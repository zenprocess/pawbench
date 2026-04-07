"""PawBench CLI — run benchmarks against any OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import requests

from pawbench import __version__
from pawbench.ablation import ablate
from pawbench.banner import print_banner
from pawbench.capture import capture_model_card, scrape_server_metrics
from pawbench.dqs import compute_dqs, dqs_spread
from pawbench.engine import run_parallel_dispatch, run_saturation_test
from pawbench.mock import (
    MockEndpoint,
    default_mock_model_card,
    load_fixture,
    save_fixture,
)
from pawbench.orchestration import OrchestrationShape, run_with_shape
from pawbench.quality import analyze_artifact
from pawbench.report import print_report
from pawbench.scoring import quality_by_tier, useful_ratio
from pawbench.types import BenchmarkReport, ScenarioReport

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
FIXTURES_DIR = Path(__file__).parent / "fixtures"

from pawbench.context_tier import apply_context_tier as _apply_context_tier  # noqa: E402


def _get_model(endpoint: str) -> str:
    try:
        return requests.get(f"{endpoint}/v1/models", timeout=5).json()["data"][0]["id"]
    except Exception:
        return "unknown"


def _load_scenarios(scenario_paths: list[Path] | None = None) -> list[dict]:
    if scenario_paths:
        scenarios = []
        for p in scenario_paths:
            with open(p) as f:
                scenarios.append(json.load(f))
        return scenarios

    # Load all built-in scenarios
    scenarios = []
    for p in sorted(SCENARIOS_DIR.glob("*.json")):
        with open(p) as f:
            scenarios.append(json.load(f))
    return scenarios


def _run_scenario(
    endpoint: str,
    model: str,
    scenario: dict,
    concurrency_levels: list[int],
    runs: int,
) -> tuple[ScenarioReport, list]:
    sr = ScenarioReport(scenario_id=scenario["id"], scenario_name=scenario["name"])
    all_cr = []
    all_agents = []

    for _ in range(runs):
        for conc in concurrency_levels:
            cr = asyncio.run(run_parallel_dispatch(endpoint, model, scenario, conc))
            all_cr.append(cr)
            for a in cr.agents:
                if not a.error:
                    all_agents.append(a)

    single_crs = [c for c in all_cr if c.concurrency == 1]
    sr.single_tok_s = sum(c.per_agent_tok_s for c in single_crs) / max(len(single_crs), 1)
    if all_cr:
        peak = max(all_cr, key=lambda c: c.aggregate_tok_s)
        sr.peak_tok_s = peak.aggregate_tok_s
        sr.peak_concurrency = peak.concurrency

    all_turns = [t for a in all_agents for t in a.turns]
    ttfts = [t.ttft_ms for t in all_turns if t.ttft_ms > 0]
    sr.avg_ttft_ms = sum(ttfts) / max(len(ttfts), 1)
    sr.avg_quality = sum(a.avg_quality for a in all_agents) / max(len(all_agents), 1)

    final_turns = [t for a in all_agents for t in a.turns if t.turn == max(tt.turn for tt in a.turns)]
    sr.format_compliance_rate = sum(1 for t in final_turns if t.format_compliant) / max(len(final_turns), 1)
    sr.tool_accuracy = sum(1 for t in all_turns if t.tool_calls) / max(len(all_turns), 1)
    sr.useful_ratio = sum(useful_ratio(t.output_text, t.tool_calls) for t in all_turns) / max(len(all_turns), 1)
    sr.total_tokens = sum(a.total_completion_tokens for a in all_agents)
    sr.tokens_per_turn = sr.total_tokens / max(len(all_turns), 1)
    sr.steering_rate = sum(1 for a in all_agents if a.steering_success) / max(len(all_agents), 1)

    steering_turns = [t for a in all_agents for t in a.turns if t.steering_followed]
    if steering_turns:
        sr.nudge_response_quality = sum(t.quality_score for t in steering_turns) / len(steering_turns)

    return sr, all_cr


def main():
    parser = argparse.ArgumentParser(
        prog="pawbench",
        description="PawBench — 4-dimensional LLM inference benchmark",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="OpenAI-compatible API endpoint (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--concurrency", default="1,2,4,8", help="Comma-separated concurrency levels for parallel sweep"
    )
    parser.add_argument("--runs", type=int, default=1, help="Runs per scenario (results averaged)")
    parser.add_argument("--tag", default="default", help="Tag for this benchmark run (used in output filename)")
    parser.add_argument("--output", default=None, help="Directory to save JSON results")
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Path to custom scenario JSON (can repeat). Omit for built-in PawStyle scenarios.",
    )
    parser.add_argument(
        "--saturation-only", action="store_true", help="Only run raw throughput saturation test (skip scenarios)"
    )
    parser.add_argument("--no-saturation", action="store_true", help="Skip raw saturation test")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of human-readable report")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use recorded fixtures instead of a live endpoint (for CI)",
    )
    parser.add_argument(
        "--record",
        default=None,
        metavar="DIR",
        help="Record API responses as fixtures to DIR for future --mock runs",
    )
    parser.add_argument(
        "--servingcard",
        default=None,
        metavar="PATH",
        help="Export results as a ServingCard file (.json or .yaml) for servingcard.dev",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    # Spec 009 — orchestration × complexity matrix
    parser.add_argument(
        "--orchestration",
        default=None,
        help="Comma-separated orchestration shapes (flat,waves,scatter-gather,team-mode,subagents). "
             "Runs the same scenario under each shape and reports per-shape DQS + spread.",
    )
    parser.add_argument(
        "--ablate",
        default=None,
        help="Comma-separated component names to ablate (quality,format_compliance,tool_accuracy,"
             "useful_ratio,steering_rate). Recomputes DQS with each component pinned to perfect.",
    )
    parser.add_argument(
        "--context-tier",
        default="standard",
        choices=["standard", "manifest-only"],
        help="Spec 009/B6 — manifest-only strips embedded code from prompts to test exploration-driven solving.",
    )
    parser.add_argument(
        "--verification-runs",
        type=int,
        default=1,
        help="Spec 009/B3 — re-score the same outputs N times to measure verifier reliability (>=1).",
    )
    parser.add_argument(
        "--no-quality-analysis",
        action="store_true",
        help="Skip artifact_quality static analysis (spec 009/B4). Useful in CI without ruff/mypy.",
    )
    args = parser.parse_args()

    conc_levels = [int(c) for c in args.concurrency.split(",")]

    # Mock / record mode setup
    mock_endpoint: MockEndpoint | None = None
    if args.mock:
        fixture_path = FIXTURES_DIR / "sample_saturation.json"
        fixture = load_fixture(fixture_path)
        mock_endpoint = MockEndpoint(fixture)
        model = mock_endpoint.model
        model_card = default_mock_model_card()
        model_card.model_name = model
    else:
        model = _get_model(args.endpoint)
        model_card = capture_model_card(args.endpoint)

    recorder: MockEndpoint | None = None
    if args.record:
        recorder = MockEndpoint()
        recorder._fixture.model = model

    if not args.json:
        print_banner()
        print(f"  Model: {model}  |  Endpoint: {args.endpoint}")
        print(f"  Concurrency: {conc_levels}  |  Runs: {args.runs}")
        print()

    scenario_reports: list[ScenarioReport] = []
    all_concurrency_data: dict[int, list] = {}

    if not args.saturation_only:
        scenario_paths = [Path(p) for p in args.scenario] if args.scenario else None
        scenarios = _load_scenarios(scenario_paths)
        # Spec 009/B6 — apply context tier transformation
        scenarios = [_apply_context_tier(s, args.context_tier) for s in scenarios]

        if not args.json:
            print(f"  Scenarios: {len(scenarios)}  |  Context tier: {args.context_tier}")

        # Per-scenario aggregate state for spec 009 reporting
        scenario_artifact_quality: list[dict] = []
        scenario_quality_by_tier: list[dict] = []
        scenario_orchestration: list[dict] = []
        scenario_dqs: list[float] = []

        orchestration_shapes: list[OrchestrationShape] = []
        if args.orchestration:
            orchestration_shapes = [OrchestrationShape.parse(s) for s in args.orchestration.split(",") if s.strip()]

        for scenario in scenarios:
            if not args.json:
                print(f"  Running: {scenario['name']} ...", end="", flush=True)

            sr, crs = _run_scenario(args.endpoint, model, scenario, conc_levels, args.runs)
            scenario_reports.append(sr)

            for cr in crs:
                all_concurrency_data.setdefault(cr.concurrency, []).append(cr)

            # Spec 009/B2 — per-tier quality breakdown
            all_agents_for_tier = [a for cr in crs for a in cr.agents if not a.error]
            tier_breakdown = quality_by_tier(all_agents_for_tier, scenario)
            if tier_breakdown:
                scenario_quality_by_tier.append({"scenario_id": scenario["id"], "by_tier": tier_breakdown})

            # Spec 009/B4 — artifact quality analysis on collected tool calls
            if not args.no_quality_analysis:
                all_tool_calls = [tc for a in all_agents_for_tier for t in a.turns for tc in t.tool_calls]
                aq = analyze_artifact(all_tool_calls)
                scenario_artifact_quality.append({"scenario_id": scenario["id"], **aq.__dict__})

            # Spec 009/B1 — orchestration matrix (re-runs scenario per shape)
            if orchestration_shapes and not args.mock:
                shape_results = []
                shape_dqs_list: list[float] = []
                for shape in orchestration_shapes:
                    res = asyncio.run(run_with_shape(args.endpoint, model, scenario, shape))
                    shape_results.append(res.to_dict())
                    # Quick DQS on the orchestration result using its avg_quality
                    bd = compute_dqs(
                        quality=res.avg_quality,
                        format_compliance=sr.format_compliance_rate,
                        tool_accuracy=sr.tool_accuracy,
                        useful_ratio=sr.useful_ratio,
                        steering_rate=sr.steering_rate,
                    )
                    shape_dqs_list.append(bd.composite)
                scenario_orchestration.append({
                    "scenario_id": scenario["id"],
                    "shapes": shape_results,
                    "dqs_per_shape": dict(zip([s.value for s in orchestration_shapes], shape_dqs_list)),
                    "dqs_spread": dqs_spread(shape_dqs_list),
                })

            # Per-scenario DQS
            sd = compute_dqs(
                quality=sr.avg_quality,
                format_compliance=sr.format_compliance_rate,
                tool_accuracy=sr.tool_accuracy,
                useful_ratio=sr.useful_ratio,
                steering_rate=sr.steering_rate,
            )
            scenario_dqs.append(sd.composite)

            if not args.json:
                print(f" tok/s={sr.single_tok_s:.1f}  quality={sr.avg_quality:.0%}  steer={sr.steering_rate:.0%}  dqs={sd.composite:.2f}")

    # Raw saturation test
    saturation_curve = []
    if not args.no_saturation:
        if not args.json:
            print("\n  Running: Raw throughput saturation ...", end="", flush=True)
        if mock_endpoint:
            saturation_curve = mock_endpoint.get_saturation_points()
        else:
            sat_levels = sorted(set(conc_levels + [1, 2, 4, 8]))
            saturation_curve = asyncio.run(run_saturation_test(args.endpoint, model, sat_levels))
            if recorder:
                for pt in saturation_curve:
                    recorder._fixture.saturation.append(
                        {
                            "concurrency": pt.concurrency,
                            "tok_s": pt.tok_s,
                            "per_agent": pt.per_agent,
                            "wall_s": pt.wall_s,
                            "total_tokens": pt.total_tokens,
                        }
                    )
        peak = max(saturation_curve, key=lambda p: p.tok_s) if saturation_curve else None
        if not args.json and peak:
            print(f" peak: {peak.tok_s:.1f} tok/s @ N={peak.concurrency}")

    # Server metrics
    if mock_endpoint:
        server_metrics: dict = {}
    else:
        server_metrics = scrape_server_metrics(args.endpoint)

    # Save recorded fixture if --record was used
    if recorder and args.record:
        record_dir = Path(args.record)
        fixture = recorder.build_fixture(name="recorded", model=model)
        fixture.saturation = recorder._fixture.saturation
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_fixture(fixture, record_dir / f"recorded_{ts}.json")
        if not args.json:
            print(f"\n  Recorded fixture saved to {record_dir}/recorded_{ts}.json")

    # Build concurrency curve from scenarios
    conc_curve = []
    for conc in sorted(all_concurrency_data.keys()):
        crs = all_concurrency_data[conc]
        conc_curve.append(
            {
                "concurrency": conc,
                "tok_s": sum(c.aggregate_tok_s for c in crs) / len(crs),
                "per_agent": sum(c.per_agent_tok_s for c in crs) / len(crs),
            }
        )

    peak_sat = max(saturation_curve, key=lambda p: p.tok_s) if saturation_curve else None
    valid_scenarios = [s for s in scenario_reports]

    # Identify scenario types for adaptability
    independent = [s for s in valid_scenarios if "independent" in s.scenario_id]
    nudged = [s for s in valid_scenarios if "nudge" in s.scenario_id]

    report = BenchmarkReport(
        tag=args.tag,
        timestamp=datetime.now(timezone.utc).isoformat(),
        endpoint=args.endpoint,
        model_card=model_card,
        runs=args.runs,
        scenarios=scenario_reports,
        saturation_curve=saturation_curve,
        concurrency_curve=conc_curve,
        server_metrics=server_metrics,
        dim1_throughput={
            "avg_single_tok_s": sum(s.single_tok_s for s in valid_scenarios) / max(len(valid_scenarios), 1),
            "avg_ttft_ms": sum(s.avg_ttft_ms for s in valid_scenarios) / max(len(valid_scenarios), 1),
            "raw_peak_tok_s": peak_sat.tok_s if peak_sat else 0,
            "raw_peak_concurrency": peak_sat.concurrency if peak_sat else 0,
        },
        dim2_quality={
            "avg_quality": sum(s.avg_quality for s in valid_scenarios) / max(len(valid_scenarios), 1),
            "format_compliance_rate": (
                sum(s.format_compliance_rate for s in valid_scenarios) / max(len(valid_scenarios), 1)
            ),
            "tool_accuracy": sum(s.tool_accuracy for s in valid_scenarios) / max(len(valid_scenarios), 1),
        },
        dim3_efficiency={
            "avg_useful_ratio": sum(s.useful_ratio for s in valid_scenarios) / max(len(valid_scenarios), 1),
            "tokens_per_turn": sum(s.tokens_per_turn for s in valid_scenarios) / max(len(valid_scenarios), 1),
            "total_tokens": sum(s.total_tokens for s in valid_scenarios),
        },
        dim4_adaptability={
            "steering_rate": sum(s.steering_rate for s in valid_scenarios) / max(len(valid_scenarios), 1),
            "nudge_quality": sum(s.nudge_response_quality for s in nudged) / max(len(nudged), 1),
            "independent_quality": sum(s.avg_quality for s in independent) / max(len(independent), 1),
            "nudged_quality": sum(s.avg_quality for s in nudged) / max(len(nudged), 1),
        },
    )

    # Spec 009 — attach orchestration × complexity matrix outputs
    if not args.saturation_only and valid_scenarios:
        # B4 — artifact quality (single aggregate row across scenarios)
        if scenario_artifact_quality:
            report.dim5_artifact_quality = {
                "version": "spec-009",
                "per_scenario": scenario_artifact_quality,
                "aggregate_score": sum(
                    r.get("score", 0.0) for r in scenario_artifact_quality
                ) / len(scenario_artifact_quality),
            }
        # B2 — quality_by_tier aggregate
        agg_tiers: dict[str, list[float]] = {}
        for entry in scenario_quality_by_tier:
            for tier_name, score in entry["by_tier"].items():
                agg_tiers.setdefault(tier_name, []).append(score)
        report.quality_by_tier = {t: sum(v) / len(v) for t, v in agg_tiers.items()}
        # B1 — orchestration matrix
        if scenario_orchestration:
            report.orchestration_results = scenario_orchestration
            spreads = [e["dqs_spread"] for e in scenario_orchestration]
            report.orchestration_dqs_spread = max(spreads) if spreads else 0.0
        # DQS aggregate
        agg_dqs = compute_dqs(
            quality=report.dim2_quality["avg_quality"],
            format_compliance=report.dim2_quality["format_compliance_rate"],
            tool_accuracy=report.dim2_quality["tool_accuracy"],
            useful_ratio=report.dim3_efficiency["avg_useful_ratio"],
            steering_rate=report.dim4_adaptability["steering_rate"],
        )
        report.dqs = agg_dqs.to_dict()
        # B7 — ablation matrix
        if args.ablate is not None:
            requested = [c.strip() for c in args.ablate.split(",") if c.strip()] if args.ablate else None
            ab = ablate(
                scenario_id="aggregate",
                quality=report.dim2_quality["avg_quality"],
                format_compliance=report.dim2_quality["format_compliance_rate"],
                tool_accuracy=report.dim2_quality["tool_accuracy"],
                useful_ratio=report.dim3_efficiency["avg_useful_ratio"],
                steering_rate=report.dim4_adaptability["steering_rate"],
                components=requested,
            )
            report.ablation = ab.to_dict()
        # B3 — verification reliability via N-run scoring agreement
        if args.verification_runs > 1:
            # Scoring is deterministic given outputs, so the only "disagreement"
            # source is non-determinism in score_turn (none today). We still
            # surface the field so downstream consumers can plug in real
            # LLM-judged verification later. Records: N runs, all agreeing.
            report.dqs.setdefault("verification", {})
            report.dqs["verification"] = {
                "runs": args.verification_runs,
                "agreement_rate": 1.0,
                "notes": "deterministic scoring — agreement_rate is 1.0 by construction. "
                         "Plug in an LLM judge to surface real verifier flake.",
            }

    if args.json:
        print(json.dumps(asdict(report), indent=2, default=str))
    else:
        print_report(report)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"pawbench_{args.tag}_{ts}.json"
        with open(out_file, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        if not args.json:
            print(f"  Saved to {out_file}")

    if args.servingcard:
        from pawbench.servingcard import export_servingcard, export_servingcard_yaml

        sc_path = Path(args.servingcard)
        if sc_path.suffix in (".yaml", ".yml"):
            export_servingcard_yaml(report, sc_path)
        else:
            export_servingcard(report, sc_path)
        if not args.json:
            print(f"  ServingCard exported to {sc_path}")
            print("  Submit to https://servingcard.dev")


if __name__ == "__main__":
    main()
