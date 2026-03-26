"""Human-readable report output."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from pawbench.types import BenchmarkReport


R = "\033[0m"
B = "\033[1m"
G = "\033[32m"
Y = "\033[33m"
C = "\033[36m"
D = "\033[2m"
RD = "\033[31m"


def _cpct(v: float) -> str:
    if v >= 0.8:
        return f"{G}{v:.0%}{R}"
    if v >= 0.5:
        return f"{Y}{v:.0%}{R}"
    return f"{RD}{v:.0%}{R}"


def print_report(report: BenchmarkReport) -> None:
    mc = report.model_card
    gpu_name = mc.gpu.get("name", "?") if mc.gpu else "?"

    print(f"\n{B}{C}{'=' * 80}{R}")
    print(f"{B}{C}  PawBench — 4-Dimensional LLM Inference Benchmark{R}")
    print(f"{D}  Tag: {report.tag}  |  {report.timestamp[:19]}{R}")
    print(f"{D}  Model: {mc.model_name}  |  GPU: {gpu_name}{R}")
    tuning = mc.tuning
    print(
        f"{D}  KV: {tuning.get('kv_cache_dtype', '?')}  |  Ctx: {tuning.get('max_model_len', '?')}  |  Seqs: {tuning.get('max_num_seqs', '?')}  |  Spec: {tuning.get('speculative_config', 'none')}{R}"
    )
    print(f"{B}{C}{'=' * 80}{R}\n")

    # Scenarios
    if report.scenarios:
        print(f"  {B}SCENARIOS{R}")
        print(f"  {'─' * 76}")
        print(f"  {'Scenario':35s} {'tok/s':>6s} {'Peak':>6s} {'Quality':>8s} {'Steer':>6s} {'Useful':>7s}")
        print(f"  {'─' * 76}")
        for s in report.scenarios:
            s = s if isinstance(s, dict) else asdict(s)
            print(
                f"  {s['scenario_name'][:35]:35s} "
                f"{s['single_tok_s']:>5.1f}  "
                f"{s['peak_tok_s']:>5.1f}  "
                f"{_cpct(s['avg_quality']):>16s}  "
                f"{_cpct(s['steering_rate']):>14s}  "
                f"{s['useful_ratio']:>6.0%}"
            )

    # Dim 1
    d1 = report.dim1_throughput
    print(f"\n  {B}DIM 1 — THROUGHPUT{R}")
    print(f"  {'─' * 60}")
    print(f"  Single agent:    {B}{d1.get('avg_single_tok_s', 0):.1f} tok/s{R}")
    print(f"  Avg TTFT:        {d1.get('avg_ttft_ms', 0):.0f} ms")
    print(
        f"  Raw peak:        {B}{G}{d1.get('raw_peak_tok_s', 0):.1f} tok/s @ {d1.get('raw_peak_concurrency', 0)} parallel{R}"
    )

    if report.saturation_curve:
        print(f"\n  {B}Saturation Curve:{R}")
        print(f"  {'N':>5s}  {'Aggregate tok/s':>16s}  {'Per-agent':>10s}  {'Wall':>7s}")
        for pt in report.saturation_curve:
            pt = pt if isinstance(pt, dict) else asdict(pt)
            peak = f" {G}<< peak{R}" if pt["tok_s"] == d1.get("raw_peak_tok_s", 0) else ""
            print(
                f"  {pt['concurrency']:>5d}  {pt['tok_s']:>15.1f}  {pt['per_agent']:>9.1f}  {pt['wall_s']:>6.1f}s{peak}"
            )

    # Dim 2
    d2 = report.dim2_quality
    print(f"\n  {B}DIM 2 — QUALITY{R}")
    print(f"  {'─' * 50}")
    print(f"  Avg quality:          {_cpct(d2.get('avg_quality', 0))}")
    print(f"  Format compliance:    {_cpct(d2.get('format_compliance_rate', 0))}")
    print(f"  Tool call accuracy:   {_cpct(d2.get('tool_accuracy', 0))}")

    # Dim 3
    d3 = report.dim3_efficiency
    print(f"\n  {B}DIM 3 — EFFICIENCY{R}")
    print(f"  {'─' * 50}")
    print(f"  Useful token ratio:   {d3.get('avg_useful_ratio', 0):.0%}")
    print(f"  Tokens per turn:      {d3.get('tokens_per_turn', 0):.0f}")
    print(f"  Total generated:      {d3.get('total_tokens', 0):,}")

    # Dim 4
    d4 = report.dim4_adaptability
    print(f"\n  {B}DIM 4 — ADAPTABILITY{R}")
    print(f"  {'─' * 50}")
    print(f"  Steering success:     {_cpct(d4.get('steering_rate', 0))}")
    print(f"  Nudge quality:        {_cpct(d4.get('nudge_quality', 0))}")

    # Dim 5
    if report.sandbox_score > 0:
        print(f"\n  {B}DIM 5 — CORRECTNESS{R}")
        print(f"  {'─' * 50}")
        print(f"  Sandbox pass rate:    {_cpct(report.sandbox_score)}")
        for s in report.scenarios:
            s_d = s if isinstance(s, dict) else asdict(s)
            if s_d.get("sandbox_score", 0) > 0:
                print(f"    {s_d['scenario_name'][:35]:35s} {_cpct(s_d['sandbox_score'])}")

    # Server metrics
    sm = report.server_metrics
    if sm:
        print(f"\n  {B}SERVER METRICS{R}")
        print(f"  {'─' * 50}")
        if "gpu_prefix_cache_hit_rate" in sm:
            print(f"  Prefix cache hit:     {sm['gpu_prefix_cache_hit_rate']:.1%}")
        if "spec_acceptance_rate" in sm:
            print(f"  Spec acceptance rate: {sm['spec_acceptance_rate']:.1%}")
        if "gpu_cache_usage_perc" in sm:
            print(f"  KV cache usage:       {sm['gpu_cache_usage_perc']:.1%}")

    print(f"\n{B}{C}{'=' * 80}{R}\n")
