"""Export PawBench results to ServingCard format (servingcard.dev)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pawbench.types import BenchmarkReport


def to_servingcard(report: BenchmarkReport) -> dict[str, Any]:
    """Convert a BenchmarkReport to ServingCard YAML-compatible dict."""
    mc = report.model_card
    mcfg = mc.model_config if mc.model_config else {}
    tuning = mc.tuning if mc.tuning else {}
    gpu = mc.gpu if mc.gpu else {}
    d1 = report.dim1_throughput
    d2 = report.dim2_quality
    d3 = report.dim3_efficiency
    d4 = report.dim4_adaptability
    sm = report.server_metrics

    # Determine quantization from tuning config
    quant = tuning.get("kv_cache_dtype", "unknown")
    spec = tuning.get("speculative_config", "none")

    # Build variant name
    variant_parts = [quant.replace("fp8_e4m3", "fp8")]
    if spec and spec != "none":
        if "eagle3" in str(spec).lower():
            variant_parts.append("eagle3")
        elif "mtp" in str(spec).lower():
            variant_parts.append("mtp")
    variant = "-".join(variant_parts)

    card: dict[str, Any] = {
        # Identity
        "model": mc.model_name,
        "variant": report.tag or variant,
        "hardware": gpu.get("name", "unknown").lower().replace(" ", "-"),
        "framework": "vllm",
        "method": "pawbench",
        "quantization": quant,
        # Architecture
        "architecture": mcfg.get("architectures", ["unknown"])[0] if mcfg.get("architectures") else "unknown",
        "num_experts": mcfg.get("num_experts", 0),
        "num_experts_per_tok": mcfg.get("num_experts_per_tok", 0),
        "num_hidden_layers": mcfg.get("num_hidden_layers", 0),
        # Serving config
        "gpu_memory_utilization": tuning.get("gpu_memory_utilization", "?"),
        "max_model_len": tuning.get("max_model_len", "?"),
        "max_num_seqs": tuning.get("max_num_seqs", "?"),
        "speculative_config": spec,
        # Dimension 1: Throughput
        "tok_s": round(d1.get("avg_single_tok_s", 0), 1),
        "peak_parallel_tok_s": round(d1.get("raw_peak_tok_s", 0), 1),
        "peak_concurrency": d1.get("raw_peak_concurrency", 0),
        "ttft_ms": round(d1.get("avg_ttft_ms", 0), 0),
        # Dimension 2: Quality
        "quality_score": round(d2.get("avg_quality", 0), 3),
        "format_compliance": round(d2.get("format_compliance_rate", 0), 3),
        "tool_call_accuracy": round(d2.get("tool_accuracy", 0), 3),
        # Dimension 3: Efficiency
        "useful_token_ratio": round(d3.get("avg_useful_ratio", 0), 3),
        "tokens_per_turn": round(d3.get("tokens_per_turn", 0), 0),
        # Dimension 4: Adaptability
        "steering_success_rate": round(d4.get("steering_rate", 0), 3),
        "nudge_quality": round(d4.get("nudge_quality", 0), 3),
        # Server metrics (if available)
        "spec_acceptance_rate": round(sm.get("spec_acceptance_rate", 0), 3) if sm else 0,
        "prefix_cache_hit_rate": round(sm.get("gpu_prefix_cache_hit_rate", 0), 3) if sm else 0,
        # Saturation curve
        "saturation_curve": [
            {"n": pt.get("concurrency", pt.get("n", 0)), "tok_s": round(pt.get("tok_s", 0), 1)}
            for pt in (report.saturation_curve if isinstance(report.saturation_curve, list) else [])
            if isinstance(pt, dict)
        ],
        # Metadata
        "pawbench_version": "1.1",
        "benchmarked_at": report.timestamp,
        "scenarios_run": len(report.scenarios),
        "runs": report.runs,
    }

    return card


def export_servingcard(report: BenchmarkReport, output_path: str | Path) -> Path:
    """Export a BenchmarkReport as a ServingCard JSON file."""
    card = to_servingcard(report)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(card, f, indent=2, default=str)
    return path


def export_servingcard_yaml(report: BenchmarkReport, output_path: str | Path) -> Path:
    """Export a BenchmarkReport as a ServingCard YAML file."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("PyYAML required for YAML export: pip install pyyaml") from e

    card = to_servingcard(report)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(card, f, default_flow_style=False, sort_keys=False)
    return path
