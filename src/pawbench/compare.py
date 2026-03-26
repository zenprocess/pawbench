"""Compare two or more benchmark result files side by side."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _delta(a: float, b: float, higher_is_better: bool = True) -> str:
    if a == 0:
        return ""
    d = ((b - a) / a) * 100
    arrow = "+" if d > 0 else ""
    good = (d > 0) == higher_is_better
    color = "\033[32m" if good else "\033[31m" if abs(d) > 5 else "\033[33m"
    return f" ({color}{arrow}{d:.1f}%\033[0m)"


def compare(paths: list[str]) -> None:
    reports = []
    for p in paths:
        with open(p) as f:
            reports.append((Path(p).stem[:30], json.load(f)))

    print(f"\n{'=' * 80}")
    print(f"  PawBench Comparison — {len(reports)} configs")
    print(f"{'=' * 80}\n")

    tags = [r[0] for r in reports]
    print(f"  {'Metric':30s}" + "".join(f"  {t:>22s}" for t in tags))
    print(f"  {'─' * (30 + 24 * len(tags))}")

    metrics = [
        ("Single tok/s", lambda r: r.get("dim1_throughput", {}).get("avg_single_tok_s", 0), True, "{:.1f}"),
        ("Raw peak tok/s", lambda r: r.get("dim1_throughput", {}).get("raw_peak_tok_s", 0), True, "{:.1f}"),
        ("Avg TTFT (ms)", lambda r: r.get("dim1_throughput", {}).get("avg_ttft_ms", 0), False, "{:.0f}"),
        ("Avg quality", lambda r: r.get("dim2_quality", {}).get("avg_quality", 0), True, "{:.1%}"),
        ("Tool accuracy", lambda r: r.get("dim2_quality", {}).get("tool_accuracy", 0), True, "{:.1%}"),
        ("Useful ratio", lambda r: r.get("dim3_efficiency", {}).get("avg_useful_ratio", 0), True, "{:.1%}"),
        ("Steering rate", lambda r: r.get("dim4_adaptability", {}).get("steering_rate", 0), True, "{:.1%}"),
    ]

    for label, extract, higher_better, fmt in metrics:
        vals = [extract(r[1]) for r in reports]
        row = f"  {label:30s}"
        for i, v in enumerate(vals):
            cell = fmt.format(v)
            if i > 0:
                cell += _delta(vals[0], v, higher_better)
            row += f"  {cell:>22s}"
        print(row)

    print(f"\n{'=' * 80}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: pawbench-compare <result1.json> [result2.json ...]")
        sys.exit(1)
    compare(sys.argv[1:])


if __name__ == "__main__":
    main()
