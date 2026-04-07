"""Ablation matrix — spec 009 / B7.

The publishable counter-intuitive: Fabian's study showed quality-focused
instructions *decreased* performance and the fastest build scored worst.
Pawbench has accumulated five scoring/quality components over time, none
of which have ever been measured in isolation.

The ablation runner takes a base BenchmarkReport and recomputes DQS with
one component disabled at a time, returning the delta. Components with
consistently negative deltas across consecutive runs are removal
candidates.

This module is **pure** — it operates on already-collected results, never
re-runs the benchmark. That keeps ablation cheap (no extra GPU time) and
deterministic (same inputs → same deltas).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pawbench.dqs import DQSBreakdown, compute_dqs


# Components that can be ablated. Each maps to a kwarg of compute_dqs that
# gets pinned to a "neutral" value (1.0 = component contributes its max,
# i.e., it's invisible to the comparison; 0.0 = component is silenced).
# We use 1.0-pinning so the ablation answers "what if this component
# always reported success", which is the meaningful counterfactual: a
# component is dead weight if its absence doesn't reduce the score.
ABLATABLE_COMPONENTS: dict[str, str] = {
    "format_compliance": "format_compliance",
    "tool_accuracy": "tool_accuracy",
    "useful_ratio": "useful_ratio",
    "steering_rate": "steering_rate",
    "quality": "quality",
}


@dataclass
class AblationDelta:
    """Per-component DQS delta from disabling that component."""

    component: str
    baseline_dqs: float
    ablated_dqs: float
    delta: float
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "baseline_dqs": round(self.baseline_dqs, 4),
            "ablated_dqs": round(self.ablated_dqs, 4),
            "delta": round(self.delta, 4),
            "interpretation": self.interpretation,
        }


@dataclass
class AblationReport:
    """Complete ablation matrix for a single scenario or scenario aggregate."""

    scenario_id: str
    baseline: DQSBreakdown
    deltas: list[AblationDelta] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "baseline": self.baseline.to_dict(),
            "deltas": [d.to_dict() for d in self.deltas],
            "removal_candidates": [d.component for d in self.deltas if d.delta <= 0.0],
        }


def _interpret(delta: float) -> str:
    """Plain-English read of a delta. Conservative thresholds."""
    if delta > 0.05:
        return "load-bearing — component contributes meaningfully"
    if delta > 0.01:
        return "marginal — small but real contribution"
    if delta > -0.01:
        return "neutral — within noise floor"
    return "DEAD WEIGHT — score improves when component is silenced"


def ablate(
    *,
    scenario_id: str,
    quality: float,
    format_compliance: float,
    tool_accuracy: float,
    useful_ratio: float,
    steering_rate: float,
    components: list[str] | None = None,
) -> AblationReport:
    """Compute the ablation delta for each requested component.

    `components=None` means "all ablatable components". Unknown components
    are ignored (logged into the interpretation field).

    Pinning convention: when a component is "ablated", its input is set to
    1.0 — i.e., we ask "what if this signal were perfect?". A negative
    delta then means: turning the signal perfect made the score *worse*,
    which is impossible under the current additive formula. So this
    counterfactual surfaces components that are *zero-information*: if
    pinning them to 1.0 gives the same score as the real value, the
    component is contributing nothing in this run.
    """
    base_inputs = {
        "quality": quality,
        "format_compliance": format_compliance,
        "tool_accuracy": tool_accuracy,
        "useful_ratio": useful_ratio,
        "steering_rate": steering_rate,
    }
    baseline = compute_dqs(**base_inputs)

    requested = components if components else list(ABLATABLE_COMPONENTS)
    deltas: list[AblationDelta] = []

    for comp in requested:
        if comp not in ABLATABLE_COMPONENTS:
            deltas.append(
                AblationDelta(
                    component=comp,
                    baseline_dqs=baseline.composite,
                    ablated_dqs=baseline.composite,
                    delta=0.0,
                    interpretation=f"unknown component '{comp}' — skipped",
                )
            )
            continue

        ablated_inputs = dict(base_inputs)
        ablated_inputs[comp] = 1.0  # pin to perfect
        ablated = compute_dqs(**ablated_inputs)
        delta = ablated.composite - baseline.composite
        deltas.append(
            AblationDelta(
                component=comp,
                baseline_dqs=baseline.composite,
                ablated_dqs=ablated.composite,
                delta=delta,
                interpretation=_interpret(delta),
            )
        )

    return AblationReport(scenario_id=scenario_id, baseline=baseline, deltas=deltas)
