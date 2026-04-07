"""Dispatch Quality Score (DQS) — composite scoring for Pawbench results.

DQS is intentionally simple, transparent, and version-pinned. Every change
to the formula bumps the version so historical results stay comparable.

This is *not* the same DQS that lives in Switchyard's optimizer — it's the
Pawbench-side composite that aggregates the four existing dimensions plus
the new spec 009 axes (complexity tier, artifact quality, verifier
agreement). The Switchyard DQS is per-dispatch; the Pawbench DQS is
per-scenario-run.

Formula (DQS v1):
    DQS = 0.50 * quality
        + 0.20 * format_compliance
        + 0.15 * tool_accuracy
        + 0.10 * useful_ratio
        + 0.05 * steering_rate

Artifact quality and verifier agreement are reported alongside DQS but
NOT folded in until calibration data justifies it (spec 009 §B4 explicit
requirement).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DQS_VERSION = "1.0.0"


@dataclass
class DQSBreakdown:
    """Auditable view of a DQS computation."""

    quality: float = 0.0
    format_compliance: float = 0.0
    tool_accuracy: float = 0.0
    useful_ratio: float = 0.0
    steering_rate: float = 0.0
    composite: float = 0.0
    version: str = DQS_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "composite": round(self.composite, 4),
            "components": {
                "quality": round(self.quality, 4),
                "format_compliance": round(self.format_compliance, 4),
                "tool_accuracy": round(self.tool_accuracy, 4),
                "useful_ratio": round(self.useful_ratio, 4),
                "steering_rate": round(self.steering_rate, 4),
            },
            "weights": {
                "quality": 0.50,
                "format_compliance": 0.20,
                "tool_accuracy": 0.15,
                "useful_ratio": 0.10,
                "steering_rate": 0.05,
            },
        }


def compute_dqs(
    *,
    quality: float,
    format_compliance: float,
    tool_accuracy: float,
    useful_ratio: float,
    steering_rate: float,
) -> DQSBreakdown:
    """Compute DQS from per-scenario aggregate metrics. All inputs in 0..1."""
    def _clamp(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    q = _clamp(quality)
    f = _clamp(format_compliance)
    t = _clamp(tool_accuracy)
    u = _clamp(useful_ratio)
    s = _clamp(steering_rate)

    composite = 0.50 * q + 0.20 * f + 0.15 * t + 0.10 * u + 0.05 * s
    return DQSBreakdown(
        quality=q,
        format_compliance=f,
        tool_accuracy=t,
        useful_ratio=u,
        steering_rate=s,
        composite=composite,
    )


def dqs_spread(scores: list[float]) -> float:
    """Max − min across a list of DQS values. The headline orchestration SLI.

    Spec 009 §5: high spread means orchestration shape mattered more than
    model — exactly the One-Shot Shop finding, re-derived from our data.
    """
    if not scores:
        return 0.0
    return max(scores) - min(scores)
