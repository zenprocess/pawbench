"""Spec 009 / B7 — ablation matrix tests."""
from __future__ import annotations

import pytest

from pawbench.ablation import ABLATABLE_COMPONENTS, ablate


def _common(**overrides):
    base = dict(
        scenario_id="test",
        quality=0.6,
        format_compliance=0.7,
        tool_accuracy=0.8,
        useful_ratio=0.5,
        steering_rate=0.4,
    )
    base.update(overrides)
    return base


def test_ablate_all_components_by_default():
    rep = ablate(**_common())
    assert {d.component for d in rep.deltas} == set(ABLATABLE_COMPONENTS)


def test_ablate_specific_components():
    rep = ablate(**_common(), components=["quality", "useful_ratio"])
    assert [d.component for d in rep.deltas] == ["quality", "useful_ratio"]


def test_ablating_already_perfect_signal_yields_zero_delta():
    rep = ablate(**_common(quality=1.0), components=["quality"])
    assert rep.deltas[0].delta == pytest.approx(0.0)
    assert "neutral" in rep.deltas[0].interpretation or "noise" in rep.deltas[0].interpretation


def test_ablating_low_signal_yields_positive_delta_for_high_weight_component():
    """Pinning quality (weight 0.50) from 0 to 1 must lift DQS by 0.50."""
    rep = ablate(**_common(quality=0.0), components=["quality"])
    assert rep.deltas[0].delta == pytest.approx(0.50)
    assert "load-bearing" in rep.deltas[0].interpretation


def test_unknown_component_is_skipped_with_explanation():
    rep = ablate(**_common(), components=["nonexistent"])
    assert len(rep.deltas) == 1
    assert rep.deltas[0].delta == 0.0
    assert "unknown" in rep.deltas[0].interpretation


def test_ablation_report_to_dict_lists_removal_candidates():
    rep = ablate(**_common(quality=1.0, format_compliance=1.0, tool_accuracy=1.0, useful_ratio=1.0, steering_rate=1.0))
    d = rep.to_dict()
    # Everything is already perfect → every component is a "removal candidate"
    # under the >= 0 threshold (delta exactly 0).
    assert set(d["removal_candidates"]) == set(ABLATABLE_COMPONENTS)


def test_ablation_baseline_is_unchanged_across_components():
    rep = ablate(**_common())
    baselines = {d.baseline_dqs for d in rep.deltas}
    assert len(baselines) == 1
