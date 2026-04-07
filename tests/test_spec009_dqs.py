"""Spec 009 — DQS composite + spread tests."""
from __future__ import annotations

import pytest

from pawbench.dqs import DQS_VERSION, compute_dqs, dqs_spread


def test_dqs_perfect_score():
    bd = compute_dqs(quality=1, format_compliance=1, tool_accuracy=1, useful_ratio=1, steering_rate=1)
    assert bd.composite == pytest.approx(1.0)
    assert bd.version == DQS_VERSION


def test_dqs_zero_score():
    bd = compute_dqs(quality=0, format_compliance=0, tool_accuracy=0, useful_ratio=0, steering_rate=0)
    assert bd.composite == 0.0


def test_dqs_quality_dominates():
    """Quality is weighted 50% — toggling it must move the score the most."""
    base = compute_dqs(quality=0, format_compliance=1, tool_accuracy=1, useful_ratio=1, steering_rate=1)
    with_quality = compute_dqs(quality=1, format_compliance=1, tool_accuracy=1, useful_ratio=1, steering_rate=1)
    assert with_quality.composite - base.composite == pytest.approx(0.50)


def test_dqs_clamps_inputs_above_one():
    bd = compute_dqs(quality=2, format_compliance=2, tool_accuracy=2, useful_ratio=2, steering_rate=2)
    assert bd.composite == pytest.approx(1.0)


def test_dqs_clamps_inputs_below_zero():
    bd = compute_dqs(quality=-1, format_compliance=-1, tool_accuracy=-1, useful_ratio=-1, steering_rate=-1)
    assert bd.composite == 0.0


def test_dqs_breakdown_to_dict_shape():
    bd = compute_dqs(quality=0.8, format_compliance=0.9, tool_accuracy=0.7, useful_ratio=0.6, steering_rate=0.5)
    d = bd.to_dict()
    assert d["version"] == DQS_VERSION
    assert "components" in d and "weights" in d
    assert sum(d["weights"].values()) == pytest.approx(1.0)
    assert set(d["components"]) == set(d["weights"])


def test_dqs_spread_empty():
    assert dqs_spread([]) == 0.0


def test_dqs_spread_single_value():
    assert dqs_spread([0.7]) == 0.0


def test_dqs_spread_basic():
    assert dqs_spread([0.4, 0.6, 0.85]) == pytest.approx(0.45)
