"""Spec 009 / B2 — complexity tier taxonomy tests."""

from __future__ import annotations

from pawbench.complexity import (
    ComplexityTier,
    infer_tier,
    tier_for_scenario,
    tier_for_turn,
)


def test_parse_canonical_values():
    for v in ("display", "crud", "transactional", "cross_cutting"):
        assert ComplexityTier.parse(v) is ComplexityTier(v)


def test_parse_normalizes_dashes_and_case():
    assert ComplexityTier.parse("CROSS-CUTTING") is ComplexityTier.CROSS_CUTTING
    assert ComplexityTier.parse("Display") is ComplexityTier.DISPLAY


def test_parse_returns_none_for_garbage():
    assert ComplexityTier.parse("nope") is None
    assert ComplexityTier.parse("") is None
    assert ComplexityTier.parse(None) is None  # type: ignore[arg-type]


def test_infer_prefers_more_complex_tier():
    text = "Build a checkout endpoint that validates auth tokens and rolls back on failure"
    # Both 'auth' (cross_cutting) and 'rollback' (transactional) match;
    # cross_cutting wins because it's checked first.
    assert infer_tier(text) is ComplexityTier.CROSS_CUTTING


def test_infer_falls_back_to_crud():
    assert infer_tier("do something completely unspecified") is ComplexityTier.CRUD


def test_infer_display_for_pure_render():
    assert infer_tier("render a static html page with a flexbox grid") is ComplexityTier.DISPLAY


def test_tier_for_turn_explicit_wins():
    turn = {"complexity_tier": "display", "content": "build a transactional checkout"}
    assert tier_for_turn(turn) is ComplexityTier.DISPLAY


def test_tier_for_turn_falls_back_to_inference():
    turn = {"content": "implement a payment webhook with audit log"}
    assert tier_for_turn(turn) is ComplexityTier.CROSS_CUTTING


def test_tier_for_scenario_returns_none_when_unset():
    assert tier_for_scenario({}) is None
    assert tier_for_scenario({"complexity_tier": "crud"}) is ComplexityTier.CRUD
