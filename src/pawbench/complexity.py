"""Complexity tier taxonomy — spec 009 / B2.

Stratifies scenario tasks so aggregate scores don't mask the cliff that
Fabian Wesner's One-Shot Shop study surfaced: display tasks pass everywhere,
transactional flows expose architectural weakness immediately.

Canonical vocabulary lives in Axiom §17.2.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ComplexityTier(str, Enum):
    """Canonical complexity tiers (Axiom §17.2)."""

    DISPLAY = "display"  # Read-only render of existing data
    CRUD = "crud"  # Single-entity create/read/update/delete
    TRANSACTIONAL = "transactional"  # Multi-entity flow with invariants
    CROSS_CUTTING = "cross_cutting"  # Spans multiple subsystems

    @classmethod
    def parse(cls, value: str | None) -> "ComplexityTier | None":
        if not value:
            return None
        try:
            return cls(value.lower().replace("-", "_"))
        except ValueError:
            return None


# Heuristic keyword inference for legacy scenarios that pre-date tier tagging.
# Used only as a fallback when scenarios don't carry an explicit tier.
_TIER_KEYWORDS: dict[ComplexityTier, tuple[str, ...]] = {
    ComplexityTier.CROSS_CUTTING: (
        "auth",
        "payment",
        "checkout",
        "email",
        "webhook",
        "subscription",
        "oauth",
        "saml",
        "sso",
        "rbac",
        "audit log",
    ),
    ComplexityTier.TRANSACTIONAL: (
        "transaction",
        "rollback",
        "atomic",
        "invariant",
        "transfer",
        "checkout",
        "booking",
        "reservation",
        "two-phase",
        "saga",
    ),
    ComplexityTier.CRUD: (
        "create",
        "update",
        "delete",
        "validation",
        "endpoint",
        "rest",
        "api",
        "post",
        "put",
        "patch",
        "crud",
    ),
    ComplexityTier.DISPLAY: (
        "render",
        "display",
        "list",
        "show",
        "view",
        "page",
        "grid",
        "card",
        "html",
        "css",
    ),
}


def infer_tier(text: str) -> ComplexityTier:
    """Heuristic tier inference for un-tagged scenarios.

    Walks tiers from most-complex to least-complex and returns the first
    match. Default falls back to CRUD because it's the largest bucket in
    practice — display tier is often miscounted by keyword matches alone.
    """
    if not text:
        return ComplexityTier.CRUD
    lowered = text.lower()
    for tier in (
        ComplexityTier.CROSS_CUTTING,
        ComplexityTier.TRANSACTIONAL,
        ComplexityTier.CRUD,
        ComplexityTier.DISPLAY,
    ):
        if any(kw in lowered for kw in _TIER_KEYWORDS[tier]):
            return tier
    return ComplexityTier.CRUD


def tier_for_turn(turn_spec: dict[str, Any]) -> ComplexityTier:
    """Resolve the complexity tier for a single turn spec.

    Priority:
      1. Explicit `complexity_tier` field on the turn.
      2. Inherited from the parent scenario (handled by caller).
      3. Heuristic inference from the turn's content.
    """
    explicit = ComplexityTier.parse(turn_spec.get("complexity_tier"))
    if explicit is not None:
        return explicit
    return infer_tier(turn_spec.get("content", ""))


def tier_for_scenario(scenario: dict[str, Any]) -> ComplexityTier | None:
    """Resolve a scenario-level default tier from explicit metadata."""
    return ComplexityTier.parse(scenario.get("complexity_tier"))
