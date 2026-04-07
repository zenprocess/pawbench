"""Context tier transformations — spec 009 / B6.

Manifest-only mode strips embedded code from prompts so we can test whether
agents can solve hard tasks via exploration when given only file inventories.
Standard mode is a passthrough.

Lives in its own module so tests and tools can import it without pulling in
the full benchmark engine (which depends on aiohttp).
"""

from __future__ import annotations

import copy
import re
from typing import Any

_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```")
_LONG_LITERAL_RE = re.compile(r"\{[^{}]{200,}\}")

CONTEXT_TIERS = ("standard", "manifest-only")


def strip_code_from_content(text: str) -> str:
    """Remove fenced code blocks and long inline literals from prompt text."""
    text = _FENCED_CODE_RE.sub("[code block removed — manifest-only mode]", text)
    text = _LONG_LITERAL_RE.sub("[long literal removed — manifest-only mode]", text)
    return text


def apply_context_tier(scenario: dict[str, Any], tier: str) -> dict[str, Any]:
    """Return a scenario transformed for the requested context tier.

    `standard` returns the input unchanged (identity preserved).
    `manifest-only` returns a deep copy with code stripped from every turn.
    """
    if tier == "standard":
        return scenario
    if tier not in CONTEXT_TIERS:
        raise ValueError(f"unknown context tier: {tier!r}; valid: {CONTEXT_TIERS}")
    out = copy.deepcopy(scenario)
    for agent in out.get("agents", []):
        for turn in agent.get("turns", []):
            content = turn.get("content")
            if isinstance(content, str):
                turn["content"] = strip_code_from_content(content)
    return out
