"""Quality scoring and efficiency metrics — format-agnostic."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from pawbench.types import TurnResult


# ---------------------------------------------------------------------------
# Built-in format validators (users can provide their own)
# ---------------------------------------------------------------------------


def key_value_format_validator(required_keys: list[str]) -> Callable[[str], dict[str, Any]]:
    """Validator for KEY:value line-based formats (e.g., STATUS:ok, FILES_CREATED:...)."""

    def validate(text: str) -> dict[str, Any]:
        result: dict[str, Any] = {"compliant": False, "fields": {}, "missing_keys": []}
        stripped = text.strip()
        if not stripped:
            result["missing_keys"] = required_keys
            return result

        for line in stripped.split("\n"):
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                result["fields"][key.strip()] = val.strip()

        found = set(result["fields"].keys())
        result["missing_keys"] = [k for k in required_keys if k not in found]
        result["compliant"] = len(result["missing_keys"]) == 0

        # Check first line starts with first required key
        if required_keys and stripped.split("\n")[0].strip().startswith(required_keys[0] + ":"):
            result["first_key_correct"] = True
        else:
            result["first_key_correct"] = False

        return result

    return validate


def json_format_validator(required_fields: list[str] | None = None) -> Callable[[str], dict[str, Any]]:
    """Validator for JSON output format."""

    def validate(text: str) -> dict[str, Any]:
        result: dict[str, Any] = {"compliant": False, "fields": {}, "parse_error": ""}
        stripped = text.strip()
        # Extract JSON from markdown code blocks if wrapped
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, re.DOTALL)
        if match:
            stripped = match.group(1).strip()
        try:
            parsed = json.loads(stripped)
            result["fields"] = parsed if isinstance(parsed, dict) else {"_value": parsed}
            if required_fields:
                missing = [f for f in required_fields if f not in result["fields"]]
                result["compliant"] = len(missing) == 0
                result["missing_keys"] = missing
            else:
                result["compliant"] = True
        except json.JSONDecodeError as e:
            result["parse_error"] = str(e)
        return result

    return validate


# ---------------------------------------------------------------------------
# Turn-level quality scoring
# ---------------------------------------------------------------------------


def score_turn(turn_spec: dict[str, Any], result: TurnResult) -> float:
    """Score quality 0-1 based on expected outcomes defined in the scenario."""
    expect = turn_spec.get("expect", {})
    scores: list[float] = []

    if "tool_calls_min" in expect:
        scores.append(1.0 if len(result.tool_calls) >= expect["tool_calls_min"] else 0.0)

    if "tool_name_any" in expect:
        names = {tc.get("function", {}).get("name", "") for tc in result.tool_calls}
        scores.append(1.0 if names & set(expect["tool_name_any"]) else 0.0)

    if "output_mentions" in expect:
        text_lower = result.output_text.lower()
        tool_text = json.dumps(result.tool_calls).lower() if result.tool_calls else ""
        combined = text_lower + " " + tool_text
        found = sum(1 for kw in expect["output_mentions"] if kw.lower() in combined)
        scores.append(found / len(expect["output_mentions"]))

    if "steering_followed" in expect:
        steering_keywords = expect.get("steering_keywords", ["size-guide", "size_guide", "wishlist"])
        all_text = json.dumps(result.tool_calls).lower() + " " + result.output_text.lower()
        result.steering_followed = any(kw in all_text for kw in steering_keywords)
        scores.append(1.0 if result.steering_followed else 0.0)

    return sum(scores) / len(scores) if scores else 1.0


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------


def useful_ratio(text: str, tool_calls: list[dict[str, Any]] | None = None) -> float:
    """Ratio of useful content. Tool call arguments (code) count as 100% useful."""
    tc_chars = sum(len(tc.get("function", {}).get("arguments", "")) for tc in (tool_calls or []))
    text_chars = len(text.strip())
    total = tc_chars + text_chars
    if total == 0:
        return 0.0

    useful_chars = tc_chars
    if text.strip():
        filler = re.compile(r"^(Sure|Here|I'll|Let me|Of course|Certainly|Great|This|The above|Below)")
        useful_chars += sum(len(l) for l in text.strip().split("\n") if l.strip() and not filler.match(l.strip()))

    return min(useful_chars / max(total, 1), 1.0)
