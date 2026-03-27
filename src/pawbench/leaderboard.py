"""PawBench leaderboard — validate submissions, render tables, submit results."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "leaderboard" / "schema.json"


def _load_schema() -> dict:
    """Load the leaderboard JSON schema."""
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def _validate_type(value: Any, prop_schema: dict[str, Any], field_name: str) -> list[str]:
    """Validate a single value against its property schema. Returns list of errors."""
    errors: list[str] = []
    expected_type = prop_schema.get("type")

    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    if expected_type and expected_type in type_map:
        py_type = type_map[expected_type]
        # Allow int for number type
        if not isinstance(value, py_type):
            errors.append(f"'{field_name}' must be type {expected_type}, got {type(value).__name__}")
            return errors

    if expected_type == "string":
        min_len = prop_schema.get("minLength")
        if min_len is not None and len(value) < min_len:
            errors.append(f"'{field_name}' must have minimum length {min_len}")

    if expected_type in ("number", "integer"):
        minimum = prop_schema.get("minimum")
        maximum = prop_schema.get("maximum")
        if minimum is not None and value < minimum:
            errors.append(f"'{field_name}' must be >= {minimum}, got {value}")
        if maximum is not None and value > maximum:
            errors.append(f"'{field_name}' must be <= {maximum}, got {value}")

    if expected_type == "array":
        items_schema = prop_schema.get("items", {})
        for i, item in enumerate(value):
            errors.extend(_validate_type(item, items_schema, f"{field_name}[{i}]"))

    if expected_type == "object":
        nested_required = prop_schema.get("required", [])
        nested_props = prop_schema.get("properties", {})
        for req in nested_required:
            if req not in value:
                errors.append(f"'{field_name}' missing required field '{req}'")
        for k, v in value.items():
            if k in nested_props:
                errors.extend(_validate_type(v, nested_props[k], f"{field_name}.{k}"))

    return errors


def validate_submission(path: str | Path) -> list[str]:
    """Validate a JSON result file against the leaderboard schema.

    Returns a list of error strings. Empty list means valid.
    """
    path = Path(path)
    errors: list[str] = []

    if not path.exists():
        return [f"File not found: {path}"]

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    if not isinstance(data, dict):
        return ["Submission must be a JSON object"]

    schema = _load_schema()

    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")

    # Check additionalProperties
    if not schema.get("additionalProperties", True):
        allowed = set(schema.get("properties", {}).keys())
        for key in data:
            if key not in allowed:
                errors.append(f"Unknown field: '{key}'")

    # Validate types and constraints
    properties = schema.get("properties", {})
    for field, value in data.items():
        if field in properties:
            errors.extend(_validate_type(value, properties[field], field))

    return errors


def render_leaderboard(results_dir: str | Path) -> str:
    """Load all submissions from results_dir, sort by peak_tok_s desc, return markdown table."""
    results_dir = Path(results_dir)
    submissions: list[dict] = []

    if not results_dir.exists():
        return "No results directory found."

    for p in sorted(results_dir.glob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            data["_file"] = p.name
            submissions.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    if not submissions:
        return "No valid submissions found."

    # Sort by peak throughput descending
    submissions.sort(key=lambda s: s.get("peak_tok_s", 0), reverse=True)

    # Build markdown table
    lines: list[str] = []
    lines.append("# PawBench Leaderboard")
    lines.append("")
    lines.append("| Rank | Model | GPU | Config | Single tok/s | Peak tok/s | Quality | TTFT (ms) |")
    lines.append("|------|-------|-----|--------|-------------|-----------|---------|-----------|")

    for i, s in enumerate(submissions, 1):
        model = s.get("model", "?")
        gpu = s.get("gpu", "?")
        config = s.get("config", {})
        engine = config.get("serving_engine", "?")
        quant = config.get("quantization", "")
        config_str = f"{engine}"
        if quant:
            config_str += f" ({quant})"
        single = s.get("single_tok_s", 0)
        peak = s.get("peak_tok_s", 0)
        quality = s.get("avg_quality", 0)
        ttft = s.get("avg_ttft_ms", 0)

        lines.append(
            f"| {i} | {model} | {gpu} | {config_str} | {single:.1f} | {peak:.1f} | {quality:.0%} | {ttft:.0f} |"
        )

    lines.append("")
    lines.append(f"*{len(submissions)} submission(s)*")
    return "\n".join(lines)


def submit(result_path: str | Path, results_dir: str | Path) -> list[str]:
    """Validate a result file and copy it to the results directory.

    Returns a list of errors. Empty list means success.
    """
    result_path = Path(result_path)
    results_dir = Path(results_dir)

    errors = validate_submission(result_path)
    if errors:
        return errors

    results_dir.mkdir(parents=True, exist_ok=True)
    dest = results_dir / result_path.name
    shutil.copy2(result_path, dest)
    return []
