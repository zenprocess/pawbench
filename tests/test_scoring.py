"""Tests for pawbench.scoring module."""

from __future__ import annotations

from pawbench.scoring import json_format_validator, key_value_format_validator, score_turn, useful_ratio
from pawbench.types import TurnResult

# ---------------------------------------------------------------------------
# key_value_format_validator
# ---------------------------------------------------------------------------


class TestKeyValueFormatValidator:
    def test_compliant_input(self) -> None:
        validator = key_value_format_validator(["STATUS", "FILES_CREATED", "LEARNED"])
        text = "STATUS:ok\nFILES_CREATED:index.html\nLEARNED:built the page"
        result = validator(text)
        assert result["compliant"] is True
        assert result["missing_keys"] == []
        assert result["fields"]["STATUS"] == "ok"
        assert result["fields"]["FILES_CREATED"] == "index.html"

    def test_non_compliant_missing_keys(self) -> None:
        validator = key_value_format_validator(["STATUS", "FILES_CREATED", "LEARNED"])
        text = "STATUS:ok\nFILES_CREATED:index.html"
        result = validator(text)
        assert result["compliant"] is False
        assert "LEARNED" in result["missing_keys"]

    def test_empty_input(self) -> None:
        validator = key_value_format_validator(["STATUS"])
        result = validator("")
        assert result["compliant"] is False
        assert result["missing_keys"] == ["STATUS"]

    def test_whitespace_handling(self) -> None:
        validator = key_value_format_validator(["STATUS"])
        text = "  STATUS : ok  "
        result = validator(text)
        assert result["compliant"] is True
        assert result["fields"]["STATUS"] == "ok"

    def test_first_key_correct(self) -> None:
        validator = key_value_format_validator(["STATUS", "BUILD"])
        text = "STATUS:ok\nBUILD:pass"
        result = validator(text)
        assert result["first_key_correct"] is True

    def test_first_key_wrong_order(self) -> None:
        validator = key_value_format_validator(["STATUS", "BUILD"])
        text = "BUILD:pass\nSTATUS:ok"
        result = validator(text)
        assert result["compliant"] is True  # all keys present
        assert result["first_key_correct"] is False

    def test_extra_keys_allowed(self) -> None:
        validator = key_value_format_validator(["STATUS"])
        text = "STATUS:ok\nEXTRA:value"
        result = validator(text)
        assert result["compliant"] is True
        assert "EXTRA" in result["fields"]


# ---------------------------------------------------------------------------
# json_format_validator
# ---------------------------------------------------------------------------


class TestJsonFormatValidator:
    def test_valid_json_no_required(self) -> None:
        validator = json_format_validator()
        result = validator('{"name": "test", "value": 42}')
        assert result["compliant"] is True
        assert result["fields"]["name"] == "test"

    def test_valid_json_with_required_fields(self) -> None:
        validator = json_format_validator(["name", "value"])
        result = validator('{"name": "test", "value": 42}')
        assert result["compliant"] is True

    def test_missing_required_fields(self) -> None:
        validator = json_format_validator(["name", "value", "missing_field"])
        result = validator('{"name": "test", "value": 42}')
        assert result["compliant"] is False
        assert "missing_field" in result["missing_keys"]

    def test_invalid_json(self) -> None:
        validator = json_format_validator()
        result = validator("not valid json at all")
        assert result["compliant"] is False
        assert result["parse_error"] != ""

    def test_json_in_code_block(self) -> None:
        validator = json_format_validator(["status"])
        text = '```json\n{"status": "ok"}\n```'
        result = validator(text)
        assert result["compliant"] is True
        assert result["fields"]["status"] == "ok"

    def test_json_array(self) -> None:
        validator = json_format_validator()
        result = validator("[1, 2, 3]")
        assert result["compliant"] is True
        assert result["fields"]["_value"] == [1, 2, 3]

    def test_empty_input(self) -> None:
        validator = json_format_validator()
        result = validator("")
        assert result["compliant"] is False


# ---------------------------------------------------------------------------
# score_turn
# ---------------------------------------------------------------------------


class TestScoreTurn:
    def test_tool_calls_min_pass(self) -> None:
        tr = TurnResult(turn=1, tool_calls=[{"function": {"name": "write_file", "arguments": "{}"}}])
        spec = {"expect": {"tool_calls_min": 1}}
        assert score_turn(spec, tr) == 1.0

    def test_tool_calls_min_fail(self) -> None:
        tr = TurnResult(turn=1, tool_calls=[])
        spec = {"expect": {"tool_calls_min": 1}}
        assert score_turn(spec, tr) == 0.0

    def test_tool_name_any_match(self) -> None:
        tr = TurnResult(turn=1, tool_calls=[{"function": {"name": "write_file", "arguments": "{}"}}])
        spec = {"expect": {"tool_name_any": ["write_file", "read_file"]}}
        assert score_turn(spec, tr) == 1.0

    def test_tool_name_any_no_match(self) -> None:
        tr = TurnResult(turn=1, tool_calls=[{"function": {"name": "delete_file", "arguments": "{}"}}])
        spec = {"expect": {"tool_name_any": ["write_file", "read_file"]}}
        assert score_turn(spec, tr) == 0.0

    def test_output_mentions_partial(self) -> None:
        tr = TurnResult(turn=1, output_text="I used html and flexbox to build it")
        spec = {"expect": {"output_mentions": ["html", "flexbox", "grid"]}}
        score = score_turn(spec, tr)
        assert abs(score - 2.0 / 3.0) < 0.01  # 2 of 3 keywords found

    def test_output_mentions_all(self) -> None:
        tr = TurnResult(turn=1, output_text="html flexbox grid layout")
        spec = {"expect": {"output_mentions": ["html", "flexbox", "grid"]}}
        assert score_turn(spec, tr) == 1.0

    def test_steering_followed(self) -> None:
        tr = TurnResult(turn=1, output_text="Added size-guide component")
        spec = {"expect": {"steering_followed": True, "steering_keywords": ["size-guide"]}}
        score = score_turn(spec, tr)
        assert score == 1.0
        assert tr.steering_followed is True

    def test_steering_not_followed(self) -> None:
        tr = TurnResult(turn=1, output_text="I built a basic page")
        spec = {"expect": {"steering_followed": True, "steering_keywords": ["size-guide"]}}
        score = score_turn(spec, tr)
        assert score == 0.0
        assert tr.steering_followed is False

    def test_no_expect_block(self) -> None:
        tr = TurnResult(turn=1, output_text="hello")
        spec = {}
        assert score_turn(spec, tr) == 1.0

    def test_combined_expectations(self) -> None:
        tr = TurnResult(
            turn=1,
            tool_calls=[{"function": {"name": "write_file", "arguments": "{}"}}],
            output_text="Created html file with flexbox grid",
        )
        spec = {
            "expect": {
                "tool_calls_min": 1,
                "tool_name_any": ["write_file"],
                "output_mentions": ["html", "flexbox", "grid"],
            }
        }
        assert score_turn(spec, tr) == 1.0


# ---------------------------------------------------------------------------
# useful_ratio
# ---------------------------------------------------------------------------


class TestUsefulRatio:
    def test_text_only_useful(self) -> None:
        text = "STATUS:ok\nFILES_CREATED:index.html"
        ratio = useful_ratio(text)
        # Ratio is slightly < 1.0 because total includes newline chars but
        # per-line sum does not. All lines are useful (no filler).
        assert ratio > 0.95

    def test_text_with_filler(self) -> None:
        text = "Sure, I'll help you!\nSTATUS:ok\nHere is the result"
        ratio = useful_ratio(text)
        assert 0.0 < ratio < 1.0

    def test_tool_calls_only(self) -> None:
        tool_calls = [{"function": {"name": "write_file", "arguments": '{"path":"a.txt","content":"hello"}'}}]
        ratio = useful_ratio("", tool_calls)
        assert ratio == 1.0

    def test_mixed_text_and_tools(self) -> None:
        tool_calls = [{"function": {"name": "write_file", "arguments": '{"path":"a.txt"}'}}]
        text = "Sure, let me help\nDone building"
        ratio = useful_ratio(text, tool_calls)
        assert 0.0 < ratio <= 1.0

    def test_empty_everything(self) -> None:
        assert useful_ratio("", []) == 0.0
        assert useful_ratio("") == 0.0

    def test_only_filler(self) -> None:
        text = "Sure thing!\nLet me help you with that.\nCertainly!"
        ratio = useful_ratio(text)
        assert ratio == 0.0

    def test_whitespace_only(self) -> None:
        assert useful_ratio("   \n  \n  ") == 0.0
