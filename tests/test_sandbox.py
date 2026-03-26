"""Tests for sandbox execution-based correctness scoring."""

from __future__ import annotations

import json
from unittest.mock import patch

from pawbench.endpoint_spec import (
    EndpointSpec,
    get_endpoint_specs,
    validate_response,
)
from pawbench.sandbox import (
    SandboxEvaluator,
    SandboxResult,
    evaluate_agent,
    extract_files,
)
from pawbench.types import AgentResult, TurnResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    agent_id: str = "test-agent",
    files: dict[str, str] | None = None,
) -> AgentResult:
    """Build an AgentResult with write_file tool calls for given files."""
    turns = []
    if files:
        tool_calls = []
        for path, content in files.items():
            tool_calls.append(
                {
                    "id": f"call_{path}",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({"path": path, "content": content}),
                    },
                }
            )
        turns.append(
            TurnResult(
                turn=1,
                tool_calls=tool_calls,
                output_text="Files written.",
            )
        )
    return AgentResult(agent_id=agent_id, agent_name="Test Agent", turns=turns)


# ---------------------------------------------------------------------------
# Tests — file extraction
# ---------------------------------------------------------------------------


class TestExtractFiles:
    def test_extracts_single_file(self):
        agent = _make_agent(files={"server.py": "print('hello')"})
        files = extract_files(agent)
        assert files == {"server.py": "print('hello')"}

    def test_extracts_multiple_files(self):
        agent = _make_agent(
            files={
                "server.py": "import http.server",
                "utils.py": "def helper(): pass",
            }
        )
        files = extract_files(agent)
        assert len(files) == 2
        assert "server.py" in files
        assert "utils.py" in files

    def test_ignores_non_write_file_calls(self):
        agent = AgentResult(
            agent_id="a",
            agent_name="A",
            turns=[
                TurnResult(
                    turn=1,
                    tool_calls=[
                        {
                            "id": "call_1",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": "foo.py"}),
                            },
                        }
                    ],
                )
            ],
        )
        assert extract_files(agent) == {}

    def test_handles_malformed_arguments(self):
        agent = AgentResult(
            agent_id="a",
            agent_name="A",
            turns=[
                TurnResult(
                    turn=1,
                    tool_calls=[
                        {
                            "id": "call_1",
                            "function": {
                                "name": "write_file",
                                "arguments": "not valid json {{{",
                            },
                        }
                    ],
                )
            ],
        )
        assert extract_files(agent) == {}

    def test_extracts_across_multiple_turns(self):
        agent = AgentResult(
            agent_id="a",
            agent_name="A",
            turns=[
                TurnResult(
                    turn=1,
                    tool_calls=[
                        {
                            "id": "call_1",
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps({"path": "a.py", "content": "aaa"}),
                            },
                        }
                    ],
                ),
                TurnResult(
                    turn=2,
                    tool_calls=[
                        {
                            "id": "call_2",
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps({"path": "b.py", "content": "bbb"}),
                            },
                        }
                    ],
                ),
            ],
        )
        files = extract_files(agent)
        assert len(files) == 2
        assert files["a.py"] == "aaa"
        assert files["b.py"] == "bbb"


# ---------------------------------------------------------------------------
# Tests — endpoint spec validation
# ---------------------------------------------------------------------------


class TestValidateResponse:
    def test_status_match_no_json_keys(self):
        spec = EndpointSpec(method="GET", path="/health", expected_status=200)
        assert validate_response(spec, 200, {"status": "ok"}) is True

    def test_status_mismatch(self):
        spec = EndpointSpec(method="GET", path="/health", expected_status=200)
        assert validate_response(spec, 500, {}) is False

    def test_required_keys_present(self):
        spec = EndpointSpec(
            method="GET",
            path="/api/products/1",
            expected_status=200,
            required_json_keys=["id", "name", "price"],
        )
        assert validate_response(spec, 200, {"id": 1, "name": "X", "price": 10.0}) is True

    def test_required_keys_missing(self):
        spec = EndpointSpec(
            method="GET",
            path="/api/products/1",
            expected_status=200,
            required_json_keys=["id", "name", "price"],
        )
        assert validate_response(spec, 200, {"id": 1}) is False

    def test_expect_array_with_list(self):
        spec = EndpointSpec(
            method="GET",
            path="/api/products",
            expected_status=200,
            expect_array=True,
        )
        assert validate_response(spec, 200, [{"id": 1}]) is True

    def test_expect_array_with_dict(self):
        spec = EndpointSpec(
            method="GET",
            path="/api/products",
            expected_status=200,
            expect_array=True,
        )
        assert validate_response(spec, 200, {"products": []}) is False

    def test_required_keys_body_not_dict(self):
        spec = EndpointSpec(
            method="GET",
            path="/api/products/1",
            expected_status=200,
            required_json_keys=["id"],
        )
        assert validate_response(spec, 200, [1, 2, 3]) is False


# ---------------------------------------------------------------------------
# Tests — scoring calculation
# ---------------------------------------------------------------------------


class TestScoringCalculation:
    def test_all_pass(self):
        result = SandboxResult(
            agent_id="a",
            endpoints_tested=3,
            endpoints_passed=3,
            score=1.0,
        )
        assert result.score == 1.0

    def test_partial_pass(self):
        result = SandboxResult(
            agent_id="a",
            endpoints_tested=3,
            endpoints_passed=1,
            score=1 / 3,
        )
        assert abs(result.score - 1 / 3) < 1e-9

    def test_none_pass(self):
        result = SandboxResult(
            agent_id="a",
            endpoints_tested=3,
            endpoints_passed=0,
            score=0.0,
        )
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# Tests — endpoint registry
# ---------------------------------------------------------------------------


class TestEndpointRegistry:
    def test_pawstyle_exists(self):
        spec = get_endpoint_specs("pawstyle-dispatch")
        assert spec is not None
        assert len(spec.endpoints) == 3

    def test_unknown_returns_none(self):
        assert get_endpoint_specs("nonexistent-scenario") is None


# ---------------------------------------------------------------------------
# Tests — evaluate_agent edge cases
# ---------------------------------------------------------------------------


class TestEvaluateAgent:
    def test_no_endpoint_specs(self):
        agent = _make_agent(files={"server.py": "pass"})
        result = evaluate_agent(agent, "nonexistent")
        assert result.score == 0.0
        assert "No endpoint specs" in result.error

    def test_no_files_extracted(self):
        agent = AgentResult(agent_id="a", agent_name="A", turns=[])
        result = evaluate_agent(agent, "pawstyle-dispatch")
        assert result.score == 0.0
        assert "No files extracted" in result.error

    def test_no_server_file_found(self):
        agent = _make_agent(files={"readme.txt": "hello"})
        result = evaluate_agent(agent, "pawstyle-dispatch")
        assert result.score == 0.0
        assert "No server file" in result.error


# ---------------------------------------------------------------------------
# Tests — SandboxEvaluator
# ---------------------------------------------------------------------------


class TestSandboxEvaluator:
    def test_average_score_empty(self):
        evaluator = SandboxEvaluator("pawstyle-dispatch")
        assert evaluator.average_score([]) == 0.0

    @patch("pawbench.sandbox.evaluate_agent")
    def test_average_score_mocked(self, mock_eval):
        mock_eval.side_effect = [
            SandboxResult(agent_id="a", score=1.0, endpoints_tested=3, endpoints_passed=3),
            SandboxResult(agent_id="b", score=0.5, endpoints_tested=2, endpoints_passed=1),
        ]
        evaluator = SandboxEvaluator("pawstyle-dispatch")
        agents = [_make_agent("a"), _make_agent("b")]
        avg = evaluator.average_score(agents)
        assert abs(avg - 0.75) < 1e-9
