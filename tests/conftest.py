"""Shared fixtures for PawBench tests."""
from __future__ import annotations

import pytest

from pawbench.types import TurnResult


@pytest.fixture
def mock_endpoint() -> str:
    return "http://localhost:9999"


@pytest.fixture
def sample_turn_result() -> TurnResult:
    return TurnResult(
        turn=1,
        ttft_ms=120.0,
        e2e_ms=850.0,
        prompt_tokens=100,
        completion_tokens=200,
        decode_tok_s=45.5,
        tool_calls=[
            {
                "id": "call_1",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "index.html", "content": "<html></html>"}',
                },
            }
        ],
        tool_call_correct=True,
        output_text=(
            "STATUS:ok\nFILES_CREATED:index.html\nFILES_MODIFIED:\nTESTS:pass:0\nBUILD:pass\nLEARNED:built the page"
        ),
        format_compliant=True,
        quality_score=0.85,
    )


@pytest.fixture
def sample_scenario() -> dict:
    return {
        "id": "test-scenario",
        "name": "Test Scenario",
        "agents": [
            {
                "id": "test-agent",
                "name": "Test Agent",
                "turns": [
                    {
                        "turn": 1,
                        "role": "user",
                        "content": "Write hello world",
                        "tools": ["write_file"],
                        "expect": {
                            "tool_calls_min": 1,
                            "tool_name_any": ["write_file"],
                            "output_mentions": ["hello"],
                        },
                    }
                ],
            }
        ],
        "tools_schema": [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            }
        ],
    }
