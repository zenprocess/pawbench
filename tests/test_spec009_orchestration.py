"""Spec 009 / B1 — orchestration shape tests (no network)."""

from __future__ import annotations

import pytest

from pawbench.orchestration import (
    OrchestrationResult,
    OrchestrationShape,
    _build_merge_agent,
)
from pawbench.types import AgentResult, TurnResult


def test_shape_vocabulary_is_canonical():
    assert {s.value for s in OrchestrationShape} == {
        "flat",
        "waves",
        "scatter-gather",
        "team-mode",
        "subagents",
    }


def test_shape_parse_accepts_canonical():
    assert OrchestrationShape.parse("flat") is OrchestrationShape.FLAT
    assert OrchestrationShape.parse("scatter-gather") is OrchestrationShape.SCATTER_GATHER


def test_shape_parse_rejects_unknown():
    with pytest.raises(ValueError, match="unknown orchestration shape"):
        OrchestrationShape.parse("megamerge")


def _agent_with_text(name: str, text: str) -> AgentResult:
    return AgentResult(
        agent_id=name,
        agent_name=name,
        turns=[TurnResult(turn=1, output_text=text)],
    )


def test_merge_agent_embeds_every_worker_summary():
    scenario = {"id": "S1", "agents": [{"id": "a"}, {"id": "b"}], "tools_schema": []}
    workers = [_agent_with_text("Frontend", "FE-OUTPUT"), _agent_with_text("Backend", "BE-OUTPUT")]
    merge = _build_merge_agent(scenario, workers)
    content = merge["turns"][0]["content"]
    assert "Frontend" in content and "FE-OUTPUT" in content
    assert "Backend" in content and "BE-OUTPUT" in content
    assert merge["turns"][0]["complexity_tier"] == "cross_cutting"


def test_merge_agent_skips_errored_workers():
    scenario = {"id": "S1", "agents": [], "tools_schema": []}
    workers = [
        _agent_with_text("Good", "OK"),
        AgentResult(agent_id="bad", agent_name="Bad", error="boom"),
    ]
    merge = _build_merge_agent(scenario, workers)
    content = merge["turns"][0]["content"]
    assert "OK" in content
    assert "Bad" not in content


def test_merge_agent_id_is_namespaced_to_scenario():
    scenario = {"id": "pawstyle-orchestration-matrix", "agents": [], "tools_schema": []}
    merge = _build_merge_agent(scenario, [])
    assert merge["id"].startswith("pawstyle-orchestration-matrix")


def test_orchestration_result_to_dict_shape():
    out = OrchestrationResult(shape="flat", scenario_id="X", avg_quality=0.8, total_tokens=42)
    d = out.to_dict()
    assert d["shape"] == "flat"
    assert d["avg_quality"] == 0.8
    assert d["had_merge_turn"] is False
