"""Tests for pawbench.types module."""

from __future__ import annotations

from dataclasses import asdict

from pawbench.types import (
    AgentResult,
    BenchmarkReport,
    ConcurrencyResult,
    ModelCard,
    SaturationPoint,
    ScenarioReport,
    TurnResult,
)


class TestTurnResult:
    def test_default_creation(self) -> None:
        tr = TurnResult(turn=1)
        assert tr.turn == 1
        assert tr.role == "assistant"
        assert tr.ttft_ms == 0.0
        assert tr.tool_calls == []
        assert tr.output_text == ""
        assert tr.error == ""

    def test_with_values(self) -> None:
        tr = TurnResult(
            turn=2,
            ttft_ms=100.0,
            e2e_ms=500.0,
            completion_tokens=150,
            decode_tok_s=30.0,
            output_text="STATUS:ok",
            format_compliant=True,
            quality_score=0.9,
        )
        assert tr.turn == 2
        assert tr.ttft_ms == 100.0
        assert tr.quality_score == 0.9

    def test_serialization(self) -> None:
        tr = TurnResult(turn=1, output_text="hello")
        d = asdict(tr)
        assert d["turn"] == 1
        assert d["output_text"] == "hello"
        assert isinstance(d["tool_calls"], list)


class TestAgentResult:
    def test_default_creation(self) -> None:
        ar = AgentResult(agent_id="a1", agent_name="Test Agent")
        assert ar.agent_id == "a1"
        assert ar.turns == []
        assert ar.error == ""

    def test_with_turns(self) -> None:
        t1 = TurnResult(turn=1, completion_tokens=100)
        t2 = TurnResult(turn=2, completion_tokens=200)
        ar = AgentResult(
            agent_id="a1",
            agent_name="Test",
            turns=[t1, t2],
            total_completion_tokens=300,
            avg_quality=0.8,
        )
        assert len(ar.turns) == 2
        assert ar.total_completion_tokens == 300


class TestBenchmarkReport:
    def test_minimal_creation(self) -> None:
        report = BenchmarkReport(
            tag="test",
            timestamp="2026-03-27T00:00:00Z",
            endpoint="http://localhost:8000",
        )
        assert report.tag == "test"
        assert report.scenarios == []
        assert report.dim1_throughput == {}

    def test_full_serialization(self) -> None:
        scenario = ScenarioReport(
            scenario_id="test-1",
            scenario_name="Test Scenario",
            single_tok_s=45.0,
            peak_tok_s=120.0,
            avg_quality=0.85,
        )
        sat = SaturationPoint(concurrency=4, tok_s=120.0, per_agent=30.0, wall_s=2.5, total_tokens=1000)
        card = ModelCard(model_name="test-model")
        report = BenchmarkReport(
            tag="test",
            timestamp="2026-03-27T00:00:00Z",
            endpoint="http://localhost:8000",
            model_card=card,
            scenarios=[scenario],
            saturation_curve=[sat],
            dim1_throughput={"avg_single_tok_s": 45.0},
            dim2_quality={"avg_quality": 0.85},
            dim3_efficiency={"avg_useful_ratio": 0.7},
            dim4_adaptability={"steering_rate": 0.6},
        )
        d = asdict(report)
        assert d["tag"] == "test"
        assert len(d["scenarios"]) == 1
        assert d["scenarios"][0]["scenario_name"] == "Test Scenario"
        assert d["saturation_curve"][0]["tok_s"] == 120.0
        assert d["model_card"]["model_name"] == "test-model"
        assert d["dim1_throughput"]["avg_single_tok_s"] == 45.0

    def test_round_trip_dict(self) -> None:
        """Ensure asdict produces JSON-serializable output."""
        import json

        report = BenchmarkReport(
            tag="rt",
            timestamp="2026-01-01T00:00:00Z",
            endpoint="http://localhost:8000",
        )
        d = asdict(report)
        serialized = json.dumps(d, default=str)
        deserialized = json.loads(serialized)
        assert deserialized["tag"] == "rt"


class TestConcurrencyResult:
    def test_creation(self) -> None:
        cr = ConcurrencyResult(concurrency=4)
        assert cr.concurrency == 4
        assert cr.agents == []
        assert cr.total_tokens == 0


class TestSaturationPoint:
    def test_creation(self) -> None:
        sp = SaturationPoint(concurrency=8, tok_s=200.0, per_agent=25.0, wall_s=3.2, total_tokens=2000)
        assert sp.concurrency == 8
        assert sp.tok_s == 200.0
