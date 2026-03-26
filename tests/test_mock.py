"""Tests for pawbench.mock module."""

from __future__ import annotations

from pathlib import Path

from pawbench.mock import (
    Fixture,
    MockEndpoint,
    RecordedResponse,
    default_mock_model_card,
    load_fixture,
    load_fixtures_dir,
    mock_chat_completion_response,
    save_fixture,
)
from pawbench.types import SaturationPoint

FIXTURES_DIR = Path(__file__).parent.parent / "src" / "pawbench" / "fixtures"


class TestLoadFixture:
    def test_load_sample_saturation(self) -> None:
        fixture = load_fixture(FIXTURES_DIR / "sample_saturation.json")
        assert fixture.name == "sample_saturation"
        assert fixture.model == "mock-model-fp8"
        assert len(fixture.responses) == 3

    def test_load_fixture_responses_have_usage(self) -> None:
        fixture = load_fixture(FIXTURES_DIR / "sample_saturation.json")
        for resp in fixture.responses:
            assert resp.status == 200
            usage = resp.body["usage"]
            assert usage["completion_tokens"] > 0
            assert usage["prompt_tokens"] > 0

    def test_load_fixture_saturation_points(self) -> None:
        fixture = load_fixture(FIXTURES_DIR / "sample_saturation.json")
        assert len(fixture.saturation) == 4
        assert fixture.saturation[0]["concurrency"] == 1
        assert fixture.saturation[-1]["concurrency"] == 8


class TestSaveFixture:
    def test_roundtrip(self, tmp_path: Path) -> None:
        original = Fixture(
            name="test-roundtrip",
            description="roundtrip test",
            model="test-model",
            responses=[
                RecordedResponse(
                    status=200,
                    headers={"content-type": "application/json"},
                    body=mock_chat_completion_response("test-model"),
                    latency_ms=33.3,
                )
            ],
            saturation=[{"concurrency": 1, "tok_s": 50.0, "per_agent": 50.0, "wall_s": 1.0, "total_tokens": 256}],
        )
        out = tmp_path / "roundtrip.json"
        save_fixture(original, out)
        loaded = load_fixture(out)
        assert loaded.name == "test-roundtrip"
        assert loaded.model == "test-model"
        assert len(loaded.responses) == 1
        assert loaded.responses[0].latency_ms == 33.3
        assert loaded.saturation[0]["tok_s"] == 50.0

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "fixture.json"
        save_fixture(Fixture(name="deep"), deep)
        assert deep.exists()


class TestMockEndpoint:
    def test_next_response_cycles(self) -> None:
        fixture = Fixture(
            responses=[
                RecordedResponse(body={"id": "r1"}),
                RecordedResponse(body={"id": "r2"}),
            ]
        )
        ep = MockEndpoint(fixture)
        assert ep.next_response().body["id"] == "r1"
        assert ep.next_response().body["id"] == "r2"
        # Cycles back to first
        assert ep.next_response().body["id"] == "r1"

    def test_next_response_default_when_empty(self) -> None:
        ep = MockEndpoint(Fixture(model="empty-model"))
        resp = ep.next_response()
        assert resp.status == 200
        assert resp.body["model"] == "empty-model"
        assert "choices" in resp.body

    def test_record_and_build_fixture(self) -> None:
        ep = MockEndpoint()
        ep.record_response(
            status=200,
            headers={"content-type": "application/json"},
            body=mock_chat_completion_response(),
            latency_ms=42.0,
        )
        ep.record_response(
            status=200,
            headers={},
            body=mock_chat_completion_response(content="second"),
            latency_ms=38.0,
        )
        fixture = ep.build_fixture(name="test-record", model="rec-model")
        assert fixture.name == "test-record"
        assert fixture.model == "rec-model"
        assert len(fixture.responses) == 2
        assert fixture.responses[0].latency_ms == 42.0

    def test_recorded_responses_list(self) -> None:
        ep = MockEndpoint()
        assert ep.recorded_responses == []
        ep.record_response(status=200, headers={}, body={}, latency_ms=10.0)
        assert len(ep.recorded_responses) == 1

    def test_get_saturation_points_from_fixture(self) -> None:
        fixture = load_fixture(FIXTURES_DIR / "sample_saturation.json")
        ep = MockEndpoint(fixture)
        points = ep.get_saturation_points()
        assert len(points) == 4
        assert all(isinstance(p, SaturationPoint) for p in points)
        assert points[0].concurrency == 1
        assert points[-1].concurrency == 8

    def test_get_saturation_points_defaults(self) -> None:
        ep = MockEndpoint(Fixture())
        points = ep.get_saturation_points()
        assert len(points) >= 1
        assert all(isinstance(p, SaturationPoint) for p in points)


class TestDefaultMockModelCard:
    def test_has_required_fields(self) -> None:
        card = default_mock_model_card()
        assert card.model_name == "mock-model"
        assert card.serving["mode"] == "mock"
        assert "architectures" in card.model_config


class TestLoadFixturesDir:
    def test_load_builtin_fixtures(self) -> None:
        fixtures = load_fixtures_dir(FIXTURES_DIR)
        assert "sample_saturation" in fixtures
        assert fixtures["sample_saturation"].model == "mock-model-fp8"

    def test_load_empty_dir(self, tmp_path: Path) -> None:
        fixtures = load_fixtures_dir(tmp_path)
        assert fixtures == {}

    def test_load_nonexistent_dir(self, tmp_path: Path) -> None:
        fixtures = load_fixtures_dir(tmp_path / "nope")
        assert fixtures == {}


class TestMockChatCompletionResponse:
    def test_default_response_structure(self) -> None:
        resp = mock_chat_completion_response()
        assert resp["model"] == "mock-model"
        assert len(resp["choices"]) == 1
        assert resp["usage"]["completion_tokens"] == 42
        assert resp["choices"][0]["message"]["role"] == "assistant"

    def test_custom_params(self) -> None:
        resp = mock_chat_completion_response(
            model="custom",
            content="hello",
            completion_tokens=10,
            prompt_tokens=5,
        )
        assert resp["model"] == "custom"
        assert resp["choices"][0]["message"]["content"] == "hello"
        assert resp["usage"]["total_tokens"] == 15
