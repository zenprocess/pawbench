"""Mock mode for CI — replay recorded API responses or record live ones."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pawbench.types import ModelCard, SaturationPoint


@dataclass
class RecordedResponse:
    """A single recorded API response."""

    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 50.0


@dataclass
class Fixture:
    """A fixture file containing recorded responses for replay."""

    name: str = ""
    description: str = ""
    model: str = "mock-model"
    responses: list[RecordedResponse] = field(default_factory=list)
    model_card: dict[str, Any] = field(default_factory=dict)
    saturation: list[dict[str, Any]] = field(default_factory=list)


def load_fixture(path: Path) -> Fixture:
    """Load a fixture from a JSON file."""
    with open(path) as f:
        raw = json.load(f)

    responses = []
    for r in raw.get("responses", []):
        responses.append(
            RecordedResponse(
                status=r.get("status", 200),
                headers=r.get("headers", {}),
                body=r.get("body", {}),
                latency_ms=r.get("latency_ms", 50.0),
            )
        )

    return Fixture(
        name=raw.get("name", ""),
        description=raw.get("description", ""),
        model=raw.get("model", "mock-model"),
        responses=responses,
        model_card=raw.get("model_card", {}),
        saturation=raw.get("saturation", []),
    )


def save_fixture(fixture: Fixture, path: Path) -> None:
    """Save a fixture to a JSON file."""
    data = {
        "name": fixture.name,
        "description": fixture.description,
        "model": fixture.model,
        "responses": [asdict(r) for r in fixture.responses],
        "model_card": fixture.model_card,
        "saturation": fixture.saturation,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def default_mock_model_card() -> ModelCard:
    """Return a default ModelCard for mock mode (no live endpoint needed)."""
    return ModelCard(
        model_name="mock-model",
        serving={"mode": "mock"},
        model_config={"model_type": "mock", "architectures": ["MockForCausalLM"]},
        tuning={
            "gpu_memory_utilization": "N/A",
            "max_model_len": "N/A",
            "max_num_seqs": "N/A",
            "max_num_batched_tokens": "N/A",
            "kv_cache_dtype": "N/A",
            "speculative_config": "none",
        },
        gpu={"name": "mock", "driver": "N/A", "temp_c": "N/A", "power_w": "N/A"},
        memory={"total_gb": 0, "used_gb": 0},
    )


class MockEndpoint:
    """Serves recorded responses from a fixture, cycling through them.

    Can also record live responses for later replay.
    """

    def __init__(self, fixture: Fixture | None = None) -> None:
        self._fixture = fixture or Fixture()
        self._index = 0
        self._recorded: list[RecordedResponse] = []

    @property
    def model(self) -> str:
        return self._fixture.model

    @property
    def fixture(self) -> Fixture:
        return self._fixture

    @property
    def recorded_responses(self) -> list[RecordedResponse]:
        return list(self._recorded)

    def next_response(self) -> RecordedResponse:
        """Return the next recorded response, cycling if exhausted."""
        if not self._fixture.responses:
            return RecordedResponse(
                status=200,
                body=_default_chat_completion(self._fixture.model),
            )
        resp = self._fixture.responses[self._index % len(self._fixture.responses)]
        self._index += 1
        return resp

    def record_response(
        self,
        status: int,
        headers: dict[str, str],
        body: dict[str, Any],
        latency_ms: float,
    ) -> None:
        """Record a live API response for later fixture generation."""
        rec = RecordedResponse(
            status=status,
            headers=headers,
            body=body,
            latency_ms=latency_ms,
        )
        self._recorded.append(rec)

    def build_fixture(self, name: str = "", description: str = "", model: str = "") -> Fixture:
        """Build a Fixture from all recorded responses."""
        return Fixture(
            name=name or self._fixture.name or "recorded",
            description=description or self._fixture.description,
            model=model or self._fixture.model,
            responses=list(self._recorded),
            model_card=asdict(default_mock_model_card()),
        )

    def get_saturation_points(self) -> list[SaturationPoint]:
        """Return mock saturation points from the fixture."""
        points = []
        for entry in self._fixture.saturation:
            points.append(
                SaturationPoint(
                    concurrency=entry.get("concurrency", 1),
                    tok_s=entry.get("tok_s", 100.0),
                    per_agent=entry.get("per_agent", 100.0),
                    wall_s=entry.get("wall_s", 1.0),
                    total_tokens=entry.get("total_tokens", 512),
                )
            )
        if not points:
            points = [
                SaturationPoint(concurrency=1, tok_s=100.0, per_agent=100.0, wall_s=1.0, total_tokens=512),
                SaturationPoint(concurrency=2, tok_s=180.0, per_agent=90.0, wall_s=1.1, total_tokens=1024),
                SaturationPoint(concurrency=4, tok_s=300.0, per_agent=75.0, wall_s=1.4, total_tokens=2048),
                SaturationPoint(concurrency=8, tok_s=400.0, per_agent=50.0, wall_s=2.0, total_tokens=4096),
            ]
        return points


def mock_chat_completion_response(
    model: str = "mock-model",
    content: str = (
        "STATUS:ok\nFILES_CREATED:main.py\nFILES_MODIFIED:\nTESTS:pass:1\nBUILD:pass\nLEARNED:Implemented quicksort"
    ),
    completion_tokens: int = 42,
    prompt_tokens: int = 100,
) -> dict[str, Any]:
    """Build a mock non-streaming chat completion response body."""
    return {
        "id": "mock-cmpl-001",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _default_chat_completion(model: str = "mock-model") -> dict[str, Any]:
    """Default chat completion body when no fixture responses exist."""
    return mock_chat_completion_response(model=model)


def load_fixtures_dir(directory: Path) -> dict[str, Fixture]:
    """Load all fixture JSON files from a directory."""
    fixtures: dict[str, Fixture] = {}
    if not directory.is_dir():
        return fixtures
    for p in sorted(directory.glob("*.json")):
        try:
            fixture = load_fixture(p)
            fixtures[p.stem] = fixture
        except (json.JSONDecodeError, KeyError):
            continue
    return fixtures
