"""Endpoint specifications for sandbox correctness evaluation.

Each scenario defines expected HTTP endpoints that the generated server code
should expose. Specs are defined in Python (not JSON) for type safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EndpointSpec:
    """Expected behavior of a single HTTP endpoint."""

    method: str
    path: str
    expected_status: int
    required_json_keys: list[str] = field(default_factory=list)
    expect_array: bool = False
    description: str = ""


@dataclass
class ScenarioEndpoints:
    """All expected endpoints for a given scenario."""

    scenario_id: str
    server_file: str  # filename pattern to identify the server entry point
    port: int = 8080
    endpoints: list[EndpointSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PawStyle backend endpoint specifications
# ---------------------------------------------------------------------------

PAWSTYLE_BACKEND_ENDPOINTS = ScenarioEndpoints(
    scenario_id="pawstyle-dispatch",
    server_file="server.py",
    port=8080,
    endpoints=[
        EndpointSpec(
            method="GET",
            path="/api/products",
            expected_status=200,
            expect_array=True,
            description="List all products as JSON array",
        ),
        EndpointSpec(
            method="GET",
            path="/api/products/1",
            expected_status=200,
            required_json_keys=["id", "name", "price"],
            description="Get single product by ID",
        ),
        EndpointSpec(
            method="GET",
            path="/api/health",
            expected_status=200,
            required_json_keys=["status"],
            description="Health check endpoint",
        ),
    ],
)


# Registry mapping scenario IDs to their endpoint specs.
ENDPOINT_REGISTRY: dict[str, ScenarioEndpoints] = {
    "pawstyle-dispatch": PAWSTYLE_BACKEND_ENDPOINTS,
}


def get_endpoint_specs(scenario_id: str) -> ScenarioEndpoints | None:
    """Look up endpoint specs for a scenario, returning None if not defined."""
    return ENDPOINT_REGISTRY.get(scenario_id)


def validate_response(
    spec: EndpointSpec,
    status_code: int,
    body: Any,
) -> bool:
    """Check whether a response satisfies an endpoint spec.

    Returns True if the status code matches and required JSON structure is present.
    """
    if status_code != spec.expected_status:
        return False

    if spec.expect_array:
        if not isinstance(body, list):
            return False

    if spec.required_json_keys:
        if not isinstance(body, dict):
            return False
        for key in spec.required_json_keys:
            if key not in body:
                return False

    return True
