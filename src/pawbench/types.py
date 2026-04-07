"""Data types for benchmark results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnResult:
    """Result from a single conversation turn."""

    turn: int
    role: str = "assistant"
    ttft_ms: float = 0.0
    e2e_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    decode_tok_s: float = 0.0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_correct: bool = False
    output_text: str = ""
    format_compliant: bool = False
    format_fields: dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    steering_followed: bool = False
    error: str = ""
    # Spec 009 / B2 — complexity stratification
    complexity_tier: str | None = None


@dataclass
class AgentResult:
    """Result from a full multi-turn agent conversation."""

    agent_id: str
    agent_name: str
    turns: list[TurnResult] = field(default_factory=list)
    total_e2e_ms: float = 0.0
    total_completion_tokens: int = 0
    total_prompt_tokens: int = 0
    avg_decode_tok_s: float = 0.0
    avg_quality: float = 0.0
    format_compliance_rate: float = 0.0
    steering_success: bool = False
    error: str = ""


@dataclass
class ConcurrencyResult:
    """Result from one concurrency level."""

    concurrency: int
    wall_time_ms: float = 0.0
    total_tokens: int = 0
    aggregate_tok_s: float = 0.0
    per_agent_tok_s: float = 0.0
    agents: list[AgentResult] = field(default_factory=list)


@dataclass
class SaturationPoint:
    """One point on the throughput saturation curve."""

    concurrency: int
    tok_s: float
    per_agent: float
    wall_s: float
    total_tokens: int


@dataclass
class ScenarioReport:
    """Results from one scenario across all concurrency levels."""

    scenario_id: str
    scenario_name: str
    single_tok_s: float = 0.0
    peak_tok_s: float = 0.0
    peak_concurrency: int = 0
    avg_ttft_ms: float = 0.0
    avg_quality: float = 0.0
    format_compliance_rate: float = 0.0
    tool_accuracy: float = 0.0
    useful_ratio: float = 0.0
    total_tokens: int = 0
    tokens_per_turn: float = 0.0
    steering_rate: float = 0.0
    nudge_response_quality: float = 0.0
    sandbox_score: float = 0.0


@dataclass
class ModelCard:
    """Captured model + serving config for reproducibility."""

    model_name: str = "unknown"
    serving: dict[str, Any] = field(default_factory=dict)
    model_config: dict[str, Any] = field(default_factory=dict)
    tuning: dict[str, Any] = field(default_factory=dict)
    gpu: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Full 4-dimensional benchmark report."""

    tag: str
    timestamp: str
    endpoint: str
    model_card: ModelCard = field(default_factory=ModelCard)
    runs: int = 1
    scenarios: list[ScenarioReport] = field(default_factory=list)
    dim1_throughput: dict[str, Any] = field(default_factory=dict)
    dim2_quality: dict[str, Any] = field(default_factory=dict)
    dim3_efficiency: dict[str, Any] = field(default_factory=dict)
    dim4_adaptability: dict[str, Any] = field(default_factory=dict)
    saturation_curve: list[SaturationPoint] = field(default_factory=list)
    concurrency_curve: list[dict[str, Any]] = field(default_factory=list)
    sandbox_score: float = 0.0
    server_metrics: dict[str, Any] = field(default_factory=dict)
    # Spec 009 — orchestration × complexity matrix additions
    dim5_artifact_quality: dict[str, Any] = field(default_factory=dict)
    quality_by_tier: dict[str, float] = field(default_factory=dict)
    orchestration_results: list[dict[str, Any]] = field(default_factory=list)
    orchestration_dqs_spread: float = 0.0
    ablation: dict[str, Any] = field(default_factory=dict)
    dqs: dict[str, Any] = field(default_factory=dict)
    spec_version: str = "spec-009"
