"""Orchestration shapes — spec 009 / B1.

The headline borrowable from Fabian Wesner's One-Shot Shop study:
**orchestration architecture beats model choice** (Team Mode 85% vs
Sub-Agents 57% on the same model). To re-derive that finding inside our
own benchmark we vary orchestration shape as a first-class axis.

Canonical vocabulary lives in Axiom §17.1. Pawbench implements three
shapes with distinct execution semantics today and stubs the other two
to fall through to `subagents` until we nail their operational contract:

  flat            — single dispatch, single agent, no parallelism. Baseline.
  subagents       — N agents in parallel, no merge. Pawbench's classic mode.
  scatter-gather  — N agents in parallel, then a synthesis turn that sees
                    every agent's outputs (a merge step). The presence of
                    the merge is what differentiates "Team Mode" from
                    "Sub-Agents" in Fabian's study.
  waves           — currently identical to subagents; reserved for future
                    DAG-aware execution (cluster_tasks-style coloring).
  team-mode       — currently identical to scatter-gather; reserved for
                    real shared-scratchpad coordination.

Pawbench's value-add here is **measurement**, not orchestration product
features. We do not claim to *be* a multi-agent framework; we claim to
score them on a level field.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pawbench.types import AgentResult

# aiohttp + engine are imported lazily inside run_with_shape so this module's
# pure helpers (parse, _build_merge_agent, OrchestrationResult) can be unit
# tested without the network stack.


class OrchestrationShape(str, Enum):
    """Canonical orchestration vocabulary (Axiom §17.1)."""

    FLAT = "flat"
    WAVES = "waves"
    SCATTER_GATHER = "scatter-gather"
    TEAM_MODE = "team-mode"
    SUBAGENTS = "subagents"

    @classmethod
    def parse(cls, value: str) -> "OrchestrationShape":
        try:
            return cls(value.lower())
        except ValueError as e:
            valid = ", ".join(s.value for s in cls)
            raise ValueError(f"unknown orchestration shape '{value}'; valid: {valid}") from e


@dataclass
class OrchestrationResult:
    """Per-shape execution outcome on a single scenario."""

    shape: str
    scenario_id: str
    wall_time_ms: float = 0.0
    agents: list[AgentResult] = field(default_factory=list)
    merge_turn: AgentResult | None = None
    avg_quality: float = 0.0
    total_tokens: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "scenario_id": self.scenario_id,
            "wall_time_ms": round(self.wall_time_ms, 2),
            "avg_quality": round(self.avg_quality, 4),
            "total_tokens": self.total_tokens,
            "agent_count": len(self.agents),
            "had_merge_turn": self.merge_turn is not None,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Shape executors
# ---------------------------------------------------------------------------


async def _run_flat(
    session: Any,
    endpoint: str,
    model: str,
    scenario: dict[str, Any],
    system_prompt: Any,
) -> list[AgentResult]:
    """Sequential execution — agent k+1 starts only after agent k finishes."""
    from pawbench.engine import run_agent  # lazy: avoids aiohttp at import time

    results: list[AgentResult] = []
    tools_schema = scenario["tools_schema"]
    for agent in scenario["agents"]:
        result = await run_agent(session, endpoint, model, agent, tools_schema, system_prompt)
        results.append(result)
    return results


async def _run_parallel(
    session: Any,
    endpoint: str,
    model: str,
    scenario: dict[str, Any],
    system_prompt: Any,
) -> list[AgentResult]:
    """Parallel execution — all agents launched simultaneously, no coordination."""
    import asyncio

    from pawbench.engine import run_agent  # lazy

    tools_schema = scenario["tools_schema"]
    tasks = [run_agent(session, endpoint, model, a, tools_schema, system_prompt) for a in scenario["agents"]]
    raw = await asyncio.gather(*tasks, return_exceptions=True)
    results: list[AgentResult] = []
    for item in raw:
        if isinstance(item, BaseException):
            results.append(AgentResult(agent_id="error", agent_name="error", error=str(item)[:200]))
        else:
            assert isinstance(item, AgentResult)
            results.append(item)
    return results


def _build_merge_agent(scenario: dict[str, Any], parallel_results: list[AgentResult]) -> dict[str, Any]:
    """Synthesize a merge-turn agent that sees every parallel worker's output.

    The merge agent runs one final user turn whose content embeds a compact
    summary of every worker. This is the structural difference between
    `subagents` and `scatter-gather`/`team-mode` — the merge step.
    """
    summaries: list[str] = []
    for ar in parallel_results:
        if ar.error or not ar.turns:
            continue
        last = ar.turns[-1].output_text or "<no output>"
        summaries.append(f"## {ar.agent_name}\n{last[:1500]}")

    merge_content = (
        "You are the integration coordinator for the parallel workers below. "
        "Your job is to verify the work fits together as a coherent system, "
        "flag any integration gaps, and emit a final CACP block summarizing "
        "the merged state. Do NOT rewrite the workers' code — only verify.\n\n"
        + "\n\n".join(summaries)
    )

    return {
        "id": f"{scenario['id']}-merge",
        "name": "Integration Coordinator",
        "turns": [
            {
                "turn": 1,
                "role": "user",
                "content": merge_content,
                "tools": [],
                "expect": {"output_mentions": ["status"]},
                "complexity_tier": "cross_cutting",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_with_shape(
    endpoint: str,
    model: str,
    scenario: dict[str, Any],
    shape: OrchestrationShape,
    system_prompt: str | None = None,
) -> OrchestrationResult:
    """Execute a scenario under a specific orchestration shape."""
    import aiohttp  # lazy

    from pawbench.engine import DEFAULT_SYSTEM_PROMPT, run_agent  # lazy

    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    out = OrchestrationResult(shape=shape.value, scenario_id=scenario["id"])
    wall_start = time.perf_counter()

    try:
        async with aiohttp.ClientSession() as session:
            if shape is OrchestrationShape.FLAT:
                out.agents = await _run_flat(session, endpoint, model, scenario, system_prompt)

            elif shape is OrchestrationShape.SUBAGENTS or shape is OrchestrationShape.WAVES:
                # WAVES currently degenerates to SUBAGENTS — no DAG yet.
                out.agents = await _run_parallel(session, endpoint, model, scenario, system_prompt)

            elif shape is OrchestrationShape.SCATTER_GATHER or shape is OrchestrationShape.TEAM_MODE:
                # Parallel workers + merge turn. The merge turn is the
                # structural differentiator vs SUBAGENTS.
                out.agents = await _run_parallel(session, endpoint, model, scenario, system_prompt)
                merge_agent = _build_merge_agent(scenario, out.agents)
                out.merge_turn = await run_agent(
                    session, endpoint, model, merge_agent,
                    scenario["tools_schema"], system_prompt,
                )
            else:  # pragma: no cover - exhaustive
                out.error = f"unhandled shape: {shape}"
    except Exception as e:  # network/endpoint failure
        out.error = str(e)[:200]

    out.wall_time_ms = (time.perf_counter() - wall_start) * 1000

    valid = [a for a in out.agents if not a.error]
    if valid:
        out.avg_quality = sum(a.avg_quality for a in valid) / len(valid)
    out.total_tokens = sum(a.total_completion_tokens for a in out.agents)
    if out.merge_turn:
        out.total_tokens += out.merge_turn.total_completion_tokens

    return out
