"""Spec 009 — CLI helper tests (context tier, scenario load, scoring/by-tier)."""
from __future__ import annotations

import json
from pathlib import Path

from pawbench.context_tier import apply_context_tier, strip_code_from_content
from pawbench.scoring import quality_by_tier
from pawbench.types import AgentResult, TurnResult

# Test-local aliases preserve the original test names
_strip_code_from_content = strip_code_from_content
_apply_context_tier = apply_context_tier

SCENARIOS_DIR = Path(__file__).resolve().parent.parent / "src" / "pawbench" / "scenarios"


def test_strip_code_removes_fenced_blocks():
    text = "before\n```python\nprint('x')\n```\nafter"
    out = _strip_code_from_content(text)
    assert "print" not in out
    assert "before" in out and "after" in out
    assert "manifest-only mode" in out


def test_strip_code_removes_long_inline_literals():
    text = "do this {" + "a" * 250 + "} now"
    out = _strip_code_from_content(text)
    assert "a" * 250 not in out
    assert "manifest-only mode" in out


def test_apply_context_tier_standard_is_passthrough():
    s = {"agents": [{"turns": [{"content": "```code```"}]}]}
    out = _apply_context_tier(s, "standard")
    assert out is s  # identity preserved


def test_apply_context_tier_manifest_only_strips_all_turns():
    s = {
        "agents": [
            {"turns": [{"content": "```py\nx=1\n```"}, {"content": "plain text"}]},
            {"turns": [{"content": "```js\ny=2\n```"}]},
        ]
    }
    out = _apply_context_tier(s, "manifest-only")
    assert s["agents"][0]["turns"][0]["content"].startswith("```")  # original untouched
    flat = " ".join(t["content"] for a in out["agents"] for t in a["turns"])
    assert "x=1" not in flat and "y=2" not in flat
    assert "plain text" in flat


def test_orchestration_matrix_scenario_validates():
    p = SCENARIOS_DIR / "pawstyle-orchestration-matrix.json"
    obj = json.loads(p.read_text())
    assert obj["complexity_tier"] == "cross_cutting"
    tiers = {a["complexity_tier"] for a in obj["agents"]}
    assert tiers == {"display", "crud", "transactional", "cross_cutting"}
    assert len(obj["agents"]) == 4


def test_existing_pawstyle_scenarios_carry_tier_tags():
    for name in ("pawstyle.json", "pawstyle-independent.json", "pawstyle-nudge.json"):
        obj = json.loads((SCENARIOS_DIR / name).read_text())
        assert obj.get("complexity_tier"), f"{name} missing complexity_tier"


def test_quality_by_tier_aggregates_per_tier():
    scenario = {
        "id": "S",
        "agents": [
            {
                "id": "agent-1",
                "complexity_tier": "display",
                "turns": [{"complexity_tier": "display"}, {"complexity_tier": "crud"}],
            }
        ],
    }
    ar = AgentResult(
        agent_id="agent-1",
        agent_name="Agent 1",
        turns=[
            TurnResult(turn=1, quality_score=0.9),
            TurnResult(turn=2, quality_score=0.5),
        ],
    )
    out = quality_by_tier([ar], scenario)
    assert out == {"display": 0.9, "crud": 0.5}


def test_quality_by_tier_handles_parallel_dispatch_id_suffix():
    scenario = {
        "id": "S",
        "agents": [{"id": "ts-fullstack", "complexity_tier": "crud", "turns": [{}]}],
    }
    ar = AgentResult(
        agent_id="ts-fullstack-3",  # parallel-dispatch suffix
        agent_name="Agent",
        turns=[TurnResult(turn=1, quality_score=0.8)],
    )
    out = quality_by_tier([ar], scenario)
    assert out == {"crud": 0.8}
