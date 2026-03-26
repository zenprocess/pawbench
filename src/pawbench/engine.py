"""Benchmark execution engine — streaming API calls, multi-turn, parallel dispatch."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp

from typing import Callable
from pawbench.scoring import score_turn, useful_ratio
from pawbench.types import AgentResult, ConcurrencyResult, SaturationPoint, TurnResult

DEFAULT_SYSTEM_PROMPT = """You are a coding agent. Use tools to build what is asked.

MANDATORY OUTPUT FORMAT — violation causes automatic rejection:
When you are done using tools, your response MUST be:
STATUS:ok
FILES_CREATED:file1,file2
FILES_MODIFIED:
TESTS:pass:0
BUILD:pass
LEARNED:one sentence

Rules:
- First line MUST start with STATUS:
- Do NOT write any text before STATUS:
- Do NOT summarize your work
- Do NOT explain what you did
- ONLY the CACP fields, nothing else"""


async def execute_turn(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    timeout: int = 300,
) -> TurnResult:
    """Execute a single streaming API call and measure throughput."""
    result = TurnResult(turn=0)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 4096,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools

    first_token_time = None
    chunks_text: list[str] = []
    tool_calls_acc: dict[int, dict[str, Any]] = {}

    try:
        start = time.perf_counter()
        async with session.post(
            f"{endpoint}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"]
                    usage = chunk.get("usage")

                    if first_token_time is None and (delta.get("content") or delta.get("tool_calls")):
                        first_token_time = time.perf_counter()

                    if delta.get("content"):
                        chunks_text.append(delta["content"])

                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                            if tc.get("id"):
                                tool_calls_acc[idx]["id"] = tc["id"]
                            fn = tc.get("function", {})
                            if fn.get("name"):
                                tool_calls_acc[idx]["function"]["name"] = fn["name"]
                            if fn.get("arguments"):
                                tool_calls_acc[idx]["function"]["arguments"] += fn["arguments"]

                    if usage:
                        result.prompt_tokens = usage.get("prompt_tokens", 0)
                        result.completion_tokens = usage.get("completion_tokens", 0)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        end = time.perf_counter()
        result.e2e_ms = (end - start) * 1000
        if first_token_time:
            result.ttft_ms = (first_token_time - start) * 1000

        result.output_text = "".join(chunks_text)
        result.tool_calls = list(tool_calls_acc.values())

        # Estimate tokens if server didn't provide usage
        if result.completion_tokens == 0:
            text_tokens = len(result.output_text) // 4
            tc_tokens = sum(len(tc.get("function", {}).get("arguments", "")) // 4 for tc in result.tool_calls)
            result.completion_tokens = max(text_tokens + tc_tokens, 1)

        decode_time = (result.e2e_ms - result.ttft_ms) / 1000 if result.ttft_ms > 0 else result.e2e_ms / 1000
        if decode_time > 0 and result.completion_tokens > 0:
            result.decode_tok_s = result.completion_tokens / decode_time

    except Exception as e:
        result.error = str(e)[:200]

    return result


async def run_agent(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    agent: dict[str, Any],
    tools_schema: list[dict[str, Any]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    format_validator: Callable[[str], dict[str, Any]] | None = None,
) -> AgentResult:
    """Execute a full multi-turn agent conversation."""
    ar = AgentResult(agent_id=agent["id"], agent_name=agent["name"])
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    agent_start = time.perf_counter()

    for turn_spec in agent["turns"]:
        turn_num = turn_spec["turn"]

        # Inject tool results from previous turn
        if "inject_tool_results" in turn_spec:
            for tr in turn_spec["inject_tool_results"]:
                tool_call_id = tr.get("tool_call_id", "auto")
                if tool_call_id == "auto" and ar.turns:
                    last_turn = ar.turns[-1]
                    if last_turn.tool_calls:
                        tool_call_id = last_turn.tool_calls[0].get("id", f"call_{turn_num}")
                    else:
                        tool_call_id = f"call_{turn_num}"

                if ar.turns and ar.turns[-1].tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": ar.turns[-1].output_text or None,
                        "tool_calls": [
                            {"id": tc.get("id", tool_call_id), "type": "function", "function": tc["function"]}
                            for tc in ar.turns[-1].tool_calls
                        ],
                    })
                    messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": tr["content"]})

        if turn_spec.get("content"):
            messages.append({"role": "user", "content": turn_spec["content"]})

        turn_tools = [ts for ts in tools_schema if ts["function"]["name"] in turn_spec.get("tools", [])]

        turn_result = await execute_turn(session, endpoint, model, messages, turn_tools)
        turn_result.turn = turn_num
        turn_result.quality_score = score_turn(turn_spec, turn_result)

        if turn_num == len(agent["turns"]) and format_validator:
            parsed = format_validator(turn_result.output_text)
            turn_result.format_compliant = parsed.get("compliant", False)
            turn_result.format_fields = parsed

        ar.turns.append(turn_result)
        ar.total_completion_tokens += turn_result.completion_tokens
        ar.total_prompt_tokens += turn_result.prompt_tokens

    ar.total_e2e_ms = (time.perf_counter() - agent_start) * 1000
    valid_turns = [t for t in ar.turns if t.decode_tok_s > 0]
    ar.avg_decode_tok_s = sum(t.decode_tok_s for t in valid_turns) / max(len(valid_turns), 1)
    ar.avg_quality = sum(t.quality_score for t in ar.turns) / max(len(ar.turns), 1)
    ar.steering_success = any(t.steering_followed for t in ar.turns)

    return ar


async def run_parallel_dispatch(
    endpoint: str,
    model: str,
    scenario: dict[str, Any],
    concurrency: int,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> ConcurrencyResult:
    """Run N copies of the scenario agents in parallel."""
    cr = ConcurrencyResult(concurrency=concurrency)
    agents = scenario["agents"]
    tools_schema = scenario["tools_schema"]

    dispatch_agents = []
    for i in range(concurrency):
        agent = agents[i % len(agents)].copy()
        agent["id"] = f"{agent['id']}-{i}"
        dispatch_agents.append(agent)

    wall_start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [run_agent(session, endpoint, model, a, tools_schema, system_prompt) for a in dispatch_agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    cr.wall_time_ms = (time.perf_counter() - wall_start) * 1000

    for r in results:
        if isinstance(r, Exception):
            cr.agents.append(AgentResult(agent_id="error", agent_name="error", error=str(r)[:200]))
        else:
            cr.agents.append(r)
            cr.total_tokens += r.total_completion_tokens

    if cr.wall_time_ms > 0:
        cr.aggregate_tok_s = cr.total_tokens / (cr.wall_time_ms / 1000)
    valid = [a for a in cr.agents if not a.error]
    cr.per_agent_tok_s = sum(a.avg_decode_tok_s for a in valid) / max(len(valid), 1)

    return cr


async def run_saturation_test(
    endpoint: str,
    model: str,
    concurrency_levels: list[int],
    max_tokens: int = 512,
) -> list[SaturationPoint]:
    """Raw parallel throughput test — fire identical requests, measure aggregate tok/s."""
    prompt = "Write a Python function that implements quicksort with type hints. Just the code, nothing else."
    results: list[SaturationPoint] = []

    async def _one_request(session: aiohttp.ClientSession) -> dict[str, Any]:
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0, "max_tokens": max_tokens}
        start = time.perf_counter()
        async with session.post(f"{endpoint}/v1/chat/completions", json=payload,
                                timeout=aiohttp.ClientTimeout(total=120)) as r:
            data = await r.json()
            return {"comp": data.get("usage", {}).get("completion_tokens", 0),
                    "e2e": time.perf_counter() - start}

    for n in concurrency_levels:
        async with aiohttp.ClientSession() as session:
            wall_start = time.perf_counter()
            raw = await asyncio.gather(*[_one_request(session) for _ in range(n)], return_exceptions=True)
            wall_s = time.perf_counter() - wall_start
            valid = [r for r in raw if isinstance(r, dict)]
            total = sum(r["comp"] for r in valid)
            agg = total / wall_s if wall_s > 0 else 0
            per = sum(r["comp"] / r["e2e"] for r in valid) / max(len(valid), 1)
            results.append(SaturationPoint(concurrency=n, tok_s=agg, per_agent=per, wall_s=wall_s, total_tokens=total))

    return results
