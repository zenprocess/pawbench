<p align="center">
  <img src="assets/pawbench.png" alt="PawBench" width="200">
</p>

<h1 align="center">PawBench</h1>

<p align="center">
  <strong>Because your model deserves a benchmark with more bark than bite.</strong>
</p>

<p align="center">
  4-dimensional LLM inference benchmark.<br>
  Multi-turn, multi-agent, parallel dispatch with tool calling.
</p>

<br>

<p align="center">
  <a href="https://github.com/zenprocess/pawbench/actions/workflows/ci.yml"><img src="https://github.com/zenprocess/pawbench/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://codecov.io/gh/zenprocess/pawbench"><img src="https://codecov.io/gh/zenprocess/pawbench/graph/badge.svg" alt="codecov"></a>
  <a href="https://sonarcloud.io/summary/new_code?id=zenprocess_pawbench"><img src="https://sonarcloud.io/api/project_badges/measure?project=zenprocess_pawbench&metric=alert_status" alt="Quality Gate"></a>
  <a href="https://pypi.org/project/pawbench/"><img src="https://badge.fury.io/py/pawbench.svg" alt="PyPI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python"></a>
  <a href="https://zenprocess.github.io/pawbench/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue" alt="Docs"></a>
</p>

<br>

---

<br>

## About

PawBench tests LLMs with **realistic coding agent workloads** — not synthetic single-turn completions.

It simulates what actually happens when you deploy coding agents: multi-turn conversations, parallel tool calling, mid-task steering events, and cross-agent coordination. Then it measures four dimensions: **throughput**, **quality**, **efficiency**, and **adaptability**.

Works against any OpenAI-compatible endpoint — vLLM, TGI, OpenAI, Ollama, LMStudio.

<br>

## Meet Lola

PawBench is inspired by **Lola** ([@_justlolathings](https://www.instagram.com/_justlolathings/)) — the most fashionable pup on Instagram.

The built-in scenarios revolve around building her boutique dog apparel store, *PawStyle by Lola*. Every product, every size guide, every "Lola's Pick" badge traces back to this style icon on four legs.

Follow Lola: [instagram.com/_justlolathings](https://www.instagram.com/_justlolathings/)

<br>

## Install

```bash
pip install pawbench
# or
uv pip install pawbench
```

<br>

## Quick Start

```bash
# Benchmark your local vLLM
pawbench --endpoint http://localhost:8000

# Against any OpenAI-compatible endpoint
pawbench --endpoint https://api.openai.com/v1 --tag gpt4o

# Just throughput saturation (no scenarios)
pawbench --saturation-only --concurrency 1,2,4,8,16

# JSON output for CI/autoresearch
pawbench --json --output results/

# Custom scenario
pawbench --scenario my_scenario.json
```

<br>

## What It Measures

### 4 Dimensions

| Dimension | Metrics |
|---|---|
| **Throughput** | Single-agent tok/s, parallel saturation curve (1->N), TTFT, peak concurrency |
| **Quality** | Tool call accuracy, instruction following, format compliance, keyword matching |
| **Efficiency** | Useful token ratio (code in tool args vs filler preamble), tokens per turn |
| **Adaptability** | Steering event response, mid-conversation context injection, nudge quality delta |

<br>

### Built-in Scenarios: PawStyle by Lola

Two parallel agents build Lola's boutique dog apparel e-commerce store — *"Where every pup is a fashionista"*:

- **`pawstyle-independent`** — Frontend and backend work independently on Lola's shop. Pure parallel throughput + quality baseline.
- **`pawstyle`** — Backend gets a steering event mid-task ("frontend added a Size Guide button — implement Lola's breed-specific sizing endpoint").
- **`pawstyle-nudge`** — Frontend adds Lola's Favorites (wishlist) and Compare features that require backend changes. Backend receives nudges and adapts.

Each scenario is 3 turns x 2 agents, with tool calls (`write_file`, `read_file`, `run_command`) and injected tool results. Products include Lola's Signature Bandana, Cozy Knit Sweater, Rainy Day Raincoat, Adventure Booties, Dapper Bow Tie, and Walk-in-Style Harness — with "Lola's Pick" badges on her personal favorites.

<br>

### Server Metrics (optional)

If the endpoint exposes `/metrics` (vLLM, TGI), PawBench scrapes:

- KV cache usage and prefix cache hit rate
- Speculative decoding acceptance rate
- GPU cache pressure

<br>

## Custom Scenarios

Scenarios are JSON files:

```json
{
  "id": "my-scenario",
  "name": "My Custom Scenario",
  "agents": [
    {
      "id": "agent-1",
      "name": "My Agent",
      "turns": [
        {
          "turn": 1,
          "role": "user",
          "content": "Build a REST API with Flask...",
          "tools": ["write_file"],
          "expect": {
            "tool_calls_min": 1,
            "tool_name_any": ["write_file"],
            "output_mentions": ["flask", "api"]
          }
        }
      ]
    }
  ],
  "tools_schema": [...]
}
```

<br>

## Comparing Configs

```bash
pawbench --tag baseline --output results/
# ... change model config ...
pawbench --tag eagle3 --output results/

pawbench-compare results/pawbench_baseline_*.json results/pawbench_eagle3_*.json
```

<br>

## Output Format

JSON results include full model card (architecture, quantization, GPU, serving params) for reproducibility:

```json
{
  "tag": "fp8-eagle3-spec3",
  "model_card": {
    "model_name": "qwen3-coder",
    "model_config": {"architectures": ["Qwen3NextForCausalLM"], "num_experts": 512},
    "tuning": {"kv_cache_dtype": "fp8_e4m3", "speculative_config": "eagle3"},
    "gpu": {"name": "NVIDIA GB10"}
  },
  "dim1_throughput": {"avg_single_tok_s": 69.0, "raw_peak_tok_s": 469.3},
  "dim2_quality": {"avg_quality": 0.81, "tool_accuracy": 0.96},
  "saturation_curve": [{"concurrency": 1, "tok_s": 69.3}, {"concurrency": 8, "tok_s": 469.3}],
  "server_metrics": {"spec_acceptance_rate": 0.72, "gpu_prefix_cache_hit_rate": 0.92}
}
```

<br>

## Why PawBench Exists

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) showed that an AI agent can autonomously run ML experiments overnight — modify, train, evaluate, repeat. PawBench extends that idea to **inference serving**: what if an agent could autonomously tune your model config, benchmark it, and keep the best result?

The problem is that LLM serving optimization is gatekept. The best configs — speculative decoding heads, MoE kernel tuning, KV cache quantization strategies — live in private Discord channels and undocumented tribal knowledge. A team with an H100 cluster can spend weeks finding the right settings. A solo dev with a single GPU doesn't have that luxury.

PawBench is the benchmark harness for that loop. Run it, change your config, run it again, compare. The [Serving Card](https://servingcard.dev) initiative takes it further — standardizing how model serving configs are documented and shared, so the community can build on each other's work instead of rediscovering the same optimizations in isolation.

Democratize the configs. Benchmark everything. Share what works.

<br>

## Disclaimer

This project has been entirely vibe coded. Two humans, several AI agents, one very fashionable dog, and a mass of mass energy that mass produced some mass code. If something breaks, it was probably the cat's fault (see [commit history](https://github.com/zenprocess/pawbench/commit/9d36c56)).

<br>

## License

MIT
