# PawBench

4-dimensional LLM inference benchmark for OpenAI-compatible endpoints. Multi-turn, multi-agent, parallel dispatch with tool calling.

Tests your model with realistic coding agent workloads — not synthetic single-turn completions.

## Install

```bash
pip install pawbench
# or
uv pip install pawbench
```

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

## What It Measures

### 4 Dimensions

| Dimension | Metrics |
|---|---|
| **Throughput** | Single-agent tok/s, parallel saturation curve (1→N), TTFT, peak concurrency |
| **Quality** | Tool call accuracy, instruction following, format compliance, keyword matching |
| **Efficiency** | Useful token ratio (code in tool args vs filler preamble), tokens per turn |
| **Adaptability** | Steering event response, mid-conversation context injection, nudge quality delta |

### Built-in Scenarios: PawStyle Dog Apparel Store

Two parallel agents build an e-commerce store:

- **`pawstyle-independent`** — Frontend and backend work independently. Pure parallel throughput + quality baseline.
- **`pawstyle`** — Backend gets a steering event mid-task ("frontend added a Size Guide button — implement the endpoint").
- **`pawstyle-nudge`** — Frontend adds features (wishlist, compare) that require backend changes. Backend receives nudges and adapts.

Each scenario is 3 turns × 2 agents, with tool calls (write_file, read_file, run_command) and injected tool results.

### Server Metrics (optional)

If the endpoint exposes `/metrics` (vLLM, TGI), PawBench scrapes:
- KV cache usage and prefix cache hit rate
- Speculative decoding acceptance rate
- GPU cache pressure

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

## Comparing Configs

```bash
pawbench --tag baseline --output results/
# ... change model config ...
pawbench --tag eagle3 --output results/

python -m pawbench.compare results/pawbench_baseline_*.json results/pawbench_eagle3_*.json
```

## Output Format

JSON results include full model card (architecture, quantization, GPU, serving params) for reproducibility:

```json
{
  "tag": "fp8-eagle3-spec3",
  "model_card": {
    "model_name": "qwen3-coder",
    "model_config": {"architectures": ["Qwen3NextForCausalLM"], "num_experts": 512, ...},
    "tuning": {"kv_cache_dtype": "fp8_e4m3", "speculative_config": "eagle3", ...},
    "gpu": {"name": "NVIDIA GB10", ...}
  },
  "dim1_throughput": {"avg_single_tok_s": 69.0, "raw_peak_tok_s": 469.3, ...},
  "dim2_quality": {"avg_quality": 0.81, "tool_accuracy": 0.96, ...},
  "saturation_curve": [{"concurrency": 1, "tok_s": 69.3}, {"concurrency": 8, "tok_s": 469.3}],
  "server_metrics": {"spec_acceptance_rate": 0.72, "gpu_prefix_cache_hit_rate": 0.92}
}
```

## License

MIT
