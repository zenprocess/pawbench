# Getting Started

## Installation

### pip

```bash
pip install pawbench
```

### uv

```bash
uv pip install pawbench
```

### From source

```bash
git clone https://github.com/zenprocess/pawbench.git
cd pawbench
pip install -e ".[dev]"
```

## Prerequisites

PawBench requires an OpenAI-compatible API endpoint. This includes:

- [vLLM](https://docs.vllm.ai/)
- [TGI](https://huggingface.co/docs/text-generation-inference/)
- [OpenAI API](https://platform.openai.com/)
- Any server implementing the `/v1/chat/completions` endpoint with tool calling support

## First Run

Point PawBench at your endpoint:

```bash
pawbench --endpoint http://localhost:8000
```

PawBench will:

1. Query `/v1/models` to identify the served model
2. Capture the model card (architecture, quantization, GPU info, serving params)
3. Run all built-in PawStyle scenarios at concurrency levels 1, 2, 4, 8
4. Run a raw throughput saturation test
5. Scrape `/metrics` for server-side stats (if available)
6. Print a 4-dimensional report

### Common Options

```bash
# Tag your run for comparison later
pawbench --endpoint http://localhost:8000 --tag my-config

# Custom concurrency sweep
pawbench --concurrency 1,2,4,8,16,32

# Multiple runs (results are averaged)
pawbench --runs 3

# Save JSON output for programmatic analysis
pawbench --output results/ --tag baseline

# JSON to stdout (for CI pipelines)
pawbench --json

# Only run raw throughput saturation (skip scenarios)
pawbench --saturation-only --concurrency 1,2,4,8,16

# Skip saturation test (only run scenarios)
pawbench --no-saturation

# Use a custom scenario
pawbench --scenario my_scenario.json
```

## Interpreting Output

PawBench prints a human-readable report with four sections:

### Throughput (Dim 1)

- **Single tok/s** -- Average decode throughput with one agent (no parallelism)
- **Raw peak tok/s** -- Maximum aggregate throughput across all saturation levels
- **Peak concurrency** -- The concurrency level that achieved peak throughput
- **Avg TTFT** -- Average time-to-first-token across all turns

Higher single tok/s means faster individual responses. Higher peak tok/s means the endpoint scales well under load. If peak concurrency is low (e.g., 2), the endpoint may be bottlenecked.

### Quality (Dim 2)

- **Avg quality** -- Composite score (0-1) across all turns and agents
- **Tool accuracy** -- Fraction of turns where the model made at least one tool call when expected
- **Format compliance** -- Whether the model's final output matches expected structure

A quality score of 0.80+ is solid. Below 0.60 suggests the model struggles with the multi-turn tool-calling workload.

### Efficiency (Dim 3)

- **Useful ratio** -- Fraction of output that is useful content (code, tool arguments) vs filler ("Sure, I'll help you with that...")
- **Tokens per turn** -- Average completion tokens per conversation turn

Higher useful ratio means less wasted tokens. Lower tokens per turn (while maintaining quality) means the model is concise.

### Adaptability (Dim 4)

- **Steering rate** -- Fraction of agents that successfully followed a mid-conversation steering event
- **Nudge quality** -- Quality score specifically on turns that received nudge/steering events
- **Independent vs nudged quality** -- Comparison of quality with and without steering events

This dimension only applies to scenarios with steering or nudge events (`pawstyle` and `pawstyle-nudge`). The `pawstyle-independent` scenario has no steering events and serves as the quality baseline.
