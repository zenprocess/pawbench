# Dimensions

PawBench evaluates LLM inference endpoints across four dimensions. Each dimension captures a different aspect of real-world performance that matters for production coding agent workloads.

## Dimension 1: Throughput

Throughput measures how fast your endpoint can generate tokens under various load conditions.

### Metrics

| Metric | Unit | Description |
|---|---|---|
| `avg_single_tok_s` | tok/s | Average decode throughput with a single agent (no parallelism) |
| `raw_peak_tok_s` | tok/s | Maximum aggregate throughput across all saturation levels |
| `raw_peak_concurrency` | count | Concurrency level that achieved peak throughput |
| `avg_ttft_ms` | ms | Average time-to-first-token across all turns |

### What the Numbers Mean

**Single tok/s** is the most intuitive metric -- it's how fast one user sees tokens streaming back. This is primarily limited by the model's decode speed and any speculative decoding optimizations.

**Raw peak tok/s** shows the endpoint's maximum aggregate capacity. PawBench sweeps concurrency levels (default: 1, 2, 4, 8) and reports the peak. If peak occurs at concurrency 2, the endpoint may be memory-bandwidth-bound. If it keeps scaling to 8+, the endpoint has headroom.

**TTFT** is critical for interactive use. High TTFT means users wait before seeing any output. This is affected by prompt length, KV cache state, and prefix caching.

### Saturation Curve

PawBench also runs a raw saturation test (independent of scenarios) that sends simple completions at increasing concurrency. The saturation curve shows how throughput scales:

- **Linear scaling** -- throughput doubles when concurrency doubles (ideal)
- **Sublinear scaling** -- throughput increases but not proportionally (normal)
- **Plateau** -- throughput stops increasing (endpoint is saturated)
- **Degradation** -- throughput decreases at high concurrency (overloaded)

## Dimension 2: Quality

Quality measures whether the model produces correct, well-structured output for the given coding tasks.

### Metrics

| Metric | Range | Description |
|---|---|---|
| `avg_quality` | 0.0 - 1.0 | Composite quality score averaged across all turns and agents |
| `tool_accuracy` | 0.0 - 1.0 | Fraction of turns where the model made tool calls when expected |
| `format_compliance_rate` | 0.0 - 1.0 | Fraction of final turns with correctly structured output |

### How Quality is Scored

Each turn in a scenario has an `expect` block defining what the model should produce. The scoring function (`score_turn`) evaluates:

1. **Tool call count** -- Did the model make at least the minimum expected number of tool calls?
2. **Tool name match** -- Did the model call the right tools (e.g., `write_file` when asked to write a file)?
3. **Keyword matching** -- Do the output text and tool call arguments mention expected keywords?
4. **Steering compliance** -- For steering turns, did the model incorporate the injected requirement?

Each criterion scores 0.0 or 1.0 (except keyword matching, which is proportional). The turn's quality score is the average of all applicable criteria.

### Interpreting Quality Scores

| Score Range | Interpretation |
|---|---|
| 0.90 - 1.00 | Excellent -- model follows instructions precisely and uses tools correctly |
| 0.75 - 0.89 | Good -- minor issues like missing keywords or extra preamble |
| 0.60 - 0.74 | Fair -- model sometimes misses tool calls or ignores parts of the prompt |
| Below 0.60 | Poor -- model struggles with the multi-turn tool-calling workload |

## Dimension 3: Efficiency

Efficiency measures how much of the model's output is actually useful content versus filler text.

### Metrics

| Metric | Range / Unit | Description |
|---|---|---|
| `avg_useful_ratio` | 0.0 - 1.0 | Average fraction of output that is useful content |
| `tokens_per_turn` | count | Average completion tokens per conversation turn |
| `total_tokens` | count | Total completion tokens across all agents and turns |

### Useful Token Ratio

The `useful_ratio` function classifies output content:

- **Tool call arguments** (code) count as 100% useful
- **Text lines** are useful unless they match filler patterns:
    - "Sure, I'll help you with that..."
    - "Here is the code..."
    - "Let me create..."
    - "Of course, I can do that..."
    - "Certainly! Below is..."

A model that outputs `write_file(path="app.py", content="...")` with no preamble achieves a useful ratio near 1.0. A model that writes three paragraphs of explanation before each tool call will score lower.

### Why Efficiency Matters

In production inference, you pay for every token generated. A model with 0.95 useful ratio at the same quality as one with 0.60 useful ratio is significantly cheaper to run. Efficiency also directly impacts throughput -- fewer wasted tokens means faster end-to-end completion.

## Dimension 4: Adaptability

Adaptability measures the model's ability to handle mid-conversation changes -- steering events and cross-agent nudges that alter requirements during a task.

### Metrics

| Metric | Range | Description |
|---|---|---|
| `steering_rate` | 0.0 - 1.0 | Fraction of agents that followed a steering event |
| `nudge_quality` | 0.0 - 1.0 | Quality score on turns with nudge/steering events |
| `independent_quality` | 0.0 - 1.0 | Baseline quality from independent (no-steering) scenarios |
| `nudged_quality` | 0.0 - 1.0 | Quality from scenarios with nudge events |

### Steering Events

A steering event is a mid-conversation injection that changes the agent's task. For example, in the `pawstyle` scenario, the backend agent is told at turn 3 that the frontend added a Size Guide button, and it must now implement a breed-specific sizing endpoint.

The model must:

1. Recognize the new requirement
2. Incorporate it into its existing plan
3. Produce output that addresses both the original task and the steering event

### Nudge Events

Nudge events (in `pawstyle-nudge`) simulate cross-agent communication. The frontend agent adds features (wishlist, compare) that require backend changes. The backend receives nudges and must adapt.

This tests a harder scenario: the model must understand context from another agent's work and modify its approach accordingly.

### Interpreting Adaptability

Compare `independent_quality` with `nudged_quality`:

- **nudged >= independent** -- The model handles steering well without quality loss
- **nudged slightly < independent** -- Normal; steering adds complexity
- **nudged << independent** -- The model struggles to adapt; it may ignore steering events or produce confused output

A high `steering_rate` with low `nudge_quality` means the model attempts to follow steering but does it poorly. A low `steering_rate` means the model often ignores the injected requirement entirely.
