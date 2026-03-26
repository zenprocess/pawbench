# API Reference

This page documents PawBench's key data types, scoring functions, and format validators.

## Data Types

All result types are defined in `pawbench.types` as Python dataclasses.

### TurnResult

Result from a single conversation turn.

```python
from pawbench.types import TurnResult
```

| Field | Type | Description |
|---|---|---|
| `turn` | `int` | Turn number (1-indexed) |
| `role` | `str` | Always `"assistant"` for results |
| `ttft_ms` | `float` | Time-to-first-token in milliseconds |
| `e2e_ms` | `float` | End-to-end latency for this turn in milliseconds |
| `prompt_tokens` | `int` | Number of prompt tokens sent |
| `completion_tokens` | `int` | Number of completion tokens generated |
| `decode_tok_s` | `float` | Decode throughput (completion tokens / decode time) |
| `tool_calls` | `list[dict]` | Raw tool call objects from the model response |
| `tool_call_correct` | `bool` | Whether tool calls matched expectations |
| `output_text` | `str` | The model's text output (excluding tool calls) |
| `format_compliant` | `bool` | Whether the output matches expected format |
| `format_fields` | `dict` | Parsed format fields (from format validator) |
| `quality_score` | `float` | Composite quality score for this turn (0.0 - 1.0) |
| `steering_followed` | `bool` | Whether a steering event was successfully followed |
| `error` | `str` | Error message if the turn failed |

### AgentResult

Result from a full multi-turn agent conversation.

```python
from pawbench.types import AgentResult
```

| Field | Type | Description |
|---|---|---|
| `agent_id` | `str` | Agent identifier from the scenario |
| `agent_name` | `str` | Agent display name |
| `turns` | `list[TurnResult]` | Results for each conversation turn |
| `total_e2e_ms` | `float` | Total wall time for the full conversation |
| `total_completion_tokens` | `int` | Sum of completion tokens across all turns |
| `total_prompt_tokens` | `int` | Sum of prompt tokens across all turns |
| `avg_decode_tok_s` | `float` | Average decode throughput across turns |
| `avg_quality` | `float` | Average quality score across turns |
| `format_compliance_rate` | `float` | Fraction of turns with compliant format |
| `steering_success` | `bool` | Whether the agent successfully followed steering events |
| `error` | `str` | Error message if the agent conversation failed |

### BenchmarkReport

The top-level report containing all four dimensions.

```python
from pawbench.types import BenchmarkReport
```

| Field | Type | Description |
|---|---|---|
| `tag` | `str` | User-provided tag for this run |
| `timestamp` | `str` | ISO 8601 timestamp |
| `endpoint` | `str` | The API endpoint URL |
| `model_card` | `ModelCard` | Captured model and serving configuration |
| `runs` | `int` | Number of runs (results averaged) |
| `scenarios` | `list[ScenarioReport]` | Per-scenario results |
| `dim1_throughput` | `dict` | Throughput dimension metrics |
| `dim2_quality` | `dict` | Quality dimension metrics |
| `dim3_efficiency` | `dict` | Efficiency dimension metrics |
| `dim4_adaptability` | `dict` | Adaptability dimension metrics |
| `saturation_curve` | `list[SaturationPoint]` | Raw throughput saturation data |
| `concurrency_curve` | `list[dict]` | Scenario-based concurrency data |
| `server_metrics` | `dict` | Scraped server-side metrics (if available) |

### Supporting Types

#### ModelCard

```python
from pawbench.types import ModelCard
```

Captured model and serving configuration for reproducibility.

| Field | Type | Description |
|---|---|---|
| `model_name` | `str` | Model identifier |
| `serving` | `dict` | Serving engine parameters |
| `model_config` | `dict` | Model architecture config |
| `tuning` | `dict` | Tuning parameters (KV cache dtype, speculative config, etc.) |
| `gpu` | `dict` | GPU information |
| `memory` | `dict` | Memory configuration |

#### SaturationPoint

One data point on the throughput saturation curve.

| Field | Type | Description |
|---|---|---|
| `concurrency` | `int` | Number of parallel requests |
| `tok_s` | `float` | Aggregate tokens per second |
| `per_agent` | `float` | Per-agent tokens per second |
| `wall_s` | `float` | Wall clock time in seconds |
| `total_tokens` | `int` | Total tokens generated |

#### ScenarioReport

Aggregated results from one scenario across all concurrency levels.

| Field | Type | Description |
|---|---|---|
| `scenario_id` | `str` | Scenario identifier |
| `scenario_name` | `str` | Scenario display name |
| `single_tok_s` | `float` | Single-agent throughput |
| `peak_tok_s` | `float` | Peak aggregate throughput |
| `peak_concurrency` | `int` | Concurrency level at peak |
| `avg_ttft_ms` | `float` | Average time-to-first-token |
| `avg_quality` | `float` | Average quality score |
| `format_compliance_rate` | `float` | Format compliance rate |
| `tool_accuracy` | `float` | Tool call accuracy |
| `useful_ratio` | `float` | Useful token ratio |
| `total_tokens` | `int` | Total completion tokens |
| `tokens_per_turn` | `float` | Average tokens per turn |
| `steering_rate` | `float` | Steering success rate |
| `nudge_response_quality` | `float` | Quality on nudge turns |

## Scoring Functions

### score_turn

```python
from pawbench.scoring import score_turn

score: float = score_turn(turn_spec, turn_result)
```

Scores a single turn's quality from 0.0 to 1.0 based on the scenario's `expect` block.

**Parameters:**

- `turn_spec` (`dict`) -- The turn definition from the scenario JSON, including the `expect` block
- `result` (`TurnResult`) -- The actual result from running the turn

**Scoring criteria** (each scores 0.0 or 1.0, averaged):

- `tool_calls_min` -- Were at least N tool calls made?
- `tool_name_any` -- Was at least one expected tool name used?
- `output_mentions` -- Proportional: fraction of expected keywords found in output text and tool arguments
- `steering_followed` -- Were steering keywords present in the output?

### useful_ratio

```python
from pawbench.scoring import useful_ratio

ratio: float = useful_ratio(text, tool_calls)
```

Calculates the fraction of output that is useful content.

**Parameters:**

- `text` (`str`) -- The model's text output
- `tool_calls` (`list[dict] | None`) -- Tool call objects (arguments count as 100% useful)

**Returns:** Float from 0.0 to 1.0.

Tool call argument characters are always counted as useful. Text lines are useful unless they match filler patterns (e.g., "Sure, I'll help...", "Here is the...", "Let me...").

## Format Validators

PawBench includes two built-in format validators. Validators are factory functions that return a validation callable.

### key_value_format_validator

```python
from pawbench.scoring import key_value_format_validator

validator = key_value_format_validator(required_keys=["STATUS", "FILES_CREATED"])
result: dict = validator(output_text)
```

Validates `KEY:value` line-based output formats.

**Returns a dict with:**

- `compliant` (`bool`) -- Whether all required keys were found
- `fields` (`dict`) -- Parsed key-value pairs
- `missing_keys` (`list[str]`) -- Keys that were not found
- `first_key_correct` (`bool`) -- Whether the first line starts with the first required key

### json_format_validator

```python
from pawbench.scoring import json_format_validator

validator = json_format_validator(required_fields=["status", "files"])
result: dict = validator(output_text)
```

Validates JSON output format. Automatically extracts JSON from markdown code blocks if wrapped in `` ```json ... ``` ``.

**Returns a dict with:**

- `compliant` (`bool`) -- Whether the output is valid JSON with all required fields
- `fields` (`dict`) -- Parsed JSON object
- `missing_keys` (`list[str]`) -- Required fields not found (if `required_fields` specified)
- `parse_error` (`str`) -- JSON parse error message (if invalid)
