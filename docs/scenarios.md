# Scenarios

Scenarios define the multi-turn, multi-agent workloads that PawBench runs against your endpoint. Each scenario simulates a realistic coding task with tool calls, injected tool results, and quality expectations.

## How Scenarios Work

A scenario is a JSON file describing:

1. **Agents** -- One or more parallel agents, each with a role and conversation turns
2. **Turns** -- Sequential conversation steps with user prompts, tool results, and expected outcomes
3. **Tools** -- The tool schemas available to the model (e.g., `write_file`, `read_file`, `run_command`)
4. **Expectations** -- What the model should produce at each turn (tool calls, keywords, format)

PawBench dispatches all agents in a scenario in parallel at each concurrency level, measures throughput and latency, then scores quality against the expectations.

## Built-in Scenarios: PawStyle by Lola

All built-in scenarios revolve around building **PawStyle by Lola** -- a boutique dog apparel e-commerce store inspired by Lola ([@_justlolathings](https://www.instagram.com/_justlolathings/)).

Two parallel agents (frontend and backend) collaborate to build the store. Products include Lola's Signature Bandana, Cozy Knit Sweater, Rainy Day Raincoat, Adventure Booties, Dapper Bow Tie, and Walk-in-Style Harness -- with "Lola's Pick" badges on her personal favorites.

### `pawstyle-independent`

**Purpose:** Pure parallel throughput + quality baseline with no cross-agent communication.

Frontend and backend work independently on Lola's shop. No steering events. This scenario establishes the quality baseline -- how well the model performs when each agent works in isolation.

- **Agents:** Frontend (HTML/CSS/JS product page, cart, checkout) + Backend (Python REST API with products, orders, health endpoints)
- **Turns per agent:** 3
- **Steering events:** None
- **Measures:** Throughput, quality, efficiency (baseline)

### `pawstyle`

**Purpose:** Tests adaptability via a mid-task steering event.

Same as independent, but the backend agent receives a steering event at turn 3: "Frontend added a Size Guide button -- implement Lola's breed-specific sizing endpoint." The model must adapt its plan and incorporate the new requirement.

- **Agents:** Frontend (TypeScript fullstack) + Backend (receives steering event)
- **Turns per agent:** 3
- **Steering events:** 1 (backend turn 3)
- **Measures:** Throughput, quality, efficiency, steering response

### `pawstyle-nudge`

**Purpose:** Tests cross-agent coordination via nudge events.

Frontend adds Lola's Favorites (wishlist) and Compare features that require backend changes. Backend receives nudge events simulating cross-agent communication. This tests the model's ability to handle context injection and adapt to requirements driven by another agent.

- **Agents:** Frontend (driver, adds wishlist + compare) + Backend (receives nudges to add supporting endpoints)
- **Turns per agent:** 3
- **Steering events:** Multiple nudges
- **Measures:** All four dimensions, with emphasis on adaptability

## JSON Format Specification

Scenarios follow this JSON schema:

```json
{
  "id": "my-scenario",
  "name": "Human-readable scenario name",
  "description": "What this scenario tests",
  "agents": [
    {
      "id": "agent-1",
      "name": "Agent Display Name",
      "turns": [
        {
          "turn": 1,
          "role": "user",
          "content": "The user prompt for this turn...",
          "tools": ["write_file", "read_file"],
          "expect": {
            "tool_calls_min": 1,
            "tool_name_any": ["write_file"],
            "output_mentions": ["keyword1", "keyword2"],
            "steering_followed": true,
            "steering_keywords": ["size-guide", "size_guide"]
          }
        },
        {
          "turn": 2,
          "role": "tool_result",
          "inject_tool_results": [
            {
              "tool_call_id": "auto",
              "name": "write_file",
              "content": "{\"status\": \"ok\", \"path\": \"file.py\"}"
            }
          ],
          "content": "Next instruction after tool result...",
          "tools": ["write_file"],
          "expect": {
            "tool_calls_min": 1,
            "output_mentions": ["expected_keyword"]
          }
        }
      ]
    }
  ],
  "tools_schema": [
    {
      "type": "function",
      "function": {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {"type": "string", "description": "File path"},
            "content": {"type": "string", "description": "File content"}
          },
          "required": ["path", "content"]
        }
      }
    }
  ]
}
```

### Fields

| Field | Required | Description |
|---|---|---|
| `id` | Yes | Unique scenario identifier (used in filenames and reports) |
| `name` | Yes | Human-readable name displayed in output |
| `description` | No | Longer description of what the scenario tests |
| `agents` | Yes | Array of agent definitions |
| `agents[].id` | Yes | Unique agent identifier within the scenario |
| `agents[].name` | Yes | Display name for the agent |
| `agents[].turns` | Yes | Array of conversation turns |
| `tools_schema` | Yes | OpenAI-format tool definitions available to all agents |

### Turn Fields

| Field | Required | Description |
|---|---|---|
| `turn` | Yes | Turn number (1-indexed) |
| `role` | Yes | `"user"` for prompts, `"tool_result"` for injected tool results |
| `content` | Yes | The prompt text or follow-up instruction |
| `tools` | No | Which tools from `tools_schema` are available this turn |
| `inject_tool_results` | No | Simulated tool results to inject before this turn's prompt |
| `expect` | No | Quality expectations for scoring |

### Expect Fields

| Field | Description |
|---|---|
| `tool_calls_min` | Minimum number of tool calls expected |
| `tool_name_any` | At least one tool call must use one of these function names |
| `output_mentions` | Keywords that should appear in output text or tool call arguments |
| `steering_followed` | Whether this turn tests a steering/nudge response |
| `steering_keywords` | Keywords indicating the model followed the steering event |

## Writing Custom Scenarios

1. Create a JSON file following the format above
2. Include at least one agent with 2+ turns
3. Add tool calls and `expect` blocks for quality measurement
4. For steering tests, use `inject_tool_results` to simulate cross-agent events

Run your custom scenario:

```bash
pawbench --scenario my_scenario.json
```

You can combine custom and built-in scenarios:

```bash
pawbench --scenario custom1.json --scenario custom2.json
```

!!! tip "Scenario Design Tips"
    - Use 3+ turns per agent for meaningful multi-turn measurement
    - Include tool calls in every turn (this is what differentiates PawBench from chat benchmarks)
    - Add `inject_tool_results` between turns to simulate realistic tool-use workflows
    - Include at least one steering variant to test adaptability
    - Keep prompts detailed and specific -- vague prompts produce noisy quality scores
