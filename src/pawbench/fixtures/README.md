# PawBench Fixtures

Recorded API responses for mock/CI mode. Each fixture is a JSON file with this schema:

```json
{
  "name": "human-readable name",
  "description": "what this fixture tests",
  "model": "model-name-used-during-recording",
  "responses": [
    {
      "status": 200,
      "headers": {"content-type": "application/json"},
      "body": { "...": "full OpenAI-compatible response body" },
      "latency_ms": 50.0
    }
  ],
  "model_card": { "...": "captured model card dict (optional)" },
  "saturation": [
    {
      "concurrency": 1,
      "tok_s": 100.0,
      "per_agent": 100.0,
      "wall_s": 1.0,
      "total_tokens": 512
    }
  ]
}
```

## Fields

- **responses**: Array of recorded API responses, replayed in order (cycles when exhausted).
- **model_card**: Optional snapshot of the model card at recording time. Used instead of live capture in mock mode.
- **saturation**: Optional array of saturation curve points. If omitted, default mock values are used.

## Recording fixtures

```bash
pawbench --endpoint http://localhost:8000 --record ./my-fixtures --saturation-only
```

This saves responses to `./my-fixtures/` as JSON files.

## Replaying fixtures

```bash
pawbench --mock --saturation-only
```

Uses the built-in `sample_saturation.json` fixture. To use custom fixtures, place them in this directory or specify `--mock` with a `--scenario` pointing at fixtures.
