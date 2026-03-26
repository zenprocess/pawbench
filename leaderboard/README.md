# PawBench Leaderboard

Public leaderboard for PawBench benchmark results.

## Submitting Results

1. Run PawBench against your endpoint:

   ```bash
   pawbench --endpoint http://localhost:8000 --output results/
   ```

2. Create a submission JSON file that conforms to `schema.json`. See `example_submission.json` for reference.

   You can validate your submission locally:

   ```bash
   pawbench-leaderboard validate my_result.json
   ```

3. Open a pull request adding your result file to `leaderboard/results/`. Use the naming convention:

   ```
   leaderboard/results/<model>_<gpu>_<date>.json
   ```

   For example: `leaderboard/results/qwen3-80b-fp8_gb10_20260325.json`

4. CI will validate the submission against the schema automatically.

## Viewing the Leaderboard

### CLI

```bash
pawbench-leaderboard render leaderboard/results/
```

### Web

Open `leaderboard/site/index.html` in a browser, or visit the GitHub Pages deployment.

## Schema

All submissions must validate against `schema.json`. Required fields:

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model name |
| `gpu` | string | GPU hardware used |
| `config` | object | Serving configuration (engine, quantization, etc.) |
| `single_tok_s` | number | Average single-agent throughput (tok/s) |
| `peak_tok_s` | number | Peak aggregate throughput (tok/s) |
| `avg_quality` | number | Average quality score (0-1) |
| `avg_ttft_ms` | number | Average time-to-first-token (ms) |
| `timestamp` | string | ISO 8601 timestamp |
| `pawbench_version` | string | PawBench version used |

See `schema.json` for the full specification including optional fields.
