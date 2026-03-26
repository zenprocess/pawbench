# Comparing Configs

PawBench includes a comparison tool (`pawbench-compare`) for A/B testing different model configurations, quantization levels, or serving parameters.

## Workflow

### 1. Run Baseline

```bash
pawbench --endpoint http://localhost:8000 --tag baseline --output results/
```

### 2. Change Configuration

Modify your serving config -- switch quantization, enable speculative decoding, change batch size, swap models, etc.

### 3. Run Variant

```bash
pawbench --endpoint http://localhost:8000 --tag variant --output results/
```

### 4. Compare

```bash
pawbench-compare results/pawbench_baseline_*.json results/pawbench_variant_*.json
```

This prints a side-by-side table with all key metrics and percentage deltas:

```
================================================================================
  PawBench Comparison -- 2 configs
================================================================================

  Metric                          pawbench_baseline       pawbench_variant
  -----------------------------------------------------------------------
  Single tok/s                             45.2                  69.0 (+52.7%)
  Raw peak tok/s                          312.0                 469.3 (+50.4%)
  Avg TTFT (ms)                            180                   120 (-33.3%)
  Avg quality                             78.0%                 81.0% (+3.8%)
  Tool accuracy                           92.0%                 96.0% (+4.3%)
  Useful ratio                            71.0%                 75.0% (+5.6%)
  Steering rate                           60.0%                 70.0% (+16.7%)

================================================================================
```

Deltas are color-coded in the terminal:

- **Green** -- improvement (higher is better for throughput/quality; lower is better for TTFT)
- **Red** -- regression greater than 5%
- **Yellow** -- minor regression (less than 5%)

## Comparing More Than Two Configs

You can compare any number of configurations:

```bash
pawbench-compare results/pawbench_fp16_*.json \
                 results/pawbench_fp8_*.json \
                 results/pawbench_nvfp4_*.json
```

The first file is always the baseline; all others show deltas relative to it.

## Common Comparisons

### Quantization Impact

```bash
pawbench --tag fp16 --output results/
# Switch to FP8
pawbench --tag fp8 --output results/
# Switch to NVFP4
pawbench --tag nvfp4 --output results/

pawbench-compare results/pawbench_fp16_*.json \
                 results/pawbench_fp8_*.json \
                 results/pawbench_nvfp4_*.json
```

### Speculative Decoding

```bash
pawbench --tag no-spec --output results/
# Enable speculative decoding
pawbench --tag eagle3-spec3 --output results/

pawbench-compare results/pawbench_no-spec_*.json results/pawbench_eagle3-spec3_*.json
```

### Concurrency Scaling

```bash
pawbench --concurrency 1 --tag single --output results/
pawbench --concurrency 1,2,4,8,16 --tag sweep --output results/

pawbench-compare results/pawbench_single_*.json results/pawbench_sweep_*.json
```

## Programmatic Access

Result files are standard JSON. You can process them with any tool:

```bash
# Extract single metric with jq
jq '.dim1_throughput.avg_single_tok_s' results/pawbench_baseline_*.json

# Build a CSV of all runs
for f in results/pawbench_*.json; do
  jq -r '[.tag, .dim1_throughput.avg_single_tok_s, .dim2_quality.avg_quality] | @csv' "$f"
done
```

## CI Integration

Use JSON output mode for automated comparisons:

```bash
pawbench --json --tag pr-$PR_NUMBER --output results/
```

The JSON output contains all four dimensions, the saturation curve, per-scenario breakdowns, and the full model card for reproducibility.
