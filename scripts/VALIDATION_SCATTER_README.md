# Validation Scatter

Reproduce the predicted-vs-measured TTFT/ITL scatter plot used as Figure 1
in the MASCOTS 2026 paper *"An approximate parameterized queueing model of
LLM inference serving"*.

## What it does

`validation_scatter.py` runs a single joint Nelder-Mead fit per model
across all sweep files in `experiments/exp1/data` (Llama-3.1-8B, JSON) and
`experiments/exp2/data` (Qwen2.5-14B, HTML). For each experiment it:

1. Globs `sweep-i*-o*.*` files and joins their paths with `$`.
2. Invokes `go run ./demos/guidellm-multiple <joined-paths>` from the
   repo root, which reads every file with the GuideLLM JSON, CSV, or
   HTML reader, merges them into one DataSet, runs the joint optimizer
   over all 112 points, and prints predicted vs. measured TTFT and ITL
   for every point.
3. Parses the optimizer's stdout for the fitted `(alpha, beta, gamma)`
   and the per-point predicted/measured columns.
4. Computes the paper's relative error metric, `mean(|pred - meas|) /
   mean(meas)`, separately for TTFT and ITL.
5. Plots a two-panel scatter with both models overlaid (Llama as blue
   circles, Qwen as red triangles, dashed `y = x`) and saves it as PDF.

## Requirements

- Go (matching the repo's `go.mod`).
- Python 3.10+ with the packages in `requirements.txt`.

```bash
pip install -r scripts/requirements.txt
```

## Run

From the repo root:

```bash
python scripts/validation_scatter.py
```

This writes the figure to `experiments/validation-scatter.pdf` and prints
the fitted parameters and errors. To override the output path:

```bash
python scripts/validation_scatter.py -o /path/to/figure.pdf
```

## Expected output

The fit reproduces Table 1 of the paper:

| model         | alpha | beta       | gamma      | err_TTFT | err_ITL |
|---------------|------:|-----------:|-----------:|---------:|--------:|
| Llama-3.1-8B  | 6.58  | 1.93e-02   | 5.46e-05   | 0.146    | 0.060   |
| Qwen2.5-14B   | 9.87  | 3.58e-02   | 8.63e-05   | 0.163    | 0.080   |

## Notes

- The paper's relative error metric is computed in Python from the
  measured/predicted columns and is the mean absolute deviation divided
  by the mean measurement. The Go optimizer minimises the per-point sum
  of squared relative errors, `((TTFT_pred - TTFT_obs)/TTFT_obs)^2 +
  ((ITL_pred - ITL_obs)/ITL_obs)^2` (see `pkg/utils/utils.go`), so the
  optimizer's `avgErrWeighted` field is not the same number as either
  `err_TTFT` or `err_ITL` reported here.
- The HTML fallback in `demos/guidellm-multiple/main.go` is what lets
  exp2's GuideLLM HTML files participate in the joint fit alongside
  exp1's JSON files.
