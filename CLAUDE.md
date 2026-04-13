# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build
go build ./...

# Run all tests
go test ./...

# Run tests in a specific package
go test ./pkg/core/...

# Run a single test
go test ./pkg/core/... -run TestOptimizer_Optimize

# Run with verbose output
go test -v ./pkg/core/...

# Run benchmarks
go test -bench=. ./pkg/core/...

# Run the HTTP service locally
go run main.go

# Run a demo directly
cd demos/simple && go run main.go

# Docker build and run
docker build -t model-trainer .
docker run -d -p 8080:8080 --name model-trainer model-trainer

# Submit a training request to the running service
curl -X POST http://localhost:8080/train -d @./samples/data.json
```

## Architecture

The system estimates three queueing model parameters (Alpha, Beta, Gamma) that best fit observed LLM benchmark data. It delegates the physics of queue analysis to the external `github.com/llm-inferno/queue-analysis` package.

### Data Flow

```
benchmark data (JSON/CSV/HTML)
    → reader.Reader  (pkg/reader/)      — parses format-specific files into DataSet
    → core.DataSet   (pkg/core/)        — holds []DataPoint (measured benchmark rows)
    → core.Optimizer (pkg/core/)        — calls gonum optimize.Minimize with LossFunction
    → core.Model     (pkg/core/)        — wraps queue-analysis LLMQueueAnalyzer
    → OptimizationResult                — {OptimizedParms, AnalysisResults}
```

### Key packages

- **`pkg/config/`** — shared types (`ModelParams`, `InputVars`, `OutputVars`, `ErrorVars`, `AnalysisResults`) and constants (default batch sizes, optimizer iterations, TTFT/ITL error weight).

- **`pkg/core/`** — the heart of the system:
  - `DataPoint` / `DataSet` — data structures; `Fix()` fills missing fields with defaults, `ToMSecs()` converts seconds to ms.
  - `Model()` — the forward model: given `InputVars` + `ModelParams`, calls `queue-analysis` and returns predicted `OutputVars`.
  - `LossFunction()` — computes mean weighted deviation error across a dataset.
  - `Optimizer` — wraps `gonum/optimize.Minimize` (explicitly Nelder-Mead) to minimize `LossFunction`. Variables are scaled by their initial values before optimization so the simplex is well-conditioned across parameters of different magnitudes.
  - `Analyzer` — runs `LossFunction` with fixed parameters (no optimization) to score a trained model.

- **`pkg/reader/`** — format adapters implementing the `Reader` interface (`ReadFrom`, `CreateDataSet`). Formats: GuideLLM JSON, GuideLLM CSV, GuideLLM HTML. GuideLLM uses median TTFT (not mean) because TTFT has a long tail.

- **`pkg/service/`** — Gin HTTP server exposing `POST /train`. Accepts a `DataSet` JSON body, runs the optimizer with hard-coded initial parameters `{Alpha:1, Beta:0, Gamma:0}`, returns `OptimizationResult`.

- **`pkg/utils/`** — conversions between `ModelParams` struct ↔ `[]float64` (required by gonum), and `DeviationError` weighted as `(errTTFT * 0.5 + errITL) / 1.5`.

### Key design decisions

- **Parameters are non-negative**: `utils.CheckParmsValid` rejects negative Alpha/Beta/Gamma; the model returns `math.Inf(1)` as loss when parameters are invalid, steering the optimizer away.
- **DataPoint.Fix()** is called lazily on `GetInOutVars()`, so callers don't need to remember to call it.
- **MaxBatchSize/MaxNumTokens** default to 256/8192 when not present in benchmark data (common for GuideLLM output).
- **`queue-analysis` dependency** is versioned; when pointing at a local path for development, update go.mod to use `replace` directive then revert before tagging.
