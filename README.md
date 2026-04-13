# LLM Inferno Model Trainer

A queueing model parameter estimation tool for LLM (Large Language Model) inference systems. This tool uses performance benchmark data to estimate optimal parameters for a mathematical model that predicts LLM serving behavior.

## Overview

The Model Trainer trains a queueing model to predict LLM inference performance metrics (Time To First Token and Inter-Token Latency) based on workload characteristics. It uses numerical optimization to find optimal values for three model parameters that best fit observed performance data.

### Model Parameters

The system estimates three key parameters:

- **Alpha (α)**: Base time component - represents fixed overhead
- **Beta (β)**: Slope for compute time - scales with computational work
- **Gamma (γ)**: Slope for memory access time - scales with memory operations

### How It Works

1. **Input**: Takes benchmark data containing request rates, token counts, and measured performance metrics
2. **Optimization**: Uses numerical optimization (via gonum) to find best-fitting parameters
3. **Validation**: Computes prediction errors to assess model accuracy
4. **Output**: Returns optimized parameters and analysis results

## Project Structure

- **pkg/core/**: Core model logic
  - Model implementation using LLM queue analyzer
  - Parameter optimization engine
  - Dataset and datapoint structures

- **pkg/reader/**: Data readers for different formats
  - GuideLLM JSON format
  - GuideLLM CSV format
  - Custom data formats

- **pkg/service/**: HTTP API service (Gin framework)

- **demos/**: Example programs
  - Simple direct API usage
  - GuideLLM data integration
  - Multiple dataset training

## Input Data Format

Each data point represents one benchmark measurement:

```json
{
    "requestRate": 76.95,
    "inputTokens": 64,
    "outputTokens": 64,
    "avgTTFTTime": 20.47,
    "avgITLTime": 8.63,
    "maxBatchSize": 512,
    "maxNumTokens": 8192
}
```

**Field notes:**

- `avgTTFTTime` is optional. If absent or zero, it is computed as `avgWaitTime + avgPrefillTime` (both of which must then be provided in milliseconds).
- `maxBatchSize` defaults to `256` when omitted.
- `maxNumTokens` defaults to `8192` when omitted.

**Alternative using wait + prefill times:**

```json
{
    "requestRate": 76.95,
    "inputTokens": 64,
    "outputTokens": 64,
    "avgWaitTime": 0.0,
    "avgPrefillTime": 20.47,
    "avgITLTime": 8.63,
    "maxBatchSize": 512
}
```

## Sample Output Result

```text
Optimization completed successfully!
-------------------------------
Name of data set: sample_dataset
Number of data points: 7
Initial parameters: {"alpha":10,"beta":0,"gamma":0}
Estimated parameters:
{"OptimizedParms":{"alpha":6.7605062059162675,"beta":0.025362709109593,"gamma":2.575997037645202e-9},"AnalysisResults":{"avgErrTTFT":1.5372373199462892,"avgErrITL":0.799758480616978,"avgErrWeighted":1.0455847603934152}}
```

> **Note:** The optimizer scales each parameter by its initial value to keep the search space well-conditioned. Initial parameters should be **positive and non-zero** — a zero initial value disables scaling for that parameter, which can degrade convergence when parameters span multiple orders of magnitude.

**Results Interpretation:**

- **OptimizedParms**: The estimated model parameters (alpha, beta, gamma)
- **AnalysisResults**: Prediction accuracy metrics
  - `avgErrTTFT`: Average absolute error for Time To First Token (milliseconds)
  - `avgErrITL`: Average absolute error for Inter-Token Latency (milliseconds)
  - `avgErrWeighted`: Weighted average error — computed as `(avgErrTTFT × 0.5 + avgErrITL) / 1.5`, giving ITL twice the weight of TTFT

## Usage

### Demos

Each demo under `demos/` shows a different usage pattern:

| Demo | Description |
|------|-------------|
| [`demos/simple`](./demos/simple/main.go) | Load a native `DataSet` JSON file and run the optimizer |
| [`demos/guidellm`](./demos/guidellm/main.go) | Load a GuideLLM JSON or CSV benchmark file via the reader package |
| [`demos/guidellm-multiple`](./demos/guidellm-multiple/main.go) | Merge multiple GuideLLM files (JSON or CSV) into one dataset before training; files are passed as `$`-separated paths |
| [`demos/guidellm-html`](./demos/guidellm-html/main.go) | Load a GuideLLM HTML benchmark report |

Run any demo directly:

```bash
cd demos/simple
go run main.go                         # uses samples/data.json

cd demos/guidellm
go run main.go                         # uses samples/guidellm.json
go run main.go path/to/benchmarks.json

cd demos/guidellm-multiple
go run main.go file1.json$file2.json   # merge multiple files

cd demos/guidellm-html
go run main.go path/to/benchmarks.html
```

### GuideLLM reader formats

The `pkg/reader` package supports three GuideLLM output formats. Use the appropriate reader when loading benchmark files programmatically:

```go
// JSON format (guidellm sweep output)
dataReader := reader.NewGuideLLMData()

// CSV format
dataReader := reader.NewGuideLLMCSVData()

// HTML report format
dataReader := reader.NewGuideLLMHTMLData()

dataBytes, _ := os.ReadFile("benchmarks.json")
dataReader.ReadFrom(dataBytes)
dataSet := dataReader.CreateDataSet()
```

The `demos/guidellm` demo automatically falls back from JSON to CSV if the JSON parse fails.

### Docker

    Build and run the image

    ``` bash
    docker build -t model-trainer .
    docker run -d -p 8080:8080 --name model-trainer model-trainer
    ```

    Submit a train request with a data set

    ``` bash
    curl -X POST http://localhost:8080/train -d @./samples/data.json
    ```

    ```json
    {
        "OptimizedParms": {
            "alpha": 6.7605062059162675,
            "beta": 0.025362709109593,
            "gamma": 2.575997037645202e-9
        },
        "AnalysisResults": {
            "avgErrTTFT": 1.5372373199462892,
            "avgErrITL": 0.799758480616978,
            "avgErrWeighted": 1.0455847603934152
        }
    }
    ```
