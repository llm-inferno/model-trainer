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

## Sample Data Point

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

**Results Interpretation:**

- **OptimizedParms**: The estimated model parameters (alpha, beta, gamma)
- **AnalysisResults**: Prediction accuracy metrics
  - `avgErrTTFT`: Average error for Time To First Token (milliseconds)
  - `avgErrITL`: Average error for Inter-Token Latency (milliseconds)
  - `avgErrWeighted`: Weighted average error across both metrics

## Usage

- Direct call
  
    Example [main.go](./demos/simple/main.go):

    ``` bash
    cd demos/simple
    go run main.go
    ```

- Docker

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
