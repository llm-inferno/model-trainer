# Random Demo

Generates synthetic training data using `LLMQueueAnalyzer` with known parameters, then runs the optimizer to verify it recovers those parameters. Serves as an end-to-end validation of the training pipeline.

## Usage

```
go run main.go [num_points] [noise_%] [seed]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `num_points` | 20 | Number of data points to generate |
| `noise_%` | 2.0 | Gaussian noise applied to TTFT and ITL (std = noise% × value). Use 0 for noiseless data. Must be ≥ 0. |
| `seed` | time-based | RNG seed for reproducibility. Always printed so any run can be replayed. |

```bash
go run main.go              # 20 points, 2% noise, random seed
go run main.go 50           # 50 points, 2% noise, random seed
go run main.go 20 0 42      # 20 points, no noise, seed=42 (exact parameter recovery)
go run main.go 20 2 42      # 20 points, 2% noise, seed=42
go run main.go 20 10 42     # 20 points, 10% noise, seed=42
```

## Configuration

| | Value |
|-|-------|
| True parameters | α=6, β=0.02, γ=0.00005 |
| Initial parameters | α=1, β=0.01, γ=0.00001 |
| maxBatchSize | 256 |
| maxNumTokens | 8192 |
| maxQueueSize | 256 |
| inputTokens range | [200, 1000] |
| outputTokens range | [200, 1000] |
| request rate range | [1, 6] rps |

## Effect of Noise

Noise adds Gaussian perturbation to the TTFT and ITL values output by the queue analyzer before they are passed to the optimizer.

| Noise | Behavior |
|-------|----------|
| 0% | Exact recovery — optimizer finds true parameters with zero error |
| 2% (default) | Harder — optimizer may land in a local minimum depending on initial parameters (see below) |
| 10% | Noisy but recoverable — optimizer converges to parameters close to true values with moderate error |

## Sensitivity to Initial Parameters

With noisy data the loss landscape develops local minima. The optimizer (Nelder-Mead) is sensitive to where it starts:

| Initial α | 2% noise result |
|-----------|----------------|
| 1 (default) | Converges well — scaling makes this robust |
| 2–6 | Also converges well |

With **0% noise**, even α=1 converges to the true parameters exactly. The local-minimum trap is a consequence of noise distorting the loss landscape, not a bug.

**Practical takeaway:** the optimizer scales each variable by its initial value so the Nelder-Mead simplex is well-conditioned regardless of parameter magnitude. For this to work, all initial values must be **positive and non-zero** — a zero initial value disables scaling for that parameter, which can cause the simplex to be degenerate in that dimension.
