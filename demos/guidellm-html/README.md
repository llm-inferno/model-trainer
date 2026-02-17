# GuideLLM HTML Demo

This demo shows how to use the `GuideLLMHTMLData` reader to process GuideLLM benchmark data from HTML report files.

## Data Format

The HTML format contains embedded JavaScript with benchmark data in the `window.benchmarks` array and overall token statistics in `window.workloadDetails`. Each benchmark includes:

- Inter-token latency (ITL) statistics
- Time to first token (TTFT) statistics
- Throughput statistics
- Requests per second (RPS)
- Time per request (latency) statistics

## Limitations

The HTML format has some limitations compared to the JSON format:

1. **No per-benchmark token counts**: The HTML only contains overall token statistics, not per-benchmark values. The reader uses the overall mean token counts for all benchmarks.

2. **No explicit strategy names**: Strategy types (synchronous, throughput, constant) are inferred based on benchmark position:
   - First benchmark → "synchronous"
   - Second benchmark → "throughput" (automatically filtered during dataset creation)
   - Remaining benchmarks → "constant"

3. **TPOT equals ITL**: In the HTML format, TPOT (Time Per Output Token) is set equal to ITL (Inter Token Latency) mean, as the HTML provides ITL statistics directly.

## Usage

### With default test data

```bash
go run main.go
```

### With custom data file

```bash
go run main.go path/to/your/benchmarks.html
```

### Build and run

```bash
go build
./guidellm-html [optional-path-to-html-file]
```

## Example Data Location

The default data file is located at:

```bash
../../experiments/exp2/csv/sweep-i64-o64/benchmarks.html
```

## Output

The demo will:

1. Extract benchmark data from the HTML file
2. Extract overall token statistics from workloadDetails
3. Display all benchmarks with inferred strategy names
4. Create a dataset (filtering out throughput benchmarks)
5. Run optimization to estimate model parameters (alpha, beta, gamma)
6. Display optimization results including error metrics

## How It Works

The reader:

1. Uses **bracket-counting logic** to properly extract the `window.benchmarks` JavaScript array (handles deeply nested JSON objects and arrays)
2. Parses it as JSON to get timing metrics (ITL, TTFT, throughput, RPS, latency)
3. Extracts token statistics from `window.workloadDetails` using similar bracket-counting for the nested JSON object
4. Applies the overall token means to all benchmarks (uses prompt token mean for input, generation token mean for output)
5. Infers strategy names based on benchmark position:
   - First benchmark → "synchronous"
   - Second benchmark → "throughput" (automatically filtered in dataset creation)
   - Remaining benchmarks → "constant"

### Technical Note

The reader uses a **bracket-counting algorithm** instead of simple regex patterns to handle the deeply nested JSON structures in the HTML file. This algorithm:

- Tracks opening and closing brackets/braces (`[`, `]`, `{`, `}`)
- Properly handles strings (ignores brackets inside quoted strings)
- Handles escape sequences within strings
- Ensures complete extraction of complex nested structures without premature termination

This approach is more robust than non-greedy regex (`.*?`) which can stop too early when encountering the first closing bracket in a deeply nested structure.
