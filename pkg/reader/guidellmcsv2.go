package reader

import (
	"encoding/json"
	"fmt"

	"github.com/llm-inferno/model-trainer/pkg/config"
	"github.com/llm-inferno/model-trainer/pkg/core"
	"github.com/llm-inferno/model-trainer/pkg/utils"
)

// data in a GuideLLM CSV results file (v2 format with pipe-separated hierarchical fields)
type GuideLLMCSV2Data struct {
	Benchmarks []BenchmarkCSV2
}

// benchmark data in GuideLLM CSV results file (v2 format)
type BenchmarkCSV2 struct {
	ID           string  `json:"Benchmark | ID"`
	Name         string  `json:"Benchmark | Strategy"`
	RPS          float64 `json:"Server Throughput | Successful Requests/Sec | Mean"`
	Concurrency  float64 `json:"Server Throughput | Successful Concurrency | Mean"`
	Latency      float64 `json:"Request Latency | Successful Sec | Mean"`
	InputTokens  float64 `json:"Token Metrics | Successful Input Tokens | Mean"`
	OutputTokens float64 `json:"Token Metrics | Successful Output Tokens | Mean"`
	TTFT         float64 `json:"Time to First Token | Successful ms | Median"`
	TPOT         float64 `json:"Time per Output Token | Successful ms | Mean"`
	ITL          float64 `json:"Inter Token Latency | Successful ms | Mean"`
}

func NewGuideLLMCSV2Data() *GuideLLMCSV2Data {
	return &GuideLLMCSV2Data{}
}

func (g *GuideLLMCSV2Data) ReadFrom(dataBytes []byte) error {
	benchmarks, err := utils.FromDataToSpec(dataBytes, []BenchmarkCSV2{})
	if err != nil {
		return err
	}
	g.Benchmarks = *benchmarks
	return nil
}

func (g *GuideLLMCSV2Data) CreateDataSet() *core.DataSet {
	dataSet := core.NewDataSet("GuideLLM CSV v2 benchmark data")
	for _, benchmark := range g.Benchmarks {
		if benchmark.Name == "throughput" {
			// skip throughput benchmark data point
			continue
		}
		if dataSet.Size() >= DefaultLimitNumDataPoints {
			// limit number of data points for model training
			break
		}
		dataPoint := &core.DataPoint{
			RequestRate:  benchmark.RPS,
			InputTokens:  benchmark.InputTokens,
			OutputTokens: benchmark.OutputTokens,
			AvgTTFTTime:  benchmark.TTFT, // using median instead of mean since TTFT has a long tail
			AvgITLTime:   benchmark.ITL,
			// TODO: how to get the max batch size and max num tokens from the data?
			MaxBatchSize: config.DefaultMaxBatchSize,
			MaxNumTokens: config.DefaultMaxNumTokens,
		}
		dataSet.AppendDataPoint(dataPoint)
	}
	return dataSet
}

func (g *GuideLLMCSV2Data) Print() {
	for _, benchmark := range g.Benchmarks {
		fmt.Printf("Benchmark ID: %s\n", benchmark.ID)
		fmt.Printf("  Strategy: %s\n", benchmark.Name)
		fmt.Printf("  RPS: Mean=%.2f\n", benchmark.RPS)
		fmt.Printf("  Concurrency: Mean=%.2f\n", benchmark.Concurrency)
		fmt.Printf("  Latency: Mean=%.2f\n", benchmark.Latency)
		fmt.Printf("  Input Tokens: Mean=%.2f\n", benchmark.InputTokens)
		fmt.Printf("  Output Tokens: Mean=%.2f\n", benchmark.OutputTokens)
		fmt.Printf("  TTFT: Median=%.2f\n", benchmark.TTFT)
		fmt.Printf("  TPOT: Mean=%.2f\n", benchmark.TPOT)
		fmt.Printf("  ITL: Mean=%.2f\n", benchmark.ITL)
	}
}

func (g *GuideLLMCSV2Data) Dump() string {
	if jsonStr, err := json.Marshal(g); err == nil {
		return fmt.Sprintf("GuideLLM CSV v2 Benchmarks data: %v\n", string(jsonStr))
	}
	return "GuideLLM CSV v2 Benchmarks data: <unavailable>"
}
