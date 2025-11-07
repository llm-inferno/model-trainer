package reader

import (
	"encoding/json"
	"fmt"

	"github.com/llm-inferno/model-trainer/pkg/config"
	"github.com/llm-inferno/model-trainer/pkg/utils"
)

// data in a GuideLLM CSV results file
type GuideLLMCSVData struct {
	Benchmarks []BenchmarkCSV
}

type BenchmarkCSV struct {
	ID           string  `json:"Id"`
	Name         string  `json:"Name"`
	RPS          float64 `json:"Successful Requests per second mean"`
	Concurrency  float64 `json:"Successful Request concurrency mean"`
	Latency      float64 `json:"Successful Request latency mean"`
	InputTokens  float64 `json:"Successful Prompt token count mean"`
	OutputTokens float64 `json:"Successful Output token count mean"`
	TTFT         float64 `json:"Successful Time to first token ms median"`
	TPOT         float64 `json:"Successful Time per output token ms mean"`
	ITL          float64 `json:"Successful Inter token latency ms mean"`
}

func NewGuideLLMCSVData() *GuideLLMCSVData {
	return &GuideLLMCSVData{}
}

func (g *GuideLLMCSVData) ReadFrom(dataBytes []byte) error {
	benchmarks, err := utils.FromDataToSpec(dataBytes, []BenchmarkCSV{})
	if err != nil {
		return err
	}
	g.Benchmarks = *benchmarks
	return nil
}

func (g *GuideLLMCSVData) CreateDataSet() *config.DataSet {
	dataSet := &config.DataSet{
		Name: "GuideLLM CSV benchmark data",
		Data: []config.DataPoint{},
	}
	for _, benchmark := range g.Benchmarks {
		dataPoint := &config.DataPoint{
			RequestRate:  benchmark.RPS,
			InputTokens:  benchmark.InputTokens,
			OutputTokens: benchmark.OutputTokens,
			// TODO: split TTFT into waiting and prefill time components
			AvgWaitTime:    0,
			AvgPrefillTime: benchmark.TTFT, // using median instead of mean since TTFT has a long tail
			AvgITLTime:     benchmark.ITL,
			// TODO: how to get the max batch size from the data?
			MaxBatchSize: 512,
		}
		dataSet.Data = append(dataSet.Data, *dataPoint)
	}
	return dataSet
}

func (g *GuideLLMCSVData) Print() {
	for _, benchmark := range g.Benchmarks {
		fmt.Printf("Benchmark ID: %s\n", benchmark.ID)
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

func (g *GuideLLMCSVData) Dump() string {
	if jsonStr, err := json.Marshal(g); err == nil {
		return fmt.Sprintf("GuideLLM CSV Benchmarks data: %v\n", string(jsonStr))
	}
	return "GuideLLM CSV Benchmarks data: <unavailable>"
}
