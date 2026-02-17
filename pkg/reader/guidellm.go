package reader

import (
	"encoding/json"
	"fmt"

	"github.com/llm-inferno/model-trainer/pkg/config"
	"github.com/llm-inferno/model-trainer/pkg/core"
	"github.com/llm-inferno/model-trainer/pkg/utils"
)

// data in a GuideLLM sweep json results file
type GuideLLMData struct {
	Benchmarks []Benchmark `json:"benchmarks"`
}

type Benchmark struct {
	ID      string  `json:"id_"`
	Metrics Metrics `json:"metrics"`
}

type Metrics struct {
	RPS          MetricCategories `json:"requests_per_second"`
	Concurrency  MetricCategories `json:"request_concurrency"`
	Latency      MetricCategories `json:"request_latency"`
	InputTokens  MetricCategories `json:"prompt_token_count"`
	OutputTokens MetricCategories `json:"output_token_count"`
	TTFT         MetricCategories `json:"time_to_first_token_ms"`
	TPOT         MetricCategories `json:"time_per_output_token_ms"`
	ITL          MetricCategories `json:"inter_token_latency_ms"`
}

type MetricCategories struct {
	Successful MetricMeasures `json:"successful"`
	Total      MetricMeasures `json:"total"`
}

type MetricMeasures struct {
	Mean   float64 `json:"mean"`
	Median float64 `json:"median"`
	STDev  float64 `json:"std_dev"`
}

func NewGuideLLMData() *GuideLLMData {
	return &GuideLLMData{}
}

// get benchmark data from data bytes in json file
func (g *GuideLLMData) ReadFrom(dataBytes []byte) error {
	benchmarks, err := utils.FromDataToSpec(dataBytes, GuideLLMData{})
	if err != nil {
		return err
	}
	g.Benchmarks = benchmarks.Benchmarks
	return nil
}

// create a data set object from benchmark data
func (g *GuideLLMData) CreateDataSet() *core.DataSet {
	dataSet := core.NewDataSet("GuideLLM benchmark data")

	for _, benchmark := range g.Benchmarks {
		metrics := benchmark.Metrics
		dataPoint := &core.DataPoint{
			RequestRate:  metrics.RPS.Successful.Mean,
			InputTokens:  metrics.InputTokens.Successful.Mean,
			OutputTokens: metrics.OutputTokens.Successful.Mean,
			AvgTTFTTime:  metrics.TTFT.Successful.Median, // using median instead of mean since TTFT has a long tail
			AvgITLTime:   metrics.ITL.Successful.Mean,
			// TODO: how to get the max batch size and max num tokens from the data?
			MaxBatchSize: config.DefaultMaxBatchSize,
			MaxNumTokens: config.DefaultMaxNumTokens,
		}
		dataSet.AppendDataPoint(dataPoint)
	}
	return dataSet
}

func (g *GuideLLMData) Print() {
	for _, benchmark := range g.Benchmarks {
		metrics := benchmark.Metrics
		rps := metrics.RPS.Successful
		concurrency := metrics.Concurrency.Successful
		latency := metrics.Latency.Successful
		inputTokens := metrics.InputTokens.Successful
		outputTokens := metrics.OutputTokens.Successful
		ttft := metrics.TTFT.Successful
		tpot := metrics.TPOT.Successful
		itl := metrics.ITL.Successful

		fmt.Printf("Benchmark ID: %s\n", benchmark.ID)
		fmt.Printf("  RPS: Mean=%.2f, Median=%.2f, StdDev=%.2f\n", rps.Mean, rps.Median, rps.STDev)
		fmt.Printf("  Concurrency: Mean=%.2f, Median=%.2f, StdDev=%.2f\n", concurrency.Mean, concurrency.Median, concurrency.STDev)
		fmt.Printf("  Latency: Mean=%.2f, Median=%.2f, StdDev=%.2f\n", latency.Mean, latency.Median, latency.STDev)
		fmt.Printf("  Input Tokens: Mean=%.2f, Median=%.2f, StdDev=%.2f\n", inputTokens.Mean, inputTokens.Median, inputTokens.STDev)
		fmt.Printf("  Output Tokens: Mean=%.2f, Median=%.2f, StdDev=%.2f\n", outputTokens.Mean, outputTokens.Median, outputTokens.STDev)
		fmt.Printf("  TTFT: Mean=%.2f, Median=%.2f, StdDev=%.2f\n", ttft.Mean, ttft.Median, ttft.STDev)
		fmt.Printf("  TPOT: Mean=%.2f, Median=%.2f, StdDev=%.2f\n", tpot.Mean, tpot.Median, tpot.STDev)
		fmt.Printf("  ITL: Mean=%.2f, Median=%.2f, StdDev=%.2f\n", itl.Mean, itl.Median, itl.STDev)
	}
}

func (g *GuideLLMData) Dump() string {
	if jsonStr, err := json.Marshal(g.Benchmarks); err == nil {
		return fmt.Sprintf("GuideLLM Benchmarks data: %v\n", string(jsonStr))
	}
	return "GuideLLM Benchmarks data: <unavailable>"
}
