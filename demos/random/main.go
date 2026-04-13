package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/llm-inferno/model-trainer/pkg/config"
	"github.com/llm-inferno/model-trainer/pkg/core"
	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
)

const (
	DefaultNumPoints = 20

	// known "true" parameters used to generate synthetic data
	TrueAlpha = 6.0
	TrueBeta  = 0.02
	TrueGamma = 0.00005

	// fixed queue configuration
	MaxBatchSize = 256
	MaxNumTokens = 8192
	MaxQueueSize = 256

	// random ranges for input variables
	MinInputTokens  = 200.0
	MaxInputTokens  = 1000.0
	MinOutputTokens = 200.0
	MaxOutputTokens = 1000.0
	MinRPS          = 1.0
	MaxRPS          = 6.0
)

func generateDataSet(numPoints int, rng *rand.Rand) *core.DataSet {
	trueParms := &analyzer.ServiceParms{
		Alpha: TrueAlpha,
		Beta:  TrueBeta,
		Gamma: TrueGamma,
	}

	randInRange := func(min, max float64) float64 {
		return min + rng.Float64()*(max-min)
	}

	dataSet := core.NewDataSet("random synthetic data")
	for dataSet.Size() < numPoints {
		inputTokens := randInRange(MinInputTokens, MaxInputTokens)
		outputTokens := randInRange(MinOutputTokens, MaxOutputTokens)
		requestRate := randInRange(MinRPS, MaxRPS)

		queueConfig := &analyzer.Configuration{
			MaxBatchSize: MaxBatchSize,
			MaxNumTokens: MaxNumTokens,
			MaxQueueSize: MaxQueueSize,
			ServiceParms: trueParms,
		}
		requestSize := &analyzer.RequestSize{
			AvgInputTokens:  float32(inputTokens),
			AvgOutputTokens: float32(outputTokens),
		}

		queueAnalyzer, err := analyzer.NewLLMQueueAnalyzer(queueConfig, requestSize)
		if err != nil {
			continue
		}
		metrics, err := queueAnalyzer.Analyze(float32(requestRate))
		if err != nil {
			continue
		}

		dataSet.AppendDataPoint(&core.DataPoint{
			RequestRate:  requestRate,
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
			AvgTTFTTime:  float64(metrics.AvgTTFT),
			AvgITLTime:   float64(metrics.AvgTokenTime),
			MaxBatchSize: MaxBatchSize,
			MaxNumTokens: MaxNumTokens,
		})
	}
	return dataSet
}

func main() {
	numPoints := DefaultNumPoints
	if len(os.Args) > 1 {
		n, err := strconv.Atoi(os.Args[1])
		if err != nil || n <= 0 {
			fmt.Fprintf(os.Stderr, "invalid number of data points: %s\n", os.Args[1])
			os.Exit(1)
		}
		numPoints = n
	}

	var seed int64
	if len(os.Args) > 2 {
		s, err := strconv.ParseInt(os.Args[2], 10, 64)
		if err != nil {
			fmt.Fprintf(os.Stderr, "invalid seed: %s\n", os.Args[2])
			os.Exit(1)
		}
		seed = s
	} else {
		seed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(seed))

	fmt.Printf("Generating %d synthetic data points (seed=%d)...\n", numPoints, seed)
	dataSet := generateDataSet(numPoints, rng)

	trueParms := &config.ModelParams{Alpha: TrueAlpha, Beta: TrueBeta, Gamma: TrueGamma}
	initParms := &config.ModelParams{Alpha: 1.0, Beta: 0.01, Gamma: 0.00001}

	optimizer := core.NewOptimizer(initParms)
	optimizerResult, err := optimizer.Optimize(dataSet, core.Model)
	if err != nil {
		fmt.Println("Optimization failed:", err)
		os.Exit(1)
	}

	fmt.Println("Optimization completed successfully!")
	fmt.Println("-------------------------------")
	fmt.Printf("Name of data set: %s\n", dataSet.Name)
	fmt.Printf("Number of data points: %d\n", dataSet.Size())
	if jsonStr, err := json.Marshal(trueParms); err == nil {
		fmt.Printf("True parameters:    %v\n", string(jsonStr))
	}
	if jsonStr, err := json.Marshal(initParms); err == nil {
		fmt.Printf("Initial parameters: %v\n", string(jsonStr))
	}
	fmt.Println("Estimated parameters:")
	if jsonStr, err := json.Marshal(optimizerResult); err == nil {
		fmt.Println(string(jsonStr))
	}
}
