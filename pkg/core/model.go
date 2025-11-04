package core

import (
	"fmt"
	"math"

	"github.com/llm-inferno/model-trainer/pkg/config"
	"github.com/llm-inferno/model-trainer/pkg/utils"
	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
)

type ModelFunction func(x *config.InputVars, params *config.ModelParams) (*config.OutputVars, error)

func Model(x *config.InputVars, params *config.ModelParams) (*config.OutputVars, error) {

	// check parameter validity
	if !utils.CheckParmsValid(params) {
		return nil, fmt.Errorf("invalid parameters")
	}

	// create queue analyzer
	queueConfig := &analyzer.Configuration{
		MaxBatchSize: x.MaxBatchSize,
		MaxQueueSize: config.MaxQueueToMaxBatchRatio * x.MaxBatchSize,
		ServiceParms: &analyzer.ServiceParms{
			Prefill: &analyzer.PrefillParms{
				Gamma: float32(params.Gamma),
				Delta: float32(params.Delta),
			},
			Decode: &analyzer.DecodeParms{
				Alpha: float32(params.Alpha),
				Beta:  float32(params.Beta),
			},
		},
	}

	requestSize := &analyzer.RequestSize{
		AvgInputTokens:  int(math.Ceil(x.InputTokens)),
		AvgOutputTokens: int(math.Ceil(x.OutputTokens)),
	}

	queueAnalyzer, err := analyzer.NewLLMQueueAnalyzer(queueConfig, requestSize)
	if err != nil {
		return nil, fmt.Errorf("NewLLMQueueAnalyzer() failed: %v", err)
	}

	// analyze queue
	metrics, err := queueAnalyzer.Analyze(float32(x.RequestRate))
	if err != nil {
		return nil, fmt.Errorf("Analyze() %v", err)
	}

	return &config.OutputVars{
		AvgWaitTime:    float64(metrics.AvgWaitTime),
		AvgPrefillTime: float64(metrics.AvgPrefillTime),
		AvgITLTime:     float64(metrics.AvgTokenTime),
	}, nil
}

func Cost(params *config.ModelParams,
	xData []*config.InputVars,
	yData []*config.OutputVars,
	model ModelFunction) float64 {

	sumOfSquares := 0.0
	for i := range xData {
		predictedY, err := model(xData[i], params)
		if err != nil {
			fmt.Printf("%d: %s\n", i, err)
			return math.Inf(1)
		}
		sumOfSquares += utils.RelativeDeviationSquared(predictedY, yData[i])
	}
	return sumOfSquares
}
