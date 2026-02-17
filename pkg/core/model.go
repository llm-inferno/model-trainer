package core

import (
	"fmt"
	"math"

	"github.com/llm-inferno/model-trainer/pkg/config"
	"github.com/llm-inferno/model-trainer/pkg/utils"
	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
)

// non-linear function, representing the model mapping input variables
// to output variables (performance metrics), given the model parameters
type ModelFunction func(x *config.InputVars, params *config.ModelParams) (*config.OutputVars, error)

// implementation of the model function using the LLM queue analyzer
func Model(x *config.InputVars, params *config.ModelParams) (*config.OutputVars, error) {

	// check parameter validity
	if !utils.CheckParmsValid(params) {
		return nil, fmt.Errorf("invalid parameters")
	}

	// create LLM queue analyzer
	queueConfig := &analyzer.Configuration{
		MaxBatchSize: x.MaxBatchSize,
		MaxNumTokens: x.MaxNumTokens,
		MaxQueueSize: config.DefaultMaxQueueToMaxBatchRatio * x.MaxBatchSize,
		ServiceParms: &analyzer.ServiceParms{
			Alpha: float32(params.Alpha),
			Beta:  float32(params.Beta),
			Gamma: float32(params.Gamma),
		},
	}

	requestSize := &analyzer.RequestSize{
		AvgInputTokens:  float32(x.InputTokens),
		AvgOutputTokens: float32(x.OutputTokens),
	}

	queueAnalyzer, err := analyzer.NewLLMQueueAnalyzer(queueConfig, requestSize)
	if err != nil {
		return nil, fmt.Errorf("NewLLMQueueAnalyzer() failed: %v", err)
	}

	// analyze queue
	metrics, err := queueAnalyzer.Analyze(float32(x.RequestRate))

	// approximate model
	// metrics, err := Analyze(queueAnalyzer, float32(x.RequestRate))

	if err != nil {
		return nil, fmt.Errorf("Analyze() %v", err)
	}

	return &config.OutputVars{
		AvgTTFTTime: float64(metrics.AvgTTFT),
		AvgITLTime:  float64(metrics.AvgTokenTime),
	}, nil
}

// loss function to compute the cost (average deviation error from observations) of using the model with a given parameter values
func LossFunction(params *config.ModelParams,
	xData []*config.InputVars,
	yData []*config.OutputVars,
	model ModelFunction,
	errVars *config.ErrorVars,
	isPrint bool,
) float64 {

	if len(xData) != len(yData) || len(xData) == 0 {
		return 0.0
	}
	sumErrors := 0.0

	if isPrint {
		fmt.Printf("  rps \t inToken \t outToken \t TTFTMeas \t TTFTPred \t ITLMeas \t ITLPred \t")
		fmt.Println()
	}

	for i := range xData {
		predictedY, err := model(xData[i], params)
		if err != nil {
			// fmt.Printf("%d: %s\n", i, err)
			return math.Inf(1)
		}

		if isPrint {
			fmt.Printf("%6.2f \t %8.2f \t %8.2f \t %8.2f \t %8.2f \t %8.2f \t %8.2f \t \n",
				xData[i].RequestRate, xData[i].InputTokens, xData[i].OutputTokens,
				yData[i].AvgTTFTTime, predictedY.AvgTTFTTime,
				yData[i].AvgITLTime, predictedY.AvgITLTime)
		}

		sumErrors += utils.DeviationError(predictedY, yData[i], errVars)
	}
	return sumErrors / float64(len(xData))
}
