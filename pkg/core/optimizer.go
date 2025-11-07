package core

import (
	"fmt"

	"github.com/llm-inferno/model-trainer/pkg/config"
	"github.com/llm-inferno/model-trainer/pkg/utils"
	"gonum.org/v1/gonum/optimize"
)

// optimizer to perform parameter estimation
type Optimizer struct {
	// initial values of model parameters
	InitParms *config.ModelParams
}

// result of optimization
type OptimizationResult struct {
	//optimal values of model parameters
	OptimizedParms *config.ModelParams
	// mean squared relative error (MSRE) due to optimal parameters
	MSRE float64
}

func NewOptimizer(initParms *config.ModelParams) *Optimizer {
	return &Optimizer{
		InitParms: initParms,
	}
}

// optimize model parameters to fit the data set using the given model function
func (opt *Optimizer) Optimize(dataSet *config.DataSet, model ModelFunction) (*OptimizationResult, error) {
	// prepare data
	xData, yData := utils.CreateInOutVarsFromDataSet(dataSet)
	initParms := utils.CreateParmsSliceFromModelParams(opt.InitParms)

	// Create a problem for the optimizer
	problem := optimize.Problem{
		Func: func(p []float64) float64 {
			params := utils.CreateModelParamsFromParmsSlice(p)
			return LossFunction(params, xData, yData, model)
		},
	}

	// Run the optimizer
	settings := &optimize.Settings{
		GradientThreshold: 1e-6,
		MajorIterations:   1000,
	}
	result, err := optimize.Minimize(problem, initParms, settings, nil)
	if err != nil {
		return nil, fmt.Errorf("optimization error: %w", err)
	}

	optimizedParms := utils.CreateModelParamsFromParmsSlice(result.X)
	return &OptimizationResult{
		OptimizedParms: optimizedParms,
		MSRE:           result.F,
	}, nil
}
