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
	// average errors due to optimal parameters
	AnalysisResults *config.AnalysisResults
}

func NewOptimizer(initParms *config.ModelParams) *Optimizer {
	return &Optimizer{
		InitParms: initParms,
	}
}

// optimize model parameters to fit the data set using the given model function
func (opt *Optimizer) Optimize(dataSet *DataSet, model ModelFunction) (*OptimizationResult, error) {
	// prepare data
	xData, yData := dataSet.GetInOutVars()
	initParms := utils.CreateParmsSliceFromModelParams(opt.InitParms)
	errVars := &config.ErrorVars{}

	// Create a problem for the optimizer
	problem := optimize.Problem{
		Func: func(p []float64) float64 {
			params := utils.CreateModelParamsFromParmsSlice(p)
			return LossFunction(params, xData, yData, model, errVars, false)
		},
	}

	// Run the optimizer
	settings := &optimize.Settings{
		MajorIterations: config.DefaultNumberOptimizationIterations,
	}
	result, err := optimize.Minimize(problem, initParms, settings, nil)
	if err != nil {
		return nil, fmt.Errorf("optimization error: %w", err)
	}
	optimizedParms := utils.CreateModelParamsFromParmsSlice(result.X)
	fmt.Printf("Optimization completed. Objective value: %f\n", result.F)

	// Create analysis results using optimal solution
	errVars = &config.ErrorVars{} // start with clean error vars
	LossFunction(optimizedParms, xData, yData, model, errVars, true)
	analysisResults := utils.CreateAnalysisResultsFromErrorVars(errVars)
	return &OptimizationResult{
		OptimizedParms:  optimizedParms,
		AnalysisResults: analysisResults,
	}, nil
}
