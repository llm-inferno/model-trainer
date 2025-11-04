package core

import (
	"fmt"

	"github.com/llm-inferno/model-trainer/pkg/config"
	"github.com/llm-inferno/model-trainer/pkg/utils"
	"gonum.org/v1/gonum/optimize"
)

type Optimizer struct {
	InitParms *config.ModelParams
}

type OptimizationResult struct {
	OptimizedParms *config.ModelParams
	Cost           float64
}

func NewOptimizer(initParms *config.ModelParams) *Optimizer {
	return &Optimizer{
		InitParms: initParms,
	}
}

func (opt *Optimizer) Optimize(dataSet *config.DataSet, model ModelFunction) (*OptimizationResult, error) {
	// prepare data
	xData, yData := utils.CreateInOutVarsFromDataSet(dataSet)
	initParms := utils.CreateParmsSliceFromModelParams(opt.InitParms)

	// Create a problem for the optimizer
	problem := optimize.Problem{
		Func: func(p []float64) float64 {
			params := utils.CreateModelParamsFromParmsSlice(p)
			return Cost(params, xData, yData, model)
		},
	}

	// Run the optimization
	settings := &optimize.Settings{
		GradientThreshold: 1e-6,
		MajorIterations:   1000,
	}
	result, err := optimize.Minimize(problem, initParms, settings, nil)
	if err != nil {
		return nil, fmt.Errorf("optimization error: %w", err)
	}

	fmt.Printf("Optimized parameters: %v\n", result.X)
	fmt.Printf("Minimum cost: %f\n", result.F)
	optimizedParms := utils.CreateModelParamsFromParmsSlice(result.X)
	return &OptimizationResult{
		OptimizedParms: optimizedParms,
		Cost:           result.F,
	}, nil
}
