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

// scaleSlice divides each element of raw by the corresponding scale factor.
// A scale factor of 0 is treated as 1 to avoid division by zero.
func scaleSlice(raw, scale []float64) []float64 {
	out := make([]float64, len(raw))
	for i := range raw {
		if scale[i] != 0 {
			out[i] = raw[i] / scale[i]
		} else {
			out[i] = raw[i]
		}
	}
	return out
}

// unscaleSlice multiplies each element of scaled by the corresponding scale factor.
func unscaleSlice(scaled, scale []float64) []float64 {
	out := make([]float64, len(scaled))
	for i := range scaled {
		out[i] = scaled[i] * scale[i]
	}
	return out
}

// optimize model parameters to fit the data set using the given model function
func (opt *Optimizer) Optimize(dataSet *DataSet, model ModelFunction) (*OptimizationResult, error) {
	// prepare data
	xData, yData := dataSet.GetInOutVars()
	errVars := &config.ErrorVars{}

	// Scale variables by their initial values so the optimizer sees O(1) quantities.
	// This prevents the initial Nelder-Mead simplex from being degenerate when
	// parameters span multiple orders of magnitude.
	scale := utils.CreateParmsSliceFromModelParams(opt.InitParms)
	scaledInit := scaleSlice(scale, scale) // raw=init, so result is all 1s (or 1 for zero inits)

	// Create a problem for the optimizer (operates in scaled space)
	problem := optimize.Problem{
		Func: func(p []float64) float64 {
			unscaled := unscaleSlice(p, scale)
			params := utils.CreateModelParamsFromParmsSlice(unscaled)
			return LossFunction(params, xData, yData, model, errVars, false)
		},
	}

	// Run the optimizer
	settings := &optimize.Settings{
		MajorIterations: config.DefaultNumberOptimizationIterations,
	}
	result, err := optimize.Minimize(problem, scaledInit, settings, &optimize.NelderMead{})
	if err != nil {
		return nil, fmt.Errorf("optimization error: %w", err)
	}
	optimizedParms := utils.CreateModelParamsFromParmsSlice(unscaleSlice(result.X, scale))
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
