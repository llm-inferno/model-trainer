package utils

import (
	"encoding/json"
	"math"

	"github.com/llm-inferno/model-trainer/pkg/config"
)

// cost function to compute the error deviation between estimated
// and actual output variables (performance metrics)
func DeviationError(estimate, actual *config.OutputVars, err *config.ErrorVars) float64 {

	errTTFT := math.Abs(estimate.AvgTTFTTime - actual.AvgTTFTTime)
	errITL := math.Abs(estimate.AvgITLTime - actual.AvgITLTime)
	errWeightedAvg := (errTTFT*config.TTFT2ITLWeight + errITL) / (config.TTFT2ITLWeight + 1.0)

	// update the error vars
	err.Count++
	err.CumErrorTTFT += errTTFT
	err.CumErrorITL += errITL
	err.CumErrorWeightedAvg += errWeightedAvg
	return errWeightedAvg
}

// converting from model parameters struct to parameters array
func CreateParmsSliceFromModelParams(params *config.ModelParams) []float64 {
	return []float64{params.Alpha, params.Beta, params.Gamma}
}

// converting from parameters array to model parameters struct
func CreateModelParamsFromParmsSlice(parms []float64) *config.ModelParams {
	return &config.ModelParams{
		Alpha: parms[config.IndexAlpha],
		Beta:  parms[config.IndexBeta],
		Gamma: parms[config.IndexGamma],
	}
}

// check if model parameters are valid (non-negative)
func CheckParmsValid(params *config.ModelParams) bool {
	if params.Alpha < 0 || params.Beta < 0 || params.Gamma < 0 {
		return false
	}
	return true
}

// create analysis results from error variables
func CreateAnalysisResultsFromErrorVars(err *config.ErrorVars) *config.AnalysisResults {
	analysisResults := &config.AnalysisResults{}
	if err.Count > 0 {
		count := float64(err.Count)
		analysisResults.AvgErrTTFT = err.CumErrorTTFT / count
		analysisResults.AvgErrITL = err.CumErrorITL / count
		analysisResults.AvgErrWeighted = err.CumErrorWeightedAvg / count
	}
	return analysisResults
}

// unmarshal a byte array to its corresponding object
func FromDataToSpec[T any](byteValue []byte, t T) (*T, error) {
	var d T
	if err := json.Unmarshal(byteValue, &d); err != nil {
		return nil, err
	}
	return &d, nil
}
