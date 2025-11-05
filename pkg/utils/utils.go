package utils

import (
	"encoding/json"

	"github.com/llm-inferno/model-trainer/pkg/config"
)

// cost function to compute the relative deviation squared between estimated
// and actual output variables (performance metrics)
func RelativeDeviationSquared(estimate, actual *config.OutputVars) float64 {
	var wait, prefill, itl float64
	if actual.AvgWaitTime > 0 {
		wait = (estimate.AvgWaitTime - actual.AvgWaitTime) / actual.AvgWaitTime
	}
	if actual.AvgPrefillTime > 0 {
		prefill = (estimate.AvgPrefillTime - actual.AvgPrefillTime) / actual.AvgPrefillTime
	}
	if actual.AvgITLTime > 0 {
		itl = (estimate.AvgITLTime - actual.AvgITLTime) / actual.AvgITLTime
	}
	return (wait*wait + prefill*prefill + itl*itl) / 3.0
}

// converting from a data point struct to input and output variables
func CreateInOutVarsFromDataPoint(dataPoint *config.DataPoint) (x *config.InputVars, y *config.OutputVars) {
	x = &config.InputVars{
		RequestRate:  dataPoint.RequestRate,
		InputTokens:  dataPoint.InputTokens,
		OutputTokens: dataPoint.OutputTokens,
		MaxBatchSize: dataPoint.MaxBatchSize,
	}
	y = &config.OutputVars{
		AvgWaitTime:    dataPoint.AvgWaitTime,
		AvgPrefillTime: dataPoint.AvgPrefillTime,
		AvgITLTime:     dataPoint.AvgITLTime,
	}
	return x, y
}

// converting from a data set struct to input and output variable arrays
func CreateInOutVarsFromDataSet(dataSet *config.DataSet) (xData []*config.InputVars, yData []*config.OutputVars) {
	for _, dp := range dataSet.Data {
		x, y := CreateInOutVarsFromDataPoint(&dp)
		xData = append(xData, x)
		yData = append(yData, y)
	}
	return xData, yData
}

// converting from model parameters struct to parameters array
func CreateParmsSliceFromModelParams(params *config.ModelParams) []float64 {
	return []float64{params.Alpha, params.Beta, params.Gamma, params.Delta}
}

// converting from parameters array to model parameters struct
func CreateModelParamsFromParmsSlice(parms []float64) *config.ModelParams {
	return &config.ModelParams{
		Alpha: parms[config.IndexAlpha],
		Beta:  parms[config.IndexBeta],
		Gamma: parms[config.IndexGamma],
		Delta: parms[config.IndexDelta],
	}
}

// check if model parameters are valid (non-negative)
func CheckParmsValid(params *config.ModelParams) bool {
	if params.Alpha < 0 || params.Beta < 0 || params.Gamma < 0 || params.Delta < 0 {
		return false
	}
	return true
}

// unmarshal a byte array to its corresponding object
func FromDataToSpec[T any](byteValue []byte, t T) (*T, error) {
	var d T
	if err := json.Unmarshal(byteValue, &d); err != nil {
		return nil, err
	}
	return &d, nil
}
