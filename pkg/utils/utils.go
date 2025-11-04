package utils

import (
	"encoding/json"

	"github.com/llm-inferno/model-trainer/pkg/config"
)

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

func CreateInOutVarsFromDataSet(dataSet *config.DataSet) (xData []*config.InputVars, yData []*config.OutputVars) {
	for _, dp := range dataSet.Data {
		x, y := CreateInOutVarsFromDataPoint(&dp)
		xData = append(xData, x)
		yData = append(yData, y)
	}
	return xData, yData
}

func CreateParmsSliceFromModelParams(params *config.ModelParams) []float64 {
	return []float64{params.Alpha, params.Beta, params.Gamma, params.Delta}
}

func CreateModelParamsFromParmsSlice(parms []float64) *config.ModelParams {
	return &config.ModelParams{
		Alpha: parms[0],
		Beta:  parms[1],
		Gamma: parms[2],
		Delta: parms[3],
	}
}

// unmarshal a byte array to its corresponding object
func FromDataToSpec[T any](byteValue []byte, t T) (*T, error) {
	var d T
	if err := json.Unmarshal(byteValue, &d); err != nil {
		return nil, err
	}
	return &d, nil
}

func CheckParmsValid(params *config.ModelParams) bool {
	if params.Alpha < 0 || params.Beta < 0 || params.Gamma < 0 || params.Delta < 0 {
		return false
	}
	return true
}
