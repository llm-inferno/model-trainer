package core

import "github.com/llm-inferno/model-trainer/pkg/config"

// data point representing a single experiment (benchmark) of input and output variables
type DataPoint struct {
	RequestRate  float64 `json:"requestRate"`  // request arrival rate (requests/sec)
	InputTokens  float64 `json:"inputTokens"`  // average number of input tokens per request
	OutputTokens float64 `json:"outputTokens"` // average number of output tokens per request
	AvgITLTime   float64 `json:"avgITLTime"`   // average inter-token latency (msec)
	AvgTTFTTime  float64 `json:"avgTTFTTime"`  // average time to first token (msec)

	// if TTFT not available, then use wait time + prefill time
	AvgWaitTime    float64 `json:"avgWaitTime"`    // average queueing time (msec)
	AvgPrefillTime float64 `json:"avgPrefillTime"` // average prefill time (msec)

	MaxBatchSize int `json:"maxBatchSize"` // maximum batch size
	MaxNumTokens int `json:"maxNumTokens"` // maximum number of tokens in a batch
}

// converting from a data point struct to input and output variables
func (dataPoint *DataPoint) GetInOutVars() (x *config.InputVars, y *config.OutputVars) {
	dataPoint.Fix()
	x = &config.InputVars{
		RequestRate:  dataPoint.RequestRate,
		InputTokens:  dataPoint.InputTokens,
		OutputTokens: dataPoint.OutputTokens,
		MaxBatchSize: dataPoint.MaxBatchSize,
		MaxNumTokens: dataPoint.MaxNumTokens,
	}
	y = &config.OutputVars{
		AvgTTFTTime: dataPoint.AvgTTFTTime,
		AvgITLTime:  dataPoint.AvgITLTime,
	}
	return x, y
}

// fix any missing or invalid fields in the data point
func (dataPoint *DataPoint) Fix() {
	if dataPoint.MaxBatchSize <= 0 {
		dataPoint.MaxBatchSize = config.DefaultMaxBatchSize
	}
	if dataPoint.MaxNumTokens <= 0 {
		dataPoint.MaxNumTokens = config.DefaultMaxNumTokens
	}
	if dataPoint.AvgTTFTTime <= 0 {
		dataPoint.AvgTTFTTime = dataPoint.AvgWaitTime + dataPoint.AvgPrefillTime
	}
}

// convert time fields from seconds to milliseconds
func (dataPoint *DataPoint) ToMSecs() {
	dataPoint.AvgTTFTTime *= 1000
	dataPoint.AvgITLTime *= 1000
	dataPoint.AvgWaitTime *= 1000
	dataPoint.AvgPrefillTime *= 1000
}
