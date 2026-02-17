package config

// model parameters, unknown quantities to be estimated
type ModelParams struct {
	Alpha float64 `json:"alpha"` // base
	Beta  float64 `json:"beta"`  // slope for compute time
	Gamma float64 `json:"gamma"` // slope for memory access time
}

// input variables representing an experiment input
type InputVars struct {
	RequestRate  float64 `json:"requestRate"`  // request arrival rate (requests/sec)
	InputTokens  float64 `json:"inputTokens"`  // average number of input tokens per request
	OutputTokens float64 `json:"outputTokens"` // average number of output tokens per request
	MaxBatchSize int     `json:"maxBatchSize"` // maximum batch size
	MaxNumTokens int     `json:"maxNumTokens"` // maximum number of tokens in a batch
}

// output variables representing an experiment output (performance metrics)
type OutputVars struct {
	AvgTTFTTime float64 `json:"avgTTFTTime"` // average time to first token (msec)
	AvgITLTime  float64 `json:"avgITLTime"`  // average inter-token latency (msec)
}

// error variables representing the (absolute) difference between predicted and observed output
type ErrorVars struct {
	Count               int     `json:"count"`               // number of data points
	CumErrorTTFT        float64 `json:"cumErrorTTFT"`        // Cumulative error for TTFT time (msec)
	CumErrorITL         float64 `json:"cumErrorITL"`         // Cumulative error for ITL time (msec)
	CumErrorWeightedAvg float64 `json:"cumErrorWeightedAvg"` // Cumulative weighted error (msec)
}

// analysis results after processing error variables
type AnalysisResults struct {
	AvgErrTTFT     float64 `json:"avgErrTTFT"`     // Average error for TTFT time (msec)
	AvgErrITL      float64 `json:"avgErrITL"`      // Average error for ITL time (msec)
	AvgErrWeighted float64 `json:"avgErrWeighted"` // Weighted average average error (msec)
}
