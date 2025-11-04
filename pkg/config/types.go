package config

type DataPoint struct {
	RequestRate    float64 `json:"requestRate"`    // request arrival rate (requests/sec)
	InputTokens    float64 `json:"inputTokens"`    // average number of input tokens per request
	OutputTokens   float64 `json:"outputTokens"`   // average number of output tokens per request
	AvgWaitTime    float64 `json:"avgWaitTime"`    // average queueing time (msec)
	AvgPrefillTime float64 `json:"avgPrefillTime"` // average prefill time (msec)
	AvgITLTime     float64 `json:"avgITLTime"`     // average inter-token latency (msec)
	MaxBatchSize   int     `json:"maxBatchSize"`   // maximum batch size
}

type DataSet struct {
	Name string      `json:"name"`
	Data []DataPoint `json:"data"`
}

type ModelParams struct {
	Alpha float64 `json:"alpha"` // decode parameter: base
	Beta  float64 `json:"beta"`  // decode parameter: slope
	Gamma float64 `json:"gamma"` // prefill parameter: base
	Delta float64 `json:"delta"` // prefill parameter: slope
}

type InputVars struct {
	RequestRate  float64 `json:"requestRate"`  // request arrival rate (requests/sec)
	InputTokens  float64 `json:"inputTokens"`  // average number of input tokens per request
	OutputTokens float64 `json:"outputTokens"` // average number of output tokens per request
	MaxBatchSize int     `json:"maxBatchSize"` // maximum batch size
}

type OutputVars struct {
	AvgWaitTime    float64 `json:"avgWaitTime"`    // average queueing time (msec)
	AvgPrefillTime float64 `json:"avgPrefillTime"` // average prefill time (msec)
	AvgITLTime     float64 `json:"avgITLTime"`     // average inter-token latency (msec)
}
