package config

const (
	// default maximum ratio of queue size (waiting room before receiving service)
	// to max batch size (processing concurrency capacity)
	MaxQueueToMaxBatchRatio = 1
)

// indexes of parameters in the oparameters array
type ParamIndex int

const (
	IndexAlpha ParamIndex = iota
	IndexBeta
	IndexGamma
	IndexDelta
)
