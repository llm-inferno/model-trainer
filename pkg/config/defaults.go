package config

const (
	// default maximum ratio of queue size (waiting room before receiving service)
	// to max batch size (processing concurrency capacity)
	DefaultMaxQueueToMaxBatchRatio = 1

	// default maximum batch size of requests in an iteration
	DefaultMaxBatchSize = 256

	// default maximum number of tokens (input and output) in an iteration
	DefaultMaxNumTokens = 8192

	// default maximum number of iterations allowed in the optimizer
	DefaultNumberOptimizationIterations = 1000
)

// indexes of parameters in the parameters array
type ParamIndex int

const (
	IndexAlpha ParamIndex = iota
	IndexBeta
	IndexGamma
)
