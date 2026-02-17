package config

const (
	// default maximum ratio of queue size (waiting room before receiving service)
	// to max batch size (processing concurrency capacity)
	DefaultMaxQueueToMaxBatchRatio = 1

	// default maximum batch size of requests in an iteration
	DefaultMaxBatchSize = 256

	// default maximum number of tokens (input and output) in an iteration
	DefaultMaxNumTokens = 8192

	// default weight for TTFT2ITL in calculating average weighted deviation error
	// errWeightedAvg = (errTTFT * TTFT2ITLWeight + errITL) / (TTFT2ITLWeight + 1)
	TTFT2ITLWeight = 0.5

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
