package core

import (
	"fmt"
	"math"

	"github.com/llm-inferno/queue-analysis/pkg/analyzer"
)

func Analyze(qa *analyzer.LLMQueueAnalyzer, requestRate float32) (metrics *analyzer.AnalysisMetrics, err error) {
	if requestRate <= 0 {
		return nil, fmt.Errorf("invalid request rate %v", requestRate)
	}
	rateRange := qa.RateRange
	if requestRate > rateRange.Max {
		err = fmt.Errorf("rate=%v, max allowed rate=%v", requestRate, rateRange.Max)
		return nil, err
	}

	alpha := float64(qa.ServiceParms.Alpha)
	beta := float64(qa.ServiceParms.Beta)
	gamma := float64(qa.ServiceParms.Gamma)
	inTokens := float64(qa.RequestSize.AvgInputTokens)
	outTokens := float64(qa.RequestSize.AvgOutputTokens)

	lambda := float64(requestRate / 1000) // convert to req/ms
	maxTokens := float64(8192)
	m := math.Ceil((inTokens + outTokens) / maxTokens)
	load1 := lambda * beta * (inTokens + outTokens)
	load2 := lambda * gamma * (inTokens + outTokens/2) * (outTokens - 1)
	denom := 1 - (load1 + load2)
	if denom <= 0 {
		err = fmt.Errorf("system unstable at rate=%v", requestRate)
		return nil, err
	}
	avgT := alpha / denom
	avgITL := beta + avgT
	avgTTFT := beta*(inTokens) + (m+1)*avgT

	avgServTime := avgTTFT + ((outTokens)-1)*avgITL
	avgNumInServ := lambda * avgServTime
	rho := avgNumInServ / float64(qa.MaxBatchSize)
	rho = min(max(rho, 0), 1)

	// return solution
	metrics = &analyzer.AnalysisMetrics{
		Throughput:     requestRate,
		AvgRespTime:    float32(avgServTime),
		AvgWaitTime:    0,
		AvgNumInServ:   float32(avgNumInServ),
		AvgPrefillTime: float32(avgTTFT),
		AvgTokenTime:   float32(avgITL),
		MaxRate:        rateRange.Max,
		Rho:            float32(rho),
	}
	return metrics, nil
}
