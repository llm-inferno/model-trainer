package core

import (
	"github.com/llm-inferno/model-trainer/pkg/config"
	"github.com/llm-inferno/model-trainer/pkg/utils"
)

// Analyzer performs data analysis using a parametrized model
type Analyzer struct {
	Parms *config.ModelParams
}

func NewAnalyzer(parms *config.ModelParams) *Analyzer {
	return &Analyzer{
		Parms: parms,
	}
}

// Analyze computes the error metrics for the given dataset and model function
func (a *Analyzer) Analyze(dataSet *DataSet, model ModelFunction) *config.AnalysisResults {
	xData, yData := dataSet.GetInOutVars()
	errVars := &config.ErrorVars{}
	LossFunction(a.Parms, xData, yData, model, errVars, true)
	return utils.CreateAnalysisResultsFromErrorVars(errVars)
}
