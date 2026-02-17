package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/llm-inferno/model-trainer/pkg/config"
	"github.com/llm-inferno/model-trainer/pkg/core"
	"github.com/llm-inferno/model-trainer/pkg/reader"
)

const DefaultFileName = "../../experiments/exp2/csv/sweep-i64-o64/benchmarks.html"

func main() {
	filePath := DefaultFileName
	if len(os.Args) > 1 {
		filePath = os.Args[1]
	}

	dataBytes, err_acc := os.ReadFile(filePath)
	if err_acc != nil {
		fmt.Println("Error reading file:", err_acc)
		return
	}

	var dataReader reader.Reader
	dataReader = reader.NewGuideLLMHTMLData()
	if err := dataReader.ReadFrom(dataBytes); err != nil {
		fmt.Println("Error reading HTML data:", err)
		return
	}

	fmt.Println("Successfully read GuideLLM HTML data")
	fmt.Println("===============================" + "==============================")
	// dataReader.Print()
	// fmt.Println("===============================" + "==============================")

	dataSet := dataReader.CreateDataSet()
	// fmt.Println(dataSet.DataSetPrettyPrint())

	initParms := &config.ModelParams{
		Alpha: 10.0,
		Beta:  0.0,
		Gamma: 0.0,
	}

	optimizer := core.NewOptimizer(initParms)
	optimizerResult, err := optimizer.Optimize(dataSet, core.Model)
	if err != nil {
		fmt.Println("Optimization failed:", err)
		return
	}

	fmt.Println("Optimization completed successfully!")
	fmt.Println("-------------------------------")
	fmt.Printf("Name of data set: %s\n", dataSet.Name)
	fmt.Printf("Number of data points: %d\n", len(dataSet.Data))
	if jsonStr, err := json.Marshal(initParms); err == nil {
		fmt.Printf("Initial parameters: %v\n", string(jsonStr))
	}
	fmt.Println("Estimated parameters:")
	if jsonStr, err := json.Marshal(optimizerResult); err == nil {
		fmt.Println(string(jsonStr))
	}
	// Print a summary line
	fmt.Printf("Summary: alpha: %.6f, beta: %.9f, gamma: %.12f, errTTFT: %.6f, errITL: %.6f, errWeightedAvg: %.6f\n",
		optimizerResult.OptimizedParms.Alpha,
		optimizerResult.OptimizedParms.Beta,
		optimizerResult.OptimizedParms.Gamma,
		optimizerResult.AnalysisResults.AvgErrTTFT,
		optimizerResult.AnalysisResults.AvgErrITL,
		optimizerResult.AnalysisResults.AvgErrWeighted)
}
