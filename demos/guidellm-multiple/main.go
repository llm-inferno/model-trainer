package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/llm-inferno/model-trainer/pkg/config"
	"github.com/llm-inferno/model-trainer/pkg/core"
	"github.com/llm-inferno/model-trainer/pkg/reader"
)

const (
	FileNameSeparator  = "$"
	DefaultFileName    = "../../samples/guidellm.json"
	DefaultDataSetName = "GuideLLM multiple data sets"
)

func main() {
	fileNames := []string{DefaultFileName}
	if len(os.Args) > 1 {
		fileNames = strings.Split(os.Args[1], FileNameSeparator)
	}

	var dataReader reader.Reader
	dataSet := core.NewDataSet(DefaultDataSetName)

	for _, fileName := range fileNames {
		fn := fileName
		fmt.Printf("Processing file: %s\n", fn)
		dataBytes, err_acc := os.ReadFile(fn)
		if err_acc != nil {
			fmt.Println(err_acc)
			continue
		}

		dataReader = reader.NewGuideLLMData()
		if err := dataReader.ReadFrom(dataBytes); err != nil {
			fmt.Println("GuideLLM data not in json output format, Trying CSV ...")
			dataReader = reader.NewGuideLLMCSVData()
			if err := dataReader.ReadFrom(dataBytes); err != nil {
				fmt.Println(err)
				continue
			}
		}
		dataSet.Merge(dataReader.CreateDataSet())
	}

	if len(dataSet.Data) == 0 {
		fmt.Println("No data to process.")
		return
	}
	// Print the data set
	dataSet.Fix()
	// fmt.Println(dataSet.DataSetPrettyPrint())

	// initParms := &config.ModelParams{
	// 	Alpha: 6.0,
	// 	Beta:  0.0,
	// 	Gamma: 0.0,
	// }

	initParms := &config.ModelParams{
		Alpha: 6.7,
		Beta:  0.024,
		Gamma: 0.000048,
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
