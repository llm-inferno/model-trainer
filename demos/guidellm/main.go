package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/llm-inferno/model-trainer/pkg/config"
	"github.com/llm-inferno/model-trainer/pkg/core"
	"github.com/llm-inferno/model-trainer/pkg/reader"
	"github.com/llm-inferno/model-trainer/pkg/utils"
)

func main() {
	filePath := "../../samples/guidellm.json"
	if len(os.Args) > 1 {
		filePath = os.Args[1]
	}

	dataBytes, err_acc := os.ReadFile(filePath)
	if err_acc != nil {
		fmt.Println(err_acc)
		return
	}

	reader := reader.NewGuideLLMData()
	if err := reader.ReadFrom(dataBytes); err != nil {
		fmt.Println(err)
		return
	}
	dataSet := reader.CreateDataSet()
	fmt.Println(utils.DataSetPrettyPrint(dataSet))

	initParms := &config.ModelParams{
		Alpha: 1.0,
		Beta:  0.1,
		Gamma: 10.0,
		Delta: 0.01,
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
}
