package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/llm-inferno/model-trainer/pkg/config"
	"github.com/llm-inferno/model-trainer/pkg/core"
	"github.com/llm-inferno/model-trainer/pkg/utils"
)

func main() {
	filePath := "../../samples/qm_test_s4.json"
	if len(os.Args) > 1 {
		filePath = os.Args[1]
	}

	bytes_acc, err_acc := os.ReadFile(filePath)
	if err_acc != nil {
		fmt.Println(err_acc)
	}

	dataSet, err := utils.FromDataToSpec(bytes_acc, core.DataSet{})
	if err != nil {
		fmt.Println(err)
		return
	}
	dataSet.Fix()
	dataSet.ToMSecs()
	// fmt.Println(dataSet.DataSetPrettyPrint())

	parms := &config.ModelParams{
		Alpha: 6.037069773706014,
		Beta:  0.012351457184339894,
		Gamma: 0.00003754977789772998,
	}

	analyzer := core.NewAnalyzer(parms)
	analyzerResults := analyzer.Analyze(dataSet, core.Model)

	fmt.Println("Testing completed successfully!")
	fmt.Println("-------------------------------")
	fmt.Printf("Name of data set: %s\n", dataSet.Name)
	fmt.Printf("Number of data points: %d\n", len(dataSet.Data))
	if jsonStr, err := json.Marshal(parms); err == nil {
		fmt.Printf("Parameters used: %v\n", string(jsonStr))
	}
	fmt.Println("Testing analysis results:")
	if jsonStr, err := json.Marshal(analyzerResults); err == nil {
		fmt.Println(string(jsonStr))
	}
}
