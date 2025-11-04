package main

import (
	"fmt"
	"os"

	"github.com/llm-inferno/model-trainer/pkg/config"
	"github.com/llm-inferno/model-trainer/pkg/core"
	"github.com/llm-inferno/model-trainer/pkg/utils"
)

func main() {
	filePath := "../../samples/data.json"
	if len(os.Args) > 1 {
		filePath = os.Args[1]
	}

	bytes_acc, err_acc := os.ReadFile(filePath)
	if err_acc != nil {
		fmt.Println(err_acc)
	}

	dataSet, err := utils.FromDataToSpec(bytes_acc, config.DataSet{})
	if err != nil {
		fmt.Println(err)
		return
	}

	initParms := &config.ModelParams{
		Alpha: 8.0,
		Beta:  0.05,
		Gamma: 16.0,
		Delta: 0.005,
	}

	optimizer := core.NewOptimizer(initParms)
	optimizerResult, err := optimizer.Optimize(dataSet, core.Model)
	if err != nil {
		fmt.Println("Optimization failed:", err)
		return
	}
	fmt.Printf("Final optimized parameters: %+v\n", optimizerResult.OptimizedParms)
	fmt.Printf("Final cost: %f\n", optimizerResult.Cost)
}
