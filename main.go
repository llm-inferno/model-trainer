package main

import "github.com/llm-inferno/model-trainer/pkg/service"

// create and run a model trainer service
func main() {
	trainer := service.NewTrainer()
	trainer.Run()
}
