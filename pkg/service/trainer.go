package service

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/llm-inferno/model-trainer/pkg/config"
	"github.com/llm-inferno/model-trainer/pkg/core"
)

// REST server for the model trainer
type Trainer struct {
	router *gin.Engine
}

// create a new Trainer
func NewTrainer() *Trainer {
	trainer := &Trainer{
		router: gin.Default(),
	}
	trainer.router.POST("/train", train)
	return trainer
}

// start service
func (trainer *Trainer) Run() {
	trainer.router.Run(":8080")
}

// train using a data set
func train(c *gin.Context) {
	// get data set
	dataSet := config.DataSet{}
	if err := c.BindJSON(&dataSet); err != nil {
		c.IndentedJSON(http.StatusBadRequest, gin.H{"message": "binding error: " + err.Error()})
		return
	}

	// TODO: provide means to set initial values
	initParms := &config.ModelParams{
		Alpha: 1.0,
		Beta:  0.05,
		Gamma: 10.0,
		Delta: 0.005,
	}

	optimizer := core.NewOptimizer(initParms)
	optimizerResult, err := optimizer.Optimize(&dataSet, core.Model)
	if err != nil {
		c.IndentedJSON(http.StatusInternalServerError,
			gin.H{"message": "optimization failed: " + err.Error()})
		return

	}
	c.IndentedJSON(http.StatusOK, optimizerResult)
}
