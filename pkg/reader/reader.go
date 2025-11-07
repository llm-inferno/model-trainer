package reader

import "github.com/llm-inferno/model-trainer/pkg/config"

// Converts from a given data format to the Data Set format
type Reader interface {
	ReadFrom(dataBytes []byte) error
	CreateDataSet() *config.DataSet
	Print()
	Dump() string
}
