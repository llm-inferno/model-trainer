package reader

import "github.com/llm-inferno/model-trainer/pkg/core"

// Converts from a given data format to the Data Set format
type Reader interface {
	ReadFrom(dataBytes []byte) error
	CreateDataSet() *core.DataSet
	Print()
	Dump() string
}
