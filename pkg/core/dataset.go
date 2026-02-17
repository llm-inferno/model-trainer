package core

import (
	"bytes"
	"fmt"

	"github.com/llm-inferno/model-trainer/pkg/config"
)

// a collection of data points representing a data set
type DataSet struct {
	Name string      `json:"name"`
	Data []DataPoint `json:"data"`
}

func NewDataSet(name string) *DataSet {
	return &DataSet{
		Name: name,
		Data: []DataPoint{}}
}

// append a data point to the data set
func (dataSet *DataSet) AppendDataPoint(dataPoint *DataPoint) {
	dataSet.Data = append(dataSet.Data, *dataPoint)
}

// converting from a data set struct to input and output variable arrays
func (dataSet *DataSet) GetInOutVars() (xData []*config.InputVars, yData []*config.OutputVars) {
	for _, dp := range dataSet.Data {
		x, y := dp.GetInOutVars()
		xData = append(xData, x)
		yData = append(yData, y)
	}
	return xData, yData
}

// merge another data set into the current data set
func (dataSet *DataSet) Merge(other *DataSet) {
	dataSet.Data = append(dataSet.Data, other.Data...)
}

// get the size of the data set
func (dataSet *DataSet) Size() int {
	return len(dataSet.Data)
}

// fix data points in the data set
func (dataSet *DataSet) Fix() {
	for i := range dataSet.Data {
		dataSet.Data[i].Fix()
	}
}

// convert time units in the data set to milliseconds
func (dataSet *DataSet) ToMSecs() {
	for i := range dataSet.Data {
		dataSet.Data[i].ToMSecs()
	}
}

// pretty print a data set
func (ds *DataSet) DataSetPrettyPrint() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "DataSet: %s \n", ds.Name)
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "  rps \t inToken \t outToken \t TTFT(msec) \t ITL(msec) \t maxBatch \t maxTokens\n")
	fmt.Fprintln(&b)
	for _, p := range ds.Data {
		fmt.Fprintf(&b, "%6.2f \t %8.2f \t %8.2f \t %8.2f \t %8.2f \t %d \t\t %d\n",
			p.RequestRate, p.InputTokens, p.OutputTokens, p.AvgTTFTTime,
			p.AvgITLTime, p.MaxBatchSize, p.MaxNumTokens)
	}
	return b.String()
}
