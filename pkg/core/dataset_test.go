package core

import (
	"strings"
	"testing"
)

func TestNewDataSet(t *testing.T) {
	tests := []struct {
		name         string
		dataSetName  string
		wantName     string
		wantDataSize int
	}{
		{
			name:         "create dataset with name",
			dataSetName:  "test-dataset",
			wantName:     "test-dataset",
			wantDataSize: 0,
		},
		{
			name:         "create dataset with empty name",
			dataSetName:  "",
			wantName:     "",
			wantDataSize: 0,
		},
		{
			name:         "create dataset with long name",
			dataSetName:  "very-long-dataset-name-with-many-characters",
			wantName:     "very-long-dataset-name-with-many-characters",
			wantDataSize: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ds := NewDataSet(tt.dataSetName)

			if ds == nil {
				t.Fatal("NewDataSet returned nil")
			}

			if ds.Name != tt.wantName {
				t.Errorf("Name = %v, want %v", ds.Name, tt.wantName)
			}

			if ds.Data == nil {
				t.Error("Data slice is nil")
			}

			if len(ds.Data) != tt.wantDataSize {
				t.Errorf("Data size = %v, want %v", len(ds.Data), tt.wantDataSize)
			}
		})
	}
}

func TestDataSet_AppendDataPoint(t *testing.T) {
	tests := []struct {
		name       string
		initial    []DataPoint
		toAppend   []DataPoint
		wantSize   int
		validateFn func(t *testing.T, ds *DataSet)
	}{
		{
			name:     "append single data point to empty dataset",
			initial:  []DataPoint{},
			toAppend: []DataPoint{
				{
					RequestRate:  10.0,
					InputTokens:  100.0,
					OutputTokens: 50.0,
					AvgTTFTTime:  10.0,
					AvgITLTime:   5.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				},
			},
			wantSize: 1,
			validateFn: func(t *testing.T, ds *DataSet) {
				if ds.Data[0].RequestRate != 10.0 {
					t.Errorf("RequestRate = %v, want 10.0", ds.Data[0].RequestRate)
				}
			},
		},
		{
			name: "append multiple data points",
			initial: []DataPoint{
				{
					RequestRate:  5.0,
					InputTokens:  50.0,
					OutputTokens: 25.0,
				},
			},
			toAppend: []DataPoint{
				{
					RequestRate:  10.0,
					InputTokens:  100.0,
					OutputTokens: 50.0,
				},
				{
					RequestRate:  15.0,
					InputTokens:  150.0,
					OutputTokens: 75.0,
				},
			},
			wantSize: 3,
			validateFn: func(t *testing.T, ds *DataSet) {
				if ds.Data[0].RequestRate != 5.0 {
					t.Errorf("First data point RequestRate = %v, want 5.0", ds.Data[0].RequestRate)
				}
				if ds.Data[2].RequestRate != 15.0 {
					t.Errorf("Third data point RequestRate = %v, want 15.0", ds.Data[2].RequestRate)
				}
			},
		},
		{
			name:     "append data point with zero values",
			initial:  []DataPoint{},
			toAppend: []DataPoint{
				{
					RequestRate:  0.0,
					InputTokens:  0.0,
					OutputTokens: 0.0,
				},
			},
			wantSize: 1,
			validateFn: func(t *testing.T, ds *DataSet) {
				if ds.Data[0].RequestRate != 0.0 {
					t.Errorf("RequestRate = %v, want 0.0", ds.Data[0].RequestRate)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ds := NewDataSet("test")

			// Add initial data points
			for i := range tt.initial {
				ds.AppendDataPoint(&tt.initial[i])
			}

			// Append test data points
			for i := range tt.toAppend {
				ds.AppendDataPoint(&tt.toAppend[i])
			}

			if ds.Size() != tt.wantSize {
				t.Errorf("Size = %v, want %v", ds.Size(), tt.wantSize)
			}

			if tt.validateFn != nil {
				tt.validateFn(t, ds)
			}
		})
	}
}

func TestDataSet_GetInOutVars(t *testing.T) {
	tests := []struct {
		name       string
		dataPoints []DataPoint
		wantXSize  int
		wantYSize  int
		validateFn func(t *testing.T, ds *DataSet)
	}{
		{
			name:       "empty dataset",
			dataPoints: []DataPoint{},
			wantXSize:  0,
			wantYSize:  0,
		},
		{
			name: "single data point",
			dataPoints: []DataPoint{
				{
					RequestRate:  10.0,
					InputTokens:  100.0,
					OutputTokens: 50.0,
					AvgTTFTTime:  15.0,
					AvgITLTime:   5.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				},
			},
			wantXSize: 1,
			wantYSize: 1,
			validateFn: func(t *testing.T, ds *DataSet) {
				xData, yData := ds.GetInOutVars()
				if xData[0].RequestRate != 10.0 {
					t.Errorf("xData[0].RequestRate = %v, want 10.0", xData[0].RequestRate)
				}
				if yData[0].AvgTTFTTime != 15.0 {
					t.Errorf("yData[0].AvgTTFTTime = %v, want 15.0", yData[0].AvgTTFTTime)
				}
			},
		},
		{
			name: "multiple data points",
			dataPoints: []DataPoint{
				{
					RequestRate:  10.0,
					InputTokens:  100.0,
					OutputTokens: 50.0,
					AvgTTFTTime:  10.0,
					AvgITLTime:   5.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				},
				{
					RequestRate:  20.0,
					InputTokens:  200.0,
					OutputTokens: 100.0,
					AvgTTFTTime:  20.0,
					AvgITLTime:   10.0,
					MaxBatchSize: 64,
					MaxNumTokens: 4096,
				},
				{
					RequestRate:  30.0,
					InputTokens:  300.0,
					OutputTokens: 150.0,
					AvgTTFTTime:  30.0,
					AvgITLTime:   15.0,
					MaxBatchSize: 128,
					MaxNumTokens: 8192,
				},
			},
			wantXSize: 3,
			wantYSize: 3,
			validateFn: func(t *testing.T, ds *DataSet) {
				xData, yData := ds.GetInOutVars()

				// Check first data point
				if xData[0].RequestRate != 10.0 {
					t.Errorf("xData[0].RequestRate = %v, want 10.0", xData[0].RequestRate)
				}
				if yData[0].AvgITLTime != 5.0 {
					t.Errorf("yData[0].AvgITLTime = %v, want 5.0", yData[0].AvgITLTime)
				}

				// Check last data point
				if xData[2].RequestRate != 30.0 {
					t.Errorf("xData[2].RequestRate = %v, want 30.0", xData[2].RequestRate)
				}
				if yData[2].AvgTTFTTime != 30.0 {
					t.Errorf("yData[2].AvgTTFTTime = %v, want 30.0", yData[2].AvgTTFTTime)
				}
			},
		},
		{
			name: "data point with missing fields - should be fixed",
			dataPoints: []DataPoint{
				{
					RequestRate:    10.0,
					InputTokens:    100.0,
					OutputTokens:   50.0,
					AvgTTFTTime:    0.0,
					AvgITLTime:     5.0,
					AvgWaitTime:    3.0,
					AvgPrefillTime: 2.0,
					MaxBatchSize:   0,
					MaxNumTokens:   0,
				},
			},
			wantXSize: 1,
			wantYSize: 1,
			validateFn: func(t *testing.T, ds *DataSet) {
				xData, yData := ds.GetInOutVars()
				// Should use default values
				if xData[0].MaxBatchSize == 0 {
					t.Error("MaxBatchSize should be set to default value")
				}
				// Should compute TTFT from wait + prefill
				if yData[0].AvgTTFTTime != 5.0 {
					t.Errorf("AvgTTFTTime = %v, want 5.0 (computed from wait+prefill)", yData[0].AvgTTFTTime)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ds := NewDataSet("test")
			for i := range tt.dataPoints {
				ds.AppendDataPoint(&tt.dataPoints[i])
			}

			xData, yData := ds.GetInOutVars()

			if len(xData) != tt.wantXSize {
				t.Errorf("xData size = %v, want %v", len(xData), tt.wantXSize)
			}

			if len(yData) != tt.wantYSize {
				t.Errorf("yData size = %v, want %v", len(yData), tt.wantYSize)
			}

			if tt.validateFn != nil {
				tt.validateFn(t, ds)
			}
		})
	}
}

func TestDataSet_Merge(t *testing.T) {
	tests := []struct {
		name       string
		dataset1   []DataPoint
		dataset2   []DataPoint
		wantSize   int
		validateFn func(t *testing.T, ds *DataSet)
	}{
		{
			name: "merge two non-empty datasets",
			dataset1: []DataPoint{
				{RequestRate: 10.0, InputTokens: 100.0},
				{RequestRate: 20.0, InputTokens: 200.0},
			},
			dataset2: []DataPoint{
				{RequestRate: 30.0, InputTokens: 300.0},
				{RequestRate: 40.0, InputTokens: 400.0},
			},
			wantSize: 4,
			validateFn: func(t *testing.T, ds *DataSet) {
				if ds.Data[0].RequestRate != 10.0 {
					t.Errorf("Data[0].RequestRate = %v, want 10.0", ds.Data[0].RequestRate)
				}
				if ds.Data[3].RequestRate != 40.0 {
					t.Errorf("Data[3].RequestRate = %v, want 40.0", ds.Data[3].RequestRate)
				}
			},
		},
		{
			name: "merge empty dataset into non-empty",
			dataset1: []DataPoint{
				{RequestRate: 10.0, InputTokens: 100.0},
			},
			dataset2: []DataPoint{},
			wantSize: 1,
			validateFn: func(t *testing.T, ds *DataSet) {
				if ds.Data[0].RequestRate != 10.0 {
					t.Errorf("Data[0].RequestRate = %v, want 10.0", ds.Data[0].RequestRate)
				}
			},
		},
		{
			name:     "merge non-empty dataset into empty",
			dataset1: []DataPoint{},
			dataset2: []DataPoint{
				{RequestRate: 30.0, InputTokens: 300.0},
			},
			wantSize: 1,
			validateFn: func(t *testing.T, ds *DataSet) {
				if ds.Data[0].RequestRate != 30.0 {
					t.Errorf("Data[0].RequestRate = %v, want 30.0", ds.Data[0].RequestRate)
				}
			},
		},
		{
			name:     "merge two empty datasets",
			dataset1: []DataPoint{},
			dataset2: []DataPoint{},
			wantSize: 0,
		},
		{
			name: "merge large datasets",
			dataset1: []DataPoint{
				{RequestRate: 1.0},
				{RequestRate: 2.0},
				{RequestRate: 3.0},
				{RequestRate: 4.0},
				{RequestRate: 5.0},
			},
			dataset2: []DataPoint{
				{RequestRate: 6.0},
				{RequestRate: 7.0},
				{RequestRate: 8.0},
				{RequestRate: 9.0},
				{RequestRate: 10.0},
			},
			wantSize: 10,
			validateFn: func(t *testing.T, ds *DataSet) {
				if ds.Data[4].RequestRate != 5.0 {
					t.Errorf("Data[4].RequestRate = %v, want 5.0", ds.Data[4].RequestRate)
				}
				if ds.Data[9].RequestRate != 10.0 {
					t.Errorf("Data[9].RequestRate = %v, want 10.0", ds.Data[9].RequestRate)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ds1 := NewDataSet("dataset1")
			for i := range tt.dataset1 {
				ds1.AppendDataPoint(&tt.dataset1[i])
			}

			ds2 := NewDataSet("dataset2")
			for i := range tt.dataset2 {
				ds2.AppendDataPoint(&tt.dataset2[i])
			}

			ds1.Merge(ds2)

			if ds1.Size() != tt.wantSize {
				t.Errorf("Size after merge = %v, want %v", ds1.Size(), tt.wantSize)
			}

			if tt.validateFn != nil {
				tt.validateFn(t, ds1)
			}
		})
	}
}

func TestDataSet_Size(t *testing.T) {
	tests := []struct {
		name       string
		dataPoints []DataPoint
		wantSize   int
	}{
		{
			name:       "empty dataset",
			dataPoints: []DataPoint{},
			wantSize:   0,
		},
		{
			name: "single data point",
			dataPoints: []DataPoint{
				{RequestRate: 10.0},
			},
			wantSize: 1,
		},
		{
			name: "multiple data points",
			dataPoints: []DataPoint{
				{RequestRate: 10.0},
				{RequestRate: 20.0},
				{RequestRate: 30.0},
				{RequestRate: 40.0},
				{RequestRate: 50.0},
			},
			wantSize: 5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ds := NewDataSet("test")
			for i := range tt.dataPoints {
				ds.AppendDataPoint(&tt.dataPoints[i])
			}

			if ds.Size() != tt.wantSize {
				t.Errorf("Size() = %v, want %v", ds.Size(), tt.wantSize)
			}

			// Also verify with len(Data) directly
			if len(ds.Data) != tt.wantSize {
				t.Errorf("len(Data) = %v, want %v", len(ds.Data), tt.wantSize)
			}
		})
	}
}

func TestDataSet_Fix(t *testing.T) {
	tests := []struct {
		name       string
		dataPoints []DataPoint
		validateFn func(t *testing.T, ds *DataSet)
	}{
		{
			name: "fix missing batch size in all data points",
			dataPoints: []DataPoint{
				{RequestRate: 10.0, MaxBatchSize: 0, MaxNumTokens: 2048, AvgTTFTTime: 10.0},
				{RequestRate: 20.0, MaxBatchSize: 0, MaxNumTokens: 2048, AvgTTFTTime: 20.0},
			},
			validateFn: func(t *testing.T, ds *DataSet) {
				for i, dp := range ds.Data {
					if dp.MaxBatchSize == 0 {
						t.Errorf("Data[%d].MaxBatchSize should be fixed, got 0", i)
					}
				}
			},
		},
		{
			name: "fix missing TTFT from wait and prefill times",
			dataPoints: []DataPoint{
				{
					RequestRate:    10.0,
					AvgTTFTTime:    0.0,
					AvgWaitTime:    5.0,
					AvgPrefillTime: 3.0,
					MaxBatchSize:   32,
					MaxNumTokens:   2048,
				},
				{
					RequestRate:    20.0,
					AvgTTFTTime:    0.0,
					AvgWaitTime:    7.0,
					AvgPrefillTime: 4.0,
					MaxBatchSize:   32,
					MaxNumTokens:   2048,
				},
			},
			validateFn: func(t *testing.T, ds *DataSet) {
				if ds.Data[0].AvgTTFTTime != 8.0 {
					t.Errorf("Data[0].AvgTTFTTime = %v, want 8.0", ds.Data[0].AvgTTFTTime)
				}
				if ds.Data[1].AvgTTFTTime != 11.0 {
					t.Errorf("Data[1].AvgTTFTTime = %v, want 11.0", ds.Data[1].AvgTTFTTime)
				}
			},
		},
		{
			name: "fix mixed issues",
			dataPoints: []DataPoint{
				{
					RequestRate:    10.0,
					AvgTTFTTime:    0.0,
					AvgWaitTime:    2.0,
					AvgPrefillTime: 3.0,
					MaxBatchSize:   0,
					MaxNumTokens:   0,
				},
			},
			validateFn: func(t *testing.T, ds *DataSet) {
				if ds.Data[0].MaxBatchSize == 0 {
					t.Error("MaxBatchSize should be fixed")
				}
				if ds.Data[0].MaxNumTokens == 0 {
					t.Error("MaxNumTokens should be fixed")
				}
				if ds.Data[0].AvgTTFTTime != 5.0 {
					t.Errorf("AvgTTFTTime = %v, want 5.0", ds.Data[0].AvgTTFTTime)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ds := NewDataSet("test")
			for i := range tt.dataPoints {
				ds.AppendDataPoint(&tt.dataPoints[i])
			}

			ds.Fix()

			if tt.validateFn != nil {
				tt.validateFn(t, ds)
			}
		})
	}
}

func TestDataSet_ToMSecs(t *testing.T) {
	tests := []struct {
		name       string
		dataPoints []DataPoint
		validateFn func(t *testing.T, ds *DataSet)
	}{
		{
			name: "convert single data point time fields",
			dataPoints: []DataPoint{
				{
					RequestRate:    10.0,
					AvgTTFTTime:    1.5,
					AvgITLTime:     0.5,
					AvgWaitTime:    0.75,
					AvgPrefillTime: 0.25,
				},
			},
			validateFn: func(t *testing.T, ds *DataSet) {
				if ds.Data[0].AvgTTFTTime != 1500.0 {
					t.Errorf("AvgTTFTTime = %v, want 1500.0", ds.Data[0].AvgTTFTTime)
				}
				if ds.Data[0].AvgITLTime != 500.0 {
					t.Errorf("AvgITLTime = %v, want 500.0", ds.Data[0].AvgITLTime)
				}
				if ds.Data[0].AvgWaitTime != 750.0 {
					t.Errorf("AvgWaitTime = %v, want 750.0", ds.Data[0].AvgWaitTime)
				}
				if ds.Data[0].AvgPrefillTime != 250.0 {
					t.Errorf("AvgPrefillTime = %v, want 250.0", ds.Data[0].AvgPrefillTime)
				}
			},
		},
		{
			name: "convert multiple data points",
			dataPoints: []DataPoint{
				{
					RequestRate: 10.0,
					AvgTTFTTime: 1.0,
					AvgITLTime:  0.5,
				},
				{
					RequestRate: 20.0,
					AvgTTFTTime: 2.0,
					AvgITLTime:  1.0,
				},
			},
			validateFn: func(t *testing.T, ds *DataSet) {
				if ds.Data[0].AvgTTFTTime != 1000.0 {
					t.Errorf("Data[0].AvgTTFTTime = %v, want 1000.0", ds.Data[0].AvgTTFTTime)
				}
				if ds.Data[1].AvgTTFTTime != 2000.0 {
					t.Errorf("Data[1].AvgTTFTTime = %v, want 2000.0", ds.Data[1].AvgTTFTTime)
				}
			},
		},
		{
			name: "verify non-time fields unchanged",
			dataPoints: []DataPoint{
				{
					RequestRate:  10.0,
					InputTokens:  100.0,
					OutputTokens: 50.0,
					AvgTTFTTime:  1.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				},
			},
			validateFn: func(t *testing.T, ds *DataSet) {
				if ds.Data[0].RequestRate != 10.0 {
					t.Error("RequestRate should not change")
				}
				if ds.Data[0].InputTokens != 100.0 {
					t.Error("InputTokens should not change")
				}
				if ds.Data[0].MaxBatchSize != 32 {
					t.Error("MaxBatchSize should not change")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ds := NewDataSet("test")
			for i := range tt.dataPoints {
				ds.AppendDataPoint(&tt.dataPoints[i])
			}

			ds.ToMSecs()

			if tt.validateFn != nil {
				tt.validateFn(t, ds)
			}
		})
	}
}

func TestDataSet_DataSetPrettyPrint(t *testing.T) {
	tests := []struct {
		name           string
		dataSetName    string
		dataPoints     []DataPoint
		wantContains   []string
		wantNotContain []string
	}{
		{
			name:        "empty dataset",
			dataSetName: "empty-test",
			dataPoints:  []DataPoint{},
			wantContains: []string{
				"DataSet: empty-test",
				"rps",
				"inToken",
				"outToken",
				"TTFT(msec)",
				"ITL(msec)",
				"maxBatch",
				"maxTokens",
			},
		},
		{
			name:        "single data point",
			dataSetName: "single-test",
			dataPoints: []DataPoint{
				{
					RequestRate:  10.50,
					InputTokens:  100.00,
					OutputTokens: 50.00,
					AvgTTFTTime:  15.25,
					AvgITLTime:   5.75,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				},
			},
			wantContains: []string{
				"DataSet: single-test",
				"10.50",
				"100.00",
				"50.00",
				"15.25",
				"5.75",
				"32",
				"2048",
			},
		},
		{
			name:        "multiple data points",
			dataSetName: "multi-test",
			dataPoints: []DataPoint{
				{
					RequestRate:  5.0,
					InputTokens:  50.0,
					OutputTokens: 25.0,
					AvgTTFTTime:  10.0,
					AvgITLTime:   3.0,
					MaxBatchSize: 16,
					MaxNumTokens: 1024,
				},
				{
					RequestRate:  10.0,
					InputTokens:  100.0,
					OutputTokens: 50.0,
					AvgTTFTTime:  15.0,
					AvgITLTime:   5.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				},
			},
			wantContains: []string{
				"DataSet: multi-test",
				"5.00",
				"10.00",
				"50.00",
				"100.00",
				"16",
				"32",
				"1024",
				"2048",
			},
		},
		{
			name:        "dataset with zero values",
			dataSetName: "zero-test",
			dataPoints: []DataPoint{
				{
					RequestRate:  0.0,
					InputTokens:  0.0,
					OutputTokens: 0.0,
					AvgTTFTTime:  0.0,
					AvgITLTime:   0.0,
					MaxBatchSize: 0,
					MaxNumTokens: 0,
				},
			},
			wantContains: []string{
				"DataSet: zero-test",
				"0.00",
			},
		},
		{
			name:        "dataset with large values",
			dataSetName: "large-test",
			dataPoints: []DataPoint{
				{
					RequestRate:  999.99,
					InputTokens:  10000.00,
					OutputTokens: 5000.00,
					AvgTTFTTime:  999.99,
					AvgITLTime:   999.99,
					MaxBatchSize: 256,
					MaxNumTokens: 8192,
				},
			},
			wantContains: []string{
				"999.99",
				"10000.00",
				"5000.00",
				"256",
				"8192",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ds := NewDataSet(tt.dataSetName)
			for i := range tt.dataPoints {
				ds.AppendDataPoint(&tt.dataPoints[i])
			}

			output := ds.DataSetPrettyPrint()

			// Check that output contains expected strings
			for _, want := range tt.wantContains {
				if !strings.Contains(output, want) {
					t.Errorf("Output does not contain %q\nGot:\n%s", want, output)
				}
			}

			// Check that output doesn't contain unwanted strings
			for _, notWant := range tt.wantNotContain {
				if strings.Contains(output, notWant) {
					t.Errorf("Output should not contain %q\nGot:\n%s", notWant, output)
				}
			}

			// Verify output is not empty
			if len(output) == 0 {
				t.Error("DataSetPrettyPrint returned empty string")
			}
		})
	}
}

func TestDataSet_ChainedOperations(t *testing.T) {
	t.Run("append, merge, fix, and convert", func(t *testing.T) {
		// Create first dataset
		ds1 := NewDataSet("dataset1")
		ds1.AppendDataPoint(&DataPoint{
			RequestRate:    10.0,
			InputTokens:    100.0,
			AvgTTFTTime:    1.0, // seconds
			MaxBatchSize:   0,   // needs fix
			MaxNumTokens:   2048,
		})

		// Create second dataset
		ds2 := NewDataSet("dataset2")
		ds2.AppendDataPoint(&DataPoint{
			RequestRate:    20.0,
			InputTokens:    200.0,
			AvgTTFTTime:    2.0, // seconds
			MaxBatchSize:   32,
			MaxNumTokens:   0, // needs fix
		})

		// Merge
		ds1.Merge(ds2)

		if ds1.Size() != 2 {
			t.Fatalf("Size after merge = %v, want 2", ds1.Size())
		}

		// Fix
		ds1.Fix()

		if ds1.Data[0].MaxBatchSize == 0 {
			t.Error("First data point MaxBatchSize should be fixed")
		}
		if ds1.Data[1].MaxNumTokens == 0 {
			t.Error("Second data point MaxNumTokens should be fixed")
		}

		// Convert to milliseconds
		ds1.ToMSecs()

		if ds1.Data[0].AvgTTFTTime != 1000.0 {
			t.Errorf("First data point AvgTTFTTime = %v, want 1000.0", ds1.Data[0].AvgTTFTTime)
		}
		if ds1.Data[1].AvgTTFTTime != 2000.0 {
			t.Errorf("Second data point AvgTTFTTime = %v, want 2000.0", ds1.Data[1].AvgTTFTTime)
		}

		// Get input/output vars
		xData, yData := ds1.GetInOutVars()

		if len(xData) != 2 {
			t.Errorf("xData length = %v, want 2", len(xData))
		}
		if len(yData) != 2 {
			t.Errorf("yData length = %v, want 2", len(yData))
		}
	})
}

// Benchmark for AppendDataPoint
func BenchmarkDataSet_AppendDataPoint(b *testing.B) {
	ds := NewDataSet("test")
	dp := DataPoint{
		RequestRate:  10.0,
		InputTokens:  100.0,
		OutputTokens: 50.0,
		AvgTTFTTime:  10.0,
		AvgITLTime:   5.0,
		MaxBatchSize: 32,
		MaxNumTokens: 2048,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ds.AppendDataPoint(&dp)
	}
}

// Benchmark for GetInOutVars
func BenchmarkDataSet_GetInOutVars(b *testing.B) {
	ds := NewDataSet("test")
	for i := 0; i < 100; i++ {
		ds.AppendDataPoint(&DataPoint{
			RequestRate:  float64(i),
			InputTokens:  float64(i * 10),
			OutputTokens: float64(i * 5),
			AvgTTFTTime:  10.0,
			AvgITLTime:   5.0,
			MaxBatchSize: 32,
			MaxNumTokens: 2048,
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = ds.GetInOutVars()
	}
}

// Benchmark for Merge
func BenchmarkDataSet_Merge(b *testing.B) {
	ds1 := NewDataSet("dataset1")
	for i := 0; i < 50; i++ {
		ds1.AppendDataPoint(&DataPoint{RequestRate: float64(i)})
	}

	ds2 := NewDataSet("dataset2")
	for i := 0; i < 50; i++ {
		ds2.AppendDataPoint(&DataPoint{RequestRate: float64(i + 50)})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Make a copy to avoid growing indefinitely
		dsCopy := NewDataSet("copy")
		dsCopy.Data = append(dsCopy.Data, ds1.Data...)
		dsCopy.Merge(ds2)
	}
}

// Benchmark for DataSetPrettyPrint
func BenchmarkDataSet_DataSetPrettyPrint(b *testing.B) {
	ds := NewDataSet("test")
	for i := 0; i < 20; i++ {
		ds.AppendDataPoint(&DataPoint{
			RequestRate:  float64(i),
			InputTokens:  float64(i * 10),
			OutputTokens: float64(i * 5),
			AvgTTFTTime:  10.0,
			AvgITLTime:   5.0,
			MaxBatchSize: 32,
			MaxNumTokens: 2048,
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ds.DataSetPrettyPrint()
	}
}
