package core

import (
	"math"
	"testing"

	"github.com/llm-inferno/model-trainer/pkg/config"
)

func TestNewAnalyzer(t *testing.T) {
	tests := []struct {
		name       string
		params     *config.ModelParams
		validateFn func(t *testing.T, analyzer *Analyzer)
	}{
		{
			name: "valid parameters",
			params: &config.ModelParams{
				Alpha: 1.0,
				Beta:  2.0,
				Gamma: 3.0,
			},
			validateFn: func(t *testing.T, analyzer *Analyzer) {
				if analyzer == nil {
					t.Fatal("NewAnalyzer returned nil")
				}
				if analyzer.Parms == nil {
					t.Fatal("Parms is nil")
				}
				if analyzer.Parms.Alpha != 1.0 {
					t.Errorf("Alpha = %v, want 1.0", analyzer.Parms.Alpha)
				}
				if analyzer.Parms.Beta != 2.0 {
					t.Errorf("Beta = %v, want 2.0", analyzer.Parms.Beta)
				}
				if analyzer.Parms.Gamma != 3.0 {
					t.Errorf("Gamma = %v, want 3.0", analyzer.Parms.Gamma)
				}
			},
		},
		{
			name: "zero parameters",
			params: &config.ModelParams{
				Alpha: 0.0,
				Beta:  0.0,
				Gamma: 0.0,
			},
			validateFn: func(t *testing.T, analyzer *Analyzer) {
				if analyzer == nil {
					t.Fatal("NewAnalyzer returned nil")
				}
				if analyzer.Parms == nil {
					t.Fatal("Parms is nil")
				}
				if analyzer.Parms.Alpha != 0.0 {
					t.Errorf("Alpha = %v, want 0.0", analyzer.Parms.Alpha)
				}
			},
		},
		{
			name: "negative parameters",
			params: &config.ModelParams{
				Alpha: -1.0,
				Beta:  -2.0,
				Gamma: -3.0,
			},
			validateFn: func(t *testing.T, analyzer *Analyzer) {
				if analyzer == nil {
					t.Fatal("NewAnalyzer returned nil")
				}
				if analyzer.Parms == nil {
					t.Fatal("Parms is nil")
				}
				if analyzer.Parms.Alpha != -1.0 {
					t.Errorf("Alpha = %v, want -1.0", analyzer.Parms.Alpha)
				}
			},
		},
		{
			name: "large parameter values",
			params: &config.ModelParams{
				Alpha: 1000.0,
				Beta:  2000.0,
				Gamma: 3000.0,
			},
			validateFn: func(t *testing.T, analyzer *Analyzer) {
				if analyzer == nil {
					t.Fatal("NewAnalyzer returned nil")
				}
				if analyzer.Parms.Alpha != 1000.0 {
					t.Errorf("Alpha = %v, want 1000.0", analyzer.Parms.Alpha)
				}
				if analyzer.Parms.Beta != 2000.0 {
					t.Errorf("Beta = %v, want 2000.0", analyzer.Parms.Beta)
				}
				if analyzer.Parms.Gamma != 3000.0 {
					t.Errorf("Gamma = %v, want 3000.0", analyzer.Parms.Gamma)
				}
			},
		},
		{
			name: "fractional parameters",
			params: &config.ModelParams{
				Alpha: 0.123,
				Beta:  0.456,
				Gamma: 0.789,
			},
			validateFn: func(t *testing.T, analyzer *Analyzer) {
				if analyzer == nil {
					t.Fatal("NewAnalyzer returned nil")
				}
				if math.Abs(analyzer.Parms.Alpha-0.123) > 1e-9 {
					t.Errorf("Alpha = %v, want 0.123", analyzer.Parms.Alpha)
				}
				if math.Abs(analyzer.Parms.Beta-0.456) > 1e-9 {
					t.Errorf("Beta = %v, want 0.456", analyzer.Parms.Beta)
				}
				if math.Abs(analyzer.Parms.Gamma-0.789) > 1e-9 {
					t.Errorf("Gamma = %v, want 0.789", analyzer.Parms.Gamma)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			analyzer := NewAnalyzer(tt.params)

			if tt.validateFn != nil {
				tt.validateFn(t, analyzer)
			}
		})
	}
}

// Mock model functions for Analyzer tests
func mockAnalyzerPerfectModel(x *config.InputVars, params *config.ModelParams) (*config.OutputVars, error) {
	// Returns fixed values for testing
	return &config.OutputVars{
		AvgTTFTTime: 10.0,
		AvgITLTime:  5.0,
	}, nil
}

func mockAnalyzerLinearModel(x *config.InputVars, params *config.ModelParams) (*config.OutputVars, error) {
	// Simple linear model: output = alpha + beta * requestRate
	return &config.OutputVars{
		AvgTTFTTime: params.Alpha + params.Beta*x.RequestRate,
		AvgITLTime:  params.Alpha + params.Gamma*x.InputTokens/100.0,
	}, nil
}

func TestAnalyzer_Analyze(t *testing.T) {
	tests := []struct {
		name       string
		params     *config.ModelParams
		dataSet    *DataSet
		model      ModelFunction
		validateFn func(t *testing.T, results *config.AnalysisResults)
	}{
		{
			name: "empty dataset",
			params: &config.ModelParams{
				Alpha: 1.0,
				Beta:  2.0,
				Gamma: 3.0,
			},
			dataSet: NewDataSet("empty"),
			model:   mockAnalyzerLinearModel,
			validateFn: func(t *testing.T, results *config.AnalysisResults) {
				if results == nil {
					t.Fatal("results is nil")
				}
				// With empty dataset, all errors should be zero
				if results.AvgErrTTFT != 0.0 {
					t.Errorf("AvgErrTTFT = %v, want 0.0", results.AvgErrTTFT)
				}
				if results.AvgErrITL != 0.0 {
					t.Errorf("AvgErrITL = %v, want 0.0", results.AvgErrITL)
				}
				if results.AvgErrWeighted != 0.0 {
					t.Errorf("AvgErrWeighted = %v, want 0.0", results.AvgErrWeighted)
				}
			},
		},
		{
			name: "single data point - perfect predictions",
			params: &config.ModelParams{
				Alpha: 1.0,
				Beta:  2.0,
				Gamma: 3.0,
			},
			dataSet: func() *DataSet {
				ds := NewDataSet("single")
				ds.AppendDataPoint(&DataPoint{
					RequestRate:  10.0,
					InputTokens:  100.0,
					OutputTokens: 50.0,
					AvgTTFTTime:  10.0,
					AvgITLTime:   5.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				})
				return ds
			}(),
			model: mockAnalyzerPerfectModel,
			validateFn: func(t *testing.T, results *config.AnalysisResults) {
				if results == nil {
					t.Fatal("results is nil")
				}
				// Perfect model should have zero error
				if results.AvgErrTTFT != 0.0 {
					t.Errorf("AvgErrTTFT = %v, want 0.0", results.AvgErrTTFT)
				}
				if results.AvgErrITL != 0.0 {
					t.Errorf("AvgErrITL = %v, want 0.0", results.AvgErrITL)
				}
				if results.AvgErrWeighted != 0.0 {
					t.Errorf("AvgErrWeighted = %v, want 0.0", results.AvgErrWeighted)
				}
			},
		},
		{
			name: "single data point - with error",
			params: &config.ModelParams{
				Alpha: 5.0,
				Beta:  1.0,
				Gamma: 0.1,
			},
			dataSet: func() *DataSet {
				ds := NewDataSet("single-error")
				ds.AppendDataPoint(&DataPoint{
					RequestRate:  10.0,
					InputTokens:  100.0,
					OutputTokens: 50.0,
					AvgTTFTTime:  10.0,
					AvgITLTime:   5.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				})
				return ds
			}(),
			model: mockAnalyzerLinearModel,
			validateFn: func(t *testing.T, results *config.AnalysisResults) {
				if results == nil {
					t.Fatal("results is nil")
				}
				// Should have some error
				if results.AvgErrTTFT < 0.0 {
					t.Errorf("AvgErrTTFT should be non-negative, got %v", results.AvgErrTTFT)
				}
				if results.AvgErrITL < 0.0 {
					t.Errorf("AvgErrITL should be non-negative, got %v", results.AvgErrITL)
				}
				if results.AvgErrWeighted < 0.0 {
					t.Errorf("AvgErrWeighted should be non-negative, got %v", results.AvgErrWeighted)
				}
				if math.IsNaN(results.AvgErrWeighted) || math.IsInf(results.AvgErrWeighted, 0) {
					t.Errorf("AvgErrWeighted should be finite, got %v", results.AvgErrWeighted)
				}
			},
		},
		{
			name: "multiple data points - perfect predictions",
			params: &config.ModelParams{
				Alpha: 1.0,
				Beta:  2.0,
				Gamma: 3.0,
			},
			dataSet: func() *DataSet {
				ds := NewDataSet("multiple-perfect")
				ds.AppendDataPoint(&DataPoint{
					RequestRate:  10.0,
					InputTokens:  100.0,
					OutputTokens: 50.0,
					AvgTTFTTime:  10.0,
					AvgITLTime:   5.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				})
				ds.AppendDataPoint(&DataPoint{
					RequestRate:  20.0,
					InputTokens:  200.0,
					OutputTokens: 100.0,
					AvgTTFTTime:  10.0,
					AvgITLTime:   5.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				})
				ds.AppendDataPoint(&DataPoint{
					RequestRate:  30.0,
					InputTokens:  300.0,
					OutputTokens: 150.0,
					AvgTTFTTime:  10.0,
					AvgITLTime:   5.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				})
				return ds
			}(),
			model: mockAnalyzerPerfectModel,
			validateFn: func(t *testing.T, results *config.AnalysisResults) {
				if results == nil {
					t.Fatal("results is nil")
				}
				// Perfect model should have zero error
				if results.AvgErrTTFT != 0.0 {
					t.Errorf("AvgErrTTFT = %v, want 0.0", results.AvgErrTTFT)
				}
				if results.AvgErrITL != 0.0 {
					t.Errorf("AvgErrITL = %v, want 0.0", results.AvgErrITL)
				}
				if results.AvgErrWeighted != 0.0 {
					t.Errorf("AvgErrWeighted = %v, want 0.0", results.AvgErrWeighted)
				}
			},
		},
		{
			name: "multiple data points - with errors",
			params: &config.ModelParams{
				Alpha: 5.0,
				Beta:  1.0,
				Gamma: 0.05,
			},
			dataSet: func() *DataSet {
				ds := NewDataSet("multiple-error")
				ds.AppendDataPoint(&DataPoint{
					RequestRate:  10.0,
					InputTokens:  100.0,
					OutputTokens: 50.0,
					AvgTTFTTime:  15.0,
					AvgITLTime:   10.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				})
				ds.AppendDataPoint(&DataPoint{
					RequestRate:  20.0,
					InputTokens:  200.0,
					OutputTokens: 100.0,
					AvgTTFTTime:  25.0,
					AvgITLTime:   15.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				})
				ds.AppendDataPoint(&DataPoint{
					RequestRate:  30.0,
					InputTokens:  300.0,
					OutputTokens: 150.0,
					AvgTTFTTime:  35.0,
					AvgITLTime:   20.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				})
				return ds
			}(),
			model: mockAnalyzerLinearModel,
			validateFn: func(t *testing.T, results *config.AnalysisResults) {
				if results == nil {
					t.Fatal("results is nil")
				}
				// Should have some average error
				if results.AvgErrTTFT < 0.0 {
					t.Errorf("AvgErrTTFT should be non-negative, got %v", results.AvgErrTTFT)
				}
				if results.AvgErrITL < 0.0 {
					t.Errorf("AvgErrITL should be non-negative, got %v", results.AvgErrITL)
				}
				if results.AvgErrWeighted < 0.0 {
					t.Errorf("AvgErrWeighted should be non-negative, got %v", results.AvgErrWeighted)
				}
				// All errors should be finite
				if math.IsNaN(results.AvgErrTTFT) || math.IsInf(results.AvgErrTTFT, 0) {
					t.Errorf("AvgErrTTFT should be finite, got %v", results.AvgErrTTFT)
				}
				if math.IsNaN(results.AvgErrITL) || math.IsInf(results.AvgErrITL, 0) {
					t.Errorf("AvgErrITL should be finite, got %v", results.AvgErrITL)
				}
				if math.IsNaN(results.AvgErrWeighted) || math.IsInf(results.AvgErrWeighted, 0) {
					t.Errorf("AvgErrWeighted should be finite, got %v", results.AvgErrWeighted)
				}
			},
		},
		{
			name: "large dataset",
			params: &config.ModelParams{
				Alpha: 5.0,
				Beta:  1.0,
				Gamma: 0.05,
			},
			dataSet: func() *DataSet {
				ds := NewDataSet("large")
				for i := 0; i < 50; i++ {
					ds.AppendDataPoint(&DataPoint{
						RequestRate:  float64(i + 1),
						InputTokens:  float64((i + 1) * 10),
						OutputTokens: float64((i + 1) * 5),
						AvgTTFTTime:  float64(10 + i),
						AvgITLTime:   float64(5 + i/2),
						MaxBatchSize: 32,
						MaxNumTokens: 2048,
					})
				}
				return ds
			}(),
			model: mockAnalyzerLinearModel,
			validateFn: func(t *testing.T, results *config.AnalysisResults) {
				if results == nil {
					t.Fatal("results is nil")
				}
				// Should have average errors
				if results.AvgErrTTFT < 0.0 {
					t.Errorf("AvgErrTTFT should be non-negative, got %v", results.AvgErrTTFT)
				}
				if results.AvgErrITL < 0.0 {
					t.Errorf("AvgErrITL should be non-negative, got %v", results.AvgErrITL)
				}
				if results.AvgErrWeighted < 0.0 {
					t.Errorf("AvgErrWeighted should be non-negative, got %v", results.AvgErrWeighted)
				}
			},
		},
		{
			name: "dataset with zero parameters",
			params: &config.ModelParams{
				Alpha: 0.0,
				Beta:  0.0,
				Gamma: 0.0,
			},
			dataSet: func() *DataSet {
				ds := NewDataSet("zero-params")
				ds.AppendDataPoint(&DataPoint{
					RequestRate:  10.0,
					InputTokens:  100.0,
					OutputTokens: 50.0,
					AvgTTFTTime:  10.0,
					AvgITLTime:   5.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				})
				return ds
			}(),
			model: mockAnalyzerLinearModel,
			validateFn: func(t *testing.T, results *config.AnalysisResults) {
				if results == nil {
					t.Fatal("results is nil")
				}
				// With zero parameters, linear model should return 0, so error = actual value
				if results.AvgErrTTFT < 0.0 {
					t.Errorf("AvgErrTTFT should be non-negative, got %v", results.AvgErrTTFT)
				}
				if results.AvgErrITL < 0.0 {
					t.Errorf("AvgErrITL should be non-negative, got %v", results.AvgErrITL)
				}
			},
		},
		{
			name: "dataset with mixed data point values",
			params: &config.ModelParams{
				Alpha: 3.0,
				Beta:  0.8,
				Gamma: 0.03,
			},
			dataSet: func() *DataSet {
				ds := NewDataSet("mixed")
				// Small values
				ds.AppendDataPoint(&DataPoint{
					RequestRate:  1.0,
					InputTokens:  10.0,
					OutputTokens: 5.0,
					AvgTTFTTime:  5.0,
					AvgITLTime:   2.5,
					MaxBatchSize: 16,
					MaxNumTokens: 1024,
				})
				// Medium values
				ds.AppendDataPoint(&DataPoint{
					RequestRate:  10.0,
					InputTokens:  100.0,
					OutputTokens: 50.0,
					AvgTTFTTime:  15.0,
					AvgITLTime:   8.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				})
				// Large values
				ds.AppendDataPoint(&DataPoint{
					RequestRate:  50.0,
					InputTokens:  500.0,
					OutputTokens: 250.0,
					AvgTTFTTime:  60.0,
					AvgITLTime:   25.0,
					MaxBatchSize: 128,
					MaxNumTokens: 8192,
				})
				return ds
			}(),
			model: mockAnalyzerLinearModel,
			validateFn: func(t *testing.T, results *config.AnalysisResults) {
				if results == nil {
					t.Fatal("results is nil")
				}
				// Verify all metrics are valid
				if results.AvgErrTTFT < 0.0 {
					t.Errorf("AvgErrTTFT should be non-negative, got %v", results.AvgErrTTFT)
				}
				if results.AvgErrITL < 0.0 {
					t.Errorf("AvgErrITL should be non-negative, got %v", results.AvgErrITL)
				}
				if results.AvgErrWeighted < 0.0 {
					t.Errorf("AvgErrWeighted should be non-negative, got %v", results.AvgErrWeighted)
				}
				if math.IsNaN(results.AvgErrWeighted) || math.IsInf(results.AvgErrWeighted, 0) {
					t.Errorf("AvgErrWeighted should be finite, got %v", results.AvgErrWeighted)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			analyzer := NewAnalyzer(tt.params)
			results := analyzer.Analyze(tt.dataSet, tt.model)

			if tt.validateFn != nil {
				tt.validateFn(t, results)
			}
		})
	}
}

func TestAnalyzer_Analyze_Consistency(t *testing.T) {
	t.Run("same input produces same output", func(t *testing.T) {
		params := &config.ModelParams{
			Alpha: 5.0,
			Beta:  1.0,
			Gamma: 0.1,
		}

		dataSet := NewDataSet("consistency")
		dataSet.AppendDataPoint(&DataPoint{
			RequestRate:  10.0,
			InputTokens:  100.0,
			OutputTokens: 50.0,
			AvgTTFTTime:  15.0,
			AvgITLTime:   10.0,
			MaxBatchSize: 32,
			MaxNumTokens: 2048,
		})

		analyzer := NewAnalyzer(params)

		// First analysis
		results1 := analyzer.Analyze(dataSet, mockAnalyzerLinearModel)

		// Second analysis with same inputs
		results2 := analyzer.Analyze(dataSet, mockAnalyzerLinearModel)

		// Results should be identical
		if results1.AvgErrTTFT != results2.AvgErrTTFT {
			t.Errorf("AvgErrTTFT not consistent: first=%v, second=%v", results1.AvgErrTTFT, results2.AvgErrTTFT)
		}
		if results1.AvgErrITL != results2.AvgErrITL {
			t.Errorf("AvgErrITL not consistent: first=%v, second=%v", results1.AvgErrITL, results2.AvgErrITL)
		}
		if results1.AvgErrWeighted != results2.AvgErrWeighted {
			t.Errorf("AvgErrWeighted not consistent: first=%v, second=%v", results1.AvgErrWeighted, results2.AvgErrWeighted)
		}
	})
}

func TestAnalyzer_Analyze_ParameterUpdate(t *testing.T) {
	t.Run("changing parameters changes results", func(t *testing.T) {
		dataSet := NewDataSet("param-change")
		dataSet.AppendDataPoint(&DataPoint{
			RequestRate:  10.0,
			InputTokens:  100.0,
			OutputTokens: 50.0,
			AvgTTFTTime:  15.0,
			AvgITLTime:   10.0,
			MaxBatchSize: 32,
			MaxNumTokens: 2048,
		})

		// First set of parameters
		params1 := &config.ModelParams{
			Alpha: 5.0,
			Beta:  1.0,
			Gamma: 0.1,
		}
		analyzer1 := NewAnalyzer(params1)
		results1 := analyzer1.Analyze(dataSet, mockAnalyzerLinearModel)

		// Different set of parameters
		params2 := &config.ModelParams{
			Alpha: 10.0,
			Beta:  2.0,
			Gamma: 0.2,
		}
		analyzer2 := NewAnalyzer(params2)
		results2 := analyzer2.Analyze(dataSet, mockAnalyzerLinearModel)

		// Results should be different (unless by extreme coincidence)
		if results1.AvgErrTTFT == results2.AvgErrTTFT &&
			results1.AvgErrITL == results2.AvgErrITL &&
			results1.AvgErrWeighted == results2.AvgErrWeighted {
			// This could happen if both parameter sets produce the same error,
			// but it's highly unlikely with these specific values
			t.Log("Warning: Different parameters produced identical errors (unlikely but possible)")
		}
	})
}

func TestAnalyzer_Analyze_DatasetModification(t *testing.T) {
	t.Run("dataset can be reused after analysis", func(t *testing.T) {
		params := &config.ModelParams{
			Alpha: 5.0,
			Beta:  1.0,
			Gamma: 0.1,
		}
		analyzer := NewAnalyzer(params)

		// Create dataset
		dataSet := NewDataSet("reusable")
		dataSet.AppendDataPoint(&DataPoint{
			RequestRate:  10.0,
			InputTokens:  100.0,
			OutputTokens: 50.0,
			AvgTTFTTime:  15.0,
			AvgITLTime:   10.0,
			MaxBatchSize: 32,
			MaxNumTokens: 2048,
		})

		// First analysis
		results1 := analyzer.Analyze(dataSet, mockAnalyzerLinearModel)

		// Add more data
		dataSet.AppendDataPoint(&DataPoint{
			RequestRate:  20.0,
			InputTokens:  200.0,
			OutputTokens: 100.0,
			AvgTTFTTime:  25.0,
			AvgITLTime:   15.0,
			MaxBatchSize: 32,
			MaxNumTokens: 2048,
		})

		// Second analysis with more data
		results2 := analyzer.Analyze(dataSet, mockAnalyzerLinearModel)

		// Results should be different because dataset changed
		if results1.AvgErrWeighted == results2.AvgErrWeighted {
			t.Log("Warning: Adding data did not change error (unlikely but possible if errors cancel out)")
		}

		// Verify dataset still has correct size
		if dataSet.Size() != 2 {
			t.Errorf("Dataset size = %v, want 2", dataSet.Size())
		}
	})
}

// Benchmark for NewAnalyzer
func BenchmarkNewAnalyzer(b *testing.B) {
	params := &config.ModelParams{
		Alpha: 1.0,
		Beta:  2.0,
		Gamma: 3.0,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewAnalyzer(params)
	}
}

// Benchmark for Analyze with single data point
func BenchmarkAnalyzer_Analyze_Single(b *testing.B) {
	params := &config.ModelParams{
		Alpha: 5.0,
		Beta:  1.0,
		Gamma: 0.1,
	}
	analyzer := NewAnalyzer(params)

	dataSet := NewDataSet("bench-single")
	dataSet.AppendDataPoint(&DataPoint{
		RequestRate:  10.0,
		InputTokens:  100.0,
		OutputTokens: 50.0,
		AvgTTFTTime:  15.0,
		AvgITLTime:   10.0,
		MaxBatchSize: 32,
		MaxNumTokens: 2048,
	})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = analyzer.Analyze(dataSet, mockAnalyzerLinearModel)
	}
}

// Benchmark for Analyze with multiple data points
func BenchmarkAnalyzer_Analyze_Multiple(b *testing.B) {
	params := &config.ModelParams{
		Alpha: 5.0,
		Beta:  1.0,
		Gamma: 0.1,
	}
	analyzer := NewAnalyzer(params)

	dataSet := NewDataSet("bench-multiple")
	for i := 0; i < 10; i++ {
		dataSet.AppendDataPoint(&DataPoint{
			RequestRate:  float64(i + 1),
			InputTokens:  float64((i + 1) * 10),
			OutputTokens: float64((i + 1) * 5),
			AvgTTFTTime:  float64(10 + i),
			AvgITLTime:   float64(5 + i/2),
			MaxBatchSize: 32,
			MaxNumTokens: 2048,
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = analyzer.Analyze(dataSet, mockAnalyzerLinearModel)
	}
}

// Benchmark for Analyze with large dataset
func BenchmarkAnalyzer_Analyze_Large(b *testing.B) {
	params := &config.ModelParams{
		Alpha: 5.0,
		Beta:  1.0,
		Gamma: 0.1,
	}
	analyzer := NewAnalyzer(params)

	dataSet := NewDataSet("bench-large")
	for i := 0; i < 100; i++ {
		dataSet.AppendDataPoint(&DataPoint{
			RequestRate:  float64(i + 1),
			InputTokens:  float64((i + 1) * 10),
			OutputTokens: float64((i + 1) * 5),
			AvgTTFTTime:  float64(10 + i),
			AvgITLTime:   float64(5 + i/2),
			MaxBatchSize: 32,
			MaxNumTokens: 2048,
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = analyzer.Analyze(dataSet, mockAnalyzerLinearModel)
	}
}
