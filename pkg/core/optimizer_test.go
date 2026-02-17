package core

import (
	"fmt"
	"math"
	"testing"

	"github.com/llm-inferno/model-trainer/pkg/config"
)

func TestNewOptimizer(t *testing.T) {
	tests := []struct {
		name       string
		initParams *config.ModelParams
	}{
		{
			name: "valid parameters",
			initParams: &config.ModelParams{
				Alpha: 1.0,
				Beta:  2.0,
				Gamma: 3.0,
			},
		},
		{
			name: "zero parameters",
			initParams: &config.ModelParams{
				Alpha: 0.0,
				Beta:  0.0,
				Gamma: 0.0,
			},
		},
		{
			name: "negative parameters",
			initParams: &config.ModelParams{
				Alpha: -1.0,
				Beta:  -2.0,
				Gamma: -3.0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			optimizer := NewOptimizer(tt.initParams)

			if optimizer == nil {
				t.Fatal("NewOptimizer returned nil")
			}

			if optimizer.InitParms == nil {
				t.Fatal("InitParms is nil")
			}

			if optimizer.InitParms.Alpha != tt.initParams.Alpha {
				t.Errorf("Alpha = %v, want %v", optimizer.InitParms.Alpha, tt.initParams.Alpha)
			}

			if optimizer.InitParms.Beta != tt.initParams.Beta {
				t.Errorf("Beta = %v, want %v", optimizer.InitParms.Beta, tt.initParams.Beta)
			}

			if optimizer.InitParms.Gamma != tt.initParams.Gamma {
				t.Errorf("Gamma = %v, want %v", optimizer.InitParms.Gamma, tt.initParams.Gamma)
			}
		})
	}
}

// mockLinearModel is a simple linear model for testing: y = alpha + beta*x
func mockLinearModel(x *config.InputVars, params *config.ModelParams) (*config.OutputVars, error) {
	// Simple linear model for testing
	ttft := params.Alpha + params.Beta*x.RequestRate
	itl := params.Alpha + params.Gamma*x.InputTokens

	return &config.OutputVars{
		AvgTTFTTime: ttft,
		AvgITLTime:  itl,
	}, nil
}

// mockErrorModel always returns an error
func mockErrorModel(x *config.InputVars, params *config.ModelParams) (*config.OutputVars, error) {
	return nil, fmt.Errorf("mock error")
}

// mockInvalidParamsModel returns error for invalid parameters
func mockInvalidParamsModel(x *config.InputVars, params *config.ModelParams) (*config.OutputVars, error) {
	if params.Alpha < 0 || params.Beta < 0 || params.Gamma < 0 {
		return nil, fmt.Errorf("invalid parameters")
	}
	return mockLinearModel(x, params)
}

func TestOptimizer_Optimize(t *testing.T) {
	tests := []struct {
		name          string
		initParams    *config.ModelParams
		dataSet       *DataSet
		model         ModelFunction
		expectError   bool
		validateFunc  func(t *testing.T, result *OptimizationResult)
	}{
		{
			name: "successful optimization with linear model",
			initParams: &config.ModelParams{
				Alpha: 1.0,
				Beta:  1.0,
				Gamma: 1.0,
			},
			dataSet: createTestDataSet([]DataPoint{
				{
					RequestRate:  10.0,
					InputTokens:  100.0,
					OutputTokens: 50.0,
					AvgTTFTTime:  15.0,
					AvgITLTime:   105.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				},
				{
					RequestRate:  20.0,
					InputTokens:  200.0,
					OutputTokens: 100.0,
					AvgTTFTTime:  25.0,
					AvgITLTime:   205.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				},
			}),
			model:       mockLinearModel,
			expectError: false,
			validateFunc: func(t *testing.T, result *OptimizationResult) {
				if result == nil {
					t.Fatal("result is nil")
				}
				if result.OptimizedParms == nil {
					t.Fatal("OptimizedParms is nil")
				}
				if result.AnalysisResults == nil {
					t.Fatal("AnalysisResults is nil")
				}
				// Verify optimized parameters are reasonable
				if math.IsNaN(result.OptimizedParms.Alpha) || math.IsInf(result.OptimizedParms.Alpha, 0) {
					t.Errorf("Alpha is invalid: %v", result.OptimizedParms.Alpha)
				}
				if math.IsNaN(result.OptimizedParms.Beta) || math.IsInf(result.OptimizedParms.Beta, 0) {
					t.Errorf("Beta is invalid: %v", result.OptimizedParms.Beta)
				}
				if math.IsNaN(result.OptimizedParms.Gamma) || math.IsInf(result.OptimizedParms.Gamma, 0) {
					t.Errorf("Gamma is invalid: %v", result.OptimizedParms.Gamma)
				}
			},
		},
		{
			name: "optimization with single data point",
			initParams: &config.ModelParams{
				Alpha: 1.0,
				Beta:  1.0,
				Gamma: 1.0,
			},
			dataSet: createTestDataSet([]DataPoint{
				{
					RequestRate:  10.0,
					InputTokens:  100.0,
					OutputTokens: 50.0,
					AvgTTFTTime:  15.0,
					AvgITLTime:   105.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				},
			}),
			model:       mockLinearModel,
			expectError: false,
			validateFunc: func(t *testing.T, result *OptimizationResult) {
				if result == nil {
					t.Fatal("result is nil")
				}
				if result.OptimizedParms == nil {
					t.Fatal("OptimizedParms is nil")
				}
			},
		},
		{
			name: "optimization with model that returns errors",
			initParams: &config.ModelParams{
				Alpha: 1.0,
				Beta:  1.0,
				Gamma: 1.0,
			},
			dataSet: createTestDataSet([]DataPoint{
				{
					RequestRate:  10.0,
					InputTokens:  100.0,
					OutputTokens: 50.0,
					AvgTTFTTime:  15.0,
					AvgITLTime:   105.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				},
			}),
			model:       mockErrorModel,
			expectError: true,
			validateFunc: func(t *testing.T, result *OptimizationResult) {
				// Should not reach here
			},
		},
		{
			name: "optimization with multiple data points",
			initParams: &config.ModelParams{
				Alpha: 1.0,
				Beta:  1.0,
				Gamma: 1.0,
			},
			dataSet: createTestDataSet([]DataPoint{
				{
					RequestRate:  5.0,
					InputTokens:  50.0,
					OutputTokens: 25.0,
					AvgTTFTTime:  10.0,
					AvgITLTime:   55.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				},
				{
					RequestRate:  10.0,
					InputTokens:  100.0,
					OutputTokens: 50.0,
					AvgTTFTTime:  15.0,
					AvgITLTime:   105.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				},
				{
					RequestRate:  20.0,
					InputTokens:  200.0,
					OutputTokens: 100.0,
					AvgTTFTTime:  25.0,
					AvgITLTime:   205.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				},
				{
					RequestRate:  30.0,
					InputTokens:  300.0,
					OutputTokens: 150.0,
					AvgTTFTTime:  35.0,
					AvgITLTime:   305.0,
					MaxBatchSize: 32,
					MaxNumTokens: 2048,
				},
			}),
			model:       mockLinearModel,
			expectError: false,
			validateFunc: func(t *testing.T, result *OptimizationResult) {
				if result == nil {
					t.Fatal("result is nil")
				}
				// For a linear model with linear data, optimized parameters should be close to [5, 1, 1]
				// (based on the pattern in test data)
				if result.OptimizedParms.Beta < 0 {
					t.Errorf("Beta should be positive for this test case, got %v", result.OptimizedParms.Beta)
				}
				if result.OptimizedParms.Gamma < 0 {
					t.Errorf("Gamma should be positive for this test case, got %v", result.OptimizedParms.Gamma)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			optimizer := NewOptimizer(tt.initParams)
			result, err := optimizer.Optimize(tt.dataSet, tt.model)

			if tt.expectError {
				if err == nil {
					t.Error("expected error but got none")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if tt.validateFunc != nil {
				tt.validateFunc(t, result)
			}
		})
	}
}

func TestOptimizer_OptimizeWithInvalidParams(t *testing.T) {
	initParams := &config.ModelParams{
		Alpha: -10.0,
		Beta:  -10.0,
		Gamma: -10.0,
	}

	dataSet := createTestDataSet([]DataPoint{
		{
			RequestRate:  10.0,
			InputTokens:  100.0,
			OutputTokens: 50.0,
			AvgTTFTTime:  15.0,
			AvgITLTime:   105.0,
			MaxBatchSize: 32,
			MaxNumTokens: 2048,
		},
	})

	optimizer := NewOptimizer(initParams)
	result, err := optimizer.Optimize(dataSet, mockInvalidParamsModel)

	// The optimizer should either return an error or converge to valid parameters
	if err == nil && result != nil {
		// If no error, check that optimizer found better parameters
		if result.OptimizedParms.Alpha < 0 && result.OptimizedParms.Beta < 0 && result.OptimizedParms.Gamma < 0 {
			t.Error("optimizer did not improve parameters from invalid starting point")
		}
	}
}

// Helper function to create a test dataset
func createTestDataSet(dataPoints []DataPoint) *DataSet {
	dataSet := NewDataSet("test")
	for _, dp := range dataPoints {
		dataSet.AppendDataPoint(&dp)
	}
	return dataSet
}

// Benchmark for Optimize method
func BenchmarkOptimizer_Optimize(b *testing.B) {
	initParams := &config.ModelParams{
		Alpha: 1.0,
		Beta:  1.0,
		Gamma: 1.0,
	}

	dataSet := createTestDataSet([]DataPoint{
		{
			RequestRate:  10.0,
			InputTokens:  100.0,
			OutputTokens: 50.0,
			AvgTTFTTime:  15.0,
			AvgITLTime:   105.0,
			MaxBatchSize: 32,
			MaxNumTokens: 2048,
		},
		{
			RequestRate:  20.0,
			InputTokens:  200.0,
			OutputTokens: 100.0,
			AvgTTFTTime:  25.0,
			AvgITLTime:   205.0,
			MaxBatchSize: 32,
			MaxNumTokens: 2048,
		},
	})

	optimizer := NewOptimizer(initParams)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := optimizer.Optimize(dataSet, mockLinearModel)
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
	}
}
