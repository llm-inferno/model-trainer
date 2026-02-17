package core

import (
	"fmt"
	"math"
	"testing"

	"github.com/llm-inferno/model-trainer/pkg/config"
)

func TestModel(t *testing.T) {
	tests := []struct {
		name        string
		inputVars   *config.InputVars
		params      *config.ModelParams
		expectError bool
		validateFn  func(t *testing.T, output *config.OutputVars, err error)
	}{
		{
			name: "valid parameters and inputs - realistic scenario",
			inputVars: &config.InputVars{
				RequestRate:  0.1,
				InputTokens:  100.0,
				OutputTokens: 50.0,
				MaxBatchSize: 32,
				MaxNumTokens: 2048,
			},
			params: &config.ModelParams{
				Alpha: 0.5,
				Beta:  1.0,
				Gamma: 0.5,
			},
			expectError: false,
			validateFn: func(t *testing.T, output *config.OutputVars, err error) {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if output == nil {
					t.Fatal("output is nil")
				}
				if output.AvgTTFTTime <= 0 {
					t.Errorf("AvgTTFTTime should be positive, got %v", output.AvgTTFTTime)
				}
				if output.AvgITLTime <= 0 {
					t.Errorf("AvgITLTime should be positive, got %v", output.AvgITLTime)
				}
			},
		},
		{
			name: "invalid parameters - negative alpha",
			inputVars: &config.InputVars{
				RequestRate:  10.0,
				InputTokens:  100.0,
				OutputTokens: 50.0,
				MaxBatchSize: 32,
				MaxNumTokens: 2048,
			},
			params: &config.ModelParams{
				Alpha: -1.0,
				Beta:  2.0,
				Gamma: 3.0,
			},
			expectError: true,
			validateFn: func(t *testing.T, output *config.OutputVars, err error) {
				if err == nil {
					t.Error("expected error for negative alpha, got nil")
				}
				if output != nil {
					t.Error("output should be nil when error occurs")
				}
			},
		},
		{
			name: "invalid parameters - negative beta",
			inputVars: &config.InputVars{
				RequestRate:  10.0,
				InputTokens:  100.0,
				OutputTokens: 50.0,
				MaxBatchSize: 32,
				MaxNumTokens: 2048,
			},
			params: &config.ModelParams{
				Alpha: 1.0,
				Beta:  -2.0,
				Gamma: 3.0,
			},
			expectError: true,
			validateFn: func(t *testing.T, output *config.OutputVars, err error) {
				if err == nil {
					t.Error("expected error for negative beta, got nil")
				}
			},
		},
		{
			name: "invalid parameters - negative gamma",
			inputVars: &config.InputVars{
				RequestRate:  10.0,
				InputTokens:  100.0,
				OutputTokens: 50.0,
				MaxBatchSize: 32,
				MaxNumTokens: 2048,
			},
			params: &config.ModelParams{
				Alpha: 1.0,
				Beta:  2.0,
				Gamma: -3.0,
			},
			expectError: true,
			validateFn: func(t *testing.T, output *config.OutputVars, err error) {
				if err == nil {
					t.Error("expected error for negative gamma, got nil")
				}
			},
		},
		{
			name: "invalid parameters - all negative",
			inputVars: &config.InputVars{
				RequestRate:  10.0,
				InputTokens:  100.0,
				OutputTokens: 50.0,
				MaxBatchSize: 32,
				MaxNumTokens: 2048,
			},
			params: &config.ModelParams{
				Alpha: -1.0,
				Beta:  -2.0,
				Gamma: -3.0,
			},
			expectError: true,
			validateFn: func(t *testing.T, output *config.OutputVars, err error) {
				if err == nil {
					t.Error("expected error for all negative parameters, got nil")
				}
			},
		},
		{
			name: "zero parameters",
			inputVars: &config.InputVars{
				RequestRate:  10.0,
				InputTokens:  100.0,
				OutputTokens: 50.0,
				MaxBatchSize: 32,
				MaxNumTokens: 2048,
			},
			params: &config.ModelParams{
				Alpha: 0.0,
				Beta:  0.0,
				Gamma: 0.0,
			},
			expectError: false,
			validateFn: func(t *testing.T, output *config.OutputVars, err error) {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if output == nil {
					t.Fatal("output is nil")
				}
			},
		},
		{
			name: "low request rate scenario",
			inputVars: &config.InputVars{
				RequestRate:  0.5,
				InputTokens:  50.0,
				OutputTokens: 25.0,
				MaxBatchSize: 32,
				MaxNumTokens: 2048,
			},
			params: &config.ModelParams{
				Alpha: 0.5,
				Beta:  1.0,
				Gamma: 0.5,
			},
			expectError: false,
			validateFn: func(t *testing.T, output *config.OutputVars, err error) {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if output == nil {
					t.Fatal("output is nil")
				}
			},
		},
		{
			name: "moderate request rate",
			inputVars: &config.InputVars{
				RequestRate:  0.05,
				InputTokens:  200.0,
				OutputTokens: 100.0,
				MaxBatchSize: 64,
				MaxNumTokens: 4096,
			},
			params: &config.ModelParams{
				Alpha: 0.3,
				Beta:  0.8,
				Gamma: 0.4,
			},
			expectError: false,
			validateFn: func(t *testing.T, output *config.OutputVars, err error) {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if output == nil {
					t.Fatal("output is nil")
				}
			},
		},
		{
			name: "large batch size scenario",
			inputVars: &config.InputVars{
				RequestRate:  0.1,
				InputTokens:  100.0,
				OutputTokens: 50.0,
				MaxBatchSize: 256,
				MaxNumTokens: 8192,
			},
			params: &config.ModelParams{
				Alpha: 0.5,
				Beta:  1.0,
				Gamma: 0.5,
			},
			expectError: false,
			validateFn: func(t *testing.T, output *config.OutputVars, err error) {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if output == nil {
					t.Fatal("output is nil")
				}
			},
		},
		{
			name: "many tokens scenario",
			inputVars: &config.InputVars{
				RequestRate:  0.001,
				InputTokens:  1000.0,
				OutputTokens: 500.0,
				MaxBatchSize: 32,
				MaxNumTokens: 2048,
			},
			params: &config.ModelParams{
				Alpha: 0.2,
				Beta:  0.5,
				Gamma: 0.3,
			},
			expectError: false,
			validateFn: func(t *testing.T, output *config.OutputVars, err error) {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if output == nil {
					t.Fatal("output is nil")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			output, err := Model(tt.inputVars, tt.params)

			if tt.expectError {
				if err == nil {
					t.Error("expected error but got nil")
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			}

			if tt.validateFn != nil {
				tt.validateFn(t, output, err)
			}
		})
	}
}

// Mock model functions for testing LossFunction
func mockPerfectModel(x *config.InputVars, params *config.ModelParams) (*config.OutputVars, error) {
	// Returns exactly what's expected (for testing zero loss)
	return &config.OutputVars{
		AvgTTFTTime: 10.0,
		AvgITLTime:  5.0,
	}, nil
}

func mockLinearModelForLoss(x *config.InputVars, params *config.ModelParams) (*config.OutputVars, error) {
	// Simple linear model: output = alpha + beta * requestRate
	return &config.OutputVars{
		AvgTTFTTime: params.Alpha + params.Beta*x.RequestRate,
		AvgITLTime:  params.Alpha + params.Gamma*x.InputTokens/100.0,
	}, nil
}

func mockAlwaysErrorModel(x *config.InputVars, params *config.ModelParams) (*config.OutputVars, error) {
	return nil, fmt.Errorf("mock model error")
}

func mockConditionalErrorModel(x *config.InputVars, params *config.ModelParams) (*config.OutputVars, error) {
	// Error on high request rates
	if x.RequestRate > 50.0 {
		return nil, fmt.Errorf("request rate too high")
	}
	return &config.OutputVars{
		AvgTTFTTime: 10.0,
		AvgITLTime:  5.0,
	}, nil
}

func TestLossFunction(t *testing.T) {
	tests := []struct {
		name       string
		params     *config.ModelParams
		xData      []*config.InputVars
		yData      []*config.OutputVars
		model      ModelFunction
		isPrint    bool
		wantLoss   float64
		validateFn func(t *testing.T, loss float64, errVars *config.ErrorVars)
	}{
		{
			name: "empty data sets",
			params: &config.ModelParams{
				Alpha: 1.0,
				Beta:  2.0,
				Gamma: 3.0,
			},
			xData:    []*config.InputVars{},
			yData:    []*config.OutputVars{},
			model:    mockLinearModelForLoss,
			isPrint:  false,
			wantLoss: 0.0,
			validateFn: func(t *testing.T, loss float64, errVars *config.ErrorVars) {
				if loss != 0.0 {
					t.Errorf("loss = %v, want 0.0 for empty data", loss)
				}
			},
		},
		{
			name: "mismatched data lengths",
			params: &config.ModelParams{
				Alpha: 1.0,
				Beta:  2.0,
				Gamma: 3.0,
			},
			xData: []*config.InputVars{
				{RequestRate: 10.0, InputTokens: 100.0},
			},
			yData:    []*config.OutputVars{},
			model:    mockLinearModelForLoss,
			isPrint:  false,
			wantLoss: 0.0,
			validateFn: func(t *testing.T, loss float64, errVars *config.ErrorVars) {
				if loss != 0.0 {
					t.Errorf("loss = %v, want 0.0 for mismatched data", loss)
				}
			},
		},
		{
			name: "perfect model - zero loss",
			params: &config.ModelParams{
				Alpha: 1.0,
				Beta:  2.0,
				Gamma: 3.0,
			},
			xData: []*config.InputVars{
				{RequestRate: 10.0, InputTokens: 100.0},
			},
			yData: []*config.OutputVars{
				{AvgTTFTTime: 10.0, AvgITLTime: 5.0},
			},
			model:    mockPerfectModel,
			isPrint:  false,
			wantLoss: 0.0,
			validateFn: func(t *testing.T, loss float64, errVars *config.ErrorVars) {
				if loss != 0.0 {
					t.Errorf("loss = %v, want 0.0 for perfect model", loss)
				}
				if errVars.Count != 1 {
					t.Errorf("errVars.Count = %v, want 1", errVars.Count)
				}
			},
		},
		{
			name: "single data point with error",
			params: &config.ModelParams{
				Alpha: 5.0,
				Beta:  1.0,
				Gamma: 0.1,
			},
			xData: []*config.InputVars{
				{RequestRate: 10.0, InputTokens: 100.0},
			},
			yData: []*config.OutputVars{
				{AvgTTFTTime: 10.0, AvgITLTime: 5.0},
			},
			model:   mockLinearModelForLoss,
			isPrint: false,
			validateFn: func(t *testing.T, loss float64, errVars *config.ErrorVars) {
				if loss <= 0.0 {
					t.Error("loss should be positive for imperfect model")
				}
				if math.IsNaN(loss) || math.IsInf(loss, 0) {
					t.Errorf("loss should be finite, got %v", loss)
				}
				if errVars.Count != 1 {
					t.Errorf("errVars.Count = %v, want 1", errVars.Count)
				}
			},
		},
		{
			name: "multiple data points",
			params: &config.ModelParams{
				Alpha: 5.0,
				Beta:  1.0,
				Gamma: 0.05,
			},
			xData: []*config.InputVars{
				{RequestRate: 10.0, InputTokens: 100.0},
				{RequestRate: 20.0, InputTokens: 200.0},
				{RequestRate: 30.0, InputTokens: 300.0},
			},
			yData: []*config.OutputVars{
				{AvgTTFTTime: 15.0, AvgITLTime: 10.0},
				{AvgTTFTTime: 25.0, AvgITLTime: 15.0},
				{AvgTTFTTime: 35.0, AvgITLTime: 20.0},
			},
			model:   mockLinearModelForLoss,
			isPrint: false,
			validateFn: func(t *testing.T, loss float64, errVars *config.ErrorVars) {
				if loss < 0.0 {
					t.Error("loss should be non-negative")
				}
				if math.IsNaN(loss) || math.IsInf(loss, 0) {
					t.Errorf("loss should be finite, got %v", loss)
				}
				if errVars.Count != 3 {
					t.Errorf("errVars.Count = %v, want 3", errVars.Count)
				}
				// Verify cumulative errors are non-negative (can be zero if perfect)
				if errVars.CumErrorTTFT < 0 {
					t.Error("CumErrorTTFT should be non-negative")
				}
				if errVars.CumErrorITL < 0 {
					t.Error("CumErrorITL should be non-negative")
				}
			},
		},
		{
			name: "model that returns error",
			params: &config.ModelParams{
				Alpha: 1.0,
				Beta:  2.0,
				Gamma: 3.0,
			},
			xData: []*config.InputVars{
				{RequestRate: 10.0, InputTokens: 100.0},
			},
			yData: []*config.OutputVars{
				{AvgTTFTTime: 10.0, AvgITLTime: 5.0},
			},
			model:    mockAlwaysErrorModel,
			isPrint:  false,
			wantLoss: math.Inf(1),
			validateFn: func(t *testing.T, loss float64, errVars *config.ErrorVars) {
				if !math.IsInf(loss, 1) {
					t.Errorf("loss = %v, want +Inf for error model", loss)
				}
			},
		},
		{
			name: "model that errors on some data points",
			params: &config.ModelParams{
				Alpha: 1.0,
				Beta:  2.0,
				Gamma: 3.0,
			},
			xData: []*config.InputVars{
				{RequestRate: 10.0, InputTokens: 100.0},
				{RequestRate: 100.0, InputTokens: 100.0}, // This will cause error
			},
			yData: []*config.OutputVars{
				{AvgTTFTTime: 10.0, AvgITLTime: 5.0},
				{AvgTTFTTime: 20.0, AvgITLTime: 10.0},
			},
			model:    mockConditionalErrorModel,
			isPrint:  false,
			wantLoss: math.Inf(1),
			validateFn: func(t *testing.T, loss float64, errVars *config.ErrorVars) {
				if !math.IsInf(loss, 1) {
					t.Errorf("loss = %v, want +Inf when model errors", loss)
				}
			},
		},
		{
			name: "test with print mode enabled",
			params: &config.ModelParams{
				Alpha: 5.0,
				Beta:  1.0,
				Gamma: 0.1,
			},
			xData: []*config.InputVars{
				{RequestRate: 10.0, InputTokens: 100.0, OutputTokens: 50.0},
			},
			yData: []*config.OutputVars{
				{AvgTTFTTime: 15.0, AvgITLTime: 10.0},
			},
			model:   mockLinearModelForLoss,
			isPrint: true, // This will print to stdout during test
			validateFn: func(t *testing.T, loss float64, errVars *config.ErrorVars) {
				if loss <= 0.0 {
					t.Error("loss should be positive")
				}
				if errVars.Count != 1 {
					t.Errorf("errVars.Count = %v, want 1", errVars.Count)
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
			xData: func() []*config.InputVars {
				data := make([]*config.InputVars, 100)
				for i := 0; i < 100; i++ {
					data[i] = &config.InputVars{
						RequestRate: float64(i + 1),
						InputTokens: float64((i + 1) * 10),
					}
				}
				return data
			}(),
			yData: func() []*config.OutputVars {
				data := make([]*config.OutputVars, 100)
				for i := 0; i < 100; i++ {
					data[i] = &config.OutputVars{
						AvgTTFTTime: float64(5 + i),
						AvgITLTime:  float64(5 + i/2),
					}
				}
				return data
			}(),
			model:   mockLinearModelForLoss,
			isPrint: false,
			validateFn: func(t *testing.T, loss float64, errVars *config.ErrorVars) {
				if loss <= 0.0 {
					t.Error("loss should be positive")
				}
				if errVars.Count != 100 {
					t.Errorf("errVars.Count = %v, want 100", errVars.Count)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			errVars := &config.ErrorVars{}
			loss := LossFunction(tt.params, tt.xData, tt.yData, tt.model, errVars, tt.isPrint)

			if tt.wantLoss != 0.0 {
				if loss != tt.wantLoss && !(math.IsInf(tt.wantLoss, 1) && math.IsInf(loss, 1)) {
					t.Errorf("loss = %v, want %v", loss, tt.wantLoss)
				}
			}

			if tt.validateFn != nil {
				tt.validateFn(t, loss, errVars)
			}
		})
	}
}

func TestLossFunction_ErrorVarsAccumulation(t *testing.T) {
	t.Run("error variables accumulate correctly", func(t *testing.T) {
		params := &config.ModelParams{
			Alpha: 5.0,
			Beta:  1.0,
			Gamma: 0.1,
		}

		xData := []*config.InputVars{
			{RequestRate: 10.0, InputTokens: 100.0},
			{RequestRate: 20.0, InputTokens: 200.0},
		}

		yData := []*config.OutputVars{
			{AvgTTFTTime: 15.0, AvgITLTime: 10.0},
			{AvgTTFTTime: 25.0, AvgITLTime: 15.0},
		}

		errVars := &config.ErrorVars{}
		loss := LossFunction(params, xData, yData, mockLinearModelForLoss, errVars, false)

		// Verify loss is computed
		if loss < 0.0 {
			t.Error("loss should be non-negative")
		}

		// Verify error variables are accumulated
		if errVars.Count != 2 {
			t.Errorf("Count = %v, want 2", errVars.Count)
		}

		if errVars.CumErrorTTFT < 0 {
			t.Error("CumErrorTTFT should be non-negative")
		}

		if errVars.CumErrorITL < 0 {
			t.Error("CumErrorITL should be non-negative")
		}

		if errVars.CumErrorWeightedAvg < 0 {
			t.Error("CumErrorWeightedAvg should be non-negative")
		}

		// Verify average is computed correctly
		avgLoss := errVars.CumErrorWeightedAvg / float64(errVars.Count)
		if math.Abs(avgLoss-loss) > 1e-9 {
			t.Errorf("computed loss %v doesn't match average from errVars %v", loss, avgLoss)
		}
	})
}

func TestLossFunction_Consistency(t *testing.T) {
	t.Run("same input produces same output", func(t *testing.T) {
		params := &config.ModelParams{
			Alpha: 5.0,
			Beta:  1.0,
			Gamma: 0.1,
		}

		xData := []*config.InputVars{
			{RequestRate: 10.0, InputTokens: 100.0},
		}

		yData := []*config.OutputVars{
			{AvgTTFTTime: 15.0, AvgITLTime: 10.0},
		}

		// First run
		errVars1 := &config.ErrorVars{}
		loss1 := LossFunction(params, xData, yData, mockLinearModelForLoss, errVars1, false)

		// Second run
		errVars2 := &config.ErrorVars{}
		loss2 := LossFunction(params, xData, yData, mockLinearModelForLoss, errVars2, false)

		if loss1 != loss2 {
			t.Errorf("loss not consistent: first=%v, second=%v", loss1, loss2)
		}

		if errVars1.Count != errVars2.Count {
			t.Error("error vars count not consistent")
		}
	})
}

// Benchmark for Model function
func BenchmarkModel(b *testing.B) {
	inputVars := &config.InputVars{
		RequestRate:  0.1,
		InputTokens:  100.0,
		OutputTokens: 50.0,
		MaxBatchSize: 32,
		MaxNumTokens: 2048,
	}

	params := &config.ModelParams{
		Alpha: 0.5,
		Beta:  1.0,
		Gamma: 0.5,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := Model(inputVars, params)
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
	}
}

// Benchmark for LossFunction
func BenchmarkLossFunction(b *testing.B) {
	params := &config.ModelParams{
		Alpha: 5.0,
		Beta:  1.0,
		Gamma: 0.1,
	}

	xData := []*config.InputVars{
		{RequestRate: 10.0, InputTokens: 100.0},
		{RequestRate: 20.0, InputTokens: 200.0},
		{RequestRate: 30.0, InputTokens: 300.0},
	}

	yData := []*config.OutputVars{
		{AvgTTFTTime: 15.0, AvgITLTime: 10.0},
		{AvgTTFTTime: 25.0, AvgITLTime: 15.0},
		{AvgTTFTTime: 35.0, AvgITLTime: 20.0},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errVars := &config.ErrorVars{}
		_ = LossFunction(params, xData, yData, mockLinearModelForLoss, errVars, false)
	}
}

// Benchmark for LossFunction with large dataset
func BenchmarkLossFunction_LargeDataset(b *testing.B) {
	params := &config.ModelParams{
		Alpha: 5.0,
		Beta:  1.0,
		Gamma: 0.1,
	}

	// Create 100 data points
	xData := make([]*config.InputVars, 100)
	yData := make([]*config.OutputVars, 100)
	for i := 0; i < 100; i++ {
		xData[i] = &config.InputVars{
			RequestRate: float64(i + 1),
			InputTokens: float64((i + 1) * 10),
		}
		yData[i] = &config.OutputVars{
			AvgTTFTTime: float64(10 + i),
			AvgITLTime:  float64(5 + i/2),
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		errVars := &config.ErrorVars{}
		_ = LossFunction(params, xData, yData, mockLinearModelForLoss, errVars, false)
	}
}
