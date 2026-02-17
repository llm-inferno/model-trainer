package core

import (
	"testing"

	"github.com/llm-inferno/model-trainer/pkg/config"
)

func TestDataPoint_GetInOutVars(t *testing.T) {
	tests := []struct {
		name      string
		dataPoint DataPoint
		wantX     *config.InputVars
		wantY     *config.OutputVars
	}{
		{
			name: "valid data point with all fields",
			dataPoint: DataPoint{
				RequestRate:  10.5,
				InputTokens:  100.0,
				OutputTokens: 50.0,
				AvgITLTime:   5.5,
				AvgTTFTTime:  10.2,
				MaxBatchSize: 32,
				MaxNumTokens: 2048,
			},
			wantX: &config.InputVars{
				RequestRate:  10.5,
				InputTokens:  100.0,
				OutputTokens: 50.0,
				MaxBatchSize: 32,
				MaxNumTokens: 2048,
			},
			wantY: &config.OutputVars{
				AvgTTFTTime: 10.2,
				AvgITLTime:  5.5,
			},
		},
		{
			name: "data point with missing MaxBatchSize",
			dataPoint: DataPoint{
				RequestRate:  20.0,
				InputTokens:  200.0,
				OutputTokens: 100.0,
				AvgITLTime:   8.0,
				AvgTTFTTime:  15.0,
				MaxBatchSize: 0,
				MaxNumTokens: 4096,
			},
			wantX: &config.InputVars{
				RequestRate:  20.0,
				InputTokens:  200.0,
				OutputTokens: 100.0,
				MaxBatchSize: config.DefaultMaxBatchSize,
				MaxNumTokens: 4096,
			},
			wantY: &config.OutputVars{
				AvgTTFTTime: 15.0,
				AvgITLTime:  8.0,
			},
		},
		{
			name: "data point with missing MaxNumTokens",
			dataPoint: DataPoint{
				RequestRate:  15.0,
				InputTokens:  150.0,
				OutputTokens: 75.0,
				AvgITLTime:   6.0,
				AvgTTFTTime:  12.0,
				MaxBatchSize: 64,
				MaxNumTokens: 0,
			},
			wantX: &config.InputVars{
				RequestRate:  15.0,
				InputTokens:  150.0,
				OutputTokens: 75.0,
				MaxBatchSize: 64,
				MaxNumTokens: config.DefaultMaxNumTokens,
			},
			wantY: &config.OutputVars{
				AvgTTFTTime: 12.0,
				AvgITLTime:  6.0,
			},
		},
		{
			name: "data point with missing AvgTTFTTime - computed from wait and prefill",
			dataPoint: DataPoint{
				RequestRate:    25.0,
				InputTokens:    250.0,
				OutputTokens:   125.0,
				AvgITLTime:     7.0,
				AvgTTFTTime:    0.0,
				AvgWaitTime:    5.0,
				AvgPrefillTime: 8.0,
				MaxBatchSize:   128,
				MaxNumTokens:   4096,
			},
			wantX: &config.InputVars{
				RequestRate:  25.0,
				InputTokens:  250.0,
				OutputTokens: 125.0,
				MaxBatchSize: 128,
				MaxNumTokens: 4096,
			},
			wantY: &config.OutputVars{
				AvgTTFTTime: 13.0, // 5.0 + 8.0
				AvgITLTime:  7.0,
			},
		},
		{
			name: "data point with all missing fields",
			dataPoint: DataPoint{
				RequestRate:    5.0,
				InputTokens:    50.0,
				OutputTokens:   25.0,
				AvgITLTime:     3.0,
				AvgTTFTTime:    0.0,
				AvgWaitTime:    2.0,
				AvgPrefillTime: 3.0,
				MaxBatchSize:   0,
				MaxNumTokens:   0,
			},
			wantX: &config.InputVars{
				RequestRate:  5.0,
				InputTokens:  50.0,
				OutputTokens: 25.0,
				MaxBatchSize: config.DefaultMaxBatchSize,
				MaxNumTokens: config.DefaultMaxNumTokens,
			},
			wantY: &config.OutputVars{
				AvgTTFTTime: 5.0, // 2.0 + 3.0
				AvgITLTime:  3.0,
			},
		},
		{
			name: "data point with zero values",
			dataPoint: DataPoint{
				RequestRate:  0.0,
				InputTokens:  0.0,
				OutputTokens: 0.0,
				AvgITLTime:   0.0,
				AvgTTFTTime:  0.0,
				MaxBatchSize: 0,
				MaxNumTokens: 0,
			},
			wantX: &config.InputVars{
				RequestRate:  0.0,
				InputTokens:  0.0,
				OutputTokens: 0.0,
				MaxBatchSize: config.DefaultMaxBatchSize,
				MaxNumTokens: config.DefaultMaxNumTokens,
			},
			wantY: &config.OutputVars{
				AvgTTFTTime: 0.0,
				AvgITLTime:  0.0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotX, gotY := tt.dataPoint.GetInOutVars()

			// Check InputVars
			if gotX.RequestRate != tt.wantX.RequestRate {
				t.Errorf("InputVars.RequestRate = %v, want %v", gotX.RequestRate, tt.wantX.RequestRate)
			}
			if gotX.InputTokens != tt.wantX.InputTokens {
				t.Errorf("InputVars.InputTokens = %v, want %v", gotX.InputTokens, tt.wantX.InputTokens)
			}
			if gotX.OutputTokens != tt.wantX.OutputTokens {
				t.Errorf("InputVars.OutputTokens = %v, want %v", gotX.OutputTokens, tt.wantX.OutputTokens)
			}
			if gotX.MaxBatchSize != tt.wantX.MaxBatchSize {
				t.Errorf("InputVars.MaxBatchSize = %v, want %v", gotX.MaxBatchSize, tt.wantX.MaxBatchSize)
			}
			if gotX.MaxNumTokens != tt.wantX.MaxNumTokens {
				t.Errorf("InputVars.MaxNumTokens = %v, want %v", gotX.MaxNumTokens, tt.wantX.MaxNumTokens)
			}

			// Check OutputVars
			if gotY.AvgTTFTTime != tt.wantY.AvgTTFTTime {
				t.Errorf("OutputVars.AvgTTFTTime = %v, want %v", gotY.AvgTTFTTime, tt.wantY.AvgTTFTTime)
			}
			if gotY.AvgITLTime != tt.wantY.AvgITLTime {
				t.Errorf("OutputVars.AvgITLTime = %v, want %v", gotY.AvgITLTime, tt.wantY.AvgITLTime)
			}
		})
	}
}

func TestDataPoint_Fix(t *testing.T) {
	tests := []struct {
		name     string
		initial  DataPoint
		expected DataPoint
	}{
		{
			name: "fix missing MaxBatchSize",
			initial: DataPoint{
				RequestRate:  10.0,
				MaxBatchSize: 0,
				MaxNumTokens: 2048,
				AvgTTFTTime:  10.0,
			},
			expected: DataPoint{
				RequestRate:  10.0,
				MaxBatchSize: config.DefaultMaxBatchSize,
				MaxNumTokens: 2048,
				AvgTTFTTime:  10.0,
			},
		},
		{
			name: "fix negative MaxBatchSize",
			initial: DataPoint{
				RequestRate:  10.0,
				MaxBatchSize: -5,
				MaxNumTokens: 2048,
				AvgTTFTTime:  10.0,
			},
			expected: DataPoint{
				RequestRate:  10.0,
				MaxBatchSize: config.DefaultMaxBatchSize,
				MaxNumTokens: 2048,
				AvgTTFTTime:  10.0,
			},
		},
		{
			name: "fix missing MaxNumTokens",
			initial: DataPoint{
				RequestRate:  10.0,
				MaxBatchSize: 32,
				MaxNumTokens: 0,
				AvgTTFTTime:  10.0,
			},
			expected: DataPoint{
				RequestRate:  10.0,
				MaxBatchSize: 32,
				MaxNumTokens: config.DefaultMaxNumTokens,
				AvgTTFTTime:  10.0,
			},
		},
		{
			name: "fix negative MaxNumTokens",
			initial: DataPoint{
				RequestRate:  10.0,
				MaxBatchSize: 32,
				MaxNumTokens: -100,
				AvgTTFTTime:  10.0,
			},
			expected: DataPoint{
				RequestRate:  10.0,
				MaxBatchSize: 32,
				MaxNumTokens: config.DefaultMaxNumTokens,
				AvgTTFTTime:  10.0,
			},
		},
		{
			name: "fix missing AvgTTFTTime with wait and prefill times",
			initial: DataPoint{
				RequestRate:    10.0,
				MaxBatchSize:   32,
				MaxNumTokens:   2048,
				AvgTTFTTime:    0.0,
				AvgWaitTime:    5.5,
				AvgPrefillTime: 7.3,
			},
			expected: DataPoint{
				RequestRate:    10.0,
				MaxBatchSize:   32,
				MaxNumTokens:   2048,
				AvgTTFTTime:    12.8, // 5.5 + 7.3
				AvgWaitTime:    5.5,
				AvgPrefillTime: 7.3,
			},
		},
		{
			name: "fix negative AvgTTFTTime with wait and prefill times",
			initial: DataPoint{
				RequestRate:    10.0,
				MaxBatchSize:   32,
				MaxNumTokens:   2048,
				AvgTTFTTime:    -1.0,
				AvgWaitTime:    3.0,
				AvgPrefillTime: 4.0,
			},
			expected: DataPoint{
				RequestRate:    10.0,
				MaxBatchSize:   32,
				MaxNumTokens:   2048,
				AvgTTFTTime:    7.0, // 3.0 + 4.0
				AvgWaitTime:    3.0,
				AvgPrefillTime: 4.0,
			},
		},
		{
			name: "fix all missing fields",
			initial: DataPoint{
				RequestRate:    10.0,
				MaxBatchSize:   0,
				MaxNumTokens:   0,
				AvgTTFTTime:    0.0,
				AvgWaitTime:    2.5,
				AvgPrefillTime: 3.5,
			},
			expected: DataPoint{
				RequestRate:    10.0,
				MaxBatchSize:   config.DefaultMaxBatchSize,
				MaxNumTokens:   config.DefaultMaxNumTokens,
				AvgTTFTTime:    6.0, // 2.5 + 3.5
				AvgWaitTime:    2.5,
				AvgPrefillTime: 3.5,
			},
		},
		{
			name: "no fixes needed - all fields valid",
			initial: DataPoint{
				RequestRate:  10.0,
				MaxBatchSize: 64,
				MaxNumTokens: 4096,
				AvgTTFTTime:  15.0,
			},
			expected: DataPoint{
				RequestRate:  10.0,
				MaxBatchSize: 64,
				MaxNumTokens: 4096,
				AvgTTFTTime:  15.0,
			},
		},
		{
			name: "missing AvgTTFTTime but no wait/prefill times",
			initial: DataPoint{
				RequestRate:    10.0,
				MaxBatchSize:   32,
				MaxNumTokens:   2048,
				AvgTTFTTime:    0.0,
				AvgWaitTime:    0.0,
				AvgPrefillTime: 0.0,
			},
			expected: DataPoint{
				RequestRate:    10.0,
				MaxBatchSize:   32,
				MaxNumTokens:   2048,
				AvgTTFTTime:    0.0, // 0.0 + 0.0
				AvgWaitTime:    0.0,
				AvgPrefillTime: 0.0,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dp := tt.initial
			dp.Fix()

			if dp.MaxBatchSize != tt.expected.MaxBatchSize {
				t.Errorf("MaxBatchSize = %v, want %v", dp.MaxBatchSize, tt.expected.MaxBatchSize)
			}
			if dp.MaxNumTokens != tt.expected.MaxNumTokens {
				t.Errorf("MaxNumTokens = %v, want %v", dp.MaxNumTokens, tt.expected.MaxNumTokens)
			}
			if dp.AvgTTFTTime != tt.expected.AvgTTFTTime {
				t.Errorf("AvgTTFTTime = %v, want %v", dp.AvgTTFTTime, tt.expected.AvgTTFTTime)
			}
		})
	}
}

func TestDataPoint_ToMSecs(t *testing.T) {
	tests := []struct {
		name     string
		initial  DataPoint
		expected DataPoint
	}{
		{
			name: "convert all time fields from seconds to milliseconds",
			initial: DataPoint{
				AvgTTFTTime:    1.5,
				AvgITLTime:     0.5,
				AvgWaitTime:    0.75,
				AvgPrefillTime: 0.25,
			},
			expected: DataPoint{
				AvgTTFTTime:    1500.0,
				AvgITLTime:     500.0,
				AvgWaitTime:    750.0,
				AvgPrefillTime: 250.0,
			},
		},
		{
			name: "convert zero time fields",
			initial: DataPoint{
				AvgTTFTTime:    0.0,
				AvgITLTime:     0.0,
				AvgWaitTime:    0.0,
				AvgPrefillTime: 0.0,
			},
			expected: DataPoint{
				AvgTTFTTime:    0.0,
				AvgITLTime:     0.0,
				AvgWaitTime:    0.0,
				AvgPrefillTime: 0.0,
			},
		},
		{
			name: "convert fractional time fields",
			initial: DataPoint{
				AvgTTFTTime:    0.0123,
				AvgITLTime:     0.0045,
				AvgWaitTime:    0.0067,
				AvgPrefillTime: 0.0089,
			},
			expected: DataPoint{
				AvgTTFTTime:    12.3,
				AvgITLTime:     4.5,
				AvgWaitTime:    6.7,
				AvgPrefillTime: 8.9,
			},
		},
		{
			name: "convert large time values",
			initial: DataPoint{
				AvgTTFTTime:    100.5,
				AvgITLTime:     50.25,
				AvgWaitTime:    75.75,
				AvgPrefillTime: 25.125,
			},
			expected: DataPoint{
				AvgTTFTTime:    100500.0,
				AvgITLTime:     50250.0,
				AvgWaitTime:    75750.0,
				AvgPrefillTime: 25125.0,
			},
		},
		{
			name: "convert mixed positive time values",
			initial: DataPoint{
				RequestRate:    10.0,
				InputTokens:    100.0,
				OutputTokens:   50.0,
				AvgTTFTTime:    2.0,
				AvgITLTime:     1.0,
				AvgWaitTime:    0.5,
				AvgPrefillTime: 1.5,
				MaxBatchSize:   32,
				MaxNumTokens:   2048,
			},
			expected: DataPoint{
				RequestRate:    10.0,
				InputTokens:    100.0,
				OutputTokens:   50.0,
				AvgTTFTTime:    2000.0,
				AvgITLTime:     1000.0,
				AvgWaitTime:    500.0,
				AvgPrefillTime: 1500.0,
				MaxBatchSize:   32,
				MaxNumTokens:   2048,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dp := tt.initial
			dp.ToMSecs()

			if dp.AvgTTFTTime != tt.expected.AvgTTFTTime {
				t.Errorf("AvgTTFTTime = %v, want %v", dp.AvgTTFTTime, tt.expected.AvgTTFTTime)
			}
			if dp.AvgITLTime != tt.expected.AvgITLTime {
				t.Errorf("AvgITLTime = %v, want %v", dp.AvgITLTime, tt.expected.AvgITLTime)
			}
			if dp.AvgWaitTime != tt.expected.AvgWaitTime {
				t.Errorf("AvgWaitTime = %v, want %v", dp.AvgWaitTime, tt.expected.AvgWaitTime)
			}
			if dp.AvgPrefillTime != tt.expected.AvgPrefillTime {
				t.Errorf("AvgPrefillTime = %v, want %v", dp.AvgPrefillTime, tt.expected.AvgPrefillTime)
			}

			// Verify non-time fields are unchanged
			if dp.RequestRate != tt.expected.RequestRate {
				t.Errorf("RequestRate should not change: got %v, want %v", dp.RequestRate, tt.expected.RequestRate)
			}
			if dp.InputTokens != tt.expected.InputTokens {
				t.Errorf("InputTokens should not change: got %v, want %v", dp.InputTokens, tt.expected.InputTokens)
			}
			if dp.OutputTokens != tt.expected.OutputTokens {
				t.Errorf("OutputTokens should not change: got %v, want %v", dp.OutputTokens, tt.expected.OutputTokens)
			}
			if dp.MaxBatchSize != tt.expected.MaxBatchSize {
				t.Errorf("MaxBatchSize should not change: got %v, want %v", dp.MaxBatchSize, tt.expected.MaxBatchSize)
			}
			if dp.MaxNumTokens != tt.expected.MaxNumTokens {
				t.Errorf("MaxNumTokens should not change: got %v, want %v", dp.MaxNumTokens, tt.expected.MaxNumTokens)
			}
		})
	}
}

func TestDataPoint_FixIdempotency(t *testing.T) {
	dp := DataPoint{
		RequestRate:    10.0,
		MaxBatchSize:   0,
		MaxNumTokens:   0,
		AvgTTFTTime:    0.0,
		AvgWaitTime:    5.0,
		AvgPrefillTime: 3.0,
	}

	// First fix
	dp.Fix()
	first := dp

	// Second fix
	dp.Fix()
	second := dp

	// Results should be the same (idempotent)
	if first != second {
		t.Error("Fix() is not idempotent")
	}
}

func TestDataPoint_ToMSecsNotIdempotent(t *testing.T) {
	dp := DataPoint{
		AvgTTFTTime:    1.0,
		AvgITLTime:     0.5,
		AvgWaitTime:    0.25,
		AvgPrefillTime: 0.75,
	}

	// First conversion
	dp.ToMSecs()
	if dp.AvgTTFTTime != 1000.0 {
		t.Errorf("First conversion: AvgTTFTTime = %v, want 1000.0", dp.AvgTTFTTime)
	}

	// Second conversion (multiplies again)
	dp.ToMSecs()
	if dp.AvgTTFTTime != 1000000.0 {
		t.Errorf("Second conversion: AvgTTFTTime = %v, want 1000000.0", dp.AvgTTFTTime)
	}
}

// Benchmark for GetInOutVars
func BenchmarkDataPoint_GetInOutVars(b *testing.B) {
	dp := DataPoint{
		RequestRate:  10.0,
		InputTokens:  100.0,
		OutputTokens: 50.0,
		AvgITLTime:   5.5,
		AvgTTFTTime:  10.2,
		MaxBatchSize: 32,
		MaxNumTokens: 2048,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = dp.GetInOutVars()
	}
}

// Benchmark for Fix
func BenchmarkDataPoint_Fix(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dp := DataPoint{
			RequestRate:    10.0,
			MaxBatchSize:   0,
			MaxNumTokens:   0,
			AvgTTFTTime:    0.0,
			AvgWaitTime:    5.0,
			AvgPrefillTime: 3.0,
		}
		dp.Fix()
	}
}

// Benchmark for ToMSecs
func BenchmarkDataPoint_ToMSecs(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dp := DataPoint{
			AvgTTFTTime:    1.5,
			AvgITLTime:     0.5,
			AvgWaitTime:    0.75,
			AvgPrefillTime: 0.25,
		}
		dp.ToMSecs()
	}
}
