package reader

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/llm-inferno/model-trainer/pkg/config"
	"github.com/llm-inferno/model-trainer/pkg/core"
)

// data in a GuideLLM HTML results file
type GuideLLMHTMLData struct {
	Benchmarks       []BenchmarkHTML
	PromptTokenStats *TokenStats
	OutputTokenStats *TokenStats
}

// benchmark data extracted from GuideLLM HTML results file
type BenchmarkHTML struct {
	Strategy     string
	RPS          float64
	Latency      float64
	InputTokens  float64
	OutputTokens float64
	TTFT         float64
	TPOT         float64
	ITL          float64
}

// Statistical data structure from HTML
type StatisticsData struct {
	Mean   float64 `json:"mean"`
	Median float64 `json:"median"`
	StdDev float64 `json:"stdDev"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
}

// Token statistics from workloadDetails
type TokenStats struct {
	Mean   float64
	Median float64
	StdDev float64
}

// Raw benchmark structure from HTML
type RawBenchmark struct {
	ITL               StatisticsData `json:"itl"`
	TTFT              StatisticsData `json:"ttft"`
	Throughput        StatisticsData `json:"throughput"`
	RequestsPerSecond float64        `json:"requestsPerSecond"`
	TimePerRequest    StatisticsData `json:"timePerRequest"`
}

func NewGuideLLMHTMLData() *GuideLLMHTMLData {
	return &GuideLLMHTMLData{}
}

func (g *GuideLLMHTMLData) ReadFrom(dataBytes []byte) error {
	htmlContent := string(dataBytes)

	// Extract window.benchmarks array using bracket counting for nested JSON
	benchmarksJSON, err := extractJSONArray(htmlContent, "window.benchmarks")
	if err != nil {
		return fmt.Errorf("could not find window.benchmarks in HTML: %w", err)
	}

	var rawBenchmarks []RawBenchmark
	if err := json.Unmarshal([]byte(benchmarksJSON), &rawBenchmarks); err != nil {
		return fmt.Errorf("failed to parse benchmarks JSON: %w", err)
	}

	// Extract token statistics from workloadDetails
	g.extractTokenStats(htmlContent)

	// Convert raw benchmarks to our format
	for i, raw := range rawBenchmarks {
		strategy := g.inferStrategy(i)

		// Use average token counts from overall statistics
		inputTokens := 64.0  // default
		outputTokens := 64.0 // default
		if g.PromptTokenStats != nil {
			inputTokens = g.PromptTokenStats.Mean
		}
		if g.OutputTokenStats != nil {
			outputTokens = g.OutputTokenStats.Mean
		}

		benchmark := BenchmarkHTML{
			Strategy:     strategy,
			RPS:          raw.RequestsPerSecond,
			Latency:      raw.TimePerRequest.Mean,
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
			TTFT:         raw.TTFT.Median,
			TPOT:         raw.ITL.Mean,
			ITL:          raw.ITL.Mean,
		}
		g.Benchmarks = append(g.Benchmarks, benchmark)
	}

	return nil
}

func (g *GuideLLMHTMLData) extractTokenStats(htmlContent string) {
	// Try to extract token distribution statistics from workloadDetails
	workloadJSON, err := extractJSONObject(htmlContent, "window.workloadDetails")
	if err != nil {
		return
	}

	var workload map[string]interface{}
	if err := json.Unmarshal([]byte(workloadJSON), &workload); err != nil {
		return
	}

	// Extract prompt token statistics
	if prompts, ok := workload["prompts"].(map[string]interface{}); ok {
		if tokenDist, ok := prompts["tokenDistributions"].(map[string]interface{}); ok {
			if stats, ok := tokenDist["statistics"].(map[string]interface{}); ok {
				g.PromptTokenStats = &TokenStats{
					Mean:   getFloat(stats, "mean"),
					Median: getFloat(stats, "median"),
					StdDev: getFloat(stats, "stdDev"),
				}
			}
		}
	}

	// Extract generation/output token statistics
	if generations, ok := workload["generations"].(map[string]interface{}); ok {
		if tokenDist, ok := generations["tokenDistributions"].(map[string]interface{}); ok {
			if stats, ok := tokenDist["statistics"].(map[string]interface{}); ok {
				g.OutputTokenStats = &TokenStats{
					Mean:   getFloat(stats, "mean"),
					Median: getFloat(stats, "median"),
					StdDev: getFloat(stats, "stdDev"),
				}
			}
		}
	}
}

func getFloat(m map[string]interface{}, key string) float64 {
	if val, ok := m[key].(float64); ok {
		return val
	}
	return 0.0
}

// extractJSONArray extracts a JSON array from HTML by finding the variable and properly handling nested brackets
func extractJSONArray(content, varName string) (string, error) {
	// Find the start of the assignment
	pattern := varName + `\s*=\s*\[`
	re := regexp.MustCompile(pattern)
	loc := re.FindStringIndex(content)
	if loc == nil {
		return "", fmt.Errorf("variable %s not found", varName)
	}

	// Find where the array starts (after the '=')
	startIdx := strings.Index(content[loc[0]:], "[")
	if startIdx == -1 {
		return "", fmt.Errorf("opening bracket not found")
	}
	startIdx += loc[0]

	// Count brackets to find the matching closing bracket
	depth := 0
	inString := false
	escape := false

	for i := startIdx; i < len(content); i++ {
		c := content[i]

		// Handle escape sequences
		if escape {
			escape = false
			continue
		}
		if c == '\\' {
			escape = true
			continue
		}

		// Handle strings (ignore brackets inside strings)
		if c == '"' {
			inString = !inString
			continue
		}

		if !inString {
			if c == '[' || c == '{' {
				depth++
			} else if c == ']' || c == '}' {
				depth--
				if depth == 0 && c == ']' {
					// Found the matching closing bracket
					return content[startIdx : i+1], nil
				}
			}
		}
	}

	return "", fmt.Errorf("matching closing bracket not found")
}

// extractJSONObject extracts a JSON object from HTML by finding the variable and properly handling nested brackets
func extractJSONObject(content, varName string) (string, error) {
	// Find the start of the assignment
	pattern := varName + `\s*=\s*\{`
	re := regexp.MustCompile(pattern)
	loc := re.FindStringIndex(content)
	if loc == nil {
		return "", fmt.Errorf("variable %s not found", varName)
	}

	// Find where the object starts (after the '=')
	startIdx := strings.Index(content[loc[0]:], "{")
	if startIdx == -1 {
		return "", fmt.Errorf("opening brace not found")
	}
	startIdx += loc[0]

	// Count brackets to find the matching closing brace
	depth := 0
	inString := false
	escape := false

	for i := startIdx; i < len(content); i++ {
		c := content[i]

		// Handle escape sequences
		if escape {
			escape = false
			continue
		}
		if c == '\\' {
			escape = true
			continue
		}

		// Handle strings (ignore braces inside strings)
		if c == '"' {
			inString = !inString
			continue
		}

		if !inString {
			if c == '[' || c == '{' {
				depth++
			} else if c == ']' || c == '}' {
				depth--
				if depth == 0 && c == '}' {
					// Found the matching closing brace
					return content[startIdx : i+1], nil
				}
			}
		}
	}

	return "", fmt.Errorf("matching closing brace not found")
}

func (g *GuideLLMHTMLData) inferStrategy(index int) string {
	// First benchmark is typically synchronous
	if index == 0 {
		return "synchronous"
	}

	// Second benchmark is typically throughput
	if index == 1 {
		return "throughput"
	}

	// Everything else is constant rate
	return "constant"
}

func (g *GuideLLMHTMLData) CreateDataSet() *core.DataSet {
	dataSet := core.NewDataSet("GuideLLM HTML benchmark data")
	for _, benchmark := range g.Benchmarks {
		if strings.EqualFold(benchmark.Strategy, "throughput") {
			// skip throughput benchmark data point
			continue
		}
		if dataSet.Size() >= DefaultLimitNumDataPoints {
			// limit number of data points for model training
			break
		}
		dataPoint := &core.DataPoint{
			RequestRate:  benchmark.RPS,
			InputTokens:  benchmark.InputTokens,
			OutputTokens: benchmark.OutputTokens,
			AvgTTFTTime:  benchmark.TTFT,
			AvgITLTime:   benchmark.ITL,
			MaxBatchSize: config.DefaultMaxBatchSize,
			MaxNumTokens: config.DefaultMaxNumTokens,
		}
		dataSet.AppendDataPoint(dataPoint)
	}
	return dataSet
}

func (g *GuideLLMHTMLData) Print() {
	fmt.Println("Token Statistics:")
	if g.PromptTokenStats != nil {
		fmt.Printf("  Prompt Tokens: Mean=%.2f, Median=%.2f, StdDev=%.2f\n",
			g.PromptTokenStats.Mean, g.PromptTokenStats.Median, g.PromptTokenStats.StdDev)
	}
	if g.OutputTokenStats != nil {
		fmt.Printf("  Output Tokens: Mean=%.2f, Median=%.2f, StdDev=%.2f\n",
			g.OutputTokenStats.Mean, g.OutputTokenStats.Median, g.OutputTokenStats.StdDev)
	}
	fmt.Println("\nBenchmarks:")
	for i, benchmark := range g.Benchmarks {
		fmt.Printf("Benchmark %d:\n", i)
		fmt.Printf("  Strategy: %s\n", benchmark.Strategy)
		fmt.Printf("  RPS: %.2f\n", benchmark.RPS)
		fmt.Printf("  Latency: %.2f sec\n", benchmark.Latency)
		fmt.Printf("  Input Tokens: %.2f\n", benchmark.InputTokens)
		fmt.Printf("  Output Tokens: %.2f\n", benchmark.OutputTokens)
		fmt.Printf("  TTFT: %.2f ms\n", benchmark.TTFT)
		fmt.Printf("  TPOT: %.2f ms\n", benchmark.TPOT)
		fmt.Printf("  ITL: %.2f ms\n", benchmark.ITL)
	}
}

func (g *GuideLLMHTMLData) Dump() string {
	if jsonStr, err := json.Marshal(g); err == nil {
		return fmt.Sprintf("GuideLLM HTML Benchmarks data: %v\n", string(jsonStr))
	}
	return "GuideLLM HTML Benchmarks data: <unavailable>"
}
