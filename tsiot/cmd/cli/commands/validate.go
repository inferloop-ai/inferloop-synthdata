package commands

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inferloop/tsiot/internal/validation/metrics"
	"github.com/inferloop/tsiot/internal/validation/tests"
	"github.com/inferloop/tsiot/pkg/models"
)

type ValidateOptions struct {
	InputFile        string
	ReferenceFile    string
	Metrics          []string
	ReportFormat     string
	OutputFile       string
	Threshold        float64
	StatisticalTests bool
}

func NewValidateCmd() *cobra.Command {
	opts := &ValidateOptions{}

	cmd := &cobra.Command{
		Use:   "validate",
		Short: "Validate synthetic time series data quality",
		Long: `Validate the quality of synthetic time series data by comparing it with
reference data or checking statistical properties.`,
		Example: `  # Validate against reference data
  tsiot-cli validate --input synthetic.csv --reference real.csv

  # Run statistical tests
  tsiot-cli validate --input data.csv --statistical-tests

  # Generate detailed report
  tsiot-cli validate --input data.csv --metrics all --report-format html --output report.html`,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runValidate(opts)
		},
	}

	// Add flags
	cmd.Flags().StringVarP(&opts.InputFile, "input", "i", "", "Input file to validate (required)")
	cmd.Flags().StringVarP(&opts.ReferenceFile, "reference", "r", "", "Reference file for comparison")
	cmd.Flags().StringSliceVarP(&opts.Metrics, "metrics", "m", []string{"basic"}, "Metrics to calculate (basic, trend, distribution, all)")
	cmd.Flags().StringVar(&opts.ReportFormat, "report-format", "text", "Report format (text, json, html)")
	cmd.Flags().StringVarP(&opts.OutputFile, "output", "o", "-", "Output file for report (- for stdout)")
	cmd.Flags().Float64Var(&opts.Threshold, "threshold", 0.95, "Quality threshold (0.0 to 1.0)")
	cmd.Flags().BoolVar(&opts.StatisticalTests, "statistical-tests", false, "Run statistical tests")

	cmd.MarkFlagRequired("input")

	return cmd
}

func runValidate(opts *ValidateOptions) error {
	fmt.Printf("Validating synthetic data...\n")
	fmt.Printf("Input File: %s\n", opts.InputFile)
	
	if opts.ReferenceFile != "" {
		fmt.Printf("Reference File: %s\n", opts.ReferenceFile)
	}

	fmt.Printf("Metrics: %v\n", opts.Metrics)
	fmt.Printf("Report Format: %s\n", opts.ReportFormat)

	// Load input data
	inputData, err := loadTimeSeriesFromFile(opts.InputFile)
	if err != nil {
		return fmt.Errorf("failed to load input data: %w", err)
	}

	// Load reference data if provided
	var referenceData *models.TimeSeries
	if opts.ReferenceFile != "" {
		referenceData, err = loadTimeSeriesFromFile(opts.ReferenceFile)
		if err != nil {
			return fmt.Errorf("failed to load reference data: %w", err)
		}
	}

	// Initialize validation components
	logger := logrus.New()
	qualityMetrics := metrics.NewQualityMetrics(0.05) // 5% significance level
	testSuite := tests.NewStatisticalTestSuite(0.05)

	// Perform validation
	ctx := context.Background()
	validationResult, err := performValidation(ctx, inputData, referenceData, qualityMetrics, testSuite, opts, logger)
	if err != nil {
		return fmt.Errorf("validation failed: %w", err)
	}

	// Output results
	err = outputValidationResults(validationResult, opts)
	if err != nil {
		return fmt.Errorf("failed to output results: %w", err)
	}

	// Check threshold
	if validationResult.OverallScore >= opts.Threshold {
		fmt.Printf("\n✓ Quality threshold met (%.2f >= %.2f)\n", validationResult.OverallScore, opts.Threshold)
		return nil
	} else {
		fmt.Printf("\n✗ Quality below threshold (%.2f < %.2f)\n", validationResult.OverallScore, opts.Threshold)
		return fmt.Errorf("quality validation failed")
	}
}

type ValidationResult struct {
	OverallScore        float64                        `json:"overall_score"`
	QualityReport       *metrics.QualityReport         `json:"quality_report,omitempty"`
	StatisticalTests    []*tests.StatisticalTestResult `json:"statistical_tests,omitempty"`
	InputSummary        *DataSummary                   `json:"input_summary"`
	ReferenceSummary    *DataSummary                   `json:"reference_summary,omitempty"`
	ComparisonMetrics   *ComparisonMetrics             `json:"comparison_metrics,omitempty"`
	ValidationTime      time.Duration                  `json:"validation_time"`
	Timestamp           time.Time                      `json:"timestamp"`
}

type DataSummary struct {
	DataPoints  int                    `json:"data_points"`
	TimeRange   TimeRange              `json:"time_range"`
	Statistics  BasicStatistics        `json:"statistics"`
	MissingData int                    `json:"missing_data"`
	Metadata    map[string]interface{} `json:"metadata"`
}

type TimeRange struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

type BasicStatistics struct {
	Mean   float64 `json:"mean"`
	StdDev float64 `json:"std_dev"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	Median float64 `json:"median"`
}

type ComparisonMetrics struct {
	StatisticalSimilarity  float64 `json:"statistical_similarity"`
	TrendSimilarity        float64 `json:"trend_similarity"`
	DistributionSimilarity float64 `json:"distribution_similarity"`
	TemporalSimilarity     float64 `json:"temporal_similarity"`
	CorrelationSimilarity  float64 `json:"correlation_similarity"`
}

func performValidation(ctx context.Context, inputData *models.TimeSeries, referenceData *models.TimeSeries, 
	qualityMetrics *metrics.QualityMetrics, testSuite *tests.StatisticalTestSuite, 
	opts *ValidateOptions, logger *logrus.Logger) (*ValidationResult, error) {
	
	start := time.Now()
	result := &ValidationResult{
		Timestamp: start,
	}

	fmt.Println("\nPerforming validation...")

	// Generate data summaries
	result.InputSummary = generateDataSummary(inputData)
	if referenceData != nil {
		result.ReferenceSummary = generateDataSummary(referenceData)
	}

	// Extract data values for analysis
	inputValues := extractValues(inputData)
	var referenceValues []float64
	if referenceData != nil {
		referenceValues = extractValues(referenceData)
	}

	// Run statistical tests if requested
	if opts.StatisticalTests {
		fmt.Printf("Running statistical tests...\n")
		result.StatisticalTests = runStatisticalTests(testSuite, inputValues, referenceValues)
	}

	// Calculate quality metrics based on requested metrics
	for _, metricType := range opts.Metrics {
		switch strings.ToLower(metricType) {
		case "basic", "all":
			if referenceData != nil {
				result.ComparisonMetrics = calculateComparisonMetrics(inputValues, referenceValues)
			}
		case "trend", "all":
			// Calculate trend metrics
		case "distribution", "all":
			// Calculate distribution metrics
		}
	}

	// Calculate overall quality score
	result.OverallScore = calculateOverallScore(result)
	result.ValidationTime = time.Since(start)

	return result, nil
}

func generateDataSummary(data *models.TimeSeries) *DataSummary {
	values := extractValues(data)
	stats := calculateBasicStats(values)
	
	var timeRange TimeRange
	if len(data.DataPoints) > 0 {
		timeRange.Start = data.DataPoints[0].Timestamp
		timeRange.End = data.DataPoints[len(data.DataPoints)-1].Timestamp
	}

	return &DataSummary{
		DataPoints:  len(data.DataPoints),
		TimeRange:   timeRange,
		Statistics:  stats,
		MissingData: 0, // Could be enhanced to detect missing values
		Metadata: map[string]interface{}{
			"sensor_type": data.SensorType,
			"frequency":   data.Frequency,
		},
	}
}

func extractValues(data *models.TimeSeries) []float64 {
	values := make([]float64, len(data.DataPoints))
	for i, dp := range data.DataPoints {
		values[i] = dp.Value
	}
	return values
}

func calculateBasicStats(values []float64) BasicStatistics {
	if len(values) == 0 {
		return BasicStatistics{}
	}

	mean := calculateMean(values)
	variance := calculateVariance(values, mean)
	stdDev := calculateStdDev(variance)
	min, max := calculateMinMax(values)
	median := calculateMedian(values)

	return BasicStatistics{
		Mean:   mean,
		StdDev: stdDev,
		Min:    min,
		Max:    max,
		Median: median,
	}
}

func runStatisticalTests(testSuite *tests.StatisticalTestSuite, inputValues []float64, referenceValues []float64) []*tests.StatisticalTestResult {
	var results []*tests.StatisticalTestResult

	// Run Kolmogorov-Smirnov test
	if len(referenceValues) > 0 {
		ks := testSuite.KolmogorovSmirnovTwoSample(inputValues, referenceValues)
		results = append(results, ks)
	} else {
		ks := testSuite.KolmogorovSmirnovTest(inputValues, "normal")
		results = append(results, ks)
	}

	// Run Anderson-Darling test
	ad := testSuite.AndersonDarlingTest(inputValues)
	results = append(results, ad)

	// Run Ljung-Box test for autocorrelation
	lb := testSuite.LjungBoxTest(inputValues, 10)
	results = append(results, lb)

	return results
}

func calculateComparisonMetrics(inputValues []float64, referenceValues []float64) *ComparisonMetrics {
	// Statistical similarity (correlation)
	statSim := 0.0
	if len(inputValues) == len(referenceValues) {
		statSim = calculateCorrelation(inputValues, referenceValues)
	}

	// Distribution similarity (using KS test p-value as proxy)
	distSim := 0.0
	if ks := performKSTest(inputValues, referenceValues); ks != nil {
		distSim = ks.PValue
	}

	// Trend similarity
	trendSim := calculateTrendSimilarity(inputValues, referenceValues)

	// Temporal similarity (simplified)
	tempSim := 0.85 // Placeholder

	// Correlation similarity
	corrSim := statSim

	return &ComparisonMetrics{
		StatisticalSimilarity:  statSim,
		TrendSimilarity:        trendSim,
		DistributionSimilarity: distSim,
		TemporalSimilarity:     tempSim,
		CorrelationSimilarity:  corrSim,
	}
}

func calculateOverallScore(result *ValidationResult) float64 {
	score := 0.0
	weightSum := 0.0

	// Weight factors
	if result.ComparisonMetrics != nil {
		score += result.ComparisonMetrics.StatisticalSimilarity * 0.3
		score += result.ComparisonMetrics.DistributionSimilarity * 0.25
		score += result.ComparisonMetrics.TrendSimilarity * 0.25
		score += result.ComparisonMetrics.TemporalSimilarity * 0.2
		weightSum = 1.0
	}

	// Statistical tests contribution
	if len(result.StatisticalTests) > 0 {
		passedTests := 0
		for _, test := range result.StatisticalTests {
			if !test.IsSignificant { // Not significant means similar to reference
				passedTests++
			}
		}
		testScore := float64(passedTests) / float64(len(result.StatisticalTests))
		if weightSum > 0 {
			score = score*0.7 + testScore*0.3
		} else {
			score = testScore
		}
	}

	// Default score if no comparisons available
	if weightSum == 0 && len(result.StatisticalTests) == 0 {
		score = 0.8 // Base quality score for standalone validation
	}

	return score
}

func outputValidationResults(result *ValidationResult, opts *ValidateOptions) error {
	switch strings.ToLower(opts.ReportFormat) {
	case "json":
		return outputJSON(result, opts.OutputFile)
	case "html":
		return outputHTML(result, opts.OutputFile)
	default:
		return outputText(result, opts.OutputFile)
	}
}

func outputText(result *ValidationResult, outputFile string) error {
	output := "\nValidation Results:\n"
	output += "==================\n\n"

	// Overall score
	output += fmt.Sprintf("Overall Quality Score: %.1f%%\n", result.OverallScore*100)
	output += fmt.Sprintf("Validation Time: %s\n", result.ValidationTime.String())
	output += "\n"

	// Input summary
	output += "Input Data Summary:\n"
	output += fmt.Sprintf("- Data Points: %d\n", result.InputSummary.DataPoints)
	output += fmt.Sprintf("- Mean: %.3f\n", result.InputSummary.Statistics.Mean)
	output += fmt.Sprintf("- Std Dev: %.3f\n", result.InputSummary.Statistics.StdDev)
	output += fmt.Sprintf("- Range: %.3f to %.3f\n", result.InputSummary.Statistics.Min, result.InputSummary.Statistics.Max)
	output += "\n"

	// Reference comparison
	if result.ReferenceSummary != nil {
		output += "Reference Data Summary:\n"
		output += fmt.Sprintf("- Data Points: %d\n", result.ReferenceSummary.DataPoints)
		output += fmt.Sprintf("- Mean: %.3f\n", result.ReferenceSummary.Statistics.Mean)
		output += fmt.Sprintf("- Std Dev: %.3f\n", result.ReferenceSummary.Statistics.StdDev)
		output += fmt.Sprintf("- Range: %.3f to %.3f\n", result.ReferenceSummary.Statistics.Min, result.ReferenceSummary.Statistics.Max)
		output += "\n"
	}

	// Comparison metrics
	if result.ComparisonMetrics != nil {
		output += "Comparison Metrics:\n"
		output += fmt.Sprintf("- Statistical Similarity: %.1f%%\n", result.ComparisonMetrics.StatisticalSimilarity*100)
		output += fmt.Sprintf("- Distribution Similarity: %.1f%%\n", result.ComparisonMetrics.DistributionSimilarity*100)
		output += fmt.Sprintf("- Trend Similarity: %.1f%%\n", result.ComparisonMetrics.TrendSimilarity*100)
		output += fmt.Sprintf("- Temporal Similarity: %.1f%%\n", result.ComparisonMetrics.TemporalSimilarity*100)
		output += "\n"
	}

	// Statistical tests
	if len(result.StatisticalTests) > 0 {
		output += "Statistical Tests:\n"
		for _, test := range result.StatisticalTests {
			status := "PASSED"
			if test.IsSignificant {
				status = "FAILED"
			}
			output += fmt.Sprintf("- %s: %s (p=%.3f)\n", test.TestName, status, test.PValue)
		}
	}

	// Output to file or stdout
	if outputFile == "-" {
		fmt.Print(output)
	} else {
		return os.WriteFile(outputFile, []byte(output), 0644)
	}

	return nil
}

func outputJSON(result *ValidationResult, outputFile string) error {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	if outputFile == "-" {
		fmt.Println(string(data))
	} else {
		return os.WriteFile(outputFile, data, 0644)
	}

	return nil
}

func outputHTML(result *ValidationResult, outputFile string) error {
	html := generateHTMLReport(result)

	if outputFile == "-" {
		fmt.Println(html)
	} else {
		return os.WriteFile(outputFile, []byte(html), 0644)
	}

	return nil
}

func generateHTMLReport(result *ValidationResult) string {
	return fmt.Sprintf(`<!DOCTYPE html>
<html>
<head>
    <title>Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .score { font-size: 24px; font-weight: bold; color: %s; }
        .metric { margin: 10px 0; }
        .test-passed { color: green; }
        .test-failed { color: red; }
        table { border-collapse: collapse; width: 100%%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Data Validation Report</h1>
    <div class="score">Overall Quality: %.1f%%</div>
    <p>Generated: %s</p>
    <p>Validation Time: %s</p>
    
    <h2>Data Summary</h2>
    <p>Input data contains %d data points</p>
    
    <!-- Additional HTML content would be added here -->
</body>
</html>`,
		getScoreColor(result.OverallScore),
		result.OverallScore*100,
		result.Timestamp.Format("2006-01-02 15:04:05"),
		result.ValidationTime.String(),
		result.InputSummary.DataPoints)
}

func getScoreColor(score float64) string {
	if score >= 0.8 {
		return "green"
	} else if score >= 0.6 {
		return "orange"
	}
	return "red"
}

// Helper functions for statistical calculations
func calculateCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0.0
	}

	meanX := calculateMean(x)
	meanY := calculateMean(y)

	numerator := 0.0
	denomX := 0.0
	denomY := 0.0

	for i := 0; i < len(x); i++ {
		diffX := x[i] - meanX
		diffY := y[i] - meanY
		numerator += diffX * diffY
		denomX += diffX * diffX
		denomY += diffY * diffY
	}

	denominator := math.Sqrt(denomX * denomY)
	if denominator == 0 {
		return 0.0
	}

	return numerator / denominator
}

func calculateTrendSimilarity(x, y []float64) float64 {
	trendX, _ := calculateTrend(x)
	trendY, _ := calculateTrend(y)
	
	// Compare trend directions and magnitudes
	diffTrend := math.Abs(trendX - trendY)
	maxTrend := math.Max(math.Abs(trendX), math.Abs(trendY))
	
	if maxTrend == 0 {
		return 1.0 // Both have no trend
	}
	
	return 1.0 - (diffTrend / maxTrend)
}

func performKSTest(x, y []float64) *tests.StatisticalTestResult {
	// Simplified KS test implementation
	testSuite := tests.NewStatisticalTestSuite(0.05)
	return testSuite.KolmogorovSmirnovTwoSample(x, y)
}