package validation

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/stat"

	"github.com/inferloop/tsiot/internal/validation/tests"
	"github.com/inferloop/tsiot/pkg/constants"
	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
)

// StatisticalValidator implements statistical validation methods
type StatisticalValidator struct {
	logger           *logrus.Logger
	config           *StatisticalValidatorConfig
	significanceLevel float64
	supportedTests   []string
	supportedMetrics []string
}

// StatisticalValidatorConfig contains configuration for statistical validation
type StatisticalValidatorConfig struct {
	SignificanceLevel    float64  `json:"significance_level"`
	EnabledTests         []string `json:"enabled_tests"`
	QualityThreshold     float64  `json:"quality_threshold"`
	AutoSelectTests      bool     `json:"auto_select_tests"`
	MaxSampleSize        int      `json:"max_sample_size"`
	MinSampleSize        int      `json:"min_sample_size"`
	BootstrapIterations  int      `json:"bootstrap_iterations"`
	ConfidenceInterval   float64  `json:"confidence_interval"`
}

// NewStatisticalValidator creates a new statistical validator
func NewStatisticalValidator(config *StatisticalValidatorConfig, logger *logrus.Logger) interfaces.StatisticalValidator {
	if config == nil {
		config = getDefaultStatisticalValidatorConfig()
	}
	
	if logger == nil {
		logger = logrus.New()
	}

	return &StatisticalValidator{
		logger:            logger,
		config:            config,
		significanceLevel: config.SignificanceLevel,
		supportedTests: []string{
			constants.TestKolmogorovSmirnov,
			constants.TestAndersonDarling,
			constants.TestLjungBox,
			constants.TestShapiro,
			constants.TestJarqueBera,
		},
		supportedMetrics: []string{
			constants.MetricBasic,
			constants.MetricStatistical,
			constants.MetricDistribution,
		},
	}
}

// GetType returns the validator type
func (v *StatisticalValidator) GetType() string {
	return constants.ValidatorTypeStatistical
}

// GetName returns a human-readable name for the validator
func (v *StatisticalValidator) GetName() string {
	return "Statistical Validator"
}

// GetDescription returns a description of the validator
func (v *StatisticalValidator) GetDescription() string {
	return "Validates synthetic data using statistical tests and distribution comparisons"
}

// GetSupportedMetrics returns the metrics this validator supports
func (v *StatisticalValidator) GetSupportedMetrics() []string {
	return v.supportedMetrics
}

// ValidateParameters validates the validation parameters
func (v *StatisticalValidator) ValidateParameters(params models.ValidationParameters) error {
	if params.ReferenceData == nil {
		return errors.NewValidationError("MISSING_REFERENCE", "Reference data is required for statistical validation")
	}

	if params.SyntheticData == nil {
		return errors.NewValidationError("MISSING_SYNTHETIC", "Synthetic data is required for validation")
	}

	if len(params.ReferenceData.DataPoints) < v.config.MinSampleSize {
		return errors.NewValidationError("INSUFFICIENT_REFERENCE_DATA", 
			fmt.Sprintf("Reference data must have at least %d data points", v.config.MinSampleSize))
	}

	if len(params.SyntheticData.DataPoints) < v.config.MinSampleSize {
		return errors.NewValidationError("INSUFFICIENT_SYNTHETIC_DATA", 
			fmt.Sprintf("Synthetic data must have at least %d data points", v.config.MinSampleSize))
	}

	return nil
}

// Validate validates synthetic data against reference data
func (v *StatisticalValidator) Validate(ctx context.Context, req *models.ValidationRequest) (*models.ValidationResult, error) {
	if req == nil {
		return nil, errors.NewValidationError("INVALID_REQUEST", "Validation request is required")
	}

	if err := v.ValidateParameters(req.Parameters); err != nil {
		return nil, err
	}

	v.logger.WithFields(logrus.Fields{
		"request_id":        req.ID,
		"reference_points":  len(req.Parameters.ReferenceData.DataPoints),
		"synthetic_points":  len(req.Parameters.SyntheticData.DataPoints),
		"significance_level": v.significanceLevel,
	}).Info("Starting statistical validation")

	start := time.Now()

	// Extract values from time series
	refValues := extractValues(req.Parameters.ReferenceData)
	synValues := extractValues(req.Parameters.SyntheticData)

	// Sample data if too large
	if len(refValues) > v.config.MaxSampleSize {
		refValues = sampleData(refValues, v.config.MaxSampleSize)
	}
	if len(synValues) > v.config.MaxSampleSize {
		synValues = sampleData(synValues, v.config.MaxSampleSize)
	}

	// Run statistical tests
	testResults, err := v.RunStatisticalTests(ctx, req.Parameters.SyntheticData, req.Parameters.ReferenceData, v.config.EnabledTests)
	if err != nil {
		return nil, err
	}

	// Calculate basic metrics
	metrics := v.calculateBasicMetrics(synValues, refValues)

	// Calculate overall quality score
	qualityScore := v.calculateQualityScore(testResults, metrics)

	// Determine validation status
	status := "passed"
	if qualityScore < v.config.QualityThreshold {
		status = "failed"
	}

	duration := time.Since(start)

	result := &models.ValidationResult{
		ID:              req.ID,
		Status:          status,
		QualityScore:    qualityScore,
		ValidationDetails: &models.ValidationDetails{
			TotalTests:    len(testResults.Results),
			PassedTests:   v.countPassedTests(testResults),
			FailedTests:   len(testResults.Results) - v.countPassedTests(testResults),
		},
		Metrics: metrics,
		StatisticalTests: testResults,
		Duration:    duration,
		ValidatedAt: time.Now(),
		ValidatorType: v.GetType(),
		Metadata: map[string]interface{}{
			"significance_level":  v.significanceLevel,
			"sample_sizes": map[string]int{
				"reference": len(refValues),
				"synthetic": len(synValues),
			},
			"tests_run": len(testResults.Results),
		},
	}

	v.logger.WithFields(logrus.Fields{
		"request_id":    req.ID,
		"status":        status,
		"quality_score": qualityScore,
		"duration":      duration,
		"tests_passed":  result.ValidationDetails.PassedTests,
		"tests_failed":  result.ValidationDetails.FailedTests,
	}).Info("Statistical validation completed")

	return result, nil
}

// RunStatisticalTests runs statistical tests comparing datasets
func (v *StatisticalValidator) RunStatisticalTests(ctx context.Context, synthetic, reference *models.TimeSeries, testsToRun []string) (*models.StatisticalTestResults, error) {
	if len(testsToRun) == 0 {
		testsToRun = v.config.EnabledTests
	}

	results := &models.StatisticalTestResults{
		Results:    make(map[string]*models.StatisticalTestResult),
		Summary:    make(map[string]interface{}),
		RunAt:      time.Now(),
	}

	refValues := extractValues(reference)
	synValues := extractValues(synthetic)

	for _, testName := range testsToRun {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		testResult, err := v.runSingleTest(testName, synValues, refValues)
		if err != nil {
			v.logger.WithFields(logrus.Fields{
				"test": testName,
				"error": err.Error(),
			}).Warn("Statistical test failed")
			continue
		}

		results.Results[testName] = testResult
	}

	// Calculate summary statistics
	results.Summary = v.calculateTestSummary(results.Results)

	return results, nil
}

// GetSupportedTests returns supported statistical tests
func (v *StatisticalValidator) GetSupportedTests() []string {
	return v.supportedTests
}

// SetSignificanceLevel sets the significance level for tests
func (v *StatisticalValidator) SetSignificanceLevel(alpha float64) error {
	if alpha <= 0 || alpha >= 1 {
		return errors.NewValidationError("INVALID_ALPHA", "Significance level must be between 0 and 1")
	}
	v.significanceLevel = alpha
	v.config.SignificanceLevel = alpha
	return nil
}

// GetSignificanceLevel returns the current significance level
func (v *StatisticalValidator) GetSignificanceLevel() float64 {
	return v.significanceLevel
}

// GetDefaultParameters returns default parameters for this validator
func (v *StatisticalValidator) GetDefaultParameters() models.ValidationParameters {
	return models.ValidationParameters{
		Metrics: []string{constants.MetricStatistical, constants.MetricDistribution},
		Tests:   v.supportedTests,
		Timeout: "30s",
	}
}

// Close cleans up resources
func (v *StatisticalValidator) Close() error {
	v.logger.Info("Closing statistical validator")
	return nil
}

// Private methods

func (v *StatisticalValidator) runSingleTest(testName string, synthetic, reference []float64) (*models.StatisticalTestResult, error) {
	switch testName {
	case constants.TestKolmogorovSmirnov:
		return tests.KolmogorovSmirnovTest(synthetic, reference, v.significanceLevel)
	case constants.TestAndersonDarling:
		return tests.AndersonDarlingTest(synthetic, reference, v.significanceLevel)
	case constants.TestLjungBox:
		return tests.LjungBoxTest(synthetic, v.significanceLevel)
	case constants.TestShapiro:
		return tests.ShapiroWilkTest(synthetic, v.significanceLevel)
	case constants.TestJarqueBera:
		return tests.JarqueBeraTest(synthetic, v.significanceLevel)
	default:
		return nil, errors.NewValidationError("UNSUPPORTED_TEST", fmt.Sprintf("Test '%s' is not supported", testName))
	}
}

func (v *StatisticalValidator) calculateBasicMetrics(synthetic, reference []float64) map[string]interface{} {
	metrics := make(map[string]interface{})

	// Basic statistics
	synMean := stat.Mean(synthetic, nil)
	refMean := stat.Mean(reference, nil)
	synStd := stat.StdDev(synthetic, nil)
	refStd := stat.StdDev(reference, nil)

	metrics["mean_difference"] = math.Abs(synMean - refMean)
	metrics["std_difference"] = math.Abs(synStd - refStd)
	metrics["mean_ratio"] = synMean / refMean
	metrics["std_ratio"] = synStd / refStd

	// Correlation
	correlation := stat.Correlation(synthetic, reference, nil)
	metrics["correlation"] = correlation

	// Distribution overlap (approximate)
	metrics["distribution_overlap"] = v.calculateDistributionOverlap(synthetic, reference)

	// Range comparison
	synMin, synMax := minMax(synthetic)
	refMin, refMax := minMax(reference)
	metrics["range_coverage"] = v.calculateRangeCoverage(synMin, synMax, refMin, refMax)

	return metrics
}

func (v *StatisticalValidator) calculateQualityScore(testResults *models.StatisticalTestResults, metrics map[string]interface{}) float64 {
	score := 0.0
	components := 0

	// Score from statistical tests (40% weight)
	if len(testResults.Results) > 0 {
		testScore := float64(v.countPassedTests(testResults)) / float64(len(testResults.Results))
		score += testScore * 0.4
		components++
	}

	// Score from correlation (30% weight)
	if corr, ok := metrics["correlation"].(float64); ok {
		corrScore := math.Abs(corr) // Higher correlation is better
		score += corrScore * 0.3
		components++
	}

	// Score from distribution overlap (20% weight)
	if overlap, ok := metrics["distribution_overlap"].(float64); ok {
		score += overlap * 0.2
		components++
	}

	// Score from range coverage (10% weight)
	if coverage, ok := metrics["range_coverage"].(float64); ok {
		score += coverage * 0.1
		components++
	}

	if components == 0 {
		return 0.0
	}

	return score / float64(components)
}

func (v *StatisticalValidator) countPassedTests(testResults *models.StatisticalTestResults) int {
	passed := 0
	for _, result := range testResults.Results {
		if result.Passed {
			passed++
		}
	}
	return passed
}

func (v *StatisticalValidator) calculateTestSummary(results map[string]*models.StatisticalTestResult) map[string]interface{} {
	summary := make(map[string]interface{})
	
	totalTests := len(results)
	passedTests := 0
	var avgPValue float64

	for _, result := range results {
		if result.Passed {
			passedTests++
		}
		avgPValue += result.PValue
	}

	if totalTests > 0 {
		avgPValue /= float64(totalTests)
	}

	summary["total_tests"] = totalTests
	summary["passed_tests"] = passedTests
	summary["failed_tests"] = totalTests - passedTests
	summary["success_rate"] = float64(passedTests) / float64(totalTests)
	summary["average_p_value"] = avgPValue

	return summary
}

func (v *StatisticalValidator) calculateDistributionOverlap(synthetic, reference []float64) float64 {
	// Simple histogram-based overlap calculation
	bins := 50
	synMin, synMax := minMax(synthetic)
	refMin, refMax := minMax(reference)
	
	globalMin := math.Min(synMin, refMin)
	globalMax := math.Max(synMax, refMax)
	binWidth := (globalMax - globalMin) / float64(bins)
	
	if binWidth == 0 {
		return 1.0 // Perfect overlap if both distributions are constant
	}

	synHist := make([]int, bins)
	refHist := make([]int, bins)

	// Create histograms
	for _, val := range synthetic {
		bin := int((val - globalMin) / binWidth)
		if bin >= bins {
			bin = bins - 1
		}
		if bin < 0 {
			bin = 0
		}
		synHist[bin]++
	}

	for _, val := range reference {
		bin := int((val - globalMin) / binWidth)
		if bin >= bins {
			bin = bins - 1
		}
		if bin < 0 {
			bin = 0
		}
		refHist[bin]++
	}

	// Calculate overlap
	overlap := 0.0
	for i := 0; i < bins; i++ {
		synFreq := float64(synHist[i]) / float64(len(synthetic))
		refFreq := float64(refHist[i]) / float64(len(reference))
		overlap += math.Min(synFreq, refFreq)
	}

	return overlap
}

func (v *StatisticalValidator) calculateRangeCoverage(synMin, synMax, refMin, refMax float64) float64 {
	// Calculate how well synthetic data covers the reference data range
	if refMax == refMin {
		return 1.0 // Perfect coverage for constant reference
	}

	refRange := refMax - refMin
	synRange := synMax - synMin

	// Calculate overlap
	overlapMin := math.Max(synMin, refMin)
	overlapMax := math.Min(synMax, refMax)
	
	if overlapMax <= overlapMin {
		return 0.0 // No overlap
	}

	overlapRange := overlapMax - overlapMin
	return overlapRange / refRange
}

// Helper functions

func extractValues(ts *models.TimeSeries) []float64 {
	values := make([]float64, len(ts.DataPoints))
	for i, dp := range ts.DataPoints {
		values[i] = dp.Value
	}
	return values
}

func sampleData(data []float64, maxSize int) []float64 {
	if len(data) <= maxSize {
		return data
	}
	
	step := len(data) / maxSize
	sampled := make([]float64, 0, maxSize)
	
	for i := 0; i < len(data); i += step {
		sampled = append(sampled, data[i])
		if len(sampled) >= maxSize {
			break
		}
	}
	
	return sampled
}

func minMax(data []float64) (float64, float64) {
	if len(data) == 0 {
		return 0, 0
	}
	
	min := data[0]
	max := data[0]
	
	for _, val := range data[1:] {
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}
	
	return min, max
}

func getDefaultStatisticalValidatorConfig() *StatisticalValidatorConfig {
	return &StatisticalValidatorConfig{
		SignificanceLevel: 0.05, // Default 5% significance level
		EnabledTests: []string{
			constants.TestKolmogorovSmirnov,
			constants.TestAndersonDarling,
			constants.TestLjungBox,
		},
		QualityThreshold:     0.7,
		AutoSelectTests:      true,
		MaxSampleSize:        10000,
		MinSampleSize:        30,
		BootstrapIterations:  1000,
		ConfidenceInterval:   0.95,
	}
}