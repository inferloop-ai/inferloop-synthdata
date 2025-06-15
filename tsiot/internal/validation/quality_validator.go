package validation

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/stat"

	"github.com/inferloop/tsiot/pkg/constants"
	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
)

// QualityValidator implements data quality assessment
type QualityValidator struct {
	logger           *logrus.Logger
	config           *QualityValidatorConfig
	qualityThreshold float64
	supportedMetrics []string
}

// QualityValidatorConfig contains configuration for quality validation
type QualityValidatorConfig struct {
	QualityThreshold    float64                `json:"quality_threshold"`
	WeightFactors       map[string]float64     `json:"weight_factors"`
	EnabledMetrics      []string               `json:"enabled_metrics"`
	OutlierThreshold    float64                `json:"outlier_threshold"`
	MissingDataThreshold float64               `json:"missing_data_threshold"`
	QualityDimensions   []string               `json:"quality_dimensions"`
}

// NewQualityValidator creates a new quality validator
func NewQualityValidator(config *QualityValidatorConfig, logger *logrus.Logger) interfaces.QualityValidator {
	if config == nil {
		config = getDefaultQualityValidatorConfig()
	}
	
	if logger == nil {
		logger = logrus.New()
	}

	return &QualityValidator{
		logger:           logger,
		config:           config,
		qualityThreshold: config.QualityThreshold,
		supportedMetrics: []string{
			constants.MetricQuality,
			constants.MetricBasic,
		},
	}
}

// GetType returns the validator type
func (v *QualityValidator) GetType() string {
	return constants.ValidatorTypeQuality
}

// GetName returns a human-readable name for the validator
func (v *QualityValidator) GetName() string {
	return "Quality Validator"
}

// GetDescription returns a description of the validator
func (v *QualityValidator) GetDescription() string {
	return "Validates synthetic data quality across multiple dimensions including completeness, accuracy, and consistency"
}

// GetSupportedMetrics returns the metrics this validator supports
func (v *QualityValidator) GetSupportedMetrics() []string {
	return v.supportedMetrics
}

// ValidateParameters validates the validation parameters
func (v *QualityValidator) ValidateParameters(params models.ValidationParameters) error {
	if params.SyntheticData == nil {
		return errors.NewValidationError("MISSING_SYNTHETIC", "Synthetic data is required for quality validation")
	}

	if len(params.SyntheticData.DataPoints) == 0 {
		return errors.NewValidationError("EMPTY_SYNTHETIC", "Synthetic data cannot be empty")
	}

	return nil
}

// Validate validates synthetic data quality
func (v *QualityValidator) Validate(ctx context.Context, req *models.ValidationRequest) (*models.ValidationResult, error) {
	if req == nil {
		return nil, errors.NewValidationError("INVALID_REQUEST", "Validation request is required")
	}

	if err := v.ValidateParameters(req.Parameters); err != nil {
		return nil, err
	}

	v.logger.WithFields(logrus.Fields{
		"request_id":       req.ID,
		"synthetic_points": len(req.Parameters.SyntheticData.DataPoints),
	}).Info("Starting quality validation")

	start := time.Now()

	// Calculate quality metrics
	qualityMetrics, err := v.GetQualityMetrics(ctx, req.Parameters.SyntheticData, req.Parameters.ReferenceData)
	if err != nil {
		return nil, err
	}

	// Calculate overall quality score
	qualityScore, err := v.CalculateQualityScore(ctx, req.Parameters.SyntheticData, req.Parameters.ReferenceData)
	if err != nil {
		return nil, err
	}

	// Determine validation status
	status := "passed"
	if qualityScore < v.qualityThreshold {
		status = "failed"
	}

	duration := time.Since(start)

	result := &models.ValidationResult{
		ID:           req.ID,
		Status:       status,
		QualityScore: qualityScore,
		ValidationDetails: &models.ValidationDetails{
			TotalTests:  len(v.config.QualityDimensions),
			PassedTests: v.countPassedDimensions(qualityMetrics),
			FailedTests: len(v.config.QualityDimensions) - v.countPassedDimensions(qualityMetrics),
		},
		Metrics:       v.convertQualityMetricsToMap(qualityMetrics),
		Duration:      duration,
		ValidatedAt:   time.Now(),
		ValidatorType: v.GetType(),
		Metadata: map[string]interface{}{
			"quality_threshold":  v.qualityThreshold,
			"dimensions_checked": v.config.QualityDimensions,
			"weight_factors":     v.config.WeightFactors,
		},
	}

	v.logger.WithFields(logrus.Fields{
		"request_id":    req.ID,
		"status":        status,
		"quality_score": qualityScore,
		"duration":      duration,
	}).Info("Quality validation completed")

	return result, nil
}

// CalculateQualityScore calculates an overall quality score (0.0 to 1.0)
func (v *QualityValidator) CalculateQualityScore(ctx context.Context, synthetic, reference *models.TimeSeries) (float64, error) {
	qualityMetrics, err := v.GetQualityMetrics(ctx, synthetic, reference)
	if err != nil {
		return 0, err
	}

	totalScore := 0.0
	totalWeight := 0.0

	// Weighted average of quality dimensions
	for dimension, score := range map[string]float64{
		"completeness": qualityMetrics.Completeness,
		"accuracy":     qualityMetrics.Accuracy,
		"consistency":  qualityMetrics.Consistency,
		"validity":     qualityMetrics.Validity,
		"uniqueness":   qualityMetrics.Uniqueness,
	} {
		weight := v.config.WeightFactors[dimension]
		if weight == 0 {
			weight = 0.2 // Default equal weight
		}
		totalScore += score * weight
		totalWeight += weight
	}

	if totalWeight == 0 {
		return 0, nil
	}

	return totalScore / totalWeight, nil
}

// GetQualityMetrics returns detailed quality metrics
func (v *QualityValidator) GetQualityMetrics(ctx context.Context, synthetic, reference *models.TimeSeries) (*models.QualityMetrics, error) {
	metrics := &models.QualityMetrics{
		CalculatedAt: time.Now(),
	}

	// Extract values
	synValues := extractValues(synthetic)

	// Completeness: Check for missing/invalid data
	metrics.Completeness = v.calculateCompleteness(synValues)

	// Accuracy: Compare against reference if available
	if reference != nil {
		refValues := extractValues(reference)
		metrics.Accuracy = v.calculateAccuracy(synValues, refValues)
	} else {
		metrics.Accuracy = 1.0 // Perfect accuracy if no reference
	}

	// Consistency: Check for outliers and anomalies
	metrics.Consistency = v.calculateConsistency(synValues)

	// Validity: Check if values are within expected ranges
	metrics.Validity = v.calculateValidity(synValues)

	// Uniqueness: Check for duplicate values
	metrics.Uniqueness = v.calculateUniqueness(synValues)

	// Additional metrics
	metrics.Outliers = v.detectOutliers(synValues)
	metrics.MissingData = v.calculateMissingDataRate(synthetic)
	metrics.DataTypes = v.validateDataTypes(synthetic)

	// Overall score
	score, _ := v.CalculateQualityScore(context.Background(), synthetic, reference)
	metrics.OverallScore = score

	return metrics, nil
}

// SetQualityThreshold sets the minimum quality threshold
func (v *QualityValidator) SetQualityThreshold(threshold float64) error {
	if threshold < 0 || threshold > 1 {
		return errors.NewValidationError("INVALID_THRESHOLD", "Quality threshold must be between 0 and 1")
	}
	v.qualityThreshold = threshold
	v.config.QualityThreshold = threshold
	return nil
}

// GetQualityThreshold returns the current quality threshold
func (v *QualityValidator) GetQualityThreshold() float64 {
	return v.qualityThreshold
}

// GetDefaultParameters returns default parameters for this validator
func (v *QualityValidator) GetDefaultParameters() models.ValidationParameters {
	return models.ValidationParameters{
		Metrics: []string{constants.MetricQuality},
		Timeout: "30s",
	}
}

// Close cleans up resources
func (v *QualityValidator) Close() error {
	v.logger.Info("Closing quality validator")
	return nil
}

// Private methods

func (v *QualityValidator) calculateCompleteness(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	validCount := 0
	for _, val := range values {
		if !math.IsNaN(val) && !math.IsInf(val, 0) {
			validCount++
		}
	}

	return float64(validCount) / float64(len(values))
}

func (v *QualityValidator) calculateAccuracy(synthetic, reference []float64) float64 {
	if len(synthetic) == 0 || len(reference) == 0 {
		return 0.0
	}

	// Calculate correlation as a measure of accuracy
	correlation := stat.Correlation(synthetic, reference, nil)
	
	// Convert correlation to accuracy score (absolute value)
	accuracy := math.Abs(correlation)
	
	// Also consider mean absolute percentage error
	mape := v.calculateMAPE(synthetic, reference)
	mapeScore := math.Max(0, 1.0-mape)
	
	// Weighted average
	return (accuracy*0.6 + mapeScore*0.4)
}

func (v *QualityValidator) calculateMAPE(synthetic, reference []float64) float64 {
	if len(synthetic) != len(reference) {
		// If lengths differ, sample to match
		minLen := math.Min(float64(len(synthetic)), float64(len(reference)))
		synthetic = synthetic[:int(minLen)]
		reference = reference[:int(minLen)]
	}

	var totalError float64
	validPoints := 0

	for i := 0; i < len(synthetic); i++ {
		if reference[i] != 0 {
			error := math.Abs((reference[i] - synthetic[i]) / reference[i])
			totalError += error
			validPoints++
		}
	}

	if validPoints == 0 {
		return 0.0
	}

	return totalError / float64(validPoints)
}

func (v *QualityValidator) calculateConsistency(values []float64) float64 {
	if len(values) < 2 {
		return 1.0
	}

	// Calculate coefficient of variation as a measure of consistency
	mean := stat.Mean(values, nil)
	stdDev := stat.StdDev(values, nil)
	
	if mean == 0 {
		return 1.0
	}

	cv := stdDev / math.Abs(mean)
	
	// Convert CV to consistency score (lower CV = higher consistency)
	consistency := 1.0 / (1.0 + cv)
	
	return consistency
}

func (v *QualityValidator) calculateValidity(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	validCount := 0
	for _, val := range values {
		// Check if value is valid (not NaN, not infinite)
		if !math.IsNaN(val) && !math.IsInf(val, 0) {
			validCount++
		}
	}

	return float64(validCount) / float64(len(values))
}

func (v *QualityValidator) calculateUniqueness(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	uniqueValues := make(map[float64]bool)
	for _, val := range values {
		uniqueValues[val] = true
	}

	return float64(len(uniqueValues)) / float64(len(values))
}

func (v *QualityValidator) detectOutliers(values []float64) []float64 {
	if len(values) < 4 {
		return []float64{}
	}

	// Using IQR method
	mean := stat.Mean(values, nil)
	stdDev := stat.StdDev(values, nil)
	
	threshold := v.config.OutlierThreshold * stdDev
	
	var outliers []float64
	for _, val := range values {
		if math.Abs(val-mean) > threshold {
			outliers = append(outliers, val)
		}
	}

	return outliers
}

func (v *QualityValidator) calculateMissingDataRate(ts *models.TimeSeries) float64 {
	if len(ts.DataPoints) == 0 {
		return 1.0 // 100% missing
	}

	missingCount := 0
	for _, dp := range ts.DataPoints {
		if math.IsNaN(dp.Value) || dp.Quality < 0.5 {
			missingCount++
		}
	}

	return float64(missingCount) / float64(len(ts.DataPoints))
}

func (v *QualityValidator) validateDataTypes(ts *models.TimeSeries) map[string]interface{} {
	dataTypes := make(map[string]interface{})
	
	if len(ts.DataPoints) == 0 {
		return dataTypes
	}

	// Check data type consistency
	allNumeric := true
	allFinite := true
	
	for _, dp := range ts.DataPoints {
		if math.IsNaN(dp.Value) || math.IsInf(dp.Value, 0) {
			allFinite = false
		}
	}

	dataTypes["all_numeric"] = allNumeric
	dataTypes["all_finite"] = allFinite
	dataTypes["data_points"] = len(ts.DataPoints)

	return dataTypes
}

func (v *QualityValidator) countPassedDimensions(metrics *models.QualityMetrics) int {
	passed := 0
	threshold := v.qualityThreshold

	if metrics.Completeness >= threshold {
		passed++
	}
	if metrics.Accuracy >= threshold {
		passed++
	}
	if metrics.Consistency >= threshold {
		passed++
	}
	if metrics.Validity >= threshold {
		passed++
	}
	if metrics.Uniqueness >= threshold {
		passed++
	}

	return passed
}

func (v *QualityValidator) convertQualityMetricsToMap(metrics *models.QualityMetrics) map[string]interface{} {
	return map[string]interface{}{
		"completeness":     metrics.Completeness,
		"accuracy":         metrics.Accuracy,
		"consistency":      metrics.Consistency,
		"validity":         metrics.Validity,
		"uniqueness":       metrics.Uniqueness,
		"overall_score":    metrics.OverallScore,
		"outliers_count":   len(metrics.Outliers),
		"missing_data_rate": metrics.MissingData,
		"data_types":       metrics.DataTypes,
	}
}

func getDefaultQualityValidatorConfig() *QualityValidatorConfig {
	return &QualityValidatorConfig{
		QualityThreshold: constants.DefaultQualityThreshold,
		WeightFactors: map[string]float64{
			"completeness": 0.25,
			"accuracy":     0.25,
			"consistency":  0.20,
			"validity":     0.20,
			"uniqueness":   0.10,
		},
		EnabledMetrics: []string{
			constants.MetricQuality,
			constants.MetricBasic,
		},
		OutlierThreshold:     3.0, // 3 standard deviations
		MissingDataThreshold: 0.05, // 5% missing data threshold
		QualityDimensions: []string{
			"completeness",
			"accuracy",
			"consistency",
			"validity",
			"uniqueness",
		},
	}
}