package validation

import (
	"context"
	"math"
	"time"

	"github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/stat"

	"github.com/inferloop/tsiot/pkg/constants"
	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
)

// TemporalValidator implements temporal pattern validation
type TemporalValidator struct {
	logger           *logrus.Logger
	config           *TemporalValidatorConfig
	supportedMetrics []string
}

// TemporalValidatorConfig contains configuration for temporal validation
type TemporalValidatorConfig struct {
	QualityThreshold     float64 `json:"quality_threshold"`
	MaxLags              int     `json:"max_lags"`
	SeasonalityThreshold float64 `json:"seasonality_threshold"`
	TrendThreshold       float64 `json:"trend_threshold"`
	AutocorrThreshold    float64 `json:"autocorr_threshold"`
}

// NewTemporalValidator creates a new temporal validator
func NewTemporalValidator(config *TemporalValidatorConfig, logger *logrus.Logger) interfaces.TemporalValidator {
	if config == nil {
		config = getDefaultTemporalValidatorConfig()
	}
	
	if logger == nil {
		logger = logrus.New()
	}

	return &TemporalValidator{
		logger: logger,
		config: config,
		supportedMetrics: []string{
			constants.MetricTemporal,
			constants.MetricBasic,
		},
	}
}

// GetType returns the validator type
func (v *TemporalValidator) GetType() string {
	return constants.ValidatorTypeTemporal
}

// GetName returns a human-readable name for the validator
func (v *TemporalValidator) GetName() string {
	return "Temporal Validator"
}

// GetDescription returns a description of the validator
func (v *TemporalValidator) GetDescription() string {
	return "Validates temporal patterns including trends, seasonality, and autocorrelation in synthetic time series data"
}

// GetSupportedMetrics returns the metrics this validator supports
func (v *TemporalValidator) GetSupportedMetrics() []string {
	return v.supportedMetrics
}

// ValidateParameters validates the validation parameters
func (v *TemporalValidator) ValidateParameters(params models.ValidationParameters) error {
	if params.SyntheticData == nil {
		return errors.NewValidationError("MISSING_SYNTHETIC", "Synthetic data is required for temporal validation")
	}

	if len(params.SyntheticData.DataPoints) < 10 {
		return errors.NewValidationError("INSUFFICIENT_DATA", "At least 10 data points required for temporal validation")
	}

	return nil
}

// Validate validates synthetic data temporal patterns
func (v *TemporalValidator) Validate(ctx context.Context, req *models.ValidationRequest) (*models.ValidationResult, error) {
	if req == nil {
		return nil, errors.NewValidationError("INVALID_REQUEST", "Validation request is required")
	}

	if err := v.ValidateParameters(req.Parameters); err != nil {
		return nil, err
	}

	v.logger.WithFields(logrus.Fields{
		"request_id":       req.ID,
		"synthetic_points": len(req.Parameters.SyntheticData.DataPoints),
	}).Info("Starting temporal validation")

	start := time.Now()

	// Validate temporal patterns
	temporalResult, err := v.ValidateTemporalPatterns(ctx, req.Parameters.SyntheticData, req.Parameters.ReferenceData)
	if err != nil {
		return nil, err
	}

	// Calculate overall quality score
	qualityScore := v.calculateTemporalQualityScore(temporalResult)

	// Determine validation status
	status := "passed"
	if qualityScore < v.config.QualityThreshold {
		status = "failed"
	}

	duration := time.Since(start)

	// Convert temporal result to metrics map
	metrics := v.convertTemporalResultToMap(temporalResult)

	result := &models.ValidationResult{
		ID:           req.ID,
		Status:       status,
		QualityScore: qualityScore,
		ValidationDetails: &models.ValidationDetails{
			TotalTests:  3, // seasonality, trend, autocorrelation
			PassedTests: v.countPassedTemporalTests(temporalResult),
			FailedTests: 3 - v.countPassedTemporalTests(temporalResult),
		},
		Metrics:       metrics,
		Duration:      duration,
		ValidatedAt:   time.Now(),
		ValidatorType: v.GetType(),
		Metadata: map[string]interface{}{
			"max_lags":              v.config.MaxLags,
			"seasonality_threshold": v.config.SeasonalityThreshold,
			"trend_threshold":       v.config.TrendThreshold,
			"autocorr_threshold":    v.config.AutocorrThreshold,
		},
	}

	v.logger.WithFields(logrus.Fields{
		"request_id":    req.ID,
		"status":        status,
		"quality_score": qualityScore,
		"duration":      duration,
	}).Info("Temporal validation completed")

	return result, nil
}

// ValidateTemporalPatterns validates temporal patterns in synthetic data
func (v *TemporalValidator) ValidateTemporalPatterns(ctx context.Context, synthetic, reference *models.TimeSeries) (*models.TemporalValidationResult, error) {
	result := &models.TemporalValidationResult{
		ValidatedAt: time.Now(),
	}

	// Check seasonality
	seasonalityResult, err := v.CheckSeasonality(ctx, synthetic, reference)
	if err != nil {
		return nil, err
	}
	result.Seasonality = seasonalityResult

	// Check trends
	trendResult, err := v.CheckTrends(ctx, synthetic, reference)
	if err != nil {
		return nil, err
	}
	result.Trend = trendResult

	// Check autocorrelation
	lags := make([]int, v.config.MaxLags)
	for i := range lags {
		lags[i] = i + 1
	}
	autocorrResult, err := v.CheckAutocorrelation(ctx, synthetic, reference, lags)
	if err != nil {
		return nil, err
	}
	result.Autocorrelation = autocorrResult

	return result, nil
}

// CheckSeasonality checks for proper seasonality preservation
func (v *TemporalValidator) CheckSeasonality(ctx context.Context, synthetic, reference *models.TimeSeries) (*models.SeasonalityResult, error) {
	result := &models.SeasonalityResult{
		Detected: false,
		Strength: 0.0,
		Period:   0,
	}

	synValues := extractValues(synthetic)
	
	// Simple seasonality detection using autocorrelation
	maxPeriod := len(synValues) / 4 // Maximum period to check
	if maxPeriod > 50 {
		maxPeriod = 50
	}

	bestPeriod := 0
	maxSeasonality := 0.0

	for period := 2; period <= maxPeriod; period++ {
		seasonality := v.calculateSeasonalityStrength(synValues, period)
		if seasonality > maxSeasonality {
			maxSeasonality = seasonality
			bestPeriod = period
		}
	}

	if maxSeasonality > v.config.SeasonalityThreshold {
		result.Detected = true
		result.Strength = maxSeasonality
		result.Period = bestPeriod
	}

	// Compare with reference if available
	if reference != nil {
		refValues := extractValues(reference)
		refSeasonality := v.calculateSeasonalityStrength(refValues, bestPeriod)
		result.Similarity = 1.0 - math.Abs(maxSeasonality-refSeasonality)
	} else {
		result.Similarity = 1.0
	}

	result.Quality = v.calculateSeasonalityQuality(result)

	return result, nil
}

// CheckTrends checks for trend preservation
func (v *TemporalValidator) CheckTrends(ctx context.Context, synthetic, reference *models.TimeSeries) (*models.TrendResult, error) {
	result := &models.TrendResult{
		Detected:  false,
		Type:      "none",
		Strength:  0.0,
		Direction: "none",
	}

	synValues := extractValues(synthetic)
	
	// Calculate trend using linear regression
	n := len(synValues)
	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i)
	}

	// Calculate linear trend
	slope, _ := stat.LinearRegression(x, synValues, nil, false)
	
	result.Strength = math.Abs(slope)
	
	if math.Abs(slope) > v.config.TrendThreshold {
		result.Detected = true
		result.Type = constants.TrendLinear
		if slope > 0 {
			result.Direction = "increasing"
		} else {
			result.Direction = "decreasing"
		}
	}

	// Compare with reference if available
	if reference != nil {
		refValues := extractValues(reference)
		refX := make([]float64, len(refValues))
		for i := range refX {
			refX[i] = float64(i)
		}
		refSlope, _ := stat.LinearRegression(refX, refValues, nil, false)
		
		// Calculate similarity based on slope comparison
		if math.Abs(refSlope) < v.config.TrendThreshold && math.Abs(slope) < v.config.TrendThreshold {
			result.Similarity = 1.0 // Both have no trend
		} else {
			slopeDiff := math.Abs(slope - refSlope)
			maxSlope := math.Max(math.Abs(slope), math.Abs(refSlope))
			if maxSlope == 0 {
				result.Similarity = 1.0
			} else {
				result.Similarity = 1.0 - slopeDiff/maxSlope
			}
		}
	} else {
		result.Similarity = 1.0
	}

	result.Quality = v.calculateTrendQuality(result)

	return result, nil
}

// CheckAutocorrelation checks autocorrelation patterns
func (v *TemporalValidator) CheckAutocorrelation(ctx context.Context, synthetic, reference *models.TimeSeries, lags []int) (*models.AutocorrelationResult, error) {
	result := &models.AutocorrelationResult{
		Lags:           lags,
		Correlations:   make([]float64, len(lags)),
		Significant:    make([]bool, len(lags)),
		MaxCorrelation: 0.0,
		OverallQuality: 0.0,
	}

	synValues := extractValues(synthetic)
	
	// Calculate autocorrelations
	for i, lag := range lags {
		if lag >= len(synValues) {
			result.Correlations[i] = 0.0
			result.Significant[i] = false
			continue
		}

		autocorr := v.calculateAutocorrelation(synValues, lag)
		result.Correlations[i] = autocorr
		result.Significant[i] = math.Abs(autocorr) > v.config.AutocorrThreshold

		if math.Abs(autocorr) > math.Abs(result.MaxCorrelation) {
			result.MaxCorrelation = autocorr
		}
	}

	// Compare with reference if available
	if reference != nil {
		refValues := extractValues(reference)
		refCorrelations := make([]float64, len(lags))
		
		for i, lag := range lags {
			if lag >= len(refValues) {
				refCorrelations[i] = 0.0
				continue
			}
			refCorrelations[i] = v.calculateAutocorrelation(refValues, lag)
		}

		// Calculate similarity
		totalDiff := 0.0
		for i := range result.Correlations {
			totalDiff += math.Abs(result.Correlations[i] - refCorrelations[i])
		}
		result.Similarity = 1.0 - totalDiff/float64(len(lags))
	} else {
		result.Similarity = 1.0
	}

	result.OverallQuality = v.calculateAutocorrelationQuality(result)

	return result, nil
}

// GetDefaultParameters returns default parameters for this validator
func (v *TemporalValidator) GetDefaultParameters() models.ValidationParameters {
	return models.ValidationParameters{
		Metrics: []string{constants.MetricTemporal},
		Timeout: "30s",
	}
}

// Close cleans up resources
func (v *TemporalValidator) Close() error {
	v.logger.Info("Closing temporal validator")
	return nil
}

// Private methods

func (v *TemporalValidator) calculateSeasonalityStrength(values []float64, period int) float64 {
	if period >= len(values) || period < 2 {
		return 0.0
	}

	// Calculate autocorrelation at seasonal lag
	return math.Abs(v.calculateAutocorrelation(values, period))
}

func (v *TemporalValidator) calculateAutocorrelation(values []float64, lag int) float64 {
	if lag >= len(values) || lag <= 0 {
		return 0.0
	}

	n := len(values) - lag
	if n <= 1 {
		return 0.0
	}

	x1 := values[:n]
	x2 := values[lag:lag+n]

	return stat.Correlation(x1, x2, nil)
}

func (v *TemporalValidator) calculateSeasonalityQuality(result *models.SeasonalityResult) float64 {
	if !result.Detected {
		return 0.5 // Neutral quality for no detected seasonality
	}
	
	// Quality based on strength and similarity
	strengthScore := math.Min(result.Strength*2, 1.0) // Scale strength
	return (strengthScore + result.Similarity) / 2.0
}

func (v *TemporalValidator) calculateTrendQuality(result *models.TrendResult) float64 {
	if !result.Detected {
		return 0.5 // Neutral quality for no detected trend
	}
	
	// Quality based on strength and similarity
	strengthScore := math.Min(result.Strength*10, 1.0) // Scale strength
	return (strengthScore + result.Similarity) / 2.0
}

func (v *TemporalValidator) calculateAutocorrelationQuality(result *models.AutocorrelationResult) float64 {
	if len(result.Correlations) == 0 {
		return 0.0
	}

	// Quality based on similarity and significance
	significantCount := 0
	for _, sig := range result.Significant {
		if sig {
			significantCount++
		}
	}

	significanceScore := float64(significantCount) / float64(len(result.Significant))
	return (significanceScore + result.Similarity) / 2.0
}

func (v *TemporalValidator) calculateTemporalQualityScore(result *models.TemporalValidationResult) float64 {
	weights := map[string]float64{
		"seasonality":     0.4,
		"trend":          0.3,
		"autocorrelation": 0.3,
	}

	totalScore := 0.0
	totalWeight := 0.0

	if result.Seasonality != nil {
		totalScore += result.Seasonality.Quality * weights["seasonality"]
		totalWeight += weights["seasonality"]
	}

	if result.Trend != nil {
		totalScore += result.Trend.Quality * weights["trend"]
		totalWeight += weights["trend"]
	}

	if result.Autocorrelation != nil {
		totalScore += result.Autocorrelation.OverallQuality * weights["autocorrelation"]
		totalWeight += weights["autocorrelation"]
	}

	if totalWeight == 0 {
		return 0.0
	}

	return totalScore / totalWeight
}

func (v *TemporalValidator) countPassedTemporalTests(result *models.TemporalValidationResult) int {
	passed := 0
	threshold := v.config.QualityThreshold

	if result.Seasonality != nil && result.Seasonality.Quality >= threshold {
		passed++
	}
	if result.Trend != nil && result.Trend.Quality >= threshold {
		passed++
	}
	if result.Autocorrelation != nil && result.Autocorrelation.OverallQuality >= threshold {
		passed++
	}

	return passed
}

func (v *TemporalValidator) convertTemporalResultToMap(result *models.TemporalValidationResult) map[string]interface{} {
	metrics := make(map[string]interface{})

	if result.Seasonality != nil {
		metrics["seasonality_detected"] = result.Seasonality.Detected
		metrics["seasonality_strength"] = result.Seasonality.Strength
		metrics["seasonality_period"] = result.Seasonality.Period
		metrics["seasonality_quality"] = result.Seasonality.Quality
		metrics["seasonality_similarity"] = result.Seasonality.Similarity
	}

	if result.Trend != nil {
		metrics["trend_detected"] = result.Trend.Detected
		metrics["trend_type"] = result.Trend.Type
		metrics["trend_strength"] = result.Trend.Strength
		metrics["trend_direction"] = result.Trend.Direction
		metrics["trend_quality"] = result.Trend.Quality
		metrics["trend_similarity"] = result.Trend.Similarity
	}

	if result.Autocorrelation != nil {
		metrics["max_autocorrelation"] = result.Autocorrelation.MaxCorrelation
		metrics["autocorrelation_quality"] = result.Autocorrelation.OverallQuality
		metrics["autocorrelation_similarity"] = result.Autocorrelation.Similarity
		metrics["significant_lags"] = len(result.Autocorrelation.Significant)
	}

	return metrics
}

func getDefaultTemporalValidatorConfig() *TemporalValidatorConfig {
	return &TemporalValidatorConfig{
		QualityThreshold:     0.7,
		MaxLags:              20,
		SeasonalityThreshold: 0.3,
		TrendThreshold:       0.01,
		AutocorrThreshold:    0.1,
	}
}