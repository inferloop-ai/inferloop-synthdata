package privacy

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/models"
)

// DataMaskingEngine provides data masking and anonymization for time series
type DataMaskingEngine struct {
	logger     *logrus.Logger
	randSource *rand.Rand
	config     *MaskingConfig
}

// MaskingConfig contains configuration for data masking
type MaskingConfig struct {
	// Masking strategies
	Strategy           string                 `json:"strategy"`            // "suppress", "generalize", "perturb", "synthetic"
	SuppressionRate    float64                `json:"suppression_rate"`    // Fraction of values to suppress
	GeneralizationBins int                    `json:"generalization_bins"` // Number of bins for generalization
	PerturbationRange  float64                `json:"perturbation_range"`  // Range for random perturbation
	
	// Time-based masking
	TimeGranularity    string                 `json:"time_granularity"`    // "hour", "day", "week", "month"
	TimeShift          time.Duration          `json:"time_shift"`          // Random time shift range
	
	// Value-based masking
	OutlierHandling    string                 `json:"outlier_handling"`    // "suppress", "cap", "replace"
	OutlierThreshold   float64                `json:"outlier_threshold"`   // Threshold for outlier detection (in std devs)
	ReplacementMethod  string                 `json:"replacement_method"`  // "mean", "median", "mode", "interpolate"
	
	// Anonymization parameters
	KAnonymity         int                    `json:"k_anonymity"`         // K-anonymity parameter
	LDiversity         int                    `json:"l_diversity"`         // L-diversity parameter
	
	// Quality preservation
	PreserveDistribution bool                 `json:"preserve_distribution"` // Preserve overall distribution
	PreserveTrends       bool                 `json:"preserve_trends"`       // Preserve temporal trends
	PreserveSeasonality  bool                 `json:"preserve_seasonality"`  // Preserve seasonal patterns
	
	// Metadata
	Seed               int64                  `json:"seed"`
	Metadata           map[string]interface{} `json:"metadata"`
}

// MaskingResult contains the result of data masking
type MaskingResult struct {
	OriginalData       []float64              `json:"original_data,omitempty"`
	MaskedData         []float64              `json:"masked_data"`
	MaskingStrategy    string                 `json:"masking_strategy"`
	UtilityLoss        float64                `json:"utility_loss"`
	PrivacyGain        float64                `json:"privacy_gain"`
	KAnonymityLevel    int                    `json:"k_anonymity_level"`
	LDiversityLevel    int                    `json:"l_diversity_level"`
	MaskingStatistics  *MaskingStatistics     `json:"masking_statistics"`
	QualityMetrics     *QualityMetrics        `json:"quality_metrics"`
	Metadata           map[string]interface{} `json:"metadata"`
	ProcessedAt        time.Time              `json:"processed_at"`
}

// MaskingStatistics provides statistics about the masking process
type MaskingStatistics struct {
	TotalValues        int                    `json:"total_values"`
	SuppressedValues   int                    `json:"suppressed_values"`
	GeneralizedValues  int                    `json:"generalized_values"`
	PerturbedValues    int                    `json:"perturbed_values"`
	ReplacedValues     int                    `json:"replaced_values"`
	OutliersDetected   int                    `json:"outliers_detected"`
	SuppressionRate    float64                `json:"suppression_rate"`
	MaskingCoverage    float64                `json:"masking_coverage"`
}

// QualityMetrics measures quality preservation after masking
type QualityMetrics struct {
	MeanPreservation     float64 `json:"mean_preservation"`
	VariancePreservation float64 `json:"variance_preservation"`
	TrendPreservation    float64 `json:"trend_preservation"`
	CorrelationLoss      float64 `json:"correlation_loss"`
	DistributionSimilarity float64 `json:"distribution_similarity"`
	OverallQuality       float64 `json:"overall_quality"`
}

// NewDataMaskingEngine creates a new data masking engine
func NewDataMaskingEngine(config *MaskingConfig, logger *logrus.Logger) *DataMaskingEngine {
	if config == nil {
		config = getDefaultMaskingConfig()
	}
	
	if logger == nil {
		logger = logrus.New()
	}
	
	if config.Seed == 0 {
		config.Seed = time.Now().UnixNano()
	}
	
	return &DataMaskingEngine{
		logger:     logger,
		randSource: rand.New(rand.NewSource(config.Seed)),
		config:     config,
	}
}

// MaskTimeSeries applies data masking to a time series
func (dme *DataMaskingEngine) MaskTimeSeries(
	ctx context.Context,
	timeSeries *models.TimeSeries,
) (*MaskingResult, error) {
	
	if timeSeries == nil || len(timeSeries.DataPoints) == 0 {
		return nil, errors.NewValidationError("EMPTY_DATA", "Time series cannot be empty")
	}
	
	dme.logger.WithFields(logrus.Fields{
		"series_id":    timeSeries.ID,
		"data_points":  len(timeSeries.DataPoints),
		"strategy":     dme.config.Strategy,
	}).Info("Starting data masking")
	
	start := time.Now()
	
	// Extract values and timestamps
	originalValues := make([]float64, len(timeSeries.DataPoints))
	timestamps := make([]time.Time, len(timeSeries.DataPoints))
	
	for i, dp := range timeSeries.DataPoints {
		originalValues[i] = dp.Value
		timestamps[i] = dp.Timestamp
	}
	
	// Apply masking strategy
	maskedValues, err := dme.applyMaskingStrategy(ctx, originalValues, timestamps)
	if err != nil {
		return nil, err
	}
	
	// Calculate masking statistics
	stats := dme.calculateMaskingStatistics(originalValues, maskedValues)
	
	// Calculate quality metrics
	quality := dme.calculateQualityMetrics(originalValues, maskedValues)
	
	// Calculate utility loss and privacy gain
	utilityLoss := dme.calculateUtilityLoss(originalValues, maskedValues)
	privacyGain := dme.calculatePrivacyGain(originalValues, maskedValues)
	
	// Assess anonymization levels
	kLevel := dme.assessKAnonymity(maskedValues)
	lLevel := dme.assessLDiversity(maskedValues)
	
	result := &MaskingResult{
		MaskedData:         maskedValues,
		MaskingStrategy:    dme.config.Strategy,
		UtilityLoss:        utilityLoss,
		PrivacyGain:        privacyGain,
		KAnonymityLevel:    kLevel,
		LDiversityLevel:    lLevel,
		MaskingStatistics:  stats,
		QualityMetrics:     quality,
		Metadata: map[string]interface{}{
			"processing_time": time.Since(start).String(),
			"original_length": len(originalValues),
			"masked_length":   len(maskedValues),
			"strategy":        dme.config.Strategy,
			"preserve_trends": dme.config.PreserveTrends,
		},
		ProcessedAt: time.Now(),
	}
	
	dme.logger.WithFields(logrus.Fields{
		"series_id":     timeSeries.ID,
		"utility_loss":  utilityLoss,
		"privacy_gain":  privacyGain,
		"k_anonymity":   kLevel,
		"processing_time": time.Since(start),
	}).Info("Data masking completed")
	
	return result, nil
}

// MaskValues applies masking to raw values
func (dme *DataMaskingEngine) MaskValues(
	ctx context.Context,
	values []float64,
) ([]float64, error) {
	
	if len(values) == 0 {
		return []float64{}, nil
	}
	
	// Create dummy timestamps
	timestamps := make([]time.Time, len(values))
	baseTime := time.Now()
	for i := range timestamps {
		timestamps[i] = baseTime.Add(time.Duration(i) * time.Hour)
	}
	
	return dme.applyMaskingStrategy(ctx, values, timestamps)
}

// applyMaskingStrategy applies the configured masking strategy
func (dme *DataMaskingEngine) applyMaskingStrategy(
	ctx context.Context,
	values []float64,
	timestamps []time.Time,
) ([]float64, error) {
	
	switch dme.config.Strategy {
	case "suppress":
		return dme.applySuppression(values), nil
	case "generalize":
		return dme.applyGeneralization(values), nil
	case "perturb":
		return dme.applyPerturbation(values), nil
	case "synthetic":
		return dme.applySyntheticReplacement(values), nil
	case "hybrid":
		return dme.applyHybridMasking(values), nil
	default:
		return nil, errors.NewValidationError("UNKNOWN_STRATEGY", 
			fmt.Sprintf("Unknown masking strategy: %s", dme.config.Strategy))
	}
}

// applySuppression suppresses (removes or nullifies) values
func (dme *DataMaskingEngine) applySuppression(values []float64) []float64 {
	result := make([]float64, len(values))
	copy(result, values)
	
	// Detect outliers first
	outlierIndices := dme.detectOutliers(values)
	
	// Suppress outliers
	for _, idx := range outlierIndices {
		if dme.config.OutlierHandling == "suppress" {
			result[idx] = math.NaN() // Mark as suppressed
		}
	}
	
	// Random suppression
	numToSuppress := int(float64(len(values)) * dme.config.SuppressionRate)
	suppressedIndices := dme.selectRandomIndices(len(values), numToSuppress)
	
	for _, idx := range suppressedIndices {
		if !math.IsNaN(result[idx]) { // Don't suppress already suppressed values
			result[idx] = math.NaN()
		}
	}
	
	// Handle NaN values based on replacement method
	result = dme.handleSuppressedValues(result, values)
	
	return result
}

// applyGeneralization applies value generalization (binning)
func (dme *DataMaskingEngine) applyGeneralization(values []float64) []float64 {
	if len(values) == 0 {
		return values
	}
	
	// Calculate value range
	min, max := values[0], values[0]
	for _, v := range values[1:] {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	
	// Create bins
	numBins := dme.config.GeneralizationBins
	if numBins <= 0 {
		numBins = 10
	}
	
	binWidth := (max - min) / float64(numBins)
	if binWidth == 0 {
		return values // All values are the same
	}
	
	result := make([]float64, len(values))
	
	for i, value := range values {
		// Determine bin
		binIndex := int((value - min) / binWidth)
		if binIndex >= numBins {
			binIndex = numBins - 1
		}
		
		// Replace with bin center
		binCenter := min + (float64(binIndex)+0.5)*binWidth
		result[i] = binCenter
	}
	
	return result
}

// applyPerturbation applies random perturbation to values
func (dme *DataMaskingEngine) applyPerturbation(values []float64) []float64 {
	result := make([]float64, len(values))
	
	for i, value := range values {
		// Add random perturbation
		perturbation := (dme.randSource.Float64() - 0.5) * 2 * dme.config.PerturbationRange
		result[i] = value + perturbation
	}
	
	return result
}

// applySyntheticReplacement replaces values with synthetic ones
func (dme *DataMaskingEngine) applySyntheticReplacement(values []float64) []float64 {
	if len(values) == 0 {
		return values
	}
	
	// Calculate distribution parameters
	mean := dme.calculateMean(values)
	stdDev := dme.calculateStdDev(values)
	
	result := make([]float64, len(values))
	
	for i := range values {
		// Generate synthetic value from normal distribution
		result[i] = dme.randSource.NormFloat64()*stdDev + mean
	}
	
	// Preserve trends if configured
	if dme.config.PreserveTrends {
		result = dme.preserveTrends(values, result)
	}
	
	return result
}

// applyHybridMasking applies a combination of masking strategies
func (dme *DataMaskingEngine) applyHybridMasking(values []float64) []float64 {
	result := make([]float64, len(values))
	copy(result, values)
	
	// Apply different strategies to different portions of the data
	third := len(values) / 3
	
	// First third: suppression
	if third > 0 {
		portion1 := dme.applySuppression(result[:third])
		copy(result[:third], portion1)
	}
	
	// Second third: generalization
	if third > 0 && 2*third <= len(values) {
		portion2 := dme.applyGeneralization(result[third:2*third])
		copy(result[third:2*third], portion2)
	}
	
	// Final third: perturbation
	if 2*third < len(values) {
		portion3 := dme.applyPerturbation(result[2*third:])
		copy(result[2*third:], portion3)
	}
	
	return result
}

// Helper methods

func (dme *DataMaskingEngine) detectOutliers(values []float64) []int {
	if len(values) < 4 {
		return []int{}
	}
	
	mean := dme.calculateMean(values)
	stdDev := dme.calculateStdDev(values)
	threshold := dme.config.OutlierThreshold * stdDev
	
	outliers := []int{}
	for i, value := range values {
		if math.Abs(value-mean) > threshold {
			outliers = append(outliers, i)
		}
	}
	
	return outliers
}

func (dme *DataMaskingEngine) selectRandomIndices(total, count int) []int {
	if count >= total {
		indices := make([]int, total)
		for i := range indices {
			indices[i] = i
		}
		return indices
	}
	
	indices := make([]int, count)
	selected := make(map[int]bool)
	
	for i := 0; i < count; i++ {
		for {
			idx := dme.randSource.Intn(total)
			if !selected[idx] {
				indices[i] = idx
				selected[idx] = true
				break
			}
		}
	}
	
	return indices
}

func (dme *DataMaskingEngine) handleSuppressedValues(result, original []float64) []float64 {
	for i, val := range result {
		if math.IsNaN(val) {
			switch dme.config.ReplacementMethod {
			case "mean":
				result[i] = dme.calculateMean(original)
			case "median":
				result[i] = dme.calculateMedian(original)
			case "interpolate":
				result[i] = dme.interpolateValue(original, i)
			default:
				result[i] = 0 // Default to zero
			}
		}
	}
	
	return result
}

func (dme *DataMaskingEngine) preserveTrends(original, synthetic []float64) []float64 {
	if len(original) != len(synthetic) || len(original) < 2 {
		return synthetic
	}
	
	// Calculate trend in original data
	originalTrend := make([]float64, len(original)-1)
	for i := 1; i < len(original); i++ {
		originalTrend[i-1] = original[i] - original[i-1]
	}
	
	// Apply similar trend to synthetic data
	result := make([]float64, len(synthetic))
	result[0] = synthetic[0]
	
	for i := 1; i < len(result); i++ {
		trendIdx := i - 1
		if trendIdx < len(originalTrend) {
			result[i] = result[i-1] + originalTrend[trendIdx]
		} else {
			result[i] = synthetic[i]
		}
	}
	
	return result
}

func (dme *DataMaskingEngine) calculateMaskingStatistics(original, masked []float64) *MaskingStatistics {
	stats := &MaskingStatistics{
		TotalValues: len(original),
	}
	
	// Count different types of transformations
	for i := 0; i < len(original) && i < len(masked); i++ {
		if math.IsNaN(masked[i]) {
			stats.SuppressedValues++
		} else if original[i] != masked[i] {
			// Determine type of transformation based on magnitude of change
			diff := math.Abs(original[i] - masked[i])
			if diff > 0.1*math.Abs(original[i]) {
				stats.PerturbedValues++
			} else {
				stats.GeneralizedValues++
			}
		}
	}
	
	stats.SuppressionRate = float64(stats.SuppressedValues) / float64(stats.TotalValues)
	stats.MaskingCoverage = float64(stats.SuppressedValues+stats.GeneralizedValues+stats.PerturbedValues) / float64(stats.TotalValues)
	
	return stats
}

func (dme *DataMaskingEngine) calculateQualityMetrics(original, masked []float64) *QualityMetrics {
	if len(original) == 0 || len(masked) == 0 {
		return &QualityMetrics{}
	}
	
	// Calculate preservation metrics
	origMean := dme.calculateMean(original)
	maskedMean := dme.calculateMean(masked)
	meanPreservation := 1.0 - math.Abs(origMean-maskedMean)/math.Abs(origMean+1e-10)
	
	origVar := dme.calculateVariance(original)
	maskedVar := dme.calculateVariance(masked)
	varPreservation := 1.0 - math.Abs(origVar-maskedVar)/math.Abs(origVar+1e-10)
	
	// Calculate correlation
	correlation := dme.calculateCorrelation(original, masked)
	correlationLoss := 1.0 - math.Abs(correlation)
	
	// Overall quality score
	overallQuality := (meanPreservation + varPreservation + (1.0-correlationLoss)) / 3.0
	
	return &QualityMetrics{
		MeanPreservation:     meanPreservation,
		VariancePreservation: varPreservation,
		TrendPreservation:    0.8, // Placeholder
		CorrelationLoss:      correlationLoss,
		DistributionSimilarity: 0.8, // Placeholder
		OverallQuality:       overallQuality,
	}
}

func (dme *DataMaskingEngine) calculateUtilityLoss(original, masked []float64) float64 {
	if len(original) != len(masked) || len(original) == 0 {
		return 1.0
	}
	
	// Calculate MSE-based utility loss
	var mse float64
	for i := 0; i < len(original); i++ {
		diff := original[i] - masked[i]
		mse += diff * diff
	}
	mse /= float64(len(original))
	
	// Normalize by data variance
	variance := dme.calculateVariance(original)
	utilityLoss := mse / (variance + 1e-10)
	
	return math.Min(1.0, utilityLoss)
}

func (dme *DataMaskingEngine) calculatePrivacyGain(original, masked []float64) float64 {
	// Simple privacy gain metric based on how much data was changed
	if len(original) != len(masked) || len(original) == 0 {
		return 0.0
	}
	
	changedValues := 0
	for i := 0; i < len(original); i++ {
		if math.Abs(original[i]-masked[i]) > 1e-10 {
			changedValues++
		}
	}
	
	return float64(changedValues) / float64(len(original))
}

func (dme *DataMaskingEngine) assessKAnonymity(values []float64) int {
	// Simplified k-anonymity assessment
	// Count frequency of each unique value
	frequency := make(map[float64]int)
	for _, v := range values {
		frequency[v]++
	}
	
	// Find minimum frequency (k-anonymity level)
	minFreq := len(values)
	for _, freq := range frequency {
		if freq < minFreq {
			minFreq = freq
		}
	}
	
	return minFreq
}

func (dme *DataMaskingEngine) assessLDiversity(values []float64) int {
	// Simplified l-diversity assessment
	// Count number of unique values
	unique := make(map[float64]bool)
	for _, v := range values {
		unique[v] = true
	}
	
	return len(unique)
}

// Utility calculation methods

func (dme *DataMaskingEngine) calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	var sum float64
	count := 0
	for _, v := range values {
		if !math.IsNaN(v) {
			sum += v
			count++
		}
	}
	
	if count == 0 {
		return 0
	}
	
	return sum / float64(count)
}

func (dme *DataMaskingEngine) calculateStdDev(values []float64) float64 {
	return math.Sqrt(dme.calculateVariance(values))
}

func (dme *DataMaskingEngine) calculateVariance(values []float64) float64 {
	if len(values) <= 1 {
		return 0
	}
	
	mean := dme.calculateMean(values)
	var sum float64
	count := 0
	
	for _, v := range values {
		if !math.IsNaN(v) {
			diff := v - mean
			sum += diff * diff
			count++
		}
	}
	
	if count <= 1 {
		return 0
	}
	
	return sum / float64(count-1)
}

func (dme *DataMaskingEngine) calculateMedian(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	// Filter out NaN values
	filtered := make([]float64, 0, len(values))
	for _, v := range values {
		if !math.IsNaN(v) {
			filtered = append(filtered, v)
		}
	}
	
	if len(filtered) == 0 {
		return 0
	}
	
	sort.Float64s(filtered)
	n := len(filtered)
	
	if n%2 == 0 {
		return (filtered[n/2-1] + filtered[n/2]) / 2
	}
	return filtered[n/2]
}

func (dme *DataMaskingEngine) interpolateValue(values []float64, index int) float64 {
	if index < 0 || index >= len(values) {
		return 0
	}
	
	// Find nearest non-NaN values
	left, right := -1, -1
	
	for i := index - 1; i >= 0; i-- {
		if !math.IsNaN(values[i]) {
			left = i
			break
		}
	}
	
	for i := index + 1; i < len(values); i++ {
		if !math.IsNaN(values[i]) {
			right = i
			break
		}
	}
	
	if left >= 0 && right >= 0 {
		// Linear interpolation
		t := float64(index-left) / float64(right-left)
		return values[left] + t*(values[right]-values[left])
	} else if left >= 0 {
		return values[left]
	} else if right >= 0 {
		return values[right]
	} else {
		return 0
	}
}

func (dme *DataMaskingEngine) calculateCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) < 2 {
		return 0
	}
	
	meanX := dme.calculateMean(x)
	meanY := dme.calculateMean(y)
	
	var numerator, sumSquareX, sumSquareY float64
	count := 0
	
	for i := 0; i < len(x); i++ {
		if !math.IsNaN(x[i]) && !math.IsNaN(y[i]) {
			diffX := x[i] - meanX
			diffY := y[i] - meanY
			numerator += diffX * diffY
			sumSquareX += diffX * diffX
			sumSquareY += diffY * diffY
			count++
		}
	}
	
	if count < 2 {
		return 0
	}
	
	denominator := math.Sqrt(sumSquareX * sumSquareY)
	if denominator == 0 {
		return 0
	}
	
	return numerator / denominator
}

func getDefaultMaskingConfig() *MaskingConfig {
	return &MaskingConfig{
		Strategy:           "perturb",
		SuppressionRate:    0.1,
		GeneralizationBins: 10,
		PerturbationRange:  0.1,
		TimeGranularity:    "hour",
		TimeShift:          time.Hour,
		OutlierHandling:    "cap",
		OutlierThreshold:   3.0,
		ReplacementMethod:  "interpolate",
		KAnonymity:         5,
		LDiversity:         3,
		PreserveDistribution: true,
		PreserveTrends:     true,
		PreserveSeasonality: true,
		Seed:               time.Now().UnixNano(),
		Metadata:           make(map[string]interface{}),
	}
}