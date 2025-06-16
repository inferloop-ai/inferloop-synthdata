package analytics

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
	"github.com/inferloop/tsiot/pkg/errors"
)

// Engine provides time series analytics capabilities
type Engine struct {
	storage interfaces.Storage
	logger  *logrus.Logger
	cache   *AnalyticsCache
	config  *EngineConfig
}

// EngineConfig contains configuration for the analytics engine
type EngineConfig struct {
	CacheEnabled     bool          `json:"cache_enabled" yaml:"cache_enabled"`
	CacheTTL         time.Duration `json:"cache_ttl" yaml:"cache_ttl"`
	MaxCacheSize     int           `json:"max_cache_size" yaml:"max_cache_size"`
	ParallelWorkers  int           `json:"parallel_workers" yaml:"parallel_workers"`
	BatchSize        int           `json:"batch_size" yaml:"batch_size"`
	SmoothingFactor  float64       `json:"smoothing_factor" yaml:"smoothing_factor"`
	SeasonalityLags  []int         `json:"seasonality_lags" yaml:"seasonality_lags"`
	AnomalyThreshold float64       `json:"anomaly_threshold" yaml:"anomaly_threshold"`
}

// AnalyticsCache provides caching for analytics results
type AnalyticsCache struct {
	data map[string]*CacheEntry
	mu   sync.RWMutex
	ttl  time.Duration
}

// CacheEntry represents a cached analytics result
type CacheEntry struct {
	Data      interface{}
	CreatedAt time.Time
	ExpiresAt time.Time
}

// AnalysisRequest represents a request for analytics
type AnalysisRequest struct {
	SeriesID     string                 `json:"series_id"`
	StartTime    *time.Time             `json:"start_time,omitempty"`
	EndTime      *time.Time             `json:"end_time,omitempty"`
	AnalysisType []string               `json:"analysis_type"`
	Parameters   map[string]interface{} `json:"parameters,omitempty"`
}

// AnalysisResult contains the results of time series analysis
type AnalysisResult struct {
	SeriesID         string                    `json:"series_id"`
	AnalysisType     string                    `json:"analysis_type"`
	BasicStats       *BasicStatistics          `json:"basic_stats,omitempty"`
	TrendAnalysis    *TrendAnalysis            `json:"trend_analysis,omitempty"`
	SeasonalityInfo  *SeasonalityAnalysis      `json:"seasonality_info,omitempty"`
	AnomalyDetection *AnomalyDetectionResult   `json:"anomaly_detection,omitempty"`
	Forecasting      *ForecastingResult        `json:"forecasting,omitempty"`
	PatternAnalysis  *PatternAnalysisResult    `json:"pattern_analysis,omitempty"`
	CorrelationInfo  *CorrelationAnalysis      `json:"correlation_info,omitempty"`
	QualityMetrics   *QualityAssessment        `json:"quality_metrics,omitempty"`
	ProcessingTime   time.Duration             `json:"processing_time"`
	Timestamp        time.Time                 `json:"timestamp"`
}

// BasicStatistics contains basic statistical measures
type BasicStatistics struct {
	Count          int64     `json:"count"`
	Mean           float64   `json:"mean"`
	Median         float64   `json:"median"`
	Mode           float64   `json:"mode"`
	StandardDev    float64   `json:"standard_deviation"`
	Variance       float64   `json:"variance"`
	Skewness       float64   `json:"skewness"`
	Kurtosis       float64   `json:"kurtosis"`
	Min            float64   `json:"minimum"`
	Max            float64   `json:"maximum"`
	Range          float64   `json:"range"`
	Q1             float64   `json:"q1"`
	Q3             float64   `json:"q3"`
	IQR            float64   `json:"iqr"`
	CV             float64   `json:"coefficient_of_variation"`
	MAD            float64   `json:"mean_absolute_deviation"`
	StartTime      time.Time `json:"start_time"`
	EndTime        time.Time `json:"end_time"`
	Duration       string    `json:"duration"`
	SamplingRate   float64   `json:"sampling_rate"`
	MissingValues  int       `json:"missing_values"`
	OutlierCount   int       `json:"outlier_count"`
}

// TrendAnalysis contains trend analysis results
type TrendAnalysis struct {
	Direction      string    `json:"direction"`        // "increasing", "decreasing", "stable"
	Strength       float64   `json:"strength"`         // 0-1 scale
	Slope          float64   `json:"slope"`
	Intercept      float64   `json:"intercept"`
	RSquared       float64   `json:"r_squared"`
	PValue         float64   `json:"p_value"`
	Significance   string    `json:"significance"`     // "significant", "not_significant"
	TrendPoints    []Point   `json:"trend_points"`
	ChangePoints   []Point   `json:"change_points"`
	LinearFit      LinearFit `json:"linear_fit"`
}

// SeasonalityAnalysis contains seasonality detection results
type SeasonalityAnalysis struct {
	HasSeasonality     bool                  `json:"has_seasonality"`
	SeasonalStrength   float64               `json:"seasonal_strength"`
	DominantPeriods    []SeasonalPeriod      `json:"dominant_periods"`
	SeasonalIndices    []float64             `json:"seasonal_indices"`
	DecomposedSeries   *SeriesDecomposition  `json:"decomposed_series,omitempty"`
	SpectralAnalysis   *SpectralAnalysis     `json:"spectral_analysis,omitempty"`
	AutocorrelationACF []float64             `json:"autocorrelation_acf"`
}

// AnomalyDetectionResult contains anomaly detection results
type AnomalyDetectionResult struct {
	Method           string         `json:"method"`
	AnomaliesFound   []Anomaly      `json:"anomalies_found"`
	AnomalyRate      float64        `json:"anomaly_rate"`
	Threshold        float64        `json:"threshold"`
	ConfidenceLevel  float64        `json:"confidence_level"`
	ModelParameters  map[string]interface{} `json:"model_parameters"`
}

// ForecastingResult contains forecasting results
type ForecastingResult struct {
	Method              string            `json:"method"`
	ForecastHorizon     int               `json:"forecast_horizon"`
	PredictedValues     []ForecastPoint   `json:"predicted_values"`
	ConfidenceIntervals []ConfidenceInterval `json:"confidence_intervals"`
	Accuracy            *AccuracyMetrics  `json:"accuracy,omitempty"`
	ModelDiagnostics    *ModelDiagnostics `json:"model_diagnostics,omitempty"`
}

// PatternAnalysisResult contains pattern recognition results
type PatternAnalysisResult struct {
	DetectedPatterns []Pattern         `json:"detected_patterns"`
	PatternFrequency map[string]int    `json:"pattern_frequency"`
	SimilarityMatrix [][]float64       `json:"similarity_matrix,omitempty"`
	MotifAnalysis    *MotifAnalysis    `json:"motif_analysis,omitempty"`
}

// CorrelationAnalysis contains correlation analysis results
type CorrelationAnalysis struct {
	AutoCorrelation  []float64           `json:"auto_correlation"`
	CrossCorrelation map[string][]float64 `json:"cross_correlation,omitempty"`
	LagAnalysis      []LagCorrelation    `json:"lag_analysis"`
	CorrelationTests []CorrelationTest   `json:"correlation_tests"`
}

// QualityAssessment contains data quality metrics
type QualityAssessment struct {
	CompletenessScore  float64           `json:"completeness_score"`
	ConsistencyScore   float64           `json:"consistency_score"`
	AccuracyScore      float64           `json:"accuracy_score"`
	ValidityScore      float64           `json:"validity_score"`
	OverallScore       float64           `json:"overall_score"`
	QualityIssues      []QualityIssue    `json:"quality_issues"`
	Recommendations    []string          `json:"recommendations"`
}

// Supporting types
type Point struct {
	Timestamp time.Time `json:"timestamp"`
	Value     float64   `json:"value"`
}

type LinearFit struct {
	Slope     float64 `json:"slope"`
	Intercept float64 `json:"intercept"`
	RSquared  float64 `json:"r_squared"`
}

type SeasonalPeriod struct {
	Period    int     `json:"period"`
	Strength  float64 `json:"strength"`
	Type      string  `json:"type"` // "daily", "weekly", "monthly", "yearly", "custom"
}

type SeriesDecomposition struct {
	Trend     []float64 `json:"trend"`
	Seasonal  []float64 `json:"seasonal"`
	Residual  []float64 `json:"residual"`
	Method    string    `json:"method"`
}

type SpectralAnalysis struct {
	DominantFrequencies []float64 `json:"dominant_frequencies"`
	PowerSpectrum      []float64 `json:"power_spectrum"`
	Method            string    `json:"method"`
}

type Peak struct {
	Frequency float64 `json:"frequency"`
	Power     float64 `json:"power"`
	Period    float64 `json:"period"`
}

type Anomaly struct {
	Index       int       `json:"index"`
	Timestamp   time.Time `json:"timestamp"`
	Value       float64   `json:"value"`
	Score       float64   `json:"score"`
	Type        string    `json:"type"`
	Severity    string    `json:"severity"`
	Description string    `json:"description"`
}

type ForecastPoint struct {
	Index     int       `json:"index"`
	Timestamp time.Time `json:"timestamp"`
	Value     float64   `json:"value"`
}

type ConfidenceInterval struct {
	Level      float64 `json:"level"`
	Lower      float64 `json:"lower"`
	Upper      float64 `json:"upper"`
	Prediction float64 `json:"prediction"`
}

type AccuracyMetrics struct {
	MAE  float64 `json:"mae"`
	MAPE float64 `json:"mape"`
	RMSE float64 `json:"rmse"`
	MSE  float64 `json:"mse"`
}

type ModelDiagnostics struct {
	Residuals        []float64 `json:"residuals"`
	ResidualsVariance float64   `json:"residuals_variance"`
	Normality        string    `json:"normality"`
	Autocorrelation  []float64 `json:"autocorrelation"`
}

type Pattern struct {
	Type        string                 `json:"type"`
	StartIndex  int                    `json:"start_index"`
	EndIndex    int                    `json:"end_index"`
	Strength    float64                `json:"strength"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

type MotifAnalysis struct {
	MotifCount       int     `json:"motif_count"`
	AverageLength    float64 `json:"average_length"`
	MaxSimilarity    float64 `json:"max_similarity"`
	DiscoveryMethod  string  `json:"discovery_method"`
}

type Motif struct {
	Pattern    []float64 `json:"pattern"`
	Locations  []int     `json:"locations"`
	Frequency  int       `json:"frequency"`
	Confidence float64   `json:"confidence"`
}

type Discord struct {
	Location   int     `json:"location"`
	Length     int     `json:"length"`
	Score      float64 `json:"score"`
	Pattern    []float64 `json:"pattern"`
}

type LagCorrelation struct {
	Lag          int     `json:"lag"`
	Correlation  float64 `json:"correlation"`
	Significance string  `json:"significance"`
	PValue       float64 `json:"p_value"`
}

type CorrelationTest struct {
	TestName     string  `json:"test_name"`
	Statistic    float64 `json:"statistic"`
	PValue       float64 `json:"p_value"`
	Conclusion   string  `json:"conclusion"`
	Significance string  `json:"significance"`
}

type QualityIssue struct {
	Type           string `json:"type"`
	Severity       string `json:"severity"`
	Description    string `json:"description"`
	AffectedPoints int    `json:"affected_points"`
}

type Location struct {
	StartTime time.Time `json:"start_time"`
	EndTime   time.Time `json:"end_time"`
	Indices   []int     `json:"indices,omitempty"`
}

type StatisticalTest struct {
	TestName    string  `json:"test_name"`
	Statistic   float64 `json:"statistic"`
	PValue      float64 `json:"p_value"`
	Critical    float64 `json:"critical_value"`
	Significant bool    `json:"significant"`
	Alpha       float64 `json:"alpha"`
}

// NewEngine creates a new analytics engine
func NewEngine(storage interfaces.Storage, logger *logrus.Logger) *Engine {
	if logger == nil {
		logger = logrus.New()
	}

	config := &EngineConfig{
		CacheEnabled:     true,
		CacheTTL:         30 * time.Minute,
		MaxCacheSize:     1000,
		ParallelWorkers:  4,
		BatchSize:        10000,
		SmoothingFactor:  0.3,
		SeasonalityLags:  []int{24, 168, 8760}, // hourly, daily, weekly, yearly patterns
		AnomalyThreshold: 2.5,
	}

	cache := &AnalyticsCache{
		data: make(map[string]*CacheEntry),
		ttl:  config.CacheTTL,
	}

	return &Engine{
		storage: storage,
		logger:  logger,
		cache:   cache,
		config:  config,
	}
}

// PerformAnalysis executes the requested analysis on time series data
func (e *Engine) PerformAnalysis(ctx context.Context, request *AnalysisRequest) (*AnalysisResult, error) {
	start := time.Now()

	// Check cache first
	if e.config.CacheEnabled {
		if cached := e.getCachedResult(request); cached != nil {
			e.logger.WithField("series_id", request.SeriesID).Debug("Returning cached analytics result")
			return cached, nil
		}
	}

	// Load time series data
	timeSeries, err := e.loadTimeSeries(ctx, request)
	if err != nil {
		return nil, err
	}

	if len(timeSeries.DataPoints) == 0 {
		return nil, errors.NewValidationError("NO_DATA", "No data points available for analysis")
	}

	// Initialize result
	result := &AnalysisResult{
		SeriesID:  request.SeriesID,
		Timestamp: time.Now(),
	}

	// Extract values for analysis
	values := make([]float64, len(timeSeries.DataPoints))
	timestamps := make([]time.Time, len(timeSeries.DataPoints))
	for i, dp := range timeSeries.DataPoints {
		values[i] = dp.Value
		timestamps[i] = dp.Timestamp
	}

	// Perform requested analyses
	for _, analysisType := range request.AnalysisType {
		switch analysisType {
		case "basic", "all":
			result.BasicStats = e.calculateBasicStatistics(values, timestamps)
		case "trend", "all":
			result.TrendAnalysis = e.analyzeTrend(values, timestamps)
		case "seasonality", "all":
			result.SeasonalityInfo = e.analyzeSeasonality(values, timestamps)
		case "anomaly", "all":
			result.AnomalyDetection = e.detectAnomalies(values, timestamps)
		case "forecast", "all":
			if params, ok := request.Parameters["forecast_horizon"].(float64); ok {
				result.Forecasting = e.generateForecast(values, timestamps, int(params))
			} else {
				result.Forecasting = e.generateForecast(values, timestamps, 24) // Default 24 periods
			}
		case "pattern", "all":
			result.PatternAnalysis = e.analyzePatterns(values, timestamps)
		case "correlation", "all":
			result.CorrelationInfo = e.analyzeCorrelation(values, timestamps)
		case "quality", "all":
			result.QualityMetrics = e.assessQuality(values, timestamps, timeSeries)
		}
	}

	result.ProcessingTime = time.Since(start)

	// Cache result
	if e.config.CacheEnabled {
		e.cacheResult(request, result)
	}

	return result, nil
}

// calculateBasicStatistics computes basic statistical measures
func (e *Engine) calculateBasicStatistics(values []float64, timestamps []time.Time) *BasicStatistics {
	n := len(values)
	if n == 0 {
		return &BasicStatistics{}
	}

	// Sort values for percentile calculations
	sorted := make([]float64, n)
	copy(sorted, values)
	sort.Float64s(sorted)

	// Basic measures
	mean := e.calculateMean(values)
	variance := e.calculateVariance(values, mean)
	stdDev := math.Sqrt(variance)
	
	median := e.calculateMedian(sorted)
	mode := e.calculateMode(values)
	min, max := sorted[0], sorted[n-1]
	
	// Percentiles
	q1 := e.calculatePercentile(sorted, 0.25)
	q3 := e.calculatePercentile(sorted, 0.75)
	iqr := q3 - q1
	
	// Advanced statistics
	skewness := e.calculateSkewness(values, mean, stdDev)
	kurtosis := e.calculateKurtosis(values, mean, stdDev)
	cv := stdDev / math.Abs(mean) * 100
	mad := e.calculateMAD(values, mean)
	
	// Time-based statistics
	duration := timestamps[n-1].Sub(timestamps[0])
	samplingRate := float64(n-1) / duration.Seconds()
	
	// Quality indicators
	missingValues := e.countMissingValues(values)
	outlierCount := e.countOutliers(values, q1, q3, iqr)

	return &BasicStatistics{
		Count:         int64(n),
		Mean:          mean,
		Median:        median,
		Mode:          mode,
		StandardDev:   stdDev,
		Variance:      variance,
		Skewness:      skewness,
		Kurtosis:      kurtosis,
		Min:           min,
		Max:           max,
		Range:         max - min,
		Q1:            q1,
		Q3:            q3,
		IQR:           iqr,
		CV:            cv,
		MAD:           mad,
		StartTime:     timestamps[0],
		EndTime:       timestamps[n-1],
		Duration:      duration.String(),
		SamplingRate:  samplingRate,
		MissingValues: missingValues,
		OutlierCount:  outlierCount,
	}
}

// Helper statistical functions
func (e *Engine) calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func (e *Engine) calculateVariance(values []float64, mean float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		diff := v - mean
		sum += diff * diff
	}
	return sum / float64(len(values))
}

func (e *Engine) calculateMedian(sortedValues []float64) float64 {
	n := len(sortedValues)
	if n == 0 {
		return 0
	}
	if n%2 == 0 {
		return (sortedValues[n/2-1] + sortedValues[n/2]) / 2
	}
	return sortedValues[n/2]
}

func (e *Engine) calculateMode(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	frequency := make(map[float64]int)
	for _, v := range values {
		frequency[v]++
	}
	
	var mode float64
	maxFreq := 0
	for value, freq := range frequency {
		if freq > maxFreq {
			maxFreq = freq
			mode = value
		}
	}
	
	return mode
}

func (e *Engine) calculatePercentile(sortedValues []float64, percentile float64) float64 {
	n := len(sortedValues)
	if n == 0 {
		return 0
	}
	
	index := percentile * float64(n-1)
	lower := int(math.Floor(index))
	upper := int(math.Ceil(index))
	
	if lower == upper {
		return sortedValues[lower]
	}
	
	weight := index - float64(lower)
	return sortedValues[lower]*(1-weight) + sortedValues[upper]*weight
}

func (e *Engine) calculateSkewness(values []float64, mean, stdDev float64) float64 {
	if len(values) == 0 || stdDev == 0 {
		return 0
	}
	
	sum := 0.0
	for _, v := range values {
		normalized := (v - mean) / stdDev
		sum += math.Pow(normalized, 3)
	}
	
	return sum / float64(len(values))
}

func (e *Engine) calculateKurtosis(values []float64, mean, stdDev float64) float64 {
	if len(values) == 0 || stdDev == 0 {
		return 0
	}
	
	sum := 0.0
	for _, v := range values {
		normalized := (v - mean) / stdDev
		sum += math.Pow(normalized, 4)
	}
	
	return sum/float64(len(values)) - 3 // Subtract 3 for excess kurtosis
}

func (e *Engine) calculateMAD(values []float64, mean float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	sum := 0.0
	for _, v := range values {
		sum += math.Abs(v - mean)
	}
	
	return sum / float64(len(values))
}

func (e *Engine) countMissingValues(values []float64) int {
	count := 0
	for _, v := range values {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			count++
		}
	}
	return count
}

func (e *Engine) countOutliers(values []float64, q1, q3, iqr float64) int {
	lowerBound := q1 - 1.5*iqr
	upperBound := q3 + 1.5*iqr
	
	count := 0
	for _, v := range values {
		if v < lowerBound || v > upperBound {
			count++
		}
	}
	return count
}

// analyzeTrend performs trend analysis
func (e *Engine) analyzeTrend(values []float64, timestamps []time.Time) *TrendAnalysis {
	n := len(values)
	if n < 2 {
		return &TrendAnalysis{Direction: "insufficient_data"}
	}

	// Linear regression for trend
	slope, intercept, rSquared := e.linearRegression(values)
	
	// Determine trend direction and strength
	direction := "stable"
	if slope > 0.01 {
		direction = "increasing"
	} else if slope < -0.01 {
		direction = "decreasing"
	}
	
	strength := math.Abs(slope) / (e.calculateMean(values) + 1e-10) // Normalize by mean
	if strength > 1 {
		strength = 1
	}
	
	// Statistical significance (simplified)
	pValue := e.calculateTrendPValue(values, slope)
	significance := "not_significant"
	if pValue < 0.05 {
		significance = "significant"
	}
	
	// Generate trend points
	trendPoints := make([]Point, n)
	for i := 0; i < n; i++ {
		trendValue := intercept + slope*float64(i)
		trendPoints[i] = Point{
			Timestamp: timestamps[i],
			Value:     trendValue,
		}
	}
	
	// Detect change points (simplified)
	changePoints := e.detectChangePoints(values, timestamps)
	
	return &TrendAnalysis{
		Direction:    direction,
		Strength:     strength,
		Slope:        slope,
		Intercept:    intercept,
		RSquared:     rSquared,
		PValue:       pValue,
		Significance: significance,
		TrendPoints:  trendPoints,
		ChangePoints: changePoints,
		LinearFit: LinearFit{
			Slope:     slope,
			Intercept: intercept,
			RSquared:  rSquared,
		},
	}
}

func (e *Engine) linearRegression(values []float64) (slope, intercept, rSquared float64) {
	n := float64(len(values))
	if n < 2 {
		return 0, 0, 0
	}
	
	// Calculate means
	sumX, sumY, sumXY, sumX2, sumY2 := 0.0, 0.0, 0.0, 0.0, 0.0
	
	for i, y := range values {
		x := float64(i)
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
		sumY2 += y * y
	}
	
	// Calculate slope and intercept
	slope = (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)
	intercept = (sumY - slope*sumX) / n
	
	// Calculate R-squared
	meanY := sumY / n
	ssTot := sumY2 - n*meanY*meanY
	ssRes := 0.0
	
	for i, y := range values {
		predicted := intercept + slope*float64(i)
		ssRes += (y - predicted) * (y - predicted)
	}
	
	if ssTot > 0 {
		rSquared = 1 - ssRes/ssTot
	}
	
	return slope, intercept, rSquared
}

func (e *Engine) calculateTrendPValue(values []float64, slope float64) float64 {
	// Simplified p-value calculation
	// In a real implementation, you would use proper statistical tests
	n := len(values)
	if n < 3 {
		return 1.0
	}
	
	// Estimate standard error (simplified)
	_, _, rSquared := e.linearRegression(values)
	standardError := math.Sqrt((1 - rSquared) / float64(n-2))
	
	// Calculate t-statistic
	tStat := math.Abs(slope) / (standardError + 1e-10)
	
	// Convert to approximate p-value (very simplified)
	if tStat > 2.576 {
		return 0.01 // p < 0.01
	} else if tStat > 1.96 {
		return 0.05 // p < 0.05
	} else if tStat > 1.645 {
		return 0.10 // p < 0.10
	}
	return 0.20 // p >= 0.10
}

func (e *Engine) detectChangePoints(values []float64, timestamps []time.Time) []Point {
	// Simplified change point detection using moving average differences
	var changePoints []Point
	windowSize := 10
	threshold := 1.5
	
	if len(values) < windowSize*2 {
		return changePoints
	}
	
	for i := windowSize; i < len(values)-windowSize; i++ {
		// Calculate moving averages before and after point
		beforeAvg := e.calculateMean(values[i-windowSize : i])
		afterAvg := e.calculateMean(values[i : i+windowSize])
		
		// Check for significant change
		if math.Abs(afterAvg-beforeAvg) > threshold*e.calculateStdDev(values) {
			changePoints = append(changePoints, Point{
				Timestamp: timestamps[i],
				Value:     values[i],
			})
		}
	}
	
	return changePoints
}

func (e *Engine) calculateStdDev(values []float64) float64 {
	mean := e.calculateMean(values)
	variance := e.calculateVariance(values, mean)
	return math.Sqrt(variance)
}

// Additional implementations for other analysis methods
func (e *Engine) analyzeSeasonality(values []float64, timestamps []time.Time) *SeasonalityAnalysis {
	if len(values) < 24 {
		return &SeasonalityAnalysis{
			HasSeasonality:   false,
			SeasonalStrength: 0.0,
			DominantPeriods:  []SeasonalPeriod{},
		}
	}

	// Calculate autocorrelation function
	acf := e.calculateAutocorrelation(values, min(len(values)/4, 100))
	
	// Detect potential seasonal periods
	periods := e.detectSeasonalPeriods(acf, timestamps)
	
	// Calculate seasonal strength
	seasonalStrength := e.calculateSeasonalStrength(values, periods)
	
	// Perform spectral analysis
	spectralResult := e.performSpectralAnalysis(values)
	
	// Decompose series if seasonality is detected
	var decomposition *SeriesDecomposition
	if len(periods) > 0 && seasonalStrength > 0.1 {
		decomposition = e.decomposeTimeSeries(values, periods[0].Period)
	}
	
	return &SeasonalityAnalysis{
		HasSeasonality:     seasonalStrength > 0.1 && len(periods) > 0,
		SeasonalStrength:   seasonalStrength,
		DominantPeriods:    periods,
		SeasonalIndices:    e.calculateSeasonalIndices(values, periods),
		DecomposedSeries:   decomposition,
		SpectralAnalysis:   spectralResult,
		AutocorrelationACF: acf,
	}
}

func (e *Engine) detectAnomalies(values []float64, timestamps []time.Time) *AnomalyDetectionResult {
	if len(values) == 0 {
		return &AnomalyDetectionResult{
			Method:         "statistical_zscore",
			AnomaliesFound: []Anomaly{},
			AnomalyRate:    0.0,
		}
	}

	// Calculate statistics
	mean := e.calculateMean(values)
	stdDev := e.calculateStandardDeviation(values)
	
	// Use configurable threshold or default
	threshold := e.config.AnomalyThreshold
	if threshold == 0 {
		threshold = 2.0 // 2 standard deviations
	}
	
	var anomalies []Anomaly
	
	// Z-score based anomaly detection
	for i, val := range values {
		zScore := math.Abs(val-mean) / stdDev
		if zScore > threshold {
			anomaly := Anomaly{
				Index:       i,
				Timestamp:   timestamps[i],
				Value:       val,
				Score:       zScore,
				Type:        "outlier",
				Severity:    e.classifyAnomalySeverity(zScore, threshold),
				Description: fmt.Sprintf("Z-score %.2f exceeds threshold %.2f", zScore, threshold),
			}
			anomalies = append(anomalies, anomaly)
		}
	}
	
	// Additional methods: IQR-based detection
	iqrAnomalies := e.detectIQRAnomalies(values, timestamps)
	anomalies = append(anomalies, iqrAnomalies...)
	
	// Moving average based detection for trend anomalies
	if len(values) > 10 {
		maAnomalies := e.detectMovingAverageAnomalies(values, timestamps, 5)
		anomalies = append(anomalies, maAnomalies...)
	}
	
	anomalyRate := float64(len(anomalies)) / float64(len(values))
	
	return &AnomalyDetectionResult{
		Method:          "multi_method",
		AnomaliesFound:  anomalies,
		AnomalyRate:     anomalyRate,
		Threshold:       threshold,
		ConfidenceLevel: 0.95,
		ModelParameters: map[string]interface{}{
			"mean":      mean,
			"std_dev":   stdDev,
			"threshold": threshold,
			"methods":   []string{"zscore", "iqr", "moving_average"},
		},
	}
}

func (e *Engine) generateForecast(values []float64, timestamps []time.Time, horizon int) *ForecastingResult {
	if len(values) < 2 || horizon <= 0 {
		return &ForecastingResult{
			Method:          "linear_trend",
			ForecastHorizon: horizon,
			PredictedValues: []ForecastPoint{},
		}
	}

	// Determine the best forecasting method based on data characteristics
	method := e.selectForecastingMethod(values)
	
	var predictions []ForecastPoint
	var accuracy *AccuracyMetrics
	var diagnostics *ModelDiagnostics
	
	switch method {
	case "linear_trend":
		predictions, accuracy = e.linearTrendForecast(values, timestamps, horizon)
	case "exponential_smoothing":
		predictions, accuracy = e.exponentialSmoothingForecast(values, timestamps, horizon)
	case "seasonal_naive":
		predictions, accuracy = e.seasonalNaiveForecast(values, timestamps, horizon)
	default:
		predictions, accuracy = e.linearTrendForecast(values, timestamps, horizon)
	}
	
	// Calculate confidence intervals
	confidenceIntervals := e.calculateConfidenceIntervals(predictions, values)
	
	// Generate model diagnostics
	diagnostics = e.generateModelDiagnostics(values, predictions[:min(len(predictions), len(values))])
	
	return &ForecastingResult{
		Method:              method,
		ForecastHorizon:     horizon,
		PredictedValues:     predictions,
		ConfidenceIntervals: confidenceIntervals,
		Accuracy:            accuracy,
		ModelDiagnostics:    diagnostics,
	}
}

func (e *Engine) analyzePatterns(values []float64, timestamps []time.Time) *PatternAnalysisResult {
	if len(values) < 10 {
		return &PatternAnalysisResult{
			DetectedPatterns: []Pattern{},
			PatternFrequency: make(map[string]int),
		}
	}

	var detectedPatterns []Pattern
	patternFrequency := make(map[string]int)
	
	// Detect trend patterns
	trendPatterns := e.detectTrendPatterns(values, timestamps)
	detectedPatterns = append(detectedPatterns, trendPatterns...)
	for _, pattern := range trendPatterns {
		patternFrequency[pattern.Type]++
	}
	
	// Detect repeating patterns (motifs)
	motifs := e.detectMotifs(values, timestamps, 5) // window size of 5
	detectedPatterns = append(detectedPatterns, motifs...)
	for _, motif := range motifs {
		patternFrequency[motif.Type]++
	}
	
	// Detect level shifts
	levelShifts := e.detectLevelShifts(values, timestamps)
	detectedPatterns = append(detectedPatterns, levelShifts...)
	for _, shift := range levelShifts {
		patternFrequency[shift.Type]++
	}
	
	// Detect cyclical patterns
	cyclicalPatterns := e.detectCyclicalPatterns(values, timestamps)
	detectedPatterns = append(detectedPatterns, cyclicalPatterns...)
	for _, cycle := range cyclicalPatterns {
		patternFrequency[cycle.Type]++
	}
	
	// Generate similarity matrix for pattern comparison
	similarityMatrix := e.calculatePatternSimilarity(detectedPatterns)
	
	// Perform motif analysis
	motifAnalysis := e.performMotifAnalysis(values, detectedPatterns)
	
	return &PatternAnalysisResult{
		DetectedPatterns: detectedPatterns,
		PatternFrequency: patternFrequency,
		SimilarityMatrix: similarityMatrix,
		MotifAnalysis:    motifAnalysis,
	}
}

func (e *Engine) analyzeCorrelation(values []float64, timestamps []time.Time) *CorrelationAnalysis {
	if len(values) < 3 {
		return &CorrelationAnalysis{
			AutoCorrelation: []float64{},
			LagAnalysis:     []LagCorrelation{},
		}
	}

	// Calculate autocorrelation function
	maxLag := min(len(values)/4, 50)
	autocorrelation := e.calculateAutocorrelation(values, maxLag)
	
	// Perform lag analysis
	lagAnalysis := make([]LagCorrelation, maxLag)
	for lag := 1; lag <= maxLag; lag++ {
		correlation := e.calculateLagCorrelation(values, lag)
		significance := e.testCorrelationSignificance(correlation, len(values), lag)
		
		lagAnalysis[lag-1] = LagCorrelation{
			Lag:          lag,
			Correlation:  correlation,
			Significance: significance,
			PValue:       e.calculateCorrelationPValue(correlation, len(values)),
		}
	}
	
	// Perform statistical tests
	correlationTests := []CorrelationTest{
		{
			TestName:     "Ljung-Box",
			Statistic:    e.ljungBoxTest(values, 10),
			PValue:       e.ljungBoxPValue(values, 10),
			Conclusion:   "Test for autocorrelation in residuals",
			Significance: "5%",
		},
		{
			TestName:     "Durbin-Watson",
			Statistic:    e.durbinWatsonTest(values),
			PValue:       0.0, // approximation
			Conclusion:   "Test for first-order autocorrelation",
			Significance: "5%",
		},
	}
	
	return &CorrelationAnalysis{
		AutoCorrelation:  autocorrelation,
		LagAnalysis:      lagAnalysis,
		CorrelationTests: correlationTests,
	}
}

func (e *Engine) assessQuality(values []float64, timestamps []time.Time, timeSeries *models.TimeSeries) *QualityAssessment {
	var qualityIssues []QualityIssue
	var recommendations []string
	
	// Assess completeness
	totalExpectedPoints := len(timestamps)
	actualPoints := len(values)
	missingPoints := e.countMissingValues(values)
	completenessScore := float64(actualPoints-missingPoints) / float64(totalExpectedPoints)
	
	if completenessScore < 0.95 {
		qualityIssues = append(qualityIssues, QualityIssue{
			Type:        "completeness",
			Severity:    e.classifyQualitySeverity(completenessScore),
			Description: fmt.Sprintf("%.1f%% data completeness, %d missing values", completenessScore*100, missingPoints),
			AffectedPoints: missingPoints,
		})
		recommendations = append(recommendations, "Consider data imputation for missing values")
	}
	
	// Assess consistency (duplicates, time ordering)
	duplicates := e.countDuplicateTimestamps(timestamps)
	unorderedPoints := e.countUnorderedTimestamps(timestamps)
	consistencyScore := 1.0 - float64(duplicates+unorderedPoints)/float64(len(timestamps))
	
	if consistencyScore < 0.95 {
		qualityIssues = append(qualityIssues, QualityIssue{
			Type:        "consistency",
			Severity:    e.classifyQualitySeverity(consistencyScore),
			Description: fmt.Sprintf("%.1f%% consistency, %d duplicates, %d unordered points", consistencyScore*100, duplicates, unorderedPoints),
			AffectedPoints: duplicates + unorderedPoints,
		})
		recommendations = append(recommendations, "Remove duplicate timestamps and ensure proper time ordering")
	}
	
	// Assess accuracy (outliers, statistical anomalies)
	outliers := e.countOutliers(values, 0, 0, 0) // will be calculated inside
	mean := e.calculateMean(values)
	stdDev := e.calculateStandardDeviation(values)
	q1, q3 := e.calculateQuartiles(values)
	iqr := q3 - q1
	actualOutliers := e.countOutliers(values, q1, q3, iqr)
	
	accuracyScore := 1.0 - float64(actualOutliers)/float64(len(values))
	
	if accuracyScore < 0.90 {
		qualityIssues = append(qualityIssues, QualityIssue{
			Type:        "accuracy",
			Severity:    e.classifyQualitySeverity(accuracyScore),
			Description: fmt.Sprintf("%.1f%% accuracy, %d outliers detected", accuracyScore*100, actualOutliers),
			AffectedPoints: actualOutliers,
		})
		recommendations = append(recommendations, "Review and potentially clean outlier values")
	}
	
	// Assess validity (data range, distribution)
	validityScore := 1.0
	negativeValues := 0
	infiniteValues := 0
	
	for _, val := range values {
		if math.IsInf(val, 0) || math.IsNaN(val) {
			infiniteValues++
		}
		// Add more validity checks based on domain knowledge
	}
	
	if infiniteValues > 0 {
		validityScore = 1.0 - float64(infiniteValues)/float64(len(values))
		qualityIssues = append(qualityIssues, QualityIssue{
			Type:        "validity",
			Severity:    "high",
			Description: fmt.Sprintf("%d infinite or NaN values found", infiniteValues),
			AffectedPoints: infiniteValues,
		})
		recommendations = append(recommendations, "Remove or replace infinite and NaN values")
	}
	
	// Calculate overall score
	overallScore := (completenessScore + consistencyScore + accuracyScore + validityScore) / 4.0
	
	// Add general recommendations based on overall score
	if overallScore < 0.8 {
		recommendations = append(recommendations, "Consider comprehensive data cleaning and validation")
	}
	if len(values) < 100 {
		recommendations = append(recommendations, "Consider collecting more data points for better analysis")
	}
	
	return &QualityAssessment{
		CompletenessScore: completenessScore,
		ConsistencyScore:  consistencyScore,
		AccuracyScore:     accuracyScore,
		ValidityScore:     validityScore,
		OverallScore:      overallScore,
		QualityIssues:     qualityIssues,
		Recommendations:   recommendations,
	}
}

// Cache management methods
func (e *Engine) getCachedResult(request *AnalysisRequest) *AnalysisResult {
	e.cache.mu.RLock()
	defer e.cache.mu.RUnlock()
	
	key := e.generateCacheKey(request)
	entry, exists := e.cache.data[key]
	
	if !exists || time.Now().After(entry.ExpiresAt) {
		return nil
	}
	
	if result, ok := entry.Data.(*AnalysisResult); ok {
		return result
	}
	
	return nil
}

func (e *Engine) cacheResult(request *AnalysisRequest, result *AnalysisResult) {
	e.cache.mu.Lock()
	defer e.cache.mu.Unlock()
	
	key := e.generateCacheKey(request)
	entry := &CacheEntry{
		Data:      result,
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(e.cache.ttl),
	}
	
	e.cache.data[key] = entry
	
	// Simple cache eviction if size exceeded
	if len(e.cache.data) > e.config.MaxCacheSize {
		e.evictOldestEntry()
	}
}

func (e *Engine) generateCacheKey(request *AnalysisRequest) string {
	return fmt.Sprintf("%s_%s_%v", request.SeriesID, 
		strings.Join(request.AnalysisType, ","), request.Parameters)
}

func (e *Engine) evictOldestEntry() {
	var oldestKey string
	var oldestTime time.Time = time.Now()
	
	for key, entry := range e.cache.data {
		if entry.CreatedAt.Before(oldestTime) {
			oldestTime = entry.CreatedAt
			oldestKey = key
		}
	}
	
	if oldestKey != "" {
		delete(e.cache.data, oldestKey)
	}
}

func (e *Engine) loadTimeSeries(ctx context.Context, request *AnalysisRequest) (*models.TimeSeries, error) {
	if e.storage == nil {
		return nil, errors.NewStorageError("NO_STORAGE", "No storage backend configured")
	}
	
	return e.storage.Read(ctx, request.SeriesID)
}

// Public API methods for individual analysis types

// AnalyzeBasicStatistics performs basic statistical analysis on time series data
func (e *Engine) AnalyzeBasicStatistics(ctx context.Context, timeSeries *models.TimeSeries) (*AnalysisResult, error) {
	if len(timeSeries.Points) == 0 {
		return nil, errors.NewValidationError("NO_DATA", "No data points available for analysis")
	}

	values, timestamps := e.extractValuesAndTimestamps(timeSeries)
	
	result := &AnalysisResult{
		SeriesID:       timeSeries.ID,
		AnalysisType:   "basic_stats",
		BasicStats:     e.calculateBasicStatistics(values, timestamps),
		ProcessingTime: 0, // will be set by caller
		Timestamp:      time.Now(),
	}

	return result, nil
}

// AnalyzeTrend performs trend analysis on time series data
func (e *Engine) AnalyzeTrend(ctx context.Context, timeSeries *models.TimeSeries) (*AnalysisResult, error) {
	if len(timeSeries.Points) == 0 {
		return nil, errors.NewValidationError("NO_DATA", "No data points available for analysis")
	}

	values, timestamps := e.extractValuesAndTimestamps(timeSeries)
	
	result := &AnalysisResult{
		SeriesID:       timeSeries.ID,
		AnalysisType:   "trend",
		TrendAnalysis:  e.analyzeTrend(values, timestamps),
		ProcessingTime: 0, // will be set by caller
		Timestamp:      time.Now(),
	}

	return result, nil
}

// AnalyzeSeasonality performs seasonality analysis on time series data
func (e *Engine) AnalyzeSeasonality(ctx context.Context, timeSeries *models.TimeSeries) (*AnalysisResult, error) {
	if len(timeSeries.Points) == 0 {
		return nil, errors.NewValidationError("NO_DATA", "No data points available for analysis")
	}

	values, timestamps := e.extractValuesAndTimestamps(timeSeries)
	
	result := &AnalysisResult{
		SeriesID:        timeSeries.ID,
		AnalysisType:    "seasonality",
		SeasonalityInfo: e.analyzeSeasonality(values, timestamps),
		ProcessingTime:  0, // will be set by caller
		Timestamp:       time.Now(),
	}

	return result, nil
}

// DetectAnomalies performs anomaly detection on time series data
func (e *Engine) DetectAnomalies(ctx context.Context, timeSeries *models.TimeSeries) (*AnalysisResult, error) {
	if len(timeSeries.Points) == 0 {
		return nil, errors.NewValidationError("NO_DATA", "No data points available for analysis")
	}

	values, timestamps := e.extractValuesAndTimestamps(timeSeries)
	
	result := &AnalysisResult{
		SeriesID:         timeSeries.ID,
		AnalysisType:     "anomalies",
		AnomalyDetection: e.detectAnomalies(values, timestamps),
		ProcessingTime:   0, // will be set by caller
		Timestamp:        time.Now(),
	}

	return result, nil
}

// GenerateForecast generates forecasts for time series data
func (e *Engine) GenerateForecast(ctx context.Context, timeSeries *models.TimeSeries, horizon int) (*AnalysisResult, error) {
	if len(timeSeries.Points) == 0 {
		return nil, errors.NewValidationError("NO_DATA", "No data points available for analysis")
	}

	values, timestamps := e.extractValuesAndTimestamps(timeSeries)
	
	result := &AnalysisResult{
		SeriesID:       timeSeries.ID,
		AnalysisType:   "forecast",
		Forecasting:    e.generateForecast(values, timestamps, horizon),
		ProcessingTime: 0, // will be set by caller
		Timestamp:      time.Now(),
	}

	return result, nil
}

// AnalyzePatterns performs pattern analysis on time series data
func (e *Engine) AnalyzePatterns(ctx context.Context, timeSeries *models.TimeSeries) (*AnalysisResult, error) {
	if len(timeSeries.Points) == 0 {
		return nil, errors.NewValidationError("NO_DATA", "No data points available for analysis")
	}

	values, timestamps := e.extractValuesAndTimestamps(timeSeries)
	
	result := &AnalysisResult{
		SeriesID:        timeSeries.ID,
		AnalysisType:    "patterns",
		PatternAnalysis: e.analyzePatterns(values, timestamps),
		ProcessingTime:  0, // will be set by caller
		Timestamp:       time.Now(),
	}

	return result, nil
}

// AssessQuality performs quality assessment on time series data
func (e *Engine) AssessQuality(ctx context.Context, timeSeries *models.TimeSeries) (*AnalysisResult, error) {
	if len(timeSeries.Points) == 0 {
		return nil, errors.NewValidationError("NO_DATA", "No data points available for analysis")
	}

	values, timestamps := e.extractValuesAndTimestamps(timeSeries)
	
	result := &AnalysisResult{
		SeriesID:       timeSeries.ID,
		AnalysisType:   "quality",
		QualityMetrics: e.assessQuality(values, timestamps, timeSeries),
		ProcessingTime: 0, // will be set by caller
		Timestamp:      time.Now(),
	}

	return result, nil
}

// Helper method to extract values and timestamps from TimeSeries
func (e *Engine) extractValuesAndTimestamps(timeSeries *models.TimeSeries) ([]float64, []time.Time) {
	values := make([]float64, len(timeSeries.Points))
	timestamps := make([]time.Time, len(timeSeries.Points))
	
	for i, point := range timeSeries.Points {
		if val, ok := point.Value.(float64); ok {
			values[i] = val
		} else if val, ok := point.Value.(int); ok {
			values[i] = float64(val)
		} else if val, ok := point.Value.(int64); ok {
			values[i] = float64(val)
		} else {
			values[i] = 0.0 // fallback for non-numeric values
		}
		timestamps[i] = point.Timestamp
	}
	
	return values, timestamps
}

// Helper functions for analytics implementations

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (e *Engine) calculateAutocorrelation(values []float64, maxLag int) []float64 {
	n := len(values)
	if maxLag > n-1 {
		maxLag = n - 1
	}
	
	mean := e.calculateMean(values)
	variance := e.calculateVariance(values, mean)
	
	autocorr := make([]float64, maxLag+1)
	autocorr[0] = 1.0 // lag 0 is always 1
	
	for lag := 1; lag <= maxLag; lag++ {
		covariance := 0.0
		count := n - lag
		
		for i := 0; i < count; i++ {
			covariance += (values[i] - mean) * (values[i+lag] - mean)
		}
		
		if variance > 0 {
			autocorr[lag] = (covariance / float64(count)) / variance
		}
	}
	
	return autocorr
}

func (e *Engine) detectSeasonalPeriods(acf []float64, timestamps []time.Time) []SeasonalPeriod {
	var periods []SeasonalPeriod
	threshold := 0.2
	
	// Look for significant peaks in autocorrelation
	for i := 2; i < len(acf); i++ {
		if acf[i] > threshold && acf[i] > acf[i-1] && acf[i] > acf[i+1] {
			periodType := e.classifyPeriodType(i)
			periods = append(periods, SeasonalPeriod{
				Period:   i,
				Strength: acf[i],
				Type:     periodType,
			})
		}
	}
	
	return periods
}

func (e *Engine) classifyPeriodType(period int) string {
	switch {
	case period >= 8760: // yearly (hourly data)
		return "yearly"
	case period >= 168: // weekly (hourly data)
		return "weekly"
	case period >= 24: // daily (hourly data)
		return "daily"
	case period >= 12: // half-daily
		return "half_daily"
	default:
		return "custom"
	}
}

func (e *Engine) calculateSeasonalStrength(values []float64, periods []SeasonalPeriod) float64 {
	if len(periods) == 0 {
		return 0.0
	}
	
	// Use the strongest autocorrelation as seasonal strength
	maxStrength := 0.0
	for _, period := range periods {
		if period.Strength > maxStrength {
			maxStrength = period.Strength
		}
	}
	
	return maxStrength
}

func (e *Engine) performSpectralAnalysis(values []float64) *SpectralAnalysis {
	// Simplified spectral analysis - in practice would use FFT
	return &SpectralAnalysis{
		DominantFrequencies: []float64{},
		PowerSpectrum:      []float64{},
		Method:            "simplified",
	}
}

func (e *Engine) decomposeTimeSeries(values []float64, period int) *SeriesDecomposition {
	n := len(values)
	trend := make([]float64, n)
	seasonal := make([]float64, n)
	residual := make([]float64, n)
	
	// Simple moving average for trend
	for i := 0; i < n; i++ {
		windowStart := max(0, i-period/2)
		windowEnd := min(n-1, i+period/2)
		sum := 0.0
		count := 0
		
		for j := windowStart; j <= windowEnd; j++ {
			sum += values[j]
			count++
		}
		trend[i] = sum / float64(count)
	}
	
	// Calculate seasonal component
	seasonalAvg := make([]float64, period)
	seasonalCount := make([]int, period)
	
	for i := 0; i < n; i++ {
		detrended := values[i] - trend[i]
		seasonalIndex := i % period
		seasonalAvg[seasonalIndex] += detrended
		seasonalCount[seasonalIndex]++
	}
	
	for i := 0; i < period; i++ {
		if seasonalCount[i] > 0 {
			seasonalAvg[i] /= float64(seasonalCount[i])
		}
	}
	
	// Apply seasonal component and calculate residuals
	for i := 0; i < n; i++ {
		seasonal[i] = seasonalAvg[i%period]
		residual[i] = values[i] - trend[i] - seasonal[i]
	}
	
	return &SeriesDecomposition{
		Trend:    trend,
		Seasonal: seasonal,
		Residual: residual,
		Method:   "additive",
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (e *Engine) calculateSeasonalIndices(values []float64, periods []SeasonalPeriod) []float64 {
	if len(periods) == 0 {
		return []float64{}
	}
	
	period := periods[0].Period
	indices := make([]float64, period)
	
	// Calculate average for each position in the cycle
	sums := make([]float64, period)
	counts := make([]int, period)
	
	for i, val := range values {
		pos := i % period
		sums[pos] += val
		counts[pos]++
	}
	
	overallMean := e.calculateMean(values)
	for i := 0; i < period; i++ {
		if counts[i] > 0 {
			cycleMean := sums[i] / float64(counts[i])
			indices[i] = cycleMean / overallMean
		} else {
			indices[i] = 1.0
		}
	}
	
	return indices
}

func (e *Engine) classifyAnomalySeverity(score, threshold float64) string {
	ratio := score / threshold
	switch {
	case ratio > 3:
		return "high"
	case ratio > 2:
		return "medium"
	default:
		return "low"
	}
}

func (e *Engine) detectIQRAnomalies(values []float64, timestamps []time.Time) []Anomaly {
	var anomalies []Anomaly
	
	q1, q3 := e.calculateQuartiles(values)
	iqr := q3 - q1
	lowerBound := q1 - 1.5*iqr
	upperBound := q3 + 1.5*iqr
	
	for i, val := range values {
		if val < lowerBound || val > upperBound {
			anomaly := Anomaly{
				Index:       i,
				Timestamp:   timestamps[i],
				Value:       val,
				Score:       math.Max(math.Abs(val-lowerBound), math.Abs(val-upperBound)),
				Type:        "iqr_outlier",
				Severity:    "medium",
				Description: fmt.Sprintf("Value %.2f outside IQR bounds [%.2f, %.2f]", val, lowerBound, upperBound),
			}
			anomalies = append(anomalies, anomaly)
		}
	}
	
	return anomalies
}

func (e *Engine) calculateQuartiles(values []float64) (float64, float64) {
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	
	n := len(sorted)
	q1 := e.calculatePercentile(sorted, 25.0)
	q3 := e.calculatePercentile(sorted, 75.0)
	
	return q1, q3
}

func (e *Engine) detectMovingAverageAnomalies(values []float64, timestamps []time.Time, window int) []Anomaly {
	var anomalies []Anomaly
	
	if len(values) < window {
		return anomalies
	}
	
	// Calculate moving average and standard deviation
	for i := window; i < len(values); i++ {
		windowValues := values[i-window : i]
		ma := e.calculateMean(windowValues)
		std := e.calculateStandardDeviation(windowValues)
		
		deviation := math.Abs(values[i] - ma)
		if deviation > 2*std {
			anomaly := Anomaly{
				Index:       i,
				Timestamp:   timestamps[i],
				Value:       values[i],
				Score:       deviation / std,
				Type:        "trend_anomaly",
				Severity:    e.classifyAnomalySeverity(deviation/std, 2.0),
				Description: fmt.Sprintf("Trend deviation: %.2f standard deviations from moving average", deviation/std),
			}
			anomalies = append(anomalies, anomaly)
		}
	}
	
	return anomalies
}

func (e *Engine) selectForecastingMethod(values []float64) string {
	// Simple method selection based on data characteristics
	trend := e.analyzeTrend(values, nil)
	
	if trend.RSquared > 0.7 {
		return "linear_trend"
	}
	
	if len(values) > 24 {
		acf := e.calculateAutocorrelation(values, min(len(values)/4, 24))
		maxAcf := 0.0
		for _, a := range acf[1:] {
			if math.Abs(a) > maxAcf {
				maxAcf = math.Abs(a)
			}
		}
		
		if maxAcf > 0.3 {
			return "seasonal_naive"
		}
	}
	
	return "exponential_smoothing"
}

func (e *Engine) linearTrendForecast(values []float64, timestamps []time.Time, horizon int) ([]ForecastPoint, *AccuracyMetrics) {
	slope, intercept, _ := e.linearRegression(values)
	
	predictions := make([]ForecastPoint, horizon)
	n := len(values)
	
	for i := 0; i < horizon; i++ {
		prediction := intercept + slope*float64(n+i)
		predictions[i] = ForecastPoint{
			Index:     n + i,
			Value:     prediction,
			Timestamp: timestamps[len(timestamps)-1].Add(time.Duration(i+1) * time.Hour), // assuming hourly data
		}
	}
	
	// Calculate accuracy on in-sample predictions
	accuracy := e.calculateForecastAccuracy(values, predictions[:min(len(predictions), len(values))])
	
	return predictions, accuracy
}

func (e *Engine) exponentialSmoothingForecast(values []float64, timestamps []time.Time, horizon int) ([]ForecastPoint, *AccuracyMetrics) {
	alpha := 0.3 // smoothing parameter
	
	// Initialize with first value
	smoothed := values[0]
	predictions := make([]ForecastPoint, horizon)
	
	// Apply exponential smoothing
	for i := 1; i < len(values); i++ {
		smoothed = alpha*values[i] + (1-alpha)*smoothed
	}
	
	// Generate forecasts
	for i := 0; i < horizon; i++ {
		predictions[i] = ForecastPoint{
			Index:     len(values) + i,
			Value:     smoothed,
			Timestamp: timestamps[len(timestamps)-1].Add(time.Duration(i+1) * time.Hour),
		}
	}
	
	accuracy := e.calculateForecastAccuracy(values, predictions[:min(len(predictions), len(values))])
	
	return predictions, accuracy
}

func (e *Engine) seasonalNaiveForecast(values []float64, timestamps []time.Time, horizon int) ([]ForecastPoint, *AccuracyMetrics) {
	// Use seasonal pattern from previous year/cycle
	seasonLength := 24 // assuming daily seasonality with hourly data
	if len(values) < seasonLength {
		seasonLength = len(values)
	}
	
	predictions := make([]ForecastPoint, horizon)
	
	for i := 0; i < horizon; i++ {
		seasonIndex := (len(values) + i) % seasonLength
		if seasonIndex < len(values) {
			predictions[i] = ForecastPoint{
				Index:     len(values) + i,
				Value:     values[len(values)-seasonLength+seasonIndex],
				Timestamp: timestamps[len(timestamps)-1].Add(time.Duration(i+1) * time.Hour),
			}
		} else {
			predictions[i] = ForecastPoint{
				Index:     len(values) + i,
				Value:     values[seasonIndex],
				Timestamp: timestamps[len(timestamps)-1].Add(time.Duration(i+1) * time.Hour),
			}
		}
	}
	
	accuracy := e.calculateForecastAccuracy(values, predictions[:min(len(predictions), len(values))])
	
	return predictions, accuracy
}

func (e *Engine) calculateForecastAccuracy(actual []float64, forecasts []ForecastPoint) *AccuracyMetrics {
	if len(actual) == 0 || len(forecasts) == 0 {
		return &AccuracyMetrics{}
	}
	
	n := min(len(actual), len(forecasts))
	mae, mse, mape := 0.0, 0.0, 0.0
	
	for i := 0; i < n; i++ {
		error := actual[i] - forecasts[i].Value
		mae += math.Abs(error)
		mse += error * error
		if actual[i] != 0 {
			mape += math.Abs(error / actual[i])
		}
	}
	
	mae /= float64(n)
	mse /= float64(n)
	mape = (mape / float64(n)) * 100
	rmse := math.Sqrt(mse)
	
	return &AccuracyMetrics{
		MAE:  mae,
		MSE:  mse,
		RMSE: rmse,
		MAPE: mape,
	}
}

func (e *Engine) calculateConfidenceIntervals(predictions []ForecastPoint, values []float64) []ConfidenceInterval {
	intervals := make([]ConfidenceInterval, len(predictions))
	
	// Simple confidence interval based on historical variance
	variance := e.calculateVariance(values, e.calculateMean(values))
	stdError := math.Sqrt(variance)
	
	for i, pred := range predictions {
		// 95% confidence interval
		margin := 1.96 * stdError * math.Sqrt(float64(i+1)) // increasing uncertainty over time
		intervals[i] = ConfidenceInterval{
			Lower:      pred.Value - margin,
			Upper:      pred.Value + margin,
			Level:      0.95,
			Prediction: pred.Value,
		}
	}
	
	return intervals
}

func (e *Engine) generateModelDiagnostics(values []float64, predictions []ForecastPoint) *ModelDiagnostics {
	// Calculate residuals and diagnostics
	residuals := make([]float64, min(len(values), len(predictions)))
	for i := 0; i < len(residuals); i++ {
		residuals[i] = values[i] - predictions[i].Value
	}
	
	return &ModelDiagnostics{
		Residuals:        residuals,
		ResidualsVariance: e.calculateVariance(residuals, e.calculateMean(residuals)),
		Normality:        e.testNormality(residuals),
		Autocorrelation:  e.calculateAutocorrelation(residuals, min(len(residuals)/4, 10)),
	}
}

func (e *Engine) testNormality(values []float64) string {
	// Simplified normality test - in practice would use Shapiro-Wilk or Jarque-Bera
	skewness := e.calculateSkewness(values, e.calculateMean(values), e.calculateStandardDeviation(values))
	
	if math.Abs(skewness) < 0.5 {
		return "normal"
	} else if math.Abs(skewness) < 1.0 {
		return "moderately_skewed"
	}
	return "highly_skewed"
}

// Pattern detection helper functions

func (e *Engine) detectTrendPatterns(values []float64, timestamps []time.Time) []Pattern {
	var patterns []Pattern
	
	// Detect trend changes
	trendAnalysis := e.analyzeTrend(values, timestamps)
	
	if trendAnalysis.RSquared > 0.5 {
		pattern := Pattern{
			Type:        "trend",
			StartIndex:  0,
			EndIndex:    len(values) - 1,
			Strength:    trendAnalysis.RSquared,
			Description: fmt.Sprintf("%s trend with R² = %.3f", trendAnalysis.Direction, trendAnalysis.RSquared),
			Parameters: map[string]interface{}{
				"slope":     trendAnalysis.Slope,
				"intercept": trendAnalysis.Intercept,
				"direction": trendAnalysis.Direction,
			},
		}
		patterns = append(patterns, pattern)
	}
	
	return patterns
}

func (e *Engine) detectMotifs(values []float64, timestamps []time.Time, windowSize int) []Pattern {
	var patterns []Pattern
	
	if len(values) < windowSize*2 {
		return patterns
	}
	
	// Simple motif detection - find repeating subsequences
	threshold := 0.1
	
	for i := 0; i <= len(values)-windowSize*2; i++ {
		window1 := values[i : i+windowSize]
		
		for j := i + windowSize; j <= len(values)-windowSize; j++ {
			window2 := values[j : j+windowSize]
			
			similarity := e.calculateSequenceSimilarity(window1, window2)
			if similarity > (1.0 - threshold) {
				pattern := Pattern{
					Type:        "motif",
					StartIndex:  i,
					EndIndex:    j + windowSize - 1,
					Strength:    similarity,
					Description: fmt.Sprintf("Repeated motif with %.1f%% similarity", similarity*100),
					Parameters: map[string]interface{}{
						"window_size":   windowSize,
						"first_start":   i,
						"second_start":  j,
						"similarity":    similarity,
					},
				}
				patterns = append(patterns, pattern)
			}
		}
	}
	
	return patterns
}

func (e *Engine) calculateSequenceSimilarity(seq1, seq2 []float64) float64 {
	if len(seq1) != len(seq2) {
		return 0.0
	}
	
	// Normalized Euclidean distance
	sumSquaredDiff := 0.0
	sumSquared1 := 0.0
	sumSquared2 := 0.0
	
	for i := 0; i < len(seq1); i++ {
		diff := seq1[i] - seq2[i]
		sumSquaredDiff += diff * diff
		sumSquared1 += seq1[i] * seq1[i]
		sumSquared2 += seq2[i] * seq2[i]
	}
	
	if sumSquared1 == 0 || sumSquared2 == 0 {
		return 0.0
	}
	
	normalizedDistance := sumSquaredDiff / (math.Sqrt(sumSquared1) * math.Sqrt(sumSquared2))
	similarity := 1.0 / (1.0 + normalizedDistance)
	
	return similarity
}

func (e *Engine) detectLevelShifts(values []float64, timestamps []time.Time) []Pattern {
	var patterns []Pattern
	
	if len(values) < 10 {
		return patterns
	}
	
	// Detect significant level changes using change point detection
	windowSize := min(len(values)/4, 20)
	threshold := 2.0 // threshold in standard deviations
	
	for i := windowSize; i < len(values)-windowSize; i++ {
		beforeWindow := values[i-windowSize : i]
		afterWindow := values[i : i+windowSize]
		
		beforeMean := e.calculateMean(beforeWindow)
		afterMean := e.calculateMean(afterWindow)
		
		pooledStd := math.Sqrt((e.calculateVariance(beforeWindow, beforeMean) + e.calculateVariance(afterWindow, afterMean)) / 2)
		
		if pooledStd > 0 {
			tStat := math.Abs(beforeMean-afterMean) / (pooledStd * math.Sqrt(2.0/float64(windowSize)))
			
			if tStat > threshold {
				pattern := Pattern{
					Type:        "level_shift",
					StartIndex:  i - windowSize,
					EndIndex:    i + windowSize,
					Strength:    tStat / threshold,
					Description: fmt.Sprintf("Level shift detected at index %d, change = %.3f", i, afterMean-beforeMean),
					Parameters: map[string]interface{}{
						"change_point":  i,
						"before_level":  beforeMean,
						"after_level":   afterMean,
						"change_amount": afterMean - beforeMean,
						"t_statistic":   tStat,
					},
				}
				patterns = append(patterns, pattern)
			}
		}
	}
	
	return patterns
}

func (e *Engine) detectCyclicalPatterns(values []float64, timestamps []time.Time) []Pattern {
	var patterns []Pattern
	
	// Use autocorrelation to detect cyclical patterns
	acf := e.calculateAutocorrelation(values, min(len(values)/3, 100))
	
	threshold := 0.3
	for lag := 5; lag < len(acf); lag++ {
		if acf[lag] > threshold {
			pattern := Pattern{
				Type:        "cyclical",
				StartIndex:  0,
				EndIndex:    len(values) - 1,
				Strength:    acf[lag],
				Description: fmt.Sprintf("Cyclical pattern with period %d, correlation %.3f", lag, acf[lag]),
				Parameters: map[string]interface{}{
					"period":      lag,
					"correlation": acf[lag],
				},
			}
			patterns = append(patterns, pattern)
		}
	}
	
	return patterns
}

func (e *Engine) calculatePatternSimilarity(patterns []Pattern) [][]float64 {
	n := len(patterns)
	if n == 0 {
		return [][]float64{}
	}
	
	similarity := make([][]float64, n)
	for i := range similarity {
		similarity[i] = make([]float64, n)
		similarity[i][i] = 1.0 // self-similarity
	}
	
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			sim := e.calculatePatternPairSimilarity(patterns[i], patterns[j])
			similarity[i][j] = sim
			similarity[j][i] = sim
		}
	}
	
	return similarity
}

func (e *Engine) calculatePatternPairSimilarity(p1, p2 Pattern) float64 {
	// Simple similarity based on type and overlap
	if p1.Type != p2.Type {
		return 0.0
	}
	
	// Calculate overlap
	start := max(p1.StartIndex, p2.StartIndex)
	end := min(p1.EndIndex, p2.EndIndex)
	
	if start >= end {
		return 0.0
	}
	
	overlap := float64(end - start)
	length1 := float64(p1.EndIndex - p1.StartIndex)
	length2 := float64(p2.EndIndex - p2.StartIndex)
	
	overlapRatio := overlap / math.Max(length1, length2)
	strengthSimilarity := 1.0 - math.Abs(p1.Strength-p2.Strength)
	
	return (overlapRatio + strengthSimilarity) / 2.0
}

func (e *Engine) performMotifAnalysis(values []float64, patterns []Pattern) *MotifAnalysis {
	motifCount := 0
	avgLength := 0.0
	
	for _, pattern := range patterns {
		if pattern.Type == "motif" {
			motifCount++
			avgLength += float64(pattern.EndIndex - pattern.StartIndex + 1)
		}
	}
	
	if motifCount > 0 {
		avgLength /= float64(motifCount)
	}
	
	return &MotifAnalysis{
		MotifCount:       motifCount,
		AverageLength:    avgLength,
		MaxSimilarity:    e.findMaxPatternSimilarity(patterns),
		DiscoveryMethod:  "sliding_window",
	}
}

func (e *Engine) findMaxPatternSimilarity(patterns []Pattern) float64 {
	maxSim := 0.0
	
	for i := 0; i < len(patterns); i++ {
		for j := i + 1; j < len(patterns); j++ {
			sim := e.calculatePatternPairSimilarity(patterns[i], patterns[j])
			if sim > maxSim {
				maxSim = sim
			}
		}
	}
	
	return maxSim
}

// Correlation analysis helper functions

func (e *Engine) calculateLagCorrelation(values []float64, lag int) float64 {
	if lag >= len(values) {
		return 0.0
	}
	
	n := len(values) - lag
	mean1 := e.calculateMean(values[:n])
	mean2 := e.calculateMean(values[lag:])
	
	numerator := 0.0
	var1 := 0.0
	var2 := 0.0
	
	for i := 0; i < n; i++ {
		diff1 := values[i] - mean1
		diff2 := values[i+lag] - mean2
		
		numerator += diff1 * diff2
		var1 += diff1 * diff1
		var2 += diff2 * diff2
	}
	
	if var1 == 0 || var2 == 0 {
		return 0.0
	}
	
	return numerator / math.Sqrt(var1*var2)
}

func (e *Engine) testCorrelationSignificance(correlation float64, n, lag int) string {
	// Simple significance test using critical value
	criticalValue := 1.96 / math.Sqrt(float64(n-lag))
	
	if math.Abs(correlation) > criticalValue {
		return "significant"
	}
	return "not_significant"
}

func (e *Engine) calculateCorrelationPValue(correlation float64, n int) float64 {
	// Simplified p-value calculation
	if n < 3 {
		return 1.0
	}
	
	t := correlation * math.Sqrt(float64(n-2)/(1-correlation*correlation))
	// This is a simplified approximation
	return 2.0 * (1.0 - math.Abs(t)/3.0) // rough approximation
}

func (e *Engine) ljungBoxTest(values []float64, h int) float64 {
	n := len(values)
	if n <= h {
		return 0.0
	}
	
	// Calculate autocorrelations
	acf := e.calculateAutocorrelation(values, h)
	
	// Ljung-Box statistic
	stat := 0.0
	for k := 1; k <= h; k++ {
		stat += acf[k] * acf[k] / float64(n-k)
	}
	
	return float64(n) * float64(n+2) * stat
}

func (e *Engine) ljungBoxPValue(values []float64, h int) float64 {
	// Simplified p-value - in practice would use chi-square distribution
	stat := e.ljungBoxTest(values, h)
	// Rough approximation for chi-square with h degrees of freedom
	return math.Max(0.0, 1.0-stat/float64(h*3))
}

func (e *Engine) durbinWatsonTest(values []float64) float64 {
	if len(values) < 2 {
		return 2.0
	}
	
	numerator := 0.0
	denominator := 0.0
	
	for i := 1; i < len(values); i++ {
		diff := values[i] - values[i-1]
		numerator += diff * diff
		denominator += values[i] * values[i]
	}
	
	if denominator == 0 {
		return 2.0
	}
	
	return numerator / denominator
}

// Quality assessment helper functions

func (e *Engine) classifyQualitySeverity(score float64) string {
	switch {
	case score >= 0.9:
		return "low"
	case score >= 0.7:
		return "medium"
	default:
		return "high"
	}
}

func (e *Engine) countDuplicateTimestamps(timestamps []time.Time) int {
	seen := make(map[time.Time]bool)
	duplicates := 0
	
	for _, ts := range timestamps {
		if seen[ts] {
			duplicates++
		} else {
			seen[ts] = true
		}
	}
	
	return duplicates
}

func (e *Engine) countUnorderedTimestamps(timestamps []time.Time) int {
	unordered := 0
	
	for i := 1; i < len(timestamps); i++ {
		if timestamps[i].Before(timestamps[i-1]) {
			unordered++
		}
	}
	
	return unordered
}