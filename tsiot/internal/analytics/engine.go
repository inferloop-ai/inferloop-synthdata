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
	Frequencies     []float64 `json:"frequencies"`
	PowerSpectrum   []float64 `json:"power_spectrum"`
	DominantFreqs   []float64 `json:"dominant_frequencies"`
	SpectralPeaks   []Peak    `json:"spectral_peaks"`
}

type Peak struct {
	Frequency float64 `json:"frequency"`
	Power     float64 `json:"power"`
	Period    float64 `json:"period"`
}

type Anomaly struct {
	Timestamp   time.Time `json:"timestamp"`
	Value       float64   `json:"value"`
	ExpectedValue float64 `json:"expected_value"`
	Deviation   float64   `json:"deviation"`
	Severity    string    `json:"severity"` // "low", "medium", "high", "critical"
	Type        string    `json:"type"`     // "spike", "drop", "shift", "drift"
	Confidence  float64   `json:"confidence"`
}

type ForecastPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Value     float64   `json:"value"`
	Lower     float64   `json:"lower_bound"`
	Upper     float64   `json:"upper_bound"`
}

type ConfidenceInterval struct {
	Level float64 `json:"level"`
	Lower float64 `json:"lower"`
	Upper float64 `json:"upper"`
}

type AccuracyMetrics struct {
	MAE   float64 `json:"mae"`   // Mean Absolute Error
	MAPE  float64 `json:"mape"`  // Mean Absolute Percentage Error
	RMSE  float64 `json:"rmse"`  // Root Mean Square Error
	SMAPE float64 `json:"smape"` // Symmetric Mean Absolute Percentage Error
}

type ModelDiagnostics struct {
	Residuals         []float64 `json:"residuals"`
	ResidualStats     BasicStatistics `json:"residual_stats"`
	LjungBoxTest      *StatisticalTest `json:"ljung_box_test"`
	AutocorrelationTest *StatisticalTest `json:"autocorrelation_test"`
}

type Pattern struct {
	Type        string    `json:"type"`
	StartTime   time.Time `json:"start_time"`
	EndTime     time.Time `json:"end_time"`
	Duration    string    `json:"duration"`
	Confidence  float64   `json:"confidence"`
	Parameters  map[string]interface{} `json:"parameters"`
}

type MotifAnalysis struct {
	Motifs          []Motif   `json:"motifs"`
	DiscordAnalysis []Discord `json:"discord_analysis"`
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
	Lag         int     `json:"lag"`
	Correlation float64 `json:"correlation"`
	Significant bool    `json:"significant"`
}

type CorrelationTest struct {
	TestName   string  `json:"test_name"`
	Statistic  float64 `json:"statistic"`
	PValue     float64 `json:"p_value"`
	Critical   float64 `json:"critical_value"`
	Significant bool   `json:"significant"`
}

type QualityIssue struct {
	Type        string    `json:"type"`
	Severity    string    `json:"severity"`
	Description string    `json:"description"`
	Location    *Location `json:"location,omitempty"`
	Count       int       `json:"count"`
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

// Additional stub implementations for other analysis methods
func (e *Engine) analyzeSeasonality(values []float64, timestamps []time.Time) *SeasonalityAnalysis {
	// Implement seasonality analysis
	return &SeasonalityAnalysis{
		HasSeasonality:   false,
		SeasonalStrength: 0.0,
		DominantPeriods:  []SeasonalPeriod{},
	}
}

func (e *Engine) detectAnomalies(values []float64, timestamps []time.Time) *AnomalyDetectionResult {
	// Implement anomaly detection
	return &AnomalyDetectionResult{
		Method:         "statistical_zscore",
		AnomaliesFound: []Anomaly{},
		AnomalyRate:    0.0,
	}
}

func (e *Engine) generateForecast(values []float64, timestamps []time.Time, horizon int) *ForecastingResult {
	// Implement forecasting
	return &ForecastingResult{
		Method:          "linear_trend",
		ForecastHorizon: horizon,
		PredictedValues: []ForecastPoint{},
	}
}

func (e *Engine) analyzePatterns(values []float64, timestamps []time.Time) *PatternAnalysisResult {
	// Implement pattern analysis
	return &PatternAnalysisResult{
		DetectedPatterns: []Pattern{},
		PatternFrequency: make(map[string]int),
	}
}

func (e *Engine) analyzeCorrelation(values []float64, timestamps []time.Time) *CorrelationAnalysis {
	// Implement correlation analysis
	return &CorrelationAnalysis{
		AutoCorrelation: []float64{},
		LagAnalysis:     []LagCorrelation{},
	}
}

func (e *Engine) assessQuality(values []float64, timestamps []time.Time, timeSeries *models.TimeSeries) *QualityAssessment {
	// Implement quality assessment
	return &QualityAssessment{
		CompletenessScore: 1.0,
		ConsistencyScore:  1.0,
		AccuracyScore:     1.0,
		ValidityScore:     1.0,
		OverallScore:      1.0,
		QualityIssues:     []QualityIssue{},
		Recommendations:   []string{},
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