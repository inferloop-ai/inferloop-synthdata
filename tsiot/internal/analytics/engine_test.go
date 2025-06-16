package analytics

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inferloop/tsiot/pkg/models"
)

func TestNewEngine(t *testing.T) {
	logger := logrus.New()
	engine := NewEngine(nil, logger)
	
	assert.NotNil(t, engine)
	assert.Equal(t, logger, engine.logger)
}

func TestAnalyzeBasicStatistics(t *testing.T) {
	engine := NewEngine(nil, logrus.New())
	ctx := context.Background()

	// Create test time series
	timeSeries := createTestTimeSeries()

	result, err := engine.AnalyzeBasicStatistics(ctx, timeSeries)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.BasicStats)

	stats := result.BasicStats
	assert.Equal(t, int64(5), stats.Count)
	assert.Equal(t, 30.0, stats.Mean)
	assert.Equal(t, 10.0, stats.Min)
	assert.Equal(t, 50.0, stats.Max)
	assert.InDelta(t, 15.811, stats.StandardDev, 0.01)
	assert.Equal(t, 30.0, stats.Median)
}

func TestAnalyzeTrend(t *testing.T) {
	engine := NewEngine(nil, logrus.New())
	ctx := context.Background()

	// Create ascending trend
	timeSeries := createTrendTimeSeries()

	result, err := engine.AnalyzeTrend(ctx, timeSeries)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.TrendAnalysis)

	trend := result.TrendAnalysis
	assert.Equal(t, "upward", trend.Direction)
	assert.Greater(t, trend.Strength, 0.8)
	assert.Greater(t, trend.Slope, 0.0)
}

func TestAnalyzeSeasonality(t *testing.T) {
	engine := NewEngine(nil, logrus.New())
	ctx := context.Background()

	// Create seasonal time series
	timeSeries := createSeasonalTimeSeries()

	result, err := engine.AnalyzeSeasonality(ctx, timeSeries)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.SeasonalityInfo)

	seasonality := result.SeasonalityInfo
	assert.True(t, seasonality.HasSeasonality)
	assert.Greater(t, seasonality.Strength, 0.5)
	assert.Contains(t, seasonality.DominantPeriods, 24)
}

func TestDetectAnomalies(t *testing.T) {
	engine := NewEngine(nil, logrus.New())
	ctx := context.Background()

	// Create time series with anomalies
	timeSeries := createAnomalousTimeSeries()

	result, err := engine.DetectAnomalies(ctx, timeSeries)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.AnomalyDetection)

	anomalies := result.AnomalyDetection
	assert.Greater(t, len(anomalies.Anomalies), 0)
	assert.Equal(t, "statistical", anomalies.Method)
	
	// Check that the anomaly at index 25 (value 100) is detected
	foundAnomaly := false
	for _, anomaly := range anomalies.Anomalies {
		if anomaly.Index == 25 {
			foundAnomaly = true
			assert.Greater(t, anomaly.Severity, 2.0)
			break
		}
	}
	assert.True(t, foundAnomaly, "Expected anomaly at index 25 not found")
}

func TestGenerateForecast(t *testing.T) {
	engine := NewEngine(nil, logrus.New())
	ctx := context.Background()

	// Create predictable time series
	timeSeries := createTrendTimeSeries()
	periods := 10

	result, err := engine.GenerateForecast(ctx, timeSeries, periods)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.Forecasting)

	forecast := result.Forecasting
	assert.Equal(t, periods, len(forecast.Predictions))
	assert.Equal(t, "linear_trend", forecast.Method)
	assert.Greater(t, forecast.ConfidenceLevel, 0.8)
	
	// Check that forecast points have increasing values (due to upward trend)
	for i := 1; i < len(forecast.Predictions); i++ {
		assert.Greater(t, forecast.Predictions[i].Value, forecast.Predictions[i-1].Value)
	}
}

func TestAnalyzePatterns(t *testing.T) {
	engine := NewEngine(nil, logrus.New())
	ctx := context.Background()

	// Create time series with patterns
	timeSeries := createPatternedTimeSeries()

	result, err := engine.AnalyzePatterns(ctx, timeSeries)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.PatternAnalysis)

	patterns := result.PatternAnalysis
	assert.Greater(t, len(patterns.Patterns), 0)
	
	// Check for repeating pattern detection
	foundPattern := false
	for _, pattern := range patterns.Patterns {
		if pattern.Type == "repeating" {
			foundPattern = true
			assert.Greater(t, pattern.Confidence, 0.5)
			break
		}
	}
	assert.True(t, foundPattern, "Expected repeating pattern not found")
}

func TestAnalyzeCorrelation(t *testing.T) {
	engine := NewEngine(nil, logrus.New())
	ctx := context.Background()

	// Create two correlated time series
	series1 := createTestTimeSeries()
	series2 := createCorrelatedTimeSeries()

	result, err := engine.AnalyzeCorrelation(ctx, []*models.TimeSeries{series1, series2})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.CorrelationInfo)

	correlation := result.CorrelationInfo
	assert.Greater(t, len(correlation.Correlations), 0)
	
	// Check for high correlation
	foundHighCorrelation := false
	for _, corr := range correlation.Correlations {
		if corr.Coefficient > 0.8 {
			foundHighCorrelation = true
			break
		}
	}
	assert.True(t, foundHighCorrelation, "Expected high correlation not found")
}

func TestAssessQuality(t *testing.T) {
	engine := NewEngine(nil, logrus.New())
	ctx := context.Background()

	// Create time series with quality issues
	timeSeries := createQualityIssueTimeSeries()

	result, err := engine.AssessQuality(ctx, timeSeries)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.QualityMetrics)

	quality := result.QualityMetrics
	assert.Less(t, quality.OverallScore, 1.0)
	assert.Greater(t, len(quality.Issues), 0)
	
	// Check for missing values issue
	foundMissingValues := false
	for _, issue := range quality.Issues {
		if issue.Type == "missing_values" {
			foundMissingValues = true
			assert.Greater(t, issue.Severity, 0.0)
			break
		}
	}
	assert.True(t, foundMissingValues, "Expected missing values issue not found")
}

// Helper functions to create test data

func createTestTimeSeries() *models.TimeSeries {
	now := time.Now()
	dataPoints := []models.DataPoint{
		{Timestamp: now.Add(-4 * time.Hour), Value: 10.0, Quality: 1.0},
		{Timestamp: now.Add(-3 * time.Hour), Value: 20.0, Quality: 1.0},
		{Timestamp: now.Add(-2 * time.Hour), Value: 30.0, Quality: 1.0},
		{Timestamp: now.Add(-1 * time.Hour), Value: 40.0, Quality: 1.0},
		{Timestamp: now, Value: 50.0, Quality: 1.0},
	}

	return &models.TimeSeries{
		ID:          "test-series-1",
		Name:        "Test Series",
		Description: "Test time series for unit tests",
		SensorType:  "temperature",
		DataPoints:  dataPoints,
		StartTime:   dataPoints[0].Timestamp,
		EndTime:     dataPoints[len(dataPoints)-1].Timestamp,
		Frequency:   "1h",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
}

func createTrendTimeSeries() *models.TimeSeries {
	now := time.Now()
	var dataPoints []models.DataPoint

	// Create clear upward trend
	for i := 0; i < 50; i++ {
		dataPoints = append(dataPoints, models.DataPoint{
			Timestamp: now.Add(time.Duration(i) * time.Hour),
			Value:     float64(i*2) + 10.0, // Linear increase
			Quality:   1.0,
		})
	}

	return &models.TimeSeries{
		ID:          "trend-series",
		Name:        "Trend Series",
		Description: "Time series with clear upward trend",
		SensorType:  "temperature",
		DataPoints:  dataPoints,
		StartTime:   dataPoints[0].Timestamp,
		EndTime:     dataPoints[len(dataPoints)-1].Timestamp,
		Frequency:   "1h",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
}

func createSeasonalTimeSeries() *models.TimeSeries {
	now := time.Now()
	var dataPoints []models.DataPoint

	// Create seasonal pattern (24-hour cycle)
	for i := 0; i < 72; i++ { // 3 days of hourly data
		hour := i % 24
		// Temperature pattern: lower at night, higher during day
		seasonal := 20 + 10*math.Sin(2*math.Pi*float64(hour)/24.0)
		
		dataPoints = append(dataPoints, models.DataPoint{
			Timestamp: now.Add(time.Duration(i) * time.Hour),
			Value:     seasonal,
			Quality:   1.0,
		})
	}

	return &models.TimeSeries{
		ID:          "seasonal-series",
		Name:        "Seasonal Series",
		Description: "Time series with 24-hour seasonality",
		SensorType:  "temperature",
		DataPoints:  dataPoints,
		StartTime:   dataPoints[0].Timestamp,
		EndTime:     dataPoints[len(dataPoints)-1].Timestamp,
		Frequency:   "1h",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
}

func createAnomalousTimeSeries() *models.TimeSeries {
	now := time.Now()
	var dataPoints []models.DataPoint

	// Create normal data with one clear anomaly
	for i := 0; i < 50; i++ {
		value := 20.0 + math.Sin(float64(i)*0.1)*5.0 // Normal oscillation
		
		// Insert anomaly at index 25
		if i == 25 {
			value = 100.0 // Clear outlier
		}

		dataPoints = append(dataPoints, models.DataPoint{
			Timestamp: now.Add(time.Duration(i) * time.Hour),
			Value:     value,
			Quality:   1.0,
		})
	}

	return &models.TimeSeries{
		ID:          "anomalous-series",
		Name:        "Anomalous Series",
		Description: "Time series with injected anomalies",
		SensorType:  "temperature",
		DataPoints:  dataPoints,
		StartTime:   dataPoints[0].Timestamp,
		EndTime:     dataPoints[len(dataPoints)-1].Timestamp,
		Frequency:   "1h",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
}

func createPatternedTimeSeries() *models.TimeSeries {
	now := time.Now()
	var dataPoints []models.DataPoint

	// Create repeating pattern every 10 points
	pattern := []float64{10, 15, 20, 25, 30, 25, 20, 15, 10, 5}
	
	for i := 0; i < 50; i++ {
		value := pattern[i%len(pattern)]
		
		dataPoints = append(dataPoints, models.DataPoint{
			Timestamp: now.Add(time.Duration(i) * time.Hour),
			Value:     value,
			Quality:   1.0,
		})
	}

	return &models.TimeSeries{
		ID:          "patterned-series",
		Name:        "Patterned Series",
		Description: "Time series with repeating patterns",
		SensorType:  "temperature",
		DataPoints:  dataPoints,
		StartTime:   dataPoints[0].Timestamp,
		EndTime:     dataPoints[len(dataPoints)-1].Timestamp,
		Frequency:   "1h",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
}

func createCorrelatedTimeSeries() *models.TimeSeries {
	now := time.Now()
	var dataPoints []models.DataPoint

	// Create series that correlates with the original test series
	baseValues := []float64{10.0, 20.0, 30.0, 40.0, 50.0}
	
	for i := 0; i < len(baseValues); i++ {
		// Highly correlated: same pattern with slight offset
		value := baseValues[i] + 5.0
		
		dataPoints = append(dataPoints, models.DataPoint{
			Timestamp: now.Add(time.Duration(-4+i) * time.Hour),
			Value:     value,
			Quality:   1.0,
		})
	}

	return &models.TimeSeries{
		ID:          "correlated-series",
		Name:        "Correlated Series",
		Description: "Time series correlated with test series",
		SensorType:  "temperature",
		DataPoints:  dataPoints,
		StartTime:   dataPoints[0].Timestamp,
		EndTime:     dataPoints[len(dataPoints)-1].Timestamp,
		Frequency:   "1h",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
}

func createQualityIssueTimeSeries() *models.TimeSeries {
	now := time.Now()
	var dataPoints []models.DataPoint

	// Create series with various quality issues
	for i := 0; i < 20; i++ {
		var quality float64 = 1.0
		value := float64(i * 2)
		
		// Introduce missing values (represented as NaN)
		if i%7 == 0 {
			value = math.NaN()
			quality = 0.0
		}
		
		// Introduce low quality points
		if i%5 == 0 {
			quality = 0.3
		}

		dataPoints = append(dataPoints, models.DataPoint{
			Timestamp: now.Add(time.Duration(i) * time.Hour),
			Value:     value,
			Quality:   quality,
		})
	}

	return &models.TimeSeries{
		ID:          "quality-issue-series",
		Name:        "Quality Issue Series",
		Description: "Time series with data quality issues",
		SensorType:  "temperature",
		DataPoints:  dataPoints,
		StartTime:   dataPoints[0].Timestamp,
		EndTime:     dataPoints[len(dataPoints)-1].Timestamp,
		Frequency:   "1h",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
}

// Benchmark tests
func BenchmarkAnalyzeBasicStatistics(b *testing.B) {
	engine := NewEngine(nil, logrus.New())
	ctx := context.Background()
	timeSeries := createLargeTimeSeries(10000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := engine.AnalyzeBasicStatistics(ctx, timeSeries)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDetectAnomalies(b *testing.B) {
	engine := NewEngine(nil, logrus.New())
	ctx := context.Background()
	timeSeries := createLargeTimeSeries(5000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := engine.DetectAnomalies(ctx, timeSeries)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func createLargeTimeSeries(size int) *models.TimeSeries {
	now := time.Now()
	var dataPoints []models.DataPoint

	for i := 0; i < size; i++ {
		value := 20.0 + math.Sin(float64(i)*0.01)*10.0 + math.Sin(float64(i)*0.1)*3.0
		
		dataPoints = append(dataPoints, models.DataPoint{
			Timestamp: now.Add(time.Duration(i) * time.Minute),
			Value:     value,
			Quality:   1.0,
		})
	}

	return &models.TimeSeries{
		ID:          "large-series",
		Name:        "Large Series",
		Description: "Large time series for benchmarking",
		SensorType:  "temperature",
		DataPoints:  dataPoints,
		StartTime:   dataPoints[0].Timestamp,
		EndTime:     dataPoints[len(dataPoints)-1].Timestamp,
		Frequency:   "1m",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
}