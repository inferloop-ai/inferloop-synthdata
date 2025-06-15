package helpers

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TimeSeriesPoint represents a single time series data point
type TimeSeriesPoint struct {
	Timestamp time.Time
	Value     float64
	Tags      map[string]string
}

// TimeSeries represents a collection of time series points
type TimeSeries []TimeSeriesPoint

// AssertTimeSeries provides assertions for time series data
func AssertTimeSeries(t *testing.T, expected, actual TimeSeries, tolerance float64) {
	t.Helper()
	
	require.Equal(t, len(expected), len(actual), "time series length mismatch")
	
	for i, expectedPoint := range expected {
		actualPoint := actual[i]
		
		// Assert timestamp
		assert.True(t, expectedPoint.Timestamp.Equal(actualPoint.Timestamp),
			"timestamp mismatch at index %d: expected %v, got %v",
			i, expectedPoint.Timestamp, actualPoint.Timestamp)
		
		// Assert value with tolerance
		AssertFloatEquals(t, expectedPoint.Value, actualPoint.Value, tolerance,
			"value mismatch at index %d", i)
		
		// Assert tags
		assert.Equal(t, expectedPoint.Tags, actualPoint.Tags,
			"tags mismatch at index %d", i)
	}
}

// AssertFloatEquals asserts that two floats are equal within tolerance
func AssertFloatEquals(t *testing.T, expected, actual, tolerance float64, msgAndArgs ...interface{}) {
	t.Helper()
	
	if math.IsNaN(expected) && math.IsNaN(actual) {
		return
	}
	
	if math.IsInf(expected, 0) && math.IsInf(actual, 0) {
		assert.Equal(t, math.Signbit(expected), math.Signbit(actual), msgAndArgs...)
		return
	}
	
	diff := math.Abs(expected - actual)
	assert.True(t, diff <= tolerance,
		"expected %f to be within %f of %f (diff: %f). %s",
		actual, tolerance, expected, diff, fmt.Sprint(msgAndArgs...))
}

// AssertFloatSliceEquals asserts that two float slices are equal within tolerance
func AssertFloatSliceEquals(t *testing.T, expected, actual []float64, tolerance float64, msgAndArgs ...interface{}) {
	t.Helper()
	
	require.Equal(t, len(expected), len(actual), "slice length mismatch. %s", fmt.Sprint(msgAndArgs...))
	
	for i := range expected {
		AssertFloatEquals(t, expected[i], actual[i], tolerance,
			"element %d: %s", i, fmt.Sprint(msgAndArgs...))
	}
}

// AssertStatisticalProperties asserts statistical properties of data
func AssertStatisticalProperties(t *testing.T, data []float64, expectedMean, expectedStdDev, tolerance float64) {
	t.Helper()
	
	require.NotEmpty(t, data, "data cannot be empty")
	
	// Calculate mean
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	mean := sum / float64(len(data))
	
	// Calculate standard deviation
	variance := 0.0
	for _, v := range data {
		variance += (v - mean) * (v - mean)
	}
	variance /= float64(len(data))
	stdDev := math.Sqrt(variance)
	
	AssertFloatEquals(t, expectedMean, mean, tolerance, "mean mismatch")
	AssertFloatEquals(t, expectedStdDev, stdDev, tolerance, "standard deviation mismatch")
}

// AssertTrendDirection asserts the trend direction of time series data
func AssertTrendDirection(t *testing.T, data []float64, expectedTrend string) {
	t.Helper()
	
	require.GreaterOrEqual(t, len(data), 2, "need at least 2 data points")
	
	increasing := 0
	decreasing := 0
	
	for i := 1; i < len(data); i++ {
		if data[i] > data[i-1] {
			increasing++
		} else if data[i] < data[i-1] {
			decreasing++
		}
	}
	
	switch strings.ToLower(expectedTrend) {
	case "increasing", "up":
		assert.Greater(t, increasing, decreasing, "data should be increasing")
	case "decreasing", "down":
		assert.Greater(t, decreasing, increasing, "data should be decreasing")
	case "stable", "flat":
		assert.Equal(t, increasing, decreasing, "data should be stable")
	default:
		t.Fatalf("unknown trend direction: %s", expectedTrend)
	}
}

// AssertWithinRange asserts that all values are within a specified range
func AssertWithinRange(t *testing.T, data []float64, min, max float64) {
	t.Helper()
	
	for i, v := range data {
		assert.GreaterOrEqual(t, v, min, "value at index %d is below minimum", i)
		assert.LessOrEqual(t, v, max, "value at index %d is above maximum", i)
	}
}

// AssertNoOutliers asserts that data has no outliers beyond z-score threshold
func AssertNoOutliers(t *testing.T, data []float64, zScoreThreshold float64) {
	t.Helper()
	
	if len(data) < 3 {
		return // Not enough data for outlier detection
	}
	
	// Calculate mean and standard deviation
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	mean := sum / float64(len(data))
	
	variance := 0.0
	for _, v := range data {
		variance += (v - mean) * (v - mean)
	}
	variance /= float64(len(data))
	stdDev := math.Sqrt(variance)
	
	if stdDev == 0 {
		return // No variation, no outliers
	}
	
	// Check for outliers
	for i, v := range data {
		zScore := math.Abs((v - mean) / stdDev)
		assert.LessOrEqual(t, zScore, zScoreThreshold,
			"outlier detected at index %d: value=%f, z-score=%f", i, v, zScore)
	}
}

// AssertPeriodicPattern asserts that data exhibits periodic behavior
func AssertPeriodicPattern(t *testing.T, data []float64, expectedPeriod int, tolerance float64) {
	t.Helper()
	
	require.GreaterOrEqual(t, len(data), expectedPeriod*2, "need at least 2 periods of data")
	
	// Compare values at periodic intervals
	for i := expectedPeriod; i < len(data); i++ {
		baseIndex := i % expectedPeriod
		AssertFloatEquals(t, data[baseIndex], data[i], tolerance,
			"periodic pattern broken at index %d (period %d)", i, expectedPeriod)
	}
}

// AssertCorrelation asserts correlation between two data series
func AssertCorrelation(t *testing.T, x, y []float64, expectedCorrelation, tolerance float64) {
	t.Helper()
	
	require.Equal(t, len(x), len(y), "data series must have same length")
	require.NotEmpty(t, x, "data cannot be empty")
	
	n := float64(len(x))
	
	// Calculate means
	meanX := 0.0
	meanY := 0.0
	for i := range x {
		meanX += x[i]
		meanY += y[i]
	}
	meanX /= n
	meanY /= n
	
	// Calculate correlation
	numerator := 0.0
	sumXSquared := 0.0
	sumYSquared := 0.0
	
	for i := range x {
		deltaX := x[i] - meanX
		deltaY := y[i] - meanY
		numerator += deltaX * deltaY
		sumXSquared += deltaX * deltaX
		sumYSquared += deltaY * deltaY
	}
	
	denominator := math.Sqrt(sumXSquared * sumYSquared)
	if denominator == 0 {
		t.Fatal("cannot calculate correlation: zero variance")
	}
	
	correlation := numerator / denominator
	AssertFloatEquals(t, expectedCorrelation, correlation, tolerance, "correlation mismatch")
}

// AssertHTTPResponse asserts HTTP response properties
func AssertHTTPResponse(t *testing.T, statusCode int, body []byte, expectedStatus int, expectedBodyContains ...string) {
	t.Helper()
	
	assert.Equal(t, expectedStatus, statusCode, "HTTP status code mismatch")
	
	bodyStr := string(body)
	for _, expected := range expectedBodyContains {
		assert.Contains(t, bodyStr, expected, "response body should contain expected text")
	}
}

// AssertJSONResponse asserts JSON response structure
func AssertJSONResponse(t *testing.T, body []byte, expectedFields map[string]interface{}) {
	t.Helper()
	
	var actual map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &actual), "response should be valid JSON")
	
	for field, expectedValue := range expectedFields {
		actualValue, exists := actual[field]
		assert.True(t, exists, "field %s should exist in response", field)
		
		if exists {
			assert.Equal(t, expectedValue, actualValue, "field %s value mismatch", field)
		}
	}
}

// AssertEventuallyTrue asserts that condition becomes true within timeout
func AssertEventuallyTrue(t *testing.T, condition func() bool, timeout time.Duration, interval time.Duration, msgAndArgs ...interface{}) {
	t.Helper()
	
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if condition() {
			return
		}
		time.Sleep(interval)
	}
	
	t.Fatalf("condition did not become true within %v. %s", timeout, fmt.Sprint(msgAndArgs...))
}

// AssertEventuallyEqual asserts that getter eventually returns expected value
func AssertEventuallyEqual(t *testing.T, expected interface{}, getter func() interface{}, timeout time.Duration, interval time.Duration, msgAndArgs ...interface{}) {
	t.Helper()
	
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if reflect.DeepEqual(expected, getter()) {
			return
		}
		time.Sleep(interval)
	}
	
	actual := getter()
	t.Fatalf("expected %v, got %v after %v. %s", expected, actual, timeout, fmt.Sprint(msgAndArgs...))
}

// AssertConcurrent runs assertions concurrently and waits for all to complete
func AssertConcurrent(t *testing.T, assertions ...func(t *testing.T)) {
	t.Helper()
	
	done := make(chan bool, len(assertions))
	
	for _, assertion := range assertions {
		go func(a func(t *testing.T)) {
			defer func() { done <- true }()
			a(t)
		}(assertion)
	}
	
	// Wait for all assertions to complete
	for i := 0; i < len(assertions); i++ {
		<-done
	}
}

// AssertPanic asserts that function panics
func AssertPanic(t *testing.T, fn func()) {
	t.Helper()
	
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected function to panic")
		}
	}()
	
	fn()
}

// AssertNoPanic asserts that function does not panic
func AssertNoPanic(t *testing.T, fn func()) {
	t.Helper()
	
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("unexpected panic: %v", r)
		}
	}()
	
	fn()
}

// AssertContextTimeout asserts that operation times out
func AssertContextTimeout(t *testing.T, timeout time.Duration, operation func(ctx context.Context) error) {
	t.Helper()
	
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	
	err := operation(ctx)
	assert.Error(t, err, "operation should timeout")
	assert.True(t, isContextTimeoutError(err), "error should be context timeout")
}

// AssertValidationErrors asserts that validation returns expected errors
func AssertValidationErrors(t *testing.T, err error, expectedErrors ...string) {
	t.Helper()
	
	require.Error(t, err, "validation should return errors")
	
	errorStr := err.Error()
	for _, expectedError := range expectedErrors {
		assert.Contains(t, errorStr, expectedError, "should contain validation error")
	}
}

// AssertMetricsIncreased asserts that metrics have increased
func AssertMetricsIncreased(t *testing.T, beforeMetrics, afterMetrics map[string]float64, expectedIncreases map[string]float64) {
	t.Helper()
	
	for metric, expectedIncrease := range expectedIncreases {
		before, beforeExists := beforeMetrics[metric]
		after, afterExists := afterMetrics[metric]
		
		require.True(t, beforeExists, "before metric %s should exist", metric)
		require.True(t, afterExists, "after metric %s should exist", metric)
		
		actualIncrease := after - before
		AssertFloatEquals(t, expectedIncrease, actualIncrease, 0.001,
			"metric %s increase mismatch", metric)
	}
}

// AssertFileExists asserts that file exists and optionally checks content
func AssertFileExists(t *testing.T, filepath string, expectedContent ...string) {
	t.Helper()
	
	assert.FileExists(t, filepath, "file should exist")
	
	if len(expectedContent) > 0 {
		content, err := os.ReadFile(filepath)
		require.NoError(t, err, "should be able to read file")
		
		contentStr := string(content)
		for _, expected := range expectedContent {
			assert.Contains(t, contentStr, expected, "file should contain expected content")
		}
	}
}

// Helper function to check if error is context timeout
func isContextTimeoutError(err error) bool {
	return err == context.DeadlineExceeded || strings.Contains(err.Error(), "context deadline exceeded")
}