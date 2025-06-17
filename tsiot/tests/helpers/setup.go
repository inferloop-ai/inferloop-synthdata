package helpers

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/require"

	"github.com/inferloop/tsiot/pkg/models"
)

// TestConfig contains configuration for test setup
type TestConfig struct {
	LogLevel        string
	TimeoutDuration time.Duration
	DataDir         string
	TempDir         string
}

// TestEnvironment provides a test environment with common utilities
type TestEnvironment struct {
	Config  *TestConfig
	Logger  *logrus.Logger
	Context context.Context
	Cancel  context.CancelFunc
	T       *testing.T
}

// NewTestEnvironment creates a new test environment
func NewTestEnvironment(t *testing.T) *TestEnvironment {
	config := &TestConfig{
		LogLevel:        "debug",
		TimeoutDuration: 30 * time.Second,
		DataDir:         "./testdata",
		TempDir:         t.TempDir(),
	}

	logger := logrus.New()
	logger.SetLevel(logrus.DebugLevel)
	logger.SetFormatter(&logrus.TextFormatter{
		DisableColors: true,
		FullTimestamp: true,
	})

	ctx, cancel := context.WithTimeout(context.Background(), config.TimeoutDuration)

	return &TestEnvironment{
		Config:  config,
		Logger:  logger,
		Context: ctx,
		Cancel:  cancel,
		T:       t,
	}
}

// Cleanup performs test cleanup
func (env *TestEnvironment) Cleanup() {
	if env.Cancel != nil {
		env.Cancel()
	}
}

// GenerateTestTimeSeries creates a test time series with specified parameters
func (env *TestEnvironment) GenerateTestTimeSeries(id string, numPoints int, pattern string) *models.TimeSeries {
	points := make([]models.DataPoint, numPoints)
	now := time.Now()
	
	for i := 0; i < numPoints; i++ {
		timestamp := now.Add(time.Duration(i) * time.Minute).Unix()
		value := env.generateValue(i, pattern)
		
		points[i] = models.DataPoint{
			Timestamp: timestamp,
			Value:     value,
			Quality:   0.9 + rand.Float64()*0.1, // Quality between 0.9 and 1.0
		}
	}

	return &models.TimeSeries{
		ID:     id,
		Name:   "Test Series " + id,
		Points: points,
		Metadata: map[string]interface{}{
			"test":         true,
			"pattern":      pattern,
			"generated_at": now.Format(time.RFC3339),
		},
		Properties: map[string]interface{}{
			"unit":        "units",
			"description": "Test time series for " + pattern + " pattern",
		},
		Tags: []string{"test", pattern, id},
	}
}

// generateValue generates a test value based on pattern
func (env *TestEnvironment) generateValue(index int, pattern string) float64 {
	t := float64(index)
	
	switch pattern {
	case "sine":
		return 50 + 20*math.Sin(2*math.Pi*t/100)
	case "linear":
		return t * 0.5
	case "random":
		return rand.Float64() * 100
	case "constant":
		return 42.0
	case "exponential":
		return math.Exp(t / 100)
	case "step":
		if int(t)%50 == 0 {
			return 100
		}
		return 10
	default:
		return t
	}
}

// GenerateTestDataset creates a dataset with multiple time series
func (env *TestEnvironment) GenerateTestDataset(numSeries, pointsPerSeries int, patterns []string) []*models.TimeSeries {
	dataset := make([]*models.TimeSeries, numSeries)
	
	for i := 0; i < numSeries; i++ {
		pattern := patterns[i%len(patterns)]
		id := fmt.Sprintf("test_series_%d", i)
		dataset[i] = env.GenerateTestTimeSeries(id, pointsPerSeries, pattern)
	}
	
	return dataset
}

// AssertTimeSeriesValid checks if a time series is valid
func (env *TestEnvironment) AssertTimeSeriesValid(ts *models.TimeSeries) {
	require.NotNil(env.T, ts, "time series should not be nil")
	require.NotEmpty(env.T, ts.ID, "time series ID should not be empty")
	require.NotEmpty(env.T, ts.Name, "time series name should not be empty")
	require.NotEmpty(env.T, ts.Points, "time series should have data points")
	
	// Check data points
	for i, point := range ts.Points {
		require.False(env.T, math.IsNaN(point.Value), "data point %d should not be NaN", i)
		require.False(env.T, math.IsInf(point.Value, 0), "data point %d should not be infinite", i)
		require.True(env.T, point.Quality >= 0 && point.Quality <= 1, "quality should be between 0 and 1")
		require.True(env.T, point.Timestamp > 0, "timestamp should be positive")
	}
}

// AssertDatasetValid checks if a dataset is valid
func (env *TestEnvironment) AssertDatasetValid(dataset []*models.TimeSeries) {
	require.NotEmpty(env.T, dataset, "dataset should not be empty")
	
	for i, ts := range dataset {
		require.NotNil(env.T, ts, "time series %d should not be nil", i)
		env.AssertTimeSeriesValid(ts)
	}
}

// CreateMockValidationRequest creates a mock validation request for testing
func (env *TestEnvironment) CreateMockValidationRequest(synthetic, reference *models.TimeSeries) *models.ValidationRequest {
	return &models.ValidationRequest{
		ID:        "test_validation_" + time.Now().Format("20060102150405"),
		Timestamp: time.Now().Unix(),
		Parameters: models.ValidationParameters{
			SyntheticData: synthetic,
			ReferenceData: reference,
			Metrics:       []string{"basic", "statistical", "quality"},
			Timeout:       "30s",
		},
	}
}

// WaitForCondition waits for a condition to be true with timeout
func (env *TestEnvironment) WaitForCondition(condition func() bool, timeout time.Duration, message string) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			env.T.Fatalf("Timeout waiting for condition: %s", message)
		case <-ticker.C:
			if condition() {
				return
			}
		}
	}
}

// RunWithTimeout runs a function with a timeout
func (env *TestEnvironment) RunWithTimeout(fn func() error, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	
	done := make(chan error, 1)
	go func() {
		done <- fn()
	}()
	
	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-done:
		return err
	}
}

// GetTestLogger returns a test logger
func GetTestLogger(t *testing.T) *logrus.Logger {
	logger := logrus.New()
	logger.SetLevel(logrus.DebugLevel)
	logger.SetFormatter(&logrus.TextFormatter{
		DisableColors: true,
		FullTimestamp: true,
	})
	return logger
}

// GetTestContext returns a test context with timeout
func GetTestContext(timeout time.Duration) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), timeout)
}

// SkipIfShort skips the test if testing.Short() is true
func SkipIfShort(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}
}

// RequireNoError is a helper for require.NoError with custom message
func RequireNoError(t *testing.T, err error, msgAndArgs ...interface{}) {
	require.NoError(t, err, msgAndArgs...)
}

// RequireEqual is a helper for require.Equal with custom message
func RequireEqual(t *testing.T, expected, actual interface{}, msgAndArgs ...interface{}) {
	require.Equal(t, expected, actual, msgAndArgs...)
}

// Float64SlicesEqual compares two float64 slices with tolerance
func Float64SlicesEqual(a, b []float64, tolerance float64) bool {
	if len(a) != len(b) {
		return false
	}
	
	for i := range a {
		if math.Abs(a[i]-b[i]) > tolerance {
			return false
		}
	}
	
	return true
}

// AssertFloat64SlicesEqual asserts that two float64 slices are equal within tolerance
func AssertFloat64SlicesEqual(t *testing.T, expected, actual []float64, tolerance float64, msgAndArgs ...interface{}) {
	require.True(t, Float64SlicesEqual(expected, actual, tolerance), msgAndArgs...)
}