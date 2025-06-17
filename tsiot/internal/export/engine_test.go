package export

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inferloop/tsiot/pkg/models"
)

func TestNewExportEngine(t *testing.T) {
	logger := logrus.New()
	config := &ExportConfig{
		Enabled:           true,
		MaxConcurrentJobs: 5,
		JobTimeout:        30 * time.Second,
		OutputDirectory:   "/tmp/tsiot/export",
		EnableCompression: true,
	}

	engine, err := NewExportEngine(config, logger)
	require.NoError(t, err)
	require.NotNil(t, engine)

	assert.Equal(t, config, engine.config)
	assert.Equal(t, logger, engine.logger)
	assert.NotNil(t, engine.exporters)
	assert.NotNil(t, engine.jobs)
	assert.NotNil(t, engine.jobQueue)
}

func TestExportTimeSeriesCSV(t *testing.T) {
	engine, err := NewExportEngine(nil, logrus.New())
	require.NoError(t, err)

	timeSeries := createTestTimeSeries()
	var buf bytes.Buffer
	
	options := ExportOptions{
		IncludeHeaders: true,
		DateFormat:     time.RFC3339,
		CSVOptions: CSVOptions{
			Delimiter: ",",
			Quote:     "\"",
		},
	}

	ctx := context.Background()
	err = engine.ExportTimeSeries(ctx, []*models.TimeSeries{timeSeries}, FormatCSV, &buf, options)
	require.NoError(t, err)

	// Parse the CSV output
	reader := csv.NewReader(strings.NewReader(buf.String()))
	records, err := reader.ReadAll()
	require.NoError(t, err)

	// Check header
	assert.Equal(t, []string{"timestamp", "value", "quality"}, records[0])
	
	// Check data rows
	assert.Equal(t, 4, len(records)) // 1 header + 3 data points
	assert.Equal(t, "10", records[1][1]) // First value
	assert.Equal(t, "30", records[2][1]) // Second value
	assert.Equal(t, "50", records[3][1]) // Third value
}

func TestExportTimeSeriesJSON(t *testing.T) {
	engine, err := NewExportEngine(nil, logrus.New())
	require.NoError(t, err)

	timeSeries := createTestTimeSeries()
	var buf bytes.Buffer
	
	options := ExportOptions{
		DateFormat: time.RFC3339,
		JSONOptions: JSONOptions{
			Pretty:      true,
			ArrayFormat: false,
		},
	}

	ctx := context.Background()
	err = engine.ExportTimeSeries(ctx, []*models.TimeSeries{timeSeries}, FormatJSON, &buf, options)
	require.NoError(t, err)

	// Parse the JSON output
	var result []*models.TimeSeries
	err = json.Unmarshal(buf.Bytes(), &result)
	require.NoError(t, err)

	assert.Equal(t, 1, len(result))
	assert.Equal(t, timeSeries.ID, result[0].ID)
	assert.Equal(t, len(timeSeries.DataPoints), len(result[0].DataPoints))
}

func TestSubmitExportJob(t *testing.T) {
	engine, err := NewExportEngine(nil, logrus.New())
	require.NoError(t, err)

	timeSeries := createTestTimeSeries()
	job := &ExportJob{
		Format:     FormatCSV,
		TimeSeries: []*models.TimeSeries{timeSeries},
		Options: ExportOptions{
			IncludeHeaders: true,
		},
	}

	err = engine.SubmitExportJob(job)
	require.NoError(t, err)

	assert.NotEmpty(t, job.ID)
	assert.Equal(t, JobStatusPending, job.Status)
	assert.False(t, job.CreatedAt.IsZero())
}

func TestGetJobStatus(t *testing.T) {
	engine, err := NewExportEngine(nil, logrus.New())
	require.NoError(t, err)

	timeSeries := createTestTimeSeries()
	job := &ExportJob{
		ID:         "test-job-1",
		Format:     FormatJSON,
		TimeSeries: []*models.TimeSeries{timeSeries},
	}

	err = engine.SubmitExportJob(job)
	require.NoError(t, err)

	retrievedJob, err := engine.GetJobStatus("test-job-1")
	require.NoError(t, err)
	assert.Equal(t, job.ID, retrievedJob.ID)
	assert.Equal(t, job.Format, retrievedJob.Format)
}

func TestGetJobStatusNotFound(t *testing.T) {
	engine, err := NewExportEngine(nil, logrus.New())
	require.NoError(t, err)

	_, err = engine.GetJobStatus("non-existent-job")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestCancelJob(t *testing.T) {
	engine, err := NewExportEngine(nil, logrus.New())
	require.NoError(t, err)

	timeSeries := createTestTimeSeries()
	job := &ExportJob{
		ID:         "test-job-cancel",
		Format:     FormatCSV,
		TimeSeries: []*models.TimeSeries{timeSeries},
	}

	err = engine.SubmitExportJob(job)
	require.NoError(t, err)

	err = engine.CancelJob("test-job-cancel")
	require.NoError(t, err)

	retrievedJob, err := engine.GetJobStatus("test-job-cancel")
	require.NoError(t, err)
	assert.Equal(t, JobStatusCancelled, retrievedJob.Status)
}

func TestApplyFilters(t *testing.T) {
	engine, err := NewExportEngine(nil, logrus.New())
	require.NoError(t, err)

	timeSeries := createTestTimeSeriesWithTags()
	
	// Test time range filter
	now := time.Now()
	startTime := now.Add(-2 * time.Hour)
	endTime := now.Add(-1 * time.Hour)
	
	filter := FilterOptions{
		StartTime: &startTime,
		EndTime:   &endTime,
		MinValue:  nil,
		MaxValue:  nil,
	}

	filtered := engine.applyFilters([]*models.TimeSeries{timeSeries}, filter)
	require.Equal(t, 1, len(filtered))
	
	// Should only include points within the time range
	assert.Less(t, len(filtered[0].DataPoints), len(timeSeries.DataPoints))
	
	for _, point := range filtered[0].DataPoints {
		assert.True(t, point.Timestamp.After(startTime) || point.Timestamp.Equal(startTime))
		assert.True(t, point.Timestamp.Before(endTime) || point.Timestamp.Equal(endTime))
	}
}

func TestApplyAggregation(t *testing.T) {
	engine, err := NewExportEngine(nil, logrus.New())
	require.NoError(t, err)

	timeSeries := createLargeTestTimeSeries(100)
	
	agg := AggregationOptions{
		Enabled:  true,
		Method:   "time_bucket",
		Interval: 1 * time.Hour,
		Function: "mean",
	}

	aggregated, err := engine.applyAggregation([]*models.TimeSeries{timeSeries}, agg)
	require.NoError(t, err)
	require.Equal(t, 1, len(aggregated))

	// Aggregated series should have fewer points
	assert.Less(t, len(aggregated[0].DataPoints), len(timeSeries.DataPoints))
	assert.Contains(t, aggregated[0].ID, "_agg")
}

func TestApplyAggregationFunction(t *testing.T) {
	engine, err := NewExportEngine(nil, logrus.New())
	require.NoError(t, err)

	points := []models.DataPoint{
		{Value: 10.0},
		{Value: 20.0},
		{Value: 30.0},
		{Value: 40.0},
		{Value: 50.0},
	}

	// Test mean
	result, err := engine.applyAggregationFunction(points, "mean")
	require.NoError(t, err)
	assert.Equal(t, 30.0, result)

	// Test sum
	result, err = engine.applyAggregationFunction(points, "sum")
	require.NoError(t, err)
	assert.Equal(t, 150.0, result)

	// Test min
	result, err = engine.applyAggregationFunction(points, "min")
	require.NoError(t, err)
	assert.Equal(t, 10.0, result)

	// Test max
	result, err = engine.applyAggregationFunction(points, "max")
	require.NoError(t, err)
	assert.Equal(t, 50.0, result)

	// Test median
	result, err = engine.applyAggregationFunction(points, "median")
	require.NoError(t, err)
	assert.Equal(t, 30.0, result)

	// Test count
	result, err = engine.applyAggregationFunction(points, "count")
	require.NoError(t, err)
	assert.Equal(t, 5.0, result)
}

func TestGetSupportedFormats(t *testing.T) {
	engine, err := NewExportEngine(nil, logrus.New())
	require.NoError(t, err)

	formats := engine.GetSupportedFormats()
	assert.Greater(t, len(formats), 0)
	
	// Check for default formats
	formatMap := make(map[ExportFormat]bool)
	for _, format := range formats {
		formatMap[format] = true
	}
	
	assert.True(t, formatMap[FormatCSV])
	assert.True(t, formatMap[FormatJSON])
}

func TestAggregateByTimeBucket(t *testing.T) {
	engine, err := NewExportEngine(nil, logrus.New())
	require.NoError(t, err)

	now := time.Now()
	points := []models.DataPoint{
		{Timestamp: now, Value: 10.0},
		{Timestamp: now.Add(30 * time.Minute), Value: 20.0},
		{Timestamp: now.Add(1 * time.Hour), Value: 30.0},
		{Timestamp: now.Add(90 * time.Minute), Value: 40.0},
		{Timestamp: now.Add(2 * time.Hour), Value: 50.0},
	}

	agg := AggregationOptions{
		Interval: 1 * time.Hour,
		Function: "mean",
	}

	result, err := engine.aggregateByTimeBucket(points, agg)
	require.NoError(t, err)

	// Should have 3 buckets (0-1h, 1-2h, 2-3h)
	assert.Equal(t, 3, len(result))
	
	// First bucket should average 10 and 20
	assert.Equal(t, 15.0, result[0].Value)
	
	// Second bucket should average 30 and 40
	assert.Equal(t, 35.0, result[1].Value)
	
	// Third bucket should have 50
	assert.Equal(t, 50.0, result[2].Value)
}

func TestAggregateByCount(t *testing.T) {
	engine, err := NewExportEngine(nil, logrus.New())
	require.NoError(t, err)

	points := []models.DataPoint{
		{Value: 10.0},
		{Value: 20.0},
		{Value: 30.0},
		{Value: 40.0},
		{Value: 50.0},
	}

	agg := AggregationOptions{
		Count:    2,
		Function: "mean",
	}

	result, err := engine.aggregateByCount(points, agg)
	require.NoError(t, err)

	// Should have 3 groups: [10,20], [30,40], [50]
	assert.Equal(t, 3, len(result))
	assert.Equal(t, 15.0, result[0].Value) // (10+20)/2
	assert.Equal(t, 35.0, result[1].Value) // (30+40)/2
	assert.Equal(t, 50.0, result[2].Value) // 50
}

func TestAggregateBySlidingWindow(t *testing.T) {
	engine, err := NewExportEngine(nil, logrus.New())
	require.NoError(t, err)

	points := []models.DataPoint{
		{Value: 10.0},
		{Value: 20.0},
		{Value: 30.0},
		{Value: 40.0},
		{Value: 50.0},
	}

	agg := AggregationOptions{
		Count:    3, // Window size
		Function: "mean",
	}

	result, err := engine.aggregateBySlidingWindow(points, agg)
	require.NoError(t, err)

	// Should have 3 windows: [10,20,30], [20,30,40], [30,40,50]
	assert.Equal(t, 3, len(result))
	assert.Equal(t, 20.0, result[0].Value) // (10+20+30)/3
	assert.Equal(t, 30.0, result[1].Value) // (20+30+40)/3
	assert.Equal(t, 40.0, result[2].Value) // (30+40+50)/3
}

func TestCreateOutputFile(t *testing.T) {
	engine, err := NewExportEngine(nil, logrus.New())
	require.NoError(t, err)

	// Test regular file
	writer, err := engine.createOutputFile("/tmp/test.csv")
	require.NoError(t, err)
	require.NotNil(t, writer)
	writer.Close()

	// Test gzip file
	writer, err = engine.createOutputFile("/tmp/test.csv.gz")
	require.NoError(t, err)
	require.NotNil(t, writer)
	writer.Close()

	// Test zip file
	writer, err = engine.createOutputFile("/tmp/test.csv.zip")
	require.NoError(t, err)
	require.NotNil(t, writer)
	writer.Close()
}

// Helper functions for tests

func createTestTimeSeries() *models.TimeSeries {
	now := time.Now()
	dataPoints := []models.DataPoint{
		{Timestamp: now.Add(-2 * time.Hour), Value: 10.0, Quality: 1.0},
		{Timestamp: now.Add(-1 * time.Hour), Value: 30.0, Quality: 1.0},
		{Timestamp: now, Value: 50.0, Quality: 1.0},
	}

	return &models.TimeSeries{
		ID:          "test-series-export",
		Name:        "Test Export Series",
		Description: "Test time series for export tests",
		SensorType:  "temperature",
		DataPoints:  dataPoints,
		StartTime:   dataPoints[0].Timestamp,
		EndTime:     dataPoints[len(dataPoints)-1].Timestamp,
		Frequency:   "1h",
		Tags: map[string]string{
			"location": "datacenter",
			"type":     "sensor",
		},
		Metadata: map[string]interface{}{
			"version": "1.0",
			"source":  "test",
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
}

func createTestTimeSeriesWithTags() *models.TimeSeries {
	now := time.Now()
	var dataPoints []models.DataPoint

	// Create 6 hours of data points
	for i := 0; i < 6; i++ {
		dataPoints = append(dataPoints, models.DataPoint{
			Timestamp: now.Add(time.Duration(-5+i) * time.Hour),
			Value:     float64(i * 10),
			Quality:   1.0,
		})
	}

	return &models.TimeSeries{
		ID:          "test-series-tags",
		Name:        "Test Series with Tags",
		Description: "Test time series with tags for filtering",
		SensorType:  "temperature",
		DataPoints:  dataPoints,
		StartTime:   dataPoints[0].Timestamp,
		EndTime:     dataPoints[len(dataPoints)-1].Timestamp,
		Frequency:   "1h",
		Tags: map[string]string{
			"location": "datacenter",
			"rack":     "A1",
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
}

func createLargeTestTimeSeries(size int) *models.TimeSeries {
	now := time.Now()
	var dataPoints []models.DataPoint

	for i := 0; i < size; i++ {
		dataPoints = append(dataPoints, models.DataPoint{
			Timestamp: now.Add(time.Duration(i) * time.Minute),
			Value:     float64(i%50) + 10.0, // Creates a pattern
			Quality:   1.0,
		})
	}

	return &models.TimeSeries{
		ID:          "large-test-series",
		Name:        "Large Test Series",
		Description: "Large time series for aggregation tests",
		SensorType:  "temperature",
		DataPoints:  dataPoints,
		StartTime:   dataPoints[0].Timestamp,
		EndTime:     dataPoints[len(dataPoints)-1].Timestamp,
		Frequency:   "1m",
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
}

// Benchmark tests
func BenchmarkExportCSV(b *testing.B) {
	engine, _ := NewExportEngine(nil, logrus.New())
	timeSeries := createLargeTestTimeSeries(10000)
	options := ExportOptions{IncludeHeaders: true}
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var buf bytes.Buffer
		err := engine.ExportTimeSeries(ctx, []*models.TimeSeries{timeSeries}, FormatCSV, &buf, options)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkExportJSON(b *testing.B) {
	engine, _ := NewExportEngine(nil, logrus.New())
	timeSeries := createLargeTestTimeSeries(10000)
	options := ExportOptions{}
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var buf bytes.Buffer
		err := engine.ExportTimeSeries(ctx, []*models.TimeSeries{timeSeries}, FormatJSON, &buf, options)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAggregation(b *testing.B) {
	engine, _ := NewExportEngine(nil, logrus.New())
	timeSeries := createLargeTestTimeSeries(10000)
	
	agg := AggregationOptions{
		Enabled:  true,
		Method:   "time_bucket",
		Interval: 1 * time.Hour,
		Function: "mean",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := engine.applyAggregation([]*models.TimeSeries{timeSeries}, agg)
		if err != nil {
			b.Fatal(err)
		}
	}
}