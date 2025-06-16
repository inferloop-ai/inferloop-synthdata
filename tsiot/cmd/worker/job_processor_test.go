package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inferloop/tsiot/pkg/models"
)

func TestNewJobProcessor(t *testing.T) {
	config := &WorkerConfig{
		Concurrency: 2,
	}
	logger := logrus.New()

	jp := NewJobProcessor(config, logger)

	require.NotNil(t, jp)
	assert.Equal(t, config, jp.config)
	assert.Equal(t, logger, jp.logger)
	assert.NotNil(t, jp.generatorFactory)
	assert.NotNil(t, jp.storageFactory)
	assert.NotNil(t, jp.validationEngine)
	assert.NotNil(t, jp.analyticsEngine)
	assert.Equal(t, "/tmp/tsiot-output", jp.outputDir)

	// Check that output directory was created
	_, err := os.Stat(jp.outputDir)
	assert.NoError(t, err)
}

func TestJobProcessorSetScheduler(t *testing.T) {
	jp := NewJobProcessor(&WorkerConfig{}, logrus.New())
	scheduler := &Scheduler{}

	jp.SetScheduler(scheduler)
	assert.Equal(t, scheduler, jp.scheduler)
}

func TestJobProcessorMetrics(t *testing.T) {
	jp := NewJobProcessor(&WorkerConfig{}, logrus.New())

	// Test initial values
	assert.Equal(t, int32(0), jp.ActiveJobs())
	assert.Equal(t, int64(0), jp.CompletedJobs())
	assert.Equal(t, int64(0), jp.FailedJobs())
}

func TestJobProcessorProcessGenerateJob(t *testing.T) {
	jp := NewJobProcessor(&WorkerConfig{}, logrus.New())
	
	// Create temporary output directory for test
	tempDir, err := os.MkdirTemp("", "test-job-processor")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)
	jp.outputDir = tempDir

	ctx := context.Background()
	job := &Job{
		ID:   "test-generate-job",
		Type: JobTypeGenerate,
		Parameters: map[string]interface{}{
			"generator":     "statistical",
			"sensor_type":   "temperature",
			"count":         100.0,
			"frequency":     "1m",
			"duration":      "1h",
			"output_format": "json",
		},
	}

	result, err := jp.processGenerateJob(ctx, job)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify result structure
	resultMap, ok := result.(map[string]interface{})
	require.True(t, ok)

	assert.Contains(t, resultMap, "records_generated")
	assert.Contains(t, resultMap, "time_range")
	assert.Contains(t, resultMap, "output_location")
	assert.Contains(t, resultMap, "output_format")
	assert.Contains(t, resultMap, "generation_time")
	assert.Contains(t, resultMap, "statistics")

	// Check that output file was created
	outputLocation, ok := resultMap["output_location"].(string)
	require.True(t, ok)
	_, err = os.Stat(outputLocation)
	assert.NoError(t, err)
}

func TestJobProcessorProcessGenerateJobDefaults(t *testing.T) {
	jp := NewJobProcessor(&WorkerConfig{}, logrus.New())
	
	tempDir, err := os.MkdirTemp("", "test-job-processor")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)
	jp.outputDir = tempDir

	ctx := context.Background()
	job := &Job{
		ID:         "test-generate-defaults",
		Type:       JobTypeGenerate,
		Parameters: map[string]interface{}{},
	}

	result, err := jp.processGenerateJob(ctx, job)
	require.NoError(t, err)

	resultMap, ok := result.(map[string]interface{})
	require.True(t, ok)

	assert.Equal(t, "json", resultMap["output_format"])
	assert.Contains(t, resultMap["output_location"], "temperature")
}

func TestJobProcessorProcessValidateJob(t *testing.T) {
	jp := NewJobProcessor(&WorkerConfig{}, logrus.New())
	
	tempDir, err := os.MkdirTemp("", "test-job-processor")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)
	jp.outputDir = tempDir

	// Create test input file
	testData := createTestTimeSeriesForFile()
	inputFile := filepath.Join(tempDir, "test-input.json")
	
	data, err := json.MarshalIndent(testData, "", "  ")
	require.NoError(t, err)
	err = os.WriteFile(inputFile, data, 0644)
	require.NoError(t, err)

	ctx := context.Background()
	job := &Job{
		ID:   "test-validate-job",
		Type: JobTypeValidate,
		Parameters: map[string]interface{}{
			"input_file":        inputFile,
			"quality_threshold": 0.7,
			"validators":        []interface{}{"statistical"},
		},
	}

	result, err := jp.processValidateJob(ctx, job)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify result structure
	resultMap, ok := result.(map[string]interface{})
	require.True(t, ok)

	assert.Contains(t, resultMap, "overall_quality_score")
	assert.Contains(t, resultMap, "quality_threshold")
	assert.Contains(t, resultMap, "passed")
	assert.Contains(t, resultMap, "validation_time")
	assert.Contains(t, resultMap, "validators_run")
	assert.Contains(t, resultMap, "detailed_results")
	assert.Contains(t, resultMap, "summary")

	assert.Equal(t, 0.7, resultMap["quality_threshold"])
}

func TestJobProcessorProcessValidateJobMissingFile(t *testing.T) {
	jp := NewJobProcessor(&WorkerConfig{}, logrus.New())

	ctx := context.Background()
	job := &Job{
		ID:   "test-validate-missing",
		Type: JobTypeValidate,
		Parameters: map[string]interface{}{
			"input_file": "",
		},
	}

	_, err := jp.processValidateJob(ctx, job)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "input_file parameter is required")
}

func TestJobProcessorProcessAnalyzeJob(t *testing.T) {
	jp := NewJobProcessor(&WorkerConfig{}, logrus.New())
	
	tempDir, err := os.MkdirTemp("", "test-job-processor")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)
	jp.outputDir = tempDir

	// Create test input file
	testData := createTestTimeSeriesForFile()
	inputFile := filepath.Join(tempDir, "test-analyze-input.json")
	
	data, err := json.MarshalIndent(testData, "", "  ")
	require.NoError(t, err)
	err = os.WriteFile(inputFile, data, 0644)
	require.NoError(t, err)

	ctx := context.Background()
	job := &Job{
		ID:   "test-analyze-job",
		Type: JobTypeAnalyze,
		Parameters: map[string]interface{}{
			"input_file":     inputFile,
			"analysis_type":  []interface{}{"basic_stats", "trend", "seasonality"},
			"output_file":    "analysis-results.json",
		},
	}

	result, err := jp.processAnalyzeJob(ctx, job)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify result structure
	resultMap, ok := result.(map[string]interface{})
	require.True(t, ok)

	assert.Contains(t, resultMap, "series_id")
	assert.Contains(t, resultMap, "data_points")
	assert.Contains(t, resultMap, "time_range")
	assert.Contains(t, resultMap, "analysis_types")
	assert.Contains(t, resultMap, "processing_time")
	assert.Contains(t, resultMap, "results")
	assert.Contains(t, resultMap, "summary")

	// Check specific analysis types were processed
	results, ok := resultMap["results"].(map[string]interface{})
	require.True(t, ok)
	assert.Contains(t, results, "basic_stats")
	assert.Contains(t, results, "trend")

	// Check that output file was created
	assert.Contains(t, resultMap, "output_file")
	outputFile, ok := resultMap["output_file"].(string)
	require.True(t, ok)
	_, err = os.Stat(outputFile)
	assert.NoError(t, err)
}

func TestJobProcessorProcessAnalyzeJobDefaults(t *testing.T) {
	jp := NewJobProcessor(&WorkerConfig{}, logrus.New())
	
	tempDir, err := os.MkdirTemp("", "test-job-processor")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)
	jp.outputDir = tempDir

	// Create test input file
	testData := createTestTimeSeriesForFile()
	inputFile := filepath.Join(tempDir, "test-analyze-defaults.json")
	
	data, err := json.MarshalIndent(testData, "", "  ")
	require.NoError(t, err)
	err = os.WriteFile(inputFile, data, 0644)
	require.NoError(t, err)

	ctx := context.Background()
	job := &Job{
		ID:   "test-analyze-defaults",
		Type: JobTypeAnalyze,
		Parameters: map[string]interface{}{
			"input_file": inputFile,
		},
	}

	result, err := jp.processAnalyzeJob(ctx, job)
	require.NoError(t, err)

	resultMap, ok := result.(map[string]interface{})
	require.True(t, ok)

	// Check default analysis types
	analysisTypes, ok := resultMap["analysis_types"].([]string)
	require.True(t, ok)
	assert.Contains(t, analysisTypes, "basic_stats")
	assert.Contains(t, analysisTypes, "trend")
	assert.Contains(t, analysisTypes, "seasonality")
}

func TestJobProcessorProcessMigrateJob(t *testing.T) {
	jp := NewJobProcessor(&WorkerConfig{}, logrus.New())
	
	tempDir, err := os.MkdirTemp("", "test-job-processor")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create source file
	sourceFile := filepath.Join(tempDir, "source.json")
	destFile := filepath.Join(tempDir, "dest.json")
	
	sourceData := map[string]interface{}{"test": "data"}
	data, err := json.Marshal(sourceData)
	require.NoError(t, err)
	err = os.WriteFile(sourceFile, data, 0644)
	require.NoError(t, err)

	ctx := context.Background()
	job := &Job{
		ID:   "test-migrate-job",
		Type: JobTypeMigrate,
		Parameters: map[string]interface{}{
			"source":         sourceFile,
			"destination":    destFile,
			"batch_size":     500.0,
			"migration_type": "file_to_file",
			"dry_run":        false,
		},
	}

	result, err := jp.processMigrateJob(ctx, job)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Verify result structure
	resultMap, ok := result.(map[string]interface{})
	require.True(t, ok)

	assert.Contains(t, resultMap, "records_migrated")
	assert.Contains(t, resultMap, "duration_seconds")
	assert.Contains(t, resultMap, "duration")
	assert.Contains(t, resultMap, "average_rate")
	assert.Contains(t, resultMap, "transferred_bytes")
	assert.Contains(t, resultMap, "batch_size")
	assert.Contains(t, resultMap, "errors_count")
	assert.Contains(t, resultMap, "migration_type")

	assert.Equal(t, 500, resultMap["batch_size"])
	assert.Equal(t, "file_to_file", resultMap["migration_type"])
	assert.Equal(t, false, resultMap["dry_run"])

	// Check that destination file was created
	_, err = os.Stat(destFile)
	assert.NoError(t, err)
}

func TestJobProcessorProcessMigrateJobDryRun(t *testing.T) {
	jp := NewJobProcessor(&WorkerConfig{}, logrus.New())

	ctx := context.Background()
	job := &Job{
		ID:   "test-migrate-dry-run",
		Type: JobTypeMigrate,
		Parameters: map[string]interface{}{
			"source":         "dummy-source",
			"destination":    "dummy-dest",
			"migration_type": "storage_to_storage",
			"dry_run":        true,
		},
	}

	result, err := jp.processMigrateJob(ctx, job)
	require.NoError(t, err)

	resultMap, ok := result.(map[string]interface{})
	require.True(t, ok)

	assert.Equal(t, true, resultMap["dry_run"])
	assert.Greater(t, resultMap["records_migrated"], int64(0))
}

func TestJobProcessorProcessMigrateJobMissingParams(t *testing.T) {
	jp := NewJobProcessor(&WorkerConfig{}, logrus.New())

	ctx := context.Background()
	job := &Job{
		ID:   "test-migrate-missing",
		Type: JobTypeMigrate,
		Parameters: map[string]interface{}{
			"source": "",
		},
	}

	_, err := jp.processMigrateJob(ctx, job)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "source and destination parameters are required")
}

func TestJobProcessorLoadTimeSeriesFromFile(t *testing.T) {
	jp := NewJobProcessor(&WorkerConfig{}, logrus.New())
	
	tempDir, err := os.MkdirTemp("", "test-load-timeseries")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)
	jp.outputDir = tempDir

	t.Run("Load TimeSeries format", func(t *testing.T) {
		// Create TimeSeries format file
		testData := createTestTimeSeriesForFile()
		filename := filepath.Join(tempDir, "timeseries.json")
		
		data, err := json.MarshalIndent(testData, "", "  ")
		require.NoError(t, err)
		err = os.WriteFile(filename, data, 0644)
		require.NoError(t, err)

		ts, err := jp.loadTimeSeriesFromFile(filename)
		require.NoError(t, err)
		require.NotNil(t, ts)

		assert.Equal(t, testData.ID, ts.ID)
		assert.Equal(t, len(testData.Points), len(ts.Points))
	})

	t.Run("Load array format", func(t *testing.T) {
		// Create array format file
		arrayData := []map[string]interface{}{
			{"timestamp": "2023-01-01T00:00:00Z", "value": 10.0},
			{"timestamp": "2023-01-01T01:00:00Z", "value": 15.0},
			{"timestamp": "2023-01-01T02:00:00Z", "value": 20.0},
		}
		filename := filepath.Join(tempDir, "array.json")
		
		data, err := json.MarshalIndent(arrayData, "", "  ")
		require.NoError(t, err)
		err = os.WriteFile(filename, data, 0644)
		require.NoError(t, err)

		ts, err := jp.loadTimeSeriesFromFile(filename)
		require.NoError(t, err)
		require.NotNil(t, ts)

		assert.Equal(t, "array.json", ts.ID)
		assert.Equal(t, 3, len(ts.Points))
	})

	t.Run("Load single data point", func(t *testing.T) {
		// Create single point format file
		pointData := map[string]interface{}{
			"timestamp": "2023-01-01T00:00:00Z",
			"value":     25.0,
		}
		filename := filepath.Join(tempDir, "point.json")
		
		data, err := json.MarshalIndent(pointData, "", "  ")
		require.NoError(t, err)
		err = os.WriteFile(filename, data, 0644)
		require.NoError(t, err)

		ts, err := jp.loadTimeSeriesFromFile(filename)
		require.NoError(t, err)
		require.NotNil(t, ts)

		assert.Equal(t, "point.json", ts.ID)
		assert.Equal(t, 1, len(ts.Points))
	})

	t.Run("File not found", func(t *testing.T) {
		_, err := jp.loadTimeSeriesFromFile("nonexistent.json")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to read file")
	})
}

func TestJobProcessorProcessJobUnknownType(t *testing.T) {
	jp := NewJobProcessor(&WorkerConfig{}, logrus.New())
	
	// Create mock scheduler
	mockScheduler := &MockScheduler{}
	jp.SetScheduler(mockScheduler)

	ctx := context.Background()
	job := &Job{
		ID:   "test-unknown",
		Type: "unknown_type",
	}

	jp.processJob(ctx, job, 0)

	// Check that job was marked as failed
	assert.True(t, mockScheduler.UpdateStatusCalled)
	assert.Equal(t, "failed", mockScheduler.LastStatus)
	assert.Contains(t, mockScheduler.LastError, "unknown job type")
}

func TestJobProcessorMigrationHelpers(t *testing.T) {
	jp := NewJobProcessor(&WorkerConfig{}, logrus.New())
	ctx := context.Background()

	t.Run("migrateStorageToStorage", func(t *testing.T) {
		records, bytes, errors := jp.migrateStorageToStorage(ctx, "source", "dest", 1000, false)
		assert.Equal(t, int64(10000), records)
		assert.Equal(t, int64(1024*1024), bytes)
		assert.Empty(t, errors)
	})

	t.Run("migrateFileToStorage", func(t *testing.T) {
		// Create temporary file
		tempFile, err := os.CreateTemp("", "migrate-test")
		require.NoError(t, err)
		defer os.Remove(tempFile.Name())
		
		testData := []byte("test data for migration")
		_, err = tempFile.Write(testData)
		require.NoError(t, err)
		tempFile.Close()

		records, bytes, errors := jp.migrateFileToStorage(ctx, tempFile.Name(), "dest", 1000, false)
		assert.Equal(t, int64(5000), records)
		assert.Equal(t, int64(len(testData)), bytes)
		assert.Empty(t, errors)
	})

	t.Run("migrateStorageToFile", func(t *testing.T) {
		tempFile, err := os.CreateTemp("", "migrate-dest")
		require.NoError(t, err)
		defer os.Remove(tempFile.Name())
		tempFile.Close()

		records, bytes, errors := jp.migrateStorageToFile(ctx, "source", tempFile.Name(), 1000, false)
		assert.Equal(t, int64(7500), records)
		assert.Greater(t, bytes, int64(0))
		assert.Empty(t, errors)

		// Check that file was created with data
		_, err = os.Stat(tempFile.Name())
		assert.NoError(t, err)
	})

	t.Run("migrateDatabaseSchema", func(t *testing.T) {
		records, bytes, errors := jp.migrateDatabaseSchema(ctx, "source", "dest", false)
		assert.Equal(t, int64(1), records)
		assert.Equal(t, int64(2048), bytes)
		assert.Empty(t, errors)
	})

	t.Run("autoMigrate", func(t *testing.T) {
		// Test file to file detection
		records, bytes, errors := jp.autoMigrate(ctx, "/tmp/source.json", "/tmp/dest.json", 1000, true)
		assert.Greater(t, records, int64(0))
		assert.Greater(t, bytes, int64(0))
		assert.Empty(t, errors)

		// Test storage to storage detection
		records, bytes, errors = jp.autoMigrate(ctx, "redis://source", "s3://dest", 1000, true)
		assert.Greater(t, records, int64(0))
		assert.Greater(t, bytes, int64(0))
		assert.Empty(t, errors)
	})
}

func TestCalculateBasicStats(t *testing.T) {
	t.Run("Normal values", func(t *testing.T) {
		values := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
		stats := calculateBasicStats(values)

		assert.Equal(t, 5.0, stats["count"])
		assert.Equal(t, 3.0, stats["mean"])
		assert.Equal(t, 1.0, stats["min"])
		assert.Equal(t, 5.0, stats["max"])
		assert.Greater(t, stats["std"], 0.0)
	})

	t.Run("Empty values", func(t *testing.T) {
		values := []float64{}
		stats := calculateBasicStats(values)

		assert.Equal(t, 0.0, stats["count"])
		assert.Equal(t, 0.0, stats["mean"])
		assert.Equal(t, 0.0, stats["min"])
		assert.Equal(t, 0.0, stats["max"])
		assert.Equal(t, 0.0, stats["std"])
	})

	t.Run("Single value", func(t *testing.T) {
		values := []float64{42.0}
		stats := calculateBasicStats(values)

		assert.Equal(t, 1.0, stats["count"])
		assert.Equal(t, 42.0, stats["mean"])
		assert.Equal(t, 42.0, stats["min"])
		assert.Equal(t, 42.0, stats["max"])
	})
}

func TestJobProcessorConcurrentWorkers(t *testing.T) {
	config := &WorkerConfig{
		Concurrency: 3,
	}
	jp := NewJobProcessor(config, logrus.New())
	
	// Create mock scheduler with job queue
	mockScheduler := NewMockSchedulerWithQueue()
	jp.SetScheduler(mockScheduler)

	// Add jobs to queue
	for i := 0; i < 5; i++ {
		job := &Job{
			ID:   fmt.Sprintf("concurrent-job-%d", i),
			Type: JobTypeGenerate,
			Parameters: map[string]interface{}{
				"generator":     "statistical",
				"sensor_type":   "temperature",
				"count":         10.0,
				"output_format": "json",
			},
		}
		mockScheduler.AddJob(job)
	}

	// Close the queue to signal no more jobs
	mockScheduler.CloseQueue()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Start processing (this will block until all workers finish)
	go jp.Start(ctx)

	// Wait a moment for processing
	time.Sleep(2 * time.Second)

	// Verify that jobs were processed
	assert.Greater(t, jp.CompletedJobs(), int64(0))
}

// Helper functions and mocks

func createTestTimeSeriesForFile() *models.TimeSeries {
	now := time.Now()
	return &models.TimeSeries{
		ID:        "test-file-series",
		Name:      "Test File Series",
		StartTime: now.Add(-2 * time.Hour),
		EndTime:   now,
		Points: []models.DataPoint{
			{
				Timestamp: now.Add(-2 * time.Hour),
				Value:     10.0,
			},
			{
				Timestamp: now.Add(-1 * time.Hour),
				Value:     15.0,
			},
			{
				Timestamp: now,
				Value:     20.0,
			},
		},
		Metadata: map[string]interface{}{
			"source": "test",
		},
	}
}

// MockScheduler for testing
type MockScheduler struct {
	UpdateStatusCalled bool
	LastJobID          string
	LastStatus         string
	LastResult         interface{}
	LastError          string
	jobQueue           chan *Job
	queueClosed        bool
}

func (m *MockScheduler) UpdateJobStatus(ctx context.Context, jobID, status string, result interface{}, errorMsg string) error {
	m.UpdateStatusCalled = true
	m.LastJobID = jobID
	m.LastStatus = status
	m.LastResult = result
	m.LastError = errorMsg
	return nil
}

func (m *MockScheduler) GetJobQueue() <-chan *Job {
	if m.jobQueue == nil {
		m.jobQueue = make(chan *Job, 10)
	}
	return m.jobQueue
}

// MockSchedulerWithQueue for concurrent testing
type MockSchedulerWithQueue struct {
	*MockScheduler
}

func NewMockSchedulerWithQueue() *MockSchedulerWithQueue {
	return &MockSchedulerWithQueue{
		MockScheduler: &MockScheduler{
			jobQueue: make(chan *Job, 10),
		},
	}
}

func (m *MockSchedulerWithQueue) AddJob(job *Job) {
	if !m.queueClosed {
		m.jobQueue <- job
	}
}

func (m *MockSchedulerWithQueue) CloseQueue() {
	if !m.queueClosed {
		close(m.jobQueue)
		m.queueClosed = true
	}
}

// Benchmark tests
func BenchmarkJobProcessorGenerateJob(b *testing.B) {
	jp := NewJobProcessor(&WorkerConfig{}, logrus.New())
	
	tempDir, err := os.MkdirTemp("", "bench-job-processor")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(tempDir)
	jp.outputDir = tempDir

	ctx := context.Background()
	job := &Job{
		ID:   "bench-generate-job",
		Type: JobTypeGenerate,
		Parameters: map[string]interface{}{
			"generator":     "statistical",
			"sensor_type":   "temperature",
			"count":         100.0,
			"output_format": "json",
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		job.ID = fmt.Sprintf("bench-generate-job-%d", i)
		_, err := jp.processGenerateJob(ctx, job)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkJobProcessorProcessJob(b *testing.B) {
	jp := NewJobProcessor(&WorkerConfig{}, logrus.New())
	
	tempDir, err := os.MkdirTemp("", "bench-job-processor")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(tempDir)
	jp.outputDir = tempDir

	// Create mock scheduler
	mockScheduler := &MockScheduler{}
	jp.SetScheduler(mockScheduler)

	ctx := context.Background()
	job := &Job{
		ID:   "bench-process-job",
		Type: JobTypeGenerate,
		Parameters: map[string]interface{}{
			"generator":     "statistical",
			"sensor_type":   "temperature",
			"count":         50.0,
			"output_format": "json",
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		job.ID = fmt.Sprintf("bench-process-job-%d", i)
		jp.processJob(ctx, job, 0)
	}
}

func BenchmarkCalculateBasicStats(b *testing.B) {
	values := make([]float64, 1000)
	for i := range values {
		values[i] = float64(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		calculateBasicStats(values)
	}
}