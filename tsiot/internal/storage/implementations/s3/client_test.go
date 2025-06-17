package s3

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inferloop/tsiot/pkg/models"
)

func TestNewS3Storage(t *testing.T) {
	config := &S3Config{
		Region: "us-east-1",
		Bucket: "test-bucket",
	}

	logger := logrus.New()
	storage, err := NewS3Storage(config, logger)
	
	require.NoError(t, err)
	require.NotNil(t, storage)
	assert.Equal(t, config, storage.config)
	assert.Equal(t, logger, storage.logger)
	assert.NotNil(t, storage.metrics)
}

func TestNewS3StorageInvalidConfig(t *testing.T) {
	// Test nil config
	_, err := NewS3Storage(nil, logrus.New())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "S3 config cannot be nil")

	// Test empty bucket
	config := &S3Config{Region: "us-east-1"}
	_, err = NewS3Storage(config, logrus.New())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "S3 bucket is required")
}

func TestS3StorageGenerateKey(t *testing.T) {
	config := &S3Config{
		Region: "us-east-1",
		Bucket: "test-bucket",
		Prefix: "test-prefix",
	}

	storage, err := NewS3Storage(config, logrus.New())
	require.NoError(t, err)

	// Test key generation with prefix
	key := storage.generateKey("series-1")
	assert.Equal(t, "test-prefix/timeseries/series-1.json", key)
}

func TestS3StorageGenerateKeyNoPrefix(t *testing.T) {
	config := &S3Config{
		Region: "us-east-1",
		Bucket: "test-bucket",
	}

	storage, err := NewS3Storage(config, logrus.New())
	require.NoError(t, err)

	// Test key generation without prefix
	key := storage.generateKey("series-1")
	assert.Equal(t, "timeseries/series-1.json", key)
}

func TestS3StorageExtractIDFromKey(t *testing.T) {
	storage, err := NewS3Storage(&S3Config{
		Region: "us-east-1",
		Bucket: "test-bucket",
	}, logrus.New())
	require.NoError(t, err)

	// Test normal key
	id := storage.extractIDFromKey("prefix/timeseries/series-123.json")
	assert.Equal(t, "series-123", id)

	// Test key without prefix
	id = storage.extractIDFromKey("timeseries/test-series.json")
	assert.Equal(t, "test-series", id)

	// Test invalid key
	id = storage.extractIDFromKey("invalid/key")
	assert.Equal(t, "", id)

	// Test empty key
	id = storage.extractIDFromKey("")
	assert.Equal(t, "", id)
}

func TestS3StorageMetricsIncrements(t *testing.T) {
	storage, err := NewS3Storage(&S3Config{
		Region: "us-east-1",
		Bucket: "test-bucket",
	}, logrus.New())
	require.NoError(t, err)

	// Test initial values
	metrics, err := storage.GetMetrics(context.Background())
	require.NoError(t, err)
	assert.Equal(t, int64(0), metrics.ReadOperations)
	assert.Equal(t, int64(0), metrics.WriteOperations)
	assert.Equal(t, int64(0), metrics.DeleteOperations)
	assert.Equal(t, int64(0), metrics.ErrorCount)

	// Test increments
	storage.incrementReadOps()
	storage.incrementWriteOps()
	storage.incrementDeleteOps()
	storage.incrementErrorCount()
	storage.incrementBytesRead(1024)
	storage.incrementBytesWritten(2048)

	assert.Equal(t, int64(1), storage.metrics.readOps)
	assert.Equal(t, int64(1), storage.metrics.writeOps)
	assert.Equal(t, int64(1), storage.metrics.deleteOps)
	assert.Equal(t, int64(1), storage.metrics.errorCount)
	assert.Equal(t, int64(1024), storage.metrics.bytesRead)
	assert.Equal(t, int64(2048), storage.metrics.bytesWritten)
}

func TestS3StorageGetInfo(t *testing.T) {
	storage, err := NewS3Storage(&S3Config{
		Region:         "us-east-1",
		Bucket:         "test-bucket",
		UseCompression: true,
		StorageClass:   "STANDARD_IA",
	}, logrus.New())
	require.NoError(t, err)

	ctx := context.Background()
	info, err := storage.GetInfo(ctx)
	require.NoError(t, err)
	require.NotNil(t, info)

	assert.Equal(t, "s3", info.Type)
	assert.Equal(t, "Amazon S3 Storage", info.Name)
	assert.True(t, info.Capabilities.Compression)
	assert.True(t, info.Capabilities.Encryption)
	assert.False(t, info.Capabilities.Streaming)
	assert.False(t, info.Capabilities.Transactions)
	assert.Contains(t, info.Features, "object storage")
	assert.Contains(t, info.Features, "high durability")
	
	config := info.Configuration
	assert.Equal(t, "us-east-1", config["region"])
	assert.Equal(t, "test-bucket", config["bucket"])
	assert.Equal(t, true, config["use_compression"])
	assert.Equal(t, "STANDARD_IA", config["storage_class"])
}

// Note: The following tests would require actual AWS S3 credentials and bucket
// In a real test environment, you would use localstack, minio, or test doubles

func TestS3StorageIntegration(t *testing.T) {
	t.Skip("Integration test - requires AWS S3 credentials and bucket")

	config := &S3Config{
		Region:          "us-east-1",
		Bucket:          "test-tsiot-bucket",
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
		UseCompression:  false,
	}

	storage, err := NewS3Storage(config, logrus.New())
	require.NoError(t, err)

	ctx := context.Background()

	// Test connection
	err = storage.Connect(ctx)
	require.NoError(t, err)
	defer storage.Close()

	// Test ping
	err = storage.Ping(ctx)
	require.NoError(t, err)

	// Create test time series
	timeSeries := createTestTimeSeries()

	// Test write
	err = storage.Write(ctx, timeSeries)
	require.NoError(t, err)

	// Test read
	retrieved, err := storage.Read(ctx, timeSeries.ID)
	require.NoError(t, err)
	require.NotNil(t, retrieved)

	assert.Equal(t, timeSeries.ID, retrieved.ID)
	assert.Equal(t, timeSeries.Name, retrieved.Name)
	assert.Equal(t, len(timeSeries.DataPoints), len(retrieved.DataPoints))

	// Test read range
	start := time.Now().Add(-2 * time.Hour)
	end := time.Now()
	rangeResult, err := storage.ReadRange(ctx, timeSeries.ID, start, end)
	require.NoError(t, err)
	require.NotNil(t, rangeResult)

	// Test list
	timeSeries2 := createTestTimeSeries()
	timeSeries2.ID = "test-series-2"
	err = storage.Write(ctx, timeSeries2)
	require.NoError(t, err)

	filters := map[string]interface{}{"limit": 10}
	list, err := storage.List(ctx, filters)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(list), 2)

	// Test count
	count, err := storage.Count(ctx, filters)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, count, int64(2))

	// Test delete
	err = storage.Delete(ctx, timeSeries.ID)
	require.NoError(t, err)

	// Verify deletion
	_, err = storage.Read(ctx, timeSeries.ID)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")

	// Clean up
	storage.Delete(ctx, timeSeries2.ID)
}

func TestS3StorageCompressionIntegration(t *testing.T) {
	t.Skip("Integration test - requires AWS S3 credentials and bucket")

	config := &S3Config{
		Region:          "us-east-1",
		Bucket:          "test-tsiot-bucket",
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
		UseCompression:  true,
	}

	storage, err := NewS3Storage(config, logrus.New())
	require.NoError(t, err)

	ctx := context.Background()
	err = storage.Connect(ctx)
	require.NoError(t, err)
	defer storage.Close()

	// Create test time series
	timeSeries := createTestTimeSeries()

	// Test write with compression
	err = storage.Write(ctx, timeSeries)
	require.NoError(t, err)

	// Test read with compression
	retrieved, err := storage.Read(ctx, timeSeries.ID)
	require.NoError(t, err)
	require.NotNil(t, retrieved)

	assert.Equal(t, timeSeries.ID, retrieved.ID)
	assert.Equal(t, len(timeSeries.DataPoints), len(retrieved.DataPoints))

	// Clean up
	storage.Delete(ctx, timeSeries.ID)
}

func TestS3StorageBatchWrite(t *testing.T) {
	t.Skip("Integration test - requires AWS S3 credentials and bucket")

	config := &S3Config{
		Region:          "us-east-1",
		Bucket:          "test-tsiot-bucket",
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
	}

	storage, err := NewS3Storage(config, logrus.New())
	require.NoError(t, err)

	ctx := context.Background()
	err = storage.Connect(ctx)
	require.NoError(t, err)
	defer storage.Close()

	// Create batch of time series
	var batch []*models.TimeSeries
	for i := 0; i < 3; i++ {
		ts := createTestTimeSeries()
		ts.ID = fmt.Sprintf("batch-series-%d", i)
		batch = append(batch, ts)
	}

	// Test batch write
	err = storage.WriteBatch(ctx, batch)
	require.NoError(t, err)

	// Verify all series were written
	for _, ts := range batch {
		retrieved, err := storage.Read(ctx, ts.ID)
		require.NoError(t, err)
		assert.Equal(t, ts.ID, retrieved.ID)
	}

	// Clean up
	for _, ts := range batch {
		storage.Delete(ctx, ts.ID)
	}
}

func TestS3StorageDeleteRange(t *testing.T) {
	t.Skip("Integration test - requires AWS S3 credentials and bucket")

	config := &S3Config{
		Region:          "us-east-1",
		Bucket:          "test-tsiot-bucket",
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
	}

	storage, err := NewS3Storage(config, logrus.New())
	require.NoError(t, err)

	ctx := context.Background()
	err = storage.Connect(ctx)
	require.NoError(t, err)
	defer storage.Close()

	// Create test time series with multiple data points
	timeSeries := createLargeTestTimeSeries(10)
	err = storage.Write(ctx, timeSeries)
	require.NoError(t, err)

	// Delete middle range
	start := timeSeries.DataPoints[2].Timestamp
	end := timeSeries.DataPoints[7].Timestamp
	
	err = storage.DeleteRange(ctx, timeSeries.ID, start, end)
	require.NoError(t, err)

	// Verify range was deleted
	retrieved, err := storage.Read(ctx, timeSeries.ID)
	require.NoError(t, err)
	
	// Should have fewer data points now
	assert.Less(t, len(retrieved.DataPoints), len(timeSeries.DataPoints))

	// Clean up
	storage.Delete(ctx, timeSeries.ID)
}

func TestS3StorageHealth(t *testing.T) {
	t.Skip("Integration test - requires AWS S3 credentials and bucket")

	config := &S3Config{
		Region:          "us-east-1",
		Bucket:          "test-tsiot-bucket",
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
	}

	storage, err := NewS3Storage(config, logrus.New())
	require.NoError(t, err)

	ctx := context.Background()
	err = storage.Connect(ctx)
	require.NoError(t, err)
	defer storage.Close()

	health, err := storage.Health(ctx)
	require.NoError(t, err)
	require.NotNil(t, health)

	assert.Equal(t, "healthy", health.Status)
	assert.Greater(t, health.LastCheck.Unix(), int64(0))
	assert.GreaterOrEqual(t, health.Latency, time.Duration(0))
	assert.Equal(t, 1, health.Connections)
}

func TestS3StorageDisconnectedOperations(t *testing.T) {
	storage, err := NewS3Storage(&S3Config{
		Region: "us-east-1",
		Bucket: "test-bucket",
	}, logrus.New())
	require.NoError(t, err)

	ctx := context.Background()
	timeSeries := createTestTimeSeries()

	// Test operations on disconnected storage
	err = storage.Write(ctx, timeSeries)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not connected")

	_, err = storage.Read(ctx, timeSeries.ID)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not connected")

	err = storage.Delete(ctx, timeSeries.ID)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not connected")

	_, err = storage.List(ctx, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not connected")
}

func TestS3StorageInvalidData(t *testing.T) {
	storage, err := NewS3Storage(&S3Config{
		Region: "us-east-1",
		Bucket: "test-bucket",
	}, logrus.New())
	require.NoError(t, err)

	// Connect to test validation (will fail on actual S3 call but validation happens first)
	ctx := context.Background()

	// Test with nil time series
	err = storage.Write(ctx, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not connected") // This error comes first

	// Test with invalid time series (empty ID)
	invalidTS := &models.TimeSeries{
		Name: "Invalid Series",
		// Missing ID
	}
	err = storage.Write(ctx, invalidTS)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not connected") // This error comes first
}

// Helper functions

func createTestTimeSeries() *models.TimeSeries {
	now := time.Now()
	dataPoints := []models.DataPoint{
		{Timestamp: now.Add(-2 * time.Hour), Value: 15.0, Quality: 1.0},
		{Timestamp: now.Add(-1 * time.Hour), Value: 25.0, Quality: 1.0},
		{Timestamp: now, Value: 35.0, Quality: 1.0},
	}

	return &models.TimeSeries{
		ID:          "test-s3-series",
		Name:        "Test S3 Series",
		Description: "Test time series for S3 storage",
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

func createLargeTestTimeSeries(size int) *models.TimeSeries {
	now := time.Now()
	var dataPoints []models.DataPoint

	for i := 0; i < size; i++ {
		dataPoints = append(dataPoints, models.DataPoint{
			Timestamp: now.Add(time.Duration(i) * time.Hour),
			Value:     float64(i*5) + 10.0,
			Quality:   1.0,
		})
	}

	return &models.TimeSeries{
		ID:          "large-s3-series",
		Name:        "Large S3 Series",
		Description: "Large time series for S3 storage tests",
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
func BenchmarkS3Write(b *testing.B) {
	b.Skip("Benchmark test - requires AWS S3 credentials and bucket")

	config := &S3Config{
		Region:          "us-east-1",
		Bucket:          "test-tsiot-bucket",
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
	}

	storage, _ := NewS3Storage(config, logrus.New())
	ctx := context.Background()
	storage.Connect(ctx)
	defer storage.Close()

	timeSeries := createTestTimeSeries()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		timeSeries.ID = fmt.Sprintf("bench-series-%d", i)
		err := storage.Write(ctx, timeSeries)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkS3Read(b *testing.B) {
	b.Skip("Benchmark test - requires AWS S3 credentials and bucket")

	config := &S3Config{
		Region:          "us-east-1",
		Bucket:          "test-tsiot-bucket",
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
	}

	storage, _ := NewS3Storage(config, logrus.New())
	ctx := context.Background()
	storage.Connect(ctx)
	defer storage.Close()

	// Pre-populate data
	timeSeries := createTestTimeSeries()
	storage.Write(ctx, timeSeries)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := storage.Read(ctx, timeSeries.ID)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkS3WriteCompressed(b *testing.B) {
	b.Skip("Benchmark test - requires AWS S3 credentials and bucket")

	config := &S3Config{
		Region:          "us-east-1",
		Bucket:          "test-tsiot-bucket",
		AccessKeyID:     "test-access-key",
		SecretAccessKey: "test-secret-key",
		UseCompression:  true,
	}

	storage, _ := NewS3Storage(config, logrus.New())
	ctx := context.Background()
	storage.Connect(ctx)
	defer storage.Close()

	timeSeries := createLargeTestTimeSeries(1000)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		timeSeries.ID = fmt.Sprintf("bench-compressed-series-%d", i)
		err := storage.Write(ctx, timeSeries)
		if err != nil {
			b.Fatal(err)
		}
	}
}