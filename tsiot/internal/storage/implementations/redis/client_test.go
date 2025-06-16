package redis

import (
	"context"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inferloop/tsiot/pkg/models"
)

func TestNewRedisStorage(t *testing.T) {
	config := &RedisConfig{
		Addr:     "localhost:6379",
		Password: "",
		DB:       0,
	}

	logger := logrus.New()
	storage, err := NewRedisStorage(config, logger)
	
	require.NoError(t, err)
	require.NotNil(t, storage)
	assert.Equal(t, config, storage.config)
	assert.Equal(t, logger, storage.logger)
}

func TestNewRedisStorageInvalidConfig(t *testing.T) {
	// Test nil config
	_, err := NewRedisStorage(nil, logrus.New())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "config cannot be nil")

	// Test empty address
	config := &RedisConfig{}
	_, err = NewRedisStorage(config, logrus.New())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "address or cluster addresses are required")
}

func TestRedisStorageGenerateKeys(t *testing.T) {
	config := &RedisConfig{
		Addr:      "localhost:6379",
		KeyPrefix: "test",
	}

	storage, err := NewRedisStorage(config, logrus.New())
	require.NoError(t, err)

	// Test metadata key generation
	metadataKey := storage.generateMetadataKey("series-1")
	assert.Equal(t, "test:metadata:series-1", metadataKey)

	// Test data key generation
	dataKey := storage.generateDataKey("series-1")
	assert.Equal(t, "test:data:series-1", dataKey)

	// Test stream key generation
	streamKey := storage.generateStreamKey("series-1")
	assert.Equal(t, "test:stream:series-1", streamKey)
}

func TestRedisStorageGenerateKeysNoPrefix(t *testing.T) {
	config := &RedisConfig{
		Addr: "localhost:6379",
	}

	storage, err := NewRedisStorage(config, logrus.New())
	require.NoError(t, err)

	// Test without prefix
	metadataKey := storage.generateMetadataKey("series-1")
	assert.Equal(t, "metadata:series-1", metadataKey)

	dataKey := storage.generateDataKey("series-1")
	assert.Equal(t, "data:series-1", dataKey)

	streamKey := storage.generateStreamKey("series-1")
	assert.Equal(t, "stream:series-1", streamKey)
}

func TestRedisStorageMetricsIncrements(t *testing.T) {
	storage, err := NewRedisStorage(&RedisConfig{Addr: "localhost:6379"}, logrus.New())
	require.NoError(t, err)

	// Test initial values
	assert.Equal(t, int64(0), storage.metrics.readOps)
	assert.Equal(t, int64(0), storage.metrics.writeOps)
	assert.Equal(t, int64(0), storage.metrics.deleteOps)
	assert.Equal(t, int64(0), storage.metrics.errorCount)

	// Test increments
	storage.incrementReadOps()
	storage.incrementWriteOps()
	storage.incrementDeleteOps()
	storage.incrementErrorCount()

	assert.Equal(t, int64(1), storage.metrics.readOps)
	assert.Equal(t, int64(1), storage.metrics.writeOps)
	assert.Equal(t, int64(1), storage.metrics.deleteOps)
	assert.Equal(t, int64(1), storage.metrics.errorCount)
}

// Note: The following tests would require a running Redis instance
// In a real test environment, you would use Docker containers or test doubles

func TestRedisStorageIntegration(t *testing.T) {
	t.Skip("Integration test - requires running Redis instance")

	config := &RedisConfig{
		Addr:     "localhost:6379",
		DB:       15, // Use test database
		TTL:      1 * time.Hour,
		UseStreams: false,
	}

	storage, err := NewRedisStorage(config, logrus.New())
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

func TestRedisStorageStreamsIntegration(t *testing.T) {
	t.Skip("Integration test - requires running Redis instance")

	config := &RedisConfig{
		Addr:         "localhost:6379",
		DB:           15, // Use test database
		UseStreams:   true,
		StreamMaxLen: 1000,
	}

	storage, err := NewRedisStorage(config, logrus.New())
	require.NoError(t, err)

	ctx := context.Background()
	err = storage.Connect(ctx)
	require.NoError(t, err)
	defer storage.Close()

	// Create test time series
	timeSeries := createTestTimeSeries()

	// Test write with streams
	err = storage.Write(ctx, timeSeries)
	require.NoError(t, err)

	// Test read with streams
	retrieved, err := storage.Read(ctx, timeSeries.ID)
	require.NoError(t, err)
	require.NotNil(t, retrieved)

	assert.Equal(t, timeSeries.ID, retrieved.ID)
	assert.Equal(t, len(timeSeries.DataPoints), len(retrieved.DataPoints))

	// Clean up
	storage.Delete(ctx, timeSeries.ID)
}

func TestRedisStorageBatchWrite(t *testing.T) {
	t.Skip("Integration test - requires running Redis instance")

	config := &RedisConfig{
		Addr: "localhost:6379",
		DB:   15,
	}

	storage, err := NewRedisStorage(config, logrus.New())
	require.NoError(t, err)

	ctx := context.Background()
	err = storage.Connect(ctx)
	require.NoError(t, err)
	defer storage.Close()

	// Create batch of time series
	var batch []*models.TimeSeries
	for i := 0; i < 5; i++ {
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

func TestRedisStorageGetInfo(t *testing.T) {
	storage, err := NewRedisStorage(&RedisConfig{Addr: "localhost:6379"}, logrus.New())
	require.NoError(t, err)

	ctx := context.Background()
	info, err := storage.GetInfo(ctx)
	require.NoError(t, err)
	require.NotNil(t, info)

	assert.Equal(t, "redis", info.Type)
	assert.Equal(t, "Redis Storage", info.Name)
	assert.True(t, info.Capabilities.Streaming)
	assert.True(t, info.Capabilities.Transactions)
	assert.Contains(t, info.Features, "in-memory storage")
}

func TestRedisStorageHealth(t *testing.T) {
	t.Skip("Integration test - requires running Redis instance")

	config := &RedisConfig{
		Addr: "localhost:6379",
		DB:   15,
	}

	storage, err := NewRedisStorage(config, logrus.New())
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
}

func TestRedisStorageGetMetrics(t *testing.T) {
	storage, err := NewRedisStorage(&RedisConfig{Addr: "localhost:6379"}, logrus.New())
	require.NoError(t, err)

	ctx := context.Background()
	metrics, err := storage.GetMetrics(ctx)
	require.NoError(t, err)
	require.NotNil(t, metrics)

	assert.Equal(t, int64(0), metrics.ReadOperations)
	assert.Equal(t, int64(0), metrics.WriteOperations)
	assert.Equal(t, int64(0), metrics.DeleteOperations)
	assert.Equal(t, int64(0), metrics.ErrorCount)
}

// Helper functions

func createTestTimeSeries() *models.TimeSeries {
	now := time.Now()
	dataPoints := []models.DataPoint{
		{Timestamp: now.Add(-2 * time.Hour), Value: 10.0, Quality: 1.0},
		{Timestamp: now.Add(-1 * time.Hour), Value: 20.0, Quality: 1.0},
		{Timestamp: now, Value: 30.0, Quality: 1.0},
	}

	return &models.TimeSeries{
		ID:          "test-redis-series",
		Name:        "Test Redis Series",
		Description: "Test time series for Redis storage",
		SensorType:  "temperature",
		DataPoints:  dataPoints,
		StartTime:   dataPoints[0].Timestamp,
		EndTime:     dataPoints[len(dataPoints)-1].Timestamp,
		Frequency:   "1h",
		Tags: map[string]string{
			"location": "server-room",
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

// Benchmark tests
func BenchmarkRedisWrite(b *testing.B) {
	b.Skip("Benchmark test - requires running Redis instance")

	config := &RedisConfig{
		Addr: "localhost:6379",
		DB:   15,
	}

	storage, _ := NewRedisStorage(config, logrus.New())
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

func BenchmarkRedisRead(b *testing.B) {
	b.Skip("Benchmark test - requires running Redis instance")

	config := &RedisConfig{
		Addr: "localhost:6379",
		DB:   15,
	}

	storage, _ := NewRedisStorage(config, logrus.New())
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