package clickhouse

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

func TestNewClickHouseStorage(t *testing.T) {
	config := &ClickHouseStorageConfig{
		Host:     "localhost",
		Port:     9000,
		Database: "test_db",
		Username: "default",
		Password: "",
	}

	logger := logrus.New()
	storage, err := NewClickHouseStorage(config, logger)
	
	require.NoError(t, err)
	require.NotNil(t, storage)
	assert.Equal(t, config, storage.config)
	assert.Equal(t, logger, storage.logger)
	assert.False(t, storage.connected)
}

func TestNewClickHouseStorageNilConfig(t *testing.T) {
	_, err := NewClickHouseStorage(nil, logrus.New())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "ClickHouseStorageConfig cannot be nil")
}

func TestClickHouseStorageConfigDefaults(t *testing.T) {
	// Test with minimal config
	config := &ClickHouseStorageConfig{}
	
	storage, err := NewClickHouseStorage(config, nil)
	require.NoError(t, err)
	
	// Check that defaults were applied
	assert.Equal(t, "localhost", config.Host)
	assert.Equal(t, 9000, config.Port)
	assert.Equal(t, "default", config.Database)
	assert.Equal(t, 10*time.Second, config.ConnectTimeout)
	assert.Equal(t, 30*time.Second, config.QueryTimeout)
	assert.Equal(t, 10, config.MaxConnections)
	assert.Equal(t, 10000, config.BatchSize)
	assert.Equal(t, 30*time.Second, config.FlushInterval)
	assert.Equal(t, "MergeTree", config.Engine)
	assert.Equal(t, "(series_id, timestamp)", config.OrderBy)
	assert.Equal(t, "toYYYYMM(timestamp)", config.PartitionBy)
}

func TestClickHouseStorageSerializeTags(t *testing.T) {
	storage, err := NewClickHouseStorage(&ClickHouseStorageConfig{}, logrus.New())
	require.NoError(t, err)

	// Test empty tags
	result := storage.serializeTags(nil)
	assert.Equal(t, "", result)

	result = storage.serializeTags(map[string]string{})
	assert.Equal(t, "", result)

	// Test with tags
	tags := map[string]string{
		"location": "datacenter",
		"type":     "sensor",
	}
	result = storage.serializeTags(tags)
	
	// Should contain both tags (order may vary)
	assert.Contains(t, result, "location=datacenter")
	assert.Contains(t, result, "type=sensor")
	assert.Contains(t, result, ",")
}

func TestClickHouseStorageSerializeMetadata(t *testing.T) {
	storage, err := NewClickHouseStorage(&ClickHouseStorageConfig{}, logrus.New())
	require.NoError(t, err)

	metadata := models.Metadata{
		Generator: "test-generator",
		Version:   "1.0",
	}
	
	result := storage.serializeMetadata(metadata)
	expected := "generator=test-generator,version=1.0"
	assert.Equal(t, expected, result)
}

func TestClickHouseStorageGetTTLClause(t *testing.T) {
	// Test without TTL
	storage, err := NewClickHouseStorage(&ClickHouseStorageConfig{}, logrus.New())
	require.NoError(t, err)

	ttlClause := storage.getTTLClause()
	assert.Equal(t, "", ttlClause)

	// Test with TTL
	config := &ClickHouseStorageConfig{
		TTL: "30 DAY",
	}
	storage, err = NewClickHouseStorage(config, logrus.New())
	require.NoError(t, err)

	ttlClause = storage.getTTLClause()
	assert.Equal(t, "TTL timestamp + INTERVAL 30 DAY", ttlClause)
}

func TestClickHouseStorageNotConnectedOperations(t *testing.T) {
	storage, err := NewClickHouseStorage(&ClickHouseStorageConfig{}, logrus.New())
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

	_, err = storage.List(ctx, 10, 0)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not connected")

	err = storage.HealthCheck(ctx)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not connected")

	_, err = storage.GetStats(ctx)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not connected")
}

func TestClickHouseStorageWriteValidation(t *testing.T) {
	storage, err := NewClickHouseStorage(&ClickHouseStorageConfig{}, logrus.New())
	require.NoError(t, err)

	ctx := context.Background()

	// Test with nil time series
	err = storage.Write(ctx, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "TimeSeries cannot be nil")
}

func TestClickHouseStorageAddToBatch(t *testing.T) {
	config := &ClickHouseStorageConfig{
		BatchSize: 2,
	}
	storage, err := NewClickHouseStorage(config, logrus.New())
	require.NoError(t, err)

	timeSeries := createTestTimeSeries()

	// Add to batch (should not trigger flush yet)
	err = storage.addToBatch(timeSeries)
	require.NoError(t, err)
	assert.Equal(t, 1, len(storage.pendingBatch))

	// Add another (should trigger auto-flush due to batch size)
	err = storage.addToBatch(timeSeries)
	require.NoError(t, err)
	
	// Give some time for async flush
	time.Sleep(100 * time.Millisecond)
}

func TestClickHouseStorageCleanup(t *testing.T) {
	storage, err := NewClickHouseStorage(&ClickHouseStorageConfig{}, logrus.New())
	require.NoError(t, err)

	// Test cleanup without connections (should not panic)
	storage.cleanup()
	assert.Nil(t, storage.conn)
	assert.Nil(t, storage.db)
}

// Note: The following tests would require a running ClickHouse instance
// In a real test environment, you would use Docker containers or test doubles

func TestClickHouseStorageIntegration(t *testing.T) {
	t.Skip("Integration test - requires running ClickHouse instance")

	config := &ClickHouseStorageConfig{
		Host:     "localhost",
		Port:     9000,
		Database: "test_tsiot",
		Username: "default",
		Password: "",
	}

	storage, err := NewClickHouseStorage(config, logrus.New())
	require.NoError(t, err)

	ctx := context.Background()

	// Test connection
	err = storage.Connect(ctx)
	require.NoError(t, err)
	defer storage.Close()

	// Test health check
	err = storage.HealthCheck(ctx)
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

	// Test list
	list, err := storage.List(ctx, 10, 0)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(list), 1)

	// Test stats
	stats, err := storage.GetStats(ctx)
	require.NoError(t, err)
	require.NotNil(t, stats)
	assert.Greater(t, stats["total_rows"], int64(0))

	// Test query
	rows, err := storage.Query(ctx, "SELECT count() FROM time_series_data WHERE series_id = ?", timeSeries.ID)
	require.NoError(t, err)
	defer rows.Close()

	var count int64
	if rows.Next() {
		rows.Scan(&count)
	}
	assert.Greater(t, count, int64(0))

	// Test delete
	err = storage.Delete(ctx, timeSeries.ID)
	require.NoError(t, err)

	// Note: ClickHouse deletes are asynchronous, so we can't immediately verify
}

func TestClickHouseStorageBatchProcessing(t *testing.T) {
	t.Skip("Integration test - requires running ClickHouse instance")

	config := &ClickHouseStorageConfig{
		Host:            "localhost",
		Port:            9000,
		Database:        "test_tsiot",
		Username:        "default",
		Password:        "",
		BatchSize:       3,
		UseAsyncInserts: true,
	}

	storage, err := NewClickHouseStorage(config, logrus.New())
	require.NoError(t, err)

	ctx := context.Background()
	err = storage.Connect(ctx)
	require.NoError(t, err)
	defer storage.Close()

	// Write multiple time series to trigger batching
	for i := 0; i < 5; i++ {
		timeSeries := createTestTimeSeries()
		timeSeries.ID = fmt.Sprintf("batch-series-%d", i)
		
		err = storage.Write(ctx, timeSeries)
		require.NoError(t, err)
	}

	// Wait for batch processing
	time.Sleep(2 * time.Second)

	// Verify data was written
	list, err := storage.List(ctx, 10, 0)
	require.NoError(t, err)
	assert.GreaterOrEqual(t, len(list), 5)

	// Clean up
	for i := 0; i < 5; i++ {
		storage.Delete(ctx, fmt.Sprintf("batch-series-%d", i))
	}
}

func TestClickHouseStorageCompressionSettings(t *testing.T) {
	t.Skip("Integration test - requires running ClickHouse instance")

	// Test with LZ4 compression
	config := &ClickHouseStorageConfig{
		Host:        "localhost",
		Port:        9000,
		Database:    "test_tsiot",
		Username:    "default",
		Password:    "",
		Compression: "lz4",
	}

	storage, err := NewClickHouseStorage(config, logrus.New())
	require.NoError(t, err)

	ctx := context.Background()
	err = storage.Connect(ctx)
	require.NoError(t, err)
	defer storage.Close()

	timeSeries := createTestTimeSeries()
	err = storage.Write(ctx, timeSeries)
	require.NoError(t, err)

	// Clean up
	storage.Delete(ctx, timeSeries.ID)
}

func TestClickHouseStorageReplicationSettings(t *testing.T) {
	t.Skip("Integration test - requires ClickHouse cluster")

	config := &ClickHouseStorageConfig{
		Host:              "localhost",
		Port:              9000,
		Database:          "test_tsiot",
		Username:          "default",
		Password:          "",
		Engine:            "ReplicatedMergeTree",
		ReplicationFactor: 2,
	}

	storage, err := NewClickHouseStorage(config, logrus.New())
	require.NoError(t, err)

	ctx := context.Background()
	err = storage.Connect(ctx)
	require.NoError(t, err)
	defer storage.Close()
}

func TestClickHouseStorageCustomPartitioning(t *testing.T) {
	t.Skip("Integration test - requires running ClickHouse instance")

	config := &ClickHouseStorageConfig{
		Host:        "localhost",
		Port:        9000,
		Database:    "test_tsiot",
		Username:    "default",
		Password:    "",
		PartitionBy: "toYYYYMMDD(timestamp)",
		OrderBy:     "(series_id, timestamp, value)",
	}

	storage, err := NewClickHouseStorage(config, logrus.New())
	require.NoError(t, err)

	ctx := context.Background()
	err = storage.Connect(ctx)
	require.NoError(t, err)
	defer storage.Close()
}

func TestClickHouseStoragePerformanceSettings(t *testing.T) {
	config := &ClickHouseStorageConfig{
		MaxBlockSize:     65536,
		MaxInsertThreads: 4,
		UseAsyncInserts:  true,
	}

	storage, err := NewClickHouseStorage(config, logrus.New())
	require.NoError(t, err)

	assert.Equal(t, 65536, storage.config.MaxBlockSize)
	assert.Equal(t, 4, storage.config.MaxInsertThreads)
	assert.True(t, storage.config.UseAsyncInserts)
}

// Helper functions

func createTestTimeSeries() *models.TimeSeries {
	now := time.Now()
	dataPoints := []models.DataPoint{
		{Timestamp: now.Add(-2 * time.Hour), Value: 20.0, Quality: 1.0},
		{Timestamp: now.Add(-1 * time.Hour), Value: 25.0, Quality: 1.0},
		{Timestamp: now, Value: 30.0, Quality: 1.0},
	}

	return &models.TimeSeries{
		ID:          "test-clickhouse-series",
		Name:        "Test ClickHouse Series",
		Description: "Test time series for ClickHouse storage",
		SensorType:  "temperature",
		DataPoints:  dataPoints,
		StartTime:   dataPoints[0].Timestamp,
		EndTime:     dataPoints[len(dataPoints)-1].Timestamp,
		Frequency:   "1h",
		Tags: map[string]string{
			"location": "server-room",
			"type":     "sensor",
		},
		Metadata: models.Metadata{
			Generator: "test-generator",
			Version:   "1.0",
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
			Value:     float64(i%100) + 20.0,
			Quality:   1.0,
		})
	}

	return &models.TimeSeries{
		ID:          "large-clickhouse-series",
		Name:        "Large ClickHouse Series",
		Description: "Large time series for ClickHouse tests",
		SensorType:  "temperature",
		DataPoints:  dataPoints,
		StartTime:   dataPoints[0].Timestamp,
		EndTime:     dataPoints[len(dataPoints)-1].Timestamp,
		Frequency:   "1m",
		Metadata: models.Metadata{
			Generator: "test-generator",
			Version:   "1.0",
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
}

// Benchmark tests
func BenchmarkClickHouseWrite(b *testing.B) {
	b.Skip("Benchmark test - requires running ClickHouse instance")

	config := &ClickHouseStorageConfig{
		Host:     "localhost",
		Port:     9000,
		Database: "test_tsiot",
		Username: "default",
	}

	storage, _ := NewClickHouseStorage(config, logrus.New())
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

func BenchmarkClickHouseBatchWrite(b *testing.B) {
	b.Skip("Benchmark test - requires running ClickHouse instance")

	config := &ClickHouseStorageConfig{
		Host:            "localhost",
		Port:            9000,
		Database:        "test_tsiot",
		Username:        "default",
		BatchSize:       1000,
		UseAsyncInserts: true,
	}

	storage, _ := NewClickHouseStorage(config, logrus.New())
	ctx := context.Background()
	storage.Connect(ctx)
	defer storage.Close()

	timeSeries := createLargeTestTimeSeries(100)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		timeSeries.ID = fmt.Sprintf("bench-batch-series-%d", i)
		err := storage.Write(ctx, timeSeries)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkClickHouseRead(b *testing.B) {
	b.Skip("Benchmark test - requires running ClickHouse instance")

	config := &ClickHouseStorageConfig{
		Host:     "localhost",
		Port:     9000,
		Database: "test_tsiot",
		Username: "default",
	}

	storage, _ := NewClickHouseStorage(config, logrus.New())
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

func BenchmarkClickHouseQuery(b *testing.B) {
	b.Skip("Benchmark test - requires running ClickHouse instance")

	config := &ClickHouseStorageConfig{
		Host:     "localhost",
		Port:     9000,
		Database: "test_tsiot",
		Username: "default",
	}

	storage, _ := NewClickHouseStorage(config, logrus.New())
	ctx := context.Background()
	storage.Connect(ctx)
	defer storage.Close()

	query := "SELECT count() FROM time_series_data WHERE timestamp >= now() - INTERVAL 1 HOUR"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rows, err := storage.Query(ctx, query)
		if err != nil {
			b.Fatal(err)
		}
		rows.Close()
	}
}