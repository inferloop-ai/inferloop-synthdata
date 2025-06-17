package redis

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
	"github.com/inferloop/tsiot/pkg/errors"
)

// RedisConfig holds configuration for Redis storage
type RedisConfig struct {
	Addr            string        `json:"addr"`
	Password        string        `json:"password"`
	DB              int           `json:"db"`
	DialTimeout     time.Duration `json:"dial_timeout"`
	ReadTimeout     time.Duration `json:"read_timeout"`
	WriteTimeout    time.Duration `json:"write_timeout"`
	PoolSize        int           `json:"pool_size"`
	MinIdleConns    int           `json:"min_idle_conns"`
	MaxRetries      int           `json:"max_retries"`
	RetryBackoff    time.Duration `json:"retry_backoff"`
	IdleTimeout     time.Duration `json:"idle_timeout"`
	TTL             time.Duration `json:"ttl"`
	KeyPrefix       string        `json:"key_prefix"`
	UseStreams      bool          `json:"use_streams"`
	StreamMaxLen    int64         `json:"stream_max_len"`
	UseClustering   bool          `json:"use_clustering"`
	ClusterAddrs    []string      `json:"cluster_addrs"`
}

// RedisStorage implements the Storage interface for Redis
type RedisStorage struct {
	config      *RedisConfig
	client      redis.UniversalClient
	logger      *logrus.Logger
	mu          sync.RWMutex
	metrics     *storageMetrics
	closed      bool
}

type storageMetrics struct {
	readOps      int64
	writeOps     int64
	deleteOps    int64
	errorCount   int64
	hitCount     int64
	missCount    int64
	startTime    time.Time
	mu           sync.RWMutex
}

// NewRedisStorage creates a new Redis storage instance
func NewRedisStorage(config *RedisConfig, logger *logrus.Logger) (*RedisStorage, error) {
	if config == nil {
		return nil, errors.NewStorageError("INVALID_CONFIG", "Redis config cannot be nil")
	}

	if config.Addr == "" && len(config.ClusterAddrs) == 0 {
		return nil, errors.NewStorageError("INVALID_CONFIG", "Redis address or cluster addresses are required")
	}

	if logger == nil {
		logger = logrus.New()
	}

	storage := &RedisStorage{
		config: config,
		logger: logger,
		metrics: &storageMetrics{
			startTime: time.Now(),
		},
	}

	return storage, nil
}

// Connect establishes connection to Redis
func (r *RedisStorage) Connect(ctx context.Context) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.client != nil {
		return nil // Already connected
	}

	var client redis.UniversalClient

	if r.config.UseClustering && len(r.config.ClusterAddrs) > 0 {
		// Redis Cluster
		client = redis.NewClusterClient(&redis.ClusterOptions{
			Addrs:        r.config.ClusterAddrs,
			Password:     r.config.Password,
			DialTimeout:  r.config.DialTimeout,
			ReadTimeout:  r.config.ReadTimeout,
			WriteTimeout: r.config.WriteTimeout,
			PoolSize:     r.config.PoolSize,
			MinIdleConns: r.config.MinIdleConns,
			MaxRetries:   r.config.MaxRetries,
			IdleTimeout:  r.config.IdleTimeout,
		})
	} else {
		// Single Redis instance
		client = redis.NewClient(&redis.Options{
			Addr:         r.config.Addr,
			Password:     r.config.Password,
			DB:           r.config.DB,
			DialTimeout:  r.config.DialTimeout,
			ReadTimeout:  r.config.ReadTimeout,
			WriteTimeout: r.config.WriteTimeout,
			PoolSize:     r.config.PoolSize,
			MinIdleConns: r.config.MinIdleConns,
			MaxRetries:   r.config.MaxRetries,
			IdleTimeout:  r.config.IdleTimeout,
		})
	}

	// Test connection
	_, err := client.Ping(ctx).Result()
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "CONNECTION_FAILED", "Failed to connect to Redis")
	}

	r.client = client

	r.logger.WithFields(logrus.Fields{
		"addr":       r.config.Addr,
		"db":         r.config.DB,
		"clustering": r.config.UseClustering,
	}).Info("Connected to Redis")

	return nil
}

// Close closes the Redis connection
func (r *RedisStorage) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return nil
	}

	if r.client != nil {
		err := r.client.Close()
		r.client = nil
		r.closed = true
		
		if err != nil {
			return errors.WrapError(err, errors.ErrorTypeStorage, "CLOSE_FAILED", "Failed to close Redis connection")
		}
	}

	r.logger.Info("Redis connection closed")
	return nil
}

// Ping tests the Redis connection
func (r *RedisStorage) Ping(ctx context.Context) error {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.closed || r.client == nil {
		return errors.NewStorageError("NOT_CONNECTED", "Redis not connected")
	}

	_, err := r.client.Ping(ctx).Result()
	if err != nil {
		r.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "PING_FAILED", "Redis ping failed")
	}

	return nil
}

// GetInfo returns information about the Redis storage
func (r *RedisStorage) GetInfo(ctx context.Context) (*interfaces.StorageInfo, error) {
	version := "unknown"
	if info, err := r.client.Info(ctx, "server").Result(); err == nil {
		lines := strings.Split(info, "\n")
		for _, line := range lines {
			if strings.HasPrefix(line, "redis_version:") {
				version = strings.TrimPrefix(line, "redis_version:")
				version = strings.TrimSpace(version)
				break
			}
		}
	}

	return &interfaces.StorageInfo{
		Type:        "redis",
		Version:     version,
		Name:        "Redis Storage",
		Description: "In-memory data structure store for caching and real-time applications",
		Features: []string{
			"in-memory storage",
			"high performance",
			"pub/sub messaging",
			"streams",
			"clustering",
			"persistence",
			"atomic operations",
		},
		Capabilities: interfaces.StorageCapabilities{
			Streaming:      true,
			Transactions:   true,
			Compression:    false,
			Encryption:     false,
			Replication:    true,
			Clustering:     r.config.UseClustering,
			Backup:         true,
			Archival:       false,
			TimeBasedQuery: true,
			Aggregation:    false,
		},
		Configuration: map[string]interface{}{
			"addr":         r.config.Addr,
			"db":           r.config.DB,
			"key_prefix":   r.config.KeyPrefix,
			"ttl":          r.config.TTL.String(),
			"use_streams":  r.config.UseStreams,
			"clustering":   r.config.UseClustering,
		},
	}, nil
}

// Health returns the health status of the storage
func (r *RedisStorage) Health(ctx context.Context) (*interfaces.HealthStatus, error) {
	start := time.Now()
	status := "healthy"
	var errors []string
	var warnings []string

	// Test connection
	if err := r.Ping(ctx); err != nil {
		status = "unhealthy"
		errors = append(errors, fmt.Sprintf("Connection failed: %v", err))
	}

	latency := time.Since(start)

	// Check for warnings
	if latency > 50*time.Millisecond {
		warnings = append(warnings, "High latency detected")
	}

	// Get Redis info
	connections := 0
	memoryUsed := int64(0)
	
	if info, err := r.client.Info(ctx, "clients", "memory").Result(); err == nil {
		lines := strings.Split(info, "\n")
		for _, line := range lines {
			if strings.HasPrefix(line, "connected_clients:") {
				if val, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, "connected_clients:"))); err == nil {
					connections = val
				}
			}
			if strings.HasPrefix(line, "used_memory:") {
				if val, err := strconv.ParseInt(strings.TrimSpace(strings.TrimPrefix(line, "used_memory:")), 10, 64); err == nil {
					memoryUsed = val
				}
			}
		}
	}

	return &interfaces.HealthStatus{
		Status:      status,
		LastCheck:   time.Now(),
		Latency:     latency,
		Connections: connections,
		Errors:      errors,
		Warnings:    warnings,
		Metadata: map[string]interface{}{
			"memory_used": memoryUsed,
			"key_count":   r.getKeyCount(ctx),
		},
	}, nil
}

// Write writes a time series to Redis
func (r *RedisStorage) Write(ctx context.Context, data *models.TimeSeries) error {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.closed || r.client == nil {
		return errors.NewStorageError("NOT_CONNECTED", "Redis not connected")
	}

	if err := data.Validate(); err != nil {
		return errors.WrapError(err, errors.ErrorTypeValidation, "INVALID_DATA", "Time series validation failed")
	}

	start := time.Now()
	defer func() {
		r.incrementWriteOps()
		r.logger.WithField("duration", time.Since(start)).Debug("Write operation completed")
	}()

	// Serialize time series metadata
	metadataKey := r.generateMetadataKey(data.ID)
	metadataJSON, err := json.Marshal(data)
	if err != nil {
		r.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "SERIALIZATION_FAILED", "Failed to serialize time series metadata")
	}

	pipe := r.client.Pipeline()

	// Store metadata
	pipe.Set(ctx, metadataKey, metadataJSON, r.config.TTL)

	if r.config.UseStreams {
		// Store data points in Redis Streams
		streamKey := r.generateStreamKey(data.ID)
		
		for _, point := range data.DataPoints {
			fields := map[string]interface{}{
				"value":     point.Value,
				"quality":   point.Quality,
				"timestamp": point.Timestamp.Unix(),
			}
			
			if point.Tags != nil {
				tagsJSON, _ := json.Marshal(point.Tags)
				fields["tags"] = string(tagsJSON)
			}
			
			if point.Metadata != nil {
				metaJSON, _ := json.Marshal(point.Metadata)
				fields["metadata"] = string(metaJSON)
			}

			pipe.XAdd(ctx, &redis.XAddArgs{
				Stream: streamKey,
				MaxLen: r.config.StreamMaxLen,
				Approx: true,
				Values: fields,
			})
		}
	} else {
		// Store data points as sorted set (timestamp as score)
		dataKey := r.generateDataKey(data.ID)
		
		for _, point := range data.DataPoints {
			pointJSON, _ := json.Marshal(point)
			score := float64(point.Timestamp.Unix())
			pipe.ZAdd(ctx, dataKey, &redis.Z{
				Score:  score,
				Member: string(pointJSON),
			})
		}
		
		if r.config.TTL > 0 {
			pipe.Expire(ctx, dataKey, r.config.TTL)
		}
	}

	// Execute pipeline
	_, err = pipe.Exec(ctx)
	if err != nil {
		r.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "WRITE_FAILED", "Failed to write to Redis")
	}

	return nil
}

// WriteBatch writes multiple time series in a batch
func (r *RedisStorage) WriteBatch(ctx context.Context, batch []*models.TimeSeries) error {
	if len(batch) == 0 {
		return nil
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.closed || r.client == nil {
		return errors.NewStorageError("NOT_CONNECTED", "Redis not connected")
	}

	start := time.Now()
	defer func() {
		r.metrics.mu.Lock()
		r.metrics.writeOps += int64(len(batch))
		r.metrics.mu.Unlock()
		r.logger.WithFields(logrus.Fields{
			"count":    len(batch),
			"duration": time.Since(start),
		}).Debug("Batch write operation completed")
	}()

	pipe := r.client.Pipeline()

	for _, timeSeries := range batch {
		if err := timeSeries.Validate(); err != nil {
			r.incrementErrorCount()
			return errors.WrapError(err, errors.ErrorTypeValidation, "INVALID_DATA", 
				fmt.Sprintf("Time series validation failed for ID: %s", timeSeries.ID))
		}

		// Store metadata
		metadataKey := r.generateMetadataKey(timeSeries.ID)
		metadataJSON, err := json.Marshal(timeSeries)
		if err != nil {
			r.incrementErrorCount()
			return errors.WrapError(err, errors.ErrorTypeStorage, "SERIALIZATION_FAILED", "Failed to serialize time series metadata")
		}

		pipe.Set(ctx, metadataKey, metadataJSON, r.config.TTL)

		// Store data points
		if r.config.UseStreams {
			streamKey := r.generateStreamKey(timeSeries.ID)
			for _, point := range timeSeries.DataPoints {
				fields := map[string]interface{}{
					"value":     point.Value,
					"quality":   point.Quality,
					"timestamp": point.Timestamp.Unix(),
				}
				
				if point.Tags != nil {
					tagsJSON, _ := json.Marshal(point.Tags)
					fields["tags"] = string(tagsJSON)
				}
				
				if point.Metadata != nil {
					metaJSON, _ := json.Marshal(point.Metadata)
					fields["metadata"] = string(metaJSON)
				}

				pipe.XAdd(ctx, &redis.XAddArgs{
					Stream: streamKey,
					MaxLen: r.config.StreamMaxLen,
					Approx: true,
					Values: fields,
				})
			}
		} else {
			dataKey := r.generateDataKey(timeSeries.ID)
			for _, point := range timeSeries.DataPoints {
				pointJSON, _ := json.Marshal(point)
				score := float64(point.Timestamp.Unix())
				pipe.ZAdd(ctx, dataKey, &redis.Z{
					Score:  score,
					Member: string(pointJSON),
				})
			}
			
			if r.config.TTL > 0 {
				pipe.Expire(ctx, dataKey, r.config.TTL)
			}
		}
	}

	// Execute pipeline
	_, err := pipe.Exec(ctx)
	if err != nil {
		r.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "BATCH_WRITE_FAILED", "Failed to write batch to Redis")
	}

	return nil
}

// Read reads a time series by ID
func (r *RedisStorage) Read(ctx context.Context, id string) (*models.TimeSeries, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.closed || r.client == nil {
		return nil, errors.NewStorageError("NOT_CONNECTED", "Redis not connected")
	}

	start := time.Now()
	defer func() {
		r.incrementReadOps()
		r.logger.WithField("duration", time.Since(start)).Debug("Read operation completed")
	}()

	// Read metadata
	metadataKey := r.generateMetadataKey(id)
	metadataJSON, err := r.client.Get(ctx, metadataKey).Result()
	if err != nil {
		if err == redis.Nil {
			r.incrementMissCount()
			return nil, errors.NewStorageError("NOT_FOUND", fmt.Sprintf("Time series '%s' not found", id))
		}
		r.incrementErrorCount()
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "READ_FAILED", "Failed to read metadata from Redis")
	}

	r.incrementHitCount()

	var timeSeries models.TimeSeries
	if err := json.Unmarshal([]byte(metadataJSON), &timeSeries); err != nil {
		r.incrementErrorCount()
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "DESERIALIZATION_FAILED", "Failed to deserialize time series metadata")
	}

	// Read data points
	dataPoints, err := r.readDataPoints(ctx, id, nil, nil)
	if err != nil {
		r.incrementErrorCount()
		return nil, err
	}

	timeSeries.DataPoints = dataPoints
	return &timeSeries, nil
}

// ReadRange reads time series data within a time range
func (r *RedisStorage) ReadRange(ctx context.Context, id string, start, end time.Time) (*models.TimeSeries, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.closed || r.client == nil {
		return nil, errors.NewStorageError("NOT_CONNECTED", "Redis not connected")
	}

	startOp := time.Now()
	defer func() {
		r.incrementReadOps()
		r.logger.WithField("duration", time.Since(startOp)).Debug("Read range operation completed")
	}()

	// Read metadata
	metadataKey := r.generateMetadataKey(id)
	metadataJSON, err := r.client.Get(ctx, metadataKey).Result()
	if err != nil {
		if err == redis.Nil {
			r.incrementMissCount()
			return nil, errors.NewStorageError("NOT_FOUND", fmt.Sprintf("Time series '%s' not found", id))
		}
		r.incrementErrorCount()
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "READ_FAILED", "Failed to read metadata from Redis")
	}

	r.incrementHitCount()

	var timeSeries models.TimeSeries
	if err := json.Unmarshal([]byte(metadataJSON), &timeSeries); err != nil {
		r.incrementErrorCount()
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "DESERIALIZATION_FAILED", "Failed to deserialize time series metadata")
	}

	// Read data points within range
	dataPoints, err := r.readDataPoints(ctx, id, &start, &end)
	if err != nil {
		r.incrementErrorCount()
		return nil, err
	}

	timeSeries.DataPoints = dataPoints
	return &timeSeries, nil
}

// Query queries time series data with filters
func (r *RedisStorage) Query(ctx context.Context, query *models.TimeSeriesQuery) ([]*models.TimeSeries, error) {
	// Redis doesn't support complex queries natively
	// This is a simplified implementation
	filters := make(map[string]interface{})
	if query.Limit > 0 {
		filters["limit"] = query.Limit
	}
	
	return r.List(ctx, filters)
}

// Delete deletes a time series by ID
func (r *RedisStorage) Delete(ctx context.Context, id string) error {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.closed || r.client == nil {
		return errors.NewStorageError("NOT_CONNECTED", "Redis not connected")
	}

	start := time.Now()
	defer func() {
		r.incrementDeleteOps()
		r.logger.WithField("duration", time.Since(start)).Debug("Delete operation completed")
	}()

	pipe := r.client.Pipeline()

	// Delete metadata
	metadataKey := r.generateMetadataKey(id)
	pipe.Del(ctx, metadataKey)

	// Delete data
	if r.config.UseStreams {
		streamKey := r.generateStreamKey(id)
		pipe.Del(ctx, streamKey)
	} else {
		dataKey := r.generateDataKey(id)
		pipe.Del(ctx, dataKey)
	}

	_, err := pipe.Exec(ctx)
	if err != nil {
		r.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "DELETE_FAILED", "Failed to delete from Redis")
	}

	return nil
}

// DeleteRange deletes time series data within a time range
func (r *RedisStorage) DeleteRange(ctx context.Context, id string, start, end time.Time) error {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.closed || r.client == nil {
		return errors.NewStorageError("NOT_CONNECTED", "Redis not connected")
	}

	startOp := time.Now()
	defer func() {
		r.incrementDeleteOps()
		r.logger.WithField("duration", time.Since(startOp)).Debug("Delete range operation completed")
	}()

	if r.config.UseStreams {
		// For streams, we can't delete specific entries easily
		// Would need to read, filter, and recreate the stream
		return errors.NewStorageError("NOT_SUPPORTED", "Range deletion not supported for Redis streams")
	} else {
		// For sorted sets, we can delete by score range
		dataKey := r.generateDataKey(id)
		minScore := float64(start.Unix())
		maxScore := float64(end.Unix())
		
		_, err := r.client.ZRemRangeByScore(ctx, dataKey, fmt.Sprintf("%f", minScore), fmt.Sprintf("%f", maxScore)).Result()
		if err != nil {
			r.incrementErrorCount()
			return errors.WrapError(err, errors.ErrorTypeStorage, "DELETE_RANGE_FAILED", "Failed to delete range from Redis")
		}
	}

	return nil
}

// List lists available time series
func (r *RedisStorage) List(ctx context.Context, filters map[string]interface{}) ([]*models.TimeSeries, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.closed || r.client == nil {
		return nil, errors.NewStorageError("NOT_CONNECTED", "Redis not connected")
	}

	start := time.Now()
	defer func() {
		r.incrementReadOps()
		r.logger.WithField("duration", time.Since(start)).Debug("List operation completed")
	}()

	// Find all metadata keys
	pattern := r.generateMetadataKey("*")
	keys, err := r.client.Keys(ctx, pattern).Result()
	if err != nil {
		r.incrementErrorCount()
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "LIST_FAILED", "Failed to list keys from Redis")
	}

	// Apply limit
	limit := len(keys)
	if limitVal, ok := filters["limit"]; ok {
		if l, ok := limitVal.(int); ok && l > 0 && l < limit {
			limit = l
			keys = keys[:limit]
		}
	}

	var result []*models.TimeSeries
	for _, key := range keys {
		metadataJSON, err := r.client.Get(ctx, key).Result()
		if err != nil {
			r.logger.WithError(err).Warn("Failed to read metadata during list")
			continue
		}

		var timeSeries models.TimeSeries
		if err := json.Unmarshal([]byte(metadataJSON), &timeSeries); err != nil {
			r.logger.WithError(err).Warn("Failed to deserialize metadata during list")
			continue
		}

		// For efficiency, don't load data points during list
		timeSeries.DataPoints = nil
		result = append(result, &timeSeries)
	}

	return result, nil
}

// Count returns the count of time series matching filters
func (r *RedisStorage) Count(ctx context.Context, filters map[string]interface{}) (int64, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.closed || r.client == nil {
		return 0, errors.NewStorageError("NOT_CONNECTED", "Redis not connected")
	}

	pattern := r.generateMetadataKey("*")
	keys, err := r.client.Keys(ctx, pattern).Result()
	if err != nil {
		r.incrementErrorCount()
		return 0, errors.WrapError(err, errors.ErrorTypeStorage, "COUNT_FAILED", "Failed to count keys in Redis")
	}

	return int64(len(keys)), nil
}

// GetMetrics returns storage metrics
func (r *RedisStorage) GetMetrics(ctx context.Context) (*interfaces.StorageMetrics, error) {
	r.metrics.mu.RLock()
	defer r.metrics.mu.RUnlock()

	return &interfaces.StorageMetrics{
		ReadOperations:    r.metrics.readOps,
		WriteOperations:   r.metrics.writeOps,
		DeleteOperations:  r.metrics.deleteOps,
		AverageReadTime:   time.Millisecond * 5,  // Redis is very fast
		AverageWriteTime:  time.Millisecond * 10, // Simplified
		ErrorCount:        r.metrics.errorCount,
		ConnectionsActive: 1, // Simplified
		ConnectionsIdle:   0,
		DataSize:          0, // Would need to calculate from Redis info
		RecordCount:       r.metrics.writeOps,
		Uptime:            time.Since(r.metrics.startTime),
	}, nil
}

// Helper methods

func (r *RedisStorage) generateMetadataKey(id string) string {
	if r.config.KeyPrefix != "" {
		return fmt.Sprintf("%s:metadata:%s", r.config.KeyPrefix, id)
	}
	return fmt.Sprintf("metadata:%s", id)
}

func (r *RedisStorage) generateDataKey(id string) string {
	if r.config.KeyPrefix != "" {
		return fmt.Sprintf("%s:data:%s", r.config.KeyPrefix, id)
	}
	return fmt.Sprintf("data:%s", id)
}

func (r *RedisStorage) generateStreamKey(id string) string {
	if r.config.KeyPrefix != "" {
		return fmt.Sprintf("%s:stream:%s", r.config.KeyPrefix, id)
	}
	return fmt.Sprintf("stream:%s", id)
}

func (r *RedisStorage) readDataPoints(ctx context.Context, id string, start, end *time.Time) ([]models.DataPoint, error) {
	if r.config.UseStreams {
		return r.readDataPointsFromStream(ctx, id, start, end)
	} else {
		return r.readDataPointsFromSortedSet(ctx, id, start, end)
	}
}

func (r *RedisStorage) readDataPointsFromStream(ctx context.Context, id string, start, end *time.Time) ([]models.DataPoint, error) {
	streamKey := r.generateStreamKey(id)
	
	startID := "-"
	endID := "+"
	
	if start != nil {
		// Convert timestamp to Redis stream ID format
		startID = fmt.Sprintf("%d-0", start.Unix()*1000)
	}
	
	if end != nil {
		endID = fmt.Sprintf("%d-0", end.Unix()*1000)
	}

	entries, err := r.client.XRange(ctx, streamKey, startID, endID).Result()
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "STREAM_READ_FAILED", "Failed to read from Redis stream")
	}

	var dataPoints []models.DataPoint
	for _, entry := range entries {
		point := models.DataPoint{}
		
		if val, ok := entry.Values["value"]; ok {
			if f, err := strconv.ParseFloat(val.(string), 64); err == nil {
				point.Value = f
			}
		}
		
		if val, ok := entry.Values["quality"]; ok {
			if f, err := strconv.ParseFloat(val.(string), 64); err == nil {
				point.Quality = f
			}
		}
		
		if val, ok := entry.Values["timestamp"]; ok {
			if i, err := strconv.ParseInt(val.(string), 10, 64); err == nil {
				point.Timestamp = time.Unix(i, 0)
			}
		}
		
		if val, ok := entry.Values["tags"]; ok {
			if tagsStr, ok := val.(string); ok {
				json.Unmarshal([]byte(tagsStr), &point.Tags)
			}
		}
		
		if val, ok := entry.Values["metadata"]; ok {
			if metaStr, ok := val.(string); ok {
				json.Unmarshal([]byte(metaStr), &point.Metadata)
			}
		}

		dataPoints = append(dataPoints, point)
	}

	return dataPoints, nil
}

func (r *RedisStorage) readDataPointsFromSortedSet(ctx context.Context, id string, start, end *time.Time) ([]models.DataPoint, error) {
	dataKey := r.generateDataKey(id)
	
	var members []string
	var err error
	
	if start != nil && end != nil {
		minScore := float64(start.Unix())
		maxScore := float64(end.Unix())
		members, err = r.client.ZRangeByScore(ctx, dataKey, &redis.ZRangeBy{
			Min: fmt.Sprintf("%f", minScore),
			Max: fmt.Sprintf("%f", maxScore),
		}).Result()
	} else {
		members, err = r.client.ZRange(ctx, dataKey, 0, -1).Result()
	}
	
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "ZSET_READ_FAILED", "Failed to read from Redis sorted set")
	}

	var dataPoints []models.DataPoint
	for _, member := range members {
		var point models.DataPoint
		if err := json.Unmarshal([]byte(member), &point); err != nil {
			r.logger.WithError(err).Warn("Failed to deserialize data point")
			continue
		}
		dataPoints = append(dataPoints, point)
	}

	return dataPoints, nil
}

func (r *RedisStorage) getKeyCount(ctx context.Context) int64 {
	if info, err := r.client.Info(ctx, "keyspace").Result(); err == nil {
		lines := strings.Split(info, "\n")
		for _, line := range lines {
			if strings.Contains(line, "keys=") {
				parts := strings.Split(line, ",")
				for _, part := range parts {
					if strings.HasPrefix(part, "keys=") {
						if count, err := strconv.ParseInt(strings.TrimPrefix(part, "keys="), 10, 64); err == nil {
							return count
						}
					}
				}
			}
		}
	}
	return 0
}

func (r *RedisStorage) incrementReadOps() {
	r.metrics.mu.Lock()
	r.metrics.readOps++
	r.metrics.mu.Unlock()
}

func (r *RedisStorage) incrementWriteOps() {
	r.metrics.mu.Lock()
	r.metrics.writeOps++
	r.metrics.mu.Unlock()
}

func (r *RedisStorage) incrementDeleteOps() {
	r.metrics.mu.Lock()
	r.metrics.deleteOps++
	r.metrics.mu.Unlock()
}

func (r *RedisStorage) incrementErrorCount() {
	r.metrics.mu.Lock()
	r.metrics.errorCount++
	r.metrics.mu.Unlock()
}

func (r *RedisStorage) incrementHitCount() {
	r.metrics.mu.Lock()
	r.metrics.hitCount++
	r.metrics.mu.Unlock()
}

func (r *RedisStorage) incrementMissCount() {
	r.metrics.mu.Lock()
	r.metrics.missCount++
	r.metrics.mu.Unlock()
}