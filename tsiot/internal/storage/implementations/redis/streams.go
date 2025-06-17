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

	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/models"
)

// StreamManager manages Redis Streams for time series data
type StreamManager struct {
	client       redis.UniversalClient
	logger       *logrus.Logger
	config       *StreamConfig
	mu           sync.RWMutex
	consumers    map[string]*ConsumerGroup
	producers    map[string]*Producer
	streams      map[string]*StreamInfo
	closed       bool
}

// StreamConfig contains configuration for Redis Streams
type StreamConfig struct {
	// Stream settings
	MaxLength            int64         `json:"max_length"`
	ApproxMaxLength      bool          `json:"approx_max_length"`
	DefaultTTL           time.Duration `json:"default_ttl"`
	
	// Consumer group settings
	ConsumerGroupName    string        `json:"consumer_group_name"`
	ConsumerTimeout      time.Duration `json:"consumer_timeout"`
	MaxRetries           int           `json:"max_retries"`
	RetryBackoff         time.Duration `json:"retry_backoff"`
	AckTimeout           time.Duration `json:"ack_timeout"`
	
	// Producer settings
	ProducerBatchSize    int           `json:"producer_batch_size"`
	ProducerFlushTimeout time.Duration `json:"producer_flush_timeout"`
	
	// Monitoring settings
	MonitoringInterval   time.Duration `json:"monitoring_interval"`
	MetricsRetention     time.Duration `json:"metrics_retention"`
}

// StreamInfo contains information about a Redis stream
type StreamInfo struct {
	Name            string              `json:"name"`
	Length          int64               `json:"length"`
	RadixTreeKeys   int64               `json:"radix_tree_keys"`
	RadixTreeNodes  int64               `json:"radix_tree_nodes"`
	Groups          int                 `json:"groups"`
	LastGeneratedID string              `json:"last_generated_id"`
	FirstEntry      *redis.XMessage     `json:"first_entry,omitempty"`
	LastEntry       *redis.XMessage     `json:"last_entry,omitempty"`
	Consumers       []*ConsumerInfo     `json:"consumers"`
	CreatedAt       time.Time           `json:"created_at"`
	LastUpdated     time.Time           `json:"last_updated"`
}

// ConsumerGroup represents a Redis streams consumer group
type ConsumerGroup struct {
	Name           string              `json:"name"`
	Stream         string              `json:"stream"`
	LastDeliveredID string             `json:"last_delivered_id"`
	Pending        int64               `json:"pending"`
	Consumers      []*ConsumerInfo     `json:"consumers"`
	Config         *ConsumerConfig     `json:"config"`
	mu             sync.RWMutex
}

// ConsumerInfo contains information about a consumer
type ConsumerInfo struct {
	Name       string    `json:"name"`
	Pending    int64     `json:"pending"`
	Idle       int64     `json:"idle"`
	LastSeen   time.Time `json:"last_seen"`
	Active     bool      `json:"active"`
}

// ConsumerConfig contains consumer configuration
type ConsumerConfig struct {
	Name            string        `json:"name"`
	Group           string        `json:"group"`
	Stream          string        `json:"stream"`
	StartID         string        `json:"start_id"`
	BlockTime       time.Duration `json:"block_time"`
	Count           int64         `json:"count"`
	ProcessTimeout  time.Duration `json:"process_timeout"`
	AutoAck         bool          `json:"auto_ack"`
	RetryFailedMsgs bool          `json:"retry_failed_msgs"`
}

// Producer manages producing messages to Redis streams
type Producer struct {
	stream       string
	manager      *StreamManager
	batchBuffer  []*ProducerMessage
	lastFlush    time.Time
	mu           sync.Mutex
}

// ProducerMessage represents a message to be produced
type ProducerMessage struct {
	Stream string                 `json:"stream"`
	Fields map[string]interface{} `json:"fields"`
	ID     string                 `json:"id,omitempty"`
}

// StreamMetrics contains metrics for stream operations
type StreamMetrics struct {
	MessagesProduced    int64         `json:"messages_produced"`
	MessagesConsumed    int64         `json:"messages_consumed"`
	MessagesPending     int64         `json:"messages_pending"`
	MessagesAcked       int64         `json:"messages_acked"`
	ConsumerLag         time.Duration `json:"consumer_lag"`
	ProducerThroughput  float64       `json:"producer_throughput"`
	ConsumerThroughput  float64       `json:"consumer_throughput"`
	ErrorCount          int64         `json:"error_count"`
	LastUpdate          time.Time     `json:"last_update"`
}

// NewStreamManager creates a new Redis streams manager
func NewStreamManager(client redis.UniversalClient, config *StreamConfig, logger *logrus.Logger) *StreamManager {
	if config == nil {
		config = getDefaultStreamConfig()
	}
	
	if logger == nil {
		logger = logrus.New()
	}
	
	return &StreamManager{
		client:    client,
		logger:    logger,
		config:    config,
		consumers: make(map[string]*ConsumerGroup),
		producers: make(map[string]*Producer),
		streams:   make(map[string]*StreamInfo),
	}
}

// CreateStream creates a new Redis stream
func (sm *StreamManager) CreateStream(ctx context.Context, streamName string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	if sm.closed {
		return errors.NewStorageError("MANAGER_CLOSED", "Stream manager is closed")
	}
	
	// Check if stream already exists
	exists, err := sm.client.Exists(ctx, streamName).Result()
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "STREAM_CHECK_FAILED", "Failed to check if stream exists")
	}
	
	if exists == 1 {
		sm.logger.WithField("stream", streamName).Debug("Stream already exists")
		return nil
	}
	
	// Create stream by adding a dummy entry and then removing it
	_, err = sm.client.XAdd(ctx, &redis.XAddArgs{
		Stream: streamName,
		Values: map[string]interface{}{
			"_init": "true",
		},
	}).Result()
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "STREAM_CREATE_FAILED", "Failed to create stream")
	}
	
	// Get stream info
	streamInfo, err := sm.getStreamInfo(ctx, streamName)
	if err != nil {
		sm.logger.WithError(err).Warn("Failed to get stream info after creation")
	} else {
		sm.streams[streamName] = streamInfo
	}
	
	sm.logger.WithField("stream", streamName).Info("Redis stream created")
	return nil
}

// CreateConsumerGroup creates a consumer group for a stream
func (sm *StreamManager) CreateConsumerGroup(ctx context.Context, streamName, groupName, startID string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	if sm.closed {
		return errors.NewStorageError("MANAGER_CLOSED", "Stream manager is closed")
	}
	
	// Create consumer group
	err := sm.client.XGroupCreate(ctx, streamName, groupName, startID).Err()
	if err != nil {
		// Check if group already exists
		if strings.Contains(err.Error(), "BUSYGROUP") {
			sm.logger.WithFields(logrus.Fields{
				"stream": streamName,
				"group":  groupName,
			}).Debug("Consumer group already exists")
			return nil
		}
		return errors.WrapError(err, errors.ErrorTypeStorage, "CONSUMER_GROUP_CREATE_FAILED", "Failed to create consumer group")
	}
	
	// Create consumer group info
	consumerGroup := &ConsumerGroup{
		Name:            groupName,
		Stream:          streamName,
		LastDeliveredID: startID,
		Pending:         0,
		Consumers:       make([]*ConsumerInfo, 0),
		Config: &ConsumerConfig{
			Group:   groupName,
			Stream:  streamName,
			StartID: startID,
		},
	}
	
	sm.consumers[fmt.Sprintf("%s:%s", streamName, groupName)] = consumerGroup
	
	sm.logger.WithFields(logrus.Fields{
		"stream":   streamName,
		"group":    groupName,
		"start_id": startID,
	}).Info("Consumer group created")
	
	return nil
}

// GetProducer returns a producer for the specified stream
func (sm *StreamManager) GetProducer(streamName string) *Producer {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	if producer, exists := sm.producers[streamName]; exists {
		return producer
	}
	
	producer := &Producer{
		stream:      streamName,
		manager:     sm,
		batchBuffer: make([]*ProducerMessage, 0, sm.config.ProducerBatchSize),
		lastFlush:   time.Now(),
	}
	
	sm.producers[streamName] = producer
	return producer
}

// ProduceTimeSeries produces a time series to a Redis stream
func (sm *StreamManager) ProduceTimeSeries(ctx context.Context, streamName string, timeSeries *models.TimeSeries) error {
	producer := sm.GetProducer(streamName)
	
	for _, point := range timeSeries.DataPoints {
		fields := map[string]interface{}{
			"series_id": timeSeries.ID,
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
		
		if err := producer.Produce(ctx, fields); err != nil {
			return err
		}
	}
	
	return producer.Flush(ctx)
}

// ConsumeTimeSeries consumes time series data from a Redis stream
func (sm *StreamManager) ConsumeTimeSeries(ctx context.Context, config *ConsumerConfig, handler func(*models.TimeSeries) error) error {
	if config == nil {
		return errors.NewValidationError("INVALID_CONFIG", "Consumer configuration is required")
	}
	
	sm.logger.WithFields(logrus.Fields{
		"consumer": config.Name,
		"group":    config.Group,
		"stream":   config.Stream,
	}).Info("Starting time series consumer")
	
	for {
		select {
		case <-ctx.Done():
			sm.logger.WithField("consumer", config.Name).Info("Consumer stopped by context")
			return ctx.Err()
		default:
			if err := sm.processMessages(ctx, config, handler); err != nil {
				sm.logger.WithError(err).Error("Error processing messages")
				time.Sleep(config.ProcessTimeout)
			}
		}
	}
}

// GetStreamInfo returns information about a stream
func (sm *StreamManager) GetStreamInfo(ctx context.Context, streamName string) (*StreamInfo, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	return sm.getStreamInfo(ctx, streamName)
}

// GetStreamMetrics returns metrics for a stream
func (sm *StreamManager) GetStreamMetrics(ctx context.Context, streamName string) (*StreamMetrics, error) {
	streamInfo, err := sm.GetStreamInfo(ctx, streamName)
	if err != nil {
		return nil, err
	}
	
	// Get consumer group info
	groups, err := sm.client.XInfoGroups(ctx, streamName).Result()
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "STREAM_GROUPS_INFO_FAILED", "Failed to get stream groups info")
	}
	
	var totalPending int64
	for _, group := range groups {
		if pending, ok := group.Pending.(int64); ok {
			totalPending += pending
		}
	}
	
	metrics := &StreamMetrics{
		MessagesProduced:   streamInfo.Length,
		MessagesPending:    totalPending,
		LastUpdate:         time.Now(),
	}
	
	return metrics, nil
}

// ListStreams returns a list of all managed streams
func (sm *StreamManager) ListStreams(ctx context.Context) ([]string, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	var streams []string
	for streamName := range sm.streams {
		streams = append(streams, streamName)
	}
	
	return streams, nil
}

// DeleteStream deletes a Redis stream
func (sm *StreamManager) DeleteStream(ctx context.Context, streamName string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	if sm.closed {
		return errors.NewStorageError("MANAGER_CLOSED", "Stream manager is closed")
	}
	
	// Delete the stream
	err := sm.client.Del(ctx, streamName).Err()
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "STREAM_DELETE_FAILED", "Failed to delete stream")
	}
	
	// Remove from local tracking
	delete(sm.streams, streamName)
	
	// Remove associated producers and consumers
	for key := range sm.producers {
		if strings.HasPrefix(key, streamName) {
			delete(sm.producers, key)
		}
	}
	
	for key := range sm.consumers {
		if strings.Contains(key, streamName) {
			delete(sm.consumers, key)
		}
	}
	
	sm.logger.WithField("stream", streamName).Info("Stream deleted")
	return nil
}

// Close closes the stream manager
func (sm *StreamManager) Close() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	sm.closed = true
	
	// Flush all producers
	for _, producer := range sm.producers {
		if err := producer.Flush(context.Background()); err != nil {
			sm.logger.WithError(err).Warn("Failed to flush producer during close")
		}
	}
	
	sm.logger.Info("Stream manager closed")
	return nil
}

// Producer methods

// Produce adds a message to the producer's batch buffer
func (p *Producer) Produce(ctx context.Context, fields map[string]interface{}) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	message := &ProducerMessage{
		Stream: p.stream,
		Fields: fields,
	}
	
	p.batchBuffer = append(p.batchBuffer, message)
	
	// Flush if batch is full or timeout reached
	if len(p.batchBuffer) >= p.manager.config.ProducerBatchSize ||
		time.Since(p.lastFlush) >= p.manager.config.ProducerFlushTimeout {
		return p.flush(ctx)
	}
	
	return nil
}

// Flush flushes the producer's batch buffer
func (p *Producer) Flush(ctx context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	return p.flush(ctx)
}

func (p *Producer) flush(ctx context.Context) error {
	if len(p.batchBuffer) == 0 {
		return nil
	}
	
	pipe := p.manager.client.Pipeline()
	
	for _, message := range p.batchBuffer {
		pipe.XAdd(ctx, &redis.XAddArgs{
			Stream: message.Stream,
			MaxLen: p.manager.config.MaxLength,
			Approx: p.manager.config.ApproxMaxLength,
			Values: message.Fields,
		})
	}
	
	_, err := pipe.Exec(ctx)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "BATCH_PRODUCE_FAILED", "Failed to produce batch to stream")
	}
	
	p.manager.logger.WithFields(logrus.Fields{
		"stream":    p.stream,
		"batch_size": len(p.batchBuffer),
	}).Debug("Batch produced to stream")
	
	p.batchBuffer = p.batchBuffer[:0] // Clear buffer
	p.lastFlush = time.Now()
	
	return nil
}

// Internal methods

func (sm *StreamManager) getStreamInfo(ctx context.Context, streamName string) (*StreamInfo, error) {
	info, err := sm.client.XInfoStream(ctx, streamName).Result()
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "STREAM_INFO_FAILED", "Failed to get stream info")
	}
	
	streamInfo := &StreamInfo{
		Name:            streamName,
		Length:          info.Length,
		RadixTreeKeys:   info.RadixTreeKeys,
		RadixTreeNodes:  info.RadixTreeNodes,
		Groups:          info.Groups,
		LastGeneratedID: info.LastGeneratedID,
		LastUpdated:     time.Now(),
	}
	
	if info.FirstEntry != nil {
		streamInfo.FirstEntry = info.FirstEntry
	}
	
	if info.LastEntry != nil {
		streamInfo.LastEntry = info.LastEntry
	}
	
	return streamInfo, nil
}

func (sm *StreamManager) processMessages(ctx context.Context, config *ConsumerConfig, handler func(*models.TimeSeries) error) error {
	// Read messages from stream
	messages, err := sm.client.XReadGroup(ctx, &redis.XReadGroupArgs{
		Group:    config.Group,
		Consumer: config.Name,
		Streams:  []string{config.Stream, ">"},
		Count:    config.Count,
		Block:    config.BlockTime,
	}).Result()
	
	if err != nil {
		if err == redis.Nil {
			return nil // No messages available
		}
		return errors.WrapError(err, errors.ErrorTypeStorage, "MESSAGE_READ_FAILED", "Failed to read messages from stream")
	}
	
	for _, message := range messages {
		for _, msg := range message.Messages {
			if err := sm.processMessage(ctx, config, msg, handler); err != nil {
				sm.logger.WithError(err).Error("Failed to process message")
				
				if !config.AutoAck {
					// Add message to pending list for retry
					continue
				}
			}
			
			// Acknowledge message
			if err := sm.client.XAck(ctx, config.Stream, config.Group, msg.ID).Err(); err != nil {
				sm.logger.WithError(err).Warn("Failed to acknowledge message")
			}
		}
	}
	
	return nil
}

func (sm *StreamManager) processMessage(ctx context.Context, config *ConsumerConfig, msg redis.XMessage, handler func(*models.TimeSeries) error) error {
	// Extract time series data from message
	timeSeries := &models.TimeSeries{
		DataPoints: make([]models.DataPoint, 0, 1),
	}
	
	dataPoint := models.DataPoint{}
	
	// Extract fields
	if seriesID, ok := msg.Values["series_id"].(string); ok {
		timeSeries.ID = seriesID
	}
	
	if value, ok := msg.Values["value"].(string); ok {
		if v, err := strconv.ParseFloat(value, 64); err == nil {
			dataPoint.Value = v
		}
	}
	
	if quality, ok := msg.Values["quality"].(string); ok {
		if q, err := strconv.ParseFloat(quality, 64); err == nil {
			dataPoint.Quality = q
		}
	}
	
	if timestamp, ok := msg.Values["timestamp"].(string); ok {
		if ts, err := strconv.ParseInt(timestamp, 10, 64); err == nil {
			dataPoint.Timestamp = time.Unix(ts, 0)
		}
	}
	
	if tags, ok := msg.Values["tags"].(string); ok {
		json.Unmarshal([]byte(tags), &dataPoint.Tags)
	}
	
	if metadata, ok := msg.Values["metadata"].(string); ok {
		json.Unmarshal([]byte(metadata), &dataPoint.Metadata)
	}
	
	timeSeries.DataPoints = append(timeSeries.DataPoints, dataPoint)
	
	// Call handler
	return handler(timeSeries)
}

func getDefaultStreamConfig() *StreamConfig {
	return &StreamConfig{
		MaxLength:            10000,
		ApproxMaxLength:      true,
		DefaultTTL:           24 * time.Hour,
		ConsumerGroupName:    "tsiot-consumers",
		ConsumerTimeout:      30 * time.Second,
		MaxRetries:           3,
		RetryBackoff:         time.Second,
		AckTimeout:           60 * time.Second,
		ProducerBatchSize:    100,
		ProducerFlushTimeout: 5 * time.Second,
		MonitoringInterval:   30 * time.Second,
		MetricsRetention:     7 * 24 * time.Hour,
	}
}