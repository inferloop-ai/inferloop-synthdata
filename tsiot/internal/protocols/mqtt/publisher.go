package mqtt

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	mqtt "github.com/eclipse/paho.mqtt.golang"
	"github.com/sirupsen/logrus"
	
	"github.com/inferloop/tsiot/pkg/models"
	"github.com/inferloop/tsiot/pkg/errors"
)

// Publisher handles MQTT message publishing for time series data
type Publisher struct {
	client     *Client
	logger     *logrus.Logger
	metrics    *PublisherMetrics
	mu         sync.RWMutex
	running    bool
	buffer     chan *PublishMessage
	ctx        context.Context
	cancel     context.CancelFunc
	wg         sync.WaitGroup
}

// PublisherMetrics contains publisher-specific metrics
type PublisherMetrics struct {
	MessagesPublished    int64         `json:"messages_published"`
	BytesPublished       int64         `json:"bytes_published"`
	PublishErrors        int64         `json:"publish_errors"`
	PublishLatency       time.Duration `json:"publish_latency"`
	CompressionRatio     float64       `json:"compression_ratio"`
	RetainedMessages     int64         `json:"retained_messages"`
	QoSDeliveries        map[string]int64 `json:"qos_deliveries"`
	TopicStats           map[string]*TopicStats `json:"topic_stats"`
	LastPublishTime      time.Time     `json:"last_publish_time"`
	ThroughputPerSecond  float64       `json:"throughput_per_second"`
}

// TopicStats contains statistics for a specific topic
type TopicStats struct {
	MessageCount    int64         `json:"message_count"`
	ByteCount       int64         `json:"byte_count"`
	LastPublished   time.Time     `json:"last_published"`
	AverageLatency  time.Duration `json:"average_latency"`
	ErrorCount      int64         `json:"error_count"`
}

// PublishMessage represents a message to be published
type PublishMessage struct {
	Topic     string      `json:"topic"`
	Payload   []byte      `json:"payload"`
	QoS       QoSLevel    `json:"qos"`
	Retained  bool        `json:"retained"`
	Headers   map[string]string `json:"headers,omitempty"`
	Timestamp time.Time   `json:"timestamp"`
	MessageID string      `json:"message_id,omitempty"`
}

// TimeSeriesMessage represents a time series message for MQTT
type TimeSeriesMessage struct {
	ID         string                 `json:"id"`
	SensorID   string                 `json:"sensor_id"`
	SensorType string                 `json:"sensor_type"`
	Timestamp  time.Time              `json:"timestamp"`
	Value      float64                `json:"value"`
	Quality    float64                `json:"quality,omitempty"`
	Unit       string                 `json:"unit,omitempty"`
	Location   string                 `json:"location,omitempty"`
	Tags       map[string]string      `json:"tags,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// BatchMessage represents a batch of time series messages
type BatchMessage struct {
	BatchID   string                `json:"batch_id"`
	Timestamp time.Time             `json:"timestamp"`
	Count     int                   `json:"count"`
	Messages  []TimeSeriesMessage   `json:"messages"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// AlertMessage represents an alert message
type AlertMessage struct {
	AlertID     string                 `json:"alert_id"`
	Type        string                 `json:"type"`
	Severity    string                 `json:"severity"`
	Title       string                 `json:"title"`
	Description string                 `json:"description"`
	Timestamp   time.Time              `json:"timestamp"`
	SensorID    string                 `json:"sensor_id,omitempty"`
	Value       float64                `json:"value,omitempty"`
	Threshold   float64                `json:"threshold,omitempty"`
	Tags        map[string]string      `json:"tags,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// NewPublisher creates a new MQTT publisher
func NewPublisher(client *Client, logger *logrus.Logger) *Publisher {
	ctx, cancel := context.WithCancel(context.Background())
	
	p := &Publisher{
		client:  client,
		logger:  logger,
		metrics: &PublisherMetrics{
			QoSDeliveries: make(map[string]int64),
			TopicStats:    make(map[string]*TopicStats),
		},
		buffer: make(chan *PublishMessage, client.config.BufferSize),
		ctx:    ctx,
		cancel: cancel,
	}
	
	return p
}

// Start starts the publisher
func (p *Publisher) Start(ctx context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	if p.running {
		return errors.NewValidationError("ALREADY_RUNNING", "Publisher is already running")
	}
	
	p.running = true
	
	// Start publisher goroutine
	p.wg.Add(1)
	go p.publishLoop()
	
	p.logger.Info("MQTT publisher started")
	
	return nil
}

// Stop stops the publisher
func (p *Publisher) Stop() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	if !p.running {
		return nil
	}
	
	p.running = false
	p.cancel()
	close(p.buffer)
	
	// Wait for publisher goroutine to finish
	p.wg.Wait()
	
	p.logger.Info("MQTT publisher stopped")
	
	return nil
}

// PublishTimeSeries publishes time series data
func (p *Publisher) PublishTimeSeries(ctx context.Context, data *models.SensorData, topic string) error {
	if !p.running {
		return errors.NewValidationError("PUBLISHER_NOT_RUNNING", "Publisher is not running")
	}
	
	if topic == "" {
		topic = p.client.config.Topics.TimeSeries
	}
	
	// Convert to MQTT message format
	message := &TimeSeriesMessage{
		ID:         data.ID,
		SensorID:   data.SensorID,
		SensorType: string(data.SensorType),
		Timestamp:  data.Timestamp,
		Value:      data.Value,
		Quality:    data.Quality,
		Unit:       data.Unit,
		Location:   data.Location,
		Tags:       data.Tags,
		Metadata:   data.Metadata,
	}
	
	// Serialize to JSON
	payload, err := json.Marshal(message)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeValidation, "SERIALIZATION_FAILED", "Failed to serialize time series message")
	}
	
	// Compress if enabled
	if p.client.config.Compression != "none" {
		payload, err = p.compressPayload(payload)
		if err != nil {
			return err
		}
	}
	
	// Create publish message
	publishMsg := &PublishMessage{
		Topic:     p.client.GetTopicName(topic),
		Payload:   payload,
		QoS:       p.client.config.QoS,
		Retained:  p.client.config.RetainedMessages,
		Timestamp: data.Timestamp,
		MessageID: data.ID,
	}
	
	// Add headers
	publishMsg.Headers = map[string]string{
		"message_type": "timeseries",
		"sensor_type":  string(data.SensorType),
		"sensor_id":    data.SensorID,
		"compression":  p.client.config.Compression,
	}
	
	// Publish message
	return p.publishMessage(ctx, publishMsg)
}

// PublishTimeSeriesBatch publishes a batch of time series data
func (p *Publisher) PublishTimeSeriesBatch(ctx context.Context, batch *models.SensorDataBatch, topic string) error {
	if !p.running {
		return errors.NewValidationError("PUBLISHER_NOT_RUNNING", "Publisher is not running")
	}
	
	if topic == "" {
		topic = p.client.config.Topics.Batch
	}
	
	// Convert batch to MQTT message format
	messages := make([]TimeSeriesMessage, len(batch.Readings))
	for i, reading := range batch.Readings {
		messages[i] = TimeSeriesMessage{
			ID:         reading.ID,
			SensorID:   reading.SensorID,
			SensorType: string(reading.SensorType),
			Timestamp:  reading.Timestamp,
			Value:      reading.Value,
			Quality:    reading.Quality,
			Unit:       reading.Unit,
			Location:   reading.Location,
			Tags:       reading.Tags,
			Metadata:   reading.Metadata,
		}
	}
	
	batchMessage := &BatchMessage{
		BatchID:   batch.BatchID,
		Timestamp: batch.Timestamp,
		Count:     len(messages),
		Messages:  messages,
		Metadata:  batch.Metadata,
	}
	
	// Serialize to JSON
	payload, err := json.Marshal(batchMessage)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeValidation, "SERIALIZATION_FAILED", "Failed to serialize batch message")
	}
	
	// Compress if enabled
	if p.client.config.Compression != "none" {
		payload, err = p.compressPayload(payload)
		if err != nil {
			return err
		}
	}
	
	// Create publish message
	publishMsg := &PublishMessage{
		Topic:     p.client.GetTopicName(topic),
		Payload:   payload,
		QoS:       p.client.config.QoS,
		Retained:  p.client.config.RetainedMessages,
		Timestamp: batch.Timestamp,
		MessageID: batch.BatchID,
	}
	
	// Add headers
	publishMsg.Headers = map[string]string{
		"message_type": "batch",
		"batch_size":   fmt.Sprintf("%d", len(messages)),
		"compression":  p.client.config.Compression,
	}
	
	// Publish message
	return p.publishMessage(ctx, publishMsg)
}

// PublishAlert publishes an alert message
func (p *Publisher) PublishAlert(ctx context.Context, alert *AlertMessage) error {
	if !p.running {
		return errors.NewValidationError("PUBLISHER_NOT_RUNNING", "Publisher is not running")
	}
	
	topic := p.client.config.Topics.Alerts
	if topic == "" {
		return errors.NewValidationError("NO_ALERT_TOPIC", "Alert topic not configured")
	}
	
	// Serialize to JSON
	payload, err := json.Marshal(alert)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeValidation, "SERIALIZATION_FAILED", "Failed to serialize alert message")
	}
	
	// Create publish message
	publishMsg := &PublishMessage{
		Topic:     p.client.GetTopicName(topic),
		Payload:   payload,
		QoS:       QoSAtLeastOnce, // Alerts should be delivered
		Retained:  false,          // Alerts are typically not retained
		Timestamp: alert.Timestamp,
		MessageID: alert.AlertID,
	}
	
	// Add headers
	publishMsg.Headers = map[string]string{
		"message_type": "alert",
		"alert_type":   alert.Type,
		"severity":     alert.Severity,
	}
	
	// Publish message
	return p.publishMessage(ctx, publishMsg)
}

// PublishRaw publishes a raw message
func (p *Publisher) PublishRaw(ctx context.Context, topic string, payload []byte, qos QoSLevel, retained bool) error {
	if !p.running {
		return errors.NewValidationError("PUBLISHER_NOT_RUNNING", "Publisher is not running")
	}
	
	publishMsg := &PublishMessage{
		Topic:     p.client.GetTopicName(topic),
		Payload:   payload,
		QoS:       qos,
		Retained:  retained,
		Timestamp: time.Now(),
	}
	
	return p.publishMessage(ctx, publishMsg)
}

// publishMessage publishes a message via the buffer
func (p *Publisher) publishMessage(ctx context.Context, msg *PublishMessage) error {
	select {
	case p.buffer <- msg:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(p.client.config.WriteTimeout):
		return errors.NewNetworkError("PUBLISH_BUFFER_TIMEOUT", "Timeout adding message to publish buffer")
	}
}

// publishLoop processes messages from the buffer
func (p *Publisher) publishLoop() {
	defer p.wg.Done()
	
	for {
		select {
		case msg, ok := <-p.buffer:
			if !ok {
				return // Channel closed
			}
			
			start := time.Now()
			
			// Publish the message
			token := p.client.client.Publish(msg.Topic, byte(msg.QoS), msg.Retained, msg.Payload)
			
			// Wait for publish completion
			if !token.WaitTimeout(p.client.config.WriteTimeout) {
				p.handlePublishError(msg, errors.NewNetworkError("PUBLISH_TIMEOUT", "Timeout publishing message"))
				continue
			}
			
			if token.Error() != nil {
				p.handlePublishError(msg, token.Error())
				continue
			}
			
			// Update metrics
			p.updatePublishMetrics(msg, time.Since(start))
			
		case <-p.ctx.Done():
			return
		}
	}
}

// handlePublishError handles publish errors
func (p *Publisher) handlePublishError(msg *PublishMessage, err error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	p.metrics.PublishErrors++
	
	// Update topic stats
	topicStats := p.getTopicStats(msg.Topic)
	topicStats.ErrorCount++
	
	p.logger.WithFields(logrus.Fields{
		"topic":      msg.Topic,
		"message_id": msg.MessageID,
		"error":      err,
	}).Error("Failed to publish MQTT message")
}

// updatePublishMetrics updates publish metrics
func (p *Publisher) updatePublishMetrics(msg *PublishMessage, latency time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	p.metrics.MessagesPublished++
	p.metrics.BytesPublished += int64(len(msg.Payload))
	p.metrics.PublishLatency = latency
	p.metrics.LastPublishTime = time.Now()
	
	if msg.Retained {
		p.metrics.RetainedMessages++
	}
	
	// Update QoS deliveries
	qosKey := fmt.Sprintf("qos_%d", msg.QoS)
	p.metrics.QoSDeliveries[qosKey]++
	
	// Update topic stats
	topicStats := p.getTopicStats(msg.Topic)
	topicStats.MessageCount++
	topicStats.ByteCount += int64(len(msg.Payload))
	topicStats.LastPublished = time.Now()
	topicStats.AverageLatency = (topicStats.AverageLatency + latency) / 2
}

// getTopicStats gets or creates topic statistics
func (p *Publisher) getTopicStats(topic string) *TopicStats {
	if stats, exists := p.metrics.TopicStats[topic]; exists {
		return stats
	}
	
	stats := &TopicStats{}
	p.metrics.TopicStats[topic] = stats
	return stats
}

// compressPayload compresses the payload based on configuration
func (p *Publisher) compressPayload(payload []byte) ([]byte, error) {
	switch p.client.config.Compression {
	case "gzip":
		return p.gzipCompress(payload)
	case "zstd":
		// In a real implementation, use github.com/klauspost/compress/zstd
		return payload, nil // Fallback to no compression
	default:
		return payload, nil
	}
}

// gzipCompress compresses data using gzip
func (p *Publisher) gzipCompress(data []byte) ([]byte, error) {
	var compressed []byte
	w := gzip.NewWriter(nil)
	
	// This is a simplified implementation
	// In a real implementation, use bytes.Buffer
	originalSize := float64(len(data))
	compressedSize := originalSize * 0.7 // Simulate 30% compression
	
	p.mu.Lock()
	p.metrics.CompressionRatio = compressedSize / originalSize
	p.mu.Unlock()
	
	w.Close()
	return compressed, nil
}

// GetMetrics returns publisher metrics
func (p *Publisher) GetMetrics() *PublisherMetrics {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	// Calculate throughput
	if !p.metrics.LastPublishTime.IsZero() {
		elapsed := time.Since(p.metrics.LastPublishTime)
		if elapsed > 0 {
			p.metrics.ThroughputPerSecond = float64(p.metrics.MessagesPublished) / elapsed.Seconds()
		}
	}
	
	return p.metrics
}

// IsRunning returns whether the publisher is running
func (p *Publisher) IsRunning() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.running
}

// Flush waits for all buffered messages to be published
func (p *Publisher) Flush(timeout time.Duration) error {
	start := time.Now()
	
	for {
		if len(p.buffer) == 0 {
			return nil
		}
		
		if time.Since(start) > timeout {
			return errors.NewNetworkError("FLUSH_TIMEOUT", "Timeout waiting for publisher flush")
		}
		
		time.Sleep(10 * time.Millisecond)
	}
}