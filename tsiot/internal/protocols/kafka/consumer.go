package kafka

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/Shopify/sarama"
	"github.com/sirupsen/logrus"
	
	"github.com/inferloop/tsiot/pkg/models"
	"github.com/inferloop/tsiot/pkg/errors"
)

// Consumer represents a Kafka consumer for time series data
type Consumer struct {
	config          *ConsumerConfig
	consumerGroup   sarama.ConsumerGroup
	logger          *logrus.Logger
	handler         MessageHandler
	metrics         *ConsumerMetrics
	mu              sync.RWMutex
	running         bool
	ctx             context.Context
	cancel          context.CancelFunc
	wg              sync.WaitGroup
}

// ConsumerConfig contains configuration for Kafka consumer
type ConsumerConfig struct {
	Brokers         []string      `json:"brokers"`
	GroupID         string        `json:"group_id"`
	Topics          []string      `json:"topics"`
	ClientID        string        `json:"client_id"`
	SessionTimeout  time.Duration `json:"session_timeout"`
	HeartbeatInterval time.Duration `json:"heartbeat_interval"`
	RebalanceTimeout time.Duration `json:"rebalance_timeout"`
	OffsetInitial   string        `json:"offset_initial"`    // "oldest", "newest"
	OffsetCommitInterval time.Duration `json:"offset_commit_interval"`
	MaxProcessingTime time.Duration `json:"max_processing_time"`
	FetchMinBytes   int32         `json:"fetch_min_bytes"`
	FetchMaxBytes   int32         `json:"fetch_max_bytes"`
	FetchMaxWait    time.Duration `json:"fetch_max_wait"`
	ChannelBufferSize int         `json:"channel_buffer_size"`
	ReturnErrors    bool          `json:"return_errors"`
	Security        SecurityConfig `json:"security"`
	Metadata        MetadataConfig `json:"metadata"`
}

// ConsumerMetrics contains consumer metrics
type ConsumerMetrics struct {
	MessagesConsumed  int64     `json:"messages_consumed"`
	MessagesProcessed int64     `json:"messages_processed"`
	MessagesFailed    int64     `json:"messages_failed"`
	BatchesProcessed  int64     `json:"batches_processed"`
	BytesConsumed     int64     `json:"bytes_consumed"`
	ErrorCount        int64     `json:"error_count"`
	LastError         error     `json:"-"`
	LastErrorTime     time.Time `json:"last_error_time"`
	ProcessingLatency time.Duration `json:"processing_latency"`
	ConsumerLag       int64     `json:"consumer_lag"`
}

// MessageHandler defines the interface for handling consumed messages
type MessageHandler interface {
	HandleMessage(ctx context.Context, message *ConsumedMessage) error
	HandleBatch(ctx context.Context, messages []*ConsumedMessage) error
	HandleError(ctx context.Context, err error)
}

// ConsumedMessage represents a consumed Kafka message
type ConsumedMessage struct {
	Topic       string            `json:"topic"`
	Partition   int32             `json:"partition"`
	Offset      int64             `json:"offset"`
	Key         string            `json:"key"`
	Value       []byte            `json:"value"`
	Headers     map[string]string `json:"headers"`
	Timestamp   time.Time         `json:"timestamp"`
	TimeSeries  *TimeSeriesMessage `json:"time_series,omitempty"`
	Batch       *BatchMessage     `json:"batch,omitempty"`
}

// ConsumerGroupHandler implements sarama.ConsumerGroupHandler
type ConsumerGroupHandler struct {
	consumer *Consumer
}

// NewConsumer creates a new Kafka consumer
func NewConsumer(config *ConsumerConfig, handler MessageHandler, logger *logrus.Logger) (*Consumer, error) {
	if config == nil {
		return nil, errors.NewValidationError("INVALID_CONFIG", "Consumer config cannot be nil")
	}
	
	if len(config.Brokers) == 0 {
		return nil, errors.NewValidationError("INVALID_BROKERS", "At least one broker must be specified")
	}
	
	if config.GroupID == "" {
		return nil, errors.NewValidationError("INVALID_GROUP_ID", "Consumer group ID must be specified")
	}
	
	if len(config.Topics) == 0 {
		return nil, errors.NewValidationError("INVALID_TOPICS", "At least one topic must be specified")
	}
	
	if handler == nil {
		return nil, errors.NewValidationError("INVALID_HANDLER", "Message handler cannot be nil")
	}
	
	// Create Sarama config
	saramaConfig := sarama.NewConfig()
	
	// Set consumer configuration
	saramaConfig.Consumer.Group.Session.Timeout = config.SessionTimeout
	saramaConfig.Consumer.Group.Heartbeat.Interval = config.HeartbeatInterval
	saramaConfig.Consumer.Group.Rebalance.Timeout = config.RebalanceTimeout
	saramaConfig.Consumer.MaxProcessingTime = config.MaxProcessingTime
	saramaConfig.Consumer.Fetch.Min = config.FetchMinBytes
	saramaConfig.Consumer.Fetch.Max = config.FetchMaxBytes
	saramaConfig.Consumer.MaxWaitTime = config.FetchMaxWait
	saramaConfig.ChannelBufferSize = config.ChannelBufferSize
	saramaConfig.Consumer.Return.Errors = config.ReturnErrors
	
	// Set offset configuration
	switch config.OffsetInitial {
	case "oldest":
		saramaConfig.Consumer.Offsets.Initial = sarama.OffsetOldest
	case "newest":
		saramaConfig.Consumer.Offsets.Initial = sarama.OffsetNewest
	default:
		saramaConfig.Consumer.Offsets.Initial = sarama.OffsetNewest
	}
	
	// Set offset commit interval
	if config.OffsetCommitInterval > 0 {
		saramaConfig.Consumer.Offsets.AutoCommit.Interval = config.OffsetCommitInterval
	}
	
	// Set client ID
	if config.ClientID != "" {
		saramaConfig.ClientID = config.ClientID
	}
	
	// Set metadata configuration
	saramaConfig.Metadata.RefreshFrequency = config.Metadata.RefreshFrequency
	saramaConfig.Metadata.Full = config.Metadata.FullRefresh
	saramaConfig.Metadata.Retry.Max = config.Metadata.RetryMax
	saramaConfig.Metadata.Retry.Backoff = config.Metadata.RetryBackoff
	saramaConfig.Metadata.Timeout = config.Metadata.Timeout
	
	// Configure security
	if config.Security.Enabled {
		configureSecurity(saramaConfig, &config.Security)
	}
	
	// Create consumer group
	consumerGroup, err := sarama.NewConsumerGroup(config.Brokers, config.GroupID, saramaConfig)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeNetwork, "KAFKA_CONSUMER_CREATE_FAILED", "Failed to create Kafka consumer")
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	consumer := &Consumer{
		config:        config,
		consumerGroup: consumerGroup,
		logger:        logger,
		handler:       handler,
		metrics:       &ConsumerMetrics{},
		ctx:           ctx,
		cancel:        cancel,
	}
	
	return consumer, nil
}

// Start starts the consumer
func (c *Consumer) Start(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if c.running {
		return errors.NewValidationError("ALREADY_RUNNING", "Consumer is already running")
	}
	
	c.running = true
	
	// Start consumer group session
	c.wg.Add(1)
	go func() {
		defer c.wg.Done()
		c.consume(ctx)
	}()
	
	// Start error handler
	c.wg.Add(1)
	go func() {
		defer c.wg.Done()
		c.handleErrors()
	}()
	
	c.logger.WithFields(logrus.Fields{
		"group_id": c.config.GroupID,
		"topics":   c.config.Topics,
	}).Info("Kafka consumer started")
	
	return nil
}

// Stop stops the consumer
func (c *Consumer) Stop() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if !c.running {
		return nil
	}
	
	c.running = false
	c.cancel()
	
	// Wait for goroutines to finish
	c.wg.Wait()
	
	if err := c.consumerGroup.Close(); err != nil {
		c.logger.WithError(err).Error("Error closing Kafka consumer")
		return err
	}
	
	c.logger.Info("Kafka consumer stopped")
	return nil
}

// consume runs the main consumer loop
func (c *Consumer) consume(ctx context.Context) {
	handler := &ConsumerGroupHandler{consumer: c}
	
	for {
		select {
		case <-ctx.Done():
			return
		default:
			// Consume messages
			if err := c.consumerGroup.Consume(ctx, c.config.Topics, handler); err != nil {
				c.metrics.ErrorCount++
				c.metrics.LastError = err
				c.metrics.LastErrorTime = time.Now()
				c.handler.HandleError(ctx, err)
				
				c.logger.WithError(err).Error("Error in consumer group session")
				
				// Brief pause before retrying
				select {
				case <-time.After(5 * time.Second):
				case <-ctx.Done():
					return
				}
			}
		}
	}
}

// handleErrors handles consumer errors
func (c *Consumer) handleErrors() {
	for {
		select {
		case err := <-c.consumerGroup.Errors():
			if err != nil {
				c.metrics.ErrorCount++
				c.metrics.LastError = err
				c.metrics.LastErrorTime = time.Now()
				c.handler.HandleError(c.ctx, err)
				
				c.logger.WithError(err).Error("Consumer group error")
			}
		case <-c.ctx.Done():
			return
		}
	}
}

// Setup implements sarama.ConsumerGroupHandler
func (h *ConsumerGroupHandler) Setup(sarama.ConsumerGroupSession) error {
	h.consumer.logger.Info("Consumer group session setup")
	return nil
}

// Cleanup implements sarama.ConsumerGroupHandler
func (h *ConsumerGroupHandler) Cleanup(sarama.ConsumerGroupSession) error {
	h.consumer.logger.Info("Consumer group session cleanup")
	return nil
}

// ConsumeClaim implements sarama.ConsumerGroupHandler
func (h *ConsumerGroupHandler) ConsumeClaim(session sarama.ConsumerGroupSession, claim sarama.ConsumerGroupClaim) error {
	for {
		select {
		case message := <-claim.Messages():
			if message == nil {
				return nil
			}
			
			start := time.Now()
			
			// Convert Sarama message to our format
			consumedMessage := &ConsumedMessage{
				Topic:     message.Topic,
				Partition: message.Partition,
				Offset:    message.Offset,
				Key:       string(message.Key),
				Value:     message.Value,
				Headers:   make(map[string]string),
				Timestamp: message.Timestamp,
			}
			
			// Convert headers
			for _, header := range message.Headers {
				consumedMessage.Headers[string(header.Key)] = string(header.Value)
			}
			
			// Try to parse the message
			if err := h.parseMessage(consumedMessage); err != nil {
				h.consumer.logger.WithError(err).Warn("Failed to parse message, treating as raw")
			}
			
			// Handle the message
			if err := h.consumer.handler.HandleMessage(session.Context(), consumedMessage); err != nil {
				h.consumer.metrics.MessagesFailed++
				h.consumer.metrics.LastError = err
				h.consumer.metrics.LastErrorTime = time.Now()
				h.consumer.handler.HandleError(session.Context(), err)
				h.consumer.logger.WithError(err).Error("Failed to handle message")
			} else {
				h.consumer.metrics.MessagesProcessed++
			}
			
			// Update metrics
			h.consumer.metrics.MessagesConsumed++
			h.consumer.metrics.BytesConsumed += int64(len(message.Value))
			h.consumer.metrics.ProcessingLatency = time.Since(start)
			
			// Mark message as processed
			session.MarkMessage(message, "")
			
		case <-session.Context().Done():
			return nil
		}
	}
}

// parseMessage attempts to parse the message as time series or batch data
func (h *ConsumerGroupHandler) parseMessage(message *ConsumedMessage) error {
	messageType := message.Headers["message_type"]
	
	switch messageType {
	case "batch":
		var batch BatchMessage
		if err := json.Unmarshal(message.Value, &batch); err != nil {
			return err
		}
		message.Batch = &batch
		
	default:
		// Try to parse as single time series message
		var timeSeries TimeSeriesMessage
		if err := json.Unmarshal(message.Value, &timeSeries); err != nil {
			return err
		}
		message.TimeSeries = &timeSeries
	}
	
	return nil
}

// GetMetrics returns consumer metrics
func (c *Consumer) GetMetrics() *ConsumerMetrics {
	return c.metrics
}

// IsRunning returns whether the consumer is running
func (c *Consumer) IsRunning() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.running
}

// GetConsumerLag returns the consumer lag for each topic partition
func (c *Consumer) GetConsumerLag(ctx context.Context) (map[string]map[int32]int64, error) {
	if !c.running {
		return nil, errors.NewValidationError("CONSUMER_NOT_RUNNING", "Consumer is not running")
	}
	
	// This is a simplified implementation
	// In a real implementation, you would query Kafka for the latest offsets
	// and compare with the consumer's current offsets
	
	lag := make(map[string]map[int32]int64)
	
	for _, topic := range c.config.Topics {
		lag[topic] = make(map[int32]int64)
		// Placeholder implementation
		lag[topic][0] = c.metrics.ConsumerLag
	}
	
	return lag, nil
}

// DefaultMessageHandler provides a basic message handler implementation
type DefaultMessageHandler struct {
	logger        *logrus.Logger
	processor     func(*ConsumedMessage) error
	batchProcessor func([]*ConsumedMessage) error
	errorHandler  func(error)
}

// NewDefaultMessageHandler creates a new default message handler
func NewDefaultMessageHandler(logger *logrus.Logger) *DefaultMessageHandler {
	return &DefaultMessageHandler{
		logger: logger,
	}
}

// SetProcessor sets the message processor function
func (h *DefaultMessageHandler) SetProcessor(processor func(*ConsumedMessage) error) {
	h.processor = processor
}

// SetBatchProcessor sets the batch processor function
func (h *DefaultMessageHandler) SetBatchProcessor(processor func([]*ConsumedMessage) error) {
	h.batchProcessor = processor
}

// SetErrorHandler sets the error handler function
func (h *DefaultMessageHandler) SetErrorHandler(handler func(error)) {
	h.errorHandler = handler
}

// HandleMessage implements MessageHandler
func (h *DefaultMessageHandler) HandleMessage(ctx context.Context, message *ConsumedMessage) error {
	if h.processor != nil {
		return h.processor(message)
	}
	
	// Default behavior: log the message
	if message.TimeSeries != nil {
		h.logger.WithFields(logrus.Fields{
			"sensor_id":   message.TimeSeries.SensorID,
			"sensor_type": message.TimeSeries.SensorType,
			"timestamp":   message.TimeSeries.Timestamp,
			"value":       message.TimeSeries.Value,
		}).Info("Received time series message")
	} else if message.Batch != nil {
		h.logger.WithFields(logrus.Fields{
			"batch_id": message.Batch.BatchID,
			"count":    message.Batch.Count,
			"timestamp": message.Batch.Timestamp,
		}).Info("Received batch message")
	} else {
		h.logger.WithFields(logrus.Fields{
			"topic":   message.Topic,
			"key":     message.Key,
			"size":    len(message.Value),
		}).Info("Received raw message")
	}
	
	return nil
}

// HandleBatch implements MessageHandler
func (h *DefaultMessageHandler) HandleBatch(ctx context.Context, messages []*ConsumedMessage) error {
	if h.batchProcessor != nil {
		return h.batchProcessor(messages)
	}
	
	// Default behavior: process each message individually
	for _, message := range messages {
		if err := h.HandleMessage(ctx, message); err != nil {
			return err
		}
	}
	
	return nil
}

// HandleError implements MessageHandler
func (h *DefaultMessageHandler) HandleError(ctx context.Context, err error) {
	if h.errorHandler != nil {
		h.errorHandler(err)
		return
	}
	
	// Default behavior: log the error
	h.logger.WithError(err).Error("Consumer error")
}

// ConvertToSensorData converts a time series message to sensor data
func (msg *TimeSeriesMessage) ConvertToSensorData() *models.SensorData {
	return &models.SensorData{
		ID:         msg.ID,
		SensorID:   msg.SensorID,
		SensorType: models.SensorType(msg.SensorType),
		Location:   msg.Location,
		Unit:       msg.Unit,
		Value:      msg.Value,
		Timestamp:  msg.Timestamp,
		Quality:    msg.Quality,
		Tags:       msg.Tags,
		Metadata:   msg.Metadata,
		CreatedAt:  time.Now(),
	}
}

// ConvertToSensorDataBatch converts a batch message to sensor data batch
func (msg *BatchMessage) ConvertToSensorDataBatch() *models.SensorDataBatch {
	readings := make([]models.SensorData, len(msg.Messages))
	for i, tsMsg := range msg.Messages {
		readings[i] = *tsMsg.ConvertToSensorData()
	}
	
	return &models.SensorDataBatch{
		BatchID:   msg.BatchID,
		Readings:  readings,
		Timestamp: msg.Timestamp,
		Source:    "kafka",
		Metadata:  msg.Metadata,
	}
}

// DefaultConsumerConfig returns a default consumer configuration
func DefaultConsumerConfig() *ConsumerConfig {
	return &ConsumerConfig{
		Brokers:           []string{"localhost:9092"},
		GroupID:           "tsiot-consumer",
		Topics:            []string{"timeseries"},
		ClientID:          "tsiot-consumer",
		SessionTimeout:    30 * time.Second,
		HeartbeatInterval: 3 * time.Second,
		RebalanceTimeout:  60 * time.Second,
		OffsetInitial:     "newest",
		OffsetCommitInterval: 1 * time.Second,
		MaxProcessingTime: 30 * time.Second,
		FetchMinBytes:     1,
		FetchMaxBytes:     1024 * 1024, // 1MB
		FetchMaxWait:      500 * time.Millisecond,
		ChannelBufferSize: 256,
		ReturnErrors:      true,
		Security: SecurityConfig{
			Enabled: false,
		},
		Metadata: MetadataConfig{
			RefreshFrequency: 10 * time.Minute,
			FullRefresh:      true,
			RetryMax:         3,
			RetryBackoff:     250 * time.Millisecond,
			Timeout:          60 * time.Second,
		},
	}
}