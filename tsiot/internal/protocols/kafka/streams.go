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

// StreamProcessor provides Kafka Streams-like functionality for time series data
type StreamProcessor struct {
	config          *StreamConfig
	consumer        *Consumer
	producer        *Producer
	logger          *logrus.Logger
	processors      map[string]StreamProcessorFunc
	mu              sync.RWMutex
	running         bool
	ctx             context.Context
	cancel          context.CancelFunc
	wg              sync.WaitGroup
	metrics         *StreamMetrics
}

// StreamConfig contains configuration for stream processing
type StreamConfig struct {
	ApplicationID   string        `json:"application_id"`
	BootstrapServers []string     `json:"bootstrap_servers"`
	InputTopics     []string      `json:"input_topics"`
	OutputTopic     string        `json:"output_topic"`
	ErrorTopic      string        `json:"error_topic,omitempty"`
	ProcessorConfig ProcessorConfig `json:"processor_config"`
	ConsumerConfig  *ConsumerConfig `json:"consumer_config"`
	ProducerConfig  *ProducerConfig `json:"producer_config"`
	StateStore      StateStoreConfig `json:"state_store"`
	Windowing       WindowConfig    `json:"windowing"`
	Parallelism     int             `json:"parallelism"`
}

// ProcessorConfig contains processor-specific configuration
type ProcessorConfig struct {
	BufferSize        int           `json:"buffer_size"`
	FlushInterval     time.Duration `json:"flush_interval"`
	MaxProcessingTime time.Duration `json:"max_processing_time"`
	ErrorHandling     string        `json:"error_handling"` // "ignore", "log", "fail", "retry"
	RetryAttempts     int           `json:"retry_attempts"`
	RetryDelay        time.Duration `json:"retry_delay"`
}

// StateStoreConfig contains state store configuration
type StateStoreConfig struct {
	Enabled     bool          `json:"enabled"`
	Type        string        `json:"type"`        // "memory", "rocksdb", "redis"
	Directory   string        `json:"directory,omitempty"`
	TTL         time.Duration `json:"ttl"`
	CleanupInterval time.Duration `json:"cleanup_interval"`
}

// WindowConfig contains windowing configuration
type WindowConfig struct {
	Type        string        `json:"type"`        // "tumbling", "hopping", "session"
	Size        time.Duration `json:"size"`
	Advance     time.Duration `json:"advance,omitempty"` // For hopping windows
	GracePeriod time.Duration `json:"grace_period"`
	RetentionTime time.Duration `json:"retention_time"`
}

// StreamMetrics contains stream processing metrics
type StreamMetrics struct {
	MessagesProcessed   int64         `json:"messages_processed"`
	MessagesFailed      int64         `json:"messages_failed"`
	MessagesSkipped     int64         `json:"messages_skipped"`
	ProcessingLatency   time.Duration `json:"processing_latency"`
	ThroughputPerSecond float64       `json:"throughput_per_second"`
	ErrorRate           float64       `json:"error_rate"`
	StateStoreSize      int64         `json:"state_store_size"`
	WindowCount         int64         `json:"window_count"`
	LastProcessedTime   time.Time     `json:"last_processed_time"`
}

// StreamProcessorFunc defines the signature for stream processor functions
type StreamProcessorFunc func(ctx context.Context, input *ProcessorInput) (*ProcessorOutput, error)

// ProcessorInput contains input data for stream processors
type ProcessorInput struct {
	Message       *ConsumedMessage      `json:"message"`
	TimeSeries    *TimeSeriesMessage    `json:"time_series,omitempty"`
	Batch         *BatchMessage         `json:"batch,omitempty"`
	Window        *WindowData           `json:"window,omitempty"`
	StateStore    StateStore            `json:"-"`
	ProcessingTime time.Time            `json:"processing_time"`
}

// ProcessorOutput contains output data from stream processors
type ProcessorOutput struct {
	Messages      []*OutputMessage      `json:"messages,omitempty"`
	StateUpdates  []StateUpdate         `json:"state_updates,omitempty"`
	Metrics       map[string]float64    `json:"metrics,omitempty"`
	ForwardToTopic string               `json:"forward_to_topic,omitempty"`
	Skip          bool                  `json:"skip"`
}

// OutputMessage represents a message to be produced
type OutputMessage struct {
	Topic     string            `json:"topic"`
	Key       string            `json:"key"`
	Value     interface{}       `json:"value"`
	Headers   map[string]string `json:"headers,omitempty"`
	Timestamp time.Time         `json:"timestamp"`
}

// StateUpdate represents a state store update
type StateUpdate struct {
	Key       string      `json:"key"`
	Value     interface{} `json:"value"`
	Operation string      `json:"operation"` // "put", "delete", "increment"
	TTL       time.Duration `json:"ttl,omitempty"`
}

// WindowData represents windowed data
type WindowData struct {
	WindowStart time.Time             `json:"window_start"`
	WindowEnd   time.Time             `json:"window_end"`
	Messages    []*ConsumedMessage    `json:"messages"`
	TimeSeries  []*TimeSeriesMessage  `json:"time_series"`
	Aggregates  map[string]float64    `json:"aggregates"`
}

// StateStore interface for state management
type StateStore interface {
	Put(key string, value interface{}, ttl time.Duration) error
	Get(key string) (interface{}, error)
	Delete(key string) error
	Increment(key string, delta float64) (float64, error)
	Range(startKey, endKey string) (map[string]interface{}, error)
	Close() error
}

// MemoryStateStore provides an in-memory state store implementation
type MemoryStateStore struct {
	data map[string]stateEntry
	mu   sync.RWMutex
}

type stateEntry struct {
	value     interface{}
	timestamp time.Time
	ttl       time.Duration
}

// NewStreamProcessor creates a new stream processor
func NewStreamProcessor(config *StreamConfig, logger *logrus.Logger) (*StreamProcessor, error) {
	if config == nil {
		return nil, errors.NewValidationError("INVALID_CONFIG", "Stream config cannot be nil")
	}
	
	if config.ApplicationID == "" {
		return nil, errors.NewValidationError("INVALID_APPLICATION_ID", "Application ID cannot be empty")
	}
	
	if len(config.InputTopics) == 0 {
		return nil, errors.NewValidationError("INVALID_INPUT_TOPICS", "At least one input topic must be specified")
	}
	
	// Create consumer config
	consumerConfig := config.ConsumerConfig
	if consumerConfig == nil {
		consumerConfig = DefaultConsumerConfig()
	}
	consumerConfig.GroupID = config.ApplicationID
	consumerConfig.Topics = config.InputTopics
	
	// Create producer config
	producerConfig := config.ProducerConfig
	if producerConfig == nil {
		producerConfig = DefaultProducerConfig()
	}
	producerConfig.DefaultTopic = config.OutputTopic
	
	ctx, cancel := context.WithCancel(context.Background())
	
	sp := &StreamProcessor{
		config:     config,
		logger:     logger,
		processors: make(map[string]StreamProcessorFunc),
		metrics:    &StreamMetrics{},
		ctx:        ctx,
		cancel:     cancel,
	}
	
	// Create message handler
	messageHandler := &StreamMessageHandler{processor: sp}
	
	// Create consumer
	consumer, err := NewConsumer(consumerConfig, messageHandler, logger)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeNetwork, "CONSUMER_CREATE_FAILED", "Failed to create consumer")
	}
	sp.consumer = consumer
	
	// Create producer
	producer, err := NewProducer(producerConfig, logger)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeNetwork, "PRODUCER_CREATE_FAILED", "Failed to create producer")
	}
	sp.producer = producer
	
	return sp, nil
}

// RegisterProcessor registers a stream processor function
func (sp *StreamProcessor) RegisterProcessor(name string, processor StreamProcessorFunc) {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	sp.processors[name] = processor
}

// Start starts the stream processor
func (sp *StreamProcessor) Start(ctx context.Context) error {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	
	if sp.running {
		return errors.NewValidationError("ALREADY_RUNNING", "Stream processor is already running")
	}
	
	// Start producer
	if err := sp.producer.Start(ctx); err != nil {
		return err
	}
	
	// Start consumer
	if err := sp.consumer.Start(ctx); err != nil {
		sp.producer.Stop()
		return err
	}
	
	sp.running = true
	
	sp.logger.WithFields(logrus.Fields{
		"application_id": sp.config.ApplicationID,
		"input_topics":   sp.config.InputTopics,
		"output_topic":   sp.config.OutputTopic,
	}).Info("Stream processor started")
	
	return nil
}

// Stop stops the stream processor
func (sp *StreamProcessor) Stop() error {
	sp.mu.Lock()
	defer sp.mu.Unlock()
	
	if !sp.running {
		return nil
	}
	
	sp.running = false
	sp.cancel()
	
	// Stop consumer and producer
	var consumerErr, producerErr error
	
	if sp.consumer != nil {
		consumerErr = sp.consumer.Stop()
	}
	
	if sp.producer != nil {
		producerErr = sp.producer.Stop()
	}
	
	// Wait for all goroutines to finish
	sp.wg.Wait()
	
	sp.logger.Info("Stream processor stopped")
	
	// Return first error encountered
	if consumerErr != nil {
		return consumerErr
	}
	return producerErr
}

// GetMetrics returns stream processing metrics
func (sp *StreamProcessor) GetMetrics() *StreamMetrics {
	return sp.metrics
}

// IsRunning returns whether the stream processor is running
func (sp *StreamProcessor) IsRunning() bool {
	sp.mu.RLock()
	defer sp.mu.RUnlock()
	return sp.running
}

// StreamMessageHandler handles messages for stream processing
type StreamMessageHandler struct {
	processor *StreamProcessor
}

// HandleMessage implements MessageHandler
func (h *StreamMessageHandler) HandleMessage(ctx context.Context, message *ConsumedMessage) error {
	start := time.Now()
	
	// Create processor input
	input := &ProcessorInput{
		Message:        message,
		TimeSeries:     message.TimeSeries,
		Batch:          message.Batch,
		ProcessingTime: start,
	}
	
	// Process the message through all registered processors
	h.processor.mu.RLock()
	processors := make(map[string]StreamProcessorFunc)
	for name, proc := range h.processor.processors {
		processors[name] = proc
	}
	h.processor.mu.RUnlock()
	
	for name, processor := range processors {
		output, err := processor(ctx, input)
		if err != nil {
			h.processor.metrics.MessagesFailed++
			h.processor.logger.WithError(err).WithField("processor", name).Error("Processor failed")
			
			switch h.processor.config.ProcessorConfig.ErrorHandling {
			case "ignore":
				continue
			case "log":
				continue
			case "fail":
				return err
			case "retry":
				// Implement retry logic
				for i := 0; i < h.processor.config.ProcessorConfig.RetryAttempts; i++ {
					time.Sleep(h.processor.config.ProcessorConfig.RetryDelay)
					output, err = processor(ctx, input)
					if err == nil {
						break
					}
				}
				if err != nil {
					return err
				}
			}
		}
		
		// Handle processor output
		if output != nil {
			if err := h.handleProcessorOutput(ctx, output); err != nil {
				h.processor.logger.WithError(err).Error("Failed to handle processor output")
				return err
			}
		}
	}
	
	// Update metrics
	h.processor.metrics.MessagesProcessed++
	h.processor.metrics.ProcessingLatency = time.Since(start)
	h.processor.metrics.LastProcessedTime = start
	
	return nil
}

// HandleBatch implements MessageHandler
func (h *StreamMessageHandler) HandleBatch(ctx context.Context, messages []*ConsumedMessage) error {
	for _, message := range messages {
		if err := h.HandleMessage(ctx, message); err != nil {
			return err
		}
	}
	return nil
}

// HandleError implements MessageHandler
func (h *StreamMessageHandler) HandleError(ctx context.Context, err error) {
	h.processor.logger.WithError(err).Error("Stream processing error")
}

// handleProcessorOutput handles the output from a processor
func (h *StreamMessageHandler) handleProcessorOutput(ctx context.Context, output *ProcessorOutput) error {
	if output.Skip {
		h.processor.metrics.MessagesSkipped++
		return nil
	}
	
	// Send output messages
	for _, msg := range output.Messages {
		topic := msg.Topic
		if topic == "" {
			topic = h.processor.config.OutputTopic
		}
		
		// Convert to time series data format
		if timeSeries, ok := msg.Value.(*TimeSeriesMessage); ok {
			sensorData := timeSeries.ConvertToSensorData()
			if err := h.processor.producer.SendTimeSeries(ctx, sensorData, topic); err != nil {
				return err
			}
		} else {
			// For other types, serialize to JSON and send as raw message
			payload, err := json.Marshal(msg.Value)
			if err != nil {
				return err
			}
			
			// This would require extending the producer to handle raw messages
			h.processor.logger.WithField("payload_size", len(payload)).Debug("Would send raw message")
		}
	}
	
	// Handle state updates (if state store is configured)
	for _, update := range output.StateUpdates {
		h.processor.logger.WithFields(logrus.Fields{
			"key":       update.Key,
			"operation": update.Operation,
		}).Debug("State update")
		// State store operations would be implemented here
	}
	
	return nil
}

// Built-in processors

// AggregationProcessor creates an aggregation processor for time series data
func AggregationProcessor(windowSize time.Duration, aggregateFunc func([]*TimeSeriesMessage) map[string]float64) StreamProcessorFunc {
	return func(ctx context.Context, input *ProcessorInput) (*ProcessorOutput, error) {
		if input.TimeSeries == nil {
			return &ProcessorOutput{Skip: true}, nil
		}
		
		// Simple aggregation example (in a real implementation, this would use windowing)
		aggregates := map[string]float64{
			"count": 1,
			"value": input.TimeSeries.Value,
		}
		
		// Create aggregated time series message
		aggregatedMsg := &TimeSeriesMessage{
			ID:         fmt.Sprintf("agg_%s_%d", input.TimeSeries.SensorID, time.Now().Unix()),
			SensorID:   input.TimeSeries.SensorID,
			SensorType: input.TimeSeries.SensorType + "_aggregated",
			Timestamp:  input.TimeSeries.Timestamp,
			Value:      aggregates["value"],
			Tags:       input.TimeSeries.Tags,
			Metadata:   map[string]interface{}{"aggregates": aggregates},
		}
		
		return &ProcessorOutput{
			Messages: []*OutputMessage{
				{
					Key:       input.TimeSeries.SensorID,
					Value:     aggregatedMsg,
					Timestamp: input.TimeSeries.Timestamp,
				},
			},
		}, nil
	}
}

// FilterProcessor creates a filter processor
func FilterProcessor(filterFunc func(*TimeSeriesMessage) bool) StreamProcessorFunc {
	return func(ctx context.Context, input *ProcessorInput) (*ProcessorOutput, error) {
		if input.TimeSeries == nil {
			return &ProcessorOutput{Skip: true}, nil
		}
		
		if !filterFunc(input.TimeSeries) {
			return &ProcessorOutput{Skip: true}, nil
		}
		
		return &ProcessorOutput{
			Messages: []*OutputMessage{
				{
					Key:       input.TimeSeries.SensorID,
					Value:     input.TimeSeries,
					Timestamp: input.TimeSeries.Timestamp,
				},
			},
		}, nil
	}
}

// TransformProcessor creates a transform processor
func TransformProcessor(transformFunc func(*TimeSeriesMessage) *TimeSeriesMessage) StreamProcessorFunc {
	return func(ctx context.Context, input *ProcessorInput) (*ProcessorOutput, error) {
		if input.TimeSeries == nil {
			return &ProcessorOutput{Skip: true}, nil
		}
		
		transformed := transformFunc(input.TimeSeries)
		if transformed == nil {
			return &ProcessorOutput{Skip: true}, nil
		}
		
		return &ProcessorOutput{
			Messages: []*OutputMessage{
				{
					Key:       transformed.SensorID,
					Value:     transformed,
					Timestamp: transformed.Timestamp,
				},
			},
		}, nil
	}
}

// AnomalyDetectionProcessor creates an anomaly detection processor
func AnomalyDetectionProcessor(threshold float64) StreamProcessorFunc {
	return func(ctx context.Context, input *ProcessorInput) (*ProcessorOutput, error) {
		if input.TimeSeries == nil {
			return &ProcessorOutput{Skip: true}, nil
		}
		
		// Simple threshold-based anomaly detection
		isAnomaly := input.TimeSeries.Value > threshold || input.TimeSeries.Value < -threshold
		
		if isAnomaly {
			anomalyMsg := &TimeSeriesMessage{
				ID:         fmt.Sprintf("anomaly_%s_%d", input.TimeSeries.SensorID, time.Now().Unix()),
				SensorID:   input.TimeSeries.SensorID,
				SensorType: "anomaly",
				Timestamp:  input.TimeSeries.Timestamp,
				Value:      input.TimeSeries.Value,
				Tags:       input.TimeSeries.Tags,
				Metadata: map[string]interface{}{
					"original_sensor_type": input.TimeSeries.SensorType,
					"threshold":           threshold,
					"anomaly_detected":    true,
				},
			}
			
			return &ProcessorOutput{
				Messages: []*OutputMessage{
					{
						Topic:     "", // Will use error topic or default topic
						Key:       input.TimeSeries.SensorID,
						Value:     anomalyMsg,
						Timestamp: input.TimeSeries.Timestamp,
						Headers: map[string]string{
							"anomaly": "true",
						},
					},
				},
			}, nil
		}
		
		return &ProcessorOutput{Skip: true}, nil
	}
}

// Memory state store implementation
func NewMemoryStateStore() *MemoryStateStore {
	return &MemoryStateStore{
		data: make(map[string]stateEntry),
	}
}

func (s *MemoryStateStore) Put(key string, value interface{}, ttl time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	s.data[key] = stateEntry{
		value:     value,
		timestamp: time.Now(),
		ttl:       ttl,
	}
	
	return nil
}

func (s *MemoryStateStore) Get(key string) (interface{}, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	entry, exists := s.data[key]
	if !exists {
		return nil, errors.NewValidationError("KEY_NOT_FOUND", "Key not found in state store")
	}
	
	// Check TTL
	if entry.ttl > 0 && time.Since(entry.timestamp) > entry.ttl {
		delete(s.data, key)
		return nil, errors.NewValidationError("KEY_EXPIRED", "Key has expired")
	}
	
	return entry.value, nil
}

func (s *MemoryStateStore) Delete(key string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	delete(s.data, key)
	return nil
}

func (s *MemoryStateStore) Increment(key string, delta float64) (float64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	entry, exists := s.data[key]
	var currentValue float64
	
	if exists {
		if val, ok := entry.value.(float64); ok {
			currentValue = val
		}
	}
	
	newValue := currentValue + delta
	s.data[key] = stateEntry{
		value:     newValue,
		timestamp: time.Now(),
		ttl:       entry.ttl,
	}
	
	return newValue, nil
}

func (s *MemoryStateStore) Range(startKey, endKey string) (map[string]interface{}, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	result := make(map[string]interface{})
	
	for key, entry := range s.data {
		if key >= startKey && key <= endKey {
			// Check TTL
			if entry.ttl > 0 && time.Since(entry.timestamp) > entry.ttl {
				continue
			}
			result[key] = entry.value
		}
	}
	
	return result, nil
}

func (s *MemoryStateStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	s.data = make(map[string]stateEntry)
	return nil
}

// DefaultStreamConfig returns a default stream configuration
func DefaultStreamConfig() *StreamConfig {
	return &StreamConfig{
		ApplicationID:    "tsiot-stream-processor",
		BootstrapServers: []string{"localhost:9092"},
		InputTopics:      []string{"timeseries"},
		OutputTopic:      "processed-timeseries",
		ProcessorConfig: ProcessorConfig{
			BufferSize:        1000,
			FlushInterval:     1 * time.Second,
			MaxProcessingTime: 30 * time.Second,
			ErrorHandling:     "log",
			RetryAttempts:     3,
			RetryDelay:        1 * time.Second,
		},
		StateStore: StateStoreConfig{
			Enabled:         false,
			Type:            "memory",
			TTL:             1 * time.Hour,
			CleanupInterval: 10 * time.Minute,
		},
		Windowing: WindowConfig{
			Type:          "tumbling",
			Size:          1 * time.Minute,
			GracePeriod:   30 * time.Second,
			RetentionTime: 1 * time.Hour,
		},
		Parallelism: 1,
	}
}