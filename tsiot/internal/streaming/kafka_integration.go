package streaming

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/models"
)

// KafkaSource implements StreamSource for Apache Kafka
type KafkaSource struct {
	logger   *logrus.Logger
	config   *KafkaSourceConfig
	consumer KafkaConsumer
	metrics  *SourceMetrics
	mu       sync.RWMutex
}

// KafkaSourceConfig configures Kafka source
type KafkaSourceConfig struct {
	Brokers              []string          `json:"brokers"`
	Topic                string            `json:"topic"`
	GroupID              string            `json:"group_id"`
	StartOffset          string            `json:"start_offset"` // earliest, latest
	EnableAutoCommit     bool              `json:"enable_auto_commit"`
	AutoCommitInterval   time.Duration     `json:"auto_commit_interval"`
	SessionTimeout       time.Duration     `json:"session_timeout"`
	HeartbeatInterval    time.Duration     `json:"heartbeat_interval"`
	MaxPollRecords       int               `json:"max_poll_records"`
	FetchMinBytes        int               `json:"fetch_min_bytes"`
	FetchMaxWait         time.Duration     `json:"fetch_max_wait"`
	SecurityProtocol     string            `json:"security_protocol"`
	SASLMechanism        string            `json:"sasl_mechanism"`
	SASLUsername         string            `json:"sasl_username"`
	SASLPassword         string            `json:"sasl_password"`
	SSLCertFile          string            `json:"ssl_cert_file"`
	SSLKeyFile           string            `json:"ssl_key_file"`
	SSLCACertFile        string            `json:"ssl_ca_cert_file"`
	EnableSchemaRegistry bool              `json:"enable_schema_registry"`
	SchemaRegistryURL    string            `json:"schema_registry_url"`
	KeyDeserializer      string            `json:"key_deserializer"`
	ValueDeserializer    string            `json:"value_deserializer"`
	Properties           map[string]string `json:"properties"`
}

// KafkaConsumer interface for Kafka consumer operations
type KafkaConsumer interface {
	Subscribe(topics []string) error
	Poll(timeout time.Duration) (*KafkaMessage, error)
	Commit() error
	Close() error
	GetLag() (int64, error)
}

// KafkaMessage represents a Kafka message
type KafkaMessage struct {
	Topic     string
	Partition int32
	Offset    int64
	Key       []byte
	Value     []byte
	Headers   map[string]string
	Timestamp time.Time
}

// KafkaSink implements StreamSink for Apache Kafka
type KafkaSink struct {
	logger   *logrus.Logger
	config   *KafkaSinkConfig
	producer KafkaProducer
	metrics  *SinkMetrics
	mu       sync.RWMutex
}

// KafkaSinkConfig configures Kafka sink
type KafkaSinkConfig struct {
	Brokers            []string          `json:"brokers"`
	Topic              string            `json:"topic"`
	Partitioner        string            `json:"partitioner"` // round-robin, hash, manual
	RequiredAcks       int               `json:"required_acks"`
	Timeout            time.Duration     `json:"timeout"`
	CompressionType    string            `json:"compression_type"`
	BatchSize          int               `json:"batch_size"`
	LingerMS           time.Duration     `json:"linger_ms"`
	BufferMemory       int64             `json:"buffer_memory"`
	RetryBackoff       time.Duration     `json:"retry_backoff"`
	MaxRetries         int               `json:"max_retries"`
	EnableIdempotence  bool              `json:"enable_idempotence"`
	SecurityProtocol   string            `json:"security_protocol"`
	SASLMechanism      string            `json:"sasl_mechanism"`
	SASLUsername       string            `json:"sasl_username"`
	SASLPassword       string            `json:"sasl_password"`
	SSLCertFile        string            `json:"ssl_cert_file"`
	SSLKeyFile         string            `json:"ssl_key_file"`
	SSLCACertFile      string            `json:"ssl_ca_cert_file"`
	KeySerializer      string            `json:"key_serializer"`
	ValueSerializer    string            `json:"value_serializer"`
	Properties         map[string]string `json:"properties"`
}

// KafkaProducer interface for Kafka producer operations
type KafkaProducer interface {
	Send(message *ProducerMessage) error
	SendAsync(message *ProducerMessage) <-chan *ProducerResult
	Flush(timeout time.Duration) error
	Close() error
}

// ProducerMessage represents a message to be sent to Kafka
type ProducerMessage struct {
	Topic     string
	Partition int32
	Key       []byte
	Value     []byte
	Headers   map[string]string
	Timestamp time.Time
}

// ProducerResult represents the result of sending a message
type ProducerResult struct {
	Message   *ProducerMessage
	Partition int32
	Offset    int64
	Error     error
}

// NewKafkaSource creates a new Kafka source
func NewKafkaSource(logger *logrus.Logger) *KafkaSource {
	return &KafkaSource{
		logger: logger,
		metrics: &SourceMetrics{},
	}
}

// Name returns the source name
func (ks *KafkaSource) Name() string {
	return "kafka"
}

// Start starts the Kafka source
func (ks *KafkaSource) Start(ctx context.Context, config SourceConfig) (<-chan *StreamMessage, error) {
	kafkaConfig, err := ks.parseConfig(config.Properties)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Kafka config: %w", err)
	}

	ks.config = kafkaConfig

	// Create Kafka consumer
	consumer, err := ks.createConsumer()
	if err != nil {
		return nil, fmt.Errorf("failed to create Kafka consumer: %w", err)
	}

	ks.consumer = consumer

	// Subscribe to topic
	if err := consumer.Subscribe([]string{kafkaConfig.Topic}); err != nil {
		return nil, fmt.Errorf("failed to subscribe to topic %s: %w", kafkaConfig.Topic, err)
	}

	// Create message channel
	messageCh := make(chan *StreamMessage, 1000)

	// Start consuming messages
	go ks.consumeMessages(ctx, messageCh)

	ks.logger.WithFields(logrus.Fields{
		"topic":    kafkaConfig.Topic,
		"group_id": kafkaConfig.GroupID,
	}).Info("Started Kafka source")

	return messageCh, nil
}

// Stop stops the Kafka source
func (ks *KafkaSource) Stop() error {
	if ks.consumer != nil {
		return ks.consumer.Close()
	}
	return nil
}

// GetMetrics returns source metrics
func (ks *KafkaSource) GetMetrics() SourceMetrics {
	ks.mu.RLock()
	defer ks.mu.RUnlock()
	return *ks.metrics
}

func (ks *KafkaSource) parseConfig(properties map[string]interface{}) (*KafkaSourceConfig, error) {
	configBytes, err := json.Marshal(properties)
	if err != nil {
		return nil, err
	}

	var config KafkaSourceConfig
	if err := json.Unmarshal(configBytes, &config); err != nil {
		return nil, err
	}

	// Set defaults
	if config.StartOffset == "" {
		config.StartOffset = "latest"
	}
	if config.GroupID == "" {
		config.GroupID = "tsiot-consumer-group"
	}
	if config.SessionTimeout == 0 {
		config.SessionTimeout = 30 * time.Second
	}
	if config.HeartbeatInterval == 0 {
		config.HeartbeatInterval = 3 * time.Second
	}
	if config.MaxPollRecords == 0 {
		config.MaxPollRecords = 500
	}

	return &config, nil
}

func (ks *KafkaSource) createConsumer() (KafkaConsumer, error) {
	// In a real implementation, this would create an actual Kafka consumer
	// For this example, we'll use a mock consumer
	return NewMockKafkaConsumer(ks.config), nil
}

func (ks *KafkaSource) consumeMessages(ctx context.Context, messageCh chan<- *StreamMessage) {
	defer close(messageCh)

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		// Poll for messages
		kafkaMsg, err := ks.consumer.Poll(100 * time.Millisecond)
		if err != nil {
			ks.logger.WithError(err).Error("Failed to poll Kafka message")
			continue
		}

		if kafkaMsg == nil {
			continue // No message available
		}

		// Convert to StreamMessage
		streamMsg := ks.convertToStreamMessage(kafkaMsg)

		// Update metrics
		ks.updateMetrics(kafkaMsg)

		// Send to channel
		select {
		case messageCh <- streamMsg:
		case <-ctx.Done():
			return
		}

		// Commit offset if auto-commit is disabled
		if !ks.config.EnableAutoCommit {
			if err := ks.consumer.Commit(); err != nil {
				ks.logger.WithError(err).Error("Failed to commit Kafka offset")
			}
		}
	}
}

func (ks *KafkaSource) convertToStreamMessage(kafkaMsg *KafkaMessage) *StreamMessage {
	streamMsg := &StreamMessage{
		ID:        fmt.Sprintf("%s-%d-%d", kafkaMsg.Topic, kafkaMsg.Partition, kafkaMsg.Offset),
		Topic:     kafkaMsg.Topic,
		Partition: kafkaMsg.Partition,
		Offset:    kafkaMsg.Offset,
		Key:       kafkaMsg.Key,
		Value:     kafkaMsg.Value,
		Headers:   kafkaMsg.Headers,
		Timestamp: kafkaMsg.Timestamp,
		Metadata: map[string]interface{}{
			"source": "kafka",
		},
	}

	// Try to deserialize as TimeSeries
	if timeSeries, err := ks.deserializeTimeSeries(kafkaMsg.Value); err == nil {
		streamMsg.TimeSeries = timeSeries
	}

	return streamMsg
}

func (ks *KafkaSource) deserializeTimeSeries(data []byte) (*models.TimeSeries, error) {
	var timeSeries models.TimeSeries
	if err := json.Unmarshal(data, &timeSeries); err != nil {
		return nil, err
	}
	return &timeSeries, nil
}

func (ks *KafkaSource) updateMetrics(kafkaMsg *KafkaMessage) {
	ks.mu.Lock()
	defer ks.mu.Unlock()

	ks.metrics.MessagesRead++
	ks.metrics.BytesRead += int64(len(kafkaMsg.Value))
	ks.metrics.LastReadTimestamp = time.Now()

	// Get lag from consumer
	if lag, err := ks.consumer.GetLag(); err == nil {
		ks.metrics.Lag = lag
	}
}

// NewKafkaSink creates a new Kafka sink
func NewKafkaSink(logger *logrus.Logger) *KafkaSink {
	return &KafkaSink{
		logger:  logger,
		metrics: &SinkMetrics{},
	}
}

// Name returns the sink name
func (ks *KafkaSink) Name() string {
	return "kafka"
}

// Write writes a message to Kafka
func (ks *KafkaSink) Write(ctx context.Context, message *StreamMessage) error {
	if ks.producer == nil {
		return fmt.Errorf("Kafka producer not initialized")
	}

	// Convert StreamMessage to ProducerMessage
	producerMsg := &ProducerMessage{
		Topic:     ks.config.Topic,
		Key:       message.Key,
		Value:     message.Value,
		Headers:   message.Headers,
		Timestamp: message.Timestamp,
	}

	// Serialize TimeSeries if present
	if message.TimeSeries != nil {
		serialized, err := ks.serializeTimeSeries(message.TimeSeries)
		if err != nil {
			return fmt.Errorf("failed to serialize time series: %w", err)
		}
		producerMsg.Value = serialized
	}

	// Send message
	if err := ks.producer.Send(producerMsg); err != nil {
		ks.updateErrorMetrics()
		return fmt.Errorf("failed to send message to Kafka: %w", err)
	}

	ks.updateSuccessMetrics(producerMsg)
	return nil
}

// Flush flushes any buffered messages
func (ks *KafkaSink) Flush(ctx context.Context) error {
	if ks.producer == nil {
		return nil
	}

	timeout := 30 * time.Second
	if err := ks.producer.Flush(timeout); err != nil {
		return fmt.Errorf("failed to flush Kafka producer: %w", err)
	}

	ks.mu.Lock()
	ks.metrics.FlushCount++
	ks.mu.Unlock()

	return nil
}

// Close closes the Kafka sink
func (ks *KafkaSink) Close() error {
	if ks.producer != nil {
		return ks.producer.Close()
	}
	return nil
}

// GetMetrics returns sink metrics
func (ks *KafkaSink) GetMetrics() SinkMetrics {
	ks.mu.RLock()
	defer ks.mu.RUnlock()
	return *ks.metrics
}

func (ks *KafkaSink) serializeTimeSeries(timeSeries *models.TimeSeries) ([]byte, error) {
	return json.Marshal(timeSeries)
}

func (ks *KafkaSink) updateSuccessMetrics(msg *ProducerMessage) {
	ks.mu.Lock()
	defer ks.mu.Unlock()

	ks.metrics.MessagesWritten++
	ks.metrics.BytesWritten += int64(len(msg.Value))
	ks.metrics.LastWriteTimestamp = time.Now()
}

func (ks *KafkaSink) updateErrorMetrics() {
	ks.mu.Lock()
	defer ks.mu.Unlock()

	ks.metrics.Errors++
}

// Mock implementations for demonstration

// MockKafkaConsumer implements KafkaConsumer for testing
type MockKafkaConsumer struct {
	config    *KafkaSourceConfig
	topics    []string
	offset    int64
	closed    bool
	messages  []*KafkaMessage
	msgIndex  int
}

func NewMockKafkaConsumer(config *KafkaSourceConfig) *MockKafkaConsumer {
	// Create some mock messages
	messages := make([]*KafkaMessage, 10)
	for i := 0; i < 10; i++ {
		timeSeries := &models.TimeSeries{
			ID:         fmt.Sprintf("mock-ts-%d", i),
			Name:       fmt.Sprintf("Mock Time Series %d", i),
			SensorType: "temperature",
			DataPoints: []models.DataPoint{
				{
					Timestamp: time.Now().Add(time.Duration(-i) * time.Minute),
					Value:     20.0 + float64(i),
					Quality:   1.0,
				},
			},
		}

		data, _ := json.Marshal(timeSeries)
		
		messages[i] = &KafkaMessage{
			Topic:     config.Topic,
			Partition: 0,
			Offset:    int64(i),
			Value:     data,
			Timestamp: time.Now(),
		}
	}

	return &MockKafkaConsumer{
		config:   config,
		messages: messages,
	}
}

func (mc *MockKafkaConsumer) Subscribe(topics []string) error {
	mc.topics = topics
	return nil
}

func (mc *MockKafkaConsumer) Poll(timeout time.Duration) (*KafkaMessage, error) {
	if mc.closed || mc.msgIndex >= len(mc.messages) {
		time.Sleep(timeout)
		return nil, nil
	}

	msg := mc.messages[mc.msgIndex]
	mc.msgIndex++
	return msg, nil
}

func (mc *MockKafkaConsumer) Commit() error {
	return nil
}

func (mc *MockKafkaConsumer) Close() error {
	mc.closed = true
	return nil
}

func (mc *MockKafkaConsumer) GetLag() (int64, error) {
	return int64(len(mc.messages) - mc.msgIndex), nil
}

// MockKafkaProducer implements KafkaProducer for testing
type MockKafkaProducer struct {
	config   *KafkaSinkConfig
	messages []*ProducerMessage
	closed   bool
}

func NewMockKafkaProducer(config *KafkaSinkConfig) *MockKafkaProducer {
	return &MockKafkaProducer{
		config:   config,
		messages: make([]*ProducerMessage, 0),
	}
}

func (mp *MockKafkaProducer) Send(message *ProducerMessage) error {
	if mp.closed {
		return fmt.Errorf("producer is closed")
	}

	mp.messages = append(mp.messages, message)
	return nil
}

func (mp *MockKafkaProducer) SendAsync(message *ProducerMessage) <-chan *ProducerResult {
	resultCh := make(chan *ProducerResult, 1)
	
	go func() {
		defer close(resultCh)
		
		if err := mp.Send(message); err != nil {
			resultCh <- &ProducerResult{
				Message: message,
				Error:   err,
			}
		} else {
			resultCh <- &ProducerResult{
				Message:   message,
				Partition: 0,
				Offset:    int64(len(mp.messages) - 1),
			}
		}
	}()

	return resultCh
}

func (mp *MockKafkaProducer) Flush(timeout time.Duration) error {
	// Mock flush - nothing to do
	return nil
}

func (mp *MockKafkaProducer) Close() error {
	mp.closed = true
	return nil
}

// PulsarSource implements StreamSource for Apache Pulsar
type PulsarSource struct {
	logger  *logrus.Logger
	config  *PulsarSourceConfig
	metrics *SourceMetrics
}

type PulsarSourceConfig struct {
	ServiceURL    string            `json:"service_url"`
	Topic         string            `json:"topic"`
	Subscription  string            `json:"subscription"`
	SubscriptionType string         `json:"subscription_type"`
	Properties    map[string]string `json:"properties"`
}

func NewPulsarSource(logger *logrus.Logger) *PulsarSource {
	return &PulsarSource{
		logger:  logger,
		metrics: &SourceMetrics{},
	}
}

func (ps *PulsarSource) Name() string {
	return "pulsar"
}

func (ps *PulsarSource) Start(ctx context.Context, config SourceConfig) (<-chan *StreamMessage, error) {
	// Mock implementation - would integrate with real Pulsar client
	messageCh := make(chan *StreamMessage, 100)
	
	go func() {
		defer close(messageCh)
		
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		
		counter := 0
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				// Generate mock message
				timeSeries := &models.TimeSeries{
					ID:         fmt.Sprintf("pulsar-ts-%d", counter),
					Name:       fmt.Sprintf("Pulsar Time Series %d", counter),
					SensorType: "pressure",
					DataPoints: []models.DataPoint{
						{
							Timestamp: time.Now(),
							Value:     100.0 + float64(counter),
							Quality:   1.0,
						},
					},
				}

				data, _ := json.Marshal(timeSeries)
				
				message := &StreamMessage{
					ID:         fmt.Sprintf("pulsar-%d", counter),
					Topic:      config.Properties["topic"].(string),
					Value:      data,
					Timestamp:  time.Now(),
					TimeSeries: timeSeries,
					Metadata: map[string]interface{}{
						"source": "pulsar",
					},
				}

				select {
				case messageCh <- message:
					counter++
				case <-ctx.Done():
					return
				}
			}
		}
	}()

	return messageCh, nil
}

func (ps *PulsarSource) Stop() error {
	return nil
}

func (ps *PulsarSource) GetMetrics() SourceMetrics {
	return *ps.metrics
}