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

// Producer represents a Kafka producer for time series data
type Producer struct {
	config          *ProducerConfig
	producer        sarama.AsyncProducer
	logger          *logrus.Logger
	topics          map[string]bool
	topicsMu        sync.RWMutex
	metrics         *ProducerMetrics
	mu              sync.RWMutex
	running         bool
	ctx             context.Context
	cancel          context.CancelFunc
	errorHandler    func(error)
	successHandler  func(*sarama.ProducerMessage)
}

// ProducerConfig contains configuration for Kafka producer
type ProducerConfig struct {
	Brokers              []string      `json:"brokers"`
	ClientID             string        `json:"client_id"`
	DefaultTopic         string        `json:"default_topic"`
	Compression          string        `json:"compression"`          // "none", "gzip", "snappy", "lz4", "zstd"
	BatchSize            int           `json:"batch_size"`
	BatchTimeout         time.Duration `json:"batch_timeout"`
	MaxMessageBytes      int           `json:"max_message_bytes"`
	RequiredAcks         int16         `json:"required_acks"`        // 0, 1, -1
	Timeout              time.Duration `json:"timeout"`
	Retry                RetryConfig   `json:"retry"`
	Security             SecurityConfig `json:"security"`
	EnableIdempotence    bool          `json:"enable_idempotence"`
	TransactionID        string        `json:"transaction_id,omitempty"`
	PartitionStrategy    string        `json:"partition_strategy"`   // "round_robin", "hash", "manual"
	Metadata             MetadataConfig `json:"metadata"`
}

// RetryConfig contains retry configuration
type RetryConfig struct {
	MaxRetries int           `json:"max_retries"`
	RetryDelay time.Duration `json:"retry_delay"`
	Backoff    BackoffConfig `json:"backoff"`
}

// BackoffConfig contains backoff configuration
type BackoffConfig struct {
	Type       string        `json:"type"`        // "fixed", "exponential"
	InitialDelay time.Duration `json:"initial_delay"`
	MaxDelay   time.Duration `json:"max_delay"`
	Multiplier float64       `json:"multiplier"`
}

// SecurityConfig contains security configuration
type SecurityConfig struct {
	Enabled       bool          `json:"enabled"`
	Protocol      string        `json:"protocol"`       // "SASL_PLAINTEXT", "SASL_SSL", "SSL"
	SASL          SASLConfig    `json:"sasl"`
	TLS           TLSConfig     `json:"tls"`
}

// SASLConfig contains SASL configuration
type SASLConfig struct {
	Mechanism string `json:"mechanism"` // "PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"
	Username  string `json:"username"`
	Password  string `json:"password"`
}

// TLSConfig contains TLS configuration
type TLSConfig struct {
	Enabled            bool   `json:"enabled"`
	CertFile           string `json:"cert_file,omitempty"`
	KeyFile            string `json:"key_file,omitempty"`
	CAFile             string `json:"ca_file,omitempty"`
	InsecureSkipVerify bool   `json:"insecure_skip_verify"`
}

// MetadataConfig contains metadata configuration
type MetadataConfig struct {
	RefreshFrequency time.Duration `json:"refresh_frequency"`
	FullRefresh      bool          `json:"full_refresh"`
	RetryMax         int           `json:"retry_max"`
	RetryBackoff     time.Duration `json:"retry_backoff"`
	Timeout          time.Duration `json:"timeout"`
}

// ProducerMetrics contains producer metrics
type ProducerMetrics struct {
	MessagesSent      int64 `json:"messages_sent"`
	MessagesDelivered int64 `json:"messages_delivered"`
	MessagesFailed    int64 `json:"messages_failed"`
	BytesSent         int64 `json:"bytes_sent"`
	BatchesSent       int64 `json:"batches_sent"`
	ErrorCount        int64 `json:"error_count"`
	LastError         error `json:"-"`
	LastErrorTime     time.Time `json:"last_error_time"`
}

// TimeSeriesMessage represents a time series message for Kafka
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

// NewProducer creates a new Kafka producer
func NewProducer(config *ProducerConfig, logger *logrus.Logger) (*Producer, error) {
	if config == nil {
		return nil, errors.NewValidationError("INVALID_CONFIG", "Producer config cannot be nil")
	}
	
	if len(config.Brokers) == 0 {
		return nil, errors.NewValidationError("INVALID_BROKERS", "At least one broker must be specified")
	}
	
	// Create Sarama config
	saramaConfig := sarama.NewConfig()
	
	// Set client ID
	if config.ClientID != "" {
		saramaConfig.ClientID = config.ClientID
	}
	
	// Set producer configuration
	saramaConfig.Producer.Return.Successes = true
	saramaConfig.Producer.Return.Errors = true
	saramaConfig.Producer.RequiredAcks = sarama.RequiredAcks(config.RequiredAcks)
	saramaConfig.Producer.Timeout = config.Timeout
	
	// Set compression
	switch config.Compression {
	case "gzip":
		saramaConfig.Producer.Compression = sarama.CompressionGZIP
	case "snappy":
		saramaConfig.Producer.Compression = sarama.CompressionSnappy
	case "lz4":
		saramaConfig.Producer.Compression = sarama.CompressionLZ4
	case "zstd":
		saramaConfig.Producer.Compression = sarama.CompressionZSTD
	default:
		saramaConfig.Producer.Compression = sarama.CompressionNone
	}
	
	// Set batch configuration
	if config.BatchSize > 0 {
		saramaConfig.Producer.Flush.Messages = config.BatchSize
	}
	if config.BatchTimeout > 0 {
		saramaConfig.Producer.Flush.Frequency = config.BatchTimeout
	}
	if config.MaxMessageBytes > 0 {
		saramaConfig.Producer.MaxMessageBytes = config.MaxMessageBytes
	}
	
	// Set idempotence
	saramaConfig.Producer.Idempotent = config.EnableIdempotence
	saramaConfig.Net.MaxOpenRequests = 1 // Required for idempotence
	
	// Set partitioning strategy
	switch config.PartitionStrategy {
	case "hash":
		saramaConfig.Producer.Partitioner = sarama.NewHashPartitioner
	case "manual":
		saramaConfig.Producer.Partitioner = sarama.NewManualPartitioner
	default:
		saramaConfig.Producer.Partitioner = sarama.NewRoundRobinPartitioner
	}
	
	// Set retry configuration
	saramaConfig.Producer.Retry.Max = config.Retry.MaxRetries
	saramaConfig.Producer.Retry.Backoff = config.Retry.RetryDelay
	
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
	
	// Create async producer
	producer, err := sarama.NewAsyncProducer(config.Brokers, saramaConfig)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeNetwork, "KAFKA_PRODUCER_CREATE_FAILED", "Failed to create Kafka producer")
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	p := &Producer{
		config:   config,
		producer: producer,
		logger:   logger,
		topics:   make(map[string]bool),
		metrics:  &ProducerMetrics{},
		ctx:      ctx,
		cancel:   cancel,
	}
	
	// Start background goroutines
	go p.handleSuccesses()
	go p.handleErrors()
	
	return p, nil
}

// Start starts the producer
func (p *Producer) Start(ctx context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	if p.running {
		return errors.NewValidationError("ALREADY_RUNNING", "Producer is already running")
	}
	
	p.running = true
	p.logger.Info("Kafka producer started")
	
	return nil
}

// Stop stops the producer
func (p *Producer) Stop() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	if !p.running {
		return nil
	}
	
	p.running = false
	p.cancel()
	
	if err := p.producer.Close(); err != nil {
		p.logger.WithError(err).Error("Error closing Kafka producer")
		return err
	}
	
	p.logger.Info("Kafka producer stopped")
	return nil
}

// SendTimeSeries sends a time series data point
func (p *Producer) SendTimeSeries(ctx context.Context, data *models.SensorData, topic string) error {
	if !p.running {
		return errors.NewValidationError("PRODUCER_NOT_RUNNING", "Producer is not running")
	}
	
	if topic == "" {
		topic = p.config.DefaultTopic
	}
	
	if topic == "" {
		return errors.NewValidationError("NO_TOPIC", "No topic specified and no default topic configured")
	}
	
	// Convert to Kafka message
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
		return errors.WrapError(err, errors.ErrorTypeValidation, "SERIALIZATION_FAILED", "Failed to serialize message")
	}
	
	// Create Kafka message
	kafkaMessage := &sarama.ProducerMessage{
		Topic: topic,
		Key:   sarama.StringEncoder(data.SensorID), // Use sensor ID as key for partitioning
		Value: sarama.ByteEncoder(payload),
		Headers: []sarama.RecordHeader{
			{
				Key:   []byte("sensor_type"),
				Value: []byte(data.SensorType),
			},
			{
				Key:   []byte("timestamp"),
				Value: []byte(data.Timestamp.Format(time.RFC3339Nano)),
			},
		},
		Timestamp: data.Timestamp,
	}
	
	// Send message
	select {
	case p.producer.Input() <- kafkaMessage:
		p.metrics.MessagesSent++
		p.metrics.BytesSent += int64(len(payload))
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(p.config.Timeout):
		return errors.NewNetworkError("SEND_TIMEOUT", "Timeout sending message to Kafka")
	}
}

// SendTimeSeriesBatch sends a batch of time series data
func (p *Producer) SendTimeSeriesBatch(ctx context.Context, batch *models.SensorDataBatch, topic string) error {
	if !p.running {
		return errors.NewValidationError("PRODUCER_NOT_RUNNING", "Producer is not running")
	}
	
	if topic == "" {
		topic = p.config.DefaultTopic
	}
	
	if topic == "" {
		return errors.NewValidationError("NO_TOPIC", "No topic specified and no default topic configured")
	}
	
	// Convert batch to Kafka message format
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
	
	// Create Kafka message
	kafkaMessage := &sarama.ProducerMessage{
		Topic: topic,
		Key:   sarama.StringEncoder(batch.BatchID), // Use batch ID as key
		Value: sarama.ByteEncoder(payload),
		Headers: []sarama.RecordHeader{
			{
				Key:   []byte("message_type"),
				Value: []byte("batch"),
			},
			{
				Key:   []byte("batch_size"),
				Value: []byte(fmt.Sprintf("%d", len(messages))),
			},
			{
				Key:   []byte("timestamp"),
				Value: []byte(batch.Timestamp.Format(time.RFC3339Nano)),
			},
		},
		Timestamp: batch.Timestamp,
	}
	
	// Send message
	select {
	case p.producer.Input() <- kafkaMessage:
		p.metrics.MessagesSent++
		p.metrics.BytesSent += int64(len(payload))
		p.metrics.BatchesSent++
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(p.config.Timeout):
		return errors.NewNetworkError("SEND_TIMEOUT", "Timeout sending batch message to Kafka")
	}
}

// handleSuccesses handles successful message deliveries
func (p *Producer) handleSuccesses() {
	for {
		select {
		case success := <-p.producer.Successes():
			p.metrics.MessagesDelivered++
			if p.successHandler != nil {
				p.successHandler(success)
			}
			p.logger.WithFields(logrus.Fields{
				"topic":     success.Topic,
				"partition": success.Partition,
				"offset":    success.Offset,
			}).Debug("Message delivered successfully")
		case <-p.ctx.Done():
			return
		}
	}
}

// handleErrors handles message delivery errors
func (p *Producer) handleErrors() {
	for {
		select {
		case err := <-p.producer.Errors():
			p.metrics.MessagesFailed++
			p.metrics.ErrorCount++
			p.metrics.LastError = err.Err
			p.metrics.LastErrorTime = time.Now()
			
			if p.errorHandler != nil {
				p.errorHandler(err.Err)
			}
			
			p.logger.WithFields(logrus.Fields{
				"topic":     err.Msg.Topic,
				"partition": err.Msg.Partition,
				"error":     err.Err,
			}).Error("Failed to deliver message")
		case <-p.ctx.Done():
			return
		}
	}
}

// SetErrorHandler sets the error handler
func (p *Producer) SetErrorHandler(handler func(error)) {
	p.errorHandler = handler
}

// SetSuccessHandler sets the success handler
func (p *Producer) SetSuccessHandler(handler func(*sarama.ProducerMessage)) {
	p.successHandler = handler
}

// GetMetrics returns producer metrics
func (p *Producer) GetMetrics() *ProducerMetrics {
	return p.metrics
}

// IsRunning returns whether the producer is running
func (p *Producer) IsRunning() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.running
}

// Flush flushes any pending messages
func (p *Producer) Flush(timeout time.Duration) error {
	if !p.running {
		return errors.NewValidationError("PRODUCER_NOT_RUNNING", "Producer is not running")
	}
	
	// Kafka async producer doesn't have a direct flush method
	// We implement this by waiting for all pending messages to be processed
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			select {
			case <-p.producer.Successes():
				// Message processed
			case <-p.producer.Errors():
				// Message failed
			case <-time.After(100 * time.Millisecond):
				// Check if we're done (this is a simplified approach)
				return
			}
		}
	}()
	
	select {
	case <-done:
		return nil
	case <-time.After(timeout):
		return errors.NewNetworkError("FLUSH_TIMEOUT", "Timeout waiting for flush to complete")
	}
}

// configureSecurity configures security settings
func configureSecurity(config *sarama.Config, security *SecurityConfig) {
	if security.TLS.Enabled {
		config.Net.TLS.Enable = true
		// Additional TLS configuration would go here
	}
	
	if security.SASL.Username != "" {
		config.Net.SASL.Enable = true
		config.Net.SASL.User = security.SASL.Username
		config.Net.SASL.Password = security.SASL.Password
		
		switch security.SASL.Mechanism {
		case "SCRAM-SHA-256":
			config.Net.SASL.Mechanism = sarama.SASLTypeSCRAMSHA256
		case "SCRAM-SHA-512":
			config.Net.SASL.Mechanism = sarama.SASLTypeSCRAMSHA512
		default:
			config.Net.SASL.Mechanism = sarama.SASLTypePlaintext
		}
	}
}

// DefaultProducerConfig returns a default producer configuration
func DefaultProducerConfig() *ProducerConfig {
	return &ProducerConfig{
		Brokers:           []string{"localhost:9092"},
		ClientID:          "tsiot-producer",
		DefaultTopic:      "timeseries",
		Compression:       "snappy",
		BatchSize:         1000,
		BatchTimeout:      100 * time.Millisecond,
		MaxMessageBytes:   1000000, // 1MB
		RequiredAcks:      1,
		Timeout:           30 * time.Second,
		EnableIdempotence: false,
		PartitionStrategy: "hash",
		Retry: RetryConfig{
			MaxRetries: 3,
			RetryDelay: 100 * time.Millisecond,
			Backoff: BackoffConfig{
				Type:         "exponential",
				InitialDelay: 100 * time.Millisecond,
				MaxDelay:     1 * time.Second,
				Multiplier:   2.0,
			},
		},
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