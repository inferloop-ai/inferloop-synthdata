package mqtt

import (
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

// Subscriber handles MQTT message subscription for time series data
type Subscriber struct {
	client       *Client
	logger       *logrus.Logger
	metrics      *SubscriberMetrics
	handlers     map[string]MessageHandler
	subscriptions map[string]*Subscription
	mu           sync.RWMutex
	running      bool
	ctx          context.Context
	cancel       context.CancelFunc
	wg           sync.WaitGroup
}

// SubscriberMetrics contains subscriber-specific metrics
type SubscriberMetrics struct {
	MessagesReceived     int64                    `json:"messages_received"`
	BytesReceived        int64                    `json:"bytes_received"`
	ProcessingErrors     int64                    `json:"processing_errors"`
	ProcessingLatency    time.Duration            `json:"processing_latency"`
	SubscriptionCount    int                      `json:"subscription_count"`
	TopicStats           map[string]*TopicStats   `json:"topic_stats"`
	HandlerStats         map[string]*HandlerStats `json:"handler_stats"`
	LastMessageTime      time.Time                `json:"last_message_time"`
	ThroughputPerSecond  float64                  `json:"throughput_per_second"`
}

// HandlerStats contains statistics for message handlers
type HandlerStats struct {
	MessagesHandled   int64         `json:"messages_handled"`
	ProcessingTime    time.Duration `json:"processing_time"`
	ErrorCount        int64         `json:"error_count"`
	LastProcessedTime time.Time     `json:"last_processed_time"`
}

// Subscription represents an active MQTT subscription
type Subscription struct {
	Topic      string         `json:"topic"`
	QoS        QoSLevel       `json:"qos"`
	Handler    MessageHandler `json:"-"`
	CreatedAt  time.Time      `json:"created_at"`
	MessageCount int64        `json:"message_count"`
	Active     bool           `json:"active"`
}

// MessageHandler defines the interface for handling received messages
type MessageHandler interface {
	HandleMessage(ctx context.Context, message *ReceivedMessage) error
	HandleBatch(ctx context.Context, messages []*ReceivedMessage) error
	HandleError(ctx context.Context, err error)
}

// ReceivedMessage represents a received MQTT message
type ReceivedMessage struct {
	Topic       string            `json:"topic"`
	Payload     []byte            `json:"payload"`
	QoS         QoSLevel          `json:"qos"`
	Retained    bool              `json:"retained"`
	Headers     map[string]string `json:"headers,omitempty"`
	Timestamp   time.Time         `json:"timestamp"`
	MessageID   string            `json:"message_id,omitempty"`
	TimeSeries  *TimeSeriesMessage `json:"time_series,omitempty"`
	Batch       *BatchMessage     `json:"batch,omitempty"`
	Alert       *AlertMessage     `json:"alert,omitempty"`
}

// DefaultMessageHandler provides a basic message handler implementation
type DefaultMessageHandler struct {
	logger        *logrus.Logger
	processor     func(*ReceivedMessage) error
	batchProcessor func([]*ReceivedMessage) error
	errorHandler  func(error)
}

// NewSubscriber creates a new MQTT subscriber
func NewSubscriber(client *Client, logger *logrus.Logger) *Subscriber {
	ctx, cancel := context.WithCancel(context.Background())
	
	s := &Subscriber{
		client:        client,
		logger:        logger,
		metrics:       &SubscriberMetrics{
			TopicStats:   make(map[string]*TopicStats),
			HandlerStats: make(map[string]*HandlerStats),
		},
		handlers:      make(map[string]MessageHandler),
		subscriptions: make(map[string]*Subscription),
		ctx:           ctx,
		cancel:        cancel,
	}
	
	return s
}

// Start starts the subscriber
func (s *Subscriber) Start(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if s.running {
		return errors.NewValidationError("ALREADY_RUNNING", "Subscriber is already running")
	}
	
	s.running = true
	
	s.logger.Info("MQTT subscriber started")
	
	return nil
}

// Stop stops the subscriber
func (s *Subscriber) Stop() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if !s.running {
		return nil
	}
	
	// Unsubscribe from all topics
	for topic := range s.subscriptions {
		s.client.client.Unsubscribe(topic)
	}
	
	s.running = false
	s.cancel()
	s.wg.Wait()
	
	s.logger.Info("MQTT subscriber stopped")
	
	return nil
}

// Subscribe subscribes to topics with a message handler
func (s *Subscriber) Subscribe(topics []string, handler MessageHandler) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if !s.running {
		return errors.NewValidationError("SUBSCRIBER_NOT_RUNNING", "Subscriber is not running")
	}
	
	if handler == nil {
		return errors.NewValidationError("INVALID_HANDLER", "Message handler cannot be nil")
	}
	
	for _, topic := range topics {
		fullTopic := s.client.GetTopicName(topic)
		
		// Create MQTT message handler
		mqttHandler := func(client mqtt.Client, msg mqtt.Message) {
			s.handleMQTTMessage(handler, msg)
		}
		
		// Subscribe to topic
		token := s.client.client.Subscribe(fullTopic, byte(s.client.config.QoS), mqttHandler)
		if !token.WaitTimeout(s.client.config.ConnectTimeout) {
			return errors.NewNetworkError("SUBSCRIBE_TIMEOUT", fmt.Sprintf("Timeout subscribing to topic '%s'", fullTopic))
		}
		
		if token.Error() != nil {
			return errors.WrapError(token.Error(), errors.ErrorTypeNetwork, "SUBSCRIBE_FAILED", fmt.Sprintf("Failed to subscribe to topic '%s'", fullTopic))
		}
		
		// Track subscription
		s.subscriptions[fullTopic] = &Subscription{
			Topic:     fullTopic,
			QoS:       s.client.config.QoS,
			Handler:   handler,
			CreatedAt: time.Now(),
			Active:    true,
		}
		
		s.handlers[fullTopic] = handler
		s.metrics.SubscriptionCount++
		
		s.logger.WithField("topic", fullTopic).Info("Subscribed to MQTT topic")
	}
	
	return nil
}

// Unsubscribe unsubscribes from topics
func (s *Subscriber) Unsubscribe(topics []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if !s.running {
		return errors.NewValidationError("SUBSCRIBER_NOT_RUNNING", "Subscriber is not running")
	}
	
	for _, topic := range topics {
		fullTopic := s.client.GetTopicName(topic)
		
		// Unsubscribe from topic
		token := s.client.client.Unsubscribe(fullTopic)
		if !token.WaitTimeout(s.client.config.ConnectTimeout) {
			return errors.NewNetworkError("UNSUBSCRIBE_TIMEOUT", fmt.Sprintf("Timeout unsubscribing from topic '%s'", fullTopic))
		}
		
		if token.Error() != nil {
			return errors.WrapError(token.Error(), errors.ErrorTypeNetwork, "UNSUBSCRIBE_FAILED", fmt.Sprintf("Failed to unsubscribe from topic '%s'", fullTopic))
		}
		
		// Remove subscription tracking
		if subscription, exists := s.subscriptions[fullTopic]; exists {
			subscription.Active = false
			delete(s.subscriptions, fullTopic)
			delete(s.handlers, fullTopic)
			s.metrics.SubscriptionCount--
		}
		
		s.logger.WithField("topic", fullTopic).Info("Unsubscribed from MQTT topic")
	}
	
	return nil
}

// SubscribeToDefaults subscribes to default topics
func (s *Subscriber) SubscribeToDefaults(handler MessageHandler) error {
	return s.Subscribe(s.client.config.Topics.DefaultSubscribe, handler)
}

// handleMQTTMessage handles an incoming MQTT message
func (s *Subscriber) handleMQTTMessage(handler MessageHandler, msg mqtt.Message) {
	start := time.Now()
	
	// Create received message
	receivedMsg := &ReceivedMessage{
		Topic:     msg.Topic(),
		Payload:   msg.Payload(),
		QoS:       QoSLevel(msg.Qos()),
		Retained:  msg.Retained(),
		Timestamp: time.Now(),
	}
	
	// Try to parse the message
	if err := s.parseMessage(receivedMsg); err != nil {
		s.logger.WithError(err).Warn("Failed to parse MQTT message, treating as raw")
	}
	
	// Handle the message
	if err := handler.HandleMessage(s.ctx, receivedMsg); err != nil {
		s.handleMessageError(receivedMsg, handler, err)
		return
	}
	
	// Update metrics
	s.updateReceiveMetrics(receivedMsg, time.Since(start))
	
	// Acknowledge message (if QoS > 0, this is handled automatically by the client)
	msg.Ack()
}

// parseMessage attempts to parse the message payload
func (s *Subscriber) parseMessage(msg *ReceivedMessage) error {
	// Try to determine message type from topic or headers
	messageType := s.determineMessageType(msg.Topic, msg.Headers)
	
	switch messageType {
	case "timeseries":
		var timeSeries TimeSeriesMessage
		if err := json.Unmarshal(msg.Payload, &timeSeries); err != nil {
			return err
		}
		msg.TimeSeries = &timeSeries
		msg.MessageID = timeSeries.ID
		
	case "batch":
		var batch BatchMessage
		if err := json.Unmarshal(msg.Payload, &batch); err != nil {
			return err
		}
		msg.Batch = &batch
		msg.MessageID = batch.BatchID
		
	case "alert":
		var alert AlertMessage
		if err := json.Unmarshal(msg.Payload, &alert); err != nil {
			return err
		}
		msg.Alert = &alert
		msg.MessageID = alert.AlertID
		
	default:
		// Try to parse as time series by default
		var timeSeries TimeSeriesMessage
		if err := json.Unmarshal(msg.Payload, &timeSeries); err != nil {
			return err // Unable to parse
		}
		msg.TimeSeries = &timeSeries
		msg.MessageID = timeSeries.ID
	}
	
	return nil
}

// determineMessageType determines the message type from topic or headers
func (s *Subscriber) determineMessageType(topic string, headers map[string]string) string {
	// Check headers first
	if msgType, exists := headers["message_type"]; exists {
		return msgType
	}
	
	// Infer from topic
	if topic == s.client.GetTopicName(s.client.config.Topics.TimeSeries) {
		return "timeseries"
	}
	if topic == s.client.GetTopicName(s.client.config.Topics.Batch) {
		return "batch"
	}
	if topic == s.client.GetTopicName(s.client.config.Topics.Alerts) {
		return "alert"
	}
	
	// Default to timeseries
	return "timeseries"
}

// handleMessageError handles message processing errors
func (s *Subscriber) handleMessageError(msg *ReceivedMessage, handler MessageHandler, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	s.metrics.ProcessingErrors++
	
	// Update topic stats
	topicStats := s.getTopicStats(msg.Topic)
	topicStats.ErrorCount++
	
	// Update handler stats
	handlerName := fmt.Sprintf("%T", handler)
	handlerStats := s.getHandlerStats(handlerName)
	handlerStats.ErrorCount++
	
	// Call error handler
	handler.HandleError(s.ctx, err)
	
	s.logger.WithFields(logrus.Fields{
		"topic":      msg.Topic,
		"message_id": msg.MessageID,
		"error":      err,
	}).Error("Failed to process MQTT message")
}

// updateReceiveMetrics updates receive metrics
func (s *Subscriber) updateReceiveMetrics(msg *ReceivedMessage, latency time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	s.metrics.MessagesReceived++
	s.metrics.BytesReceived += int64(len(msg.Payload))
	s.metrics.ProcessingLatency = latency
	s.metrics.LastMessageTime = time.Now()
	
	// Update topic stats
	topicStats := s.getTopicStats(msg.Topic)
	topicStats.MessageCount++
	topicStats.ByteCount += int64(len(msg.Payload))
	topicStats.LastPublished = time.Now()
	topicStats.AverageLatency = (topicStats.AverageLatency + latency) / 2
	
	// Update subscription stats
	if subscription, exists := s.subscriptions[msg.Topic]; exists {
		subscription.MessageCount++
	}
}

// getTopicStats gets or creates topic statistics
func (s *Subscriber) getTopicStats(topic string) *TopicStats {
	if stats, exists := s.metrics.TopicStats[topic]; exists {
		return stats
	}
	
	stats := &TopicStats{}
	s.metrics.TopicStats[topic] = stats
	return stats
}

// getHandlerStats gets or creates handler statistics
func (s *Subscriber) getHandlerStats(handlerName string) *HandlerStats {
	if stats, exists := s.metrics.HandlerStats[handlerName]; exists {
		return stats
	}
	
	stats := &HandlerStats{}
	s.metrics.HandlerStats[handlerName] = stats
	return stats
}

// GetMetrics returns subscriber metrics
func (s *Subscriber) GetMetrics() *SubscriberMetrics {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	// Calculate throughput
	if !s.metrics.LastMessageTime.IsZero() {
		elapsed := time.Since(s.metrics.LastMessageTime)
		if elapsed > 0 {
			s.metrics.ThroughputPerSecond = float64(s.metrics.MessagesReceived) / elapsed.Seconds()
		}
	}
	
	return s.metrics
}

// GetSubscriptions returns active subscriptions
func (s *Subscriber) GetSubscriptions() map[string]*Subscription {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	subscriptions := make(map[string]*Subscription)
	for topic, subscription := range s.subscriptions {
		subscriptions[topic] = subscription
	}
	
	return subscriptions
}

// IsRunning returns whether the subscriber is running
func (s *Subscriber) IsRunning() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.running
}

// NewDefaultMessageHandler creates a new default message handler
func NewDefaultMessageHandler(logger *logrus.Logger) *DefaultMessageHandler {
	return &DefaultMessageHandler{
		logger: logger,
	}
}

// SetProcessor sets the message processor function
func (h *DefaultMessageHandler) SetProcessor(processor func(*ReceivedMessage) error) {
	h.processor = processor
}

// SetBatchProcessor sets the batch processor function
func (h *DefaultMessageHandler) SetBatchProcessor(processor func([]*ReceivedMessage) error) {
	h.batchProcessor = processor
}

// SetErrorHandler sets the error handler function
func (h *DefaultMessageHandler) SetErrorHandler(handler func(error)) {
	h.errorHandler = handler
}

// HandleMessage implements MessageHandler
func (h *DefaultMessageHandler) HandleMessage(ctx context.Context, message *ReceivedMessage) error {
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
			"batch_id":  message.Batch.BatchID,
			"count":     message.Batch.Count,
			"timestamp": message.Batch.Timestamp,
		}).Info("Received batch message")
	} else if message.Alert != nil {
		h.logger.WithFields(logrus.Fields{
			"alert_id":  message.Alert.AlertID,
			"type":      message.Alert.Type,
			"severity":  message.Alert.Severity,
			"timestamp": message.Alert.Timestamp,
		}).Info("Received alert message")
	} else {
		h.logger.WithFields(logrus.Fields{
			"topic": message.Topic,
			"size":  len(message.Payload),
		}).Info("Received raw message")
	}
	
	return nil
}

// HandleBatch implements MessageHandler
func (h *DefaultMessageHandler) HandleBatch(ctx context.Context, messages []*ReceivedMessage) error {
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
	h.logger.WithError(err).Error("Message processing error")
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
		Source:    "mqtt",
		Metadata:  msg.Metadata,
	}
}