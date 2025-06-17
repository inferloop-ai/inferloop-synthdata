package mqtt

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	mqtt "github.com/eclipse/paho.mqtt.golang"
	"github.com/sirupsen/logrus"
	
	"github.com/inferloop/tsiot/pkg/errors"
)

// RetainedMessageManager manages MQTT retained messages
type RetainedMessageManager struct {
	client         *Client
	logger         *logrus.Logger
	retainedMessages map[string]*RetainedMessage
	mu             sync.RWMutex
	config         *RetainedMessageConfig
}

// RetainedMessageConfig contains configuration for retained messages
type RetainedMessageConfig struct {
	Enabled             bool                       `json:"enabled"`
	MaxRetainedMessages int                        `json:"max_retained_messages"`
	TTL                 time.Duration              `json:"ttl"`
	CleanupInterval     time.Duration              `json:"cleanup_interval"`
	Topics              map[string]*TopicRetentionConfig `json:"topics"`
	Policies            map[string]*RetentionPolicy `json:"policies"`
}

// TopicRetentionConfig contains retention configuration for a specific topic
type TopicRetentionConfig struct {
	Enabled       bool          `json:"enabled"`
	TTL           time.Duration `json:"ttl"`
	MaxMessages   int           `json:"max_messages"`
	Policy        string        `json:"policy"` // "latest", "oldest", "all"
	Compression   bool          `json:"compression"`
	Priority      int           `json:"priority"`
}

// RetentionPolicy defines retention behavior
type RetentionPolicy struct {
	Name         string        `json:"name"`
	Description  string        `json:"description"`
	TTL          time.Duration `json:"ttl"`
	MaxSize      int64         `json:"max_size"`      // Max size in bytes
	MaxMessages  int           `json:"max_messages"`
	CleanupRule  string        `json:"cleanup_rule"` // "lru", "fifo", "size", "ttl"
	Conditions   []string      `json:"conditions"`   // Additional conditions
}

// RetainedMessage represents a retained MQTT message
type RetainedMessage struct {
	Topic       string                 `json:"topic"`
	Payload     []byte                 `json:"payload"`
	QoS         QoSLevel               `json:"qos"`
	Timestamp   time.Time              `json:"timestamp"`
	ExpiresAt   time.Time              `json:"expires_at"`
	Size        int64                  `json:"size"`
	MessageType string                 `json:"message_type"`
	Metadata    map[string]interface{} `json:"metadata"`
	Version     int                    `json:"version"`
	Compressed  bool                   `json:"compressed"`
}

// MessageSnapshot represents a snapshot of messages at a point in time
type MessageSnapshot struct {
	Topic       string            `json:"topic"`
	Messages    []*RetainedMessage `json:"messages"`
	Timestamp   time.Time         `json:"timestamp"`
	Count       int               `json:"count"`
	TotalSize   int64             `json:"total_size"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// RetentionStats contains statistics about retained messages
type RetentionStats struct {
	TotalMessages       int                    `json:"total_messages"`
	TotalSize           int64                  `json:"total_size"`
	MessagesByTopic     map[string]int         `json:"messages_by_topic"`
	SizeByTopic         map[string]int64       `json:"size_by_topic"`
	MessagesByType      map[string]int         `json:"messages_by_type"`
	OldestMessage       time.Time              `json:"oldest_message"`
	NewestMessage       time.Time              `json:"newest_message"`
	ExpiredMessages     int                    `json:"expired_messages"`
	CleanupOperations   int64                  `json:"cleanup_operations"`
	LastCleanup         time.Time              `json:"last_cleanup"`
	CompressionRatio    float64                `json:"compression_ratio"`
}

// NewRetainedMessageManager creates a new retained message manager
func NewRetainedMessageManager(client *Client, logger *logrus.Logger, config *RetainedMessageConfig) *RetainedMessageManager {
	if config == nil {
		config = DefaultRetainedMessageConfig()
	}
	
	rmm := &RetainedMessageManager{
		client:           client,
		logger:           logger,
		config:           config,
		retainedMessages: make(map[string]*RetainedMessage),
	}
	
	// Start cleanup goroutine if enabled
	if config.Enabled && config.CleanupInterval > 0 {
		go rmm.cleanupLoop()
	}
	
	return rmm
}

// PublishRetained publishes a retained message
func (rmm *RetainedMessageManager) PublishRetained(ctx context.Context, topic string, payload []byte, qos QoSLevel, ttl time.Duration) error {
	if !rmm.config.Enabled {
		return errors.NewValidationError("RETAINED_MESSAGES_DISABLED", "Retained messages are disabled")
	}
	
	// Check topic retention configuration
	topicConfig := rmm.getTopicConfig(topic)
	if topicConfig != nil && !topicConfig.Enabled {
		return errors.NewValidationError("TOPIC_RETENTION_DISABLED", fmt.Sprintf("Retention disabled for topic '%s'", topic))
	}
	
	// Create retained message
	retainedMsg := &RetainedMessage{
		Topic:       topic,
		Payload:     payload,
		QoS:         qos,
		Timestamp:   time.Now(),
		Size:        int64(len(payload)),
		MessageType: rmm.determineMessageType(payload),
		Metadata:    make(map[string]interface{}),
		Version:     1,
	}
	
	// Set expiration
	if ttl > 0 {
		retainedMsg.ExpiresAt = time.Now().Add(ttl)
	} else if topicConfig != nil && topicConfig.TTL > 0 {
		retainedMsg.ExpiresAt = time.Now().Add(topicConfig.TTL)
	} else if rmm.config.TTL > 0 {
		retainedMsg.ExpiresAt = time.Now().Add(rmm.config.TTL)
	}
	
	// Compress if configured
	if topicConfig != nil && topicConfig.Compression {
		compressedPayload, err := rmm.compressPayload(payload)
		if err != nil {
			rmm.logger.WithError(err).Warn("Failed to compress retained message")
		} else {
			retainedMsg.Payload = compressedPayload
			retainedMsg.Compressed = true
			retainedMsg.Size = int64(len(compressedPayload))
		}
	}
	
	// Publish the message
	token := rmm.client.client.Publish(topic, byte(qos), true, retainedMsg.Payload)
	if !token.WaitTimeout(rmm.client.config.WriteTimeout) {
		return errors.NewNetworkError("RETAINED_PUBLISH_TIMEOUT", "Timeout publishing retained message")
	}
	
	if token.Error() != nil {
		return errors.WrapError(token.Error(), errors.ErrorTypeNetwork, "RETAINED_PUBLISH_FAILED", "Failed to publish retained message")
	}
	
	// Store in local cache
	rmm.storeRetainedMessage(retainedMsg)
	
	rmm.logger.WithFields(logrus.Fields{
		"topic":      topic,
		"size":       retainedMsg.Size,
		"compressed": retainedMsg.Compressed,
		"expires_at": retainedMsg.ExpiresAt,
	}).Debug("Published retained message")
	
	return nil
}

// PublishLatestSensorValue publishes the latest sensor value as retained
func (rmm *RetainedMessageManager) PublishLatestSensorValue(ctx context.Context, data *TimeSeriesMessage) error {
	topic := fmt.Sprintf("%s/latest/%s", rmm.client.config.Topics.TimeSeries, data.SensorID)
	
	payload, err := json.Marshal(data)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeValidation, "SENSOR_VALUE_MARSHAL_FAILED", "Failed to marshal sensor value")
	}
	
	// Use sensor-specific TTL or default
	ttl := rmm.config.TTL
	if topicConfig := rmm.getTopicConfig(topic); topicConfig != nil && topicConfig.TTL > 0 {
		ttl = topicConfig.TTL
	}
	
	return rmm.PublishRetained(ctx, topic, payload, QoSAtLeastOnce, ttl)
}

// PublishDeviceStatus publishes device status as retained
func (rmm *RetainedMessageManager) PublishDeviceStatus(ctx context.Context, deviceID, status string, metadata map[string]interface{}) error {
	topic := fmt.Sprintf("%s/devices/%s/status", rmm.client.config.Topics.Status, deviceID)
	
	statusMsg := map[string]interface{}{
		"device_id":  deviceID,
		"status":     status,
		"timestamp":  time.Now(),
		"metadata":   metadata,
	}
	
	payload, err := json.Marshal(statusMsg)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeValidation, "DEVICE_STATUS_MARSHAL_FAILED", "Failed to marshal device status")
	}
	
	return rmm.PublishRetained(ctx, topic, payload, QoSAtLeastOnce, rmm.config.TTL)
}

// ClearRetained clears a retained message by publishing an empty message
func (rmm *RetainedMessageManager) ClearRetained(ctx context.Context, topic string) error {
	// Publish empty message with retain flag to clear
	token := rmm.client.client.Publish(topic, 0, true, []byte{})
	if !token.WaitTimeout(rmm.client.config.WriteTimeout) {
		return errors.NewNetworkError("CLEAR_RETAINED_TIMEOUT", "Timeout clearing retained message")
	}
	
	if token.Error() != nil {
		return errors.WrapError(token.Error(), errors.ErrorTypeNetwork, "CLEAR_RETAINED_FAILED", "Failed to clear retained message")
	}
	
	// Remove from local cache
	rmm.removeRetainedMessage(topic)
	
	rmm.logger.WithField("topic", topic).Debug("Cleared retained message")
	return nil
}

// GetRetainedMessage retrieves a retained message from local cache
func (rmm *RetainedMessageManager) GetRetainedMessage(topic string) (*RetainedMessage, bool) {
	rmm.mu.RLock()
	defer rmm.mu.RUnlock()
	
	msg, exists := rmm.retainedMessages[topic]
	if !exists {
		return nil, false
	}
	
	// Check if expired
	if !msg.ExpiresAt.IsZero() && time.Now().After(msg.ExpiresAt) {
		return nil, false
	}
	
	return msg, true
}

// ListRetainedMessages lists all retained messages
func (rmm *RetainedMessageManager) ListRetainedMessages() map[string]*RetainedMessage {
	rmm.mu.RLock()
	defer rmm.mu.RUnlock()
	
	messages := make(map[string]*RetainedMessage)
	now := time.Now()
	
	for topic, msg := range rmm.retainedMessages {
		// Skip expired messages
		if !msg.ExpiresAt.IsZero() && now.After(msg.ExpiresAt) {
			continue
		}
		messages[topic] = msg
	}
	
	return messages
}

// CreateSnapshot creates a snapshot of retained messages for a topic pattern
func (rmm *RetainedMessageManager) CreateSnapshot(topicPattern string) (*MessageSnapshot, error) {
	rmm.mu.RLock()
	defer rmm.mu.RUnlock()
	
	var messages []*RetainedMessage
	var totalSize int64
	
	now := time.Now()
	
	for topic, msg := range rmm.retainedMessages {
		// Simple pattern matching (in a real implementation, use proper wildcard matching)
		if rmm.matchesTopic(topic, topicPattern) {
			// Skip expired messages
			if !msg.ExpiresAt.IsZero() && now.After(msg.ExpiresAt) {
				continue
			}
			messages = append(messages, msg)
			totalSize += msg.Size
		}
	}
	
	snapshot := &MessageSnapshot{
		Topic:     topicPattern,
		Messages:  messages,
		Timestamp: now,
		Count:     len(messages),
		TotalSize: totalSize,
		Metadata: map[string]interface{}{
			"client_id": rmm.client.config.ClientID,
			"version":   "1.0",
		},
	}
	
	return snapshot, nil
}

// GetRetentionStats returns retention statistics
func (rmm *RetainedMessageManager) GetRetentionStats() *RetentionStats {
	rmm.mu.RLock()
	defer rmm.mu.RUnlock()
	
	stats := &RetentionStats{
		MessagesByTopic: make(map[string]int),
		SizeByTopic:     make(map[string]int64),
		MessagesByType:  make(map[string]int),
	}
	
	now := time.Now()
	var totalSize int64
	var compressedSize int64
	var originalSize int64
	
	for _, msg := range rmm.retainedMessages {
		// Skip expired messages
		if !msg.ExpiresAt.IsZero() && now.After(msg.ExpiresAt) {
			stats.ExpiredMessages++
			continue
		}
		
		stats.TotalMessages++
		totalSize += msg.Size
		
		stats.MessagesByTopic[msg.Topic]++
		stats.SizeByTopic[msg.Topic] += msg.Size
		stats.MessagesByType[msg.MessageType]++
		
		if msg.Compressed {
			compressedSize += msg.Size
		} else {
			originalSize += msg.Size
		}
		
		// Track oldest and newest
		if stats.OldestMessage.IsZero() || msg.Timestamp.Before(stats.OldestMessage) {
			stats.OldestMessage = msg.Timestamp
		}
		if stats.NewestMessage.IsZero() || msg.Timestamp.After(stats.NewestMessage) {
			stats.NewestMessage = msg.Timestamp
		}
	}
	
	stats.TotalSize = totalSize
	
	if originalSize > 0 {
		stats.CompressionRatio = compressedSize / originalSize
	}
	
	return stats
}

// storeRetainedMessage stores a retained message in local cache
func (rmm *RetainedMessageManager) storeRetainedMessage(msg *RetainedMessage) {
	rmm.mu.Lock()
	defer rmm.mu.Unlock()
	
	// Check limits
	if rmm.config.MaxRetainedMessages > 0 && len(rmm.retainedMessages) >= rmm.config.MaxRetainedMessages {
		// Remove oldest message
		rmm.removeOldestMessage()
	}
	
	rmm.retainedMessages[msg.Topic] = msg
}

// removeRetainedMessage removes a retained message from local cache
func (rmm *RetainedMessageManager) removeRetainedMessage(topic string) {
	rmm.mu.Lock()
	defer rmm.mu.Unlock()
	
	delete(rmm.retainedMessages, topic)
}

// removeOldestMessage removes the oldest retained message
func (rmm *RetainedMessageManager) removeOldestMessage() {
	var oldestTopic string
	var oldestTime time.Time
	
	for topic, msg := range rmm.retainedMessages {
		if oldestTime.IsZero() || msg.Timestamp.Before(oldestTime) {
			oldestTime = msg.Timestamp
			oldestTopic = topic
		}
	}
	
	if oldestTopic != "" {
		delete(rmm.retainedMessages, oldestTopic)
		rmm.logger.WithField("topic", oldestTopic).Debug("Removed oldest retained message")
	}
}

// cleanupLoop runs periodic cleanup of expired messages
func (rmm *RetainedMessageManager) cleanupLoop() {
	ticker := time.NewTicker(rmm.config.CleanupInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			rmm.cleanup()
		case <-rmm.client.ctx.Done():
			return
		}
	}
}

// cleanup removes expired retained messages
func (rmm *RetainedMessageManager) cleanup() {
	rmm.mu.Lock()
	defer rmm.mu.Unlock()
	
	now := time.Now()
	var expiredTopics []string
	
	for topic, msg := range rmm.retainedMessages {
		if !msg.ExpiresAt.IsZero() && now.After(msg.ExpiresAt) {
			expiredTopics = append(expiredTopics, topic)
		}
	}
	
	for _, topic := range expiredTopics {
		delete(rmm.retainedMessages, topic)
	}
	
	if len(expiredTopics) > 0 {
		rmm.logger.WithField("count", len(expiredTopics)).Debug("Cleaned up expired retained messages")
	}
}

// getTopicConfig gets the retention configuration for a topic
func (rmm *RetainedMessageManager) getTopicConfig(topic string) *TopicRetentionConfig {
	if rmm.config.Topics == nil {
		return nil
	}
	
	// Exact match first
	if config, exists := rmm.config.Topics[topic]; exists {
		return config
	}
	
	// Pattern matching (simplified)
	for pattern, config := range rmm.config.Topics {
		if rmm.matchesTopic(topic, pattern) {
			return config
		}
	}
	
	return nil
}

// determineMessageType determines the message type from payload
func (rmm *RetainedMessageManager) determineMessageType(payload []byte) string {
	// Try to parse as JSON and detect type
	var obj map[string]interface{}
	if err := json.Unmarshal(payload, &obj); err != nil {
		return "raw"
	}
	
	if _, exists := obj["sensor_id"]; exists {
		return "timeseries"
	}
	if _, exists := obj["batch_id"]; exists {
		return "batch"
	}
	if _, exists := obj["alert_id"]; exists {
		return "alert"
	}
	if _, exists := obj["status"]; exists {
		return "status"
	}
	
	return "json"
}

// compressPayload compresses the payload (simplified implementation)
func (rmm *RetainedMessageManager) compressPayload(payload []byte) ([]byte, error) {
	// In a real implementation, use proper compression
	return payload, nil
}

// matchesTopic checks if a topic matches a pattern (simplified implementation)
func (rmm *RetainedMessageManager) matchesTopic(topic, pattern string) bool {
	// Simple wildcard matching
	if pattern == "#" || pattern == topic {
		return true
	}
	
	// In a real implementation, implement proper MQTT topic matching
	return false
}

// DefaultRetainedMessageConfig returns a default retained message configuration
func DefaultRetainedMessageConfig() *RetainedMessageConfig {
	return &RetainedMessageConfig{
		Enabled:             true,
		MaxRetainedMessages: 10000,
		TTL:                 24 * time.Hour,
		CleanupInterval:     1 * time.Hour,
		Topics: map[string]*TopicRetentionConfig{
			"tsiot/timeseries/latest/+": {
				Enabled:     true,
				TTL:         1 * time.Hour,
				MaxMessages: 1,
				Policy:      "latest",
				Compression: true,
				Priority:    1,
			},
			"tsiot/devices/+/status": {
				Enabled:     true,
				TTL:         24 * time.Hour,
				MaxMessages: 1,
				Policy:      "latest",
				Compression: false,
				Priority:    2,
			},
		},
		Policies: map[string]*RetentionPolicy{
			"sensor_latest": {
				Name:        "sensor_latest",
				Description: "Keep only the latest value for each sensor",
				TTL:         1 * time.Hour,
				MaxMessages: 1,
				CleanupRule: "latest",
			},
			"device_status": {
				Name:        "device_status",
				Description: "Keep device status messages",
				TTL:         24 * time.Hour,
				MaxMessages: 1,
				CleanupRule: "latest",
			},
		},
	}
}