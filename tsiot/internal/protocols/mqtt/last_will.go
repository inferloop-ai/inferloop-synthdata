package mqtt

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/sirupsen/logrus"
	
	"github.com/inferloop/tsiot/pkg/errors"
)

// LastWillManager manages MQTT Last Will and Testament functionality
type LastWillManager struct {
	client *Client
	logger *logrus.Logger
	config *LastWillConfig
}

// LastWillMessage represents a Last Will and Testament message
type LastWillMessage struct {
	ClientID      string                 `json:"client_id"`
	Status        string                 `json:"status"`
	Timestamp     time.Time              `json:"timestamp"`
	Reason        string                 `json:"reason,omitempty"`
	LastSeen      time.Time              `json:"last_seen"`
	SessionInfo   *SessionInfo           `json:"session_info,omitempty"`
	Metrics       *ClientMetrics         `json:"metrics,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// SessionInfo contains information about the client session
type SessionInfo struct {
	ConnectedAt       time.Time     `json:"connected_at"`
	Duration          time.Duration `json:"duration"`
	MessagesPublished int64         `json:"messages_published"`
	MessagesReceived  int64         `json:"messages_received"`
	BytesTransferred  int64         `json:"bytes_transferred"`
	TopicsSubscribed  []string      `json:"topics_subscribed"`
	LastActivity      time.Time     `json:"last_activity"`
}

// LastWillTemplate defines a template for Last Will messages
type LastWillTemplate struct {
	Topic    string                 `json:"topic"`
	QoS      QoSLevel               `json:"qos"`
	Retained bool                   `json:"retained"`
	Template map[string]interface{} `json:"template"`
}

// PresenceManager manages client presence using Last Will
type PresenceManager struct {
	client      *Client
	logger      *logrus.Logger
	presenceTopic string
	statusTopic   string
	heartbeatInterval time.Duration
	lastWillManager   *LastWillManager
}

// NewLastWillManager creates a new Last Will manager
func NewLastWillManager(client *Client, logger *logrus.Logger) *LastWillManager {
	return &LastWillManager{
		client: client,
		logger: logger,
		config: client.config.LastWill,
	}
}

// CreateLastWillMessage creates a Last Will message for the client
func (lwm *LastWillManager) CreateLastWillMessage(reason string) (*LastWillMessage, error) {
	if lwm.config == nil {
		return nil, errors.NewValidationError("NO_LAST_WILL_CONFIG", "Last Will configuration not set")
	}
	
	sessionInfo := lwm.getSessionInfo()
	
	message := &LastWillMessage{
		ClientID:    lwm.client.config.ClientID,
		Status:      "offline",
		Timestamp:   time.Now(),
		Reason:      reason,
		LastSeen:    time.Now(),
		SessionInfo: sessionInfo,
		Metrics:     lwm.client.GetMetrics(),
		Metadata: map[string]interface{}{
			"version":     "1.0",
			"protocol":    "mqtt",
			"broker":      lwm.client.config.Broker,
			"clean_session": lwm.client.config.CleanSession,
		},
	}
	
	return message, nil
}

// SetLastWillMessage sets up the Last Will message for the client
func (lwm *LastWillManager) SetLastWillMessage(reason string) error {
	if lwm.config == nil {
		return errors.NewValidationError("NO_LAST_WILL_CONFIG", "Last Will configuration not set")
	}
	
	// Create Last Will message
	lastWillMsg, err := lwm.CreateLastWillMessage(reason)
	if err != nil {
		return err
	}
	
	// Serialize to JSON
	payload, err := json.Marshal(lastWillMsg)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeValidation, "LAST_WILL_MARSHAL_FAILED", "Failed to marshal Last Will message")
	}
	
	lwm.logger.WithFields(logrus.Fields{
		"topic":   lwm.config.Topic,
		"qos":     lwm.config.QoS,
		"retained": lwm.config.Retained,
		"reason":  reason,
	}).Info("Last Will message configured")
	
	return nil
}

// CreateCustomLastWill creates a custom Last Will message from template
func (lwm *LastWillManager) CreateCustomLastWill(template *LastWillTemplate, variables map[string]interface{}) ([]byte, error) {
	if template == nil {
		return nil, errors.NewValidationError("INVALID_TEMPLATE", "Last Will template cannot be nil")
	}
	
	// Merge template with variables
	message := make(map[string]interface{})
	for key, value := range template.Template {
		message[key] = value
	}
	
	// Replace variables
	for key, value := range variables {
		message[key] = value
	}
	
	// Add standard fields
	message["client_id"] = lwm.client.config.ClientID
	message["timestamp"] = time.Now()
	message["last_seen"] = time.Now()
	
	// Serialize to JSON
	payload, err := json.Marshal(message)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeValidation, "CUSTOM_LAST_WILL_MARSHAL_FAILED", "Failed to marshal custom Last Will message")
	}
	
	return payload, nil
}

// getSessionInfo collects session information
func (lwm *LastWillManager) getSessionInfo() *SessionInfo {
	metrics := lwm.client.GetMetrics()
	subscriptions := lwm.client.GetSubscriber().GetSubscriptions()
	
	topics := make([]string, 0, len(subscriptions))
	for topic := range subscriptions {
		topics = append(topics, topic)
	}
	
	return &SessionInfo{
		ConnectedAt:       metrics.LastConnectedTime,
		Duration:          lwm.client.GetUptime(),
		MessagesPublished: metrics.MessagesPublished,
		MessagesReceived:  metrics.MessagesReceived,
		BytesTransferred:  metrics.BytesPublished + metrics.BytesReceived,
		TopicsSubscribed:  topics,
		LastActivity:      time.Now(),
	}
}

// PublishOnlineStatus publishes an online status message
func (lwm *LastWillManager) PublishOnlineStatus() error {
	if lwm.config == nil {
		return nil // No Last Will configured
	}
	
	onlineMessage := &LastWillMessage{
		ClientID:    lwm.client.config.ClientID,
		Status:      "online",
		Timestamp:   time.Now(),
		LastSeen:    time.Now(),
		SessionInfo: lwm.getSessionInfo(),
		Metrics:     lwm.client.GetMetrics(),
		Metadata: map[string]interface{}{
			"version":       "1.0",
			"protocol":      "mqtt",
			"broker":        lwm.client.config.Broker,
			"clean_session": lwm.client.config.CleanSession,
		},
	}
	
	payload, err := json.Marshal(onlineMessage)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeValidation, "ONLINE_STATUS_MARSHAL_FAILED", "Failed to marshal online status")
	}
	
	// Publish to the same topic as Last Will but with "online" status
	token := lwm.client.client.Publish(lwm.config.Topic, byte(lwm.config.QoS), lwm.config.Retained, payload)
	if !token.WaitTimeout(lwm.client.config.WriteTimeout) {
		return errors.NewNetworkError("ONLINE_STATUS_TIMEOUT", "Timeout publishing online status")
	}
	
	if token.Error() != nil {
		return errors.WrapError(token.Error(), errors.ErrorTypeNetwork, "ONLINE_STATUS_PUBLISH_FAILED", "Failed to publish online status")
	}
	
	lwm.logger.Info("Published online status")
	return nil
}

// NewPresenceManager creates a new presence manager
func NewPresenceManager(client *Client, logger *logrus.Logger, heartbeatInterval time.Duration) *PresenceManager {
	return &PresenceManager{
		client:            client,
		logger:            logger,
		presenceTopic:     fmt.Sprintf("%s/presence", client.config.Topics.Prefix),
		statusTopic:       fmt.Sprintf("%s/status", client.config.Topics.Prefix),
		heartbeatInterval: heartbeatInterval,
		lastWillManager:   NewLastWillManager(client, logger),
	}
}

// Start starts the presence manager
func (pm *PresenceManager) Start() error {
	// Set up Last Will
	if err := pm.lastWillManager.SetLastWillMessage("unexpected_disconnect"); err != nil {
		return err
	}
	
	// Publish initial online status
	if err := pm.PublishPresence("online", "connected"); err != nil {
		return err
	}
	
	// Start heartbeat goroutine
	go pm.heartbeatLoop()
	
	pm.logger.Info("Presence manager started")
	return nil
}

// Stop stops the presence manager
func (pm *PresenceManager) Stop() error {
	// Publish offline status
	if err := pm.PublishPresence("offline", "graceful_disconnect"); err != nil {
		pm.logger.WithError(err).Warn("Failed to publish offline status")
	}
	
	pm.logger.Info("Presence manager stopped")
	return nil
}

// PublishPresence publishes a presence message
func (pm *PresenceManager) PublishPresence(status, reason string) error {
	presence := map[string]interface{}{
		"client_id":  pm.client.config.ClientID,
		"status":     status,
		"timestamp":  time.Now(),
		"reason":     reason,
		"broker":     pm.client.config.Broker,
		"uptime":     pm.client.GetUptime().Seconds(),
		"metrics": map[string]interface{}{
			"messages_published": pm.client.GetMetrics().MessagesPublished,
			"messages_received":  pm.client.GetMetrics().MessagesReceived,
			"bytes_published":    pm.client.GetMetrics().BytesPublished,
			"bytes_received":     pm.client.GetMetrics().BytesReceived,
		},
	}
	
	payload, err := json.Marshal(presence)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeValidation, "PRESENCE_MARSHAL_FAILED", "Failed to marshal presence message")
	}
	
	token := pm.client.client.Publish(pm.presenceTopic, byte(QoSAtLeastOnce), true, payload)
	if !token.WaitTimeout(pm.client.config.WriteTimeout) {
		return errors.NewNetworkError("PRESENCE_TIMEOUT", "Timeout publishing presence")
	}
	
	if token.Error() != nil {
		return errors.WrapError(token.Error(), errors.ErrorTypeNetwork, "PRESENCE_PUBLISH_FAILED", "Failed to publish presence")
	}
	
	return nil
}

// heartbeatLoop sends periodic heartbeat messages
func (pm *PresenceManager) heartbeatLoop() {
	ticker := time.NewTicker(pm.heartbeatInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			if pm.client.IsConnected() {
				if err := pm.client.PublishHeartbeat(pm.client.ctx); err != nil {
					pm.logger.WithError(err).Warn("Failed to publish heartbeat")
				}
				
				// Also update presence
				if err := pm.PublishPresence("online", "heartbeat"); err != nil {
					pm.logger.WithError(err).Warn("Failed to update presence")
				}
			}
			
		case <-pm.client.ctx.Done():
			return
		}
	}
}

// GetLastWillManager returns the Last Will manager
func (pm *PresenceManager) GetLastWillManager() *LastWillManager {
	return pm.lastWillManager
}

// DefaultLastWillConfig returns a default Last Will configuration
func DefaultLastWillConfig(clientID string) *LastWillConfig {
	return &LastWillConfig{
		Topic:    fmt.Sprintf("tsiot/clients/%s/status", clientID),
		Message:  fmt.Sprintf(`{"client_id":"%s","status":"offline","timestamp":"%s","reason":"unexpected_disconnect"}`, clientID, time.Now().Format(time.RFC3339)),
		QoS:      QoSAtLeastOnce,
		Retained: true,
	}
}

// CreateLastWillTemplate creates a Last Will template
func CreateLastWillTemplate(topic string, qos QoSLevel, retained bool) *LastWillTemplate {
	return &LastWillTemplate{
		Topic:    topic,
		QoS:      qos,
		Retained: retained,
		Template: map[string]interface{}{
			"status":    "offline",
			"timestamp": "{{timestamp}}",
			"reason":    "{{reason}}",
			"version":   "1.0",
			"protocol":  "mqtt",
		},
	}
}