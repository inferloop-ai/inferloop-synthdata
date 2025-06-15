package mqtt

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	mqtt "github.com/eclipse/paho.mqtt.golang"
	"github.com/sirupsen/logrus"
	
	"github.com/inferloop/tsiot/pkg/models"
	"github.com/inferloop/tsiot/pkg/errors"
)

// Client represents an MQTT client for time series data
type Client struct {
	config       *ClientConfig
	client       mqtt.Client
	logger       *logrus.Logger
	publisher    *Publisher
	subscriber   *Subscriber
	metrics      *ClientMetrics
	mu           sync.RWMutex
	running      bool
	ctx          context.Context
	cancel       context.CancelFunc
	wg           sync.WaitGroup
}

// ClientConfig contains MQTT client configuration
type ClientConfig struct {
	Broker             string            `json:"broker"`
	Port               int               `json:"port"`
	ClientID           string            `json:"client_id"`
	Username           string            `json:"username,omitempty"`
	Password           string            `json:"password,omitempty"`
	CleanSession       bool              `json:"clean_session"`
	KeepAlive          time.Duration     `json:"keep_alive"`
	ConnectTimeout     time.Duration     `json:"connect_timeout"`
	WriteTimeout       time.Duration     `json:"write_timeout"`
	PingTimeout        time.Duration     `json:"ping_timeout"`
	MaxReconnectInterval time.Duration   `json:"max_reconnect_interval"`
	AutoReconnect      bool              `json:"auto_reconnect"`
	TLS                TLSConfig         `json:"tls"`
	LastWill           *LastWillConfig   `json:"last_will,omitempty"`
	RetainedMessages   bool              `json:"retained_messages"`
	QoS                QoSLevel          `json:"qos"`
	Topics             TopicConfig       `json:"topics"`
	Compression        string            `json:"compression"` // "none", "gzip", "zstd"
	MaxMessageSize     int               `json:"max_message_size"`
	BufferSize         int               `json:"buffer_size"`
}

// TLSConfig contains TLS configuration for MQTT
type TLSConfig struct {
	Enabled            bool   `json:"enabled"`
	CertFile           string `json:"cert_file,omitempty"`
	KeyFile            string `json:"key_file,omitempty"`
	CAFile             string `json:"ca_file,omitempty"`
	InsecureSkipVerify bool   `json:"insecure_skip_verify"`
	ServerName         string `json:"server_name,omitempty"`
}

// LastWillConfig contains last will and testament configuration
type LastWillConfig struct {
	Topic    string   `json:"topic"`
	Message  string   `json:"message"`
	QoS      QoSLevel `json:"qos"`
	Retained bool     `json:"retained"`
}

// TopicConfig contains topic configuration
type TopicConfig struct {
	DefaultPublish   string            `json:"default_publish"`
	DefaultSubscribe []string          `json:"default_subscribe"`
	TimeSeries       string            `json:"time_series"`
	Batch            string            `json:"batch"`
	Alerts           string            `json:"alerts"`
	Status           string            `json:"status"`
	Heartbeat        string            `json:"heartbeat"`
	Prefix           string            `json:"prefix,omitempty"`
	Templates        map[string]string `json:"templates,omitempty"`
}

// QoSLevel represents MQTT Quality of Service levels
type QoSLevel byte

const (
	QoSAtMostOnce  QoSLevel = 0
	QoSAtLeastOnce QoSLevel = 1
	QoSExactlyOnce QoSLevel = 2
)

// ClientMetrics contains MQTT client metrics
type ClientMetrics struct {
	MessagesPublished   int64     `json:"messages_published"`
	MessagesReceived    int64     `json:"messages_received"`
	BytesPublished      int64     `json:"bytes_published"`
	BytesReceived       int64     `json:"bytes_received"`
	ConnectionCount     int64     `json:"connection_count"`
	DisconnectionCount  int64     `json:"disconnection_count"`
	ReconnectionCount   int64     `json:"reconnection_count"`
	PublishErrors       int64     `json:"publish_errors"`
	SubscribeErrors     int64     `json:"subscribe_errors"`
	LastError           error     `json:"-"`
	LastErrorTime       time.Time `json:"last_error_time"`
	ConnectionState     string    `json:"connection_state"`
	LastConnectedTime   time.Time `json:"last_connected_time"`
	LastDisconnectedTime time.Time `json:"last_disconnected_time"`
	Uptime              time.Duration `json:"uptime"`
}

// ConnectionHandler defines the interface for handling connection events
type ConnectionHandler interface {
	OnConnect(client mqtt.Client)
	OnConnectionLost(client mqtt.Client, err error)
	OnReconnect(client mqtt.Client, opts *mqtt.ClientOptions)
}

// DefaultConnectionHandler provides a basic connection handler
type DefaultConnectionHandler struct {
	client *Client
}

// NewClient creates a new MQTT client
func NewClient(config *ClientConfig, logger *logrus.Logger) (*Client, error) {
	if config == nil {
		return nil, errors.NewValidationError("INVALID_CONFIG", "MQTT client config cannot be nil")
	}
	
	if config.Broker == "" {
		return nil, errors.NewValidationError("INVALID_BROKER", "MQTT broker must be specified")
	}
	
	if config.ClientID == "" {
		return nil, errors.NewValidationError("INVALID_CLIENT_ID", "MQTT client ID must be specified")
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	client := &Client{
		config:  config,
		logger:  logger,
		metrics: &ClientMetrics{ConnectionState: "disconnected"},
		ctx:     ctx,
		cancel:  cancel,
	}
	
	// Create MQTT client options
	opts := mqtt.NewClientOptions()
	opts.AddBroker(fmt.Sprintf("tcp://%s:%d", config.Broker, config.Port))
	opts.SetClientID(config.ClientID)
	opts.SetCleanSession(config.CleanSession)
	opts.SetKeepAlive(config.KeepAlive)
	opts.SetPingTimeout(config.PingTimeout)
	opts.SetConnectTimeout(config.ConnectTimeout)
	opts.SetWriteTimeout(config.WriteTimeout)
	opts.SetMaxReconnectInterval(config.MaxReconnectInterval)
	opts.SetAutoReconnect(config.AutoReconnect)
	
	// Set authentication
	if config.Username != "" {
		opts.SetUsername(config.Username)
		if config.Password != "" {
			opts.SetPassword(config.Password)
		}
	}
	
	// Configure TLS
	if config.TLS.Enabled {
		tlsConfig := &tls.Config{
			InsecureSkipVerify: config.TLS.InsecureSkipVerify,
		}
		
		if config.TLS.ServerName != "" {
			tlsConfig.ServerName = config.TLS.ServerName
		}
		
		opts.SetTLSConfig(tlsConfig)
	}
	
	// Set last will and testament
	if config.LastWill != nil {
		opts.SetWill(
			config.LastWill.Topic,
			config.LastWill.Message,
			byte(config.LastWill.QoS),
			config.LastWill.Retained,
		)
	}
	
	// Set connection handlers
	connectionHandler := &DefaultConnectionHandler{client: client}
	opts.SetOnConnectHandler(connectionHandler.OnConnect)
	opts.SetConnectionLostHandler(connectionHandler.OnConnectionLost)
	opts.SetReconnectingHandler(connectionHandler.OnReconnect)
	
	// Create MQTT client
	client.client = mqtt.NewClient(opts)
	
	// Create publisher and subscriber
	client.publisher = NewPublisher(client, logger)
	client.subscriber = NewSubscriber(client, logger)
	
	return client, nil
}

// Connect connects to the MQTT broker
func (c *Client) Connect(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if c.running {
		return errors.NewValidationError("ALREADY_CONNECTED", "MQTT client is already connected")
	}
	
	// Connect to broker
	token := c.client.Connect()
	
	// Wait for connection with timeout
	if !token.WaitTimeout(c.config.ConnectTimeout) {
		return errors.NewNetworkError("CONNECTION_TIMEOUT", "Timeout connecting to MQTT broker")
	}
	
	if token.Error() != nil {
		c.metrics.LastError = token.Error()
		c.metrics.LastErrorTime = time.Now()
		return errors.WrapError(token.Error(), errors.ErrorTypeNetwork, "MQTT_CONNECTION_FAILED", "Failed to connect to MQTT broker")
	}
	
	c.running = true
	c.metrics.ConnectionCount++
	c.metrics.ConnectionState = "connected"
	c.metrics.LastConnectedTime = time.Now()
	
	c.logger.WithFields(logrus.Fields{
		"broker":    c.config.Broker,
		"port":      c.config.Port,
		"client_id": c.config.ClientID,
	}).Info("Connected to MQTT broker")
	
	return nil
}

// Disconnect disconnects from the MQTT broker
func (c *Client) Disconnect() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if !c.running {
		return nil
	}
	
	// Stop publisher and subscriber
	if c.publisher != nil {
		c.publisher.Stop()
	}
	if c.subscriber != nil {
		c.subscriber.Stop()
	}
	
	// Disconnect from broker
	c.client.Disconnect(250) // 250ms quiesce time
	
	c.running = false
	c.cancel()
	c.wg.Wait()
	
	c.metrics.ConnectionState = "disconnected"
	c.metrics.LastDisconnectedTime = time.Now()
	c.metrics.DisconnectionCount++
	
	if !c.metrics.LastConnectedTime.IsZero() {
		c.metrics.Uptime += time.Since(c.metrics.LastConnectedTime)
	}
	
	c.logger.Info("Disconnected from MQTT broker")
	
	return nil
}

// IsConnected returns whether the client is connected
func (c *Client) IsConnected() bool {
	return c.client.IsConnected()
}

// PublishTimeSeries publishes time series data
func (c *Client) PublishTimeSeries(ctx context.Context, data *models.SensorData, topic string) error {
	if !c.IsConnected() {
		return errors.NewValidationError("NOT_CONNECTED", "MQTT client is not connected")
	}
	
	return c.publisher.PublishTimeSeries(ctx, data, topic)
}

// PublishTimeSeriesBatch publishes a batch of time series data
func (c *Client) PublishTimeSeriesBatch(ctx context.Context, batch *models.SensorDataBatch, topic string) error {
	if !c.IsConnected() {
		return errors.NewValidationError("NOT_CONNECTED", "MQTT client is not connected")
	}
	
	return c.publisher.PublishTimeSeriesBatch(ctx, batch, topic)
}

// Subscribe subscribes to topics and sets up message handling
func (c *Client) Subscribe(topics []string, handler MessageHandler) error {
	if !c.IsConnected() {
		return errors.NewValidationError("NOT_CONNECTED", "MQTT client is not connected")
	}
	
	return c.subscriber.Subscribe(topics, handler)
}

// Unsubscribe unsubscribes from topics
func (c *Client) Unsubscribe(topics []string) error {
	if !c.IsConnected() {
		return errors.NewValidationError("NOT_CONNECTED", "MQTT client is not connected")
	}
	
	return c.subscriber.Unsubscribe(topics)
}

// PublishHeartbeat publishes a heartbeat message
func (c *Client) PublishHeartbeat(ctx context.Context) error {
	if c.config.Topics.Heartbeat == "" {
		return nil // Heartbeat not configured
	}
	
	heartbeat := map[string]interface{}{
		"client_id":  c.config.ClientID,
		"timestamp":  time.Now(),
		"status":     "alive",
		"uptime":     c.GetUptime(),
		"metrics":    c.GetMetrics(),
	}
	
	payload, err := json.Marshal(heartbeat)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeValidation, "HEARTBEAT_MARSHAL_FAILED", "Failed to marshal heartbeat")
	}
	
	token := c.client.Publish(c.config.Topics.Heartbeat, byte(c.config.QoS), false, payload)
	if !token.WaitTimeout(c.config.WriteTimeout) {
		return errors.NewNetworkError("HEARTBEAT_TIMEOUT", "Timeout publishing heartbeat")
	}
	
	if token.Error() != nil {
		return errors.WrapError(token.Error(), errors.ErrorTypeNetwork, "HEARTBEAT_PUBLISH_FAILED", "Failed to publish heartbeat")
	}
	
	return nil
}

// GetMetrics returns client metrics
func (c *Client) GetMetrics() *ClientMetrics {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	// Calculate current uptime
	if c.running && !c.metrics.LastConnectedTime.IsZero() {
		c.metrics.Uptime = time.Since(c.metrics.LastConnectedTime)
	}
	
	return c.metrics
}

// GetUptime returns the current uptime
func (c *Client) GetUptime() time.Duration {
	if !c.running || c.metrics.LastConnectedTime.IsZero() {
		return 0
	}
	return time.Since(c.metrics.LastConnectedTime)
}

// GetPublisher returns the publisher
func (c *Client) GetPublisher() *Publisher {
	return c.publisher
}

// GetSubscriber returns the subscriber
func (c *Client) GetSubscriber() *Subscriber {
	return c.subscriber
}

// GetTopicName returns the full topic name with prefix
func (c *Client) GetTopicName(topic string) string {
	if c.config.Topics.Prefix != "" {
		return fmt.Sprintf("%s/%s", c.config.Topics.Prefix, topic)
	}
	return topic
}

// OnConnect implements ConnectionHandler
func (h *DefaultConnectionHandler) OnConnect(client mqtt.Client) {
	h.client.metrics.ConnectionCount++
	h.client.metrics.ConnectionState = "connected"
	h.client.metrics.LastConnectedTime = time.Now()
	
	h.client.logger.WithField("client_id", h.client.config.ClientID).Info("MQTT client connected")
}

// OnConnectionLost implements ConnectionHandler
func (h *DefaultConnectionHandler) OnConnectionLost(client mqtt.Client, err error) {
	h.client.metrics.DisconnectionCount++
	h.client.metrics.ConnectionState = "disconnected"
	h.client.metrics.LastDisconnectedTime = time.Now()
	h.client.metrics.LastError = err
	h.client.metrics.LastErrorTime = time.Now()
	
	if !h.client.metrics.LastConnectedTime.IsZero() {
		h.client.metrics.Uptime += time.Since(h.client.metrics.LastConnectedTime)
	}
	
	h.client.logger.WithError(err).Error("MQTT connection lost")
}

// OnReconnect implements ConnectionHandler
func (h *DefaultConnectionHandler) OnReconnect(client mqtt.Client, opts *mqtt.ClientOptions) {
	h.client.metrics.ReconnectionCount++
	h.client.metrics.ConnectionState = "reconnecting"
	
	h.client.logger.Info("MQTT client reconnecting")
}

// DefaultClientConfig returns a default MQTT client configuration
func DefaultClientConfig() *ClientConfig {
	return &ClientConfig{
		Broker:               "localhost",
		Port:                 1883,
		ClientID:             "tsiot-client",
		CleanSession:         true,
		KeepAlive:            60 * time.Second,
		ConnectTimeout:       30 * time.Second,
		WriteTimeout:         10 * time.Second,
		PingTimeout:          10 * time.Second,
		MaxReconnectInterval: 60 * time.Second,
		AutoReconnect:        true,
		TLS: TLSConfig{
			Enabled: false,
		},
		RetainedMessages: false,
		QoS:              QoSAtLeastOnce,
		Topics: TopicConfig{
			DefaultPublish:   "tsiot/timeseries",
			DefaultSubscribe: []string{"tsiot/timeseries"},
			TimeSeries:       "tsiot/timeseries",
			Batch:            "tsiot/batch",
			Alerts:           "tsiot/alerts",
			Status:           "tsiot/status",
			Heartbeat:        "tsiot/heartbeat",
		},
		Compression:    "none",
		MaxMessageSize: 1024 * 1024, // 1MB
		BufferSize:     1000,
	}
}