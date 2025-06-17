package grpc

import (
	"context"
	"crypto/tls"
	"fmt"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/health/grpc_health_v1"
	"github.com/sirupsen/logrus"
	
	"github.com/inferloop/tsiot/pkg/errors"
)

// Client represents a gRPC client for time series operations
type Client struct {
	config      *ClientConfig
	conn        *grpc.ClientConn
	logger      *logrus.Logger
	clients     map[string]interface{}
	metrics     *ClientMetrics
	mu          sync.RWMutex
	connected   bool
	ctx         context.Context
	cancel      context.CancelFunc
	wg          sync.WaitGroup
}

// ClientConfig contains gRPC client configuration
type ClientConfig struct {
	Address                string        `json:"address"`
	Port                   int           `json:"port"`
	TLS                    TLSConfig     `json:"tls"`
	KeepAlive              time.Duration `json:"keep_alive"`
	KeepAliveTimeout       time.Duration `json:"keep_alive_timeout"`
	PermitWithoutStream    bool          `json:"permit_without_stream"`
	MaxRecvMsgSize         int           `json:"max_recv_msg_size"`
	MaxSendMsgSize         int           `json:"max_send_msg_size"`
	InitialWindowSize      int32         `json:"initial_window_size"`
	InitialConnWindowSize  int32         `json:"initial_conn_window_size"`
	MaxHeaderListSize      uint32        `json:"max_header_list_size"`
	ConnectTimeout         time.Duration `json:"connect_timeout"`
	RequestTimeout         time.Duration `json:"request_timeout"`
	RetryPolicy            RetryPolicy   `json:"retry_policy"`
	EnableHealthCheck      bool          `json:"enable_health_check"`
	HealthCheckInterval    time.Duration `json:"health_check_interval"`
	EnableMetrics          bool          `json:"enable_metrics"`
	EnableTracing          bool          `json:"enable_tracing"`
	EnableLogging          bool          `json:"enable_logging"`
	UserAgent              string        `json:"user_agent"`
	Authority              string        `json:"authority"`
}

// RetryPolicy configures retry behavior
type RetryPolicy struct {
	Enabled            bool          `json:"enabled"`
	MaxAttempts        int           `json:"max_attempts"`
	InitialBackoff     time.Duration `json:"initial_backoff"`
	MaxBackoff         time.Duration `json:"max_backoff"`
	BackoffMultiplier  float64       `json:"backoff_multiplier"`
	RetryableStatusCodes []string     `json:"retryable_status_codes"`
}

// ClientMetrics contains client metrics
type ClientMetrics struct {
	ConnectionsTotal     int64         `json:"connections_total"`
	ConnectionsActive    int64         `json:"connections_active"`
	ConnectionsFailed    int64         `json:"connections_failed"`
	RequestsTotal        int64         `json:"requests_total"`
	RequestsInFlight     int64         `json:"requests_in_flight"`
	RequestsSuccessful   int64         `json:"requests_successful"`
	RequestsFailed       int64         `json:"requests_failed"`
	RequestDuration      time.Duration `json:"request_duration"`
	BytesSent            int64         `json:"bytes_sent"`
	BytesReceived        int64         `json:"bytes_received"`
	LastRequestTime      time.Time     `json:"last_request_time"`
	ConnectedAt          time.Time     `json:"connected_at"`
	LastError            error         `json:"-"`
	LastErrorTime        time.Time     `json:"last_error_time"`
	HealthCheckStatus    string        `json:"health_check_status"`
	ServiceMetrics       map[string]*ClientServiceMetrics `json:"service_metrics"`
}

// ClientServiceMetrics contains metrics for individual services
type ClientServiceMetrics struct {
	ServiceName       string        `json:"service_name"`
	RequestsTotal     int64         `json:"requests_total"`
	RequestsSuccessful int64        `json:"requests_successful"`
	RequestsFailed    int64         `json:"requests_failed"`
	AverageLatency    time.Duration `json:"average_latency"`
	LastRequestTime   time.Time     `json:"last_request_time"`
}

// NewClient creates a new gRPC client
func NewClient(config *ClientConfig, logger *logrus.Logger) (*Client, error) {
	if config == nil {
		return nil, errors.NewValidationError("INVALID_CONFIG", "Client config cannot be nil")
	}
	
	if config.Address == "" {
		return nil, errors.NewValidationError("INVALID_ADDRESS", "Address cannot be empty")
	}
	
	if config.Port <= 0 || config.Port > 65535 {
		return nil, errors.NewValidationError("INVALID_PORT", "Port must be between 1 and 65535")
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	c := &Client{
		config:  config,
		logger:  logger,
		clients: make(map[string]interface{}),
		metrics: &ClientMetrics{
			ServiceMetrics: make(map[string]*ClientServiceMetrics),
		},
		ctx:    ctx,
		cancel: cancel,
	}
	
	return c, nil
}

// Connect establishes a connection to the gRPC server
func (c *Client) Connect(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if c.connected {
		return errors.NewValidationError("ALREADY_CONNECTED", "Client is already connected")
	}
	
	// Create connection options
	opts, err := c.createDialOptions()
	if err != nil {
		return err
	}
	
	// Create connection context with timeout
	connectCtx := ctx
	if c.config.ConnectTimeout > 0 {
		var connectCancel context.CancelFunc
		connectCtx, connectCancel = context.WithTimeout(ctx, c.config.ConnectTimeout)
		defer connectCancel()
	}
	
	// Establish connection
	address := fmt.Sprintf("%s:%d", c.config.Address, c.config.Port)
	conn, err := grpc.DialContext(connectCtx, address, opts...)
	if err != nil {
		c.metrics.ConnectionsFailed++
		return errors.WrapError(err, errors.ErrorTypeNetwork, "GRPC_CONNECT_FAILED", "Failed to connect to gRPC server")
	}
	
	c.conn = conn
	c.connected = true
	c.metrics.ConnectionsTotal++
	c.metrics.ConnectionsActive++
	c.metrics.ConnectedAt = time.Now()
	
	// Start health checking if enabled
	if c.config.EnableHealthCheck {
		c.wg.Add(1)
		go c.healthChecker()
	}
	
	// Start metrics collection if enabled
	if c.config.EnableMetrics {
		c.wg.Add(1)
		go c.metricsCollector()
	}
	
	c.logger.WithField("address", address).Info("Connected to gRPC server")
	
	return nil
}

// Disconnect closes the connection to the gRPC server
func (c *Client) Disconnect() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if !c.connected {
		return nil
	}
	
	c.connected = false
	c.cancel()
	
	// Close connection
	if c.conn != nil {
		if err := c.conn.Close(); err != nil {
			c.logger.WithError(err).Error("Error closing gRPC connection")
		}
		c.conn = nil
	}
	
	// Wait for goroutines to finish
	c.wg.Wait()
	
	c.metrics.ConnectionsActive--
	
	c.logger.Info("Disconnected from gRPC server")
	
	return nil
}

// createDialOptions creates gRPC dial options
func (c *Client) createDialOptions() ([]grpc.DialOption, error) {
	var opts []grpc.DialOption
	
	// Configure credentials
	if c.config.TLS.Enabled {
		creds, err := c.createTLSCredentials()
		if err != nil {
			return nil, err
		}
		opts = append(opts, grpc.WithTransportCredentials(creds))
	} else {
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}
	
	// Configure keepalive
	if c.config.KeepAlive > 0 {
		opts = append(opts, grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                c.config.KeepAlive,
			Timeout:             c.config.KeepAliveTimeout,
			PermitWithoutStream: c.config.PermitWithoutStream,
		}))
	}
	
	// Configure message size limits
	if c.config.MaxRecvMsgSize > 0 {
		opts = append(opts, grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(c.config.MaxRecvMsgSize)))
	}
	if c.config.MaxSendMsgSize > 0 {
		opts = append(opts, grpc.WithDefaultCallOptions(grpc.MaxCallSendMsgSize(c.config.MaxSendMsgSize)))
	}
	
	// Configure window sizes
	if c.config.InitialWindowSize > 0 {
		opts = append(opts, grpc.WithInitialWindowSize(c.config.InitialWindowSize))
	}
	if c.config.InitialConnWindowSize > 0 {
		opts = append(opts, grpc.WithInitialConnWindowSize(c.config.InitialConnWindowSize))
	}
	
	// Configure header list size
	if c.config.MaxHeaderListSize > 0 {
		opts = append(opts, grpc.WithMaxHeaderListSize(c.config.MaxHeaderListSize))
	}
	
	// Configure user agent
	if c.config.UserAgent != "" {
		opts = append(opts, grpc.WithUserAgent(c.config.UserAgent))
	}
	
	// Configure authority
	if c.config.Authority != "" {
		opts = append(opts, grpc.WithAuthority(c.config.Authority))
	}
	
	// Add interceptors
	interceptors := c.createInterceptors()
	if len(interceptors.unary) > 0 {
		opts = append(opts, grpc.WithChainUnaryInterceptor(interceptors.unary...))
	}
	if len(interceptors.stream) > 0 {
		opts = append(opts, grpc.WithChainStreamInterceptor(interceptors.stream...))
	}
	
	return opts, nil
}

// createTLSCredentials creates TLS credentials for the client
func (c *Client) createTLSCredentials() (credentials.TransportCredentials, error) {
	tlsConfig := &tls.Config{
		InsecureSkipVerify: c.config.TLS.InsecureSkipVerify,
	}
	
	if c.config.TLS.ServerName != "" {
		tlsConfig.ServerName = c.config.TLS.ServerName
	}
	
	// Load client certificate if provided
	if c.config.TLS.CertFile != "" && c.config.TLS.KeyFile != "" {
		cert, err := tls.LoadX509KeyPair(c.config.TLS.CertFile, c.config.TLS.KeyFile)
		if err != nil {
			return nil, errors.WrapError(err, errors.ErrorTypeNetwork, "TLS_CERT_LOAD_FAILED", "Failed to load client TLS certificate")
		}
		tlsConfig.Certificates = []tls.Certificate{cert}
	}
	
	return credentials.NewTLS(tlsConfig), nil
}

// GetConnection returns the gRPC connection
func (c *Client) GetConnection() *grpc.ClientConn {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.conn
}

// IsConnected returns whether the client is connected
func (c *Client) IsConnected() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.connected && c.conn != nil
}

// RegisterServiceClient registers a service client
func (c *Client) RegisterServiceClient(name string, client interface{}) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	c.clients[name] = client
	c.metrics.ServiceMetrics[name] = &ClientServiceMetrics{
		ServiceName: name,
	}
	
	c.logger.WithField("service", name).Info("Registered gRPC service client")
	
	return nil
}

// GetServiceClient returns a registered service client
func (c *Client) GetServiceClient(name string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	client, exists := c.clients[name]
	return client, exists
}

// GetMetrics returns client metrics
func (c *Client) GetMetrics() *ClientMetrics {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	// Create a copy to avoid race conditions
	metrics := &ClientMetrics{
		ConnectionsTotal:     c.metrics.ConnectionsTotal,
		ConnectionsActive:    c.metrics.ConnectionsActive,
		ConnectionsFailed:    c.metrics.ConnectionsFailed,
		RequestsTotal:        c.metrics.RequestsTotal,
		RequestsInFlight:     c.metrics.RequestsInFlight,
		RequestsSuccessful:   c.metrics.RequestsSuccessful,
		RequestsFailed:       c.metrics.RequestsFailed,
		RequestDuration:      c.metrics.RequestDuration,
		BytesSent:            c.metrics.BytesSent,
		BytesReceived:        c.metrics.BytesReceived,
		LastRequestTime:      c.metrics.LastRequestTime,
		ConnectedAt:          c.metrics.ConnectedAt,
		LastError:            c.metrics.LastError,
		LastErrorTime:        c.metrics.LastErrorTime,
		HealthCheckStatus:    c.metrics.HealthCheckStatus,
		ServiceMetrics:       make(map[string]*ClientServiceMetrics),
	}
	
	for name, sm := range c.metrics.ServiceMetrics {
		metrics.ServiceMetrics[name] = &ClientServiceMetrics{
			ServiceName:        sm.ServiceName,
			RequestsTotal:      sm.RequestsTotal,
			RequestsSuccessful: sm.RequestsSuccessful,
			RequestsFailed:     sm.RequestsFailed,
			AverageLatency:     sm.AverageLatency,
			LastRequestTime:    sm.LastRequestTime,
		}
	}
	
	return metrics
}

// Ping performs a health check ping
func (c *Client) Ping(ctx context.Context) error {
	if !c.IsConnected() {
		return errors.NewValidationError("NOT_CONNECTED", "Client is not connected")
	}
	
	if !c.config.EnableHealthCheck {
		return errors.NewValidationError("HEALTH_CHECK_DISABLED", "Health check is not enabled")
	}
	
	client := grpc_health_v1.NewHealthClient(c.conn)
	
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	
	resp, err := client.Check(ctx, &grpc_health_v1.HealthCheckRequest{})
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeNetwork, "HEALTH_CHECK_FAILED", "Health check failed")
	}
	
	if resp.Status != grpc_health_v1.HealthCheckResponse_SERVING {
		return errors.NewNetworkError("SERVER_NOT_SERVING", "Server is not serving")
	}
	
	return nil
}

// healthChecker performs periodic health checks
func (c *Client) healthChecker() {
	defer c.wg.Done()
	
	ticker := time.NewTicker(c.config.HealthCheckInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			if c.IsConnected() {
				err := c.Ping(c.ctx)
				
				c.mu.Lock()
				if err != nil {
					c.metrics.HealthCheckStatus = "UNHEALTHY"
					c.metrics.LastError = err
					c.metrics.LastErrorTime = time.Now()
					c.logger.WithError(err).Warn("Health check failed")
				} else {
					c.metrics.HealthCheckStatus = "HEALTHY"
				}
				c.mu.Unlock()
			}
			
		case <-c.ctx.Done():
			return
		}
	}
}

// metricsCollector collects client metrics
func (c *Client) metricsCollector() {
	defer c.wg.Done()
	
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			c.collectMetrics()
		case <-c.ctx.Done():
			return
		}
	}
}

// collectMetrics collects and updates client metrics
func (c *Client) collectMetrics() {
	// This is a simplified metrics collection
	// In a real implementation, you would collect actual metrics from the gRPC client
	
	c.mu.Lock()
	defer c.mu.Unlock()
	
	// Update connection state
	if c.connected && c.conn != nil {
		state := c.conn.GetState()
		c.logger.WithField("state", state.String()).Debug("gRPC connection state")
	}
}

// Wait waits for the connection state to change
func (c *Client) Wait(ctx context.Context) error {
	if !c.IsConnected() {
		return errors.NewValidationError("NOT_CONNECTED", "Client is not connected")
	}
	
	return c.conn.WaitForStateChange(ctx, c.conn.GetState())
}

// Close closes the client and cleans up resources
func (c *Client) Close() error {
	return c.Disconnect()
}

// DefaultClientConfig returns a default client configuration
func DefaultClientConfig() *ClientConfig {
	return &ClientConfig{
		Address:                "localhost",
		Port:                   50051,
		KeepAlive:              30 * time.Second,
		KeepAliveTimeout:       5 * time.Second,
		PermitWithoutStream:    true,
		MaxRecvMsgSize:         4 * 1024 * 1024, // 4MB
		MaxSendMsgSize:         4 * 1024 * 1024, // 4MB
		InitialWindowSize:      64 * 1024,       // 64KB
		InitialConnWindowSize:  64 * 1024,       // 64KB
		MaxHeaderListSize:      8 * 1024,        // 8KB
		ConnectTimeout:         10 * time.Second,
		RequestTimeout:         30 * time.Second,
		EnableHealthCheck:      true,
		HealthCheckInterval:    30 * time.Second,
		EnableMetrics:          true,
		EnableTracing:          false,
		EnableLogging:          true,
		UserAgent:              "tsiot-grpc-client/1.0",
		TLS: TLSConfig{
			Enabled: false,
		},
		RetryPolicy: RetryPolicy{
			Enabled:           true,
			MaxAttempts:       3,
			InitialBackoff:    100 * time.Millisecond,
			MaxBackoff:        30 * time.Second,
			BackoffMultiplier: 2.0,
			RetryableStatusCodes: []string{
				"UNAVAILABLE",
				"DEADLINE_EXCEEDED",
				"RESOURCE_EXHAUSTED",
			},
		},
	}
}