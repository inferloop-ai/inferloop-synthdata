package grpc

import (
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/reflection"
	"github.com/sirupsen/logrus"
	
	"github.com/inferloop/tsiot/pkg/errors"
)

// Server represents a gRPC server for time series operations
type Server struct {
	config      *ServerConfig
	server      *grpc.Server
	listener    net.Listener
	logger      *logrus.Logger
	healthCheck *health.Server
	services    map[string]interface{}
	metrics     *ServerMetrics
	mu          sync.RWMutex
	running     bool
	ctx         context.Context
	cancel      context.CancelFunc
	wg          sync.WaitGroup
}

// ServerConfig contains gRPC server configuration
type ServerConfig struct {
	Address               string             `json:"address"`
	Port                  int                `json:"port"`
	TLS                   TLSConfig          `json:"tls"`
	MaxConnectionIdle     time.Duration      `json:"max_connection_idle"`
	MaxConnectionAge      time.Duration      `json:"max_connection_age"`
	MaxConnectionAgeGrace time.Duration      `json:"max_connection_age_grace"`
	Time                  time.Duration      `json:"time"`
	Timeout               time.Duration      `json:"timeout"`
	MaxRecvMsgSize        int                `json:"max_recv_msg_size"`
	MaxSendMsgSize        int                `json:"max_send_msg_size"`
	MaxConcurrentStreams  uint32             `json:"max_concurrent_streams"`
	EnableReflection      bool               `json:"enable_reflection"`
	EnableHealthCheck     bool               `json:"enable_health_check"`
	EnableMetrics         bool               `json:"enable_metrics"`
	EnableTracing         bool               `json:"enable_tracing"`
	EnableLogging         bool               `json:"enable_logging"`
	ServiceConfigs        map[string]interface{} `json:"service_configs"`
}

// TLSConfig contains TLS configuration
type TLSConfig struct {
	Enabled            bool   `json:"enabled"`
	CertFile           string `json:"cert_file"`
	KeyFile            string `json:"key_file"`
	CAFile             string `json:"ca_file"`
	ClientAuth         string `json:"client_auth"` // "none", "request", "require", "verify"
	InsecureSkipVerify bool   `json:"insecure_skip_verify"`
	ServerName         string `json:"server_name"`
}

// ServerMetrics contains server metrics
type ServerMetrics struct {
	ActiveConnections    int64         `json:"active_connections"`
	TotalConnections     int64         `json:"total_connections"`
	RequestsTotal        int64         `json:"requests_total"`
	RequestsInFlight     int64         `json:"requests_in_flight"`
	RequestDuration      time.Duration `json:"request_duration"`
	BytesReceived        int64         `json:"bytes_received"`
	BytesSent            int64         `json:"bytes_sent"`
	ErrorsTotal          int64         `json:"errors_total"`
	StartTime            time.Time     `json:"start_time"`
	LastRequestTime      time.Time     `json:"last_request_time"`
	ServiceMetrics       map[string]*ServiceMetrics `json:"service_metrics"`
}

// ServiceMetrics contains metrics for individual services
type ServiceMetrics struct {
	ServiceName     string        `json:"service_name"`
	RequestsTotal   int64         `json:"requests_total"`
	ErrorsTotal     int64         `json:"errors_total"`
	AverageLatency  time.Duration `json:"average_latency"`
	LastRequestTime time.Time     `json:"last_request_time"`
}

// NewServer creates a new gRPC server
func NewServer(config *ServerConfig, logger *logrus.Logger) (*Server, error) {
	if config == nil {
		return nil, errors.NewValidationError("INVALID_CONFIG", "Server config cannot be nil")
	}
	
	if config.Port <= 0 || config.Port > 65535 {
		return nil, errors.NewValidationError("INVALID_PORT", "Port must be between 1 and 65535")
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	s := &Server{
		config:   config,
		logger:   logger,
		services: make(map[string]interface{}),
		metrics:  &ServerMetrics{
			ServiceMetrics: make(map[string]*ServiceMetrics),
			StartTime:      time.Now(),
		},
		ctx:    ctx,
		cancel: cancel,
	}
	
	// Create gRPC server with options
	grpcServer, err := s.createGRPCServer()
	if err != nil {
		return nil, err
	}
	s.server = grpcServer
	
	// Setup health check service
	if config.EnableHealthCheck {
		s.healthCheck = health.NewServer()
		grpc_health_v1.RegisterHealthServer(s.server, s.healthCheck)
	}
	
	// Enable reflection if configured
	if config.EnableReflection {
		reflection.Register(s.server)
	}
	
	return s, nil
}

// createGRPCServer creates and configures the gRPC server
func (s *Server) createGRPCServer() (*grpc.Server, error) {
	var opts []grpc.ServerOption
	
	// Configure keepalive parameters
	opts = append(opts, grpc.KeepaliveParams(keepalive.ServerParameters{
		MaxConnectionIdle:     s.config.MaxConnectionIdle,
		MaxConnectionAge:      s.config.MaxConnectionAge,
		MaxConnectionAgeGrace: s.config.MaxConnectionAgeGrace,
		Time:                  s.config.Time,
		Timeout:               s.config.Timeout,
	}))
	
	// Configure message size limits
	if s.config.MaxRecvMsgSize > 0 {
		opts = append(opts, grpc.MaxRecvMsgSize(s.config.MaxRecvMsgSize))
	}
	if s.config.MaxSendMsgSize > 0 {
		opts = append(opts, grpc.MaxSendMsgSize(s.config.MaxSendMsgSize))
	}
	
	// Configure concurrent streams
	if s.config.MaxConcurrentStreams > 0 {
		opts = append(opts, grpc.MaxConcurrentStreams(s.config.MaxConcurrentStreams))
	}
	
	// Configure TLS
	if s.config.TLS.Enabled {
		creds, err := s.createTLSCredentials()
		if err != nil {
			return nil, err
		}
		opts = append(opts, grpc.Creds(creds))
	}
	
	// Add interceptors
	interceptors := s.createInterceptors()
	if len(interceptors.unary) > 0 {
		opts = append(opts, grpc.ChainUnaryInterceptor(interceptors.unary...))
	}
	if len(interceptors.stream) > 0 {
		opts = append(opts, grpc.ChainStreamInterceptor(interceptors.stream...))
	}
	
	return grpc.NewServer(opts...), nil
}

// createTLSCredentials creates TLS credentials
func (s *Server) createTLSCredentials() (credentials.TransportCredentials, error) {
	if s.config.TLS.CertFile == "" || s.config.TLS.KeyFile == "" {
		return nil, errors.NewValidationError("INVALID_TLS_CONFIG", "TLS cert and key files must be specified")
	}
	
	cert, err := tls.LoadX509KeyPair(s.config.TLS.CertFile, s.config.TLS.KeyFile)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeNetwork, "TLS_CERT_LOAD_FAILED", "Failed to load TLS certificate")
	}
	
	tlsConfig := &tls.Config{
		Certificates:       []tls.Certificate{cert},
		InsecureSkipVerify: s.config.TLS.InsecureSkipVerify,
	}
	
	if s.config.TLS.ServerName != "" {
		tlsConfig.ServerName = s.config.TLS.ServerName
	}
	
	// Configure client authentication
	switch s.config.TLS.ClientAuth {
	case "request":
		tlsConfig.ClientAuth = tls.RequestClientCert
	case "require":
		tlsConfig.ClientAuth = tls.RequireAnyClientCert
	case "verify":
		tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
	default:
		tlsConfig.ClientAuth = tls.NoClientCert
	}
	
	return credentials.NewTLS(tlsConfig), nil
}

// Start starts the gRPC server
func (s *Server) Start(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if s.running {
		return errors.NewValidationError("ALREADY_RUNNING", "Server is already running")
	}
	
	// Create listener
	address := fmt.Sprintf("%s:%d", s.config.Address, s.config.Port)
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeNetwork, "LISTENER_CREATE_FAILED", "Failed to create listener")
	}
	s.listener = listener
	
	// Set health check status
	if s.healthCheck != nil {
		s.healthCheck.SetServingStatus("", grpc_health_v1.HealthCheckResponse_SERVING)
		for serviceName := range s.services {
			s.healthCheck.SetServingStatus(serviceName, grpc_health_v1.HealthCheckResponse_SERVING)
		}
	}
	
	s.running = true
	s.metrics.StartTime = time.Now()
	
	// Start server in goroutine
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		
		s.logger.WithFields(logrus.Fields{
			"address": address,
			"tls":     s.config.TLS.Enabled,
		}).Info("Starting gRPC server")
		
		if err := s.server.Serve(listener); err != nil {
			if s.running {
				s.logger.WithError(err).Error("gRPC server error")
			}
		}
	}()
	
	// Start metrics collection if enabled
	if s.config.EnableMetrics {
		s.wg.Add(1)
		go s.metricsCollector()
	}
	
	s.logger.WithField("address", address).Info("gRPC server started")
	
	return nil
}

// Stop stops the gRPC server
func (s *Server) Stop() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if !s.running {
		return nil
	}
	
	s.running = false
	s.cancel()
	
	// Set health check status to not serving
	if s.healthCheck != nil {
		s.healthCheck.SetServingStatus("", grpc_health_v1.HealthCheckResponse_NOT_SERVING)
		for serviceName := range s.services {
			s.healthCheck.SetServingStatus(serviceName, grpc_health_v1.HealthCheckResponse_NOT_SERVING)
		}
	}
	
	// Graceful shutdown
	stopped := make(chan struct{})
	go func() {
		s.server.GracefulStop()
		close(stopped)
	}()
	
	// Wait for graceful shutdown or force stop after timeout
	select {
	case <-stopped:
		s.logger.Info("gRPC server stopped gracefully")
	case <-time.After(30 * time.Second):
		s.server.Stop()
		s.logger.Info("gRPC server force stopped")
	}
	
	// Wait for all goroutines to finish
	s.wg.Wait()
	
	s.logger.Info("gRPC server stopped")
	
	return nil
}

// RegisterService registers a service with the server
func (s *Server) RegisterService(name string, service interface{}) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	if s.running {
		return errors.NewValidationError("SERVER_RUNNING", "Cannot register services while server is running")
	}
	
	s.services[name] = service
	s.metrics.ServiceMetrics[name] = &ServiceMetrics{
		ServiceName: name,
	}
	
	s.logger.WithField("service", name).Info("Registered gRPC service")
	
	return nil
}

// GetService returns a registered service
func (s *Server) GetService(name string) (interface{}, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	service, exists := s.services[name]
	return service, exists
}

// IsRunning returns whether the server is running
func (s *Server) IsRunning() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.running
}

// GetMetrics returns server metrics
func (s *Server) GetMetrics() *ServerMetrics {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	// Create a copy to avoid race conditions
	metrics := &ServerMetrics{
		ActiveConnections: s.metrics.ActiveConnections,
		TotalConnections:  s.metrics.TotalConnections,
		RequestsTotal:     s.metrics.RequestsTotal,
		RequestsInFlight:  s.metrics.RequestsInFlight,
		RequestDuration:   s.metrics.RequestDuration,
		BytesReceived:     s.metrics.BytesReceived,
		BytesSent:         s.metrics.BytesSent,
		ErrorsTotal:       s.metrics.ErrorsTotal,
		StartTime:         s.metrics.StartTime,
		LastRequestTime:   s.metrics.LastRequestTime,
		ServiceMetrics:    make(map[string]*ServiceMetrics),
	}
	
	for name, sm := range s.metrics.ServiceMetrics {
		metrics.ServiceMetrics[name] = &ServiceMetrics{
			ServiceName:     sm.ServiceName,
			RequestsTotal:   sm.RequestsTotal,
			ErrorsTotal:     sm.ErrorsTotal,
			AverageLatency:  sm.AverageLatency,
			LastRequestTime: sm.LastRequestTime,
		}
	}
	
	return metrics
}

// GetAddress returns the server address
func (s *Server) GetAddress() string {
	if s.listener != nil {
		return s.listener.Addr().String()
	}
	return fmt.Sprintf("%s:%d", s.config.Address, s.config.Port)
}

// UpdateServiceHealth updates health status for a service
func (s *Server) UpdateServiceHealth(serviceName string, status grpc_health_v1.HealthCheckResponse_ServingStatus) {
	if s.healthCheck != nil {
		s.healthCheck.SetServingStatus(serviceName, status)
		
		statusStr := "UNKNOWN"
		switch status {
		case grpc_health_v1.HealthCheckResponse_SERVING:
			statusStr = "SERVING"
		case grpc_health_v1.HealthCheckResponse_NOT_SERVING:
			statusStr = "NOT_SERVING"
		case grpc_health_v1.HealthCheckResponse_SERVICE_UNKNOWN:
			statusStr = "SERVICE_UNKNOWN"
		}
		
		s.logger.WithFields(logrus.Fields{
			"service": serviceName,
			"status":  statusStr,
		}).Info("Updated service health status")
	}
}

// metricsCollector collects server metrics
func (s *Server) metricsCollector() {
	defer s.wg.Done()
	
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			s.collectMetrics()
		case <-s.ctx.Done():
			return
		}
	}
}

// collectMetrics collects and updates server metrics
func (s *Server) collectMetrics() {
	// This is a simplified metrics collection
	// In a real implementation, you would collect actual metrics from the gRPC server
	
	s.mu.Lock()
	defer s.mu.Unlock()
	
	// Update uptime and other derived metrics
	if s.running {
		// Placeholder for actual metrics collection
		s.logger.Debug("Collecting gRPC server metrics")
	}
}

// DefaultServerConfig returns a default server configuration
func DefaultServerConfig() *ServerConfig {
	return &ServerConfig{
		Address:               "localhost",
		Port:                  50051,
		MaxConnectionIdle:     15 * time.Minute,
		MaxConnectionAge:      30 * time.Minute,
		MaxConnectionAgeGrace: 5 * time.Minute,
		Time:                  5 * time.Minute,
		Timeout:               1 * time.Minute,
		MaxRecvMsgSize:        4 * 1024 * 1024, // 4MB
		MaxSendMsgSize:        4 * 1024 * 1024, // 4MB
		MaxConcurrentStreams:  1000,
		EnableReflection:      true,
		EnableHealthCheck:     true,
		EnableMetrics:         true,
		EnableTracing:         false,
		EnableLogging:         true,
		TLS: TLSConfig{
			Enabled:    false,
			ClientAuth: "none",
		},
		ServiceConfigs: make(map[string]interface{}),
	}
}