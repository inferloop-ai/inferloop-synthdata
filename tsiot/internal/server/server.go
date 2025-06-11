package server

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/internal/api"
	"github.com/inferloop/tsiot/pkg/constants"
)

// Server represents the HTTP server
type Server struct {
	httpServer    *http.Server
	metricsServer *http.Server
	router        *mux.Router
	logger        *logrus.Logger
	config        *Config
	handlers      *api.Handlers
}

// Config contains server configuration
type Config struct {
	Host               string        `yaml:"host" json:"host"`
	Port               int           `yaml:"port" json:"port"`
	MetricsPort        int           `yaml:"metrics_port" json:"metrics_port"`
	ReadTimeout        time.Duration `yaml:"read_timeout" json:"read_timeout"`
	WriteTimeout       time.Duration `yaml:"write_timeout" json:"write_timeout"`
	IdleTimeout        time.Duration `yaml:"idle_timeout" json:"idle_timeout"`
	ShutdownTimeout    time.Duration `yaml:"shutdown_timeout" json:"shutdown_timeout"`
	EnableMetrics      bool          `yaml:"enable_metrics" json:"enable_metrics"`
	EnableProfiling    bool          `yaml:"enable_profiling" json:"enable_profiling"`
	EnableCORS         bool          `yaml:"enable_cors" json:"enable_cors"`
	MaxRequestSize     int64         `yaml:"max_request_size" json:"max_request_size"`
	TLSCertFile        string        `yaml:"tls_cert_file,omitempty" json:"tls_cert_file,omitempty"`
	TLSKeyFile         string        `yaml:"tls_key_file,omitempty" json:"tls_key_file,omitempty"`
}

// NewServer creates a new HTTP server instance
func NewServer(config *Config, logger *logrus.Logger) (*Server, error) {
	if config == nil {
		config = getDefaultConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	router := mux.NewRouter()

	// Create handlers
	handlers, err := api.NewHandlers(logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create API handlers: %w", err)
	}

	server := &Server{
		router:   router,
		logger:   logger,
		config:   config,
		handlers: handlers,
	}

	// Setup routes
	server.setupRoutes()

	// Setup middleware
	server.setupMiddleware()

	// Create HTTP server
	server.httpServer = &http.Server{
		Addr:         fmt.Sprintf("%s:%d", config.Host, config.Port),
		Handler:      router,
		ReadTimeout:  config.ReadTimeout,
		WriteTimeout: config.WriteTimeout,
		IdleTimeout:  config.IdleTimeout,
	}

	// Create metrics server if enabled
	if config.EnableMetrics {
		server.setupMetricsServer()
	}

	return server, nil
}

// Start starts the HTTP server
func (s *Server) Start(ctx context.Context) error {
	s.logger.Infof("Starting HTTP server on %s:%d", s.config.Host, s.config.Port)

	// Start metrics server if enabled
	if s.config.EnableMetrics && s.metricsServer != nil {
		go func() {
			s.logger.Infof("Starting metrics server on port %d", s.config.MetricsPort)
			if err := s.metricsServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				s.logger.Errorf("Metrics server error: %v", err)
			}
		}()
	}

	// Start main server
	if s.config.TLSCertFile != "" && s.config.TLSKeyFile != "" {
		s.logger.Info("Starting HTTPS server")
		return s.httpServer.ListenAndServeTLS(s.config.TLSCertFile, s.config.TLSKeyFile)
	}

	return s.httpServer.ListenAndServe()
}

// Stop gracefully stops the HTTP server
func (s *Server) Stop(ctx context.Context) error {
	s.logger.Info("Shutting down HTTP server...")

	// Create context with timeout for shutdown
	shutdownCtx, cancel := context.WithTimeout(ctx, s.config.ShutdownTimeout)
	defer cancel()

	// Shutdown metrics server if running
	if s.metricsServer != nil {
		if err := s.metricsServer.Shutdown(shutdownCtx); err != nil {
			s.logger.Errorf("Error shutting down metrics server: %v", err)
		}
	}

	// Shutdown main server
	if err := s.httpServer.Shutdown(shutdownCtx); err != nil {
		s.logger.Errorf("Error shutting down HTTP server: %v", err)
		return err
	}

	s.logger.Info("HTTP server stopped")
	return nil
}

// setupRoutes sets up the HTTP routes
func (s *Server) setupRoutes() {
	// API routes
	apiRouter := s.router.PathPrefix(constants.APIPrefix).Subrouter()

	// Health endpoints
	s.router.HandleFunc("/health", s.handlers.Health).Methods("GET")
	s.router.HandleFunc("/health/ready", s.handlers.Ready).Methods("GET")
	s.router.HandleFunc("/health/live", s.handlers.Live).Methods("GET")

	// Version endpoint
	s.router.HandleFunc("/version", s.handlers.Version).Methods("GET")

	// Generation endpoints
	apiRouter.HandleFunc("/generate", s.handlers.Generate).Methods("POST")
	apiRouter.HandleFunc("/generate/{id}", s.handlers.GetGeneration).Methods("GET")
	apiRouter.HandleFunc("/generate/{id}/status", s.handlers.GetGenerationStatus).Methods("GET")
	apiRouter.HandleFunc("/generate/{id}/cancel", s.handlers.CancelGeneration).Methods("POST")

	// Validation endpoints
	apiRouter.HandleFunc("/validate", s.handlers.Validate).Methods("POST")
	apiRouter.HandleFunc("/validate/{id}", s.handlers.GetValidation).Methods("GET")
	apiRouter.HandleFunc("/validate/{id}/report", s.handlers.GetValidationReport).Methods("GET")

	// Time series endpoints
	apiRouter.HandleFunc("/timeseries", s.handlers.ListTimeSeries).Methods("GET")
	apiRouter.HandleFunc("/timeseries", s.handlers.CreateTimeSeries).Methods("POST")
	apiRouter.HandleFunc("/timeseries/{id}", s.handlers.GetTimeSeries).Methods("GET")
	apiRouter.HandleFunc("/timeseries/{id}", s.handlers.UpdateTimeSeries).Methods("PUT")
	apiRouter.HandleFunc("/timeseries/{id}", s.handlers.DeleteTimeSeries).Methods("DELETE")
	apiRouter.HandleFunc("/timeseries/{id}/data", s.handlers.GetTimeSeriesData).Methods("GET")
	apiRouter.HandleFunc("/timeseries/{id}/metrics", s.handlers.GetTimeSeriesMetrics).Methods("GET")

	// Generator endpoints
	apiRouter.HandleFunc("/generators", s.handlers.ListGenerators).Methods("GET")
	apiRouter.HandleFunc("/generators/{type}", s.handlers.GetGenerator).Methods("GET")
	apiRouter.HandleFunc("/generators/{type}/config", s.handlers.GetGeneratorConfig).Methods("GET")
	apiRouter.HandleFunc("/generators/{type}/config", s.handlers.UpdateGeneratorConfig).Methods("PUT")

	// Validator endpoints
	apiRouter.HandleFunc("/validators", s.handlers.ListValidators).Methods("GET")
	apiRouter.HandleFunc("/validators/{type}", s.handlers.GetValidator).Methods("GET")
	apiRouter.HandleFunc("/validators/{type}/config", s.handlers.GetValidatorConfig).Methods("GET")
	apiRouter.HandleFunc("/validators/{type}/config", s.handlers.UpdateValidatorConfig).Methods("PUT")

	// Agent endpoints
	apiRouter.HandleFunc("/agents", s.handlers.ListAgents).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}", s.handlers.GetAgent).Methods("GET")
	apiRouter.HandleFunc("/agents/{id}/status", s.handlers.GetAgentStatus).Methods("GET")

	// Job endpoints
	apiRouter.HandleFunc("/jobs", s.handlers.ListJobs).Methods("GET")
	apiRouter.HandleFunc("/jobs/{id}", s.handlers.GetJob).Methods("GET")
	apiRouter.HandleFunc("/jobs/{id}/cancel", s.handlers.CancelJob).Methods("POST")
	apiRouter.HandleFunc("/jobs/{id}/retry", s.handlers.RetryJob).Methods("POST")

	// MCP endpoints
	apiRouter.HandleFunc("/mcp/tools", s.handlers.ListMCPTools).Methods("GET")
	apiRouter.HandleFunc("/mcp/tools/{name}", s.handlers.CallMCPTool).Methods("POST")
	apiRouter.HandleFunc("/mcp/resources", s.handlers.ListMCPResources).Methods("GET")
	apiRouter.HandleFunc("/mcp/resources/{uri}", s.handlers.GetMCPResource).Methods("GET")

	// WebSocket endpoint for MCP
	s.router.HandleFunc("/mcp", s.handlers.MCPWebSocket).Methods("GET")

	// Static file serving for documentation
	s.router.PathPrefix("/docs/").Handler(http.StripPrefix("/docs/", http.FileServer(http.Dir("./docs/"))))

	// Catch-all for 404
	s.router.NotFoundHandler = http.HandlerFunc(s.handlers.NotFound)
}

// setupMiddleware sets up HTTP middleware
func (s *Server) setupMiddleware() {
	// Request logging middleware
	s.router.Use(s.loggingMiddleware)

	// Recovery middleware
	s.router.Use(s.recoveryMiddleware)

	// Request ID middleware
	s.router.Use(s.requestIDMiddleware)

	// CORS middleware
	if s.config.EnableCORS {
		s.router.Use(s.corsMiddleware)
	}

	// Request size limiting middleware
	s.router.Use(s.requestSizeLimitMiddleware)

	// Security headers middleware
	s.router.Use(s.securityHeadersMiddleware)

	// Rate limiting middleware
	s.router.Use(s.rateLimitMiddleware)
}

// setupMetricsServer sets up the metrics server
func (s *Server) setupMetricsServer() {
	metricsRouter := mux.NewRouter()
	
	// Prometheus metrics endpoint
	metricsRouter.Handle("/metrics", s.handlers.Metrics()).Methods("GET")
	
	// Health check for metrics server
	metricsRouter.HandleFunc("/health", s.handlers.Health).Methods("GET")

	// Profiling endpoints if enabled
	if s.config.EnableProfiling {
		s.setupProfilingRoutes(metricsRouter)
	}

	s.metricsServer = &http.Server{
		Addr:         fmt.Sprintf("%s:%d", s.config.Host, s.config.MetricsPort),
		Handler:      metricsRouter,
		ReadTimeout:  s.config.ReadTimeout,
		WriteTimeout: s.config.WriteTimeout,
		IdleTimeout:  s.config.IdleTimeout,
	}
}

// setupProfilingRoutes sets up profiling routes
func (s *Server) setupProfilingRoutes(router *mux.Router) {
	// pprof endpoints
	router.HandleFunc("/debug/pprof/", s.handlers.PprofIndex)
	router.HandleFunc("/debug/pprof/cmdline", s.handlers.PprofCmdline)
	router.HandleFunc("/debug/pprof/profile", s.handlers.PprofProfile)
	router.HandleFunc("/debug/pprof/symbol", s.handlers.PprofSymbol)
	router.HandleFunc("/debug/pprof/trace", s.handlers.PprofTrace)
	router.Handle("/debug/pprof/goroutine", s.handlers.PprofHandler("goroutine"))
	router.Handle("/debug/pprof/heap", s.handlers.PprofHandler("heap"))
	router.Handle("/debug/pprof/threadcreate", s.handlers.PprofHandler("threadcreate"))
	router.Handle("/debug/pprof/block", s.handlers.PprofHandler("block"))
}

// GetRouter returns the HTTP router
func (s *Server) GetRouter() *mux.Router {
	return s.router
}

// GetConfig returns the server configuration
func (s *Server) GetConfig() *Config {
	return s.config
}

// getDefaultConfig returns the default server configuration
func getDefaultConfig() *Config {
	return &Config{
		Host:            constants.DefaultHost,
		Port:            constants.DefaultPort,
		MetricsPort:     constants.DefaultMetricsPort,
		ReadTimeout:     constants.DefaultReadTimeout,
		WriteTimeout:    constants.DefaultWriteTimeout,
		IdleTimeout:     constants.DefaultIdleTimeout,
		ShutdownTimeout: constants.DefaultShutdownTimeout,
		EnableMetrics:   true,
		EnableProfiling: false,
		EnableCORS:      true,
		MaxRequestSize:  constants.MaxUploadSize,
	}
}