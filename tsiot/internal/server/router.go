package server

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-contrib/gzip"
	"github.com/gin-contrib/pprof"
	"github.com/gin-contrib/requestid"
	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/sirupsen/logrus"

	"github.com/your-org/tsiot/internal/middleware"
)

// Router wraps the HTTP router with configuration
type Router struct {
	engine   *gin.Engine
	handlers *Handlers
	config   *Config
	logger   *logrus.Logger
}

// NewRouter creates a new HTTP router
func NewRouter(handlers *Handlers, config *Config, logger *logrus.Logger) *Router {
	// Set Gin mode based on config
	if config.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	} else {
		gin.SetMode(gin.DebugMode)
	}

	engine := gin.New()

	return &Router{
		engine:   engine,
		handlers: handlers,
		config:   config,
		logger:   logger,
	}
}

// SetupRoutes configures all routes and middleware
func (r *Router) SetupRoutes() {
	// Global middleware
	r.setupGlobalMiddleware()

	// Health and monitoring routes (no auth required)
	r.setupHealthRoutes()

	// API routes with authentication
	r.setupAPIRoutes()

	// Development and debug routes
	if r.config.Environment != "production" {
		r.setupDebugRoutes()
	}
}

// setupGlobalMiddleware configures global middleware
func (r *Router) setupGlobalMiddleware() {
	// Recovery middleware
	r.engine.Use(gin.Recovery())

	// Request ID middleware
	r.engine.Use(requestid.New())

	// Custom logging middleware
	r.engine.Use(r.loggingMiddleware())

	// CORS middleware
	if r.config.CORS.Enabled {
		corsConfig := cors.Config{
			AllowOrigins:     r.config.CORS.AllowedOrigins,
			AllowMethods:     []string{"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"},
			AllowHeaders:     []string{"Origin", "Content-Length", "Content-Type", "Authorization", "X-API-Key", "X-Request-ID"},
			ExposeHeaders:    []string{"Content-Length", "X-Request-ID"},
			AllowCredentials: true,
			MaxAge:           12 * time.Hour,
		}
		r.engine.Use(cors.New(corsConfig))
	}

	// Compression middleware
	r.engine.Use(gzip.Gzip(gzip.DefaultCompression))

	// Rate limiting middleware
	if r.config.RateLimit.Enabled {
		r.engine.Use(middleware.RateLimit(r.config.RateLimit))
	}

	// Security headers middleware
	r.engine.Use(middleware.SecurityHeaders())

	// Request timeout middleware
	r.engine.Use(middleware.TimeoutHandler(r.config.API.RequestTimeout))
}

// setupHealthRoutes configures health and monitoring endpoints
func (r *Router) setupHealthRoutes() {
	// Basic health endpoints
	r.engine.GET("/health", r.handlers.HealthCheck)
	r.engine.GET("/ready", r.handlers.ReadinessCheck)
	r.engine.GET("/ping", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"message": "pong"})
	})

	// Metrics endpoint
	r.engine.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// Version info
	r.engine.GET("/version", r.handlers.Info)
}

// setupAPIRoutes configures main API routes
func (r *Router) setupAPIRoutes() {
	// API v1 group
	v1 := r.engine.Group("/api/v1")

	// Apply authentication middleware to API routes
	v1.Use(middleware.AuthenticationHandler(r.config.Auth))

	// Service info (authenticated)
	v1.GET("/info", r.handlers.Info)

	// Generator routes
	generators := v1.Group("/generators")
	{
		generators.GET("", r.handlers.ListGenerators)
	}

	// Generation routes
	generation := v1.Group("")
	{
		generation.POST("/generate", r.handlers.GenerateTimeSeries)
		generation.POST("/batch/generate", r.handlers.BatchGenerateTimeSeries)
	}

	// Validation routes
	validation := v1.Group("")
	{
		validation.POST("/validate", r.handlers.ValidateTimeSeries)
		validation.GET("/validators", r.handlers.ListValidators)
	}

	// Analytics routes
	analytics := v1.Group("/analytics")
	{
		analytics.POST("/analyze", r.handlers.AnalyzeTimeSeries)
		analytics.GET("/capabilities", r.handlers.GetAnalyticsCapabilities)
	}

	// Export routes
	export := v1.Group("")
	{
		export.POST("/export", r.handlers.ExportTimeSeries)
	}

	// Data management routes
	timeseries := v1.Group("/timeseries")
	{
		timeseries.GET("", r.handlers.ListTimeSeries)
		timeseries.GET("/:id", r.handlers.GetTimeSeries)
		timeseries.DELETE("/:id", r.handlers.DeleteTimeSeries)
	}

	// Admin routes (require admin privileges)
	admin := v1.Group("/admin")
	admin.Use(middleware.RequireRole("admin"))
	{
		admin.GET("/stats", r.systemStatsHandler)
		admin.POST("/maintenance", r.maintenanceHandler)
		admin.GET("/logs", r.logsHandler)
	}
}

// setupDebugRoutes configures debug and development routes
func (r *Router) setupDebugRoutes() {
	// pprof endpoints for profiling
	pprof.Register(r.engine)

	// Debug group
	debug := r.engine.Group("/debug")
	{
		debug.GET("/config", r.configHandler)
		debug.GET("/routes", r.routesHandler)
		debug.GET("/middleware", r.middlewareHandler)
	}
}

// GetEngine returns the underlying Gin engine
func (r *Router) GetEngine() *gin.Engine {
	return r.engine
}

// Start starts the HTTP server
func (r *Router) Start(ctx context.Context) error {
	server := &http.Server{
		Addr:           fmt.Sprintf("%s:%d", r.config.Server.Host, r.config.Server.Port),
		Handler:        r.engine,
		ReadTimeout:    r.config.Server.ReadTimeout,
		WriteTimeout:   r.config.Server.WriteTimeout,
		MaxHeaderBytes: r.config.Server.MaxHeaderBytes,
	}

	// Start server in a goroutine
	go func() {
		r.logger.WithFields(logrus.Fields{
			"addr": server.Addr,
		}).Info("Starting HTTP server")

		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			r.logger.WithError(err).Fatal("Failed to start HTTP server")
		}
	}()

	// Wait for context cancellation
	<-ctx.Done()

	// Graceful shutdown
	r.logger.Info("Shutting down HTTP server...")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	return server.Shutdown(shutdownCtx)
}

// Custom middleware implementations

func (r *Router) loggingMiddleware() gin.HandlerFunc {
	return gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		// Custom log format
		r.logger.WithFields(logrus.Fields{
			"method":     param.Method,
			"path":       param.Path,
			"status":     param.StatusCode,
			"latency":    param.Latency,
			"client_ip":  param.ClientIP,
			"user_agent": param.Request.UserAgent(),
			"request_id": param.Request.Header.Get("X-Request-ID"),
		}).Info("HTTP request")

		return ""
	})
}

// Admin handler implementations

func (r *Router) systemStatsHandler(c *gin.Context) {
	stats := map[string]interface{}{
		"timestamp":    time.Now().UTC(),
		"goroutines":   r.getGoroutineCount(),
		"memory_usage": r.getMemoryUsage(),
		"uptime":       time.Since(r.config.StartTime).String(),
		"version":      r.config.Version,
	}

	c.JSON(http.StatusOK, stats)
}

func (r *Router) maintenanceHandler(c *gin.Context) {
	var req struct {
		Action string `json:"action" binding:"required"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	switch req.Action {
	case "gc":
		r.performGC()
		c.JSON(http.StatusOK, gin.H{"message": "Garbage collection performed"})
	case "cache_clear":
		r.clearCache()
		c.JSON(http.StatusOK, gin.H{"message": "Cache cleared"})
	default:
		c.JSON(http.StatusBadRequest, gin.H{"error": "Unknown maintenance action"})
	}
}

func (r *Router) logsHandler(c *gin.Context) {
	level := c.DefaultQuery("level", "info")
	lines := c.DefaultQuery("lines", "100")

	// This is a placeholder - in a real implementation you'd read from log files
	logs := map[string]interface{}{
		"level":     level,
		"lines":     lines,
		"logs":      []string{"Log entry 1", "Log entry 2", "Log entry 3"},
		"timestamp": time.Now().UTC(),
	}

	c.JSON(http.StatusOK, logs)
}

// Debug handler implementations

func (r *Router) configHandler(c *gin.Context) {
	// Return sanitized config (remove sensitive data)
	config := map[string]interface{}{
		"server": map[string]interface{}{
			"host": r.config.Server.Host,
			"port": r.config.Server.Port,
		},
		"environment": r.config.Environment,
		"version":     r.config.Version,
		"build_time":  r.config.BuildTime,
		"generators": map[string]interface{}{
			"max_length": r.config.Generators.MaxLength,
		},
		"api": map[string]interface{}{
			"max_batch_size":  r.config.API.MaxBatchSize,
			"request_timeout": r.config.API.RequestTimeout.String(),
		},
	}

	c.JSON(http.StatusOK, config)
}

func (r *Router) routesHandler(c *gin.Context) {
	routes := r.engine.Routes()
	routeInfo := make([]map[string]string, len(routes))

	for i, route := range routes {
		routeInfo[i] = map[string]string{
			"method": route.Method,
			"path":   route.Path,
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"routes": routeInfo,
		"count":  len(routes),
	})
}

func (r *Router) middlewareHandler(c *gin.Context) {
	middleware := []string{
		"Recovery",
		"RequestID",
		"Logging",
		"CORS",
		"Compression",
		"RateLimit",
		"SecurityHeaders",
		"Timeout",
		"Authentication",
	}

	c.JSON(http.StatusOK, gin.H{
		"middleware": middleware,
		"count":      len(middleware),
	})
}

// Helper functions

func (r *Router) getGoroutineCount() int {
	// Placeholder - implement actual goroutine counting
	return 10
}

func (r *Router) getMemoryUsage() map[string]interface{} {
	// Placeholder - implement actual memory usage tracking
	return map[string]interface{}{
		"alloc":      "10MB",
		"total_alloc": "50MB",
		"sys":        "20MB",
	}
}

func (r *Router) performGC() {
	// Placeholder - implement garbage collection
	r.logger.Info("Performing garbage collection")
}

func (r *Router) clearCache() {
	// Placeholder - implement cache clearing
	r.logger.Info("Clearing cache")
}