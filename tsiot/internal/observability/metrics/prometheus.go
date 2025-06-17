package metrics

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/constants"
)

// PrometheusMetrics provides Prometheus-based metrics collection
type PrometheusMetrics struct {
	logger   *logrus.Logger
	registry *prometheus.Registry
	server   *http.Server
	config   *PrometheusConfig
	mu       sync.RWMutex

	// Core metrics
	httpRequestsTotal       *prometheus.CounterVec
	httpRequestDuration     *prometheus.HistogramVec
	generationRequestsTotal *prometheus.CounterVec
	generationDuration      *prometheus.HistogramVec
	generationActive        prometheus.Gauge
	validationRequestsTotal *prometheus.CounterVec
	validationDuration      *prometheus.HistogramVec
	storageOperationsTotal  *prometheus.CounterVec
	storageDuration         *prometheus.HistogramVec
	mcpConnectionsActive    prometheus.Gauge
	mcpMessagesTotal        *prometheus.CounterVec
	workerJobsTotal         *prometheus.CounterVec
	workerJobDuration       *prometheus.HistogramVec
	validationQualityScore  *prometheus.GaugeVec

	// System metrics
	systemMemoryUsage       prometheus.Gauge
	systemCPUUsage          prometheus.Gauge
	systemDiskUsage         *prometheus.GaugeVec
	systemNetworkBytesTotal *prometheus.CounterVec

	// Application metrics
	agentsActive            *prometheus.GaugeVec
	modelsLoaded            *prometheus.GaugeVec
	cacheHitRate            *prometheus.GaugeVec
	errorRate               *prometheus.CounterVec
	healthStatus            *prometheus.GaugeVec
}

// PrometheusConfig configures Prometheus metrics
type PrometheusConfig struct {
	Enabled     bool   `json:"enabled"`
	Port        int    `json:"port"`
	Path        string `json:"path"`
	Namespace   string `json:"namespace"`
	Subsystem   string `json:"subsystem"`
	EnablePush  bool   `json:"enable_push"`
	PushGateway string `json:"push_gateway"`
	PushInterval time.Duration `json:"push_interval"`
	Labels      map[string]string `json:"labels"`
}

// NewPrometheusMetrics creates a new Prometheus metrics instance
func NewPrometheusMetrics(config *PrometheusConfig, logger *logrus.Logger) (*PrometheusMetrics, error) {
	if config == nil {
		config = getDefaultPrometheusConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	registry := prometheus.NewRegistry()

	pm := &PrometheusMetrics{
		logger:   logger,
		registry: registry,
		config:   config,
	}

	// Initialize metrics
	if err := pm.initializeMetrics(); err != nil {
		return nil, fmt.Errorf("failed to initialize metrics: %w", err)
	}

	// Register metrics with registry
	if err := pm.registerMetrics(); err != nil {
		return nil, fmt.Errorf("failed to register metrics: %w", err)
	}

	return pm, nil
}

// Start starts the Prometheus metrics server
func (pm *PrometheusMetrics) Start(ctx context.Context) error {
	if !pm.config.Enabled {
		pm.logger.Info("Prometheus metrics disabled")
		return nil
	}

	mux := http.NewServeMux()
	mux.Handle(pm.config.Path, promhttp.HandlerFor(pm.registry, promhttp.HandlerOpts{
		EnableOpenMetrics: true,
	}))

	pm.server = &http.Server{
		Addr:    fmt.Sprintf(":%d", pm.config.Port),
		Handler: mux,
	}

	pm.logger.WithFields(logrus.Fields{
		"port": pm.config.Port,
		"path": pm.config.Path,
	}).Info("Starting Prometheus metrics server")

	go func() {
		if err := pm.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			pm.logger.WithError(err).Error("Prometheus metrics server error")
		}
	}()

	// Start background collectors
	go pm.collectSystemMetrics(ctx)

	return nil
}

// Stop stops the Prometheus metrics server
func (pm *PrometheusMetrics) Stop(ctx context.Context) error {
	if pm.server == nil {
		return nil
	}

	pm.logger.Info("Stopping Prometheus metrics server")
	return pm.server.Shutdown(ctx)
}

// HTTP Metrics
func (pm *PrometheusMetrics) RecordHTTPRequest(method, path, status string, duration time.Duration) {
	pm.httpRequestsTotal.WithLabelValues(method, path, status).Inc()
	pm.httpRequestDuration.WithLabelValues(method, path).Observe(duration.Seconds())
}

// Generation Metrics
func (pm *PrometheusMetrics) RecordGenerationRequest(generator, status string, duration time.Duration) {
	pm.generationRequestsTotal.WithLabelValues(generator, status).Inc()
	pm.generationDuration.WithLabelValues(generator).Observe(duration.Seconds())
}

func (pm *PrometheusMetrics) SetActiveGenerations(count float64) {
	pm.generationActive.Set(count)
}

// Validation Metrics
func (pm *PrometheusMetrics) RecordValidationRequest(validator, status string, duration time.Duration) {
	pm.validationRequestsTotal.WithLabelValues(validator, status).Inc()
	pm.validationDuration.WithLabelValues(validator).Observe(duration.Seconds())
}

func (pm *PrometheusMetrics) SetValidationQualityScore(validator string, score float64) {
	pm.validationQualityScore.WithLabelValues(validator).Set(score)
}

// Storage Metrics
func (pm *PrometheusMetrics) RecordStorageOperation(backend, operation, status string, duration time.Duration) {
	pm.storageOperationsTotal.WithLabelValues(backend, operation, status).Inc()
	pm.storageDuration.WithLabelValues(backend, operation).Observe(duration.Seconds())
}

// MCP Metrics
func (pm *PrometheusMetrics) SetMCPConnectionsActive(count float64) {
	pm.mcpConnectionsActive.Set(count)
}

func (pm *PrometheusMetrics) RecordMCPMessage(messageType, direction string) {
	pm.mcpMessagesTotal.WithLabelValues(messageType, direction).Inc()
}

// Worker Metrics
func (pm *PrometheusMetrics) RecordWorkerJob(jobType, status string, duration time.Duration) {
	pm.workerJobsTotal.WithLabelValues(jobType, status).Inc()
	pm.workerJobDuration.WithLabelValues(jobType).Observe(duration.Seconds())
}

// Agent Metrics
func (pm *PrometheusMetrics) SetAgentsActive(agentType string, count float64) {
	pm.agentsActive.WithLabelValues(agentType).Set(count)
}

// Model Metrics
func (pm *PrometheusMetrics) SetModelsLoaded(modelType string, count float64) {
	pm.modelsLoaded.WithLabelValues(modelType).Set(count)
}

// Cache Metrics
func (pm *PrometheusMetrics) SetCacheHitRate(cacheType string, rate float64) {
	pm.cacheHitRate.WithLabelValues(cacheType).Set(rate)
}

// Error Metrics
func (pm *PrometheusMetrics) RecordError(component, errorType string) {
	pm.errorRate.WithLabelValues(component, errorType).Inc()
}

// Health Metrics
func (pm *PrometheusMetrics) SetHealthStatus(component, status string, value float64) {
	pm.healthStatus.WithLabelValues(component, status).Set(value)
}

// initializeMetrics initializes all Prometheus metrics
func (pm *PrometheusMetrics) initializeMetrics() error {
	namespace := pm.config.Namespace
	subsystem := pm.config.Subsystem

	// HTTP metrics
	pm.httpRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "http_requests_total",
			Help:      "Total number of HTTP requests",
		},
		[]string{"method", "path", "status"},
	)

	pm.httpRequestDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "http_request_duration_seconds",
			Help:      "HTTP request duration in seconds",
			Buckets:   prometheus.DefBuckets,
		},
		[]string{"method", "path"},
	)

	// Generation metrics
	pm.generationRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "generation_requests_total",
			Help:      "Total number of generation requests",
		},
		[]string{"generator", "status"},
	)

	pm.generationDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "generation_duration_seconds",
			Help:      "Generation request duration in seconds",
			Buckets:   []float64{0.1, 0.5, 1, 5, 10, 30, 60, 120, 300},
		},
		[]string{"generator"},
	)

	pm.generationActive = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "generation_active",
			Help:      "Number of active generation requests",
		},
	)

	// Validation metrics
	pm.validationRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "validation_requests_total",
			Help:      "Total number of validation requests",
		},
		[]string{"validator", "status"},
	)

	pm.validationDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "validation_duration_seconds",
			Help:      "Validation request duration in seconds",
			Buckets:   []float64{0.1, 0.5, 1, 5, 10, 30, 60},
		},
		[]string{"validator"},
	)

	pm.validationQualityScore = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "validation_quality_score",
			Help:      "Validation quality score",
		},
		[]string{"validator"},
	)

	// Storage metrics
	pm.storageOperationsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "storage_operations_total",
			Help:      "Total number of storage operations",
		},
		[]string{"backend", "operation", "status"},
	)

	pm.storageDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "storage_operation_duration_seconds",
			Help:      "Storage operation duration in seconds",
			Buckets:   []float64{0.001, 0.01, 0.1, 0.5, 1, 5},
		},
		[]string{"backend", "operation"},
	)

	// MCP metrics
	pm.mcpConnectionsActive = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "mcp_connections_active",
			Help:      "Number of active MCP connections",
		},
	)

	pm.mcpMessagesTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "mcp_messages_total",
			Help:      "Total number of MCP messages",
		},
		[]string{"type", "direction"},
	)

	// Worker metrics
	pm.workerJobsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "worker_jobs_total",
			Help:      "Total number of worker jobs",
		},
		[]string{"type", "status"},
	)

	pm.workerJobDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "worker_job_duration_seconds",
			Help:      "Worker job duration in seconds",
			Buckets:   []float64{0.1, 1, 10, 60, 300, 1800, 3600},
		},
		[]string{"type"},
	)

	// System metrics
	pm.systemMemoryUsage = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "system",
			Name:      "memory_usage_bytes",
			Help:      "System memory usage in bytes",
		},
	)

	pm.systemCPUUsage = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "system",
			Name:      "cpu_usage_percent",
			Help:      "System CPU usage percentage",
		},
	)

	pm.systemDiskUsage = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "system",
			Name:      "disk_usage_bytes",
			Help:      "System disk usage in bytes",
		},
		[]string{"device", "type"},
	)

	pm.systemNetworkBytesTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "system",
			Name:      "network_bytes_total",
			Help:      "Total network bytes",
		},
		[]string{"device", "direction"},
	)

	// Application metrics
	pm.agentsActive = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "agents_active",
			Help:      "Number of active agents",
		},
		[]string{"type"},
	)

	pm.modelsLoaded = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "models_loaded",
			Help:      "Number of loaded models",
		},
		[]string{"type"},
	)

	pm.cacheHitRate = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "cache_hit_rate",
			Help:      "Cache hit rate",
		},
		[]string{"type"},
	)

	pm.errorRate = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "errors_total",
			Help:      "Total number of errors",
		},
		[]string{"component", "type"},
	)

	pm.healthStatus = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "health_status",
			Help:      "Health status of components",
		},
		[]string{"component", "status"},
	)

	return nil
}

// registerMetrics registers all metrics with the Prometheus registry
func (pm *PrometheusMetrics) registerMetrics() error {
	metrics := []prometheus.Collector{
		pm.httpRequestsTotal,
		pm.httpRequestDuration,
		pm.generationRequestsTotal,
		pm.generationDuration,
		pm.generationActive,
		pm.validationRequestsTotal,
		pm.validationDuration,
		pm.validationQualityScore,
		pm.storageOperationsTotal,
		pm.storageDuration,
		pm.mcpConnectionsActive,
		pm.mcpMessagesTotal,
		pm.workerJobsTotal,
		pm.workerJobDuration,
		pm.systemMemoryUsage,
		pm.systemCPUUsage,
		pm.systemDiskUsage,
		pm.systemNetworkBytesTotal,
		pm.agentsActive,
		pm.modelsLoaded,
		pm.cacheHitRate,
		pm.errorRate,
		pm.healthStatus,
	}

	for _, metric := range metrics {
		if err := pm.registry.Register(metric); err != nil {
			return fmt.Errorf("failed to register metric: %w", err)
		}
	}

	return nil
}

// collectSystemMetrics collects system-level metrics
func (pm *PrometheusMetrics) collectSystemMetrics(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			pm.updateSystemMetrics()
		}
	}
}

// updateSystemMetrics updates system metrics (simplified implementation)
func (pm *PrometheusMetrics) updateSystemMetrics() {
	// These would be replaced with actual system metric collection
	// For now, using placeholder values
	pm.systemMemoryUsage.Set(1024 * 1024 * 1024) // 1GB placeholder
	pm.systemCPUUsage.Set(15.5)                   // 15.5% placeholder
}

// GetRegistry returns the Prometheus registry
func (pm *PrometheusMetrics) GetRegistry() *prometheus.Registry {
	return pm.registry
}

// GetConfig returns the configuration
func (pm *PrometheusMetrics) GetConfig() *PrometheusConfig {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	return pm.config
}

func getDefaultPrometheusConfig() *PrometheusConfig {
	return &PrometheusConfig{
		Enabled:     true,
		Port:        constants.DefaultMetricsPort,
		Path:        "/metrics",
		Namespace:   "tsiot",
		Subsystem:   "server",
		EnablePush:  false,
		PushGateway: "",
		PushInterval: 30 * time.Second,
		Labels:      make(map[string]string),
	}
}