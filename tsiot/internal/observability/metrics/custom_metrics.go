package metrics

import (
	"context"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// CustomMetrics provides application-specific metrics collection
type CustomMetrics struct {
	logger *logrus.Logger
	config *CustomMetricsConfig
	mu     sync.RWMutex

	// Time series metrics
	timeSeriesProcessed   int64
	timeSeriesGenerated   int64
	timeSeriesValidated   int64
	timeSeriesFailed      int64

	// Quality metrics
	qualityScores         map[string]float64
	validationResults     map[string]*ValidationMetrics
	generationResults     map[string]*GenerationMetrics

	// Performance metrics
	processingTimes       []time.Duration
	memoryUsage          []float64
	cpuUsage             []float64
	diskIO               []float64

	// Business metrics
	activeUsers          int64
	apiRequestCounts     map[string]int64
	errorCounts          map[string]int64
	featureUsage         map[string]int64

	// System health metrics
	componentHealth      map[string]HealthMetric
	dependencyStatus     map[string]DependencyStatus
	alertHistory         []AlertMetric

	// Custom counters and gauges
	customCounters       map[string]int64
	customGauges         map[string]float64
	customHistograms     map[string][]float64
}

// CustomMetricsConfig configures custom metrics collection
type CustomMetricsConfig struct {
	Enabled              bool          `json:"enabled"`
	CollectionInterval   time.Duration `json:"collection_interval"`
	RetentionPeriod      time.Duration `json:"retention_period"`
	MaxHistorySize       int           `json:"max_history_size"`
	EnableSystemMetrics  bool          `json:"enable_system_metrics"`
	EnableBusinessMetrics bool         `json:"enable_business_metrics"`
	CustomTags           map[string]string `json:"custom_tags"`
}

// ValidationMetrics tracks validation-specific metrics
type ValidationMetrics struct {
	TestsRun        int       `json:"tests_run"`
	TestsPassed     int       `json:"tests_passed"`
	TestsFailed     int       `json:"tests_failed"`
	QualityScore    float64   `json:"quality_score"`
	Duration        time.Duration `json:"duration"`
	Timestamp       time.Time `json:"timestamp"`
	ValidationID    string    `json:"validation_id"`
	DatasetSize     int       `json:"dataset_size"`
}

// GenerationMetrics tracks generation-specific metrics
type GenerationMetrics struct {
	DataPointsGenerated int       `json:"data_points_generated"`
	ModelAccuracy       float64   `json:"model_accuracy"`
	GenerationTime      time.Duration `json:"generation_time"`
	ModelType           string    `json:"model_type"`
	Timestamp           time.Time `json:"timestamp"`
	RequestID           string    `json:"request_id"`
	QualityScore        float64   `json:"quality_score"`
}

// HealthMetric represents component health status
type HealthMetric struct {
	Status        string            `json:"status"`
	LastCheck     time.Time         `json:"last_check"`
	ResponseTime  time.Duration     `json:"response_time"`
	ErrorCount    int64             `json:"error_count"`
	Details       map[string]string `json:"details"`
}

// DependencyStatus tracks external dependency status
type DependencyStatus struct {
	Name          string        `json:"name"`
	Status        string        `json:"status"`
	LastCheck     time.Time     `json:"last_check"`
	ResponseTime  time.Duration `json:"response_time"`
	ErrorMessage  string        `json:"error_message"`
	Version       string        `json:"version"`
}

// AlertMetric tracks alert history
type AlertMetric struct {
	AlertID     string            `json:"alert_id"`
	Severity    string            `json:"severity"`
	Component   string            `json:"component"`
	Message     string            `json:"message"`
	Timestamp   time.Time         `json:"timestamp"`
	Resolved    bool              `json:"resolved"`
	ResolvedAt  *time.Time        `json:"resolved_at,omitempty"`
	Tags        map[string]string `json:"tags"`
}

// NewCustomMetrics creates a new custom metrics instance
func NewCustomMetrics(config *CustomMetricsConfig, logger *logrus.Logger) *CustomMetrics {
	if config == nil {
		config = getDefaultCustomMetricsConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	return &CustomMetrics{
		logger:               logger,
		config:               config,
		qualityScores:        make(map[string]float64),
		validationResults:    make(map[string]*ValidationMetrics),
		generationResults:    make(map[string]*GenerationMetrics),
		processingTimes:      make([]time.Duration, 0),
		memoryUsage:          make([]float64, 0),
		cpuUsage:             make([]float64, 0),
		diskIO:               make([]float64, 0),
		apiRequestCounts:     make(map[string]int64),
		errorCounts:          make(map[string]int64),
		featureUsage:         make(map[string]int64),
		componentHealth:      make(map[string]HealthMetric),
		dependencyStatus:     make(map[string]DependencyStatus),
		alertHistory:         make([]AlertMetric, 0),
		customCounters:       make(map[string]int64),
		customGauges:         make(map[string]float64),
		customHistograms:     make(map[string][]float64),
	}
}

// Start starts custom metrics collection
func (cm *CustomMetrics) Start(ctx context.Context) error {
	if !cm.config.Enabled {
		cm.logger.Info("Custom metrics collection disabled")
		return nil
	}

	cm.logger.Info("Starting custom metrics collection")
	go cm.collectMetrics(ctx)
	return nil
}

// IncrementTimeSeriesProcessed increments processed time series counter
func (cm *CustomMetrics) IncrementTimeSeriesProcessed() {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.timeSeriesProcessed++
}

// RecordQualityScore records a quality score for a specific metric
func (cm *CustomMetrics) RecordQualityScore(metric string, score float64) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.qualityScores[metric] = score
}

// IncrementCounter increments a custom counter
func (cm *CustomMetrics) IncrementCounter(name string) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.customCounters[name]++
}

// collectMetrics runs background metric collection
func (cm *CustomMetrics) collectMetrics(ctx context.Context) {
	ticker := time.NewTicker(cm.config.CollectionInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			cm.logger.Debug("Collecting custom metrics")
		}
	}
}

func getDefaultCustomMetricsConfig() *CustomMetricsConfig {
	return &CustomMetricsConfig{
		Enabled:               true,
		CollectionInterval:    time.Minute,
		RetentionPeriod:       24 * time.Hour,
		MaxHistorySize:        1000,
		EnableSystemMetrics:   true,
		EnableBusinessMetrics: true,
		CustomTags:            make(map[string]string),
	}
}