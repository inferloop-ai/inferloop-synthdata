package agents

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/stat"

	"github.com/inferloop/tsiot/internal/agents/base"
	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/models"
)

// AnomalyAgent detects anomalies in time series data using multiple detection methods
type AnomalyAgent struct {
	*base.BaseAgent
	mu               sync.RWMutex
	detectors        map[string]AnomalyDetector
	monitors         map[string]*TimeSeriesMonitor
	alertHandlers    []AlertHandler
	config           *AnomalyConfig
	anomalyHistory   []AnomalyEvent
	modelStore       map[string]*DetectionModel
	activeJobs       map[string]*DetectionJob
	jobQueue         chan *DetectionJob
	workers          chan struct{}
	stopCh           chan struct{}
	wg               sync.WaitGroup
}

// AnomalyConfig contains configuration for anomaly detection
type AnomalyConfig struct {
	// Detection settings
	EnableStatisticalDetection bool                    `json:"enable_statistical_detection"`
	EnableMLDetection         bool                    `json:"enable_ml_detection"`
	EnableEnsembleDetection   bool                    `json:"enable_ensemble_detection"`
	
	// Thresholds
	StatisticalThreshold      float64                 `json:"statistical_threshold"`      // Standard deviations
	MLThreshold              float64                 `json:"ml_threshold"`              // ML model confidence
	EnsembleThreshold        float64                 `json:"ensemble_threshold"`        // Ensemble vote threshold
	
	// Window settings
	WindowSize               int                     `json:"window_size"`               // Rolling window size
	MinWindowSize            int                     `json:"min_window_size"`           // Minimum window for detection
	StepSize                 int                     `json:"step_size"`                 // Step size for sliding window
	
	// Model settings
	ModelUpdateInterval      time.Duration           `json:"model_update_interval"`     // How often to retrain models
	ModelRetentionPeriod     time.Duration           `json:"model_retention_period"`    // How long to keep old models
	TrainingDataSize         int                     `json:"training_data_size"`        // Size of training dataset
	
	// Alert settings
	AlertCooldown           time.Duration           `json:"alert_cooldown"`            // Min time between alerts
	MaxAlertsPerHour        int                     `json:"max_alerts_per_hour"`       // Rate limiting
	AlertSeverityThreshold  AnomalySeverity         `json:"alert_severity_threshold"`  // Min severity to alert
	
	// Performance settings
	MaxConcurrentJobs       int                     `json:"max_concurrent_jobs"`       // Max parallel detection jobs
	JobTimeout              time.Duration           `json:"job_timeout"`               // Detection job timeout
	WorkerPoolSize          int                     `json:"worker_pool_size"`          // Number of worker goroutines
	
	// Historical settings
	HistoryRetentionPeriod  time.Duration           `json:"history_retention_period"`  // How long to keep anomaly history
	MaxHistorySize          int                     `json:"max_history_size"`          // Max number of historical events
}

// AnomalyDetector interface for different detection algorithms
type AnomalyDetector interface {
	GetName() string
	GetDescription() string
	Train(ctx context.Context, data []float64) error
	Detect(ctx context.Context, data []float64) (*AnomalyResult, error)
	UpdateModel(ctx context.Context, data []float64) error
	GetThreshold() float64
	SetThreshold(threshold float64)
}

// TimeSeriesMonitor monitors a specific time series for anomalies
type TimeSeriesMonitor struct {
	SeriesID        string              `json:"series_id"`
	Config          *MonitorConfig      `json:"config"`
	DetectionModel  *DetectionModel     `json:"detection_model"`
	LastDetection   time.Time           `json:"last_detection"`
	AlertCount      int                 `json:"alert_count"`
	LastAlert       time.Time           `json:"last_alert"`
	Status          MonitorStatus       `json:"status"`
	CreatedAt       time.Time           `json:"created_at"`
	UpdatedAt       time.Time           `json:"updated_at"`
	mu              sync.RWMutex
}

// MonitorConfig contains configuration for monitoring a specific time series
type MonitorConfig struct {
	SeriesID          string           `json:"series_id"`
	DetectionMethods  []string         `json:"detection_methods"`  // List of detector names
	SamplingInterval  time.Duration    `json:"sampling_interval"`  // How often to check for anomalies
	EnableRealTime    bool             `json:"enable_real_time"`   // Real-time vs batch detection
	AlertEnabled      bool             `json:"alert_enabled"`      // Enable alerting
	AutoRetraining    bool             `json:"auto_retraining"`    // Automatically retrain models
	CustomThresholds  map[string]float64 `json:"custom_thresholds"` // Per-detector thresholds
}

// DetectionJob represents an anomaly detection task
type DetectionJob struct {
	ID           string              `json:"id"`
	SeriesID     string              `json:"series_id"`
	Data         []float64           `json:"data"`
	Timestamps   []time.Time         `json:"timestamps"`
	DetectorName string              `json:"detector_name"`
	Priority     int                 `json:"priority"`
	CreatedAt    time.Time           `json:"created_at"`
	StartedAt    *time.Time          `json:"started_at,omitempty"`
	CompletedAt  *time.Time          `json:"completed_at,omitempty"`
	Result       *AnomalyResult      `json:"result,omitempty"`
	Error        error               `json:"error,omitempty"`
	Context      context.Context     `json:"-"`
	CancelFunc   context.CancelFunc  `json:"-"`
}

// DetectionModel stores trained models and metadata
type DetectionModel struct {
	SeriesID        string                 `json:"series_id"`
	DetectorName    string                 `json:"detector_name"`
	ModelData       map[string]interface{} `json:"model_data"`
	TrainingData    []float64              `json:"training_data,omitempty"`
	TrainedAt       time.Time              `json:"trained_at"`
	LastUsed        time.Time              `json:"last_used"`
	AccuracyScore   float64                `json:"accuracy_score"`
	Threshold       float64                `json:"threshold"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// AnomalyResult contains the result of anomaly detection
type AnomalyResult struct {
	SeriesID        string                 `json:"series_id"`
	DetectorName    string                 `json:"detector_name"`
	IsAnomaly       bool                   `json:"is_anomaly"`
	AnomalyScore    float64                `json:"anomaly_score"`
	Threshold       float64                `json:"threshold"`
	Confidence      float64                `json:"confidence"`
	Severity        AnomalySeverity        `json:"severity"`
	AnomalyType     AnomalyType            `json:"anomaly_type"`
	AnomalyPoints   []AnomalyPoint         `json:"anomaly_points"`
	Context         *AnomalyContext        `json:"context"`
	Explanation     string                 `json:"explanation"`
	DetectedAt      time.Time              `json:"detected_at"`
	ProcessingTime  time.Duration          `json:"processing_time"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// AnomalyEvent represents a detected anomaly event
type AnomalyEvent struct {
	ID              string                 `json:"id"`
	SeriesID        string                 `json:"series_id"`
	DetectorName    string                 `json:"detector_name"`
	Severity        AnomalySeverity        `json:"severity"`
	AnomalyType     AnomalyType            `json:"anomaly_type"`
	AnomalyScore    float64                `json:"anomaly_score"`
	Description     string                 `json:"description"`
	AnomalyPoints   []AnomalyPoint         `json:"anomaly_points"`
	Context         *AnomalyContext        `json:"context"`
	AlertSent       bool                   `json:"alert_sent"`
	Acknowledged    bool                   `json:"acknowledged"`
	ResolvedAt      *time.Time             `json:"resolved_at,omitempty"`
	CreatedAt       time.Time              `json:"created_at"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// AnomalyPoint represents a specific anomalous data point
type AnomalyPoint struct {
	Index       int       `json:"index"`
	Timestamp   time.Time `json:"timestamp"`
	Value       float64   `json:"value"`
	Expected    float64   `json:"expected"`
	Deviation   float64   `json:"deviation"`
	Score       float64   `json:"score"`
}

// AnomalyContext provides additional context about the anomaly
type AnomalyContext struct {
	WindowSize      int                    `json:"window_size"`
	WindowMean      float64                `json:"window_mean"`
	WindowStdDev    float64                `json:"window_std_dev"`
	WindowMin       float64                `json:"window_min"`
	WindowMax       float64                `json:"window_max"`
	TrendDirection  string                 `json:"trend_direction"`
	Seasonality     map[string]float64     `json:"seasonality"`
	PreviousValues  []float64              `json:"previous_values"`
	ModelFeatures   map[string]interface{} `json:"model_features"`
}

// Enums and types

type AnomalySeverity int

const (
	SeverityInfo AnomalySeverity = iota
	SeverityLow
	SeverityMedium
	SeverityHigh
	SeverityCritical
)

func (s AnomalySeverity) String() string {
	switch s {
	case SeverityInfo:
		return "info"
	case SeverityLow:
		return "low"
	case SeverityMedium:
		return "medium"
	case SeverityHigh:
		return "high"
	case SeverityCritical:
		return "critical"
	default:
		return "unknown"
	}
}

type AnomalyType int

const (
	TypePointAnomaly AnomalyType = iota
	TypeContextualAnomaly
	TypeCollectiveAnomaly
	TypeTrendAnomaly
	TypeSeasonalAnomaly
	TypeLevelShift
	TypeVarianceChange
)

func (t AnomalyType) String() string {
	switch t {
	case TypePointAnomaly:
		return "point_anomaly"
	case TypeContextualAnomaly:
		return "contextual_anomaly"
	case TypeCollectiveAnomaly:
		return "collective_anomaly"
	case TypeTrendAnomaly:
		return "trend_anomaly"
	case TypeSeasonalAnomaly:
		return "seasonal_anomaly"
	case TypeLevelShift:
		return "level_shift"
	case TypeVarianceChange:
		return "variance_change"
	default:
		return "unknown"
	}
}

type MonitorStatus int

const (
	MonitorStatusActive MonitorStatus = iota
	MonitorStatusPaused
	MonitorStatusStopped
	MonitorStatusError
)

func (s MonitorStatus) String() string {
	switch s {
	case MonitorStatusActive:
		return "active"
	case MonitorStatusPaused:
		return "paused"
	case MonitorStatusStopped:
		return "stopped"
	case MonitorStatusError:
		return "error"
	default:
		return "unknown"
	}
}

// AlertHandler interface for handling anomaly alerts
type AlertHandler interface {
	HandleAlert(ctx context.Context, event *AnomalyEvent) error
	GetName() string
	IsEnabled() bool
}

// NewAnomalyAgent creates a new anomaly detection agent
func NewAnomalyAgent(config *AnomalyConfig, logger *logrus.Logger) (*AnomalyAgent, error) {
	if config == nil {
		config = getDefaultAnomalyConfig()
	}
	
	baseAgent, err := base.NewBaseAgent("anomaly-agent", logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create base agent: %w", err)
	}
	
	agent := &AnomalyAgent{
		BaseAgent:      baseAgent,
		detectors:      make(map[string]AnomalyDetector),
		monitors:       make(map[string]*TimeSeriesMonitor),
		alertHandlers:  make([]AlertHandler, 0),
		config:         config,
		anomalyHistory: make([]AnomalyEvent, 0),
		modelStore:     make(map[string]*DetectionModel),
		activeJobs:     make(map[string]*DetectionJob),
		jobQueue:       make(chan *DetectionJob, config.MaxConcurrentJobs*2),
		workers:        make(chan struct{}, config.WorkerPoolSize),
		stopCh:         make(chan struct{}),
	}
	
	// Register default detectors
	if err := agent.registerDefaultDetectors(); err != nil {
		return nil, fmt.Errorf("failed to register detectors: %w", err)
	}
	
	// Start worker pool
	agent.startWorkers()
	
	return agent, nil
}

// RegisterDetector registers a new anomaly detector
func (aa *AnomalyAgent) RegisterDetector(detector AnomalyDetector) error {
	aa.mu.Lock()
	defer aa.mu.Unlock()
	
	if detector == nil {
		return errors.NewValidationError("INVALID_DETECTOR", "Detector cannot be nil")
	}
	
	name := detector.GetName()
	if name == "" {
		return errors.NewValidationError("INVALID_DETECTOR_NAME", "Detector name cannot be empty")
	}
	
	aa.detectors[name] = detector
	
	aa.Logger.WithFields(logrus.Fields{
		"detector_name": name,
		"description":   detector.GetDescription(),
	}).Info("Anomaly detector registered")
	
	return nil
}

// CreateMonitor creates a new time series monitor
func (aa *AnomalyAgent) CreateMonitor(ctx context.Context, config *MonitorConfig) error {
	if config == nil {
		return errors.NewValidationError("INVALID_CONFIG", "Monitor configuration is required")
	}
	
	if config.SeriesID == "" {
		return errors.NewValidationError("INVALID_SERIES_ID", "Series ID is required")
	}
	
	aa.mu.Lock()
	defer aa.mu.Unlock()
	
	// Check if monitor already exists
	if _, exists := aa.monitors[config.SeriesID]; exists {
		return errors.NewValidationError("MONITOR_EXISTS", fmt.Sprintf("Monitor for series %s already exists", config.SeriesID))
	}
	
	// Validate detection methods
	for _, method := range config.DetectionMethods {
		if _, exists := aa.detectors[method]; !exists {
			return errors.NewValidationError("UNKNOWN_DETECTOR", fmt.Sprintf("Detector %s not found", method))
		}
	}
	
	monitor := &TimeSeriesMonitor{
		SeriesID:      config.SeriesID,
		Config:        config,
		Status:        MonitorStatusActive,
		CreatedAt:     time.Now(),
		UpdatedAt:     time.Now(),
	}
	
	aa.monitors[config.SeriesID] = monitor
	
	aa.Logger.WithFields(logrus.Fields{
		"series_id":        config.SeriesID,
		"detection_methods": config.DetectionMethods,
		"real_time":        config.EnableRealTime,
	}).Info("Time series monitor created")
	
	return nil
}

// DetectAnomalies detects anomalies in time series data
func (aa *AnomalyAgent) DetectAnomalies(ctx context.Context, seriesID string, data []float64, timestamps []time.Time) ([]*AnomalyResult, error) {
	if len(data) == 0 {
		return nil, errors.NewValidationError("EMPTY_DATA", "Data cannot be empty")
	}
	
	if len(data) != len(timestamps) {
		return nil, errors.NewValidationError("MISMATCHED_LENGTHS", "Data and timestamps must have same length")
	}
	
	aa.mu.RLock()
	monitor, exists := aa.monitors[seriesID]
	aa.mu.RUnlock()
	
	if !exists {
		return nil, errors.NewValidationError("MONITOR_NOT_FOUND", fmt.Sprintf("No monitor found for series %s", seriesID))
	}
	
	if monitor.Status != MonitorStatusActive {
		return nil, errors.NewValidationError("MONITOR_INACTIVE", fmt.Sprintf("Monitor for series %s is not active", seriesID))
	}
	
	var results []*AnomalyResult
	
	// Run detection with each configured detector
	for _, detectorName := range monitor.Config.DetectionMethods {
		detector, exists := aa.detectors[detectorName]
		if !exists {
			aa.Logger.WithField("detector", detectorName).Warn("Detector not found, skipping")
			continue
		}
		
		// Set custom threshold if configured
		if threshold, ok := monitor.Config.CustomThresholds[detectorName]; ok {
			detector.SetThreshold(threshold)
		}
		
		// Submit detection job
		jobID, err := aa.submitDetectionJob(ctx, seriesID, data, timestamps, detectorName)
		if err != nil {
			aa.Logger.WithError(err).WithField("detector", detectorName).Error("Failed to submit detection job")
			continue
		}
		
		// Wait for job completion (simplified - in production might be async)
		if result := aa.waitForJobCompletion(ctx, jobID); result != nil {
			results = append(results, result)
		}
	}
	
	// Process ensemble results if enabled
	if aa.config.EnableEnsembleDetection && len(results) > 1 {
		ensembleResult := aa.combineResults(ctx, seriesID, results)
		if ensembleResult != nil {
			results = append(results, ensembleResult)
		}
	}
	
	// Handle anomaly events
	for _, result := range results {
		if result.IsAnomaly {
			aa.handleAnomalyEvent(ctx, result)
		}
	}
	
	return results, nil
}

// GetAnomalyHistory returns historical anomaly events
func (aa *AnomalyAgent) GetAnomalyHistory(ctx context.Context, seriesID string, limit int) ([]AnomalyEvent, error) {
	aa.mu.RLock()
	defer aa.mu.RUnlock()
	
	var events []AnomalyEvent
	
	for _, event := range aa.anomalyHistory {
		if seriesID == "" || event.SeriesID == seriesID {
			events = append(events, event)
		}
	}
	
	// Sort by creation time (most recent first)
	sort.Slice(events, func(i, j int) bool {
		return events[i].CreatedAt.After(events[j].CreatedAt)
	})
	
	// Apply limit
	if limit > 0 && limit < len(events) {
		events = events[:limit]
	}
	
	return events, nil
}

// GetMonitorStatus returns the status of a monitor
func (aa *AnomalyAgent) GetMonitorStatus(seriesID string) (*TimeSeriesMonitor, error) {
	aa.mu.RLock()
	defer aa.mu.RUnlock()
	
	monitor, exists := aa.monitors[seriesID]
	if !exists {
		return nil, errors.NewValidationError("MONITOR_NOT_FOUND", fmt.Sprintf("No monitor found for series %s", seriesID))
	}
	
	// Return a copy to avoid race conditions
	monitorCopy := *monitor
	return &monitorCopy, nil
}

// UpdateMonitor updates monitor configuration
func (aa *AnomalyAgent) UpdateMonitor(ctx context.Context, seriesID string, config *MonitorConfig) error {
	aa.mu.Lock()
	defer aa.mu.Unlock()
	
	monitor, exists := aa.monitors[seriesID]
	if !exists {
		return errors.NewValidationError("MONITOR_NOT_FOUND", fmt.Sprintf("No monitor found for series %s", seriesID))
	}
	
	// Validate new configuration
	for _, method := range config.DetectionMethods {
		if _, exists := aa.detectors[method]; !exists {
			return errors.NewValidationError("UNKNOWN_DETECTOR", fmt.Sprintf("Detector %s not found", method))
		}
	}
	
	monitor.Config = config
	monitor.UpdatedAt = time.Now()
	
	aa.Logger.WithField("series_id", seriesID).Info("Monitor configuration updated")
	return nil
}

// DeleteMonitor deletes a monitor
func (aa *AnomalyAgent) DeleteMonitor(seriesID string) error {
	aa.mu.Lock()
	defer aa.mu.Unlock()
	
	if _, exists := aa.monitors[seriesID]; !exists {
		return errors.NewValidationError("MONITOR_NOT_FOUND", fmt.Sprintf("No monitor found for series %s", seriesID))
	}
	
	delete(aa.monitors, seriesID)
	
	// Clean up associated models
	for key := range aa.modelStore {
		if fmt.Sprintf("%s_", seriesID) == key[:len(seriesID)+1] {
			delete(aa.modelStore, key)
		}
	}
	
	aa.Logger.WithField("series_id", seriesID).Info("Monitor deleted")
	return nil
}

// AddAlertHandler adds an alert handler
func (aa *AnomalyAgent) AddAlertHandler(handler AlertHandler) {
	aa.mu.Lock()
	defer aa.mu.Unlock()
	
	aa.alertHandlers = append(aa.alertHandlers, handler)
	
	aa.Logger.WithField("handler", handler.GetName()).Info("Alert handler added")
}

// Training and model management methods

func (aa *AnomalyAgent) TrainDetector(ctx context.Context, seriesID, detectorName string, trainingData []float64) error {
	detector, exists := aa.detectors[detectorName]
	if !exists {
		return errors.NewValidationError("UNKNOWN_DETECTOR", fmt.Sprintf("Detector %s not found", detectorName))
	}
	
	if len(trainingData) < aa.config.MinWindowSize {
		return errors.NewValidationError("INSUFFICIENT_DATA", fmt.Sprintf("Need at least %d data points for training", aa.config.MinWindowSize))
	}
	
	// Train the detector
	if err := detector.Train(ctx, trainingData); err != nil {
		return errors.WrapError(err, errors.ErrorTypeProcessing, "TRAINING_FAILED", "Failed to train detector")
	}
	
	// Store the trained model
	modelKey := fmt.Sprintf("%s_%s", seriesID, detectorName)
	model := &DetectionModel{
		SeriesID:      seriesID,
		DetectorName:  detectorName,
		TrainingData:  trainingData,
		TrainedAt:     time.Now(),
		LastUsed:      time.Now(),
		Threshold:     detector.GetThreshold(),
		Metadata:      make(map[string]interface{}),
	}
	
	aa.mu.Lock()
	aa.modelStore[modelKey] = model
	aa.mu.Unlock()
	
	aa.Logger.WithFields(logrus.Fields{
		"series_id":      seriesID,
		"detector":       detectorName,
		"training_size":  len(trainingData),
	}).Info("Detector trained successfully")
	
	return nil
}

// Worker methods

func (aa *AnomalyAgent) startWorkers() {
	for i := 0; i < aa.config.WorkerPoolSize; i++ {
		aa.wg.Add(1)
		go aa.worker()
	}
}

func (aa *AnomalyAgent) worker() {
	defer aa.wg.Done()
	
	for {
		select {
		case job := <-aa.jobQueue:
			aa.workers <- struct{}{} // Acquire worker slot
			aa.processDetectionJob(job)
			<-aa.workers // Release worker slot
		case <-aa.stopCh:
			return
		}
	}
}

func (aa *AnomalyAgent) submitDetectionJob(ctx context.Context, seriesID string, data []float64, timestamps []time.Time, detectorName string) (string, error) {
	jobCtx, cancelFunc := context.WithTimeout(ctx, aa.config.JobTimeout)
	
	job := &DetectionJob{
		ID:           aa.generateJobID(),
		SeriesID:     seriesID,
		Data:         data,
		Timestamps:   timestamps,
		DetectorName: detectorName,
		Priority:     1,
		CreatedAt:    time.Now(),
		Context:      jobCtx,
		CancelFunc:   cancelFunc,
	}
	
	aa.mu.Lock()
	aa.activeJobs[job.ID] = job
	aa.mu.Unlock()
	
	select {
	case aa.jobQueue <- job:
		return job.ID, nil
	default:
		aa.mu.Lock()
		delete(aa.activeJobs, job.ID)
		aa.mu.Unlock()
		cancelFunc()
		return "", errors.NewProcessingError("QUEUE_FULL", "Detection job queue is full")
	}
}

func (aa *AnomalyAgent) processDetectionJob(job *DetectionJob) {
	start := time.Now()
	job.StartedAt = &start
	
	detector, exists := aa.detectors[job.DetectorName]
	if !exists {
		job.Error = fmt.Errorf("detector %s not found", job.DetectorName)
		aa.completeJob(job)
		return
	}
	
	// Perform detection
	result, err := detector.Detect(job.Context, job.Data)
	if err != nil {
		job.Error = err
		aa.completeJob(job)
		return
	}
	
	// Enhance result with additional information
	result.SeriesID = job.SeriesID
	result.DetectorName = job.DetectorName
	result.DetectedAt = time.Now()
	result.ProcessingTime = time.Since(start)
	
	// Add anomaly points if anomaly detected
	if result.IsAnomaly {
		result.AnomalyPoints = aa.identifyAnomalyPoints(job.Data, job.Timestamps, result)
		result.Context = aa.buildAnomalyContext(job.Data)
	}
	
	job.Result = result
	aa.completeJob(job)
}

func (aa *AnomalyAgent) completeJob(job *DetectionJob) {
	completed := time.Now()
	job.CompletedAt = &completed
	
	if job.CancelFunc != nil {
		job.CancelFunc()
	}
	
	aa.Logger.WithFields(logrus.Fields{
		"job_id":      job.ID,
		"series_id":   job.SeriesID,
		"detector":    job.DetectorName,
		"duration":    completed.Sub(job.CreatedAt),
		"error":       job.Error != nil,
		"anomaly":     job.Result != nil && job.Result.IsAnomaly,
	}).Debug("Detection job completed")
}

func (aa *AnomalyAgent) waitForJobCompletion(ctx context.Context, jobID string) *AnomalyResult {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			aa.mu.RLock()
			job, exists := aa.activeJobs[jobID]
			aa.mu.RUnlock()
			
			if !exists {
				return nil
			}
			
			if job.CompletedAt != nil {
				aa.mu.Lock()
				delete(aa.activeJobs, jobID)
				aa.mu.Unlock()
				return job.Result
			}
		}
	}
}

// Helper methods

func (aa *AnomalyAgent) registerDefaultDetectors() error {
	// Register Z-Score detector
	zscore := NewZScoreDetector(aa.config.StatisticalThreshold)
	if err := aa.RegisterDetector(zscore); err != nil {
		return err
	}
	
	// Register IQR detector
	iqr := NewIQRDetector(1.5)
	if err := aa.RegisterDetector(iqr); err != nil {
		return err
	}
	
	// Register moving average detector
	movingAvg := NewMovingAverageDetector(aa.config.WindowSize, aa.config.StatisticalThreshold)
	if err := aa.RegisterDetector(movingAvg); err != nil {
		return err
	}
	
	return nil
}

func (aa *AnomalyAgent) combineResults(ctx context.Context, seriesID string, results []*AnomalyResult) *AnomalyResult {
	if len(results) == 0 {
		return nil
	}
	
	// Simple ensemble voting
	anomalyVotes := 0
	totalScore := 0.0
	maxSeverity := SeverityInfo
	
	for _, result := range results {
		if result.IsAnomaly {
			anomalyVotes++
		}
		totalScore += result.AnomalyScore
		
		if result.Severity > maxSeverity {
			maxSeverity = result.Severity
		}
	}
	
	avgScore := totalScore / float64(len(results))
	voteRatio := float64(anomalyVotes) / float64(len(results))
	isAnomaly := voteRatio >= aa.config.EnsembleThreshold
	
	return &AnomalyResult{
		SeriesID:       seriesID,
		DetectorName:   "ensemble",
		IsAnomaly:      isAnomaly,
		AnomalyScore:   avgScore,
		Threshold:      aa.config.EnsembleThreshold,
		Confidence:     voteRatio,
		Severity:       maxSeverity,
		AnomalyType:    TypeCollectiveAnomaly,
		Explanation:    fmt.Sprintf("Ensemble result: %d/%d detectors voted anomaly", anomalyVotes, len(results)),
		DetectedAt:     time.Now(),
		Metadata: map[string]interface{}{
			"vote_ratio":     voteRatio,
			"detector_count": len(results),
			"avg_score":      avgScore,
		},
	}
}

func (aa *AnomalyAgent) handleAnomalyEvent(ctx context.Context, result *AnomalyResult) {
	event := &AnomalyEvent{
		ID:            aa.generateEventID(),
		SeriesID:      result.SeriesID,
		DetectorName:  result.DetectorName,
		Severity:      result.Severity,
		AnomalyType:   result.AnomalyType,
		AnomalyScore:  result.AnomalyScore,
		Description:   result.Explanation,
		AnomalyPoints: result.AnomalyPoints,
		Context:       result.Context,
		CreatedAt:     time.Now(),
		Metadata:      result.Metadata,
	}
	
	// Add to history
	aa.mu.Lock()
	aa.anomalyHistory = append(aa.anomalyHistory, *event)
	
	// Trim history if needed
	if len(aa.anomalyHistory) > aa.config.MaxHistorySize {
		aa.anomalyHistory = aa.anomalyHistory[len(aa.anomalyHistory)-aa.config.MaxHistorySize:]
	}
	aa.mu.Unlock()
	
	// Send alerts if configured
	if result.Severity >= aa.config.AlertSeverityThreshold {
		for _, handler := range aa.alertHandlers {
			if handler.IsEnabled() {
				if err := handler.HandleAlert(ctx, event); err != nil {
					aa.Logger.WithError(err).WithField("handler", handler.GetName()).Error("Failed to send alert")
				} else {
					event.AlertSent = true
				}
			}
		}
	}
	
	aa.Logger.WithFields(logrus.Fields{
		"event_id":     event.ID,
		"series_id":    event.SeriesID,
		"detector":     event.DetectorName,
		"severity":     event.Severity.String(),
		"anomaly_type": event.AnomalyType.String(),
		"score":        event.AnomalyScore,
	}).Info("Anomaly event generated")
}

func (aa *AnomalyAgent) identifyAnomalyPoints(data []float64, timestamps []time.Time, result *AnomalyResult) []AnomalyPoint {
	var points []AnomalyPoint
	
	// Simple implementation - in practice this would be detector-specific
	threshold := result.Threshold
	mean := stat.Mean(data, nil)
	stddev := math.Sqrt(stat.Variance(data, nil))
	
	for i, value := range data {
		deviation := math.Abs(value - mean)
		if deviation > threshold*stddev {
			point := AnomalyPoint{
				Index:     i,
				Value:     value,
				Expected:  mean,
				Deviation: deviation,
				Score:     deviation / stddev,
			}
			
			if i < len(timestamps) {
				point.Timestamp = timestamps[i]
			}
			
			points = append(points, point)
		}
	}
	
	return points
}

func (aa *AnomalyAgent) buildAnomalyContext(data []float64) *AnomalyContext {
	if len(data) == 0 {
		return nil
	}
	
	mean := stat.Mean(data, nil)
	variance := stat.Variance(data, nil)
	stddev := math.Sqrt(variance)
	
	min := data[0]
	max := data[0]
	for _, v := range data[1:] {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	
	// Determine trend direction (simplified)
	trendDirection := "stable"
	if len(data) >= 2 {
		if data[len(data)-1] > data[0] {
			trendDirection = "increasing"
		} else if data[len(data)-1] < data[0] {
			trendDirection = "decreasing"
		}
	}
	
	return &AnomalyContext{
		WindowSize:     len(data),
		WindowMean:     mean,
		WindowStdDev:   stddev,
		WindowMin:      min,
		WindowMax:      max,
		TrendDirection: trendDirection,
		Seasonality:    make(map[string]float64),
		PreviousValues: data,
		ModelFeatures:  make(map[string]interface{}),
	}
}

func (aa *AnomalyAgent) generateJobID() string {
	return fmt.Sprintf("job_%d", time.Now().UnixNano())
}

func (aa *AnomalyAgent) generateEventID() string {
	return fmt.Sprintf("event_%d", time.Now().UnixNano())
}

// Shutdown stops the anomaly agent
func (aa *AnomalyAgent) Shutdown(ctx context.Context) error {
	aa.Logger.Info("Shutting down anomaly agent")
	
	close(aa.stopCh)
	
	// Cancel all active jobs
	aa.mu.Lock()
	for _, job := range aa.activeJobs {
		if job.CancelFunc != nil {
			job.CancelFunc()
		}
	}
	aa.mu.Unlock()
	
	// Wait for workers to finish
	done := make(chan struct{})
	go func() {
		aa.wg.Wait()
		close(done)
	}()
	
	select {
	case <-done:
		aa.Logger.Info("All anomaly detection workers stopped")
	case <-ctx.Done():
		aa.Logger.Warn("Shutdown timeout, some workers may still be running")
	}
	
	return nil
}

func getDefaultAnomalyConfig() *AnomalyConfig {
	return &AnomalyConfig{
		EnableStatisticalDetection: true,
		EnableMLDetection:         false,
		EnableEnsembleDetection:   true,
		StatisticalThreshold:      3.0,
		MLThreshold:              0.8,
		EnsembleThreshold:        0.6,
		WindowSize:               50,
		MinWindowSize:            10,
		StepSize:                 1,
		ModelUpdateInterval:      24 * time.Hour,
		ModelRetentionPeriod:     7 * 24 * time.Hour,
		TrainingDataSize:         1000,
		AlertCooldown:           5 * time.Minute,
		MaxAlertsPerHour:        10,
		AlertSeverityThreshold:  SeverityMedium,
		MaxConcurrentJobs:       10,
		JobTimeout:              30 * time.Second,
		WorkerPoolSize:          4,
		HistoryRetentionPeriod:  30 * 24 * time.Hour,
		MaxHistorySize:          10000,
	}
}