package batch

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/your-org/tsiot/internal/storage"
)

// Processor handles batch processing of time series data
type Processor struct {
	logger      *logrus.Logger
	config      *Config
	workers     []*Worker
	jobQueue    chan *Job
	resultQueue chan *Result
	mu          sync.RWMutex
	running     bool
	wg          sync.WaitGroup
}

// Config contains batch processor configuration
type Config struct {
	MaxWorkers      int           `json:"max_workers" yaml:"max_workers"`
	QueueSize       int           `json:"queue_size" yaml:"queue_size"`
	BatchSize       int           `json:"batch_size" yaml:"batch_size"`
	ProcessTimeout  time.Duration `json:"process_timeout" yaml:"process_timeout"`
	RetryAttempts   int           `json:"retry_attempts" yaml:"retry_attempts"`
	RetryDelay      time.Duration `json:"retry_delay" yaml:"retry_delay"`
	EnableMetrics   bool          `json:"enable_metrics" yaml:"enable_metrics"`
	ChunkSize       int           `json:"chunk_size" yaml:"chunk_size"`
	CompressionType string        `json:"compression_type" yaml:"compression_type"`
}

// Job represents a batch processing job
type Job struct {
	ID          string                 `json:"id"`
	Type        JobType                `json:"type"`
	TimeSeries  []*storage.TimeSeries  `json:"time_series"`
	Parameters  map[string]interface{} `json:"parameters"`
	Priority    Priority               `json:"priority"`
	CreatedAt   time.Time              `json:"created_at"`
	StartedAt   *time.Time             `json:"started_at,omitempty"`
	CompletedAt *time.Time             `json:"completed_at,omitempty"`
	Attempts    int                    `json:"attempts"`
	MaxAttempts int                    `json:"max_attempts"`
	Status      JobStatus              `json:"status"`
	Error       string                 `json:"error,omitempty"`
	Result      interface{}            `json:"result,omitempty"`
}

// JobType represents the type of batch job
type JobType string

const (
	JobTypeAggregation    JobType = "aggregation"
	JobTypeTransformation JobType = "transformation"
	JobTypeValidation     JobType = "validation"
	JobTypeAnalytics      JobType = "analytics"
	JobTypeExport         JobType = "export"
	JobTypeCleanup        JobType = "cleanup"
)

// Priority represents job priority
type Priority int

const (
	PriorityLow Priority = iota
	PriorityNormal
	PriorityHigh
	PriorityCritical
)

// JobStatus represents job status
type JobStatus string

const (
	StatusPending   JobStatus = "pending"
	StatusRunning   JobStatus = "running"
	StatusCompleted JobStatus = "completed"
	StatusFailed    JobStatus = "failed"
	StatusCancelled JobStatus = "cancelled"
)

// Result represents the result of a batch processing job
type Result struct {
	JobID       string                 `json:"job_id"`
	Status      JobStatus              `json:"status"`
	Data        interface{}            `json:"data,omitempty"`
	Error       string                 `json:"error,omitempty"`
	Metrics     *ProcessingMetrics     `json:"metrics,omitempty"`
	CompletedAt time.Time              `json:"completed_at"`
}

// ProcessingMetrics contains metrics about batch processing
type ProcessingMetrics struct {
	ProcessingTime   time.Duration `json:"processing_time"`
	RecordsProcessed int           `json:"records_processed"`
	RecordsSkipped   int           `json:"records_skipped"`
	RecordsFailed    int           `json:"records_failed"`
	MemoryUsed       int64         `json:"memory_used"`
	CPUTime          time.Duration `json:"cpu_time"`
}

// NewProcessor creates a new batch processor
func NewProcessor(logger *logrus.Logger, config *Config) *Processor {
	if config == nil {
		config = DefaultConfig()
	}

	return &Processor{
		logger:      logger,
		config:      config,
		jobQueue:    make(chan *Job, config.QueueSize),
		resultQueue: make(chan *Result, config.QueueSize),
		workers:     make([]*Worker, 0, config.MaxWorkers),
	}
}

// DefaultConfig returns default batch processor configuration
func DefaultConfig() *Config {
	return &Config{
		MaxWorkers:      4,
		QueueSize:       1000,
		BatchSize:       10000,
		ProcessTimeout:  30 * time.Minute,
		RetryAttempts:   3,
		RetryDelay:      5 * time.Second,
		EnableMetrics:   true,
		ChunkSize:       1000,
		CompressionType: "gzip",
	}
}

// Start starts the batch processor
func (p *Processor) Start(ctx context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.running {
		return fmt.Errorf("batch processor is already running")
	}

	p.logger.Info("Starting batch processor")

	// Start workers
	for i := 0; i < p.config.MaxWorkers; i++ {
		worker := NewWorker(i, p.logger, p.config)
		p.workers = append(p.workers, worker)

		p.wg.Add(1)
		go func(w *Worker) {
			defer p.wg.Done()
			w.Start(ctx, p.jobQueue, p.resultQueue)
		}(worker)
	}

	p.running = true
	p.logger.WithField("workers", p.config.MaxWorkers).Info("Batch processor started")

	return nil
}

// Stop stops the batch processor
func (p *Processor) Stop() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.running {
		return fmt.Errorf("batch processor is not running")
	}

	p.logger.Info("Stopping batch processor")

	// Close job queue to signal workers to stop
	close(p.jobQueue)

	// Wait for all workers to finish
	p.wg.Wait()

	// Close result queue
	close(p.resultQueue)

	p.running = false
	p.logger.Info("Batch processor stopped")

	return nil
}

// SubmitJob submits a job for batch processing
func (p *Processor) SubmitJob(job *Job) error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	if !p.running {
		return fmt.Errorf("batch processor is not running")
	}

	if job.ID == "" {
		job.ID = generateJobID()
	}

	job.CreatedAt = time.Now()
	job.Status = StatusPending
	job.MaxAttempts = p.config.RetryAttempts

	select {
	case p.jobQueue <- job:
		p.logger.WithFields(logrus.Fields{
			"job_id":   job.ID,
			"job_type": job.Type,
			"priority": job.Priority,
		}).Info("Job submitted for batch processing")
		return nil
	default:
		return fmt.Errorf("job queue is full")
	}
}

// GetResults returns a channel for receiving processing results
func (p *Processor) GetResults() <-chan *Result {
	return p.resultQueue
}

// ProcessAggregation processes time series aggregation
func (p *Processor) ProcessAggregation(ctx context.Context, timeSeries []*storage.TimeSeries, parameters map[string]interface{}) (*Result, error) {
	startTime := time.Now()
	
	// Extract aggregation parameters
	aggType, _ := parameters["type"].(string)
	if aggType == "" {
		aggType = "mean"
	}
	
	interval, _ := parameters["interval"].(string)
	if interval == "" {
		interval = "1h"
	}

	// Parse interval
	duration, err := time.ParseDuration(interval)
	if err != nil {
		return nil, fmt.Errorf("invalid interval: %v", err)
	}

	aggregatedSeries := make([]*storage.TimeSeries, 0)
	recordsProcessed := 0

	for _, ts := range timeSeries {
		aggregated, err := p.aggregateTimeSeries(ts, aggType, duration)
		if err != nil {
			p.logger.WithError(err).WithField("series_id", ts.SeriesID).Error("Failed to aggregate time series")
			continue
		}

		aggregatedSeries = append(aggregatedSeries, aggregated)
		recordsProcessed += len(ts.Values)
	}

	metrics := &ProcessingMetrics{
		ProcessingTime:   time.Since(startTime),
		RecordsProcessed: recordsProcessed,
		RecordsSkipped:   0,
		RecordsFailed:    len(timeSeries) - len(aggregatedSeries),
	}

	return &Result{
		Status:      StatusCompleted,
		Data:        aggregatedSeries,
		Metrics:     metrics,
		CompletedAt: time.Now(),
	}, nil
}

// ProcessTransformation processes time series transformation
func (p *Processor) ProcessTransformation(ctx context.Context, timeSeries []*storage.TimeSeries, parameters map[string]interface{}) (*Result, error) {
	startTime := time.Now()
	
	transformType, _ := parameters["type"].(string)
	if transformType == "" {
		return nil, fmt.Errorf("transformation type is required")
	}

	transformedSeries := make([]*storage.TimeSeries, 0)
	recordsProcessed := 0

	for _, ts := range timeSeries {
		transformed, err := p.transformTimeSeries(ts, transformType, parameters)
		if err != nil {
			p.logger.WithError(err).WithField("series_id", ts.SeriesID).Error("Failed to transform time series")
			continue
		}

		transformedSeries = append(transformedSeries, transformed)
		recordsProcessed += len(ts.Values)
	}

	metrics := &ProcessingMetrics{
		ProcessingTime:   time.Since(startTime),
		RecordsProcessed: recordsProcessed,
		RecordsSkipped:   0,
		RecordsFailed:    len(timeSeries) - len(transformedSeries),
	}

	return &Result{
		Status:      StatusCompleted,
		Data:        transformedSeries,
		Metrics:     metrics,
		CompletedAt: time.Now(),
	}, nil
}

// ProcessValidation processes batch validation
func (p *Processor) ProcessValidation(ctx context.Context, timeSeries []*storage.TimeSeries, parameters map[string]interface{}) (*Result, error) {
	startTime := time.Now()
	
	validationResults := make(map[string]interface{})
	recordsProcessed := 0

	for _, ts := range timeSeries {
		results, err := p.validateTimeSeries(ts, parameters)
		if err != nil {
			p.logger.WithError(err).WithField("series_id", ts.SeriesID).Error("Failed to validate time series")
			continue
		}

		validationResults[ts.SeriesID] = results
		recordsProcessed += len(ts.Values)
	}

	metrics := &ProcessingMetrics{
		ProcessingTime:   time.Since(startTime),
		RecordsProcessed: recordsProcessed,
		RecordsSkipped:   0,
		RecordsFailed:    0,
	}

	return &Result{
		Status:      StatusCompleted,
		Data:        validationResults,
		Metrics:     metrics,
		CompletedAt: time.Now(),
	}, nil
}

// Helper methods

func (p *Processor) aggregateTimeSeries(ts *storage.TimeSeries, aggType string, interval time.Duration) (*storage.TimeSeries, error) {
	if len(ts.Values) == 0 {
		return ts, nil
	}

	aggregated := &storage.TimeSeries{
		SeriesID:   ts.SeriesID + "_agg_" + aggType,
		Name:       ts.Name + " (Aggregated)",
		Generator:  "batch_aggregator",
		Metadata:   make(map[string]interface{}),
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}

	// Copy metadata
	for k, v := range ts.Metadata {
		aggregated.Metadata[k] = v
	}
	aggregated.Metadata["aggregation_type"] = aggType
	aggregated.Metadata["aggregation_interval"] = interval.String()
	aggregated.Metadata["original_series_id"] = ts.SeriesID

	// Group data by intervals
	groups := make(map[int64][]float64)
	timestamps := make(map[int64]time.Time)

	for i, timestamp := range ts.Timestamps {
		if i >= len(ts.Values) {
			break
		}

		// Calculate interval bucket
		bucket := timestamp.Unix() / int64(interval.Seconds())
		
		if _, exists := groups[bucket]; !exists {
			groups[bucket] = make([]float64, 0)
			timestamps[bucket] = timestamp.Truncate(interval)
		}
		
		groups[bucket] = append(groups[bucket], ts.Values[i])
	}

	// Apply aggregation function
	for bucket, values := range groups {
		timestamp := timestamps[bucket]
		aggregatedValue := p.applyAggregation(values, aggType)
		
		aggregated.Timestamps = append(aggregated.Timestamps, timestamp)
		aggregated.Values = append(aggregated.Values, aggregatedValue)
	}

	return aggregated, nil
}

func (p *Processor) transformTimeSeries(ts *storage.TimeSeries, transformType string, parameters map[string]interface{}) (*storage.TimeSeries, error) {
	transformed := &storage.TimeSeries{
		SeriesID:   ts.SeriesID + "_transform_" + transformType,
		Name:       ts.Name + " (Transformed)",
		Generator:  "batch_transformer",
		Timestamps: make([]time.Time, len(ts.Timestamps)),
		Values:     make([]float64, len(ts.Values)),
		Metadata:   make(map[string]interface{}),
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}

	// Copy basic data
	copy(transformed.Timestamps, ts.Timestamps)
	copy(transformed.Values, ts.Values)

	// Copy and update metadata
	for k, v := range ts.Metadata {
		transformed.Metadata[k] = v
	}
	transformed.Metadata["transformation_type"] = transformType
	transformed.Metadata["original_series_id"] = ts.SeriesID

	// Apply transformation
	switch transformType {
	case "normalize":
		return p.normalizeTimeSeries(transformed), nil
	case "standardize":
		return p.standardizeTimeSeries(transformed), nil
	case "log":
		return p.logTransformTimeSeries(transformed), nil
	case "diff":
		return p.differenceTimeSeries(transformed), nil
	default:
		return nil, fmt.Errorf("unknown transformation type: %s", transformType)
	}
}

func (p *Processor) validateTimeSeries(ts *storage.TimeSeries, parameters map[string]interface{}) (map[string]interface{}, error) {
	results := make(map[string]interface{})

	// Basic validation checks
	results["has_data"] = len(ts.Values) > 0
	results["timestamps_count"] = len(ts.Timestamps)
	results["values_count"] = len(ts.Values)
	results["timestamps_values_match"] = len(ts.Timestamps) == len(ts.Values)

	if len(ts.Values) > 0 {
		// Statistical validation
		results["has_null_values"] = p.hasNullValues(ts.Values)
		results["has_infinite_values"] = p.hasInfiniteValues(ts.Values)
		results["min_value"] = p.minValue(ts.Values)
		results["max_value"] = p.maxValue(ts.Values)
		results["mean_value"] = p.meanValue(ts.Values)
	}

	if len(ts.Timestamps) > 1 {
		// Temporal validation
		results["is_chronologically_ordered"] = p.isChronologicallyOrdered(ts.Timestamps)
		results["has_duplicate_timestamps"] = p.hasDuplicateTimestamps(ts.Timestamps)
		results["average_interval"] = p.averageInterval(ts.Timestamps).String()
	}

	return results, nil
}

func (p *Processor) applyAggregation(values []float64, aggType string) float64 {
	if len(values) == 0 {
		return 0
	}

	switch aggType {
	case "mean", "avg":
		sum := 0.0
		for _, v := range values {
			sum += v
		}
		return sum / float64(len(values))
	case "sum":
		sum := 0.0
		for _, v := range values {
			sum += v
		}
		return sum
	case "min":
		min := values[0]
		for _, v := range values[1:] {
			if v < min {
				min = v
			}
		}
		return min
	case "max":
		max := values[0]
		for _, v := range values[1:] {
			if v > max {
				max = v
			}
		}
		return max
	case "count":
		return float64(len(values))
	default:
		return p.meanValue(values)
	}
}

func (p *Processor) normalizeTimeSeries(ts *storage.TimeSeries) *storage.TimeSeries {
	if len(ts.Values) == 0 {
		return ts
	}

	min := p.minValue(ts.Values)
	max := p.maxValue(ts.Values)
	
	if max == min {
		return ts
	}

	for i := range ts.Values {
		ts.Values[i] = (ts.Values[i] - min) / (max - min)
	}

	return ts
}

func (p *Processor) standardizeTimeSeries(ts *storage.TimeSeries) *storage.TimeSeries {
	if len(ts.Values) == 0 {
		return ts
	}

	mean := p.meanValue(ts.Values)
	stddev := p.standardDeviation(ts.Values, mean)
	
	if stddev == 0 {
		return ts
	}

	for i := range ts.Values {
		ts.Values[i] = (ts.Values[i] - mean) / stddev
	}

	return ts
}

func (p *Processor) logTransformTimeSeries(ts *storage.TimeSeries) *storage.TimeSeries {
	if len(ts.Values) == 0 {
		return ts
	}

	for i := range ts.Values {
		if ts.Values[i] > 0 {
			ts.Values[i] = math.Log(ts.Values[i])
		}
	}

	return ts
}

func (p *Processor) differenceTimeSeries(ts *storage.TimeSeries) *storage.TimeSeries {
	if len(ts.Values) <= 1 {
		return ts
	}

	newValues := make([]float64, len(ts.Values)-1)
	newTimestamps := make([]time.Time, len(ts.Timestamps)-1)

	for i := 1; i < len(ts.Values); i++ {
		newValues[i-1] = ts.Values[i] - ts.Values[i-1]
		newTimestamps[i-1] = ts.Timestamps[i]
	}

	ts.Values = newValues
	ts.Timestamps = newTimestamps

	return ts
}

// Utility functions
func (p *Processor) hasNullValues(values []float64) bool {
	for _, v := range values {
		if math.IsNaN(v) {
			return true
		}
	}
	return false
}

func (p *Processor) hasInfiniteValues(values []float64) bool {
	for _, v := range values {
		if math.IsInf(v, 0) {
			return true
		}
	}
	return false
}

func (p *Processor) minValue(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	min := values[0]
	for _, v := range values[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

func (p *Processor) maxValue(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	max := values[0]
	for _, v := range values[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

func (p *Processor) meanValue(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func (p *Processor) standardDeviation(values []float64, mean float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sumSquares := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquares += diff * diff
	}
	return math.Sqrt(sumSquares / float64(len(values)))
}

func (p *Processor) isChronologicallyOrdered(timestamps []time.Time) bool {
	for i := 1; i < len(timestamps); i++ {
		if timestamps[i].Before(timestamps[i-1]) {
			return false
		}
	}
	return true
}

func (p *Processor) hasDuplicateTimestamps(timestamps []time.Time) bool {
	seen := make(map[int64]bool)
	for _, ts := range timestamps {
		unix := ts.Unix()
		if seen[unix] {
			return true
		}
		seen[unix] = true
	}
	return false
}

func (p *Processor) averageInterval(timestamps []time.Time) time.Duration {
	if len(timestamps) <= 1 {
		return 0
	}
	
	totalDuration := timestamps[len(timestamps)-1].Sub(timestamps[0])
	return totalDuration / time.Duration(len(timestamps)-1)
}

func generateJobID() string {
	return fmt.Sprintf("job_%d", time.Now().UnixNano())
}