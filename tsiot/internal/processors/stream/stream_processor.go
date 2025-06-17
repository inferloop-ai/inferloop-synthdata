package stream

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/your-org/tsiot/internal/storage"
)

// Processor handles real-time stream processing of time series data
type Processor struct {
	logger    *logrus.Logger
	config    *Config
	pipelines map[string]*Pipeline
	mu        sync.RWMutex
	running   bool
	wg        sync.WaitGroup
}

// Config contains stream processor configuration
type Config struct {
	MaxPipelines     int           `json:"max_pipelines" yaml:"max_pipelines"`
	BufferSize       int           `json:"buffer_size" yaml:"buffer_size"`
	FlushInterval    time.Duration `json:"flush_interval" yaml:"flush_interval"`
	ProcessTimeout   time.Duration `json:"process_timeout" yaml:"process_timeout"`
	EnableMetrics    bool          `json:"enable_metrics" yaml:"enable_metrics"`
	WindowSize       time.Duration `json:"window_size" yaml:"window_size"`
	CompressionType  string        `json:"compression_type" yaml:"compression_type"`
	CheckpointInterval time.Duration `json:"checkpoint_interval" yaml:"checkpoint_interval"`
}

// Pipeline represents a stream processing pipeline
type Pipeline struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	InputQueue  chan *StreamEvent      `json:"-"`
	OutputQueue chan *StreamResult     `json:"-"`
	Processors  []StreamProcessorFunc  `json:"-"`
	Config      *PipelineConfig        `json:"config"`
	Metrics     *PipelineMetrics       `json:"metrics"`
	State       PipelineState          `json:"state"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	mu          sync.RWMutex           `json:"-"`
}

// PipelineConfig contains configuration for a pipeline
type PipelineConfig struct {
	WindowSize      time.Duration          `json:"window_size"`
	BufferSize      int                    `json:"buffer_size"`
	Parallelism     int                    `json:"parallelism"`
	CheckpointInterval time.Duration       `json:"checkpoint_interval"`
	Parameters      map[string]interface{} `json:"parameters"`
}

// PipelineMetrics contains metrics for a pipeline
type PipelineMetrics struct {
	EventsProcessed   int64         `json:"events_processed"`
	EventsSkipped     int64         `json:"events_skipped"`
	EventsFailed      int64         `json:"events_failed"`
	ProcessingTime    time.Duration `json:"processing_time"`
	LastProcessedTime time.Time     `json:"last_processed_time"`
	Throughput        float64       `json:"throughput"` // events per second
}

// PipelineState represents the state of a pipeline
type PipelineState string

const (
	StateInitialized PipelineState = "initialized"
	StateRunning     PipelineState = "running"
	StatePaused      PipelineState = "paused"
	StateStopped     PipelineState = "stopped"
	StateError       PipelineState = "error"
)

// StreamEvent represents an event in the stream
type StreamEvent struct {
	ID        string                 `json:"id"`
	Type      EventType              `json:"type"`
	Timestamp time.Time              `json:"timestamp"`
	SeriesID  string                 `json:"series_id"`
	Data      interface{}            `json:"data"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// EventType represents the type of stream event
type EventType string

const (
	EventTypeDataPoint     EventType = "data_point"
	EventTypeTimeSeries    EventType = "time_series"
	EventTypeAggregation   EventType = "aggregation"
	EventTypeTransformation EventType = "transformation"
	EventTypeAlert         EventType = "alert"
	EventTypeHeartbeat     EventType = "heartbeat"
)

// StreamResult represents the result of stream processing
type StreamResult struct {
	EventID     string                 `json:"event_id"`
	PipelineID  string                 `json:"pipeline_id"`
	Status      ResultStatus           `json:"status"`
	Data        interface{}            `json:"data"`
	Error       string                 `json:"error,omitempty"`
	ProcessedAt time.Time              `json:"processed_at"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// ResultStatus represents the status of a processing result
type ResultStatus string

const (
	StatusSuccess ResultStatus = "success"
	StatusError   ResultStatus = "error"
	StatusSkipped ResultStatus = "skipped"
)

// StreamProcessorFunc is a function that processes stream events
type StreamProcessorFunc func(ctx context.Context, event *StreamEvent) (*StreamResult, error)

// DataPoint represents a single data point in a time series
type DataPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Value     float64   `json:"value"`
	Quality   float64   `json:"quality,omitempty"`
}

// Window represents a time window for stream processing
type Window struct {
	Start     time.Time    `json:"start"`
	End       time.Time    `json:"end"`
	DataPoints []*DataPoint `json:"data_points"`
	Metadata   map[string]interface{} `json:"metadata"`
}

// NewProcessor creates a new stream processor
func NewProcessor(logger *logrus.Logger, config *Config) *Processor {
	if config == nil {
		config = DefaultConfig()
	}

	return &Processor{
		logger:    logger,
		config:    config,
		pipelines: make(map[string]*Pipeline),
	}
}

// DefaultConfig returns default stream processor configuration
func DefaultConfig() *Config {
	return &Config{
		MaxPipelines:       10,
		BufferSize:         10000,
		FlushInterval:      1 * time.Second,
		ProcessTimeout:     30 * time.Second,
		EnableMetrics:      true,
		WindowSize:         5 * time.Minute,
		CompressionType:    "gzip",
		CheckpointInterval: 30 * time.Second,
	}
}

// Start starts the stream processor
func (p *Processor) Start(ctx context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.running {
		return fmt.Errorf("stream processor is already running")
	}

	p.logger.Info("Starting stream processor")
	p.running = true

	// Start checkpoint manager
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		p.checkpointManager(ctx)
	}()

	p.logger.Info("Stream processor started")
	return nil
}

// Stop stops the stream processor
func (p *Processor) Stop() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.running {
		return fmt.Errorf("stream processor is not running")
	}

	p.logger.Info("Stopping stream processor")

	// Stop all pipelines
	for _, pipeline := range p.pipelines {
		p.stopPipeline(pipeline)
	}

	p.running = false

	// Wait for all goroutines to finish
	p.wg.Wait()

	p.logger.Info("Stream processor stopped")
	return nil
}

// CreatePipeline creates a new stream processing pipeline
func (p *Processor) CreatePipeline(id, name string, config *PipelineConfig) (*Pipeline, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if _, exists := p.pipelines[id]; exists {
		return nil, fmt.Errorf("pipeline with ID %s already exists", id)
	}

	if config == nil {
		config = &PipelineConfig{
			WindowSize:         p.config.WindowSize,
			BufferSize:         p.config.BufferSize,
			Parallelism:        1,
			CheckpointInterval: p.config.CheckpointInterval,
			Parameters:         make(map[string]interface{}),
		}
	}

	pipeline := &Pipeline{
		ID:          id,
		Name:        name,
		InputQueue:  make(chan *StreamEvent, config.BufferSize),
		OutputQueue: make(chan *StreamResult, config.BufferSize),
		Processors:  make([]StreamProcessorFunc, 0),
		Config:      config,
		Metrics:     &PipelineMetrics{},
		State:       StateInitialized,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	p.pipelines[id] = pipeline

	p.logger.WithFields(logrus.Fields{
		"pipeline_id":   id,
		"pipeline_name": name,
	}).Info("Pipeline created")

	return pipeline, nil
}

// AddProcessor adds a processor function to a pipeline
func (p *Processor) AddProcessor(pipelineID string, processor StreamProcessorFunc) error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	pipeline, exists := p.pipelines[pipelineID]
	if !exists {
		return fmt.Errorf("pipeline %s not found", pipelineID)
	}

	pipeline.mu.Lock()
	defer pipeline.mu.Unlock()

	pipeline.Processors = append(pipeline.Processors, processor)
	pipeline.UpdatedAt = time.Now()

	return nil
}

// StartPipeline starts a specific pipeline
func (p *Processor) StartPipeline(pipelineID string, ctx context.Context) error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	pipeline, exists := p.pipelines[pipelineID]
	if !exists {
		return fmt.Errorf("pipeline %s not found", pipelineID)
	}

	pipeline.mu.Lock()
	defer pipeline.mu.Unlock()

	if pipeline.State == StateRunning {
		return fmt.Errorf("pipeline %s is already running", pipelineID)
	}

	pipeline.State = StateRunning

	// Start pipeline workers
	for i := 0; i < pipeline.Config.Parallelism; i++ {
		p.wg.Add(1)
		go func(workerID int) {
			defer p.wg.Done()
			p.pipelineWorker(ctx, pipeline, workerID)
		}(i)
	}

	p.logger.WithField("pipeline_id", pipelineID).Info("Pipeline started")
	return nil
}

// SendEvent sends an event to a pipeline for processing
func (p *Processor) SendEvent(pipelineID string, event *StreamEvent) error {
	p.mu.RLock()
	defer p.mu.RUnlock()

	pipeline, exists := p.pipelines[pipelineID]
	if !exists {
		return fmt.Errorf("pipeline %s not found", pipelineID)
	}

	if pipeline.State != StateRunning {
		return fmt.Errorf("pipeline %s is not running", pipelineID)
	}

	select {
	case pipeline.InputQueue <- event:
		return nil
	default:
		return fmt.Errorf("pipeline %s input queue is full", pipelineID)
	}
}

// GetResults returns a channel for receiving processing results from a pipeline
func (p *Processor) GetResults(pipelineID string) (<-chan *StreamResult, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	pipeline, exists := p.pipelines[pipelineID]
	if !exists {
		return nil, fmt.Errorf("pipeline %s not found", pipelineID)
	}

	return pipeline.OutputQueue, nil
}

// pipelineWorker processes events in a pipeline
func (p *Processor) pipelineWorker(ctx context.Context, pipeline *Pipeline, workerID int) {
	logger := p.logger.WithFields(logrus.Fields{
		"pipeline_id": pipeline.ID,
		"worker_id":   workerID,
	})

	logger.Info("Pipeline worker started")
	defer logger.Info("Pipeline worker stopped")

	for {
		select {
		case <-ctx.Done():
			return
		case event, ok := <-pipeline.InputQueue:
			if !ok {
				return
			}

			p.processEvent(ctx, pipeline, event, logger)
		}
	}
}

// processEvent processes a single event through the pipeline
func (p *Processor) processEvent(ctx context.Context, pipeline *Pipeline, event *StreamEvent, logger *logrus.Entry) {
	startTime := time.Now()

	// Create processing context with timeout
	processCtx, cancel := context.WithTimeout(ctx, p.config.ProcessTimeout)
	defer cancel()

	var result *StreamResult
	var err error

	// Process event through all processors in the pipeline
	for i, processor := range pipeline.Processors {
		logger.WithField("processor_index", i).Debug("Processing event")
		
		result, err = processor(processCtx, event)
		if err != nil {
			logger.WithError(err).Error("Processor failed")
			result = &StreamResult{
				EventID:     event.ID,
				PipelineID:  pipeline.ID,
				Status:      StatusError,
				Error:       err.Error(),
				ProcessedAt: time.Now(),
			}
			break
		}

		// If processor returns a result, use it as input for next processor
		if result != nil && result.Data != nil {
			// Update event data for next processor
			event.Data = result.Data
		}
	}

	// Update metrics
	pipeline.mu.Lock()
	pipeline.Metrics.EventsProcessed++
	pipeline.Metrics.ProcessingTime += time.Since(startTime)
	pipeline.Metrics.LastProcessedTime = time.Now()
	
	if err != nil {
		pipeline.Metrics.EventsFailed++
	}
	
	// Calculate throughput
	if pipeline.Metrics.EventsProcessed > 0 {
		duration := time.Since(pipeline.CreatedAt).Seconds()
		if duration > 0 {
			pipeline.Metrics.Throughput = float64(pipeline.Metrics.EventsProcessed) / duration
		}
	}
	pipeline.mu.Unlock()

	// Send result if available
	if result != nil {
		select {
		case pipeline.OutputQueue <- result:
			// Result sent successfully
		case <-processCtx.Done():
			logger.Warn("Context cancelled while sending result")
		default:
			logger.Warn("Output queue is full, dropping result")
		}
	}
}

// stopPipeline stops a specific pipeline
func (p *Processor) stopPipeline(pipeline *Pipeline) {
	pipeline.mu.Lock()
	defer pipeline.mu.Unlock()

	if pipeline.State == StateStopped {
		return
	}

	pipeline.State = StateStopped
	close(pipeline.InputQueue)
	close(pipeline.OutputQueue)

	p.logger.WithField("pipeline_id", pipeline.ID).Info("Pipeline stopped")
}

// checkpointManager manages periodic checkpoints
func (p *Processor) checkpointManager(ctx context.Context) {
	ticker := time.NewTicker(p.config.CheckpointInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.createCheckpoint()
		}
	}
}

// createCheckpoint creates a checkpoint of the current state
func (p *Processor) createCheckpoint() {
	p.mu.RLock()
	defer p.mu.RUnlock()

	p.logger.Info("Creating checkpoint")

	// In a real implementation, this would persist pipeline states,
	// metrics, and processing positions to allow for recovery
	for pipelineID, pipeline := range p.pipelines {
		pipeline.mu.RLock()
		p.logger.WithFields(logrus.Fields{
			"pipeline_id":      pipelineID,
			"events_processed": pipeline.Metrics.EventsProcessed,
			"state":            pipeline.State,
		}).Debug("Pipeline checkpoint")
		pipeline.mu.RUnlock()
	}
}

// GetPipelineStatus returns the status of a pipeline
func (p *Processor) GetPipelineStatus(pipelineID string) (map[string]interface{}, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	pipeline, exists := p.pipelines[pipelineID]
	if !exists {
		return nil, fmt.Errorf("pipeline %s not found", pipelineID)
	}

	pipeline.mu.RLock()
	defer pipeline.mu.RUnlock()

	return map[string]interface{}{
		"id":              pipeline.ID,
		"name":            pipeline.Name,
		"state":           pipeline.State,
		"metrics":         pipeline.Metrics,
		"created_at":      pipeline.CreatedAt,
		"updated_at":      pipeline.UpdatedAt,
		"input_queue_size": len(pipeline.InputQueue),
		"output_queue_size": len(pipeline.OutputQueue),
	}, nil
}

// GetAllPipelineStatus returns the status of all pipelines
func (p *Processor) GetAllPipelineStatus() map[string]interface{} {
	p.mu.RLock()
	defer p.mu.RUnlock()

	status := make(map[string]interface{})
	for pipelineID := range p.pipelines {
		if pipelineStatus, err := p.GetPipelineStatus(pipelineID); err == nil {
			status[pipelineID] = pipelineStatus
		}
	}

	return status
}

// Built-in stream processors

// AggregationProcessor creates a processor that aggregates data points within time windows
func AggregationProcessor(windowSize time.Duration, aggType string) StreamProcessorFunc {
	return func(ctx context.Context, event *StreamEvent) (*StreamResult, error) {
		if event.Type != EventTypeDataPoint {
			return &StreamResult{
				EventID:     event.ID,
				Status:      StatusSkipped,
				ProcessedAt: time.Now(),
			}, nil
		}

		// In a real implementation, this would maintain state across events
		// and aggregate data points within the specified window
		return &StreamResult{
			EventID: event.ID,
			Status:  StatusSuccess,
			Data: map[string]interface{}{
				"aggregation_type": aggType,
				"window_size":      windowSize.String(),
				"processed_at":     time.Now(),
			},
			ProcessedAt: time.Now(),
		}, nil
	}
}

// FilterProcessor creates a processor that filters events based on criteria
func FilterProcessor(filterFunc func(event *StreamEvent) bool) StreamProcessorFunc {
	return func(ctx context.Context, event *StreamEvent) (*StreamResult, error) {
		if !filterFunc(event) {
			return &StreamResult{
				EventID:     event.ID,
				Status:      StatusSkipped,
				ProcessedAt: time.Now(),
			}, nil
		}

		return &StreamResult{
			EventID:     event.ID,
			Status:      StatusSuccess,
			Data:        event.Data,
			ProcessedAt: time.Now(),
		}, nil
	}
}

// TransformProcessor creates a processor that transforms event data
func TransformProcessor(transformFunc func(data interface{}) (interface{}, error)) StreamProcessorFunc {
	return func(ctx context.Context, event *StreamEvent) (*StreamResult, error) {
		transformedData, err := transformFunc(event.Data)
		if err != nil {
			return &StreamResult{
				EventID:     event.ID,
				Status:      StatusError,
				Error:       err.Error(),
				ProcessedAt: time.Now(),
			}, nil
		}

		return &StreamResult{
			EventID:     event.ID,
			Status:      StatusSuccess,
			Data:        transformedData,
			ProcessedAt: time.Now(),
		}, nil
	}
}