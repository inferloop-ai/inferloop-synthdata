package streaming

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/models"
)

// StreamingPipeline orchestrates real-time data streaming operations
type StreamingPipeline struct {
	logger          *logrus.Logger
	config          *PipelineConfig
	mu              sync.RWMutex
	sources         map[string]StreamSource
	processors      map[string]StreamProcessor
	sinks           map[string]StreamSink
	pipelines       map[string]*Pipeline
	metrics         *PipelineMetrics
	eventBus        EventBus
	stateManager    StateManager
	errorHandler    ErrorHandler
	checkpointMgr   CheckpointManager
	activeStreams   map[string]*StreamContext
	stopCh          chan struct{}
	wg              sync.WaitGroup
}

// PipelineConfig configures the streaming pipeline
type PipelineConfig struct {
	Enabled                bool          `json:"enabled"`
	MaxConcurrentPipelines int           `json:"max_concurrent_pipelines"`
	BufferSize             int           `json:"buffer_size"`
	BatchSize              int           `json:"batch_size"`
	FlushInterval          time.Duration `json:"flush_interval"`
	CheckpointInterval     time.Duration `json:"checkpoint_interval"`
	RetryAttempts          int           `json:"retry_attempts"`
	RetryDelay             time.Duration `json:"retry_delay"`
	EnableMetrics          bool          `json:"enable_metrics"`
	EnableCheckpointing    bool          `json:"enable_checkpointing"`
	EnableBackpressure     bool          `json:"enable_backpressure"`
	EnableDeadLetterQueue  bool          `json:"enable_dead_letter_queue"`
	SerializationFormat    string        `json:"serialization_format"`
	CompressionEnabled     bool          `json:"compression_enabled"`
	CompressionAlgorithm   string        `json:"compression_algorithm"`
}

// Pipeline represents a streaming data pipeline
type Pipeline struct {
	ID                string            `json:"id"`
	Name              string            `json:"name"`
	Description       string            `json:"description"`
	Source            SourceConfig      `json:"source"`
	Processors        []ProcessorConfig `json:"processors"`
	Sink              SinkConfig        `json:"sink"`
	ErrorHandling     ErrorConfig       `json:"error_handling"`
	Parallelism       int               `json:"parallelism"`
	Enabled           bool              `json:"enabled"`
	CreatedAt         time.Time         `json:"created_at"`
	UpdatedAt         time.Time         `json:"updated_at"`
}

// SourceConfig configures a stream source
type SourceConfig struct {
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties"`
	SchemaID   string                 `json:"schema_id,omitempty"`
}

// ProcessorConfig configures a stream processor
type ProcessorConfig struct {
	Type       string                 `json:"type"`
	Name       string                 `json:"name"`
	Properties map[string]interface{} `json:"properties"`
	Enabled    bool                   `json:"enabled"`
}

// SinkConfig configures a stream sink
type SinkConfig struct {
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties"`
	SchemaID   string                 `json:"schema_id,omitempty"`
}

// ErrorConfig configures error handling
type ErrorConfig struct {
	Strategy          ErrorStrategy `json:"strategy"`
	MaxRetries        int           `json:"max_retries"`
	RetryBackoff      time.Duration `json:"retry_backoff"`
	DeadLetterTopic   string        `json:"dead_letter_topic,omitempty"`
	IgnoreErrors      bool          `json:"ignore_errors"`
}

// ErrorStrategy defines error handling strategies
type ErrorStrategy string

const (
	ErrorStrategyRetry      ErrorStrategy = "retry"
	ErrorStrategySkip       ErrorStrategy = "skip"
	ErrorStrategyDeadLetter ErrorStrategy = "dead_letter"
	ErrorStrategyFail       ErrorStrategy = "fail"
)

// StreamContext maintains state for an active stream
type StreamContext struct {
	PipelineID   string
	StartTime    time.Time
	ProcessedMsg int64
	ErrorCount   int64
	LastMsg      time.Time
	State        StreamState
	Checkpoint   *Checkpoint
	CancelFunc   context.CancelFunc
}

// StreamState represents the current state of a stream
type StreamState string

const (
	StreamStateStarting StreamState = "starting"
	StreamStateRunning  StreamState = "running"
	StreamStatePaused   StreamState = "paused"
	StreamStateStopping StreamState = "stopping"
	StreamStateStopped  StreamState = "stopped"
	StreamStateError    StreamState = "error"
)

// StreamSource defines the interface for stream sources
type StreamSource interface {
	Name() string
	Start(ctx context.Context, config SourceConfig) (<-chan *StreamMessage, error)
	Stop() error
	GetMetrics() SourceMetrics
}

// StreamProcessor defines the interface for stream processors
type StreamProcessor interface {
	Name() string
	Process(ctx context.Context, message *StreamMessage) (*StreamMessage, error)
	GetMetrics() ProcessorMetrics
}

// StreamSink defines the interface for stream sinks
type StreamSink interface {
	Name() string
	Write(ctx context.Context, message *StreamMessage) error
	Flush(ctx context.Context) error
	Close() error
	GetMetrics() SinkMetrics
}

// StreamMessage represents a message in the streaming pipeline
type StreamMessage struct {
	ID          string                 `json:"id"`
	Topic       string                 `json:"topic"`
	Partition   int32                  `json:"partition"`
	Offset      int64                  `json:"offset"`
	Key         []byte                 `json:"key"`
	Value       []byte                 `json:"value"`
	Headers     map[string]string      `json:"headers"`
	Timestamp   time.Time              `json:"timestamp"`
	SchemaID    string                 `json:"schema_id,omitempty"`
	Metadata    map[string]interface{} `json:"metadata"`
	TimeSeries  *models.TimeSeries     `json:"time_series,omitempty"`
}

// EventBus handles pipeline events
type EventBus interface {
	Publish(event PipelineEvent) error
	Subscribe(eventType string, handler EventHandler) error
	Unsubscribe(eventType string, handler EventHandler) error
}

// PipelineEvent represents a pipeline event
type PipelineEvent struct {
	Type        string                 `json:"type"`
	PipelineID  string                 `json:"pipeline_id"`
	Timestamp   time.Time              `json:"timestamp"`
	Data        map[string]interface{} `json:"data"`
}

// EventHandler handles pipeline events
type EventHandler func(event PipelineEvent) error

// StateManager manages pipeline state
type StateManager interface {
	SaveState(pipelineID string, state interface{}) error
	LoadState(pipelineID string) (interface{}, error)
	DeleteState(pipelineID string) error
}

// ErrorHandler handles pipeline errors
type ErrorHandler interface {
	HandleError(ctx context.Context, err error, message *StreamMessage) error
}

// CheckpointManager manages stream checkpoints
type CheckpointManager interface {
	SaveCheckpoint(streamID string, checkpoint *Checkpoint) error
	LoadCheckpoint(streamID string) (*Checkpoint, error)
	DeleteCheckpoint(streamID string) error
}

// Checkpoint represents a stream processing checkpoint
type Checkpoint struct {
	StreamID     string                 `json:"stream_id"`
	Offset       int64                  `json:"offset"`
	Partition    int32                  `json:"partition"`
	Timestamp    time.Time              `json:"timestamp"`
	ProcessedMsg int64                  `json:"processed_messages"`
	Metadata     map[string]interface{} `json:"metadata"`
}

// PipelineMetrics contains pipeline metrics
type PipelineMetrics struct {
	ActivePipelines    int64             `json:"active_pipelines"`
	TotalMessages      int64             `json:"total_messages"`
	MessagesPerSecond  float64           `json:"messages_per_second"`
	ErrorRate          float64           `json:"error_rate"`
	Latency            time.Duration     `json:"latency"`
	Throughput         float64           `json:"throughput"`
	BackpressureEvents int64             `json:"backpressure_events"`
	Sources            map[string]SourceMetrics    `json:"sources"`
	Processors         map[string]ProcessorMetrics `json:"processors"`
	Sinks              map[string]SinkMetrics      `json:"sinks"`
}

// SourceMetrics contains source-specific metrics
type SourceMetrics struct {
	MessagesRead      int64         `json:"messages_read"`
	BytesRead         int64         `json:"bytes_read"`
	ReadRate          float64       `json:"read_rate"`
	Lag               int64         `json:"lag"`
	LastReadTimestamp time.Time     `json:"last_read_timestamp"`
	Errors            int64         `json:"errors"`
}

// ProcessorMetrics contains processor-specific metrics
type ProcessorMetrics struct {
	MessagesProcessed int64         `json:"messages_processed"`
	ProcessingTime    time.Duration `json:"processing_time"`
	ProcessingRate    float64       `json:"processing_rate"`
	Errors            int64         `json:"errors"`
	Transformations   int64         `json:"transformations"`
}

// SinkMetrics contains sink-specific metrics
type SinkMetrics struct {
	MessagesWritten    int64         `json:"messages_written"`
	BytesWritten       int64         `json:"bytes_written"`
	WriteRate          float64       `json:"write_rate"`
	FlushCount         int64         `json:"flush_count"`
	LastWriteTimestamp time.Time     `json:"last_write_timestamp"`
	Errors             int64         `json:"errors"`
}

// NewStreamingPipeline creates a new streaming pipeline
func NewStreamingPipeline(config *PipelineConfig, logger *logrus.Logger) (*StreamingPipeline, error) {
	if config == nil {
		config = getDefaultPipelineConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	pipeline := &StreamingPipeline{
		logger:        logger,
		config:        config,
		sources:       make(map[string]StreamSource),
		processors:    make(map[string]StreamProcessor),
		sinks:         make(map[string]StreamSink),
		pipelines:     make(map[string]*Pipeline),
		activeStreams: make(map[string]*StreamContext),
		stopCh:        make(chan struct{}),
	}

	// Initialize components
	pipeline.metrics = NewPipelineMetrics()
	pipeline.eventBus = NewInMemoryEventBus()
	pipeline.stateManager = NewInMemoryStateManager()
	pipeline.errorHandler = NewDefaultErrorHandler(config)
	
	if config.EnableCheckpointing {
		pipeline.checkpointMgr = NewInMemoryCheckpointManager()
	}

	// Register default sources, processors, and sinks
	pipeline.registerDefaultComponents()

	return pipeline, nil
}

// Start starts the streaming pipeline engine
func (sp *StreamingPipeline) Start(ctx context.Context) error {
	if !sp.config.Enabled {
		sp.logger.Info("Streaming pipeline disabled")
		return nil
	}

	sp.logger.Info("Starting streaming pipeline engine")

	// Start metrics collection
	if sp.config.EnableMetrics {
		go sp.metricsCollectionLoop(ctx)
	}

	// Start checkpoint routine
	if sp.config.EnableCheckpointing {
		go sp.checkpointLoop(ctx)
	}

	return nil
}

// Stop stops the streaming pipeline engine
func (sp *StreamingPipeline) Stop(ctx context.Context) error {
	sp.logger.Info("Stopping streaming pipeline engine")

	close(sp.stopCh)

	// Stop all active streams
	sp.mu.Lock()
	for streamID, streamCtx := range sp.activeStreams {
		if streamCtx.CancelFunc != nil {
			streamCtx.CancelFunc()
		}
		sp.logger.WithField("stream_id", streamID).Info("Stopped stream")
	}
	sp.mu.Unlock()

	sp.wg.Wait()

	return nil
}

// RegisterPipeline registers a new pipeline definition
func (sp *StreamingPipeline) RegisterPipeline(pipeline *Pipeline) error {
	if err := sp.validatePipeline(pipeline); err != nil {
		return fmt.Errorf("invalid pipeline: %w", err)
	}

	sp.mu.Lock()
	defer sp.mu.Unlock()

	pipeline.CreatedAt = time.Now()
	pipeline.UpdatedAt = time.Now()
	sp.pipelines[pipeline.ID] = pipeline

	sp.logger.WithField("pipeline_id", pipeline.ID).Info("Registered pipeline")
	return nil
}

// StartPipeline starts a registered pipeline
func (sp *StreamingPipeline) StartPipeline(ctx context.Context, pipelineID string) error {
	sp.mu.RLock()
	pipeline, exists := sp.pipelines[pipelineID]
	sp.mu.RUnlock()

	if !exists {
		return fmt.Errorf("pipeline not found: %s", pipelineID)
	}

	if !pipeline.Enabled {
		return fmt.Errorf("pipeline is disabled: %s", pipelineID)
	}

	// Check if already running
	sp.mu.RLock()
	_, running := sp.activeStreams[pipelineID]
	sp.mu.RUnlock()

	if running {
		return fmt.Errorf("pipeline already running: %s", pipelineID)
	}

	// Create stream context
	streamCtx, cancel := context.WithCancel(ctx)
	streamContext := &StreamContext{
		PipelineID: pipelineID,
		StartTime:  time.Now(),
		State:      StreamStateStarting,
		CancelFunc: cancel,
	}

	sp.mu.Lock()
	sp.activeStreams[pipelineID] = streamContext
	sp.mu.Unlock()

	// Start pipeline execution
	sp.wg.Add(1)
	go sp.executePipeline(streamCtx, pipeline, streamContext)

	sp.logger.WithField("pipeline_id", pipelineID).Info("Started pipeline")
	return nil
}

// StopPipeline stops a running pipeline
func (sp *StreamingPipeline) StopPipeline(pipelineID string) error {
	sp.mu.RLock()
	streamCtx, exists := sp.activeStreams[pipelineID]
	sp.mu.RUnlock()

	if !exists {
		return fmt.Errorf("pipeline not running: %s", pipelineID)
	}

	streamCtx.State = StreamStateStopping
	if streamCtx.CancelFunc != nil {
		streamCtx.CancelFunc()
	}

	sp.logger.WithField("pipeline_id", pipelineID).Info("Stopped pipeline")
	return nil
}

// GetPipelineStatus returns the status of a pipeline
func (sp *StreamingPipeline) GetPipelineStatus(pipelineID string) (*StreamContext, error) {
	sp.mu.RLock()
	defer sp.mu.RUnlock()

	streamCtx, exists := sp.activeStreams[pipelineID]
	if !exists {
		return nil, fmt.Errorf("pipeline not found or not running: %s", pipelineID)
	}

	return streamCtx, nil
}

// GetMetrics returns pipeline metrics
func (sp *StreamingPipeline) GetMetrics() *PipelineMetrics {
	return sp.metrics
}

// executePipeline executes a streaming pipeline
func (sp *StreamingPipeline) executePipeline(ctx context.Context, pipeline *Pipeline, streamCtx *StreamContext) {
	defer sp.wg.Done()
	defer func() {
		sp.mu.Lock()
		delete(sp.activeStreams, pipeline.ID)
		sp.mu.Unlock()
	}()

	streamCtx.State = StreamStateRunning

	// Get source
	source, exists := sp.sources[pipeline.Source.Type]
	if !exists {
		sp.logger.WithError(fmt.Errorf("source not found: %s", pipeline.Source.Type)).Error("Failed to start pipeline")
		streamCtx.State = StreamStateError
		return
	}

	// Start source
	messageCh, err := source.Start(ctx, pipeline.Source)
	if err != nil {
		sp.logger.WithError(err).Error("Failed to start source")
		streamCtx.State = StreamStateError
		return
	}

	// Process messages
	for {
		select {
		case <-ctx.Done():
			streamCtx.State = StreamStateStopped
			return
		case message, ok := <-messageCh:
			if !ok {
				streamCtx.State = StreamStateStopped
				return
			}

			if err := sp.processMessage(ctx, pipeline, message, streamCtx); err != nil {
				sp.logger.WithError(err).Error("Failed to process message")
				streamCtx.ErrorCount++
				
				if err := sp.errorHandler.HandleError(ctx, err, message); err != nil {
					sp.logger.WithError(err).Error("Error handler failed")
				}
			}

			streamCtx.ProcessedMsg++
			streamCtx.LastMsg = time.Now()
		}
	}
}

// processMessage processes a single message through the pipeline
func (sp *StreamingPipeline) processMessage(ctx context.Context, pipeline *Pipeline, message *StreamMessage, streamCtx *StreamContext) error {
	currentMessage := message

	// Apply processors in sequence
	for _, processorConfig := range pipeline.Processors {
		if !processorConfig.Enabled {
			continue
		}

		processor, exists := sp.processors[processorConfig.Type]
		if !exists {
			return fmt.Errorf("processor not found: %s", processorConfig.Type)
		}

		processedMessage, err := processor.Process(ctx, currentMessage)
		if err != nil {
			return fmt.Errorf("processor %s failed: %w", processorConfig.Name, err)
		}

		currentMessage = processedMessage
	}

	// Write to sink
	sink, exists := sp.sinks[pipeline.Sink.Type]
	if !exists {
		return fmt.Errorf("sink not found: %s", pipeline.Sink.Type)
	}

	if err := sink.Write(ctx, currentMessage); err != nil {
		return fmt.Errorf("sink write failed: %w", err)
	}

	return nil
}

// Helper methods

func (sp *StreamingPipeline) validatePipeline(pipeline *Pipeline) error {
	if pipeline.ID == "" {
		return fmt.Errorf("pipeline ID is required")
	}

	if pipeline.Source.Type == "" {
		return fmt.Errorf("source type is required")
	}

	if pipeline.Sink.Type == "" {
		return fmt.Errorf("sink type is required")
	}

	return nil
}

func (sp *StreamingPipeline) registerDefaultComponents() {
	// Register default sources
	sp.sources["kafka"] = NewKafkaSource(sp.logger)
	sp.sources["pulsar"] = NewPulsarSource(sp.logger)
	sp.sources["file"] = NewFileSource(sp.logger)

	// Register default processors
	sp.processors["filter"] = NewFilterProcessor()
	sp.processors["transform"] = NewTransformProcessor()
	sp.processors["aggregate"] = NewAggregateProcessor()

	// Register default sinks
	sp.sinks["kafka"] = NewKafkaSink(sp.logger)
	sp.sinks["file"] = NewFileSink(sp.logger)
	sp.sinks["database"] = NewDatabaseSink(sp.logger)
}

func (sp *StreamingPipeline) metricsCollectionLoop(ctx context.Context) {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			sp.collectMetrics()
		}
	}
}

func (sp *StreamingPipeline) collectMetrics() {
	sp.mu.RLock()
	defer sp.mu.RUnlock()

	sp.metrics.ActivePipelines = int64(len(sp.activeStreams))

	var totalMessages int64
	var totalErrors int64

	for _, streamCtx := range sp.activeStreams {
		totalMessages += streamCtx.ProcessedMsg
		totalErrors += streamCtx.ErrorCount
	}

	sp.metrics.TotalMessages = totalMessages
	if totalMessages > 0 {
		sp.metrics.ErrorRate = float64(totalErrors) / float64(totalMessages) * 100
	}
}

func (sp *StreamingPipeline) checkpointLoop(ctx context.Context) {
	ticker := time.NewTicker(sp.config.CheckpointInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			sp.createCheckpoints()
		}
	}
}

func (sp *StreamingPipeline) createCheckpoints() {
	sp.mu.RLock()
	defer sp.mu.RUnlock()

	for streamID, streamCtx := range sp.activeStreams {
		checkpoint := &Checkpoint{
			StreamID:     streamID,
			Timestamp:    time.Now(),
			ProcessedMsg: streamCtx.ProcessedMsg,
		}

		if err := sp.checkpointMgr.SaveCheckpoint(streamID, checkpoint); err != nil {
			sp.logger.WithError(err).WithField("stream_id", streamID).Error("Failed to save checkpoint")
		}
	}
}

func getDefaultPipelineConfig() *PipelineConfig {
	return &PipelineConfig{
		Enabled:                true,
		MaxConcurrentPipelines: 10,
		BufferSize:             1000,
		BatchSize:              100,
		FlushInterval:          time.Second * 5,
		CheckpointInterval:     time.Minute * 5,
		RetryAttempts:          3,
		RetryDelay:             time.Second * 5,
		EnableMetrics:          true,
		EnableCheckpointing:    true,
		EnableBackpressure:     true,
		EnableDeadLetterQueue:  true,
		SerializationFormat:    "json",
		CompressionEnabled:     true,
		CompressionAlgorithm:   "gzip",
	}
}