package streaming

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/models"
)

// FileSource implements StreamSource for file-based input
type FileSource struct {
	logger   *logrus.Logger
	config   *FileSourceConfig
	metrics  *SourceMetrics
	file     *os.File
	decoder  *json.Decoder
	mu       sync.RWMutex
}

// FileSourceConfig configures file source
type FileSourceConfig struct {
	FilePath     string `json:"file_path"`
	Format       string `json:"format"` // json, csv, jsonlines
	ReadMode     string `json:"read_mode"` // stream, batch
	PollInterval time.Duration `json:"poll_interval"`
	FollowFile   bool   `json:"follow_file"` // Like tail -f
}

// NewFileSource creates a new file source
func NewFileSource(logger *logrus.Logger) *FileSource {
	return &FileSource{
		logger:  logger,
		metrics: &SourceMetrics{},
	}
}

// Name returns the source name
func (fs *FileSource) Name() string {
	return "file"
}

// Start starts the file source
func (fs *FileSource) Start(ctx context.Context, config SourceConfig) (<-chan *StreamMessage, error) {
	fileConfig, err := fs.parseConfig(config.Properties)
	if err != nil {
		return nil, fmt.Errorf("failed to parse file config: %w", err)
	}

	fs.config = fileConfig

	// Open file
	file, err := os.Open(fileConfig.FilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %s: %w", fileConfig.FilePath, err)
	}

	fs.file = file

	if fileConfig.Format == "json" || fileConfig.Format == "jsonlines" {
		fs.decoder = json.NewDecoder(file)
	}

	// Create message channel
	messageCh := make(chan *StreamMessage, 100)

	// Start reading messages
	go fs.readMessages(ctx, messageCh)

	fs.logger.WithField("file_path", fileConfig.FilePath).Info("Started file source")

	return messageCh, nil
}

// Stop stops the file source
func (fs *FileSource) Stop() error {
	if fs.file != nil {
		return fs.file.Close()
	}
	return nil
}

// GetMetrics returns source metrics
func (fs *FileSource) GetMetrics() SourceMetrics {
	fs.mu.RLock()
	defer fs.mu.RUnlock()
	return *fs.metrics
}

func (fs *FileSource) parseConfig(properties map[string]interface{}) (*FileSourceConfig, error) {
	configBytes, err := json.Marshal(properties)
	if err != nil {
		return nil, err
	}

	var config FileSourceConfig
	if err := json.Unmarshal(configBytes, &config); err != nil {
		return nil, err
	}

	// Set defaults
	if config.Format == "" {
		config.Format = "json"
	}
	if config.ReadMode == "" {
		config.ReadMode = "stream"
	}
	if config.PollInterval == 0 {
		config.PollInterval = time.Second
	}

	return &config, nil
}

func (fs *FileSource) readMessages(ctx context.Context, messageCh chan<- *StreamMessage) {
	defer close(messageCh)

	counter := 0
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		var message *StreamMessage
		var err error

		switch fs.config.Format {
		case "json", "jsonlines":
			message, err = fs.readJSONMessage(counter)
		default:
			err = fmt.Errorf("unsupported format: %s", fs.config.Format)
		}

		if err != nil {
			if err.Error() == "EOF" {
				if fs.config.FollowFile {
					// Wait and try again for new data
					time.Sleep(fs.config.PollInterval)
					continue
				} else {
					// End of file, stop reading
					return
				}
			}
			fs.logger.WithError(err).Error("Failed to read message from file")
			continue
		}

		if message != nil {
			fs.updateMetrics(message)

			select {
			case messageCh <- message:
				counter++
			case <-ctx.Done():
				return
			}
		}

		// Add small delay for streaming mode
		if fs.config.ReadMode == "stream" {
			time.Sleep(fs.config.PollInterval)
		}
	}
}

func (fs *FileSource) readJSONMessage(counter int) (*StreamMessage, error) {
	var data interface{}
	if err := fs.decoder.Decode(&data); err != nil {
		return nil, err
	}

	// Convert to JSON bytes
	jsonBytes, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}

	message := &StreamMessage{
		ID:        fmt.Sprintf("file-%d", counter),
		Topic:     "file-input",
		Value:     jsonBytes,
		Timestamp: time.Now(),
		Metadata: map[string]interface{}{
			"source":    "file",
			"file_path": fs.config.FilePath,
		},
	}

	// Try to parse as TimeSeries
	var timeSeries models.TimeSeries
	if err := json.Unmarshal(jsonBytes, &timeSeries); err == nil {
		message.TimeSeries = &timeSeries
	}

	return message, nil
}

func (fs *FileSource) updateMetrics(message *StreamMessage) {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	fs.metrics.MessagesRead++
	fs.metrics.BytesRead += int64(len(message.Value))
	fs.metrics.LastReadTimestamp = time.Now()
}

// FileSink implements StreamSink for file-based output
type FileSink struct {
	logger   *logrus.Logger
	config   *FileSinkConfig
	metrics  *SinkMetrics
	file     *os.File
	encoder  *json.Encoder
	mu       sync.RWMutex
}

// FileSinkConfig configures file sink
type FileSinkConfig struct {
	FilePath      string `json:"file_path"`
	Format        string `json:"format"` // json, csv, jsonlines
	WriteMode     string `json:"write_mode"` // append, overwrite
	FlushInterval time.Duration `json:"flush_interval"`
	BufferSize    int    `json:"buffer_size"`
}

// NewFileSink creates a new file sink
func NewFileSink(logger *logrus.Logger) *FileSink {
	return &FileSink{
		logger:  logger,
		metrics: &SinkMetrics{},
	}
}

// Name returns the sink name
func (fs *FileSink) Name() string {
	return "file"
}

// Write writes a message to file
func (fs *FileSink) Write(ctx context.Context, message *StreamMessage) error {
	if fs.file == nil {
		return fmt.Errorf("file sink not initialized")
	}

	var data interface{}
	if message.TimeSeries != nil {
		data = message.TimeSeries
	} else {
		// Try to parse message value as JSON
		if err := json.Unmarshal(message.Value, &data); err != nil {
			data = string(message.Value)
		}
	}

	if err := fs.encoder.Encode(data); err != nil {
		fs.updateErrorMetrics()
		return fmt.Errorf("failed to encode message: %w", err)
	}

	fs.updateSuccessMetrics(message)
	return nil
}

// Flush flushes any buffered data
func (fs *FileSink) Flush(ctx context.Context) error {
	if fs.file != nil {
		if err := fs.file.Sync(); err != nil {
			return fmt.Errorf("failed to flush file: %w", err)
		}

		fs.mu.Lock()
		fs.metrics.FlushCount++
		fs.mu.Unlock()
	}
	return nil
}

// Close closes the file sink
func (fs *FileSink) Close() error {
	if fs.file != nil {
		return fs.file.Close()
	}
	return nil
}

// GetMetrics returns sink metrics
func (fs *FileSink) GetMetrics() SinkMetrics {
	fs.mu.RLock()
	defer fs.mu.RUnlock()
	return *fs.metrics
}

func (fs *FileSink) updateSuccessMetrics(message *StreamMessage) {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	fs.metrics.MessagesWritten++
	fs.metrics.BytesWritten += int64(len(message.Value))
	fs.metrics.LastWriteTimestamp = time.Now()
}

func (fs *FileSink) updateErrorMetrics() {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	fs.metrics.Errors++
}

// DatabaseSink implements StreamSink for database output
type DatabaseSink struct {
	logger   *logrus.Logger
	config   *DatabaseSinkConfig
	metrics  *SinkMetrics
	mu       sync.RWMutex
}

// DatabaseSinkConfig configures database sink
type DatabaseSinkConfig struct {
	Driver       string `json:"driver"` // postgres, mysql, mongodb
	ConnectionString string `json:"connection_string"`
	Table        string `json:"table"`
	BatchSize    int    `json:"batch_size"`
	FlushInterval time.Duration `json:"flush_interval"`
}

// NewDatabaseSink creates a new database sink
func NewDatabaseSink(logger *logrus.Logger) *DatabaseSink {
	return &DatabaseSink{
		logger:  logger,
		metrics: &SinkMetrics{},
	}
}

// Name returns the sink name
func (ds *DatabaseSink) Name() string {
	return "database"
}

// Write writes a message to database
func (ds *DatabaseSink) Write(ctx context.Context, message *StreamMessage) error {
	// Mock implementation - would integrate with actual database
	ds.logger.WithFields(logrus.Fields{
		"message_id": message.ID,
		"topic":      message.Topic,
	}).Debug("Writing message to database")

	ds.updateSuccessMetrics(message)
	return nil
}

// Flush flushes any buffered data
func (ds *DatabaseSink) Flush(ctx context.Context) error {
	ds.mu.Lock()
	ds.metrics.FlushCount++
	ds.mu.Unlock()
	return nil
}

// Close closes the database sink
func (ds *DatabaseSink) Close() error {
	return nil
}

// GetMetrics returns sink metrics
func (ds *DatabaseSink) GetMetrics() SinkMetrics {
	ds.mu.RLock()
	defer ds.mu.RUnlock()
	return *ds.metrics
}

func (ds *DatabaseSink) updateSuccessMetrics(message *StreamMessage) {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	ds.metrics.MessagesWritten++
	ds.metrics.BytesWritten += int64(len(message.Value))
	ds.metrics.LastWriteTimestamp = time.Now()
}

// Helper components

// NewPipelineMetrics creates new pipeline metrics
func NewPipelineMetrics() *PipelineMetrics {
	return &PipelineMetrics{
		Sources:    make(map[string]SourceMetrics),
		Processors: make(map[string]ProcessorMetrics),
		Sinks:      make(map[string]SinkMetrics),
	}
}

// InMemoryEventBus implements EventBus using in-memory channels
type InMemoryEventBus struct {
	subscribers map[string][]EventHandler
	mu          sync.RWMutex
}

// NewInMemoryEventBus creates a new in-memory event bus
func NewInMemoryEventBus() *InMemoryEventBus {
	return &InMemoryEventBus{
		subscribers: make(map[string][]EventHandler),
	}
}

// Publish publishes an event
func (eb *InMemoryEventBus) Publish(event PipelineEvent) error {
	eb.mu.RLock()
	handlers, exists := eb.subscribers[event.Type]
	eb.mu.RUnlock()

	if !exists {
		return nil // No subscribers
	}

	for _, handler := range handlers {
		go func(h EventHandler) {
			if err := h(event); err != nil {
				// Log error but don't fail the publish
				fmt.Printf("Event handler error: %v\n", err)
			}
		}(handler)
	}

	return nil
}

// Subscribe subscribes to events
func (eb *InMemoryEventBus) Subscribe(eventType string, handler EventHandler) error {
	eb.mu.Lock()
	defer eb.mu.Unlock()

	eb.subscribers[eventType] = append(eb.subscribers[eventType], handler)
	return nil
}

// Unsubscribe unsubscribes from events
func (eb *InMemoryEventBus) Unsubscribe(eventType string, handler EventHandler) error {
	eb.mu.Lock()
	defer eb.mu.Unlock()

	handlers := eb.subscribers[eventType]
	for i, h := range handlers {
		// Note: This is a simplified comparison; in practice, you'd need a better way to identify handlers
		if fmt.Sprintf("%p", h) == fmt.Sprintf("%p", handler) {
			eb.subscribers[eventType] = append(handlers[:i], handlers[i+1:]...)
			break
		}
	}

	return nil
}

// InMemoryStateManager implements StateManager using in-memory storage
type InMemoryStateManager struct {
	states map[string]interface{}
	mu     sync.RWMutex
}

// NewInMemoryStateManager creates a new in-memory state manager
func NewInMemoryStateManager() *InMemoryStateManager {
	return &InMemoryStateManager{
		states: make(map[string]interface{}),
	}
}

// SaveState saves pipeline state
func (sm *InMemoryStateManager) SaveState(pipelineID string, state interface{}) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.states[pipelineID] = state
	return nil
}

// LoadState loads pipeline state
func (sm *InMemoryStateManager) LoadState(pipelineID string) (interface{}, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	state, exists := sm.states[pipelineID]
	if !exists {
		return nil, fmt.Errorf("state not found for pipeline: %s", pipelineID)
	}

	return state, nil
}

// DeleteState deletes pipeline state
func (sm *InMemoryStateManager) DeleteState(pipelineID string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	delete(sm.states, pipelineID)
	return nil
}

// DefaultErrorHandler implements ErrorHandler with basic error handling strategies
type DefaultErrorHandler struct {
	config *PipelineConfig
	logger *logrus.Logger
}

// NewDefaultErrorHandler creates a new default error handler
func NewDefaultErrorHandler(config *PipelineConfig) *DefaultErrorHandler {
	return &DefaultErrorHandler{
		config: config,
		logger: logrus.New(),
	}
}

// HandleError handles pipeline errors
func (eh *DefaultErrorHandler) HandleError(ctx context.Context, err error, message *StreamMessage) error {
	eh.logger.WithFields(logrus.Fields{
		"error":      err.Error(),
		"message_id": message.ID,
		"topic":      message.Topic,
	}).Error("Pipeline error occurred")

	// Implement basic retry logic
	if eh.config.RetryAttempts > 0 {
		// In a real implementation, this would retry the operation
		time.Sleep(eh.config.RetryDelay)
	}

	if eh.config.EnableDeadLetterQueue {
		// In a real implementation, this would send to dead letter queue
		eh.logger.WithField("message_id", message.ID).Info("Message sent to dead letter queue")
	}

	return nil
}

// InMemoryCheckpointManager implements CheckpointManager using in-memory storage
type InMemoryCheckpointManager struct {
	checkpoints map[string]*Checkpoint
	mu          sync.RWMutex
}

// NewInMemoryCheckpointManager creates a new in-memory checkpoint manager
func NewInMemoryCheckpointManager() *InMemoryCheckpointManager {
	return &InMemoryCheckpointManager{
		checkpoints: make(map[string]*Checkpoint),
	}
}

// SaveCheckpoint saves a stream checkpoint
func (cm *InMemoryCheckpointManager) SaveCheckpoint(streamID string, checkpoint *Checkpoint) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.checkpoints[streamID] = checkpoint
	return nil
}

// LoadCheckpoint loads a stream checkpoint
func (cm *InMemoryCheckpointManager) LoadCheckpoint(streamID string) (*Checkpoint, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	checkpoint, exists := cm.checkpoints[streamID]
	if !exists {
		return nil, fmt.Errorf("checkpoint not found for stream: %s", streamID)
	}

	return checkpoint, nil
}

// DeleteCheckpoint deletes a stream checkpoint
func (cm *InMemoryCheckpointManager) DeleteCheckpoint(streamID string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	delete(cm.checkpoints, streamID)
	return nil
}

// CronScheduler implements Scheduler using cron expressions
type CronScheduler struct {
	logger         *logrus.Logger
	scheduledItems map[string]*ScheduledWorkflow
	mu             sync.RWMutex
}

// NewCronScheduler creates a new cron scheduler
func NewCronScheduler(logger *logrus.Logger) *CronScheduler {
	return &CronScheduler{
		logger:         logger,
		scheduledItems: make(map[string]*ScheduledWorkflow),
	}
}

// Schedule schedules a workflow
func (cs *CronScheduler) Schedule(workflow *WorkflowDefinition) error {
	if workflow.Schedule == nil || !workflow.Schedule.Enabled {
		return nil
	}

	cs.mu.Lock()
	defer cs.mu.Unlock()

	scheduled := &ScheduledWorkflow{
		WorkflowID: workflow.ID,
		NextRun:    time.Now().Add(time.Minute), // Simplified scheduling
		Enabled:    true,
	}

	cs.scheduledItems[workflow.ID] = scheduled
	cs.logger.WithField("workflow_id", workflow.ID).Info("Scheduled workflow")

	return nil
}

// Unschedule unschedules a workflow
func (cs *CronScheduler) Unschedule(workflowID string) error {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	delete(cs.scheduledItems, workflowID)
	cs.logger.WithField("workflow_id", workflowID).Info("Unscheduled workflow")

	return nil
}

// GetScheduled returns all scheduled workflows
func (cs *CronScheduler) GetScheduled() []*ScheduledWorkflow {
	cs.mu.RLock()
	defer cs.mu.RUnlock()

	scheduled := make([]*ScheduledWorkflow, 0, len(cs.scheduledItems))
	for _, item := range cs.scheduledItems {
		scheduled = append(scheduled, item)
	}

	return scheduled
}

// WorkflowDefinition represents a workflow definition (from pipeline.go)
type WorkflowDefinition struct {
	ID          string                    `json:"id"`
	Name        string                    `json:"name"`
	Description string                    `json:"description"`
	Schedule    *ScheduleSpec             `json:"schedule,omitempty"`
	Enabled     bool                      `json:"enabled"`
}

// ScheduleSpec defines workflow scheduling (from pipeline.go)
type ScheduleSpec struct {
	CronExpression string     `json:"cron_expression"`
	Timezone       string     `json:"timezone"`
	StartDate      *time.Time `json:"start_date,omitempty"`
	EndDate        *time.Time `json:"end_date,omitempty"`
	Enabled        bool       `json:"enabled"`
}