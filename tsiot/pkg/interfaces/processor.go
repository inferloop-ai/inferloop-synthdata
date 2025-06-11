package interfaces

import (
	"context"
	"time"

	"github.com/inferloop/tsiot/pkg/models"
)

// Processor defines the interface for data processing
type Processor interface {
	// GetType returns the processor type
	GetType() string

	// GetName returns a human-readable name for the processor
	GetName() string

	// GetDescription returns a description of the processor
	GetDescription() string

	// Process processes the input data
	Process(ctx context.Context, data *models.TimeSeries, params models.ProcessingParameters) (*models.TimeSeries, error)

	// ValidateParameters validates the processing parameters
	ValidateParameters(params models.ProcessingParameters) error

	// GetDefaultParameters returns default parameters for this processor
	GetDefaultParameters() models.ProcessingParameters

	// Close cleans up resources
	Close() error
}

// DataCleaner extends Processor for data cleaning operations
type DataCleaner interface {
	Processor

	// RemoveOutliers removes outliers from the data
	RemoveOutliers(ctx context.Context, data *models.TimeSeries, method string, threshold float64) (*models.TimeSeries, error)

	// FillMissingValues fills missing values in the data
	FillMissingValues(ctx context.Context, data *models.TimeSeries, method string) (*models.TimeSeries, error)

	// SmoothData applies smoothing to the data
	SmoothData(ctx context.Context, data *models.TimeSeries, method string, window int) (*models.TimeSeries, error)

	// GetSupportedMethods returns supported cleaning methods
	GetSupportedMethods() []string
}

// DataTransformer extends Processor for data transformation operations
type DataTransformer interface {
	Processor

	// Normalize normalizes the data to a specific range
	Normalize(ctx context.Context, data *models.TimeSeries, method string, targetMin, targetMax float64) (*models.TimeSeries, error)

	// Standardize standardizes the data (zero mean, unit variance)
	Standardize(ctx context.Context, data *models.TimeSeries) (*models.TimeSeries, error)

	// ApplyLog applies logarithmic transformation
	ApplyLog(ctx context.Context, data *models.TimeSeries, base float64) (*models.TimeSeries, error)

	// ApplyDifferencing applies differencing to make data stationary
	ApplyDifferencing(ctx context.Context, data *models.TimeSeries, order int) (*models.TimeSeries, error)

	// GetSupportedTransformations returns supported transformation methods
	GetSupportedTransformations() []string
}

// FeatureExtractor extends Processor for feature extraction
type FeatureExtractor interface {
	Processor

	// ExtractFeatures extracts features from time series data
	ExtractFeatures(ctx context.Context, data *models.TimeSeries, features []string) (*models.FeatureSet, error)

	// ExtractStatisticalFeatures extracts statistical features
	ExtractStatisticalFeatures(ctx context.Context, data *models.TimeSeries) (*models.StatisticalFeatures, error)

	// ExtractTemporalFeatures extracts temporal features
	ExtractTemporalFeatures(ctx context.Context, data *models.TimeSeries) (*models.TemporalFeatures, error)

	// ExtractFrequencyFeatures extracts frequency domain features
	ExtractFrequencyFeatures(ctx context.Context, data *models.TimeSeries) (*models.FrequencyFeatures, error)

	// GetAvailableFeatures returns available feature types
	GetAvailableFeatures() []string
}

// DataAggregator extends Processor for data aggregation
type DataAggregator interface {
	Processor

	// Aggregate aggregates data over specified time windows
	Aggregate(ctx context.Context, data *models.TimeSeries, window string, method string) (*models.TimeSeries, error)

	// Resample resamples data to a different frequency
	Resample(ctx context.Context, data *models.TimeSeries, frequency string, method string) (*models.TimeSeries, error)

	// GroupBy groups data by specified criteria
	GroupBy(ctx context.Context, data *models.TimeSeries, groupBy []string) (map[string]*models.TimeSeries, error)

	// GetSupportedAggregations returns supported aggregation methods
	GetSupportedAggregations() []string
}

// StreamProcessor extends Processor for streaming data processing
type StreamProcessor interface {
	Processor

	// ProcessStream processes streaming data
	ProcessStream(ctx context.Context, dataStream <-chan *models.DataPoint, params models.ProcessingParameters) (<-chan *models.DataPoint, <-chan error)

	// SetBufferSize sets the buffer size for streaming
	SetBufferSize(size int) error

	// GetBufferSize returns the current buffer size
	GetBufferSize() int

	// SetProcessingWindow sets the processing window for streaming
	SetProcessingWindow(window time.Duration) error
}

// BatchProcessor extends Processor for batch processing operations
type BatchProcessor interface {
	Processor

	// ProcessBatch processes multiple time series in a single operation
	ProcessBatch(ctx context.Context, batch []*models.TimeSeries, params models.ProcessingParameters) ([]*models.TimeSeries, error)

	// GetMaxBatchSize returns the maximum supported batch size
	GetMaxBatchSize() int

	// SupportsBatch returns true if batch processing is supported
	SupportsBatch() bool
}

// ProcessorFactory creates processor instances
type ProcessorFactory interface {
	// CreateProcessor creates a new processor instance
	CreateProcessor(processorType string) (Processor, error)

	// GetAvailableProcessors returns all available processor types
	GetAvailableProcessors() []string

	// RegisterProcessor registers a new processor type
	RegisterProcessor(processorType string, createFunc ProcessorCreateFunc) error

	// IsSupported checks if a processor type is supported
	IsSupported(processorType string) bool
}

// ProcessorCreateFunc is a function that creates a processor instance
type ProcessorCreateFunc func() Processor

// ProcessorRegistry manages available processors
type ProcessorRegistry interface {
	// Register registers a processor
	Register(processor Processor) error

	// Get retrieves a processor by type
	Get(processorType string) (Processor, error)

	// List returns all registered processors
	List() []Processor

	// Remove removes a processor from the registry
	Remove(processorType string) error

	// Count returns the number of registered processors
	Count() int
}

// ProcessingPipeline defines a sequence of processing steps
type ProcessingPipeline interface {
	// AddStep adds a processing step to the pipeline
	AddStep(processor Processor, params models.ProcessingParameters) error

	// RemoveStep removes a processing step from the pipeline
	RemoveStep(index int) error

	// Execute executes the entire pipeline
	Execute(ctx context.Context, data *models.TimeSeries) (*models.TimeSeries, error)

	// ExecutePartial executes the pipeline up to a specific step
	ExecutePartial(ctx context.Context, data *models.TimeSeries, endStep int) (*models.TimeSeries, error)

	// GetSteps returns all processing steps
	GetSteps() []ProcessingStep

	// Clear clears all processing steps
	Clear() error

	// Clone creates a copy of the pipeline
	Clone() ProcessingPipeline
}

// ProcessingStep represents a single step in a processing pipeline
type ProcessingStep struct {
	Processor  Processor                  `json:"-"`
	Type       string                     `json:"type"`
	Parameters models.ProcessingParameters `json:"parameters"`
	Enabled    bool                       `json:"enabled"`
	Name       string                     `json:"name,omitempty"`
}

// ProcessingMetrics provides metrics about processing performance
type ProcessingMetrics interface {
	// GetProcessingCount returns the total number of processing operations
	GetProcessingCount(processorType string) int64

	// GetAverageDuration returns the average processing duration
	GetAverageDuration(processorType string) time.Duration

	// GetSuccessRate returns the success rate (0.0 to 1.0)
	GetSuccessRate(processorType string) float64

	// GetLastError returns the last error for a processor type
	GetLastError(processorType string) error

	// Reset resets metrics for a processor type
	Reset(processorType string) error
}

// ProcessorPool manages a pool of processor instances
type ProcessorPool interface {
	// Get gets a processor from the pool
	Get(ctx context.Context, processorType string) (Processor, error)

	// Put returns a processor to the pool
	Put(processor Processor) error

	// Close closes the pool and all processors
	Close() error

	// Stats returns pool statistics
	Stats() ProcessorPoolStats
}

// ProcessorPoolStats contains processor pool statistics
type ProcessorPoolStats struct {
	ActiveProcessors int `json:"active_processors"`
	IdleProcessors   int `json:"idle_processors"`
	TotalCreated     int `json:"total_created"`
	TotalReused      int `json:"total_reused"`
}

// ProcessorConfig provides configuration for processors
type ProcessorConfig interface {
	// GetConfig returns configuration for a processor type
	GetConfig(processorType string) (*ProcessorConfiguration, error)

	// SetConfig sets configuration for a processor type
	SetConfig(processorType string, config *ProcessorConfiguration) error

	// GetDefaultConfig returns default configuration
	GetDefaultConfig(processorType string) (*ProcessorConfiguration, error)

	// ValidateConfig validates a configuration
	ValidateConfig(processorType string, config *ProcessorConfiguration) error
}

// ProcessorConfiguration contains processor-specific configuration
type ProcessorConfiguration struct {
	Enabled        bool                       `json:"enabled"`
	MaxConcurrency int                        `json:"max_concurrency"`
	Timeout        string                     `json:"timeout"`
	RetryAttempts  int                        `json:"retry_attempts"`
	BufferSize     int                        `json:"buffer_size,omitempty"`
	Parameters     models.ProcessingParameters `json:"parameters"`
	Resources      ProcessorResourceLimits    `json:"resources"`
	Metadata       map[string]interface{}     `json:"metadata,omitempty"`
}

// ProcessorResourceLimits defines resource limits for processors
type ProcessorResourceLimits struct {
	MaxMemory  string `json:"max_memory"`  // e.g., "1GB"
	MaxCPU     string `json:"max_cpu"`     // e.g., "2.0"
	MaxRuntime string `json:"max_runtime"` // e.g., "30m"
}

// DataQualityChecker extends Processor for data quality assessment
type DataQualityChecker interface {
	Processor

	// CheckQuality checks the quality of the data
	CheckQuality(ctx context.Context, data *models.TimeSeries) (*models.DataQualityReport, error)

	// CheckCompleteness checks data completeness
	CheckCompleteness(ctx context.Context, data *models.TimeSeries) (*models.CompletenessReport, error)

	// CheckConsistency checks data consistency
	CheckConsistency(ctx context.Context, data *models.TimeSeries) (*models.ConsistencyReport, error)

	// CheckAccuracy checks data accuracy (if reference data is available)
	CheckAccuracy(ctx context.Context, data, reference *models.TimeSeries) (*models.AccuracyReport, error)

	// GetQualityThresholds returns current quality thresholds
	GetQualityThresholds() *models.QualityThresholds

	// SetQualityThresholds sets quality thresholds
	SetQualityThresholds(thresholds *models.QualityThresholds) error
}

// AnomalyDetector extends Processor for anomaly detection
type AnomalyDetector interface {
	Processor

	// DetectAnomalies detects anomalies in time series data
	DetectAnomalies(ctx context.Context, data *models.TimeSeries, method string, params map[string]interface{}) (*models.AnomalyDetectionResult, error)

	// DetectPointAnomalies detects point anomalies
	DetectPointAnomalies(ctx context.Context, data *models.TimeSeries, threshold float64) (*models.PointAnomalies, error)

	// DetectContextualAnomalies detects contextual anomalies
	DetectContextualAnomalies(ctx context.Context, data *models.TimeSeries, window int) (*models.ContextualAnomalies, error)

	// DetectCollectiveAnomalies detects collective anomalies
	DetectCollectiveAnomalies(ctx context.Context, data *models.TimeSeries, minSize int) (*models.CollectiveAnomalies, error)

	// GetSupportedMethods returns supported anomaly detection methods
	GetSupportedMethods() []string
}

// ProcessingOrchestrator orchestrates multiple processors
type ProcessingOrchestrator interface {
	// CreatePipeline creates a new processing pipeline
	CreatePipeline(name string) (ProcessingPipeline, error)

	// GetPipeline gets an existing pipeline by name
	GetPipeline(name string) (ProcessingPipeline, error)

	// ListPipelines lists all available pipelines
	ListPipelines() []string

	// DeletePipeline deletes a pipeline
	DeletePipeline(name string) error

	// ExecutePipeline executes a named pipeline
	ExecutePipeline(ctx context.Context, pipelineName string, data *models.TimeSeries) (*models.TimeSeries, error)

	// RegisterProcessor registers a processor for use in pipelines
	RegisterProcessor(processor Processor) error

	// GetAvailableProcessors returns all available processors
	GetAvailableProcessors() []string
}

// ProcessingMonitor monitors processing operations
type ProcessingMonitor interface {
	// Start starts monitoring
	Start(ctx context.Context) error

	// Stop stops monitoring
	Stop() error

	// GetMetrics returns current metrics
	GetMetrics() *ProcessingMetrics

	// Subscribe subscribes to processing events
	Subscribe() (<-chan *ProcessingEvent, error)

	// SetThresholds sets alert thresholds
	SetThresholds(thresholds *ProcessingThresholds) error
}

// ProcessingEvent represents a processing event
type ProcessingEvent struct {
	Type         string                 `json:"type"`
	ProcessorType string                `json:"processor_type"`
	Timestamp    time.Time              `json:"timestamp"`
	Duration     time.Duration          `json:"duration"`
	DataSize     int64                  `json:"data_size"`
	Error        error                  `json:"error,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// ProcessingThresholds defines alert thresholds for processing monitoring
type ProcessingThresholds struct {
	MaxDuration       time.Duration `json:"max_duration"`
	MaxMemoryUsage    float64       `json:"max_memory_usage"`
	MaxErrorRate      float64       `json:"max_error_rate"`
	MinSuccessRate    float64       `json:"min_success_rate"`
	MaxDataSize       int64         `json:"max_data_size"`
}

// CustomProcessor allows for custom processing operations
type CustomProcessor interface {
	Processor

	// RegisterCustomOperation registers a custom processing operation
	RegisterCustomOperation(name string, operation CustomOperation) error

	// RemoveCustomOperation removes a custom operation
	RemoveCustomOperation(name string) error

	// GetCustomOperations returns all registered custom operations
	GetCustomOperations() map[string]CustomOperation

	// ExecuteCustomOperation executes a custom operation
	ExecuteCustomOperation(ctx context.Context, data *models.TimeSeries, operationName string, params map[string]interface{}) (*models.TimeSeries, error)
}

// CustomOperation defines a custom processing operation
type CustomOperation interface {
	// GetName returns the operation name
	GetName() string

	// GetDescription returns the operation description
	GetDescription() string

	// Execute executes the custom operation
	Execute(ctx context.Context, data *models.TimeSeries, params map[string]interface{}) (*models.TimeSeries, error)

	// ValidateParameters validates operation parameters
	ValidateParameters(params map[string]interface{}) error

	// GetParameterSchema returns the parameter schema
	GetParameterSchema() map[string]interface{}
}