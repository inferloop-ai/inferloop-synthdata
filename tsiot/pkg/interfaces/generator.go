package interfaces

import (
	"context"
	"io"
	"time"

	"github.com/inferloop/tsiot/pkg/models"
)

// Generator defines the interface for synthetic data generators
type Generator interface {
	// GetType returns the generator type
	GetType() models.GeneratorType

	// GetName returns a human-readable name for the generator
	GetName() string

	// GetDescription returns a description of the generator
	GetDescription() string

	// GetSupportedSensorTypes returns the sensor types this generator supports
	GetSupportedSensorTypes() []models.SensorType

	// ValidateParameters validates the generation parameters
	ValidateParameters(params models.GenerationParameters) error

	// Generate generates synthetic data based on the request
	Generate(ctx context.Context, req *models.GenerationRequest) (*models.GenerationResult, error)

	// Train trains the generator with reference data (if applicable)
	Train(ctx context.Context, data *models.TimeSeries, params models.GenerationParameters) error

	// IsTrainable returns true if the generator requires/supports training
	IsTrainable() bool

	// GetDefaultParameters returns default parameters for this generator
	GetDefaultParameters() models.GenerationParameters

	// EstimateDuration estimates how long generation will take
	EstimateDuration(req *models.GenerationRequest) (time.Duration, error)

	// Cancel cancels an ongoing generation
	Cancel(ctx context.Context, requestID string) error

	// GetProgress returns the progress of an ongoing generation (0.0 to 1.0)
	GetProgress(requestID string) (float64, error)

	// Close cleans up resources
	Close() error
}

// TrainableGenerator extends Generator for generators that support training
type TrainableGenerator interface {
	Generator

	// LoadModel loads a pre-trained model
	LoadModel(ctx context.Context, modelPath string) error

	// SaveModel saves the trained model
	SaveModel(ctx context.Context, modelPath string) error

	// GetModelInfo returns information about the current model
	GetModelInfo() (*ModelInfo, error)

	// IsModelLoaded returns true if a model is currently loaded
	IsModelLoaded() bool
}

// StreamingGenerator extends Generator for generators that support streaming output
type StreamingGenerator interface {
	Generator

	// GenerateStream generates data as a stream
	GenerateStream(ctx context.Context, req *models.GenerationRequest) (<-chan *models.DataPoint, <-chan error)

	// GetStreamBufferSize returns the buffer size for streaming
	GetStreamBufferSize() int

	// SetStreamBufferSize sets the buffer size for streaming
	SetStreamBufferSize(size int)
}

// BatchGenerator extends Generator for generators that support batch processing
type BatchGenerator interface {
	Generator

	// GenerateBatch generates multiple time series in a single operation
	GenerateBatch(ctx context.Context, requests []*models.GenerationRequest) ([]*models.GenerationResult, error)

	// GetMaxBatchSize returns the maximum supported batch size
	GetMaxBatchSize() int

	// SupportsBatch returns true if batch processing is supported
	SupportsBatch() bool
}

// ModelInfo contains information about a trained model
type ModelInfo struct {
	Type         string                 `json:"type"`
	Version      string                 `json:"version"`
	TrainedAt    string                 `json:"trained_at"`
	TrainingSize int64                  `json:"training_size"`
	Accuracy     float64                `json:"accuracy,omitempty"`
	Parameters   map[string]interface{} `json:"parameters"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// GeneratorFactory creates generator instances
type GeneratorFactory interface {
	// CreateGenerator creates a new generator instance
	CreateGenerator(generatorType models.GeneratorType) (Generator, error)

	// GetAvailableGenerators returns all available generator types
	GetAvailableGenerators() []models.GeneratorType

	// RegisterGenerator registers a new generator type
	RegisterGenerator(generatorType models.GeneratorType, createFunc GeneratorCreateFunc) error

	// IsSupported checks if a generator type is supported
	IsSupported(generatorType models.GeneratorType) bool
}

// GeneratorCreateFunc is a function that creates a generator instance
type GeneratorCreateFunc func() Generator

// GeneratorRegistry manages available generators
type GeneratorRegistry interface {
	// Register registers a generator
	Register(generator Generator) error

	// Get retrieves a generator by type
	Get(generatorType models.GeneratorType) (Generator, error)

	// List returns all registered generators
	List() []Generator

	// Remove removes a generator from the registry
	Remove(generatorType models.GeneratorType) error

	// Count returns the number of registered generators
	Count() int
}

// ParameterValidator validates generation parameters
type ParameterValidator interface {
	// Validate validates parameters for a specific generator type
	Validate(generatorType models.GeneratorType, params models.GenerationParameters) error

	// GetConstraints returns parameter constraints for a generator type
	GetConstraints(generatorType models.GeneratorType) (*ParameterConstraints, error)

	// SuggestParameters suggests parameters based on data characteristics
	SuggestParameters(generatorType models.GeneratorType, data *models.TimeSeries) (models.GenerationParameters, error)
}

// ParameterConstraints defines constraints for generator parameters
type ParameterConstraints struct {
	Epochs      *IntConstraint    `json:"epochs,omitempty"`
	BatchSize   *IntConstraint    `json:"batch_size,omitempty"`
	LearningRate *FloatConstraint `json:"learning_rate,omitempty"`
	HiddenDim   *IntConstraint    `json:"hidden_dim,omitempty"`
	NumLayers   *IntConstraint    `json:"num_layers,omitempty"`
	Custom      map[string]interface{} `json:"custom,omitempty"`
}

// IntConstraint defines constraints for integer parameters
type IntConstraint struct {
	Min     *int   `json:"min,omitempty"`
	Max     *int   `json:"max,omitempty"`
	Default int    `json:"default"`
	Values  []int  `json:"values,omitempty"` // allowed values
}

// FloatConstraint defines constraints for float parameters
type FloatConstraint struct {
	Min     *float64   `json:"min,omitempty"`
	Max     *float64   `json:"max,omitempty"`
	Default float64    `json:"default"`
	Values  []float64  `json:"values,omitempty"` // allowed values
}

// StringConstraint defines constraints for string parameters
type StringConstraint struct {
	Default string   `json:"default"`
	Values  []string `json:"values,omitempty"` // allowed values
	Pattern string   `json:"pattern,omitempty"` // regex pattern
}

// DataPreprocessor preprocesses data before generation
type DataPreprocessor interface {
	// Preprocess preprocesses the input data
	Preprocess(ctx context.Context, data *models.TimeSeries, params models.GenerationParameters) (*models.TimeSeries, error)

	// GetPreprocessingSteps returns the preprocessing steps that will be applied
	GetPreprocessingSteps(params models.GenerationParameters) []string

	// SupportsStreaming returns true if streaming preprocessing is supported
	SupportsStreaming() bool
}

// PostProcessor processes generated data
type PostProcessor interface {
	// Process applies post-processing to generated data
	Process(ctx context.Context, data *models.TimeSeries, params models.PostProcessingParams) (*models.TimeSeries, error)

	// GetProcessingSteps returns the post-processing steps that will be applied
	GetProcessingSteps(params models.PostProcessingParams) []string

	// SupportsInPlace returns true if in-place processing is supported
	SupportsInPlace() bool
}

// GeneratorMetrics provides metrics about generator performance
type GeneratorMetrics interface {
	// GetGenerationCount returns the total number of generations
	GetGenerationCount(generatorType models.GeneratorType) int64

	// GetAverageDuration returns the average generation duration
	GetAverageDuration(generatorType models.GeneratorType) float64

	// GetSuccessRate returns the success rate (0.0 to 1.0)
	GetSuccessRate(generatorType models.GeneratorType) float64

	// GetLastError returns the last error for a generator type
	GetLastError(generatorType models.GeneratorType) error

	// Reset resets metrics for a generator type
	Reset(generatorType models.GeneratorType) error
}

// GeneratorPool manages a pool of generator instances
type GeneratorPool interface {
	// Get gets a generator from the pool
	Get(ctx context.Context, generatorType models.GeneratorType) (Generator, error)

	// Put returns a generator to the pool
	Put(generator Generator) error

	// Close closes the pool and all generators
	Close() error

	// Stats returns pool statistics
	Stats() PoolStats
}

// PoolStats contains pool statistics
type PoolStats struct {
	ActiveConnections int `json:"active_connections"`
	IdleConnections   int `json:"idle_connections"`
	TotalCreated      int `json:"total_created"`
	TotalReused       int `json:"total_reused"`
}

// OutputWriter writes generated data to various formats
type OutputWriter interface {
	// Write writes time series data
	Write(ctx context.Context, data *models.TimeSeries, config models.OutputConfiguration) error

	// WriteStream writes streaming data
	WriteStream(ctx context.Context, stream <-chan *models.DataPoint, config models.OutputConfiguration) error

	// SupportedFormats returns supported output formats
	SupportedFormats() []string

	// GetWriter returns an io.Writer for the specified format
	GetWriter(format string, config models.OutputConfiguration) (io.Writer, error)

	// Close closes the writer and any open resources
	Close() error
}

// GeneratorConfig provides configuration for generators
type GeneratorConfig interface {
	// GetConfig returns configuration for a generator type
	GetConfig(generatorType models.GeneratorType) (*GeneratorConfiguration, error)

	// SetConfig sets configuration for a generator type
	SetConfig(generatorType models.GeneratorType, config *GeneratorConfiguration) error

	// GetDefaultConfig returns default configuration
	GetDefaultConfig(generatorType models.GeneratorType) (*GeneratorConfiguration, error)

	// ValidateConfig validates a configuration
	ValidateConfig(generatorType models.GeneratorType, config *GeneratorConfiguration) error
}

// GeneratorConfiguration contains generator-specific configuration
type GeneratorConfiguration struct {
	Enabled          bool                   `json:"enabled"`
	MaxConcurrency   int                    `json:"max_concurrency"`
	Timeout          string                 `json:"timeout"`
	RetryAttempts    int                    `json:"retry_attempts"`
	ModelPath        string                 `json:"model_path,omitempty"`
	Parameters       models.GenerationParameters `json:"parameters"`
	Resources        ResourceLimits         `json:"resources"`
	Metadata         map[string]interface{} `json:"metadata,omitempty"`
}

// ResourceLimits defines resource limits for generators
type ResourceLimits struct {
	MaxMemory  string `json:"max_memory"`  // e.g., "1GB"
	MaxCPU     string `json:"max_cpu"`     // e.g., "2.0"
	MaxGPU     string `json:"max_gpu,omitempty"`
	MaxDisk    string `json:"max_disk"`    // e.g., "10GB"
	MaxRuntime string `json:"max_runtime"` // e.g., "30m"
}