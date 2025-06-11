package interfaces

import (
	"context"
	"time"

	"github.com/inferloop/tsiot/pkg/models"
)

// Validator defines the interface for data validation
type Validator interface {
	// GetType returns the validator type
	GetType() string

	// GetName returns a human-readable name for the validator
	GetName() string

	// GetDescription returns a description of the validator
	GetDescription() string

	// GetSupportedMetrics returns the metrics this validator supports
	GetSupportedMetrics() []string

	// ValidateParameters validates the validation parameters
	ValidateParameters(params models.ValidationParameters) error

	// Validate validates synthetic data against reference data
	Validate(ctx context.Context, req *models.ValidationRequest) (*models.ValidationResult, error)

	// GetDefaultParameters returns default parameters for this validator
	GetDefaultParameters() models.ValidationParameters

	// Close cleans up resources
	Close() error
}

// QualityValidator extends Validator for data quality assessment
type QualityValidator interface {
	Validator

	// CalculateQualityScore calculates an overall quality score (0.0 to 1.0)
	CalculateQualityScore(ctx context.Context, synthetic, reference *models.TimeSeries) (float64, error)

	// GetQualityMetrics returns detailed quality metrics
	GetQualityMetrics(ctx context.Context, synthetic, reference *models.TimeSeries) (*models.QualityMetrics, error)

	// SetQualityThreshold sets the minimum quality threshold
	SetQualityThreshold(threshold float64) error

	// GetQualityThreshold returns the current quality threshold
	GetQualityThreshold() float64
}

// StatisticalValidator extends Validator for statistical validation
type StatisticalValidator interface {
	Validator

	// RunStatisticalTests runs statistical tests comparing datasets
	RunStatisticalTests(ctx context.Context, synthetic, reference *models.TimeSeries, tests []string) (*models.StatisticalTestResults, error)

	// GetSupportedTests returns supported statistical tests
	GetSupportedTests() []string

	// SetSignificanceLevel sets the significance level for tests
	SetSignificanceLevel(alpha float64) error

	// GetSignificanceLevel returns the current significance level
	GetSignificanceLevel() float64
}

// PrivacyValidator extends Validator for privacy validation
type PrivacyValidator interface {
	Validator

	// ValidatePrivacy validates privacy protection of synthetic data
	ValidatePrivacy(ctx context.Context, synthetic, reference *models.TimeSeries, params models.PrivacyParameters) (*models.PrivacyValidationResult, error)

	// CalculatePrivacyRisk calculates privacy risk metrics
	CalculatePrivacyRisk(ctx context.Context, synthetic, reference *models.TimeSeries) (*models.PrivacyRisk, error)

	// CheckReidentificationRisk checks for reidentification risks
	CheckReidentificationRisk(ctx context.Context, synthetic, reference *models.TimeSeries) (*models.ReidentificationRisk, error)

	// GetSupportedPrivacyTechniques returns supported privacy techniques
	GetSupportedPrivacyTechniques() []string
}

// TemporalValidator extends Validator for temporal pattern validation
type TemporalValidator interface {
	Validator

	// ValidateTemporalPatterns validates temporal patterns in synthetic data
	ValidateTemporalPatterns(ctx context.Context, synthetic, reference *models.TimeSeries) (*models.TemporalValidationResult, error)

	// CheckSeasonality checks for proper seasonality preservation
	CheckSeasonality(ctx context.Context, synthetic, reference *models.TimeSeries) (*models.SeasonalityResult, error)

	// CheckTrends checks for trend preservation
	CheckTrends(ctx context.Context, synthetic, reference *models.TimeSeries) (*models.TrendResult, error)

	// CheckAutocorrelation checks autocorrelation patterns
	CheckAutocorrelation(ctx context.Context, synthetic, reference *models.TimeSeries, lags []int) (*models.AutocorrelationResult, error)
}

// BatchValidator extends Validator for batch validation operations
type BatchValidator interface {
	Validator

	// ValidateBatch validates multiple time series in a single operation
	ValidateBatch(ctx context.Context, requests []*models.ValidationRequest) ([]*models.ValidationResult, error)

	// GetMaxBatchSize returns the maximum supported batch size
	GetMaxBatchSize() int

	// SupportsBatch returns true if batch processing is supported
	SupportsBatch() bool
}

// ValidatorFactory creates validator instances
type ValidatorFactory interface {
	// CreateValidator creates a new validator instance
	CreateValidator(validatorType string) (Validator, error)

	// GetAvailableValidators returns all available validator types
	GetAvailableValidators() []string

	// RegisterValidator registers a new validator type
	RegisterValidator(validatorType string, createFunc ValidatorCreateFunc) error

	// IsSupported checks if a validator type is supported
	IsSupported(validatorType string) bool
}

// ValidatorCreateFunc is a function that creates a validator instance
type ValidatorCreateFunc func() Validator

// ValidatorRegistry manages available validators
type ValidatorRegistry interface {
	// Register registers a validator
	Register(validator Validator) error

	// Get retrieves a validator by type
	Get(validatorType string) (Validator, error)

	// List returns all registered validators
	List() []Validator

	// Remove removes a validator from the registry
	Remove(validatorType string) error

	// Count returns the number of registered validators
	Count() int
}

// ValidationMetrics provides metrics about validation performance
type ValidationMetrics interface {
	// GetValidationCount returns the total number of validations
	GetValidationCount(validatorType string) int64

	// GetAverageDuration returns the average validation duration
	GetAverageDuration(validatorType string) time.Duration

	// GetSuccessRate returns the success rate (0.0 to 1.0)
	GetSuccessRate(validatorType string) float64

	// GetLastError returns the last error for a validator type
	GetLastError(validatorType string) error

	// Reset resets metrics for a validator type
	Reset(validatorType string) error
}

// ValidationPool manages a pool of validator instances
type ValidationPool interface {
	// Get gets a validator from the pool
	Get(ctx context.Context, validatorType string) (Validator, error)

	// Put returns a validator to the pool
	Put(validator Validator) error

	// Close closes the pool and all validators
	Close() error

	// Stats returns pool statistics
	Stats() ValidationPoolStats
}

// ValidationPoolStats contains validation pool statistics
type ValidationPoolStats struct {
	ActiveValidators int `json:"active_validators"`
	IdleValidators   int `json:"idle_validators"`
	TotalCreated     int `json:"total_created"`
	TotalReused      int `json:"total_reused"`
}

// ValidationReporter generates validation reports
type ValidationReporter interface {
	// GenerateReport generates a validation report
	GenerateReport(ctx context.Context, results []*models.ValidationResult, config *models.ReportConfig) (*models.ValidationReport, error)

	// GenerateQualityReport generates a quality-focused report
	GenerateQualityReport(ctx context.Context, qualityMetrics *models.QualityMetrics, config *models.ReportConfig) (*models.QualityReport, error)

	// GetSupportedFormats returns supported report formats
	GetSupportedFormats() []string

	// ExportReport exports a report to a specific format
	ExportReport(ctx context.Context, report interface{}, format string, destination string) error
}

// ValidationConfig provides configuration for validators
type ValidationConfig interface {
	// GetConfig returns configuration for a validator type
	GetConfig(validatorType string) (*ValidatorConfiguration, error)

	// SetConfig sets configuration for a validator type
	SetConfig(validatorType string, config *ValidatorConfiguration) error

	// GetDefaultConfig returns default configuration
	GetDefaultConfig(validatorType string) (*ValidatorConfiguration, error)

	// ValidateConfig validates a configuration
	ValidateConfig(validatorType string, config *ValidatorConfiguration) error
}

// ValidatorConfiguration contains validator-specific configuration
type ValidatorConfiguration struct {
	Enabled          bool                       `json:"enabled"`
	QualityThreshold float64                    `json:"quality_threshold"`
	StatisticalAlpha float64                    `json:"statistical_alpha"`
	PrivacyBudget    float64                    `json:"privacy_budget,omitempty"`
	Metrics          []string                   `json:"metrics"`
	Tests            []string                   `json:"tests,omitempty"`
	Timeout          string                     `json:"timeout"`
	RetryAttempts    int                        `json:"retry_attempts"`
	Parameters       models.ValidationParameters `json:"parameters"`
	Metadata         map[string]interface{}     `json:"metadata,omitempty"`
}

// ValidationEngine orchestrates multiple validators
type ValidationEngine interface {
	// AddValidator adds a validator to the engine
	AddValidator(validator Validator) error

	// RemoveValidator removes a validator from the engine
	RemoveValidator(validatorType string) error

	// ValidateWithMultiple validates using multiple validators
	ValidateWithMultiple(ctx context.Context, req *models.ValidationRequest, validatorTypes []string) ([]*models.ValidationResult, error)

	// GetAggregatedResult aggregates results from multiple validators
	GetAggregatedResult(results []*models.ValidationResult) (*models.AggregatedValidationResult, error)

	// GetAvailableValidators returns available validators in the engine
	GetAvailableValidators() []string

	// Close closes the engine and all validators
	Close() error
}

// ValidationMonitor monitors validation operations
type ValidationMonitor interface {
	// Start starts monitoring
	Start(ctx context.Context) error

	// Stop stops monitoring
	Stop() error

	// GetMetrics returns current metrics
	GetMetrics() *ValidationMetrics

	// Subscribe subscribes to validation events
	Subscribe() (<-chan *ValidationEvent, error)

	// SetThresholds sets alert thresholds
	SetThresholds(thresholds *ValidationThresholds) error
}

// ValidationEvent represents a validation event
type ValidationEvent struct {
	Type        string                 `json:"type"`
	ValidatorType string               `json:"validator_type"`
	Timestamp   time.Time              `json:"timestamp"`
	Duration    time.Duration          `json:"duration"`
	Result      *models.ValidationResult `json:"result,omitempty"`
	Error       error                  `json:"error,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// ValidationThresholds defines alert thresholds for validation monitoring
type ValidationThresholds struct {
	MinQualityScore   float64       `json:"min_quality_score"`
	MaxDuration       time.Duration `json:"max_duration"`
	MaxErrorRate      float64       `json:"max_error_rate"`
	MinSuccessRate    float64       `json:"min_success_rate"`
}

// RealTimeValidator validates data in real-time streaming scenarios
type RealTimeValidator interface {
	Validator

	// ValidateStream validates streaming data
	ValidateStream(ctx context.Context, dataStream <-chan *models.DataPoint, referenceData *models.TimeSeries) (<-chan *models.ValidationResult, <-chan error)

	// SetValidationWindow sets the time window for validation
	SetValidationWindow(window time.Duration) error

	// GetValidationWindow returns the current validation window
	GetValidationWindow() time.Duration

	// SetUpdateInterval sets how often validation results are updated
	SetUpdateInterval(interval time.Duration) error
}

// CustomMetricValidator allows for custom validation metrics
type CustomMetricValidator interface {
	Validator

	// RegisterCustomMetric registers a custom validation metric
	RegisterCustomMetric(name string, metric CustomMetric) error

	// RemoveCustomMetric removes a custom metric
	RemoveCustomMetric(name string) error

	// GetCustomMetrics returns all registered custom metrics
	GetCustomMetrics() map[string]CustomMetric

	// ValidateWithCustomMetrics validates using custom metrics
	ValidateWithCustomMetrics(ctx context.Context, synthetic, reference *models.TimeSeries, metrics []string) (*models.CustomValidationResult, error)
}

// CustomMetric defines a custom validation metric
type CustomMetric interface {
	// GetName returns the metric name
	GetName() string

	// GetDescription returns the metric description
	GetDescription() string

	// Calculate calculates the metric value
	Calculate(synthetic, reference *models.TimeSeries) (float64, error)

	// GetRange returns the expected range of values
	GetRange() (min, max float64)

	// IsBetterHigher returns true if higher values are better
	IsBetterHigher() bool
}