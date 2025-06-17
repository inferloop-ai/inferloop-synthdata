package errors

import (
	"errors"
	"fmt"
)

// Common application errors
var (
	// Validation errors
	ErrInvalidTimeSeriesName = errors.New("invalid time series name")
	ErrInvalidSensorType     = errors.New("invalid sensor type")
	ErrInvalidTimeRange      = errors.New("invalid time range: start time must be before end time")
	ErrInvalidGenerator      = errors.New("invalid generator type")
	ErrInvalidFormat         = errors.New("invalid output format")
	ErrInvalidDestination    = errors.New("invalid output destination")
	ErrInvalidInputData      = errors.New("invalid input data")
	ErrInvalidThreshold      = errors.New("invalid threshold: must be between 0 and 1")
	ErrNoFeatures           = errors.New("no features specified")
	ErrNoMetrics            = errors.New("no metrics specified")

	// Generation errors
	ErrGenerationFailed       = errors.New("data generation failed")
	ErrGeneratorNotFound      = errors.New("generator not found")
	ErrGeneratorNotSupported  = errors.New("generator not supported")
	ErrInvalidParameters      = errors.New("invalid generation parameters")
	ErrInsufficientData       = errors.New("insufficient training data")
	ErrModelLoadFailed        = errors.New("failed to load model")
	ErrModelTrainingFailed    = errors.New("model training failed")
	ErrGenerationTimeout      = errors.New("generation timeout")
	ErrGenerationCancelled    = errors.New("generation cancelled")

	// Storage errors
	ErrStorageNotFound        = errors.New("storage backend not found")
	ErrStorageConnectionFailed = errors.New("storage connection failed")
	ErrStorageWriteFailed     = errors.New("storage write failed")
	ErrStorageReadFailed      = errors.New("storage read failed")
	ErrStorageTimeout         = errors.New("storage operation timeout")
	ErrDataNotFound           = errors.New("data not found")
	ErrDuplicateData          = errors.New("duplicate data")

	// Validation errors
	ErrValidationFailed       = errors.New("validation failed")
	ErrValidatorNotFound      = errors.New("validator not found")
	ErrInvalidMetric          = errors.New("invalid metric")
	ErrInvalidTest            = errors.New("invalid statistical test")
	ErrQualityBelowThreshold  = errors.New("data quality below threshold")

	// Network/Protocol errors
	ErrConnectionFailed       = errors.New("connection failed")
	ErrNetworkTimeout         = errors.New("network timeout")
	ErrProtocolError          = errors.New("protocol error")
	ErrMCPError              = errors.New("MCP protocol error")
	ErrMessageTooLarge        = errors.New("message too large")
	ErrInvalidMessage         = errors.New("invalid message format")

	// Authentication/Authorization errors
	ErrUnauthorized          = errors.New("unauthorized")
	ErrForbidden             = errors.New("forbidden")
	ErrInvalidCredentials    = errors.New("invalid credentials")
	ErrTokenExpired          = errors.New("token expired")
	ErrInvalidToken          = errors.New("invalid token")

	// Rate limiting errors
	ErrRateLimitExceeded     = errors.New("rate limit exceeded")
	ErrQuotaExceeded         = errors.New("quota exceeded")
	ErrTooManyRequests       = errors.New("too many requests")

	// Configuration errors
	ErrInvalidConfiguration  = errors.New("invalid configuration")
	ErrMissingConfiguration  = errors.New("missing configuration")
	ErrConfigurationLoad     = errors.New("failed to load configuration")

	// Resource errors
	ErrInsufficientMemory    = errors.New("insufficient memory")
	ErrInsufficientDisk      = errors.New("insufficient disk space")
	ErrResourceExhausted     = errors.New("resource exhausted")
	ErrConcurrencyLimit      = errors.New("concurrency limit exceeded")

	// Privacy errors
	ErrPrivacyViolation      = errors.New("privacy violation")
	ErrInsufficientPrivacy   = errors.New("insufficient privacy protection")
	ErrPrivacyBudgetExceeded = errors.New("privacy budget exceeded")
	ErrReidentificationRisk  = errors.New("reidentification risk too high")

	// Agent errors
	ErrAgentNotFound         = errors.New("agent not found")
	ErrAgentNotAvailable     = errors.New("agent not available")
	ErrAgentError            = errors.New("agent error")
	ErrCapabilityNotSupported = errors.New("capability not supported")

	// Job/Task errors
	ErrJobNotFound           = errors.New("job not found")
	ErrJobFailed             = errors.New("job failed")
	ErrJobTimeout            = errors.New("job timeout")
	ErrJobCancelled          = errors.New("job cancelled")
	ErrJobAlreadyExists      = errors.New("job already exists")

	// Internal errors
	ErrInternal              = errors.New("internal error")
	ErrNotImplemented        = errors.New("not implemented")
	ErrUnavailable           = errors.New("service unavailable")
	ErrMaintenance           = errors.New("service under maintenance")
)

// ErrorType represents different categories of errors
type ErrorType string

const (
	ErrorTypeValidation     ErrorType = "validation"
	ErrorTypeGeneration     ErrorType = "generation"
	ErrorTypeStorage        ErrorType = "storage"
	ErrorTypeValidationJob  ErrorType = "validation_job"
	ErrorTypeNetwork        ErrorType = "network"
	ErrorTypeAuth           ErrorType = "auth"
	ErrorTypeRateLimit      ErrorType = "rate_limit"
	ErrorTypeConfiguration  ErrorType = "configuration"
	ErrorTypeResource       ErrorType = "resource"
	ErrorTypePrivacy        ErrorType = "privacy"
	ErrorTypeAgent          ErrorType = "agent"
	ErrorTypeJob            ErrorType = "job"
	ErrorTypeInternal       ErrorType = "internal"
)

// AppError represents an application-specific error with additional context
type AppError struct {
	Type        ErrorType              `json:"type"`
	Code        string                 `json:"code"`
	Message     string                 `json:"message"`
	Details     string                 `json:"details,omitempty"`
	Cause       error                  `json:"-"`
	Context     map[string]interface{} `json:"context,omitempty"`
	Retryable   bool                   `json:"retryable"`
	HTTPStatus  int                    `json:"-"`
}

// Error implements the error interface
func (e *AppError) Error() string {
	if e.Details != "" {
		return fmt.Sprintf("%s: %s - %s", e.Code, e.Message, e.Details)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

// Unwrap returns the underlying error
func (e *AppError) Unwrap() error {
	return e.Cause
}

// Is checks if the error matches the target
func (e *AppError) Is(target error) bool {
	t, ok := target.(*AppError)
	if !ok {
		return false
	}
	return e.Type == t.Type && e.Code == t.Code
}

// WithContext adds context to the error
func (e *AppError) WithContext(key string, value interface{}) *AppError {
	if e.Context == nil {
		e.Context = make(map[string]interface{})
	}
	e.Context[key] = value
	return e
}

// WithDetails adds details to the error
func (e *AppError) WithDetails(details string) *AppError {
	e.Details = details
	return e
}

// NewAppError creates a new application error
func NewAppError(errType ErrorType, code, message string) *AppError {
	return &AppError{
		Type:       errType,
		Code:       code,
		Message:    message,
		Retryable:  false,
		HTTPStatus: getDefaultHTTPStatus(errType),
	}
}

// WrapError wraps an existing error with application context
func WrapError(err error, errType ErrorType, code, message string) *AppError {
	return &AppError{
		Type:       errType,
		Code:       code,
		Message:    message,
		Cause:      err,
		Retryable:  isRetryable(err),
		HTTPStatus: getDefaultHTTPStatus(errType),
	}
}

// NewValidationError creates a validation error
func NewValidationError(code, message string) *AppError {
	return NewAppError(ErrorTypeValidation, code, message)
}

// NewGenerationError creates a generation error
func NewGenerationError(code, message string) *AppError {
	return NewAppError(ErrorTypeGeneration, code, message)
}

// NewStorageError creates a storage error
func NewStorageError(code, message string) *AppError {
	return NewAppError(ErrorTypeStorage, code, message)
}

// NewNetworkError creates a network error
func NewNetworkError(code, message string) *AppError {
	return &AppError{
		Type:       ErrorTypeNetwork,
		Code:       code,
		Message:    message,
		Retryable:  true,
		HTTPStatus: 503,
	}
}

// NewAuthError creates an authentication/authorization error
func NewAuthError(code, message string) *AppError {
	return &AppError{
		Type:       ErrorTypeAuth,
		Code:       code,
		Message:    message,
		Retryable:  false,
		HTTPStatus: 401,
	}
}

// NewRateLimitError creates a rate limit error
func NewRateLimitError(message string) *AppError {
	return &AppError{
		Type:       ErrorTypeRateLimit,
		Code:       "RATE_LIMIT_EXCEEDED",
		Message:    message,
		Retryable:  true,
		HTTPStatus: 429,
	}
}

// NewInternalError creates an internal error
func NewInternalError(message string) *AppError {
	return &AppError{
		Type:       ErrorTypeInternal,
		Code:       "INTERNAL_ERROR",
		Message:    message,
		Retryable:  false,
		HTTPStatus: 500,
	}
}

// getDefaultHTTPStatus returns the default HTTP status for an error type
func getDefaultHTTPStatus(errType ErrorType) int {
	switch errType {
	case ErrorTypeValidation:
		return 400
	case ErrorTypeAuth:
		return 401
	case ErrorTypeResource, ErrorTypePrivacy:
		return 403
	case ErrorTypeStorage, ErrorTypeAgent, ErrorTypeJob:
		return 404
	case ErrorTypeRateLimit:
		return 429
	case ErrorTypeInternal, ErrorTypeGeneration:
		return 500
	case ErrorTypeNetwork, ErrorTypeConfiguration:
		return 503
	default:
		return 500
	}
}

// isRetryable determines if an error is retryable
func isRetryable(err error) bool {
	if err == nil {
		return false
	}

	// Check for specific retryable errors
	switch {
	case errors.Is(err, ErrNetworkTimeout):
		return true
	case errors.Is(err, ErrConnectionFailed):
		return true
	case errors.Is(err, ErrStorageTimeout):
		return true
	case errors.Is(err, ErrRateLimitExceeded):
		return true
	case errors.Is(err, ErrUnavailable):
		return true
	case errors.Is(err, ErrResourceExhausted):
		return true
	default:
		return false
	}
}

// ErrorResponse represents an error response for APIs
type ErrorResponse struct {
	Error     *AppError `json:"error"`
	RequestID string    `json:"request_id,omitempty"`
	Timestamp string    `json:"timestamp"`
	Path      string    `json:"path,omitempty"`
}

// ValidationErrorDetail represents detailed validation error information
type ValidationErrorDetail struct {
	Field   string      `json:"field"`
	Value   interface{} `json:"value,omitempty"`
	Message string      `json:"message"`
	Code    string      `json:"code"`
}

// ValidationErrors represents multiple validation errors
type ValidationErrors struct {
	Message string                   `json:"message"`
	Errors  []ValidationErrorDetail  `json:"errors"`
}

// Error implements the error interface for ValidationErrors
func (ve *ValidationErrors) Error() string {
	return ve.Message
}

// Add adds a validation error
func (ve *ValidationErrors) Add(field, code, message string, value interface{}) {
	ve.Errors = append(ve.Errors, ValidationErrorDetail{
		Field:   field,
		Value:   value,
		Message: message,
		Code:    code,
	})
}

// HasErrors checks if there are any validation errors
func (ve *ValidationErrors) HasErrors() bool {
	return len(ve.Errors) > 0
}

// NewValidationErrors creates a new ValidationErrors instance
func NewValidationErrors() *ValidationErrors {
	return &ValidationErrors{
		Message: "Validation failed",
		Errors:  make([]ValidationErrorDetail, 0),
	}
}

// Error codes for different error scenarios
const (
	// Validation error codes
	CodeInvalidInput        = "INVALID_INPUT"
	CodeMissingField        = "MISSING_FIELD"
	CodeInvalidFormat       = "INVALID_FORMAT"
	CodeOutOfRange          = "OUT_OF_RANGE"
	CodeInvalidTimeRange    = "INVALID_TIME_RANGE"
	CodeInvalidSensorType   = "INVALID_SENSOR_TYPE"
	CodeInvalidGenerator    = "INVALID_GENERATOR"

	// Generation error codes
	CodeGenerationFailed    = "GENERATION_FAILED"
	CodeModelNotFound       = "MODEL_NOT_FOUND"
	CodeModelLoadFailed     = "MODEL_LOAD_FAILED"
	CodeTrainingFailed      = "TRAINING_FAILED"
	CodeInsufficientData    = "INSUFFICIENT_DATA"
	CodeGenerationTimeout   = "GENERATION_TIMEOUT"

	// Storage error codes
	CodeStorageError        = "STORAGE_ERROR"
	CodeConnectionFailed    = "CONNECTION_FAILED"
	CodeDataNotFound        = "DATA_NOT_FOUND"
	CodeWriteFailed         = "WRITE_FAILED"
	CodeReadFailed          = "READ_FAILED"
	CodeStorageTimeout      = "STORAGE_TIMEOUT"

	// Network error codes
	CodeNetworkError        = "NETWORK_ERROR"
	CodeTimeout             = "TIMEOUT"
	CodeProtocolError       = "PROTOCOL_ERROR"
	CodeMessageTooLarge     = "MESSAGE_TOO_LARGE"

	// Authentication error codes
	CodeUnauthorized        = "UNAUTHORIZED"
	CodeForbidden           = "FORBIDDEN"
	CodeInvalidCredentials  = "INVALID_CREDENTIALS"
	CodeTokenExpired        = "TOKEN_EXPIRED"
	CodeInvalidToken        = "INVALID_TOKEN"

	// Rate limiting error codes
	CodeRateLimitExceeded   = "RATE_LIMIT_EXCEEDED"
	CodeQuotaExceeded       = "QUOTA_EXCEEDED"
	CodeTooManyRequests     = "TOO_MANY_REQUESTS"

	// Resource error codes
	CodeInsufficientMemory  = "INSUFFICIENT_MEMORY"
	CodeInsufficientDisk    = "INSUFFICIENT_DISK"
	CodeResourceExhausted   = "RESOURCE_EXHAUSTED"
	CodeConcurrencyLimit    = "CONCURRENCY_LIMIT"

	// Privacy error codes
	CodePrivacyViolation    = "PRIVACY_VIOLATION"
	CodePrivacyBudgetExceeded = "PRIVACY_BUDGET_EXCEEDED"
	CodeReidentificationRisk = "REIDENTIFICATION_RISK"

	// Agent error codes
	CodeAgentNotFound       = "AGENT_NOT_FOUND"
	CodeAgentUnavailable    = "AGENT_UNAVAILABLE"
	CodeCapabilityNotSupported = "CAPABILITY_NOT_SUPPORTED"

	// Job error codes
	CodeJobNotFound         = "JOB_NOT_FOUND"
	CodeJobFailed           = "JOB_FAILED"
	CodeJobTimeout          = "JOB_TIMEOUT"
	CodeJobCancelled        = "JOB_CANCELLED"

	// Internal error codes
	CodeInternalError       = "INTERNAL_ERROR"
	CodeNotImplemented      = "NOT_IMPLEMENTED"
	CodeServiceUnavailable  = "SERVICE_UNAVAILABLE"
	CodeMaintenance         = "MAINTENANCE"
)