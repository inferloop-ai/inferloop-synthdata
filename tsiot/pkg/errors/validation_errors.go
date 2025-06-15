package errors

import (
	"fmt"
	"strings"
	"time"
)

// Validation-specific error definitions
var (
	// Data validation errors
	ErrValidationFieldRequired     = NewValidationError("FIELD_REQUIRED", "required field is missing")
	ErrValidationFieldInvalid      = NewValidationError("FIELD_INVALID", "field value is invalid")
	ErrValidationFieldTooLong      = NewValidationError("FIELD_TOO_LONG", "field value is too long")
	ErrValidationFieldTooShort     = NewValidationError("FIELD_TOO_SHORT", "field value is too short")
	ErrValidationFieldOutOfRange   = NewValidationError("FIELD_OUT_OF_RANGE", "field value is out of range")
	ErrValidationFieldWrongType    = NewValidationError("FIELD_WRONG_TYPE", "field value has wrong type")
	ErrValidationFieldPattern      = NewValidationError("FIELD_PATTERN_MISMATCH", "field value doesn't match pattern")
	
	// Time series validation errors
	ErrValidationTimeSeriesEmpty   = NewValidationError("TIMESERIES_EMPTY", "time series is empty")
	ErrValidationTimeRangeInvalid  = NewValidationError("TIME_RANGE_INVALID", "time range is invalid")
	ErrValidationFrequencyInvalid  = NewValidationError("FREQUENCY_INVALID", "frequency is invalid")
	ErrValidationDataPointsInvalid = NewValidationError("DATA_POINTS_INVALID", "data points are invalid")
	ErrValidationTimeGapsExist     = NewValidationError("TIME_GAPS_EXIST", "time gaps exist in series")
	ErrValidationDuplicateTimestamps = NewValidationError("DUPLICATE_TIMESTAMPS", "duplicate timestamps found")
	ErrValidationUnsortedTimestamps = NewValidationError("UNSORTED_TIMESTAMPS", "timestamps are not sorted")
	
	// Sensor validation errors
	ErrValidationSensorTypeInvalid = NewValidationError("SENSOR_TYPE_INVALID", "sensor type is invalid")
	ErrValidationSensorIDInvalid   = NewValidationError("SENSOR_ID_INVALID", "sensor ID is invalid")
	ErrValidationUnitInvalid       = NewValidationError("UNIT_INVALID", "unit is invalid")
	ErrValidationValueOutOfRange   = NewValidationError("VALUE_OUT_OF_RANGE", "sensor value is out of range")
	ErrValidationQualityInvalid    = NewValidationError("QUALITY_INVALID", "quality value is invalid")
	ErrValidationCalibrationExpired = NewValidationError("CALIBRATION_EXPIRED", "sensor calibration expired")
	
	// Generation parameter validation errors
	ErrValidationGeneratorInvalid  = NewValidationError("GENERATOR_INVALID", "generator type is invalid")
	ErrValidationParametersInvalid = NewValidationError("PARAMETERS_INVALID", "generation parameters are invalid")
	ErrValidationEpochsInvalid     = NewValidationError("EPOCHS_INVALID", "epochs value is invalid")
	ErrValidationBatchSizeInvalid  = NewValidationError("BATCH_SIZE_INVALID", "batch size is invalid")
	ErrValidationLearningRateInvalid = NewValidationError("LEARNING_RATE_INVALID", "learning rate is invalid")
	ErrValidationSequenceLengthInvalid = NewValidationError("SEQUENCE_LENGTH_INVALID", "sequence length is invalid")
	
	// Privacy validation errors
	ErrValidationEpsilonInvalid    = NewValidationError("EPSILON_INVALID", "epsilon value is invalid")
	ErrValidationDeltaInvalid      = NewValidationError("DELTA_INVALID", "delta value is invalid")
	ErrValidationKValueInvalid     = NewValidationError("K_VALUE_INVALID", "k-anonymity value is invalid")
	ErrValidationLValueInvalid     = NewValidationError("L_VALUE_INVALID", "l-diversity value is invalid")
	ErrValidationTValueInvalid     = NewValidationError("T_VALUE_INVALID", "t-closeness value is invalid")
	ErrValidationPrivacyBudgetExceeded = NewValidationError("PRIVACY_BUDGET_EXCEEDED", "privacy budget exceeded")
	
	// Schema validation errors
	ErrValidationSchemaInvalid     = NewValidationError("SCHEMA_INVALID", "schema is invalid")
	ErrValidationSchemaVersionMismatch = NewValidationError("SCHEMA_VERSION_MISMATCH", "schema version mismatch")
	ErrValidationSchemaMissing     = NewValidationError("SCHEMA_MISSING", "schema is missing")
	ErrValidationSchemaConstraintViolation = NewValidationError("SCHEMA_CONSTRAINT_VIOLATION", "schema constraint violation")
	
	// File format validation errors
	ErrValidationFormatUnsupported = NewValidationError("FORMAT_UNSUPPORTED", "file format is unsupported")
	ErrValidationFormatCorrupted   = NewValidationError("FORMAT_CORRUPTED", "file format is corrupted")
	ErrValidationHeaderMissing     = NewValidationError("HEADER_MISSING", "file header is missing")
	ErrValidationHeaderInvalid     = NewValidationError("HEADER_INVALID", "file header is invalid")
	ErrValidationEncodingInvalid   = NewValidationError("ENCODING_INVALID", "file encoding is invalid")
	
	// Statistical validation errors
	ErrValidationStatisticalTestFailed = NewValidationError("STATISTICAL_TEST_FAILED", "statistical test failed")
	ErrValidationDistributionMismatch = NewValidationError("DISTRIBUTION_MISMATCH", "distribution mismatch")
	ErrValidationCorrelationInvalid = NewValidationError("CORRELATION_INVALID", "correlation is invalid")
	ErrValidationStationarityViolated = NewValidationError("STATIONARITY_VIOLATED", "stationarity assumption violated")
	ErrValidationSeasonalityMismatch = NewValidationError("SEASONALITY_MISMATCH", "seasonality pattern mismatch")
	ErrValidationTrendMismatch     = NewValidationError("TREND_MISMATCH", "trend pattern mismatch")
	
	// Quality validation errors
	ErrValidationQualityBelowThreshold = NewValidationError("QUALITY_BELOW_THRESHOLD", "data quality below threshold")
	ErrValidationCompletenessInsufficient = NewValidationError("COMPLETENESS_INSUFFICIENT", "data completeness insufficient")
	ErrValidationConsistencyViolated = NewValidationError("CONSISTENCY_VIOLATED", "data consistency violated")
	ErrValidationAccuracyInsufficient = NewValidationError("ACCURACY_INSUFFICIENT", "data accuracy insufficient")
	ErrValidationIntegrityCompromised = NewValidationError("INTEGRITY_COMPROMISED", "data integrity compromised")
)

// ValidationError represents a validation-specific error with field-level details
type ValidationError struct {
	*AppError
	Field       string                 `json:"field,omitempty"`       // Field that failed validation
	Value       interface{}            `json:"value,omitempty"`       // Invalid value
	Expected    interface{}            `json:"expected,omitempty"`    // Expected value or format
	Rule        string                 `json:"rule,omitempty"`        // Validation rule that failed
	Path        string                 `json:"path,omitempty"`        // JSON path to the field
	LineNumber  int                    `json:"line_number,omitempty"` // Line number in file
	ColumnNumber int                   `json:"column_number,omitempty"` // Column number in file
	TestName    string                 `json:"test_name,omitempty"`   // Statistical test name
	TestResult  *TestResult            `json:"test_result,omitempty"` // Statistical test result
	Threshold   float64                `json:"threshold,omitempty"`   // Quality threshold
	ActualValue float64                `json:"actual_value,omitempty"` // Actual quality value
	Severity    ValidationSeverity     `json:"severity"`              // Severity level
}

// ValidationSeverity represents the severity of a validation error
type ValidationSeverity string

const (
	SeverityInfo     ValidationSeverity = "info"
	SeverityWarning  ValidationSeverity = "warning"
	SeverityError    ValidationSeverity = "error"
	SeverityCritical ValidationSeverity = "critical"
)

// TestResult represents the result of a statistical test
type TestResult struct {
	TestName   string                 `json:"test_name"`
	Statistic  float64                `json:"statistic"`
	PValue     float64                `json:"p_value"`
	Critical   float64                `json:"critical_value"`
	Alpha      float64                `json:"alpha"`
	Passed     bool                   `json:"passed"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

// MultiValidationError represents multiple validation errors
type MultiValidationError struct {
	*AppError
	Errors      []*ValidationError     `json:"errors"`
	FieldErrors map[string][]string    `json:"field_errors"`
	Summary     *ValidationSummary     `json:"summary"`
}

// ValidationSummary provides a summary of validation results
type ValidationSummary struct {
	TotalFields      int                           `json:"total_fields"`
	ValidFields      int                           `json:"valid_fields"`
	InvalidFields    int                           `json:"invalid_fields"`
	ErrorsByType     map[string]int                `json:"errors_by_type"`
	ErrorsBySeverity map[ValidationSeverity]int    `json:"errors_by_severity"`
	OverallScore     float64                       `json:"overall_score"`
	PassedTests      int                           `json:"passed_tests"`
	FailedTests      int                           `json:"failed_tests"`
	QualityMetrics   map[string]float64            `json:"quality_metrics"`
}

// NewValidationError creates a new validation error
func NewValidationError(code, message string) *AppError {
	return &AppError{
		Type:       ErrorTypeValidation,
		Code:       code,
		Message:    message,
		Retryable:  false,
		HTTPStatus: 400,
	}
}

// WrapValidationError wraps a validation error with additional context
func WrapValidationError(err error, field string, value interface{}) *ValidationError {
	if err == nil {
		return nil
	}
	
	return &ValidationError{
		AppError: WrapError(err, ErrorTypeValidation, "VALIDATION_ERROR", "Validation failed"),
		Field:    field,
		Value:    value,
		Severity: SeverityError,
	}
}

// NewFieldValidationError creates a field-specific validation error
func NewFieldValidationError(field, rule string, value, expected interface{}) *ValidationError {
	return &ValidationError{
		AppError: &AppError{
			Type:       ErrorTypeValidation,
			Code:       "FIELD_VALIDATION_ERROR",
			Message:    fmt.Sprintf("Field '%s' validation failed: %s", field, rule),
			Retryable:  false,
			HTTPStatus: 400,
		},
		Field:    field,
		Value:    value,
		Expected: expected,
		Rule:     rule,
		Severity: SeverityError,
	}
}

// NewTimeSeriesValidationError creates a time series validation error
func NewTimeSeriesValidationError(code, message string, field string, value interface{}) *ValidationError {
	return &ValidationError{
		AppError: &AppError{
			Type:       ErrorTypeValidation,
			Code:       code,
			Message:    message,
			Retryable:  false,
			HTTPStatus: 400,
		},
		Field:    field,
		Value:    value,
		Severity: SeverityError,
	}
}

// NewStatisticalValidationError creates a statistical validation error
func NewStatisticalValidationError(testName string, result *TestResult) *ValidationError {
	return &ValidationError{
		AppError: &AppError{
			Type:       ErrorTypeValidation,
			Code:       "STATISTICAL_TEST_FAILED",
			Message:    fmt.Sprintf("Statistical test '%s' failed: p-value=%.6f", testName, result.PValue),
			Retryable:  false,
			HTTPStatus: 400,
		},
		TestName:   testName,
		TestResult: result,
		Severity:   SeverityWarning, // Statistical tests are often warnings
	}
}

// NewQualityValidationError creates a quality validation error
func NewQualityValidationError(metric string, actual, threshold float64) *ValidationError {
	return &ValidationError{
		AppError: &AppError{
			Type:       ErrorTypeValidation,
			Code:       "QUALITY_BELOW_THRESHOLD",
			Message:    fmt.Sprintf("Quality metric '%s' (%.3f) below threshold (%.3f)", metric, actual, threshold),
			Retryable:  false,
			HTTPStatus: 400,
		},
		Field:       metric,
		ActualValue: actual,
		Threshold:   threshold,
		Severity:    SeverityError,
	}
}

// NewMultiValidationError creates a multi-validation error
func NewMultiValidationError(errors []*ValidationError) *MultiValidationError {
	fieldErrors := make(map[string][]string)
	errorsByType := make(map[string]int)
	errorsBySeverity := make(map[ValidationSeverity]int)
	
	for _, err := range errors {
		// Group by field
		if err.Field != "" {
			fieldErrors[err.Field] = append(fieldErrors[err.Field], err.Message)
		}
		
		// Count by type
		errorsByType[err.Code]++
		
		// Count by severity
		errorsBySeverity[err.Severity]++
	}
	
	summary := &ValidationSummary{
		InvalidFields:    len(fieldErrors),
		ErrorsByType:     errorsByType,
		ErrorsBySeverity: errorsBySeverity,
		FailedTests:      len(errors),
	}
	
	return &MultiValidationError{
		AppError: &AppError{
			Type:       ErrorTypeValidation,
			Code:       "MULTIPLE_VALIDATION_ERRORS",
			Message:    fmt.Sprintf("Multiple validation errors: %d errors", len(errors)),
			Retryable:  false,
			HTTPStatus: 400,
		},
		Errors:      errors,
		FieldErrors: fieldErrors,
		Summary:     summary,
	}
}

// WithField adds field information to the validation error
func (ve *ValidationError) WithField(field string) *ValidationError {
	ve.Field = field
	return ve
}

// WithValue adds value information to the validation error
func (ve *ValidationError) WithValue(value interface{}) *ValidationError {
	ve.Value = value
	return ve
}

// WithExpected adds expected value information to the validation error
func (ve *ValidationError) WithExpected(expected interface{}) *ValidationError {
	ve.Expected = expected
	return ve
}

// WithRule adds validation rule information to the validation error
func (ve *ValidationError) WithRule(rule string) *ValidationError {
	ve.Rule = rule
	return ve
}

// WithPath adds JSON path information to the validation error
func (ve *ValidationError) WithPath(path string) *ValidationError {
	ve.Path = path
	return ve
}

// WithLocation adds file location information to the validation error
func (ve *ValidationError) WithLocation(line, column int) *ValidationError {
	ve.LineNumber = line
	ve.ColumnNumber = column
	return ve
}

// WithTestResult adds statistical test result to the validation error
func (ve *ValidationError) WithTestResult(testName string, result *TestResult) *ValidationError {
	ve.TestName = testName
	ve.TestResult = result
	return ve
}

// WithThreshold adds quality threshold information to the validation error
func (ve *ValidationError) WithThreshold(threshold, actual float64) *ValidationError {
	ve.Threshold = threshold
	ve.ActualValue = actual
	return ve
}

// WithSeverity adds severity information to the validation error
func (ve *ValidationError) WithSeverity(severity ValidationSeverity) *ValidationError {
	ve.Severity = severity
	return ve
}

// GetFieldPath returns the full path to the field (including JSON path)
func (ve *ValidationError) GetFieldPath() string {
	if ve.Path != "" && ve.Field != "" {
		return fmt.Sprintf("%s.%s", ve.Path, ve.Field)
	}
	if ve.Path != "" {
		return ve.Path
	}
	return ve.Field
}

// GetLocation returns the file location as a string
func (ve *ValidationError) GetLocation() string {
	if ve.LineNumber > 0 && ve.ColumnNumber > 0 {
		return fmt.Sprintf("line %d, column %d", ve.LineNumber, ve.ColumnNumber)
	}
	if ve.LineNumber > 0 {
		return fmt.Sprintf("line %d", ve.LineNumber)
	}
	return ""
}

// IsCritical checks if the validation error is critical
func (ve *ValidationError) IsCritical() bool {
	return ve.Severity == SeverityCritical
}

// IsWarning checks if the validation error is a warning
func (ve *ValidationError) IsWarning() bool {
	return ve.Severity == SeverityWarning
}

// AddError adds a validation error to the multi-validation error
func (mve *MultiValidationError) AddError(err *ValidationError) {
	mve.Errors = append(mve.Errors, err)
	
	// Update field errors
	if err.Field != "" {
		mve.FieldErrors[err.Field] = append(mve.FieldErrors[err.Field], err.Message)
	}
	
	// Update summary
	mve.Summary.InvalidFields = len(mve.FieldErrors)
	mve.Summary.FailedTests = len(mve.Errors)
	mve.Summary.ErrorsByType[err.Code]++
	mve.Summary.ErrorsBySeverity[err.Severity]++
}

// HasCriticalErrors checks if there are any critical validation errors
func (mve *MultiValidationError) HasCriticalErrors() bool {
	return mve.Summary.ErrorsBySeverity[SeverityCritical] > 0
}

// HasErrors checks if there are any validation errors (excluding warnings)
func (mve *MultiValidationError) HasErrors() bool {
	return mve.Summary.ErrorsBySeverity[SeverityError] > 0 || 
		   mve.Summary.ErrorsBySeverity[SeverityCritical] > 0
}

// GetErrorsByField returns validation errors grouped by field
func (mve *MultiValidationError) GetErrorsByField() map[string][]*ValidationError {
	fieldErrors := make(map[string][]*ValidationError)
	for _, err := range mve.Errors {
		if err.Field != "" {
			fieldErrors[err.Field] = append(fieldErrors[err.Field], err)
		}
	}
	return fieldErrors
}

// GetErrorsBySeverity returns validation errors grouped by severity
func (mve *MultiValidationError) GetErrorsBySeverity() map[ValidationSeverity][]*ValidationError {
	severityErrors := make(map[ValidationSeverity][]*ValidationError)
	for _, err := range mve.Errors {
		severityErrors[err.Severity] = append(severityErrors[err.Severity], err)
	}
	return severityErrors
}

// ValidationBuilder helps build validation errors incrementally
type ValidationBuilder struct {
	errors []ValidationError
	field  string
	path   string
}

// NewValidationBuilder creates a new validation builder
func NewValidationBuilder() *ValidationBuilder {
	return &ValidationBuilder{
		errors: make([]ValidationError, 0),
	}
}

// SetField sets the current field for subsequent validations
func (vb *ValidationBuilder) SetField(field string) *ValidationBuilder {
	vb.field = field
	return vb
}

// SetPath sets the current JSON path for subsequent validations
func (vb *ValidationBuilder) SetPath(path string) *ValidationBuilder {
	vb.path = path
	return vb
}

// Required validates that a value is not nil/empty
func (vb *ValidationBuilder) Required(value interface{}) *ValidationBuilder {
	if isEmpty(value) {
		vb.errors = append(vb.errors, ValidationError{
			AppError: &AppError{
				Type:    ErrorTypeValidation,
				Code:    "FIELD_REQUIRED",
				Message: fmt.Sprintf("Field '%s' is required", vb.field),
			},
			Field:    vb.field,
			Path:     vb.path,
			Value:    value,
			Rule:     "required",
			Severity: SeverityError,
		})
	}
	return vb
}

// MinLength validates minimum string length
func (vb *ValidationBuilder) MinLength(value string, min int) *ValidationBuilder {
	if len(value) < min {
		vb.errors = append(vb.errors, ValidationError{
			AppError: &AppError{
				Type:    ErrorTypeValidation,
				Code:    "FIELD_TOO_SHORT",
				Message: fmt.Sprintf("Field '%s' is too short (min: %d)", vb.field, min),
			},
			Field:    vb.field,
			Path:     vb.path,
			Value:    value,
			Expected: min,
			Rule:     "min_length",
			Severity: SeverityError,
		})
	}
	return vb
}

// MaxLength validates maximum string length
func (vb *ValidationBuilder) MaxLength(value string, max int) *ValidationBuilder {
	if len(value) > max {
		vb.errors = append(vb.errors, ValidationError{
			AppError: &AppError{
				Type:    ErrorTypeValidation,
				Code:    "FIELD_TOO_LONG",
				Message: fmt.Sprintf("Field '%s' is too long (max: %d)", vb.field, max),
			},
			Field:    vb.field,
			Path:     vb.path,
			Value:    value,
			Expected: max,
			Rule:     "max_length",
			Severity: SeverityError,
		})
	}
	return vb
}

// Range validates numeric range
func (vb *ValidationBuilder) Range(value, min, max float64) *ValidationBuilder {
	if value < min || value > max {
		vb.errors = append(vb.errors, ValidationError{
			AppError: &AppError{
				Type:    ErrorTypeValidation,
				Code:    "FIELD_OUT_OF_RANGE",
				Message: fmt.Sprintf("Field '%s' is out of range [%.2f, %.2f]", vb.field, min, max),
			},
			Field:    vb.field,
			Path:     vb.path,
			Value:    value,
			Expected: fmt.Sprintf("[%.2f, %.2f]", min, max),
			Rule:     "range",
			Severity: SeverityError,
		})
	}
	return vb
}

// Pattern validates against a regular expression pattern
func (vb *ValidationBuilder) Pattern(value, pattern string) *ValidationBuilder {
	// Simple pattern matching (in a real implementation, use regexp package)
	if !matchesPattern(value, pattern) {
		vb.errors = append(vb.errors, ValidationError{
			AppError: &AppError{
				Type:    ErrorTypeValidation,
				Code:    "FIELD_PATTERN_MISMATCH",
				Message: fmt.Sprintf("Field '%s' doesn't match pattern", vb.field),
			},
			Field:    vb.field,
			Path:     vb.path,
			Value:    value,
			Expected: pattern,
			Rule:     "pattern",
			Severity: SeverityError,
		})
	}
	return vb
}

// Build returns the validation errors or nil if no errors
func (vb *ValidationBuilder) Build() error {
	if len(vb.errors) == 0 {
		return nil
	}
	
	if len(vb.errors) == 1 {
		return &vb.errors[0]
	}
	
	// Convert to slice of pointers
	errorPtrs := make([]*ValidationError, len(vb.errors))
	for i := range vb.errors {
		errorPtrs[i] = &vb.errors[i]
	}
	
	return NewMultiValidationError(errorPtrs)
}

// isEmpty checks if a value is considered empty
func isEmpty(value interface{}) bool {
	if value == nil {
		return true
	}
	
	switch v := value.(type) {
	case string:
		return strings.TrimSpace(v) == ""
	case []interface{}:
		return len(v) == 0
	case map[string]interface{}:
		return len(v) == 0
	case int, int32, int64:
		return v == 0
	case float32, float64:
		return v == 0.0
	case bool:
		return !v
	case time.Time:
		return v.IsZero()
	default:
		return false
	}
}

// matchesPattern performs simple pattern matching
func matchesPattern(value, pattern string) bool {
	// This is a simplified implementation
	// In a real implementation, use the regexp package
	
	// Handle some common patterns
	switch pattern {
	case "email":
		return strings.Contains(value, "@") && strings.Contains(value, ".")
	case "uuid":
		return len(value) == 36 && strings.Count(value, "-") == 4
	case "numeric":
		for _, r := range value {
			if r < '0' || r > '9' {
				return false
			}
		}
		return true
	default:
		return true // Assume valid for unknown patterns
	}
}