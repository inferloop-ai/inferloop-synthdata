package errors

import (
	"fmt"
	"time"
)

// Storage-specific error definitions
var (
	// Connection errors
	ErrStorageConnectionLost      = NewStorageError("STORAGE_CONNECTION_LOST", "storage connection lost")
	ErrStorageConnectionTimeout   = NewStorageError("STORAGE_CONNECTION_TIMEOUT", "storage connection timeout")
	ErrStorageConnectionPool      = NewStorageError("STORAGE_CONNECTION_POOL", "storage connection pool exhausted")
	ErrStorageConnectionRefused   = NewStorageError("STORAGE_CONNECTION_REFUSED", "storage connection refused")
	ErrStorageConnectionReset     = NewStorageError("STORAGE_CONNECTION_RESET", "storage connection reset")
	
	// Authentication and authorization errors
	ErrStorageAuthenticationFailed = NewStorageError("STORAGE_AUTH_FAILED", "storage authentication failed")
	ErrStorageUnauthorized        = NewStorageError("STORAGE_UNAUTHORIZED", "storage access unauthorized")
	ErrStoragePermissionDenied    = NewStorageError("STORAGE_PERMISSION_DENIED", "storage permission denied")
	ErrStorageInvalidCredentials  = NewStorageError("STORAGE_INVALID_CREDENTIALS", "storage invalid credentials")
	ErrStorageTokenExpired        = NewStorageError("STORAGE_TOKEN_EXPIRED", "storage token expired")
	
	// Data operation errors
	ErrStorageRecordNotFound      = NewStorageError("STORAGE_RECORD_NOT_FOUND", "storage record not found")
	ErrStorageDuplicateKey        = NewStorageError("STORAGE_DUPLICATE_KEY", "storage duplicate key")
	ErrStorageConstraintViolation = NewStorageError("STORAGE_CONSTRAINT_VIOLATION", "storage constraint violation")
	ErrStorageDataCorrupted       = NewStorageError("STORAGE_DATA_CORRUPTED", "storage data corrupted")
	ErrStorageDataTooLarge        = NewStorageError("STORAGE_DATA_TOO_LARGE", "storage data too large")
	ErrStorageInvalidData         = NewStorageError("STORAGE_INVALID_DATA", "storage invalid data format")
	ErrStorageSchemaMismatch      = NewStorageError("STORAGE_SCHEMA_MISMATCH", "storage schema mismatch")
	
	// Query and transaction errors
	ErrStorageQueryFailed         = NewStorageError("STORAGE_QUERY_FAILED", "storage query failed")
	ErrStorageQueryTimeout        = NewStorageError("STORAGE_QUERY_TIMEOUT", "storage query timeout")
	ErrStorageQuerySyntaxError    = NewStorageError("STORAGE_QUERY_SYNTAX_ERROR", "storage query syntax error")
	ErrStorageTransactionFailed   = NewStorageError("STORAGE_TRANSACTION_FAILED", "storage transaction failed")
	ErrStorageTransactionAborted  = NewStorageError("STORAGE_TRANSACTION_ABORTED", "storage transaction aborted")
	ErrStorageDeadlock            = NewStorageError("STORAGE_DEADLOCK", "storage deadlock detected")
	ErrStorageLockTimeout         = NewStorageError("STORAGE_LOCK_TIMEOUT", "storage lock timeout")
	
	// Capacity and resource errors
	ErrStorageCapacityExceeded    = NewStorageError("STORAGE_CAPACITY_EXCEEDED", "storage capacity exceeded")
	ErrStorageDiskFull            = NewStorageError("STORAGE_DISK_FULL", "storage disk full")
	ErrStorageMemoryExhausted     = NewStorageError("STORAGE_MEMORY_EXHAUSTED", "storage memory exhausted")
	ErrStorageQuotaExceeded       = NewStorageError("STORAGE_QUOTA_EXCEEDED", "storage quota exceeded")
	ErrStorageRateLimitExceeded   = NewStorageError("STORAGE_RATE_LIMIT_EXCEEDED", "storage rate limit exceeded")
	ErrStorageConnectionLimitExceeded = NewStorageError("STORAGE_CONNECTION_LIMIT_EXCEEDED", "storage connection limit exceeded")
	
	// Backup and recovery errors
	ErrStorageBackupFailed        = NewStorageError("STORAGE_BACKUP_FAILED", "storage backup failed")
	ErrStorageRestoreFailed       = NewStorageError("STORAGE_RESTORE_FAILED", "storage restore failed")
	ErrStorageCheckpointFailed    = NewStorageError("STORAGE_CHECKPOINT_FAILED", "storage checkpoint failed")
	ErrStorageReplicationFailed   = NewStorageError("STORAGE_REPLICATION_FAILED", "storage replication failed")
	ErrStorageSnapshotFailed      = NewStorageError("STORAGE_SNAPSHOT_FAILED", "storage snapshot failed")
	
	// Configuration and schema errors
	ErrStorageConfigurationInvalid = NewStorageError("STORAGE_CONFIG_INVALID", "storage configuration invalid")
	ErrStorageSchemaVersionMismatch = NewStorageError("STORAGE_SCHEMA_VERSION_MISMATCH", "storage schema version mismatch")
	ErrStorageMigrationFailed     = NewStorageError("STORAGE_MIGRATION_FAILED", "storage migration failed")
	ErrStorageIndexCorrupted      = NewStorageError("STORAGE_INDEX_CORRUPTED", "storage index corrupted")
	ErrStorageMetadataCorrupted   = NewStorageError("STORAGE_METADATA_CORRUPTED", "storage metadata corrupted")
	
	// Time series specific errors
	ErrStorageRetentionPolicyFailed = NewStorageError("STORAGE_RETENTION_POLICY_FAILED", "storage retention policy failed")
	ErrStorageCompactionFailed    = NewStorageError("STORAGE_COMPACTION_FAILED", "storage compaction failed")
	ErrStoragePartitionFull       = NewStorageError("STORAGE_PARTITION_FULL", "storage partition full")
	ErrStorageTimeRangeInvalid    = NewStorageError("STORAGE_TIME_RANGE_INVALID", "storage time range invalid")
	ErrStorageSeriesNotFound      = NewStorageError("STORAGE_SERIES_NOT_FOUND", "storage time series not found")
	
	// Cloud storage specific errors
	ErrStorageRegionUnavailable   = NewStorageError("STORAGE_REGION_UNAVAILABLE", "storage region unavailable")
	ErrStorageServiceUnavailable  = NewStorageError("STORAGE_SERVICE_UNAVAILABLE", "storage service unavailable")
	ErrStorageAPIRateLimited      = NewStorageError("STORAGE_API_RATE_LIMITED", "storage API rate limited")
	ErrStorageAccessKeyInvalid    = NewStorageError("STORAGE_ACCESS_KEY_INVALID", "storage access key invalid")
	ErrStorageBucketNotFound      = NewStorageError("STORAGE_BUCKET_NOT_FOUND", "storage bucket not found")
	ErrStorageObjectNotFound      = NewStorageError("STORAGE_OBJECT_NOT_FOUND", "storage object not found")
)

// StorageError represents a storage-specific error with additional context
type StorageError struct {
	*AppError
	StorageType   string        `json:"storage_type,omitempty"`   // "influxdb", "timescaledb", "s3", etc.
	Database      string        `json:"database,omitempty"`      // Database/bucket name
	Table         string        `json:"table,omitempty"`         // Table/measurement name
	Operation     string        `json:"operation,omitempty"`     // "read", "write", "delete", "query"
	Query         string        `json:"query,omitempty"`         // SQL/query string (sanitized)
	Duration      time.Duration `json:"duration,omitempty"`      // Operation duration
	RowsAffected  int64         `json:"rows_affected,omitempty"` // Number of rows affected
	BytesRead     int64         `json:"bytes_read,omitempty"`    // Bytes read
	BytesWritten  int64         `json:"bytes_written,omitempty"` // Bytes written
	RetryAttempt  int           `json:"retry_attempt,omitempty"` // Retry attempt number
	Recoverable   bool          `json:"recoverable"`             // Whether error is recoverable
	Transient     bool          `json:"transient"`               // Whether error is transient
}

// NewStorageError creates a new storage error
func NewStorageError(code, message string) *AppError {
	return &AppError{
		Type:       ErrorTypeStorage,
		Code:       code,
		Message:    message,
		Retryable:  isRetryableStorageError(code),
		HTTPStatus: getStorageErrorHTTPStatus(code),
	}
}

// WrapStorageError wraps a storage error with additional context
func WrapStorageError(err error, operation, storageType string) *StorageError {
	if err == nil {
		return nil
	}
	
	storageErr := &StorageError{
		AppError:    WrapError(err, ErrorTypeStorage, "STORAGE_ERROR", "Storage operation failed"),
		StorageType: storageType,
		Operation:   operation,
		Recoverable: isRecoverableStorageError(err),
		Transient:   isTransientStorageError(err),
	}
	
	return storageErr
}

// NewStorageConnectionError creates a storage connection error
func NewStorageConnectionError(storageType, database string, err error) *StorageError {
	return &StorageError{
		AppError: &AppError{
			Type:       ErrorTypeStorage,
			Code:       "STORAGE_CONNECTION_ERROR",
			Message:    fmt.Sprintf("Failed to connect to %s database %s", storageType, database),
			Cause:      err,
			Retryable:  true,
			HTTPStatus: 503,
		},
		StorageType: storageType,
		Database:    database,
		Operation:   "connect",
		Recoverable: true,
		Transient:   true,
	}
}

// NewStorageQueryError creates a storage query error
func NewStorageQueryError(storageType, database, query string, err error) *StorageError {
	// Sanitize query for logging (remove sensitive data)
	sanitizedQuery := sanitizeQuery(query)
	
	return &StorageError{
		AppError: &AppError{
			Type:       ErrorTypeStorage,
			Code:       "STORAGE_QUERY_ERROR",
			Message:    fmt.Sprintf("Query failed in %s database %s", storageType, database),
			Cause:      err,
			Retryable:  false,
			HTTPStatus: 400,
		},
		StorageType: storageType,
		Database:    database,
		Query:       sanitizedQuery,
		Operation:   "query",
		Recoverable: false,
		Transient:   false,
	}
}

// NewStorageTimeoutError creates a storage timeout error
func NewStorageTimeoutError(storageType, operation string, duration time.Duration) *StorageError {
	return &StorageError{
		AppError: &AppError{
			Type:       ErrorTypeStorage,
			Code:       "STORAGE_TIMEOUT_ERROR",
			Message:    fmt.Sprintf("Storage %s operation timed out after %v", operation, duration),
			Retryable:  true,
			HTTPStatus: 408,
		},
		StorageType: storageType,
		Operation:   operation,
		Duration:    duration,
		Recoverable: true,
		Transient:   true,
	}
}

// NewStorageCapacityError creates a storage capacity error
func NewStorageCapacityError(storageType, database string, usage, limit int64) *StorageError {
	return &StorageError{
		AppError: &AppError{
			Type:       ErrorTypeStorage,
			Code:       "STORAGE_CAPACITY_ERROR",
			Message:    fmt.Sprintf("Storage capacity exceeded in %s database %s: %d/%d bytes", storageType, database, usage, limit),
			Retryable:  false,
			HTTPStatus: 507, // Insufficient Storage
		},
		StorageType: storageType,
		Database:    database,
		Operation:   "write",
		Recoverable: false,
		Transient:   false,
	}
}

// NewStoragePermissionError creates a storage permission error
func NewStoragePermissionError(storageType, database, operation string) *StorageError {
	return &StorageError{
		AppError: &AppError{
			Type:       ErrorTypeStorage,
			Code:       "STORAGE_PERMISSION_ERROR",
			Message:    fmt.Sprintf("Permission denied for %s operation in %s database %s", operation, storageType, database),
			Retryable:  false,
			HTTPStatus: 403,
		},
		StorageType: storageType,
		Database:    database,
		Operation:   operation,
		Recoverable: false,
		Transient:   false,
	}
}

// WithStorageType adds storage type information to the storage error
func (se *StorageError) WithStorageType(storageType string) *StorageError {
	se.StorageType = storageType
	return se
}

// WithDatabase adds database information to the storage error
func (se *StorageError) WithDatabase(database string) *StorageError {
	se.Database = database
	return se
}

// WithTable adds table information to the storage error
func (se *StorageError) WithTable(table string) *StorageError {
	se.Table = table
	return se
}

// WithOperation adds operation information to the storage error
func (se *StorageError) WithOperation(operation string) *StorageError {
	se.Operation = operation
	return se
}

// WithQuery adds query information to the storage error (sanitized)
func (se *StorageError) WithQuery(query string) *StorageError {
	se.Query = sanitizeQuery(query)
	return se
}

// WithDuration adds duration information to the storage error
func (se *StorageError) WithDuration(duration time.Duration) *StorageError {
	se.Duration = duration
	return se
}

// WithRowsAffected adds rows affected information to the storage error
func (se *StorageError) WithRowsAffected(rows int64) *StorageError {
	se.RowsAffected = rows
	return se
}

// WithBytesTransferred adds bytes transferred information to the storage error
func (se *StorageError) WithBytesTransferred(read, written int64) *StorageError {
	se.BytesRead = read
	se.BytesWritten = written
	return se
}

// WithRetryAttempt adds retry attempt information to the storage error
func (se *StorageError) WithRetryAttempt(attempt int) *StorageError {
	se.RetryAttempt = attempt
	return se
}

// IsRecoverable checks if the storage error is recoverable
func (se *StorageError) IsRecoverable() bool {
	return se.Recoverable
}

// IsTransient checks if the storage error is transient
func (se *StorageError) IsTransient() bool {
	return se.Transient
}

// ShouldRetry checks if the operation should be retried based on the error
func (se *StorageError) ShouldRetry() bool {
	return se.Retryable && (se.Recoverable || se.Transient)
}

// GetRetryDelay calculates the appropriate retry delay based on the error and attempt number
func (se *StorageError) GetRetryDelay(attempt int) time.Duration {
	if !se.ShouldRetry() {
		return 0
	}
	
	// Base delay with exponential backoff
	baseDelay := 500 * time.Millisecond
	maxDelay := 30 * time.Second
	
	// Different delays for different error types
	switch se.Code {
	case "STORAGE_CONNECTION_TIMEOUT", "STORAGE_QUERY_TIMEOUT":
		baseDelay = 2 * time.Second
	case "STORAGE_RATE_LIMIT_EXCEEDED":
		baseDelay = 5 * time.Second
	case "STORAGE_CONNECTION_LOST", "STORAGE_CONNECTION_REFUSED":
		baseDelay = 1 * time.Second
	case "STORAGE_DEADLOCK":
		baseDelay = 100 * time.Millisecond // Faster retry for deadlocks
	}
	
	// Calculate exponential backoff: baseDelay * 2^attempt
	delay := time.Duration(float64(baseDelay) * (1 << uint(attempt)))
	
	if delay > maxDelay {
		delay = maxDelay
	}
	
	return delay
}

// isRetryableStorageError determines if a storage error code is retryable
func isRetryableStorageError(code string) bool {
	retryableErrors := map[string]bool{
		"STORAGE_CONNECTION_LOST":           true,
		"STORAGE_CONNECTION_TIMEOUT":        true,
		"STORAGE_CONNECTION_POOL":           true,
		"STORAGE_CONNECTION_REFUSED":        true,
		"STORAGE_CONNECTION_RESET":          true,
		"STORAGE_QUERY_TIMEOUT":             true,
		"STORAGE_TRANSACTION_ABORTED":       true,
		"STORAGE_DEADLOCK":                  true,
		"STORAGE_LOCK_TIMEOUT":              true,
		"STORAGE_RATE_LIMIT_EXCEEDED":       true,
		"STORAGE_SERVICE_UNAVAILABLE":       true,
		"STORAGE_REGION_UNAVAILABLE":        true,
		"STORAGE_API_RATE_LIMITED":          true,
		"STORAGE_CONNECTION_LIMIT_EXCEEDED": true,
	}
	
	return retryableErrors[code]
}

// getStorageErrorHTTPStatus returns the appropriate HTTP status for a storage error code
func getStorageErrorHTTPStatus(code string) int {
	statusMap := map[string]int{
		"STORAGE_RECORD_NOT_FOUND":          404,
		"STORAGE_AUTH_FAILED":               401,
		"STORAGE_UNAUTHORIZED":              401,
		"STORAGE_PERMISSION_DENIED":         403,
		"STORAGE_INVALID_CREDENTIALS":       401,
		"STORAGE_TOKEN_EXPIRED":             401,
		"STORAGE_DUPLICATE_KEY":             409,
		"STORAGE_CONSTRAINT_VIOLATION":      400,
		"STORAGE_INVALID_DATA":              400,
		"STORAGE_SCHEMA_MISMATCH":           400,
		"STORAGE_QUERY_SYNTAX_ERROR":        400,
		"STORAGE_CAPACITY_EXCEEDED":         507,
		"STORAGE_DISK_FULL":                 507,
		"STORAGE_QUOTA_EXCEEDED":            507,
		"STORAGE_RATE_LIMIT_EXCEEDED":       429,
		"STORAGE_CONNECTION_TIMEOUT":        408,
		"STORAGE_QUERY_TIMEOUT":             408,
		"STORAGE_LOCK_TIMEOUT":              408,
		"STORAGE_SERVICE_UNAVAILABLE":       503,
		"STORAGE_REGION_UNAVAILABLE":        503,
		"STORAGE_BUCKET_NOT_FOUND":          404,
		"STORAGE_OBJECT_NOT_FOUND":          404,
		"STORAGE_SERIES_NOT_FOUND":          404,
	}
	
	if status, exists := statusMap[code]; exists {
		return status
	}
	
	return 500 // Default to internal server error
}

// isRecoverableStorageError determines if a storage error is recoverable
func isRecoverableStorageError(err error) bool {
	if err == nil {
		return false
	}
	
	// Check error message for common recoverable patterns
	errStr := err.Error()
	
	recoverablePatterns := []string{
		"connection",
		"timeout",
		"temporary",
		"deadlock",
		"lock",
		"rate limit",
		"unavailable",
		"network",
	}
	
	for _, pattern := range recoverablePatterns {
		if contains(errStr, pattern) {
			return true
		}
	}
	
	return false
}

// isTransientStorageError determines if a storage error is transient
func isTransientStorageError(err error) bool {
	if err == nil {
		return false
	}
	
	errStr := err.Error()
	
	transientPatterns := []string{
		"timeout",
		"temporary",
		"rate limit",
		"connection reset",
		"service unavailable",
		"deadlock",
	}
	
	for _, pattern := range transientPatterns {
		if contains(errStr, pattern) {
			return true
		}
	}
	
	return false
}

// sanitizeQuery removes sensitive information from query strings
func sanitizeQuery(query string) string {
	if len(query) > 500 {
		query = query[:500] + "... [truncated]"
	}
	
	// TODO: Implement more sophisticated sanitization
	// Remove potential passwords, tokens, etc.
	// This is a basic implementation
	
	return query
}

// contains checks if a string contains a substring (case-insensitive)
func contains(str, substr string) bool {
	// Simple case-insensitive contains check
	for i := 0; i <= len(str)-len(substr); i++ {
		match := true
		for j := 0; j < len(substr); j++ {
			if str[i+j] != substr[j] && str[i+j] != substr[j]-32 && str[i+j] != substr[j]+32 {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

// StorageErrorSummary provides a summary of storage error statistics
type StorageErrorSummary struct {
	TotalErrors        int                    `json:"total_errors"`
	ErrorsByType       map[string]int         `json:"errors_by_type"`
	ErrorsByStorage    map[string]int         `json:"errors_by_storage"`
	ErrorsByOperation  map[string]int         `json:"errors_by_operation"`
	RecoverableErrors  int                    `json:"recoverable_errors"`
	TransientErrors    int                    `json:"transient_errors"`
	RetryableErrors    int                    `json:"retryable_errors"`
	AverageRetries     float64                `json:"average_retries"`
	AverageDuration    time.Duration          `json:"average_duration"`
	TotalBytesAffected int64                  `json:"total_bytes_affected"`
	TopErrors          []string               `json:"top_errors"`
}

// AnalyzeStorageErrors analyzes a collection of storage errors and returns a summary
func AnalyzeStorageErrors(errors []*StorageError) *StorageErrorSummary {
	summary := &StorageErrorSummary{
		ErrorsByType:      make(map[string]int),
		ErrorsByStorage:   make(map[string]int),
		ErrorsByOperation: make(map[string]int),
	}
	
	totalRetries := 0
	totalDuration := time.Duration(0)
	errorCounts := make(map[string]int)
	
	for _, err := range errors {
		summary.TotalErrors++
		
		// Count by error code
		summary.ErrorsByType[err.Code]++
		errorCounts[err.Code]++
		
		// Count by storage type
		if err.StorageType != "" {
			summary.ErrorsByStorage[err.StorageType]++
		}
		
		// Count by operation
		if err.Operation != "" {
			summary.ErrorsByOperation[err.Operation]++
		}
		
		// Count recoverable, transient, and retryable errors
		if err.IsRecoverable() {
			summary.RecoverableErrors++
		}
		
		if err.IsTransient() {
			summary.TransientErrors++
		}
		
		if err.ShouldRetry() {
			summary.RetryableErrors++
		}
		
		totalRetries += err.RetryAttempt
		totalDuration += err.Duration
		summary.TotalBytesAffected += err.BytesRead + err.BytesWritten
	}
	
	// Calculate averages
	if summary.TotalErrors > 0 {
		summary.AverageRetries = float64(totalRetries) / float64(summary.TotalErrors)
		summary.AverageDuration = totalDuration / time.Duration(summary.TotalErrors)
	}
	
	// Find top errors (up to 5)
	type errorCount struct {
		code  string
		count int
	}
	
	var errorList []errorCount
	for code, count := range errorCounts {
		errorList = append(errorList, errorCount{code, count})
	}
	
	// Sort by count (descending)
	for i := 0; i < len(errorList)-1; i++ {
		for j := i + 1; j < len(errorList); j++ {
			if errorList[j].count > errorList[i].count {
				errorList[i], errorList[j] = errorList[j], errorList[i]
			}
		}
	}
	
	// Take top 5
	maxTop := 5
	if len(errorList) < maxTop {
		maxTop = len(errorList)
	}
	
	for i := 0; i < maxTop; i++ {
		summary.TopErrors = append(summary.TopErrors, errorList[i].code)
	}
	
	return summary
}