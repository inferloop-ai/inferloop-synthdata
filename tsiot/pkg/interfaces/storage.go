package interfaces

import (
	"context"
	"time"

	"github.com/inferloop/tsiot/pkg/models"
)

// Storage defines the interface for time series data storage
type Storage interface {
	// Connect establishes connection to the storage backend
	Connect(ctx context.Context) error

	// Close closes the connection and cleans up resources
	Close() error

	// Ping tests the connection
	Ping(ctx context.Context) error

	// GetInfo returns information about the storage backend
	GetInfo(ctx context.Context) (*StorageInfo, error)

	// Health returns health status of the storage
	Health(ctx context.Context) (*HealthStatus, error)
}

// TimeSeriesStorage extends Storage for time series operations
type TimeSeriesStorage interface {
	Storage

	// Write writes time series data
	Write(ctx context.Context, data *models.TimeSeries) error

	// WriteBatch writes multiple time series in a batch
	WriteBatch(ctx context.Context, batch []*models.TimeSeries) error

	// Read reads time series data by ID
	Read(ctx context.Context, id string) (*models.TimeSeries, error)

	// ReadRange reads time series data within a time range
	ReadRange(ctx context.Context, id string, start, end time.Time) (*models.TimeSeries, error)

	// Query queries time series data with filters
	Query(ctx context.Context, query *models.TimeSeriesQuery) ([]*models.TimeSeries, error)

	// Delete deletes time series data by ID
	Delete(ctx context.Context, id string) error

	// DeleteRange deletes time series data within a time range
	DeleteRange(ctx context.Context, id string, start, end time.Time) error

	// List lists available time series
	List(ctx context.Context, filters map[string]interface{}) ([]*models.TimeSeries, error)

	// Count returns the count of time series matching filters
	Count(ctx context.Context, filters map[string]interface{}) (int64, error)

	// GetMetrics returns storage metrics
	GetMetrics(ctx context.Context) (*StorageMetrics, error)
}

// SensorDataStorage extends Storage for sensor data operations
type SensorDataStorage interface {
	Storage

	// WriteSensorData writes sensor data
	WriteSensorData(ctx context.Context, data *models.SensorData) error

	// WriteSensorDataBatch writes multiple sensor readings in a batch
	WriteSensorDataBatch(ctx context.Context, batch *models.SensorDataBatch) error

	// ReadSensorData reads sensor data by ID
	ReadSensorData(ctx context.Context, id string) (*models.SensorData, error)

	// QuerySensorData queries sensor data with filters
	QuerySensorData(ctx context.Context, sensorID string, start, end time.Time) ([]*models.SensorData, error)

	// GetSensorConfig gets sensor configuration
	GetSensorConfig(ctx context.Context, sensorID string) (*models.SensorConfig, error)

	// SetSensorConfig sets sensor configuration
	SetSensorConfig(ctx context.Context, config *models.SensorConfig) error

	// ListSensors lists available sensors
	ListSensors(ctx context.Context) ([]*models.SensorConfig, error)
}

// StreamingStorage extends Storage for streaming operations
type StreamingStorage interface {
	Storage

	// WriteStream writes streaming data
	WriteStream(ctx context.Context, stream <-chan *models.DataPoint) error

	// ReadStream reads streaming data
	ReadStream(ctx context.Context, query *models.TimeSeriesQuery) (<-chan *models.DataPoint, <-chan error)

	// Subscribe subscribes to real-time data updates
	Subscribe(ctx context.Context, pattern string) (<-chan *models.DataPoint, error)

	// Unsubscribe unsubscribes from data updates
	Unsubscribe(ctx context.Context, pattern string) error
}

// ArchivalStorage extends Storage for data archival operations
type ArchivalStorage interface {
	Storage

	// Archive archives old data
	Archive(ctx context.Context, cutoffTime time.Time) (*ArchivalResult, error)

	// Restore restores archived data
	Restore(ctx context.Context, archiveID string) (*RestoreResult, error)

	// ListArchives lists available archives
	ListArchives(ctx context.Context) ([]*ArchiveInfo, error)

	// DeleteArchive deletes an archive
	DeleteArchive(ctx context.Context, archiveID string) error

	// GetArchivalPolicy returns the current archival policy
	GetArchivalPolicy(ctx context.Context) (*ArchivalPolicy, error)

	// SetArchivalPolicy sets the archival policy
	SetArchivalPolicy(ctx context.Context, policy *ArchivalPolicy) error
}

// TransactionalStorage extends Storage for transactional operations
type TransactionalStorage interface {
	Storage

	// Begin begins a transaction
	Begin(ctx context.Context) (Transaction, error)

	// SupportsTransactions returns true if transactions are supported
	SupportsTransactions() bool
}

// Transaction represents a storage transaction
type Transaction interface {
	// Write writes data within the transaction
	Write(ctx context.Context, data *models.TimeSeries) error

	// Read reads data within the transaction
	Read(ctx context.Context, id string) (*models.TimeSeries, error)

	// Delete deletes data within the transaction
	Delete(ctx context.Context, id string) error

	// Commit commits the transaction
	Commit(ctx context.Context) error

	// Rollback rolls back the transaction
	Rollback(ctx context.Context) error

	// IsActive returns true if the transaction is active
	IsActive() bool
}

// StorageFactory creates storage instances
type StorageFactory interface {
	// CreateStorage creates a new storage instance
	CreateStorage(storageType string, config StorageConfig) (Storage, error)

	// GetSupportedTypes returns supported storage types
	GetSupportedTypes() []string

	// RegisterStorage registers a new storage type
	RegisterStorage(storageType string, createFunc StorageCreateFunc) error

	// IsSupported checks if a storage type is supported
	IsSupported(storageType string) bool
}

// StorageCreateFunc is a function that creates a storage instance
type StorageCreateFunc func(config StorageConfig) (Storage, error)

// StorageConfig contains storage configuration
type StorageConfig struct {
	Type             string                 `json:"type"`
	ConnectionString string                 `json:"connection_string"`
	Database         string                 `json:"database,omitempty"`
	Username         string                 `json:"username,omitempty"`
	Password         string                 `json:"password,omitempty"`
	Timeout          time.Duration          `json:"timeout"`
	MaxConnections   int                    `json:"max_connections"`
	RetentionPolicy  string                 `json:"retention_policy,omitempty"`
	BatchSize        int                    `json:"batch_size"`
	Compression      bool                   `json:"compression"`
	TLS              *TLSConfig             `json:"tls,omitempty"`
	Metadata         map[string]interface{} `json:"metadata,omitempty"`
}

// TLSConfig contains TLS configuration
type TLSConfig struct {
	Enabled            bool   `json:"enabled"`
	CertFile           string `json:"cert_file,omitempty"`
	KeyFile            string `json:"key_file,omitempty"`
	CAFile             string `json:"ca_file,omitempty"`
	InsecureSkipVerify bool   `json:"insecure_skip_verify"`
}

// StorageInfo contains information about the storage backend
type StorageInfo struct {
	Type          string                 `json:"type"`
	Version       string                 `json:"version"`
	Name          string                 `json:"name"`
	Description   string                 `json:"description"`
	Features      []string               `json:"features"`
	Capabilities  StorageCapabilities    `json:"capabilities"`
	Configuration map[string]interface{} `json:"configuration,omitempty"`
}

// StorageCapabilities describes what the storage backend supports
type StorageCapabilities struct {
	Streaming      bool `json:"streaming"`
	Transactions   bool `json:"transactions"`
	Compression    bool `json:"compression"`
	Encryption     bool `json:"encryption"`
	Replication    bool `json:"replication"`
	Clustering     bool `json:"clustering"`
	Backup         bool `json:"backup"`
	Archival       bool `json:"archival"`
	TimeBasedQuery bool `json:"time_based_query"`
	Aggregation    bool `json:"aggregation"`
}

// HealthStatus represents storage health status
type HealthStatus struct {
	Status      string                 `json:"status"` // "healthy", "degraded", "unhealthy"
	LastCheck   time.Time              `json:"last_check"`
	Latency     time.Duration          `json:"latency"`
	Connections int                    `json:"connections"`
	Errors      []string               `json:"errors,omitempty"`
	Warnings    []string               `json:"warnings,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// StorageMetrics contains storage performance metrics
type StorageMetrics struct {
	ReadOperations    int64         `json:"read_operations"`
	WriteOperations   int64         `json:"write_operations"`
	DeleteOperations  int64         `json:"delete_operations"`
	AverageReadTime   time.Duration `json:"average_read_time"`
	AverageWriteTime  time.Duration `json:"average_write_time"`
	ErrorCount        int64         `json:"error_count"`
	ConnectionsActive int           `json:"connections_active"`
	ConnectionsIdle   int           `json:"connections_idle"`
	DataSize          int64         `json:"data_size"`
	RecordCount       int64         `json:"record_count"`
	LastError         string        `json:"last_error,omitempty"`
	Uptime            time.Duration `json:"uptime"`
}

// ArchivalResult contains results of an archival operation
type ArchivalResult struct {
	ArchiveID      string    `json:"archive_id"`
	RecordsArchived int64    `json:"records_archived"`
	DataSize       int64     `json:"data_size"`
	CompressionRatio float64 `json:"compression_ratio"`
	StartTime      time.Time `json:"start_time"`
	EndTime        time.Time `json:"end_time"`
	Duration       time.Duration `json:"duration"`
	Location       string    `json:"location"`
}

// RestoreResult contains results of a restore operation
type RestoreResult struct {
	ArchiveID       string        `json:"archive_id"`
	RecordsRestored int64         `json:"records_restored"`
	DataSize        int64         `json:"data_size"`
	Duration        time.Duration `json:"duration"`
	Success         bool          `json:"success"`
	Errors          []string      `json:"errors,omitempty"`
}

// ArchiveInfo contains information about an archive
type ArchiveInfo struct {
	ID               string    `json:"id"`
	Name             string    `json:"name"`
	Description      string    `json:"description,omitempty"`
	CreatedAt        time.Time `json:"created_at"`
	DataTimeRange    TimeRange `json:"data_time_range"`
	RecordCount      int64     `json:"record_count"`
	DataSize         int64     `json:"data_size"`
	CompressionRatio float64   `json:"compression_ratio"`
	Location         string    `json:"location"`
	Checksum         string    `json:"checksum"`
	Status           string    `json:"status"` // "active", "archived", "deleted"
}

// TimeRange represents a time range
type TimeRange struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

// ArchivalPolicy defines how data should be archived
type ArchivalPolicy struct {
	Enabled         bool          `json:"enabled"`
	RetentionPeriod time.Duration `json:"retention_period"`
	ArchiveAfter    time.Duration `json:"archive_after"`
	CompressionType string        `json:"compression_type"`
	Location        string        `json:"location"`
	Schedule        string        `json:"schedule"` // cron expression
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
}

// StoragePool manages a pool of storage connections
type StoragePool interface {
	// Get gets a storage connection from the pool
	Get(ctx context.Context) (Storage, error)

	// Put returns a storage connection to the pool
	Put(storage Storage) error

	// Close closes the pool and all connections
	Close() error

	// Stats returns pool statistics
	Stats() *StoragePoolStats
}

// StoragePoolStats contains storage pool statistics
type StoragePoolStats struct {
	ActiveConnections int `json:"active_connections"`
	IdleConnections   int `json:"idle_connections"`
	TotalCreated      int `json:"total_created"`
	TotalReused       int `json:"total_reused"`
	TotalErrors       int `json:"total_errors"`
}

// StorageMonitor monitors storage performance and health
type StorageMonitor interface {
	// Start starts monitoring
	Start(ctx context.Context) error

	// Stop stops monitoring
	Stop() error

	// GetMetrics returns current metrics
	GetMetrics() *StorageMetrics

	// GetHealth returns current health status
	GetHealth() *HealthStatus

	// Subscribe subscribes to metric updates
	Subscribe() (<-chan *StorageMetrics, error)

	// SetThresholds sets alert thresholds
	SetThresholds(thresholds *StorageThresholds) error
}

// StorageThresholds defines alert thresholds for storage monitoring
type StorageThresholds struct {
	MaxLatency        time.Duration `json:"max_latency"`
	MaxErrorRate      float64       `json:"max_error_rate"`
	MinConnections    int           `json:"min_connections"`
	MaxConnections    int           `json:"max_connections"`
	MaxDiskUsage      float64       `json:"max_disk_usage"`
	MaxMemoryUsage    float64       `json:"max_memory_usage"`
}

// BackupStorage extends Storage for backup operations
type BackupStorage interface {
	Storage

	// Backup creates a backup
	Backup(ctx context.Context, config *BackupConfig) (*BackupResult, error)

	// ListBackups lists available backups
	ListBackups(ctx context.Context) ([]*BackupInfo, error)

	// RestoreBackup restores from a backup
	RestoreBackup(ctx context.Context, backupID string) (*RestoreResult, error)

	// DeleteBackup deletes a backup
	DeleteBackup(ctx context.Context, backupID string) error

	// VerifyBackup verifies backup integrity
	VerifyBackup(ctx context.Context, backupID string) (*VerificationResult, error)
}

// BackupConfig contains backup configuration
type BackupConfig struct {
	Name            string                 `json:"name"`
	Description     string                 `json:"description,omitempty"`
	IncludeData     bool                   `json:"include_data"`
	IncludeSchema   bool                   `json:"include_schema"`
	CompressionType string                 `json:"compression_type"`
	Encryption      *EncryptionConfig      `json:"encryption,omitempty"`
	Location        string                 `json:"location"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
}

// EncryptionConfig contains encryption configuration
type EncryptionConfig struct {
	Enabled   bool   `json:"enabled"`
	Algorithm string `json:"algorithm"`
	KeyFile   string `json:"key_file,omitempty"`
	Password  string `json:"password,omitempty"`
}

// BackupResult contains results of a backup operation
type BackupResult struct {
	BackupID         string        `json:"backup_id"`
	RecordsBacked    int64         `json:"records_backed"`
	DataSize         int64         `json:"data_size"`
	CompressedSize   int64         `json:"compressed_size"`
	CompressionRatio float64       `json:"compression_ratio"`
	Duration         time.Duration `json:"duration"`
	Location         string        `json:"location"`
	Checksum         string        `json:"checksum"`
}

// BackupInfo contains information about a backup
type BackupInfo struct {
	ID               string    `json:"id"`
	Name             string    `json:"name"`
	Description      string    `json:"description,omitempty"`
	CreatedAt        time.Time `json:"created_at"`
	Size             int64     `json:"size"`
	CompressedSize   int64     `json:"compressed_size"`
	CompressionRatio float64   `json:"compression_ratio"`
	Location         string    `json:"location"`
	Checksum         string    `json:"checksum"`
	Status           string    `json:"status"` // "active", "archived", "deleted"
	Encrypted        bool      `json:"encrypted"`
}

// VerificationResult contains results of backup verification
type VerificationResult struct {
	BackupID    string   `json:"backup_id"`
	Valid       bool     `json:"valid"`
	ChecksumOK  bool     `json:"checksum_ok"`
	Readable    bool     `json:"readable"`
	Complete    bool     `json:"complete"`
	Errors      []string `json:"errors,omitempty"`
	Warnings    []string `json:"warnings,omitempty"`
	VerifiedAt  time.Time `json:"verified_at"`
}