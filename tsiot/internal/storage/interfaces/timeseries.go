package interfaces

import (
	"context"
	"time"

	"github.com/inferloop/tsiot/pkg/models"
)

// TimeSeriesStorage defines the interface for time series storage operations
type TimeSeriesStorage interface {
	// Basic operations
	Write(ctx context.Context, data *models.TimeSeries) error
	WriteBatch(ctx context.Context, batch []*models.TimeSeries) error
	Read(ctx context.Context, id string) (*models.TimeSeries, error)
	ReadRange(ctx context.Context, id string, start, end time.Time) (*models.TimeSeries, error)
	Query(ctx context.Context, query *models.TimeSeriesQuery) ([]*models.TimeSeries, error)
	Delete(ctx context.Context, id string) error
	DeleteRange(ctx context.Context, id string, start, end time.Time) error
	List(ctx context.Context, filters map[string]interface{}) ([]*models.TimeSeries, error)
	Count(ctx context.Context, filters map[string]interface{}) (int64, error)

	// Metadata operations
	GetMetadata(ctx context.Context, id string) (map[string]interface{}, error)
	SetMetadata(ctx context.Context, id string, metadata map[string]interface{}) error
	UpdateMetadata(ctx context.Context, id string, updates map[string]interface{}) error

	// Schema operations
	CreateSchema(ctx context.Context, schema *TimeSeriesSchema) error
	GetSchema(ctx context.Context, schemaID string) (*TimeSeriesSchema, error)
	UpdateSchema(ctx context.Context, schema *TimeSeriesSchema) error
	DeleteSchema(ctx context.Context, schemaID string) error
	ListSchemas(ctx context.Context) ([]*TimeSeriesSchema, error)

	// Index operations
	CreateIndex(ctx context.Context, index *TimeSeriesIndex) error
	DeleteIndex(ctx context.Context, indexName string) error
	ListIndexes(ctx context.Context) ([]*TimeSeriesIndex, error)
	RebuildIndex(ctx context.Context, indexName string) error

	// Health and monitoring
	Health(ctx context.Context) (*TimeSeriesHealth, error)
	GetMetrics(ctx context.Context) (*TimeSeriesMetrics, error)
}

// StreamingTimeSeriesStorage extends TimeSeriesStorage with streaming capabilities
type StreamingTimeSeriesStorage interface {
	TimeSeriesStorage

	// WriteStream writes data from a stream
	WriteStream(ctx context.Context, stream <-chan *models.DataPoint) error

	// ReadStream reads data as a stream
	ReadStream(ctx context.Context, query *models.TimeSeriesQuery) (<-chan *models.DataPoint, error)

	// Subscribe subscribes to real-time data updates
	Subscribe(ctx context.Context, pattern string) (<-chan *TimeSeriesEvent, error)

	// Unsubscribe unsubscribes from data updates
	Unsubscribe(ctx context.Context, subscriptionID string) error

	// Publish publishes real-time data
	Publish(ctx context.Context, event *TimeSeriesEvent) error
}

// AggregatedTimeSeriesStorage extends TimeSeriesStorage with aggregation capabilities
type AggregatedTimeSeriesStorage interface {
	TimeSeriesStorage

	// CreateAggregation creates a pre-computed aggregation
	CreateAggregation(ctx context.Context, aggregation *TimeSeriesAggregation) error

	// GetAggregation retrieves aggregated data
	GetAggregation(ctx context.Context, id string, aggregationType string, 
		start, end time.Time, interval time.Duration) (*models.TimeSeries, error)

	// DeleteAggregation removes an aggregation
	DeleteAggregation(ctx context.Context, aggregationID string) error

	// ListAggregations lists available aggregations
	ListAggregations(ctx context.Context) ([]*TimeSeriesAggregation, error)

	// RefreshAggregation manually refreshes an aggregation
	RefreshAggregation(ctx context.Context, aggregationID string) error

	// Aggregate performs on-demand aggregation
	Aggregate(ctx context.Context, query *AggregationQuery) (*models.TimeSeries, error)
}

// CompressedTimeSeriesStorage extends TimeSeriesStorage with compression
type CompressedTimeSeriesStorage interface {
	TimeSeriesStorage

	// WriteCompressed writes compressed time series data
	WriteCompressed(ctx context.Context, data *models.TimeSeries, algorithm string) error

	// ReadDecompressed reads and decompresses time series data
	ReadDecompressed(ctx context.Context, id string) (*models.TimeSeries, error)

	// GetCompressionStats returns compression statistics
	GetCompressionStats(ctx context.Context) (*CompressionStats, error)

	// SetCompressionPolicy sets the compression policy
	SetCompressionPolicy(ctx context.Context, policy *CompressionPolicy) error

	// GetCompressionPolicy gets the current compression policy
	GetCompressionPolicy(ctx context.Context) (*CompressionPolicy, error)
}

// TimeSeriesSchema defines the schema for time series data
type TimeSeriesSchema struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Version     string                 `json:"version"`
	Fields      []*TimeSeriesField     `json:"fields"`
	Tags        []*TimeSeriesTag       `json:"tags"`
	Indexes     []*TimeSeriesIndex     `json:"indexes"`
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// TimeSeriesField defines a field in the time series schema
type TimeSeriesField struct {
	Name        string                 `json:"name"`
	Type        string                 `json:"type"`        // timestamp, float64, int64, string, bool
	Required    bool                   `json:"required"`
	Description string                 `json:"description"`
	Units       string                 `json:"units,omitempty"`
	Range       *ValueRange            `json:"range,omitempty"`
	Precision   int                    `json:"precision,omitempty"`
	Constraints map[string]interface{} `json:"constraints,omitempty"`
}

// ValueRange defines the valid range for a field
type ValueRange struct {
	Min interface{} `json:"min,omitempty"`
	Max interface{} `json:"max,omitempty"`
}

// TimeSeriesTag defines a tag in the time series schema
type TimeSeriesTag struct {
	Name        string   `json:"name"`
	Type        string   `json:"type"`        // string, int, float, bool
	Required    bool     `json:"required"`
	Description string   `json:"description"`
	Values      []string `json:"values,omitempty"` // allowed values
	Indexed     bool     `json:"indexed"`
}

// TimeSeriesIndex defines an index for query optimization
type TimeSeriesIndex struct {
	Name        string                 `json:"name"`
	Type        string                 `json:"type"`        // btree, hash, gin, gist
	Fields      []string               `json:"fields"`
	Unique      bool                   `json:"unique"`
	Partial     string                 `json:"partial,omitempty"` // partial index condition
	Options     map[string]interface{} `json:"options,omitempty"`
	CreatedAt   time.Time              `json:"created_at"`
}

// TimeSeriesEvent represents a real-time time series event
type TimeSeriesEvent struct {
	Type        string                 `json:"type"`        // insert, update, delete
	SeriesID    string                 `json:"series_id"`
	Timestamp   time.Time              `json:"timestamp"`
	DataPoint   *models.DataPoint      `json:"data_point,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	Source      string                 `json:"source"`
}

// TimeSeriesAggregation defines a pre-computed aggregation
type TimeSeriesAggregation struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	SeriesID     string                 `json:"series_id"`
	Function     string                 `json:"function"`     // avg, sum, min, max, count, stddev
	Interval     time.Duration          `json:"interval"`     // aggregation interval
	Window       time.Duration          `json:"window"`       // sliding window size
	GroupBy      []string               `json:"group_by"`     // group by tags
	Filter       map[string]interface{} `json:"filter"`       // filter conditions
	Retention    time.Duration          `json:"retention"`    // how long to keep aggregated data
	Schedule     string                 `json:"schedule"`     // cron schedule for refresh
	Enabled      bool                   `json:"enabled"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
	LastRefresh  time.Time              `json:"last_refresh"`
}

// AggregationQuery defines a query for on-demand aggregation
type AggregationQuery struct {
	SeriesIDs    []string               `json:"series_ids"`
	StartTime    time.Time              `json:"start_time"`
	EndTime      time.Time              `json:"end_time"`
	Function     string                 `json:"function"`     // avg, sum, min, max, count, stddev
	Interval     time.Duration          `json:"interval"`     // aggregation interval
	GroupBy      []string               `json:"group_by"`     // group by tags
	Filter       map[string]interface{} `json:"filter"`       // filter conditions
	Fill         string                 `json:"fill"`         // null, previous, linear, zero
	Limit        int                    `json:"limit"`
	Offset       int                    `json:"offset"`
}

// TimeSeriesHealth represents the health status of time series storage
type TimeSeriesHealth struct {
	Status           string                 `json:"status"` // healthy, degraded, unhealthy
	LastCheck        time.Time              `json:"last_check"`
	ResponseTime     time.Duration          `json:"response_time"`
	StorageSize      int64                  `json:"storage_size"`
	IndexSize        int64                  `json:"index_size"`
	SeriesCount      int64                  `json:"series_count"`
	DataPointCount   int64                  `json:"data_point_count"`
	MemoryUsage      int64                  `json:"memory_usage"`
	DiskUsage        int64                  `json:"disk_usage"`
	ConnectionCount  int                    `json:"connection_count"`
	QueriesPerSecond float64                `json:"queries_per_second"`
	WritesPerSecond  float64                `json:"writes_per_second"`
	Errors           []string               `json:"errors,omitempty"`
	Warnings         []string               `json:"warnings,omitempty"`
	Details          map[string]interface{} `json:"details,omitempty"`
}

// TimeSeriesMetrics contains detailed performance metrics
type TimeSeriesMetrics struct {
	// Operation counts
	ReadOperations    int64 `json:"read_operations"`
	WriteOperations   int64 `json:"write_operations"`
	DeleteOperations  int64 `json:"delete_operations"`
	QueryOperations   int64 `json:"query_operations"`

	// Performance metrics
	AverageReadTime   time.Duration `json:"average_read_time"`
	AverageWriteTime  time.Duration `json:"average_write_time"`
	AverageQueryTime  time.Duration `json:"average_query_time"`
	P95ReadTime       time.Duration `json:"p95_read_time"`
	P95WriteTime      time.Duration `json:"p95_write_time"`
	P95QueryTime      time.Duration `json:"p95_query_time"`

	// Error metrics
	ErrorCount        int64   `json:"error_count"`
	ErrorRate         float64 `json:"error_rate"`
	TimeoutCount      int64   `json:"timeout_count"`
	RetryCount        int64   `json:"retry_count"`

	// Resource metrics
	ConnectionsActive int   `json:"connections_active"`
	ConnectionsIdle   int   `json:"connections_idle"`
	MemoryUsage       int64 `json:"memory_usage"`
	DiskUsage         int64 `json:"disk_usage"`
	NetworkBytesIn    int64 `json:"network_bytes_in"`
	NetworkBytesOut   int64 `json:"network_bytes_out"`

	// Data metrics
	SeriesCount       int64 `json:"series_count"`
	DataPointCount    int64 `json:"data_point_count"`
	IndexCount        int64 `json:"index_count"`
	CompressionRatio  float64 `json:"compression_ratio"`

	// Cache metrics
	CacheHitRate      float64 `json:"cache_hit_rate"`
	CacheMissRate     float64 `json:"cache_miss_rate"`
	CacheEvictions    int64   `json:"cache_evictions"`

	// Query metrics
	SlowQueries       int64   `json:"slow_queries"`
	FullTableScans    int64   `json:"full_table_scans"`
	IndexHitRate      float64 `json:"index_hit_rate"`

	// Aggregation metrics
	AggregationCount  int64 `json:"aggregation_count"`
	AggregationTime   time.Duration `json:"aggregation_time"`

	// Replication metrics (if applicable)
	ReplicationLag    time.Duration `json:"replication_lag,omitempty"`
	ReplicationErrors int64         `json:"replication_errors,omitempty"`

	// Timestamp of metrics collection
	CollectedAt       time.Time `json:"collected_at"`
	Uptime            time.Duration `json:"uptime"`
}

// CompressionStats contains compression-related statistics
type CompressionStats struct {
	Algorithm         string    `json:"algorithm"`
	CompressedSize    int64     `json:"compressed_size"`
	UncompressedSize  int64     `json:"uncompressed_size"`
	CompressionRatio  float64   `json:"compression_ratio"`
	CompressionTime   time.Duration `json:"compression_time"`
	DecompressionTime time.Duration `json:"decomprression_time"`
	CompressedSeries  int64     `json:"compressed_series"`
	UncompressedSeries int64    `json:"uncompressed_series"`
	SpaceSaved        int64     `json:"space_saved"`
	LastCompression   time.Time `json:"last_compression"`
}

// CompressionPolicy defines compression settings for time series data
type CompressionPolicy struct {
	Enabled           bool          `json:"enabled"`
	Algorithm         string        `json:"algorithm"`         // gzip, lz4, snappy, zstd
	CompressionLevel  int           `json:"compression_level"` // algorithm-specific level
	MinAge            time.Duration `json:"min_age"`           // minimum age before compression
	MinSize           int64         `json:"min_size"`          // minimum size before compression
	ChunkSize         int64         `json:"chunk_size"`        // compression chunk size
	Schedule          string        `json:"schedule"`          // cron schedule for compression
	AutoCompress      bool          `json:"auto_compress"`     // automatic compression
	PreserveOriginal  bool          `json:"preserve_original"` // keep uncompressed copy
}

// TimeSeriesRetentionPolicy defines data retention settings
type TimeSeriesRetentionPolicy struct {
	ID            string        `json:"id"`
	Name          string        `json:"name"`
	Duration      time.Duration `json:"duration"`      // how long to keep data
	Resolution    time.Duration `json:"resolution"`    // data resolution/granularity
	Downsampling  *DownsamplingConfig `json:"downsampling,omitempty"`
	SeriesPattern string        `json:"series_pattern"` // pattern to match series
	TagFilters    map[string]string `json:"tag_filters,omitempty"`
	Action        string        `json:"action"`        // delete, archive, downsample
	Enabled       bool          `json:"enabled"`
	CreatedAt     time.Time     `json:"created_at"`
	UpdatedAt     time.Time     `json:"updated_at"`
}

// DownsamplingConfig defines downsampling parameters
type DownsamplingConfig struct {
	Interval     time.Duration `json:"interval"`     // downsampling interval
	Function     string        `json:"function"`     // avg, sum, min, max, last, first
	FillPolicy   string        `json:"fill_policy"`  // null, previous, linear, zero
	KeepOriginal bool          `json:"keep_original"` // whether to keep original data
}

// TimeSeriesBackup represents a backup operation
type TimeSeriesBackup struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Type         string                 `json:"type"`         // full, incremental, differential
	Status       string                 `json:"status"`       // pending, running, completed, failed
	StartTime    time.Time              `json:"start_time"`
	EndTime      time.Time              `json:"end_time"`
	Duration     time.Duration          `json:"duration"`
	Size         int64                  `json:"size"`
	SeriesCount  int64                  `json:"series_count"`
	Location     string                 `json:"location"`     // backup storage location
	Compression  string                 `json:"compression"`  // compression algorithm used
	Encryption   bool                   `json:"encryption"`   // whether backup is encrypted
	Checksum     string                 `json:"checksum"`     // backup integrity checksum
	Metadata     map[string]interface{} `json:"metadata"`
	ErrorMessage string                 `json:"error_message,omitempty"`
}

// TimeSeriesRestore represents a restore operation
type TimeSeriesRestore struct {
	ID           string                 `json:"id"`
	BackupID     string                 `json:"backup_id"`
	Status       string                 `json:"status"`       // pending, running, completed, failed
	StartTime    time.Time              `json:"start_time"`
	EndTime      time.Time              `json:"end_time"`
	Duration     time.Duration          `json:"duration"`
	SeriesCount  int64                  `json:"series_count"`
	Destination  string                 `json:"destination"`  // restore destination
	Overwrite    bool                   `json:"overwrite"`    // whether to overwrite existing data
	Metadata     map[string]interface{} `json:"metadata"`
	ErrorMessage string                 `json:"error_message,omitempty"`
}