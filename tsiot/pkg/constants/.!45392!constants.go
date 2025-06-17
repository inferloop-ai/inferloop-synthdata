package constants

import "time"

// Application constants
const (
	// Application metadata
	AppName        = "tsiot-server"
	AppDescription = "Time Series Synthetic Data MCP Server"
	AppVersion     = "0.1.0"
	
	// API constants
	APIVersion = "v1"
	APIPrefix  = "/api/v1"
	
	// Default configuration values
	DefaultPort                = 8080
	DefaultMetricsPort         = 9090
	DefaultHost                = "0.0.0.0"
	DefaultLogLevel            = "info"
	DefaultLogFormat           = "json"
	DefaultReadTimeout         = 15 * time.Second
	DefaultWriteTimeout        = 15 * time.Second
	DefaultIdleTimeout         = 60 * time.Second
	DefaultShutdownTimeout     = 30 * time.Second
	
	// Generation defaults
	DefaultBatchSize           = 1000
	DefaultEpochs              = 1000
	DefaultLearningRate        = 0.001
	DefaultSequenceLength      = 24
	DefaultHiddenDim           = 24
	DefaultNumLayers           = 3
	DefaultQualityThreshold    = 0.8
	
	// Storage defaults
	DefaultRetentionPeriod     = "30d"
	DefaultStorageTimeout      = 30 * time.Second
	DefaultMaxConnections      = 100
	DefaultConnectionTimeout   = 10 * time.Second
	
	// Worker defaults
	DefaultWorkerConcurrency   = 4
	DefaultWorkerPollInterval  = 5 * time.Second
	DefaultMaxRetries          = 3
	DefaultRetryDelay          = 1 * time.Second
	DefaultJobTimeout          = 30 * time.Minute
	
	// Rate limiting defaults
	DefaultRateLimit           = 100  // requests per minute
	DefaultBurstLimit          = 200
	
	// Privacy defaults
	DefaultEpsilon             = 1.0
	DefaultDelta               = 1e-5
	DefaultPrivacyBudget       = 1.0
	
	// MCP defaults
	DefaultMCPTransport        = "stdio"
	DefaultMCPMaxMessageSize   = 1048576 // 1MB
	DefaultMCPWebSocketPath    = "/mcp"
	
	// Validation defaults
	DefaultValidationMetrics   = "basic,statistical"
	DefaultStatisticalAlpha    = 0.05
	
	// File size limits
	MaxUploadSize              = 100 * 1024 * 1024 // 100MB
	MaxGenerationSize          = 1000000           // 1M data points
	MaxBatchSize               = 10000
	
	// Pagination defaults
	DefaultPageSize            = 100
	MaxPageSize                = 1000
	
	// Cache defaults
	DefaultCacheTTL            = 1 * time.Hour
	DefaultCacheSize           = 1000
)

// HTTP headers
const (
	HeaderContentType          = "Content-Type"
	HeaderAccept               = "Accept"
	HeaderAuthorization        = "Authorization"
	HeaderUserAgent            = "User-Agent"
	HeaderRequestID            = "X-Request-ID"
	HeaderCorrelationID        = "X-Correlation-ID"
	HeaderForwardedFor         = "X-Forwarded-For"
	HeaderRealIP               = "X-Real-IP"
	HeaderRateLimit            = "X-RateLimit-Limit"
	HeaderRateLimitRemaining   = "X-RateLimit-Remaining"
	HeaderRateLimitReset       = "X-RateLimit-Reset"
	HeaderRetryAfter           = "Retry-After"
	HeaderCacheControl         = "Cache-Control"
	HeaderETag                 = "ETag"
	HeaderLastModified         = "Last-Modified"
	HeaderLocation             = "Location"
)

// Content types
const (
	ContentTypeJSON            = "application/json"
	ContentTypeXML             = "application/xml"
	ContentTypeCSV             = "text/csv"
	ContentTypePlainText       = "text/plain"
	ContentTypeHTML            = "text/html"
	ContentTypeFormData        = "application/x-www-form-urlencoded"
	ContentTypeMultipartForm   = "multipart/form-data"
	ContentTypeOctetStream     = "application/octet-stream"
	ContentTypeParquet         = "application/parquet"
	ContentTypeAvro            = "application/avro"
)

// HTTP status codes (commonly used)
const (
	StatusOK                   = 200
	StatusCreated              = 201
	StatusAccepted             = 202
	StatusNoContent            = 204
	StatusBadRequest           = 400
	StatusUnauthorized         = 401
	StatusForbidden            = 403
	StatusNotFound             = 404
	StatusMethodNotAllowed     = 405
	StatusConflict             = 409
	StatusUnprocessableEntity  = 422
	StatusTooManyRequests      = 429
	StatusInternalServerError  = 500
	StatusNotImplemented       = 501
	StatusBadGateway           = 502
	StatusServiceUnavailable   = 503
	StatusGatewayTimeout       = 504
)

// Environment names
const (
	EnvDevelopment = "development"
	EnvTesting     = "testing"
	EnvStaging     = "staging"
	EnvProduction  = "production"
)

// Log levels
const (
	LogLevelDebug = "debug"
	LogLevelInfo  = "info"
	LogLevelWarn  = "warn"
	LogLevelError = "error"
	LogLevelFatal = "fatal"
)

// Log formats
const (
	LogFormatJSON = "json"
	LogFormatText = "text"
)

// Generator types
const (
	GeneratorTypeTimeGAN     = "timegan"
	GeneratorTypeARIMA       = "arima"
	GeneratorTypeRNN         = "rnn"
	GeneratorTypeLSTM        = "lstm"
	GeneratorTypeGRU         = "gru"
	GeneratorTypeStatistical = "statistical"
	GeneratorTypeYData       = "ydata"
)

// Sensor types
const (
	SensorTypeTemperature = "temperature"
	SensorTypeHumidity    = "humidity"
	SensorTypePressure    = "pressure"
	SensorTypeVibration   = "vibration"
	SensorTypePower       = "power"
	SensorTypeFlow        = "flow"
	SensorTypeLevel       = "level"
	SensorTypeSpeed       = "speed"
	SensorTypeCustom      = "custom"
)

// Storage backends
const (
	StorageTypeInfluxDB    = "influxdb"
	StorageTypeTimescaleDB = "timescaledb"
	StorageTypeClickhouse  = "clickhouse"
	StorageTypeS3          = "s3"
	StorageTypeFile        = "file"
	StorageTypeRedis       = "redis"
)

// Output formats
const (
	OutputFormatCSV     = "csv"
	OutputFormatJSON    = "json"
	OutputFormatParquet = "parquet"
	OutputFormatAvro    = "avro"
	OutputFormatInflux  = "influx"
	OutputFormatHDF5    = "hdf5"
)

// Compression types
const (
	CompressionNone   = "none"
	CompressionGzip   = "gzip"
	CompressionBZ2    = "bz2"
	CompressionXZ     = "xz"
	CompressionSnappy = "snappy"
	CompressionLZ4    = "lz4"
	CompressionZstd   = "zstd"
)

// Validation metrics
const (
	MetricBasic        = "basic"
	MetricStatistical  = "statistical"
	MetricTemporal     = "temporal"
	MetricDistribution = "distribution"
	MetricPrivacy      = "privacy"
	MetricQuality      = "quality"
)

// Statistical tests
const (
	TestKolmogorovSmirnov     = "ks"
	TestAndersonDarling       = "anderson"
	TestLjungBox              = "ljung_box"
	TestAugmentedDickeyFuller = "adf"
	TestShapiro               = "shapiro"
	TestJarqueBera            = "jarque_bera"
)

// Privacy techniques
const (
	PrivacyDifferentialPrivacy = "differential_privacy"
	PrivacyKAnonymity          = "k_anonymity"
	PrivacyLDiversity          = "l_diversity"
	PrivacyTCloseness          = "t_closeness"
	PrivacyDataMasking         = "data_masking"
	PrivacyFederatedLearning   = "federated_learning"
)

// Job statuses
const (
	JobStatusPending   = "pending"
	JobStatusQueued    = "queued"
	JobStatusRunning   = "running"
	JobStatusCompleted = "completed"
	JobStatusFailed    = "failed"
	JobStatusCancelled = "cancelled"
	JobStatusPaused    = "paused"
)

// Job priorities
const (
	PriorityLow    = "low"
	PriorityNormal = "normal"
	PriorityHigh   = "high"
	PriorityUrgent = "urgent"
)

// Quality levels
const (
	QualityUnknown   = "unknown"
	QualityPoor      = "poor"
	QualityFair      = "fair"
	QualityGood      = "good"
	QualityExcellent = "excellent"
)

// Report formats
const (
	ReportFormatJSON = "json"
	ReportFormatHTML = "html"
	ReportFormatPDF  = "pdf"
	ReportFormatText = "text"
)

// MCP protocol constants
const (
	MCPTransportStdio     = "stdio"
	MCPTransportWebSocket = "websocket"
	MCPTransportSSE       = "sse"
	
	MCPMethodInitialize        = "initialize"
	MCPMethodListTools         = "tools/list"
	MCPMethodCallTool          = "tools/call"
	MCPMethodListResources     = "resources/list"
	MCPMethodReadResource      = "resources/read"
	MCPMethodListPrompts       = "prompts/list"
	MCPMethodGetPrompt         = "prompts/get"
	MCPMethodSetLevel          = "logging/setLevel"
	MCPMethodProgress          = "notifications/progress"
	MCPMethodCancelled         = "notifications/cancelled"
	MCPMethodRootsListChanged  = "notifications/roots/list_changed"
)

// Agent types
const (
	AgentTypeGeneration   = "generation"
	AgentTypeValidation   = "validation"
	AgentTypeAnalysis     = "analysis"
	AgentTypePrivacy      = "privacy"
	AgentTypeForecasting  = "forecasting"
	AgentTypeAnomaly      = "anomaly"
	AgentTypeCoordinator  = "coordinator"
)

// Regular expressions for validation
const (
	RegexEmail        = `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`
	RegexUUID         = `^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`
	RegexSensorID     = `^[a-zA-Z0-9_-]+$`
	RegexTimestamp    = `^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z?$`
