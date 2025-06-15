package clickhouse

import (
	"context"
	"database/sql"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ClickHouse/clickhouse-go/v2"
	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
	"github.com/inferloop/tsiot/pkg/errors"
)

// ClickHouseStorageConfig contains configuration for ClickHouse storage
type ClickHouseStorageConfig struct {
	Host            string        `json:"host" yaml:"host"`
	Port            int           `json:"port" yaml:"port"`
	Database        string        `json:"database" yaml:"database"`
	Username        string        `json:"username" yaml:"username"`
	Password        string        `json:"password" yaml:"password"`
	SSLMode         string        `json:"ssl_mode" yaml:"ssl_mode"`
	ConnectTimeout  time.Duration `json:"connect_timeout" yaml:"connect_timeout"`
	QueryTimeout    time.Duration `json:"query_timeout" yaml:"query_timeout"`
	MaxConnections  int           `json:"max_connections" yaml:"max_connections"`
	MaxIdleConns    int           `json:"max_idle_conns" yaml:"max_idle_conns"`
	ConnMaxLifetime time.Duration `json:"conn_max_lifetime" yaml:"conn_max_lifetime"`
	
	// ClickHouse specific settings
	Compression       string `json:"compression" yaml:"compression"`         // "lz4", "zstd", "none"
	BatchSize         int    `json:"batch_size" yaml:"batch_size"`           // rows per batch insert
	FlushInterval     time.Duration `json:"flush_interval" yaml:"flush_interval"` // auto-flush interval
	Engine            string `json:"engine" yaml:"engine"`                   // "MergeTree", "ReplicatedMergeTree"
	PartitionBy       string `json:"partition_by" yaml:"partition_by"`       // partition expression
	OrderBy           string `json:"order_by" yaml:"order_by"`               // order by expression
	TTL               string `json:"ttl" yaml:"ttl"`                         // data TTL expression
	ReplicationFactor int    `json:"replication_factor" yaml:"replication_factor"`
	
	// Performance settings
	MaxBlockSize      int `json:"max_block_size" yaml:"max_block_size"`
	MaxInsertThreads  int `json:"max_insert_threads" yaml:"max_insert_threads"`
	UseAsyncInserts   bool `json:"use_async_inserts" yaml:"use_async_inserts"`
}

// ClickHouseStorage implements the Storage interface for ClickHouse
type ClickHouseStorage struct {
	config    *ClickHouseStorageConfig
	logger    *logrus.Logger
	conn      clickhouse.Conn
	db        *sql.DB
	mu        sync.RWMutex
	connected bool
	
	// Batch processing
	batchMu       sync.Mutex
	pendingBatch  []batchItem
	lastFlush     time.Time
	flushTimer    *time.Timer
}

type batchItem struct {
	timeSeries *models.TimeSeries
	timestamp  time.Time
}

// NewClickHouseStorage creates a new ClickHouse storage instance
func NewClickHouseStorage(config *ClickHouseStorageConfig, logger *logrus.Logger) (*ClickHouseStorage, error) {
	if config == nil {
		return nil, errors.NewValidationError("INVALID_CONFIG", "ClickHouseStorageConfig cannot be nil")
	}

	if config.Host == "" {
		config.Host = "localhost"
	}

	if config.Port == 0 {
		config.Port = 9000
	}

	if config.Database == "" {
		config.Database = "default"
	}

	if config.ConnectTimeout == 0 {
		config.ConnectTimeout = 10 * time.Second
	}

	if config.QueryTimeout == 0 {
		config.QueryTimeout = 30 * time.Second
	}

	if config.MaxConnections == 0 {
		config.MaxConnections = 10
	}

	if config.BatchSize == 0 {
		config.BatchSize = 10000
	}

	if config.FlushInterval == 0 {
		config.FlushInterval = 30 * time.Second
	}

	if config.Engine == "" {
		config.Engine = "MergeTree"
	}

	if config.OrderBy == "" {
		config.OrderBy = "(series_id, timestamp)"
	}

	if config.PartitionBy == "" {
		config.PartitionBy = "toYYYYMM(timestamp)"
	}

	if logger == nil {
		logger = logrus.New()
	}

	return &ClickHouseStorage{
		config:       config,
		logger:       logger,
		connected:    false,
		pendingBatch: make([]batchItem, 0, config.BatchSize),
		lastFlush:    time.Now(),
	}, nil
}

// Connect establishes connection to ClickHouse
func (ch *ClickHouseStorage) Connect(ctx context.Context) error {
	ch.mu.Lock()
	defer ch.mu.Unlock()

	if ch.connected {
		return nil
	}

	// Build connection options
	options := &clickhouse.Options{
		Addr: []string{fmt.Sprintf("%s:%d", ch.config.Host, ch.config.Port)},
		Auth: clickhouse.Auth{
			Database: ch.config.Database,
			Username: ch.config.Username,
			Password: ch.config.Password,
		},
		ConnMaxLifetime: ch.config.ConnMaxLifetime,
		MaxOpenConns:    ch.config.MaxConnections,
		MaxIdleConns:    ch.config.MaxIdleConns,
		ConnOpenStrategy: clickhouse.ConnOpenInOrder,
		
		// Performance settings
		BlockBufferSize: 10,
		MaxCompressionBuffer: 10240,
	}

	// Set compression
	switch strings.ToLower(ch.config.Compression) {
	case "lz4":
		options.Compression = &clickhouse.Compression{
			Method: clickhouse.CompressionLZ4,
		}
	case "zstd":
		options.Compression = &clickhouse.Compression{
			Method: clickhouse.CompressionZSTD,
		}
	}

	// Set timeouts
	if ch.config.ConnectTimeout > 0 {
		options.DialTimeout = ch.config.ConnectTimeout
	}

	// Create connection
	conn, err := clickhouse.Open(options)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "CONNECTION_FAILED", 
			fmt.Sprintf("Failed to connect to ClickHouse at %s:%d", ch.config.Host, ch.config.Port))
	}

	// Test connection
	if err := conn.Ping(ctx); err != nil {
		conn.Close()
		return errors.WrapError(err, errors.ErrorTypeStorage, "PING_FAILED", "Failed to ping ClickHouse")
	}

	ch.conn = conn

	// Open SQL interface for DDL operations
	sqlConn := clickhouse.OpenDB(options)
	sqlConn.SetMaxOpenConns(ch.config.MaxConnections)
	sqlConn.SetMaxIdleConns(ch.config.MaxIdleConns)
	sqlConn.SetConnMaxLifetime(ch.config.ConnMaxLifetime)
	ch.db = sqlConn

	// Initialize database schema
	if err := ch.initializeSchema(ctx); err != nil {
		ch.cleanup()
		return errors.WrapError(err, errors.ErrorTypeStorage, "SCHEMA_INIT_FAILED", "Failed to initialize schema")
	}

	// Start batch flushing routine
	ch.startBatchFlusher(ctx)

	ch.connected = true
	ch.logger.WithFields(logrus.Fields{
		"host":     ch.config.Host,
		"port":     ch.config.Port,
		"database": ch.config.Database,
	}).Info("ClickHouse storage connected")

	return nil
}

// Close closes the ClickHouse connection
func (ch *ClickHouseStorage) Close() error {
	ch.mu.Lock()
	defer ch.mu.Unlock()

	if !ch.connected {
		return nil
	}

	// Flush any pending batches
	ch.flushPendingBatches()

	// Stop flush timer
	if ch.flushTimer != nil {
		ch.flushTimer.Stop()
	}

	ch.cleanup()
	ch.connected = false

	ch.logger.Info("ClickHouse storage disconnected")
	return nil
}

// HealthCheck verifies the ClickHouse connection
func (ch *ClickHouseStorage) HealthCheck(ctx context.Context) error {
	if !ch.connected {
		return errors.NewStorageError("NOT_CONNECTED", "ClickHouse storage is not connected")
	}

	if err := ch.conn.Ping(ctx); err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "HEALTH_CHECK_FAILED", "ClickHouse ping failed")
	}

	return nil
}

// Write stores a time series in ClickHouse
func (ch *ClickHouseStorage) Write(ctx context.Context, timeSeries *models.TimeSeries) error {
	if !ch.connected {
		return errors.NewStorageError("NOT_CONNECTED", "ClickHouse storage is not connected")
	}

	if timeSeries == nil {
		return errors.NewValidationError("INVALID_INPUT", "TimeSeries cannot be nil")
	}

	// Use batch processing for better performance
	if ch.config.UseAsyncInserts {
		return ch.addToBatch(timeSeries)
	}

	// Direct insert for synchronous writes
	return ch.insertTimeSeries(ctx, timeSeries)
}

// Read retrieves a time series from ClickHouse
func (ch *ClickHouseStorage) Read(ctx context.Context, seriesID string) (*models.TimeSeries, error) {
	if !ch.connected {
		return nil, errors.NewStorageError("NOT_CONNECTED", "ClickHouse storage is not connected")
	}

	query := `
		SELECT series_id, name, description, timestamp, value, quality, 
		       sensor_type, frequency, tags, metadata,
		       created_at, updated_at
		FROM time_series_data 
		WHERE series_id = ?
		ORDER BY timestamp ASC
	`

	rows, err := ch.db.QueryContext(ctx, query, seriesID)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "QUERY_FAILED", 
			fmt.Sprintf("Failed to query series: %s", seriesID))
	}
	defer rows.Close()

	var dataPoints []models.DataPoint
	var timeSeries *models.TimeSeries

	for rows.Next() {
		var (
			id, name, description, sensorType, frequency, tags, metadata string
			timestamp, createdAt, updatedAt time.Time
			value, quality float64
		)

		err := rows.Scan(&id, &name, &description, &timestamp, &value, &quality,
			&sensorType, &frequency, &tags, &metadata, &createdAt, &updatedAt)
		if err != nil {
			continue
		}

		// Create time series object on first row
		if timeSeries == nil {
			timeSeries = &models.TimeSeries{
				ID:          id,
				Name:        name,
				Description: description,
				SensorType:  models.SensorType(sensorType),
				Frequency:   frequency,
				CreatedAt:   createdAt,
				UpdatedAt:   updatedAt,
				DataPoints:  make([]models.DataPoint, 0),
			}
		}

		dataPoints = append(dataPoints, models.DataPoint{
			Timestamp: timestamp,
			Value:     value,
			Quality:   quality,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "ROW_SCAN_FAILED", "Failed to scan rows")
	}

	if timeSeries == nil {
		return nil, errors.NewStorageError("NOT_FOUND", fmt.Sprintf("Series not found: %s", seriesID))
	}

	timeSeries.DataPoints = dataPoints
	if len(dataPoints) > 0 {
		timeSeries.StartTime = dataPoints[0].Timestamp
		timeSeries.EndTime = dataPoints[len(dataPoints)-1].Timestamp
	}

	return timeSeries, nil
}

// Delete removes a time series from ClickHouse
func (ch *ClickHouseStorage) Delete(ctx context.Context, seriesID string) error {
	if !ch.connected {
		return errors.NewStorageError("NOT_CONNECTED", "ClickHouse storage is not connected")
	}

	// In ClickHouse, we use mutations for deletes (which are async)
	query := `ALTER TABLE time_series_data DELETE WHERE series_id = ?`

	_, err := ch.db.ExecContext(ctx, query, seriesID)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "DELETE_FAILED", 
			fmt.Sprintf("Failed to delete series: %s", seriesID))
	}

	ch.logger.WithField("series_id", seriesID).Info("Time series delete initiated")
	return nil
}

// List returns a list of available time series
func (ch *ClickHouseStorage) List(ctx context.Context, limit, offset int) ([]*models.TimeSeries, error) {
	if !ch.connected {
		return nil, errors.NewStorageError("NOT_CONNECTED", "ClickHouse storage is not connected")
	}

	query := `
		SELECT DISTINCT series_id, name, description, sensor_type, frequency,
		                min(timestamp) as start_time, max(timestamp) as end_time,
		                count(*) as data_points
		FROM time_series_data
		GROUP BY series_id, name, description, sensor_type, frequency
		ORDER BY series_id
		LIMIT ? OFFSET ?
	`

	rows, err := ch.db.QueryContext(ctx, query, limit, offset)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "LIST_QUERY_FAILED", "Failed to list time series")
	}
	defer rows.Close()

	var seriesList []*models.TimeSeries

	for rows.Next() {
		var (
			id, name, description, sensorType, frequency string
			startTime, endTime time.Time
			dataPoints int64
		)

		err := rows.Scan(&id, &name, &description, &sensorType, &frequency, 
			&startTime, &endTime, &dataPoints)
		if err != nil {
			continue
		}

		series := &models.TimeSeries{
			ID:          id,
			Name:        name,
			Description: description,
			SensorType:  models.SensorType(sensorType),
			Frequency:   frequency,
			StartTime:   startTime,
			EndTime:     endTime,
			DataPoints:  make([]models.DataPoint, 0), // Don't load actual data points in list
		}

		seriesList = append(seriesList, series)
	}

	if err := rows.Err(); err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "ROW_SCAN_FAILED", "Failed to scan rows")
	}

	return seriesList, nil
}

// Helper methods

func (ch *ClickHouseStorage) initializeSchema(ctx context.Context) error {
	// Create the main time series table
	createTableSQL := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS time_series_data (
			series_id String,
			name String,
			description String,
			timestamp DateTime64(9),
			value Float64,
			quality Float64,
			sensor_type String,
			frequency String,
			tags String,
			metadata String,
			created_at DateTime,
			updated_at DateTime
		) ENGINE = %s
		PARTITION BY %s
		ORDER BY %s
		%s
	`, ch.config.Engine, ch.config.PartitionBy, ch.config.OrderBy, ch.getTTLClause())

	if _, err := ch.db.ExecContext(ctx, createTableSQL); err != nil {
		return fmt.Errorf("failed to create time_series_data table: %w", err)
	}

	// Create indexes for better query performance
	indexQueries := []string{
		`CREATE INDEX IF NOT EXISTS idx_series_timestamp ON time_series_data (series_id, timestamp)`,
		`CREATE INDEX IF NOT EXISTS idx_timestamp ON time_series_data (timestamp)`,
		`CREATE INDEX IF NOT EXISTS idx_sensor_type ON time_series_data (sensor_type)`,
	}

	for _, query := range indexQueries {
		if _, err := ch.db.ExecContext(ctx, query); err != nil {
			ch.logger.WithError(err).WithField("query", query).Warn("Failed to create index")
		}
	}

	ch.logger.Info("ClickHouse schema initialized")
	return nil
}

func (ch *ClickHouseStorage) getTTLClause() string {
	if ch.config.TTL != "" {
		return fmt.Sprintf("TTL timestamp + INTERVAL %s", ch.config.TTL)
	}
	return ""
}

func (ch *ClickHouseStorage) insertTimeSeries(ctx context.Context, timeSeries *models.TimeSeries) error {
	if len(timeSeries.DataPoints) == 0 {
		return nil
	}

	// Prepare batch insert
	batch, err := ch.conn.PrepareBatch(ctx, `
		INSERT INTO time_series_data (
			series_id, name, description, timestamp, value, quality,
			sensor_type, frequency, tags, metadata, created_at, updated_at
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to prepare batch: %w", err)
	}

	// Add data points to batch
	for _, dp := range timeSeries.DataPoints {
		err := batch.Append(
			timeSeries.ID,
			timeSeries.Name,
			timeSeries.Description,
			dp.Timestamp,
			dp.Value,
			dp.Quality,
			string(timeSeries.SensorType),
			timeSeries.Frequency,
			ch.serializeTags(timeSeries.Tags),
			ch.serializeMetadata(timeSeries.Metadata),
			timeSeries.CreatedAt,
			timeSeries.UpdatedAt,
		)
		if err != nil {
			return fmt.Errorf("failed to append to batch: %w", err)
		}
	}

	// Execute batch
	if err := batch.Send(); err != nil {
		return fmt.Errorf("failed to send batch: %w", err)
	}

	ch.logger.WithFields(logrus.Fields{
		"series_id":    timeSeries.ID,
		"data_points":  len(timeSeries.DataPoints),
	}).Debug("Time series inserted")

	return nil
}

func (ch *ClickHouseStorage) addToBatch(timeSeries *models.TimeSeries) error {
	ch.batchMu.Lock()
	defer ch.batchMu.Unlock()

	ch.pendingBatch = append(ch.pendingBatch, batchItem{
		timeSeries: timeSeries,
		timestamp:  time.Now(),
	})

	// Auto-flush if batch is full
	if len(ch.pendingBatch) >= ch.config.BatchSize {
		go func() {
			ctx, cancel := context.WithTimeout(context.Background(), ch.config.QueryTimeout)
			defer cancel()
			ch.flushBatch(ctx)
		}()
	}

	return nil
}

func (ch *ClickHouseStorage) startBatchFlusher(ctx context.Context) {
	ch.flushTimer = time.AfterFunc(ch.config.FlushInterval, func() {
		flushCtx, cancel := context.WithTimeout(context.Background(), ch.config.QueryTimeout)
		defer cancel()
		
		ch.flushBatch(flushCtx)
		
		// Schedule next flush
		if ch.connected {
			ch.flushTimer = time.AfterFunc(ch.config.FlushInterval, func() {
				ch.startBatchFlusher(ctx)
			})
		}
	})
}

func (ch *ClickHouseStorage) flushBatch(ctx context.Context) {
	ch.batchMu.Lock()
	
	if len(ch.pendingBatch) == 0 {
		ch.batchMu.Unlock()
		return
	}
	
	batch := ch.pendingBatch
	ch.pendingBatch = make([]batchItem, 0, ch.config.BatchSize)
	ch.batchMu.Unlock()

	// Group by series ID for efficient insertion
	seriesMap := make(map[string]*models.TimeSeries)
	
	for _, item := range batch {
		ts := item.timeSeries
		if existing, exists := seriesMap[ts.ID]; exists {
			existing.DataPoints = append(existing.DataPoints, ts.DataPoints...)
		} else {
			seriesMap[ts.ID] = &models.TimeSeries{
				ID:          ts.ID,
				Name:        ts.Name,
				Description: ts.Description,
				SensorType:  ts.SensorType,
				Frequency:   ts.Frequency,
				Tags:        ts.Tags,
				Metadata:    ts.Metadata,
				CreatedAt:   ts.CreatedAt,
				UpdatedAt:   ts.UpdatedAt,
				DataPoints:  append([]models.DataPoint{}, ts.DataPoints...),
			}
		}
	}

	// Insert each series
	for _, ts := range seriesMap {
		if err := ch.insertTimeSeries(ctx, ts); err != nil {
			ch.logger.WithError(err).WithField("series_id", ts.ID).Error("Failed to flush batch")
		}
	}

	ch.logger.WithField("batch_size", len(batch)).Debug("Batch flushed")
	ch.lastFlush = time.Now()
}

func (ch *ClickHouseStorage) flushPendingBatches() {
	ctx, cancel := context.WithTimeout(context.Background(), ch.config.QueryTimeout)
	defer cancel()
	ch.flushBatch(ctx)
}

func (ch *ClickHouseStorage) serializeTags(tags map[string]string) string {
	if len(tags) == 0 {
		return ""
	}
	
	var parts []string
	for k, v := range tags {
		parts = append(parts, fmt.Sprintf("%s=%s", k, v))
	}
	return strings.Join(parts, ",")
}

func (ch *ClickHouseStorage) serializeMetadata(metadata models.Metadata) string {
	// Simple JSON-like serialization
	// In production, you might want to use proper JSON marshaling
	return fmt.Sprintf("generator=%s,version=%s", metadata.Generator, metadata.Version)
}

func (ch *ClickHouseStorage) cleanup() {
	if ch.conn != nil {
		ch.conn.Close()
		ch.conn = nil
	}
	if ch.db != nil {
		ch.db.Close()
		ch.db = nil
	}
}

// Query executes a custom ClickHouse query (additional method for advanced users)
func (ch *ClickHouseStorage) Query(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
	if !ch.connected {
		return nil, errors.NewStorageError("NOT_CONNECTED", "ClickHouse storage is not connected")
	}

	return ch.db.QueryContext(ctx, query, args...)
}

// GetStats returns storage statistics
func (ch *ClickHouseStorage) GetStats(ctx context.Context) (map[string]interface{}, error) {
	if !ch.connected {
		return nil, errors.NewStorageError("NOT_CONNECTED", "ClickHouse storage is not connected")
	}

	stats := make(map[string]interface{})

	// Get table statistics
	query := `
		SELECT 
			count() as total_rows,
			uniq(series_id) as unique_series,
			min(timestamp) as earliest_timestamp,
			max(timestamp) as latest_timestamp,
			formatReadableSize(sum(bytes_on_disk)) as disk_usage
		FROM time_series_data
	`

	row := ch.db.QueryRowContext(ctx, query)
	var totalRows, uniqueSeries int64
	var earliestTimestamp, latestTimestamp time.Time
	var diskUsage string

	err := row.Scan(&totalRows, &uniqueSeries, &earliestTimestamp, &latestTimestamp, &diskUsage)
	if err != nil {
		return nil, err
	}

	stats["total_rows"] = totalRows
	stats["unique_series"] = uniqueSeries
	stats["earliest_timestamp"] = earliestTimestamp
	stats["latest_timestamp"] = latestTimestamp
	stats["disk_usage"] = diskUsage
	stats["pending_batch_size"] = len(ch.pendingBatch)
	stats["last_flush"] = ch.lastFlush

	return stats, nil
}