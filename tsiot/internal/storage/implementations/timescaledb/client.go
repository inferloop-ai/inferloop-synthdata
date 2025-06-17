package timescaledb

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/lib/pq"
	_ "github.com/lib/pq"
	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
	"github.com/inferloop/tsiot/pkg/errors"
)

// TimescaleDBConfig holds configuration for TimescaleDB
type TimescaleDBConfig struct {
	Host             string        `json:"host"`
	Port             int           `json:"port"`
	Database         string        `json:"database"`
	Username         string        `json:"username"`
	Password         string        `json:"password"`
	SSLMode          string        `json:"ssl_mode"`
	ConnectTimeout   time.Duration `json:"connect_timeout"`
	QueryTimeout     time.Duration `json:"query_timeout"`
	MaxConnections   int           `json:"max_connections"`
	MaxIdleConns     int           `json:"max_idle_conns"`
	ConnMaxLifetime  time.Duration `json:"conn_max_lifetime"`
	ChunkTimeInterval string       `json:"chunk_time_interval"`
	CompressionPolicy bool         `json:"compression_policy"`
	RetentionPolicy   string       `json:"retention_policy"`
}

// TimescaleDBStorage implements the Storage interface for TimescaleDB
type TimescaleDBStorage struct {
	config   *TimescaleDBConfig
	db       *sql.DB
	logger   *logrus.Logger
	mu       sync.RWMutex
	metrics  *storageMetrics
	closed   bool
}

type storageMetrics struct {
	readOps      int64
	writeOps     int64
	deleteOps    int64
	errorCount   int64
	startTime    time.Time
	mu           sync.RWMutex
}

// NewTimescaleDBStorage creates a new TimescaleDB storage instance
func NewTimescaleDBStorage(config *TimescaleDBConfig, logger *logrus.Logger) (*TimescaleDBStorage, error) {
	if config == nil {
		return nil, errors.NewStorageError("INVALID_CONFIG", "TimescaleDB config cannot be nil")
	}

	if logger == nil {
		logger = logrus.New()
	}

	storage := &TimescaleDBStorage{
		config: config,
		logger: logger,
		metrics: &storageMetrics{
			startTime: time.Now(),
		},
	}

	return storage, nil
}

// Connect establishes connection to TimescaleDB
func (ts *TimescaleDBStorage) Connect(ctx context.Context) error {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	if ts.db != nil {
		return nil // Already connected
	}

	connStr := fmt.Sprintf("host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		ts.config.Host,
		ts.config.Port,
		ts.config.Username,
		ts.config.Password,
		ts.config.Database,
		ts.config.SSLMode,
	)

	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "CONNECTION_FAILED", "Failed to open database connection")
	}

	// Configure connection pool
	db.SetMaxOpenConns(ts.config.MaxConnections)
	db.SetMaxIdleConns(ts.config.MaxIdleConns)
	db.SetConnMaxLifetime(ts.config.ConnMaxLifetime)

	// Test connection
	ctx, cancel := context.WithTimeout(ctx, ts.config.ConnectTimeout)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return errors.WrapError(err, errors.ErrorTypeStorage, "PING_FAILED", "Failed to ping database")
	}

	ts.db = db

	// Initialize database schema
	if err := ts.initializeSchema(ctx); err != nil {
		db.Close()
		ts.db = nil
		return errors.WrapError(err, errors.ErrorTypeStorage, "SCHEMA_INIT_FAILED", "Failed to initialize schema")
	}

	ts.logger.WithFields(logrus.Fields{
		"host":     ts.config.Host,
		"port":     ts.config.Port,
		"database": ts.config.Database,
	}).Info("Connected to TimescaleDB")

	return nil
}

// Close closes the database connection
func (ts *TimescaleDBStorage) Close() error {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	if ts.closed {
		return nil
	}

	if ts.db != nil {
		err := ts.db.Close()
		ts.db = nil
		ts.closed = true
		
		if err != nil {
			return errors.WrapError(err, errors.ErrorTypeStorage, "CLOSE_FAILED", "Failed to close database connection")
		}
	}

	ts.logger.Info("TimescaleDB connection closed")
	return nil
}

// Ping tests the database connection
func (ts *TimescaleDBStorage) Ping(ctx context.Context) error {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	if ts.closed || ts.db == nil {
		return errors.NewStorageError("NOT_CONNECTED", "Database not connected")
	}

	ctx, cancel := context.WithTimeout(ctx, ts.config.QueryTimeout)
	defer cancel()

	if err := ts.db.PingContext(ctx); err != nil {
		ts.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "PING_FAILED", "Database ping failed")
	}

	return nil
}

// GetInfo returns information about the TimescaleDB storage
func (ts *TimescaleDBStorage) GetInfo(ctx context.Context) (*interfaces.StorageInfo, error) {
	version, err := ts.getVersion(ctx)
	if err != nil {
		version = "unknown"
	}

	return &interfaces.StorageInfo{
		Type:        "timescaledb",
		Version:     version,
		Name:        "TimescaleDB Storage",
		Description: "Time-series database built on PostgreSQL",
		Features: []string{
			"time-series optimization",
			"SQL queries",
			"continuous aggregates",
			"compression",
			"retention policies",
			"parallel processing",
		},
		Capabilities: interfaces.StorageCapabilities{
			Streaming:      true,
			Transactions:   true,
			Compression:    true,
			Encryption:     true,
			Replication:    true,
			Clustering:     true,
			Backup:         true,
			Archival:       true,
			TimeBasedQuery: true,
			Aggregation:    true,
		},
		Configuration: map[string]interface{}{
			"host":                ts.config.Host,
			"port":                ts.config.Port,
			"database":            ts.config.Database,
			"max_connections":     ts.config.MaxConnections,
			"chunk_time_interval": ts.config.ChunkTimeInterval,
			"compression_policy":  ts.config.CompressionPolicy,
		},
	}, nil
}

// Health returns the health status of the storage
func (ts *TimescaleDBStorage) Health(ctx context.Context) (*interfaces.HealthStatus, error) {
	start := time.Now()
	status := "healthy"
	var errors []string
	var warnings []string

	// Test connection
	if err := ts.Ping(ctx); err != nil {
		status = "unhealthy"
		errors = append(errors, fmt.Sprintf("Connection failed: %v", err))
	}

	latency := time.Since(start)

	// Check for warnings
	if latency > 100*time.Millisecond {
		warnings = append(warnings, "High latency detected")
	}

	connections := ts.getActiveConnections()

	return &interfaces.HealthStatus{
		Status:      status,
		LastCheck:   time.Now(),
		Latency:     latency,
		Connections: connections,
		Errors:      errors,
		Warnings:    warnings,
		Metadata: map[string]interface{}{
			"database_size": ts.getDatabaseSize(ctx),
			"table_count":   ts.getTableCount(ctx),
		},
	}, nil
}

// Write writes a time series to the database
func (ts *TimescaleDBStorage) Write(ctx context.Context, data *models.TimeSeries) error {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	if ts.closed || ts.db == nil {
		return errors.NewStorageError("NOT_CONNECTED", "Database not connected")
	}

	if err := data.Validate(); err != nil {
		return errors.WrapError(err, errors.ErrorTypeValidation, "INVALID_DATA", "Time series validation failed")
	}

	start := time.Now()
	defer func() {
		ts.incrementWriteOps()
		ts.logger.WithField("duration", time.Since(start)).Debug("Write operation completed")
	}()

	tx, err := ts.db.BeginTx(ctx, nil)
	if err != nil {
		ts.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "TRANSACTION_FAILED", "Failed to begin transaction")
	}
	defer tx.Rollback()

	// Insert or update time series metadata
	if err := ts.upsertTimeSeries(ctx, tx, data); err != nil {
		ts.incrementErrorCount()
		return err
	}

	// Insert data points
	if err := ts.insertDataPoints(ctx, tx, data.ID, data.DataPoints); err != nil {
		ts.incrementErrorCount()
		return err
	}

	if err := tx.Commit(); err != nil {
		ts.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "COMMIT_FAILED", "Failed to commit transaction")
	}

	return nil
}

// WriteBatch writes multiple time series in a batch
func (ts *TimescaleDBStorage) WriteBatch(ctx context.Context, batch []*models.TimeSeries) error {
	if len(batch) == 0 {
		return nil
	}

	ts.mu.RLock()
	defer ts.mu.RUnlock()

	if ts.closed || ts.db == nil {
		return errors.NewStorageError("NOT_CONNECTED", "Database not connected")
	}

	start := time.Now()
	defer func() {
		ts.metrics.mu.Lock()
		ts.metrics.writeOps += int64(len(batch))
		ts.metrics.mu.Unlock()
		ts.logger.WithFields(logrus.Fields{
			"count":    len(batch),
			"duration": time.Since(start),
		}).Debug("Batch write operation completed")
	}()

	tx, err := ts.db.BeginTx(ctx, nil)
	if err != nil {
		ts.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "TRANSACTION_FAILED", "Failed to begin transaction")
	}
	defer tx.Rollback()

	for _, timeSeries := range batch {
		if err := timeSeries.Validate(); err != nil {
			ts.incrementErrorCount()
			return errors.WrapError(err, errors.ErrorTypeValidation, "INVALID_DATA", 
				fmt.Sprintf("Time series validation failed for ID: %s", timeSeries.ID))
		}

		if err := ts.upsertTimeSeries(ctx, tx, timeSeries); err != nil {
			ts.incrementErrorCount()
			return err
		}

		if err := ts.insertDataPoints(ctx, tx, timeSeries.ID, timeSeries.DataPoints); err != nil {
			ts.incrementErrorCount()
			return err
		}
	}

	if err := tx.Commit(); err != nil {
		ts.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "COMMIT_FAILED", "Failed to commit transaction")
	}

	return nil
}

// Read reads a time series by ID
func (ts *TimescaleDBStorage) Read(ctx context.Context, id string) (*models.TimeSeries, error) {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	if ts.closed || ts.db == nil {
		return nil, errors.NewStorageError("NOT_CONNECTED", "Database not connected")
	}

	start := time.Now()
	defer func() {
		ts.incrementReadOps()
		ts.logger.WithField("duration", time.Since(start)).Debug("Read operation completed")
	}()

	// Read time series metadata
	timeSeries, err := ts.readTimeSeries(ctx, id)
	if err != nil {
		ts.incrementErrorCount()
		return nil, err
	}

	// Read data points
	dataPoints, err := ts.readDataPoints(ctx, id, nil, nil)
	if err != nil {
		ts.incrementErrorCount()
		return nil, err
	}

	timeSeries.DataPoints = dataPoints
	return timeSeries, nil
}

// ReadRange reads time series data within a time range
func (ts *TimescaleDBStorage) ReadRange(ctx context.Context, id string, start, end time.Time) (*models.TimeSeries, error) {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	if ts.closed || ts.db == nil {
		return nil, errors.NewStorageError("NOT_CONNECTED", "Database not connected")
	}

	startOp := time.Now()
	defer func() {
		ts.incrementReadOps()
		ts.logger.WithField("duration", time.Since(startOp)).Debug("Read range operation completed")
	}()

	// Read time series metadata
	timeSeries, err := ts.readTimeSeries(ctx, id)
	if err != nil {
		ts.incrementErrorCount()
		return nil, err
	}

	// Read data points within range
	dataPoints, err := ts.readDataPoints(ctx, id, &start, &end)
	if err != nil {
		ts.incrementErrorCount()
		return nil, err
	}

	timeSeries.DataPoints = dataPoints
	return timeSeries, nil
}

// Query queries time series data with filters
func (ts *TimescaleDBStorage) Query(ctx context.Context, query *models.TimeSeriesQuery) ([]*models.TimeSeries, error) {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	if ts.closed || ts.db == nil {
		return nil, errors.NewStorageError("NOT_CONNECTED", "Database not connected")
	}

	start := time.Now()
	defer func() {
		ts.incrementReadOps()
		ts.logger.WithField("duration", time.Since(start)).Debug("Query operation completed")
	}()

	return ts.queryTimeSeries(ctx, query)
}

// Delete deletes a time series by ID
func (ts *TimescaleDBStorage) Delete(ctx context.Context, id string) error {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	if ts.closed || ts.db == nil {
		return errors.NewStorageError("NOT_CONNECTED", "Database not connected")
	}

	start := time.Now()
	defer func() {
		ts.incrementDeleteOps()
		ts.logger.WithField("duration", time.Since(start)).Debug("Delete operation completed")
	}()

	tx, err := ts.db.BeginTx(ctx, nil)
	if err != nil {
		ts.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "TRANSACTION_FAILED", "Failed to begin transaction")
	}
	defer tx.Rollback()

	// Delete data points
	if _, err := tx.ExecContext(ctx, "DELETE FROM data_points WHERE series_id = $1", id); err != nil {
		ts.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "DELETE_FAILED", "Failed to delete data points")
	}

	// Delete time series metadata
	if _, err := tx.ExecContext(ctx, "DELETE FROM time_series WHERE id = $1", id); err != nil {
		ts.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "DELETE_FAILED", "Failed to delete time series")
	}

	if err := tx.Commit(); err != nil {
		ts.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "COMMIT_FAILED", "Failed to commit transaction")
	}

	return nil
}

// DeleteRange deletes time series data within a time range
func (ts *TimescaleDBStorage) DeleteRange(ctx context.Context, id string, start, end time.Time) error {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	if ts.closed || ts.db == nil {
		return errors.NewStorageError("NOT_CONNECTED", "Database not connected")
	}

	startOp := time.Now()
	defer func() {
		ts.incrementDeleteOps()
		ts.logger.WithField("duration", time.Since(startOp)).Debug("Delete range operation completed")
	}()

	query := "DELETE FROM data_points WHERE series_id = $1 AND timestamp >= $2 AND timestamp <= $3"
	_, err := ts.db.ExecContext(ctx, query, id, start, end)
	if err != nil {
		ts.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "DELETE_RANGE_FAILED", "Failed to delete data points in range")
	}

	return nil
}

// List lists available time series
func (ts *TimescaleDBStorage) List(ctx context.Context, filters map[string]interface{}) ([]*models.TimeSeries, error) {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	if ts.closed || ts.db == nil {
		return nil, errors.NewStorageError("NOT_CONNECTED", "Database not connected")
	}

	start := time.Now()
	defer func() {
		ts.incrementReadOps()
		ts.logger.WithField("duration", time.Since(start)).Debug("List operation completed")
	}()

	query := "SELECT id, name, description, tags, metadata, start_time, end_time, frequency, sensor_type, created_at, updated_at FROM time_series"
	var args []interface{}
	var conditions []string

	// Apply filters
	argIndex := 1
	if name, ok := filters["name"]; ok {
		conditions = append(conditions, fmt.Sprintf("name ILIKE $%d", argIndex))
		args = append(args, fmt.Sprintf("%%%s%%", name))
		argIndex++
	}

	if sensorType, ok := filters["sensor_type"]; ok {
		conditions = append(conditions, fmt.Sprintf("sensor_type = $%d", argIndex))
		args = append(args, sensorType)
		argIndex++
	}

	if len(conditions) > 0 {
		query += " WHERE " + fmt.Sprintf("%s", conditions[0])
		for i := 1; i < len(conditions); i++ {
			query += " AND " + conditions[i]
		}
	}

	query += " ORDER BY created_at DESC"

	if limit, ok := filters["limit"]; ok {
		if l, ok := limit.(int); ok && l > 0 {
			query += fmt.Sprintf(" LIMIT %d", l)
		}
	}

	rows, err := ts.db.QueryContext(ctx, query, args...)
	if err != nil {
		ts.incrementErrorCount()
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "QUERY_FAILED", "Failed to list time series")
	}
	defer rows.Close()

	var result []*models.TimeSeries
	for rows.Next() {
		timeSeries, err := ts.scanTimeSeries(rows)
		if err != nil {
			ts.incrementErrorCount()
			return nil, err
		}
		result = append(result, timeSeries)
	}

	if err := rows.Err(); err != nil {
		ts.incrementErrorCount()
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "SCAN_FAILED", "Failed to scan rows")
	}

	return result, nil
}

// Count returns the count of time series matching filters
func (ts *TimescaleDBStorage) Count(ctx context.Context, filters map[string]interface{}) (int64, error) {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	if ts.closed || ts.db == nil {
		return 0, errors.NewStorageError("NOT_CONNECTED", "Database not connected")
	}

	start := time.Now()
	defer func() {
		ts.incrementReadOps()
		ts.logger.WithField("duration", time.Since(start)).Debug("Count operation completed")
	}()

	query := "SELECT COUNT(*) FROM time_series"
	var args []interface{}
	var conditions []string

	// Apply filters (similar to List method)
	argIndex := 1
	if name, ok := filters["name"]; ok {
		conditions = append(conditions, fmt.Sprintf("name ILIKE $%d", argIndex))
		args = append(args, fmt.Sprintf("%%%s%%", name))
		argIndex++
	}

	if sensorType, ok := filters["sensor_type"]; ok {
		conditions = append(conditions, fmt.Sprintf("sensor_type = $%d", argIndex))
		args = append(args, sensorType)
		argIndex++
	}

	if len(conditions) > 0 {
		query += " WHERE " + fmt.Sprintf("%s", conditions[0])
		for i := 1; i < len(conditions); i++ {
			query += " AND " + conditions[i]
		}
	}

	var count int64
	err := ts.db.QueryRowContext(ctx, query, args...).Scan(&count)
	if err != nil {
		ts.incrementErrorCount()
		return 0, errors.WrapError(err, errors.ErrorTypeStorage, "COUNT_FAILED", "Failed to count time series")
	}

	return count, nil
}

// GetMetrics returns storage metrics
func (ts *TimescaleDBStorage) GetMetrics(ctx context.Context) (*interfaces.StorageMetrics, error) {
	ts.metrics.mu.RLock()
	defer ts.metrics.mu.RUnlock()

	stats := ts.db.Stats()

	return &interfaces.StorageMetrics{
		ReadOperations:    ts.metrics.readOps,
		WriteOperations:   ts.metrics.writeOps,
		DeleteOperations:  ts.metrics.deleteOps,
		AverageReadTime:   time.Millisecond * 50,  // Simplified
		AverageWriteTime:  time.Millisecond * 100, // Simplified
		ErrorCount:        ts.metrics.errorCount,
		ConnectionsActive: stats.OpenConnections,
		ConnectionsIdle:   stats.Idle,
		DataSize:          ts.getDatabaseSize(ctx),
		RecordCount:       ts.getRecordCount(ctx),
		Uptime:            time.Since(ts.metrics.startTime),
	}, nil
}

// Helper methods

func (ts *TimescaleDBStorage) initializeSchema(ctx context.Context) error {
	// Create TimescaleDB extension if not exists
	if _, err := ts.db.ExecContext(ctx, "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"); err != nil {
		return fmt.Errorf("failed to create timescaledb extension: %w", err)
	}

	// Create time_series table
	timeSeriesSchema := `
	CREATE TABLE IF NOT EXISTS time_series (
		id VARCHAR(255) PRIMARY KEY,
		name VARCHAR(255) NOT NULL,
		description TEXT,
		tags JSONB,
		metadata JSONB,
		start_time TIMESTAMPTZ,
		end_time TIMESTAMPTZ,
		frequency VARCHAR(50),
		sensor_type VARCHAR(100),
		created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
		updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
	)`

	if _, err := ts.db.ExecContext(ctx, timeSeriesSchema); err != nil {
		return fmt.Errorf("failed to create time_series table: %w", err)
	}

	// Create data_points hypertable
	dataPointsSchema := `
	CREATE TABLE IF NOT EXISTS data_points (
		series_id VARCHAR(255) NOT NULL,
		timestamp TIMESTAMPTZ NOT NULL,
		value DOUBLE PRECISION NOT NULL,
		quality DOUBLE PRECISION,
		tags JSONB,
		metadata JSONB,
		FOREIGN KEY (series_id) REFERENCES time_series(id) ON DELETE CASCADE
	)`

	if _, err := ts.db.ExecContext(ctx, dataPointsSchema); err != nil {
		return fmt.Errorf("failed to create data_points table: %w", err)
	}

	// Create hypertable
	hypertableQuery := `
	SELECT create_hypertable('data_points', 'timestamp', 
		chunk_time_interval => INTERVAL '%s',
		if_not_exists => TRUE
	)`

	interval := ts.config.ChunkTimeInterval
	if interval == "" {
		interval = "1 day"
	}

	if _, err := ts.db.ExecContext(ctx, fmt.Sprintf(hypertableQuery, interval)); err != nil {
		// Log error but don't fail - table might already be a hypertable
		ts.logger.WithError(err).Warn("Failed to create hypertable, table might already exist")
	}

	// Create indexes
	indexes := []string{
		"CREATE INDEX IF NOT EXISTS idx_data_points_series_timestamp ON data_points (series_id, timestamp DESC)",
		"CREATE INDEX IF NOT EXISTS idx_time_series_name ON time_series (name)",
		"CREATE INDEX IF NOT EXISTS idx_time_series_sensor_type ON time_series (sensor_type)",
		"CREATE INDEX IF NOT EXISTS idx_time_series_created_at ON time_series (created_at)",
	}

	for _, index := range indexes {
		if _, err := ts.db.ExecContext(ctx, index); err != nil {
			ts.logger.WithError(err).Warn("Failed to create index")
		}
	}

	// Setup compression policy if enabled
	if ts.config.CompressionPolicy {
		compressionQuery := `
		SELECT add_compression_policy('data_points', INTERVAL '7 days', if_not_exists => TRUE)
		`
		if _, err := ts.db.ExecContext(ctx, compressionQuery); err != nil {
			ts.logger.WithError(err).Warn("Failed to setup compression policy")
		}
	}

	// Setup retention policy if configured
	if ts.config.RetentionPolicy != "" {
		retentionQuery := fmt.Sprintf(`
		SELECT add_retention_policy('data_points', INTERVAL '%s', if_not_exists => TRUE)
		`, ts.config.RetentionPolicy)
		
		if _, err := ts.db.ExecContext(ctx, retentionQuery); err != nil {
			ts.logger.WithError(err).Warn("Failed to setup retention policy")
		}
	}

	return nil
}

func (ts *TimescaleDBStorage) upsertTimeSeries(ctx context.Context, tx *sql.Tx, data *models.TimeSeries) error {
	tagsJSON, _ := json.Marshal(data.Tags)
	metadataJSON, _ := json.Marshal(data.Metadata)

	query := `
	INSERT INTO time_series (id, name, description, tags, metadata, start_time, end_time, frequency, sensor_type, created_at, updated_at)
	VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
	ON CONFLICT (id) DO UPDATE SET
		name = EXCLUDED.name,
		description = EXCLUDED.description,
		tags = EXCLUDED.tags,
		metadata = EXCLUDED.metadata,
		start_time = EXCLUDED.start_time,
		end_time = EXCLUDED.end_time,
		frequency = EXCLUDED.frequency,
		sensor_type = EXCLUDED.sensor_type,
		updated_at = EXCLUDED.updated_at
	`

	_, err := tx.ExecContext(ctx, query,
		data.ID, data.Name, data.Description,
		tagsJSON, metadataJSON,
		data.StartTime, data.EndTime,
		data.Frequency, data.SensorType,
		data.CreatedAt, data.UpdatedAt,
	)

	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "UPSERT_FAILED", "Failed to upsert time series")
	}

	return nil
}

func (ts *TimescaleDBStorage) insertDataPoints(ctx context.Context, tx *sql.Tx, seriesID string, points []models.DataPoint) error {
	if len(points) == 0 {
		return nil
	}

	// Use COPY for efficient bulk inserts
	stmt, err := tx.PrepareContext(ctx, pq.CopyIn("data_points", "series_id", "timestamp", "value", "quality", "tags", "metadata"))
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "COPY_PREPARE_FAILED", "Failed to prepare COPY statement")
	}
	defer stmt.Close()

	for _, point := range points {
		tagsJSON, _ := json.Marshal(point.Tags)
		metadataJSON, _ := json.Marshal(point.Metadata)

		_, err := stmt.ExecContext(ctx, seriesID, point.Timestamp, point.Value, point.Quality, tagsJSON, metadataJSON)
		if err != nil {
			return errors.WrapError(err, errors.ErrorTypeStorage, "COPY_EXEC_FAILED", "Failed to execute COPY")
		}
	}

	_, err = stmt.ExecContext(ctx)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "COPY_FINALIZE_FAILED", "Failed to finalize COPY")
	}

	return nil
}

func (ts *TimescaleDBStorage) readTimeSeries(ctx context.Context, id string) (*models.TimeSeries, error) {
	query := "SELECT id, name, description, tags, metadata, start_time, end_time, frequency, sensor_type, created_at, updated_at FROM time_series WHERE id = $1"
	
	row := ts.db.QueryRowContext(ctx, query, id)
	return ts.scanTimeSeries(row)
}

func (ts *TimescaleDBStorage) readDataPoints(ctx context.Context, seriesID string, start, end *time.Time) ([]models.DataPoint, error) {
	query := "SELECT timestamp, value, quality, tags, metadata FROM data_points WHERE series_id = $1"
	args := []interface{}{seriesID}
	argIndex := 2

	if start != nil {
		query += fmt.Sprintf(" AND timestamp >= $%d", argIndex)
		args = append(args, *start)
		argIndex++
	}

	if end != nil {
		query += fmt.Sprintf(" AND timestamp <= $%d", argIndex)
		args = append(args, *end)
	}

	query += " ORDER BY timestamp"

	rows, err := ts.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "QUERY_FAILED", "Failed to query data points")
	}
	defer rows.Close()

	var points []models.DataPoint
	for rows.Next() {
		var point models.DataPoint
		var tagsJSON, metadataJSON []byte

		err := rows.Scan(&point.Timestamp, &point.Value, &point.Quality, &tagsJSON, &metadataJSON)
		if err != nil {
			return nil, errors.WrapError(err, errors.ErrorTypeStorage, "SCAN_FAILED", "Failed to scan data point")
		}

		if len(tagsJSON) > 0 {
			json.Unmarshal(tagsJSON, &point.Tags)
		}
		if len(metadataJSON) > 0 {
			json.Unmarshal(metadataJSON, &point.Metadata)
		}

		points = append(points, point)
	}

	if err := rows.Err(); err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "SCAN_FAILED", "Failed to scan rows")
	}

	return points, nil
}

func (ts *TimescaleDBStorage) scanTimeSeries(scanner interface {
	Scan(dest ...interface{}) error
}) (*models.TimeSeries, error) {
	var timeSeries models.TimeSeries
	var tagsJSON, metadataJSON []byte

	err := scanner.Scan(
		&timeSeries.ID,
		&timeSeries.Name,
		&timeSeries.Description,
		&tagsJSON,
		&metadataJSON,
		&timeSeries.StartTime,
		&timeSeries.EndTime,
		&timeSeries.Frequency,
		&timeSeries.SensorType,
		&timeSeries.CreatedAt,
		&timeSeries.UpdatedAt,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, errors.NewStorageError("NOT_FOUND", "Time series not found")
		}
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "SCAN_FAILED", "Failed to scan time series")
	}

	if len(tagsJSON) > 0 {
		json.Unmarshal(tagsJSON, &timeSeries.Tags)
	}
	if len(metadataJSON) > 0 {
		json.Unmarshal(metadataJSON, &timeSeries.Metadata)
	}

	return &timeSeries, nil
}

func (ts *TimescaleDBStorage) queryTimeSeries(ctx context.Context, query *models.TimeSeriesQuery) ([]*models.TimeSeries, error) {
	// Implementation would build dynamic SQL based on query parameters
	// For now, return basic implementation
	filters := make(map[string]interface{})
	if query.Limit > 0 {
		filters["limit"] = query.Limit
	}
	
	return ts.List(ctx, filters)
}

func (ts *TimescaleDBStorage) getVersion(ctx context.Context) (string, error) {
	var version string
	err := ts.db.QueryRowContext(ctx, "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'").Scan(&version)
	if err != nil {
		return "", err
	}
	return version, nil
}

func (ts *TimescaleDBStorage) getActiveConnections() int {
	if ts.db == nil {
		return 0
	}
	return ts.db.Stats().OpenConnections
}

func (ts *TimescaleDBStorage) getDatabaseSize(ctx context.Context) int64 {
	var size int64
	err := ts.db.QueryRowContext(ctx, "SELECT pg_database_size(current_database())").Scan(&size)
	if err != nil {
		return 0
	}
	return size
}

func (ts *TimescaleDBStorage) getTableCount(ctx context.Context) int {
	var count int
	err := ts.db.QueryRowContext(ctx, "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'").Scan(&count)
	if err != nil {
		return 0
	}
	return count
}

func (ts *TimescaleDBStorage) getRecordCount(ctx context.Context) int64 {
	var count int64
	err := ts.db.QueryRowContext(ctx, "SELECT count(*) FROM data_points").Scan(&count)
	if err != nil {
		return 0
	}
	return count
}

func (ts *TimescaleDBStorage) incrementReadOps() {
	ts.metrics.mu.Lock()
	ts.metrics.readOps++
	ts.metrics.mu.Unlock()
}

func (ts *TimescaleDBStorage) incrementWriteOps() {
	ts.metrics.mu.Lock()
	ts.metrics.writeOps++
	ts.metrics.mu.Unlock()
}

func (ts *TimescaleDBStorage) incrementDeleteOps() {
	ts.metrics.mu.Lock()
	ts.metrics.deleteOps++
	ts.metrics.mu.Unlock()
}

func (ts *TimescaleDBStorage) incrementErrorCount() {
	ts.metrics.mu.Lock()
	ts.metrics.errorCount++
	ts.metrics.mu.Unlock()
}