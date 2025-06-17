package storage

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	influxdb2 "github.com/influxdata/influxdb-client-go/v2"
	"github.com/influxdata/influxdb-client-go/v2/api"
	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
	"github.com/inferloop/tsiot/pkg/errors"
)

// InfluxDBStorage implements the TimeSeriesStorage interface for InfluxDB
type InfluxDBStorage struct {
	client   influxdb2.Client
	writeAPI api.WriteAPI
	queryAPI api.QueryAPI
	config   *InfluxDBConfig
	logger   *logrus.Logger
	org      string
	bucket   string
}

// InfluxDBConfig contains InfluxDB-specific configuration
type InfluxDBConfig struct {
	URL            string        `yaml:"url" json:"url"`
	Token          string        `yaml:"token" json:"token"`
	Organization   string        `yaml:"organization" json:"organization"`
	Bucket         string        `yaml:"bucket" json:"bucket"`
	Timeout        time.Duration `yaml:"timeout" json:"timeout"`
	BatchSize      int           `yaml:"batch_size" json:"batch_size"`
	FlushInterval  time.Duration `yaml:"flush_interval" json:"flush_interval"`
	RetryInterval  time.Duration `yaml:"retry_interval" json:"retry_interval"`
	MaxRetries     int           `yaml:"max_retries" json:"max_retries"`
	Precision      string        `yaml:"precision" json:"precision"`
	UseGZip        bool          `yaml:"use_gzip" json:"use_gzip"`
}

// NewInfluxDBStorage creates a new InfluxDB storage instance
func NewInfluxDBStorage(config *InfluxDBConfig, logger *logrus.Logger) (*InfluxDBStorage, error) {
	if config == nil {
		return nil, errors.NewStorageError("INVALID_CONFIG", "InfluxDB configuration is required")
	}

	if logger == nil {
		logger = logrus.New()
	}

	// Set default values
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	if config.BatchSize == 0 {
		config.BatchSize = 1000
	}
	if config.FlushInterval == 0 {
		config.FlushInterval = 1 * time.Second
	}
	if config.Precision == "" {
		config.Precision = "ms"
	}

	// Create InfluxDB client
	client := influxdb2.NewClientWithOptions(
		config.URL,
		config.Token,
		influxdb2.DefaultOptions().
			SetBatchSize(uint(config.BatchSize)).
			SetFlushInterval(uint(config.FlushInterval.Milliseconds())).
			SetRetryInterval(uint(config.RetryInterval.Milliseconds())).
			SetMaxRetries(uint(config.MaxRetries)).
			SetUseGZip(config.UseGZip).
			SetPrecision(time.Millisecond),
	)

	storage := &InfluxDBStorage{
		client: client,
		config: config,
		logger: logger,
		org:    config.Organization,
		bucket: config.Bucket,
	}

	// Initialize APIs
	storage.writeAPI = client.WriteAPI(storage.org, storage.bucket)
	storage.queryAPI = client.QueryAPI(storage.org)

	// Setup error handling for write API
	go storage.handleWriteErrors()

	return storage, nil
}

// Connect establishes connection to InfluxDB
func (s *InfluxDBStorage) Connect(ctx context.Context) error {
	s.logger.Info("Connecting to InfluxDB...")

	// Test connection by pinging the server
	health, err := s.client.Health(ctx)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "CONNECTION_FAILED", "Failed to connect to InfluxDB")
	}

	if health.Status != "pass" {
		return errors.NewStorageError("CONNECTION_FAILED", fmt.Sprintf("InfluxDB health check failed: %s", health.Message))
	}

	s.logger.Info("Successfully connected to InfluxDB")
	return nil
}

// Close closes the connection to InfluxDB
func (s *InfluxDBStorage) Close() error {
	s.logger.Info("Closing InfluxDB connection...")
	
	// Flush any remaining writes
	s.writeAPI.Flush()
	
	// Close the client
	s.client.Close()
	
	s.logger.Info("InfluxDB connection closed")
	return nil
}

// Ping tests the connection to InfluxDB
func (s *InfluxDBStorage) Ping(ctx context.Context) error {
	health, err := s.client.Health(ctx)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "PING_FAILED", "Failed to ping InfluxDB")
	}

	if health.Status != "pass" {
		return errors.NewStorageError("PING_FAILED", fmt.Sprintf("InfluxDB health check failed: %s", health.Message))
	}

	return nil
}

// GetInfo returns information about the InfluxDB storage backend
func (s *InfluxDBStorage) GetInfo(ctx context.Context) (*interfaces.StorageInfo, error) {
	health, err := s.client.Health(ctx)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "INFO_FAILED", "Failed to get InfluxDB info")
	}

	return &interfaces.StorageInfo{
		Type:        "influxdb",
		Version:     health.Version,
		Name:        "InfluxDB",
		Description: "Time series database optimized for fast, high-availability storage and retrieval",
		Features:    []string{"time-series", "compression", "retention-policies", "continuous-queries"},
		Capabilities: interfaces.StorageCapabilities{
			Streaming:      true,
			Transactions:   false,
			Compression:    true,
			Encryption:     false,
			Replication:    true,
			Clustering:     true,
			Backup:         true,
			Archival:       true,
			TimeBasedQuery: true,
			Aggregation:    true,
		},
		Configuration: map[string]interface{}{
			"url":          s.config.URL,
			"organization": s.config.Organization,
			"bucket":       s.config.Bucket,
			"batch_size":   s.config.BatchSize,
		},
	}, nil
}

// Health returns the health status of InfluxDB
func (s *InfluxDBStorage) Health(ctx context.Context) (*interfaces.HealthStatus, error) {
	start := time.Now()
	health, err := s.client.Health(ctx)
	latency := time.Since(start)

	status := "healthy"
	var healthErrors []string

	if err != nil {
		status = "unhealthy"
		healthErrors = append(healthErrors, err.Error())
	} else if health.Status != "pass" {
		status = "degraded"
		if health.Message != "" {
			healthErrors = append(healthErrors, health.Message)
		}
	}

	return &interfaces.HealthStatus{
		Status:      status,
		LastCheck:   time.Now(),
		Latency:     latency,
		Connections: 1, // InfluxDB client maintains connection pool internally
		Errors:      healthErrors,
	}, nil
}

// Write writes a single time series to InfluxDB
func (s *InfluxDBStorage) Write(ctx context.Context, data *models.TimeSeries) error {
	if data == nil {
		return errors.NewValidationError("INVALID_INPUT", "Time series data is required")
	}

	if err := data.Validate(); err != nil {
		return errors.WrapError(err, errors.ErrorTypeValidation, "INVALID_DATA", "Invalid time series data")
	}

	// Convert time series to InfluxDB points
	for _, point := range data.DataPoints {
		p := influxdb2.NewPointWithMeasurement("time_series").
			AddTag("series_id", data.ID).
			AddTag("series_name", data.Name).
			AddTag("sensor_type", data.SensorType).
			AddField("value", point.Value).
			SetTime(point.Timestamp)

		// Add tags from time series
		for key, value := range data.Tags {
			p.AddTag(key, value)
		}

		// Add tags from data point
		for key, value := range point.Tags {
			p.AddTag(key, value)
		}

		// Add quality if present
		if point.Quality > 0 {
			p.AddField("quality", point.Quality)
		}

		// Write the point
		s.writeAPI.WritePoint(p)
	}

	s.logger.WithFields(logrus.Fields{
		"series_id":    data.ID,
		"series_name":  data.Name,
		"data_points":  len(data.DataPoints),
	}).Debug("Wrote time series to InfluxDB")

	return nil
}

// WriteBatch writes multiple time series to InfluxDB in a batch
func (s *InfluxDBStorage) WriteBatch(ctx context.Context, batch []*models.TimeSeries) error {
	if len(batch) == 0 {
		return errors.NewValidationError("INVALID_INPUT", "Batch cannot be empty")
	}

	totalPoints := 0
	for _, data := range batch {
		if err := s.Write(ctx, data); err != nil {
			return err
		}
		totalPoints += len(data.DataPoints)
	}

	s.logger.WithFields(logrus.Fields{
		"batch_size":   len(batch),
		"total_points": totalPoints,
	}).Debug("Wrote batch to InfluxDB")

	return nil
}

// Read reads a time series by ID from InfluxDB
func (s *InfluxDBStorage) Read(ctx context.Context, id string) (*models.TimeSeries, error) {
	if id == "" {
		return nil, errors.NewValidationError("INVALID_INPUT", "Series ID is required")
	}

	query := fmt.Sprintf(`
		from(bucket: "%s")
		|> range(start: -1y)
		|> filter(fn: (r) => r._measurement == "time_series" and r.series_id == "%s")
		|> sort(columns: ["_time"])
	`, s.bucket, id)

	result, err := s.queryAPI.Query(ctx, query)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "READ_FAILED", "Failed to read from InfluxDB")
	}
	defer result.Close()

	return s.parseQueryResult(result)
}

// ReadRange reads time series data within a time range
func (s *InfluxDBStorage) ReadRange(ctx context.Context, id string, start, end time.Time) (*models.TimeSeries, error) {
	if id == "" {
		return nil, errors.NewValidationError("INVALID_INPUT", "Series ID is required")
	}

	query := fmt.Sprintf(`
		from(bucket: "%s")
		|> range(start: %s, stop: %s)
		|> filter(fn: (r) => r._measurement == "time_series" and r.series_id == "%s")
		|> sort(columns: ["_time"])
	`, s.bucket, start.Format(time.RFC3339), end.Format(time.RFC3339), id)

	result, err := s.queryAPI.Query(ctx, query)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "READ_FAILED", "Failed to read range from InfluxDB")
	}
	defer result.Close()

	return s.parseQueryResult(result)
}

// Query queries time series data with filters
func (s *InfluxDBStorage) Query(ctx context.Context, query *models.TimeSeriesQuery) ([]*models.TimeSeries, error) {
	if query == nil {
		return nil, errors.NewValidationError("INVALID_INPUT", "Query is required")
	}

	// Build InfluxDB query
	fluxQuery := s.buildFluxQuery(query)

	result, err := s.queryAPI.Query(ctx, fluxQuery)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "QUERY_FAILED", "Failed to execute query")
	}
	defer result.Close()

	return s.parseQueryResults(result)
}

// Delete deletes a time series by ID
func (s *InfluxDBStorage) Delete(ctx context.Context, id string) error {
	if id == "" {
		return errors.NewValidationError("INVALID_INPUT", "Series ID is required")
	}

	// InfluxDB doesn't support direct delete by tag, we need to use the delete API
	// For now, we'll mark it as deleted by adding a tombstone record
	deletePoint := influxdb2.NewPointWithMeasurement("time_series_deleted").
		AddTag("series_id", id).
		AddField("deleted", true).
		SetTime(time.Now())

	s.writeAPI.WritePoint(deletePoint)

	s.logger.WithFields(logrus.Fields{
		"series_id": id,
	}).Info("Marked time series as deleted in InfluxDB")

	return nil
}

// DeleteRange deletes time series data within a time range
func (s *InfluxDBStorage) DeleteRange(ctx context.Context, id string, start, end time.Time) error {
	if id == "" {
		return errors.NewValidationError("INVALID_INPUT", "Series ID is required")
	}

	// Similar to Delete, we'll use a tombstone approach
	deletePoint := influxdb2.NewPointWithMeasurement("time_series_deleted_range").
		AddTag("series_id", id).
		AddField("deleted", true).
		AddField("start_time", start.Format(time.RFC3339)).
		AddField("end_time", end.Format(time.RFC3339)).
		SetTime(time.Now())

	s.writeAPI.WritePoint(deletePoint)

	s.logger.WithFields(logrus.Fields{
		"series_id": id,
		"start":     start,
		"end":       end,
	}).Info("Marked time series range as deleted in InfluxDB")

	return nil
}

// List lists available time series with optional filters
func (s *InfluxDBStorage) List(ctx context.Context, filters map[string]interface{}) ([]*models.TimeSeries, error) {
	query := fmt.Sprintf(`
		from(bucket: "%s")
		|> range(start: -1y)
		|> filter(fn: (r) => r._measurement == "time_series")
		|> group(columns: ["series_id"])
		|> first()
	`, s.bucket)

	result, err := s.queryAPI.Query(ctx, query)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "LIST_FAILED", "Failed to list time series")
	}
	defer result.Close()

	return s.parseQueryResults(result)
}

// Count returns the count of time series matching filters
func (s *InfluxDBStorage) Count(ctx context.Context, filters map[string]interface{}) (int64, error) {
	query := fmt.Sprintf(`
		from(bucket: "%s")
		|> range(start: -1y)
		|> filter(fn: (r) => r._measurement == "time_series")
		|> group(columns: ["series_id"])
		|> count()
	`, s.bucket)

	result, err := s.queryAPI.Query(ctx, query)
	if err != nil {
		return 0, errors.WrapError(err, errors.ErrorTypeStorage, "COUNT_FAILED", "Failed to count time series")
	}
	defer result.Close()

	count := int64(0)
	for result.Next() {
		if result.Record().Value() != nil {
			if v, ok := result.Record().Value().(int64); ok {
				count += v
			}
		}
	}

	return count, nil
}

// GetMetrics returns storage metrics
func (s *InfluxDBStorage) GetMetrics(ctx context.Context) (*interfaces.StorageMetrics, error) {
	// In a real implementation, you would collect actual metrics
	return &interfaces.StorageMetrics{
		ReadOperations:    0,
		WriteOperations:   0,
		DeleteOperations:  0,
		AverageReadTime:   0,
		AverageWriteTime:  0,
		ErrorCount:        0,
		ConnectionsActive: 1,
		ConnectionsIdle:   0,
		DataSize:          0,
		RecordCount:       0,
		Uptime:            time.Since(time.Now()),
	}, nil
}

// handleWriteErrors handles write errors from InfluxDB
func (s *InfluxDBStorage) handleWriteErrors() {
	errorsCh := s.writeAPI.Errors()
	for err := range errorsCh {
		s.logger.WithFields(logrus.Fields{
			"error": err.Error(),
		}).Error("InfluxDB write error")
	}
}

// parseQueryResult parses a single query result into a TimeSeries
func (s *InfluxDBStorage) parseQueryResult(result *api.QueryTableResult) (*models.TimeSeries, error) {
	results, err := s.parseQueryResults(result)
	if err != nil {
		return nil, err
	}

	if len(results) == 0 {
		return nil, errors.NewStorageError("DATA_NOT_FOUND", "Time series not found")
	}

	return results[0], nil
}

// parseQueryResults parses multiple query results into TimeSeries slice
func (s *InfluxDBStorage) parseQueryResults(result *api.QueryTableResult) ([]*models.TimeSeries, error) {
	seriesMap := make(map[string]*models.TimeSeries)

	for result.Next() {
		record := result.Record()
		
		seriesID := record.ValueByKey("series_id").(string)
		
		// Get or create time series
		series, exists := seriesMap[seriesID]
		if !exists {
			series = &models.TimeSeries{
				ID:         seriesID,
				Name:       getStringValue(record.ValueByKey("series_name")),
				SensorType: getStringValue(record.ValueByKey("sensor_type")),
				Tags:       make(map[string]string),
				DataPoints: make([]models.DataPoint, 0),
			}
			seriesMap[seriesID] = series
		}

		// Add data point
		dataPoint := models.DataPoint{
			Timestamp: record.Time(),
			Value:     getFloatValue(record.Value()),
			Quality:   getFloatValue(record.ValueByKey("quality")),
			Tags:      make(map[string]string),
		}

		// Copy tags from record
		for key, value := range record.Values() {
			if strings.HasPrefix(key, "_") || key == "series_id" || key == "series_name" || key == "sensor_type" {
				continue
			}
			if strValue := getStringValue(value); strValue != "" {
				dataPoint.Tags[key] = strValue
			}
		}

		series.DataPoints = append(series.DataPoints, dataPoint)
	}

	// Convert map to slice
	results := make([]*models.TimeSeries, 0, len(seriesMap))
	for _, series := range seriesMap {
		results = append(results, series)
	}

	return results, nil
}

// buildFluxQuery builds a Flux query from TimeSeriesQuery
func (s *InfluxDBStorage) buildFluxQuery(query *models.TimeSeriesQuery) string {
	var flux strings.Builder

	// Base query
	fmt.Fprintf(&flux, `from(bucket: "%s")`, s.bucket)

	// Time range
	if query.StartTime != nil && query.EndTime != nil {
		fmt.Fprintf(&flux, `|> range(start: %s, stop: %s)`, 
			query.StartTime.Format(time.RFC3339), 
			query.EndTime.Format(time.RFC3339))
	} else if query.StartTime != nil {
		fmt.Fprintf(&flux, `|> range(start: %s)`, query.StartTime.Format(time.RFC3339))
	} else {
		flux.WriteString(`|> range(start: -1y)`)
	}

	// Measurement filter
	flux.WriteString(`|> filter(fn: (r) => r._measurement == "time_series"`)

	// Series ID filter
	if query.SeriesID != "" {
		fmt.Fprintf(&flux, ` and r.series_id == "%s"`, query.SeriesID)
	}

	// Tag filters
	for key, value := range query.Tags {
		fmt.Fprintf(&flux, ` and r.%s == "%s"`, key, value)
	}

	flux.WriteString(`)`)

	// Aggregation
	if query.Aggregate != "" {
		switch query.Aggregate {
		case "mean":
			flux.WriteString(`|> aggregateWindow(every: 1m, fn: mean)`)
		case "sum":
			flux.WriteString(`|> aggregateWindow(every: 1m, fn: sum)`)
		case "count":
			flux.WriteString(`|> aggregateWindow(every: 1m, fn: count)`)
		case "min":
			flux.WriteString(`|> aggregateWindow(every: 1m, fn: min)`)
		case "max":
			flux.WriteString(`|> aggregateWindow(every: 1m, fn: max)`)
		}
	}

	// Sort
	flux.WriteString(`|> sort(columns: ["_time"])`)

	// Limit
	if query.Limit > 0 {
		fmt.Fprintf(&flux, `|> limit(n: %d`, query.Limit)
		if query.Offset > 0 {
			fmt.Fprintf(&flux, `, offset: %d`, query.Offset)
		}
		flux.WriteString(`)`)
	}

	return flux.String()
}

// getStringValue safely converts interface{} to string
func getStringValue(value interface{}) string {
	if value == nil {
		return ""
	}
	if str, ok := value.(string); ok {
		return str
	}
	return fmt.Sprintf("%v", value)
}

// getFloatValue safely converts interface{} to float64
func getFloatValue(value interface{}) float64 {
	if value == nil {
		return 0
	}
	
	switch v := value.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int64:
		return float64(v)
	case string:
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	
	return 0
}