package influxdb

import (
	"context"
	"fmt"
	"time"

	influxdb2 "github.com/influxdata/influxdb-client-go/v2"
	"github.com/influxdata/influxdb-client-go/v2/api"
	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
	"github.com/inferloop/tsiot/pkg/errors"
)

// InfluxDBConfig contains configuration for InfluxDB storage
type InfluxDBConfig struct {
	URL           string        `json:"url" yaml:"url"`
	Token         string        `json:"token" yaml:"token"`
	Organization  string        `json:"organization" yaml:"organization"`
	Bucket        string        `json:"bucket" yaml:"bucket"`
	Timeout       time.Duration `json:"timeout" yaml:"timeout"`
	BatchSize     int           `json:"batch_size" yaml:"batch_size"`
	UseGZip       bool          `json:"use_gzip" yaml:"use_gzip"`
	Precision     string        `json:"precision" yaml:"precision"` // ns, us, ms, s
	RetentionPolicy string      `json:"retention_policy" yaml:"retention_policy"`
}

// InfluxDBStorage implements the Storage interface for InfluxDB
type InfluxDBStorage struct {
	config     *InfluxDBConfig
	client     influxdb2.Client
	writeAPI   api.WriteAPI
	queryAPI   api.QueryAPI
	logger     *logrus.Logger
	connected  bool
}

// NewInfluxDBStorage creates a new InfluxDB storage instance
func NewInfluxDBStorage(config *InfluxDBConfig, logger *logrus.Logger) (*InfluxDBStorage, error) {
	if config == nil {
		return nil, errors.NewStorageError("INVALID_CONFIG", "InfluxDB config cannot be nil")
	}

	if logger == nil {
		logger = logrus.New()
	}

	// Set defaults
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	if config.BatchSize == 0 {
		config.BatchSize = 1000
	}
	if config.Precision == "" {
		config.Precision = "ns"
	}

	storage := &InfluxDBStorage{
		config: config,
		logger: logger,
	}

	return storage, nil
}

// Connect establishes connection to InfluxDB
func (s *InfluxDBStorage) Connect(ctx context.Context) error {
	if s.connected {
		return nil
	}

	// Create InfluxDB client
	options := influxdb2.DefaultOptions()
	options.SetBatchSize(uint(s.config.BatchSize))
	options.SetUseGZip(s.config.UseGZip)
	options.SetPrecision(time.Nanosecond)

	s.client = influxdb2.NewClientWithOptions(s.config.URL, s.config.Token, options)

	// Test connection by pinging the server
	ok, err := s.client.Ping(ctx)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "CONNECTION_FAILED", "Failed to connect to InfluxDB")
	}

	if !ok {
		return errors.NewStorageError("CONNECTION_FAILED", "InfluxDB ping failed")
	}

	// Initialize APIs
	s.writeAPI = s.client.WriteAPI(s.config.Organization, s.config.Bucket)
	s.queryAPI = s.client.QueryAPI(s.config.Organization)

	s.connected = true

	s.logger.WithFields(logrus.Fields{
		"url":          s.config.URL,
		"organization": s.config.Organization,
		"bucket":       s.config.Bucket,
	}).Info("Connected to InfluxDB")

	return nil
}

// Close closes the connection to InfluxDB
func (s *InfluxDBStorage) Close() error {
	if !s.connected {
		return nil
	}

	if s.writeAPI != nil {
		s.writeAPI.Flush()
	}

	if s.client != nil {
		s.client.Close()
	}

	s.connected = false
	s.logger.Info("Disconnected from InfluxDB")

	return nil
}

// IsConnected returns whether the storage is connected
func (s *InfluxDBStorage) IsConnected() bool {
	return s.connected
}

// Write writes time series data to InfluxDB
func (s *InfluxDBStorage) Write(ctx context.Context, data *models.TimeSeries) error {
	if !s.connected {
		return errors.NewStorageError("NOT_CONNECTED", "Not connected to InfluxDB")
	}

	if data == nil {
		return errors.NewStorageError("INVALID_DATA", "Time series data cannot be nil")
	}

	// Convert data points to InfluxDB points
	for _, point := range data.DataPoints {
		influxPoint := influxdb2.NewPointWithMeasurement(data.SensorType).
			AddTag("series_id", data.ID).
			AddTag("series_name", data.Name).
			AddField("value", point.Value).
			AddField("quality", point.Quality).
			SetTime(point.Timestamp)

		// Add tags from data point
		for key, value := range point.Tags {
			influxPoint.AddTag(key, value)
		}

		// Add fields from metadata
		for key, value := range point.Metadata {
			influxPoint.AddField(fmt.Sprintf("meta_%s", key), value)
		}

		// Add series tags
		for key, value := range data.Tags {
			influxPoint.AddTag(fmt.Sprintf("series_%s", key), value)
		}

		// Write point
		s.writeAPI.WritePoint(influxPoint)
	}

	// Flush to ensure data is written
	s.writeAPI.Flush()

	// Check for write errors
	if err := s.writeAPI.Errors(); err != nil {
		select {
		case writeErr := <-err:
			return errors.WrapError(writeErr, errors.ErrorTypeStorage, "WRITE_FAILED", "Failed to write to InfluxDB")
		default:
			// No errors
		}
	}

	s.logger.WithFields(logrus.Fields{
		"series_id":    data.ID,
		"series_name":  data.Name,
		"data_points":  len(data.DataPoints),
		"sensor_type":  data.SensorType,
	}).Debug("Wrote time series to InfluxDB")

	return nil
}

// Read reads time series data from InfluxDB
func (s *InfluxDBStorage) Read(ctx context.Context, query *models.TimeSeriesQuery) (*models.TimeSeries, error) {
	if !s.connected {
		return nil, errors.NewStorageError("NOT_CONNECTED", "Not connected to InfluxDB")
	}

	if query == nil {
		return nil, errors.NewStorageError("INVALID_QUERY", "Query cannot be nil")
	}

	// Build Flux query
	fluxQuery := s.buildFluxQuery(query)

	s.logger.WithFields(logrus.Fields{
		"query": fluxQuery,
	}).Debug("Executing InfluxDB query")

	// Execute query
	result, err := s.queryAPI.Query(ctx, fluxQuery)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "QUERY_FAILED", "Failed to execute InfluxDB query")
	}

	// Parse results
	timeSeries := &models.TimeSeries{
		ID:          query.SeriesID,
		DataPoints:  make([]models.DataPoint, 0),
		Tags:        make(map[string]string),
		Metadata:    make(map[string]interface{}),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	for result.Next() {
		record := result.Record()
		
		// Extract data point
		timestamp := record.Time()
		value, ok := record.Value().(float64)
		if !ok {
			continue
		}

		dataPoint := models.DataPoint{
			Timestamp: timestamp,
			Value:     value,
			Quality:   1.0, // Default quality
			Tags:      make(map[string]string),
			Metadata:  make(map[string]interface{}),
		}

		// Extract tags and fields
		for key, val := range record.Values() {
			switch key {
			case "_time", "_value", "_field", "_measurement":
				// Skip system fields
				continue
			case "quality":
				if q, ok := val.(float64); ok {
					dataPoint.Quality = q
				}
			default:
				if strVal, ok := val.(string); ok {
					dataPoint.Tags[key] = strVal
				} else {
					dataPoint.Metadata[key] = val
				}
			}
		}

		timeSeries.DataPoints = append(timeSeries.DataPoints, dataPoint)

		// Update time series metadata
		if timeSeries.SensorType == "" {
			if measurement, ok := record.Values()["_measurement"].(string); ok {
				timeSeries.SensorType = measurement
			}
		}
		
		if timeSeries.Name == "" {
			if name, ok := record.Values()["series_name"].(string); ok {
				timeSeries.Name = name
			}
		}
	}

	if result.Err() != nil {
		return nil, errors.WrapError(result.Err(), errors.ErrorTypeStorage, "RESULT_ERROR", "Error reading query results")
	}

	// Update time range
	if len(timeSeries.DataPoints) > 0 {
		timeSeries.StartTime = timeSeries.DataPoints[0].Timestamp
		timeSeries.EndTime = timeSeries.DataPoints[len(timeSeries.DataPoints)-1].Timestamp
	}

	s.logger.WithFields(logrus.Fields{
		"series_id":   timeSeries.ID,
		"data_points": len(timeSeries.DataPoints),
	}).Debug("Read time series from InfluxDB")

	return timeSeries, nil
}

// Delete deletes time series data from InfluxDB
func (s *InfluxDBStorage) Delete(ctx context.Context, seriesID string) error {
	if !s.connected {
		return errors.NewStorageError("NOT_CONNECTED", "Not connected to InfluxDB")
	}

	// InfluxDB delete requires start and end times
	// For now, we'll delete all data for the series
	deleteAPI := s.client.DeleteAPI()
	
	start := time.Unix(0, 0)
	end := time.Now()
	
	predicate := fmt.Sprintf(`series_id="%s"`, seriesID)
	
	err := deleteAPI.DeleteWithName(ctx, s.config.Organization, s.config.Bucket, start, end, predicate)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "DELETE_FAILED", "Failed to delete from InfluxDB")
	}

	s.logger.WithFields(logrus.Fields{
		"series_id": seriesID,
	}).Debug("Deleted time series from InfluxDB")

	return nil
}

// List lists available time series
func (s *InfluxDBStorage) List(ctx context.Context, limit, offset int) ([]*models.TimeSeries, error) {
	if !s.connected {
		return nil, errors.NewStorageError("NOT_CONNECTED", "Not connected to InfluxDB")
	}

	// Query to get unique series
	fluxQuery := fmt.Sprintf(`
		from(bucket: "%s")
		|> range(start: -7d)
		|> group(columns: ["series_id", "series_name", "_measurement"])
		|> first()
		|> limit(n: %d, offset: %d)
	`, s.config.Bucket, limit, offset)

	result, err := s.queryAPI.Query(ctx, fluxQuery)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "QUERY_FAILED", "Failed to list time series")
	}

	var series []*models.TimeSeries
	seriesMap := make(map[string]*models.TimeSeries)

	for result.Next() {
		record := result.Record()
		
		seriesID, ok := record.Values()["series_id"].(string)
		if !ok {
			continue
		}

		if _, exists := seriesMap[seriesID]; !exists {
			ts := &models.TimeSeries{
				ID:         seriesID,
				DataPoints: make([]models.DataPoint, 0),
				Tags:       make(map[string]string),
				Metadata:   make(map[string]interface{}),
				CreatedAt:  time.Now(),
				UpdatedAt:  time.Now(),
			}

			if name, ok := record.Values()["series_name"].(string); ok {
				ts.Name = name
			}

			if measurement, ok := record.Values()["_measurement"].(string); ok {
				ts.SensorType = measurement
			}

			seriesMap[seriesID] = ts
			series = append(series, ts)
		}
	}

	if result.Err() != nil {
		return nil, errors.WrapError(result.Err(), errors.ErrorTypeStorage, "RESULT_ERROR", "Error reading list results")
	}

	return series, nil
}

// Health checks the health of the InfluxDB connection
func (s *InfluxDBStorage) Health(ctx context.Context) error {
	if !s.connected {
		return errors.NewStorageError("NOT_CONNECTED", "Not connected to InfluxDB")
	}

	ok, err := s.client.Ping(ctx)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "HEALTH_CHECK_FAILED", "InfluxDB health check failed")
	}

	if !ok {
		return errors.NewStorageError("HEALTH_CHECK_FAILED", "InfluxDB ping returned false")
	}

	return nil
}

// GetStats returns storage statistics
func (s *InfluxDBStorage) GetStats(ctx context.Context) (map[string]interface{}, error) {
	stats := map[string]interface{}{
		"connected":     s.connected,
		"url":           s.config.URL,
		"organization":  s.config.Organization,
		"bucket":        s.config.Bucket,
		"batch_size":    s.config.BatchSize,
		"use_gzip":      s.config.UseGZip,
		"precision":     s.config.Precision,
	}

	if s.connected {
		// Get bucket statistics
		fluxQuery := fmt.Sprintf(`
			from(bucket: "%s")
			|> range(start: -24h)
			|> count()
		`, s.config.Bucket)

		result, err := s.queryAPI.Query(ctx, fluxQuery)
		if err == nil {
			var totalPoints int64
			for result.Next() {
				if count, ok := result.Record().Value().(int64); ok {
					totalPoints += count
				}
			}
			stats["total_points_24h"] = totalPoints
		}
	}

	return stats, nil
}

// buildFluxQuery builds a Flux query from a TimeSeriesQuery
func (s *InfluxDBStorage) buildFluxQuery(query *models.TimeSeriesQuery) string {
	fluxQuery := fmt.Sprintf(`from(bucket: "%s")`, s.config.Bucket)

	// Add time range
	if query.StartTime != nil && query.EndTime != nil {
		fluxQuery += fmt.Sprintf(`
			|> range(start: %s, stop: %s)`,
			query.StartTime.Format(time.RFC3339),
			query.EndTime.Format(time.RFC3339))
	} else if query.StartTime != nil {
		fluxQuery += fmt.Sprintf(`
			|> range(start: %s)`,
			query.StartTime.Format(time.RFC3339))
	} else {
		fluxQuery += `
			|> range(start: -24h)`
	}

	// Add series ID filter
	if query.SeriesID != "" {
		fluxQuery += fmt.Sprintf(`
			|> filter(fn: (r) => r.series_id == "%s")`, query.SeriesID)
	}

	// Add tag filters
	for key, value := range query.Tags {
		fluxQuery += fmt.Sprintf(`
			|> filter(fn: (r) => r.%s == "%s")`, key, value)
	}

	// Add aggregation
	if query.Aggregate != "" && query.Interval != "" {
		switch query.Aggregate {
		case "mean":
			fluxQuery += fmt.Sprintf(`
				|> aggregateWindow(every: %s, fn: mean, createEmpty: false)`, query.Interval)
		case "sum":
			fluxQuery += fmt.Sprintf(`
				|> aggregateWindow(every: %s, fn: sum, createEmpty: false)`, query.Interval)
		case "count":
			fluxQuery += fmt.Sprintf(`
				|> aggregateWindow(every: %s, fn: count, createEmpty: false)`, query.Interval)
		}
	}

	// Add limit and offset
	if query.Limit > 0 {
		fluxQuery += fmt.Sprintf(`
			|> limit(n: %d, offset: %d)`, query.Limit, query.Offset)
	}

	return fluxQuery
}