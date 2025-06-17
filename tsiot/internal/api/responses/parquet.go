package responses

import (
	"bytes"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
	"github.com/xitongsys/parquet-go-source/buffer"
	"github.com/xitongsys/parquet-go/writer"

	"github.com/your-org/tsiot/internal/storage"
	"github.com/your-org/tsiot/pkg/constants"
)

// ParquetResponse handles Parquet format responses
type ParquetResponse struct {
	logger *logrus.Logger
}

// NewParquetResponse creates a new Parquet response handler
func NewParquetResponse(logger *logrus.Logger) *ParquetResponse {
	return &ParquetResponse{
		logger: logger,
	}
}

// TimeSeriesParquetRecord represents a time series data point for Parquet
type TimeSeriesParquetRecord struct {
	SeriesID  string  `parquet:"name=series_id, type=BYTE_ARRAY, convertedtype=UTF8"`
	Timestamp int64   `parquet:"name=timestamp, type=INT64, convertedtype=TIMESTAMP_MILLIS"`
	Value     float64 `parquet:"name=value, type=DOUBLE"`
	Quality   float64 `parquet:"name=quality, type=DOUBLE"`
}

// ValidationParquetRecord represents validation results for Parquet
type ValidationParquetRecord struct {
	SeriesID       string  `parquet:"name=series_id, type=BYTE_ARRAY, convertedtype=UTF8"`
	ValidationType string  `parquet:"name=validation_type, type=BYTE_ARRAY, convertedtype=UTF8"`
	Metric         string  `parquet:"name=metric, type=BYTE_ARRAY, convertedtype=UTF8"`
	Value          float64 `parquet:"name=value, type=DOUBLE"`
	Status         string  `parquet:"name=status, type=BYTE_ARRAY, convertedtype=UTF8"`
	Timestamp      int64   `parquet:"name=timestamp, type=INT64, convertedtype=TIMESTAMP_MILLIS"`
}

// AnalyticsParquetRecord represents analytics results for Parquet
type AnalyticsParquetRecord struct {
	SeriesID     string  `parquet:"name=series_id, type=BYTE_ARRAY, convertedtype=UTF8"`
	AnalysisType string  `parquet:"name=analysis_type, type=BYTE_ARRAY, convertedtype=UTF8"`
	Metric       string  `parquet:"name=metric, type=BYTE_ARRAY, convertedtype=UTF8"`
	Value        float64 `parquet:"name=value, type=DOUBLE"`
	Timestamp    int64   `parquet:"name=timestamp, type=INT64, convertedtype=TIMESTAMP_MILLIS"`
}

// WriteTimeSeries writes a time series as Parquet to the response
func (r *ParquetResponse) WriteTimeSeries(c *gin.Context, timeSeries *storage.TimeSeries) error {
	// Set appropriate headers
	filename := fmt.Sprintf("%s_%s.parquet", timeSeries.SeriesID, time.Now().Format("20060102_150405"))
	c.Header("Content-Type", constants.MimeTypeParquet)
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", filename))
	c.Header("Cache-Control", "no-cache")

	// Create buffer for Parquet data
	buf := bytes.NewBuffer(nil)
	bufferSource := buffer.NewBufferSource(buf)

	// Create Parquet writer
	pw, err := writer.NewParquetWriterFromWriter(bufferSource, new(TimeSeriesParquetRecord), 4)
	if err != nil {
		r.logger.WithError(err).Error("Failed to create Parquet writer")
		return err
	}

	// Convert time series to Parquet records
	records := make([]*TimeSeriesParquetRecord, 0, len(timeSeries.Values))
	for i, timestamp := range timeSeries.Timestamps {
		if i >= len(timeSeries.Values) {
			break
		}

		quality := 0.0
		if timeSeries.Quality != nil && i < len(timeSeries.Quality) {
			quality = timeSeries.Quality[i]
		}

		record := &TimeSeriesParquetRecord{
			SeriesID:  timeSeries.SeriesID,
			Timestamp: timestamp.UnixMilli(),
			Value:     timeSeries.Values[i],
			Quality:   quality,
		}
		records = append(records, record)
	}

	// Write records to Parquet
	for _, record := range records {
		if err := pw.Write(record); err != nil {
			r.logger.WithError(err).Error("Failed to write Parquet record")
			pw.WriteStop()
			return err
		}
	}

	// Finalize Parquet file
	if err := pw.WriteStop(); err != nil {
		r.logger.WithError(err).Error("Failed to finalize Parquet file")
		return err
	}

	// Write buffer to response
	c.Status(http.StatusOK)
	_, err = c.Writer.Write(buf.Bytes())
	return err
}

// WriteBatchTimeSeries writes multiple time series as Parquet to the response
func (r *ParquetResponse) WriteBatchTimeSeries(c *gin.Context, timeSeriesList []*storage.TimeSeries) error {
	// Set appropriate headers
	filename := fmt.Sprintf("batch_timeseries_%s.parquet", time.Now().Format("20060102_150405"))
	c.Header("Content-Type", constants.MimeTypeParquet)
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", filename))
	c.Header("Cache-Control", "no-cache")

	// Create buffer for Parquet data
	buf := bytes.NewBuffer(nil)
	bufferSource := buffer.NewBufferSource(buf)

	// Create Parquet writer
	pw, err := writer.NewParquetWriterFromWriter(bufferSource, new(TimeSeriesParquetRecord), 4)
	if err != nil {
		r.logger.WithError(err).Error("Failed to create Parquet writer")
		return err
	}

	// Convert all time series to Parquet records
	for _, timeSeries := range timeSeriesList {
		for i, timestamp := range timeSeries.Timestamps {
			if i >= len(timeSeries.Values) {
				break
			}

			quality := 0.0
			if timeSeries.Quality != nil && i < len(timeSeries.Quality) {
				quality = timeSeries.Quality[i]
			}

			record := &TimeSeriesParquetRecord{
				SeriesID:  timeSeries.SeriesID,
				Timestamp: timestamp.UnixMilli(),
				Value:     timeSeries.Values[i],
				Quality:   quality,
			}

			if err := pw.Write(record); err != nil {
				r.logger.WithError(err).Error("Failed to write Parquet record")
				pw.WriteStop()
				return err
			}
		}
	}

	// Finalize Parquet file
	if err := pw.WriteStop(); err != nil {
		r.logger.WithError(err).Error("Failed to finalize Parquet file")
		return err
	}

	// Write buffer to response
	c.Status(http.StatusOK)
	_, err = c.Writer.Write(buf.Bytes())
	return err
}

// WriteValidationResults writes validation results as Parquet
func (r *ParquetResponse) WriteValidationResults(c *gin.Context, seriesID string, results map[string]interface{}) error {
	filename := fmt.Sprintf("validation_results_%s.parquet", time.Now().Format("20060102_150405"))
	c.Header("Content-Type", constants.MimeTypeParquet)
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", filename))
	c.Header("Cache-Control", "no-cache")

	// Create buffer for Parquet data
	buf := bytes.NewBuffer(nil)
	bufferSource := buffer.NewBufferSource(buf)

	// Create Parquet writer
	pw, err := writer.NewParquetWriterFromWriter(bufferSource, new(ValidationParquetRecord), 4)
	if err != nil {
		r.logger.WithError(err).Error("Failed to create Parquet writer")
		return err
	}

	timestamp := time.Now().UnixMilli()

	// Convert validation results to Parquet records
	for validationType, validationData := range results {
		if data, ok := validationData.(map[string]interface{}); ok {
			for metric, value := range data {
				status := "unknown"
				numericValue := 0.0

				// Handle different value types
				switch v := value.(type) {
				case bool:
					if v {
						numericValue = 1.0
						status = "passed"
					} else {
						numericValue = 0.0
						status = "failed"
					}
				case float64:
					numericValue = v
				case int:
					numericValue = float64(v)
				case string:
					// For string values, we'll use 0 as numeric value
					numericValue = 0.0
				}

				record := &ValidationParquetRecord{
					SeriesID:       seriesID,
					ValidationType: validationType,
					Metric:         metric,
					Value:          numericValue,
					Status:         status,
					Timestamp:      timestamp,
				}

				if err := pw.Write(record); err != nil {
					r.logger.WithError(err).Error("Failed to write Parquet record")
					pw.WriteStop()
					return err
				}
			}
		}
	}

	// Finalize Parquet file
	if err := pw.WriteStop(); err != nil {
		r.logger.WithError(err).Error("Failed to finalize Parquet file")
		return err
	}

	// Write buffer to response
	c.Status(http.StatusOK)
	_, err = c.Writer.Write(buf.Bytes())
	return err
}

// WriteAnalyticsResults writes analytics results as Parquet
func (r *ParquetResponse) WriteAnalyticsResults(c *gin.Context, seriesID string, results map[string]interface{}) error {
	filename := fmt.Sprintf("analytics_results_%s.parquet", time.Now().Format("20060102_150405"))
	c.Header("Content-Type", constants.MimeTypeParquet)
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", filename))
	c.Header("Cache-Control", "no-cache")

	// Create buffer for Parquet data
	buf := bytes.NewBuffer(nil)
	bufferSource := buffer.NewBufferSource(buf)

	// Create Parquet writer
	pw, err := writer.NewParquetWriterFromWriter(bufferSource, new(AnalyticsParquetRecord), 4)
	if err != nil {
		r.logger.WithError(err).Error("Failed to create Parquet writer")
		return err
	}

	timestamp := time.Now().UnixMilli()

	// Convert analytics results to Parquet records
	for analysisType, analysisData := range results {
		if data, ok := analysisData.(map[string]interface{}); ok {
			for metric, value := range data {
				numericValue := 0.0

				// Handle different value types
				switch v := value.(type) {
				case float64:
					numericValue = v
				case int:
					numericValue = float64(v)
				case bool:
					if v {
						numericValue = 1.0
					} else {
						numericValue = 0.0
					}
				}

				record := &AnalyticsParquetRecord{
					SeriesID:     seriesID,
					AnalysisType: analysisType,
					Metric:       metric,
					Value:        numericValue,
					Timestamp:    timestamp,
				}

				if err := pw.Write(record); err != nil {
					r.logger.WithError(err).Error("Failed to write Parquet record")
					pw.WriteStop()
					return err
				}
			}
		}
	}

	// Finalize Parquet file
	if err := pw.WriteStop(); err != nil {
		r.logger.WithError(err).Error("Failed to finalize Parquet file")
		return err
	}

	// Write buffer to response
	c.Status(http.StatusOK)
	_, err = c.Writer.Write(buf.Bytes())
	return err
}

// WriteError writes an error response in Parquet format
func (r *ParquetResponse) WriteError(c *gin.Context, statusCode int, message string) {
	// For errors, we'll fall back to JSON since Parquet is not suitable for error responses
	c.Header("Content-Type", constants.MimeTypeJSON)
	
	errorResponse := map[string]interface{}{
		"error":       message,
		"status_code": statusCode,
		"timestamp":   time.Now().UTC(),
		"note":        "Error response cannot be encoded in Parquet format",
	}
	
	c.JSON(statusCode, errorResponse)
}