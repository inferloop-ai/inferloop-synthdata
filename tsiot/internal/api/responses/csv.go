package responses

import (
	"encoding/csv"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"

	"github.com/your-org/tsiot/internal/storage"
	"github.com/your-org/tsiot/pkg/constants"
)

// CSVResponse handles CSV format responses
type CSVResponse struct {
	logger *logrus.Logger
}

// NewCSVResponse creates a new CSV response handler
func NewCSVResponse(logger *logrus.Logger) *CSVResponse {
	return &CSVResponse{
		logger: logger,
	}
}

// WriteTimeSeries writes a time series as CSV to the response
func (r *CSVResponse) WriteTimeSeries(c *gin.Context, timeSeries *storage.TimeSeries) error {
	// Set appropriate headers
	filename := fmt.Sprintf("%s_%s.csv", timeSeries.SeriesID, time.Now().Format("20060102_150405"))
	c.Header("Content-Type", constants.MimeTypeCSV)
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", filename))
	c.Header("Cache-Control", "no-cache")

	// Create CSV writer
	writer := csv.NewWriter(c.Writer)
	defer writer.Flush()

	// Write headers
	headers := []string{"timestamp", "value"}
	if len(timeSeries.Metadata) > 0 {
		for key := range timeSeries.Metadata {
			headers = append(headers, key)
		}
	}
	if timeSeries.Quality != nil && len(timeSeries.Quality) > 0 {
		headers = append(headers, "quality")
	}

	if err := writer.Write(headers); err != nil {
		r.logger.WithError(err).Error("Failed to write CSV headers")
		return err
	}

	// Write data rows
	for i, timestamp := range timeSeries.Timestamps {
		if i >= len(timeSeries.Values) {
			break
		}

		row := []string{
			timestamp.Format(time.RFC3339),
			strconv.FormatFloat(timeSeries.Values[i], 'f', -1, 64),
		}

		// Add metadata columns
		if len(timeSeries.Metadata) > 0 {
			for key := range timeSeries.Metadata {
				if value, exists := timeSeries.Metadata[key]; exists {
					row = append(row, fmt.Sprintf("%v", value))
				} else {
					row = append(row, "")
				}
			}
		}

		// Add quality column
		if timeSeries.Quality != nil && i < len(timeSeries.Quality) {
			row = append(row, strconv.FormatFloat(timeSeries.Quality[i], 'f', -1, 64))
		} else if len(headers) > 2 {
			// Add empty quality if header exists but no quality data
			for j := len(row); j < len(headers); j++ {
				row = append(row, "")
			}
		}

		if err := writer.Write(row); err != nil {
			r.logger.WithError(err).Error("Failed to write CSV row")
			return err
		}
	}

	c.Status(http.StatusOK)
	return nil
}

// WriteBatchTimeSeries writes multiple time series as CSV to the response
func (r *CSVResponse) WriteBatchTimeSeries(c *gin.Context, timeSeriesList []*storage.TimeSeries) error {
	// Set appropriate headers
	filename := fmt.Sprintf("batch_timeseries_%s.csv", time.Now().Format("20060102_150405"))
	c.Header("Content-Type", constants.MimeTypeCSV)
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", filename))
	c.Header("Cache-Control", "no-cache")

	// Create CSV writer
	writer := csv.NewWriter(c.Writer)
	defer writer.Flush()

	// Write headers (include series_id for batch)
	headers := []string{"series_id", "timestamp", "value", "quality"}

	if err := writer.Write(headers); err != nil {
		r.logger.WithError(err).Error("Failed to write CSV headers")
		return err
	}

	// Write data rows for each time series
	for _, timeSeries := range timeSeriesList {
		for i, timestamp := range timeSeries.Timestamps {
			if i >= len(timeSeries.Values) {
				break
			}

			qualityValue := ""
			if timeSeries.Quality != nil && i < len(timeSeries.Quality) {
				qualityValue = strconv.FormatFloat(timeSeries.Quality[i], 'f', -1, 64)
			}

			row := []string{
				timeSeries.SeriesID,
				timestamp.Format(time.RFC3339),
				strconv.FormatFloat(timeSeries.Values[i], 'f', -1, 64),
				qualityValue,
			}

			if err := writer.Write(row); err != nil {
				r.logger.WithError(err).Error("Failed to write CSV row")
				return err
			}
		}
	}

	c.Status(http.StatusOK)
	return nil
}

// WriteValidationResults writes validation results as CSV
func (r *CSVResponse) WriteValidationResults(c *gin.Context, results map[string]interface{}) error {
	filename := fmt.Sprintf("validation_results_%s.csv", time.Now().Format("20060102_150405"))
	c.Header("Content-Type", constants.MimeTypeCSV)
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", filename))
	c.Header("Cache-Control", "no-cache")

	writer := csv.NewWriter(c.Writer)
	defer writer.Flush()

	// Write headers
	headers := []string{"validation_type", "metric", "value", "status"}
	if err := writer.Write(headers); err != nil {
		r.logger.WithError(err).Error("Failed to write CSV headers")
		return err
	}

	// Flatten validation results into CSV rows
	for validationType, validationData := range results {
		if data, ok := validationData.(map[string]interface{}); ok {
			for metric, value := range data {
				status := "unknown"
				if metric == "passed" || metric == "status" {
					if passed, ok := value.(bool); ok {
						if passed {
							status = "passed"
						} else {
							status = "failed"
						}
					}
					continue
				}

				row := []string{
					validationType,
					metric,
					fmt.Sprintf("%v", value),
					status,
				}

				if err := writer.Write(row); err != nil {
					r.logger.WithError(err).Error("Failed to write CSV row")
					return err
				}
			}
		}
	}

	c.Status(http.StatusOK)
	return nil
}

// WriteAnalyticsResults writes analytics results as CSV
func (r *CSVResponse) WriteAnalyticsResults(c *gin.Context, results map[string]interface{}) error {
	filename := fmt.Sprintf("analytics_results_%s.csv", time.Now().Format("20060102_150405"))
	c.Header("Content-Type", constants.MimeTypeCSV)
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", filename))
	c.Header("Cache-Control", "no-cache")

	writer := csv.NewWriter(c.Writer)
	defer writer.Flush()

	// Write headers
	headers := []string{"analysis_type", "metric", "value"}
	if err := writer.Write(headers); err != nil {
		r.logger.WithError(err).Error("Failed to write CSV headers")
		return err
	}

	// Flatten analytics results into CSV rows
	for analysisType, analysisData := range results {
		if data, ok := analysisData.(map[string]interface{}); ok {
			for metric, value := range data {
				row := []string{
					analysisType,
					metric,
					fmt.Sprintf("%v", value),
				}

				if err := writer.Write(row); err != nil {
					r.logger.WithError(err).Error("Failed to write CSV row")
					return err
				}
			}
		}
	}

	c.Status(http.StatusOK)
	return nil
}

// WriteError writes an error response in CSV format
func (r *CSVResponse) WriteError(c *gin.Context, statusCode int, message string) {
	filename := fmt.Sprintf("error_%s.csv", time.Now().Format("20060102_150405"))
	c.Header("Content-Type", constants.MimeTypeCSV)
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", filename))

	writer := csv.NewWriter(c.Writer)
	defer writer.Flush()

	// Write error headers
	headers := []string{"error", "status_code", "timestamp"}
	writer.Write(headers)

	// Write error row
	row := []string{
		message,
		strconv.Itoa(statusCode),
		time.Now().Format(time.RFC3339),
	}
	writer.Write(row)

	c.Status(statusCode)
}