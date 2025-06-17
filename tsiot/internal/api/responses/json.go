package responses

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"

	"github.com/your-org/tsiot/internal/storage"
	"github.com/your-org/tsiot/pkg/constants"
)

// JSONResponse handles JSON format responses
type JSONResponse struct {
	logger *logrus.Logger
}

// NewJSONResponse creates a new JSON response handler
func NewJSONResponse(logger *logrus.Logger) *JSONResponse {
	return &JSONResponse{
		logger: logger,
	}
}

// TimeSeriesResponse represents a time series in JSON format
type TimeSeriesResponse struct {
	SeriesID   string                 `json:"series_id"`
	Name       string                 `json:"name,omitempty"`
	Generator  string                 `json:"generator,omitempty"`
	Length     int                    `json:"length"`
	StartTime  time.Time              `json:"start_time"`
	EndTime    time.Time              `json:"end_time"`
	Interval   string                 `json:"interval,omitempty"`
	Timestamps []time.Time            `json:"timestamps"`
	Values     []float64              `json:"values"`
	Quality    []float64              `json:"quality,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt  time.Time              `json:"created_at"`
	UpdatedAt  time.Time              `json:"updated_at"`
}

// BatchTimeSeriesResponse represents multiple time series
type BatchTimeSeriesResponse struct {
	TimeSeries []*TimeSeriesResponse `json:"time_series"`
	Count      int                   `json:"count"`
	Generated  time.Time             `json:"generated_at"`
}

// ValidationResponse represents validation results
type ValidationResponse struct {
	SeriesID          string                 `json:"series_id"`
	ValidationResults map[string]interface{} `json:"validation_results"`
	OverallStatus     string                 `json:"overall_status"`
	Timestamp         time.Time              `json:"timestamp"`
	ExecutionTime     time.Duration          `json:"execution_time_ms"`
}

// AnalyticsResponse represents analytics results
type AnalyticsResponse struct {
	SeriesID        string                 `json:"series_id"`
	AnalysisResults map[string]interface{} `json:"analysis_results"`
	Timestamp       time.Time              `json:"timestamp"`
	ExecutionTime   time.Duration          `json:"execution_time_ms"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error       string                 `json:"error"`
	Message     string                 `json:"message,omitempty"`
	Code        string                 `json:"code,omitempty"`
	Details     map[string]interface{} `json:"details,omitempty"`
	Timestamp   time.Time              `json:"timestamp"`
	RequestID   string                 `json:"request_id,omitempty"`
	StatusCode  int                    `json:"status_code"`
}

// SuccessResponse represents a generic success response
type SuccessResponse struct {
	Success   bool                   `json:"success"`
	Message   string                 `json:"message,omitempty"`
	Data      interface{}            `json:"data,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	Timestamp time.Time              `json:"timestamp"`
	RequestID string                 `json:"request_id,omitempty"`
}

// WriteTimeSeries writes a time series as JSON to the response
func (r *JSONResponse) WriteTimeSeries(c *gin.Context, timeSeries *storage.TimeSeries) error {
	c.Header("Content-Type", constants.MimeTypeJSON)
	c.Header("Cache-Control", "no-cache")

	response := r.convertToTimeSeriesResponse(timeSeries)
	c.JSON(http.StatusOK, response)
	return nil
}

// WriteBatchTimeSeries writes multiple time series as JSON to the response
func (r *JSONResponse) WriteBatchTimeSeries(c *gin.Context, timeSeriesList []*storage.TimeSeries) error {
	c.Header("Content-Type", constants.MimeTypeJSON)
	c.Header("Cache-Control", "no-cache")

	responses := make([]*TimeSeriesResponse, len(timeSeriesList))
	for i, ts := range timeSeriesList {
		responses[i] = r.convertToTimeSeriesResponse(ts)
	}

	batchResponse := &BatchTimeSeriesResponse{
		TimeSeries: responses,
		Count:      len(responses),
		Generated:  time.Now().UTC(),
	}

	c.JSON(http.StatusOK, batchResponse)
	return nil
}

// WriteValidationResults writes validation results as JSON
func (r *JSONResponse) WriteValidationResults(c *gin.Context, seriesID string, results map[string]interface{}, executionTime time.Duration) error {
	c.Header("Content-Type", constants.MimeTypeJSON)
	c.Header("Cache-Control", "no-cache")

	// Determine overall status
	overallStatus := "passed"
	for _, result := range results {
		if resultMap, ok := result.(map[string]interface{}); ok {
			if passed, exists := resultMap["passed"]; exists {
				if passedBool, ok := passed.(bool); ok && !passedBool {
					overallStatus = "failed"
					break
				}
			}
		}
	}

	response := &ValidationResponse{
		SeriesID:          seriesID,
		ValidationResults: results,
		OverallStatus:     overallStatus,
		Timestamp:         time.Now().UTC(),
		ExecutionTime:     executionTime,
	}

	c.JSON(http.StatusOK, response)
	return nil
}

// WriteAnalyticsResults writes analytics results as JSON
func (r *JSONResponse) WriteAnalyticsResults(c *gin.Context, seriesID string, results map[string]interface{}, executionTime time.Duration) error {
	c.Header("Content-Type", constants.MimeTypeJSON)
	c.Header("Cache-Control", "no-cache")

	response := &AnalyticsResponse{
		SeriesID:        seriesID,
		AnalysisResults: results,
		Timestamp:       time.Now().UTC(),
		ExecutionTime:   executionTime,
	}

	c.JSON(http.StatusOK, response)
	return nil
}

// WriteSuccess writes a generic success response
func (r *JSONResponse) WriteSuccess(c *gin.Context, statusCode int, message string, data interface{}) {
	c.Header("Content-Type", constants.MimeTypeJSON)

	response := &SuccessResponse{
		Success:   true,
		Message:   message,
		Data:      data,
		Timestamp: time.Now().UTC(),
		RequestID: c.GetString("X-Request-ID"),
	}

	c.JSON(statusCode, response)
}

// WriteError writes an error response in JSON format
func (r *JSONResponse) WriteError(c *gin.Context, statusCode int, error string, details map[string]interface{}) {
	c.Header("Content-Type", constants.MimeTypeJSON)

	response := &ErrorResponse{
		Error:      error,
		Details:    details,
		Timestamp:  time.Now().UTC(),
		RequestID:  c.GetString("X-Request-ID"),
		StatusCode: statusCode,
	}

	// Add error code based on status
	switch statusCode {
	case http.StatusBadRequest:
		response.Code = "INVALID_REQUEST"
	case http.StatusUnauthorized:
		response.Code = "UNAUTHORIZED"
	case http.StatusForbidden:
		response.Code = "FORBIDDEN"
	case http.StatusNotFound:
		response.Code = "NOT_FOUND"
	case http.StatusConflict:
		response.Code = "CONFLICT"
	case http.StatusTooManyRequests:
		response.Code = "RATE_LIMITED"
	case http.StatusInternalServerError:
		response.Code = "INTERNAL_ERROR"
	case http.StatusServiceUnavailable:
		response.Code = "SERVICE_UNAVAILABLE"
	default:
		response.Code = "UNKNOWN_ERROR"
	}

	c.JSON(statusCode, response)
}

// WriteValidationError writes a validation error response
func (r *JSONResponse) WriteValidationError(c *gin.Context, field string, message string) {
	details := map[string]interface{}{
		"field":   field,
		"message": message,
	}
	r.WriteError(c, http.StatusBadRequest, "Validation failed", details)
}

// WriteNotFound writes a not found error response
func (r *JSONResponse) WriteNotFound(c *gin.Context, resource string, id string) {
	details := map[string]interface{}{
		"resource": resource,
		"id":       id,
	}
	r.WriteError(c, http.StatusNotFound, "Resource not found", details)
}

// WriteRateLimitError writes a rate limit error response
func (r *JSONResponse) WriteRateLimitError(c *gin.Context, limit int, window string) {
	details := map[string]interface{}{
		"limit":  limit,
		"window": window,
	}
	r.WriteError(c, http.StatusTooManyRequests, "Rate limit exceeded", details)
}

// Helper methods

func (r *JSONResponse) convertToTimeSeriesResponse(ts *storage.TimeSeries) *TimeSeriesResponse {
	var startTime, endTime time.Time
	if len(ts.Timestamps) > 0 {
		startTime = ts.Timestamps[0]
		endTime = ts.Timestamps[len(ts.Timestamps)-1]
	}

	return &TimeSeriesResponse{
		SeriesID:   ts.SeriesID,
		Name:       ts.Name,
		Generator:  ts.Generator,
		Length:     len(ts.Values),
		StartTime:  startTime,
		EndTime:    endTime,
		Timestamps: ts.Timestamps,
		Values:     ts.Values,
		Quality:    ts.Quality,
		Metadata:   ts.Metadata,
		CreatedAt:  ts.CreatedAt,
		UpdatedAt:  ts.UpdatedAt,
	}
}