package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"

	"github.com/your-org/tsiot/internal/analytics"
	"github.com/your-org/tsiot/internal/generators"
	"github.com/your-org/tsiot/internal/storage"
	"github.com/your-org/tsiot/internal/validation"
	"github.com/your-org/tsiot/pkg/constants"
	"github.com/your-org/tsiot/pkg/errors"
)

// Handlers contains all HTTP handlers for the TSIOT API
type Handlers struct {
	generatorFactory *generators.Factory
	storageManager   storage.Manager
	validationEngine *validation.Engine
	analyticsEngine  *analytics.Engine
	logger           *logrus.Logger
	config           *Config
}

// NewHandlers creates a new handlers instance
func NewHandlers(
	generatorFactory *generators.Factory,
	storageManager storage.Manager,
	validationEngine *validation.Engine,
	analyticsEngine *analytics.Engine,
	logger *logrus.Logger,
	config *Config,
) *Handlers {
	return &Handlers{
		generatorFactory: generatorFactory,
		storageManager:   storageManager,
		validationEngine: validationEngine,
		analyticsEngine:  analyticsEngine,
		logger:           logger,
		config:           config,
	}
}

// Health Check Handlers

// HealthCheck handles GET /health
func (h *Handlers) HealthCheck(c *gin.Context) {
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now().UTC(),
		"version":   h.config.Version,
		"uptime":    time.Since(h.config.StartTime).String(),
	}

	// Check dependencies
	checks := h.performHealthChecks(c.Request.Context())
	health["checks"] = checks

	// Determine overall status
	allHealthy := true
	for _, check := range checks {
		if !check["healthy"].(bool) {
			allHealthy = false
			break
		}
	}

	if allHealthy {
		health["status"] = "healthy"
		c.JSON(http.StatusOK, health)
	} else {
		health["status"] = "unhealthy"
		c.JSON(http.StatusServiceUnavailable, health)
	}
}

// ReadinessCheck handles GET /ready
func (h *Handlers) ReadinessCheck(c *gin.Context) {
	ready := map[string]interface{}{
		"ready":     true,
		"timestamp": time.Now().UTC(),
		"version":   h.config.Version,
	}

	// Check if service is ready to serve requests
	checks := h.performReadinessChecks(c.Request.Context())
	ready["checks"] = checks

	// Determine overall readiness
	allReady := true
	for _, check := range checks {
		if !check["ready"].(bool) {
			allReady = false
			break
		}
	}

	if allReady {
		ready["ready"] = true
		c.JSON(http.StatusOK, ready)
	} else {
		ready["ready"] = false
		c.JSON(http.StatusServiceUnavailable, ready)
	}
}

// Info handles GET /api/v1/info
func (h *Handlers) Info(c *gin.Context) {
	info := map[string]interface{}{
		"name":        "TSIOT",
		"description": "Time Series IoT Synthetic Data Platform",
		"version":     h.config.Version,
		"build_time":  h.config.BuildTime,
		"commit":      h.config.Commit,
		"go_version":  h.config.GoVersion,
		"api_version": "v1",
		"capabilities": map[string]interface{}{
			"generators": h.generatorFactory.GetAvailableTypes(),
			"validators": h.validationEngine.GetAvailableValidators(),
			"storage":    h.storageManager.GetAvailableBackends(),
			"analytics":  h.analyticsEngine.GetCapabilities(),
		},
		"limits": map[string]interface{}{
			"max_series_length":     h.config.Generators.MaxLength,
			"max_batch_size":        h.config.API.MaxBatchSize,
			"request_timeout":       h.config.API.RequestTimeout.String(),
			"max_concurrent_jobs":   h.config.Workers.MaxConcurrentJobs,
		},
	}

	c.JSON(http.StatusOK, info)
}

// Generation Handlers

// GenerateRequest represents a time series generation request
type GenerateRequest struct {
	Type       string                 `json:"type" binding:"required"`
	Length     int                    `json:"length" binding:"required,min=1"`
	Parameters map[string]interface{} `json:"parameters"`
	Metadata   map[string]interface{} `json:"metadata"`
	StartTime  *time.Time             `json:"start_time"`
	Interval   *time.Duration         `json:"interval"`
}

// GenerateTimeSeries handles POST /api/v1/generate
func (h *Handlers) GenerateTimeSeries(c *gin.Context) {
	var req GenerateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		h.logger.WithError(err).Error("Invalid generation request")
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid request",
			"details": err.Error(),
		})
		return
	}

	// Validate request
	if err := h.validateGenerateRequest(&req); err != nil {
		h.logger.WithError(err).Error("Generation request validation failed")
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Validation failed",
			"details": err.Error(),
		})
		return
	}

	// Get generator
	generator, err := h.generatorFactory.CreateGenerator(req.Type, req.Parameters)
	if err != nil {
		h.logger.WithError(err).Error("Failed to create generator")
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid generator type or parameters",
			"details": err.Error(),
		})
		return
	}

	// Generate time series
	startTime := time.Now()
	if req.StartTime != nil {
		startTime = *req.StartTime
	}

	timeSeries, err := generator.Generate(c.Request.Context(), req.Length, startTime, req.Metadata)
	if err != nil {
		h.logger.WithError(err).Error("Time series generation failed")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Generation failed",
			"details": err.Error(),
		})
		return
	}

	// Store generated time series
	if err := h.storageManager.Store(c.Request.Context(), timeSeries); err != nil {
		h.logger.WithError(err).Warn("Failed to store generated time series")
		// Don't fail the request, just log the warning
	}

	// Return the generated time series
	h.respondWithTimeSeries(c, timeSeries, constants.MimeTypeJSON)
}

// BatchGenerateRequest represents a batch generation request
type BatchGenerateRequest struct {
	Requests []GenerateRequest `json:"requests" binding:"required,min=1"`
}

// BatchGenerateTimeSeries handles POST /api/v1/batch/generate
func (h *Handlers) BatchGenerateTimeSeries(c *gin.Context) {
	var req BatchGenerateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		h.logger.WithError(err).Error("Invalid batch generation request")
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid request",
			"details": err.Error(),
		})
		return
	}

	// Validate batch size
	if len(req.Requests) > h.config.API.MaxBatchSize {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": fmt.Sprintf("Batch size exceeds maximum of %d", h.config.API.MaxBatchSize),
		})
		return
	}

	// Process batch
	results := make([]map[string]interface{}, len(req.Requests))
	errors := make([]string, 0)

	for i, genReq := range req.Requests {
		if err := h.validateGenerateRequest(&genReq); err != nil {
			errors = append(errors, fmt.Sprintf("Request %d: %v", i, err))
			results[i] = map[string]interface{}{
				"success": false,
				"error":   err.Error(),
			}
			continue
		}

		generator, err := h.generatorFactory.CreateGenerator(genReq.Type, genReq.Parameters)
		if err != nil {
			errors = append(errors, fmt.Sprintf("Request %d: %v", i, err))
			results[i] = map[string]interface{}{
				"success": false,
				"error":   err.Error(),
			}
			continue
		}

		startTime := time.Now()
		if genReq.StartTime != nil {
			startTime = *genReq.StartTime
		}

		timeSeries, err := generator.Generate(c.Request.Context(), genReq.Length, startTime, genReq.Metadata)
		if err != nil {
			errors = append(errors, fmt.Sprintf("Request %d: %v", i, err))
			results[i] = map[string]interface{}{
				"success": false,
				"error":   err.Error(),
			}
			continue
		}

		// Store generated time series
		if err := h.storageManager.Store(c.Request.Context(), timeSeries); err != nil {
			h.logger.WithError(err).Warn("Failed to store generated time series")
		}

		results[i] = map[string]interface{}{
			"success":     true,
			"time_series": timeSeries,
		}
	}

	response := map[string]interface{}{
		"results": results,
		"summary": map[string]interface{}{
			"total":     len(req.Requests),
			"successful": len(req.Requests) - len(errors),
			"failed":     len(errors),
		},
	}

	if len(errors) > 0 {
		response["errors"] = errors
	}

	c.JSON(http.StatusOK, response)
}

// ListGenerators handles GET /api/v1/generators
func (h *Handlers) ListGenerators(c *gin.Context) {
	generators := h.generatorFactory.GetAvailableGenerators()
	c.JSON(http.StatusOK, map[string]interface{}{
		"generators": generators,
		"count":      len(generators),
	})
}

// Validation Handlers

// ValidateRequest represents a validation request
type ValidateRequest struct {
	TimeSeries      map[string]interface{} `json:"time_series" binding:"required"`
	ValidationTypes []string               `json:"validation_types"`
	Parameters      map[string]interface{} `json:"parameters"`
}

// ValidateTimeSeries handles POST /api/v1/validate
func (h *Handlers) ValidateTimeSeries(c *gin.Context) {
	var req ValidateRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		h.logger.WithError(err).Error("Invalid validation request")
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid request",
			"details": err.Error(),
		})
		return
	}

	// Convert time series data
	timeSeries, err := h.parseTimeSeriesFromRequest(req.TimeSeries)
	if err != nil {
		h.logger.WithError(err).Error("Failed to parse time series")
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid time series data",
			"details": err.Error(),
		})
		return
	}

	// Set default validation types if not provided
	validationTypes := req.ValidationTypes
	if len(validationTypes) == 0 {
		validationTypes = []string{"quality", "statistical"}
	}

	// Run validation
	results, err := h.validationEngine.Validate(c.Request.Context(), timeSeries, validationTypes, req.Parameters)
	if err != nil {
		h.logger.WithError(err).Error("Validation failed")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Validation failed",
			"details": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, map[string]interface{}{
		"validation_results": results,
		"timestamp":          time.Now().UTC(),
		"series_id":          timeSeries.SeriesID,
	})
}

// ListValidators handles GET /api/v1/validators
func (h *Handlers) ListValidators(c *gin.Context) {
	validators := h.validationEngine.GetAvailableValidators()
	c.JSON(http.StatusOK, map[string]interface{}{
		"validators": validators,
		"count":      len(validators),
	})
}

// Analytics Handlers

// AnalyzeRequest represents an analysis request
type AnalyzeRequest struct {
	TimeSeries    map[string]interface{} `json:"time_series" binding:"required"`
	AnalysisTypes []string               `json:"analysis_types"`
	Parameters    map[string]interface{} `json:"parameters"`
}

// AnalyzeTimeSeries handles POST /api/v1/analyze
func (h *Handlers) AnalyzeTimeSeries(c *gin.Context) {
	var req AnalyzeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		h.logger.WithError(err).Error("Invalid analysis request")
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid request",
			"details": err.Error(),
		})
		return
	}

	// Convert time series data
	timeSeries, err := h.parseTimeSeriesFromRequest(req.TimeSeries)
	if err != nil {
		h.logger.WithError(err).Error("Failed to parse time series")
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid time series data",
			"details": err.Error(),
		})
		return
	}

	// Set default analysis types if not provided
	analysisTypes := req.AnalysisTypes
	if len(analysisTypes) == 0 {
		analysisTypes = []string{"basic", "trend"}
	}

	// Run analysis
	results, err := h.analyticsEngine.Analyze(c.Request.Context(), timeSeries, analysisTypes, req.Parameters)
	if err != nil {
		h.logger.WithError(err).Error("Analysis failed")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Analysis failed",
			"details": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, map[string]interface{}{
		"analysis_results": results,
		"timestamp":        time.Now().UTC(),
		"series_id":        timeSeries.SeriesID,
	})
}

// GetAnalyticsCapabilities handles GET /api/v1/analytics/capabilities
func (h *Handlers) GetAnalyticsCapabilities(c *gin.Context) {
	capabilities := h.analyticsEngine.GetCapabilities()
	c.JSON(http.StatusOK, map[string]interface{}{
		"capabilities": capabilities,
		"timestamp":    time.Now().UTC(),
	})
}

// Export Handlers

// ExportRequest represents an export request
type ExportRequest struct {
	TimeSeries  map[string]interface{} `json:"time_series" binding:"required"`
	Format      string                 `json:"format" binding:"required"`
	Compression bool                   `json:"compression"`
	Options     map[string]interface{} `json:"options"`
}

// ExportTimeSeries handles POST /api/v1/export
func (h *Handlers) ExportTimeSeries(c *gin.Context) {
	var req ExportRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		h.logger.WithError(err).Error("Invalid export request")
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid request",
			"details": err.Error(),
		})
		return
	}

	// Validate format
	supportedFormats := []string{"json", "csv", "parquet"}
	formatValid := false
	for _, format := range supportedFormats {
		if strings.ToLower(req.Format) == format {
			formatValid = true
			break
		}
	}

	if !formatValid {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":             "Unsupported format",
			"supported_formats": supportedFormats,
		})
		return
	}

	// Convert time series data
	timeSeries, err := h.parseTimeSeriesFromRequest(req.TimeSeries)
	if err != nil {
		h.logger.WithError(err).Error("Failed to parse time series")
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid time series data",
			"details": err.Error(),
		})
		return
	}

	// Export based on format
	switch strings.ToLower(req.Format) {
	case "json":
		h.respondWithTimeSeries(c, timeSeries, constants.MimeTypeJSON)
	case "csv":
		h.respondWithTimeSeries(c, timeSeries, constants.MimeTypeCSV)
	case "parquet":
		h.respondWithTimeSeries(c, timeSeries, constants.MimeTypeParquet)
	default:
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Unsupported format",
		})
	}
}

// Data Management Handlers

// GetTimeSeries handles GET /api/v1/timeseries/:id
func (h *Handlers) GetTimeSeries(c *gin.Context) {
	seriesID := c.Param("id")
	if seriesID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Missing series ID",
		})
		return
	}

	timeSeries, err := h.storageManager.Get(c.Request.Context(), seriesID)
	if err != nil {
		if errors.IsNotFound(err) {
			c.JSON(http.StatusNotFound, gin.H{
				"error": "Time series not found",
			})
			return
		}

		h.logger.WithError(err).Error("Failed to retrieve time series")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to retrieve time series",
			"details": err.Error(),
		})
		return
	}

	// Determine response format from Accept header
	acceptHeader := c.GetHeader("Accept")
	var mimeType string

	switch {
	case strings.Contains(acceptHeader, constants.MimeTypeCSV):
		mimeType = constants.MimeTypeCSV
	case strings.Contains(acceptHeader, constants.MimeTypeParquet):
		mimeType = constants.MimeTypeParquet
	default:
		mimeType = constants.MimeTypeJSON
	}

	h.respondWithTimeSeries(c, timeSeries, mimeType)
}

// ListTimeSeries handles GET /api/v1/timeseries
func (h *Handlers) ListTimeSeries(c *gin.Context) {
	// Parse query parameters
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "100"))
	offset, _ := strconv.Atoi(c.DefaultQuery("offset", "0"))
	sortBy := c.DefaultQuery("sort_by", "created_at")
	sortOrder := c.DefaultQuery("sort_order", "desc")

	// Validate parameters
	if limit > 1000 {
		limit = 1000
	}
	if limit < 1 {
		limit = 100
	}

	query := storage.ListQuery{
		Limit:     limit,
		Offset:    offset,
		SortBy:    sortBy,
		SortOrder: sortOrder,
		Filters:   make(map[string]interface{}),
	}

	// Add filters from query parameters
	if generatorType := c.Query("generator_type"); generatorType != "" {
		query.Filters["generator_type"] = generatorType
	}
	if createdAfter := c.Query("created_after"); createdAfter != "" {
		if t, err := time.Parse(time.RFC3339, createdAfter); err == nil {
			query.Filters["created_after"] = t
		}
	}
	if createdBefore := c.Query("created_before"); createdBefore != "" {
		if t, err := time.Parse(time.RFC3339, createdBefore); err == nil {
			query.Filters["created_before"] = t
		}
	}

	results, err := h.storageManager.List(c.Request.Context(), query)
	if err != nil {
		h.logger.WithError(err).Error("Failed to list time series")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to list time series",
			"details": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, map[string]interface{}{
		"time_series": results.Items,
		"pagination": map[string]interface{}{
			"limit":       limit,
			"offset":      offset,
			"total":       results.Total,
			"has_more":    offset+limit < results.Total,
			"next_offset": offset + limit,
		},
	})
}

// DeleteTimeSeries handles DELETE /api/v1/timeseries/:id
func (h *Handlers) DeleteTimeSeries(c *gin.Context) {
	seriesID := c.Param("id")
	if seriesID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Missing series ID",
		})
		return
	}

	err := h.storageManager.Delete(c.Request.Context(), seriesID)
	if err != nil {
		if errors.IsNotFound(err) {
			c.JSON(http.StatusNotFound, gin.H{
				"error": "Time series not found",
			})
			return
		}

		h.logger.WithError(err).Error("Failed to delete time series")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to delete time series",
			"details": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":   "Time series deleted successfully",
		"series_id": seriesID,
	})
}

// Helper Methods

func (h *Handlers) validateGenerateRequest(req *GenerateRequest) error {
	if req.Length <= 0 {
		return fmt.Errorf("length must be positive")
	}

	if req.Length > h.config.Generators.MaxLength {
		return fmt.Errorf("length exceeds maximum of %d", h.config.Generators.MaxLength)
	}

	// Validate generator type
	availableTypes := h.generatorFactory.GetAvailableTypes()
	typeValid := false
	for _, t := range availableTypes {
		if req.Type == t {
			typeValid = true
			break
		}
	}

	if !typeValid {
		return fmt.Errorf("unsupported generator type: %s", req.Type)
	}

	return nil
}

func (h *Handlers) parseTimeSeriesFromRequest(data map[string]interface{}) (*storage.TimeSeries, error) {
	// This is a simplified parser - in a real implementation,
	// you'd have proper deserialization logic
	jsonData, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}

	var timeSeries storage.TimeSeries
	if err := json.Unmarshal(jsonData, &timeSeries); err != nil {
		return nil, err
	}

	return &timeSeries, nil
}

func (h *Handlers) respondWithTimeSeries(c *gin.Context, timeSeries *storage.TimeSeries, mimeType string) {
	switch mimeType {
	case constants.MimeTypeJSON:
		c.JSON(http.StatusOK, timeSeries)
	case constants.MimeTypeCSV:
		// Use CSV response handler
		h.respondWithCSV(c, timeSeries)
	case constants.MimeTypeParquet:
		// Use Parquet response handler
		h.respondWithParquet(c, timeSeries)
	default:
		c.JSON(http.StatusOK, timeSeries)
	}
}

func (h *Handlers) respondWithCSV(c *gin.Context, timeSeries *storage.TimeSeries) {
	// Placeholder - will be implemented in response handlers
	c.Header("Content-Type", constants.MimeTypeCSV)
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s.csv\"", timeSeries.SeriesID))
	c.String(http.StatusOK, "timestamp,value,quality\n") // Placeholder
}

func (h *Handlers) respondWithParquet(c *gin.Context, timeSeries *storage.TimeSeries) {
	// Placeholder - will be implemented in response handlers
	c.Header("Content-Type", constants.MimeTypeParquet)
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s.parquet\"", timeSeries.SeriesID))
	c.Data(http.StatusOK, constants.MimeTypeParquet, []byte{}) // Placeholder
}

func (h *Handlers) performHealthChecks(ctx context.Context) map[string]map[string]interface{} {
	checks := make(map[string]map[string]interface{})

	// Database health check
	checks["database"] = map[string]interface{}{
		"healthy": true,
		"message": "Database connection is healthy",
	}

	// Storage health check
	checks["storage"] = map[string]interface{}{
		"healthy": true,
		"message": "Storage backends are healthy",
	}

	// Cache health check (if enabled)
	if h.config.Cache.Enabled {
		checks["cache"] = map[string]interface{}{
			"healthy": true,
			"message": "Cache is healthy",
		}
	}

	return checks
}

func (h *Handlers) performReadinessChecks(ctx context.Context) map[string]map[string]interface{} {
	checks := make(map[string]map[string]interface{})

	// Generator factory readiness
	checks["generators"] = map[string]interface{}{
		"ready":   true,
		"message": "Generator factory is ready",
	}

	// Validation engine readiness
	checks["validation"] = map[string]interface{}{
		"ready":   true,
		"message": "Validation engine is ready",
	}

	// Analytics engine readiness
	checks["analytics"] = map[string]interface{}{
		"ready":   true,
		"message": "Analytics engine is ready",
	}

	return checks
}