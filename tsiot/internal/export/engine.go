package export

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/models"
)

// ExportEngine manages multi-format data exports
type ExportEngine struct {
	logger     *logrus.Logger
	config     *ExportConfig
	mu         sync.RWMutex
	exporters  map[string]Exporter
	jobs       map[string]*ExportJob
	jobQueue   chan *ExportJob
	workers    chan struct{}
	stopCh     chan struct{}
	wg         sync.WaitGroup
}

// ExportConfig configures the export engine
type ExportConfig struct {
	Enabled            bool          `json:"enabled"`
	MaxConcurrentJobs  int           `json:"max_concurrent_jobs"`
	JobTimeout         time.Duration `json:"job_timeout"`
	MaxFileSize        int64         `json:"max_file_size"`        // bytes
	TempDirectory      string        `json:"temp_directory"`
	OutputDirectory    string        `json:"output_directory"`
	EnableCompression  bool          `json:"enable_compression"`
	CompressionLevel   int           `json:"compression_level"`
	CleanupInterval    time.Duration `json:"cleanup_interval"`
	RetentionPeriod    time.Duration `json:"retention_period"`
	EnableMetrics      bool          `json:"enable_metrics"`
	BufferSize         int           `json:"buffer_size"`
}

// ExportFormat defines supported export formats
type ExportFormat string

const (
	FormatCSV     ExportFormat = "csv"
	FormatJSON    ExportFormat = "json"
	FormatParquet ExportFormat = "parquet"
	FormatAvro    ExportFormat = "avro"
	FormatHDF5    ExportFormat = "hdf5"
	FormatXLSX    ExportFormat = "xlsx"
	FormatXML     ExportFormat = "xml"
	FormatInflux  ExportFormat = "influx"
	FormatArrow   ExportFormat = "arrow"
)

// CompressionType defines compression options
type CompressionType string

const (
	CompressionNone   CompressionType = "none"
	CompressionGzip   CompressionType = "gzip"
	CompressionSnappy CompressionType = "snappy"
	CompressionLZ4    CompressionType = "lz4"
	CompressionZstd   CompressionType = "zstd"
)

// ExportJob represents an export operation
type ExportJob struct {
	ID            string              `json:"id"`
	Format        ExportFormat        `json:"format"`
	Compression   CompressionType     `json:"compression"`
	TimeSeries    []*models.TimeSeries `json:"time_series"`
	Options       ExportOptions       `json:"options"`
	Status        JobStatus           `json:"status"`
	Progress      float64             `json:"progress"`
	CreatedAt     time.Time           `json:"created_at"`
	StartedAt     *time.Time          `json:"started_at,omitempty"`
	CompletedAt   *time.Time          `json:"completed_at,omitempty"`
	FilePath      string              `json:"file_path,omitempty"`
	FileSize      int64               `json:"file_size,omitempty"`
	Error         error               `json:"error,omitempty"`
	Metadata      map[string]interface{} `json:"metadata"`
	Context       context.Context     `json:"-"`
}

// JobStatus defines export job status
type JobStatus string

const (
	JobStatusPending    JobStatus = "pending"
	JobStatusRunning    JobStatus = "running"
	JobStatusCompleted  JobStatus = "completed"
	JobStatusFailed     JobStatus = "failed"
	JobStatusCancelled  JobStatus = "cancelled"
)

// ExportOptions contains export-specific options
type ExportOptions struct {
	// Common options
	IncludeHeaders     bool              `json:"include_headers"`
	DateFormat         string            `json:"date_format"`
	TimeZone           string            `json:"timezone"`
	Precision          int               `json:"precision"`
	CustomFields       map[string]string `json:"custom_fields"`
	Filter             FilterOptions     `json:"filter"`
	Aggregation        AggregationOptions `json:"aggregation"`
	
	// Format-specific options
	CSVOptions         CSVOptions        `json:"csv_options,omitempty"`
	JSONOptions        JSONOptions       `json:"json_options,omitempty"`
	ParquetOptions     ParquetOptions    `json:"parquet_options,omitempty"`
	AvroOptions        AvroOptions       `json:"avro_options,omitempty"`
	HDF5Options        HDF5Options       `json:"hdf5_options,omitempty"`
	
	// Output options
	SplitBySize        bool              `json:"split_by_size"`
	MaxRecordsPerFile  int               `json:"max_records_per_file"`
	FilePrefix         string            `json:"file_prefix"`
	FileSuffix         string            `json:"file_suffix"`
}

// FilterOptions defines data filtering options
type FilterOptions struct {
	StartTime      *time.Time `json:"start_time,omitempty"`
	EndTime        *time.Time `json:"end_time,omitempty"`
	MinValue       *float64   `json:"min_value,omitempty"`
	MaxValue       *float64   `json:"max_value,omitempty"`
	SensorTypes    []string   `json:"sensor_types,omitempty"`
	Tags           map[string]string `json:"tags,omitempty"`
	QualityMin     *float64   `json:"quality_min,omitempty"`
}

// AggregationOptions defines data aggregation options
type AggregationOptions struct {
	Enabled    bool          `json:"enabled"`
	Interval   time.Duration `json:"interval"`
	Function   string        `json:"function"` // mean, sum, min, max, count, etc.
	GroupBy    []string      `json:"group_by"`
}

// Format-specific options
type CSVOptions struct {
	Delimiter    string `json:"delimiter"`
	Quote        string `json:"quote"`
	Escape       string `json:"escape"`
	LineEnding   string `json:"line_ending"`
	NullValue    string `json:"null_value"`
}

type JSONOptions struct {
	Pretty       bool   `json:"pretty"`
	ArrayFormat  bool   `json:"array_format"`
	StreamFormat bool   `json:"stream_format"`
}

type ParquetOptions struct {
	CompressionCodec string `json:"compression_codec"`
	RowGroupSize     int    `json:"row_group_size"`
	PageSize         int    `json:"page_size"`
	Version          string `json:"version"`
}

type AvroOptions struct {
	Schema      string `json:"schema"`
	Compression string `json:"compression"`
}

type HDF5Options struct {
	Dataset     string `json:"dataset"`
	Compression string `json:"compression"`
	ChunkSize   int    `json:"chunk_size"`
}

// ExportResult contains export operation results
type ExportResult struct {
	JobID       string                 `json:"job_id"`
	Format      ExportFormat           `json:"format"`
	Files       []ExportedFile         `json:"files"`
	TotalSize   int64                  `json:"total_size"`
	RecordCount int64                  `json:"record_count"`
	Duration    time.Duration          `json:"duration"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// ExportedFile represents an exported file
type ExportedFile struct {
	Path        string    `json:"path"`
	Size        int64     `json:"size"`
	RecordCount int64     `json:"record_count"`
	Checksum    string    `json:"checksum"`
	CreatedAt   time.Time `json:"created_at"`
}

// Exporter interface for format-specific exporters
type Exporter interface {
	Name() string
	SupportedFormats() []ExportFormat
	Export(ctx context.Context, writer io.Writer, data []*models.TimeSeries, options ExportOptions) error
	ValidateOptions(options ExportOptions) error
	EstimateSize(data []*models.TimeSeries, options ExportOptions) (int64, error)
}

// ProgressCallback defines progress reporting callback
type ProgressCallback func(jobID string, progress float64, message string)

// NewExportEngine creates a new export engine
func NewExportEngine(config *ExportConfig, logger *logrus.Logger) (*ExportEngine, error) {
	if config == nil {
		config = getDefaultExportConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	engine := &ExportEngine{
		logger:    logger,
		config:    config,
		exporters: make(map[string]Exporter),
		jobs:      make(map[string]*ExportJob),
		jobQueue:  make(chan *ExportJob, 1000),
		workers:   make(chan struct{}, config.MaxConcurrentJobs),
		stopCh:    make(chan struct{}),
	}

	// Register default exporters
	engine.registerDefaultExporters()

	return engine, nil
}

// Start starts the export engine
func (ee *ExportEngine) Start(ctx context.Context) error {
	if !ee.config.Enabled {
		ee.logger.Info("Export engine disabled")
		return nil
	}

	ee.logger.Info("Starting export engine")

	// Start worker pool
	for i := 0; i < ee.config.MaxConcurrentJobs; i++ {
		ee.wg.Add(1)
		go ee.worker(ctx)
	}

	// Start cleanup routine
	go ee.cleanupRoutine(ctx)

	return nil
}

// Stop stops the export engine
func (ee *ExportEngine) Stop(ctx context.Context) error {
	ee.logger.Info("Stopping export engine")

	close(ee.stopCh)
	close(ee.jobQueue)

	ee.wg.Wait()

	return nil
}

// RegisterExporter registers a custom exporter
func (ee *ExportEngine) RegisterExporter(exporter Exporter) {
	ee.mu.Lock()
	defer ee.mu.Unlock()

	ee.exporters[exporter.Name()] = exporter
	ee.logger.WithField("exporter", exporter.Name()).Info("Registered exporter")
}

// SubmitExportJob submits an export job
func (ee *ExportEngine) SubmitExportJob(job *ExportJob) error {
	if job.ID == "" {
		job.ID = fmt.Sprintf("export_%d", time.Now().UnixNano())
	}

	job.CreatedAt = time.Now()
	job.Status = JobStatusPending
	job.Metadata = make(map[string]interface{})

	// Validate job
	if err := ee.validateJob(job); err != nil {
		return fmt.Errorf("invalid export job: %w", err)
	}

	// Store job
	ee.mu.Lock()
	ee.jobs[job.ID] = job
	ee.mu.Unlock()

	// Queue job
	select {
	case ee.jobQueue <- job:
		ee.logger.WithFields(logrus.Fields{
			"job_id": job.ID,
			"format": job.Format,
		}).Debug("Submitted export job")
		return nil
	default:
		return fmt.Errorf("export queue is full")
	}
}

// GetJobStatus returns the status of an export job
func (ee *ExportEngine) GetJobStatus(jobID string) (*ExportJob, error) {
	ee.mu.RLock()
	defer ee.mu.RUnlock()

	job, exists := ee.jobs[jobID]
	if !exists {
		return nil, fmt.Errorf("job %s not found", jobID)
	}

	return job, nil
}

// CancelJob cancels an export job
func (ee *ExportEngine) CancelJob(jobID string) error {
	ee.mu.Lock()
	defer ee.mu.Unlock()

	job, exists := ee.jobs[jobID]
	if !exists {
		return fmt.Errorf("job %s not found", jobID)
	}

	if job.Status == JobStatusRunning {
		// Cancel the context if available
		if job.Context != nil {
			// In a real implementation, we'd have a cancellation mechanism
		}
	}

	job.Status = JobStatusCancelled
	return nil
}

// ExportTimeSeries exports time series data immediately (synchronous)
func (ee *ExportEngine) ExportTimeSeries(ctx context.Context, data []*models.TimeSeries, format ExportFormat, writer io.Writer, options ExportOptions) error {
	ee.mu.RLock()
	exporter, exists := ee.findExporterForFormat(format)
	ee.mu.RUnlock()

	if !exists {
		return fmt.Errorf("no exporter found for format %s", format)
	}

	// Validate options
	if err := exporter.ValidateOptions(options); err != nil {
		return fmt.Errorf("invalid export options: %w", err)
	}

	// Apply filtering if specified
	filteredData := ee.applyFilters(data, options.Filter)

	// Apply aggregation if specified
	if options.Aggregation.Enabled {
		aggregatedData, err := ee.applyAggregation(filteredData, options.Aggregation)
		if err != nil {
			return fmt.Errorf("aggregation failed: %w", err)
		}
		filteredData = aggregatedData
	}

	// Perform export
	start := time.Now()
	err := exporter.Export(ctx, writer, filteredData, options)
	duration := time.Since(start)

	ee.logger.WithFields(logrus.Fields{
		"format":       format,
		"series_count": len(filteredData),
		"duration":     duration,
	}).Debug("Export completed")

	return err
}

// GetSupportedFormats returns all supported export formats
func (ee *ExportEngine) GetSupportedFormats() []ExportFormat {
	ee.mu.RLock()
	defer ee.mu.RUnlock()

	formats := make(map[ExportFormat]bool)
	for _, exporter := range ee.exporters {
		for _, format := range exporter.SupportedFormats() {
			formats[format] = true
		}
	}

	result := make([]ExportFormat, 0, len(formats))
	for format := range formats {
		result = append(result, format)
	}

	return result
}

// worker processes export jobs
func (ee *ExportEngine) worker(ctx context.Context) {
	defer ee.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ee.stopCh:
			return
		case job, ok := <-ee.jobQueue:
			if !ok {
				return
			}

			ee.processJob(ctx, job)
		}
	}
}

// processJob processes a single export job
func (ee *ExportEngine) processJob(ctx context.Context, job *ExportJob) {
	jobCtx, cancel := context.WithTimeout(ctx, ee.config.JobTimeout)
	defer cancel()

	start := time.Now()
	job.Status = JobStatusRunning
	job.StartedAt = &start
	job.Context = jobCtx

	ee.logger.WithFields(logrus.Fields{
		"job_id": job.ID,
		"format": job.Format,
	}).Debug("Processing export job")

	// Find exporter
	exporter, exists := ee.findExporterForFormat(job.Format)
	if !exists {
		job.Status = JobStatusFailed
		job.Error = fmt.Errorf("no exporter found for format %s", job.Format)
		return
	}

	// Create output file
	outputPath := ee.generateOutputPath(job)
	file, err := ee.createOutputFile(outputPath)
	if err != nil {
		job.Status = JobStatusFailed
		job.Error = fmt.Errorf("failed to create output file: %w", err)
		return
	}
	defer file.Close()

	// Apply filters
	filteredData := ee.applyFilters(job.TimeSeries, job.Options.Filter)

	// Apply aggregation
	if job.Options.Aggregation.Enabled {
		aggregatedData, err := ee.applyAggregation(filteredData, job.Options.Aggregation)
		if err != nil {
			job.Status = JobStatusFailed
			job.Error = fmt.Errorf("aggregation failed: %w", err)
			return
		}
		filteredData = aggregatedData
	}

	// Export data
	err = exporter.Export(jobCtx, file, filteredData, job.Options)
	
	completed := time.Now()
	job.CompletedAt = &completed

	if err != nil {
		job.Status = JobStatusFailed
		job.Error = err
		ee.logger.WithError(err).WithField("job_id", job.ID).Error("Export job failed")
	} else {
		job.Status = JobStatusCompleted
		job.FilePath = outputPath
		
		// Get file size
		if fileInfo, err := file.Stat(); err == nil {
			job.FileSize = fileInfo.Size()
		}

		ee.logger.WithFields(logrus.Fields{
			"job_id":   job.ID,
			"duration": completed.Sub(start),
			"size":     job.FileSize,
		}).Debug("Export job completed")
	}

	job.Progress = 100.0
}

// Helper methods

func (ee *ExportEngine) findExporterForFormat(format ExportFormat) (Exporter, bool) {
	for _, exporter := range ee.exporters {
		for _, supportedFormat := range exporter.SupportedFormats() {
			if supportedFormat == format {
				return exporter, true
			}
		}
	}
	return nil, false
}

func (ee *ExportEngine) validateJob(job *ExportJob) error {
	if len(job.TimeSeries) == 0 {
		return fmt.Errorf("no time series data provided")
	}

	if job.Format == "" {
		return fmt.Errorf("export format not specified")
	}

	// Check if format is supported
	_, exists := ee.findExporterForFormat(job.Format)
	if !exists {
		return fmt.Errorf("unsupported format: %s", job.Format)
	}

	return nil
}

func (ee *ExportEngine) generateOutputPath(job *ExportJob) string {
	timestamp := time.Now().Format("20060102_150405")
	filename := fmt.Sprintf("%s_%s_%s.%s", 
		job.Options.FilePrefix, 
		job.ID, 
		timestamp, 
		job.Format)
	
	return filepath.Join(ee.config.OutputDirectory, filename)
}

func (ee *ExportEngine) createOutputFile(path string) (io.WriteCloser, error) {
	// In a real implementation, this would create the actual file
	// For now, return a dummy writer
	return &dummyWriter{}, nil
}

// dummyWriter is a placeholder writer
type dummyWriter struct{}

func (dw *dummyWriter) Write(p []byte) (n int, error) {
	return len(p), nil
}

func (dw *dummyWriter) Close() error {
	return nil
}

func (ee *ExportEngine) applyFilters(data []*models.TimeSeries, filter FilterOptions) []*models.TimeSeries {
	filtered := make([]*models.TimeSeries, 0)

	for _, ts := range data {
		// Apply sensor type filter
		if len(filter.SensorTypes) > 0 {
			found := false
			for _, sensorType := range filter.SensorTypes {
				if ts.SensorType == sensorType {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}

		// Filter data points
		filteredPoints := make([]models.DataPoint, 0)
		for _, dp := range ts.DataPoints {
			// Time range filter
			if filter.StartTime != nil && dp.Timestamp.Before(*filter.StartTime) {
				continue
			}
			if filter.EndTime != nil && dp.Timestamp.After(*filter.EndTime) {
				continue
			}

			// Value range filter
			if filter.MinValue != nil && dp.Value < *filter.MinValue {
				continue
			}
			if filter.MaxValue != nil && dp.Value > *filter.MaxValue {
				continue
			}

			// Quality filter
			if filter.QualityMin != nil && dp.Quality < *filter.QualityMin {
				continue
			}

			filteredPoints = append(filteredPoints, dp)
		}

		if len(filteredPoints) > 0 {
			filteredTS := *ts
			filteredTS.DataPoints = filteredPoints
			filtered = append(filtered, &filteredTS)
		}
	}

	return filtered
}

func (ee *ExportEngine) applyAggregation(data []*models.TimeSeries, agg AggregationOptions) ([]*models.TimeSeries, error) {
	// Simplified aggregation implementation
	// In a real implementation, this would be more sophisticated
	return data, nil
}

func (ee *ExportEngine) cleanupRoutine(ctx context.Context) {
	ticker := time.NewTicker(ee.config.CleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			ee.performCleanup()
		}
	}
}

func (ee *ExportEngine) performCleanup() {
	ee.mu.Lock()
	defer ee.mu.Unlock()

	cutoff := time.Now().Add(-ee.config.RetentionPeriod)
	
	for jobID, job := range ee.jobs {
		if job.CreatedAt.Before(cutoff) && 
		   (job.Status == JobStatusCompleted || job.Status == JobStatusFailed || job.Status == JobStatusCancelled) {
			delete(ee.jobs, jobID)
		}
	}
}

func (ee *ExportEngine) registerDefaultExporters() {
	// Register CSV exporter
	ee.RegisterExporter(&CSVExporter{})
	
	// Register JSON exporter
	ee.RegisterExporter(&JSONExporter{})
	
	// Register Parquet exporter
	ee.RegisterExporter(&ParquetExporter{})
	
	// Register other exporters would go here
}

func getDefaultExportConfig() *ExportConfig {
	return &ExportConfig{
		Enabled:           true,
		MaxConcurrentJobs: 5,
		JobTimeout:        30 * time.Minute,
		MaxFileSize:       1024 * 1024 * 1024, // 1GB
		TempDirectory:     "/tmp/tsiot/export",
		OutputDirectory:   "/tmp/tsiot/output",
		EnableCompression: true,
		CompressionLevel:  6,
		CleanupInterval:   time.Hour,
		RetentionPeriod:   24 * time.Hour,
		EnableMetrics:     true,
		BufferSize:        64 * 1024,
	}
}