package s3

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
	"github.com/inferloop/tsiot/pkg/errors"
)

// S3Config holds configuration for S3 storage
type S3Config struct {
	Region          string        `json:"region"`
	Bucket          string        `json:"bucket"`
	AccessKeyID     string        `json:"access_key_id"`
	SecretAccessKey string        `json:"secret_access_key"`
	SessionToken    string        `json:"session_token,omitempty"`
	Endpoint        string        `json:"endpoint,omitempty"`
	ForcePathStyle  bool          `json:"force_path_style"`
	DisableSSL      bool          `json:"disable_ssl"`
	Prefix          string        `json:"prefix"`
	Timeout         time.Duration `json:"timeout"`
	MaxRetries      int           `json:"max_retries"`
	PartSize        int64         `json:"part_size"`
	UseCompression  bool          `json:"use_compression"`
	StorageClass    string        `json:"storage_class"`
}

// S3Storage implements the Storage interface for AWS S3
type S3Storage struct {
	config     *S3Config
	s3Client   *s3.S3
	uploader   *s3manager.Uploader
	downloader *s3manager.Downloader
	logger     *logrus.Logger
	mu         sync.RWMutex
	metrics    *storageMetrics
	closed     bool
}

type storageMetrics struct {
	readOps      int64
	writeOps     int64
	deleteOps    int64
	errorCount   int64
	bytesRead    int64
	bytesWritten int64
	startTime    time.Time
	mu           sync.RWMutex
}

// S3Object represents a time series object in S3
type S3Object struct {
	TimeSeries *models.TimeSeries `json:"timeseries"`
	Metadata   map[string]string  `json:"metadata"`
	Version    string            `json:"version"`
	CreatedAt  time.Time         `json:"created_at"`
	UpdatedAt  time.Time         `json:"updated_at"`
}

// NewS3Storage creates a new S3 storage instance
func NewS3Storage(config *S3Config, logger *logrus.Logger) (*S3Storage, error) {
	if config == nil {
		return nil, errors.NewStorageError("INVALID_CONFIG", "S3 config cannot be nil")
	}

	if config.Bucket == "" {
		return nil, errors.NewStorageError("INVALID_CONFIG", "S3 bucket is required")
	}

	if logger == nil {
		logger = logrus.New()
	}

	storage := &S3Storage{
		config: config,
		logger: logger,
		metrics: &storageMetrics{
			startTime: time.Now(),
		},
	}

	return storage, nil
}

// Connect establishes connection to S3
func (s *S3Storage) Connect(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.s3Client != nil {
		return nil // Already connected
	}

	// Create AWS config
	awsConfig := &aws.Config{
		Region:     aws.String(s.config.Region),
		MaxRetries: aws.Int(s.config.MaxRetries),
	}

	// Set credentials if provided
	if s.config.AccessKeyID != "" && s.config.SecretAccessKey != "" {
		awsConfig.Credentials = credentials.NewStaticCredentials(
			s.config.AccessKeyID,
			s.config.SecretAccessKey,
			s.config.SessionToken,
		)
	}

	// Set custom endpoint if provided (for S3-compatible services)
	if s.config.Endpoint != "" {
		awsConfig.Endpoint = aws.String(s.config.Endpoint)
		awsConfig.S3ForcePathStyle = aws.Bool(s.config.ForcePathStyle)
	}

	if s.config.DisableSSL {
		awsConfig.DisableSSL = aws.Bool(true)
	}

	// Create session
	sess, err := session.NewSession(awsConfig)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "SESSION_FAILED", "Failed to create AWS session")
	}

	s.s3Client = s3.New(sess)
	s.uploader = s3manager.NewUploader(sess)
	s.downloader = s3manager.NewDownloader(sess)

	// Configure uploader
	if s.config.PartSize > 0 {
		s.uploader.PartSize = s.config.PartSize
	}

	// Test connection by checking if bucket exists
	_, err = s.s3Client.HeadBucketWithContext(ctx, &s3.HeadBucketInput{
		Bucket: aws.String(s.config.Bucket),
	})

	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "BUCKET_ACCESS_FAILED", 
			fmt.Sprintf("Failed to access bucket '%s'", s.config.Bucket))
	}

	s.logger.WithFields(logrus.Fields{
		"region": s.config.Region,
		"bucket": s.config.Bucket,
	}).Info("Connected to S3")

	return nil
}

// Close closes the S3 connection
func (s *S3Storage) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}

	s.s3Client = nil
	s.uploader = nil
	s.downloader = nil
	s.closed = true

	s.logger.Info("S3 connection closed")
	return nil
}

// Ping tests the S3 connection
func (s *S3Storage) Ping(ctx context.Context) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed || s.s3Client == nil {
		return errors.NewStorageError("NOT_CONNECTED", "S3 not connected")
	}

	_, err := s.s3Client.HeadBucketWithContext(ctx, &s3.HeadBucketInput{
		Bucket: aws.String(s.config.Bucket),
	})

	if err != nil {
		s.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "PING_FAILED", "S3 ping failed")
	}

	return nil
}

// GetInfo returns information about the S3 storage
func (s *S3Storage) GetInfo(ctx context.Context) (*interfaces.StorageInfo, error) {
	return &interfaces.StorageInfo{
		Type:        "s3",
		Version:     "AWS S3 API",
		Name:        "Amazon S3 Storage",
		Description: "Object storage service with high scalability and durability",
		Features: []string{
			"object storage",
			"high durability",
			"scalability",
			"versioning",
			"lifecycle management",
			"server-side encryption",
			"access control",
		},
		Capabilities: interfaces.StorageCapabilities{
			Streaming:      false,
			Transactions:   false,
			Compression:    true,
			Encryption:     true,
			Replication:    true,
			Clustering:     false,
			Backup:         true,
			Archival:       true,
			TimeBasedQuery: false,
			Aggregation:    false,
		},
		Configuration: map[string]interface{}{
			"region":          s.config.Region,
			"bucket":          s.config.Bucket,
			"prefix":          s.config.Prefix,
			"use_compression": s.config.UseCompression,
			"storage_class":   s.config.StorageClass,
		},
	}, nil
}

// Health returns the health status of the storage
func (s *S3Storage) Health(ctx context.Context) (*interfaces.HealthStatus, error) {
	start := time.Now()
	status := "healthy"
	var errors []string
	var warnings []string

	// Test connection
	if err := s.Ping(ctx); err != nil {
		status = "unhealthy"
		errors = append(errors, fmt.Sprintf("Connection failed: %v", err))
	}

	latency := time.Since(start)

	// Check for warnings
	if latency > 500*time.Millisecond {
		warnings = append(warnings, "High latency detected")
	}

	return &interfaces.HealthStatus{
		Status:      status,
		LastCheck:   time.Now(),
		Latency:     latency,
		Connections: 1, // S3 doesn't have persistent connections
		Errors:      errors,
		Warnings:    warnings,
		Metadata: map[string]interface{}{
			"bucket_region": s.config.Region,
			"object_count":  s.getObjectCount(ctx),
		},
	}, nil
}

// Write writes a time series to S3
func (s *S3Storage) Write(ctx context.Context, data *models.TimeSeries) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed || s.s3Client == nil {
		return errors.NewStorageError("NOT_CONNECTED", "S3 not connected")
	}

	if err := data.Validate(); err != nil {
		return errors.WrapError(err, errors.ErrorTypeValidation, "INVALID_DATA", "Time series validation failed")
	}

	start := time.Now()
	defer func() {
		s.incrementWriteOps()
		s.logger.WithField("duration", time.Since(start)).Debug("Write operation completed")
	}()

	// Create S3 object
	s3Object := &S3Object{
		TimeSeries: data,
		Version:    "1.0",
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
		Metadata: map[string]string{
			"content-type": "application/json",
			"series-id":    data.ID,
			"series-name":  data.Name,
		},
	}

	// Serialize to JSON
	jsonData, err := json.Marshal(s3Object)
	if err != nil {
		s.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "SERIALIZATION_FAILED", "Failed to serialize time series")
	}

	// Compress if enabled
	var body io.Reader = bytes.NewReader(jsonData)
	contentEncoding := ""
	
	if s.config.UseCompression {
		var buf bytes.Buffer
		gzWriter := gzip.NewWriter(&buf)
		if _, err := gzWriter.Write(jsonData); err != nil {
			s.incrementErrorCount()
			return errors.WrapError(err, errors.ErrorTypeStorage, "COMPRESSION_FAILED", "Failed to compress data")
		}
		gzWriter.Close()
		
		body = bytes.NewReader(buf.Bytes())
		contentEncoding = "gzip"
		s.incrementBytesWritten(int64(buf.Len()))
	} else {
		s.incrementBytesWritten(int64(len(jsonData)))
	}

	// Generate S3 key
	key := s.generateKey(data.ID)

	// Upload to S3
	uploadInput := &s3manager.UploadInput{
		Bucket:      aws.String(s.config.Bucket),
		Key:         aws.String(key),
		Body:        body,
		ContentType: aws.String("application/json"),
		Metadata: map[string]*string{
			"series-id":    aws.String(data.ID),
			"series-name":  aws.String(data.Name),
			"sensor-type":  aws.String(data.SensorType),
			"created-at":   aws.String(data.CreatedAt.Format(time.RFC3339)),
			"data-points":  aws.String(fmt.Sprintf("%d", len(data.DataPoints))),
		},
	}

	if contentEncoding != "" {
		uploadInput.ContentEncoding = aws.String(contentEncoding)
	}

	if s.config.StorageClass != "" {
		uploadInput.StorageClass = aws.String(s.config.StorageClass)
	}

	_, err = s.uploader.UploadWithContext(ctx, uploadInput)
	if err != nil {
		s.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "UPLOAD_FAILED", "Failed to upload to S3")
	}

	return nil
}

// WriteBatch writes multiple time series in a batch
func (s *S3Storage) WriteBatch(ctx context.Context, batch []*models.TimeSeries) error {
	if len(batch) == 0 {
		return nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed || s.s3Client == nil {
		return errors.NewStorageError("NOT_CONNECTED", "S3 not connected")
	}

	start := time.Now()
	defer func() {
		s.metrics.mu.Lock()
		s.metrics.writeOps += int64(len(batch))
		s.metrics.mu.Unlock()
		s.logger.WithFields(logrus.Fields{
			"count":    len(batch),
			"duration": time.Since(start),
		}).Debug("Batch write operation completed")
	}()

	// Write each time series individually
	// For better performance, could implement parallel uploads
	for _, timeSeries := range batch {
		if err := s.Write(ctx, timeSeries); err != nil {
			return errors.WrapError(err, errors.ErrorTypeStorage, "BATCH_WRITE_FAILED", 
				fmt.Sprintf("Failed to write time series %s in batch", timeSeries.ID))
		}
	}

	return nil
}

// Read reads a time series by ID
func (s *S3Storage) Read(ctx context.Context, id string) (*models.TimeSeries, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed || s.s3Client == nil {
		return nil, errors.NewStorageError("NOT_CONNECTED", "S3 not connected")
	}

	start := time.Now()
	defer func() {
		s.incrementReadOps()
		s.logger.WithField("duration", time.Since(start)).Debug("Read operation completed")
	}()

	key := s.generateKey(id)

	// Download from S3
	buf := aws.NewWriteAtBuffer([]byte{})
	_, err := s.downloader.DownloadWithContext(ctx, buf, &s3.GetObjectInput{
		Bucket: aws.String(s.config.Bucket),
		Key:    aws.String(key),
	})

	if err != nil {
		s.incrementErrorCount()
		if strings.Contains(err.Error(), "NoSuchKey") {
			return nil, errors.NewStorageError("NOT_FOUND", fmt.Sprintf("Time series '%s' not found", id))
		}
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "DOWNLOAD_FAILED", "Failed to download from S3")
	}

	s.incrementBytesRead(int64(len(buf.Bytes())))

	// Handle compression
	data := buf.Bytes()
	if s.config.UseCompression {
		gzReader, err := gzip.NewReader(bytes.NewReader(data))
		if err != nil {
			s.incrementErrorCount()
			return nil, errors.WrapError(err, errors.ErrorTypeStorage, "DECOMPRESSION_FAILED", "Failed to decompress data")
		}
		defer gzReader.Close()

		decompressed, err := io.ReadAll(gzReader)
		if err != nil {
			s.incrementErrorCount()
			return nil, errors.WrapError(err, errors.ErrorTypeStorage, "DECOMPRESSION_FAILED", "Failed to read decompressed data")
		}
		data = decompressed
	}

	// Deserialize
	var s3Object S3Object
	if err := json.Unmarshal(data, &s3Object); err != nil {
		s.incrementErrorCount()
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "DESERIALIZATION_FAILED", "Failed to deserialize time series")
	}

	return s3Object.TimeSeries, nil
}

// ReadRange reads time series data within a time range
func (s *S3Storage) ReadRange(ctx context.Context, id string, start, end time.Time) (*models.TimeSeries, error) {
	// S3 doesn't support range queries natively
	// We need to read the full time series and filter in memory
	timeSeries, err := s.Read(ctx, id)
	if err != nil {
		return nil, err
	}

	// Filter data points by time range
	filteredPoints := make([]models.DataPoint, 0)
	for _, point := range timeSeries.DataPoints {
		if (point.Timestamp.After(start) || point.Timestamp.Equal(start)) &&
			(point.Timestamp.Before(end) || point.Timestamp.Equal(end)) {
			filteredPoints = append(filteredPoints, point)
		}
	}

	timeSeries.DataPoints = filteredPoints
	return timeSeries, nil
}

// Query queries time series data with filters
func (s *S3Storage) Query(ctx context.Context, query *models.TimeSeriesQuery) ([]*models.TimeSeries, error) {
	// S3 doesn't support complex queries natively
	// This is a simplified implementation that lists and filters
	filters := make(map[string]interface{})
	if query.Limit > 0 {
		filters["limit"] = query.Limit
	}
	
	return s.List(ctx, filters)
}

// Delete deletes a time series by ID
func (s *S3Storage) Delete(ctx context.Context, id string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed || s.s3Client == nil {
		return errors.NewStorageError("NOT_CONNECTED", "S3 not connected")
	}

	start := time.Now()
	defer func() {
		s.incrementDeleteOps()
		s.logger.WithField("duration", time.Since(start)).Debug("Delete operation completed")
	}()

	key := s.generateKey(id)

	_, err := s.s3Client.DeleteObjectWithContext(ctx, &s3.DeleteObjectInput{
		Bucket: aws.String(s.config.Bucket),
		Key:    aws.String(key),
	})

	if err != nil {
		s.incrementErrorCount()
		return errors.WrapError(err, errors.ErrorTypeStorage, "DELETE_FAILED", "Failed to delete from S3")
	}

	return nil
}

// DeleteRange deletes time series data within a time range
func (s *S3Storage) DeleteRange(ctx context.Context, id string, start, end time.Time) error {
	// For S3, we need to read, filter, and write back
	timeSeries, err := s.Read(ctx, id)
	if err != nil {
		return err
	}

	// Filter out data points in the range
	filteredPoints := make([]models.DataPoint, 0)
	for _, point := range timeSeries.DataPoints {
		if point.Timestamp.Before(start) || point.Timestamp.After(end) {
			filteredPoints = append(filteredPoints, point)
		}
	}

	timeSeries.DataPoints = filteredPoints
	timeSeries.UpdatedAt = time.Now()

	return s.Write(ctx, timeSeries)
}

// List lists available time series
func (s *S3Storage) List(ctx context.Context, filters map[string]interface{}) ([]*models.TimeSeries, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed || s.s3Client == nil {
		return nil, errors.NewStorageError("NOT_CONNECTED", "S3 not connected")
	}

	start := time.Now()
	defer func() {
		s.incrementReadOps()
		s.logger.WithField("duration", time.Since(start)).Debug("List operation completed")
	}()

	prefix := s.config.Prefix
	if prefix != "" && !strings.HasSuffix(prefix, "/") {
		prefix += "/"
	}

	input := &s3.ListObjectsV2Input{
		Bucket: aws.String(s.config.Bucket),
		Prefix: aws.String(prefix),
	}

	// Apply limit if specified
	if limit, ok := filters["limit"]; ok {
		if l, ok := limit.(int); ok && l > 0 {
			input.MaxKeys = aws.Int64(int64(l))
		}
	}

	var result []*models.TimeSeries
	err := s.s3Client.ListObjectsV2PagesWithContext(ctx, input,
		func(page *s3.ListObjectsV2Output, lastPage bool) bool {
			for _, obj := range page.Contents {
				// Extract time series ID from key
				id := s.extractIDFromKey(*obj.Key)
				if id == "" {
					continue
				}

				// For efficiency, we could just return metadata here
				// For now, we'll read the full object
				timeSeries, err := s.Read(ctx, id)
				if err != nil {
					s.logger.WithError(err).Warn("Failed to read time series during list")
					continue
				}

				result = append(result, timeSeries)
			}
			return true
		})

	if err != nil {
		s.incrementErrorCount()
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "LIST_FAILED", "Failed to list objects from S3")
	}

	return result, nil
}

// Count returns the count of time series matching filters
func (s *S3Storage) Count(ctx context.Context, filters map[string]interface{}) (int64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed || s.s3Client == nil {
		return 0, errors.NewStorageError("NOT_CONNECTED", "S3 not connected")
	}

	prefix := s.config.Prefix
	if prefix != "" && !strings.HasSuffix(prefix, "/") {
		prefix += "/"
	}

	input := &s3.ListObjectsV2Input{
		Bucket: aws.String(s.config.Bucket),
		Prefix: aws.String(prefix),
	}

	var count int64
	err := s.s3Client.ListObjectsV2PagesWithContext(ctx, input,
		func(page *s3.ListObjectsV2Output, lastPage bool) bool {
			count += int64(len(page.Contents))
			return true
		})

	if err != nil {
		s.incrementErrorCount()
		return 0, errors.WrapError(err, errors.ErrorTypeStorage, "COUNT_FAILED", "Failed to count objects in S3")
	}

	return count, nil
}

// GetMetrics returns storage metrics
func (s *S3Storage) GetMetrics(ctx context.Context) (*interfaces.StorageMetrics, error) {
	s.metrics.mu.RLock()
	defer s.metrics.mu.RUnlock()

	return &interfaces.StorageMetrics{
		ReadOperations:    s.metrics.readOps,
		WriteOperations:   s.metrics.writeOps,
		DeleteOperations:  s.metrics.deleteOps,
		AverageReadTime:   time.Millisecond * 200,  // Simplified
		AverageWriteTime:  time.Millisecond * 300, // Simplified
		ErrorCount:        s.metrics.errorCount,
		ConnectionsActive: 1, // S3 doesn't have persistent connections
		ConnectionsIdle:   0,
		DataSize:          s.metrics.bytesWritten,
		RecordCount:       s.metrics.writeOps,
		Uptime:            time.Since(s.metrics.startTime),
	}, nil
}

// Helper methods

func (s *S3Storage) generateKey(id string) string {
	prefix := s.config.Prefix
	if prefix != "" && !strings.HasSuffix(prefix, "/") {
		prefix += "/"
	}
	return path.Join(prefix, "timeseries", fmt.Sprintf("%s.json", id))
}

func (s *S3Storage) extractIDFromKey(key string) string {
	// Extract ID from key like "prefix/timeseries/id.json"
	parts := strings.Split(key, "/")
	if len(parts) == 0 {
		return ""
	}
	
	filename := parts[len(parts)-1]
	if strings.HasSuffix(filename, ".json") {
		return strings.TrimSuffix(filename, ".json")
	}
	
	return ""
}

func (s *S3Storage) getObjectCount(ctx context.Context) int64 {
	count, _ := s.Count(ctx, map[string]interface{}{})
	return count
}

func (s *S3Storage) incrementReadOps() {
	s.metrics.mu.Lock()
	s.metrics.readOps++
	s.metrics.mu.Unlock()
}

func (s *S3Storage) incrementWriteOps() {
	s.metrics.mu.Lock()
	s.metrics.writeOps++
	s.metrics.mu.Unlock()
}

func (s *S3Storage) incrementDeleteOps() {
	s.metrics.mu.Lock()
	s.metrics.deleteOps++
	s.metrics.mu.Unlock()
}

func (s *S3Storage) incrementErrorCount() {
	s.metrics.mu.Lock()
	s.metrics.errorCount++
	s.metrics.mu.Unlock()
}

func (s *S3Storage) incrementBytesRead(bytes int64) {
	s.metrics.mu.Lock()
	s.metrics.bytesRead += bytes
	s.metrics.mu.Unlock()
}

func (s *S3Storage) incrementBytesWritten(bytes int64) {
	s.metrics.mu.Lock()
	s.metrics.bytesWritten += bytes
	s.metrics.mu.Unlock()
}