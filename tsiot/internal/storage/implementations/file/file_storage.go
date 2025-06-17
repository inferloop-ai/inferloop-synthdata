package file

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
	"github.com/inferloop/tsiot/pkg/errors"
)

// FileStorageConfig contains configuration for file-based storage
type FileStorageConfig struct {
	BasePath       string `json:"base_path" yaml:"base_path"`
	Format         string `json:"format" yaml:"format"`                   // "csv", "json", "parquet"
	Compression    bool   `json:"compression" yaml:"compression"`         // gzip compression
	CreateDirs     bool   `json:"create_dirs" yaml:"create_dirs"`         // auto-create directories
	FileRotation   string `json:"file_rotation" yaml:"file_rotation"`     // "daily", "hourly", "none"
	MaxFileSize    int64  `json:"max_file_size" yaml:"max_file_size"`     // bytes
	BufferSize     int    `json:"buffer_size" yaml:"buffer_size"`         // write buffer size
	SyncWrites     bool   `json:"sync_writes" yaml:"sync_writes"`         // sync after each write
	IndexEnabled   bool   `json:"index_enabled" yaml:"index_enabled"`     // maintain index files
	BackupEnabled  bool   `json:"backup_enabled" yaml:"backup_enabled"`   // create backup files
	RetentionDays  int    `json:"retention_days" yaml:"retention_days"`   // auto-cleanup after N days
}

// FileStorage implements the Storage interface for file-based storage
type FileStorage struct {
	config     *FileStorageConfig
	logger     *logrus.Logger
	mu         sync.RWMutex
	openFiles  map[string]*os.File
	writers    map[string]io.Writer
	indexes    map[string]*FileIndex
	connected  bool
}

// FileIndex maintains an index of time series files
type FileIndex struct {
	SeriesID     string            `json:"series_id"`
	Files        []FileEntry       `json:"files"`
	LastModified time.Time         `json:"last_modified"`
	Metadata     map[string]string `json:"metadata"`
}

// FileEntry represents a single file in the index
type FileEntry struct {
	Path      string    `json:"path"`
	StartTime time.Time `json:"start_time"`
	EndTime   time.Time `json:"end_time"`
	RecordCount int64   `json:"record_count"`
	Size      int64     `json:"size"`
	Format    string    `json:"format"`
	Compressed bool     `json:"compressed"`
}

// NewFileStorage creates a new file storage instance
func NewFileStorage(config *FileStorageConfig, logger *logrus.Logger) (*FileStorage, error) {
	if config == nil {
		return nil, errors.NewValidationError("INVALID_CONFIG", "FileStorageConfig cannot be nil")
	}

	if config.BasePath == "" {
		return nil, errors.NewValidationError("INVALID_CONFIG", "BasePath is required")
	}

	if config.Format == "" {
		config.Format = "csv"
	}

	if config.BufferSize <= 0 {
		config.BufferSize = 4096
	}

	if logger == nil {
		logger = logrus.New()
	}

	return &FileStorage{
		config:    config,
		logger:    logger,
		openFiles: make(map[string]*os.File),
		writers:   make(map[string]io.Writer),
		indexes:   make(map[string]*FileIndex),
		connected: false,
	}, nil
}

// Connect initializes the file storage
func (fs *FileStorage) Connect(ctx context.Context) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	if fs.connected {
		return nil
	}

	// Create base directory if it doesn't exist
	if fs.config.CreateDirs {
		if err := os.MkdirAll(fs.config.BasePath, 0755); err != nil {
			return errors.WrapError(err, errors.ErrorTypeStorage, "DIRECTORY_CREATION_FAILED", 
				fmt.Sprintf("Failed to create directory: %s", fs.config.BasePath))
		}
	}

	// Verify base path exists and is writable
	if _, err := os.Stat(fs.config.BasePath); os.IsNotExist(err) {
		return errors.NewStorageError("PATH_NOT_FOUND", fmt.Sprintf("Base path does not exist: %s", fs.config.BasePath))
	}

	// Test write permissions
	testFile := filepath.Join(fs.config.BasePath, ".write_test")
	if file, err := os.Create(testFile); err != nil {
		return errors.NewStorageError("PERMISSION_DENIED", fmt.Sprintf("Cannot write to directory: %s", fs.config.BasePath))
	} else {
		file.Close()
		os.Remove(testFile)
	}

	// Load existing indexes
	if fs.config.IndexEnabled {
		if err := fs.loadIndexes(); err != nil {
			fs.logger.WithError(err).Warn("Failed to load indexes, continuing without them")
		}
	}

	// Start cleanup routine if retention is enabled
	if fs.config.RetentionDays > 0 {
		go fs.cleanupRoutine(ctx)
	}

	fs.connected = true
	fs.logger.WithField("base_path", fs.config.BasePath).Info("File storage connected")

	return nil
}

// Close closes all open files and cleans up resources
func (fs *FileStorage) Close() error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	if !fs.connected {
		return nil
	}

	// Close all open files
	for path, file := range fs.openFiles {
		if err := file.Close(); err != nil {
			fs.logger.WithError(err).WithField("file", path).Error("Failed to close file")
		}
	}

	// Save indexes
	if fs.config.IndexEnabled {
		if err := fs.saveIndexes(); err != nil {
			fs.logger.WithError(err).Error("Failed to save indexes")
		}
	}

	fs.openFiles = make(map[string]*os.File)
	fs.writers = make(map[string]io.Writer)
	fs.connected = false

	fs.logger.Info("File storage disconnected")
	return nil
}

// HealthCheck verifies the storage is accessible
func (fs *FileStorage) HealthCheck(ctx context.Context) error {
	if !fs.connected {
		return errors.NewStorageError("NOT_CONNECTED", "File storage is not connected")
	}

	// Check if base path is still accessible
	if _, err := os.Stat(fs.config.BasePath); err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "HEALTH_CHECK_FAILED", "Base path is not accessible")
	}

	return nil
}

// Write stores a time series to file
func (fs *FileStorage) Write(ctx context.Context, timeSeries *models.TimeSeries) error {
	if !fs.connected {
		return errors.NewStorageError("NOT_CONNECTED", "File storage is not connected")
	}

	if timeSeries == nil {
		return errors.NewValidationError("INVALID_INPUT", "TimeSeries cannot be nil")
	}

	fs.mu.Lock()
	defer fs.mu.Unlock()

	// Determine file path
	filePath := fs.getFilePath(timeSeries)
	
	// Ensure directory exists
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "DIRECTORY_CREATION_FAILED", 
			fmt.Sprintf("Failed to create directory: %s", dir))
	}

	// Write data based on format
	switch strings.ToLower(fs.config.Format) {
	case "csv":
		return fs.writeCSV(filePath, timeSeries)
	case "json":
		return fs.writeJSON(filePath, timeSeries)
	default:
		return errors.NewValidationError("UNSUPPORTED_FORMAT", fmt.Sprintf("Unsupported format: %s", fs.config.Format))
	}
}

// Read retrieves a time series from file
func (fs *FileStorage) Read(ctx context.Context, seriesID string) (*models.TimeSeries, error) {
	if !fs.connected {
		return nil, errors.NewStorageError("NOT_CONNECTED", "File storage is not connected")
	}

	fs.mu.RLock()
	defer fs.mu.RUnlock()

	// Try to find files for this series
	files, err := fs.findSeriesFiles(seriesID)
	if err != nil {
		return nil, err
	}

	if len(files) == 0 {
		return nil, errors.NewStorageError("NOT_FOUND", fmt.Sprintf("No files found for series: %s", seriesID))
	}

	// Read from all files and merge
	var allDataPoints []models.DataPoint
	var metadata models.Metadata

	for _, filePath := range files {
		timeSeries, err := fs.readFile(filePath)
		if err != nil {
			fs.logger.WithError(err).WithField("file", filePath).Error("Failed to read file")
			continue
		}

		allDataPoints = append(allDataPoints, timeSeries.DataPoints...)
		if metadata.SeriesID == "" {
			metadata = timeSeries.Metadata
		}
	}

	if len(allDataPoints) == 0 {
		return nil, errors.NewStorageError("NO_DATA", fmt.Sprintf("No data found for series: %s", seriesID))
	}

	// Sort by timestamp
	sort.Slice(allDataPoints, func(i, j int) bool {
		return allDataPoints[i].Timestamp.Before(allDataPoints[j].Timestamp)
	})

	return &models.TimeSeries{
		ID:          seriesID,
		Name:        metadata.Name,
		Description: metadata.Description,
		DataPoints:  allDataPoints,
		StartTime:   allDataPoints[0].Timestamp,
		EndTime:     allDataPoints[len(allDataPoints)-1].Timestamp,
		Frequency:   metadata.Frequency,
		SensorType:  metadata.SensorType,
		Tags:        metadata.Tags,
		Metadata:    metadata,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}, nil
}

// Delete removes a time series
func (fs *FileStorage) Delete(ctx context.Context, seriesID string) error {
	if !fs.connected {
		return errors.NewStorageError("NOT_CONNECTED", "File storage is not connected")
	}

	fs.mu.Lock()
	defer fs.mu.Unlock()

	// Find all files for this series
	files, err := fs.findSeriesFiles(seriesID)
	if err != nil {
		return err
	}

	// Delete each file
	for _, filePath := range files {
		if err := os.Remove(filePath); err != nil && !os.IsNotExist(err) {
			fs.logger.WithError(err).WithField("file", filePath).Error("Failed to delete file")
		}
	}

	// Remove from index
	if fs.config.IndexEnabled {
		delete(fs.indexes, seriesID)
	}

	fs.logger.WithField("series_id", seriesID).Info("Time series deleted")
	return nil
}

// List returns a list of available time series
func (fs *FileStorage) List(ctx context.Context, limit, offset int) ([]*models.TimeSeries, error) {
	if !fs.connected {
		return nil, errors.NewStorageError("NOT_CONNECTED", "File storage is not connected")
	}

	fs.mu.RLock()
	defer fs.mu.RUnlock()

	// Use index if available
	if fs.config.IndexEnabled && len(fs.indexes) > 0 {
		return fs.listFromIndex(limit, offset)
	}

	// Scan directory for files
	return fs.listFromFiles(limit, offset)
}

// Helper methods

func (fs *FileStorage) getFilePath(timeSeries *models.TimeSeries) string {
	// Create hierarchical path: basePath/year/month/seriesID_timestamp.format
	now := time.Now()
	if len(timeSeries.DataPoints) > 0 {
		now = timeSeries.DataPoints[0].Timestamp
	}

	var fileName string
	switch fs.config.FileRotation {
	case "daily":
		fileName = fmt.Sprintf("%s_%s.%s", timeSeries.ID, now.Format("2006-01-02"), fs.config.Format)
	case "hourly":
		fileName = fmt.Sprintf("%s_%s.%s", timeSeries.ID, now.Format("2006-01-02_15"), fs.config.Format)
	default:
		fileName = fmt.Sprintf("%s.%s", timeSeries.ID, fs.config.Format)
	}

	if fs.config.Compression {
		fileName += ".gz"
	}

	return filepath.Join(fs.config.BasePath, 
		strconv.Itoa(now.Year()), 
		fmt.Sprintf("%02d", now.Month()), 
		fileName)
}

func (fs *FileStorage) writeCSV(filePath string, timeSeries *models.TimeSeries) error {
	file, err := os.OpenFile(filePath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "FILE_OPEN_FAILED", 
			fmt.Sprintf("Failed to open file: %s", filePath))
	}
	defer file.Close()

	// Check if file is empty (need header)
	stat, err := file.Stat()
	if err != nil {
		return err
	}

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header if file is new
	if stat.Size() == 0 {
		header := []string{"timestamp", "value", "quality"}
		if err := writer.Write(header); err != nil {
			return err
		}
	}

	// Write data points
	for _, dp := range timeSeries.DataPoints {
		record := []string{
			dp.Timestamp.Format(time.RFC3339Nano),
			strconv.FormatFloat(dp.Value, 'f', -1, 64),
			strconv.FormatFloat(dp.Quality, 'f', -1, 64),
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	// Update index
	if fs.config.IndexEnabled {
		fs.updateIndex(timeSeries.ID, filePath, timeSeries)
	}

	return nil
}

func (fs *FileStorage) writeJSON(filePath string, timeSeries *models.TimeSeries) error {
	// For JSON, we'll write the complete time series to the file
	file, err := os.Create(filePath)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "FILE_CREATE_FAILED", 
			fmt.Sprintf("Failed to create file: %s", filePath))
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	
	if err := encoder.Encode(timeSeries); err != nil {
		return errors.WrapError(err, errors.ErrorTypeStorage, "JSON_ENCODE_FAILED", "Failed to encode JSON")
	}

	// Update index
	if fs.config.IndexEnabled {
		fs.updateIndex(timeSeries.ID, filePath, timeSeries)
	}

	return nil
}

func (fs *FileStorage) readFile(filePath string) (*models.TimeSeries, error) {
	ext := strings.ToLower(filepath.Ext(filePath))
	
	// Handle compressed files
	if strings.HasSuffix(ext, ".gz") {
		ext = strings.TrimSuffix(ext, ".gz")
	}

	switch ext {
	case ".csv":
		return fs.readCSV(filePath)
	case ".json":
		return fs.readJSON(filePath)
	default:
		return nil, errors.NewValidationError("UNSUPPORTED_FORMAT", fmt.Sprintf("Unsupported file format: %s", ext))
	}
}

func (fs *FileStorage) readCSV(filePath string) (*models.TimeSeries, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "FILE_OPEN_FAILED", 
			fmt.Sprintf("Failed to open file: %s", filePath))
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "CSV_READ_FAILED", "Failed to read CSV")
	}

	if len(records) < 2 {
		return nil, errors.NewValidationError("INVALID_DATA", "CSV file must have at least header and one data row")
	}

	// Parse header
	header := records[0]
	timestampCol, valueCol, qualityCol := -1, -1, -1

	for i, col := range header {
		switch strings.ToLower(col) {
		case "timestamp", "time", "date":
			timestampCol = i
		case "value", "data":
			valueCol = i
		case "quality":
			qualityCol = i
		}
	}

	if timestampCol == -1 || valueCol == -1 {
		return nil, errors.NewValidationError("INVALID_FORMAT", "CSV must have timestamp and value columns")
	}

	// Parse data
	var dataPoints []models.DataPoint
	for i, record := range records[1:] {
		if len(record) <= timestampCol || len(record) <= valueCol {
			continue
		}

		timestamp, err := time.Parse(time.RFC3339Nano, record[timestampCol])
		if err != nil {
			// Try other formats
			if timestamp, err = time.Parse(time.RFC3339, record[timestampCol]); err != nil {
				fs.logger.WithError(err).WithField("row", i+2).Warn("Failed to parse timestamp")
				continue
			}
		}

		value, err := strconv.ParseFloat(record[valueCol], 64)
		if err != nil {
			fs.logger.WithError(err).WithField("row", i+2).Warn("Failed to parse value")
			continue
		}

		quality := 1.0
		if qualityCol != -1 && len(record) > qualityCol {
			if q, err := strconv.ParseFloat(record[qualityCol], 64); err == nil {
				quality = q
			}
		}

		dataPoints = append(dataPoints, models.DataPoint{
			Timestamp: timestamp,
			Value:     value,
			Quality:   quality,
		})
	}

	if len(dataPoints) == 0 {
		return nil, errors.NewValidationError("NO_DATA", "No valid data points found in CSV")
	}

	// Extract series ID from filename
	seriesID := strings.TrimSuffix(filepath.Base(filePath), filepath.Ext(filePath))
	if idx := strings.LastIndex(seriesID, "_"); idx != -1 {
		seriesID = seriesID[:idx]
	}

	return &models.TimeSeries{
		ID:         seriesID,
		Name:       seriesID,
		DataPoints: dataPoints,
		StartTime:  dataPoints[0].Timestamp,
		EndTime:    dataPoints[len(dataPoints)-1].Timestamp,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}, nil
}

func (fs *FileStorage) readJSON(filePath string) (*models.TimeSeries, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "FILE_OPEN_FAILED", 
			fmt.Sprintf("Failed to open file: %s", filePath))
	}
	defer file.Close()

	var timeSeries models.TimeSeries
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&timeSeries); err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeStorage, "JSON_DECODE_FAILED", "Failed to decode JSON")
	}

	return &timeSeries, nil
}

func (fs *FileStorage) findSeriesFiles(seriesID string) ([]string, error) {
	var files []string

	err := filepath.Walk(fs.config.BasePath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		// Check if filename starts with seriesID
		filename := filepath.Base(path)
		if strings.HasPrefix(filename, seriesID+"_") || strings.HasPrefix(filename, seriesID+".") {
			files = append(files, path)
		}

		return nil
	})

	return files, err
}

func (fs *FileStorage) listFromIndex(limit, offset int) ([]*models.TimeSeries, error) {
	var series []*models.TimeSeries
	count := 0

	for seriesID := range fs.indexes {
		if count < offset {
			count++
			continue
		}

		if len(series) >= limit {
			break
		}

		// Read basic info from index
		ts := &models.TimeSeries{
			ID:        seriesID,
			Name:      seriesID,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}

		series = append(series, ts)
		count++
	}

	return series, nil
}

func (fs *FileStorage) listFromFiles(limit, offset int) ([]*models.TimeSeries, error) {
	seriesMap := make(map[string]*models.TimeSeries)

	err := filepath.Walk(fs.config.BasePath, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}

		// Extract series ID from filename
		filename := filepath.Base(path)
		ext := filepath.Ext(filename)
		seriesID := strings.TrimSuffix(filename, ext)
		
		if idx := strings.LastIndex(seriesID, "_"); idx != -1 {
			seriesID = seriesID[:idx]
		}

		if _, exists := seriesMap[seriesID]; !exists {
			seriesMap[seriesID] = &models.TimeSeries{
				ID:        seriesID,
				Name:      seriesID,
				CreatedAt: info.ModTime(),
				UpdatedAt: info.ModTime(),
			}
		}

		return nil
	})

	if err != nil {
		return nil, err
	}

	// Convert map to slice
	var allSeries []*models.TimeSeries
	for _, ts := range seriesMap {
		allSeries = append(allSeries, ts)
	}

	// Apply offset and limit
	start := offset
	if start > len(allSeries) {
		start = len(allSeries)
	}

	end := start + limit
	if end > len(allSeries) {
		end = len(allSeries)
	}

	return allSeries[start:end], nil
}

func (fs *FileStorage) updateIndex(seriesID, filePath string, timeSeries *models.TimeSeries) {
	if !fs.config.IndexEnabled {
		return
	}

	index, exists := fs.indexes[seriesID]
	if !exists {
		index = &FileIndex{
			SeriesID: seriesID,
			Files:    make([]FileEntry, 0),
			Metadata: make(map[string]string),
		}
		fs.indexes[seriesID] = index
	}

	// Get file info
	stat, err := os.Stat(filePath)
	if err != nil {
		return
	}

	entry := FileEntry{
		Path:        filePath,
		Size:        stat.Size(),
		Format:      fs.config.Format,
		Compressed:  fs.config.Compression,
		RecordCount: int64(len(timeSeries.DataPoints)),
	}

	if len(timeSeries.DataPoints) > 0 {
		entry.StartTime = timeSeries.DataPoints[0].Timestamp
		entry.EndTime = timeSeries.DataPoints[len(timeSeries.DataPoints)-1].Timestamp
	}

	// Add or update file entry
	found := false
	for i, existing := range index.Files {
		if existing.Path == filePath {
			index.Files[i] = entry
			found = true
			break
		}
	}

	if !found {
		index.Files = append(index.Files, entry)
	}

	index.LastModified = time.Now()
}

func (fs *FileStorage) loadIndexes() error {
	indexPath := filepath.Join(fs.config.BasePath, ".indexes.json")
	
	file, err := os.Open(indexPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // No indexes to load
		}
		return err
	}
	defer file.Close()

	return json.NewDecoder(file).Decode(&fs.indexes)
}

func (fs *FileStorage) saveIndexes() error {
	indexPath := filepath.Join(fs.config.BasePath, ".indexes.json")
	
	file, err := os.Create(indexPath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(fs.indexes)
}

func (fs *FileStorage) cleanupRoutine(ctx context.Context) {
	ticker := time.NewTicker(24 * time.Hour) // Run daily
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			fs.performCleanup()
		}
	}
}

func (fs *FileStorage) performCleanup() {
	if fs.config.RetentionDays <= 0 {
		return
	}

	cutoff := time.Now().AddDate(0, 0, -fs.config.RetentionDays)

	err := filepath.Walk(fs.config.BasePath, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}

		if info.ModTime().Before(cutoff) {
			if err := os.Remove(path); err != nil {
				fs.logger.WithError(err).WithField("file", path).Error("Failed to cleanup old file")
			} else {
				fs.logger.WithField("file", path).Info("Cleaned up old file")
			}
		}

		return nil
	})

	if err != nil {
		fs.logger.WithError(err).Error("Error during cleanup routine")
	}
}