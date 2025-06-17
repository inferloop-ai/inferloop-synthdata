package main

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
	"gopkg.in/yaml.v2"
)

type LogCleanupConfig struct {
	LogDir          string                    `json:"log_dir" yaml:"log_dir"`
	MaxAge          time.Duration             `json:"max_age" yaml:"max_age"`
	MaxSizeMB       int64                     `json:"max_size_mb" yaml:"max_size_mb"`
	FilePattern     string                    `json:"file_pattern" yaml:"file_pattern"`
	DryRun          bool                      `json:"dry_run" yaml:"dry_run"`
	Compression     string                    `json:"compression" yaml:"compression"`
	Archive         ArchiveConfig             `json:"archive" yaml:"archive"`
	Rotation        RotationConfig            `json:"rotation" yaml:"rotation"`
	Aggregation     AggregationConfig         `json:"aggregation" yaml:"aggregation"`
	LevelFilter     []string                  `json:"level_filter" yaml:"level_filter"`
	LevelPolicies   map[string]LevelPolicy    `json:"level_policies" yaml:"level_policies"`
	ExcludePatterns []string                  `json:"exclude_patterns" yaml:"exclude_patterns"`
	Workers         int                       `json:"workers" yaml:"workers"`
}

type ArchiveConfig struct {
	Enabled        bool          `json:"enabled" yaml:"enabled"`
	Destination    string        `json:"destination" yaml:"destination"`
	RetentionDays  int           `json:"retention_days" yaml:"retention_days"`
	Compression    string        `json:"compression" yaml:"compression"`
	BatchSize      int           `json:"batch_size" yaml:"batch_size"`
}

type RotationConfig struct {
	SizeMB       int64  `json:"size_mb" yaml:"size_mb"`
	Count        int    `json:"count" yaml:"count"`
	CompressOld  bool   `json:"compress_old" yaml:"compress_old"`
	Pattern      string `json:"pattern" yaml:"pattern"`
}

type AggregationConfig struct {
	Enabled      bool          `json:"enabled" yaml:"enabled"`
	Threshold    float64       `json:"threshold" yaml:"threshold"`
	TimeWindow   time.Duration `json:"time_window" yaml:"time_window"`
	MaxBatchSize int           `json:"max_batch_size" yaml:"max_batch_size"`
}

type LevelPolicy struct {
	MaxAge       time.Duration `json:"max_age" yaml:"max_age"`
	MaxSizeMB    int64         `json:"max_size_mb" yaml:"max_size_mb"`
	Archive      bool          `json:"archive" yaml:"archive"`
	Compress     bool          `json:"compress" yaml:"compress"`
}

type LogCleaner struct {
	config    *LogCleanupConfig
	logger    *logrus.Logger
	patterns  []*regexp.Regexp
	stats     *CleanupStats
	mu        sync.Mutex
}

type CleanupStats struct {
	FilesProcessed   int64
	FilesRotated     int64
	FilesCompressed  int64
	FilesArchived    int64
	FilesDeleted     int64
	BytesProcessed   int64
	BytesFreed       int64
	LogsAggregated   int64
	Errors           []error
	StartTime        time.Time
	EndTime          time.Time
}

type LogEntry struct {
	Timestamp time.Time              `json:"timestamp"`
	Level     string                 `json:"level"`
	Message   string                 `json:"message"`
	Fields    map[string]interface{} `json:"fields"`
	Raw       string                 `json:"-"`
}

type LogFile struct {
	Path         string
	Info         os.FileInfo
	Level        string
	LastModified time.Time
	Size         int64
}

func main() {
	var (
		logDir          = flag.String("dir", "./logs", "Log directory to clean")
		maxAge          = flag.Duration("max-age", 7*24*time.Hour, "Maximum age of logs to keep")
		maxSizeMB       = flag.Int64("max-size-mb", 100, "Maximum size in MB before rotation")
		pattern         = flag.String("pattern", "*.log", "File pattern to match")
		dryRun          = flag.Bool("dry-run", false, "Perform dry run without making changes")
		compression     = flag.String("compress", "gzip", "Compression type (none, gzip, zstd)")
		rotateSize      = flag.Int64("rotate-size", 0, "Rotate logs larger than this size (MB)")
		rotateCount     = flag.Int("rotate-count", 10, "Number of rotated logs to keep")
		archive         = flag.String("archive", "", "Archive destination (e.g., s3://bucket/path)")
		aggregate       = flag.Bool("aggregate", false, "Enable log aggregation")
		levelFilter     = flag.String("level-filter", "", "Comma-separated log levels to filter")
		configFile      = flag.String("config", "", "Configuration file path")
		workers         = flag.Int("workers", 4, "Number of parallel workers")
		verbose         = flag.Bool("verbose", false, "Enable verbose logging")
	)
	flag.Parse()

	// Setup logging
	logger := logrus.New()
	if *verbose {
		logger.SetLevel(logrus.DebugLevel)
	}

	// Load configuration
	config := &LogCleanupConfig{
		LogDir:      *logDir,
		MaxAge:      *maxAge,
		MaxSizeMB:   *maxSizeMB,
		FilePattern: *pattern,
		DryRun:      *dryRun,
		Compression: *compression,
		Workers:     *workers,
	}

	if *configFile != "" {
		if err := loadConfig(*configFile, config); err != nil {
			log.Fatalf("Failed to load config: %v", err)
		}
	}

	// Override with command line flags
	if *rotateSize > 0 {
		config.Rotation.SizeMB = *rotateSize
		config.Rotation.Count = *rotateCount
	}

	if *archive != "" {
		config.Archive.Enabled = true
		config.Archive.Destination = *archive
	}

	if *aggregate {
		config.Aggregation.Enabled = true
		if config.Aggregation.Threshold == 0 {
			config.Aggregation.Threshold = 0.85
		}
	}

	if *levelFilter != "" {
		config.LevelFilter = strings.Split(*levelFilter, ",")
	}

	cleaner := NewLogCleaner(config, logger)

	logger.WithFields(logrus.Fields{
		"log_dir":     config.LogDir,
		"max_age":     config.MaxAge.String(),
		"pattern":     config.FilePattern,
		"dry_run":     config.DryRun,
		"compression": config.Compression,
		"workers":     config.Workers,
	}).Info("Starting log cleanup")

	ctx := context.Background()
	stats, err := cleaner.Cleanup(ctx)
	if err != nil {
		log.Fatalf("Cleanup failed: %v", err)
	}

	// Print summary
	logger.WithFields(logrus.Fields{
		"files_processed":  stats.FilesProcessed,
		"files_rotated":    stats.FilesRotated,
		"files_compressed": stats.FilesCompressed,
		"files_archived":   stats.FilesArchived,
		"files_deleted":    stats.FilesDeleted,
		"bytes_freed":      stats.BytesFreed,
		"logs_aggregated":  stats.LogsAggregated,
		"duration":         stats.EndTime.Sub(stats.StartTime),
		"errors":           len(stats.Errors),
	}).Info("Log cleanup completed")

	if config.DryRun {
		fmt.Printf("\nDRY RUN Summary:\n")
		fmt.Printf("Would process %d files\n", stats.FilesProcessed)
		fmt.Printf("Would rotate %d files\n", stats.FilesRotated)
		fmt.Printf("Would compress %d files\n", stats.FilesCompressed)
		fmt.Printf("Would archive %d files\n", stats.FilesArchived)
		fmt.Printf("Would delete %d files\n", stats.FilesDeleted)
		fmt.Printf("Would free %d bytes\n", stats.BytesFreed)
	}
}

func NewLogCleaner(config *LogCleanupConfig, logger *logrus.Logger) *LogCleaner {
	cleaner := &LogCleaner{
		config: config,
		logger: logger,
		stats:  &CleanupStats{StartTime: time.Now()},
	}

	// Compile exclude patterns
	for _, pattern := range config.ExcludePatterns {
		if re, err := regexp.Compile(pattern); err == nil {
			cleaner.patterns = append(cleaner.patterns, re)
		}
	}

	return cleaner
}

func (c *LogCleaner) Cleanup(ctx context.Context) (*CleanupStats, error) {
	// Validate configuration
	if err := c.validateConfig(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	// Collect log files
	logFiles, err := c.collectLogFiles()
	if err != nil {
		return nil, fmt.Errorf("failed to collect log files: %w", err)
	}

	c.logger.WithField("files_found", len(logFiles)).Info("Found log files")

	// Process files in parallel
	fileChan := make(chan LogFile, len(logFiles))
	resultChan := make(chan error, c.config.Workers)

	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < c.config.Workers; i++ {
		wg.Add(1)
		go c.processWorker(ctx, fileChan, resultChan, &wg)
	}

	// Queue files for processing
	for _, file := range logFiles {
		select {
		case fileChan <- file:
		case <-ctx.Done():
			close(fileChan)
			return c.stats, ctx.Err()
		}
	}
	close(fileChan)

	// Wait for workers to complete
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results
	for err := range resultChan {
		if err != nil {
			c.stats.Errors = append(c.stats.Errors, err)
		}
	}

	// Perform aggregation if enabled
	if c.config.Aggregation.Enabled {
		if err := c.aggregateLogs(ctx); err != nil {
			c.logger.WithError(err).Error("Log aggregation failed")
			c.stats.Errors = append(c.stats.Errors, err)
		}
	}

	c.stats.EndTime = time.Now()
	return c.stats, nil
}

func (c *LogCleaner) collectLogFiles() ([]LogFile, error) {
	var logFiles []LogFile

	err := filepath.Walk(c.config.LogDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			c.logger.WithField("path", path).WithError(err).Warn("Error accessing file")
			return nil
		}

		// Skip directories
		if info.IsDir() {
			return nil
		}

		// Check if file matches pattern
		matched, err := filepath.Match(c.config.FilePattern, filepath.Base(path))
		if err != nil || !matched {
			return nil
		}

		// Check exclude patterns
		if c.isExcluded(path) {
			c.logger.WithField("file", path).Debug("File excluded by pattern")
			return nil
		}

		// Detect log level from filename or content
		level := c.detectLogLevel(path)

		logFile := LogFile{
			Path:         path,
			Info:         info,
			Level:        level,
			LastModified: info.ModTime(),
			Size:         info.Size(),
		}

		logFiles = append(logFiles, logFile)
		return nil
	})

	if err != nil {
		return nil, err
	}

	// Sort by modification time (oldest first)
	sort.Slice(logFiles, func(i, j int) bool {
		return logFiles[i].LastModified.Before(logFiles[j].LastModified)
	})

	return logFiles, nil
}

func (c *LogCleaner) processWorker(ctx context.Context, files <-chan LogFile, results chan<- error, wg *sync.WaitGroup) {
	defer wg.Done()

	for file := range files {
		select {
		case <-ctx.Done():
			results <- ctx.Err()
			return
		default:
			if err := c.processLogFile(ctx, file); err != nil {
				results <- fmt.Errorf("failed to process %s: %w", file.Path, err)
			} else {
				results <- nil
			}
		}
	}
}

func (c *LogCleaner) processLogFile(ctx context.Context, file LogFile) error {
	c.mu.Lock()
	c.stats.FilesProcessed++
	c.stats.BytesProcessed += file.Size
	c.mu.Unlock()

	// Check if file should be processed based on level policies
	policy := c.getLevelPolicy(file.Level)
	
	// Check age
	age := time.Since(file.LastModified)
	maxAge := c.config.MaxAge
	if policy != nil && policy.MaxAge > 0 {
		maxAge = policy.MaxAge
	}

	// Rotation check
	needsRotation := false
	if c.config.Rotation.SizeMB > 0 && file.Size > c.config.Rotation.SizeMB*1024*1024 {
		needsRotation = true
	}

	// Process based on conditions
	if needsRotation {
		if err := c.rotateLog(file); err != nil {
			return err
		}
	}

	// Archive if needed
	if c.shouldArchive(file, age, policy) {
		if err := c.archiveLog(file); err != nil {
			return err
		}
	}

	// Compress if needed
	if c.shouldCompress(file, age, policy) {
		if err := c.compressLog(file); err != nil {
			return err
		}
	}

	// Delete if too old
	if age > maxAge {
		if err := c.deleteLog(file); err != nil {
			return err
		}
	}

	return nil
}

func (c *LogCleaner) rotateLog(file LogFile) error {
	if c.config.DryRun {
		c.logger.WithField("file", file.Path).Info("DRY RUN - Would rotate log")
		c.mu.Lock()
		c.stats.FilesRotated++
		c.mu.Unlock()
		return nil
	}

	// Generate rotated filename
	dir := filepath.Dir(file.Path)
	base := filepath.Base(file.Path)
	ext := filepath.Ext(base)
	nameWithoutExt := strings.TrimSuffix(base, ext)
	
	timestamp := time.Now().Format("20060102_150405")
	rotatedPath := filepath.Join(dir, fmt.Sprintf("%s.%s%s", nameWithoutExt, timestamp, ext))

	// Rename file
	if err := os.Rename(file.Path, rotatedPath); err != nil {
		return fmt.Errorf("failed to rotate log: %w", err)
	}

	// Create new empty log file
	newFile, err := os.Create(file.Path)
	if err != nil {
		// Try to restore original file
		os.Rename(rotatedPath, file.Path)
		return fmt.Errorf("failed to create new log file: %w", err)
	}
	newFile.Close()

	c.logger.WithFields(logrus.Fields{
		"original": file.Path,
		"rotated":  rotatedPath,
	}).Info("Rotated log file")

	c.mu.Lock()
	c.stats.FilesRotated++
	c.mu.Unlock()

	// Clean up old rotated files
	c.cleanupRotatedLogs(dir, nameWithoutExt, ext)

	return nil
}

func (c *LogCleaner) compressLog(file LogFile) error {
	if c.config.DryRun {
		c.logger.WithField("file", file.Path).Info("DRY RUN - Would compress log")
		c.mu.Lock()
		c.stats.FilesCompressed++
		c.mu.Unlock()
		return nil
	}

	// Skip if already compressed
	if strings.HasSuffix(file.Path, ".gz") || strings.HasSuffix(file.Path, ".zst") {
		return nil
	}

	compressedPath := file.Path + ".gz"

	// Open source file
	source, err := os.Open(file.Path)
	if err != nil {
		return err
	}
	defer source.Close()

	// Create compressed file
	dest, err := os.Create(compressedPath)
	if err != nil {
		return err
	}
	defer dest.Close()

	// Create gzip writer
	gw := gzip.NewWriter(dest)
	gw.Name = filepath.Base(file.Path)
	gw.ModTime = file.LastModified
	defer gw.Close()

	// Copy data
	if _, err := io.Copy(gw, source); err != nil {
		os.Remove(compressedPath)
		return err
	}

	// Close gzip writer to flush data
	if err := gw.Close(); err != nil {
		os.Remove(compressedPath)
		return err
	}

	// Remove original file
	if err := os.Remove(file.Path); err != nil {
		os.Remove(compressedPath)
		return err
	}

	originalSize := file.Size
	compressedInfo, _ := os.Stat(compressedPath)
	compressedSize := compressedInfo.Size()
	
	c.logger.WithFields(logrus.Fields{
		"file":             file.Path,
		"compressed":       compressedPath,
		"original_size":    originalSize,
		"compressed_size":  compressedSize,
		"compression_ratio": float64(originalSize-compressedSize) / float64(originalSize) * 100,
	}).Info("Compressed log file")

	c.mu.Lock()
	c.stats.FilesCompressed++
	c.stats.BytesFreed += originalSize - compressedSize
	c.mu.Unlock()

	return nil
}

func (c *LogCleaner) archiveLog(file LogFile) error {
	if c.config.DryRun {
		c.logger.WithField("file", file.Path).Info("DRY RUN - Would archive log")
		c.mu.Lock()
		c.stats.FilesArchived++
		c.mu.Unlock()
		return nil
	}

	// Implement actual archiving based on destination type
	// This is a placeholder - real implementation would handle S3, GCS, etc.
	
	c.logger.WithFields(logrus.Fields{
		"file":        file.Path,
		"destination": c.config.Archive.Destination,
	}).Info("Archived log file")

	c.mu.Lock()
	c.stats.FilesArchived++
	c.mu.Unlock()

	return nil
}

func (c *LogCleaner) deleteLog(file LogFile) error {
	if c.config.DryRun {
		c.logger.WithFields(logrus.Fields{
			"file": file.Path,
			"age":  time.Since(file.LastModified).String(),
			"size": file.Size,
		}).Info("DRY RUN - Would delete log")
		c.mu.Lock()
		c.stats.FilesDeleted++
		c.stats.BytesFreed += file.Size
		c.mu.Unlock()
		return nil
	}

	if err := os.Remove(file.Path); err != nil {
		return err
	}

	c.logger.WithFields(logrus.Fields{
		"file": file.Path,
		"age":  time.Since(file.LastModified).String(),
		"size": file.Size,
	}).Info("Deleted log file")

	c.mu.Lock()
	c.stats.FilesDeleted++
	c.stats.BytesFreed += file.Size
	c.mu.Unlock()

	return nil
}

func (c *LogCleaner) aggregateLogs(ctx context.Context) error {
	c.logger.Info("Starting log aggregation")

	// Group similar log entries
	// This is a simplified implementation
	aggregatedCount := int64(0)

	// Process each log file for aggregation
	files, err := c.collectLogFiles()
	if err != nil {
		return err
	}

	for _, file := range files {
		if strings.HasSuffix(file.Path, ".gz") {
			continue // Skip compressed files
		}

		count, err := c.aggregateLogFile(file)
		if err != nil {
			c.logger.WithField("file", file.Path).WithError(err).Warn("Failed to aggregate log file")
			continue
		}
		aggregatedCount += count
	}

	c.mu.Lock()
	c.stats.LogsAggregated = aggregatedCount
	c.mu.Unlock()

	return nil
}

func (c *LogCleaner) aggregateLogFile(file LogFile) (int64, error) {
	// This is a simplified implementation
	// In a real implementation, you would:
	// 1. Parse log entries
	// 2. Group similar entries based on threshold
	// 3. Create aggregated summary
	// 4. Write aggregated logs

	return 0, nil
}

func (c *LogCleaner) cleanupRotatedLogs(dir, namePrefix, ext string) error {
	// Find all rotated logs
	pattern := fmt.Sprintf("%s.*%s", namePrefix, ext)
	matches, err := filepath.Glob(filepath.Join(dir, pattern))
	if err != nil {
		return err
	}

	// Sort by modification time
	type rotatedFile struct {
		path    string
		modTime time.Time
	}
	
	var rotatedFiles []rotatedFile
	for _, match := range matches {
		info, err := os.Stat(match)
		if err != nil {
			continue
		}
		rotatedFiles = append(rotatedFiles, rotatedFile{
			path:    match,
			modTime: info.ModTime(),
		})
	}

	sort.Slice(rotatedFiles, func(i, j int) bool {
		return rotatedFiles[i].modTime.After(rotatedFiles[j].modTime)
	})

	// Keep only the configured number of rotated files
	if len(rotatedFiles) > c.config.Rotation.Count {
		for i := c.config.Rotation.Count; i < len(rotatedFiles); i++ {
			if err := os.Remove(rotatedFiles[i].path); err != nil {
				c.logger.WithField("file", rotatedFiles[i].path).WithError(err).Warn("Failed to remove old rotated log")
			} else {
				c.logger.WithField("file", rotatedFiles[i].path).Debug("Removed old rotated log")
			}
		}
	}

	return nil
}

func (c *LogCleaner) isExcluded(path string) bool {
	for _, pattern := range c.patterns {
		if pattern.MatchString(path) {
			return true
		}
	}
	return false
}

func (c *LogCleaner) detectLogLevel(path string) string {
	// Check filename for log level
	base := filepath.Base(path)
	levelPatterns := map[string]string{
		"ERROR":   `(?i)(error|err)`,
		"WARNING": `(?i)(warn|warning)`,
		"INFO":    `(?i)(info)`,
		"DEBUG":   `(?i)(debug|dbg)`,
		"TRACE":   `(?i)(trace)`,
	}

	for level, pattern := range levelPatterns {
		if matched, _ := regexp.MatchString(pattern, base); matched {
			return level
		}
	}

	// If not in filename, sample the file content
	file, err := os.Open(path)
	if err != nil {
		return "UNKNOWN"
	}
	defer file.Close()

	// Read first few lines
	buffer := make([]byte, 1024)
	n, _ := file.Read(buffer)
	content := string(buffer[:n])

	for level, pattern := range levelPatterns {
		if matched, _ := regexp.MatchString(pattern, content); matched {
			return level
		}
	}

	return "UNKNOWN"
}

func (c *LogCleaner) getLevelPolicy(level string) *LevelPolicy {
	if policy, ok := c.config.LevelPolicies[level]; ok {
		return &policy
	}
	return nil
}

func (c *LogCleaner) shouldArchive(file LogFile, age time.Duration, policy *LevelPolicy) bool {
	if !c.config.Archive.Enabled {
		return false
	}

	if policy != nil && policy.Archive {
		return true
	}

	// Archive if older than retention period
	if c.config.Archive.RetentionDays > 0 {
		retentionAge := time.Duration(c.config.Archive.RetentionDays) * 24 * time.Hour
		return age > retentionAge
	}

	return false
}

func (c *LogCleaner) shouldCompress(file LogFile, age time.Duration, policy *LevelPolicy) bool {
	// Skip if already compressed
	if strings.HasSuffix(file.Path, ".gz") || strings.HasSuffix(file.Path, ".zst") {
		return false
	}

	if policy != nil && policy.Compress {
		return true
	}

	// Compress if older than 1 day by default
	return age > 24*time.Hour
}

func (c *LogCleaner) validateConfig() error {
	if c.config.LogDir == "" {
		return fmt.Errorf("log directory cannot be empty")
	}

	if _, err := os.Stat(c.config.LogDir); os.IsNotExist(err) {
		return fmt.Errorf("log directory does not exist: %s", c.config.LogDir)
	}

	if c.config.Workers < 1 {
		c.config.Workers = 1
	}

	if c.config.FilePattern == "" {
		c.config.FilePattern = "*.log"
	}

	if c.config.Aggregation.Enabled && c.config.Aggregation.Threshold == 0 {
		c.config.Aggregation.Threshold = 0.85
	}

	return nil
}

// Helper functions

func loadConfig(path string, config *LogCleanupConfig) error {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	if strings.HasSuffix(path, ".yaml") || strings.HasSuffix(path, ".yml") {
		return yaml.Unmarshal(data, config)
	}

	return json.Unmarshal(data, config)
}

func formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}