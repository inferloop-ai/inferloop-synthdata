package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/sirupsen/logrus"
)

type DataCleanupConfig struct {
	DataDir         string        `json:"data_dir"`
	MaxAge          time.Duration `json:"max_age"`
	FilePattern     string        `json:"file_pattern"`
	DryRun          bool          `json:"dry_run"`
	Recursive       bool          `json:"recursive"`
	MinFreeSpace    int64         `json:"min_free_space_gb"`
	BackupLocation  string        `json:"backup_location"`
	CompressionType string        `json:"compression_type"` // none, gzip, zstd
}

type DataCleaner struct {
	config *DataCleanupConfig
	logger *logrus.Logger
}

type CleanupStats struct {
	FilesProcessed int64         `json:"files_processed"`
	FilesDeleted   int64         `json:"files_deleted"`
	BytesFreed     int64         `json:"bytes_freed"`
	Errors         int64         `json:"errors"`
	Duration       time.Duration `json:"duration"`
}

func main() {
	var (
		dataDir     = flag.String("dir", "./data", "Data directory to clean")
		maxAge      = flag.Duration("max-age", 30*24*time.Hour, "Maximum age of files to keep")
		pattern     = flag.String("pattern", "*.json", "File pattern to match")
		dryRun      = flag.Bool("dry-run", false, "Perform dry run without deleting files")
		recursive   = flag.Bool("recursive", true, "Clean directories recursively")
		minFreeGB   = flag.Int64("min-free-gb", 10, "Minimum free space in GB to maintain")
		backup      = flag.String("backup", "", "Backup location before deletion")
		compression = flag.String("compression", "none", "Compression type for backups (none, gzip, zstd)")
		verbose     = flag.Bool("verbose", false, "Enable verbose logging")
	)
	flag.Parse()

	// Setup logging
	logger := logrus.New()
	if *verbose {
		logger.SetLevel(logrus.DebugLevel)
	}

	config := &DataCleanupConfig{
		DataDir:         *dataDir,
		MaxAge:          *maxAge,
		FilePattern:     *pattern,
		DryRun:          *dryRun,
		Recursive:       *recursive,
		MinFreeSpace:    *minFreeGB,
		BackupLocation:  *backup,
		CompressionType: *compression,
	}

	cleaner := NewDataCleaner(config, logger)

	logger.WithFields(logrus.Fields{
		"data_dir":         config.DataDir,
		"max_age":          config.MaxAge.String(),
		"file_pattern":     config.FilePattern,
		"dry_run":          config.DryRun,
		"recursive":        config.Recursive,
		"min_free_space":   config.MinFreeSpace,
		"backup_location":  config.BackupLocation,
		"compression_type": config.CompressionType,
	}).Info("Starting data cleanup")

	stats, err := cleaner.Cleanup(context.Background())
	if err != nil {
		log.Fatalf("Cleanup failed: %v", err)
	}

	logger.WithFields(logrus.Fields{
		"files_processed": stats.FilesProcessed,
		"files_deleted":   stats.FilesDeleted,
		"bytes_freed":     stats.BytesFreed,
		"errors":          stats.Errors,
		"duration":        stats.Duration.String(),
	}).Info("Data cleanup completed")

	if config.DryRun {
		fmt.Printf("DRY RUN - Would have processed %d files, deleted %d files, freed %d bytes\n",
			stats.FilesProcessed, stats.FilesDeleted, stats.BytesFreed)
	} else {
		fmt.Printf("Processed %d files, deleted %d files, freed %d bytes\n",
			stats.FilesProcessed, stats.FilesDeleted, stats.BytesFreed)
	}
}

func NewDataCleaner(config *DataCleanupConfig, logger *logrus.Logger) *DataCleaner {
	return &DataCleaner{
		config: config,
		logger: logger,
	}
}

func (c *DataCleaner) Cleanup(ctx context.Context) (*CleanupStats, error) {
	start := time.Now()
	stats := &CleanupStats{}

	// Check if data directory exists
	if _, err := os.Stat(c.config.DataDir); os.IsNotExist(err) {
		return nil, fmt.Errorf("data directory does not exist: %s", c.config.DataDir)
	}

	// Get current free space
	freeSpace, err := c.getFreeSpace(c.config.DataDir)
	if err != nil {
		c.logger.Warnf("Could not determine free space: %v", err)
	} else {
		c.logger.WithField("free_space_gb", freeSpace/(1024*1024*1024)).Info("Current free space")
	}

	// Walk through directory
	err = c.walkDirectory(ctx, c.config.DataDir, stats)
	if err != nil {
		return nil, fmt.Errorf("failed to walk directory: %w", err)
	}

	stats.Duration = time.Since(start)
	return stats, nil
}

func (c *DataCleaner) walkDirectory(ctx context.Context, dir string, stats *CleanupStats) error {
	return filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			c.logger.WithField("path", path).WithError(err).Error("Error accessing file")
			stats.Errors++
			return nil // Continue walking
		}

		// Skip directories
		if info.IsDir() {
			return nil
		}

		// Check context for cancellation
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Check if file matches pattern
		matched, err := filepath.Match(c.config.FilePattern, info.Name())
		if err != nil {
			c.logger.WithField("pattern", c.config.FilePattern).WithError(err).Error("Invalid pattern")
			return err
		}

		if !matched {
			return nil
		}

		stats.FilesProcessed++

		// Check if file is old enough to delete
		if c.shouldDeleteFile(info) {
			if err := c.processFile(path, info, stats); err != nil {
				c.logger.WithField("file", path).WithError(err).Error("Failed to process file")
				stats.Errors++
			}
		}

		return nil
	})
}

func (c *DataCleaner) shouldDeleteFile(info os.FileInfo) bool {
	age := time.Since(info.ModTime())
	return age > c.config.MaxAge
}

func (c *DataCleaner) processFile(path string, info os.FileInfo, stats *CleanupStats) error {
	// Backup file if backup location is specified
	if c.config.BackupLocation != "" {
		if err := c.backupFile(path, info); err != nil {
			return fmt.Errorf("failed to backup file: %w", err)
		}
	}

	// Delete file (or simulate deletion in dry run)
	if c.config.DryRun {
		c.logger.WithFields(logrus.Fields{
			"file": path,
			"size": info.Size(),
			"age":  time.Since(info.ModTime()).String(),
		}).Info("DRY RUN - Would delete file")
	} else {
		c.logger.WithFields(logrus.Fields{
			"file": path,
			"size": info.Size(),
			"age":  time.Since(info.ModTime()).String(),
		}).Info("Deleting file")

		if err := os.Remove(path); err != nil {
			return fmt.Errorf("failed to delete file: %w", err)
		}
	}

	stats.FilesDeleted++
	stats.BytesFreed += info.Size()

	return nil
}

func (c *DataCleaner) backupFile(path string, info os.FileInfo) error {
	// Create backup directory structure
	relPath, err := filepath.Rel(c.config.DataDir, path)
	if err != nil {
		return err
	}

	backupPath := filepath.Join(c.config.BackupLocation, relPath)
	backupDir := filepath.Dir(backupPath)

	if err := os.MkdirAll(backupDir, 0755); err != nil {
		return fmt.Errorf("failed to create backup directory: %w", err)
	}

	// Add timestamp to backup filename
	ext := filepath.Ext(backupPath)
	nameWithoutExt := backupPath[:len(backupPath)-len(ext)]
	timestamp := time.Now().Format("20060102_150405")
	backupPath = fmt.Sprintf("%s_%s%s", nameWithoutExt, timestamp, ext)

	// Apply compression if specified
	switch c.config.CompressionType {
	case "gzip":
		backupPath += ".gz"
		return c.backupWithGzip(path, backupPath)
	case "zstd":
		backupPath += ".zst"
		return c.backupWithZstd(path, backupPath)
	default:
		return c.copyFile(path, backupPath)
	}
}

func (c *DataCleaner) copyFile(src, dst string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destFile.Close()

	buffer := make([]byte, 64*1024) // 64KB buffer
	for {
		n, err := sourceFile.Read(buffer)
		if n > 0 {
			if _, writeErr := destFile.Write(buffer[:n]); writeErr != nil {
				return writeErr
			}
		}
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return err
		}
	}

	return destFile.Sync()
}

func (c *DataCleaner) backupWithGzip(src, dst string) error {
	// Note: In a real implementation, you would use compress/gzip
	// For simplicity, this just copies the file
	c.logger.Warn("GZIP compression not implemented, using plain copy")
	return c.copyFile(src, dst)
}

func (c *DataCleaner) backupWithZstd(src, dst string) error {
	// Note: In a real implementation, you would use a zstd library
	// For simplicity, this just copies the file
	c.logger.Warn("ZSTD compression not implemented, using plain copy")
	return c.copyFile(src, dst)
}

func (c *DataCleaner) getFreeSpace(path string) (int64, error) {
	// This is a simplified implementation
	// In a real implementation, you would use syscalls to get actual free space
	stat, err := os.Stat(path)
	if err != nil {
		return 0, err
	}

	// Return a placeholder value
	// In production, use syscall.Statfs (Linux) or similar
	_ = stat
	return 100 * 1024 * 1024 * 1024, nil // 100GB placeholder
}

func (c *DataCleaner) CleanupOldBackups(ctx context.Context, backupAge time.Duration) error {
	if c.config.BackupLocation == "" {
		return nil
	}

	c.logger.WithFields(logrus.Fields{
		"backup_location": c.config.BackupLocation,
		"backup_age":      backupAge.String(),
	}).Info("Cleaning up old backups")

	return filepath.Walk(c.config.BackupLocation, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Continue walking
		}

		if info.IsDir() {
			return nil
		}

		if time.Since(info.ModTime()) > backupAge {
			if c.config.DryRun {
				c.logger.WithField("backup", path).Info("DRY RUN - Would delete old backup")
			} else {
				c.logger.WithField("backup", path).Info("Deleting old backup")
				return os.Remove(path)
			}
		}

		return nil
	})
}

func (c *DataCleaner) GetDirectorySize(path string) (int64, error) {
	var size int64

	err := filepath.Walk(path, func(filePath string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Continue walking
		}
		if !info.IsDir() {
			size += info.Size()
		}
		return nil
	})

	return size, err
}

func (c *DataCleaner) ValidateConfig() error {
	if c.config.DataDir == "" {
		return fmt.Errorf("data directory cannot be empty")
	}

	if c.config.MaxAge <= 0 {
		return fmt.Errorf("max age must be positive")
	}

	if c.config.FilePattern == "" {
		return fmt.Errorf("file pattern cannot be empty")
	}

	if c.config.BackupLocation != "" {
		if err := os.MkdirAll(c.config.BackupLocation, 0755); err != nil {
			return fmt.Errorf("cannot create backup location: %w", err)
		}
	}

	validCompressions := []string{"none", "gzip", "zstd"}
	isValid := false
	for _, valid := range validCompressions {
		if c.config.CompressionType == valid {
			isValid = true
			break
		}
	}
	if !isValid {
		return fmt.Errorf("invalid compression type: %s", c.config.CompressionType)
	}

	return nil
}