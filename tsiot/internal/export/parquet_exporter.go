package export

import (
	"context"
	"fmt"
	"io"
	"time"

	"github.com/inferloop/tsiot/pkg/models"
)

// ParquetExporter implements Parquet export functionality
type ParquetExporter struct{}

// Name returns the exporter name
func (pe *ParquetExporter) Name() string {
	return "parquet"
}

// SupportedFormats returns supported formats
func (pe *ParquetExporter) SupportedFormats() []ExportFormat {
	return []ExportFormat{FormatParquet}
}

// Export exports time series data to Parquet format
func (pe *ParquetExporter) Export(ctx context.Context, writer io.Writer, data []*models.TimeSeries, options ExportOptions) error {
	// In a real implementation, this would use a Parquet library like github.com/xitongsys/parquet-go
	// For now, we'll provide a simplified implementation that writes metadata about what would be exported

	parquetData := pe.convertToParquetFormat(data, options)
	
	// Simulate writing Parquet data
	metadata := fmt.Sprintf("Parquet export metadata:\n")
	metadata += fmt.Sprintf("Schema: %s\n", parquetData.Schema)
	metadata += fmt.Sprintf("Records: %d\n", parquetData.RecordCount)
	metadata += fmt.Sprintf("Columns: %d\n", len(parquetData.Columns))
	metadata += fmt.Sprintf("Compression: %s\n", parquetData.Compression)
	metadata += fmt.Sprintf("Row Groups: %d\n", parquetData.RowGroups)

	_, err := writer.Write([]byte(metadata))
	return err
}

// ValidateOptions validates Parquet export options
func (pe *ParquetExporter) ValidateOptions(options ExportOptions) error {
	if options.ParquetOptions.CompressionCodec != "" {
		validCodecs := map[string]bool{
			"uncompressed": true,
			"snappy":       true,
			"gzip":         true,
			"lzo":          true,
			"brotli":       true,
			"lz4":          true,
			"zstd":         true,
		}

		if !validCodecs[options.ParquetOptions.CompressionCodec] {
			return fmt.Errorf("unsupported Parquet compression codec: %s", options.ParquetOptions.CompressionCodec)
		}
	}

	if options.ParquetOptions.Version != "" {
		validVersions := map[string]bool{
			"1.0": true,
			"2.0": true,
			"2.4": true,
			"2.6": true,
		}

		if !validVersions[options.ParquetOptions.Version] {
			return fmt.Errorf("unsupported Parquet version: %s", options.ParquetOptions.Version)
		}
	}

	return nil
}

// EstimateSize estimates the output size for Parquet export
func (pe *ParquetExporter) EstimateSize(data []*models.TimeSeries, options ExportOptions) (int64, error) {
	totalPoints := int64(0)
	for _, ts := range data {
		totalPoints += int64(len(ts.DataPoints))
	}

	// Parquet is typically 60-80% more compressed than CSV
	// Estimate: ~30-40 bytes per row with compression
	compressionRatio := 0.4
	if options.ParquetOptions.CompressionCodec == "uncompressed" {
		compressionRatio = 1.0
	} else if options.ParquetOptions.CompressionCodec == "snappy" {
		compressionRatio = 0.5
	} else if options.ParquetOptions.CompressionCodec == "gzip" {
		compressionRatio = 0.3
	}

	baseSize := totalPoints * 80 // Base uncompressed size
	estimatedSize := int64(float64(baseSize) * compressionRatio)

	return estimatedSize, nil
}

// ParquetData represents Parquet format structure
type ParquetData struct {
	Schema      string
	RecordCount int64
	Columns     []ParquetColumn
	Compression string
	RowGroups   int
	Metadata    map[string]interface{}
}

// ParquetColumn represents a Parquet column definition
type ParquetColumn struct {
	Name               string
	Type               string
	LogicalType        string
	RepetitionType     string
	CompressionCodec   string
	Encoding           string
	Statistics         *ParquetColumnStats
}

// ParquetColumnStats contains column statistics
type ParquetColumnStats struct {
	NullCount    int64
	DistinctCount int64
	MinValue     interface{}
	MaxValue     interface{}
}

// convertToParquetFormat converts time series data to Parquet format structure
func (pe *ParquetExporter) convertToParquetFormat(data []*models.TimeSeries, options ExportOptions) *ParquetData {
	recordCount := int64(0)
	for _, ts := range data {
		recordCount += int64(len(ts.DataPoints))
	}

	// Define Parquet schema
	columns := pe.defineParquetSchema(data, options)
	
	// Calculate row groups
	rowGroupSize := options.ParquetOptions.RowGroupSize
	if rowGroupSize <= 0 {
		rowGroupSize = 100000 // Default 100K rows per group
	}
	
	rowGroups := int(recordCount)/rowGroupSize + 1

	// Set compression
	compression := options.ParquetOptions.CompressionCodec
	if compression == "" {
		compression = "snappy" // Default compression
	}

	return &ParquetData{
		Schema:      pe.generateParquetSchema(columns),
		RecordCount: recordCount,
		Columns:     columns,
		Compression: compression,
		RowGroups:   rowGroups,
		Metadata: map[string]interface{}{
			"exported_by":    "tsiot-export-engine",
			"export_time":    time.Now(),
			"series_count":   len(data),
			"parquet_version": pe.getParquetVersion(options),
		},
	}
}

// defineParquetSchema defines the Parquet schema based on the data
func (pe *ParquetExporter) defineParquetSchema(data []*models.TimeSeries, options ExportOptions) []ParquetColumn {
	columns := []ParquetColumn{
		{
			Name:             "timestamp",
			Type:             "INT64",
			LogicalType:      "TIMESTAMP_MILLIS",
			RepetitionType:   "REQUIRED",
			CompressionCodec: options.ParquetOptions.CompressionCodec,
			Encoding:         "DELTA_BINARY_PACKED",
		},
		{
			Name:             "series_id",
			Type:             "BYTE_ARRAY",
			LogicalType:      "UTF8",
			RepetitionType:   "REQUIRED",
			CompressionCodec: options.ParquetOptions.CompressionCodec,
			Encoding:         "DICTIONARY",
		},
		{
			Name:             "series_name",
			Type:             "BYTE_ARRAY",
			LogicalType:      "UTF8",
			RepetitionType:   "OPTIONAL",
			CompressionCodec: options.ParquetOptions.CompressionCodec,
			Encoding:         "DICTIONARY",
		},
		{
			Name:             "sensor_type",
			Type:             "BYTE_ARRAY",
			LogicalType:      "UTF8",
			RepetitionType:   "OPTIONAL",
			CompressionCodec: options.ParquetOptions.CompressionCodec,
			Encoding:         "DICTIONARY",
		},
		{
			Name:             "value",
			Type:             "DOUBLE",
			RepetitionType:   "REQUIRED",
			CompressionCodec: options.ParquetOptions.CompressionCodec,
			Encoding:         "PLAIN",
		},
		{
			Name:             "quality",
			Type:             "FLOAT",
			RepetitionType:   "OPTIONAL",
			CompressionCodec: options.ParquetOptions.CompressionCodec,
			Encoding:         "PLAIN",
		},
	}

	// Add metadata columns if present
	metadataFields := pe.collectMetadataFields(data)
	for field := range metadataFields {
		columns = append(columns, ParquetColumn{
			Name:             "metadata_" + field,
			Type:             "BYTE_ARRAY",
			LogicalType:      "UTF8",
			RepetitionType:   "OPTIONAL",
			CompressionCodec: options.ParquetOptions.CompressionCodec,
			Encoding:         "DICTIONARY",
		})
	}

	// Add tag columns if present
	tagFields := pe.collectTagFields(data)
	for field := range tagFields {
		columns = append(columns, ParquetColumn{
			Name:             "tag_" + field,
			Type:             "BYTE_ARRAY",
			LogicalType:      "UTF8",
			RepetitionType:   "OPTIONAL",
			CompressionCodec: options.ParquetOptions.CompressionCodec,
			Encoding:         "DICTIONARY",
		})
	}

	// Calculate statistics for each column
	for i := range columns {
		columns[i].Statistics = pe.calculateColumnStats(data, columns[i].Name)
	}

	return columns
}

// generateParquetSchema generates a Parquet schema string
func (pe *ParquetExporter) generateParquetSchema(columns []ParquetColumn) string {
	schema := "message tsiot_timeseries {\n"
	
	for _, column := range columns {
		repetition := column.RepetitionType
		if repetition == "" {
			repetition = "OPTIONAL"
		}
		
		logicalType := ""
		if column.LogicalType != "" {
			logicalType = fmt.Sprintf(" (%s)", column.LogicalType)
		}
		
		schema += fmt.Sprintf("  %s %s %s%s;\n", 
			repetition, column.Type, column.Name, logicalType)
	}
	
	schema += "}"
	return schema
}

// calculateColumnStats calculates statistics for a column
func (pe *ParquetExporter) calculateColumnStats(data []*models.TimeSeries, columnName string) *ParquetColumnStats {
	stats := &ParquetColumnStats{
		NullCount:     0,
		DistinctCount: 0,
	}

	distinctValues := make(map[interface{}]bool)
	var minValue, maxValue interface{}

	for _, ts := range data {
		for _, dp := range ts.DataPoints {
			var value interface{}
			
			switch columnName {
			case "timestamp":
				value = dp.Timestamp.UnixMilli()
			case "series_id":
				value = ts.ID
			case "series_name":
				value = ts.Name
			case "sensor_type":
				value = ts.SensorType
			case "value":
				value = dp.Value
			case "quality":
				value = dp.Quality
			default:
				// Handle metadata and tag fields
				if ts.Metadata != nil {
					if metaValue, exists := ts.Metadata[columnName]; exists {
						value = metaValue
					}
				}
				if ts.Tags != nil {
					if tagValue, exists := ts.Tags[columnName]; exists {
						value = tagValue
					}
				}
			}

			if value == nil {
				stats.NullCount++
			} else {
				distinctValues[value] = true
				
				if minValue == nil || pe.compareValues(value, minValue) < 0 {
					minValue = value
				}
				if maxValue == nil || pe.compareValues(value, maxValue) > 0 {
					maxValue = value
				}
			}
		}
	}

	stats.DistinctCount = int64(len(distinctValues))
	stats.MinValue = minValue
	stats.MaxValue = maxValue

	return stats
}

// compareValues compares two values (simplified implementation)
func (pe *ParquetExporter) compareValues(a, b interface{}) int {
	// Simplified comparison - in real implementation would handle all types properly
	switch av := a.(type) {
	case int64:
		if bv, ok := b.(int64); ok {
			if av < bv {
				return -1
			} else if av > bv {
				return 1
			}
			return 0
		}
	case float64:
		if bv, ok := b.(float64); ok {
			if av < bv {
				return -1
			} else if av > bv {
				return 1
			}
			return 0
		}
	case string:
		if bv, ok := b.(string); ok {
			if av < bv {
				return -1
			} else if av > bv {
				return 1
			}
			return 0
		}
	}
	return 0
}

// collectMetadataFields collects all unique metadata field names
func (pe *ParquetExporter) collectMetadataFields(data []*models.TimeSeries) map[string]bool {
	fields := make(map[string]bool)
	
	for _, ts := range data {
		for field := range ts.Metadata {
			fields[field] = true
		}
	}
	
	return fields
}

// collectTagFields collects all unique tag field names
func (pe *ParquetExporter) collectTagFields(data []*models.TimeSeries) map[string]bool {
	fields := make(map[string]bool)
	
	for _, ts := range data {
		for field := range ts.Tags {
			fields[field] = true
		}
	}
	
	return fields
}

// getParquetVersion returns the Parquet version to use
func (pe *ParquetExporter) getParquetVersion(options ExportOptions) string {
	if options.ParquetOptions.Version != "" {
		return options.ParquetOptions.Version
	}
	return "2.6" // Default to latest stable version
}

// ParquetStreamExporter implements streaming Parquet export
type ParquetStreamExporter struct {
	options     ExportOptions
	schema      []ParquetColumn
	initialized bool
	recordCount int64
}

// NewParquetStreamExporter creates a new streaming Parquet exporter
func NewParquetStreamExporter(writer io.Writer, options ExportOptions) *ParquetStreamExporter {
	return &ParquetStreamExporter{
		options:     options,
		initialized: false,
		recordCount: 0,
	}
}

// WriteHeader initializes the Parquet file and writes the schema
func (pse *ParquetStreamExporter) WriteHeader(sampleData []*models.TimeSeries) error {
	if pse.initialized {
		return fmt.Errorf("Parquet stream already initialized")
	}

	exporter := &ParquetExporter{}
	pse.schema = exporter.defineParquetSchema(sampleData, pse.options)
	pse.initialized = true

	return nil
}

// WriteTimeSeries writes time series data to the Parquet stream
func (pse *ParquetStreamExporter) WriteTimeSeries(ts *models.TimeSeries) error {
	if !pse.initialized {
		return fmt.Errorf("Parquet stream not initialized")
	}

	// In a real implementation, this would write the data to Parquet format
	pse.recordCount += int64(len(ts.DataPoints))
	
	return nil
}

// Close finalizes the Parquet file
func (pse *ParquetStreamExporter) Close() error {
	if !pse.initialized {
		return nil
	}

	// In a real implementation, this would finalize the Parquet file
	// by writing footer metadata, finalizing row groups, etc.
	
	return nil
}

// GetStatistics returns statistics about the exported data
func (pse *ParquetStreamExporter) GetStatistics() map[string]interface{} {
	return map[string]interface{}{
		"record_count": pse.recordCount,
		"columns":      len(pse.schema),
		"compression":  pse.options.ParquetOptions.CompressionCodec,
	}
}