package export

import (
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"strconv"
	"time"

	"github.com/inferloop/tsiot/pkg/models"
)

// CSVExporter implements CSV export functionality
type CSVExporter struct{}

// Name returns the exporter name
func (ce *CSVExporter) Name() string {
	return "csv"
}

// SupportedFormats returns supported formats
func (ce *CSVExporter) SupportedFormats() []ExportFormat {
	return []ExportFormat{FormatCSV}
}

// Export exports time series data to CSV format
func (ce *CSVExporter) Export(ctx context.Context, writer io.Writer, data []*models.TimeSeries, options ExportOptions) error {
	csvOptions := options.CSVOptions
	if csvOptions.Delimiter == "" {
		csvOptions.Delimiter = ","
	}
	if csvOptions.Quote == "" {
		csvOptions.Quote = "\""
	}
	if csvOptions.LineEnding == "" {
		csvOptions.LineEnding = "\n"
	}
	if csvOptions.NullValue == "" {
		csvOptions.NullValue = ""
	}

	csvWriter := csv.NewWriter(writer)
	csvWriter.Comma = rune(csvOptions.Delimiter[0])

	defer csvWriter.Flush()

	// Write headers if enabled
	if options.IncludeHeaders {
		headers := ce.generateHeaders(data, options)
		if err := csvWriter.Write(headers); err != nil {
			return fmt.Errorf("failed to write CSV headers: %w", err)
		}
	}

	// Write data rows
	for _, ts := range data {
		for _, dp := range ts.DataPoints {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}

			row := ce.generateRow(ts, dp, options)
			if err := csvWriter.Write(row); err != nil {
				return fmt.Errorf("failed to write CSV row: %w", err)
			}
		}
	}

	return nil
}

// ValidateOptions validates CSV export options
func (ce *CSVExporter) ValidateOptions(options ExportOptions) error {
	if options.CSVOptions.Delimiter != "" && len(options.CSVOptions.Delimiter) != 1 {
		return fmt.Errorf("CSV delimiter must be a single character")
	}
	
	if options.CSVOptions.Quote != "" && len(options.CSVOptions.Quote) != 1 {
		return fmt.Errorf("CSV quote must be a single character")
	}

	return nil
}

// EstimateSize estimates the output size for CSV export
func (ce *CSVExporter) EstimateSize(data []*models.TimeSeries, options ExportOptions) (int64, error) {
	totalPoints := int64(0)
	for _, ts := range data {
		totalPoints += int64(len(ts.DataPoints))
	}

	// Estimate: ~80 bytes per row (timestamp + value + metadata)
	estimatedSize := totalPoints * 80

	// Add header size if enabled
	if options.IncludeHeaders {
		estimatedSize += 200
	}

	return estimatedSize, nil
}

// generateHeaders generates CSV column headers
func (ce *CSVExporter) generateHeaders(data []*models.TimeSeries, options ExportOptions) []string {
	headers := []string{
		"timestamp",
		"series_id",
		"series_name",
		"sensor_type",
		"value",
		"quality",
	}

	// Add custom fields
	for fieldName := range options.CustomFields {
		headers = append(headers, fieldName)
	}

	// Add metadata fields if any series has metadata
	metadataFields := ce.collectMetadataFields(data)
	for field := range metadataFields {
		headers = append(headers, "metadata_"+field)
	}

	// Add tag fields if any series has tags
	tagFields := ce.collectTagFields(data)
	for field := range tagFields {
		headers = append(headers, "tag_"+field)
	}

	return headers
}

// generateRow generates a CSV row for a data point
func (ce *CSVExporter) generateRow(ts *models.TimeSeries, dp models.DataPoint, options ExportOptions) []string {
	// Format timestamp
	timeFormat := options.DateFormat
	if timeFormat == "" {
		timeFormat = time.RFC3339
	}

	timestamp := dp.Timestamp.Format(timeFormat)
	if options.TimeZone != "" {
		if loc, err := time.LoadLocation(options.TimeZone); err == nil {
			timestamp = dp.Timestamp.In(loc).Format(timeFormat)
		}
	}

	// Format value with precision
	valueStr := ce.formatValue(dp.Value, options.Precision)

	// Basic row
	row := []string{
		timestamp,
		ts.ID,
		ts.Name,
		ts.SensorType,
		valueStr,
		ce.formatValue(dp.Quality, options.Precision),
	}

	// Add custom fields
	for fieldName, fieldValue := range options.CustomFields {
		row = append(row, fieldValue)
	}

	// Add metadata fields
	metadataFields := ce.collectMetadataFields([]*models.TimeSeries{ts})
	for field := range metadataFields {
		if value, exists := ts.Metadata[field]; exists {
			row = append(row, fmt.Sprintf("%v", value))
		} else {
			row = append(row, options.CSVOptions.NullValue)
		}
	}

	// Add tag fields
	tagFields := ce.collectTagFields([]*models.TimeSeries{ts})
	for field := range tagFields {
		if value, exists := ts.Tags[field]; exists {
			row = append(row, value)
		} else {
			row = append(row, options.CSVOptions.NullValue)
		}
	}

	return row
}

// formatValue formats a numeric value with specified precision
func (ce *CSVExporter) formatValue(value float64, precision int) string {
	if precision <= 0 {
		precision = 6 // Default precision
	}
	
	format := fmt.Sprintf("%%.%df", precision)
	return fmt.Sprintf(format, value)
}

// collectMetadataFields collects all unique metadata field names
func (ce *CSVExporter) collectMetadataFields(data []*models.TimeSeries) map[string]bool {
	fields := make(map[string]bool)
	
	for _, ts := range data {
		for field := range ts.Metadata {
			fields[field] = true
		}
	}
	
	return fields
}

// collectTagFields collects all unique tag field names
func (ce *CSVExporter) collectTagFields(data []*models.TimeSeries) map[string]bool {
	fields := make(map[string]bool)
	
	for _, ts := range data {
		for field := range ts.Tags {
			fields[field] = true
		}
	}
	
	return fields
}

// CSVStreamExporter implements streaming CSV export for large datasets
type CSVStreamExporter struct {
	writer    *csv.Writer
	options   ExportOptions
	headerRow []string
	written   bool
}

// NewCSVStreamExporter creates a new streaming CSV exporter
func NewCSVStreamExporter(writer io.Writer, options ExportOptions) *CSVStreamExporter {
	csvWriter := csv.NewWriter(writer)
	
	delimiter := ","
	if options.CSVOptions.Delimiter != "" {
		delimiter = options.CSVOptions.Delimiter
	}
	csvWriter.Comma = rune(delimiter[0])

	return &CSVStreamExporter{
		writer:  csvWriter,
		options: options,
		written: false,
	}
}

// WriteHeader writes the CSV header row
func (cse *CSVStreamExporter) WriteHeader(sampleData []*models.TimeSeries) error {
	if cse.written {
		return fmt.Errorf("header already written")
	}

	if cse.options.IncludeHeaders {
		exporter := &CSVExporter{}
		headers := exporter.generateHeaders(sampleData, cse.options)
		cse.headerRow = headers
		
		if err := cse.writer.Write(headers); err != nil {
			return fmt.Errorf("failed to write CSV headers: %w", err)
		}
	}

	cse.written = true
	return nil
}

// WriteTimeSeries writes time series data to the CSV stream
func (cse *CSVStreamExporter) WriteTimeSeries(ts *models.TimeSeries) error {
	if !cse.written {
		return fmt.Errorf("header not written yet")
	}

	exporter := &CSVExporter{}
	
	for _, dp := range ts.DataPoints {
		row := exporter.generateRow(ts, dp, cse.options)
		if err := cse.writer.Write(row); err != nil {
			return fmt.Errorf("failed to write CSV row: %w", err)
		}
	}

	return nil
}

// WriteDataPoint writes a single data point to the CSV stream
func (cse *CSVStreamExporter) WriteDataPoint(ts *models.TimeSeries, dp models.DataPoint) error {
	if !cse.written {
		return fmt.Errorf("header not written yet")
	}

	exporter := &CSVExporter{}
	row := exporter.generateRow(ts, dp, cse.options)
	
	if err := cse.writer.Write(row); err != nil {
		return fmt.Errorf("failed to write CSV row: %w", err)
	}

	return nil
}

// Flush flushes any buffered data to the underlying writer
func (cse *CSVStreamExporter) Flush() error {
	cse.writer.Flush()
	return cse.writer.Error()
}

// Close finalizes the CSV export
func (cse *CSVStreamExporter) Close() error {
	return cse.Flush()
}

// CSVPivotExporter exports data in pivot table format
type CSVPivotExporter struct{}

// Name returns the exporter name
func (cpe *CSVPivotExporter) Name() string {
	return "csv_pivot"
}

// SupportedFormats returns supported formats
func (cpe *CSVPivotExporter) SupportedFormats() []ExportFormat {
	return []ExportFormat{FormatCSV}
}

// Export exports time series data in pivot table format
func (cpe *CSVPivotExporter) Export(ctx context.Context, writer io.Writer, data []*models.TimeSeries, options ExportOptions) error {
	// Create a pivot table with timestamps as rows and series as columns
	timestampMap := make(map[time.Time]map[string]float64)
	seriesNames := make([]string, 0)
	seriesSet := make(map[string]bool)

	// Collect all unique timestamps and series
	for _, ts := range data {
		if !seriesSet[ts.Name] {
			seriesNames = append(seriesNames, ts.Name)
			seriesSet[ts.Name] = true
		}

		for _, dp := range ts.DataPoints {
			if _, exists := timestampMap[dp.Timestamp]; !exists {
				timestampMap[dp.Timestamp] = make(map[string]float64)
			}
			timestampMap[dp.Timestamp][ts.Name] = dp.Value
		}
	}

	// Convert map to sorted slice of timestamps
	timestamps := make([]time.Time, 0, len(timestampMap))
	for timestamp := range timestampMap {
		timestamps = append(timestamps, timestamp)
	}

	// Sort timestamps
	for i := 0; i < len(timestamps)-1; i++ {
		for j := i + 1; j < len(timestamps); j++ {
			if timestamps[i].After(timestamps[j]) {
				timestamps[i], timestamps[j] = timestamps[j], timestamps[i]
			}
		}
	}

	csvWriter := csv.NewWriter(writer)
	defer csvWriter.Flush()

	// Write header
	if options.IncludeHeaders {
		header := []string{"timestamp"}
		header = append(header, seriesNames...)
		if err := csvWriter.Write(header); err != nil {
			return fmt.Errorf("failed to write pivot CSV headers: %w", err)
		}
	}

	// Write data rows
	timeFormat := options.DateFormat
	if timeFormat == "" {
		timeFormat = time.RFC3339
	}

	for _, timestamp := range timestamps {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		row := []string{timestamp.Format(timeFormat)}
		
		for _, seriesName := range seriesNames {
			if value, exists := timestampMap[timestamp][seriesName]; exists {
				row = append(row, strconv.FormatFloat(value, 'f', options.Precision, 64))
			} else {
				row = append(row, options.CSVOptions.NullValue)
			}
		}

		if err := csvWriter.Write(row); err != nil {
			return fmt.Errorf("failed to write pivot CSV row: %w", err)
		}
	}

	return nil
}

// ValidateOptions validates pivot CSV export options
func (cpe *CSVPivotExporter) ValidateOptions(options ExportOptions) error {
	return (&CSVExporter{}).ValidateOptions(options)
}

// EstimateSize estimates the output size for pivot CSV export
func (cpe *CSVPivotExporter) EstimateSize(data []*models.TimeSeries, options ExportOptions) (int64, error) {
	// Count unique timestamps
	timestampSet := make(map[time.Time]bool)
	for _, ts := range data {
		for _, dp := range ts.DataPoints {
			timestampSet[dp.Timestamp] = true
		}
	}

	rows := int64(len(timestampSet))
	cols := int64(len(data)) + 1 // +1 for timestamp column
	
	// Estimate: ~20 bytes per cell
	estimatedSize := rows * cols * 20

	return estimatedSize, nil
}