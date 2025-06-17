package export

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"time"

	"github.com/inferloop/tsiot/pkg/models"
)

// JSONExporter implements JSON export functionality
type JSONExporter struct{}

// Name returns the exporter name
func (je *JSONExporter) Name() string {
	return "json"
}

// SupportedFormats returns supported formats
func (je *JSONExporter) SupportedFormats() []ExportFormat {
	return []ExportFormat{FormatJSON}
}

// Export exports time series data to JSON format
func (je *JSONExporter) Export(ctx context.Context, writer io.Writer, data []*models.TimeSeries, options ExportOptions) error {
	encoder := json.NewEncoder(writer)
	
	if options.JSONOptions.Pretty {
		encoder.SetIndent("", "  ")
	}

	// Apply filters to data
	filteredData := je.applyFilters(data, options)

	if options.JSONOptions.StreamFormat {
		return je.exportAsJSONLines(ctx, encoder, filteredData, options)
	} else if options.JSONOptions.ArrayFormat {
		return je.exportAsDataPointArray(ctx, encoder, filteredData, options)
	} else {
		return je.exportAsTimeSeriesObjects(ctx, encoder, filteredData, options)
	}
}

// ValidateOptions validates JSON export options
func (je *JSONExporter) ValidateOptions(options ExportOptions) error {
	// JSON options are generally flexible, minimal validation needed
	return nil
}

// EstimateSize estimates the output size for JSON export
func (je *JSONExporter) EstimateSize(data []*models.TimeSeries, options ExportOptions) (int64, error) {
	totalPoints := int64(0)
	for _, ts := range data {
		totalPoints += int64(len(ts.DataPoints))
	}

	var estimatedSize int64
	if options.JSONOptions.ArrayFormat {
		// Estimate: ~150 bytes per data point in array format
		estimatedSize = totalPoints * 150
	} else if options.JSONOptions.StreamFormat {
		// Estimate: ~120 bytes per data point in streaming format
		estimatedSize = totalPoints * 120
	} else {
		// Estimate: ~200 bytes per data point + metadata overhead
		estimatedSize = totalPoints*200 + int64(len(data))*500
	}

	return estimatedSize, nil
}

// exportAsTimeSeriesObjects exports as complete time series objects
func (je *JSONExporter) exportAsTimeSeriesObjects(ctx context.Context, encoder *json.Encoder, data []*models.TimeSeries, options ExportOptions) error {
	exportData := make([]JSONTimeSeries, len(data))

	for i, ts := range data {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		jsonTS := je.convertToJSONTimeSeries(ts, options)
		exportData[i] = jsonTS
	}

	// Create wrapper object
	wrapper := JSONExportWrapper{
		ExportInfo: JSONExportInfo{
			Timestamp:    time.Now(),
			Format:       "json",
			SeriesCount:  len(data),
			TotalPoints:  je.countTotalPoints(data),
			TimeRange:    je.calculateTimeRange(data),
			ExportedBy:   "tsiot-export-engine",
			Version:      "1.0",
		},
		TimeSeries: exportData,
	}

	return encoder.Encode(wrapper)
}

// exportAsDataPointArray exports as a flat array of data points
func (je *JSONExporter) exportAsDataPointArray(ctx context.Context, encoder *json.Encoder, data []*models.TimeSeries, options ExportOptions) error {
	dataPoints := make([]JSONDataPoint, 0)

	for _, ts := range data {
		for _, dp := range ts.DataPoints {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}

			jsonDP := je.convertToJSONDataPoint(ts, dp, options)
			dataPoints = append(dataPoints, jsonDP)
		}
	}

	// Create wrapper object
	wrapper := JSONDataPointArray{
		ExportInfo: JSONExportInfo{
			Timestamp:   time.Now(),
			Format:      "json_array",
			SeriesCount: len(data),
			TotalPoints: len(dataPoints),
			TimeRange:   je.calculateTimeRange(data),
			ExportedBy:  "tsiot-export-engine",
			Version:     "1.0",
		},
		DataPoints: dataPoints,
	}

	return encoder.Encode(wrapper)
}

// exportAsJSONLines exports as JSON Lines format (one JSON object per line)
func (je *JSONExporter) exportAsJSONLines(ctx context.Context, encoder *json.Encoder, data []*models.TimeSeries, options ExportOptions) error {
	for _, ts := range data {
		for _, dp := range ts.DataPoints {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}

			jsonDP := je.convertToJSONDataPoint(ts, dp, options)
			if err := encoder.Encode(jsonDP); err != nil {
				return fmt.Errorf("failed to encode JSON line: %w", err)
			}
		}
	}

	return nil
}

// JSON data structures

// JSONExportWrapper wraps the exported time series data
type JSONExportWrapper struct {
	ExportInfo JSONExportInfo     `json:"export_info"`
	TimeSeries []JSONTimeSeries   `json:"time_series"`
}

// JSONDataPointArray wraps exported data points as an array
type JSONDataPointArray struct {
	ExportInfo JSONExportInfo   `json:"export_info"`
	DataPoints []JSONDataPoint  `json:"data_points"`
}

// JSONExportInfo contains metadata about the export
type JSONExportInfo struct {
	Timestamp   time.Time         `json:"timestamp"`
	Format      string            `json:"format"`
	SeriesCount int               `json:"series_count"`
	TotalPoints int               `json:"total_points"`
	TimeRange   JSONTimeRange     `json:"time_range"`
	ExportedBy  string            `json:"exported_by"`
	Version     string            `json:"version"`
	Options     map[string]interface{} `json:"options,omitempty"`
}

// JSONTimeRange represents a time range
type JSONTimeRange struct {
	Start *time.Time `json:"start"`
	End   *time.Time `json:"end"`
}

// JSONTimeSeries represents a time series in JSON format
type JSONTimeSeries struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	SensorType  string                 `json:"sensor_type"`
	Tags        map[string]string      `json:"tags,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	DataPoints  []JSONDataPoint        `json:"data_points"`
	Statistics  *JSONStatistics        `json:"statistics,omitempty"`
	TimeRange   JSONTimeRange          `json:"time_range"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// JSONDataPoint represents a data point in JSON format
type JSONDataPoint struct {
	Timestamp   time.Time              `json:"timestamp"`
	Value       float64                `json:"value"`
	Quality     float64                `json:"quality,omitempty"`
	SeriesID    string                 `json:"series_id,omitempty"`
	SeriesName  string                 `json:"series_name,omitempty"`
	SensorType  string                 `json:"sensor_type,omitempty"`
	Tags        map[string]string      `json:"tags,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	CustomFields map[string]string     `json:"custom_fields,omitempty"`
}

// JSONStatistics contains statistical information about the time series
type JSONStatistics struct {
	Count    int     `json:"count"`
	Min      float64 `json:"min"`
	Max      float64 `json:"max"`
	Mean     float64 `json:"mean"`
	Median   float64 `json:"median"`
	StdDev   float64 `json:"std_dev"`
	Variance float64 `json:"variance"`
}

// Helper methods

func (je *JSONExporter) convertToJSONTimeSeries(ts *models.TimeSeries, options ExportOptions) JSONTimeSeries {
	dataPoints := make([]JSONDataPoint, len(ts.DataPoints))
	for i, dp := range ts.DataPoints {
		dataPoints[i] = je.convertToJSONDataPoint(ts, dp, options)
	}

	jsonTS := JSONTimeSeries{
		ID:          ts.ID,
		Name:        ts.Name,
		Description: ts.Description,
		SensorType:  ts.SensorType,
		Tags:        ts.Tags,
		Metadata:    ts.Metadata,
		DataPoints:  dataPoints,
		TimeRange:   je.calculateSeriesTimeRange(ts),
		CreatedAt:   ts.CreatedAt,
		UpdatedAt:   ts.UpdatedAt,
	}

	// Calculate statistics if requested
	if options.JSONOptions.Pretty { // Use this as a proxy for including statistics
		jsonTS.Statistics = je.calculateStatistics(ts)
	}

	return jsonTS
}

func (je *JSONExporter) convertToJSONDataPoint(ts *models.TimeSeries, dp models.DataPoint, options ExportOptions) JSONDataPoint {
	jsonDP := JSONDataPoint{
		Timestamp: je.formatTimestamp(dp.Timestamp, options),
		Value:     je.formatValue(dp.Value, options.Precision),
		Quality:   dp.Quality,
	}

	// Include series information if in array/streaming format
	if options.JSONOptions.ArrayFormat || options.JSONOptions.StreamFormat {
		jsonDP.SeriesID = ts.ID
		jsonDP.SeriesName = ts.Name
		jsonDP.SensorType = ts.SensorType
		
		if len(ts.Tags) > 0 {
			jsonDP.Tags = ts.Tags
		}
		
		if len(ts.Metadata) > 0 {
			jsonDP.Metadata = ts.Metadata
		}
	}

	// Add custom fields
	if len(options.CustomFields) > 0 {
		jsonDP.CustomFields = options.CustomFields
	}

	return jsonDP
}

func (je *JSONExporter) formatTimestamp(timestamp time.Time, options ExportOptions) time.Time {
	if options.TimeZone != "" {
		if loc, err := time.LoadLocation(options.TimeZone); err == nil {
			return timestamp.In(loc)
		}
	}
	return timestamp
}

func (je *JSONExporter) formatValue(value float64, precision int) float64 {
	if precision <= 0 {
		return value
	}
	
	// Round to specified precision
	multiplier := 1.0
	for i := 0; i < precision; i++ {
		multiplier *= 10
	}
	
	return float64(int(value*multiplier+0.5)) / multiplier
}

func (je *JSONExporter) calculateStatistics(ts *models.TimeSeries) *JSONStatistics {
	if len(ts.DataPoints) == 0 {
		return nil
	}

	values := make([]float64, len(ts.DataPoints))
	for i, dp := range ts.DataPoints {
		values[i] = dp.Value
	}

	// Sort for median calculation
	for i := 0; i < len(values)-1; i++ {
		for j := i + 1; j < len(values); j++ {
			if values[i] > values[j] {
				values[i], values[j] = values[j], values[i]
			}
		}
	}

	min := values[0]
	max := values[len(values)-1]
	median := values[len(values)/2]

	// Calculate mean
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))

	// Calculate variance and standard deviation
	variance := 0.0
	for _, v := range values {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(len(values))

	return &JSONStatistics{
		Count:    len(values),
		Min:      min,
		Max:      max,
		Mean:     mean,
		Median:   median,
		StdDev:   je.sqrt(variance),
		Variance: variance,
	}
}

func (je *JSONExporter) sqrt(x float64) float64 {
	// Simple square root implementation
	if x < 0 {
		return 0
	}
	if x == 0 {
		return 0
	}
	
	// Newton's method
	guess := x / 2
	for i := 0; i < 10; i++ {
		guess = (guess + x/guess) / 2
	}
	return guess
}

func (je *JSONExporter) calculateTimeRange(data []*models.TimeSeries) JSONTimeRange {
	if len(data) == 0 {
		return JSONTimeRange{}
	}

	var start, end *time.Time

	for _, ts := range data {
		for _, dp := range ts.DataPoints {
			if start == nil || dp.Timestamp.Before(*start) {
				start = &dp.Timestamp
			}
			if end == nil || dp.Timestamp.After(*end) {
				end = &dp.Timestamp
			}
		}
	}

	return JSONTimeRange{
		Start: start,
		End:   end,
	}
}

func (je *JSONExporter) calculateSeriesTimeRange(ts *models.TimeSeries) JSONTimeRange {
	if len(ts.DataPoints) == 0 {
		return JSONTimeRange{}
	}

	start := ts.DataPoints[0].Timestamp
	end := ts.DataPoints[0].Timestamp

	for _, dp := range ts.DataPoints {
		if dp.Timestamp.Before(start) {
			start = dp.Timestamp
		}
		if dp.Timestamp.After(end) {
			end = dp.Timestamp
		}
	}

	return JSONTimeRange{
		Start: &start,
		End:   &end,
	}
}

func (je *JSONExporter) countTotalPoints(data []*models.TimeSeries) int {
	total := 0
	for _, ts := range data {
		total += len(ts.DataPoints)
	}
	return total
}

func (je *JSONExporter) applyFilters(data []*models.TimeSeries, options ExportOptions) []*models.TimeSeries {
	// This would apply the same filtering logic as in the main export engine
	// For now, return data as-is
	return data
}

// JSONStreamExporter implements streaming JSON export for large datasets
type JSONStreamExporter struct {
	encoder *json.Encoder
	options ExportOptions
	first   bool
}

// NewJSONStreamExporter creates a new streaming JSON exporter
func NewJSONStreamExporter(writer io.Writer, options ExportOptions) *JSONStreamExporter {
	encoder := json.NewEncoder(writer)
	
	if options.JSONOptions.Pretty {
		encoder.SetIndent("", "  ")
	}

	return &JSONStreamExporter{
		encoder: encoder,
		options: options,
		first:   true,
	}
}

// WriteHeader writes the JSON export header
func (jse *JSONStreamExporter) WriteHeader() error {
	if jse.options.JSONOptions.StreamFormat {
		// No header needed for JSON Lines format
		return nil
	}

	// Start array for array format
	if jse.options.JSONOptions.ArrayFormat {
		_, err := jse.encoder.(*json.Encoder).(*json.Encoder) // Type assertion placeholder
		return err
	}

	return nil
}

// WriteTimeSeries writes a time series to the JSON stream
func (jse *JSONStreamExporter) WriteTimeSeries(ts *models.TimeSeries) error {
	exporter := &JSONExporter{}
	
	if jse.options.JSONOptions.StreamFormat {
		// Write each data point as a separate JSON line
		for _, dp := range ts.DataPoints {
			jsonDP := exporter.convertToJSONDataPoint(ts, dp, jse.options)
			if err := jse.encoder.Encode(jsonDP); err != nil {
				return fmt.Errorf("failed to encode JSON stream data point: %w", err)
			}
		}
	} else {
		// Write complete time series object
		jsonTS := exporter.convertToJSONTimeSeries(ts, jse.options)
		if err := jse.encoder.Encode(jsonTS); err != nil {
			return fmt.Errorf("failed to encode JSON time series: %w", err)
		}
	}

	return nil
}

// Close finalizes the JSON export
func (jse *JSONStreamExporter) Close() error {
	// No cleanup needed for JSON export
	return nil
}