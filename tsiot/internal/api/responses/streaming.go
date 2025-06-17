package responses

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"

	"github.com/your-org/tsiot/internal/storage"
	"github.com/your-org/tsiot/pkg/constants"
)

// StreamingResponse handles streaming format responses
type StreamingResponse struct {
	logger *logrus.Logger
}

// NewStreamingResponse creates a new streaming response handler
func NewStreamingResponse(logger *logrus.Logger) *StreamingResponse {
	return &StreamingResponse{
		logger: logger,
	}
}

// StreamingOptions contains options for streaming responses
type StreamingOptions struct {
	ChunkSize    int           // Number of data points per chunk
	FlushDelay   time.Duration // Delay between chunks
	IncludeStats bool          // Include statistics in stream
	Format       string        // Format: "json", "csv", "ndjson"
}

// DefaultStreamingOptions returns default streaming options
func DefaultStreamingOptions() *StreamingOptions {
	return &StreamingOptions{
		ChunkSize:    1000,
		FlushDelay:   100 * time.Millisecond,
		IncludeStats: true,
		Format:       "ndjson", // Newline Delimited JSON
	}
}

// StreamDataPoint represents a single data point in the stream
type StreamDataPoint struct {
	SeriesID  string    `json:"series_id"`
	Timestamp time.Time `json:"timestamp"`
	Value     float64   `json:"value"`
	Quality   float64   `json:"quality,omitempty"`
	Index     int       `json:"index"`
}

// StreamMetadata represents metadata about the stream
type StreamMetadata struct {
	Type         string                 `json:"type"`
	SeriesID     string                 `json:"series_id,omitempty"`
	TotalPoints  int                    `json:"total_points,omitempty"`
	ChunkSize    int                    `json:"chunk_size,omitempty"`
	StartTime    time.Time              `json:"start_time,omitempty"`
	EndTime      time.Time              `json:"end_time,omitempty"`
	Generator    string                 `json:"generator,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	Timestamp    time.Time              `json:"timestamp"`
}

// StreamChunk represents a chunk of data in the stream
type StreamChunk struct {
	Type      string             `json:"type"`
	SeriesID  string             `json:"series_id"`
	ChunkID   int                `json:"chunk_id"`
	DataCount int                `json:"data_count"`
	Data      []*StreamDataPoint `json:"data"`
	Timestamp time.Time          `json:"timestamp"`
}

// StreamEnd represents the end of a stream
type StreamEnd struct {
	Type         string    `json:"type"`
	SeriesID     string    `json:"series_id,omitempty"`
	TotalPoints  int       `json:"total_points"`
	TotalChunks  int       `json:"total_chunks"`
	Duration     string    `json:"duration"`
	Timestamp    time.Time `json:"timestamp"`
}

// WriteTimeSeriesStream writes a time series as a streaming response
func (r *StreamingResponse) WriteTimeSeriesStream(c *gin.Context, timeSeries *storage.TimeSeries, options *StreamingOptions) error {
	if options == nil {
		options = DefaultStreamingOptions()
	}

	// Set streaming headers
	c.Header("Content-Type", "application/x-ndjson; charset=utf-8")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Content-Type-Options", "nosniff")
	
	// Ensure the response is flushed immediately
	c.Writer.WriteHeader(http.StatusOK)

	writer := bufio.NewWriter(c.Writer)
	flusher, canFlush := c.Writer.(http.Flusher)

	startTime := time.Now()

	// Send metadata
	if options.IncludeStats {
		metadata := &StreamMetadata{
			Type:        "metadata",
			SeriesID:    timeSeries.SeriesID,
			TotalPoints: len(timeSeries.Values),
			ChunkSize:   options.ChunkSize,
			Generator:   timeSeries.Generator,
			Metadata:    timeSeries.Metadata,
			Timestamp:   time.Now().UTC(),
		}
		
		if len(timeSeries.Timestamps) > 0 {
			metadata.StartTime = timeSeries.Timestamps[0]
			metadata.EndTime = timeSeries.Timestamps[len(timeSeries.Timestamps)-1]
		}

		if err := r.writeStreamLine(writer, metadata); err != nil {
			return err
		}
		
		if canFlush {
			writer.Flush()
			flusher.Flush()
		}
	}

	// Stream data in chunks
	totalPoints := len(timeSeries.Values)
	chunkCount := 0
	
	for i := 0; i < totalPoints; i += options.ChunkSize {
		end := i + options.ChunkSize
		if end > totalPoints {
			end = totalPoints
		}

		// Create chunk data
		chunkData := make([]*StreamDataPoint, 0, end-i)
		for j := i; j < end && j < len(timeSeries.Timestamps); j++ {
			quality := 0.0
			if timeSeries.Quality != nil && j < len(timeSeries.Quality) {
				quality = timeSeries.Quality[j]
			}

			dataPoint := &StreamDataPoint{
				SeriesID:  timeSeries.SeriesID,
				Timestamp: timeSeries.Timestamps[j],
				Value:     timeSeries.Values[j],
				Quality:   quality,
				Index:     j,
			}
			chunkData = append(chunkData, dataPoint)
		}

		// Create chunk
		chunk := &StreamChunk{
			Type:      "chunk",
			SeriesID:  timeSeries.SeriesID,
			ChunkID:   chunkCount,
			DataCount: len(chunkData),
			Data:      chunkData,
			Timestamp: time.Now().UTC(),
		}

		// Write chunk
		if err := r.writeStreamLine(writer, chunk); err != nil {
			r.logger.WithError(err).Error("Failed to write stream chunk")
			return err
		}

		chunkCount++

		// Flush and delay
		if canFlush {
			writer.Flush()
			flusher.Flush()
		}

		if options.FlushDelay > 0 {
			time.Sleep(options.FlushDelay)
		}
	}

	// Send end marker
	endMarker := &StreamEnd{
		Type:        "end",
		SeriesID:    timeSeries.SeriesID,
		TotalPoints: totalPoints,
		TotalChunks: chunkCount,
		Duration:    time.Since(startTime).String(),
		Timestamp:   time.Now().UTC(),
	}

	if err := r.writeStreamLine(writer, endMarker); err != nil {
		return err
	}

	// Final flush
	writer.Flush()
	if canFlush {
		flusher.Flush()
	}

	return nil
}

// WriteBatchTimeSeriesStream writes multiple time series as a streaming response
func (r *StreamingResponse) WriteBatchTimeSeriesStream(c *gin.Context, timeSeriesList []*storage.TimeSeries, options *StreamingOptions) error {
	if options == nil {
		options = DefaultStreamingOptions()
	}

	// Set streaming headers
	c.Header("Content-Type", "application/x-ndjson; charset=utf-8")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Content-Type-Options", "nosniff")
	
	c.Writer.WriteHeader(http.StatusOK)

	writer := bufio.NewWriter(c.Writer)
	flusher, canFlush := c.Writer.(http.Flusher)

	startTime := time.Now()

	// Send batch metadata
	if options.IncludeStats {
		totalPoints := 0
		for _, ts := range timeSeriesList {
			totalPoints += len(ts.Values)
		}

		metadata := &StreamMetadata{
			Type:        "batch_metadata",
			TotalPoints: totalPoints,
			ChunkSize:   options.ChunkSize,
			Timestamp:   time.Now().UTC(),
			Metadata: map[string]interface{}{
				"series_count": len(timeSeriesList),
				"batch_mode":   true,
			},
		}

		if err := r.writeStreamLine(writer, metadata); err != nil {
			return err
		}
		
		if canFlush {
			writer.Flush()
			flusher.Flush()
		}
	}

	// Stream each time series
	for seriesIndex, timeSeries := range timeSeriesList {
		// Send series metadata
		if options.IncludeStats {
			seriesMetadata := &StreamMetadata{
				Type:        "series_metadata",
				SeriesID:    timeSeries.SeriesID,
				TotalPoints: len(timeSeries.Values),
				Generator:   timeSeries.Generator,
				Metadata:    timeSeries.Metadata,
				Timestamp:   time.Now().UTC(),
			}

			if err := r.writeStreamLine(writer, seriesMetadata); err != nil {
				return err
			}
		}

		// Stream series data
		if err := r.streamSingleSeries(writer, flusher, canFlush, timeSeries, options, seriesIndex); err != nil {
			return err
		}
	}

	// Send batch end marker
	endMarker := &StreamEnd{
		Type:        "batch_end",
		TotalChunks: len(timeSeriesList),
		Duration:    time.Since(startTime).String(),
		Timestamp:   time.Now().UTC(),
	}

	if err := r.writeStreamLine(writer, endMarker); err != nil {
		return err
	}

	// Final flush
	writer.Flush()
	if canFlush {
		flusher.Flush()
	}

	return nil
}

// WriteValidationResultsStream writes validation results as a streaming response
func (r *StreamingResponse) WriteValidationResultsStream(c *gin.Context, seriesID string, results map[string]interface{}) error {
	// Set streaming headers
	c.Header("Content-Type", "application/x-ndjson; charset=utf-8")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	
	c.Writer.WriteHeader(http.StatusOK)

	writer := bufio.NewWriter(c.Writer)
	flusher, canFlush := c.Writer.(http.Flusher)

	// Send metadata
	metadata := &StreamMetadata{
		Type:      "validation_metadata",
		SeriesID:  seriesID,
		Timestamp: time.Now().UTC(),
		Metadata: map[string]interface{}{
			"validation_types": len(results),
		},
	}

	if err := r.writeStreamLine(writer, metadata); err != nil {
		return err
	}

	// Stream validation results
	for validationType, validationData := range results {
		resultChunk := map[string]interface{}{
			"type":            "validation_result",
			"series_id":       seriesID,
			"validation_type": validationType,
			"results":         validationData,
			"timestamp":       time.Now().UTC(),
		}

		if err := r.writeStreamLine(writer, resultChunk); err != nil {
			return err
		}

		if canFlush {
			writer.Flush()
			flusher.Flush()
		}
	}

	// Send end marker
	endMarker := &StreamEnd{
		Type:      "validation_end",
		SeriesID:  seriesID,
		Timestamp: time.Now().UTC(),
	}

	if err := r.writeStreamLine(writer, endMarker); err != nil {
		return err
	}

	writer.Flush()
	if canFlush {
		flusher.Flush()
	}

	return nil
}

// Helper methods

func (r *StreamingResponse) writeStreamLine(writer *bufio.Writer, data interface{}) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}

	_, err = writer.Write(jsonData)
	if err != nil {
		return err
	}

	_, err = writer.WriteString("\n")
	return err
}

func (r *StreamingResponse) streamSingleSeries(writer *bufio.Writer, flusher http.Flusher, canFlush bool, timeSeries *storage.TimeSeries, options *StreamingOptions, seriesIndex int) error {
	totalPoints := len(timeSeries.Values)
	chunkCount := 0

	for i := 0; i < totalPoints; i += options.ChunkSize {
		end := i + options.ChunkSize
		if end > totalPoints {
			end = totalPoints
		}

		// Create chunk data
		chunkData := make([]*StreamDataPoint, 0, end-i)
		for j := i; j < end && j < len(timeSeries.Timestamps); j++ {
			quality := 0.0
			if timeSeries.Quality != nil && j < len(timeSeries.Quality) {
				quality = timeSeries.Quality[j]
			}

			dataPoint := &StreamDataPoint{
				SeriesID:  timeSeries.SeriesID,
				Timestamp: timeSeries.Timestamps[j],
				Value:     timeSeries.Values[j],
				Quality:   quality,
				Index:     j,
			}
			chunkData = append(chunkData, dataPoint)
		}

		// Create chunk
		chunk := &StreamChunk{
			Type:      "series_chunk",
			SeriesID:  timeSeries.SeriesID,
			ChunkID:   chunkCount,
			DataCount: len(chunkData),
			Data:      chunkData,
			Timestamp: time.Now().UTC(),
		}

		// Write chunk
		if err := r.writeStreamLine(writer, chunk); err != nil {
			return err
		}

		chunkCount++

		// Flush and delay
		if canFlush {
			writer.Flush()
			flusher.Flush()
		}

		if options.FlushDelay > 0 {
			time.Sleep(options.FlushDelay)
		}
	}

	// Send series end marker
	seriesEnd := &StreamEnd{
		Type:        "series_end",
		SeriesID:    timeSeries.SeriesID,
		TotalPoints: totalPoints,
		TotalChunks: chunkCount,
		Timestamp:   time.Now().UTC(),
	}

	return r.writeStreamLine(writer, seriesEnd)
}