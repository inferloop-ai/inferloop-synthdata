package streaming

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/inferloop/tsiot/pkg/models"
)

// FilterProcessor implements filtering logic for stream processing
type FilterProcessor struct {
	mu      sync.RWMutex
	metrics *ProcessorMetrics
}

// FilterConfig configures the filter processor
type FilterConfig struct {
	Conditions []FilterCondition `json:"conditions"`
	LogicOp    string            `json:"logic_op"` // AND, OR
}

// FilterCondition represents a single filter condition
type FilterCondition struct {
	Field    string      `json:"field"`
	Operator string      `json:"operator"` // eq, ne, gt, lt, gte, lte, contains, regex
	Value    interface{} `json:"value"`
	Type     string      `json:"type"` // string, number, boolean
}

// NewFilterProcessor creates a new filter processor
func NewFilterProcessor() *FilterProcessor {
	return &FilterProcessor{
		metrics: &ProcessorMetrics{},
	}
}

// Name returns the processor name
func (fp *FilterProcessor) Name() string {
	return "filter"
}

// Process processes a message through the filter
func (fp *FilterProcessor) Process(ctx context.Context, message *StreamMessage) (*StreamMessage, error) {
	startTime := time.Now()
	defer func() {
		fp.mu.Lock()
		fp.metrics.MessagesProcessed++
		fp.metrics.ProcessingTime += time.Since(startTime)
		fp.mu.Unlock()
	}()

	// Parse filter config from message headers or use default
	filterConfig, err := fp.parseFilterConfig(message)
	if err != nil {
		fp.mu.Lock()
		fp.metrics.Errors++
		fp.mu.Unlock()
		return nil, fmt.Errorf("failed to parse filter config: %w", err)
	}

	// Apply filter conditions
	if fp.shouldFilter(message, filterConfig) {
		return nil, nil // Message filtered out
	}

	return message, nil
}

// GetMetrics returns processor metrics
func (fp *FilterProcessor) GetMetrics() ProcessorMetrics {
	fp.mu.RLock()
	defer fp.mu.RUnlock()
	return *fp.metrics
}

func (fp *FilterProcessor) parseFilterConfig(message *StreamMessage) (*FilterConfig, error) {
	// Try to get config from headers
	if configStr, exists := message.Headers["filter_config"]; exists {
		var config FilterConfig
		if err := json.Unmarshal([]byte(configStr), &config); err != nil {
			return nil, err
		}
		return &config, nil
	}

	// Default filter config
	return &FilterConfig{
		Conditions: []FilterCondition{},
		LogicOp:    "AND",
	}, nil
}

func (fp *FilterProcessor) shouldFilter(message *StreamMessage, config *FilterConfig) bool {
	if len(config.Conditions) == 0 {
		return false // No conditions, don't filter
	}

	results := make([]bool, len(config.Conditions))
	
	for i, condition := range config.Conditions {
		results[i] = fp.evaluateCondition(message, condition)
	}

	// Apply logic operator
	if config.LogicOp == "OR" {
		for _, result := range results {
			if result {
				return false // At least one condition matches, don't filter
			}
		}
		return true // No conditions match, filter out
	} else { // Default to AND
		for _, result := range results {
			if !result {
				return true // At least one condition fails, filter out
			}
		}
		return false // All conditions match, don't filter
	}
}

func (fp *FilterProcessor) evaluateCondition(message *StreamMessage, condition FilterCondition) bool {
	fieldValue := fp.getFieldValue(message, condition.Field)
	if fieldValue == nil {
		return false
	}

	switch condition.Operator {
	case "eq":
		return fp.compareValues(fieldValue, condition.Value) == 0
	case "ne":
		return fp.compareValues(fieldValue, condition.Value) != 0
	case "gt":
		return fp.compareValues(fieldValue, condition.Value) > 0
	case "lt":
		return fp.compareValues(fieldValue, condition.Value) < 0
	case "gte":
		return fp.compareValues(fieldValue, condition.Value) >= 0
	case "lte":
		return fp.compareValues(fieldValue, condition.Value) <= 0
	case "contains":
		return fp.containsValue(fieldValue, condition.Value)
	case "regex":
		return fp.matchesRegex(fieldValue, condition.Value)
	default:
		return false
	}
}

func (fp *FilterProcessor) getFieldValue(message *StreamMessage, field string) interface{} {
	switch field {
	case "topic":
		return message.Topic
	case "partition":
		return message.Partition
	case "offset":
		return message.Offset
	case "timestamp":
		return message.Timestamp
	case "value":
		if message.TimeSeries != nil && len(message.TimeSeries.DataPoints) > 0 {
			return message.TimeSeries.DataPoints[0].Value
		}
	case "sensor_type":
		if message.TimeSeries != nil {
			return message.TimeSeries.SensorType
		}
	default:
		// Check headers
		if value, exists := message.Headers[field]; exists {
			return value
		}
		// Check metadata
		if value, exists := message.Metadata[field]; exists {
			return value
		}
		// Check time series metadata
		if message.TimeSeries != nil && message.TimeSeries.Metadata != nil {
			if value, exists := message.TimeSeries.Metadata[field]; exists {
				return value
			}
		}
	}
	return nil
}

func (fp *FilterProcessor) compareValues(a, b interface{}) int {
	// Convert to comparable types
	aStr := fmt.Sprintf("%v", a)
	bStr := fmt.Sprintf("%v", b)

	// Try numeric comparison first
	if aFloat, aErr := strconv.ParseFloat(aStr, 64); aErr == nil {
		if bFloat, bErr := strconv.ParseFloat(bStr, 64); bErr == nil {
			if aFloat < bFloat {
				return -1
			} else if aFloat > bFloat {
				return 1
			}
			return 0
		}
	}

	// Fall back to string comparison
	if aStr < bStr {
		return -1
	} else if aStr > bStr {
		return 1
	}
	return 0
}

func (fp *FilterProcessor) containsValue(field, value interface{}) bool {
	fieldStr := strings.ToLower(fmt.Sprintf("%v", field))
	valueStr := strings.ToLower(fmt.Sprintf("%v", value))
	return strings.Contains(fieldStr, valueStr)
}

func (fp *FilterProcessor) matchesRegex(field, pattern interface{}) bool {
	fieldStr := fmt.Sprintf("%v", field)
	patternStr := fmt.Sprintf("%v", pattern)
	
	if regex, err := regexp.Compile(patternStr); err == nil {
		return regex.MatchString(fieldStr)
	}
	return false
}

// TransformProcessor implements data transformation logic
type TransformProcessor struct {
	mu      sync.RWMutex
	metrics *ProcessorMetrics
}

// TransformConfig configures the transform processor
type TransformConfig struct {
	Transformations []Transformation `json:"transformations"`
}

// Transformation represents a single transformation
type Transformation struct {
	Type       string                 `json:"type"` // scale, offset, unit_convert, normalize, aggregate
	Field      string                 `json:"field"`
	Parameters map[string]interface{} `json:"parameters"`
	OutputField string                `json:"output_field,omitempty"`
}

// NewTransformProcessor creates a new transform processor
func NewTransformProcessor() *TransformProcessor {
	return &TransformProcessor{
		metrics: &ProcessorMetrics{},
	}
}

// Name returns the processor name
func (tp *TransformProcessor) Name() string {
	return "transform"
}

// Process processes a message through transformations
func (tp *TransformProcessor) Process(ctx context.Context, message *StreamMessage) (*StreamMessage, error) {
	startTime := time.Now()
	defer func() {
		tp.mu.Lock()
		tp.metrics.MessagesProcessed++
		tp.metrics.ProcessingTime += time.Since(startTime)
		tp.mu.Unlock()
	}()

	// Parse transform config
	transformConfig, err := tp.parseTransformConfig(message)
	if err != nil {
		tp.mu.Lock()
		tp.metrics.Errors++
		tp.mu.Unlock()
		return nil, fmt.Errorf("failed to parse transform config: %w", err)
	}

	// Create a copy of the message to avoid modifying the original
	transformedMessage := tp.copyMessage(message)

	// Apply transformations
	for _, transformation := range transformConfig.Transformations {
		if err := tp.applyTransformation(transformedMessage, transformation); err != nil {
			tp.mu.Lock()
			tp.metrics.Errors++
			tp.mu.Unlock()
			return nil, fmt.Errorf("transformation failed: %w", err)
		}

		tp.mu.Lock()
		tp.metrics.Transformations++
		tp.mu.Unlock()
	}

	return transformedMessage, nil
}

// GetMetrics returns processor metrics
func (tp *TransformProcessor) GetMetrics() ProcessorMetrics {
	tp.mu.RLock()
	defer tp.mu.RUnlock()
	return *tp.metrics
}

func (tp *TransformProcessor) parseTransformConfig(message *StreamMessage) (*TransformConfig, error) {
	// Try to get config from headers
	if configStr, exists := message.Headers["transform_config"]; exists {
		var config TransformConfig
		if err := json.Unmarshal([]byte(configStr), &config); err != nil {
			return nil, err
		}
		return &config, nil
	}

	// Default transform config
	return &TransformConfig{
		Transformations: []Transformation{},
	}, nil
}

func (tp *TransformProcessor) copyMessage(original *StreamMessage) *StreamMessage {
	copy := &StreamMessage{
		ID:        original.ID,
		Topic:     original.Topic,
		Partition: original.Partition,
		Offset:    original.Offset,
		Key:       make([]byte, len(original.Key)),
		Value:     make([]byte, len(original.Value)),
		Headers:   make(map[string]string),
		Timestamp: original.Timestamp,
		SchemaID:  original.SchemaID,
		Metadata:  make(map[string]interface{}),
	}

	copy(copy.Key, original.Key)
	copy(copy.Value, original.Value)

	for k, v := range original.Headers {
		copy.Headers[k] = v
	}

	for k, v := range original.Metadata {
		copy.Metadata[k] = v
	}

	// Deep copy TimeSeries if present
	if original.TimeSeries != nil {
		copy.TimeSeries = tp.copyTimeSeries(original.TimeSeries)
	}

	return copy
}

func (tp *TransformProcessor) copyTimeSeries(original *models.TimeSeries) *models.TimeSeries {
	copy := &models.TimeSeries{
		ID:          original.ID,
		Name:        original.Name,
		Description: original.Description,
		SensorType:  original.SensorType,
		Frequency:   original.Frequency,
		CreatedAt:   original.CreatedAt,
		UpdatedAt:   original.UpdatedAt,
		Tags:        make(map[string]string),
		Metadata:    make(map[string]interface{}),
		DataPoints:  make([]models.DataPoint, len(original.DataPoints)),
	}

	for k, v := range original.Tags {
		copy.Tags[k] = v
	}

	for k, v := range original.Metadata {
		copy.Metadata[k] = v
	}

	copy(copy.DataPoints, original.DataPoints)

	return copy
}

func (tp *TransformProcessor) applyTransformation(message *StreamMessage, transformation Transformation) error {
	switch transformation.Type {
	case "scale":
		return tp.applyScale(message, transformation)
	case "offset":
		return tp.applyOffset(message, transformation)
	case "unit_convert":
		return tp.applyUnitConvert(message, transformation)
	case "normalize":
		return tp.applyNormalize(message, transformation)
	case "enrich":
		return tp.applyEnrich(message, transformation)
	default:
		return fmt.Errorf("unknown transformation type: %s", transformation.Type)
	}
}

func (tp *TransformProcessor) applyScale(message *StreamMessage, transformation Transformation) error {
	scale, exists := transformation.Parameters["scale"]
	if !exists {
		return fmt.Errorf("scale parameter required")
	}

	scaleFloat, ok := scale.(float64)
	if !ok {
		return fmt.Errorf("scale must be a number")
	}

	if message.TimeSeries != nil {
		for i := range message.TimeSeries.DataPoints {
			message.TimeSeries.DataPoints[i].Value *= scaleFloat
		}
	}

	return nil
}

func (tp *TransformProcessor) applyOffset(message *StreamMessage, transformation Transformation) error {
	offset, exists := transformation.Parameters["offset"]
	if !exists {
		return fmt.Errorf("offset parameter required")
	}

	offsetFloat, ok := offset.(float64)
	if !ok {
		return fmt.Errorf("offset must be a number")
	}

	if message.TimeSeries != nil {
		for i := range message.TimeSeries.DataPoints {
			message.TimeSeries.DataPoints[i].Value += offsetFloat
		}
	}

	return nil
}

func (tp *TransformProcessor) applyUnitConvert(message *StreamMessage, transformation Transformation) error {
	fromUnit, exists := transformation.Parameters["from_unit"]
	if !exists {
		return fmt.Errorf("from_unit parameter required")
	}

	toUnit, exists := transformation.Parameters["to_unit"]
	if !exists {
		return fmt.Errorf("to_unit parameter required")
	}

	conversionFactor := tp.getUnitConversionFactor(fromUnit.(string), toUnit.(string))
	if conversionFactor == 0 {
		return fmt.Errorf("unsupported unit conversion: %s to %s", fromUnit, toUnit)
	}

	if message.TimeSeries != nil {
		for i := range message.TimeSeries.DataPoints {
			message.TimeSeries.DataPoints[i].Value *= conversionFactor
		}
	}

	return nil
}

func (tp *TransformProcessor) applyNormalize(message *StreamMessage, transformation Transformation) error {
	method, exists := transformation.Parameters["method"]
	if !exists {
		method = "min_max" // Default normalization method
	}

	if message.TimeSeries == nil || len(message.TimeSeries.DataPoints) == 0 {
		return nil
	}

	switch method {
	case "min_max":
		return tp.applyMinMaxNormalization(message.TimeSeries)
	case "z_score":
		return tp.applyZScoreNormalization(message.TimeSeries)
	default:
		return fmt.Errorf("unknown normalization method: %s", method)
	}
}

func (tp *TransformProcessor) applyEnrich(message *StreamMessage, transformation Transformation) error {
	enrichments, exists := transformation.Parameters["enrichments"]
	if !exists {
		return fmt.Errorf("enrichments parameter required")
	}

	enrichMap, ok := enrichments.(map[string]interface{})
	if !ok {
		return fmt.Errorf("enrichments must be a map")
	}

	// Add enrichments to metadata
	if message.Metadata == nil {
		message.Metadata = make(map[string]interface{})
	}

	for key, value := range enrichMap {
		message.Metadata[key] = value
	}

	// Add enrichments to TimeSeries metadata if present
	if message.TimeSeries != nil {
		if message.TimeSeries.Metadata == nil {
			message.TimeSeries.Metadata = make(map[string]interface{})
		}
		
		for key, value := range enrichMap {
			message.TimeSeries.Metadata[key] = value
		}
	}

	return nil
}

func (tp *TransformProcessor) getUnitConversionFactor(fromUnit, toUnit string) float64 {
	// Temperature conversions
	if fromUnit == "celsius" && toUnit == "fahrenheit" {
		return 9.0/5.0 // Note: this is just the scaling factor, offset is handled separately
	}
	if fromUnit == "fahrenheit" && toUnit == "celsius" {
		return 5.0/9.0
	}

	// Length conversions
	conversionMap := map[string]map[string]float64{
		"mm": {"cm": 0.1, "m": 0.001, "km": 0.000001, "in": 0.0393701, "ft": 0.00328084},
		"cm": {"mm": 10, "m": 0.01, "km": 0.00001, "in": 0.393701, "ft": 0.0328084},
		"m":  {"mm": 1000, "cm": 100, "km": 0.001, "in": 39.3701, "ft": 3.28084},
		"km": {"mm": 1000000, "cm": 100000, "m": 1000, "in": 39370.1, "ft": 3280.84},
		"in": {"mm": 25.4, "cm": 2.54, "m": 0.0254, "km": 0.0000254, "ft": 0.0833333},
		"ft": {"mm": 304.8, "cm": 30.48, "m": 0.3048, "km": 0.0003048, "in": 12},
	}

	if fromMap, exists := conversionMap[fromUnit]; exists {
		if factor, exists := fromMap[toUnit]; exists {
			return factor
		}
	}

	return 0 // Unsupported conversion
}

func (tp *TransformProcessor) applyMinMaxNormalization(timeSeries *models.TimeSeries) error {
	if len(timeSeries.DataPoints) == 0 {
		return nil
	}

	// Find min and max values
	min := timeSeries.DataPoints[0].Value
	max := timeSeries.DataPoints[0].Value

	for _, dp := range timeSeries.DataPoints {
		if dp.Value < min {
			min = dp.Value
		}
		if dp.Value > max {
			max = dp.Value
		}
	}

	// Avoid division by zero
	if max == min {
		for i := range timeSeries.DataPoints {
			timeSeries.DataPoints[i].Value = 0
		}
		return nil
	}

	// Normalize values to [0, 1]
	for i := range timeSeries.DataPoints {
		timeSeries.DataPoints[i].Value = (timeSeries.DataPoints[i].Value - min) / (max - min)
	}

	return nil
}

func (tp *TransformProcessor) applyZScoreNormalization(timeSeries *models.TimeSeries) error {
	if len(timeSeries.DataPoints) == 0 {
		return nil
	}

	// Calculate mean
	sum := 0.0
	for _, dp := range timeSeries.DataPoints {
		sum += dp.Value
	}
	mean := sum / float64(len(timeSeries.DataPoints))

	// Calculate standard deviation
	squaredDiffSum := 0.0
	for _, dp := range timeSeries.DataPoints {
		diff := dp.Value - mean
		squaredDiffSum += diff * diff
	}
	stdDev := math.Sqrt(squaredDiffSum / float64(len(timeSeries.DataPoints)))

	// Avoid division by zero
	if stdDev == 0 {
		for i := range timeSeries.DataPoints {
			timeSeries.DataPoints[i].Value = 0
		}
		return nil
	}

	// Normalize values using z-score
	for i := range timeSeries.DataPoints {
		timeSeries.DataPoints[i].Value = (timeSeries.DataPoints[i].Value - mean) / stdDev
	}

	return nil
}

// AggregateProcessor implements aggregation logic for stream processing
type AggregateProcessor struct {
	mu      sync.RWMutex
	metrics *ProcessorMetrics
	windows map[string]*AggregationWindow
}

// AggregateConfig configures the aggregate processor
type AggregateConfig struct {
	WindowSize     time.Duration `json:"window_size"`
	WindowType     string        `json:"window_type"` // tumbling, sliding, session
	AggregateFunc  string        `json:"aggregate_func"` // sum, avg, min, max, count
	GroupByFields  []string      `json:"group_by_fields"`
	OutputInterval time.Duration `json:"output_interval"`
}

// AggregationWindow maintains state for a time window
type AggregationWindow struct {
	StartTime    time.Time
	EndTime      time.Time
	Values       []float64
	Count        int64
	Sum          float64
	Min          float64
	Max          float64
	LastUpdate   time.Time
	GroupKey     string
}

// NewAggregateProcessor creates a new aggregate processor
func NewAggregateProcessor() *AggregateProcessor {
	return &AggregateProcessor{
		metrics: &ProcessorMetrics{},
		windows: make(map[string]*AggregationWindow),
	}
}

// Name returns the processor name
func (ap *AggregateProcessor) Name() string {
	return "aggregate"
}

// Process processes a message through aggregation
func (ap *AggregateProcessor) Process(ctx context.Context, message *StreamMessage) (*StreamMessage, error) {
	startTime := time.Now()
	defer func() {
		ap.mu.Lock()
		ap.metrics.MessagesProcessed++
		ap.metrics.ProcessingTime += time.Since(startTime)
		ap.mu.Unlock()
	}()

	// Parse aggregate config
	aggregateConfig, err := ap.parseAggregateConfig(message)
	if err != nil {
		ap.mu.Lock()
		ap.metrics.Errors++
		ap.mu.Unlock()
		return nil, fmt.Errorf("failed to parse aggregate config: %w", err)
	}

	// Update aggregation window
	windowKey := ap.getWindowKey(message, aggregateConfig)
	window := ap.updateWindow(windowKey, message, aggregateConfig)

	// Check if window should be emitted
	if ap.shouldEmitWindow(window, aggregateConfig) {
		aggregatedMessage := ap.createAggregatedMessage(window, message, aggregateConfig)
		
		// Reset or remove window
		ap.resetWindow(windowKey, aggregateConfig)
		
		return aggregatedMessage, nil
	}

	// No output for this message
	return nil, nil
}

// GetMetrics returns processor metrics
func (ap *AggregateProcessor) GetMetrics() ProcessorMetrics {
	ap.mu.RLock()
	defer ap.mu.RUnlock()
	return *ap.metrics
}

func (ap *AggregateProcessor) parseAggregateConfig(message *StreamMessage) (*AggregateConfig, error) {
	// Try to get config from headers
	if configStr, exists := message.Headers["aggregate_config"]; exists {
		var config AggregateConfig
		if err := json.Unmarshal([]byte(configStr), &config); err != nil {
			return nil, err
		}
		return &config, nil
	}

	// Default aggregate config
	return &AggregateConfig{
		WindowSize:     time.Minute,
		WindowType:     "tumbling",
		AggregateFunc:  "avg",
		GroupByFields:  []string{},
		OutputInterval: time.Minute,
	}, nil
}

func (ap *AggregateProcessor) getWindowKey(message *StreamMessage, config *AggregateConfig) string {
	var keyParts []string

	// Add group by fields to key
	for _, field := range config.GroupByFields {
		if value := ap.getFieldValue(message, field); value != nil {
			keyParts = append(keyParts, fmt.Sprintf("%s=%v", field, value))
		}
	}

	// Add time window to key
	windowStart := ap.getWindowStart(message.Timestamp, config)
	keyParts = append(keyParts, fmt.Sprintf("window=%d", windowStart.Unix()))

	return strings.Join(keyParts, ",")
}

func (ap *AggregateProcessor) getFieldValue(message *StreamMessage, field string) interface{} {
	switch field {
	case "topic":
		return message.Topic
	case "sensor_type":
		if message.TimeSeries != nil {
			return message.TimeSeries.SensorType
		}
	default:
		if value, exists := message.Headers[field]; exists {
			return value
		}
		if value, exists := message.Metadata[field]; exists {
			return value
		}
	}
	return nil
}

func (ap *AggregateProcessor) getWindowStart(timestamp time.Time, config *AggregateConfig) time.Time {
	switch config.WindowType {
	case "tumbling":
		// Round down to window boundary
		windowSizeNanos := config.WindowSize.Nanoseconds()
		return time.Unix(0, (timestamp.UnixNano()/windowSizeNanos)*windowSizeNanos)
	case "sliding":
		// For sliding windows, use the current time minus window size
		return timestamp.Add(-config.WindowSize)
	default:
		return timestamp.Truncate(config.WindowSize)
	}
}

func (ap *AggregateProcessor) updateWindow(windowKey string, message *StreamMessage, config *AggregateConfig) *AggregationWindow {
	ap.mu.Lock()
	defer ap.mu.Unlock()

	window, exists := ap.windows[windowKey]
	if !exists {
		windowStart := ap.getWindowStart(message.Timestamp, config)
		window = &AggregationWindow{
			StartTime:  windowStart,
			EndTime:    windowStart.Add(config.WindowSize),
			Values:     make([]float64, 0),
			Count:      0,
			Sum:        0,
			Min:        math.Inf(1),
			Max:        math.Inf(-1),
			GroupKey:   windowKey,
		}
		ap.windows[windowKey] = window
	}

	// Extract value from message
	if message.TimeSeries != nil && len(message.TimeSeries.DataPoints) > 0 {
		value := message.TimeSeries.DataPoints[0].Value
		
		window.Values = append(window.Values, value)
		window.Count++
		window.Sum += value
		
		if value < window.Min {
			window.Min = value
		}
		if value > window.Max {
			window.Max = value
		}
		
		window.LastUpdate = message.Timestamp
	}

	return window
}

func (ap *AggregateProcessor) shouldEmitWindow(window *AggregationWindow, config *AggregateConfig) bool {
	now := time.Now()
	
	switch config.WindowType {
	case "tumbling":
		return now.After(window.EndTime)
	case "sliding":
		return now.Sub(window.LastUpdate) >= config.OutputInterval
	default:
		return now.After(window.EndTime)
	}
}

func (ap *AggregateProcessor) createAggregatedMessage(window *AggregationWindow, originalMessage *StreamMessage, config *AggregateConfig) *StreamMessage {
	// Calculate aggregated value
	var aggregatedValue float64
	switch config.AggregateFunc {
	case "sum":
		aggregatedValue = window.Sum
	case "avg":
		if window.Count > 0 {
			aggregatedValue = window.Sum / float64(window.Count)
		}
	case "min":
		aggregatedValue = window.Min
	case "max":
		aggregatedValue = window.Max
	case "count":
		aggregatedValue = float64(window.Count)
	default:
		aggregatedValue = window.Sum / float64(window.Count) // Default to average
	}

	// Create aggregated TimeSeries
	aggregatedTimeSeries := &models.TimeSeries{
		ID:         fmt.Sprintf("agg_%s_%d", config.AggregateFunc, window.StartTime.Unix()),
		Name:       fmt.Sprintf("Aggregated %s", config.AggregateFunc),
		SensorType: "aggregated",
		DataPoints: []models.DataPoint{
			{
				Timestamp: window.EndTime,
				Value:     aggregatedValue,
				Quality:   1.0,
			},
		},
		Metadata: map[string]interface{}{
			"aggregation_func":   config.AggregateFunc,
			"window_start":       window.StartTime,
			"window_end":         window.EndTime,
			"sample_count":       window.Count,
			"window_size":        config.WindowSize.String(),
		},
	}

	// Create aggregated message
	aggregatedMessage := &StreamMessage{
		ID:         fmt.Sprintf("agg_%s", window.GroupKey),
		Topic:      originalMessage.Topic + "_aggregated",
		Timestamp:  window.EndTime,
		TimeSeries: aggregatedTimeSeries,
		Headers: map[string]string{
			"aggregation": "true",
			"window_key":  window.GroupKey,
		},
		Metadata: map[string]interface{}{
			"source":           "aggregate_processor",
			"original_topic":   originalMessage.Topic,
			"aggregation_func": config.AggregateFunc,
		},
	}

	return aggregatedMessage
}

func (ap *AggregateProcessor) resetWindow(windowKey string, config *AggregateConfig) {
	ap.mu.Lock()
	defer ap.mu.Unlock()

	if config.WindowType == "tumbling" {
		// Remove completed tumbling window
		delete(ap.windows, windowKey)
	} else {
		// For sliding windows, just clear the values but keep the window
		if window, exists := ap.windows[windowKey]; exists {
			window.Values = window.Values[:0]
			window.Count = 0
			window.Sum = 0
			window.Min = math.Inf(1)
			window.Max = math.Inf(-1)
		}
	}
}