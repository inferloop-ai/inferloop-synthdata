package generators

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
	"github.com/inferloop/tsiot/pkg/constants"
	"github.com/inferloop/tsiot/pkg/errors"
)

// StatisticalGenerator implements statistical-based synthetic data generation
type StatisticalGenerator struct {
	logger     *logrus.Logger
	config     *StatisticalConfig
	trained    bool
	statistics *models.TimeSeriesMetrics
	randSource *rand.Rand
}

// StatisticalConfig contains configuration for statistical generation
type StatisticalConfig struct {
	Method           string  `json:"method"`           // "gaussian", "ar", "ma", "arma"
	NoiseLevel       float64 `json:"noise_level"`      // Amount of noise to add (0.0 to 1.0)
	TrendType        string  `json:"trend_type"`       // "linear", "exponential", "polynomial", "none"
	TrendStrength    float64 `json:"trend_strength"`   // Strength of trend component
	SeasonalPeriod   int     `json:"seasonal_period"`  // Period for seasonal component
	SeasonalStrength float64 `json:"seasonal_strength"` // Strength of seasonal component
	AROrder          int     `json:"ar_order"`         // Order for AR model
	MAOrder          int     `json:"ma_order"`         // Order for MA model
	Seed             int64   `json:"seed"`             // Random seed for reproducibility
}

// NewStatisticalGenerator creates a new statistical generator
func NewStatisticalGenerator(config *StatisticalConfig, logger *logrus.Logger) *StatisticalGenerator {
	if config == nil {
		config = getDefaultStatisticalConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	// Set default values
	if config.Method == "" {
		config.Method = "gaussian"
	}
	if config.NoiseLevel == 0 {
		config.NoiseLevel = 0.1
	}
	if config.TrendType == "" {
		config.TrendType = "none"
	}
	if config.Seed == 0 {
		config.Seed = time.Now().UnixNano()
	}

	return &StatisticalGenerator{
		logger:     logger,
		config:     config,
		trained:    false,
		randSource: rand.New(rand.NewSource(config.Seed)),
	}
}

// GetType returns the generator type
func (g *StatisticalGenerator) GetType() models.GeneratorType {
	return models.GeneratorType(constants.GeneratorTypeStatistical)
}

// GetName returns a human-readable name for the generator
func (g *StatisticalGenerator) GetName() string {
	return "Statistical Generator"
}

// GetDescription returns a description of the generator
func (g *StatisticalGenerator) GetDescription() string {
	return "Generates synthetic time series data using statistical methods including AR, MA, ARMA models with trend and seasonal components"
}

// GetSupportedSensorTypes returns the sensor types this generator supports
func (g *StatisticalGenerator) GetSupportedSensorTypes() []models.SensorType {
	return []models.SensorType{
		models.SensorType(constants.SensorTypeTemperature),
		models.SensorType(constants.SensorTypeHumidity),
		models.SensorType(constants.SensorTypePressure),
		models.SensorType(constants.SensorTypeVibration),
		models.SensorType(constants.SensorTypePower),
		models.SensorType(constants.SensorTypeFlow),
		models.SensorType(constants.SensorTypeLevel),
		models.SensorType(constants.SensorTypeSpeed),
		models.SensorType(constants.SensorTypeCustom),
	}
}

// ValidateParameters validates the generation parameters
func (g *StatisticalGenerator) ValidateParameters(params models.GenerationParameters) error {
	if params.Length <= 0 {
		return errors.NewValidationError("INVALID_LENGTH", "Generation length must be positive")
	}

	if params.Frequency == "" {
		return errors.NewValidationError("INVALID_FREQUENCY", "Frequency is required")
	}

	if g.config.NoiseLevel < 0 || g.config.NoiseLevel > 1 {
		return errors.NewValidationError("INVALID_NOISE_LEVEL", "Noise level must be between 0 and 1")
	}

	if g.config.AROrder < 0 || g.config.AROrder > 10 {
		return errors.NewValidationError("INVALID_AR_ORDER", "AR order must be between 0 and 10")
	}

	if g.config.MAOrder < 0 || g.config.MAOrder > 10 {
		return errors.NewValidationError("INVALID_MA_ORDER", "MA order must be between 0 and 10")
	}

	return nil
}

// Generate generates synthetic data based on the request
func (g *StatisticalGenerator) Generate(ctx context.Context, req *models.GenerationRequest) (*models.GenerationResult, error) {
	if req == nil {
		return nil, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}

	if err := g.ValidateParameters(req.Parameters); err != nil {
		return nil, err
	}

	g.logger.WithFields(logrus.Fields{
		"request_id": req.ID,
		"length":     req.Parameters.Length,
		"method":     g.config.Method,
	}).Info("Starting statistical generation")

	start := time.Now()

	// Parse frequency
	frequency, err := g.parseFrequency(req.Parameters.Frequency)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeValidation, "INVALID_FREQUENCY", "Failed to parse frequency")
	}

	// Generate timestamps
	timestamps := g.generateTimestamps(req.Parameters.StartTime, frequency, req.Parameters.Length)

	// Generate values based on method
	values, err := g.generateValues(ctx, req.Parameters.Length)
	if err != nil {
		return nil, err
	}

	// Create data points
	dataPoints := make([]models.DataPoint, len(timestamps))
	for i, timestamp := range timestamps {
		dataPoints[i] = models.DataPoint{
			Timestamp: timestamp,
			Value:     values[i],
			Quality:   1.0, // Statistical generation has perfect quality
		}
	}

	// Create time series
	timeSeries := &models.TimeSeries{
		ID:          fmt.Sprintf("statistical-%d", time.Now().UnixNano()),
		Name:        fmt.Sprintf("Statistical Generated - %s", g.config.Method),
		Description: fmt.Sprintf("Synthetic data generated using %s method", g.config.Method),
		Tags:        req.Parameters.Tags,
		Metadata:    req.Parameters.Metadata,
		DataPoints:  dataPoints,
		StartTime:   timestamps[0],
		EndTime:     timestamps[len(timestamps)-1],
		Frequency:   req.Parameters.Frequency,
		SensorType:  string(req.Parameters.SensorType),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	duration := time.Since(start)

	result := &models.GenerationResult{
		ID:           req.ID,
		Status:       "completed",
		TimeSeries:   timeSeries,
		Duration:     duration,
		GeneratedAt:  time.Now(),
		GeneratorType: string(g.GetType()),
		Quality:      1.0,
		Metadata: map[string]interface{}{
			"method":           g.config.Method,
			"noise_level":      g.config.NoiseLevel,
			"trend_type":       g.config.TrendType,
			"seasonal_period":  g.config.SeasonalPeriod,
			"data_points":      len(dataPoints),
			"generation_time":  duration.String(),
		},
	}

	g.logger.WithFields(logrus.Fields{
		"request_id":    req.ID,
		"data_points":   len(dataPoints),
		"duration":      duration,
		"method":        g.config.Method,
	}).Info("Completed statistical generation")

	return result, nil
}

// Train trains the generator with reference data
func (g *StatisticalGenerator) Train(ctx context.Context, data *models.TimeSeries, params models.GenerationParameters) error {
	if data == nil {
		return errors.NewValidationError("INVALID_DATA", "Training data is required")
	}

	g.logger.WithFields(logrus.Fields{
		"series_id":    data.ID,
		"data_points":  len(data.DataPoints),
	}).Info("Training statistical generator")

	// Calculate statistics from training data
	g.statistics = data.CalculateMetrics()
	g.trained = true

	g.logger.WithFields(logrus.Fields{
		"mean":     g.statistics.Mean,
		"std_dev":  g.statistics.StdDev,
		"min":      g.statistics.Min,
		"max":      g.statistics.Max,
	}).Info("Statistical generator training completed")

	return nil
}

// IsTrainable returns true if the generator requires/supports training
func (g *StatisticalGenerator) IsTrainable() bool {
	return true
}

// GetDefaultParameters returns default parameters for this generator
func (g *StatisticalGenerator) GetDefaultParameters() models.GenerationParameters {
	return models.GenerationParameters{
		Length:    1000,
		Frequency: "1m",
		StartTime: time.Now().Add(-24 * time.Hour),
		Tags:      make(map[string]string),
		Metadata:  make(map[string]interface{}),
	}
}

// EstimateDuration estimates how long generation will take
func (g *StatisticalGenerator) EstimateDuration(req *models.GenerationRequest) (time.Duration, error) {
	if req == nil {
		return 0, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}

	// Statistical generation is very fast, approximately 1ms per 1000 data points
	pointsPerMs := 1000.0
	estimatedMs := float64(req.Parameters.Length) / pointsPerMs
	return time.Duration(estimatedMs) * time.Millisecond, nil
}

// Cancel cancels an ongoing generation
func (g *StatisticalGenerator) Cancel(ctx context.Context, requestID string) error {
	// Statistical generation is typically too fast to cancel
	// In a real implementation, you might maintain a registry of ongoing generations
	g.logger.WithFields(logrus.Fields{
		"request_id": requestID,
	}).Info("Cancel requested for statistical generation")
	
	return nil
}

// GetProgress returns the progress of an ongoing generation
func (g *StatisticalGenerator) GetProgress(requestID string) (float64, error) {
	// Statistical generation is typically too fast to track progress
	return 1.0, nil
}

// Close cleans up resources
func (g *StatisticalGenerator) Close() error {
	g.logger.Info("Closing statistical generator")
	return nil
}

// generateValues generates synthetic values based on the configured method
func (g *StatisticalGenerator) generateValues(ctx context.Context, length int) ([]float64, error) {
	values := make([]float64, length)

	switch g.config.Method {
	case "gaussian":
		return g.generateGaussian(length), nil
	case "ar":
		return g.generateAR(length), nil
	case "ma":
		return g.generateMA(length), nil
	case "arma":
		return g.generateARMA(length), nil
	default:
		return nil, errors.NewValidationError("INVALID_METHOD", fmt.Sprintf("Unknown generation method: %s", g.config.Method))
	}
}

// generateGaussian generates values using Gaussian (normal) distribution
func (g *StatisticalGenerator) generateGaussian(length int) []float64 {
	values := make([]float64, length)
	
	mean := 0.0
	stdDev := 1.0
	
	// Use training statistics if available
	if g.trained && g.statistics != nil {
		mean = g.statistics.Mean
		stdDev = g.statistics.StdDev
	}

	for i := 0; i < length; i++ {
		// Generate base value
		baseValue := g.randSource.NormFloat64()*stdDev + mean
		
		// Add trend component
		trendValue := g.generateTrend(i, length)
		
		// Add seasonal component
		seasonalValue := g.generateSeasonal(i)
		
		// Add noise
		noiseValue := g.randSource.NormFloat64() * g.config.NoiseLevel * stdDev
		
		values[i] = baseValue + trendValue + seasonalValue + noiseValue
	}

	return values
}

// generateAR generates values using Autoregressive (AR) model
func (g *StatisticalGenerator) generateAR(length int) []float64 {
	values := make([]float64, length)
	
	// AR coefficients (simple example)
	arCoeffs := make([]float64, g.config.AROrder)
	for i := range arCoeffs {
		arCoeffs[i] = 0.5 / float64(i+1) // Decreasing coefficients
	}
	
	// Initialize with random values
	for i := 0; i < min(g.config.AROrder, length); i++ {
		values[i] = g.randSource.NormFloat64()
	}
	
	// Generate AR values
	for i := g.config.AROrder; i < length; i++ {
		value := 0.0
		for j := 0; j < g.config.AROrder; j++ {
			value += arCoeffs[j] * values[i-j-1]
		}
		
		// Add noise
		value += g.randSource.NormFloat64() * g.config.NoiseLevel
		
		// Add trend and seasonal components
		value += g.generateTrend(i, length)
		value += g.generateSeasonal(i)
		
		values[i] = value
	}

	return values
}

// generateMA generates values using Moving Average (MA) model
func (g *StatisticalGenerator) generateMA(length int) []float64 {
	values := make([]float64, length)
	errors := make([]float64, length+g.config.MAOrder)
	
	// Initialize random errors
	for i := range errors {
		errors[i] = g.randSource.NormFloat64()
	}
	
	// MA coefficients
	maCoeffs := make([]float64, g.config.MAOrder)
	for i := range maCoeffs {
		maCoeffs[i] = 0.3 / float64(i+1) // Decreasing coefficients
	}
	
	for i := 0; i < length; i++ {
		value := errors[i+g.config.MAOrder] // Current error
		
		// Add MA component
		for j := 0; j < g.config.MAOrder; j++ {
			value += maCoeffs[j] * errors[i+g.config.MAOrder-j-1]
		}
		
		// Add trend and seasonal components
		value += g.generateTrend(i, length)
		value += g.generateSeasonal(i)
		
		values[i] = value
	}

	return values
}

// generateARMA generates values using ARMA model
func (g *StatisticalGenerator) generateARMA(length int) []float64 {
	values := make([]float64, length)
	errors := make([]float64, length+g.config.MAOrder)
	
	// Initialize random errors
	for i := range errors {
		errors[i] = g.randSource.NormFloat64()
	}
	
	// AR and MA coefficients
	arCoeffs := make([]float64, g.config.AROrder)
	maCoeffs := make([]float64, g.config.MAOrder)
	
	for i := range arCoeffs {
		arCoeffs[i] = 0.4 / float64(i+1)
	}
	for i := range maCoeffs {
		maCoeffs[i] = 0.3 / float64(i+1)
	}
	
	// Initialize with random values
	for i := 0; i < min(g.config.AROrder, length); i++ {
		values[i] = g.randSource.NormFloat64()
	}
	
	// Generate ARMA values
	start := max(g.config.AROrder, g.config.MAOrder)
	for i := start; i < length; i++ {
		value := 0.0
		
		// AR component
		for j := 0; j < g.config.AROrder && i-j-1 >= 0; j++ {
			value += arCoeffs[j] * values[i-j-1]
		}
		
		// MA component
		value += errors[i+g.config.MAOrder]
		for j := 0; j < g.config.MAOrder; j++ {
			value += maCoeffs[j] * errors[i+g.config.MAOrder-j-1]
		}
		
		// Add trend and seasonal components
		value += g.generateTrend(i, length)
		value += g.generateSeasonal(i)
		
		values[i] = value
	}

	return values
}

// generateTrend generates trend component
func (g *StatisticalGenerator) generateTrend(index, length int) float64 {
	if g.config.TrendType == "none" || g.config.TrendStrength == 0 {
		return 0.0
	}
	
	t := float64(index) / float64(length)
	
	switch g.config.TrendType {
	case "linear":
		return g.config.TrendStrength * t
	case "exponential":
		return g.config.TrendStrength * (math.Exp(t) - 1)
	case "polynomial":
		return g.config.TrendStrength * t * t
	case "logarithmic":
		return g.config.TrendStrength * math.Log(1+t)
	default:
		return 0.0
	}
}

// generateSeasonal generates seasonal component
func (g *StatisticalGenerator) generateSeasonal(index int) float64 {
	if g.config.SeasonalPeriod == 0 || g.config.SeasonalStrength == 0 {
		return 0.0
	}
	
	angle := 2 * math.Pi * float64(index) / float64(g.config.SeasonalPeriod)
	return g.config.SeasonalStrength * math.Sin(angle)
}

// generateTimestamps generates a sequence of timestamps
func (g *StatisticalGenerator) generateTimestamps(start time.Time, frequency time.Duration, length int) []time.Time {
	timestamps := make([]time.Time, length)
	current := start
	
	for i := 0; i < length; i++ {
		timestamps[i] = current
		current = current.Add(frequency)
	}
	
	return timestamps
}

// parseFrequency parses a frequency string into a time.Duration
func (g *StatisticalGenerator) parseFrequency(freq string) (time.Duration, error) {
	duration, err := time.ParseDuration(freq)
	if err != nil {
		return 0, fmt.Errorf("invalid frequency format: %s", freq)
	}
	return duration, nil
}

// getDefaultStatisticalConfig returns default configuration
func getDefaultStatisticalConfig() *StatisticalConfig {
	return &StatisticalConfig{
		Method:           "gaussian",
		NoiseLevel:       0.1,
		TrendType:        "none",
		TrendStrength:    0.0,
		SeasonalPeriod:   0,
		SeasonalStrength: 0.0,
		AROrder:          1,
		MAOrder:          1,
		Seed:             time.Now().UnixNano(),
	}
}

// Helper functions
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}