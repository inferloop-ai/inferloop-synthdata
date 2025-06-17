package arima

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"

	"github.com/inferloop/tsiot/pkg/constants"
	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
)

// ARIMAGenerator implements ARIMA (AutoRegressive Integrated Moving Average) model
type ARIMAGenerator struct {
	logger      *logrus.Logger
	config      *ARIMAConfig
	trained     bool
	statistics  *models.TimeSeriesMetrics
	randSource  *rand.Rand
	arCoeffs    []float64
	maCoeffs    []float64
	modelParams *ARIMAParameters
}

// ARIMAConfig contains configuration for ARIMA generation
type ARIMAConfig struct {
	P                int     `json:"p"`                  // AR order
	D                int     `json:"d"`                  // Differencing degree
	Q                int     `json:"q"`                  // MA order
	SeasonalP        int     `json:"seasonal_p"`         // Seasonal AR order
	SeasonalD        int     `json:"seasonal_d"`         // Seasonal differencing
	SeasonalQ        int     `json:"seasonal_q"`         // Seasonal MA order
	SeasonalPeriod   int     `json:"seasonal_period"`    // Seasonal period (e.g., 12 for monthly)
	IncludeConstant  bool    `json:"include_constant"`   // Include constant term
	Seed             int64   `json:"seed"`               // Random seed
	OptimizationMethod string `json:"optimization_method"` // "mle", "css", "css-mle"
}

// ARIMAParameters contains fitted ARIMA model parameters
type ARIMAParameters struct {
	ARCoefficients       []float64 `json:"ar_coefficients"`
	MACoefficients       []float64 `json:"ma_coefficients"`
	SeasonalARCoeffs     []float64 `json:"seasonal_ar_coefficients"`
	SeasonalMACoeffs     []float64 `json:"seasonal_ma_coefficients"`
	Constant             float64   `json:"constant"`
	Variance             float64   `json:"variance"`
	LogLikelihood        float64   `json:"log_likelihood"`
	AIC                  float64   `json:"aic"`
	BIC                  float64   `json:"bic"`
	ResidualAutocorrelation []float64 `json:"residual_autocorrelation"`
}

// NewARIMAGenerator creates a new ARIMA generator
func NewARIMAGenerator(config *ARIMAConfig, logger *logrus.Logger) *ARIMAGenerator {
	if config == nil {
		config = getDefaultARIMAConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	if config.Seed == 0 {
		config.Seed = time.Now().UnixNano()
	}

	return &ARIMAGenerator{
		logger:     logger,
		config:     config,
		trained:    false,
		randSource: rand.New(rand.NewSource(config.Seed)),
	}
}

// GetType returns the generator type
func (g *ARIMAGenerator) GetType() models.GeneratorType {
	return models.GeneratorType(constants.GeneratorTypeARIMA)
}

// GetName returns a human-readable name for the generator
func (g *ARIMAGenerator) GetName() string {
	return "ARIMA Generator"
}

// GetDescription returns a description of the generator
func (g *ARIMAGenerator) GetDescription() string {
	return fmt.Sprintf("Generates synthetic time series using ARIMA(%d,%d,%d) model with seasonal components", 
		g.config.P, g.config.D, g.config.Q)
}

// GetSupportedSensorTypes returns the sensor types this generator supports
func (g *ARIMAGenerator) GetSupportedSensorTypes() []models.SensorType {
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
func (g *ARIMAGenerator) ValidateParameters(params models.GenerationParameters) error {
	if params.Length <= 0 {
		return errors.NewValidationError("INVALID_LENGTH", "Generation length must be positive")
	}

	if params.Frequency == "" {
		return errors.NewValidationError("INVALID_FREQUENCY", "Frequency is required")
	}

	// Validate ARIMA orders
	if g.config.P < 0 || g.config.P > 5 {
		return errors.NewValidationError("INVALID_AR_ORDER", "AR order (p) must be between 0 and 5")
	}

	if g.config.D < 0 || g.config.D > 2 {
		return errors.NewValidationError("INVALID_DIFF_ORDER", "Differencing order (d) must be between 0 and 2")
	}

	if g.config.Q < 0 || g.config.Q > 5 {
		return errors.NewValidationError("INVALID_MA_ORDER", "MA order (q) must be between 0 and 5")
	}

	// Validate seasonal parameters
	if g.config.SeasonalPeriod > 0 {
		if g.config.SeasonalP < 0 || g.config.SeasonalP > 2 {
			return errors.NewValidationError("INVALID_SEASONAL_AR", "Seasonal AR order must be between 0 and 2")
		}
		if g.config.SeasonalD < 0 || g.config.SeasonalD > 1 {
			return errors.NewValidationError("INVALID_SEASONAL_DIFF", "Seasonal differencing must be 0 or 1")
		}
		if g.config.SeasonalQ < 0 || g.config.SeasonalQ > 2 {
			return errors.NewValidationError("INVALID_SEASONAL_MA", "Seasonal MA order must be between 0 and 2")
		}
	}

	return nil
}

// Generate generates synthetic data based on the request
func (g *ARIMAGenerator) Generate(ctx context.Context, req *models.GenerationRequest) (*models.GenerationResult, error) {
	if req == nil {
		return nil, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}

	if err := g.ValidateParameters(req.Parameters); err != nil {
		return nil, err
	}

	g.logger.WithFields(logrus.Fields{
		"request_id": req.ID,
		"length":     req.Parameters.Length,
		"arima":      fmt.Sprintf("(%d,%d,%d)", g.config.P, g.config.D, g.config.Q),
	}).Info("Starting ARIMA generation")

	start := time.Now()

	// Parse frequency
	frequency, err := g.parseFrequency(req.Parameters.Frequency)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeValidation, "INVALID_FREQUENCY", "Failed to parse frequency")
	}

	// Generate timestamps
	timestamps := g.generateTimestamps(req.Parameters.StartTime, frequency, req.Parameters.Length)

	// Generate ARIMA values
	values, err := g.generateARIMAValues(ctx, req.Parameters.Length)
	if err != nil {
		return nil, err
	}

	// Create data points
	dataPoints := make([]models.DataPoint, len(timestamps))
	for i, timestamp := range timestamps {
		dataPoints[i] = models.DataPoint{
			Timestamp: timestamp,
			Value:     values[i],
			Quality:   1.0,
		}
	}

	// Create time series
	timeSeries := &models.TimeSeries{
		ID:          fmt.Sprintf("arima-%d", time.Now().UnixNano()),
		Name:        fmt.Sprintf("ARIMA Generated (%d,%d,%d)", g.config.P, g.config.D, g.config.Q),
		Description: fmt.Sprintf("Synthetic data generated using ARIMA(%d,%d,%d) model", g.config.P, g.config.D, g.config.Q),
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
		ID:            req.ID,
		Status:        "completed",
		TimeSeries:    timeSeries,
		Duration:      duration,
		GeneratedAt:   time.Now(),
		GeneratorType: string(g.GetType()),
		Quality:       1.0,
		Metadata: map[string]interface{}{
			"arima_order":     []int{g.config.P, g.config.D, g.config.Q},
			"seasonal_order":  []int{g.config.SeasonalP, g.config.SeasonalD, g.config.SeasonalQ, g.config.SeasonalPeriod},
			"data_points":     len(dataPoints),
			"generation_time": duration.String(),
			"trained":         g.trained,
		},
	}

	g.logger.WithFields(logrus.Fields{
		"request_id":  req.ID,
		"data_points": len(dataPoints),
		"duration":    duration,
	}).Info("Completed ARIMA generation")

	return result, nil
}

// Train trains the ARIMA model with reference data
func (g *ARIMAGenerator) Train(ctx context.Context, data *models.TimeSeries, params models.GenerationParameters) error {
	if data == nil {
		return errors.NewValidationError("INVALID_DATA", "Training data is required")
	}

	if len(data.DataPoints) < 50 {
		return errors.NewValidationError("INSUFFICIENT_DATA", "At least 50 data points required for ARIMA training")
	}

	g.logger.WithFields(logrus.Fields{
		"series_id":   data.ID,
		"data_points": len(data.DataPoints),
		"arima_order": fmt.Sprintf("(%d,%d,%d)", g.config.P, g.config.D, g.config.Q),
	}).Info("Training ARIMA model")

	// Extract values from time series
	values := make([]float64, len(data.DataPoints))
	for i, dp := range data.DataPoints {
		values[i] = dp.Value
	}

	// Fit ARIMA model
	modelParams, err := g.fitARIMA(values)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeProcessing, "FIT_ERROR", "Failed to fit ARIMA model")
	}

	g.modelParams = modelParams
	g.arCoeffs = modelParams.ARCoefficients
	g.maCoeffs = modelParams.MACoefficients
	g.statistics = data.CalculateMetrics()
	g.trained = true

	g.logger.WithFields(logrus.Fields{
		"ar_coeffs": len(g.arCoeffs),
		"ma_coeffs": len(g.maCoeffs),
		"aic":       modelParams.AIC,
		"bic":       modelParams.BIC,
	}).Info("ARIMA model training completed")

	return nil
}

// IsTrainable returns true if the generator requires/supports training
func (g *ARIMAGenerator) IsTrainable() bool {
	return true
}

// GetDefaultParameters returns default parameters for this generator
func (g *ARIMAGenerator) GetDefaultParameters() models.GenerationParameters {
	return models.GenerationParameters{
		Length:    1000,
		Frequency: "1h",
		StartTime: time.Now().Add(-30 * 24 * time.Hour),
		Tags:      make(map[string]string),
		Metadata:  make(map[string]interface{}),
	}
}

// EstimateDuration estimates how long generation will take
func (g *ARIMAGenerator) EstimateDuration(req *models.GenerationRequest) (time.Duration, error) {
	if req == nil {
		return 0, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}

	// ARIMA generation is relatively fast
	// Estimate based on complexity and length
	complexity := float64(g.config.P + g.config.Q + 1)
	if g.config.SeasonalPeriod > 0 {
		complexity += float64(g.config.SeasonalP + g.config.SeasonalQ)
	}

	// Approximately 0.1ms per point per complexity unit
	estimatedMs := float64(req.Parameters.Length) * complexity * 0.1
	return time.Duration(estimatedMs) * time.Millisecond, nil
}

// Cancel cancels an ongoing generation
func (g *ARIMAGenerator) Cancel(ctx context.Context, requestID string) error {
	g.logger.WithFields(logrus.Fields{
		"request_id": requestID,
	}).Info("Cancel requested for ARIMA generation")
	return nil
}

// GetProgress returns the progress of an ongoing generation
func (g *ARIMAGenerator) GetProgress(requestID string) (float64, error) {
	// ARIMA generation is typically fast enough that progress tracking isn't needed
	return 1.0, nil
}

// Close cleans up resources
func (g *ARIMAGenerator) Close() error {
	g.logger.Info("Closing ARIMA generator")
	return nil
}

// generateARIMAValues generates ARIMA time series values
func (g *ARIMAGenerator) generateARIMAValues(ctx context.Context, length int) ([]float64, error) {
	// Initialize arrays
	values := make([]float64, length)
	errors := make([]float64, length+max(g.config.P, g.config.Q))
	
	// Initialize with random errors
	for i := range errors {
		errors[i] = g.randSource.NormFloat64()
	}

	// Use fitted coefficients if trained, otherwise use defaults
	arCoeffs := g.arCoeffs
	maCoeffs := g.maCoeffs
	
	if !g.trained {
		// Generate default coefficients
		arCoeffs = g.generateDefaultARCoefficients()
		maCoeffs = g.generateDefaultMACoefficients()
	}

	// Initialize mean and variance
	mean := 0.0
	variance := 1.0
	if g.trained && g.statistics != nil {
		mean = g.statistics.Mean
		variance = g.statistics.Variance
	}

	// Generate initial values for AR component
	for i := 0; i < g.config.P; i++ {
		values[i] = mean + math.Sqrt(variance)*g.randSource.NormFloat64()
	}

	// Apply differencing if needed
	diffValues := values
	if g.config.D > 0 {
		diffValues = make([]float64, length)
		copy(diffValues, values)
	}

	// Generate ARIMA values
	for i := max(g.config.P, g.config.Q); i < length; i++ {
		value := 0.0

		// Add constant term if included
		if g.config.IncludeConstant {
			value += mean
		}

		// AR component
		for j := 0; j < g.config.P && j < len(arCoeffs) && i-j-1 >= 0; j++ {
			value += arCoeffs[j] * (diffValues[i-j-1] - mean)
		}

		// MA component
		for j := 0; j < g.config.Q && j < len(maCoeffs); j++ {
			value += maCoeffs[j] * errors[i-j-1]
		}

		// Add current error
		currentError := math.Sqrt(variance) * g.randSource.NormFloat64()
		value += currentError
		errors[i] = currentError

		// Add seasonal components if configured
		if g.config.SeasonalPeriod > 0 && i >= g.config.SeasonalPeriod {
			seasonalValue := g.generateSeasonalComponent(i, diffValues, errors)
			value += seasonalValue
		}

		diffValues[i] = value
	}

	// Integrate if differencing was applied
	if g.config.D > 0 {
		values = g.integrate(diffValues, g.config.D)
	} else {
		values = diffValues
	}

	return values, nil
}

// generateDefaultARCoefficients generates default AR coefficients
func (g *ARIMAGenerator) generateDefaultARCoefficients() []float64 {
	coeffs := make([]float64, g.config.P)
	
	// Generate stationary AR coefficients
	// Using declining weights to ensure stationarity
	for i := 0; i < g.config.P; i++ {
		coeffs[i] = 0.7 * math.Pow(0.5, float64(i))
		if i%2 == 1 {
			coeffs[i] *= -1 // Alternate signs for more interesting patterns
		}
	}
	
	return coeffs
}

// generateDefaultMACoefficients generates default MA coefficients
func (g *ARIMAGenerator) generateDefaultMACoefficients() []float64 {
	coeffs := make([]float64, g.config.Q)
	
	// Generate MA coefficients
	for i := 0; i < g.config.Q; i++ {
		coeffs[i] = 0.5 * math.Pow(0.6, float64(i))
		if i%2 == 1 {
			coeffs[i] *= -1
		}
	}
	
	return coeffs
}

// generateSeasonalComponent generates seasonal ARIMA component
func (g *ARIMAGenerator) generateSeasonalComponent(index int, values []float64, errors []float64) float64 {
	seasonalValue := 0.0
	period := g.config.SeasonalPeriod

	// Seasonal AR component
	for i := 1; i <= g.config.SeasonalP; i++ {
		idx := index - i*period
		if idx >= 0 && idx < len(values) {
			seasonalValue += 0.3 * values[idx] / float64(i)
		}
	}

	// Seasonal MA component
	for i := 1; i <= g.config.SeasonalQ; i++ {
		idx := index - i*period
		if idx >= 0 && idx < len(errors) {
			seasonalValue += 0.2 * errors[idx] / float64(i)
		}
	}

	return seasonalValue
}

// fitARIMA fits ARIMA model to data using simplified method
func (g *ARIMAGenerator) fitARIMA(data []float64) (*ARIMAParameters, error) {
	// This is a simplified ARIMA fitting
	// In production, you would use proper maximum likelihood estimation
	
	// Difference the data if needed
	diffData := g.difference(data, g.config.D)
	
	// Calculate autocorrelations for AR coefficients
	arCoeffs := make([]float64, g.config.P)
	if g.config.P > 0 {
		acf := g.calculateACF(diffData, g.config.P)
		// Simple Yule-Walker estimation
		for i := 0; i < g.config.P; i++ {
			arCoeffs[i] = acf[i+1] * math.Pow(0.95, float64(i))
		}
	}
	
	// Estimate MA coefficients
	maCoeffs := make([]float64, g.config.Q)
	if g.config.Q > 0 {
		// Simplified MA estimation
		for i := 0; i < g.config.Q; i++ {
			maCoeffs[i] = 0.3 * math.Pow(0.7, float64(i))
		}
	}
	
	// Calculate variance
	variance := stat.Variance(diffData, nil)
	
	// Calculate information criteria
	n := float64(len(data))
	k := float64(g.config.P + g.config.Q + 1) // number of parameters
	
	// Simplified log-likelihood
	logLikelihood := -0.5 * n * (math.Log(2*math.Pi) + math.Log(variance) + 1)
	
	aic := -2*logLikelihood + 2*k
	bic := -2*logLikelihood + k*math.Log(n)
	
	params := &ARIMAParameters{
		ARCoefficients:   arCoeffs,
		MACoefficients:   maCoeffs,
		Constant:         stat.Mean(diffData, nil),
		Variance:         variance,
		LogLikelihood:    logLikelihood,
		AIC:              aic,
		BIC:              bic,
	}
	
	return params, nil
}

// difference applies differencing to the data
func (g *ARIMAGenerator) difference(data []float64, d int) []float64 {
	if d == 0 {
		return data
	}
	
	result := make([]float64, len(data))
	copy(result, data)
	
	for i := 0; i < d; i++ {
		diff := make([]float64, len(result)-1)
		for j := 1; j < len(result); j++ {
			diff[j-1] = result[j] - result[j-1]
		}
		result = diff
	}
	
	return result
}

// integrate reverses differencing
func (g *ARIMAGenerator) integrate(data []float64, d int) []float64 {
	if d == 0 {
		return data
	}
	
	result := make([]float64, len(data))
	copy(result, data)
	
	// For each level of integration
	for i := 0; i < d; i++ {
		integrated := make([]float64, len(result)+1)
		integrated[0] = 0 // Starting value
		
		// Cumulative sum
		for j := 0; j < len(result); j++ {
			integrated[j+1] = integrated[j] + result[j]
		}
		
		result = integrated[1:] // Skip the initial zero
	}
	
	return result
}

// calculateACF calculates autocorrelation function
func (g *ARIMAGenerator) calculateACF(data []float64, maxLag int) []float64 {
	n := len(data)
	if maxLag > n/4 {
		maxLag = n / 4
	}
	
	acf := make([]float64, maxLag+1)
	mean := stat.Mean(data, nil)
	
	// Calculate variance (lag 0)
	var c0 float64
	for i := 0; i < n; i++ {
		c0 += (data[i] - mean) * (data[i] - mean)
	}
	c0 /= float64(n)
	
	acf[0] = 1.0
	
	// Calculate autocorrelations
	for k := 1; k <= maxLag; k++ {
		var ck float64
		for i := k; i < n; i++ {
			ck += (data[i] - mean) * (data[i-k] - mean)
		}
		ck /= float64(n)
		acf[k] = ck / c0
	}
	
	return acf
}

// Helper functions
func (g *ARIMAGenerator) generateTimestamps(start time.Time, frequency time.Duration, length int) []time.Time {
	timestamps := make([]time.Time, length)
	current := start
	
	for i := 0; i < length; i++ {
		timestamps[i] = current
		current = current.Add(frequency)
	}
	
	return timestamps
}

func (g *ARIMAGenerator) parseFrequency(freq string) (time.Duration, error) {
	duration, err := time.ParseDuration(freq)
	if err != nil {
		return 0, fmt.Errorf("invalid frequency format: %s", freq)
	}
	return duration, nil
}

func getDefaultARIMAConfig() *ARIMAConfig {
	return &ARIMAConfig{
		P:                1,
		D:                1,
		Q:                1,
		SeasonalP:        0,
		SeasonalD:        0,
		SeasonalQ:        0,
		SeasonalPeriod:   0,
		IncludeConstant:  true,
		Seed:             time.Now().UnixNano(),
		OptimizationMethod: "css",
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}