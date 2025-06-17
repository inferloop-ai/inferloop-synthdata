package arima

import (
	"context"
	"fmt"
	"math"
	"time"

	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/models"
)

// ForecastingEngine provides ARIMA-based forecasting capabilities
type ForecastingEngine struct {
	generator *ARIMAGenerator
}

// NewForecastingEngine creates a new forecasting engine
func NewForecastingEngine(generator *ARIMAGenerator) *ForecastingEngine {
	return &ForecastingEngine{
		generator: generator,
	}
}

// ForecastRequest contains parameters for forecasting
type ForecastRequest struct {
	ID               string                 `json:"id"`
	HistoricalData   *models.TimeSeries     `json:"historical_data"`
	ForecastHorizon  int                    `json:"forecast_horizon"`  // Number of steps to forecast
	ConfidenceLevel  float64                `json:"confidence_level"`  // For prediction intervals (e.g., 0.95)
	IncludeResiduals bool                   `json:"include_residuals"` // Include residual analysis
	Metadata         map[string]interface{} `json:"metadata,omitempty"`
}

// ForecastResult contains the forecasting results
type ForecastResult struct {
	ID                string                 `json:"id"`
	ForecastedSeries  *models.TimeSeries     `json:"forecasted_series"`
	PredictionIntervals *PredictionIntervals `json:"prediction_intervals,omitempty"`
	ModelDiagnostics  *ModelDiagnostics      `json:"model_diagnostics,omitempty"`
	Quality           float64                `json:"quality"`
	GeneratedAt       time.Time              `json:"generated_at"`
	Metadata          map[string]interface{} `json:"metadata,omitempty"`
}

// PredictionIntervals contains confidence intervals for forecasts
type PredictionIntervals struct {
	ConfidenceLevel float64              `json:"confidence_level"`
	LowerBounds     []float64            `json:"lower_bounds"`
	UpperBounds     []float64            `json:"upper_bounds"`
	StandardErrors  []float64            `json:"standard_errors"`
}

// ModelDiagnostics contains model fit diagnostics
type ModelDiagnostics struct {
	FittedValues        []float64 `json:"fitted_values"`
	Residuals           []float64 `json:"residuals"`
	StandardizedResiduals []float64 `json:"standardized_residuals"`
	ResidualACF         []float64 `json:"residual_acf"`
	LjungBoxTest        *LjungBoxTest `json:"ljung_box_test,omitempty"`
	AugmentedDickeyFuller *ADFTest   `json:"adf_test,omitempty"`
	ModelFit            *ModelFitStats `json:"model_fit"`
}

// LjungBoxTest results for residual autocorrelation
type LjungBoxTest struct {
	Statistic   float64 `json:"statistic"`
	PValue      float64 `json:"p_value"`
	DegreesOfFreedom int `json:"degrees_of_freedom"`
	IsSignificant bool   `json:"is_significant"`
}

// ADFTest results for stationarity
type ADFTest struct {
	Statistic     float64 `json:"statistic"`
	PValue        float64 `json:"p_value"`
	CriticalValues map[string]float64 `json:"critical_values"`
	IsStationary  bool    `json:"is_stationary"`
}

// ModelFitStats contains model fit statistics
type ModelFitStats struct {
	AIC              float64 `json:"aic"`
	BIC              float64 `json:"bic"`
	LogLikelihood    float64 `json:"log_likelihood"`
	MSE              float64 `json:"mse"`
	MAE              float64 `json:"mae"`
	MAPE             float64 `json:"mape"`
	RMSE             float64 `json:"rmse"`
	RSquared         float64 `json:"r_squared"`
}

// Forecast performs ARIMA forecasting
func (fe *ForecastingEngine) Forecast(ctx context.Context, req *ForecastRequest) (*ForecastResult, error) {
	if req == nil {
		return nil, errors.NewValidationError("INVALID_REQUEST", "Forecast request is required")
	}

	if req.HistoricalData == nil || len(req.HistoricalData.DataPoints) == 0 {
		return nil, errors.NewValidationError("INVALID_DATA", "Historical data is required for forecasting")
	}

	if req.ForecastHorizon <= 0 {
		return nil, errors.NewValidationError("INVALID_HORIZON", "Forecast horizon must be positive")
	}

	fe.generator.logger.WithFields(map[string]interface{}{
		"request_id":       req.ID,
		"historical_points": len(req.HistoricalData.DataPoints),
		"forecast_horizon": req.ForecastHorizon,
	}).Info("Starting ARIMA forecasting")

	start := time.Now()

	// Train the model on historical data
	params := fe.generator.GetDefaultParameters()
	err := fe.generator.Train(ctx, req.HistoricalData, params)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeProcessing, "TRAINING_ERROR", "Failed to train ARIMA model")
	}

	// Extract historical values
	historicalValues := make([]float64, len(req.HistoricalData.DataPoints))
	for i, dp := range req.HistoricalData.DataPoints {
		historicalValues[i] = dp.Value
	}

	// Generate forecasts
	forecastValues, standardErrors, err := fe.generateForecasts(historicalValues, req.ForecastHorizon)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeProcessing, "FORECAST_ERROR", "Failed to generate forecasts")
	}

	// Generate forecast timestamps
	lastTimestamp := req.HistoricalData.DataPoints[len(req.HistoricalData.DataPoints)-1].Timestamp
	frequency, err := fe.parseFrequency(req.HistoricalData.Frequency)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeValidation, "INVALID_FREQUENCY", "Failed to parse frequency")
	}

	forecastTimestamps := fe.generateForecastTimestamps(lastTimestamp, frequency, req.ForecastHorizon)

	// Create forecast data points
	forecastDataPoints := make([]models.DataPoint, req.ForecastHorizon)
	for i := 0; i < req.ForecastHorizon; i++ {
		forecastDataPoints[i] = models.DataPoint{
			Timestamp: forecastTimestamps[i],
			Value:     forecastValues[i],
			Quality:   0.9, // Slightly lower quality for forecasts
		}
	}

	// Create forecasted time series
	forecastedSeries := &models.TimeSeries{
		ID:          fmt.Sprintf("forecast-%s-%d", req.ID, time.Now().UnixNano()),
		Name:        fmt.Sprintf("ARIMA Forecast - %s", req.HistoricalData.Name),
		Description: fmt.Sprintf("ARIMA forecast with %d steps ahead", req.ForecastHorizon),
		Tags:        req.HistoricalData.Tags,
		Metadata:    req.Metadata,
		DataPoints:  forecastDataPoints,
		StartTime:   forecastTimestamps[0],
		EndTime:     forecastTimestamps[len(forecastTimestamps)-1],
		Frequency:   req.HistoricalData.Frequency,
		SensorType:  req.HistoricalData.SensorType,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	// Calculate prediction intervals if requested
	var predictionIntervals *PredictionIntervals
	if req.ConfidenceLevel > 0 {
		predictionIntervals = fe.calculatePredictionIntervals(
			forecastValues, standardErrors, req.ConfidenceLevel)
	}

	// Generate model diagnostics if requested
	var modelDiagnostics *ModelDiagnostics
	if req.IncludeResiduals {
		modelDiagnostics, err = fe.generateModelDiagnostics(historicalValues)
		if err != nil {
			fe.generator.logger.WithError(err).Warn("Failed to generate model diagnostics")
		}
	}

	// Calculate forecast quality score
	quality := fe.calculateForecastQuality(historicalValues, standardErrors)

	result := &ForecastResult{
		ID:                  req.ID,
		ForecastedSeries:    forecastedSeries,
		PredictionIntervals: predictionIntervals,
		ModelDiagnostics:    modelDiagnostics,
		Quality:             quality,
		GeneratedAt:         time.Now(),
		Metadata: map[string]interface{}{
			"forecast_horizon":   req.ForecastHorizon,
			"arima_order":       []int{fe.generator.config.P, fe.generator.config.D, fe.generator.config.Q},
			"seasonal_order":    []int{fe.generator.config.SeasonalP, fe.generator.config.SeasonalD, fe.generator.config.SeasonalQ, fe.generator.config.SeasonalPeriod},
			"historical_points": len(req.HistoricalData.DataPoints),
			"forecast_time":     time.Since(start).String(),
		},
	}

	fe.generator.logger.WithFields(map[string]interface{}{
		"request_id":        req.ID,
		"forecast_horizon":  req.ForecastHorizon,
		"quality":          quality,
		"duration":         time.Since(start),
	}).Info("Completed ARIMA forecasting")

	return result, nil
}

// generateForecasts generates point forecasts and standard errors
func (fe *ForecastingEngine) generateForecasts(historicalData []float64, horizon int) ([]float64, []float64, error) {
	forecasts := make([]float64, horizon)
	standardErrors := make([]float64, horizon)
	
	// Get model parameters
	if !fe.generator.trained || fe.generator.modelParams == nil {
		return nil, nil, errors.NewProcessingError("MODEL_NOT_TRAINED", "Model must be trained before forecasting")
	}
	
	params := fe.generator.modelParams
	
	// Difference the historical data if needed
	diffData := fe.generator.difference(historicalData, fe.generator.config.D)
	n := len(diffData)
	
	// Initialize forecast arrays
	extendedData := make([]float64, len(diffData)+horizon)
	copy(extendedData, diffData)
	
	extendedErrors := make([]float64, len(diffData)+horizon)
	// Initialize with zero errors for forecast period
	
	// Generate forecasts iteratively
	for h := 0; h < horizon; h++ {
		forecast := 0.0
		
		// Add constant term
		if fe.generator.config.IncludeConstant {
			forecast += params.Constant
		}
		
		// AR component
		for i := 0; i < fe.generator.config.P && i < len(params.ARCoefficients); i++ {
			idx := n + h - i - 1
			if idx >= 0 && idx < len(extendedData) {
				forecast += params.ARCoefficients[i] * (extendedData[idx] - params.Constant)
			}
		}
		
		// MA component (only uses known errors)
		for i := 0; i < fe.generator.config.Q && i < len(params.MACoefficients); i++ {
			idx := n + h - i - 1
			if idx >= 0 && idx < n { // Only use historical errors
				forecast += params.MACoefficients[i] * extendedErrors[idx]
			}
		}
		
		extendedData[n+h] = forecast
		forecasts[h] = forecast
		
		// Calculate forecast standard error
		// This is a simplified calculation
		standardErrors[h] = math.Sqrt(params.Variance * (1.0 + float64(h)*0.1))
	}
	
	// Integrate forecasts if differencing was applied
	if fe.generator.config.D > 0 {
		// Get the last D values from original data for integration
		lastValues := historicalData[len(historicalData)-fe.generator.config.D:]
		integratedForecasts := fe.integrateForecasts(forecasts, lastValues, fe.generator.config.D)
		forecasts = integratedForecasts
	}
	
	return forecasts, standardErrors, nil
}

// integrateForecasts applies integration to differenced forecasts
func (fe *ForecastingEngine) integrateForecasts(forecasts []float64, lastValues []float64, d int) []float64 {
	result := make([]float64, len(forecasts))
	
	// Create extended series with last values + forecasts
	extended := make([]float64, len(lastValues)+len(forecasts))
	copy(extended, lastValues)
	copy(extended[len(lastValues):], forecasts)
	
	// Apply integration d times
	for i := 0; i < d; i++ {
		integrated := make([]float64, len(extended))
		integrated[0] = extended[0]
		
		for j := 1; j < len(extended); j++ {
			integrated[j] = integrated[j-1] + extended[j]
		}
		
		extended = integrated
	}
	
	// Return only the forecast portion
	copy(result, extended[len(lastValues):])
	return result
}

// calculatePredictionIntervals calculates confidence intervals for forecasts
func (fe *ForecastingEngine) calculatePredictionIntervals(forecasts, standardErrors []float64, confidenceLevel float64) *PredictionIntervals {
	alpha := 1.0 - confidenceLevel
	// Using normal approximation (z-score)
	zScore := fe.getZScore(alpha / 2.0)
	
	lowerBounds := make([]float64, len(forecasts))
	upperBounds := make([]float64, len(forecasts))
	
	for i := 0; i < len(forecasts); i++ {
		margin := zScore * standardErrors[i]
		lowerBounds[i] = forecasts[i] - margin
		upperBounds[i] = forecasts[i] + margin
	}
	
	return &PredictionIntervals{
		ConfidenceLevel: confidenceLevel,
		LowerBounds:     lowerBounds,
		UpperBounds:     upperBounds,
		StandardErrors:  standardErrors,
	}
}

// generateModelDiagnostics generates comprehensive model diagnostics
func (fe *ForecastingEngine) generateModelDiagnostics(historicalData []float64) (*ModelDiagnostics, error) {
	if !fe.generator.trained || fe.generator.modelParams == nil {
		return nil, errors.NewProcessingError("MODEL_NOT_TRAINED", "Model must be trained for diagnostics")
	}
	
	// Calculate fitted values and residuals
	fittedValues, err := fe.calculateFittedValues(historicalData)
	if err != nil {
		return nil, err
	}
	
	// Calculate residuals
	residuals := make([]float64, len(historicalData))
	for i := 0; i < len(historicalData); i++ {
		if i < len(fittedValues) {
			residuals[i] = historicalData[i] - fittedValues[i]
		}
	}
	
	// Calculate standardized residuals
	residualVariance := fe.calculateVariance(residuals)
	standardizedResiduals := make([]float64, len(residuals))
	for i := 0; i < len(residuals); i++ {
		if residualVariance > 0 {
			standardizedResiduals[i] = residuals[i] / math.Sqrt(residualVariance)
		}
	}
	
	// Calculate residual autocorrelation function
	residualACF := fe.generator.calculateACF(residuals, min(20, len(residuals)/4))
	
	// Ljung-Box test for residual autocorrelation
	ljungBoxTest := fe.ljungBoxTest(residuals, 10)
	
	// Model fit statistics
	modelFit := fe.calculateModelFitStats(historicalData, fittedValues, residuals)
	
	return &ModelDiagnostics{
		FittedValues:          fittedValues,
		Residuals:            residuals,
		StandardizedResiduals: standardizedResiduals,
		ResidualACF:          residualACF,
		LjungBoxTest:         ljungBoxTest,
		ModelFit:             modelFit,
	}, nil
}

// calculateFittedValues calculates in-sample fitted values
func (fe *ForecastingEngine) calculateFittedValues(data []float64) ([]float64, error) {
	params := fe.generator.modelParams
	config := fe.generator.config
	
	// Difference the data if needed
	diffData := fe.generator.difference(data, config.D)
	fittedDiff := make([]float64, len(diffData))
	
	// Initialize with mean for first few values
	mean := params.Constant
	for i := 0; i < max(config.P, config.Q) && i < len(fittedDiff); i++ {
		fittedDiff[i] = mean
	}
	
	// Calculate fitted values for the rest
	errors := make([]float64, len(diffData))
	for i := max(config.P, config.Q); i < len(diffData); i++ {
		fitted := mean
		
		// AR component
		for j := 0; j < config.P && j < len(params.ARCoefficients); j++ {
			if i-j-1 >= 0 {
				fitted += params.ARCoefficients[j] * (diffData[i-j-1] - mean)
			}
		}
		
		// MA component - use estimated errors
		for j := 0; j < config.Q && j < len(params.MACoefficients); j++ {
			if i-j-1 >= 0 {
				fitted += params.MACoefficients[j] * errors[i-j-1]
			}
		}
		
		fittedDiff[i] = fitted
		
		// Calculate error for MA component
		if i < len(diffData) {
			errors[i] = diffData[i] - fitted
		}
	}
	
	// Integrate back if differencing was applied
	if config.D > 0 {
		return fe.integrateForecasts(fittedDiff, data[:config.D], config.D), nil
	}
	
	return fittedDiff, nil
}

// Helper functions
func (fe *ForecastingEngine) calculateVariance(data []float64) float64 {
	if len(data) <= 1 {
		return 0
	}
	
	mean := 0.0
	for _, v := range data {
		mean += v
	}
	mean /= float64(len(data))
	
	variance := 0.0
	for _, v := range data {
		variance += (v - mean) * (v - mean)
	}
	
	return variance / float64(len(data)-1)
}

func (fe *ForecastingEngine) ljungBoxTest(residuals []float64, lags int) *LjungBoxTest {
	n := float64(len(residuals))
	acf := fe.generator.calculateACF(residuals, lags)
	
	// Calculate Ljung-Box statistic
	statistic := 0.0
	for k := 1; k <= lags; k++ {
		statistic += acf[k] * acf[k] / (n - float64(k))
	}
	statistic *= n * (n + 2)
	
	// Simplified p-value calculation (would use chi-square distribution in practice)
	pValue := math.Exp(-statistic / 2.0) // Approximation
	
	return &LjungBoxTest{
		Statistic:        statistic,
		PValue:           pValue,
		DegreesOfFreedom: lags,
		IsSignificant:    pValue < 0.05,
	}
}

func (fe *ForecastingEngine) calculateModelFitStats(actual, fitted, residuals []float64) *ModelFitStats {
	n := len(actual)
	if n == 0 {
		return &ModelFitStats{}
	}
	
	// Calculate means
	actualMean := 0.0
	for _, v := range actual {
		actualMean += v
	}
	actualMean /= float64(n)
	
	// Calculate MSE, MAE, MAPE
	var mse, mae, mape, tss, rss float64
	
	for i := 0; i < n; i++ {
		if i < len(fitted) && i < len(residuals) {
			// MSE and MAE
			absError := math.Abs(residuals[i])
			mae += absError
			mse += residuals[i] * residuals[i]
			
			// MAPE (avoid division by zero)
			if math.Abs(actual[i]) > 1e-10 {
				mape += absError / math.Abs(actual[i])
			}
			
			// R-squared components
			rss += residuals[i] * residuals[i]
			tss += (actual[i] - actualMean) * (actual[i] - actualMean)
		}
	}
	
	mse /= float64(n)
	mae /= float64(n)
	mape /= float64(n)
	
	// R-squared
	rSquared := 0.0
	if tss > 0 {
		rSquared = 1.0 - rss/tss
	}
	
	return &ModelFitStats{
		AIC:           fe.generator.modelParams.AIC,
		BIC:           fe.generator.modelParams.BIC,
		LogLikelihood: fe.generator.modelParams.LogLikelihood,
		MSE:           mse,
		MAE:           mae,
		MAPE:          mape * 100, // Convert to percentage
		RMSE:          math.Sqrt(mse),
		RSquared:      rSquared,
	}
}

func (fe *ForecastingEngine) calculateForecastQuality(historicalData, standardErrors []float64) float64 {
	// Simple quality metric based on forecast uncertainty
	avgStdError := 0.0
	for _, se := range standardErrors {
		avgStdError += se
	}
	if len(standardErrors) > 0 {
		avgStdError /= float64(len(standardErrors))
	}
	
	// Calculate historical data range for normalization
	var minVal, maxVal float64
	if len(historicalData) > 0 {
		minVal, maxVal = historicalData[0], historicalData[0]
		for _, v := range historicalData[1:] {
			if v < minVal {
				minVal = v
			}
			if v > maxVal {
				maxVal = v
			}
		}
	}
	
	dataRange := maxVal - minVal
	if dataRange <= 0 {
		return 0.5 // Default quality for constant data
	}
	
	// Quality inversely related to relative forecast uncertainty
	relativeUncertainty := avgStdError / dataRange
	quality := math.Max(0.1, 1.0-relativeUncertainty)
	return math.Min(1.0, quality)
}

func (fe *ForecastingEngine) generateForecastTimestamps(lastTimestamp time.Time, frequency time.Duration, horizon int) []time.Time {
	timestamps := make([]time.Time, horizon)
	current := lastTimestamp.Add(frequency)
	
	for i := 0; i < horizon; i++ {
		timestamps[i] = current
		current = current.Add(frequency)
	}
	
	return timestamps
}

func (fe *ForecastingEngine) parseFrequency(freq string) (time.Duration, error) {
	duration, err := time.ParseDuration(freq)
	if err != nil {
		return 0, fmt.Errorf("invalid frequency format: %s", freq)
	}
	return duration, nil
}

func (fe *ForecastingEngine) getZScore(alpha float64) float64 {
	// Simplified z-score calculation for common confidence levels
	switch {
	case alpha <= 0.001:
		return 3.291 // 99.9%
	case alpha <= 0.005:
		return 2.807 // 99.5%
	case alpha <= 0.01:
		return 2.576 // 99%
	case alpha <= 0.025:
		return 1.96  // 95%
	case alpha <= 0.05:
		return 1.645 // 90%
	case alpha <= 0.1:
		return 1.282 // 80%
	default:
		return 1.96  // Default to 95%
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}