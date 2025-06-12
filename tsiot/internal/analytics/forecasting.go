package analytics

import (
	"fmt"
	"math"
	"time"
)

// ForecasterRegistry manages forecasting algorithms
type ForecasterRegistry struct {
	forecasters map[string]Forecaster
}

// NewForecasterRegistry creates a new forecaster registry
func NewForecasterRegistry() *ForecasterRegistry {
	registry := &ForecasterRegistry{
		forecasters: make(map[string]Forecaster),
	}

	// Register default forecasters
	registry.Register(&SimpleMovingAverageForecaster{})
	registry.Register(&ExponentialSmoothingForecaster{})
	registry.Register(&LinearRegressionForecaster{})
	registry.Register(&SeasonalNaiveForecaster{})
	registry.Register(&HoltWintersForecaster{})
	registry.Register(&ARIMAForecaster{})

	return registry
}

// Register registers a forecaster
func (fr *ForecasterRegistry) Register(forecaster Forecaster) {
	fr.forecasters[forecaster.Name()] = forecaster
}

// Get returns a forecaster by name
func (fr *ForecasterRegistry) Get(name string) (Forecaster, bool) {
	forecaster, exists := fr.forecasters[name]
	return forecaster, exists
}

// GetAll returns all registered forecasters
func (fr *ForecasterRegistry) GetAll() map[string]Forecaster {
	return fr.forecasters
}

// SimpleMovingAverageForecaster implements simple moving average forecasting
type SimpleMovingAverageForecaster struct{}

// Name returns the forecaster name
func (sma *SimpleMovingAverageForecaster) Name() string {
	return "simple_moving_average"
}

// Forecast generates forecasts using simple moving average
func (sma *SimpleMovingAverageForecaster) Forecast(data []float64, horizon int, params AnalysisParameters) ([]ForecastPoint, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("no data provided for forecasting")
	}

	windowSize := params.WindowSize
	if windowSize <= 0 || windowSize > len(data) {
		windowSize = int(math.Min(float64(len(data)), 10))
	}

	// Calculate moving average for the last window
	sum := 0.0
	for i := len(data) - windowSize; i < len(data); i++ {
		sum += data[i]
	}
	avgValue := sum / float64(windowSize)

	// Calculate variance for confidence intervals
	variance := 0.0
	for i := len(data) - windowSize; i < len(data); i++ {
		diff := data[i] - avgValue
		variance += diff * diff
	}
	variance /= float64(windowSize)
	stdDev := math.Sqrt(variance)

	// Generate forecasts
	forecasts := make([]ForecastPoint, horizon)
	baseTime := time.Now()
	confidence := params.Confidence
	if confidence <= 0 {
		confidence = 0.95
	}

	// Z-score for confidence interval
	zScore := 1.96 // 95% confidence interval
	if confidence == 0.90 {
		zScore = 1.645
	} else if confidence == 0.99 {
		zScore = 2.576
	}

	margin := zScore * stdDev

	for i := 0; i < horizon; i++ {
		forecasts[i] = ForecastPoint{
			Timestamp:  baseTime.Add(time.Duration(i) * time.Minute),
			Value:      avgValue,
			Lower:      avgValue - margin,
			Upper:      avgValue + margin,
			Confidence: confidence,
			Method:     sma.Name(),
		}
	}

	return forecasts, nil
}

// ExponentialSmoothingForecaster implements exponential smoothing
type ExponentialSmoothingForecaster struct{}

// Name returns the forecaster name
func (es *ExponentialSmoothingForecaster) Name() string {
	return "exponential_smoothing"
}

// Forecast generates forecasts using exponential smoothing
func (es *ExponentialSmoothingForecaster) Forecast(data []float64, horizon int, params AnalysisParameters) ([]ForecastPoint, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("no data provided for forecasting")
	}

	// Alpha parameter for exponential smoothing
	alpha := 0.3
	if customAlpha, exists := params.CustomParams["alpha"]; exists {
		if alphaFloat, ok := customAlpha.(float64); ok && alphaFloat > 0 && alphaFloat < 1 {
			alpha = alphaFloat
		}
	}

	// Initialize with first value
	smoothedValue := data[0]
	
	// Apply exponential smoothing
	for i := 1; i < len(data); i++ {
		smoothedValue = alpha*data[i] + (1-alpha)*smoothedValue
	}

	// Calculate error for confidence intervals
	errors := make([]float64, len(data)-1)
	tempSmoothed := data[0]
	for i := 1; i < len(data); i++ {
		errors[i-1] = math.Abs(data[i] - tempSmoothed)
		tempSmoothed = alpha*data[i] + (1-alpha)*tempSmoothed
	}

	// Calculate mean absolute error
	mae := 0.0
	for _, err := range errors {
		mae += err
	}
	mae /= float64(len(errors))

	// Generate forecasts
	forecasts := make([]ForecastPoint, horizon)
	baseTime := time.Now()
	confidence := params.Confidence
	if confidence <= 0 {
		confidence = 0.95
	}

	// Confidence interval multiplier
	multiplier := 1.96
	if confidence == 0.90 {
		multiplier = 1.645
	} else if confidence == 0.99 {
		multiplier = 2.576
	}

	for i := 0; i < horizon; i++ {
		margin := multiplier * mae * math.Sqrt(float64(i+1)) // Increasing uncertainty
		
		forecasts[i] = ForecastPoint{
			Timestamp:  baseTime.Add(time.Duration(i) * time.Minute),
			Value:      smoothedValue,
			Lower:      smoothedValue - margin,
			Upper:      smoothedValue + margin,
			Confidence: confidence * math.Exp(-float64(i)*0.1), // Decreasing confidence
			Method:     es.Name(),
		}
	}

	return forecasts, nil
}

// LinearRegressionForecaster implements linear regression forecasting
type LinearRegressionForecaster struct{}

// Name returns the forecaster name
func (lr *LinearRegressionForecaster) Name() string {
	return "linear_regression"
}

// Forecast generates forecasts using linear regression
func (lr *LinearRegressionForecaster) Forecast(data []float64, horizon int, params AnalysisParameters) ([]ForecastPoint, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf("insufficient data for linear regression")
	}

	// Calculate linear regression
	slope, intercept, r2, err := lr.calculateLinearRegression(data)
	if err != nil {
		return nil, err
	}

	// Calculate residual standard error
	rse := lr.calculateResidualStandardError(data, slope, intercept)

	// Generate forecasts
	forecasts := make([]ForecastPoint, horizon)
	baseTime := time.Now()
	confidence := params.Confidence
	if confidence <= 0 {
		confidence = 0.95
	}

	// T-score for confidence interval (approximating with z-score for simplicity)
	tScore := 1.96
	if confidence == 0.90 {
		tScore = 1.645
	} else if confidence == 0.99 {
		tScore = 2.576
	}

	for i := 0; i < horizon; i++ {
		x := float64(len(data) + i)
		predicted := slope*x + intercept
		
		// Standard error increases with distance from data
		standardError := rse * math.Sqrt(1.0+1.0/float64(len(data))+math.Pow(x-float64(len(data))/2, 2)/lr.calculateSumSquaredX(len(data)))
		margin := tScore * standardError

		forecasts[i] = ForecastPoint{
			Timestamp:  baseTime.Add(time.Duration(i) * time.Minute),
			Value:      predicted,
			Lower:      predicted - margin,
			Upper:      predicted + margin,
			Confidence: confidence * r2, // Confidence adjusted by R²
			Method:     lr.Name(),
		}
	}

	return forecasts, nil
}

// calculateLinearRegression calculates linear regression parameters
func (lr *LinearRegressionForecaster) calculateLinearRegression(data []float64) (slope, intercept, r2 float64, err error) {
	n := float64(len(data))
	sumX := n * (n - 1) / 2  // Sum of indices 0,1,2,...,n-1
	sumY := 0.0
	sumXY := 0.0
	sumX2 := n * (n - 1) * (2*n - 1) / 6  // Sum of squares of indices
	sumY2 := 0.0

	for i, y := range data {
		x := float64(i)
		sumY += y
		sumXY += x * y
		sumY2 += y * y
	}

	denominator := n*sumX2 - sumX*sumX
	if denominator == 0 {
		return 0, 0, 0, fmt.Errorf("cannot calculate regression: zero denominator")
	}

	slope = (n*sumXY - sumX*sumY) / denominator
	intercept = (sumY - slope*sumX) / n

	// Calculate R²
	yMean := sumY / n
	ssTotal := sumY2 - n*yMean*yMean
	if ssTotal == 0 {
		r2 = 1.0
	} else {
		ssRes := 0.0
		for i, y := range data {
			predicted := slope*float64(i) + intercept
			ssRes += (y - predicted) * (y - predicted)
		}
		r2 = 1.0 - ssRes/ssTotal
	}

	return slope, intercept, r2, nil
}

// calculateResidualStandardError calculates the residual standard error
func (lr *LinearRegressionForecaster) calculateResidualStandardError(data []float64, slope, intercept float64) float64 {
	sumSquaredResiduals := 0.0
	for i, y := range data {
		predicted := slope*float64(i) + intercept
		residual := y - predicted
		sumSquaredResiduals += residual * residual
	}
	
	degreesOfFreedom := float64(len(data) - 2)
	if degreesOfFreedom <= 0 {
		degreesOfFreedom = 1
	}
	
	return math.Sqrt(sumSquaredResiduals / degreesOfFreedom)
}

// calculateSumSquaredX calculates sum of squared deviations from mean for X values
func (lr *LinearRegressionForecaster) calculateSumSquaredX(n int) float64 {
	mean := float64(n-1) / 2
	sum := 0.0
	for i := 0; i < n; i++ {
		diff := float64(i) - mean
		sum += diff * diff
	}
	return sum
}

// SeasonalNaiveForecaster implements seasonal naive forecasting
type SeasonalNaiveForecaster struct{}

// Name returns the forecaster name
func (sn *SeasonalNaiveForecaster) Name() string {
	return "seasonal_naive"
}

// Forecast generates forecasts using seasonal naive method
func (sn *SeasonalNaiveForecaster) Forecast(data []float64, horizon int, params AnalysisParameters) ([]ForecastPoint, error) {
	seasonality := params.Seasonality
	if seasonality <= 0 {
		seasonality = 24 // Default to daily seasonality
	}

	if len(data) < seasonality {
		return nil, fmt.Errorf("insufficient data for seasonal forecasting: need at least %d points", seasonality)
	}

	// Calculate seasonal pattern from the last complete season
	seasonalPattern := make([]float64, seasonality)
	seasonalCounts := make([]int, seasonality)

	// Aggregate values by seasonal position
	for i, value := range data {
		pos := i % seasonality
		seasonalPattern[pos] += value
		seasonalCounts[pos]++
	}

	// Calculate averages
	for i := range seasonalPattern {
		if seasonalCounts[i] > 0 {
			seasonalPattern[i] /= float64(seasonalCounts[i])
		}
	}

	// Calculate error estimates
	errors := make([]float64, len(data))
	for i, value := range data {
		pos := i % seasonality
		errors[i] = math.Abs(value - seasonalPattern[pos])
	}

	// Calculate mean absolute error
	mae := 0.0
	for _, err := range errors {
		mae += err
	}
	mae /= float64(len(errors))

	// Generate forecasts
	forecasts := make([]ForecastPoint, horizon)
	baseTime := time.Now()
	confidence := params.Confidence
	if confidence <= 0 {
		confidence = 0.95
	}

	multiplier := 1.96
	if confidence == 0.90 {
		multiplier = 1.645
	} else if confidence == 0.99 {
		multiplier = 2.576
	}

	for i := 0; i < horizon; i++ {
		pos := (len(data) + i) % seasonality
		predicted := seasonalPattern[pos]
		margin := multiplier * mae

		forecasts[i] = ForecastPoint{
			Timestamp:  baseTime.Add(time.Duration(i) * time.Minute),
			Value:      predicted,
			Lower:      predicted - margin,
			Upper:      predicted + margin,
			Confidence: confidence,
			Method:     sn.Name(),
		}
	}

	return forecasts, nil
}

// HoltWintersForecaster implements Holt-Winters exponential smoothing
type HoltWintersForecaster struct{}

// Name returns the forecaster name
func (hw *HoltWintersForecaster) Name() string {
	return "holt_winters"
}

// Forecast generates forecasts using Holt-Winters method
func (hw *HoltWintersForecaster) Forecast(data []float64, horizon int, params AnalysisParameters) ([]ForecastPoint, error) {
	seasonality := params.Seasonality
	if seasonality <= 0 {
		seasonality = 24
	}

	if len(data) < seasonality*2 {
		return nil, fmt.Errorf("insufficient data for Holt-Winters: need at least %d points", seasonality*2)
	}

	// Parameters
	alpha := 0.3 // Level smoothing
	beta := 0.1  // Trend smoothing
	gamma := 0.2 // Seasonal smoothing

	// Extract parameters from custom params if provided
	if customAlpha, exists := params.CustomParams["alpha"]; exists {
		if alphaFloat, ok := customAlpha.(float64); ok && alphaFloat > 0 && alphaFloat < 1 {
			alpha = alphaFloat
		}
	}

	// Initialize components
	level, trend, seasonal := hw.initializeComponents(data, seasonality)

	// Apply Holt-Winters smoothing
	errors := make([]float64, 0)
	for t := seasonality; t < len(data); t++ {
		// Prediction error
		predicted := level + trend + seasonal[t%seasonality]
		errors = append(errors, math.Abs(data[t]-predicted))

		// Update components
		prevLevel := level
		level = alpha*(data[t]-seasonal[t%seasonality]) + (1-alpha)*(level+trend)
		trend = beta*(level-prevLevel) + (1-beta)*trend
		seasonal[t%seasonality] = gamma*(data[t]-level) + (1-gamma)*seasonal[t%seasonality]
	}

	// Calculate mean absolute error
	mae := 0.0
	for _, err := range errors {
		mae += err
	}
	if len(errors) > 0 {
		mae /= float64(len(errors))
	}

	// Generate forecasts
	forecasts := make([]ForecastPoint, horizon)
	baseTime := time.Now()
	confidence := params.Confidence
	if confidence <= 0 {
		confidence = 0.95
	}

	multiplier := 1.96
	if confidence == 0.90 {
		multiplier = 1.645
	} else if confidence == 0.99 {
		multiplier = 2.576
	}

	for i := 0; i < horizon; i++ {
		predicted := level + float64(i+1)*trend + seasonal[(len(data)+i)%seasonality]
		margin := multiplier * mae * math.Sqrt(float64(i+1))

		forecasts[i] = ForecastPoint{
			Timestamp:  baseTime.Add(time.Duration(i) * time.Minute),
			Value:      predicted,
			Lower:      predicted - margin,
			Upper:      predicted + margin,
			Confidence: confidence * math.Exp(-float64(i)*0.05),
			Method:     hw.Name(),
		}
	}

	return forecasts, nil
}

// initializeComponents initializes Holt-Winters components
func (hw *HoltWintersForecaster) initializeComponents(data []float64, seasonality int) (level, trend float64, seasonal []float64) {
	// Initialize level as average of first season
	level = 0.0
	for i := 0; i < seasonality; i++ {
		level += data[i]
	}
	level /= float64(seasonality)

	// Initialize trend as average difference between first two seasons
	trend = 0.0
	if len(data) >= seasonality*2 {
		firstSeason := 0.0
		secondSeason := 0.0
		for i := 0; i < seasonality; i++ {
			firstSeason += data[i]
			secondSeason += data[i+seasonality]
		}
		trend = (secondSeason - firstSeason) / float64(seasonality*seasonality)
	}

	// Initialize seasonal components
	seasonal = make([]float64, seasonality)
	for i := 0; i < seasonality; i++ {
		if len(data) >= seasonality*2 {
			// Average seasonal effect across available seasons
			sum := 0.0
			count := 0
			for j := i; j < len(data); j += seasonality {
				seasonIndex := j / seasonality
				detrended := data[j] - (level + trend*float64(seasonIndex))
				sum += detrended
				count++
			}
			if count > 0 {
				seasonal[i] = sum / float64(count)
			}
		} else {
			seasonal[i] = data[i] - level
		}
	}

	return level, trend, seasonal
}

// ARIMAForecaster implements ARIMA forecasting (simplified)
type ARIMAForecaster struct{}

// Name returns the forecaster name
func (arima *ARIMAForecaster) Name() string {
	return "arima"
}

// Forecast generates forecasts using simplified ARIMA
func (arima *ARIMAForecaster) Forecast(data []float64, horizon int, params AnalysisParameters) ([]ForecastPoint, error) {
	if len(data) < 10 {
		return nil, fmt.Errorf("insufficient data for ARIMA forecasting")
	}

	// Simplified ARIMA(1,1,1) implementation
	// First, difference the series to make it stationary
	diffData := make([]float64, len(data)-1)
	for i := 1; i < len(data); i++ {
		diffData[i-1] = data[i] - data[i-1]
	}

	// Fit AR(1) model to differenced data
	phi := arima.estimateAR1(diffData)
	
	// Estimate MA(1) parameter (simplified)
	theta := 0.3 // Simplified assumption

	// Calculate residuals and their variance
	residuals := arima.calculateResiduals(diffData, phi)
	residualVar := arima.calculateVariance(residuals)

	// Generate forecasts
	forecasts := make([]ForecastPoint, horizon)
	baseTime := time.Now()
	confidence := params.Confidence
	if confidence <= 0 {
		confidence = 0.95
	}

	lastValue := data[len(data)-1]
	lastDiff := diffData[len(diffData)-1]
	lastResidual := 0.0 // Simplified assumption

	multiplier := 1.96
	if confidence == 0.90 {
		multiplier = 1.645
	} else if confidence == 0.99 {
		multiplier = 2.576
	}

	for i := 0; i < horizon; i++ {
		// ARIMA(1,1,1) forecast
		var diffForecast float64
		if i == 0 {
			diffForecast = phi*lastDiff + theta*lastResidual
		} else {
			diffForecast = phi * diffForecast // AR component only for h > 1
		}

		forecast := lastValue + diffForecast
		lastValue = forecast

		// Forecast variance increases with horizon
		forecastVar := residualVar * float64(i+1)
		margin := multiplier * math.Sqrt(forecastVar)

		forecasts[i] = ForecastPoint{
			Timestamp:  baseTime.Add(time.Duration(i) * time.Minute),
			Value:      forecast,
			Lower:      forecast - margin,
			Upper:      forecast + margin,
			Confidence: confidence * math.Exp(-float64(i)*0.1),
			Method:     arima.Name(),
		}
	}

	return forecasts, nil
}

// estimateAR1 estimates AR(1) parameter using least squares
func (arima *ARIMAForecaster) estimateAR1(data []float64) float64 {
	if len(data) < 2 {
		return 0.0
	}

	// Calculate autocorrelation at lag 1
	n := len(data) - 1
	meanX := 0.0
	meanY := 0.0

	for i := 0; i < n; i++ {
		meanX += data[i]
		meanY += data[i+1]
	}
	meanX /= float64(n)
	meanY /= float64(n)

	numerator := 0.0
	denominatorX := 0.0

	for i := 0; i < n; i++ {
		diffX := data[i] - meanX
		diffY := data[i+1] - meanY
		numerator += diffX * diffY
		denominatorX += diffX * diffX
	}

	if denominatorX == 0 {
		return 0.0
	}

	return numerator / denominatorX
}

// calculateResiduals calculates residuals for AR(1) model
func (arima *ARIMAForecaster) calculateResiduals(data []float64, phi float64) []float64 {
	if len(data) < 2 {
		return []float64{}
	}

	residuals := make([]float64, len(data)-1)
	for i := 1; i < len(data); i++ {
		predicted := phi * data[i-1]
		residuals[i-1] = data[i] - predicted
	}

	return residuals
}

// calculateVariance calculates sample variance
func (arima *ARIMAForecaster) calculateVariance(data []float64) float64 {
	if len(data) <= 1 {
		return 1.0
	}

	mean := 0.0
	for _, v := range data {
		mean += v
	}
	mean /= float64(len(data))

	variance := 0.0
	for _, v := range data {
		diff := v - mean
		variance += diff * diff
	}

	return variance / float64(len(data)-1)
}