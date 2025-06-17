package metrics

import (
	"errors"
	"math"

	"github.com/inferloop/tsiot/pkg/models"
)

// AutocorrelationAnalysis contains autocorrelation function analysis results
type AutocorrelationAnalysis struct {
	ACF                    []float64                `json:"acf"`                      // Autocorrelation function values
	PACF                   []float64                `json:"pacf"`                     // Partial autocorrelation function values
	Lags                   []int                    `json:"lags"`                     // Lag values
	ConfidenceBounds       ConfidenceBounds         `json:"confidence_bounds"`        // Confidence intervals
	SignificantLags        []int                    `json:"significant_lags"`         // Lags with significant autocorrelation
	DecayRate              float64                  `json:"decay_rate"`               // Rate of ACF decay
	EffectiveLength        int                      `json:"effective_length"`         // Effective sample size
	StationarityIndicators StationarityIndicators   `json:"stationarity_indicators"`  // Stationarity test results
}

// ConfidenceBounds contains confidence interval information
type ConfidenceBounds struct {
	Alpha       float64   `json:"alpha"`        // Significance level
	LowerBounds []float64 `json:"lower_bounds"` // Lower confidence bounds
	UpperBounds []float64 `json:"upper_bounds"` // Upper confidence bounds
}

// StationarityIndicators contains indicators of time series stationarity
type StationarityIndicators struct {
	MeanStationarity     bool    `json:"mean_stationarity"`     // Whether mean appears stationary
	VarianceStationarity bool    `json:"variance_stationarity"` // Whether variance appears stationary
	TrendPresent         bool    `json:"trend_present"`         // Whether trend is detected
	SeasonalityPresent   bool    `json:"seasonality_present"`   // Whether seasonality is detected
	LjungBoxPValue       float64 `json:"ljung_box_p_value"`     // Ljung-Box test p-value
}

// AutocorrelationComparison contains comparison results between two time series
type AutocorrelationComparison struct {
	OriginalACF    []float64 `json:"original_acf"`    // Original series ACF
	SyntheticACF   []float64 `json:"synthetic_acf"`   // Synthetic series ACF
	Similarity     float64   `json:"similarity"`      // Overall similarity score (0-1)
	LagSimilarity  []float64 `json:"lag_similarity"`  // Similarity at each lag
	MaxDifference  float64   `json:"max_difference"`  // Maximum difference between ACFs
	MeanSquaredError float64 `json:"mean_squared_error"` // MSE between ACFs
	CorrelationScore float64 `json:"correlation_score"`  // Correlation between ACFs
}

// CalculateAutocorrelation computes the autocorrelation function for a time series
func CalculateAutocorrelation(data []float64, maxLag int) (*AutocorrelationAnalysis, error) {
	if len(data) < 10 {
		return nil, errors.New("autocorrelation analysis requires at least 10 observations")
	}

	if maxLag <= 0 || maxLag >= len(data) {
		maxLag = min(len(data)/4, 20) // Default to reasonable maximum
	}

	n := len(data)
	
	// Calculate sample mean
	mean := calculateMean(data)
	
	// Calculate autocovariances
	autocovar := make([]float64, maxLag+1)
	
	// Lag 0 (variance)
	for i := 0; i < n; i++ {
		diff := data[i] - mean
		autocovar[0] += diff * diff
	}
	autocovar[0] /= float64(n)

	// Lags 1 to maxLag
	for lag := 1; lag <= maxLag; lag++ {
		for i := lag; i < n; i++ {
			autocovar[lag] += (data[i] - mean) * (data[i-lag] - mean)
		}
		autocovar[lag] /= float64(n)
	}

	// Convert to autocorrelations
	acf := make([]float64, maxLag)
	lags := make([]int, maxLag)
	for lag := 1; lag <= maxLag; lag++ {
		if autocovar[0] != 0 {
			acf[lag-1] = autocovar[lag] / autocovar[0]
		}
		lags[lag-1] = lag
	}

	// Calculate partial autocorrelations using Yule-Walker equations
	pacf, err := calculatePACF(acf, maxLag)
	if err != nil {
		pacf = make([]float64, maxLag) // Fill with zeros on error
	}

	// Calculate confidence bounds
	confidenceBounds := calculateConfidenceBounds(n, maxLag, 0.05)

	// Find significant lags
	significantLags := findSignificantLags(acf, confidenceBounds)

	// Calculate decay rate
	decayRate := calculateDecayRate(acf)

	// Assess stationarity
	stationarityIndicators := assessStationarity(data, acf)

	analysis := &AutocorrelationAnalysis{
		ACF:                    acf,
		PACF:                   pacf,
		Lags:                   lags,
		ConfidenceBounds:       confidenceBounds,
		SignificantLags:        significantLags,
		DecayRate:              decayRate,
		EffectiveLength:        n,
		StationarityIndicators: stationarityIndicators,
	}

	return analysis, nil
}

// CalculatePACF computes partial autocorrelation function using Yule-Walker equations
func calculatePACF(acf []float64, maxLag int) ([]float64, error) {
	pacf := make([]float64, maxLag)
	
	if len(acf) == 0 {
		return pacf, nil
	}

	// PACF at lag 1 is just ACF at lag 1
	if maxLag > 0 && len(acf) > 0 {
		pacf[0] = acf[0]
	}

	// For higher lags, solve Yule-Walker equations
	for k := 2; k <= maxLag && k <= len(acf); k++ {
		// Create correlation matrix R
		R := make([][]float64, k-1)
		for i := 0; i < k-1; i++ {
			R[i] = make([]float64, k-1)
			for j := 0; j < k-1; j++ {
				lag := int(math.Abs(float64(i - j)))
				if lag == 0 {
					R[i][j] = 1.0
				} else if lag <= len(acf) {
					R[i][j] = acf[lag-1]
				}
			}
		}

		// Create correlation vector r
		r := make([]float64, k-1)
		for i := 0; i < k-1; i++ {
			lag := k - 1 - i
			if lag <= len(acf) {
				r[i] = acf[lag-1]
			}
		}

		// Solve R * phi = r using simple elimination (for small systems)
		phi, err := solveLinearSystem(R, r)
		if err != nil {
			continue // Skip this lag on error
		}

		// PACF at lag k is the last coefficient
		if len(phi) > 0 {
			pacf[k-1] = phi[len(phi)-1]
		}
	}

	return pacf, nil
}

// CompareAutocorrelations compares autocorrelation functions of two time series
func CompareAutocorrelations(original, synthetic []float64, maxLag int) (*AutocorrelationComparison, error) {
	if len(original) < 10 || len(synthetic) < 10 {
		return nil, errors.New("autocorrelation comparison requires at least 10 observations in each series")
	}

	// Calculate autocorrelations for both series
	originalAnalysis, err := CalculateAutocorrelation(original, maxLag)
	if err != nil {
		return nil, err
	}

	syntheticAnalysis, err := CalculateAutocorrelation(synthetic, maxLag)
	if err != nil {
		return nil, err
	}

	// Ensure both ACFs have the same length
	minLen := min(len(originalAnalysis.ACF), len(syntheticAnalysis.ACF))
	originalACF := originalAnalysis.ACF[:minLen]
	syntheticACF := syntheticAnalysis.ACF[:minLen]

	// Calculate lag-wise similarities
	lagSimilarity := make([]float64, minLen)
	var sumSquaredDiff float64
	
	for i := 0; i < minLen; i++ {
		diff := originalACF[i] - syntheticACF[i]
		sumSquaredDiff += diff * diff
		
		// Similarity at this lag (1 - normalized absolute difference)
		maxVal := math.Max(math.Abs(originalACF[i]), math.Abs(syntheticACF[i]))
		if maxVal > 0 {
			lagSimilarity[i] = 1.0 - math.Abs(diff)/maxVal
		} else {
			lagSimilarity[i] = 1.0
		}
	}

	// Calculate overall metrics
	mse := sumSquaredDiff / float64(minLen)
	
	// Calculate correlation between ACFs
	correlationScore := calculateCorrelation(originalACF, syntheticACF)
	
	// Calculate maximum difference
	maxDiff := 0.0
	for i := 0; i < minLen; i++ {
		diff := math.Abs(originalACF[i] - syntheticACF[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	// Calculate overall similarity (weighted combination of metrics)
	similarity := (correlationScore + calculateMean(lagSimilarity) + (1.0-maxDiff)) / 3.0
	similarity = math.Max(0, math.Min(1, similarity))

	comparison := &AutocorrelationComparison{
		OriginalACF:      originalACF,
		SyntheticACF:     syntheticACF,
		Similarity:       similarity,
		LagSimilarity:    lagSimilarity,
		MaxDifference:    maxDiff,
		MeanSquaredError: mse,
		CorrelationScore: correlationScore,
	}

	return comparison, nil
}

// AutocorrelationSimilarityScore calculates a similarity score between two ACFs
func AutocorrelationSimilarityScore(original, synthetic []float64, maxLag int) (float64, error) {
	comparison, err := CompareAutocorrelations(original, synthetic, maxLag)
	if err != nil {
		return 0, err
	}
	return comparison.Similarity, nil
}

// calculateConfidenceBounds calculates confidence bounds for autocorrelations
func calculateConfidenceBounds(n, maxLag int, alpha float64) ConfidenceBounds {
	// Under null hypothesis of white noise, ACF is approximately N(0, 1/n)
	se := 1.0 / math.Sqrt(float64(n))
	
	// Critical value for given alpha
	var zAlpha float64
	switch {
	case alpha <= 0.01:
		zAlpha = 2.576
	case alpha <= 0.05:
		zAlpha = 1.96
	case alpha <= 0.10:
		zAlpha = 1.645
	default:
		zAlpha = 1.96
	}

	lowerBounds := make([]float64, maxLag)
	upperBounds := make([]float64, maxLag)
	
	for i := 0; i < maxLag; i++ {
		bound := zAlpha * se
		lowerBounds[i] = -bound
		upperBounds[i] = bound
	}

	return ConfidenceBounds{
		Alpha:       alpha,
		LowerBounds: lowerBounds,
		UpperBounds: upperBounds,
	}
}

// findSignificantLags identifies lags with significant autocorrelation
func findSignificantLags(acf []float64, bounds ConfidenceBounds) []int {
	var significantLags []int
	
	for i, ac := range acf {
		if i < len(bounds.LowerBounds) && i < len(bounds.UpperBounds) {
			if ac < bounds.LowerBounds[i] || ac > bounds.UpperBounds[i] {
				significantLags = append(significantLags, i+1) // Lag numbers start from 1
			}
		}
	}
	
	return significantLags
}

// calculateDecayRate estimates the decay rate of the ACF
func calculateDecayRate(acf []float64) float64 {
	if len(acf) < 2 {
		return 0
	}

	// Simple approach: calculate the rate at which ACF approaches zero
	// Use exponential decay model: ACF(k) H exp(-»k)
	
	var sumLogRatio float64
	var count int
	
	for i := 1; i < len(acf) && i < 10; i++ { // Use first 10 lags
		if acf[i-1] > 0 && acf[i] > 0 {
			logRatio := math.Log(acf[i]) - math.Log(acf[i-1])
			sumLogRatio += logRatio
			count++
		}
	}
	
	if count > 0 {
		return -sumLogRatio / float64(count) // Negative because we want decay rate
	}
	
	return 0
}

// assessStationarity performs basic stationarity assessment
func assessStationarity(data []float64, acf []float64) StationarityIndicators {
	n := len(data)
	
	// Split data into chunks to assess stationarity
	chunkSize := n / 3
	if chunkSize < 5 {
		chunkSize = n / 2
	}
	
	var meanStationarity, varianceStationarity bool
	
	if chunkSize >= 5 {
		// Test mean stationarity
		chunk1Mean := calculateMean(data[:chunkSize])
		chunk2Mean := calculateMean(data[chunkSize:2*chunkSize])
		chunk3Mean := calculateMean(data[2*chunkSize:])
		
		meanDiff1 := math.Abs(chunk1Mean - chunk2Mean)
		meanDiff2 := math.Abs(chunk2Mean - chunk3Mean)
		overallMean := calculateMean(data)
		
		// If chunk means are close to overall mean, assume mean stationarity
		meanStationarity = (meanDiff1 < 0.1*math.Abs(overallMean)) && (meanDiff2 < 0.1*math.Abs(overallMean))
		
		// Test variance stationarity
		chunk1Var := calculateVariance(data[:chunkSize])
		chunk2Var := calculateVariance(data[chunkSize:2*chunkSize])
		chunk3Var := calculateVariance(data[2*chunkSize:])
		
		varRatio1 := math.Max(chunk1Var, chunk2Var) / math.Min(chunk1Var, chunk2Var)
		varRatio2 := math.Max(chunk2Var, chunk3Var) / math.Min(chunk2Var, chunk3Var)
		
		// If variance ratios are close to 1, assume variance stationarity
		varianceStationarity = varRatio1 < 2.0 && varRatio2 < 2.0
	}

	// Detect trend (simplified: check if ACF decays slowly)
	trendPresent := false
	if len(acf) > 0 && acf[0] > 0.8 {
		trendPresent = true
	}

	// Detect seasonality (simplified: check for periodic patterns in ACF)
	seasonalityPresent := false
	if len(acf) >= 12 {
		// Look for peaks at seasonal lags (e.g., 12, 24 for monthly data)
		for _, lag := range []int{11, 23} { // 0-indexed, so 11 = lag 12
			if lag < len(acf) && math.Abs(acf[lag]) > 0.3 {
				seasonalityPresent = true
				break
			}
		}
	}

	// Ljung-Box test p-value (simplified)
	ljungBoxPValue := calculateLjungBoxPValue(acf)

	return StationarityIndicators{
		MeanStationarity:     meanStationarity,
		VarianceStationarity: varianceStationarity,
		TrendPresent:         trendPresent,
		SeasonalityPresent:   seasonalityPresent,
		LjungBoxPValue:       ljungBoxPValue,
	}
}

// calculateLjungBoxPValue calculates simplified Ljung-Box test p-value
func calculateLjungBoxPValue(acf []float64) float64 {
	if len(acf) == 0 {
		return 1.0
	}

	// Simplified calculation
	var Q float64
	n := float64(len(acf) * 4) // Approximate sample size
	
	for k := 1; k <= min(10, len(acf)); k++ {
		rk := acf[k-1]
		Q += (rk * rk) / (n - float64(k))
	}
	Q *= n * (n + 2)

	// Approximate p-value using chi-square distribution
	df := float64(min(10, len(acf)))
	pValue := 1.0 - chiSquareCDF(Q, df)
	
	return math.Max(0, math.Min(1, pValue))
}

// Utility functions

func calculateMean(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sum := 0.0
	for _, x := range data {
		sum += x
	}
	return sum / float64(len(data))
}

func calculateVariance(data []float64) float64 {
	if len(data) <= 1 {
		return 0
	}
	mean := calculateMean(data)
	sum := 0.0
	for _, x := range data {
		diff := x - mean
		sum += diff * diff
	}
	return sum / float64(len(data)-1)
}

func calculateCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0
	}

	meanX := calculateMean(x)
	meanY := calculateMean(y)

	var numerator, sumXSq, sumYSq float64
	for i := 0; i < len(x); i++ {
		diffX := x[i] - meanX
		diffY := y[i] - meanY
		numerator += diffX * diffY
		sumXSq += diffX * diffX
		sumYSq += diffY * diffY
	}

	denominator := math.Sqrt(sumXSq * sumYSq)
	if denominator == 0 {
		return 0
	}

	return numerator / denominator
}

func solveLinearSystem(A [][]float64, b []float64) ([]float64, error) {
	n := len(A)
	if n == 0 || len(b) != n {
		return nil, errors.New("invalid system dimensions")
	}

	// Simple Gaussian elimination for small systems
	// Create augmented matrix
	aug := make([][]float64, n)
	for i := 0; i < n; i++ {
		aug[i] = make([]float64, n+1)
		copy(aug[i][:n], A[i])
		aug[i][n] = b[i]
	}

	// Forward elimination
	for i := 0; i < n; i++ {
		// Find pivot
		maxRow := i
		for k := i + 1; k < n; k++ {
			if math.Abs(aug[k][i]) > math.Abs(aug[maxRow][i]) {
				maxRow = k
			}
		}

		// Swap rows
		aug[i], aug[maxRow] = aug[maxRow], aug[i]

		// Check for singular matrix
		if math.Abs(aug[i][i]) < 1e-12 {
			return nil, errors.New("singular matrix")
		}

		// Eliminate column
		for k := i + 1; k < n; k++ {
			factor := aug[k][i] / aug[i][i]
			for j := i; j <= n; j++ {
				aug[k][j] -= factor * aug[i][j]
			}
		}
	}

	// Back substitution
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		x[i] = aug[i][n]
		for j := i + 1; j < n; j++ {
			x[i] -= aug[i][j] * x[j]
		}
		x[i] /= aug[i][i]
	}

	return x, nil
}

func chiSquareCDF(x, df float64) float64 {
	// Simplified approximation
	if x <= 0 || df <= 0 {
		return 0
	}
	
	// Wilson-Hilferty approximation
	h := 2.0 / (9.0 * df)
	z := (math.Pow(x/df, 1.0/3.0) - 1 + h) / math.Sqrt(h)
	
	return normalCDF(z)
}

func normalCDF(x float64) float64 {
	return 0.5 * (1 + math.Erf(x/math.Sqrt2))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// AutocorrelationValidationMetric implements validation interface
func AutocorrelationValidationMetric(original, synthetic *models.TimeSeries, config map[string]interface{}) (*models.ValidationMetric, error) {
	// Extract values from time series
	originalValues := make([]float64, len(original.DataPoints))
	syntheticValues := make([]float64, len(synthetic.DataPoints))
	
	for i, dp := range original.DataPoints {
		originalValues[i] = dp.Value
	}
	for i, dp := range synthetic.DataPoints {
		syntheticValues[i] = dp.Value
	}

	// Get max lag from config
	maxLag := 20 // default
	if ml, ok := config["max_lag"].(int); ok {
		maxLag = ml
	}

	// Compare autocorrelations
	comparison, err := CompareAutocorrelations(originalValues, syntheticValues, maxLag)
	if err != nil {
		return nil, err
	}

	return &models.ValidationMetric{
		Name:        "Autocorrelation Similarity",
		Value:       comparison.Similarity,
		Score:       comparison.Similarity,
		Passed:      comparison.Similarity > 0.7, // Threshold
		Description: "Measures similarity between autocorrelation functions",
		Details: map[string]interface{}{
			"max_difference":      comparison.MaxDifference,
			"mean_squared_error":  comparison.MeanSquaredError,
			"correlation_score":   comparison.CorrelationScore,
			"lag_similarity":      comparison.LagSimilarity,
		},
	}, nil
}