package arima

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/stat"
)

// SeasonalityDetector detects and analyzes seasonal patterns in time series
type SeasonalityDetector struct {
	maxSeasonalPeriod int
	significanceLevel float64
}

// NewSeasonalityDetector creates a new seasonality detector
func NewSeasonalityDetector(maxPeriod int, significanceLevel float64) *SeasonalityDetector {
	if maxPeriod <= 0 {
		maxPeriod = 52 // Default to weekly seasonality for daily data
	}
	if significanceLevel <= 0 || significanceLevel >= 1 {
		significanceLevel = 0.05 // Default 5% significance level
	}

	return &SeasonalityDetector{
		maxSeasonalPeriod: maxPeriod,
		significanceLevel: significanceLevel,
	}
}

// SeasonalityResult contains the results of seasonality analysis
type SeasonalityResult struct {
	IsStationary         bool                `json:"is_stationary"`
	DetectedPeriods      []int               `json:"detected_periods"`
	SeasonalStrengths    map[int]float64     `json:"seasonal_strengths"`
	PrimaryPeriod        int                 `json:"primary_period"`
	PrimaryStrength      float64             `json:"primary_strength"`
	DecompositionResult  *DecompositionResult `json:"decomposition_result"`
	AutocorrelationTest  *AutocorrelationTest `json:"autocorrelation_test"`
}

// DecompositionResult contains seasonal decomposition components
type DecompositionResult struct {
	Trend           []float64 `json:"trend"`
	Seasonal        []float64 `json:"seasonal"`
	Residual        []float64 `json:"residual"`
	DecompositionType string  `json:"decomposition_type"` // "additive" or "multiplicative"
}

// AutocorrelationTest contains results of autocorrelation-based seasonality test
type AutocorrelationTest struct {
	LaggedAutocorrelations map[int]float64 `json:"lagged_autocorrelations"`
	SignificantLags        []int           `json:"significant_lags"`
	TestStatistic          float64         `json:"test_statistic"`
	PValue                 float64         `json:"p_value"`
	IsSignificant          bool            `json:"is_significant"`
}

// DetectSeasonality analyzes a time series for seasonal patterns
func (sd *SeasonalityDetector) DetectSeasonality(data []float64) (*SeasonalityResult, error) {
	if len(data) < 2*sd.maxSeasonalPeriod {
		return &SeasonalityResult{
			IsStationary:      false,
			DetectedPeriods:   []int{},
			SeasonalStrengths: make(map[int]float64),
			PrimaryPeriod:     0,
			PrimaryStrength:   0.0,
		}, nil
	}

	result := &SeasonalityResult{
		SeasonalStrengths: make(map[int]float64),
	}

	// 1. Test for stationarity
	result.IsStationary = sd.testStationarity(data)

	// 2. Autocorrelation-based seasonality detection
	result.AutocorrelationTest = sd.autocorrelationSeasonalityTest(data)

	// 3. Detect seasonal periods using multiple methods
	result.DetectedPeriods, result.SeasonalStrengths = sd.detectSeasonalPeriods(data)

	// 4. Identify primary seasonal period
	if len(result.DetectedPeriods) > 0 {
		result.PrimaryPeriod = result.DetectedPeriods[0]
		result.PrimaryStrength = result.SeasonalStrengths[result.PrimaryPeriod]
	}

	// 5. Seasonal decomposition
	if result.PrimaryPeriod > 0 {
		decomp, err := sd.seasonalDecomposition(data, result.PrimaryPeriod)
		if err == nil {
			result.DecompositionResult = decomp
		}
	}

	return result, nil
}

// testStationarity performs an augmented Dickey-Fuller test for stationarity
func (sd *SeasonalityDetector) testStationarity(data []float64) bool {
	// Simplified stationarity test
	// In practice, would implement full ADF test
	
	n := len(data)
	if n < 10 {
		return false
	}

	// Calculate first differences
	diffs := make([]float64, n-1)
	for i := 1; i < n; i++ {
		diffs[i-1] = data[i] - data[i-1]
	}

	// Test if differences have mean around zero and bounded variance
	mean := stat.Mean(diffs, nil)
	variance := stat.Variance(diffs, nil)
	
	// Simple heuristic: stationary if |mean| is small relative to std dev
	stdDev := math.Sqrt(variance)
	if stdDev > 0 {
		return math.Abs(mean) < 0.1*stdDev
	}

	return false
}

// autocorrelationSeasonalityTest performs autocorrelation-based seasonality detection
func (sd *SeasonalityDetector) autocorrelationSeasonalityTest(data []float64) *AutocorrelationTest {
	n := len(data)
	maxLag := min(sd.maxSeasonalPeriod, n/4)
	
	// Calculate autocorrelations
	acf := sd.calculateAutocorrelation(data, maxLag)
	
	// Find significant lags
	significantLags := []int{}
	laggedACF := make(map[int]float64)
	
	// Critical value for significance (approximate)
	criticalValue := 2.0 / math.Sqrt(float64(n))
	
	testStatistic := 0.0
	for lag := 1; lag <= maxLag; lag++ {
		if lag < len(acf) {
			laggedACF[lag] = acf[lag]
			if math.Abs(acf[lag]) > criticalValue {
				significantLags = append(significantLags, lag)
				testStatistic += acf[lag] * acf[lag]
			}
		}
	}
	
	// Simplified p-value calculation
	pValue := math.Exp(-testStatistic * float64(n) / 2.0)
	
	return &AutocorrelationTest{
		LaggedAutocorrelations: laggedACF,
		SignificantLags:        significantLags,
		TestStatistic:          testStatistic,
		PValue:                 pValue,
		IsSignificant:          pValue < sd.significanceLevel,
	}
}

// detectSeasonalPeriods detects seasonal periods using multiple approaches
func (sd *SeasonalityDetector) detectSeasonalPeriods(data []float64) ([]int, map[int]float64) {
	periods := []int{}
	strengths := make(map[int]float64)
	
	// Method 1: Autocorrelation peaks
	acfPeriods := sd.detectPeriodsFromACF(data)
	
	// Method 2: Box-Jenkins seasonal identification
	bjPeriods := sd.boxJenkinsSeasonalIdentification(data)
	
	// Combine results and calculate strengths
	candidatePeriods := make(map[int]bool)
	
	for _, p := range acfPeriods {
		candidatePeriods[p] = true
	}
	for _, p := range bjPeriods {
		candidatePeriods[p] = true
	}
	
	// Calculate strength for each candidate period
	for period := range candidatePeriods {
		strength := sd.calculateSeasonalStrength(data, period)
		if strength > 0.1 { // Minimum threshold for seasonal strength
			periods = append(periods, period)
			strengths[period] = strength
		}
	}
	
	// Sort by strength (descending)
	sort.Slice(periods, func(i, j int) bool {
		return strengths[periods[i]] > strengths[periods[j]]
	})
	
	return periods, strengths
}

// detectPeriodsFromACF detects periods from autocorrelation function peaks
func (sd *SeasonalityDetector) detectPeriodsFromACF(data []float64) []int {
	maxLag := min(sd.maxSeasonalPeriod, len(data)/4)
	acf := sd.calculateAutocorrelation(data, maxLag)
	
	periods := []int{}
	n := float64(len(data))
	criticalValue := 2.0 / math.Sqrt(n)
	
	// Find local maxima in ACF that exceed critical value
	for lag := 2; lag < len(acf)-1; lag++ {
		if acf[lag] > criticalValue &&
			acf[lag] > acf[lag-1] &&
			acf[lag] > acf[lag+1] {
			periods = append(periods, lag)
		}
	}
	
	return periods
}

// boxJenkinsSeasonalIdentification uses Box-Jenkins methodology
func (sd *SeasonalityDetector) boxJenkinsSeasonalIdentification(data []float64) []int {
	periods := []int{}
	
	// Common seasonal periods to test
	commonPeriods := []int{4, 7, 12, 24, 52, 365}
	
	for _, period := range commonPeriods {
		if period <= sd.maxSeasonalPeriod && period < len(data)/2 {
			// Test seasonal difference
			seasonalDiff := sd.seasonalDifference(data, period)
			
			// Check if seasonal differencing reduces variance
			originalVar := stat.Variance(data, nil)
			diffVar := stat.Variance(seasonalDiff, nil)
			
			if diffVar < 0.8*originalVar { // 20% variance reduction threshold
				periods = append(periods, period)
			}
		}
	}
	
	return periods
}

// calculateSeasonalStrength calculates the strength of seasonality for a given period
func (sd *SeasonalityDetector) calculateSeasonalStrength(data []float64, period int) float64 {
	if period <= 1 || period >= len(data)/2 {
		return 0.0
	}
	
	// Method: Variance of seasonal means vs. total variance
	n := len(data)
	numSeasons := n / period
	
	if numSeasons < 2 {
		return 0.0
	}
	
	// Calculate seasonal means
	seasonalMeans := make([]float64, period)
	seasonalCounts := make([]int, period)
	
	for i, value := range data {
		seasonIndex := i % period
		seasonalMeans[seasonIndex] += value
		seasonalCounts[seasonIndex]++
	}
	
	for i := 0; i < period; i++ {
		if seasonalCounts[i] > 0 {
			seasonalMeans[i] /= float64(seasonalCounts[i])
		}
	}
	
	// Calculate grand mean
	grandMean := stat.Mean(data, nil)
	
	// Calculate between-season variance
	betweenVar := 0.0
	for i := 0; i < period; i++ {
		if seasonalCounts[i] > 0 {
			diff := seasonalMeans[i] - grandMean
			betweenVar += diff * diff * float64(seasonalCounts[i])
		}
	}
	betweenVar /= float64(n - 1)
	
	// Calculate total variance
	totalVar := stat.Variance(data, nil)
	
	// Seasonal strength
	if totalVar > 0 {
		return betweenVar / totalVar
	}
	
	return 0.0
}

// seasonalDecomposition performs seasonal decomposition
func (sd *SeasonalityDetector) seasonalDecomposition(data []float64, period int) (*DecompositionResult, error) {
	n := len(data)
	if period <= 1 || period >= n/2 {
		return nil, nil
	}
	
	// Initialize components
	trend := make([]float64, n)
	seasonal := make([]float64, n)
	residual := make([]float64, n)
	
	// 1. Calculate trend using moving average
	halfPeriod := period / 2
	for i := halfPeriod; i < n-halfPeriod; i++ {
		sum := 0.0
		for j := i - halfPeriod; j <= i+halfPeriod; j++ {
			sum += data[j]
		}
		trend[i] = sum / float64(period)
	}
	
	// Extrapolate trend at endpoints
	for i := 0; i < halfPeriod; i++ {
		trend[i] = trend[halfPeriod]
	}
	for i := n - halfPeriod; i < n; i++ {
		trend[i] = trend[n-halfPeriod-1]
	}
	
	// 2. Calculate seasonal component (additive model)
	detrended := make([]float64, n)
	for i := 0; i < n; i++ {
		detrended[i] = data[i] - trend[i]
	}
	
	// Calculate average seasonal pattern
	seasonalPattern := make([]float64, period)
	seasonalCounts := make([]int, period)
	
	for i, value := range detrended {
		seasonIndex := i % period
		seasonalPattern[seasonIndex] += value
		seasonalCounts[seasonIndex]++
	}
	
	for i := 0; i < period; i++ {
		if seasonalCounts[i] > 0 {
			seasonalPattern[i] /= float64(seasonalCounts[i])
		}
	}
	
	// Apply seasonal pattern
	for i := 0; i < n; i++ {
		seasonal[i] = seasonalPattern[i%period]
	}
	
	// 3. Calculate residual
	for i := 0; i < n; i++ {
		residual[i] = data[i] - trend[i] - seasonal[i]
	}
	
	return &DecompositionResult{
		Trend:             trend,
		Seasonal:          seasonal,
		Residual:          residual,
		DecompositionType: "additive",
	}, nil
}

// Helper functions

func (sd *SeasonalityDetector) calculateAutocorrelation(data []float64, maxLag int) []float64 {
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
		if c0 > 0 {
			acf[k] = ck / c0
		}
	}
	
	return acf
}

func (sd *SeasonalityDetector) seasonalDifference(data []float64, period int) []float64 {
	n := len(data)
	if period >= n {
		return data
	}
	
	diff := make([]float64, n-period)
	for i := period; i < n; i++ {
		diff[i-period] = data[i] - data[i-period]
	}
	
	return diff
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}