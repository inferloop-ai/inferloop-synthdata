package tests

import (
	"errors"
	"fmt"
	"math"

	"github.com/inferloop/tsiot/pkg/models"
)

// LjungBoxTestResult contains detailed results of the Ljung-Box test
type LjungBoxTestResult struct {
	*StatisticalTestResult
	Lags               int                    `json:"lags"`
	SampleSize         int                    `json:"sample_size"`
	Autocorrelations   []float64              `json:"autocorrelations"`
	BoxPierceStatistic float64                `json:"box_pierce_statistic"`
	DegreesOfFreedom   int                    `json:"degrees_of_freedom"`
	EffectiveSampleSize int                   `json:"effective_sample_size"`
}

// AutocorrelationResult contains autocorrelation analysis
type AutocorrelationResult struct {
	Lag          int     `json:"lag"`
	Correlation  float64 `json:"correlation"`
	StandardError float64 `json:"standard_error"`
	IsSignificant bool   `json:"is_significant"`
	Bounds       struct {
		Lower float64 `json:"lower"`
		Upper float64 `json:"upper"`
	} `json:"bounds"`
}

// LjungBoxTest performs the Ljung-Box test for autocorrelation
func LjungBoxTest(data []float64, lags int, alpha float64) (*models.StatisticalTestResult, error) {
	if len(data) < 10 {
		return nil, errors.New("Ljung-Box test requires at least 10 observations")
	}

	if lags <= 0 {
		// Default number of lags
		lags = min(int(math.Log(float64(len(data)))), len(data)/4)
		if lags < 1 {
			lags = 1
		}
	}

	if lags >= len(data) {
		return nil, errors.New("number of lags must be less than sample size")
	}

	result, err := performLjungBoxTest(data, lags, alpha)
	if err != nil {
		return nil, err
	}

	return convertLBToModelResult(result), nil
}

// BoxPierceTest performs the Box-Pierce test (simpler version of Ljung-Box)
func BoxPierceTest(data []float64, lags int, alpha float64) (*models.StatisticalTestResult, error) {
	if len(data) < 10 {
		return nil, errors.New("Box-Pierce test requires at least 10 observations")
	}

	if lags <= 0 {
		lags = min(int(math.Log(float64(len(data)))), len(data)/4)
		if lags < 1 {
			lags = 1
		}
	}

	result, err := performBoxPierceTest(data, lags, alpha)
	if err != nil {
		return nil, err
	}

	return convertLBToModelResult(result), nil
}

// performLjungBoxTest is the core implementation of the Ljung-Box test
func performLjungBoxTest(data []float64, lags int, alpha float64) (*LjungBoxTestResult, error) {
	n := len(data)
	
	// Calculate autocorrelations
	autocorr, err := calculateAutocorrelations(data, lags)
	if err != nil {
		return nil, err
	}

	// Calculate Ljung-Box statistic
	var Q float64
	for k := 1; k <= lags; k++ {
		rk := autocorr[k-1]
		Q += (rk * rk) / float64(n-k)
	}
	Q *= float64(n) * (float64(n) + 2)

	// Also calculate Box-Pierce statistic for comparison
	var BP float64
	for k := 1; k <= lags; k++ {
		rk := autocorr[k-1]
		BP += rk * rk
	}
	BP *= float64(n)

	// Degrees of freedom
	df := lags

	// Calculate p-value using chi-square distribution
	pValue := 1 - chiSquareCDF(Q, float64(df))

	// Critical value for given alpha level
	criticalValue := chiSquareQuantile(1-alpha, float64(df))

	isSignificant := Q > criticalValue

	// Create interpretation
	interpretation := generateLjungBoxInterpretation(isSignificant, Q, pValue, lags, n)

	result := &LjungBoxTestResult{
		StatisticalTestResult: &StatisticalTestResult{
			TestName:       "Ljung-Box Test",
			Statistic:      Q,
			PValue:         pValue,
			CriticalValue:  criticalValue,
			IsSignificant:  isSignificant,
			AlphaLevel:     alpha,
			Description:    "Tests for autocorrelation in time series residuals",
			Interpretation: interpretation,
		},
		Lags:                lags,
		SampleSize:          n,
		Autocorrelations:    autocorr,
		BoxPierceStatistic:  BP,
		DegreesOfFreedom:    df,
		EffectiveSampleSize: n - lags,
	}

	return result, nil
}

// performBoxPierceTest is the core implementation of the Box-Pierce test
func performBoxPierceTest(data []float64, lags int, alpha float64) (*LjungBoxTestResult, error) {
	n := len(data)
	
	// Calculate autocorrelations
	autocorr, err := calculateAutocorrelations(data, lags)
	if err != nil {
		return nil, err
	}

	// Calculate Box-Pierce statistic
	var BP float64
	for k := 1; k <= lags; k++ {
		rk := autocorr[k-1]
		BP += rk * rk
	}
	BP *= float64(n)

	// Degrees of freedom
	df := lags

	// Calculate p-value using chi-square distribution
	pValue := 1 - chiSquareCDF(BP, float64(df))

	// Critical value for given alpha level
	criticalValue := chiSquareQuantile(1-alpha, float64(df))

	isSignificant := BP > criticalValue

	// Create interpretation
	interpretation := generateBoxPierceInterpretation(isSignificant, BP, pValue, lags, n)

	result := &LjungBoxTestResult{
		StatisticalTestResult: &StatisticalTestResult{
			TestName:       "Box-Pierce Test",
			Statistic:      BP,
			PValue:         pValue,
			CriticalValue:  criticalValue,
			IsSignificant:  isSignificant,
			AlphaLevel:     alpha,
			Description:    "Tests for autocorrelation in time series data",
			Interpretation: interpretation,
		},
		Lags:                lags,
		SampleSize:          n,
		Autocorrelations:    autocorr,
		BoxPierceStatistic:  BP,
		DegreesOfFreedom:    df,
		EffectiveSampleSize: n - lags,
	}

	return result, nil
}

// calculateAutocorrelations calculates sample autocorrelations up to specified lag
func calculateAutocorrelations(data []float64, maxLag int) ([]float64, error) {
	n := len(data)
	if maxLag >= n {
		return nil, errors.New("maximum lag must be less than sample size")
	}

	// Calculate sample mean
	mean := 0.0
	for _, x := range data {
		mean += x
	}
	mean /= float64(n)

	// Calculate sample autocovariances
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
	autocorr := make([]float64, maxLag)
	for lag := 1; lag <= maxLag; lag++ {
		if autocovar[0] != 0 {
			autocorr[lag-1] = autocovar[lag] / autocovar[0]
		}
	}

	return autocorr, nil
}

// calculateAutocorrelationBounds calculates confidence bounds for autocorrelations
func calculateAutocorrelationBounds(n int, alpha float64) (float64, float64) {
	// Standard error for autocorrelations under null hypothesis of white noise
	se := 1.0 / math.Sqrt(float64(n))
	
	// Critical value for normal distribution
	var zAlpha float64
	if alpha <= 0.01 {
		zAlpha = 2.576
	} else if alpha <= 0.05 {
		zAlpha = 1.96
	} else if alpha <= 0.10 {
		zAlpha = 1.645
	} else {
		zAlpha = 1.96 // Default to 5%
	}

	bound := zAlpha * se
	return -bound, bound
}

// AutocorrelationAnalysis performs detailed autocorrelation analysis
func AutocorrelationAnalysis(data []float64, maxLag int, alpha float64) ([]AutocorrelationResult, error) {
	if len(data) < 10 {
		return nil, errors.New("autocorrelation analysis requires at least 10 observations")
	}

	if maxLag <= 0 {
		maxLag = min(int(math.Log(float64(len(data)))), len(data)/4)
	}

	autocorr, err := calculateAutocorrelations(data, maxLag)
	if err != nil {
		return nil, err
	}

	lowerBound, upperBound := calculateAutocorrelationBounds(len(data), alpha)
	
	results := make([]AutocorrelationResult, maxLag)
	for i := 0; i < maxLag; i++ {
		lag := i + 1
		corr := autocorr[i]
		se := 1.0 / math.Sqrt(float64(len(data)))
		
		results[i] = AutocorrelationResult{
			Lag:           lag,
			Correlation:   corr,
			StandardError: se,
			IsSignificant: corr < lowerBound || corr > upperBound,
			Bounds: struct {
				Lower float64 `json:"lower"`
				Upper float64 `json:"upper"`
			}{
				Lower: lowerBound,
				Upper: upperBound,
			},
		}
	}

	return results, nil
}

// SerialCorrelationTest performs a comprehensive serial correlation test
func SerialCorrelationTest(data []float64, alpha float64) (*models.StatisticalTestResult, error) {
	if len(data) < 10 {
		return nil, errors.New("serial correlation test requires at least 10 observations")
	}

	// Determine optimal number of lags using various criteria
	n := len(data)
	
	// Rule of thumb: log(n) or n/4, whichever is smaller
	optimalLags := min(int(math.Log(float64(n))), n/4)
	if optimalLags < 1 {
		optimalLags = 1
	}

	// Perform Ljung-Box test
	result, err := LjungBoxTest(data, optimalLags, alpha)
	if err != nil {
		return nil, err
	}

	// Perform autocorrelation analysis
	autocorrResults, err := AutocorrelationAnalysis(data, optimalLags, alpha)
	if err != nil {
		return nil, err
	}

	// Count significant autocorrelations
	significantLags := 0
	for _, ar := range autocorrResults {
		if ar.IsSignificant {
			significantLags++
		}
	}

	// Modify result to include additional information
	result.TestName = "Serial Correlation Test"
	result.Description = "Comprehensive test for serial correlation using Ljung-Box and autocorrelation analysis"
	
	if result.Metadata == nil {
		result.Metadata = make(map[string]interface{})
	}
	result.Metadata["optimal_lags"] = optimalLags
	result.Metadata["significant_lags"] = significantLags
	result.Metadata["autocorrelation_results"] = autocorrResults

	return result, nil
}

// PortmanteauTest performs a portmanteau test (alias for Ljung-Box)
func PortmanteauTest(data []float64, lags int, alpha float64) (*models.StatisticalTestResult, error) {
	result, err := LjungBoxTest(data, lags, alpha)
	if err != nil {
		return nil, err
	}

	result.TestName = "Portmanteau Test"
	result.Description = "Tests for autocorrelation using portmanteau statistic (Ljung-Box variant)"

	return result, nil
}

// generateLjungBoxInterpretation creates interpretation for Ljung-Box test
func generateLjungBoxInterpretation(isSignificant bool, Q, pValue float64, lags, n int) string {
	if isSignificant {
		return fmt.Sprintf("Reject null hypothesis: Significant autocorrelation detected "+
			"(Q = %.4f, p = %.4f, lags = %d, n = %d). The data shows serial dependence.", 
			Q, pValue, lags, n)
	} else {
		return fmt.Sprintf("Fail to reject null hypothesis: No significant autocorrelation detected "+
			"(Q = %.4f, p = %.4f, lags = %d, n = %d). The data appears to be serially independent.", 
			Q, pValue, lags, n)
	}
}

// generateBoxPierceInterpretation creates interpretation for Box-Pierce test
func generateBoxPierceInterpretation(isSignificant bool, BP, pValue float64, lags, n int) string {
	if isSignificant {
		return fmt.Sprintf("Reject null hypothesis: Significant autocorrelation detected "+
			"(BP = %.4f, p = %.4f, lags = %d, n = %d). The data shows serial dependence.", 
			BP, pValue, lags, n)
	} else {
		return fmt.Sprintf("Fail to reject null hypothesis: No significant autocorrelation detected "+
			"(BP = %.4f, p = %.4f, lags = %d, n = %d). The data appears to be serially independent.", 
			BP, pValue, lags, n)
	}
}

// convertLBToModelResult converts internal result to model result
func convertLBToModelResult(result *LjungBoxTestResult) *models.StatisticalTestResult {
	return &models.StatisticalTestResult{
		TestName:      result.TestName,
		Statistic:     result.Statistic,
		PValue:        result.PValue,
		CriticalValue: result.CriticalValue,
		Passed:        !result.IsSignificant,
		Alpha:         result.AlphaLevel,
		Description:   result.Description,
		Metadata: map[string]interface{}{
			"interpretation":         result.Interpretation,
			"lags":                  result.Lags,
			"sample_size":           result.SampleSize,
			"autocorrelations":      result.Autocorrelations,
			"box_pierce_statistic":  result.BoxPierceStatistic,
			"degrees_of_freedom":    result.DegreesOfFreedom,
			"effective_sample_size": result.EffectiveSampleSize,
		},
	}
}

// Utility functions for chi-square distribution
// These are simplified implementations - in production, use a proper stats library

func chiSquareCDF(x, df float64) float64 {
	if x <= 0 {
		return 0
	}
	if df <= 0 {
		return 0
	}
	
	// Simplified approximation using gamma function
	// For a proper implementation, use incomplete gamma function
	
	// Wilson-Hilferty approximation
	h := 2.0 / (9.0 * df)
	z := (math.Pow(x/df, 1.0/3.0) - 1 + h) / math.Sqrt(h)
	
	return normalCDF(z)
}

func chiSquareQuantile(p, df float64) float64 {
	if p <= 0 {
		return 0
	}
	if p >= 1 {
		return math.Inf(1)
	}
	
	// Simplified approximation
	// Wilson-Hilferty approximation
	h := 2.0 / (9.0 * df)
	z := normalQuantile(p)
	x := df * math.Pow(1-h+z*math.Sqrt(h), 3)
	
	return math.Max(0, x)
}

func normalQuantile(p float64) float64 {
	// Simplified approximation of normal quantile
	// Acklam's approximation
	if p <= 0 {
		return math.Inf(-1)
	}
	if p >= 1 {
		return math.Inf(1)
	}
	
	if p == 0.5 {
		return 0
	}
	
	// Coefficients for Acklam's approximation
	a := []float64{-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00}
	b := []float64{-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01}
	c := []float64{-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00}
	d := []float64{7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00}
	
	var x, r float64
	
	if p < 0.02425 {
		// Lower tail
		q := math.Sqrt(-2 * math.Log(p))
		x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
	} else if p <= 0.97575 {
		// Central region
		q := p - 0.5
		r = q * q
		x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
	} else {
		// Upper tail
		q := math.Sqrt(-2 * math.Log(1-p))
		x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
	}
	
	return x
}