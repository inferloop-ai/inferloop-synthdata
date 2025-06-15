package tests

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/inferloop/tsiot/pkg/models"
)

// AndersonDarlingTestResult contains detailed results of the Anderson-Darling test
type AndersonDarlingTestResult struct {
	*StatisticalTestResult
	AdjustedStatistic float64                `json:"adjusted_statistic"`
	SampleSize        int                    `json:"sample_size"`
	StandardizedData  []float64              `json:"standardized_data,omitempty"`
	TestType          string                 `json:"test_type"` // "normality", "exponential", "uniform"
}

// AndersonDarlingCriticalValues contains critical values for different significance levels
type AndersonDarlingCriticalValues struct {
	Alpha01  float64 `json:"alpha_0_01"`  // 1% significance level
	Alpha025 float64 `json:"alpha_0_025"` // 2.5% significance level
	Alpha05  float64 `json:"alpha_0_05"`  // 5% significance level
	Alpha10  float64 `json:"alpha_0_10"`  // 10% significance level
}

// AndersonDarlingNormalityTest performs Anderson-Darling test for normality
func AndersonDarlingNormalityTest(data []float64, alpha float64) (*models.StatisticalTestResult, error) {
	if len(data) < 8 {
		return nil, errors.New("Anderson-Darling test requires at least 8 observations")
	}

	result, err := performAndersonDarlingTest(data, "normality", alpha)
	if err != nil {
		return nil, err
	}

	return convertToModelResult(result), nil
}

// AndersonDarlingExponentialTest performs Anderson-Darling test for exponentiality
func AndersonDarlingExponentialTest(data []float64, alpha float64) (*models.StatisticalTestResult, error) {
	if len(data) < 8 {
		return nil, errors.New("Anderson-Darling test requires at least 8 observations")
	}

	// Check that all data is positive for exponential test
	for _, x := range data {
		if x <= 0 {
			return nil, errors.New("Anderson-Darling exponential test requires all positive values")
		}
	}

	result, err := performAndersonDarlingTest(data, "exponential", alpha)
	if err != nil {
		return nil, err
	}

	return convertToModelResult(result), nil
}

// AndersonDarlingUniformTest performs Anderson-Darling test for uniformity
func AndersonDarlingUniformTest(data []float64, alpha float64) (*models.StatisticalTestResult, error) {
	if len(data) < 8 {
		return nil, errors.New("Anderson-Darling test requires at least 8 observations")
	}

	result, err := performAndersonDarlingTest(data, "uniform", alpha)
	if err != nil {
		return nil, err
	}

	return convertToModelResult(result), nil
}

// performAndersonDarlingTest is the core implementation
func performAndersonDarlingTest(data []float64, testType string, alpha float64) (*AndersonDarlingTestResult, error) {
	n := len(data)
	
	// Sort the data
	sortedData := make([]float64, n)
	copy(sortedData, data)
	sort.Float64s(sortedData)

	var A2 float64
	var transformedData []float64

	switch testType {
	case "normality":
		A2, transformedData = calculateA2Normality(sortedData)
	case "exponential":
		A2, transformedData = calculateA2Exponential(sortedData)
	case "uniform":
		A2, transformedData = calculateA2Uniform(sortedData)
	default:
		return nil, errors.New("unsupported test type: " + testType)
	}

	// Adjust A2 for sample size
	A2Star := adjustA2ForSampleSize(A2, n, testType)

	// Get critical values and p-value
	criticalValues := getCriticalValues(testType)
	pValue := calculatePValue(A2Star, testType)
	criticalValue := getCriticalValueForAlpha(criticalValues, alpha)

	isSignificant := A2Star > criticalValue

	// Create interpretation
	interpretation := generateInterpretation(testType, isSignificant, A2Star, pValue)

	result := &AndersonDarlingTestResult{
		StatisticalTestResult: &StatisticalTestResult{
			TestName:       "Anderson-Darling Test (" + testType + ")",
			Statistic:      A2,
			PValue:         pValue,
			CriticalValue:  criticalValue,
			IsSignificant:  isSignificant,
			AlphaLevel:     alpha,
			Description:    "Tests goodness of fit to " + testType + " distribution",
			Interpretation: interpretation,
		},
		AdjustedStatistic: A2Star,
		SampleSize:        n,
		StandardizedData:  transformedData,
		TestType:          testType,
	}

	return result, nil
}

// calculateA2Normality calculates A² statistic for normality test
func calculateA2Normality(sortedData []float64) (float64, []float64) {
	n := len(sortedData)
	
	// Calculate sample mean and standard deviation
	mean := calculateMean(sortedData)
	stdDev := calculateStdDev(sortedData, mean)

	// Standardize the data
	standardized := make([]float64, n)
	for i, x := range sortedData {
		standardized[i] = (x - mean) / stdDev
	}

	// Calculate A² statistic
	var A2 float64
	for i := 0; i < n; i++ {
		zi := standardized[i]
		Fi := normalCDF(zi)
		Fni := normalCDF(standardized[n-1-i])

		// Avoid log(0) and log(1) by clamping values
		Fi = clamp(Fi, 1e-15, 1-1e-15)
		Fni = clamp(Fni, 1e-15, 1-1e-15)

		term := float64(2*i+1) * (math.Log(Fi) + math.Log(1-Fni))
		A2 += term
	}

	A2 = -float64(n) - A2/float64(n)
	return A2, standardized
}

// calculateA2Exponential calculates A² statistic for exponentiality test
func calculateA2Exponential(sortedData []float64) (float64, []float64) {
	n := len(sortedData)
	
	// Estimate rate parameter (lambda = 1/mean)
	mean := calculateMean(sortedData)
	lambda := 1.0 / mean

	// Transform data using exponential CDF
	transformed := make([]float64, n)
	for i, x := range sortedData {
		transformed[i] = 1 - math.Exp(-lambda*x)
	}

	// Calculate A² statistic
	var A2 float64
	for i := 0; i < n; i++ {
		Fi := transformed[i]
		Fni := transformed[n-1-i]

		// Avoid log(0) and log(1)
		Fi = clamp(Fi, 1e-15, 1-1e-15)
		Fni = clamp(Fni, 1e-15, 1-1e-15)

		term := float64(2*i+1) * (math.Log(Fi) + math.Log(1-Fni))
		A2 += term
	}

	A2 = -float64(n) - A2/float64(n)
	return A2, transformed
}

// calculateA2Uniform calculates A² statistic for uniformity test
func calculateA2Uniform(sortedData []float64) (float64, []float64) {
	n := len(sortedData)
	
	// Find range
	min := sortedData[0]
	max := sortedData[n-1]
	
	// Transform data to uniform [0,1]
	transformed := make([]float64, n)
	for i, x := range sortedData {
		transformed[i] = (x - min) / (max - min)
	}

	// Calculate A² statistic
	var A2 float64
	for i := 0; i < n; i++ {
		Fi := transformed[i]
		Fni := transformed[n-1-i]

		// Avoid log(0) and log(1)
		Fi = clamp(Fi, 1e-15, 1-1e-15)
		Fni = clamp(Fni, 1e-15, 1-1e-15)

		term := float64(2*i+1) * (math.Log(Fi) + math.Log(1-Fni))
		A2 += term
	}

	A2 = -float64(n) - A2/float64(n)
	return A2, transformed
}

// adjustA2ForSampleSize adjusts A² statistic for finite sample size
func adjustA2ForSampleSize(A2 float64, n int, testType string) float64 {
	switch testType {
	case "normality":
		return A2 * (1 + 0.75/float64(n) + 2.25/(float64(n)*float64(n)))
	case "exponential":
		return A2 * (1 + 0.6/float64(n))
	case "uniform":
		return A2 * (1 + 0.5/float64(n))
	default:
		return A2
	}
}

// getCriticalValues returns critical values for different test types
func getCriticalValues(testType string) *AndersonDarlingCriticalValues {
	switch testType {
	case "normality":
		return &AndersonDarlingCriticalValues{
			Alpha01:  1.035,
			Alpha025: 0.873,
			Alpha05:  0.752,
			Alpha10:  0.631,
		}
	case "exponential":
		return &AndersonDarlingCriticalValues{
			Alpha01:  1.321,
			Alpha025: 1.078,
			Alpha05:  0.922,
			Alpha10:  0.775,
		}
	case "uniform":
		return &AndersonDarlingCriticalValues{
			Alpha01:  2.816,
			Alpha025: 2.310,
			Alpha05:  1.933,
			Alpha10:  1.610,
		}
	default:
		return &AndersonDarlingCriticalValues{}
	}
}

// getCriticalValueForAlpha interpolates critical value for given alpha
func getCriticalValueForAlpha(cv *AndersonDarlingCriticalValues, alpha float64) float64 {
	switch {
	case alpha <= 0.01:
		return cv.Alpha01
	case alpha <= 0.025:
		// Linear interpolation between 0.01 and 0.025
		t := (alpha - 0.01) / (0.025 - 0.01)
		return cv.Alpha01 + t*(cv.Alpha025-cv.Alpha01)
	case alpha <= 0.05:
		// Linear interpolation between 0.025 and 0.05
		t := (alpha - 0.025) / (0.05 - 0.025)
		return cv.Alpha025 + t*(cv.Alpha05-cv.Alpha025)
	case alpha <= 0.10:
		// Linear interpolation between 0.05 and 0.10
		t := (alpha - 0.05) / (0.10 - 0.05)
		return cv.Alpha05 + t*(cv.Alpha10-cv.Alpha05)
	default:
		return cv.Alpha10
	}
}

// calculatePValue calculates approximate p-value
func calculatePValue(A2Star float64, testType string) float64 {
	switch testType {
	case "normality":
		return calculatePValueNormality(A2Star)
	case "exponential":
		return calculatePValueExponential(A2Star)
	case "uniform":
		return calculatePValueUniform(A2Star)
	default:
		return 0.5
	}
}

// calculatePValueNormality calculates p-value for normality test
func calculatePValueNormality(A2Star float64) float64 {
	if A2Star < 0.2 {
		return 1 - math.Exp(-13.436+101.14*A2Star-223.73*A2Star*A2Star)
	} else if A2Star < 0.34 {
		return 1 - math.Exp(-8.318+42.796*A2Star-59.938*A2Star*A2Star)
	} else if A2Star < 0.6 {
		return math.Exp(0.9177-4.279*A2Star+1.38*A2Star*A2Star)
	} else {
		return math.Exp(1.2937-5.709*A2Star+0.0186*A2Star*A2Star)
	}
}

// calculatePValueExponential calculates p-value for exponentiality test
func calculatePValueExponential(A2Star float64) float64 {
	// Simplified approximation for exponential distribution
	if A2Star < 0.5 {
		return 1 - math.Exp(-8*A2Star)
	} else if A2Star < 1.0 {
		return math.Exp(-2*A2Star)
	} else {
		return math.Exp(-A2Star*A2Star)
	}
}

// calculatePValueUniform calculates p-value for uniformity test
func calculatePValueUniform(A2Star float64) float64 {
	// Simplified approximation for uniform distribution
	if A2Star < 1.0 {
		return 1 - math.Exp(-3*A2Star)
	} else if A2Star < 2.0 {
		return math.Exp(-1.5*A2Star)
	} else {
		return math.Exp(-A2Star)
	}
}

// generateInterpretation creates human-readable interpretation
func generateInterpretation(testType string, isSignificant bool, A2Star, pValue float64) string {
	var nullHypothesis string
	switch testType {
	case "normality":
		nullHypothesis = "data follows a normal distribution"
	case "exponential":
		nullHypothesis = "data follows an exponential distribution"
	case "uniform":
		nullHypothesis = "data follows a uniform distribution"
	}

	if isSignificant {
		return "Reject null hypothesis: " + nullHypothesis + " is not supported by the data (p = " + 
			   formatFloat(pValue, 4) + ", A² = " + formatFloat(A2Star, 4) + ")"
	} else {
		return "Fail to reject null hypothesis: " + nullHypothesis + " is consistent with the data (p = " + 
			   formatFloat(pValue, 4) + ", A² = " + formatFloat(A2Star, 4) + ")"
	}
}

// Utility functions

func calculateMean(data []float64) float64 {
	sum := 0.0
	for _, x := range data {
		sum += x
	}
	return sum / float64(len(data))
}

func calculateStdDev(data []float64, mean float64) float64 {
	sum := 0.0
	for _, x := range data {
		diff := x - mean
		sum += diff * diff
	}
	return math.Sqrt(sum / float64(len(data)-1))
}

func clamp(x, min, max float64) float64 {
	if x < min {
		return min
	}
	if x > max {
		return max
	}
	return x
}

func normalCDF(x float64) float64 {
	return 0.5 * (1 + math.Erf(x/math.Sqrt2))
}

func formatFloat(x float64, precision int) string {
	format := "%." + fmt.Sprintf("%d", precision) + "f"
	return fmt.Sprintf(format, x)
}

func convertToModelResult(result *AndersonDarlingTestResult) *models.StatisticalTestResult {
	return &models.StatisticalTestResult{
		TestName:      result.TestName,
		Statistic:     result.Statistic,
		PValue:        result.PValue,
		CriticalValue: result.CriticalValue,
		Passed:        !result.IsSignificant,
		Alpha:         result.AlphaLevel,
		Description:   result.Description,
		Metadata: map[string]interface{}{
			"interpretation":      result.Interpretation,
			"adjusted_statistic":  result.AdjustedStatistic,
			"sample_size":         result.SampleSize,
			"test_type":          result.TestType,
		},
	}
}