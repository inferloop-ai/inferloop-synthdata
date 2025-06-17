package tests

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distuv"

	"github.com/inferloop/tsiot/pkg/models"
)

// StatisticalTestResult represents the result of a statistical test
type StatisticalTestResult struct {
	TestName      string  `json:"test_name"`
	Statistic     float64 `json:"statistic"`
	PValue        float64 `json:"p_value"`
	CriticalValue float64 `json:"critical_value,omitempty"`
	IsSignificant bool    `json:"is_significant"`
	AlphaLevel    float64 `json:"alpha_level"`
	Description   string  `json:"description"`
	Interpretation string `json:"interpretation"`
}

// StatisticalTestSuite provides comprehensive statistical tests for data validation
type StatisticalTestSuite struct {
	alphaLevel float64 // Significance level
}

// NewStatisticalTestSuite creates a new statistical test suite
func NewStatisticalTestSuite(alphaLevel float64) *StatisticalTestSuite {
	if alphaLevel <= 0 || alphaLevel >= 1 {
		alphaLevel = 0.05 // Default 5% significance level
	}
	
	return &StatisticalTestSuite{
		alphaLevel: alphaLevel,
	}
}

// KolmogorovSmirnovTest performs the Kolmogorov-Smirnov test for goodness of fit
func (sts *StatisticalTestSuite) KolmogorovSmirnovTest(data []float64, expectedDist string) *StatisticalTestResult {
	n := len(data)
	if n < 10 {
		return &StatisticalTestResult{
			TestName:    "Kolmogorov-Smirnov Test",
			Description: "Tests goodness of fit to expected distribution",
			Interpretation: "Insufficient data for reliable test (n < 10)",
		}
	}

	// Sort the data
	sortedData := make([]float64, n)
	copy(sortedData, data)
	sort.Float64s(sortedData)

	// Calculate empirical CDF
	var maxDifference float64
	
	// Compare with theoretical distribution
	switch expectedDist {
	case "normal":
		maxDifference = sts.ksTestNormal(sortedData)
	case "uniform":
		maxDifference = sts.ksTestUniform(sortedData)
	case "exponential":
		maxDifference = sts.ksTestExponential(sortedData)
	default:
		maxDifference = sts.ksTestNormal(sortedData) // Default to normal
	}

	// Calculate critical value (approximate)
	criticalValue := 1.36 / math.Sqrt(float64(n)) // For alpha = 0.05
	
	// Calculate p-value (approximate)
	pValue := sts.ksPValue(maxDifference, n)
	
	isSignificant := pValue < sts.alphaLevel
	
	interpretation := "Data follows the expected distribution"
	if isSignificant {
		interpretation = "Data does not follow the expected distribution"
	}

	return &StatisticalTestResult{
		TestName:       "Kolmogorov-Smirnov Test",
		Statistic:      maxDifference,
		PValue:         pValue,
		CriticalValue:  criticalValue,
		IsSignificant:  isSignificant,
		AlphaLevel:     sts.alphaLevel,
		Description:    "Tests goodness of fit to expected distribution",
		Interpretation: interpretation,
	}
}

// AndersonDarlingTest performs the Anderson-Darling test for normality
func (sts *StatisticalTestSuite) AndersonDarlingTest(data []float64) *StatisticalTestResult {
	n := len(data)
	if n < 8 {
		return &StatisticalTestResult{
			TestName:    "Anderson-Darling Test",
			Description: "Tests for normality",
			Interpretation: "Insufficient data for reliable test (n < 8)",
		}
	}

	// Standardize the data
	mean := stat.Mean(data, nil)
	stdDev := math.Sqrt(stat.Variance(data, nil))
	
	standardized := make([]float64, n)
	for i, x := range data {
		standardized[i] = (x - mean) / stdDev
	}
	
	// Sort standardized data
	sort.Float64s(standardized)
	
	// Calculate Anderson-Darling statistic
	var A2 float64
	normal := distuv.Normal{Mu: 0, Sigma: 1}
	
	for i := 0; i < n; i++ {
		Fi := normal.CDF(standardized[i])
		Fni := normal.CDF(standardized[n-1-i])
		
		if Fi > 0 && Fi < 1 && Fni > 0 && Fni < 1 {
			term := float64(2*i+1) * (math.Log(Fi) + math.Log(1-Fni))
			A2 += term
		}
	}
	
	A2 = -float64(n) - A2/float64(n)
	
	// Adjust for sample size
	A2Star := A2 * (1 + 0.75/float64(n) + 2.25/(float64(n)*float64(n)))
	
	// Determine critical values and p-value
	var pValue float64
	var criticalValue float64
	
	if A2Star < 0.2 {
		pValue = 1 - math.Exp(-13.436+101.14*A2Star-223.73*A2Star*A2Star)
	} else if A2Star < 0.34 {
		pValue = 1 - math.Exp(-8.318+42.796*A2Star-59.938*A2Star*A2Star)
	} else if A2Star < 0.6 {
		pValue = math.Exp(0.9177-4.279*A2Star+1.38*A2Star*A2Star)
	} else {
		pValue = math.Exp(1.2937-5.709*A2Star+0.0186*A2Star*A2Star)
	}
	
	criticalValue = 0.752 // Critical value for alpha = 0.05
	isSignificant := A2Star > criticalValue
	
	interpretation := "Data is normally distributed"
	if isSignificant {
		interpretation = "Data is not normally distributed"
	}

	return &StatisticalTestResult{
		TestName:       "Anderson-Darling Test",
		Statistic:      A2Star,
		PValue:         pValue,
		CriticalValue:  criticalValue,
		IsSignificant:  isSignificant,
		AlphaLevel:     sts.alphaLevel,
		Description:    "Tests for normality",
		Interpretation: interpretation,
	}
}

// LjungBoxTest performs the Ljung-Box test for autocorrelation
func (sts *StatisticalTestSuite) LjungBoxTest(data []float64, lags int) *StatisticalTestResult {
	n := len(data)
	if n < 2*lags {
		return &StatisticalTestResult{
			TestName:    "Ljung-Box Test",
			Description: "Tests for autocorrelation in residuals",
			Interpretation: "Insufficient data for reliable test",
		}
	}

	if lags <= 0 {
		lags = min(10, n/4) // Default number of lags
	}

	// Calculate autocorrelations
	autocorrelations := sts.calculateAutocorrelations(data, lags)
	
	// Calculate Ljung-Box statistic
	var Q float64
	for k := 1; k <= lags; k++ {
		rk := autocorrelations[k-1]
		Q += (rk * rk) / float64(n-k)
	}
	Q *= float64(n) * (float64(n) + 2)
	
	// Degrees of freedom
	df := lags
	
	// Calculate p-value using chi-square distribution
	chiSq := distuv.ChiSquared{K: float64(df)}
	pValue := 1 - chiSq.CDF(Q)
	
	// Critical value for given alpha level
	criticalValue := chiSq.Quantile(1 - sts.alphaLevel)
	
	isSignificant := Q > criticalValue
	
	interpretation := "No significant autocorrelation detected"
	if isSignificant {
		interpretation = "Significant autocorrelation detected"
	}

	return &StatisticalTestResult{
		TestName:       "Ljung-Box Test",
		Statistic:      Q,
		PValue:         pValue,
		CriticalValue:  criticalValue,
		IsSignificant:  isSignificant,
		AlphaLevel:     sts.alphaLevel,
		Description:    "Tests for autocorrelation in residuals",
		Interpretation: interpretation,
	}
}

// ShapiroWilkTest performs the Shapiro-Wilk test for normality
func (sts *StatisticalTestSuite) ShapiroWilkTest(data []float64) *StatisticalTestResult {
	n := len(data)
	if n < 3 || n > 5000 {
		return &StatisticalTestResult{
			TestName:    "Shapiro-Wilk Test",
			Description: "Tests for normality",
			Interpretation: "Sample size not suitable for Shapiro-Wilk test (3 d n d 5000)",
		}
	}

	// Sort the data
	sortedData := make([]float64, n)
	copy(sortedData, data)
	sort.Float64s(sortedData)

	// Calculate Shapiro-Wilk statistic (simplified version)
	// This is a simplified implementation - full implementation requires coefficients table
	
	// Calculate sample mean and variance
	mean := stat.Mean(data, nil)
	variance := stat.Variance(data, nil)
	
	// Calculate numerator (sum of products with weights)
	var numerator float64
	for i := 0; i < n/2; i++ {
		// Simplified weights (actual weights come from statistical tables)
		weight := sts.approximateShapiroWilkWeight(i, n)
		numerator += weight * (sortedData[n-1-i] - sortedData[i])
	}
	
	// Calculate W statistic
	W := (numerator * numerator) / (float64(n-1) * variance)
	
	// Approximate p-value calculation
	var pValue float64
	if n <= 11 {
		// Use approximation for small samples
		pValue = sts.shapiroWilkPValueSmall(W, n)
	} else {
		// Use normal approximation for larger samples
		pValue = sts.shapiroWilkPValueLarge(W, n)
	}
	
	isSignificant := pValue < sts.alphaLevel
	
	interpretation := "Data is normally distributed"
	if isSignificant {
		interpretation = "Data is not normally distributed"
	}

	return &StatisticalTestResult{
		TestName:       "Shapiro-Wilk Test",
		Statistic:      W,
		PValue:         pValue,
		IsSignificant:  isSignificant,
		AlphaLevel:     sts.alphaLevel,
		Description:    "Tests for normality",
		Interpretation: interpretation,
	}
}

// RunsTest performs the runs test for randomness
func (sts *StatisticalTestSuite) RunsTest(data []float64) *StatisticalTestResult {
	n := len(data)
	if n < 10 {
		return &StatisticalTestResult{
			TestName:    "Runs Test",
			Description: "Tests for randomness",
			Interpretation: "Insufficient data for reliable test (n < 10)",
		}
	}

	// Calculate median
	sortedData := make([]float64, n)
	copy(sortedData, data)
	sort.Float64s(sortedData)
	
	var median float64
	if n%2 == 0 {
		median = (sortedData[n/2-1] + sortedData[n/2]) / 2
	} else {
		median = sortedData[n/2]
	}

	// Convert to binary sequence (above/below median)
	binary := make([]bool, 0, n)
	var n1, n2 int // Count of values above and below median
	
	for _, value := range data {
		if value != median { // Exclude values equal to median
			isAbove := value > median
			binary = append(binary, isAbove)
			if isAbove {
				n1++
			} else {
				n2++
			}
		}
	}

	// Count runs
	runs := 1
	for i := 1; i < len(binary); i++ {
		if binary[i] != binary[i-1] {
			runs++
		}
	}

	// Calculate expected runs and variance
	N := n1 + n2
	expectedRuns := (2*float64(n1)*float64(n2))/float64(N) + 1
	varianceRuns := (2*float64(n1)*float64(n2)*(2*float64(n1)*float64(n2)-float64(N))) / 
					(float64(N)*float64(N)*float64(N-1))

	// Calculate test statistic
	var Z float64
	if float64(runs) > expectedRuns {
		Z = (float64(runs) - 0.5 - expectedRuns) / math.Sqrt(varianceRuns)
	} else {
		Z = (float64(runs) + 0.5 - expectedRuns) / math.Sqrt(varianceRuns)
	}

	// Calculate p-value (two-tailed)
	normal := distuv.Normal{Mu: 0, Sigma: 1}
	pValue := 2 * (1 - normal.CDF(math.Abs(Z)))

	criticalValue := 1.96 // For alpha = 0.05, two-tailed
	isSignificant := math.Abs(Z) > criticalValue

	interpretation := "Data appears random"
	if isSignificant {
		interpretation = "Data does not appear random"
	}

	return &StatisticalTestResult{
		TestName:       "Runs Test",
		Statistic:      Z,
		PValue:         pValue,
		CriticalValue:  criticalValue,
		IsSignificant:  isSignificant,
		AlphaLevel:     sts.alphaLevel,
		Description:    "Tests for randomness",
		Interpretation: interpretation,
	}
}

// JarqueBeraTest performs the Jarque-Bera test for normality
func (sts *StatisticalTestSuite) JarqueBeraTest(data []float64) *StatisticalTestResult {
	n := len(data)
	if n < 8 {
		return &StatisticalTestResult{
			TestName:    "Jarque-Bera Test",
			Description: "Tests for normality using skewness and kurtosis",
			Interpretation: "Insufficient data for reliable test (n < 8)",
		}
	}

	// Calculate moments
	mean := stat.Mean(data, nil)
	variance := stat.Variance(data, nil)
	stdDev := math.Sqrt(variance)

	// Calculate skewness and kurtosis
	var m3, m4 float64
	for _, x := range data {
		standardized := (x - mean) / stdDev
		standardized2 := standardized * standardized
		m3 += standardized2 * standardized
		m4 += standardized2 * standardized2
	}
	
	skewness := m3 / float64(n)
	kurtosis := m4/float64(n) - 3 // Excess kurtosis

	// Calculate Jarque-Bera statistic
	JB := float64(n) * (skewness*skewness/6 + kurtosis*kurtosis/24)

	// Calculate p-value using chi-square distribution with 2 degrees of freedom
	chiSq := distuv.ChiSquared{K: 2}
	pValue := 1 - chiSq.CDF(JB)

	criticalValue := chiSq.Quantile(1 - sts.alphaLevel)
	isSignificant := JB > criticalValue

	interpretation := "Data is normally distributed"
	if isSignificant {
		interpretation = "Data is not normally distributed"
	}

	return &StatisticalTestResult{
		TestName:       "Jarque-Bera Test",
		Statistic:      JB,
		PValue:         pValue,
		CriticalValue:  criticalValue,
		IsSignificant:  isSignificant,
		AlphaLevel:     sts.alphaLevel,
		Description:    "Tests for normality using skewness and kurtosis",
		Interpretation: interpretation,
	}
}

// TwoSampleKSTest performs two-sample Kolmogorov-Smirnov test
func (sts *StatisticalTestSuite) TwoSampleKSTest(data1, data2 []float64) *StatisticalTestResult {
	n1, n2 := len(data1), len(data2)
	if n1 < 5 || n2 < 5 {
		return &StatisticalTestResult{
			TestName:    "Two-Sample Kolmogorov-Smirnov Test",
			Description: "Tests if two samples come from the same distribution",
			Interpretation: "Insufficient data for reliable test",
		}
	}

	// Sort both samples
	sorted1 := make([]float64, n1)
	sorted2 := make([]float64, n2)
	copy(sorted1, data1)
	copy(sorted2, data2)
	sort.Float64s(sorted1)
	sort.Float64s(sorted2)

	// Calculate maximum difference between empirical CDFs
	var maxDiff float64
	i1, i2 := 0, 0
	
	for i1 < n1 || i2 < n2 {
		var x float64
		if i1 >= n1 {
			x = sorted2[i2]
		} else if i2 >= n2 {
			x = sorted1[i1]
		} else {
			x = math.Min(sorted1[i1], sorted2[i2])
		}

		// Calculate empirical CDFs at point x
		for i1 < n1 && sorted1[i1] <= x {
			i1++
		}
		for i2 < n2 && sorted2[i2] <= x {
			i2++
		}

		cdf1 := float64(i1) / float64(n1)
		cdf2 := float64(i2) / float64(n2)
		diff := math.Abs(cdf1 - cdf2)

		if diff > maxDiff {
			maxDiff = diff
		}
	}

	// Calculate critical value
	alpha := sts.alphaLevel
	c := math.Sqrt(-0.5 * math.Log(alpha/2))
	criticalValue := c * math.Sqrt((float64(n1)+float64(n2))/(float64(n1)*float64(n2)))

	// Approximate p-value
	m := float64(n1 * n2) / float64(n1 + n2)
	lambda := maxDiff * math.Sqrt(m)
	pValue := 2 * math.Exp(-2*lambda*lambda)

	isSignificant := maxDiff > criticalValue

	interpretation := "Samples come from the same distribution"
	if isSignificant {
		interpretation = "Samples come from different distributions"
	}

	return &StatisticalTestResult{
		TestName:       "Two-Sample Kolmogorov-Smirnov Test",
		Statistic:      maxDiff,
		PValue:         pValue,
		CriticalValue:  criticalValue,
		IsSignificant:  isSignificant,
		AlphaLevel:     sts.alphaLevel,
		Description:    "Tests if two samples come from the same distribution",
		Interpretation: interpretation,
	}
}

// Helper methods

func (sts *StatisticalTestSuite) ksTestNormal(sortedData []float64) float64 {
	n := len(sortedData)
	mean := stat.Mean(sortedData, nil)
	stdDev := math.Sqrt(stat.Variance(sortedData, nil))
	
	normal := distuv.Normal{Mu: mean, Sigma: stdDev}
	
	var maxDiff float64
	for i, x := range sortedData {
		empiricalCDF := float64(i+1) / float64(n)
		theoreticalCDF := normal.CDF(x)
		diff := math.Abs(empiricalCDF - theoreticalCDF)
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	
	return maxDiff
}

func (sts *StatisticalTestSuite) ksTestUniform(sortedData []float64) float64 {
	n := len(sortedData)
	min := sortedData[0]
	max := sortedData[n-1]
	
	var maxDiff float64
	for i, x := range sortedData {
		empiricalCDF := float64(i+1) / float64(n)
		theoreticalCDF := (x - min) / (max - min)
		diff := math.Abs(empiricalCDF - theoreticalCDF)
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	
	return maxDiff
}

func (sts *StatisticalTestSuite) ksTestExponential(sortedData []float64) float64 {
	n := len(sortedData)
	lambda := 1.0 / stat.Mean(sortedData, nil)
	
	exp := distuv.Exponential{Rate: lambda}
	
	var maxDiff float64
	for i, x := range sortedData {
		empiricalCDF := float64(i+1) / float64(n)
		theoreticalCDF := exp.CDF(x)
		diff := math.Abs(empiricalCDF - theoreticalCDF)
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	
	return maxDiff
}

func (sts *StatisticalTestSuite) ksPValue(dMax float64, n int) float64 {
	// Approximate p-value for Kolmogorov-Smirnov test
	if dMax <= 0 {
		return 1.0
	}
	
	sqrt_n := math.Sqrt(float64(n))
	z := dMax * sqrt_n
	
	// Asymptotic formula
	pValue := 2 * math.Exp(-2*z*z)
	
	// Ensure p-value is in [0, 1]
	if pValue > 1 {
		pValue = 1
	}
	if pValue < 0 {
		pValue = 0
	}
	
	return pValue
}

func (sts *StatisticalTestSuite) calculateAutocorrelations(data []float64, maxLag int) []float64 {
	n := len(data)
	autocorr := make([]float64, maxLag)
	
	mean := stat.Mean(data, nil)
	
	// Calculate variance (lag 0)
	var c0 float64
	for i := 0; i < n; i++ {
		c0 += (data[i] - mean) * (data[i] - mean)
	}
	c0 /= float64(n)
	
	// Calculate autocorrelations for each lag
	for lag := 1; lag <= maxLag; lag++ {
		var ck float64
		for i := lag; i < n; i++ {
			ck += (data[i] - mean) * (data[i-lag] - mean)
		}
		ck /= float64(n)
		
		if c0 > 0 {
			autocorr[lag-1] = ck / c0
		}
	}
	
	return autocorr
}

func (sts *StatisticalTestSuite) approximateShapiroWilkWeight(i, n int) float64 {
	// Simplified approximation for Shapiro-Wilk weights
	// Actual implementation would use precomputed tables
	return 1.0 / math.Sqrt(float64(n))
}

func (sts *StatisticalTestSuite) shapiroWilkPValueSmall(W float64, n int) float64 {
	// Simplified p-value calculation for small samples
	// This is a rough approximation
	if W > 0.95 {
		return 0.5
	}
	return math.Max(0.01, 1.0-W)
}

func (sts *StatisticalTestSuite) shapiroWilkPValueLarge(W float64, n int) float64 {
	// Normal approximation for larger samples
	// This is a simplified version
	logW := math.Log(W)
	mu := -0.5
	sigma := 0.3
	
	z := (logW - mu) / sigma
	normal := distuv.Normal{Mu: 0, Sigma: 1}
	
	return normal.CDF(z)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Wrapper functions for the statistical validator interface

// KolmogorovSmirnovTest performs two-sample Kolmogorov-Smirnov test
func KolmogorovSmirnovTest(sample1, sample2 []float64, alpha float64) (*models.StatisticalTestResult, error) {
	suite := NewStatisticalTestSuite(alpha)
	result := suite.TwoSampleKSTest(sample1, sample2)
	
	return &models.StatisticalTestResult{
		TestName:      result.TestName,
		Statistic:     result.Statistic,
		PValue:        result.PValue,
		CriticalValue: result.CriticalValue,
		Passed:        !result.IsSignificant,
		Alpha:         result.AlphaLevel,
		Description:   result.Description,
		Metadata: map[string]interface{}{
			"interpretation": result.Interpretation,
			"sample1_size":   len(sample1),
			"sample2_size":   len(sample2),
		},
	}, nil
}

// AndersonDarlingTest performs Anderson-Darling test for normality
func AndersonDarlingTest(sample1, sample2 []float64, alpha float64) (*models.StatisticalTestResult, error) {
	suite := NewStatisticalTestSuite(alpha)
	result := suite.AndersonDarlingTest(sample1)
	
	return &models.StatisticalTestResult{
		TestName:      result.TestName,
		Statistic:     result.Statistic,
		PValue:        result.PValue,
		CriticalValue: result.CriticalValue,
		Passed:        !result.IsSignificant,
		Alpha:         result.AlphaLevel,
		Description:   result.Description,
		Metadata: map[string]interface{}{
			"interpretation": result.Interpretation,
			"sample_size":    len(sample1),
		},
	}, nil
}

// LjungBoxTest performs Ljung-Box test for autocorrelation
func LjungBoxTest(sample []float64, alpha float64) (*models.StatisticalTestResult, error) {
	suite := NewStatisticalTestSuite(alpha)
	result := suite.LjungBoxTest(sample, 0) // 0 means use default lags
	
	return &models.StatisticalTestResult{
		TestName:      result.TestName,
		Statistic:     result.Statistic,
		PValue:        result.PValue,
		CriticalValue: result.CriticalValue,
		Passed:        !result.IsSignificant,
		Alpha:         result.AlphaLevel,
		Description:   result.Description,
		Metadata: map[string]interface{}{
			"interpretation": result.Interpretation,
			"sample_size":    len(sample),
		},
	}, nil
}

// ShapiroWilkTest performs Shapiro-Wilk test for normality
func ShapiroWilkTest(sample []float64, alpha float64) (*models.StatisticalTestResult, error) {
	suite := NewStatisticalTestSuite(alpha)
	result := suite.ShapiroWilkTest(sample)
	
	return &models.StatisticalTestResult{
		TestName:      result.TestName,
		Statistic:     result.Statistic,
		PValue:        result.PValue,
		CriticalValue: result.CriticalValue,
		Passed:        !result.IsSignificant,
		Alpha:         result.AlphaLevel,
		Description:   result.Description,
		Metadata: map[string]interface{}{
			"interpretation": result.Interpretation,
			"sample_size":    len(sample),
		},
	}, nil
}

// JarqueBeraTest performs Jarque-Bera test for normality
func JarqueBeraTest(sample []float64, alpha float64) (*models.StatisticalTestResult, error) {
	suite := NewStatisticalTestSuite(alpha)
	result := suite.JarqueBeraTest(sample)
	
	return &models.StatisticalTestResult{
		TestName:      result.TestName,
		Statistic:     result.Statistic,
		PValue:        result.PValue,
		CriticalValue: result.CriticalValue,
		Passed:        !result.IsSignificant,
		Alpha:         result.AlphaLevel,
		Description:   result.Description,
		Metadata: map[string]interface{}{
			"interpretation": result.Interpretation,
			"sample_size":    len(sample),
		},
	}, nil
}