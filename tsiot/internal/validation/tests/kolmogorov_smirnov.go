package tests

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/inferloop/tsiot/pkg/models"
)

// KSTestResult contains detailed results of the Kolmogorov-Smirnov test
type KSTestResult struct {
	*StatisticalTestResult
	SampleSize1       int                    `json:"sample_size_1"`
	SampleSize2       int                    `json:"sample_size_2,omitempty"`
	MaxDifference     float64                `json:"max_difference"`
	DifferenceLocation float64               `json:"difference_location"`
	TestType          string                 `json:"test_type"` // "one_sample", "two_sample"
	Distribution      string                 `json:"distribution,omitempty"` // for one-sample tests
}

// Distribution types for one-sample KS test
const (
	DistributionNormal      = "normal"
	DistributionUniform     = "uniform"
	DistributionExponential = "exponential"
	DistributionLogNormal   = "lognormal"
)

// TwoSampleKSTest performs two-sample Kolmogorov-Smirnov test
func TwoSampleKSTest(sample1, sample2 []float64, alpha float64) (*models.StatisticalTestResult, error) {
	if len(sample1) < 5 || len(sample2) < 5 {
		return nil, errors.New("Kolmogorov-Smirnov test requires at least 5 observations in each sample")
	}

	result, err := performTwoSampleKSTest(sample1, sample2, alpha)
	if err != nil {
		return nil, err
	}

	return convertKSToModelResult(result), nil
}

// OneSampleKSTest performs one-sample Kolmogorov-Smirnov test against a theoretical distribution
func OneSampleKSTest(sample []float64, distribution string, alpha float64) (*models.StatisticalTestResult, error) {
	if len(sample) < 5 {
		return nil, errors.New("Kolmogorov-Smirnov test requires at least 5 observations")
	}

	result, err := performOneSampleKSTest(sample, distribution, alpha)
	if err != nil {
		return nil, err
	}

	return convertKSToModelResult(result), nil
}

// performTwoSampleKSTest is the core implementation for two-sample test
func performTwoSampleKSTest(sample1, sample2 []float64, alpha float64) (*KSTestResult, error) {
	n1, n2 := len(sample1), len(sample2)

	// Sort both samples
	sorted1 := make([]float64, n1)
	sorted2 := make([]float64, n2)
	copy(sorted1, sample1)
	copy(sorted2, sample2)
	sort.Float64s(sorted1)
	sort.Float64s(sorted2)

	// Calculate maximum difference between empirical CDFs
	maxDiff, diffLocation := calculateTwoSampleKSStatistic(sorted1, sorted2)

	// Calculate critical value
	criticalValue := calculateTwoSampleKSCriticalValue(n1, n2, alpha)

	// Calculate p-value
	pValue := calculateTwoSampleKSPValue(maxDiff, n1, n2)

	isSignificant := maxDiff > criticalValue

	// Create interpretation
	interpretation := generateTwoSampleKSInterpretation(isSignificant, maxDiff, pValue, n1, n2)

	result := &KSTestResult{
		StatisticalTestResult: &StatisticalTestResult{
			TestName:       "Two-Sample Kolmogorov-Smirnov Test",
			Statistic:      maxDiff,
			PValue:         pValue,
			CriticalValue:  criticalValue,
			IsSignificant:  isSignificant,
			AlphaLevel:     alpha,
			Description:    "Tests whether two independent samples come from the same distribution",
			Interpretation: interpretation,
		},
		SampleSize1:        n1,
		SampleSize2:        n2,
		MaxDifference:      maxDiff,
		DifferenceLocation: diffLocation,
		TestType:           "two_sample",
	}

	return result, nil
}

// performOneSampleKSTest is the core implementation for one-sample test
func performOneSampleKSTest(sample []float64, distribution string, alpha float64) (*KSTestResult, error) {
	n := len(sample)

	// Sort the sample
	sorted := make([]float64, n)
	copy(sorted, sample)
	sort.Float64s(sorted)

	// Calculate maximum difference between empirical and theoretical CDFs
	maxDiff, diffLocation, err := calculateOneSampleKSStatistic(sorted, distribution)
	if err != nil {
		return nil, err
	}

	// Calculate critical value
	criticalValue := calculateOneSampleKSCriticalValue(n, alpha)

	// Calculate p-value
	pValue := calculateOneSampleKSPValue(maxDiff, n)

	isSignificant := maxDiff > criticalValue

	// Create interpretation
	interpretation := generateOneSampleKSInterpretation(distribution, isSignificant, maxDiff, pValue, n)

	result := &KSTestResult{
		StatisticalTestResult: &StatisticalTestResult{
			TestName:       "One-Sample Kolmogorov-Smirnov Test",
			Statistic:      maxDiff,
			PValue:         pValue,
			CriticalValue:  criticalValue,
			IsSignificant:  isSignificant,
			AlphaLevel:     alpha,
			Description:    fmt.Sprintf("Tests goodness of fit to %s distribution", distribution),
			Interpretation: interpretation,
		},
		SampleSize1:        n,
		MaxDifference:      maxDiff,
		DifferenceLocation: diffLocation,
		TestType:           "one_sample",
		Distribution:       distribution,
	}

	return result, nil
}

// calculateTwoSampleKSStatistic calculates the KS statistic for two samples
func calculateTwoSampleKSStatistic(sorted1, sorted2 []float64) (float64, float64) {
	n1, n2 := len(sorted1), len(sorted2)
	var maxDiff float64
	var diffLocation float64

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
			diffLocation = x
		}
	}

	return maxDiff, diffLocation
}

// calculateOneSampleKSStatistic calculates the KS statistic for one sample against theoretical distribution
func calculateOneSampleKSStatistic(sorted []float64, distribution string) (float64, float64, error) {
	n := len(sorted)
	var maxDiff float64
	var diffLocation float64

	// Estimate parameters for the theoretical distribution
	params, err := estimateDistributionParameters(sorted, distribution)
	if err != nil {
		return 0, 0, err
	}

	for i, x := range sorted {
		empiricalCDF := float64(i+1) / float64(n)
		theoreticalCDF, err := calculateTheoreticalCDF(x, distribution, params)
		if err != nil {
			return 0, 0, err
		}

		diff := math.Abs(empiricalCDF - theoreticalCDF)
		if diff > maxDiff {
			maxDiff = diff
			diffLocation = x
		}
	}

	return maxDiff, diffLocation, nil
}

// estimateDistributionParameters estimates parameters for different distributions
func estimateDistributionParameters(data []float64, distribution string) (map[string]float64, error) {
	params := make(map[string]float64)

	switch distribution {
	case DistributionNormal:
		mean := calculateMean(data)
		stdDev := calculateStdDev(data, mean)
		params["mean"] = mean
		params["std"] = stdDev

	case DistributionUniform:
		params["min"] = data[0] // data is sorted
		params["max"] = data[len(data)-1]

	case DistributionExponential:
		// Check for positive values
		for _, x := range data {
			if x <= 0 {
				return nil, errors.New("exponential distribution requires positive values")
			}
		}
		mean := calculateMean(data)
		params["lambda"] = 1.0 / mean

	case DistributionLogNormal:
		// Check for positive values
		for _, x := range data {
			if x <= 0 {
				return nil, errors.New("log-normal distribution requires positive values")
			}
		}
		// Transform to log scale
		logData := make([]float64, len(data))
		for i, x := range data {
			logData[i] = math.Log(x)
		}
		logMean := calculateMean(logData)
		logStd := calculateStdDev(logData, logMean)
		params["mu"] = logMean
		params["sigma"] = logStd

	default:
		return nil, errors.New("unsupported distribution: " + distribution)
	}

	return params, nil
}

// calculateTheoreticalCDF calculates the CDF value for different distributions
func calculateTheoreticalCDF(x float64, distribution string, params map[string]float64) (float64, error) {
	switch distribution {
	case DistributionNormal:
		mean := params["mean"]
		std := params["std"]
		z := (x - mean) / std
		return normalCDF(z), nil

	case DistributionUniform:
		min := params["min"]
		max := params["max"]
		if x <= min {
			return 0, nil
		} else if x >= max {
			return 1, nil
		} else {
			return (x - min) / (max - min), nil
		}

	case DistributionExponential:
		lambda := params["lambda"]
		if x <= 0 {
			return 0, nil
		}
		return 1 - math.Exp(-lambda*x), nil

	case DistributionLogNormal:
		mu := params["mu"]
		sigma := params["sigma"]
		if x <= 0 {
			return 0, nil
		}
		z := (math.Log(x) - mu) / sigma
		return normalCDF(z), nil

	default:
		return 0, errors.New("unsupported distribution: " + distribution)
	}
}

// calculateTwoSampleKSCriticalValue calculates critical value for two-sample test
func calculateTwoSampleKSCriticalValue(n1, n2 int, alpha float64) float64 {
	// Critical value for two-sample KS test
	var cAlpha float64
	switch {
	case alpha <= 0.01:
		cAlpha = 1.63
	case alpha <= 0.05:
		cAlpha = 1.36
	case alpha <= 0.10:
		cAlpha = 1.22
	default:
		cAlpha = 1.36 // Default to 5% level
	}

	return cAlpha * math.Sqrt((float64(n1)+float64(n2))/(float64(n1)*float64(n2)))
}

// calculateOneSampleKSCriticalValue calculates critical value for one-sample test
func calculateOneSampleKSCriticalValue(n int, alpha float64) float64 {
	// Critical value for one-sample KS test
	var cAlpha float64
	switch {
	case alpha <= 0.01:
		cAlpha = 1.63
	case alpha <= 0.05:
		cAlpha = 1.36
	case alpha <= 0.10:
		cAlpha = 1.22
	default:
		cAlpha = 1.36 // Default to 5% level
	}

	return cAlpha / math.Sqrt(float64(n))
}

// calculateTwoSampleKSPValue calculates p-value for two-sample test
func calculateTwoSampleKSPValue(dMax float64, n1, n2 int) float64 {
	if dMax <= 0 {
		return 1.0
	}

	// Effective sample size
	ne := float64(n1*n2) / float64(n1+n2)
	lambda := dMax * math.Sqrt(ne)

	// Asymptotic approximation
	pValue := 2 * math.Exp(-2*lambda*lambda)

	// Ensure p-value is in [0, 1]
	return math.Max(0, math.Min(1, pValue))
}

// calculateOneSampleKSPValue calculates p-value for one-sample test
func calculateOneSampleKSPValue(dMax float64, n int) float64 {
	if dMax <= 0 {
		return 1.0
	}

	z := dMax * math.Sqrt(float64(n))
	
	// Asymptotic approximation
	pValue := 2 * math.Exp(-2*z*z)

	// Ensure p-value is in [0, 1]
	return math.Max(0, math.Min(1, pValue))
}

// generateTwoSampleKSInterpretation creates interpretation for two-sample test
func generateTwoSampleKSInterpretation(isSignificant bool, maxDiff, pValue float64, n1, n2 int) string {
	if isSignificant {
		return fmt.Sprintf("Reject null hypothesis: The two samples come from different distributions "+
			"(D = %.4f, p = %.4f, n1 = %d, n2 = %d)", maxDiff, pValue, n1, n2)
	} else {
		return fmt.Sprintf("Fail to reject null hypothesis: The two samples appear to come from the same distribution "+
			"(D = %.4f, p = %.4f, n1 = %d, n2 = %d)", maxDiff, pValue, n1, n2)
	}
}

// generateOneSampleKSInterpretation creates interpretation for one-sample test
func generateOneSampleKSInterpretation(distribution string, isSignificant bool, maxDiff, pValue float64, n int) string {
	if isSignificant {
		return fmt.Sprintf("Reject null hypothesis: The sample does not follow a %s distribution "+
			"(D = %.4f, p = %.4f, n = %d)", distribution, maxDiff, pValue, n)
	} else {
		return fmt.Sprintf("Fail to reject null hypothesis: The sample is consistent with a %s distribution "+
			"(D = %.4f, p = %.4f, n = %d)", distribution, maxDiff, pValue, n)
	}
}

// convertKSToModelResult converts internal result to model result
func convertKSToModelResult(result *KSTestResult) *models.StatisticalTestResult {
	metadata := map[string]interface{}{
		"interpretation":       result.Interpretation,
		"sample_size_1":        result.SampleSize1,
		"max_difference":       result.MaxDifference,
		"difference_location":  result.DifferenceLocation,
		"test_type":           result.TestType,
	}

	if result.SampleSize2 > 0 {
		metadata["sample_size_2"] = result.SampleSize2
	}
	if result.Distribution != "" {
		metadata["distribution"] = result.Distribution
	}

	return &models.StatisticalTestResult{
		TestName:      result.TestName,
		Statistic:     result.Statistic,
		PValue:        result.PValue,
		CriticalValue: result.CriticalValue,
		Passed:        !result.IsSignificant,
		Alpha:         result.AlphaLevel,
		Description:   result.Description,
		Metadata:      metadata,
	}
}

// KSTestMultipleDistributions tests sample against multiple distributions
func KSTestMultipleDistributions(sample []float64, alpha float64) ([]*models.StatisticalTestResult, error) {
	distributions := []string{DistributionNormal, DistributionUniform, DistributionExponential}
	
	// Only test log-normal if all values are positive
	allPositive := true
	for _, x := range sample {
		if x <= 0 {
			allPositive = false
			break
		}
	}
	if allPositive {
		distributions = append(distributions, DistributionLogNormal)
	}

	var results []*models.StatisticalTestResult
	for _, dist := range distributions {
		result, err := OneSampleKSTest(sample, dist, alpha)
		if err == nil {
			results = append(results, result)
		}
	}

	return results, nil
}

// KSTestGoodnesOfFit performs comprehensive goodness-of-fit testing
func KSTestGoodnesOfFit(sample []float64, alpha float64) (*models.StatisticalTestResult, error) {
	results, err := KSTestMultipleDistributions(sample, alpha)
	if err != nil {
		return nil, err
	}

	if len(results) == 0 {
		return nil, errors.New("no valid distribution tests could be performed")
	}

	// Find the best-fitting distribution (highest p-value among non-rejected)
	var bestResult *models.StatisticalTestResult
	var bestPValue float64 = -1

	for _, result := range results {
		if result.Passed && result.PValue > bestPValue {
			bestPValue = result.PValue
			bestResult = result
		}
	}

	// If no distribution fits, return the one with highest p-value
	if bestResult == nil {
		for _, result := range results {
			if result.PValue > bestPValue {
				bestPValue = result.PValue
				bestResult = result
			}
		}
	}

	// Modify the result to indicate it's a comprehensive test
	bestResult.TestName = "Kolmogorov-Smirnov Goodness-of-Fit Test"
	bestResult.Description = "Comprehensive goodness-of-fit test against multiple distributions"
	
	if bestResult.Metadata == nil {
		bestResult.Metadata = make(map[string]interface{})
	}
	bestResult.Metadata["tested_distributions"] = len(results)
	bestResult.Metadata["all_results"] = results

	return bestResult, nil
}

// KSTestComparison performs KS test with additional diagnostic information
func KSTestComparison(sample1, sample2 []float64, alpha float64) (*models.StatisticalTestResult, error) {
	result, err := TwoSampleKSTest(sample1, sample2, alpha)
	if err != nil {
		return nil, err
	}

	// Add additional diagnostic information
	if result.Metadata == nil {
		result.Metadata = make(map[string]interface{})
	}

	// Calculate basic statistics for both samples
	mean1 := calculateMean(sample1)
	mean2 := calculateMean(sample2)
	std1 := calculateStdDev(sample1, mean1)
	std2 := calculateStdDev(sample2, mean2)

	result.Metadata["sample1_mean"] = mean1
	result.Metadata["sample1_std"] = std1
	result.Metadata["sample2_mean"] = mean2
	result.Metadata["sample2_std"] = std2
	result.Metadata["mean_difference"] = math.Abs(mean1 - mean2)
	result.Metadata["std_difference"] = math.Abs(std1 - std2)

	return result, nil
}