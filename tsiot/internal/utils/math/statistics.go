package math

import (
	"math"
	"sort"
)

// Mean calculates the arithmetic mean of a slice of float64 values
func Mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// Median calculates the median of a slice of float64 values
func Median(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	
	n := len(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2
	}
	return sorted[n/2]
}

// Mode calculates the mode (most frequent value) of a slice of float64 values
func Mode(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	frequency := make(map[float64]int)
	for _, v := range values {
		frequency[v]++
	}
	
	var mode float64
	maxFreq := 0
	for val, freq := range frequency {
		if freq > maxFreq {
			maxFreq = freq
			mode = val
		}
	}
	
	return mode
}

// Variance calculates the variance of a slice of float64 values
func Variance(values []float64) float64 {
	if len(values) <= 1 {
		return 0
	}
	
	mean := Mean(values)
	sumSquaredDiff := 0.0
	
	for _, v := range values {
		diff := v - mean
		sumSquaredDiff += diff * diff
	}
	
	return sumSquaredDiff / float64(len(values)-1)
}

// StandardDeviation calculates the standard deviation of a slice of float64 values
func StandardDeviation(values []float64) float64 {
	return math.Sqrt(Variance(values))
}

// Skewness calculates the skewness (asymmetry) of a distribution
func Skewness(values []float64) float64 {
	if len(values) < 3 {
		return 0
	}
	
	mean := Mean(values)
	std := StandardDeviation(values)
	
	if std == 0 {
		return 0
	}
	
	sum := 0.0
	n := float64(len(values))
	
	for _, v := range values {
		standardized := (v - mean) / std
		sum += standardized * standardized * standardized
	}
	
	return (n / ((n - 1) * (n - 2))) * sum
}

// Kurtosis calculates the kurtosis (tail heaviness) of a distribution
func Kurtosis(values []float64) float64 {
	if len(values) < 4 {
		return 0
	}
	
	mean := Mean(values)
	std := StandardDeviation(values)
	
	if std == 0 {
		return 0
	}
	
	sum := 0.0
	n := float64(len(values))
	
	for _, v := range values {
		standardized := (v - mean) / std
		sum += math.Pow(standardized, 4)
	}
	
	kurtosis := (n*(n+1))/((n-1)*(n-2)*(n-3))*sum - (3*(n-1)*(n-1))/((n-2)*(n-3))
	return kurtosis
}

// Correlation calculates the Pearson correlation coefficient between two variables
func Correlation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0
	}
	
	meanX := Mean(x)
	meanY := Mean(y)
	
	numerator := 0.0
	sumXSq := 0.0
	sumYSq := 0.0
	
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

// Covariance calculates the covariance between two variables
func Covariance(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0
	}
	
	meanX := Mean(x)
	meanY := Mean(y)
	
	sum := 0.0
	for i := 0; i < len(x); i++ {
		sum += (x[i] - meanX) * (y[i] - meanY)
	}
	
	return sum / float64(len(x)-1)
}

// Percentile calculates the p-th percentile of a slice of values
func Percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	if p < 0 || p > 100 {
		return 0
	}
	
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	
	if p == 0 {
		return sorted[0]
	}
	if p == 100 {
		return sorted[len(sorted)-1]
	}
	
	index := (p / 100.0) * float64(len(sorted)-1)
	lower := int(math.Floor(index))
	upper := int(math.Ceil(index))
	
	if lower == upper {
		return sorted[lower]
	}
	
	weight := index - float64(lower)
	return sorted[lower]*(1-weight) + sorted[upper]*weight
}

// Quantile calculates quantiles (quartiles, quintiles, etc.)
func Quantile(values []float64, q float64) float64 {
	return Percentile(values, q*100)
}

// IQR calculates the Interquartile Range (Q3 - Q1)
func IQR(values []float64) float64 {
	q1 := Quantile(values, 0.25)
	q3 := Quantile(values, 0.75)
	return q3 - q1
}

// OutlierBounds calculates the lower and upper bounds for outlier detection using IQR method
func OutlierBounds(values []float64) (lower, upper float64) {
	q1 := Quantile(values, 0.25)
	q3 := Quantile(values, 0.75)
	iqr := q3 - q1
	
	lower = q1 - 1.5*iqr
	upper = q3 + 1.5*iqr
	return lower, upper
}

// DetectOutliers returns indices of outliers using the IQR method
func DetectOutliers(values []float64) []int {
	lower, upper := OutlierBounds(values)
	
	var outliers []int
	for i, v := range values {
		if v < lower || v > upper {
			outliers = append(outliers, i)
		}
	}
	
	return outliers
}

// ZScore calculates the z-score for each value
func ZScore(values []float64) []float64 {
	if len(values) == 0 {
		return nil
	}
	
	mean := Mean(values)
	std := StandardDeviation(values)
	
	if std == 0 {
		return make([]float64, len(values))
	}
	
	zScores := make([]float64, len(values))
	for i, v := range values {
		zScores[i] = (v - mean) / std
	}
	
	return zScores
}

// MovingAverage calculates simple moving average with given window size
func MovingAverage(values []float64, window int) []float64 {
	if window <= 0 || len(values) == 0 {
		return nil
	}
	
	if window > len(values) {
		window = len(values)
	}
	
	result := make([]float64, len(values)-window+1)
	
	for i := 0; i <= len(values)-window; i++ {
		sum := 0.0
		for j := i; j < i+window; j++ {
			sum += values[j]
		}
		result[i] = sum / float64(window)
	}
	
	return result
}

// ExponentialMovingAverage calculates exponential moving average
func ExponentialMovingAverage(values []float64, alpha float64) []float64 {
	if len(values) == 0 || alpha <= 0 || alpha > 1 {
		return nil
	}
	
	result := make([]float64, len(values))
	result[0] = values[0]
	
	for i := 1; i < len(values); i++ {
		result[i] = alpha*values[i] + (1-alpha)*result[i-1]
	}
	
	return result
}

// CumulativeSum calculates cumulative sum
func CumulativeSum(values []float64) []float64 {
	if len(values) == 0 {
		return nil
	}
	
	result := make([]float64, len(values))
	result[0] = values[0]
	
	for i := 1; i < len(values); i++ {
		result[i] = result[i-1] + values[i]
	}
	
	return result
}

// Diff calculates the difference between consecutive elements
func Diff(values []float64) []float64 {
	if len(values) < 2 {
		return nil
	}
	
	result := make([]float64, len(values)-1)
	for i := 1; i < len(values); i++ {
		result[i-1] = values[i] - values[i-1]
	}
	
	return result
}

// AutoCorrelation calculates autocorrelation at given lag
func AutoCorrelation(values []float64, lag int) float64 {
	if lag >= len(values) || lag < 0 {
		return 0
	}
	
	n := len(values) - lag
	if n <= 1 {
		return 0
	}
	
	x1 := values[:n]
	x2 := values[lag : lag+n]
	
	return Correlation(x1, x2)
}

// CrossCorrelation calculates cross-correlation between two series at given lag
func CrossCorrelation(x, y []float64, lag int) float64 {
	if len(x) != len(y) || lag >= len(x) || lag < -len(x) {
		return 0
	}
	
	var x1, y1 []float64
	
	if lag >= 0 {
		x1 = x[:len(x)-lag]
		y1 = y[lag:]
	} else {
		x1 = x[-lag:]
		y1 = y[:len(y)+lag]
	}
	
	return Correlation(x1, y1)
}