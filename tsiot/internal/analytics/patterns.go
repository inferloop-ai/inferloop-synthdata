package analytics

import (
	"fmt"
	"math"
	"sort"
)

// PatternDetectorRegistry manages pattern detectors
type PatternDetectorRegistry struct {
	detectors map[string]PatternDetector
}

// NewPatternDetectorRegistry creates a new pattern detector registry
func NewPatternDetectorRegistry() *PatternDetectorRegistry {
	registry := &PatternDetectorRegistry{
		detectors: make(map[string]PatternDetector),
	}

	// Register default detectors
	registry.Register(&TrendPatternDetector{})
	registry.Register(&CyclicPatternDetector{})
	registry.Register(&SeasonalPatternDetector{})
	registry.Register(&SpikePatternDetector{})
	registry.Register(&PlateauPatternDetector{})

	return registry
}

// Register registers a pattern detector
func (pdr *PatternDetectorRegistry) Register(detector PatternDetector) {
	pdr.detectors[detector.Name()] = detector
}

// Get returns a pattern detector by name
func (pdr *PatternDetectorRegistry) Get(name string) (PatternDetector, bool) {
	detector, exists := pdr.detectors[name]
	return detector, exists
}

// GetAll returns all registered detectors
func (pdr *PatternDetectorRegistry) GetAll() map[string]PatternDetector {
	return pdr.detectors
}

// TrendPatternDetector detects trending patterns
type TrendPatternDetector struct{}

// Name returns the detector name
func (tpd *TrendPatternDetector) Name() string {
	return "trend"
}

// DetectPatterns detects trend patterns in the data
func (tpd *TrendPatternDetector) DetectPatterns(data []float64, params AnalysisParameters) ([]Pattern, error) {
	if len(data) < params.WindowSize {
		return nil, fmt.Errorf("insufficient data for trend detection")
	}

	patterns := []Pattern{}
	windowSize := params.WindowSize
	if windowSize <= 0 {
		windowSize = 10
	}

	// Sliding window approach
	for i := 0; i <= len(data)-windowSize; i++ {
		window := data[i : i+windowSize]
		
		// Calculate linear regression for the window
		slope, r2, err := tpd.linearRegression(window)
		if err != nil {
			continue
		}

		// Determine trend type and strength
		if math.Abs(slope) > params.Threshold && r2 > 0.7 {
			direction := PatternTypeTrending
			confidence := r2
			strength := math.Abs(slope)

			pattern := Pattern{
				ID:          fmt.Sprintf("trend_%d_%d", i, i+windowSize-1),
				Type:        direction,
				Description: fmt.Sprintf("Trend with slope %.4f (R²=%.3f)", slope, r2),
				StartIndex:  i,
				EndIndex:    i + windowSize - 1,
				Confidence:  confidence,
				Strength:    strength,
				Parameters: map[string]interface{}{
					"slope":     slope,
					"r_squared": r2,
					"direction": tpd.getTrendDirection(slope),
				},
			}

			patterns = append(patterns, pattern)
		}
	}

	return tpd.mergeSimilarPatterns(patterns), nil
}

// linearRegression calculates linear regression for a data window
func (tpd *TrendPatternDetector) linearRegression(data []float64) (slope, r2 float64, err error) {
	n := float64(len(data))
	if n < 2 {
		return 0, 0, fmt.Errorf("insufficient data for regression")
	}

	// Calculate sums
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

	// Calculate slope
	denominator := n*sumX2 - sumX*sumX
	if denominator == 0 {
		return 0, 0, fmt.Errorf("cannot calculate slope: zero denominator")
	}
	slope = (n*sumXY - sumX*sumY) / denominator

	// Calculate R²
	yMean := sumY / n
	ssTotal := sumY2 - n*yMean*yMean
	if ssTotal == 0 {
		r2 = 1.0 // Perfect fit for constant data
	} else {
		ssRes := 0.0
		for i, y := range data {
			predicted := slope*float64(i) + (sumY-slope*sumX)/n
			ssRes += (y - predicted) * (y - predicted)
		}
		r2 = 1.0 - ssRes/ssTotal
	}

	return slope, r2, nil
}

// getTrendDirection determines trend direction from slope
func (tpd *TrendPatternDetector) getTrendDirection(slope float64) string {
	if slope > 0.01 {
		return "increasing"
	} else if slope < -0.01 {
		return "decreasing"
	}
	return "stable"
}

// mergeSimilarPatterns merges overlapping or similar trend patterns
func (tpd *TrendPatternDetector) mergeSimilarPatterns(patterns []Pattern) []Pattern {
	if len(patterns) <= 1 {
		return patterns
	}

	merged := []Pattern{}
	sort.Slice(patterns, func(i, j int) bool {
		return patterns[i].StartIndex < patterns[j].StartIndex
	})

	current := patterns[0]
	for i := 1; i < len(patterns); i++ {
		next := patterns[i]
		
		// Check for overlap
		if next.StartIndex <= current.EndIndex {
			// Merge patterns
			current.EndIndex = int(math.Max(float64(current.EndIndex), float64(next.EndIndex)))
			current.Confidence = (current.Confidence + next.Confidence) / 2
			current.Strength = (current.Strength + next.Strength) / 2
			current.Description = fmt.Sprintf("Merged trend pattern from %d to %d", current.StartIndex, current.EndIndex)
		} else {
			merged = append(merged, current)
			current = next
		}
	}
	merged = append(merged, current)

	return merged
}

// CyclicPatternDetector detects cyclic patterns
type CyclicPatternDetector struct{}

// Name returns the detector name
func (cpd *CyclicPatternDetector) Name() string {
	return "cyclic"
}

// DetectPatterns detects cyclic patterns using autocorrelation
func (cpd *CyclicPatternDetector) DetectPatterns(data []float64, params AnalysisParameters) ([]Pattern, error) {
	if len(data) < 20 {
		return nil, fmt.Errorf("insufficient data for cyclic pattern detection")
	}

	patterns := []Pattern{}
	maxLag := int(math.Min(float64(len(data)/2), 100))

	// Calculate autocorrelation for different lags
	autocorr := cpd.calculateAutocorrelation(data, maxLag)
	
	// Find peaks in autocorrelation (indicating cycles)
	peaks := cpd.findPeaks(autocorr, params.Threshold)

	for _, peak := range peaks {
		if peak.Value > 0.3 && peak.Lag > 1 { // Minimum correlation threshold
			pattern := Pattern{
				ID:          fmt.Sprintf("cycle_%d", peak.Lag),
				Type:        PatternTypeCyclic,
				Description: fmt.Sprintf("Cyclic pattern with period %d (correlation=%.3f)", peak.Lag, peak.Value),
				StartIndex:  0,
				EndIndex:    len(data) - 1,
				Confidence:  peak.Value,
				Strength:    peak.Value,
				Frequency:   1.0 / float64(peak.Lag),
				Parameters: map[string]interface{}{
					"period":      peak.Lag,
					"correlation": peak.Value,
				},
			}
			patterns = append(patterns, pattern)
		}
	}

	return patterns, nil
}

// Peak represents a peak in autocorrelation
type Peak struct {
	Lag   int
	Value float64
}

// calculateAutocorrelation calculates autocorrelation for different lags
func (cpd *CyclicPatternDetector) calculateAutocorrelation(data []float64, maxLag int) []float64 {
	n := len(data)
	autocorr := make([]float64, maxLag)

	// Calculate mean
	mean := 0.0
	for _, v := range data {
		mean += v
	}
	mean /= float64(n)

	// Calculate variance
	variance := 0.0
	for _, v := range data {
		diff := v - mean
		variance += diff * diff
	}
	variance /= float64(n)

	// Calculate autocorrelation for each lag
	for lag := 1; lag < maxLag && lag < n; lag++ {
		covariance := 0.0
		count := 0

		for i := 0; i < n-lag; i++ {
			covariance += (data[i] - mean) * (data[i+lag] - mean)
			count++
		}

		if count > 0 && variance > 0 {
			autocorr[lag] = (covariance / float64(count)) / variance
		}
	}

	return autocorr
}

// findPeaks finds peaks in the autocorrelation function
func (cpd *CyclicPatternDetector) findPeaks(autocorr []float64, threshold float64) []Peak {
	peaks := []Peak{}

	for i := 1; i < len(autocorr)-1; i++ {
		if autocorr[i] > autocorr[i-1] && autocorr[i] > autocorr[i+1] && autocorr[i] > threshold {
			peaks = append(peaks, Peak{
				Lag:   i,
				Value: autocorr[i],
			})
		}
	}

	// Sort peaks by value (descending)
	sort.Slice(peaks, func(i, j int) bool {
		return peaks[i].Value > peaks[j].Value
	})

	return peaks
}

// SeasonalPatternDetector detects seasonal patterns
type SeasonalPatternDetector struct{}

// Name returns the detector name
func (spd *SeasonalPatternDetector) Name() string {
	return "seasonal"
}

// DetectPatterns detects seasonal patterns
func (spd *SeasonalPatternDetector) DetectPatterns(data []float64, params AnalysisParameters) ([]Pattern, error) {
	if len(data) < 24 {
		return nil, fmt.Errorf("insufficient data for seasonal pattern detection")
	}

	patterns := []Pattern{}
	
	// Common seasonal periods to check
	periods := []int{7, 24, 168, 720} // daily, hourly, weekly, monthly (assuming hourly data)
	if params.Seasonality > 0 {
		periods = []int{params.Seasonality}
	}

	for _, period := range periods {
		if len(data) >= period*2 {
			correlation := spd.calculateSeasonalCorrelation(data, period)
			if correlation > 0.3 {
				pattern := Pattern{
					ID:          fmt.Sprintf("seasonal_%d", period),
					Type:        PatternTypeSeasonal,
					Description: fmt.Sprintf("Seasonal pattern with period %d (strength=%.3f)", period, correlation),
					StartIndex:  0,
					EndIndex:    len(data) - 1,
					Confidence:  correlation,
					Strength:    correlation,
					Frequency:   1.0 / float64(period),
					Parameters: map[string]interface{}{
						"period":   period,
						"strength": correlation,
					},
				}
				patterns = append(patterns, pattern)
			}
		}
	}

	return patterns, nil
}

// calculateSeasonalCorrelation calculates correlation for seasonal patterns
func (spd *SeasonalPatternDetector) calculateSeasonalCorrelation(data []float64, period int) float64 {
	if len(data) < period*2 {
		return 0.0
	}

	// Calculate average pattern for one period
	avgPattern := make([]float64, period)
	counts := make([]int, period)

	for i, v := range data {
		pos := i % period
		avgPattern[pos] += v
		counts[pos]++
	}

	for i := range avgPattern {
		if counts[i] > 0 {
			avgPattern[i] /= float64(counts[i])
		}
	}

	// Calculate correlation between actual data and repeated average pattern
	totalCorr := 0.0
	validPeriods := 0

	for start := 0; start+period <= len(data); start += period {
		periodData := data[start : start+period]
		corr := spd.calculateCorrelation(periodData, avgPattern)
		if !math.IsNaN(corr) {
			totalCorr += corr
			validPeriods++
		}
	}

	if validPeriods > 0 {
		return totalCorr / float64(validPeriods)
	}
	return 0.0
}

// calculateCorrelation calculates Pearson correlation coefficient
func (spd *SeasonalPatternDetector) calculateCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0.0
	}

	n := float64(len(x))
	sumX, sumY, sumXY, sumX2, sumY2 := 0.0, 0.0, 0.0, 0.0, 0.0

	for i := 0; i < len(x); i++ {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumX2 += x[i] * x[i]
		sumY2 += y[i] * y[i]
	}

	numerator := n*sumXY - sumX*sumY
	denominator := math.Sqrt((n*sumX2 - sumX*sumX) * (n*sumY2 - sumY*sumY))

	if denominator == 0 {
		return 0.0
	}

	return numerator / denominator
}

// SpikePatternDetector detects spike patterns
type SpikePatternDetector struct{}

// Name returns the detector name
func (spd *SpikePatternDetector) Name() string {
	return "spike"
}

// DetectPatterns detects spike patterns (sudden increases/decreases)
func (spd *SpikePatternDetector) DetectPatterns(data []float64, params AnalysisParameters) ([]Pattern, error) {
	if len(data) < 3 {
		return nil, fmt.Errorf("insufficient data for spike detection")
	}

	patterns := []Pattern{}
	threshold := params.Threshold
	if threshold <= 0 {
		threshold = 2.0 // Default to 2 standard deviations
	}

	// Calculate rolling statistics
	windowSize := 5
	if params.WindowSize > 0 {
		windowSize = params.WindowSize
	}

	for i := windowSize; i < len(data)-windowSize; i++ {
		// Calculate local mean and std dev
		localMean, localStd := spd.calculateLocalStats(data, i, windowSize)
		
		if localStd > 0 {
			zScore := math.Abs(data[i]-localMean) / localStd
			
			if zScore > threshold {
				patternType := PatternTypeSpike
				if data[i] < localMean {
					patternType = PatternTypeDrop
				}

				pattern := Pattern{
					ID:          fmt.Sprintf("%s_%d", patternType, i),
					Type:        patternType,
					Description: fmt.Sprintf("%s at index %d (z-score=%.2f)", patternType, i, zScore),
					StartIndex:  i,
					EndIndex:    i,
					Confidence:  math.Min(zScore/threshold, 1.0),
					Strength:    zScore,
					Parameters: map[string]interface{}{
						"z_score":    zScore,
						"value":      data[i],
						"local_mean": localMean,
						"local_std":  localStd,
					},
				}
				patterns = append(patterns, pattern)
			}
		}
	}

	return patterns, nil
}

// calculateLocalStats calculates local mean and standard deviation
func (spd *SpikePatternDetector) calculateLocalStats(data []float64, center, windowSize int) (mean, std float64) {
	start := int(math.Max(0, float64(center-windowSize)))
	end := int(math.Min(float64(len(data)), float64(center+windowSize+1)))
	
	window := data[start:end]
	
	// Exclude the center point from local statistics
	localData := append(window[:center-start], window[center-start+1:]...)
	
	if len(localData) == 0 {
		return 0, 0
	}

	// Calculate mean
	sum := 0.0
	for _, v := range localData {
		sum += v
	}
	mean = sum / float64(len(localData))

	// Calculate standard deviation
	sumSq := 0.0
	for _, v := range localData {
		diff := v - mean
		sumSq += diff * diff
	}
	std = math.Sqrt(sumSq / float64(len(localData)))

	return mean, std
}

// PlateauPatternDetector detects plateau patterns (flat regions)
type PlateauPatternDetector struct{}

// Name returns the detector name
func (ppd *PlateauPatternDetector) Name() string {
	return "plateau"
}

// DetectPatterns detects plateau patterns
func (ppd *PlateauPatternDetector) DetectPatterns(data []float64, params AnalysisParameters) ([]Pattern, error) {
	if len(data) < params.WindowSize {
		return nil, fmt.Errorf("insufficient data for plateau detection")
	}

	patterns := []Pattern{}
	windowSize := params.WindowSize
	if windowSize <= 0 {
		windowSize = 10
	}

	threshold := params.Threshold
	if threshold <= 0 {
		threshold = 0.1 // Default threshold for plateau detection
	}

	for i := 0; i <= len(data)-windowSize; i++ {
		window := data[i : i+windowSize]
		
		// Calculate coefficient of variation (std dev / mean)
		mean, std := ppd.calculateMeanStd(window)
		
		if mean != 0 {
			cv := std / math.Abs(mean)
			
			if cv < threshold {
				pattern := Pattern{
					ID:          fmt.Sprintf("plateau_%d_%d", i, i+windowSize-1),
					Type:        PatternTypePlateau,
					Description: fmt.Sprintf("Plateau from %d to %d (CV=%.4f)", i, i+windowSize-1, cv),
					StartIndex:  i,
					EndIndex:    i + windowSize - 1,
					Confidence:  1.0 - cv,
					Strength:    1.0 - cv,
					Parameters: map[string]interface{}{
						"coefficient_of_variation": cv,
						"mean":                     mean,
						"std_dev":                  std,
					},
				}
				patterns = append(patterns, pattern)
			}
		}
	}

	// Merge overlapping plateaus
	return ppd.mergeOverlappingPlateaus(patterns), nil
}

// calculateMeanStd calculates mean and standard deviation
func (ppd *PlateauPatternDetector) calculateMeanStd(data []float64) (mean, std float64) {
	if len(data) == 0 {
		return 0, 0
	}

	sum := 0.0
	for _, v := range data {
		sum += v
	}
	mean = sum / float64(len(data))

	sumSq := 0.0
	for _, v := range data {
		diff := v - mean
		sumSq += diff * diff
	}
	std = math.Sqrt(sumSq / float64(len(data)))

	return mean, std
}

// mergeOverlappingPlateaus merges overlapping plateau patterns
func (ppd *PlateauPatternDetector) mergeOverlappingPlateaus(patterns []Pattern) []Pattern {
	if len(patterns) <= 1 {
		return patterns
	}

	merged := []Pattern{}
	sort.Slice(patterns, func(i, j int) bool {
		return patterns[i].StartIndex < patterns[j].StartIndex
	})

	current := patterns[0]
	for i := 1; i < len(patterns); i++ {
		next := patterns[i]
		
		// Check for overlap or adjacency
		if next.StartIndex <= current.EndIndex+1 {
			// Merge patterns
			current.EndIndex = int(math.Max(float64(current.EndIndex), float64(next.EndIndex)))
			current.Confidence = (current.Confidence + next.Confidence) / 2
			current.Strength = (current.Strength + next.Strength) / 2
			current.Description = fmt.Sprintf("Merged plateau from %d to %d", current.StartIndex, current.EndIndex)
		} else {
			merged = append(merged, current)
			current = next
		}
	}
	merged = append(merged, current)

	return merged
}