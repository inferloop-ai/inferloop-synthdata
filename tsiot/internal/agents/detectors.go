package agents

import (
	"context"
	"fmt"
	"math"
	"sort"

	"gonum.org/v1/gonum/stat"
)

// ZScoreDetector implements Z-score based anomaly detection
type ZScoreDetector struct {
	threshold float64
	mean      float64
	stddev    float64
	trained   bool
}

// NewZScoreDetector creates a new Z-score detector
func NewZScoreDetector(threshold float64) *ZScoreDetector {
	return &ZScoreDetector{
		threshold: threshold,
		trained:   false,
	}
}

func (z *ZScoreDetector) GetName() string {
	return "zscore"
}

func (z *ZScoreDetector) GetDescription() string {
	return "Z-score based anomaly detection using statistical deviation from mean"
}

func (z *ZScoreDetector) Train(ctx context.Context, data []float64) error {
	if len(data) < 2 {
		return fmt.Errorf("insufficient data for training: need at least 2 points")
	}
	
	z.mean = stat.Mean(data, nil)
	z.stddev = math.Sqrt(stat.Variance(data, nil))
	z.trained = true
	
	return nil
}

func (z *ZScoreDetector) Detect(ctx context.Context, data []float64) (*AnomalyResult, error) {
	if !z.trained && len(data) >= 2 {
		if err := z.Train(ctx, data); err != nil {
			return nil, err
		}
	}
	
	if !z.trained {
		return nil, fmt.Errorf("detector not trained and insufficient data")
	}
	
	var maxZScore float64
	var anomalyCount int
	
	for _, value := range data {
		zScore := math.Abs(value-z.mean) / z.stddev
		if zScore > maxZScore {
			maxZScore = zScore
		}
		if zScore > z.threshold {
			anomalyCount++
		}
	}
	
	isAnomaly := maxZScore > z.threshold
	severity := z.calculateSeverity(maxZScore)
	
	return &AnomalyResult{
		IsAnomaly:    isAnomaly,
		AnomalyScore: maxZScore,
		Threshold:    z.threshold,
		Confidence:   math.Min(maxZScore/z.threshold, 1.0),
		Severity:     severity,
		AnomalyType:  TypePointAnomaly,
		Explanation:  fmt.Sprintf("Z-score %.2f exceeds threshold %.2f", maxZScore, z.threshold),
		Metadata: map[string]interface{}{
			"max_zscore":     maxZScore,
			"mean":           z.mean,
			"stddev":         z.stddev,
			"anomaly_count":  anomalyCount,
			"anomaly_ratio":  float64(anomalyCount) / float64(len(data)),
		},
	}, nil
}

func (z *ZScoreDetector) UpdateModel(ctx context.Context, data []float64) error {
	return z.Train(ctx, data)
}

func (z *ZScoreDetector) GetThreshold() float64 {
	return z.threshold
}

func (z *ZScoreDetector) SetThreshold(threshold float64) {
	z.threshold = threshold
}

func (z *ZScoreDetector) calculateSeverity(zScore float64) AnomalySeverity {
	if zScore < z.threshold {
		return SeverityInfo
	} else if zScore < z.threshold*1.5 {
		return SeverityLow
	} else if zScore < z.threshold*2.0 {
		return SeverityMedium
	} else if zScore < z.threshold*3.0 {
		return SeverityHigh
	} else {
		return SeverityCritical
	}
}

// IQRDetector implements Interquartile Range based anomaly detection
type IQRDetector struct {
	threshold float64
	q1        float64
	q3        float64
	iqr       float64
	trained   bool
}

// NewIQRDetector creates a new IQR detector
func NewIQRDetector(threshold float64) *IQRDetector {
	return &IQRDetector{
		threshold: threshold,
		trained:   false,
	}
}

func (i *IQRDetector) GetName() string {
	return "iqr"
}

func (i *IQRDetector) GetDescription() string {
	return "Interquartile Range based anomaly detection using quartile statistics"
}

func (i *IQRDetector) Train(ctx context.Context, data []float64) error {
	if len(data) < 4 {
		return fmt.Errorf("insufficient data for training: need at least 4 points")
	}
	
	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)
	
	n := len(sorted)
	i.q1 = sorted[n/4]
	i.q3 = sorted[3*n/4]
	i.iqr = i.q3 - i.q1
	i.trained = true
	
	return nil
}

func (i *IQRDetector) Detect(ctx context.Context, data []float64) (*AnomalyResult, error) {
	if !i.trained && len(data) >= 4 {
		if err := i.Train(ctx, data); err != nil {
			return nil, err
		}
	}
	
	if !i.trained {
		return nil, fmt.Errorf("detector not trained and insufficient data")
	}
	
	lowerBound := i.q1 - i.threshold*i.iqr
	upperBound := i.q3 + i.threshold*i.iqr
	
	var maxDeviation float64
	var anomalyCount int
	
	for _, value := range data {
		var deviation float64
		if value < lowerBound {
			deviation = (lowerBound - value) / i.iqr
		} else if value > upperBound {
			deviation = (value - upperBound) / i.iqr
		}
		
		if deviation > maxDeviation {
			maxDeviation = deviation
		}
		if deviation > 0 {
			anomalyCount++
		}
	}
	
	isAnomaly := maxDeviation > 0
	severity := i.calculateSeverity(maxDeviation)
	
	return &AnomalyResult{
		IsAnomaly:    isAnomaly,
		AnomalyScore: maxDeviation,
		Threshold:    i.threshold,
		Confidence:   math.Min(maxDeviation/i.threshold, 1.0),
		Severity:     severity,
		AnomalyType:  TypePointAnomaly,
		Explanation:  fmt.Sprintf("IQR deviation %.2f exceeds threshold %.2f", maxDeviation, i.threshold),
		Metadata: map[string]interface{}{
			"max_deviation": maxDeviation,
			"q1":            i.q1,
			"q3":            i.q3,
			"iqr":           i.iqr,
			"lower_bound":   lowerBound,
			"upper_bound":   upperBound,
			"anomaly_count": anomalyCount,
		},
	}, nil
}

func (i *IQRDetector) UpdateModel(ctx context.Context, data []float64) error {
	return i.Train(ctx, data)
}

func (i *IQRDetector) GetThreshold() float64 {
	return i.threshold
}

func (i *IQRDetector) SetThreshold(threshold float64) {
	i.threshold = threshold
}

func (i *IQRDetector) calculateSeverity(deviation float64) AnomalySeverity {
	if deviation == 0 {
		return SeverityInfo
	} else if deviation < i.threshold*0.5 {
		return SeverityLow
	} else if deviation < i.threshold {
		return SeverityMedium
	} else if deviation < i.threshold*2.0 {
		return SeverityHigh
	} else {
		return SeverityCritical
	}
}

// MovingAverageDetector implements moving average based anomaly detection
type MovingAverageDetector struct {
	windowSize    int
	threshold     float64
	movingAverage []float64
	trained       bool
}

// NewMovingAverageDetector creates a new moving average detector
func NewMovingAverageDetector(windowSize int, threshold float64) *MovingAverageDetector {
	return &MovingAverageDetector{
		windowSize: windowSize,
		threshold:  threshold,
		trained:    false,
	}
}

func (m *MovingAverageDetector) GetName() string {
	return "moving_average"
}

func (m *MovingAverageDetector) GetDescription() string {
	return fmt.Sprintf("Moving average based anomaly detection with window size %d", m.windowSize)
}

func (m *MovingAverageDetector) Train(ctx context.Context, data []float64) error {
	if len(data) < m.windowSize {
		return fmt.Errorf("insufficient data for training: need at least %d points", m.windowSize)
	}
	
	m.movingAverage = m.calculateMovingAverage(data)
	m.trained = true
	
	return nil
}

func (m *MovingAverageDetector) Detect(ctx context.Context, data []float64) (*AnomalyResult, error) {
	if !m.trained && len(data) >= m.windowSize {
		if err := m.Train(ctx, data); err != nil {
			return nil, err
		}
	}
	
	if !m.trained {
		return nil, fmt.Errorf("detector not trained and insufficient data")
	}
	
	movingAvg := m.calculateMovingAverage(data)
	
	// Calculate deviations from moving average
	var maxDeviation float64
	var anomalyCount int
	var totalDeviation float64
	
	for i, value := range data {
		if i < len(movingAvg) {
			deviation := math.Abs(value - movingAvg[i])
			totalDeviation += deviation
			
			if deviation > maxDeviation {
				maxDeviation = deviation
			}
			
			// Calculate adaptive threshold based on recent data variability
			adaptiveThreshold := m.calculateAdaptiveThreshold(data, i)
			if deviation > adaptiveThreshold {
				anomalyCount++
			}
		}
	}
	
	avgDeviation := totalDeviation / float64(len(data))
	adaptiveThreshold := m.calculateAdaptiveThreshold(data, len(data)-1)
	isAnomaly := maxDeviation > adaptiveThreshold
	severity := m.calculateSeverity(maxDeviation, adaptiveThreshold)
	
	return &AnomalyResult{
		IsAnomaly:    isAnomaly,
		AnomalyScore: maxDeviation / adaptiveThreshold,
		Threshold:    m.threshold,
		Confidence:   math.Min(maxDeviation/adaptiveThreshold, 1.0),
		Severity:     severity,
		AnomalyType:  TypeTrendAnomaly,
		Explanation:  fmt.Sprintf("Moving average deviation %.2f exceeds adaptive threshold %.2f", maxDeviation, adaptiveThreshold),
		Metadata: map[string]interface{}{
			"max_deviation":       maxDeviation,
			"avg_deviation":       avgDeviation,
			"adaptive_threshold":  adaptiveThreshold,
			"window_size":         m.windowSize,
			"anomaly_count":       anomalyCount,
			"moving_average":      movingAvg,
		},
	}, nil
}

func (m *MovingAverageDetector) UpdateModel(ctx context.Context, data []float64) error {
	return m.Train(ctx, data)
}

func (m *MovingAverageDetector) GetThreshold() float64 {
	return m.threshold
}

func (m *MovingAverageDetector) SetThreshold(threshold float64) {
	m.threshold = threshold
}

func (m *MovingAverageDetector) calculateMovingAverage(data []float64) []float64 {
	if len(data) < m.windowSize {
		return []float64{}
	}
	
	movingAvg := make([]float64, len(data)-m.windowSize+1)
	
	for i := 0; i <= len(data)-m.windowSize; i++ {
		sum := 0.0
		for j := i; j < i+m.windowSize; j++ {
			sum += data[j]
		}
		movingAvg[i] = sum / float64(m.windowSize)
	}
	
	return movingAvg
}

func (m *MovingAverageDetector) calculateAdaptiveThreshold(data []float64, index int) float64 {
	// Calculate threshold based on recent data variability
	start := index - m.windowSize
	if start < 0 {
		start = 0
	}
	
	window := data[start:index+1]
	if len(window) < 2 {
		return m.threshold
	}
	
	variance := stat.Variance(window, nil)
	stddev := math.Sqrt(variance)
	
	return m.threshold * stddev
}

func (m *MovingAverageDetector) calculateSeverity(deviation, threshold float64) AnomalySeverity {
	ratio := deviation / threshold
	
	if ratio < 1.0 {
		return SeverityInfo
	} else if ratio < 1.5 {
		return SeverityLow
	} else if ratio < 2.0 {
		return SeverityMedium
	} else if ratio < 3.0 {
		return SeverityHigh
	} else {
		return SeverityCritical
	}
}

// ExponentialSmoothingDetector implements exponential smoothing based anomaly detection
type ExponentialSmoothingDetector struct {
	alpha      float64
	threshold  float64
	forecast   float64
	level      float64
	trained    bool
}

// NewExponentialSmoothingDetector creates a new exponential smoothing detector
func NewExponentialSmoothingDetector(alpha, threshold float64) *ExponentialSmoothingDetector {
	return &ExponentialSmoothingDetector{
		alpha:     alpha,
		threshold: threshold,
		trained:   false,
	}
}

func (e *ExponentialSmoothingDetector) GetName() string {
	return "exponential_smoothing"
}

func (e *ExponentialSmoothingDetector) GetDescription() string {
	return fmt.Sprintf("Exponential smoothing based anomaly detection with alpha %.2f", e.alpha)
}

func (e *ExponentialSmoothingDetector) Train(ctx context.Context, data []float64) error {
	if len(data) < 2 {
		return fmt.Errorf("insufficient data for training: need at least 2 points")
	}
	
	// Initialize with first value
	e.level = data[0]
	e.forecast = data[0]
	
	// Update model with training data
	for i := 1; i < len(data); i++ {
		e.updateModel(data[i])
	}
	
	e.trained = true
	return nil
}

func (e *ExponentialSmoothingDetector) Detect(ctx context.Context, data []float64) (*AnomalyResult, error) {
	if !e.trained && len(data) >= 2 {
		if err := e.Train(ctx, data); err != nil {
			return nil, err
		}
	}
	
	if !e.trained {
		return nil, fmt.Errorf("detector not trained and insufficient data")
	}
	
	var maxDeviation float64
	var anomalyCount int
	errors := make([]float64, 0, len(data))
	
	// Calculate prediction errors
	currentLevel := e.level
	for _, value := range data {
		forecast := currentLevel
		error := math.Abs(value - forecast)
		errors = append(errors, error)
		
		if error > maxDeviation {
			maxDeviation = error
		}
		
		// Update level for next prediction
		currentLevel = e.alpha*value + (1-e.alpha)*currentLevel
	}
	
	// Calculate adaptive threshold based on error statistics
	errorMean := stat.Mean(errors, nil)
	errorStdDev := math.Sqrt(stat.Variance(errors, nil))
	adaptiveThreshold := errorMean + e.threshold*errorStdDev
	
	// Count anomalies
	for _, error := range errors {
		if error > adaptiveThreshold {
			anomalyCount++
		}
	}
	
	isAnomaly := maxDeviation > adaptiveThreshold
	severity := e.calculateSeverity(maxDeviation, adaptiveThreshold)
	
	return &AnomalyResult{
		IsAnomaly:    isAnomaly,
		AnomalyScore: maxDeviation / adaptiveThreshold,
		Threshold:    e.threshold,
		Confidence:   math.Min(maxDeviation/adaptiveThreshold, 1.0),
		Severity:     severity,
		AnomalyType:  TypeTrendAnomaly,
		Explanation:  fmt.Sprintf("Exponential smoothing prediction error %.2f exceeds threshold %.2f", maxDeviation, adaptiveThreshold),
		Metadata: map[string]interface{}{
			"max_deviation":        maxDeviation,
			"adaptive_threshold":   adaptiveThreshold,
			"error_mean":           errorMean,
			"error_stddev":         errorStdDev,
			"alpha":                e.alpha,
			"anomaly_count":        anomalyCount,
			"prediction_errors":    errors,
		},
	}, nil
}

func (e *ExponentialSmoothingDetector) UpdateModel(ctx context.Context, data []float64) error {
	if len(data) == 0 {
		return nil
	}
	
	for _, value := range data {
		e.updateModel(value)
	}
	
	return nil
}

func (e *ExponentialSmoothingDetector) GetThreshold() float64 {
	return e.threshold
}

func (e *ExponentialSmoothingDetector) SetThreshold(threshold float64) {
	e.threshold = threshold
}

func (e *ExponentialSmoothingDetector) updateModel(value float64) {
	e.level = e.alpha*value + (1-e.alpha)*e.level
	e.forecast = e.level
}

func (e *ExponentialSmoothingDetector) calculateSeverity(deviation, threshold float64) AnomalySeverity {
	ratio := deviation / threshold
	
	if ratio < 1.0 {
		return SeverityInfo
	} else if ratio < 1.5 {
		return SeverityLow
	} else if ratio < 2.0 {
		return SeverityMedium
	} else if ratio < 3.0 {
		return SeverityHigh
	} else {
		return SeverityCritical
	}
}

// SeasonalDetector implements seasonal pattern based anomaly detection
type SeasonalDetector struct {
	threshold       float64
	seasonLength    int
	seasonalPattern []float64
	trained         bool
}

// NewSeasonalDetector creates a new seasonal detector
func NewSeasonalDetector(seasonLength int, threshold float64) *SeasonalDetector {
	return &SeasonalDetector{
		threshold:    threshold,
		seasonLength: seasonLength,
		trained:      false,
	}
}

func (s *SeasonalDetector) GetName() string {
	return "seasonal"
}

func (s *SeasonalDetector) GetDescription() string {
	return fmt.Sprintf("Seasonal pattern based anomaly detection with season length %d", s.seasonLength)
}

func (s *SeasonalDetector) Train(ctx context.Context, data []float64) error {
	if len(data) < s.seasonLength*2 {
		return fmt.Errorf("insufficient data for training: need at least %d points", s.seasonLength*2)
	}
	
	// Calculate seasonal pattern by averaging values at same seasonal positions
	s.seasonalPattern = make([]float64, s.seasonLength)
	counts := make([]int, s.seasonLength)
	
	for i, value := range data {
		seasonPos := i % s.seasonLength
		s.seasonalPattern[seasonPos] += value
		counts[seasonPos]++
	}
	
	// Average the accumulated values
	for i := 0; i < s.seasonLength; i++ {
		if counts[i] > 0 {
			s.seasonalPattern[i] /= float64(counts[i])
		}
	}
	
	s.trained = true
	return nil
}

func (s *SeasonalDetector) Detect(ctx context.Context, data []float64) (*AnomalyResult, error) {
	if !s.trained && len(data) >= s.seasonLength*2 {
		if err := s.Train(ctx, data); err != nil {
			return nil, err
		}
	}
	
	if !s.trained {
		return nil, fmt.Errorf("detector not trained and insufficient data")
	}
	
	var maxDeviation float64
	var anomalyCount int
	deviations := make([]float64, 0, len(data))
	
	// Calculate deviations from seasonal pattern
	for i, value := range data {
		seasonPos := i % s.seasonLength
		expected := s.seasonalPattern[seasonPos]
		deviation := math.Abs(value - expected)
		deviations = append(deviations, deviation)
		
		if deviation > maxDeviation {
			maxDeviation = deviation
		}
	}
	
	// Calculate adaptive threshold based on deviation statistics
	deviationMean := stat.Mean(deviations, nil)
	deviationStdDev := math.Sqrt(stat.Variance(deviations, nil))
	adaptiveThreshold := deviationMean + s.threshold*deviationStdDev
	
	// Count anomalies
	for _, deviation := range deviations {
		if deviation > adaptiveThreshold {
			anomalyCount++
		}
	}
	
	isAnomaly := maxDeviation > adaptiveThreshold
	severity := s.calculateSeverity(maxDeviation, adaptiveThreshold)
	
	return &AnomalyResult{
		IsAnomaly:    isAnomaly,
		AnomalyScore: maxDeviation / adaptiveThreshold,
		Threshold:    s.threshold,
		Confidence:   math.Min(maxDeviation/adaptiveThreshold, 1.0),
		Severity:     severity,
		AnomalyType:  TypeSeasonalAnomaly,
		Explanation:  fmt.Sprintf("Seasonal deviation %.2f exceeds threshold %.2f", maxDeviation, adaptiveThreshold),
		Metadata: map[string]interface{}{
			"max_deviation":        maxDeviation,
			"adaptive_threshold":   adaptiveThreshold,
			"deviation_mean":       deviationMean,
			"deviation_stddev":     deviationStdDev,
			"season_length":        s.seasonLength,
			"seasonal_pattern":     s.seasonalPattern,
			"anomaly_count":        anomalyCount,
			"deviations":           deviations,
		},
	}, nil
}

func (s *SeasonalDetector) UpdateModel(ctx context.Context, data []float64) error {
	return s.Train(ctx, data)
}

func (s *SeasonalDetector) GetThreshold() float64 {
	return s.threshold
}

func (s *SeasonalDetector) SetThreshold(threshold float64) {
	s.threshold = threshold
}

func (s *SeasonalDetector) calculateSeverity(deviation, threshold float64) AnomalySeverity {
	ratio := deviation / threshold
	
	if ratio < 1.0 {
		return SeverityInfo
	} else if ratio < 1.5 {
		return SeverityLow
	} else if ratio < 2.0 {
		return SeverityMedium
	} else if ratio < 3.0 {
		return SeverityHigh
	} else {
		return SeverityCritical
	}
}