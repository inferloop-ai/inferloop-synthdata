package privacy

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"gonum.org/v1/gonum/stat/distuv"
)

// LaplaceMechanism implements the Laplace mechanism for differential privacy
type LaplaceMechanism struct {
	randSource     *rand.Rand
	clampingConfig *ClampingConfig
	postProcessConfig *PostProcessConfig
}

// GaussianMechanism implements the Gaussian mechanism for differential privacy
type GaussianMechanism struct {
	randSource     *rand.Rand
	clampingConfig *ClampingConfig
	postProcessConfig *PostProcessConfig
}

// ExponentialMechanism implements the exponential mechanism for differential privacy
type ExponentialMechanism struct {
	randSource     *rand.Rand
	clampingConfig *ClampingConfig
	postProcessConfig *PostProcessConfig
	scoringFunction func([]float64, float64) float64
}

// NewLaplaceMechanism creates a new Laplace mechanism
func NewLaplaceMechanism(randSource *rand.Rand, clampingConfig *ClampingConfig, postProcessConfig *PostProcessConfig) *LaplaceMechanism {
	if randSource == nil {
		randSource = rand.New(rand.NewSource(42))
	}
	
	return &LaplaceMechanism{
		randSource:        randSource,
		clampingConfig:    clampingConfig,
		postProcessConfig: postProcessConfig,
	}
}

// GetName returns the mechanism name
func (lm *LaplaceMechanism) GetName() string {
	return "laplace"
}

// GetDescription returns mechanism description
func (lm *LaplaceMechanism) GetDescription() string {
	return "Laplace mechanism adds noise from Laplace distribution calibrated to query sensitivity"
}

// AddNoise adds Laplace noise to a single value
func (lm *LaplaceMechanism) AddNoise(ctx context.Context, value float64, sensitivity float64, epsilon float64) (float64, error) {
	if epsilon <= 0 {
		return 0, fmt.Errorf("epsilon must be positive, got %f", epsilon)
	}
	
	if sensitivity < 0 {
		return 0, fmt.Errorf("sensitivity must be non-negative, got %f", sensitivity)
	}
	
	// Calculate noise scale (b = sensitivity / epsilon)
	scale := sensitivity / epsilon
	
	// Generate Laplace noise
	noise := lm.sampleLaplace(scale)
	
	// Add noise to value
	noisyValue := value + noise
	
	// Apply clamping if configured
	if lm.clampingConfig != nil && lm.clampingConfig.Enabled {
		noisyValue = lm.applyBounds(noisyValue)
	}
	
	return noisyValue, nil
}

// AddNoiseToSeries adds Laplace noise to a time series
func (lm *LaplaceMechanism) AddNoiseToSeries(ctx context.Context, data []float64, sensitivity float64, epsilon float64) ([]float64, error) {
	if len(data) == 0 {
		return []float64{}, nil
	}
	
	result := make([]float64, len(data))
	
	for i, value := range data {
		noisyValue, err := lm.AddNoise(ctx, value, sensitivity, epsilon)
		if err != nil {
			return nil, fmt.Errorf("error adding noise at index %d: %w", i, err)
		}
		result[i] = noisyValue
		
		// Check for cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}
	
	// Apply post-processing if configured
	if lm.postProcessConfig != nil && lm.postProcessConfig.Enabled {
		result = lm.applyPostProcessing(result)
	}
	
	return result, nil
}

// CalculateSensitivity calculates sensitivity for different query types
func (lm *LaplaceMechanism) CalculateSensitivity(data []float64, queryType QueryType) float64 {
	if len(data) == 0 {
		return 0.0
	}
	
	switch queryType {
	case QueryTypeSum:
		return lm.calculateRange(data)
	case QueryTypeMean:
		return lm.calculateRange(data) / float64(len(data))
	case QueryTypeCount:
		return 1.0
	case QueryTypeMedian:
		return lm.calculateRange(data) / 2.0
	case QueryTypeVariance:
		dataRange := lm.calculateRange(data)
		return dataRange * dataRange / float64(len(data))
	case QueryTypeRange:
		return lm.calculateRange(data)
	case QueryTypeQuantile:
		return lm.calculateRange(data) / 2.0
	default:
		// Conservative default
		return lm.calculateRange(data)
	}
}

// ValidateParameters validates epsilon and delta parameters
func (lm *LaplaceMechanism) ValidateParameters(epsilon, delta float64) error {
	if epsilon <= 0 {
		return fmt.Errorf("epsilon must be positive for Laplace mechanism, got %f", epsilon)
	}
	
	// Laplace mechanism is pure DP, delta should be 0
	if delta != 0 {
		return fmt.Errorf("delta must be 0 for pure differential privacy (Laplace mechanism), got %f", delta)
	}
	
	return nil
}

// CalculateNoiseScale calculates the noise scale parameter
func (lm *LaplaceMechanism) CalculateNoiseScale(sensitivity, epsilon float64) float64 {
	return sensitivity / epsilon
}

// Helper methods for LaplaceMechanism

func (lm *LaplaceMechanism) sampleLaplace(scale float64) float64 {
	// Sample from Laplace distribution using inverse transform
	// Laplace CDF^(-1)(p) = -b*sign(p-0.5)*ln(1-2*|p-0.5|)
	u := lm.randSource.Float64()
	
	if u < 0.5 {
		return scale * math.Log(2*u)
	} else {
		return -scale * math.Log(2*(1-u))
	}
}

func (lm *LaplaceMechanism) calculateRange(data []float64) float64 {
	if len(data) == 0 {
		return 0.0
	}
	
	min, max := data[0], data[0]
	for _, val := range data[1:] {
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}
	
	return max - min
}

func (lm *LaplaceMechanism) applyBounds(value float64) float64 {
	switch lm.clampingConfig.Strategy {
	case "clip":
		if value < lm.clampingConfig.LowerBound {
			return lm.clampingConfig.LowerBound
		}
		if value > lm.clampingConfig.UpperBound {
			return lm.clampingConfig.UpperBound
		}
		return value
	case "wrap":
		// Wrap around bounds
		range_ := lm.clampingConfig.UpperBound - lm.clampingConfig.LowerBound
		if range_ <= 0 {
			return lm.clampingConfig.LowerBound
		}
		
		if value < lm.clampingConfig.LowerBound {
			return lm.clampingConfig.UpperBound - math.Mod(lm.clampingConfig.LowerBound-value, range_)
		}
		if value > lm.clampingConfig.UpperBound {
			return lm.clampingConfig.LowerBound + math.Mod(value-lm.clampingConfig.UpperBound, range_)
		}
		return value
	default: // "reject" or default to clip
		return lm.applyBounds(value) // Use clip as fallback
	}
}

func (lm *LaplaceMechanism) applyPostProcessing(data []float64) []float64 {
	result := make([]float64, len(data))
	copy(result, data)
	
	// Apply smoothing
	if lm.postProcessConfig.SmoothingWindow > 1 {
		result = lm.applySmoothing(result, lm.postProcessConfig.SmoothingWindow)
	}
	
	// Remove outliers
	if lm.postProcessConfig.OutlierRemoval {
		result = lm.removeOutliers(result)
	}
	
	// Apply min/max bounds
	if lm.postProcessConfig.MinValue != nil || lm.postProcessConfig.MaxValue != nil {
		for i := range result {
			if lm.postProcessConfig.MinValue != nil && result[i] < *lm.postProcessConfig.MinValue {
				result[i] = *lm.postProcessConfig.MinValue
			}
			if lm.postProcessConfig.MaxValue != nil && result[i] > *lm.postProcessConfig.MaxValue {
				result[i] = *lm.postProcessConfig.MaxValue
			}
		}
	}
	
	return result
}

func (lm *LaplaceMechanism) applySmoothing(data []float64, window int) []float64 {
	if window <= 1 || len(data) < window {
		return data
	}
	
	result := make([]float64, len(data))
	halfWindow := window / 2
	
	for i := range data {
		var sum float64
		var count int
		
		for j := max(0, i-halfWindow); j <= min(len(data)-1, i+halfWindow); j++ {
			sum += data[j]
			count++
		}
		
		result[i] = sum / float64(count)
	}
	
	return result
}

func (lm *LaplaceMechanism) removeOutliers(data []float64) []float64 {
	// Simple outlier removal using IQR method
	if len(data) < 4 {
		return data
	}
	
	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)
	
	n := len(sorted)
	q1 := sorted[n/4]
	q3 := sorted[3*n/4]
	iqr := q3 - q1
	
	lowerBound := q1 - 1.5*iqr
	upperBound := q3 + 1.5*iqr
	
	result := make([]float64, len(data))
	for i, val := range data {
		if val < lowerBound {
			result[i] = q1
		} else if val > upperBound {
			result[i] = q3
		} else {
			result[i] = val
		}
	}
	
	return result
}

// GaussianMechanism implementation

func NewGaussianMechanism(randSource *rand.Rand, clampingConfig *ClampingConfig, postProcessConfig *PostProcessConfig) *GaussianMechanism {
	if randSource == nil {
		randSource = rand.New(rand.NewSource(42))
	}
	
	return &GaussianMechanism{
		randSource:        randSource,
		clampingConfig:    clampingConfig,
		postProcessConfig: postProcessConfig,
	}
}

func (gm *GaussianMechanism) GetName() string {
	return "gaussian"
}

func (gm *GaussianMechanism) GetDescription() string {
	return "Gaussian mechanism adds noise from normal distribution for (ε,δ)-differential privacy"
}

func (gm *GaussianMechanism) AddNoise(ctx context.Context, value float64, sensitivity float64, epsilon float64) (float64, error) {
	// For Gaussian mechanism, we need delta parameter
	// Using a default delta = 1e-5 if not provided
	delta := 1e-5
	return gm.AddNoiseWithDelta(ctx, value, sensitivity, epsilon, delta)
}

func (gm *GaussianMechanism) AddNoiseWithDelta(ctx context.Context, value float64, sensitivity float64, epsilon, delta float64) (float64, error) {
	if epsilon <= 0 {
		return 0, fmt.Errorf("epsilon must be positive, got %f", epsilon)
	}
	
	if delta <= 0 || delta >= 1 {
		return 0, fmt.Errorf("delta must be in (0, 1), got %f", delta)
	}
	
	if sensitivity < 0 {
		return 0, fmt.Errorf("sensitivity must be non-negative, got %f", sensitivity)
	}
	
	// Calculate noise scale for Gaussian mechanism
	// σ = sqrt(2 * ln(1.25/δ)) * sensitivity / ε
	sigma := gm.CalculateNoiseScale(sensitivity, epsilon, delta)
	
	// Generate Gaussian noise
	noise := gm.randSource.NormFloat64() * sigma
	
	// Add noise to value
	noisyValue := value + noise
	
	// Apply clamping if configured
	if gm.clampingConfig != nil && gm.clampingConfig.Enabled {
		noisyValue = gm.applyBounds(noisyValue)
	}
	
	return noisyValue, nil
}

func (gm *GaussianMechanism) AddNoiseToSeries(ctx context.Context, data []float64, sensitivity float64, epsilon float64) ([]float64, error) {
	// Use default delta for time series
	delta := 1e-5
	return gm.AddNoiseToSeriesWithDelta(ctx, data, sensitivity, epsilon, delta)
}

func (gm *GaussianMechanism) AddNoiseToSeriesWithDelta(ctx context.Context, data []float64, sensitivity float64, epsilon, delta float64) ([]float64, error) {
	if len(data) == 0 {
		return []float64{}, nil
	}
	
	result := make([]float64, len(data))
	
	for i, value := range data {
		noisyValue, err := gm.AddNoiseWithDelta(ctx, value, sensitivity, epsilon, delta)
		if err != nil {
			return nil, fmt.Errorf("error adding noise at index %d: %w", i, err)
		}
		result[i] = noisyValue
		
		// Check for cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}
	
	// Apply post-processing if configured
	if gm.postProcessConfig != nil && gm.postProcessConfig.Enabled {
		result = gm.applyPostProcessing(result)
	}
	
	return result, nil
}

func (gm *GaussianMechanism) CalculateSensitivity(data []float64, queryType QueryType) float64 {
	// Same as Laplace mechanism
	if len(data) == 0 {
		return 0.0
	}
	
	switch queryType {
	case QueryTypeSum:
		return gm.calculateRange(data)
	case QueryTypeMean:
		return gm.calculateRange(data) / float64(len(data))
	case QueryTypeCount:
		return 1.0
	case QueryTypeMedian:
		return gm.calculateRange(data) / 2.0
	case QueryTypeVariance:
		dataRange := gm.calculateRange(data)
		return dataRange * dataRange / float64(len(data))
	case QueryTypeRange:
		return gm.calculateRange(data)
	case QueryTypeQuantile:
		return gm.calculateRange(data) / 2.0
	default:
		return gm.calculateRange(data)
	}
}

func (gm *GaussianMechanism) ValidateParameters(epsilon, delta float64) error {
	if epsilon <= 0 {
		return fmt.Errorf("epsilon must be positive for Gaussian mechanism, got %f", epsilon)
	}
	
	if delta <= 0 || delta >= 1 {
		return fmt.Errorf("delta must be in (0, 1) for Gaussian mechanism, got %f", delta)
	}
	
	return nil
}

func (gm *GaussianMechanism) CalculateNoiseScale(sensitivity, epsilon, delta float64) float64 {
	// σ = sqrt(2 * ln(1.25/δ)) * sensitivity / ε
	return math.Sqrt(2*math.Log(1.25/delta)) * sensitivity / epsilon
}

// Helper methods for GaussianMechanism (similar to LaplaceMechanism)

func (gm *GaussianMechanism) calculateRange(data []float64) float64 {
	if len(data) == 0 {
		return 0.0
	}
	
	min, max := data[0], data[0]
	for _, val := range data[1:] {
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}
	
	return max - min
}

func (gm *GaussianMechanism) applyBounds(value float64) float64 {
	// Same implementation as LaplaceMechanism
	switch gm.clampingConfig.Strategy {
	case "clip":
		if value < gm.clampingConfig.LowerBound {
			return gm.clampingConfig.LowerBound
		}
		if value > gm.clampingConfig.UpperBound {
			return gm.clampingConfig.UpperBound
		}
		return value
	case "wrap":
		range_ := gm.clampingConfig.UpperBound - gm.clampingConfig.LowerBound
		if range_ <= 0 {
			return gm.clampingConfig.LowerBound
		}
		
		if value < gm.clampingConfig.LowerBound {
			return gm.clampingConfig.UpperBound - math.Mod(gm.clampingConfig.LowerBound-value, range_)
		}
		if value > gm.clampingConfig.UpperBound {
			return gm.clampingConfig.LowerBound + math.Mod(value-gm.clampingConfig.UpperBound, range_)
		}
		return value
	default:
		return gm.applyBounds(value)
	}
}

func (gm *GaussianMechanism) applyPostProcessing(data []float64) []float64 {
	// Same implementation as LaplaceMechanism
	result := make([]float64, len(data))
	copy(result, data)
	
	if gm.postProcessConfig.SmoothingWindow > 1 {
		result = gm.applySmoothing(result, gm.postProcessConfig.SmoothingWindow)
	}
	
	if gm.postProcessConfig.OutlierRemoval {
		result = gm.removeOutliers(result)
	}
	
	if gm.postProcessConfig.MinValue != nil || gm.postProcessConfig.MaxValue != nil {
		for i := range result {
			if gm.postProcessConfig.MinValue != nil && result[i] < *gm.postProcessConfig.MinValue {
				result[i] = *gm.postProcessConfig.MinValue
			}
			if gm.postProcessConfig.MaxValue != nil && result[i] > *gm.postProcessConfig.MaxValue {
				result[i] = *gm.postProcessConfig.MaxValue
			}
		}
	}
	
	return result
}

func (gm *GaussianMechanism) applySmoothing(data []float64, window int) []float64 {
	if window <= 1 || len(data) < window {
		return data
	}
	
	result := make([]float64, len(data))
	halfWindow := window / 2
	
	for i := range data {
		var sum float64
		var count int
		
		for j := max(0, i-halfWindow); j <= min(len(data)-1, i+halfWindow); j++ {
			sum += data[j]
			count++
		}
		
		result[i] = sum / float64(count)
	}
	
	return result
}

func (gm *GaussianMechanism) removeOutliers(data []float64) []float64 {
	if len(data) < 4 {
		return data
	}
	
	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)
	
	n := len(sorted)
	q1 := sorted[n/4]
	q3 := sorted[3*n/4]
	iqr := q3 - q1
	
	lowerBound := q1 - 1.5*iqr
	upperBound := q3 + 1.5*iqr
	
	result := make([]float64, len(data))
	for i, val := range data {
		if val < lowerBound {
			result[i] = q1
		} else if val > upperBound {
			result[i] = q3
		} else {
			result[i] = val
		}
	}
	
	return result
}

// ExponentialMechanism implementation

func NewExponentialMechanism(randSource *rand.Rand, clampingConfig *ClampingConfig, postProcessConfig *PostProcessConfig) *ExponentialMechanism {
	if randSource == nil {
		randSource = rand.New(rand.NewSource(42))
	}
	
	// Default scoring function (prefer values closer to median)
	defaultScoringFunction := func(data []float64, candidate float64) float64 {
		if len(data) == 0 {
			return 0.0
		}
		
		// Calculate distance to median
		sorted := make([]float64, len(data))
		copy(sorted, data)
		sort.Float64s(sorted)
		
		var median float64
		n := len(sorted)
		if n%2 == 0 {
			median = (sorted[n/2-1] + sorted[n/2]) / 2.0
		} else {
			median = sorted[n/2]
		}
		
		// Score is negative distance (higher score for closer values)
		return -math.Abs(candidate - median)
	}
	
	return &ExponentialMechanism{
		randSource:        randSource,
		clampingConfig:    clampingConfig,
		postProcessConfig: postProcessConfig,
		scoringFunction:   defaultScoringFunction,
	}
}

func (em *ExponentialMechanism) GetName() string {
	return "exponential"
}

func (em *ExponentialMechanism) GetDescription() string {
	return "Exponential mechanism selects outputs with probability proportional to their utility scores"
}

func (em *ExponentialMechanism) AddNoise(ctx context.Context, value float64, sensitivity float64, epsilon float64) (float64, error) {
	// For single values, add small perturbation using exponential mechanism
	candidates := []float64{
		value - sensitivity,
		value - sensitivity/2,
		value,
		value + sensitivity/2,
		value + sensitivity,
	}
	
	return em.selectCandidate([]float64{value}, candidates, sensitivity, epsilon), nil
}

func (em *ExponentialMechanism) AddNoiseToSeries(ctx context.Context, data []float64, sensitivity float64, epsilon float64) ([]float64, error) {
	if len(data) == 0 {
		return []float64{}, nil
	}
	
	result := make([]float64, len(data))
	
	// For each value, select from candidate set using exponential mechanism
	for i, value := range data {
		// Create candidate set around the original value
		candidates := em.generateCandidates(value, sensitivity)
		
		// Select candidate using exponential mechanism
		selected := em.selectCandidate(data, candidates, sensitivity, epsilon)
		result[i] = selected
		
		// Check for cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}
	
	// Apply post-processing if configured
	if em.postProcessConfig != nil && em.postProcessConfig.Enabled {
		result = em.applyPostProcessing(result)
	}
	
	return result, nil
}

func (em *ExponentialMechanism) CalculateSensitivity(data []float64, queryType QueryType) float64 {
	// Sensitivity for exponential mechanism is typically the sensitivity of the scoring function
	// For simplicity, use same calculation as other mechanisms
	if len(data) == 0 {
		return 0.0
	}
	
	return em.calculateRange(data)
}

func (em *ExponentialMechanism) ValidateParameters(epsilon, delta float64) error {
	if epsilon <= 0 {
		return fmt.Errorf("epsilon must be positive for exponential mechanism, got %f", epsilon)
	}
	
	// Exponential mechanism is pure DP, delta should be 0
	if delta != 0 {
		return fmt.Errorf("delta must be 0 for pure differential privacy (exponential mechanism), got %f", delta)
	}
	
	return nil
}

// Helper methods for ExponentialMechanism

func (em *ExponentialMechanism) generateCandidates(value, sensitivity float64) []float64 {
	// Generate candidate set around the value
	numCandidates := 11
	candidates := make([]float64, numCandidates)
	
	step := sensitivity / float64(numCandidates-1)
	start := value - sensitivity/2
	
	for i := 0; i < numCandidates; i++ {
		candidates[i] = start + float64(i)*step
	}
	
	return candidates
}

func (em *ExponentialMechanism) selectCandidate(data, candidates []float64, sensitivity, epsilon float64) float64 {
	if len(candidates) == 0 {
		return 0.0
	}
	
	if len(candidates) == 1 {
		return candidates[0]
	}
	
	// Calculate scores for all candidates
	scores := make([]float64, len(candidates))
	maxScore := math.Inf(-1)
	
	for i, candidate := range candidates {
		scores[i] = em.scoringFunction(data, candidate)
		if scores[i] > maxScore {
			maxScore = scores[i]
		}
	}
	
	// Calculate probabilities using exponential mechanism
	// P(output) ∝ exp(ε * score / (2 * sensitivity))
	probabilities := make([]float64, len(candidates))
	var sumProb float64
	
	for i, score := range scores {
		// Normalize scores to prevent overflow
		normalizedScore := score - maxScore
		prob := math.Exp(epsilon * normalizedScore / (2 * sensitivity))
		probabilities[i] = prob
		sumProb += prob
	}
	
	// Normalize probabilities
	for i := range probabilities {
		probabilities[i] /= sumProb
	}
	
	// Sample from categorical distribution
	return em.sampleCategorical(candidates, probabilities)
}

func (em *ExponentialMechanism) sampleCategorical(candidates []float64, probabilities []float64) float64 {
	r := em.randSource.Float64()
	cumulative := 0.0
	
	for i, prob := range probabilities {
		cumulative += prob
		if r <= cumulative {
			return candidates[i]
		}
	}
	
	// Fallback to last candidate
	return candidates[len(candidates)-1]
}

func (em *ExponentialMechanism) calculateRange(data []float64) float64 {
	if len(data) == 0 {
		return 0.0
	}
	
	min, max := data[0], data[0]
	for _, val := range data[1:] {
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}
	
	return max - min
}

func (em *ExponentialMechanism) applyPostProcessing(data []float64) []float64 {
	// Same implementation as other mechanisms
	result := make([]float64, len(data))
	copy(result, data)
	
	if em.postProcessConfig != nil {
		if em.postProcessConfig.SmoothingWindow > 1 {
			result = em.applySmoothing(result, em.postProcessConfig.SmoothingWindow)
		}
		
		if em.postProcessConfig.OutlierRemoval {
			result = em.removeOutliers(result)
		}
		
		if em.postProcessConfig.MinValue != nil || em.postProcessConfig.MaxValue != nil {
			for i := range result {
				if em.postProcessConfig.MinValue != nil && result[i] < *em.postProcessConfig.MinValue {
					result[i] = *em.postProcessConfig.MinValue
				}
				if em.postProcessConfig.MaxValue != nil && result[i] > *em.postProcessConfig.MaxValue {
					result[i] = *em.postProcessConfig.MaxValue
				}
			}
		}
	}
	
	return result
}

func (em *ExponentialMechanism) applySmoothing(data []float64, window int) []float64 {
	if window <= 1 || len(data) < window {
		return data
	}
	
	result := make([]float64, len(data))
	halfWindow := window / 2
	
	for i := range data {
		var sum float64
		var count int
		
		for j := max(0, i-halfWindow); j <= min(len(data)-1, i+halfWindow); j++ {
			sum += data[j]
			count++
		}
		
		result[i] = sum / float64(count)
	}
	
	return result
}

func (em *ExponentialMechanism) removeOutliers(data []float64) []float64 {
	if len(data) < 4 {
		return data
	}
	
	sorted := make([]float64, len(data))
	copy(sorted, data)
	sort.Float64s(sorted)
	
	n := len(sorted)
	q1 := sorted[n/4]
	q3 := sorted[3*n/4]
	iqr := q3 - q1
	
	lowerBound := q1 - 1.5*iqr
	upperBound := q3 + 1.5*iqr
	
	result := make([]float64, len(data))
	for i, val := range data {
		if val < lowerBound {
			result[i] = q1
		} else if val > upperBound {
			result[i] = q3
		} else {
			result[i] = val
		}
	}
	
	return result
}

// Utility functions

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}