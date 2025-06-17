package ydata

import (
	"math"
	"math/rand"
	"time"

	"github.com/sirupsen/logrus"
)

// DifferentialPrivacyMechanism implements differential privacy mechanisms
type DifferentialPrivacyMechanism struct {
	epsilon     float64
	delta       float64
	noiseType   string
	randSource  *rand.Rand
	logger      *logrus.Logger
}

// NoiseParameters contains parameters for noise generation
type NoiseParameters struct {
	Sensitivity float64 `json:"sensitivity"`
	Scale       float64 `json:"scale"`
	Variance    float64 `json:"variance"`
	Clipping    float64 `json:"clipping"`
}

// NewDifferentialPrivacyMechanism creates a new DP mechanism
func NewDifferentialPrivacyMechanism(epsilon, delta float64, noiseType string) *DifferentialPrivacyMechanism {
	return &DifferentialPrivacyMechanism{
		epsilon:    epsilon,
		delta:      delta,
		noiseType:  noiseType,
		randSource: rand.New(rand.NewSource(time.Now().UnixNano())),
		logger:     logrus.New(),
	}
}

// AddNoise adds differential privacy noise to data
func (dp *DifferentialPrivacyMechanism) AddNoise(data []float64) []float64 {
	switch dp.noiseType {
	case "laplace":
		return dp.addLaplaceNoise(data)
	case "gaussian":
		return dp.addGaussianNoise(data)
	case "exponential":
		return dp.addExponentialNoise(data)
	default:
		return dp.addGaussianNoise(data)
	}
}

// addLaplaceNoise adds Laplace noise for pure differential privacy
func (dp *DifferentialPrivacyMechanism) addLaplaceNoise(data []float64) []float64 {
	result := make([]float64, len(data))
	sensitivity := dp.calculateSensitivity(data)
	scale := sensitivity / dp.epsilon
	
	for i, value := range data {
		noise := dp.sampleLaplace(scale)
		result[i] = value + noise
	}
	
	return result
}

// addGaussianNoise adds Gaussian noise for approximate differential privacy
func (dp *DifferentialPrivacyMechanism) addGaussianNoise(data []float64) []float64 {
	result := make([]float64, len(data))
	sensitivity := dp.calculateSensitivity(data)
	
	// Calculate noise variance for (epsilon, delta)-DP
	variance := dp.calculateGaussianVariance(sensitivity)
	stddev := math.Sqrt(variance)
	
	for i, value := range data {
		noise := dp.randSource.NormFloat64() * stddev
		result[i] = value + noise
	}
	
	return result
}

// addExponentialNoise adds exponential mechanism noise
func (dp *DifferentialPrivacyMechanism) addExponentialNoise(data []float64) []float64 {
	result := make([]float64, len(data))
	sensitivity := dp.calculateSensitivity(data)
	scale := 2 * sensitivity / dp.epsilon
	
	for i, value := range data {
		// Exponential mechanism with utility proportional to negative squared distance
		noise := dp.sampleExponential(scale)
		result[i] = value + noise
	}
	
	return result
}

// sampleLaplace samples from Laplace distribution
func (dp *DifferentialPrivacyMechanism) sampleLaplace(scale float64) float64 {
	u := dp.randSource.Float64() - 0.5
	if u >= 0 {
		return -scale * math.Log(1-2*u)
	}
	return scale * math.Log(1+2*u)
}

// sampleExponential samples from exponential distribution
func (dp *DifferentialPrivacyMechanism) sampleExponential(scale float64) float64 {
	u := dp.randSource.Float64()
	return -scale * math.Log(u)
}

// calculateSensitivity calculates the sensitivity of the data
func (dp *DifferentialPrivacyMechanism) calculateSensitivity(data []float64) float64 {
	if len(data) <= 1 {
		return 1.0
	}
	
	// Use range as a proxy for sensitivity
	min, max := data[0], data[0]
	for _, v := range data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	
	sensitivity := max - min
	if sensitivity == 0 {
		return 1.0
	}
	return sensitivity
}

// calculateGaussianVariance calculates variance for Gaussian noise
func (dp *DifferentialPrivacyMechanism) calculateGaussianVariance(sensitivity float64) float64 {
	// For (epsilon, delta)-DP: sigma^2 >= 2 * log(1.25/delta) * sensitivity^2 / epsilon^2
	if dp.delta <= 0 {
		// Fallback to pure DP approximation
		return 2 * sensitivity * sensitivity / (dp.epsilon * dp.epsilon)
	}
	
	logTerm := math.Log(1.25 / dp.delta)
	variance := 2 * logTerm * sensitivity * sensitivity / (dp.epsilon * dp.epsilon)
	return variance
}

// ClipGradients clips gradients for gradient-based training
func (dp *DifferentialPrivacyMechanism) ClipGradients(gradients []float64, clippingNorm float64) []float64 {
	// Calculate L2 norm of gradients
	norm := 0.0
	for _, g := range gradients {
		norm += g * g
	}
	norm = math.Sqrt(norm)
	
	// Clip if norm exceeds threshold
	if norm > clippingNorm {
		scale := clippingNorm / norm
		result := make([]float64, len(gradients))
		for i, g := range gradients {
			result[i] = g * scale
		}
		return result
	}
	
	return gradients
}

// AddNoiseToGradients adds DP noise to clipped gradients
func (dp *DifferentialPrivacyMechanism) AddNoiseToGradients(gradients []float64, clippingNorm float64) []float64 {
	// First clip the gradients
	clipped := dp.ClipGradients(gradients, clippingNorm)
	
	// Calculate noise scale
	noiseScale := clippingNorm / dp.epsilon
	if dp.noiseType == "gaussian" {
		variance := dp.calculateGaussianVariance(clippingNorm)
		noiseScale = math.Sqrt(variance)
	}
	
	// Add noise
	result := make([]float64, len(clipped))
	for i, g := range clipped {
		var noise float64
		switch dp.noiseType {
		case "laplace":
			noise = dp.sampleLaplace(noiseScale)
		case "gaussian":
			noise = dp.randSource.NormFloat64() * noiseScale
		default:
			noise = dp.randSource.NormFloat64() * noiseScale
		}
		result[i] = g + noise
	}
	
	return result
}

// CalculatePrivacyLoss calculates the privacy loss for a given operation
func (dp *DifferentialPrivacyMechanism) CalculatePrivacyLoss(operation string, dataSize int) float64 {
	// Basic privacy accounting - in practice would use more sophisticated methods
	baseLoss := dp.epsilon
	
	// Adjust based on operation type and data size
	switch operation {
	case "training":
		// Training typically requires more privacy budget
		return baseLoss * math.Log(float64(dataSize))
	case "generation":
		// Generation uses less budget
		return baseLoss * 0.5
	case "validation":
		// Validation uses minimal budget
		return baseLoss * 0.1
	default:
		return baseLoss
	}
}

// GetNoiseParameters returns the current noise parameters
func (dp *DifferentialPrivacyMechanism) GetNoiseParameters(dataSize int) NoiseParameters {
	sensitivity := 1.0 // Default sensitivity
	if dataSize > 0 {
		sensitivity = math.Sqrt(float64(dataSize))
	}
	
	var scale, variance float64
	switch dp.noiseType {
	case "laplace":
		scale = sensitivity / dp.epsilon
		variance = 2 * scale * scale
	case "gaussian":
		variance = dp.calculateGaussianVariance(sensitivity)
		scale = math.Sqrt(variance)
	default:
		variance = dp.calculateGaussianVariance(sensitivity)
		scale = math.Sqrt(variance)
	}
	
	return NoiseParameters{
		Sensitivity: sensitivity,
		Scale:       scale,
		Variance:    variance,
		Clipping:    sensitivity, // Use sensitivity as default clipping
	}
}

// ValidatePrivacyParameters validates the privacy parameters
func (dp *DifferentialPrivacyMechanism) ValidatePrivacyParameters() error {
	if dp.epsilon <= 0 {
		return fmt.Errorf("epsilon must be positive, got %f", dp.epsilon)
	}
	
	if dp.delta < 0 || dp.delta >= 1 {
		return fmt.Errorf("delta must be in [0, 1), got %f", dp.delta)
	}
	
	validNoiseTypes := map[string]bool{
		"gaussian":    true,
		"laplace":     true,
		"exponential": true,
	}
	
	if !validNoiseTypes[dp.noiseType] {
		return fmt.Errorf("invalid noise type: %s", dp.noiseType)
	}
	
	return nil
}

// SetEpsilon updates the epsilon parameter
func (dp *DifferentialPrivacyMechanism) SetEpsilon(epsilon float64) error {
	if epsilon <= 0 {
		return fmt.Errorf("epsilon must be positive")
	}
	dp.epsilon = epsilon
	return nil
}

// SetDelta updates the delta parameter
func (dp *DifferentialPrivacyMechanism) SetDelta(delta float64) error {
	if delta < 0 || delta >= 1 {
		return fmt.Errorf("delta must be in [0, 1)")
	}
	dp.delta = delta
	return nil
}

// GetEpsilon returns the current epsilon value
func (dp *DifferentialPrivacyMechanism) GetEpsilon() float64 {
	return dp.epsilon
}

// GetDelta returns the current delta value
func (dp *DifferentialPrivacyMechanism) GetDelta() float64 {
	return dp.delta
}

// GetNoiseType returns the current noise type
func (dp *DifferentialPrivacyMechanism) GetNoiseType() string {
	return dp.noiseType
}
