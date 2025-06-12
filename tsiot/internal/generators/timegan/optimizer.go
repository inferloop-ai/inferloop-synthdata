package timegan

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// AdamOptimizer implements the Adam optimization algorithm
type AdamOptimizer struct {
	learningRate float64
	beta1        float64
	beta2        float64
	epsilon      float64
	t            int // time step
	m            []*mat.Dense // first moment estimate
	v            []*mat.Dense // second moment estimate
}

// NewAdamOptimizer creates a new Adam optimizer
func NewAdamOptimizer(learningRate float64) *AdamOptimizer {
	return &AdamOptimizer{
		learningRate: learningRate,
		beta1:        0.9,
		beta2:        0.999,
		epsilon:      1e-8,
		t:            0,
		m:            make([]*mat.Dense, 0),
		v:            make([]*mat.Dense, 0),
	}
}

// Step performs one optimization step
// In a real implementation, this would compute gradients and update weights
// For this simulation, we just increment the time step
func (opt *AdamOptimizer) Step() {
	opt.t++
	// In a real implementation, this would:
	// 1. Compute gradients
	// 2. Update first and second moment estimates
	// 3. Compute bias-corrected estimates
	// 4. Update parameters
}

// UpdateWeights updates network weights using Adam algorithm
func (opt *AdamOptimizer) UpdateWeights(weights []*mat.Dense, gradients []*mat.Dense) {
	opt.t++
	
	// Initialize moment estimates if needed
	if len(opt.m) != len(weights) {
		opt.initializeMoments(weights)
	}
	
	for i, weight := range weights {
		if i >= len(gradients) {
			continue
		}
		
		gradient := gradients[i]
		rows, cols := weight.Dims()
		
		// Update biased first moment estimate
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				g := gradient.At(r, c)
				m := opt.beta1*opt.m[i].At(r, c) + (1-opt.beta1)*g
				opt.m[i].Set(r, c, m)
			}
		}
		
		// Update biased second raw moment estimate
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				g := gradient.At(r, c)
				v := opt.beta2*opt.v[i].At(r, c) + (1-opt.beta2)*g*g
				opt.v[i].Set(r, c, v)
			}
		}
		
		// Compute bias-corrected first moment estimate
		mHat := mat.NewDense(rows, cols, nil)
		beta1Correction := 1 - math.Pow(opt.beta1, float64(opt.t))
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				mHat.Set(r, c, opt.m[i].At(r, c)/beta1Correction)
			}
		}
		
		// Compute bias-corrected second raw moment estimate
		vHat := mat.NewDense(rows, cols, nil)
		beta2Correction := 1 - math.Pow(opt.beta2, float64(opt.t))
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				vHat.Set(r, c, opt.v[i].At(r, c)/beta2Correction)
			}
		}
		
		// Update parameters
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				mhat := mHat.At(r, c)
				vhat := vHat.At(r, c)
				update := opt.learningRate * mhat / (math.Sqrt(vhat) + opt.epsilon)
				newWeight := weight.At(r, c) - update
				weight.Set(r, c, newWeight)
			}
		}
	}
}

// initializeMoments initializes the moment estimates
func (opt *AdamOptimizer) initializeMoments(weights []*mat.Dense) {
	opt.m = make([]*mat.Dense, len(weights))
	opt.v = make([]*mat.Dense, len(weights))
	
	for i, weight := range weights {
		rows, cols := weight.Dims()
		opt.m[i] = mat.NewDense(rows, cols, nil) // Initialize to zero
		opt.v[i] = mat.NewDense(rows, cols, nil) // Initialize to zero
	}
}

// GetLearningRate returns the current learning rate
func (opt *AdamOptimizer) GetLearningRate() float64 {
	return opt.learningRate
}

// SetLearningRate sets the learning rate
func (opt *AdamOptimizer) SetLearningRate(lr float64) {
	opt.learningRate = lr
}

// GetTimeStep returns the current time step
func (opt *AdamOptimizer) GetTimeStep() int {
	return opt.t
}

// Reset resets the optimizer state
func (opt *AdamOptimizer) Reset() {
	opt.t = 0
	opt.m = make([]*mat.Dense, 0)
	opt.v = make([]*mat.Dense, 0)
}
