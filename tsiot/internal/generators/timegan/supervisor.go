package timegan

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Supervisor predicts next timestep in latent space for temporal consistency
type Supervisor struct {
	inputDim  int
	outputDim int
	numLayers int
	weights   []*mat.Dense
	biases    []*mat.Dense
	dropout   float64
}

// NewSupervisor creates a new supervisor network
func NewSupervisor(inputDim, outputDim, numLayers int) (*Supervisor, error) {
	s := &Supervisor{
		inputDim:  inputDim,
		outputDim: outputDim,
		numLayers: numLayers,
		dropout:   0.1,
		weights:   make([]*mat.Dense, numLayers),
		biases:    make([]*mat.Dense, numLayers),
	}

	// Initialize weights and biases
	for i := 0; i < numLayers; i++ {
		var rows, cols int
		if i == 0 {
			rows, cols = inputDim, inputDim
		} else if i == numLayers-1 {
			rows, cols = outputDim, inputDim
		} else {
			rows, cols = inputDim, inputDim
		}

		// Xavier initialization
		weight := mat.NewDense(rows, cols, nil)
		scale := math.Sqrt(2.0 / float64(cols))
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				weight.Set(r, c, rand.NormFloat64()*scale)
			}
		}
		s.weights[i] = weight

		// Initialize biases to zero
		bias := mat.NewDense(rows, 1, nil)
		s.biases[i] = bias
	}

	return s, nil
}

// Forward performs forward pass through the supervisor
func (s *Supervisor) Forward(input *mat.Dense) *mat.Dense {
	activation := input

	for i := 0; i < s.numLayers; i++ {
		// Linear transformation: W * x + b
		linear := &mat.Dense{}
		linear.Mul(s.weights[i], activation)
		linear.Add(linear, s.biases[i])

		// Apply activation function
		if i < s.numLayers-1 {
			// ReLU for hidden layers
			activation = s.relu(linear)
			// Apply dropout during training
			activation = s.applyDropout(activation)
		} else {
			// Linear activation for output layer (regression task)
			activation = linear
		}
	}

	return activation
}

// relu applies ReLU activation function
func (s *Supervisor) relu(input *mat.Dense) *mat.Dense {
	rows, cols := input.Dims()
	output := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := input.At(i, j)
			if val > 0 {
				output.Set(i, j, val)
			} else {
				output.Set(i, j, 0)
			}
		}
	}

	return output
}

// applyDropout applies dropout regularization
func (s *Supervisor) applyDropout(input *mat.Dense) *mat.Dense {
	rows, cols := input.Dims()
	output := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if rand.Float64() > s.dropout {
				output.Set(i, j, input.At(i, j)/(1-s.dropout))
			} else {
				output.Set(i, j, 0)
			}
		}
	}

	return output
}

// GetWeights returns the network weights
func (s *Supervisor) GetWeights() []*mat.Dense {
	return s.weights
}

// SetWeights sets the network weights
func (s *Supervisor) SetWeights(weights []*mat.Dense) {
	s.weights = weights
}

// GetBiases returns the network biases
func (s *Supervisor) GetBiases() []*mat.Dense {
	return s.biases
}

// SetBiases sets the network biases
func (s *Supervisor) SetBiases(biases []*mat.Dense) {
	s.biases = biases
}
