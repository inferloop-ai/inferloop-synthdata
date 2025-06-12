package timegan

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Recovery maps latent space back to original space
type Recovery struct {
	inputDim  int
	outputDim int
	numLayers int
	weights   []*mat.Dense
	biases    []*mat.Dense
	dropout   float64
}

// NewRecovery creates a new recovery network
func NewRecovery(inputDim, outputDim, numLayers int) (*Recovery, error) {
	r := &Recovery{
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
		r.weights[i] = weight

		// Initialize biases to zero
		bias := mat.NewDense(rows, 1, nil)
		r.biases[i] = bias
	}

	return r, nil
}

// Forward performs forward pass through the recovery network
func (r *Recovery) Forward(input *mat.Dense) *mat.Dense {
	activation := input

	for i := 0; i < r.numLayers; i++ {
		// Linear transformation: W * x + b
		linear := &mat.Dense{}
		linear.Mul(r.weights[i], activation)
		linear.Add(linear, r.biases[i])

		// Apply activation function
		if i < r.numLayers-1 {
			// ReLU for hidden layers
			activation = r.relu(linear)
			// Apply dropout during training
			activation = r.applyDropout(activation)
		} else {
			// Sigmoid for output layer to ensure values are in [0,1]
			activation = r.sigmoid(linear)
		}
	}

	return activation
}

// relu applies ReLU activation function
func (r *Recovery) relu(input *mat.Dense) *mat.Dense {
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

// sigmoid applies sigmoid activation function
func (r *Recovery) sigmoid(input *mat.Dense) *mat.Dense {
	rows, cols := input.Dims()
	output := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := input.At(i, j)
			output.Set(i, j, 1.0/(1.0+math.Exp(-val)))
		}
	}

	return output
}

// applyDropout applies dropout regularization
func (r *Recovery) applyDropout(input *mat.Dense) *mat.Dense {
	rows, cols := input.Dims()
	output := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if rand.Float64() > r.dropout {
				output.Set(i, j, input.At(i, j)/(1-r.dropout))
			} else {
				output.Set(i, j, 0)
			}
		}
	}

	return output
}

// GetWeights returns the network weights
func (r *Recovery) GetWeights() []*mat.Dense {
	return r.weights
}

// SetWeights sets the network weights
func (r *Recovery) SetWeights(weights []*mat.Dense) {
	r.weights = weights
}

// GetBiases returns the network biases
func (r *Recovery) GetBiases() []*mat.Dense {
	return r.biases
}

// SetBiases sets the network biases
func (r *Recovery) SetBiases(biases []*mat.Dense) {
	r.biases = biases
}
