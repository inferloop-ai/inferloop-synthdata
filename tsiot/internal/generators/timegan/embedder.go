package timegan

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Embedder maps original space to latent space
type Embedder struct {
	inputDim  int
	hiddenDim int
	numLayers int
	weights   []*mat.Dense
	biases    []*mat.Dense
	dropout   float64
}

// NewEmbedder creates a new embedder network
func NewEmbedder(inputDim, hiddenDim, numLayers int) (*Embedder, error) {
	e := &Embedder{
		inputDim:  inputDim,
		hiddenDim: hiddenDim,
		numLayers: numLayers,
		dropout:   0.1,
		weights:   make([]*mat.Dense, numLayers),
		biases:    make([]*mat.Dense, numLayers),
	}

	// Initialize weights and biases
	for i := 0; i < numLayers; i++ {
		var rows, cols int
		if i == 0 {
			rows, cols = hiddenDim, inputDim
		} else {
			rows, cols = hiddenDim, hiddenDim
		}

		// Xavier initialization
		weight := mat.NewDense(rows, cols, nil)
		scale := math.Sqrt(2.0 / float64(cols))
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				weight.Set(r, c, rand.NormFloat64()*scale)
			}
		}
		e.weights[i] = weight

		// Initialize biases to zero
		bias := mat.NewDense(rows, 1, nil)
		e.biases[i] = bias
	}

	return e, nil
}

// Forward performs forward pass through the embedder
func (e *Embedder) Forward(input *mat.Dense) *mat.Dense {
	activation := input

	for i := 0; i < e.numLayers; i++ {
		// Linear transformation: W * x + b
		linear := &mat.Dense{}
		linear.Mul(e.weights[i], activation)
		linear.Add(linear, e.biases[i])

		// Apply activation function (ReLU for hidden layers, linear for output)
		if i < e.numLayers-1 {
			activation = e.relu(linear)
			// Apply dropout during training
			activation = e.applyDropout(activation)
		} else {
			activation = linear
		}
	}

	return activation
}

// relu applies ReLU activation function
func (e *Embedder) relu(input *mat.Dense) *mat.Dense {
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
func (e *Embedder) applyDropout(input *mat.Dense) *mat.Dense {
	rows, cols := input.Dims()
	output := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if rand.Float64() > e.dropout {
				output.Set(i, j, input.At(i, j)/(1-e.dropout))
			} else {
				output.Set(i, j, 0)
			}
		}
	}

	return output
}

// GetWeights returns the network weights
func (e *Embedder) GetWeights() []*mat.Dense {
	return e.weights
}

// SetWeights sets the network weights
func (e *Embedder) SetWeights(weights []*mat.Dense) {
	e.weights = weights
}

// GetBiases returns the network biases
func (e *Embedder) GetBiases() []*mat.Dense {
	return e.biases
}

// SetBiases sets the network biases
func (e *Embedder) SetBiases(biases []*mat.Dense) {
	e.biases = biases
}
