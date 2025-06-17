package timegan

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Generator creates synthetic data in latent space
type Generator struct {
	inputDim  int
	hiddenDim int
	numLayers int
	weights   []*mat.Dense
	biases    []*mat.Dense
	dropout   float64
}

// NewGenerator creates a new generator network
func NewGenerator(inputDim, hiddenDim, numLayers int) (*Generator, error) {
	g := &Generator{
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
		} else if i == numLayers-1 {
			rows, cols = hiddenDim, hiddenDim
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
		g.weights[i] = weight

		// Initialize biases to zero
		bias := mat.NewDense(rows, 1, nil)
		g.biases[i] = bias
	}

	return g, nil
}

// Forward performs forward pass through the generator
func (g *Generator) Forward(input *mat.Dense) *mat.Dense {
	activation := input

	for i := 0; i < g.numLayers; i++ {
		// Linear transformation: W * x + b
		linear := &mat.Dense{}
		linear.Mul(g.weights[i], activation)
		linear.Add(linear, g.biases[i])

		// Apply activation function
		if i < g.numLayers-1 {
			// ReLU for hidden layers
			activation = g.relu(linear)
			// Apply dropout during training
			activation = g.applyDropout(activation)
		} else {
			// Tanh for output layer to generate values in [-1,1]
			activation = g.tanh(linear)
		}
	}

	return activation
}

// relu applies ReLU activation function
func (g *Generator) relu(input *mat.Dense) *mat.Dense {
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

// tanh applies hyperbolic tangent activation function
func (g *Generator) tanh(input *mat.Dense) *mat.Dense {
	rows, cols := input.Dims()
	output := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := input.At(i, j)
			output.Set(i, j, math.Tanh(val))
		}
	}

	return output
}

// applyDropout applies dropout regularization
func (g *Generator) applyDropout(input *mat.Dense) *mat.Dense {
	rows, cols := input.Dims()
	output := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if rand.Float64() > g.dropout {
				output.Set(i, j, input.At(i, j)/(1-g.dropout))
			} else {
				output.Set(i, j, 0)
			}
		}
	}

	return output
}

// GetWeights returns the network weights
func (g *Generator) GetWeights() []*mat.Dense {
	return g.weights
}

// SetWeights sets the network weights
func (g *Generator) SetWeights(weights []*mat.Dense) {
	g.weights = weights
}

// GetBiases returns the network biases
func (g *Generator) GetBiases() []*mat.Dense {
	return g.biases
}

// SetBiases sets the network biases
func (g *Generator) SetBiases(biases []*mat.Dense) {
	g.biases = biases
}
