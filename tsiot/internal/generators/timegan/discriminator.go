package timegan

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Discriminator distinguishes between real and synthetic data in latent space
type Discriminator struct {
	inputDim  int
	outputDim int
	numLayers int
	weights   []*mat.Dense
	biases    []*mat.Dense
	dropout   float64
}

// NewDiscriminator creates a new discriminator network
func NewDiscriminator(inputDim, outputDim, numLayers int) (*Discriminator, error) {
	d := &Discriminator{
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
		d.weights[i] = weight

		// Initialize biases to zero
		bias := mat.NewDense(rows, 1, nil)
		d.biases[i] = bias
	}

	return d, nil
}

// Forward performs forward pass through the discriminator
func (d *Discriminator) Forward(input *mat.Dense) *mat.Dense {
	activation := input

	for i := 0; i < d.numLayers; i++ {
		// Linear transformation: W * x + b
		linear := &mat.Dense{}
		linear.Mul(d.weights[i], activation)
		linear.Add(linear, d.biases[i])

		// Apply activation function
		if i < d.numLayers-1 {
			// LeakyReLU for hidden layers (better for discriminator)
			activation = d.leakyRelu(linear)
			// Apply dropout during training
			activation = d.applyDropout(activation)
		} else {
			// Sigmoid for output layer (binary classification)
			activation = d.sigmoid(linear)
		}
	}

	return activation
}

// leakyRelu applies Leaky ReLU activation function
func (d *Discriminator) leakyRelu(input *mat.Dense) *mat.Dense {
	rows, cols := input.Dims()
	output := mat.NewDense(rows, cols, nil)
	alpha := 0.01 // Slope for negative values

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := input.At(i, j)
			if val > 0 {
				output.Set(i, j, val)
			} else {
				output.Set(i, j, alpha*val)
			}
		}
	}

	return output
}

// sigmoid applies sigmoid activation function
func (d *Discriminator) sigmoid(input *mat.Dense) *mat.Dense {
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
func (d *Discriminator) applyDropout(input *mat.Dense) *mat.Dense {
	rows, cols := input.Dims()
	output := mat.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if rand.Float64() > d.dropout {
				output.Set(i, j, input.At(i, j)/(1-d.dropout))
			} else {
				output.Set(i, j, 0)
			}
		}
	}

	return output
}

// GetWeights returns the network weights
func (d *Discriminator) GetWeights() []*mat.Dense {
	return d.weights
}

// SetWeights sets the network weights
func (d *Discriminator) SetWeights(weights []*mat.Dense) {
	d.weights = weights
}

// GetBiases returns the network biases
func (d *Discriminator) GetBiases() []*mat.Dense {
	return d.biases
}

// SetBiases sets the network biases
func (d *Discriminator) SetBiases(biases []*mat.Dense) {
	d.biases = biases
}
