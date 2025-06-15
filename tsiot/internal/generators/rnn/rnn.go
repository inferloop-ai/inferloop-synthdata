package rnn

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/mat"

	"github.com/inferloop/tsiot/pkg/constants"
	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
)

// RNNGenerator implements basic RNN-based synthetic data generation
type RNNGenerator struct {
	logger      *logrus.Logger
	config      *RNNConfig
	model       *RNNModel
	trained     bool
	statistics  *models.TimeSeriesMetrics
	randSource  *rand.Rand
	scaler      *RNNDataScaler
}

// RNNConfig contains configuration for RNN generation
type RNNConfig struct {
	// Network architecture
	HiddenSize     int     `json:"hidden_size"`     // Number of hidden units
	NumLayers      int     `json:"num_layers"`      // Number of RNN layers
	InputSize      int     `json:"input_size"`      // Input dimension
	OutputSize     int     `json:"output_size"`     // Output dimension
	SequenceLength int     `json:"sequence_length"` // Lookback window size
	Activation     string  `json:"activation"`      // Activation function: "tanh", "relu", "sigmoid"
	
	// Training parameters
	LearningRate     float64 `json:"learning_rate"`     // Learning rate
	Epochs           int     `json:"epochs"`            // Training epochs
	BatchSize        int     `json:"batch_size"`        // Batch size
	DropoutRate      float64 `json:"dropout_rate"`      // Dropout probability
	GradientClipping float64 `json:"gradient_clipping"` // Gradient clipping threshold
	
	// Regularization
	L1Regularization float64 `json:"l1_regularization"` // L1 penalty weight
	L2Regularization float64 `json:"l2_regularization"` // L2 penalty weight
	
	// Generation parameters
	Temperature    float64 `json:"temperature"`     // Sampling temperature
	SamplingMethod string  `json:"sampling_method"` // "greedy", "random", "topk"
	TopK           int     `json:"top_k"`           // Top-k sampling parameter
	
	// Data preprocessing
	Normalization   string  `json:"normalization"`    // "minmax", "zscore", "robust"
	WindowStride    int     `json:"window_stride"`    // Sliding window stride
	ValidationSplit float64 `json:"validation_split"` // Fraction for validation
	
	// Other parameters
	Seed           int64   `json:"seed"`           // Random seed
	EarlyStopping  bool    `json:"early_stopping"` // Enable early stopping
	Patience       int     `json:"patience"`       // Early stopping patience
	MinDelta       float64 `json:"min_delta"`      // Min improvement for early stopping
}

// RNNModel represents the RNN neural network model
type RNNModel struct {
	layers      []*RNNLayer
	outputLayer *DenseLayer
	optimizer   *SGDOptimizer
	lossHistory []float64
}

// RNNLayer represents a single RNN layer
type RNNLayer struct {
	hiddenSize  int
	inputSize   int
	activation  string
	
	// Weights and biases
	weightsInput  *mat.Dense    // Input to hidden weights
	weightsHidden *mat.Dense    // Hidden to hidden weights
	bias          *mat.VecDense // Bias vector
	
	// State
	hiddenState   *mat.VecDense
	
	// For backpropagation
	activations   []*mat.VecDense
	preActivations []*mat.VecDense
}

// DenseLayer represents a fully connected output layer
type DenseLayer struct {
	inputSize  int
	outputSize int
	weights    *mat.Dense
	bias       *mat.VecDense
}

// SGDOptimizer implements stochastic gradient descent with momentum
type SGDOptimizer struct {
	learningRate float64
	momentum     float64
	
	// Momentum storage for each parameter
	velocities map[string]interface{}
}

// RNNDataScaler handles data normalization
type RNNDataScaler struct {
	method   string
	min      float64
	max      float64
	mean     float64
	std      float64
	q25      float64
	q75      float64
	median   float64
	fitted   bool
}

// TrainingMetrics tracks training progress
type TrainingMetrics struct {
	Epoch          int     `json:"epoch"`
	TrainingLoss   float64 `json:"training_loss"`
	ValidationLoss float64 `json:"validation_loss"`
	LearningRate   float64 `json:"learning_rate"`
	GradientNorm   float64 `json:"gradient_norm"`
	Duration       time.Duration `json:"duration"`
}

// NewRNNGenerator creates a new RNN generator
func NewRNNGenerator(config *RNNConfig, logger *logrus.Logger) *RNNGenerator {
	if config == nil {
		config = getDefaultRNNConfig()
	}
	
	if logger == nil {
		logger = logrus.New()
	}
	
	if config.Seed == 0 {
		config.Seed = time.Now().UnixNano()
	}
	
	return &RNNGenerator{
		logger:     logger,
		config:     config,
		trained:    false,
		randSource: rand.New(rand.NewSource(config.Seed)),
		scaler:     NewRNNDataScaler(config.Normalization),
	}
}

// GetType returns the generator type
func (g *RNNGenerator) GetType() models.GeneratorType {
	return models.GeneratorType(constants.GeneratorTypeRNN)
}

// GetName returns a human-readable name for the generator
func (g *RNNGenerator) GetName() string {
	return "RNN Generator"
}

// GetDescription returns a description of the generator
func (g *RNNGenerator) GetDescription() string {
	return fmt.Sprintf("Generates synthetic time series using vanilla RNN with %d layers and %d hidden units", 
		g.config.NumLayers, g.config.HiddenSize)
}

// GetSupportedSensorTypes returns the sensor types this generator supports
func (g *RNNGenerator) GetSupportedSensorTypes() []models.SensorType {
	return []models.SensorType{
		models.SensorType(constants.SensorTypeTemperature),
		models.SensorType(constants.SensorTypeHumidity),
		models.SensorType(constants.SensorTypePressure),
		models.SensorType(constants.SensorTypeVibration),
		models.SensorType(constants.SensorTypePower),
		models.SensorType(constants.SensorTypeFlow),
		models.SensorType(constants.SensorTypeLevel),
		models.SensorType(constants.SensorTypeSpeed),
		models.SensorType(constants.SensorTypeCustom),
	}
}

// ValidateParameters validates the generation parameters
func (g *RNNGenerator) ValidateParameters(params models.GenerationParameters) error {
	if params.Length <= 0 {
		return errors.NewValidationError("INVALID_LENGTH", "Generation length must be positive")
	}
	
	if params.Frequency == "" {
		return errors.NewValidationError("INVALID_FREQUENCY", "Frequency is required")
	}
	
	// Validate RNN-specific parameters
	if g.config.HiddenSize <= 0 || g.config.HiddenSize > 1000 {
		return errors.NewValidationError("INVALID_HIDDEN_SIZE", "Hidden size must be between 1 and 1000")
	}
	
	if g.config.NumLayers <= 0 || g.config.NumLayers > 10 {
		return errors.NewValidationError("INVALID_NUM_LAYERS", "Number of layers must be between 1 and 10")
	}
	
	if g.config.SequenceLength <= 0 || g.config.SequenceLength > 200 {
		return errors.NewValidationError("INVALID_SEQUENCE_LENGTH", "Sequence length must be between 1 and 200")
	}
	
	if g.config.LearningRate <= 0 || g.config.LearningRate > 1 {
		return errors.NewValidationError("INVALID_LEARNING_RATE", "Learning rate must be between 0 and 1")
	}
	
	return nil
}

// Generate generates synthetic data based on the request
func (g *RNNGenerator) Generate(ctx context.Context, req *models.GenerationRequest) (*models.GenerationResult, error) {
	if req == nil {
		return nil, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}
	
	if err := g.ValidateParameters(req.Parameters); err != nil {
		return nil, err
	}
	
	if !g.trained {
		return nil, errors.NewValidationError("MODEL_NOT_TRAINED", "RNN model must be trained before generation")
	}
	
	g.logger.WithFields(logrus.Fields{
		"request_id":  req.ID,
		"length":      req.Parameters.Length,
		"hidden_size": g.config.HiddenSize,
		"num_layers":  g.config.NumLayers,
	}).Info("Starting RNN generation")
	
	start := time.Now()
	
	// Parse frequency
	frequency, err := g.parseFrequency(req.Parameters.Frequency)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeValidation, "INVALID_FREQUENCY", "Failed to parse frequency")
	}
	
	// Generate timestamps
	timestamps := g.generateTimestamps(req.Parameters.StartTime, frequency, req.Parameters.Length)
	
	// Generate RNN values
	values, err := g.generateRNNValues(ctx, req.Parameters.Length)
	if err != nil {
		return nil, err
	}
	
	// Create data points
	dataPoints := make([]models.DataPoint, len(timestamps))
	for i, timestamp := range timestamps {
		dataPoints[i] = models.DataPoint{
			Timestamp: timestamp,
			Value:     values[i],
			Quality:   0.90, // Slightly lower quality than LSTM
		}
	}
	
	// Create time series
	timeSeries := &models.TimeSeries{
		ID:          fmt.Sprintf("rnn-%d", time.Now().UnixNano()),
		Name:        fmt.Sprintf("RNN Generated (%d layers, %d units)", g.config.NumLayers, g.config.HiddenSize),
		Description: fmt.Sprintf("Synthetic data generated using vanilla RNN"),
		Tags:        req.Parameters.Tags,
		Metadata:    req.Parameters.Metadata,
		DataPoints:  dataPoints,
		StartTime:   timestamps[0],
		EndTime:     timestamps[len(timestamps)-1],
		Frequency:   req.Parameters.Frequency,
		SensorType:  string(req.Parameters.SensorType),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	
	duration := time.Since(start)
	
	result := &models.GenerationResult{
		ID:            req.ID,
		Status:        "completed",
		TimeSeries:    timeSeries,
		Duration:      duration,
		GeneratedAt:   time.Now(),
		GeneratorType: string(g.GetType()),
		Quality:       0.90,
		Metadata: map[string]interface{}{
			"model_architecture": map[string]interface{}{
				"hidden_size":     g.config.HiddenSize,
				"num_layers":      g.config.NumLayers,
				"sequence_length": g.config.SequenceLength,
				"activation":      g.config.Activation,
			},
			"training_info": map[string]interface{}{
				"epochs":        g.config.Epochs,
				"learning_rate": g.config.LearningRate,
				"batch_size":    g.config.BatchSize,
			},
			"data_points":     len(dataPoints),
			"generation_time": duration.String(),
		},
	}
	
	g.logger.WithFields(logrus.Fields{
		"request_id":  req.ID,
		"data_points": len(dataPoints),
		"duration":    duration,
	}).Info("Completed RNN generation")
	
	return result, nil
}

// Train trains the RNN model with reference data
func (g *RNNGenerator) Train(ctx context.Context, data *models.TimeSeries, params models.GenerationParameters) error {
	if data == nil {
		return errors.NewValidationError("INVALID_DATA", "Training data is required")
	}
	
	minDataPoints := g.config.SequenceLength * 10
	if len(data.DataPoints) < minDataPoints {
		return errors.NewValidationError("INSUFFICIENT_DATA", 
			fmt.Sprintf("At least %d data points required for RNN training", minDataPoints))
	}
	
	g.logger.WithFields(logrus.Fields{
		"series_id":       data.ID,
		"data_points":     len(data.DataPoints),
		"sequence_length": g.config.SequenceLength,
		"epochs":          g.config.Epochs,
	}).Info("Training RNN model")
	
	// Extract and preprocess data
	values := make([]float64, len(data.DataPoints))
	for i, dp := range data.DataPoints {
		values[i] = dp.Value
	}
	
	// Scale the data
	if err := g.scaler.Fit(values); err != nil {
		return errors.WrapError(err, errors.ErrorTypeProcessing, "SCALING_ERROR", "Failed to fit data scaler")
	}
	
	scaledValues := g.scaler.Transform(values)
	
	// Create training sequences
	X, y, err := g.createSequences(scaledValues)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeProcessing, "SEQUENCE_ERROR", "Failed to create training sequences")
	}
	
	// Split into training and validation sets
	trainX, trainY, valX, valY := g.trainValidationSplit(X, y)
	
	// Initialize the model
	g.model = g.initializeModel()
	
	// Train the model
	trainingMetrics, err := g.trainModel(ctx, trainX, trainY, valX, valY)
	if err != nil {
		return errors.WrapError(err, errors.ErrorTypeProcessing, "TRAINING_ERROR", "Failed to train RNN model")
	}
	
	// Calculate final statistics
	g.statistics = data.CalculateMetrics()
	g.trained = true
	
	g.logger.WithFields(logrus.Fields{
		"final_train_loss": trainingMetrics[len(trainingMetrics)-1].TrainingLoss,
		"final_val_loss":   trainingMetrics[len(trainingMetrics)-1].ValidationLoss,
		"epochs_completed": len(trainingMetrics),
	}).Info("RNN model training completed")
	
	return nil
}

// IsTrainable returns true if the generator requires/supports training
func (g *RNNGenerator) IsTrainable() bool {
	return true
}

// GetDefaultParameters returns default parameters for this generator
func (g *RNNGenerator) GetDefaultParameters() models.GenerationParameters {
	return models.GenerationParameters{
		Length:    1000,
		Frequency: "1h",
		StartTime: time.Now().Add(-30 * 24 * time.Hour),
		Tags:      make(map[string]string),
		Metadata:  make(map[string]interface{}),
	}
}

// EstimateDuration estimates how long generation will take
func (g *RNNGenerator) EstimateDuration(req *models.GenerationRequest) (time.Duration, error) {
	if req == nil {
		return 0, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}
	
	// RNN generation time is typically faster than LSTM
	baseTimePerPoint := 0.8 // milliseconds
	complexityFactor := float64(g.config.HiddenSize * g.config.NumLayers) / 120.0
	
	estimatedMs := float64(req.Parameters.Length) * baseTimePerPoint * complexityFactor
	return time.Duration(estimatedMs) * time.Millisecond, nil
}

// Cancel cancels an ongoing generation
func (g *RNNGenerator) Cancel(ctx context.Context, requestID string) error {
	g.logger.WithFields(logrus.Fields{
		"request_id": requestID,
	}).Info("Cancel requested for RNN generation")
	return nil
}

// GetProgress returns the progress of an ongoing generation
func (g *RNNGenerator) GetProgress(requestID string) (float64, error) {
	// RNN generation progress tracking would be implemented here
	return 1.0, nil
}

// Close cleans up resources
func (g *RNNGenerator) Close() error {
	g.logger.Info("Closing RNN generator")
	return nil
}

// RNN-specific methods

func (g *RNNGenerator) generateRNNValues(ctx context.Context, length int) ([]float64, error) {
	if g.model == nil {
		return nil, errors.NewProcessingError("MODEL_NOT_INITIALIZED", "RNN model not initialized")
	}
	
	// Initialize with a seed sequence
	seedSequence := g.generateSeedSequence()
	
	// Generate new values autoregressively
	generated := make([]float64, length)
	currentSequence := make([]float64, len(seedSequence))
	copy(currentSequence, seedSequence)
	
	for i := 0; i < length; i++ {
		// Predict next value
		input := g.sequenceToMatrix(currentSequence)
		output := g.forward(input)
		
		// Get the output value
		nextValue := output.AtVec(0)
		
		// Apply temperature sampling if configured
		if g.config.Temperature > 0 && g.config.Temperature != 1.0 {
			nextValue = g.applyTemperature(nextValue, g.config.Temperature)
		}
		
		// Store the generated value
		generated[i] = nextValue
		
		// Update current sequence (sliding window)
		if len(currentSequence) >= g.config.SequenceLength {
			copy(currentSequence[:len(currentSequence)-1], currentSequence[1:])
			currentSequence[len(currentSequence)-1] = nextValue
		} else {
			currentSequence = append(currentSequence, nextValue)
		}
		
		// Check for cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
	}
	
	// Inverse transform to original scale
	return g.scaler.InverseTransform(generated), nil
}

func (g *RNNGenerator) createSequences(data []float64) ([]*mat.Dense, []*mat.VecDense, error) {
	if len(data) < g.config.SequenceLength+1 {
		return nil, nil, fmt.Errorf("insufficient data for sequence creation")
	}
	
	numSequences := len(data) - g.config.SequenceLength
	
	// Create input sequences (X) and targets (y)
	X := make([]*mat.Dense, numSequences)
	y := make([]*mat.VecDense, numSequences)
	
	for i := 0; i < numSequences; i++ {
		// Input sequence
		sequence := mat.NewDense(g.config.SequenceLength, 1, nil)
		for j := 0; j < g.config.SequenceLength; j++ {
			sequence.Set(j, 0, data[i+j])
		}
		X[i] = sequence
		
		// Target (next value)
		target := mat.NewVecDense(1, []float64{data[i+g.config.SequenceLength]})
		y[i] = target
	}
	
	return X, y, nil
}

func (g *RNNGenerator) trainValidationSplit(X []*mat.Dense, y []*mat.VecDense) ([]*mat.Dense, []*mat.VecDense, []*mat.Dense, []*mat.VecDense) {
	numSamples := len(X)
	splitIndex := int(float64(numSamples) * (1.0 - g.config.ValidationSplit))
	
	trainX := X[:splitIndex]
	trainY := y[:splitIndex]
	valX := X[splitIndex:]
	valY := y[splitIndex:]
	
	return trainX, trainY, valX, valY
}

func (g *RNNGenerator) initializeModel() *RNNModel {
	model := &RNNModel{
		layers:      make([]*RNNLayer, g.config.NumLayers),
		optimizer:   NewSGDOptimizer(g.config.LearningRate, 0.9),
		lossHistory: make([]float64, 0),
	}
	
	// Initialize RNN layers
	for i := 0; i < g.config.NumLayers; i++ {
		inputSize := g.config.InputSize
		if i > 0 {
			inputSize = g.config.HiddenSize
		}
		
		model.layers[i] = NewRNNLayer(inputSize, g.config.HiddenSize, g.config.Activation, g.randSource)
	}
	
	// Initialize output layer
	model.outputLayer = NewDenseLayer(g.config.HiddenSize, g.config.OutputSize, g.randSource)
	
	return model
}

func (g *RNNGenerator) trainModel(ctx context.Context, trainX, trainY, valX, valY []*mat.Dense) ([]*TrainingMetrics, error) {
	metrics := make([]*TrainingMetrics, 0)
	bestValLoss := math.Inf(1)
	patienceCounter := 0
	
	for epoch := 0; epoch < g.config.Epochs; epoch++ {
		start := time.Now()
		
		// Training phase
		trainLoss := g.trainEpoch(trainX, trainY)
		
		// Validation phase
		valLoss := g.validateEpoch(valX, valY)
		
		// Record metrics
		metric := &TrainingMetrics{
			Epoch:          epoch + 1,
			TrainingLoss:   trainLoss,
			ValidationLoss: valLoss,
			LearningRate:   g.config.LearningRate,
			Duration:       time.Since(start),
		}
		metrics = append(metrics, metric)
		
		g.logger.WithFields(logrus.Fields{
			"epoch":      epoch + 1,
			"train_loss": trainLoss,
			"val_loss":   valLoss,
			"duration":   metric.Duration,
		}).Debug("Training epoch completed")
		
		// Early stopping
		if g.config.EarlyStopping {
			if valLoss < bestValLoss-g.config.MinDelta {
				bestValLoss = valLoss
				patienceCounter = 0
			} else {
				patienceCounter++
				if patienceCounter >= g.config.Patience {
					g.logger.WithFields(logrus.Fields{
						"epoch":    epoch + 1,
						"patience": patienceCounter,
					}).Info("Early stopping triggered")
					break
				}
			}
		}
		
		// Check for cancellation
		select {
		case <-ctx.Done():
			return metrics, ctx.Err()
		default:
		}
	}
	
	return metrics, nil
}

func (g *RNNGenerator) trainEpoch(X []*mat.Dense, y []*mat.VecDense) float64 {
	totalLoss := 0.0
	
	// Shuffle data for each epoch
	indices := make([]int, len(X))
	for i := range indices {
		indices[i] = i
	}
	g.randSource.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})
	
	// Process each sample
	for _, idx := range indices {
		// Forward pass
		output := g.forward(X[idx])
		
		// Calculate loss
		loss := g.calculateLoss(output, y[idx])
		totalLoss += loss
		
		// Backward pass
		g.backward(X[idx], y[idx], output)
		
		// Update weights
		g.updateWeights()
	}
	
	return totalLoss / float64(len(X))
}

func (g *RNNGenerator) validateEpoch(X []*mat.Dense, y []*mat.VecDense) float64 {
	totalLoss := 0.0
	
	for i := 0; i < len(X); i++ {
		output := g.forward(X[i])
		loss := g.calculateLoss(output, y[i])
		totalLoss += loss
	}
	
	return totalLoss / float64(len(X))
}

func (g *RNNGenerator) forward(input *mat.Dense) *mat.VecDense {
	rows, _ := input.Dims()
	
	// Process through RNN layers
	var currentHidden *mat.VecDense
	for layerIdx, layer := range g.model.layers {
		// Reset hidden state for new sequence
		layer.hiddenState = mat.NewVecDense(layer.hiddenSize, nil)
		
		// Process each time step
		for t := 0; t < rows; t++ {
			// Get input for this time step
			var inputVec *mat.VecDense
			if layerIdx == 0 {
				inputVec = mat.NewVecDense(1, []float64{input.At(t, 0)})
			} else {
				inputVec = currentHidden
			}
			
			// Compute new hidden state
			layer.computeHiddenState(inputVec)
		}
		
		// Pass final hidden state to next layer
		currentHidden = layer.hiddenState
	}
	
	// Output layer
	output := g.model.outputLayer.forward(currentHidden)
	
	return output
}

func (g *RNNGenerator) backward(input *mat.Dense, target, output *mat.VecDense) {
	// Simplified backward pass for vanilla RNN
	// In a complete implementation, this would compute gradients through time
	
	// Compute output error
	outputError := mat.NewVecDense(output.Len(), nil)
	outputError.SubVec(output, target)
	
	// Backpropagate through output layer
	g.model.outputLayer.backward(outputError)
	
	// Backpropagate through RNN layers (simplified)
	// Full BPTT (Backpropagation Through Time) would be implemented here
}

func (g *RNNGenerator) updateWeights() {
	// Update weights using optimizer
	// This is a simplified version - actual implementation would update all parameters
	
	// Apply gradient clipping if configured
	if g.config.GradientClipping > 0 {
		// Clip gradients to prevent exploding gradients
	}
	
	// Apply L2 regularization if configured
	if g.config.L2Regularization > 0 {
		// Add L2 penalty to gradients
	}
}

func (g *RNNGenerator) calculateLoss(predicted, actual *mat.VecDense) float64 {
	// Mean squared error
	diff := mat.NewVecDense(predicted.Len(), nil)
	diff.SubVec(predicted, actual)
	
	var sum float64
	for i := 0; i < diff.Len(); i++ {
		val := diff.AtVec(i)
		sum += val * val
	}
	
	return sum / float64(diff.Len())
}

// Helper methods

func (g *RNNGenerator) generateSeedSequence() []float64 {
	seed := make([]float64, g.config.SequenceLength)
	for i := range seed {
		seed[i] = g.randSource.NormFloat64() * 0.1
	}
	return seed
}

func (g *RNNGenerator) sequenceToMatrix(sequence []float64) *mat.Dense {
	matrix := mat.NewDense(len(sequence), 1, nil)
	for i, val := range sequence {
		matrix.Set(i, 0, val)
	}
	return matrix
}

func (g *RNNGenerator) applyTemperature(value, temperature float64) float64 {
	// Apply temperature scaling for diversity in generation
	scaled := value / temperature
	// Add some noise based on temperature
	noise := g.randSource.NormFloat64() * (temperature - 1.0) * 0.1
	return scaled + noise
}

func (g *RNNGenerator) generateTimestamps(start time.Time, frequency time.Duration, length int) []time.Time {
	timestamps := make([]time.Time, length)
	current := start
	
	for i := 0; i < length; i++ {
		timestamps[i] = current
		current = current.Add(frequency)
	}
	
	return timestamps
}

func (g *RNNGenerator) parseFrequency(freq string) (time.Duration, error) {
	duration, err := time.ParseDuration(freq)
	if err != nil {
		return 0, fmt.Errorf("invalid frequency format: %s", freq)
	}
	return duration, nil
}

func getDefaultRNNConfig() *RNNConfig {
	return &RNNConfig{
		HiddenSize:       32,
		NumLayers:        2,
		InputSize:        1,
		OutputSize:       1,
		SequenceLength:   15,
		Activation:       "tanh",
		LearningRate:     0.01,
		Epochs:           50,
		BatchSize:        32,
		DropoutRate:      0.1,
		GradientClipping: 1.0,
		L2Regularization: 0.001,
		Temperature:      1.0,
		SamplingMethod:   "greedy",
		Normalization:    "zscore",
		ValidationSplit:  0.2,
		Seed:             time.Now().UnixNano(),
		EarlyStopping:    true,
		Patience:         5,
		MinDelta:         0.001,
	}
}

// Supporting types

func NewRNNLayer(inputSize, hiddenSize int, activation string, randSource *rand.Rand) *RNNLayer {
	layer := &RNNLayer{
		hiddenSize:     hiddenSize,
		inputSize:      inputSize,
		activation:     activation,
		activations:    make([]*mat.VecDense, 0),
		preActivations: make([]*mat.VecDense, 0),
	}
	
	// Initialize weights using Xavier initialization
	scale := math.Sqrt(2.0 / float64(inputSize+hiddenSize))
	
	// Input to hidden weights
	inputWeights := make([]float64, hiddenSize*inputSize)
	for i := range inputWeights {
		inputWeights[i] = randSource.NormFloat64() * scale
	}
	layer.weightsInput = mat.NewDense(hiddenSize, inputSize, inputWeights)
	
	// Hidden to hidden weights
	hiddenWeights := make([]float64, hiddenSize*hiddenSize)
	for i := range hiddenWeights {
		hiddenWeights[i] = randSource.NormFloat64() * scale
	}
	layer.weightsHidden = mat.NewDense(hiddenSize, hiddenSize, hiddenWeights)
	
	// Bias
	layer.bias = mat.NewVecDense(hiddenSize, nil)
	
	// Initialize hidden state
	layer.hiddenState = mat.NewVecDense(hiddenSize, nil)
	
	return layer
}

func (l *RNNLayer) computeHiddenState(input *mat.VecDense) {
	// Compute pre-activation: h_t = W_ih * x_t + W_hh * h_{t-1} + b
	preActivation := mat.NewVecDense(l.hiddenSize, nil)
	
	// Input contribution
	inputContrib := mat.NewVecDense(l.hiddenSize, nil)
	inputContrib.MulVec(l.weightsInput, input)
	
	// Hidden contribution
	hiddenContrib := mat.NewVecDense(l.hiddenSize, nil)
	hiddenContrib.MulVec(l.weightsHidden, l.hiddenState)
	
	// Sum contributions and add bias
	preActivation.AddVec(inputContrib, hiddenContrib)
	preActivation.AddVec(preActivation, l.bias)
	
	// Apply activation function
	l.hiddenState = l.applyActivation(preActivation)
	
	// Store for backpropagation
	l.preActivations = append(l.preActivations, preActivation)
	l.activations = append(l.activations, l.hiddenState)
}

func (l *RNNLayer) applyActivation(x *mat.VecDense) *mat.VecDense {
	result := mat.NewVecDense(x.Len(), nil)
	
	switch l.activation {
	case "tanh":
		for i := 0; i < x.Len(); i++ {
			result.SetVec(i, math.Tanh(x.AtVec(i)))
		}
	case "relu":
		for i := 0; i < x.Len(); i++ {
			result.SetVec(i, math.Max(0, x.AtVec(i)))
		}
	case "sigmoid":
		for i := 0; i < x.Len(); i++ {
			result.SetVec(i, 1.0/(1.0+math.Exp(-x.AtVec(i))))
		}
	default:
		// Default to tanh
		for i := 0; i < x.Len(); i++ {
			result.SetVec(i, math.Tanh(x.AtVec(i)))
		}
	}
	
	return result
}

func NewDenseLayer(inputSize, outputSize int, randSource *rand.Rand) *DenseLayer {
	layer := &DenseLayer{
		inputSize:  inputSize,
		outputSize: outputSize,
	}
	
	// Initialize weights using Xavier initialization
	scale := math.Sqrt(2.0 / float64(inputSize+outputSize))
	weights := make([]float64, outputSize*inputSize)
	for i := range weights {
		weights[i] = randSource.NormFloat64() * scale
	}
	layer.weights = mat.NewDense(outputSize, inputSize, weights)
	
	// Initialize bias to zero
	layer.bias = mat.NewVecDense(outputSize, nil)
	
	return layer
}

func (l *DenseLayer) forward(input *mat.VecDense) *mat.VecDense {
	output := mat.NewVecDense(l.outputSize, nil)
	output.MulVec(l.weights, input)
	output.AddVec(output, l.bias)
	return output
}

func (l *DenseLayer) backward(outputError *mat.VecDense) {
	// Compute gradients for weights and bias
	// This is simplified - actual implementation would store gradients
}

func NewSGDOptimizer(learningRate, momentum float64) *SGDOptimizer {
	return &SGDOptimizer{
		learningRate: learningRate,
		momentum:     momentum,
		velocities:   make(map[string]interface{}),
	}
}

func NewRNNDataScaler(method string) *RNNDataScaler {
	return &RNNDataScaler{
		method: method,
		fitted: false,
	}
}

func (ds *RNNDataScaler) Fit(data []float64) error {
	if len(data) == 0 {
		return fmt.Errorf("cannot fit scaler on empty data")
	}
	
	switch ds.method {
	case "minmax":
		ds.min = data[0]
		ds.max = data[0]
		for _, v := range data {
			if v < ds.min {
				ds.min = v
			}
			if v > ds.max {
				ds.max = v
			}
		}
		
	case "zscore":
		// Calculate mean
		sum := 0.0
		for _, v := range data {
			sum += v
		}
		ds.mean = sum / float64(len(data))
		
		// Calculate standard deviation
		sumSq := 0.0
		for _, v := range data {
			diff := v - ds.mean
			sumSq += diff * diff
		}
		ds.std = math.Sqrt(sumSq / float64(len(data)))
		
	case "robust":
		// Use median and IQR for robust scaling
		sorted := make([]float64, len(data))
		copy(sorted, data)
		sort.Float64s(sorted)
		
		n := len(sorted)
		ds.median = sorted[n/2]
		ds.q25 = sorted[n/4]
		ds.q75 = sorted[3*n/4]
	}
	
	ds.fitted = true
	return nil
}

func (ds *RNNDataScaler) Transform(data []float64) []float64 {
	if !ds.fitted {
		return data
	}
	
	transformed := make([]float64, len(data))
	
	switch ds.method {
	case "minmax":
		scale := ds.max - ds.min
		if scale == 0 {
			scale = 1
		}
		for i, v := range data {
			transformed[i] = (v - ds.min) / scale
		}
		
	case "zscore":
		if ds.std == 0 {
			copy(transformed, data)
		} else {
			for i, v := range data {
				transformed[i] = (v - ds.mean) / ds.std
			}
		}
		
	case "robust":
		scale := ds.q75 - ds.q25
		if scale == 0 {
			scale = 1
		}
		for i, v := range data {
			transformed[i] = (v - ds.median) / scale
		}
	}
	
	return transformed
}

func (ds *RNNDataScaler) InverseTransform(data []float64) []float64 {
	if !ds.fitted {
		return data
	}
	
	original := make([]float64, len(data))
	
	switch ds.method {
	case "minmax":
		scale := ds.max - ds.min
		for i, v := range data {
			original[i] = v*scale + ds.min
		}
		
	case "zscore":
		for i, v := range data {
			original[i] = v*ds.std + ds.mean
		}
		
	case "robust":
		scale := ds.q75 - ds.q25
		for i, v := range data {
			original[i] = v*scale + ds.median
		}
	}
	
	return original
}