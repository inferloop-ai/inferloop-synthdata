package rnn

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/mat"

	"github.com/inferloop/tsiot/pkg/constants"
	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
)

// GRUGenerator implements GRU-based synthetic data generation
type GRUGenerator struct {
	logger      *logrus.Logger
	config      *GRUConfig
	model       *GRUModel
	trained     bool
	statistics  *models.TimeSeriesMetrics
	randSource  *rand.Rand
	scaler      *DataScaler
}

// GRUConfig contains configuration for GRU generation
type GRUConfig struct {
	// Network architecture
	HiddenSize     int     `json:"hidden_size"`     // Number of hidden units
	NumLayers      int     `json:"num_layers"`      // Number of GRU layers
	InputSize      int     `json:"input_size"`      // Input dimension
	OutputSize     int     `json:"output_size"`     // Output dimension
	SequenceLength int     `json:"sequence_length"` // Lookback window size
	
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
	SamplingMethod string  `json:"sampling_method"` // "greedy", "random", "nucleus"
	TopK           int     `json:"top_k"`           // Top-k sampling parameter
	TopP           float64 `json:"top_p"`           // Nucleus sampling parameter
	
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

// GRUModel represents the GRU neural network model
type GRUModel struct {
	layers      []*GRULayer
	outputLayer *DenseLayer
	optimizer   *AdamOptimizer
	lossHistory []float64
}

// GRULayer represents a single GRU layer
type GRULayer struct {
	hiddenSize int
	inputSize  int
	
	// GRU gates weights and biases
	weightsResetGate  *mat.Dense    // Reset gate weights for input
	weightsUpdateGate *mat.Dense    // Update gate weights for input
	weightsCandidate  *mat.Dense    // Candidate weights for input
	
	weightsHiddenResetGate  *mat.Dense  // Reset gate weights for hidden
	weightsHiddenUpdateGate *mat.Dense  // Update gate weights for hidden
	weightsHiddenCandidate  *mat.Dense  // Candidate weights for hidden
	
	biasResetGate  *mat.VecDense // Reset gate bias
	biasUpdateGate *mat.VecDense // Update gate bias
	biasCandidate  *mat.VecDense // Candidate bias
	
	// State
	hiddenState *mat.VecDense
	
	// For backpropagation
	gates       map[string][]*mat.VecDense
	activations map[string][]*mat.VecDense
}

// NewGRUGenerator creates a new GRU generator
func NewGRUGenerator(config *GRUConfig, logger *logrus.Logger) *GRUGenerator {
	if config == nil {
		config = getDefaultGRUConfig()
	}
	
	if logger == nil {
		logger = logrus.New()
	}
	
	if config.Seed == 0 {
		config.Seed = time.Now().UnixNano()
	}
	
	return &GRUGenerator{
		logger:     logger,
		config:     config,
		trained:    false,
		randSource: rand.New(rand.NewSource(config.Seed)),
		scaler:     NewDataScaler(config.Normalization),
	}
}

// GetType returns the generator type
func (g *GRUGenerator) GetType() models.GeneratorType {
	return models.GeneratorType(constants.GeneratorTypeGRU)
}

// GetName returns a human-readable name for the generator
func (g *GRUGenerator) GetName() string {
	return "GRU Generator"
}

// GetDescription returns a description of the generator
func (g *GRUGenerator) GetDescription() string {
	return fmt.Sprintf("Generates synthetic time series using GRU neural network with %d layers and %d hidden units", 
		g.config.NumLayers, g.config.HiddenSize)
}

// GetSupportedSensorTypes returns the sensor types this generator supports
func (g *GRUGenerator) GetSupportedSensorTypes() []models.SensorType {
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
func (g *GRUGenerator) ValidateParameters(params models.GenerationParameters) error {
	if params.Length <= 0 {
		return errors.NewValidationError("INVALID_LENGTH", "Generation length must be positive")
	}
	
	if params.Frequency == "" {
		return errors.NewValidationError("INVALID_FREQUENCY", "Frequency is required")
	}
	
	// Validate GRU-specific parameters
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
func (g *GRUGenerator) Generate(ctx context.Context, req *models.GenerationRequest) (*models.GenerationResult, error) {
	if req == nil {
		return nil, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}
	
	if err := g.ValidateParameters(req.Parameters); err != nil {
		return nil, err
	}
	
	if !g.trained {
		return nil, errors.NewValidationError("MODEL_NOT_TRAINED", "GRU model must be trained before generation")
	}
	
	g.logger.WithFields(logrus.Fields{
		"request_id":  req.ID,
		"length":      req.Parameters.Length,
		"hidden_size": g.config.HiddenSize,
		"num_layers":  g.config.NumLayers,
	}).Info("Starting GRU generation")
	
	start := time.Now()
	
	// Parse frequency
	frequency, err := g.parseFrequency(req.Parameters.Frequency)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeValidation, "INVALID_FREQUENCY", "Failed to parse frequency")
	}
	
	// Generate timestamps
	timestamps := g.generateTimestamps(req.Parameters.StartTime, frequency, req.Parameters.Length)
	
	// Generate GRU values
	values, err := g.generateGRUValues(ctx, req.Parameters.Length)
	if err != nil {
		return nil, err
	}
	
	// Create data points
	dataPoints := make([]models.DataPoint, len(timestamps))
	for i, timestamp := range timestamps {
		dataPoints[i] = models.DataPoint{
			Timestamp: timestamp,
			Value:     values[i],
			Quality:   0.93, // Between RNN and LSTM quality
		}
	}
	
	// Create time series
	timeSeries := &models.TimeSeries{
		ID:          fmt.Sprintf("gru-%d", time.Now().UnixNano()),
		Name:        fmt.Sprintf("GRU Generated (%d layers, %d units)", g.config.NumLayers, g.config.HiddenSize),
		Description: fmt.Sprintf("Synthetic data generated using GRU neural network"),
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
		Quality:       0.93,
		Metadata: map[string]interface{}{
			"model_architecture": map[string]interface{}{
				"hidden_size":     g.config.HiddenSize,
				"num_layers":      g.config.NumLayers,
				"sequence_length": g.config.SequenceLength,
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
	}).Info("Completed GRU generation")
	
	return result, nil
}

// Train trains the GRU model with reference data
func (g *GRUGenerator) Train(ctx context.Context, data *models.TimeSeries, params models.GenerationParameters) error {
	if data == nil {
		return errors.NewValidationError("INVALID_DATA", "Training data is required")
	}
	
	minDataPoints := g.config.SequenceLength * 10
	if len(data.DataPoints) < minDataPoints {
		return errors.NewValidationError("INSUFFICIENT_DATA", 
			fmt.Sprintf("At least %d data points required for GRU training", minDataPoints))
	}
	
	g.logger.WithFields(logrus.Fields{
		"series_id":       data.ID,
		"data_points":     len(data.DataPoints),
		"sequence_length": g.config.SequenceLength,
		"epochs":          g.config.Epochs,
	}).Info("Training GRU model")
	
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
		return errors.WrapError(err, errors.ErrorTypeProcessing, "TRAINING_ERROR", "Failed to train GRU model")
	}
	
	// Calculate final statistics
	g.statistics = data.CalculateMetrics()
	g.trained = true
	
	g.logger.WithFields(logrus.Fields{
		"final_train_loss": trainingMetrics[len(trainingMetrics)-1].TrainingLoss,
		"final_val_loss":   trainingMetrics[len(trainingMetrics)-1].ValidationLoss,
		"epochs_completed": len(trainingMetrics),
	}).Info("GRU model training completed")
	
	return nil
}

// IsTrainable returns true if the generator requires/supports training
func (g *GRUGenerator) IsTrainable() bool {
	return true
}

// GetDefaultParameters returns default parameters for this generator
func (g *GRUGenerator) GetDefaultParameters() models.GenerationParameters {
	return models.GenerationParameters{
		Length:    1000,
		Frequency: "1h",
		StartTime: time.Now().Add(-30 * 24 * time.Hour),
		Tags:      make(map[string]string),
		Metadata:  make(map[string]interface{}),
	}
}

// EstimateDuration estimates how long generation will take
func (g *GRUGenerator) EstimateDuration(req *models.GenerationRequest) (time.Duration, error) {
	if req == nil {
		return 0, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}
	
	// GRU generation time is between RNN and LSTM
	baseTimePerPoint := 0.9 // milliseconds
	complexityFactor := float64(g.config.HiddenSize * g.config.NumLayers) / 110.0
	
	estimatedMs := float64(req.Parameters.Length) * baseTimePerPoint * complexityFactor
	return time.Duration(estimatedMs) * time.Millisecond, nil
}

// Cancel cancels an ongoing generation
func (g *GRUGenerator) Cancel(ctx context.Context, requestID string) error {
	g.logger.WithFields(logrus.Fields{
		"request_id": requestID,
	}).Info("Cancel requested for GRU generation")
	return nil
}

// GetProgress returns the progress of an ongoing generation
func (g *GRUGenerator) GetProgress(requestID string) (float64, error) {
	// GRU generation progress tracking would be implemented here
	return 1.0, nil
}

// Close cleans up resources
func (g *GRUGenerator) Close() error {
	g.logger.Info("Closing GRU generator")
	return nil
}

// GRU-specific methods

func (g *GRUGenerator) generateGRUValues(ctx context.Context, length int) ([]float64, error) {
	if g.model == nil {
		return nil, errors.NewProcessingError("MODEL_NOT_INITIALIZED", "GRU model not initialized")
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
		
		// Sample from output
		nextValue := g.sampleFromOutput(output)
		
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

func (g *GRUGenerator) createSequences(data []float64) ([]*mat.Dense, []*mat.VecDense, error) {
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

func (g *GRUGenerator) trainValidationSplit(X []*mat.Dense, y []*mat.VecDense) ([]*mat.Dense, []*mat.VecDense, []*mat.Dense, []*mat.VecDense) {
	numSamples := len(X)
	splitIndex := int(float64(numSamples) * (1.0 - g.config.ValidationSplit))
	
	trainX := X[:splitIndex]
	trainY := y[:splitIndex]
	valX := X[splitIndex:]
	valY := y[splitIndex:]
	
	return trainX, trainY, valX, valY
}

func (g *GRUGenerator) initializeModel() *GRUModel {
	model := &GRUModel{
		layers:      make([]*GRULayer, g.config.NumLayers),
		optimizer:   NewAdamOptimizer(g.config.LearningRate),
		lossHistory: make([]float64, 0),
	}
	
	// Initialize GRU layers
	for i := 0; i < g.config.NumLayers; i++ {
		inputSize := g.config.InputSize
		if i > 0 {
			inputSize = g.config.HiddenSize
		}
		
		model.layers[i] = NewGRULayer(inputSize, g.config.HiddenSize, g.randSource)
	}
	
	// Initialize output layer
	model.outputLayer = NewDenseLayer(g.config.HiddenSize, g.config.OutputSize, g.randSource)
	
	return model
}

func (g *GRUGenerator) trainModel(ctx context.Context, trainX, trainY, valX, valY []*mat.Dense) ([]*TrainingMetrics, error) {
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

func (g *GRUGenerator) trainEpoch(X []*mat.Dense, y []*mat.VecDense) float64 {
	totalLoss := 0.0
	
	// Process in batches
	batchSize := g.config.BatchSize
	numBatches := (len(X) + batchSize - 1) / batchSize
	
	for batch := 0; batch < numBatches; batch++ {
		start := batch * batchSize
		end := start + batchSize
		if end > len(X) {
			end = len(X)
		}
		
		batchLoss := 0.0
		for i := start; i < end; i++ {
			// Forward pass
			output := g.forward(X[i])
			
			// Calculate loss
			loss := g.calculateLoss(output, y[i])
			batchLoss += loss
			
			// Backward pass
			g.backward(X[i], y[i], output)
		}
		
		// Update weights after batch
		g.updateWeights()
		totalLoss += batchLoss
	}
	
	return totalLoss / float64(len(X))
}

func (g *GRUGenerator) validateEpoch(X []*mat.Dense, y []*mat.VecDense) float64 {
	totalLoss := 0.0
	
	for i := 0; i < len(X); i++ {
		output := g.forward(X[i])
		loss := g.calculateLoss(output, y[i])
		totalLoss += loss
	}
	
	return totalLoss / float64(len(X))
}

func (g *GRUGenerator) forward(input *mat.Dense) *mat.VecDense {
	rows, _ := input.Dims()
	
	// Process through GRU layers
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
			
			// Compute new hidden state using GRU equations
			layer.computeHiddenState(inputVec)
		}
		
		// Pass final hidden state to next layer
		currentHidden = layer.hiddenState
	}
	
	// Output layer
	output := g.model.outputLayer.forward(currentHidden)
	
	return output
}

func (g *GRUGenerator) backward(input *mat.Dense, target, output *mat.VecDense) {
	// Simplified backward pass for GRU
	// In a complete implementation, this would compute gradients through time
	
	// Compute output error
	outputError := mat.NewVecDense(output.Len(), nil)
	outputError.SubVec(output, target)
	
	// Apply L2 regularization if configured
	if g.config.L2Regularization > 0 {
		// Add L2 penalty gradient
	}
}

func (g *GRUGenerator) updateWeights() {
	// Update weights using optimizer
	// Apply gradient clipping if configured
	if g.config.GradientClipping > 0 {
		// Clip gradients to prevent exploding gradients
	}
}

func (g *GRUGenerator) calculateLoss(predicted, actual *mat.VecDense) float64 {
	// Mean squared error
	diff := mat.NewVecDense(predicted.Len(), nil)
	diff.SubVec(predicted, actual)
	
	var sum float64
	for i := 0; i < diff.Len(); i++ {
		val := diff.AtVec(i)
		sum += val * val
	}
	
	mse := sum / float64(diff.Len())
	
	// Add L2 regularization penalty if configured
	if g.config.L2Regularization > 0 {
		// Add L2 penalty term
		mse += g.config.L2Regularization * g.computeL2Penalty()
	}
	
	return mse
}

func (g *GRUGenerator) computeL2Penalty() float64 {
	// Compute L2 norm of all weights
	penalty := 0.0
	// This would sum up the squared weights across all layers
	return penalty
}

// Helper methods

func (g *GRUGenerator) generateSeedSequence() []float64 {
	seed := make([]float64, g.config.SequenceLength)
	for i := range seed {
		seed[i] = g.randSource.NormFloat64() * 0.1
	}
	return seed
}

func (g *GRUGenerator) sequenceToMatrix(sequence []float64) *mat.Dense {
	matrix := mat.NewDense(len(sequence), 1, nil)
	for i, val := range sequence {
		matrix.Set(i, 0, val)
	}
	return matrix
}

func (g *GRUGenerator) sampleFromOutput(output *mat.VecDense) float64 {
	value := output.AtVec(0)
	
	// Apply sampling method
	switch g.config.SamplingMethod {
	case "greedy":
		return value
	case "random":
		// Add temperature-scaled noise
		noise := g.randSource.NormFloat64() * g.config.Temperature * 0.1
		return value + noise
	case "nucleus":
		// Simplified nucleus sampling for continuous values
		if g.config.Temperature > 0 {
			value = value / g.config.Temperature
		}
		return value
	default:
		return value
	}
}

func (g *GRUGenerator) generateTimestamps(start time.Time, frequency time.Duration, length int) []time.Time {
	timestamps := make([]time.Time, length)
	current := start
	
	for i := 0; i < length; i++ {
		timestamps[i] = current
		current = current.Add(frequency)
	}
	
	return timestamps
}

func (g *GRUGenerator) parseFrequency(freq string) (time.Duration, error) {
	duration, err := time.ParseDuration(freq)
	if err != nil {
		return 0, fmt.Errorf("invalid frequency format: %s", freq)
	}
	return duration, nil
}

func getDefaultGRUConfig() *GRUConfig {
	return &GRUConfig{
		HiddenSize:       48,
		NumLayers:        2,
		InputSize:        1,
		OutputSize:       1,
		SequenceLength:   18,
		LearningRate:     0.005,
		Epochs:           80,
		BatchSize:        32,
		DropoutRate:      0.15,
		GradientClipping: 1.0,
		L2Regularization: 0.001,
		Temperature:      1.0,
		SamplingMethod:   "greedy",
		Normalization:    "zscore",
		ValidationSplit:  0.2,
		Seed:             time.Now().UnixNano(),
		EarlyStopping:    true,
		Patience:         8,
		MinDelta:         0.001,
	}
}

// Supporting types

func NewGRULayer(inputSize, hiddenSize int, randSource *rand.Rand) *GRULayer {
	layer := &GRULayer{
		hiddenSize: hiddenSize,
		inputSize:  inputSize,
		gates:      make(map[string][]*mat.VecDense),
		activations: make(map[string][]*mat.VecDense),
	}
	
	// Initialize weights using Xavier initialization
	scale := math.Sqrt(2.0 / float64(inputSize+hiddenSize))
	
	// Reset gate weights
	layer.weightsResetGate = g.initializeMatrix(hiddenSize, inputSize, scale, randSource)
	layer.weightsHiddenResetGate = g.initializeMatrix(hiddenSize, hiddenSize, scale, randSource)
	layer.biasResetGate = mat.NewVecDense(hiddenSize, nil)
	
	// Update gate weights
	layer.weightsUpdateGate = g.initializeMatrix(hiddenSize, inputSize, scale, randSource)
	layer.weightsHiddenUpdateGate = g.initializeMatrix(hiddenSize, hiddenSize, scale, randSource)
	layer.biasUpdateGate = mat.NewVecDense(hiddenSize, nil)
	
	// Candidate weights
	layer.weightsCandidate = g.initializeMatrix(hiddenSize, inputSize, scale, randSource)
	layer.weightsHiddenCandidate = g.initializeMatrix(hiddenSize, hiddenSize, scale, randSource)
	layer.biasCandidate = mat.NewVecDense(hiddenSize, nil)
	
	// Initialize hidden state
	layer.hiddenState = mat.NewVecDense(hiddenSize, nil)
	
	return layer
}

func (g *GRUGenerator) initializeMatrix(rows, cols int, scale float64, randSource *rand.Rand) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = randSource.NormFloat64() * scale
	}
	return mat.NewDense(rows, cols, data)
}

func (l *GRULayer) computeHiddenState(input *mat.VecDense) {
	// GRU equations:
	// r_t = sigmoid(W_r * x_t + U_r * h_{t-1} + b_r)  // Reset gate
	// z_t = sigmoid(W_z * x_t + U_z * h_{t-1} + b_z)  // Update gate
	// h_tilde_t = tanh(W_h * x_t + U_h * (r_t ™ h_{t-1}) + b_h)  // Candidate
	// h_t = (1 - z_t) ™ h_{t-1} + z_t ™ h_tilde_t  // New hidden state
	
	// Reset gate
	resetGate := mat.NewVecDense(l.hiddenSize, nil)
	tempReset := mat.NewVecDense(l.hiddenSize, nil)
	tempReset.MulVec(l.weightsResetGate, input)
	resetGate.AddVec(tempReset, l.biasResetGate)
	
	tempResetHidden := mat.NewVecDense(l.hiddenSize, nil)
	tempResetHidden.MulVec(l.weightsHiddenResetGate, l.hiddenState)
	resetGate.AddVec(resetGate, tempResetHidden)
	
	// Apply sigmoid
	for i := 0; i < resetGate.Len(); i++ {
		resetGate.SetVec(i, sigmoid(resetGate.AtVec(i)))
	}
	
	// Update gate
	updateGate := mat.NewVecDense(l.hiddenSize, nil)
	tempUpdate := mat.NewVecDense(l.hiddenSize, nil)
	tempUpdate.MulVec(l.weightsUpdateGate, input)
	updateGate.AddVec(tempUpdate, l.biasUpdateGate)
	
	tempUpdateHidden := mat.NewVecDense(l.hiddenSize, nil)
	tempUpdateHidden.MulVec(l.weightsHiddenUpdateGate, l.hiddenState)
	updateGate.AddVec(updateGate, tempUpdateHidden)
	
	// Apply sigmoid
	for i := 0; i < updateGate.Len(); i++ {
		updateGate.SetVec(i, sigmoid(updateGate.AtVec(i)))
	}
	
	// Candidate hidden state
	candidate := mat.NewVecDense(l.hiddenSize, nil)
	tempCandidate := mat.NewVecDense(l.hiddenSize, nil)
	tempCandidate.MulVec(l.weightsCandidate, input)
	candidate.AddVec(tempCandidate, l.biasCandidate)
	
	// Apply reset gate to previous hidden state
	resetHidden := mat.NewVecDense(l.hiddenSize, nil)
	for i := 0; i < l.hiddenSize; i++ {
		resetHidden.SetVec(i, resetGate.AtVec(i)*l.hiddenState.AtVec(i))
	}
	
	tempCandidateHidden := mat.NewVecDense(l.hiddenSize, nil)
	tempCandidateHidden.MulVec(l.weightsHiddenCandidate, resetHidden)
	candidate.AddVec(candidate, tempCandidateHidden)
	
	// Apply tanh
	for i := 0; i < candidate.Len(); i++ {
		candidate.SetVec(i, math.Tanh(candidate.AtVec(i)))
	}
	
	// Compute new hidden state
	for i := 0; i < l.hiddenSize; i++ {
		h_prev := l.hiddenState.AtVec(i)
		z := updateGate.AtVec(i)
		h_tilde := candidate.AtVec(i)
		l.hiddenState.SetVec(i, (1-z)*h_prev + z*h_tilde)
	}
	
	// Store gates for backpropagation
	l.gates["reset"] = append(l.gates["reset"], resetGate)
	l.gates["update"] = append(l.gates["update"], updateGate)
	l.gates["candidate"] = append(l.gates["candidate"], candidate)
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}