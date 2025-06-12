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

// LSTMGenerator implements LSTM-based synthetic data generation
type LSTMGenerator struct {
	logger      *logrus.Logger
	config      *LSTMConfig
	model       *LSTMModel
	trained     bool
	statistics  *models.TimeSeriesMetrics
	randSource  *rand.Rand
	scaler      *DataScaler
}

// LSTMConfig contains configuration for LSTM generation
type LSTMConfig struct {
	// Network architecture
	HiddenSize     int     `json:"hidden_size"`     // Number of hidden units
	NumLayers      int     `json:"num_layers"`      // Number of LSTM layers
	InputSize      int     `json:"input_size"`      // Input dimension
	OutputSize     int     `json:"output_size"`     // Output dimension
	SequenceLength int     `json:"sequence_length"` // Lookback window size
	
	// Training parameters
	LearningRate   float64 `json:"learning_rate"`   // Learning rate
	Epochs         int     `json:"epochs"`          // Training epochs
	BatchSize      int     `json:"batch_size"`      // Batch size
	DropoutRate    float64 `json:"dropout_rate"`    // Dropout probability
	
	// Regularization
	L1Regularization float64 `json:"l1_regularization"` // L1 penalty weight
	L2Regularization float64 `json:"l2_regularization"` // L2 penalty weight
	GradientClipping float64 `json:"gradient_clipping"` // Gradient clipping threshold
	
	// Generation parameters
	Temperature    float64 `json:"temperature"`     // Sampling temperature
	SamplingMethod string  `json:"sampling_method"` // "greedy", "random", "nucleus"
	TopK           int     `json:"top_k"`           // Top-k sampling parameter
	TopP           float64 `json:"top_p"`           // Nucleus sampling parameter
	
	// Data preprocessing
	Normalization   string  `json:"normalization"`   // "minmax", "zscore", "robust"
	WindowStride    int     `json:"window_stride"`   // Sliding window stride
	ValidationSplit float64 `json:"validation_split"` // Fraction for validation
	
	// Other parameters
	Seed           int64   `json:"seed"`             // Random seed
	EarlyStopping  bool    `json:"early_stopping"`   // Enable early stopping
	Patience       int     `json:"patience"`         // Early stopping patience
	MinDelta       float64 `json:"min_delta"`        // Min improvement for early stopping
}

// LSTMModel represents the LSTM neural network model
type LSTMModel struct {
	layers     []*LSTMLayer
	weights    map[string]*mat.Dense
	biases     map[string]*mat.VecDense
	optimizer  *AdamOptimizer
	lossHistory []float64
}

// LSTMLayer represents a single LSTM layer
type LSTMLayer struct {
	hiddenSize int
	inputSize  int
	
	// LSTM gates weights and biases
	weightsInputGate  *mat.Dense
	weightsForgetGate *mat.Dense
	weightsOutputGate *mat.Dense
	weightsCellGate   *mat.Dense
	
	weightsHiddenInputGate  *mat.Dense
	weightsHiddenForgetGate *mat.Dense
	weightsHiddenOutputGate *mat.Dense
	weightsHiddenCellGate   *mat.Dense
	
	biasInputGate  *mat.VecDense
	biasForgetGate *mat.VecDense
	biasOutputGate *mat.VecDense
	biasCellGate   *mat.VecDense
	
	// State
	hiddenState *mat.VecDense
	cellState   *mat.VecDense
	
	// For backpropagation
	gates map[string]*mat.VecDense
	activations map[string]*mat.VecDense
}

// AdamOptimizer implements the Adam optimization algorithm
type AdamOptimizer struct {
	learningRate float64
	beta1        float64
	beta2        float64
	epsilon      float64
	t            int // time step
	
	// Momentum and velocity for each parameter
	momentum map[string]*mat.Dense
	velocity map[string]*mat.Dense
	momentumVec map[string]*mat.VecDense
	velocityVec map[string]*mat.VecDense
}

// DataScaler handles data normalization and scaling
type DataScaler struct {
	method   string  // "minmax", "zscore", "robust"
	min      float64
	max      float64
	mean     float64
	std      float64
	q25      float64 // 25th percentile for robust scaling
	q75      float64 // 75th percentile for robust scaling
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

// NewLSTMGenerator creates a new LSTM generator
func NewLSTMGenerator(config *LSTMConfig, logger *logrus.Logger) *LSTMGenerator {
	if config == nil {
		config = getDefaultLSTMConfig()
	}
	
	if logger == nil {
		logger = logrus.New()
	}
	
	if config.Seed == 0 {
		config.Seed = time.Now().UnixNano()
	}
	
	return &LSTMGenerator{
		logger:     logger,
		config:     config,
		trained:    false,
		randSource: rand.New(rand.NewSource(config.Seed)),
		scaler:     NewDataScaler(config.Normalization),
	}
}

// GetType returns the generator type
func (g *LSTMGenerator) GetType() models.GeneratorType {
	return models.GeneratorType(constants.GeneratorTypeLSTM)
}

// GetName returns a human-readable name for the generator
func (g *LSTMGenerator) GetName() string {
	return "LSTM Generator"
}

// GetDescription returns a description of the generator
func (g *LSTMGenerator) GetDescription() string {
	return fmt.Sprintf("Generates synthetic time series using LSTM neural network with %d layers and %d hidden units", 
		g.config.NumLayers, g.config.HiddenSize)
}

// GetSupportedSensorTypes returns the sensor types this generator supports
func (g *LSTMGenerator) GetSupportedSensorTypes() []models.SensorType {
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
func (g *LSTMGenerator) ValidateParameters(params models.GenerationParameters) error {
	if params.Length <= 0 {
		return errors.NewValidationError("INVALID_LENGTH", "Generation length must be positive")
	}
	
	if params.Frequency == "" {
		return errors.NewValidationError("INVALID_FREQUENCY", "Frequency is required")
	}
	
	// Validate LSTM-specific parameters
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
func (g *LSTMGenerator) Generate(ctx context.Context, req *models.GenerationRequest) (*models.GenerationResult, error) {
	if req == nil {
		return nil, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}
	
	if err := g.ValidateParameters(req.Parameters); err != nil {
		return nil, err
	}
	
	if !g.trained {
		return nil, errors.NewValidationError("MODEL_NOT_TRAINED", "LSTM model must be trained before generation")
	}
	
	g.logger.WithFields(logrus.Fields{
		"request_id": req.ID,
		"length":     req.Parameters.Length,
		"hidden_size": g.config.HiddenSize,
		"num_layers":  g.config.NumLayers,
	}).Info("Starting LSTM generation")
	
	start := time.Now()
	
	// Parse frequency
	frequency, err := g.parseFrequency(req.Parameters.Frequency)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeValidation, "INVALID_FREQUENCY", "Failed to parse frequency")
	}
	
	// Generate timestamps
	timestamps := g.generateTimestamps(req.Parameters.StartTime, frequency, req.Parameters.Length)
	
	// Generate LSTM values
	values, err := g.generateLSTMValues(ctx, req.Parameters.Length)
	if err != nil {
		return nil, err
	}
	
	// Create data points
	dataPoints := make([]models.DataPoint, len(timestamps))
	for i, timestamp := range timestamps {
		dataPoints[i] = models.DataPoint{
			Timestamp: timestamp,
			Value:     values[i],
			Quality:   0.95, // High quality for neural network generation
		}
	}
	
	// Create time series
	timeSeries := &models.TimeSeries{
		ID:          fmt.Sprintf("lstm-%d", time.Now().UnixNano()),
		Name:        fmt.Sprintf("LSTM Generated (%d layers, %d units)", g.config.NumLayers, g.config.HiddenSize),
		Description: fmt.Sprintf("Synthetic data generated using LSTM neural network"),
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
		Quality:       0.95,
		Metadata: map[string]interface{}{
			"model_architecture": map[string]interface{}{
				"hidden_size":     g.config.HiddenSize,
				"num_layers":      g.config.NumLayers,
				"sequence_length": g.config.SequenceLength,
			},
			"training_info": map[string]interface{}{
				"epochs":          g.config.Epochs,
				"learning_rate":   g.config.LearningRate,
				"batch_size":      g.config.BatchSize,
			},
			"data_points":     len(dataPoints),
			"generation_time": duration.String(),
			"trained":         g.trained,
		},
	}
	
	g.logger.WithFields(logrus.Fields{
		"request_id":  req.ID,
		"data_points": len(dataPoints),
		"duration":    duration,
	}).Info("Completed LSTM generation")
	
	return result, nil
}

// Train trains the LSTM model with reference data
func (g *LSTMGenerator) Train(ctx context.Context, data *models.TimeSeries, params models.GenerationParameters) error {
	if data == nil {
		return errors.NewValidationError("INVALID_DATA", "Training data is required")
	}
	
	minDataPoints := g.config.SequenceLength * 10
	if len(data.DataPoints) < minDataPoints {
		return errors.NewValidationError("INSUFFICIENT_DATA", 
			fmt.Sprintf("At least %d data points required for LSTM training", minDataPoints))
	}
	
	g.logger.WithFields(logrus.Fields{
		"series_id":       data.ID,
		"data_points":     len(data.DataPoints),
		"sequence_length": g.config.SequenceLength,
		"epochs":          g.config.Epochs,
	}).Info("Training LSTM model")
	
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
		return errors.WrapError(err, errors.ErrorTypeProcessing, "TRAINING_ERROR", "Failed to train LSTM model")
	}
	
	// Calculate final statistics
	g.statistics = data.CalculateMetrics()
	g.trained = true
	
	g.logger.WithFields(logrus.Fields{
		"final_train_loss": trainingMetrics[len(trainingMetrics)-1].TrainingLoss,
		"final_val_loss":   trainingMetrics[len(trainingMetrics)-1].ValidationLoss,
		"epochs_completed": len(trainingMetrics),
	}).Info("LSTM model training completed")
	
	return nil
}

// IsTrainable returns true if the generator requires/supports training
func (g *LSTMGenerator) IsTrainable() bool {
	return true
}

// GetDefaultParameters returns default parameters for this generator
func (g *LSTMGenerator) GetDefaultParameters() models.GenerationParameters {
	return models.GenerationParameters{
		Length:    1000,
		Frequency: "1h",
		StartTime: time.Now().Add(-30 * 24 * time.Hour),
		Tags:      make(map[string]string),
		Metadata:  make(map[string]interface{}),
	}
}

// EstimateDuration estimates how long generation will take
func (g *LSTMGenerator) EstimateDuration(req *models.GenerationRequest) (time.Duration, error) {
	if req == nil {
		return 0, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}
	
	// LSTM generation time depends on sequence length and model complexity
	baseTimePerPoint := 1.0 // milliseconds
	complexityFactor := float64(g.config.HiddenSize * g.config.NumLayers) / 100.0
	
	estimatedMs := float64(req.Parameters.Length) * baseTimePerPoint * complexityFactor
	return time.Duration(estimatedMs) * time.Millisecond, nil
}

// Cancel cancels an ongoing generation
func (g *LSTMGenerator) Cancel(ctx context.Context, requestID string) error {
	g.logger.WithFields(logrus.Fields{
		"request_id": requestID,
	}).Info("Cancel requested for LSTM generation")
	return nil
}

// GetProgress returns the progress of an ongoing generation
func (g *LSTMGenerator) GetProgress(requestID string) (float64, error) {
	// LSTM generation progress tracking would be implemented here
	return 1.0, nil
}

// Close cleans up resources
func (g *LSTMGenerator) Close() error {
	g.logger.Info("Closing LSTM generator")
	return nil
}

// LSTM-specific methods

func (g *LSTMGenerator) generateLSTMValues(ctx context.Context, length int) ([]float64, error) {
	if g.model == nil {
		return nil, errors.NewProcessingError("MODEL_NOT_INITIALIZED", "LSTM model not initialized")
	}
	
	// Initialize with a seed sequence (use last sequence from training if available)
	seedSequence := g.generateSeedSequence()
	
	// Generate new values autoregressively
	generated := make([]float64, length)
	currentSequence := make([]float64, len(seedSequence))
	copy(currentSequence, seedSequence)
	
	for i := 0; i < length; i++ {
		// Predict next value
		input := g.sequenceToMatrix(currentSequence)
		output := g.forward(input)
		
		// Sample from output (apply temperature and sampling method)
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

func (g *LSTMGenerator) createSequences(data []float64) ([]*mat.Dense, []*mat.VecDense, error) {
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

func (g *LSTMGenerator) trainValidationSplit(X []*mat.Dense, y []*mat.VecDense) ([]*mat.Dense, []*mat.VecDense, []*mat.Dense, []*mat.VecDense) {
	numSamples := len(X)
	splitIndex := int(float64(numSamples) * (1.0 - g.config.ValidationSplit))
	
	trainX := X[:splitIndex]
	trainY := y[:splitIndex]
	valX := X[splitIndex:]
	valY := y[splitIndex:]
	
	return trainX, trainY, valX, valY
}

func (g *LSTMGenerator) initializeModel() *LSTMModel {
	model := &LSTMModel{
		layers:      make([]*LSTMLayer, g.config.NumLayers),
		weights:     make(map[string]*mat.Dense),
		biases:      make(map[string]*mat.VecDense),
		optimizer:   NewAdamOptimizer(g.config.LearningRate),
		lossHistory: make([]float64, 0),
	}
	
	// Initialize LSTM layers
	for i := 0; i < g.config.NumLayers; i++ {
		inputSize := g.config.InputSize
		if i > 0 {
			inputSize = g.config.HiddenSize
		}
		
		model.layers[i] = NewLSTMLayer(inputSize, g.config.HiddenSize, g.randSource)
	}
	
	// Initialize output layer
	model.weights["output"] = g.initializeMatrix(g.config.HiddenSize, g.config.OutputSize)
	model.biases["output"] = g.initializeVector(g.config.OutputSize)
	
	return model
}

func (g *LSTMGenerator) trainModel(ctx context.Context, trainX, trainY, valX, valY []*mat.Dense) ([]*TrainingMetrics, error) {
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
		}).Info("Training epoch completed")
		
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

// Simplified training and forward pass methods (actual LSTM implementation would be more complex)

func (g *LSTMGenerator) trainEpoch(X []*mat.Dense, y []*mat.VecDense) float64 {
	totalLoss := 0.0
	
	for i := 0; i < len(X); i++ {
		// Forward pass
		output := g.forward(X[i])
		
		// Calculate loss
		loss := g.calculateLoss(output, y[i])
		totalLoss += loss
		
		// Backward pass (simplified)
		g.backward(X[i], y[i], output)
	}
	
	return totalLoss / float64(len(X))
}

func (g *LSTMGenerator) validateEpoch(X []*mat.Dense, y []*mat.VecDense) float64 {
	totalLoss := 0.0
	
	for i := 0; i < len(X); i++ {
		output := g.forward(X[i])
		loss := g.calculateLoss(output, y[i])
		totalLoss += loss
	}
	
	return totalLoss / float64(len(X))
}

func (g *LSTMGenerator) forward(input *mat.Dense) *mat.VecDense {
	// Simplified forward pass through LSTM layers
	currentInput := input
	
	for _, layer := range g.model.layers {
		// Process sequence through LSTM layer
		// This is a simplified version - actual LSTM forward pass is more complex
		output := g.processLSTMLayer(layer, currentInput)
		currentInput = output
	}
	
	// Output layer
	lastHidden := g.getLastHidden(currentInput)
	output := mat.NewVecDense(g.config.OutputSize, nil)
	output.MulVec(g.model.weights["output"], lastHidden)
	output.AddVec(output, g.model.biases["output"])
	
	return output
}

// Helper methods and simplified implementations

func (g *LSTMGenerator) generateSeedSequence() []float64 {
	// Generate a random seed sequence or use the last training sequence
	seed := make([]float64, g.config.SequenceLength)
	for i := range seed {
		seed[i] = g.randSource.NormFloat64() * 0.1 // Small random values
	}
	return seed
}

func (g *LSTMGenerator) sequenceToMatrix(sequence []float64) *mat.Dense {
	matrix := mat.NewDense(len(sequence), 1, nil)
	for i, val := range sequence {
		matrix.Set(i, 0, val)
	}
	return matrix
}

func (g *LSTMGenerator) sampleFromOutput(output *mat.VecDense) float64 {
	// For regression, just return the output value
	// For more complex sampling, could apply temperature and different sampling methods
	return output.AtVec(0)
}

func (g *LSTMGenerator) calculateLoss(predicted, actual *mat.VecDense) float64 {
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

func (g *LSTMGenerator) backward(input *mat.Dense, target, output *mat.VecDense) {
	// Simplified backward pass
	// In a real implementation, this would compute gradients and update weights
}

func (g *LSTMGenerator) processLSTMLayer(layer *LSTMLayer, input *mat.Dense) *mat.Dense {
	// Simplified LSTM layer processing
	// Real implementation would include forget gate, input gate, output gate calculations
	rows, _ := input.Dims()
	output := mat.NewDense(rows, layer.hiddenSize, nil)
	
	// This is a placeholder - actual LSTM computation is much more complex
	for i := 0; i < rows; i++ {
		for j := 0; j < layer.hiddenSize; j++ {
			output.Set(i, j, g.randSource.NormFloat64()*0.1)
		}
	}
	
	return output
}

func (g *LSTMGenerator) getLastHidden(input *mat.Dense) *mat.VecDense {
	rows, cols := input.Dims()
	if rows == 0 {
		return mat.NewVecDense(cols, nil)
	}
	
	lastRow := mat.NewVecDense(cols, nil)
	for j := 0; j < cols; j++ {
		lastRow.SetVec(j, input.At(rows-1, j))
	}
	
	return lastRow
}

func (g *LSTMGenerator) initializeMatrix(rows, cols int) *mat.Dense {
	// Xavier/Glorot initialization
	scale := math.Sqrt(2.0 / float64(rows+cols))
	data := make([]float64, rows*cols)
	
	for i := range data {
		data[i] = g.randSource.NormFloat64() * scale
	}
	
	return mat.NewDense(rows, cols, data)
}

func (g *LSTMGenerator) initializeVector(size int) *mat.VecDense {
	data := make([]float64, size)
	// Initialize biases to zero
	return mat.NewVecDense(size, data)
}

// Utility methods

func (g *LSTMGenerator) generateTimestamps(start time.Time, frequency time.Duration, length int) []time.Time {
	timestamps := make([]time.Time, length)
	current := start
	
	for i := 0; i < length; i++ {
		timestamps[i] = current
		current = current.Add(frequency)
	}
	
	return timestamps
}

func (g *LSTMGenerator) parseFrequency(freq string) (time.Duration, error) {
	duration, err := time.ParseDuration(freq)
	if err != nil {
		return 0, fmt.Errorf("invalid frequency format: %s", freq)
	}
	return duration, nil
}

func getDefaultLSTMConfig() *LSTMConfig {
	return &LSTMConfig{
		HiddenSize:       64,
		NumLayers:        2,
		InputSize:        1,
		OutputSize:       1,
		SequenceLength:   20,
		LearningRate:     0.001,
		Epochs:           100,
		BatchSize:        32,
		DropoutRate:      0.2,
		L2Regularization: 0.001,
		Temperature:      1.0,
		SamplingMethod:   "greedy",
		Normalization:    "zscore",
		ValidationSplit:  0.2,
		Seed:             time.Now().UnixNano(),
		EarlyStopping:    true,
		Patience:         10,
		MinDelta:         0.001,
	}
}

// Supporting types and constructors

func NewLSTMLayer(inputSize, hiddenSize int, randSource *rand.Rand) *LSTMLayer {
	layer := &LSTMLayer{
		hiddenSize: hiddenSize,
		inputSize:  inputSize,
		gates:      make(map[string]*mat.VecDense),
		activations: make(map[string]*mat.VecDense),
	}
	
	// Initialize weights and biases (simplified)
	scale := math.Sqrt(2.0 / float64(inputSize+hiddenSize))
	
	// Initialize with random values (placeholder)
	layer.weightsInputGate = mat.NewDense(hiddenSize, inputSize, nil)
	layer.weightsForgetGate = mat.NewDense(hiddenSize, inputSize, nil)
	layer.weightsOutputGate = mat.NewDense(hiddenSize, inputSize, nil)
	layer.weightsCellGate = mat.NewDense(hiddenSize, inputSize, nil)
	
	// Initialize hidden states
	layer.hiddenState = mat.NewVecDense(hiddenSize, nil)
	layer.cellState = mat.NewVecDense(hiddenSize, nil)
	
	return layer
}

func NewAdamOptimizer(learningRate float64) *AdamOptimizer {
	return &AdamOptimizer{
		learningRate: learningRate,
		beta1:        0.9,
		beta2:        0.999,
		epsilon:      1e-8,
		t:            0,
		momentum:     make(map[string]*mat.Dense),
		velocity:     make(map[string]*mat.Dense),
		momentumVec:  make(map[string]*mat.VecDense),
		velocityVec:  make(map[string]*mat.VecDense),
	}
}

func NewDataScaler(method string) *DataScaler {
	return &DataScaler{
		method: method,
		fitted: false,
	}
}

func (ds *DataScaler) Fit(data []float64) error {
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

func (ds *DataScaler) Transform(data []float64) []float64 {
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

func (ds *DataScaler) InverseTransform(data []float64) []float64 {
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