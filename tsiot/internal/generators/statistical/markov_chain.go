package statistical

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/constants"
	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/models"
)

// MarkovChainGenerator generates synthetic time series using Markov chain models
type MarkovChainGenerator struct {
	logger         *logrus.Logger
	config         *MarkovChainConfig
	trained        bool
	states         []float64                    // Discretized states
	transitionMatrix [][]float64                // Transition probability matrix
	initialDistribution []float64               // Initial state distribution
	stateBounds    []float64                    // Boundaries between states
	numStates      int
	randSource     *rand.Rand
}

// MarkovChainConfig contains configuration for Markov chain generation
type MarkovChainConfig struct {
	NumStates       int     `json:"num_states"`       // Number of discrete states
	Order           int     `json:"order"`            // Order of Markov chain (1, 2, etc.)
	Discretization  string  `json:"discretization"`   // "uniform", "quantile", "kmeans"
	SmoothingFactor float64 `json:"smoothing_factor"` // Laplace smoothing parameter
	NoiseLevel      float64 `json:"noise_level"`      // Amount of Gaussian noise to add
	MemoryLength    int     `json:"memory_length"`    // Length of memory for higher-order chains
	Seed            int64   `json:"seed"`             // Random seed
}

// NewMarkovChainGenerator creates a new Markov chain generator
func NewMarkovChainGenerator(config *MarkovChainConfig, logger *logrus.Logger) *MarkovChainGenerator {
	if config == nil {
		config = getDefaultMarkovChainConfig()
	}
	
	if logger == nil {
		logger = logrus.New()
	}
	
	if config.Seed == 0 {
		config.Seed = time.Now().UnixNano()
	}
	
	return &MarkovChainGenerator{
		logger:     logger,
		config:     config,
		trained:    false,
		randSource: rand.New(rand.NewSource(config.Seed)),
	}
}

// GetType returns the generator type
func (m *MarkovChainGenerator) GetType() models.GeneratorType {
	return models.GeneratorType(constants.GeneratorTypeMarkov)
}

// GetName returns a human-readable name for the generator
func (m *MarkovChainGenerator) GetName() string {
	return "Markov Chain Generator"
}

// GetDescription returns a description of the generator
func (m *MarkovChainGenerator) GetDescription() string {
	return "Generates synthetic time series using Markov chain models to capture temporal dependencies and state transitions"
}

// GetSupportedSensorTypes returns the sensor types this generator supports
func (m *MarkovChainGenerator) GetSupportedSensorTypes() []models.SensorType {
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
func (m *MarkovChainGenerator) ValidateParameters(params models.GenerationParameters) error {
	if params.Length <= 0 {
		return errors.NewValidationError("INVALID_LENGTH", "Generation length must be positive")
	}
	
	if params.Frequency == "" {
		return errors.NewValidationError("INVALID_FREQUENCY", "Frequency is required")
	}
	
	if !m.trained {
		return errors.NewGenerationError("MODEL_NOT_TRAINED", "Markov chain generator must be trained before generation")
	}
	
	return nil
}

// Generate generates synthetic data based on the request
func (m *MarkovChainGenerator) Generate(ctx context.Context, req *models.GenerationRequest) (*models.GenerationResult, error) {
	if req == nil {
		return nil, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}
	
	if err := m.ValidateParameters(req.Parameters); err != nil {
		return nil, err
	}
	
	m.logger.WithFields(logrus.Fields{
		"request_id": req.ID,
		"length":     req.Parameters.Length,
		"num_states": m.numStates,
		"order":      m.config.Order,
	}).Info("Starting Markov chain generation")
	
	start := time.Now()
	
	// Parse frequency
	frequency, err := time.ParseDuration(req.Parameters.Frequency)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeValidation, "INVALID_FREQUENCY", "Failed to parse frequency")
	}
	
	// Generate timestamps
	timestamps := make([]time.Time, req.Parameters.Length)
	current := req.Parameters.StartTime
	for i := 0; i < req.Parameters.Length; i++ {
		timestamps[i] = current
		current = current.Add(frequency)
	}
	
	// Generate synthetic time series using Markov chain
	values := m.generateMarkovSeries(req.Parameters.Length)
	
	// Create data points
	dataPoints := make([]models.DataPoint, len(timestamps))
	for i, timestamp := range timestamps {
		dataPoints[i] = models.DataPoint{
			Timestamp: timestamp,
			Value:     values[i],
			Quality:   0.88, // Good quality for Markov generation
		}
	}
	
	// Create time series
	timeSeries := &models.TimeSeries{
		ID:          fmt.Sprintf("markov-%d", time.Now().UnixNano()),
		Name:        "Markov Chain Generated Series",
		Description: "Synthetic time series data generated using Markov chain models",
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
		GeneratorType: string(m.GetType()),
		Quality:       0.88,
		Metadata: map[string]interface{}{
			"num_states":       m.numStates,
			"order":            m.config.Order,
			"discretization":   m.config.Discretization,
			"smoothing_factor": m.config.SmoothingFactor,
			"noise_level":      m.config.NoiseLevel,
			"data_points":      len(dataPoints),
			"generation_time":  duration.String(),
		},
	}
	
	m.logger.WithFields(logrus.Fields{
		"request_id":  req.ID,
		"data_points": len(dataPoints),
		"duration":    duration,
		"states":      m.numStates,
	}).Info("Completed Markov chain generation")
	
	return result, nil
}

// Train trains the generator with reference data by learning transition probabilities
func (m *MarkovChainGenerator) Train(ctx context.Context, data *models.TimeSeries, params models.GenerationParameters) error {
	if data == nil {
		return errors.NewValidationError("INVALID_DATA", "Training data is required")
	}
	
	if len(data.DataPoints) < m.config.NumStates*2 {
		return errors.NewValidationError("INSUFFICIENT_DATA", fmt.Sprintf("Need at least %d data points for Markov chain training", m.config.NumStates*2))
	}
	
	m.logger.WithFields(logrus.Fields{
		"series_id":      data.ID,
		"data_points":    len(data.DataPoints),
		"num_states":     m.config.NumStates,
		"discretization": m.config.Discretization,
	}).Info("Training Markov chain generator")
	
	start := time.Now()
	
	// Extract values from data points
	values := make([]float64, len(data.DataPoints))
	for i, dp := range data.DataPoints {
		values[i] = dp.Value
	}
	
	// Discretize the continuous values into states
	m.discretizeValues(values)
	
	// Convert values to state indices
	stateSequence := m.valuesToStates(values)
	
	// Learn transition probabilities
	m.learnTransitions(stateSequence)
	
	// Calculate initial state distribution
	m.calculateInitialDistribution(stateSequence)
	
	m.trained = true
	duration := time.Since(start)
	
	m.logger.WithFields(logrus.Fields{
		"series_id":         data.ID,
		"training_duration": duration,
		"num_states":        m.numStates,
		"transitions":       len(m.transitionMatrix),
	}).Info("Markov chain generator training completed")
	
	return nil
}

// IsTrainable returns true if the generator requires/supports training
func (m *MarkovChainGenerator) IsTrainable() bool {
	return true
}

// generateMarkovSeries generates a time series using learned Markov chain
func (m *MarkovChainGenerator) generateMarkovSeries(length int) []float64 {
	values := make([]float64, length)
	
	// Start with initial state
	currentState := m.sampleFromDistribution(m.initialDistribution)
	
	// Generate sequence using transition probabilities
	for i := 0; i < length; i++ {
		// Convert state index to continuous value
		value := m.stateToValue(currentState)
		
		// Add Gaussian noise if configured
		if m.config.NoiseLevel > 0 {
			noise := m.randSource.NormFloat64() * m.config.NoiseLevel
			value += noise
		}
		
		values[i] = value
		
		// Transition to next state
		if i < length-1 {
			currentState = m.sampleFromDistribution(m.transitionMatrix[currentState])
		}
	}
	
	return values
}

// discretizeValues creates discrete states from continuous values
func (m *MarkovChainGenerator) discretizeValues(values []float64) {
	m.numStates = m.config.NumStates
	m.states = make([]float64, m.numStates)
	m.stateBounds = make([]float64, m.numStates+1)
	
	switch m.config.Discretization {
	case "uniform":
		m.uniformDiscretization(values)
	case "quantile":
		m.quantileDiscretization(values)
	case "kmeans":
		m.kmeansDiscretization(values)
	default:
		m.uniformDiscretization(values)
	}
}

// uniformDiscretization creates equally-spaced states
func (m *MarkovChainGenerator) uniformDiscretization(values []float64) {
	minVal := values[0]
	maxVal := values[0]
	
	for _, v := range values {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	
	// Create uniform boundaries
	range_ := maxVal - minVal
	stepSize := range_ / float64(m.numStates)
	
	for i := 0; i <= m.numStates; i++ {
		m.stateBounds[i] = minVal + float64(i)*stepSize
	}
	
	// Set state centers
	for i := 0; i < m.numStates; i++ {
		m.states[i] = (m.stateBounds[i] + m.stateBounds[i+1]) / 2
	}
}

// quantileDiscretization creates states based on value quantiles
func (m *MarkovChainGenerator) quantileDiscretization(values []float64) {
	// Sort values for quantile calculation
	sortedValues := make([]float64, len(values))
	copy(sortedValues, values)
	sort.Float64s(sortedValues)
	
	// Calculate quantile boundaries
	for i := 0; i <= m.numStates; i++ {
		quantile := float64(i) / float64(m.numStates)
		index := int(quantile * float64(len(sortedValues)-1))
		if index >= len(sortedValues) {
			index = len(sortedValues) - 1
		}
		m.stateBounds[i] = sortedValues[index]
	}
	
	// Set state centers
	for i := 0; i < m.numStates; i++ {
		m.states[i] = (m.stateBounds[i] + m.stateBounds[i+1]) / 2
	}
}

// kmeansDiscretization uses k-means clustering for discretization
func (m *MarkovChainGenerator) kmeansDiscretization(values []float64) {
	// Simple k-means implementation
	centroids := make([]float64, m.numStates)
	
	// Initialize centroids randomly
	minVal, maxVal := values[0], values[0]
	for _, v := range values {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	
	for i := 0; i < m.numStates; i++ {
		centroids[i] = minVal + m.randSource.Float64()*(maxVal-minVal)
	}
	
	// K-means iterations
	for iter := 0; iter < 10; iter++ {
		// Assign points to nearest centroid
		clusters := make([][]float64, m.numStates)
		for i := range clusters {
			clusters[i] = make([]float64, 0)
		}
		
		for _, v := range values {
			closestCluster := 0
			minDistance := math.Abs(v - centroids[0])
			
			for j := 1; j < m.numStates; j++ {
				distance := math.Abs(v - centroids[j])
				if distance < minDistance {
					minDistance = distance
					closestCluster = j
				}
			}
			
			clusters[closestCluster] = append(clusters[closestCluster], v)
		}
		
		// Update centroids
		for i := 0; i < m.numStates; i++ {
			if len(clusters[i]) > 0 {
				sum := 0.0
				for _, v := range clusters[i] {
					sum += v
				}
				centroids[i] = sum / float64(len(clusters[i]))
			}
		}
	}
	
	// Sort centroids and create boundaries
	sort.Float64s(centroids)
	copy(m.states, centroids)
	
	// Create boundaries midway between centroids
	m.stateBounds[0] = centroids[0] - (centroids[1]-centroids[0])/2
	for i := 1; i < m.numStates; i++ {
		m.stateBounds[i] = (centroids[i-1] + centroids[i]) / 2
	}
	m.stateBounds[m.numStates] = centroids[m.numStates-1] + (centroids[m.numStates-1]-centroids[m.numStates-2])/2
}

// valuesToStates converts continuous values to discrete state indices
func (m *MarkovChainGenerator) valuesToStates(values []float64) []int {
	states := make([]int, len(values))
	
	for i, v := range values {
		states[i] = m.valueToState(v)
	}
	
	return states
}

// valueToState converts a single value to state index
func (m *MarkovChainGenerator) valueToState(value float64) int {
	for i := 0; i < m.numStates; i++ {
		if value >= m.stateBounds[i] && value < m.stateBounds[i+1] {
			return i
		}
	}
	// Handle edge case for maximum value
	return m.numStates - 1
}

// stateToValue converts state index to continuous value (state center)
func (m *MarkovChainGenerator) stateToValue(state int) float64 {
	if state >= 0 && state < m.numStates {
		return m.states[state]
	}
	return m.states[0] // Fallback
}

// learnTransitions learns transition probabilities from state sequence
func (m *MarkovChainGenerator) learnTransitions(stateSequence []int) {
	// Initialize transition count matrix
	transitionCounts := make([][]float64, m.numStates)
	for i := range transitionCounts {
		transitionCounts[i] = make([]float64, m.numStates)
	}
	
	// Count transitions
	for i := 0; i < len(stateSequence)-1; i++ {
		fromState := stateSequence[i]
		toState := stateSequence[i+1]
		transitionCounts[fromState][toState]++
	}
	
	// Convert counts to probabilities with Laplace smoothing
	m.transitionMatrix = make([][]float64, m.numStates)
	for i := range m.transitionMatrix {
		m.transitionMatrix[i] = make([]float64, m.numStates)
		
		// Calculate row sum for normalization
		rowSum := 0.0
		for j := 0; j < m.numStates; j++ {
			rowSum += transitionCounts[i][j] + m.config.SmoothingFactor
		}
		
		// Normalize to probabilities
		for j := 0; j < m.numStates; j++ {
			m.transitionMatrix[i][j] = (transitionCounts[i][j] + m.config.SmoothingFactor) / rowSum
		}
	}
}

// calculateInitialDistribution calculates initial state probabilities
func (m *MarkovChainGenerator) calculateInitialDistribution(stateSequence []int) {
	m.initialDistribution = make([]float64, m.numStates)
	
	// Count initial states (could be more sophisticated)
	stateCounts := make([]float64, m.numStates)
	for _, state := range stateSequence {
		stateCounts[state]++
	}
	
	// Convert to probabilities with smoothing
	totalCount := float64(len(stateSequence))
	for i := 0; i < m.numStates; i++ {
		m.initialDistribution[i] = (stateCounts[i] + m.config.SmoothingFactor) / (totalCount + float64(m.numStates)*m.config.SmoothingFactor)
	}
}

// sampleFromDistribution samples a state index from probability distribution
func (m *MarkovChainGenerator) sampleFromDistribution(distribution []float64) int {
	r := m.randSource.Float64()
	cumulative := 0.0
	
	for i, prob := range distribution {
		cumulative += prob
		if r <= cumulative {
			return i
		}
	}
	
	// Fallback to last state
	return len(distribution) - 1
}

// GetDefaultParameters returns default parameters for this generator
func (m *MarkovChainGenerator) GetDefaultParameters() models.GenerationParameters {
	return models.GenerationParameters{
		Length:    1000,
		Frequency: "1m",
		StartTime: time.Now().Add(-24 * time.Hour),
		Tags:      make(map[string]string),
		Metadata:  make(map[string]interface{}),
	}
}

// EstimateDuration estimates how long generation will take
func (m *MarkovChainGenerator) EstimateDuration(req *models.GenerationRequest) (time.Duration, error) {
	if req == nil {
		return 0, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}
	
	// Markov chain generation is very fast, roughly 2ms per 1000 data points
	pointsPerMs := 500.0
	estimatedMs := float64(req.Parameters.Length) / pointsPerMs
	return time.Duration(estimatedMs) * time.Millisecond, nil
}

// Cancel cancels an ongoing generation
func (m *MarkovChainGenerator) Cancel(ctx context.Context, requestID string) error {
	m.logger.WithField("request_id", requestID).Info("Cancel requested for Markov chain generation")
	return nil
}

// GetProgress returns the progress of an ongoing generation
func (m *MarkovChainGenerator) GetProgress(requestID string) (float64, error) {
	return 1.0, nil
}

// Close cleans up resources
func (m *MarkovChainGenerator) Close() error {
	m.logger.Info("Closing Markov chain generator")
	return nil
}

func getDefaultMarkovChainConfig() *MarkovChainConfig {
	return &MarkovChainConfig{
		NumStates:       10,
		Order:           1,
		Discretization:  "quantile",
		SmoothingFactor: 0.01,
		NoiseLevel:      0.05,
		MemoryLength:    5,
		Seed:            time.Now().UnixNano(),
	}
}
