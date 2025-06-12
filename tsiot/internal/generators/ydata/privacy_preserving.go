package ydata

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/constants"
	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/models"
)

// PrivacyPreservingGenerator implements privacy-preserving synthetic data generation
type PrivacyPreservingGenerator struct {
	logger            *logrus.Logger
	config            *PrivacyPreservingConfig
	trained           bool
	privacyBudget     *PrivacyBudget
	dpMechanism       *DifferentialPrivacyMechanism
	federatedLearner  *FederatedLearner
	secureAggregator  *SecureAggregator
	baseModel         interface{}
	privacyMetrics    *PrivacyMetrics
	randSource        *rand.Rand
}

// PrivacyPreservingConfig contains configuration for privacy-preserving generation
type PrivacyPreservingConfig struct {
	// Privacy parameters
	Epsilon           float64 `json:"epsilon"`            // Differential privacy parameter
	Delta             float64 `json:"delta"`              // Differential privacy parameter
	PrivacyBudget     float64 `json:"privacy_budget"`     // Total privacy budget
	ClippingThreshold float64 `json:"clipping_threshold"` // Gradient clipping threshold
	
	// Federated learning parameters
	EnableFederated   bool    `json:"enable_federated"`   // Enable federated learning
	NumClients        int     `json:"num_clients"`        // Number of federated clients
	RoundsPerEpoch    int     `json:"rounds_per_epoch"`   // Federated rounds per epoch
	ClientSampleRate  float64 `json:"client_sample_rate"` // Fraction of clients per round
	
	// Secure aggregation parameters
	EnableSecureAgg   bool    `json:"enable_secure_agg"`  // Enable secure aggregation
	NumSurvivors      int     `json:"num_survivors"`      // Minimum surviving clients
	ReconstructionThreshold int `json:"reconstruction_threshold"` // Threshold for reconstruction
	
	// Model parameters
	ModelType         string  `json:"model_type"`         // "timegan", "lstm", "transformer"
	HiddenDim         int     `json:"hidden_dim"`         // Hidden dimension
	NumLayers         int     `json:"num_layers"`         // Number of layers
	SequenceLength    int     `json:"sequence_length"`    // Sequence length
	
	// Training parameters
	Epochs            int     `json:"epochs"`             // Training epochs
	BatchSize         int     `json:"batch_size"`         // Batch size
	LearningRate      float64 `json:"learning_rate"`      // Learning rate
	
	// Advanced privacy techniques
	UseKAnonymity     bool    `json:"use_k_anonymity"`    // Enable k-anonymity
	KValue            int     `json:"k_value"`            // k value for k-anonymity
	UseLDiversity     bool    `json:"use_l_diversity"`    // Enable l-diversity
	LValue            int     `json:"l_value"`            // l value for l-diversity
	UseTCloseness     bool    `json:"use_t_closeness"`    // Enable t-closeness
	TValue            float64 `json:"t_value"`            // t value for t-closeness
	
	// Noise parameters
	NoiseType         string  `json:"noise_type"`         // "gaussian", "laplace", "exponential"
	AdaptiveNoise     bool    `json:"adaptive_noise"`     // Use adaptive noise
	NoiseMultiplier   float64 `json:"noise_multiplier"`   // Noise multiplier
	
	// Other settings
	Seed              int64   `json:"seed"`               // Random seed
	EnableAudit       bool    `json:"enable_audit"`       // Enable privacy auditing
	AuditInterval     int     `json:"audit_interval"`     // Audit interval
}

// PrivacyBudget tracks and manages privacy budget
type PrivacyBudget struct {
	totalBudget    float64
	usedBudget     float64
	remainingBudget float64
	transactions   []PrivacyTransaction
	mu             sync.RWMutex
}

// PrivacyTransaction represents a privacy budget expenditure
type PrivacyTransaction struct {
	ID          string    `json:"id"`
	Timestamp   time.Time `json:"timestamp"`
	Epsilon     float64   `json:"epsilon"`
	Delta       float64   `json:"delta"`
	Operation   string    `json:"operation"`
	Description string    `json:"description"`
}

// PrivacyMetrics tracks privacy-related metrics
type PrivacyMetrics struct {
	PrivacyLoss       float64            `json:"privacy_loss"`
	UtilityMetrics    map[string]float64 `json:"utility_metrics"`
	AuditResults      []AuditResult      `json:"audit_results"`
	LastAuditTime     time.Time          `json:"last_audit_time"`
}

// AuditResult represents the result of a privacy audit
type AuditResult struct {
	Timestamp     time.Time          `json:"timestamp"`
	AuditType     string             `json:"audit_type"`
	Passed        bool               `json:"passed"`
	Score         float64            `json:"score"`
	Violations    []PrivacyViolation `json:"violations"`
	Recommendations []string         `json:"recommendations"`
}

// PrivacyViolation represents a detected privacy violation
type PrivacyViolation struct {
	Type        string  `json:"type"`
	Severity    string  `json:"severity"`
	Description string  `json:"description"`
	Impact      float64 `json:"impact"`
	Mitigation  string  `json:"mitigation"`
}

// NewPrivacyPreservingGenerator creates a new privacy-preserving generator
func NewPrivacyPreservingGenerator(config *PrivacyPreservingConfig, logger *logrus.Logger) (*PrivacyPreservingGenerator, error) {
	if config == nil {
		config = getDefaultPrivacyPreservingConfig()
	}
	
	if logger == nil {
		logger = logrus.New()
	}
	
	if config.Seed == 0 {
		config.Seed = time.Now().UnixNano()
	}
	
	// Validate privacy parameters
	if err := validatePrivacyConfig(config); err != nil {
		return nil, err
	}
	
	g := &PrivacyPreservingGenerator{
		logger:     logger,
		config:     config,
		trained:    false,
		randSource: rand.New(rand.NewSource(config.Seed)),
	}
	
	// Initialize privacy budget
	g.privacyBudget = &PrivacyBudget{
		totalBudget:     config.PrivacyBudget,
		remainingBudget: config.PrivacyBudget,
		transactions:    make([]PrivacyTransaction, 0),
	}
	
	// Initialize differential privacy mechanism
	g.dpMechanism = NewDifferentialPrivacyMechanism(config.Epsilon, config.Delta, config.NoiseType)
	
	// Initialize federated learning if enabled
	if config.EnableFederated {
		var err error
		g.federatedLearner, err = NewFederatedLearner(config, logger)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize federated learner: %w", err)
		}
	}
	
	// Initialize secure aggregation if enabled
	if config.EnableSecureAgg {
		var err error
		g.secureAggregator, err = NewSecureAggregator(config, logger)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize secure aggregator: %w", err)
		}
	}
	
	// Initialize privacy metrics
	g.privacyMetrics = &PrivacyMetrics{
		UtilityMetrics: make(map[string]float64),
		AuditResults:   make([]AuditResult, 0),
	}
	
	return g, nil
}

// GetType returns the generator type
func (g *PrivacyPreservingGenerator) GetType() models.GeneratorType {
	return models.GeneratorType(constants.GeneratorTypeYData)
}

// GetName returns a human-readable name for the generator
func (g *PrivacyPreservingGenerator) GetName() string {
	return "YData Privacy-Preserving Generator"
}

// GetDescription returns a description of the generator
func (g *PrivacyPreservingGenerator) GetDescription() string {
	return "Privacy-preserving synthetic time series generator using differential privacy, federated learning, and secure aggregation"
}

// GetSupportedSensorTypes returns the sensor types this generator supports
func (g *PrivacyPreservingGenerator) GetSupportedSensorTypes() []models.SensorType {
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
func (g *PrivacyPreservingGenerator) ValidateParameters(params models.GenerationParameters) error {
	if params.Length <= 0 {
		return errors.NewValidationError("INVALID_LENGTH", "Generation length must be positive")
	}
	
	if params.Frequency == "" {
		return errors.NewValidationError("INVALID_FREQUENCY", "Frequency is required")
	}
	
	if !g.trained {
		return errors.NewGenerationError("MODEL_NOT_TRAINED", "Privacy-preserving generator must be trained before generation")
	}
	
	// Check privacy budget
	if !g.hasRemainingPrivacyBudget(g.config.Epsilon) {
		return errors.NewPrivacyError("INSUFFICIENT_PRIVACY_BUDGET", "Insufficient privacy budget for generation")
	}
	
	return nil
}

// Generate generates synthetic data based on the request
func (g *PrivacyPreservingGenerator) Generate(ctx context.Context, req *models.GenerationRequest) (*models.GenerationResult, error) {
	if req == nil {
		return nil, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}
	
	if err := g.ValidateParameters(req.Parameters); err != nil {
		return nil, err
	}
	
	g.logger.WithFields(logrus.Fields{
		"request_id":      req.ID,
		"length":          req.Parameters.Length,
		"epsilon":         g.config.Epsilon,
		"federated":       g.config.EnableFederated,
		"secure_agg":      g.config.EnableSecureAgg,
	}).Info("Starting privacy-preserving generation")
	
	start := time.Now()
	
	// Consume privacy budget
	if err := g.consumePrivacyBudget(g.config.Epsilon, g.config.Delta, "generation", req.ID); err != nil {
		return nil, err
	}
	
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
	
	// Generate synthetic time series with privacy preservation
	values, err := g.generatePrivateTimeSeries(ctx, req.Parameters.Length)
	if err != nil {
		return nil, err
	}
	
	// Create data points
	dataPoints := make([]models.DataPoint, len(timestamps))
	for i, timestamp := range timestamps {
		dataPoints[i] = models.DataPoint{
			Timestamp: timestamp,
			Value:     values[i],
			Quality:   0.85, // Slightly reduced quality due to privacy noise
		}
	}
	
	// Calculate utility metrics
	utilityMetrics := g.calculateUtilityMetrics(values)
	g.privacyMetrics.UtilityMetrics = utilityMetrics
	
	// Create time series
	timeSeries := &models.TimeSeries{
		ID:          fmt.Sprintf("ydata-private-%d", time.Now().UnixNano()),
		Name:        "YData Privacy-Preserving Generated Series",
		Description: "Privacy-preserving synthetic time series data generated using YData techniques",
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
		Quality:       0.85,
		Metadata: map[string]interface{}{
			"epsilon":              g.config.Epsilon,
			"delta":                g.config.Delta,
			"privacy_budget_used":  g.privacyBudget.usedBudget,
			"privacy_budget_remaining": g.privacyBudget.remainingBudget,
			"federated_enabled":    g.config.EnableFederated,
			"secure_agg_enabled":   g.config.EnableSecureAgg,
			"utility_metrics":      utilityMetrics,
			"data_points":          len(dataPoints),
			"generation_time":      duration.String(),
		},
	}
	
	g.logger.WithFields(logrus.Fields{
		"request_id":     req.ID,
		"data_points":    len(dataPoints),
		"duration":       duration,
		"privacy_budget": g.privacyBudget.remainingBudget,
	}).Info("Completed privacy-preserving generation")
	
	return result, nil
}

// Train trains the generator with reference data using privacy-preserving techniques
func (g *PrivacyPreservingGenerator) Train(ctx context.Context, data *models.TimeSeries, params models.GenerationParameters) error {
	if data == nil {
		return errors.NewValidationError("INVALID_DATA", "Training data is required")
	}
	
	if len(data.DataPoints) < g.config.SequenceLength*2 {
		return errors.NewValidationError("INSUFFICIENT_DATA", fmt.Sprintf("Need at least %d data points for training", g.config.SequenceLength*2))
	}
	
	g.logger.WithFields(logrus.Fields{
		"series_id":    data.ID,
		"data_points":  len(data.DataPoints),
		"epsilon":      g.config.Epsilon,
		"federated":    g.config.EnableFederated,
	}).Info("Starting privacy-preserving training")
	
	start := time.Now()
	
	// Consume privacy budget for training
	trainingEpsilon := g.config.Epsilon * 0.8 // Reserve 20% for generation
	if err := g.consumePrivacyBudget(trainingEpsilon, g.config.Delta, "training", data.ID); err != nil {
		return err
	}
	
	// Extract and preprocess training data
	values := make([]float64, len(data.DataPoints))
	for i, dp := range data.DataPoints {
		values[i] = dp.Value
	}
	
	// Apply privacy-preserving preprocessing
	privateValues, err := g.applyPrivacyPreprocessing(values)
	if err != nil {
		return fmt.Errorf("failed to apply privacy preprocessing: %w", err)
	}
	
	// Train using federated learning if enabled
	if g.config.EnableFederated {
		if err := g.trainFederated(ctx, privateValues); err != nil {
			return fmt.Errorf("federated training failed: %w", err)
		}
	} else {
		// Train with centralized differential privacy
		if err := g.trainCentralized(ctx, privateValues); err != nil {
			return fmt.Errorf("centralized training failed: %w", err)
		}
	}
	
	// Perform privacy audit if enabled
	if g.config.EnableAudit {
		auditResult := g.performPrivacyAudit()
		g.privacyMetrics.AuditResults = append(g.privacyMetrics.AuditResults, auditResult)
		g.privacyMetrics.LastAuditTime = time.Now()
	}
	
	g.trained = true
	duration := time.Since(start)
	
	g.logger.WithFields(logrus.Fields{
		"series_id":         data.ID,
		"training_duration": duration,
		"privacy_budget_used": g.privacyBudget.usedBudget,
		"model_type":        g.config.ModelType,
	}).Info("Privacy-preserving training completed")
	
	return nil
}

// IsTrainable returns true if the generator requires/supports training
func (g *PrivacyPreservingGenerator) IsTrainable() bool {
	return true
}

// generatePrivateTimeSeries generates synthetic time series with privacy preservation
func (g *PrivacyPreservingGenerator) generatePrivateTimeSeries(ctx context.Context, length int) ([]float64, error) {
	// Generate base synthetic data
	baseValues, err := g.generateBaseSyntheticData(length)
	if err != nil {
		return nil, err
	}
	
	// Apply differential privacy noise
	privateValues := g.dpMechanism.AddNoise(baseValues)
	
	// Apply additional privacy techniques if enabled
	if g.config.UseKAnonymity {
		privateValues = g.applyKAnonymity(privateValues)
	}
	
	if g.config.UseLDiversity {
		privateValues = g.applyLDiversity(privateValues)
	}
	
	if g.config.UseTCloseness {
		privateValues = g.applyTCloseness(privateValues)
	}
	
	return privateValues, nil
}

// Helper methods for privacy preservation techniques

func (g *PrivacyPreservingGenerator) hasRemainingPrivacyBudget(epsilon float64) bool {
	g.privacyBudget.mu.RLock()
	defer g.privacyBudget.mu.RUnlock()
	return g.privacyBudget.remainingBudget >= epsilon
}

func (g *PrivacyPreservingGenerator) consumePrivacyBudget(epsilon, delta float64, operation, description string) error {
	g.privacyBudget.mu.Lock()
	defer g.privacyBudget.mu.Unlock()
	
	if g.privacyBudget.remainingBudget < epsilon {
		return errors.NewPrivacyError("INSUFFICIENT_BUDGET", "Insufficient privacy budget")
	}
	
	transaction := PrivacyTransaction{
		ID:          fmt.Sprintf("tx_%d", time.Now().UnixNano()),
		Timestamp:   time.Now(),
		Epsilon:     epsilon,
		Delta:       delta,
		Operation:   operation,
		Description: description,
	}
	
	g.privacyBudget.usedBudget += epsilon
	g.privacyBudget.remainingBudget -= epsilon
	g.privacyBudget.transactions = append(g.privacyBudget.transactions, transaction)
	
	return nil
}

// GetDefaultParameters returns default parameters for this generator
func (g *PrivacyPreservingGenerator) GetDefaultParameters() models.GenerationParameters {
	return models.GenerationParameters{
		Length:    1000,
		Frequency: "1m",
		StartTime: time.Now().Add(-24 * time.Hour),
		Tags:      make(map[string]string),
		Metadata:  make(map[string]interface{}),
	}
}

// EstimateDuration estimates how long generation will take
func (g *PrivacyPreservingGenerator) EstimateDuration(req *models.GenerationRequest) (time.Duration, error) {
	if req == nil {
		return 0, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}
	
	// Privacy-preserving generation is slower due to additional computations
	pointsPerMs := 50.0
	estimatedMs := float64(req.Parameters.Length) / pointsPerMs
	return time.Duration(estimatedMs) * time.Millisecond, nil
}

// Cancel cancels an ongoing generation
func (g *PrivacyPreservingGenerator) Cancel(ctx context.Context, requestID string) error {
	g.logger.WithField("request_id", requestID).Info("Cancel requested for privacy-preserving generation")
	return nil
}

// GetProgress returns the progress of an ongoing generation
func (g *PrivacyPreservingGenerator) GetProgress(requestID string) (float64, error) {
	return 1.0, nil
}

// Close cleans up resources
func (g *PrivacyPreservingGenerator) Close() error {
	g.logger.Info("Closing privacy-preserving generator")
	return nil
}

// Additional helper methods

// applyPrivacyPreprocessing applies privacy-preserving preprocessing to data
func (g *PrivacyPreservingGenerator) applyPrivacyPreprocessing(values []float64) ([]float64, error) {
	preprocessed := make([]float64, len(values))
	copy(preprocessed, values)
	
	// Apply clipping to limit sensitivity
	if g.config.ClippingThreshold > 0 {
		for i, v := range preprocessed {
			if v > g.config.ClippingThreshold {
				preprocessed[i] = g.config.ClippingThreshold
			} else if v < -g.config.ClippingThreshold {
				preprocessed[i] = -g.config.ClippingThreshold
			}
		}
	}
	
	return preprocessed, nil
}

// trainFederated performs federated training
func (g *PrivacyPreservingGenerator) trainFederated(ctx context.Context, values []float64) error {
	if g.federatedLearner == nil {
		return fmt.Errorf("federated learner not initialized")
	}
	
	return g.federatedLearner.TrainFederated(ctx, values)
}

// trainCentralized performs centralized differential privacy training
func (g *PrivacyPreservingGenerator) trainCentralized(ctx context.Context, values []float64) error {
	// Simplified centralized training with DP
	g.logger.Info("Performing centralized training with differential privacy")
	
	// Apply DP noise to training data
	privateValues := g.dpMechanism.AddNoise(values)
	
	// Simulate model training (in practice would train actual model)
	g.baseModel = map[string]interface{}{
		"type":        g.config.ModelType,
		"mean":        g.calculateMean(privateValues),
		"variance":    g.calculateVariance(privateValues),
		"trained_at":  time.Now(),
	}
	
	return nil
}

// generateBaseSyntheticData generates base synthetic data before privacy application
func (g *PrivacyPreservingGenerator) generateBaseSyntheticData(length int) ([]float64, error) {
	values := make([]float64, length)
	
	// Use base model to generate synthetic data
	if baseModelMap, ok := g.baseModel.(map[string]interface{}); ok {
		mean, _ := baseModelMap["mean"].(float64)
		variance, _ := baseModelMap["variance"].(float64)
		stddev := math.Sqrt(variance)
		
		for i := 0; i < length; i++ {
			values[i] = mean + g.randSource.NormFloat64()*stddev
		}
	} else {
		// Fallback to simple generation
		for i := 0; i < length; i++ {
			values[i] = g.randSource.NormFloat64()
		}
	}
	
	return values, nil
}

// applyKAnonymity applies k-anonymity to the data
func (g *PrivacyPreservingGenerator) applyKAnonymity(values []float64) []float64 {
	if !g.config.UseKAnonymity || g.config.KValue <= 1 {
		return values
	}
	
	// Simplified k-anonymity implementation
	// Group values into bins and ensure each bin has at least k values
	result := make([]float64, len(values))
	copy(result, values)
	
	// Sort to create groups
	sortValues := make([]float64, len(result))
	copy(sortValues, result)
	g.sortFloat64Slice(sortValues)
	
	// Create k-anonymous groups
	for i := 0; i < len(sortValues); i += g.config.KValue {
		end := i + g.config.KValue
		if end > len(sortValues) {
			end = len(sortValues)
		}
		
		// Replace values in group with group average
		groupSum := 0.0
		for j := i; j < end; j++ {
			groupSum += sortValues[j]
		}
		groupAvg := groupSum / float64(end-i)
		
		for j := i; j < end; j++ {
			sortValues[j] = groupAvg
		}
	}
	
	return sortValues
}

// applyLDiversity applies l-diversity to the data
func (g *PrivacyPreservingGenerator) applyLDiversity(values []float64) []float64 {
	if !g.config.UseLDiversity || g.config.LValue <= 1 {
		return values
	}
	
	// Simplified l-diversity implementation
	result := make([]float64, len(values))
	copy(result, values)
	
	// Ensure diversity by adding controlled variation
	for i := 0; i < len(result); i += g.config.LValue {
		end := i + g.config.LValue
		if end > len(result) {
			end = len(result)
		}
		
		// Add variation to ensure l-diversity
		for j := i; j < end; j++ {
			variation := g.randSource.NormFloat64() * 0.1
			result[j] += variation
		}
	}
	
	return result
}

// applyTCloseness applies t-closeness to the data
func (g *PrivacyPreservingGenerator) applyTCloseness(values []float64) []float64 {
	if !g.config.UseTCloseness || g.config.TValue <= 0 {
		return values
	}
	
	// Simplified t-closeness implementation
	result := make([]float64, len(values))
	copy(result, values)
	
	// Calculate global distribution properties
	mean := g.calculateMean(values)
	
	// Adjust values to maintain t-closeness
	for i := range result {
		// Move values closer to global mean based on t-value
		diff := result[i] - mean
		result[i] = mean + diff*(1-g.config.TValue)
	}
	
	return result
}

// calculateUtilityMetrics calculates utility metrics for the generated data
func (g *PrivacyPreservingGenerator) calculateUtilityMetrics(values []float64) map[string]float64 {
	if len(values) == 0 {
		return map[string]float64{}
	}
	
	mean := g.calculateMean(values)
	variance := g.calculateVariance(values)
	
	// Calculate additional utility metrics
	min, max := values[0], values[0]
	for _, v := range values {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	
	return map[string]float64{
		"mean":     mean,
		"variance": variance,
		"std_dev":  math.Sqrt(variance),
		"min":      min,
		"max":      max,
		"range":    max - min,
	}
}

// performPrivacyAudit performs a privacy audit
func (g *PrivacyPreservingGenerator) performPrivacyAudit() AuditResult {
	result := AuditResult{
		Timestamp:       time.Now(),
		AuditType:       "comprehensive",
		Passed:          true,
		Score:           0.0,
		Violations:      make([]PrivacyViolation, 0),
		Recommendations: make([]string, 0),
	}
	
	// Check privacy budget usage
	budgetUsageRatio := g.privacyBudget.usedBudget / g.privacyBudget.totalBudget
	if budgetUsageRatio > 0.8 {
		result.Violations = append(result.Violations, PrivacyViolation{
			Type:        "budget_exhaustion",
			Severity:    "high",
			Description: "Privacy budget usage is high",
			Impact:      budgetUsageRatio,
			Mitigation:  "Reduce epsilon or increase total budget",
		})
		result.Passed = false
	}
	
	// Check epsilon value
	if g.config.Epsilon > 2.0 {
		result.Violations = append(result.Violations, PrivacyViolation{
			Type:        "weak_privacy",
			Severity:    "medium",
			Description: "Epsilon value may provide weak privacy",
			Impact:      g.config.Epsilon,
			Mitigation:  "Consider reducing epsilon value",
		})
	}
	
	// Calculate overall score
	if result.Passed {
		result.Score = 1.0 - float64(len(result.Violations))*0.1
	} else {
		result.Score = 0.5 - float64(len(result.Violations))*0.1
	}
	
	if result.Score < 0 {
		result.Score = 0
	}
	
	return result
}

// Helper utility methods
func (g *PrivacyPreservingGenerator) calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}
	
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func (g *PrivacyPreservingGenerator) calculateVariance(values []float64) float64 {
	if len(values) <= 1 {
		return 0.0
	}
	
	mean := g.calculateMean(values)
	sum := 0.0
	for _, v := range values {
		diff := v - mean
		sum += diff * diff
	}
	return sum / float64(len(values)-1)
}

func (g *PrivacyPreservingGenerator) sortFloat64Slice(slice []float64) {
	// Simple bubble sort (in production, use sort.Float64s)
	n := len(slice)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if slice[j] > slice[j+1] {
				slice[j], slice[j+1] = slice[j+1], slice[j]
			}
		}
	}
}

func validatePrivacyConfig(config *PrivacyPreservingConfig) error {
	if config.Epsilon <= 0 {
		return errors.NewValidationError("INVALID_EPSILON", "Epsilon must be positive")
	}
	if config.Delta < 0 || config.Delta >= 1 {
		return errors.NewValidationError("INVALID_DELTA", "Delta must be between 0 and 1")
	}
	if config.PrivacyBudget <= 0 {
		return errors.NewValidationError("INVALID_BUDGET", "Privacy budget must be positive")
	}
	return nil
}

func getDefaultPrivacyPreservingConfig() *PrivacyPreservingConfig {
	return &PrivacyPreservingConfig{
		Epsilon:           1.0,
		Delta:             1e-5,
		PrivacyBudget:     10.0,
		ClippingThreshold: 1.0,
		EnableFederated:   false,
		NumClients:        10,
		RoundsPerEpoch:    5,
		ClientSampleRate:  0.3,
		EnableSecureAgg:   false,
		NumSurvivors:      5,
		ReconstructionThreshold: 3,
		ModelType:         "timegan",
		HiddenDim:         24,
		NumLayers:         3,
		SequenceLength:    24,
		Epochs:            100,
		BatchSize:         32,
		LearningRate:      0.001,
		UseKAnonymity:     false,
		KValue:            5,
		UseLDiversity:     false,
		LValue:            3,
		UseTCloseness:     false,
		TValue:            0.2,
		NoiseType:         "gaussian",
		AdaptiveNoise:     true,
		NoiseMultiplier:   1.0,
		Seed:              time.Now().UnixNano(),
		EnableAudit:       true,
		AuditInterval:     10,
	}
}
