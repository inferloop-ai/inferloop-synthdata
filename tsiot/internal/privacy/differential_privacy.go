package privacy

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/sirupsen/logrus"
	"gonum.org/v1/gonum/stat"

	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/models"
)

// DifferentialPrivacyEngine provides differential privacy mechanisms for time series data
type DifferentialPrivacyEngine struct {
	logger       *logrus.Logger
	budgetManager *PrivacyBudgetManager
	mechanisms   map[string]PrivacyMechanism
	randSource   *rand.Rand
}

// PrivacyMechanism defines interface for different DP mechanisms
type PrivacyMechanism interface {
	GetName() string
	GetDescription() string
	AddNoise(ctx context.Context, value float64, sensitivity float64, epsilon float64) (float64, error)
	AddNoiseToSeries(ctx context.Context, data []float64, sensitivity float64, epsilon float64) ([]float64, error)
	CalculateSensitivity(data []float64, queryType QueryType) float64
	ValidateParameters(epsilon, delta float64) error
}

// QueryType represents different types of queries on time series data
type QueryType string

const (
	QueryTypeSum       QueryType = "sum"
	QueryTypeMean      QueryType = "mean"
	QueryTypeVariance  QueryType = "variance"
	QueryTypeMedian    QueryType = "median"
	QueryTypeQuantile  QueryType = "quantile"
	QueryTypeCount     QueryType = "count"
	QueryTypeRange     QueryType = "range"
	QueryTypeAutocorr  QueryType = "autocorrelation"
	QueryTypeFFT       QueryType = "fft"
	QueryTypeHistogram QueryType = "histogram"
)

// PrivacyConfig contains configuration for differential privacy
type PrivacyConfig struct {
	GlobalEpsilon    float64           `json:"global_epsilon"`    // Total privacy budget
	GlobalDelta      float64           `json:"global_delta"`      // Failure probability
	Mechanism        string            `json:"mechanism"`         // "laplace", "gaussian", "exponential"
	BudgetAllocation map[string]float64 `json:"budget_allocation"` // Per-query budget allocation
	TimeHorizon      time.Duration     `json:"time_horizon"`      // Privacy budget refresh period
	Composition      string            `json:"composition"`       // "basic", "advanced", "rdp"
	Clamping         *ClampingConfig   `json:"clamping"`         // Value clamping configuration
	PostProcessing   *PostProcessConfig `json:"post_processing"`  // Post-processing configuration
}

// ClampingConfig defines value clamping parameters
type ClampingConfig struct {
	Enabled   bool    `json:"enabled"`
	LowerBound float64 `json:"lower_bound"`
	UpperBound float64 `json:"upper_bound"`
	Strategy   string  `json:"strategy"` // "clip", "reject", "wrap"
}

// PostProcessConfig defines post-processing parameters
type PostProcessConfig struct {
	Enabled         bool    `json:"enabled"`
	SmoothingWindow int     `json:"smoothing_window"`
	OutlierRemoval  bool    `json:"outlier_removal"`
	ConsistencyCheck bool   `json:"consistency_check"`
	MinValue        *float64 `json:"min_value,omitempty"`
	MaxValue        *float64 `json:"max_value,omitempty"`
}

// PrivacyResult contains the result of applying differential privacy
type PrivacyResult struct {
	OriginalData     []float64              `json:"original_data,omitempty"`
	PrivatizedData   []float64              `json:"privatized_data"`
	EpsilonUsed      float64                `json:"epsilon_used"`
	DeltaUsed        float64                `json:"delta_used"`
	Mechanism        string                 `json:"mechanism"`
	Sensitivity      float64                `json:"sensitivity"`
	NoiseScale       float64                `json:"noise_scale"`
	UtilityLoss      float64                `json:"utility_loss"`
	PrivacyGuarantee string                 `json:"privacy_guarantee"`
	Metadata         map[string]interface{} `json:"metadata"`
	ProcessedAt      time.Time              `json:"processed_at"`
}

// PrivacyAnalysis provides analysis of privacy-utility tradeoffs
type PrivacyAnalysis struct {
	EpsilonValues    []float64 `json:"epsilon_values"`
	UtilityScores    []float64 `json:"utility_scores"`
	PrivacyRisk      float64   `json:"privacy_risk"`
	RecommendedEps   float64   `json:"recommended_epsilon"`
	DataDependence   float64   `json:"data_dependence"`
	SensitivityAnalysis *SensitivityAnalysis `json:"sensitivity_analysis"`
}

// SensitivityAnalysis contains detailed sensitivity analysis
type SensitivityAnalysis struct {
	GlobalSensitivity float64            `json:"global_sensitivity"`
	LocalSensitivity  []float64          `json:"local_sensitivity"`
	SmoothSensitivity float64            `json:"smooth_sensitivity"`
	QuerySensitivities map[QueryType]float64 `json:"query_sensitivities"`
}

// NewDifferentialPrivacyEngine creates a new DP engine
func NewDifferentialPrivacyEngine(config *PrivacyConfig, logger *logrus.Logger) (*DifferentialPrivacyEngine, error) {
	if config == nil {
		config = getDefaultPrivacyConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	// Create privacy budget manager
	budgetManager, err := NewPrivacyBudgetManager(config.GlobalEpsilon, config.GlobalDelta, config.TimeHorizon)
	if err != nil {
		return nil, fmt.Errorf("failed to create privacy budget manager: %w", err)
	}

	engine := &DifferentialPrivacyEngine{
		logger:        logger,
		budgetManager: budgetManager,
		mechanisms:    make(map[string]PrivacyMechanism),
		randSource:    rand.New(rand.NewSource(time.Now().UnixNano())),
	}

	// Register available mechanisms
	engine.registerMechanisms(config)

	return engine, nil
}

// ApplyDifferentialPrivacy applies DP to time series data
func (dpe *DifferentialPrivacyEngine) ApplyDifferentialPrivacy(
	ctx context.Context,
	data []float64,
	epsilon, delta float64,
	queryType QueryType,
	mechanismName string,
) (*PrivacyResult, error) {

	if len(data) == 0 {
		return nil, errors.NewValidationError("EMPTY_DATA", "Input data cannot be empty")
	}

	// Check budget availability
	if !dpe.budgetManager.CanSpend(epsilon, delta) {
		return nil, errors.NewPrivacyError("BUDGET_EXHAUSTED", "Insufficient privacy budget")
	}

	// Get mechanism
	mechanism, exists := dpe.mechanisms[mechanismName]
	if !exists {
		return nil, errors.NewValidationError("UNKNOWN_MECHANISM", fmt.Sprintf("Unknown mechanism: %s", mechanismName))
	}

	// Validate parameters
	if err := mechanism.ValidateParameters(epsilon, delta); err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeValidation, "INVALID_PARAMETERS", "Invalid privacy parameters")
	}

	dpe.logger.WithFields(logrus.Fields{
		"data_points": len(data),
		"epsilon":     epsilon,
		"delta":       delta,
		"mechanism":   mechanismName,
		"query_type":  string(queryType),
	}).Info("Applying differential privacy")

	start := time.Now()

	// Calculate sensitivity
	sensitivity := mechanism.CalculateSensitivity(data, queryType)

	// Apply mechanism
	privatizedData, err := mechanism.AddNoiseToSeries(ctx, data, sensitivity, epsilon)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeProcessing, "MECHANISM_ERROR", "Failed to apply privacy mechanism")
	}

	// Spend budget
	if err := dpe.budgetManager.Spend(epsilon, delta); err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeProcessing, "BUDGET_ERROR", "Failed to spend privacy budget")
	}

	// Calculate utility loss
	utilityLoss := dpe.calculateUtilityLoss(data, privatizedData)

	// Create result
	result := &PrivacyResult{
		PrivatizedData:   privatizedData,
		EpsilonUsed:      epsilon,
		DeltaUsed:        delta,
		Mechanism:        mechanismName,
		Sensitivity:      sensitivity,
		UtilityLoss:      utilityLoss,
		PrivacyGuarantee: dpe.generatePrivacyGuarantee(epsilon, delta, mechanismName),
		Metadata: map[string]interface{}{
			"query_type":       string(queryType),
			"data_points":      len(data),
			"processing_time":  time.Since(start).String(),
			"budget_remaining": dpe.budgetManager.GetRemainingBudget(),
		},
		ProcessedAt: time.Now(),
	}

	// Calculate noise scale for metadata
	if gaussianMech, ok := mechanism.(*GaussianMechanism); ok {
		result.NoiseScale = gaussianMech.CalculateNoiseScale(sensitivity, epsilon, delta)
	} else if laplaceMech, ok := mechanism.(*LaplaceMechanism); ok {
		result.NoiseScale = laplaceMech.CalculateNoiseScale(sensitivity, epsilon)
	}

	dpe.logger.WithFields(logrus.Fields{
		"epsilon_used":   epsilon,
		"sensitivity":    sensitivity,
		"utility_loss":   utilityLoss,
		"processing_time": time.Since(start),
	}).Info("Differential privacy applied successfully")

	return result, nil
}

// AnalyzePrivacyUtilityTradeoff analyzes privacy-utility tradeoffs
func (dpe *DifferentialPrivacyEngine) AnalyzePrivacyUtilityTradeoff(
	ctx context.Context,
	data []float64,
	queryType QueryType,
	mechanismName string,
	epsilonRange []float64,
) (*PrivacyAnalysis, error) {

	mechanism, exists := dpe.mechanisms[mechanismName]
	if !exists {
		return nil, errors.NewValidationError("UNKNOWN_MECHANISM", fmt.Sprintf("Unknown mechanism: %s", mechanismName))
	}

	analysis := &PrivacyAnalysis{
		EpsilonValues: epsilonRange,
		UtilityScores: make([]float64, len(epsilonRange)),
	}

	// Calculate sensitivity
	sensitivity := mechanism.CalculateSensitivity(data, queryType)

	// Analyze for each epsilon value
	for i, epsilon := range epsilonRange {
		// Apply mechanism multiple times and average utility
		var totalUtility float64
		numTrials := 10

		for trial := 0; trial < numTrials; trial++ {
			privatizedData, err := mechanism.AddNoiseToSeries(ctx, data, sensitivity, epsilon)
			if err != nil {
				continue
			}

			utilityLoss := dpe.calculateUtilityLoss(data, privatizedData)
			utility := 1.0 - utilityLoss
			totalUtility += utility
		}

		analysis.UtilityScores[i] = totalUtility / float64(numTrials)
	}

	// Calculate additional analysis metrics
	analysis.PrivacyRisk = dpe.estimatePrivacyRisk(data, epsilonRange[len(epsilonRange)-1])
	analysis.RecommendedEps = dpe.recommendEpsilon(analysis.EpsilonValues, analysis.UtilityScores)
	analysis.DataDependence = dpe.calculateDataDependence(data)

	// Perform sensitivity analysis
	analysis.SensitivityAnalysis = dpe.performSensitivityAnalysis(data, queryType)

	return analysis, nil
}

// EstimatePrivacyRisk estimates privacy risk for given parameters
func (dpe *DifferentialPrivacyEngine) EstimatePrivacyRisk(epsilon, delta float64, dataSize int) float64 {
	// Privacy risk increases with epsilon and decreases with delta
	// Risk also depends on data size
	baseRisk := epsilon / (1.0 + math.Exp(-epsilon))
	deltaFactor := 1.0 + delta
	sizeFactor := math.Log(float64(dataSize)) / 10.0

	risk := baseRisk * deltaFactor * sizeFactor
	return math.Min(1.0, math.Max(0.0, risk))
}

// RegisterMechanism registers a new privacy mechanism
func (dpe *DifferentialPrivacyEngine) RegisterMechanism(mechanism PrivacyMechanism) {
	dpe.mechanisms[mechanism.GetName()] = mechanism
}

// GetAvailableMechanisms returns list of available mechanisms
func (dpe *DifferentialPrivacyEngine) GetAvailableMechanisms() []string {
	mechanisms := make([]string, 0, len(dpe.mechanisms))
	for name := range dpe.mechanisms {
		mechanisms = append(mechanisms, name)
	}
	sort.Strings(mechanisms)
	return mechanisms
}

// GetBudgetStatus returns current budget status
func (dpe *DifferentialPrivacyEngine) GetBudgetStatus() *BudgetStatus {
	return dpe.budgetManager.GetStatus()
}

// ResetBudget resets the privacy budget
func (dpe *DifferentialPrivacyEngine) ResetBudget() error {
	return dpe.budgetManager.Reset()
}

// Helper methods

func (dpe *DifferentialPrivacyEngine) registerMechanisms(config *PrivacyConfig) {
	// Register Laplace mechanism
	laplace := NewLaplaceMechanism(dpe.randSource, config.Clamping, config.PostProcessing)
	dpe.mechanisms[laplace.GetName()] = laplace

	// Register Gaussian mechanism
	gaussian := NewGaussianMechanism(dpe.randSource, config.Clamping, config.PostProcessing)
	dpe.mechanisms[gaussian.GetName()] = gaussian

	// Register Exponential mechanism
	exponential := NewExponentialMechanism(dpe.randSource, config.Clamping, config.PostProcessing)
	dpe.mechanisms[exponential.GetName()] = exponential
}

func (dpe *DifferentialPrivacyEngine) calculateUtilityLoss(original, privatized []float64) float64 {
	if len(original) != len(privatized) {
		return 1.0 // Maximum utility loss
	}

	// Calculate multiple utility metrics
	mse := dpe.meanSquaredError(original, privatized)
	mae := dpe.meanAbsoluteError(original, privatized)
	correlation := dpe.correlation(original, privatized)

	// Normalize MSE by data variance
	variance := stat.Variance(original, nil)
	normalizedMSE := mse / (variance + 1e-10)

	// Combine metrics (correlation is similarity, others are distances)
	utilityLoss := 0.4*normalizedMSE + 0.3*mae + 0.3*(1.0-math.Abs(correlation))

	return math.Min(1.0, math.Max(0.0, utilityLoss))
}

func (dpe *DifferentialPrivacyEngine) meanSquaredError(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}

	var sum float64
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return sum / float64(len(a))
}

func (dpe *DifferentialPrivacyEngine) meanAbsoluteError(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}

	var sum float64
	for i := 0; i < len(a); i++ {
		sum += math.Abs(a[i] - b[i])
	}

	return sum / float64(len(a))
}

func (dpe *DifferentialPrivacyEngine) correlation(a, b []float64) float64 {
	if len(a) != len(b) || len(a) < 2 {
		return 0.0
	}

	meanA := stat.Mean(a, nil)
	meanB := stat.Mean(b, nil)

	var numerator, sumSquareA, sumSquareB float64
	for i := 0; i < len(a); i++ {
		diffA := a[i] - meanA
		diffB := b[i] - meanB
		numerator += diffA * diffB
		sumSquareA += diffA * diffA
		sumSquareB += diffB * diffB
	}

	denominator := math.Sqrt(sumSquareA * sumSquareB)
	if denominator == 0 {
		return 0.0
	}

	return numerator / denominator
}

func (dpe *DifferentialPrivacyEngine) generatePrivacyGuarantee(epsilon, delta float64, mechanism string) string {
	if delta == 0 {
		return fmt.Sprintf("Pure (µ=%g)-differential privacy using %s mechanism", epsilon, mechanism)
	}
	return fmt.Sprintf("Approximate (µ=%g, ´=%g)-differential privacy using %s mechanism", epsilon, delta, mechanism)
}

func (dpe *DifferentialPrivacyEngine) estimatePrivacyRisk(data []float64, epsilon float64) float64 {
	// Simplified privacy risk estimation
	// Higher epsilon means higher risk
	// Risk also depends on data characteristics
	dataRange := dpe.calculateDataRange(data)
	normalizedEpsilon := epsilon / (1.0 + dataRange/1000.0)

	risk := 1.0 - math.Exp(-normalizedEpsilon)
	return math.Min(1.0, math.Max(0.0, risk))
}

func (dpe *DifferentialPrivacyEngine) recommendEpsilon(epsilons, utilities []float64) float64 {
	if len(epsilons) != len(utilities) || len(epsilons) == 0 {
		return 1.0 // Default recommendation
	}

	// Find epsilon that maximizes utility while keeping privacy reasonable
	bestEpsilon := epsilons[0]
	bestScore := 0.0

	for i, eps := range epsilons {
		utility := utilities[i]
		privacyPenalty := eps / 10.0 // Penalize high epsilon
		score := utility - privacyPenalty

		if score > bestScore {
			bestScore = score
			bestEpsilon = eps
		}
	}

	return bestEpsilon
}

func (dpe *DifferentialPrivacyEngine) calculateDataDependence(data []float64) float64 {
	// Measure how much consecutive values depend on each other
	if len(data) < 2 {
		return 0.0
	}

	var sumSquaredDiffs float64
	var sumSquaredValues float64
	mean := stat.Mean(data, nil)

	for i := 1; i < len(data); i++ {
		diff := data[i] - data[i-1]
		sumSquaredDiffs += diff * diff

		val := data[i] - mean
		sumSquaredValues += val * val
	}

	if sumSquaredValues == 0 {
		return 0.0
	}

	// Dependence is inverse of relative variation
	relativeVariation := sumSquaredDiffs / sumSquaredValues
	dependence := 1.0 / (1.0 + relativeVariation)

	return math.Min(1.0, math.Max(0.0, dependence))
}

func (dpe *DifferentialPrivacyEngine) performSensitivityAnalysis(data []float64, queryType QueryType) *SensitivityAnalysis {
	analysis := &SensitivityAnalysis{
		QuerySensitivities: make(map[QueryType]float64),
	}

	// Calculate global sensitivity (worst-case)
	analysis.GlobalSensitivity = dpe.calculateGlobalSensitivity(data, queryType)

	// Calculate local sensitivity for each data point
	analysis.LocalSensitivity = dpe.calculateLocalSensitivity(data, queryType)

	// Calculate smooth sensitivity (adaptive)
	analysis.SmoothSensitivity = dpe.calculateSmoothSensitivity(data, queryType)

	// Calculate sensitivity for different query types
	queryTypes := []QueryType{QueryTypeSum, QueryTypeMean, QueryTypeVariance, QueryTypeMedian, QueryTypeCount}
	for _, qt := range queryTypes {
		analysis.QuerySensitivities[qt] = dpe.calculateGlobalSensitivity(data, qt)
	}

	return analysis
}

func (dpe *DifferentialPrivacyEngine) calculateGlobalSensitivity(data []float64, queryType QueryType) float64 {
	// Global sensitivity is the maximum change in query result
	// when one data point is added/removed/changed
	switch queryType {
	case QueryTypeSum:
		return dpe.calculateDataRange(data)
	case QueryTypeMean:
		return dpe.calculateDataRange(data) / float64(len(data))
	case QueryTypeCount:
		return 1.0
	case QueryTypeMedian:
		return dpe.calculateDataRange(data) / 2.0
	case QueryTypeVariance:
		return dpe.calculateDataRange(data) * dpe.calculateDataRange(data) / float64(len(data))
	default:
		return 1.0 // Conservative default
	}
}

func (dpe *DifferentialPrivacyEngine) calculateLocalSensitivity(data []float64, queryType QueryType) []float64 {
	n := len(data)
	localSens := make([]float64, n)

	for i := 0; i < n; i++ {
		// Calculate sensitivity if we remove data point i
		modifiedData := make([]float64, 0, n-1)
		for j, val := range data {
			if j != i {
				modifiedData = append(modifiedData, val)
			}
		}

		originalResult := dpe.calculateQueryResult(data, queryType)
		modifiedResult := dpe.calculateQueryResult(modifiedData, queryType)
		localSens[i] = math.Abs(originalResult - modifiedResult)
	}

	return localSens
}

func (dpe *DifferentialPrivacyEngine) calculateSmoothSensitivity(data []float64, queryType QueryType) float64 {
	// Smooth sensitivity is a data-dependent but differentially private sensitivity measure
	localSens := dpe.calculateLocalSensitivity(data, queryType)
	
	// Return maximum local sensitivity (simplified)
	maxSens := 0.0
	for _, sens := range localSens {
		if sens > maxSens {
			maxSens = sens
		}
	}

	return maxSens
}

func (dpe *DifferentialPrivacyEngine) calculateQueryResult(data []float64, queryType QueryType) float64 {
	if len(data) == 0 {
		return 0.0
	}

	switch queryType {
	case QueryTypeSum:
		var sum float64
		for _, val := range data {
			sum += val
		}
		return sum
	case QueryTypeMean:
		return stat.Mean(data, nil)
	case QueryTypeCount:
		return float64(len(data))
	case QueryTypeMedian:
		sorted := make([]float64, len(data))
		copy(sorted, data)
		sort.Float64s(sorted)
		n := len(sorted)
		if n%2 == 0 {
			return (sorted[n/2-1] + sorted[n/2]) / 2.0
		}
		return sorted[n/2]
	case QueryTypeVariance:
		return stat.Variance(data, nil)
	default:
		return 0.0
	}
}

func (dpe *DifferentialPrivacyEngine) calculateDataRange(data []float64) float64 {
	if len(data) == 0 {
		return 0.0
	}

	min, max := data[0], data[0]
	for _, val := range data[1:] {
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}

	return max - min
}

func getDefaultPrivacyConfig() *PrivacyConfig {
	return &PrivacyConfig{
		GlobalEpsilon: 1.0,
		GlobalDelta:   1e-5,
		Mechanism:     "gaussian",
		BudgetAllocation: map[string]float64{
			"generation": 0.7,
			"validation": 0.2,
			"analysis":   0.1,
		},
		TimeHorizon: 24 * time.Hour,
		Composition: "basic",
		Clamping: &ClampingConfig{
			Enabled:    true,
			LowerBound: -1000.0,
			UpperBound: 1000.0,
			Strategy:   "clip",
		},
		PostProcessing: &PostProcessConfig{
			Enabled:         true,
			SmoothingWindow: 3,
			OutlierRemoval:  true,
			ConsistencyCheck: true,
		},
	}
}

// PrivacyError represents a privacy-related error
type PrivacyError struct {
	Code    string
	Message string
}

func (e *PrivacyError) Error() string {
	return fmt.Sprintf("Privacy Error [%s]: %s", e.Code, e.Message)
}

func NewPrivacyError(code, message string) *PrivacyError {
	return &PrivacyError{Code: code, Message: message}
}