package validation

import (
	"context"
	"math"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/constants"
	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
)

// PrivacyValidator implements privacy validation methods
type PrivacyValidator struct {
	logger           *logrus.Logger
	config           *PrivacyValidatorConfig
	supportedMetrics []string
	supportedTechniques []string
}

// PrivacyValidatorConfig contains configuration for privacy validation
type PrivacyValidatorConfig struct {
	QualityThreshold          float64 `json:"quality_threshold"`
	ReidentificationThreshold float64 `json:"reidentification_threshold"`
	KAnonymityK               int     `json:"k_anonymity_k"`
	LDiversityL               int     `json:"l_diversity_l"`
	TClosenessT               float64 `json:"t_closeness_t"`
	DifferentialPrivacyEpsilon float64 `json:"dp_epsilon"`
	DifferentialPrivacyDelta   float64 `json:"dp_delta"`
}

// NewPrivacyValidator creates a new privacy validator
func NewPrivacyValidator(config *PrivacyValidatorConfig, logger *logrus.Logger) interfaces.PrivacyValidator {
	if config == nil {
		config = getDefaultPrivacyValidatorConfig()
	}
	
	if logger == nil {
		logger = logrus.New()
	}

	return &PrivacyValidator{
		logger: logger,
		config: config,
		supportedMetrics: []string{
			constants.MetricPrivacy,
			constants.MetricBasic,
		},
		supportedTechniques: []string{
			constants.PrivacyDifferentialPrivacy,
			constants.PrivacyKAnonymity,
			constants.PrivacyLDiversity,
			constants.PrivacyTCloseness,
			constants.PrivacyDataMasking,
		},
	}
}

// GetType returns the validator type
func (v *PrivacyValidator) GetType() string {
	return constants.ValidatorTypePrivacy
}

// GetName returns a human-readable name for the validator
func (v *PrivacyValidator) GetName() string {
	return "Privacy Validator"
}

// GetDescription returns a description of the validator
func (v *PrivacyValidator) GetDescription() string {
	return "Validates privacy protection of synthetic data using various privacy metrics and techniques"
}

// GetSupportedMetrics returns the metrics this validator supports
func (v *PrivacyValidator) GetSupportedMetrics() []string {
	return v.supportedMetrics
}

// ValidateParameters validates the validation parameters
func (v *PrivacyValidator) ValidateParameters(params models.ValidationParameters) error {
	if params.SyntheticData == nil {
		return errors.NewValidationError("MISSING_SYNTHETIC", "Synthetic data is required for privacy validation")
	}

	if params.ReferenceData == nil {
		return errors.NewValidationError("MISSING_REFERENCE", "Reference data is required for privacy validation")
	}

	if len(params.SyntheticData.DataPoints) == 0 {
		return errors.NewValidationError("EMPTY_SYNTHETIC", "Synthetic data cannot be empty")
	}

	if len(params.ReferenceData.DataPoints) == 0 {
		return errors.NewValidationError("EMPTY_REFERENCE", "Reference data cannot be empty")
	}

	return nil
}

// Validate validates synthetic data privacy protection
func (v *PrivacyValidator) Validate(ctx context.Context, req *models.ValidationRequest) (*models.ValidationResult, error) {
	if req == nil {
		return nil, errors.NewValidationError("INVALID_REQUEST", "Validation request is required")
	}

	if err := v.ValidateParameters(req.Parameters); err != nil {
		return nil, err
	}

	v.logger.WithFields(logrus.Fields{
		"request_id":       req.ID,
		"synthetic_points": len(req.Parameters.SyntheticData.DataPoints),
		"reference_points": len(req.Parameters.ReferenceData.DataPoints),
	}).Info("Starting privacy validation")

	start := time.Now()

	// Validate privacy
	privacyParams := models.PrivacyParameters{
		Technique: constants.PrivacyDifferentialPrivacy,
		Epsilon:   v.config.DifferentialPrivacyEpsilon,
		Delta:     v.config.DifferentialPrivacyDelta,
		K:         v.config.KAnonymityK,
		L:         v.config.LDiversityL,
		T:         v.config.TClosenessT,
	}

	privacyResult, err := v.ValidatePrivacy(ctx, req.Parameters.SyntheticData, req.Parameters.ReferenceData, privacyParams)
	if err != nil {
		return nil, err
	}

	// Calculate overall quality score
	qualityScore := v.calculatePrivacyQualityScore(privacyResult)

	// Determine validation status
	status := "passed"
	if qualityScore < v.config.QualityThreshold {
		status = "failed"
	}

	duration := time.Since(start)

	// Convert privacy result to metrics map
	metrics := v.convertPrivacyResultToMap(privacyResult)

	result := &models.ValidationResult{
		ID:           req.ID,
		Status:       status,
		QualityScore: qualityScore,
		ValidationDetails: &models.ValidationDetails{
			TotalTests:  len(v.supportedTechniques),
			PassedTests: v.countPassedPrivacyTests(privacyResult),
			FailedTests: len(v.supportedTechniques) - v.countPassedPrivacyTests(privacyResult),
		},
		Metrics:       metrics,
		Duration:      duration,
		ValidatedAt:   time.Now(),
		ValidatorType: v.GetType(),
		Metadata: map[string]interface{}{
			"reidentification_threshold": v.config.ReidentificationThreshold,
			"supported_techniques":       v.supportedTechniques,
			"dp_epsilon":                v.config.DifferentialPrivacyEpsilon,
			"dp_delta":                  v.config.DifferentialPrivacyDelta,
		},
	}

	v.logger.WithFields(logrus.Fields{
		"request_id":    req.ID,
		"status":        status,
		"quality_score": qualityScore,
		"duration":      duration,
	}).Info("Privacy validation completed")

	return result, nil
}

// ValidatePrivacy validates privacy protection of synthetic data
func (v *PrivacyValidator) ValidatePrivacy(ctx context.Context, synthetic, reference *models.TimeSeries, params models.PrivacyParameters) (*models.PrivacyValidationResult, error) {
	result := &models.PrivacyValidationResult{
		ValidatedAt: time.Now(),
		Technique:   params.Technique,
	}

	// Calculate privacy risk
	privacyRisk, err := v.CalculatePrivacyRisk(ctx, synthetic, reference)
	if err != nil {
		return nil, err
	}
	result.PrivacyRisk = privacyRisk

	// Check reidentification risk
	reidentificationRisk, err := v.CheckReidentificationRisk(ctx, synthetic, reference)
	if err != nil {
		return nil, err
	}
	result.ReidentificationRisk = reidentificationRisk

	// Validate specific privacy technique
	switch params.Technique {
	case constants.PrivacyDifferentialPrivacy:
		result.DifferentialPrivacy = v.validateDifferentialPrivacy(synthetic, reference, params.Epsilon, params.Delta)
	case constants.PrivacyKAnonymity:
		result.KAnonymity = v.validateKAnonymity(synthetic, reference, params.K)
	case constants.PrivacyLDiversity:
		result.LDiversity = v.validateLDiversity(synthetic, reference, params.L)
	case constants.PrivacyTCloseness:
		result.TCloseness = v.validateTCloseness(synthetic, reference, params.T)
	}

	// Calculate overall privacy score
	result.OverallScore = v.calculateOverallPrivacyScore(result)

	return result, nil
}

// CalculatePrivacyRisk calculates privacy risk metrics
func (v *PrivacyValidator) CalculatePrivacyRisk(ctx context.Context, synthetic, reference *models.TimeSeries) (*models.PrivacyRisk, error) {
	risk := &models.PrivacyRisk{
		CalculatedAt: time.Now(),
	}

	synValues := extractValues(synthetic)
	refValues := extractValues(reference)

	// Distance-based privacy risk
	risk.DistanceBasedRisk = v.calculateDistanceBasedRisk(synValues, refValues)

	// Membership inference risk
	risk.MembershipInferenceRisk = v.calculateMembershipInferenceRisk(synValues, refValues)

	// Attribute inference risk
	risk.AttributeInferenceRisk = v.calculateAttributeInferenceRisk(synValues, refValues)

	// Overall risk level
	risk.OverallRisk = (risk.DistanceBasedRisk + risk.MembershipInferenceRisk + risk.AttributeInferenceRisk) / 3.0

	// Risk level classification
	if risk.OverallRisk < 0.3 {
		risk.RiskLevel = "low"
	} else if risk.OverallRisk < 0.7 {
		risk.RiskLevel = "medium"
	} else {
		risk.RiskLevel = "high"
	}

	return risk, nil
}

// CheckReidentificationRisk checks for reidentification risks
func (v *PrivacyValidator) CheckReidentificationRisk(ctx context.Context, synthetic, reference *models.TimeSeries) (*models.ReidentificationRisk, error) {
	risk := &models.ReidentificationRisk{
		CheckedAt: time.Now(),
	}

	synValues := extractValues(synthetic)
	refValues := extractValues(reference)

	// Simple nearest neighbor-based reidentification risk
	risk.NearestNeighborRisk = v.calculateNearestNeighborRisk(synValues, refValues)

	// Statistical similarity risk
	risk.StatisticalSimilarityRisk = v.calculateStatisticalSimilarityRisk(synValues, refValues)

	// Pattern matching risk
	risk.PatternMatchingRisk = v.calculatePatternMatchingRisk(synValues, refValues)

	// Overall reidentification risk
	risk.OverallRisk = math.Max(risk.NearestNeighborRisk, math.Max(risk.StatisticalSimilarityRisk, risk.PatternMatchingRisk))

	// Risk assessment
	risk.RiskAssessment = "acceptable"
	if risk.OverallRisk > v.config.ReidentificationThreshold {
		risk.RiskAssessment = "unacceptable"
	}

	return risk, nil
}

// GetSupportedPrivacyTechniques returns supported privacy techniques
func (v *PrivacyValidator) GetSupportedPrivacyTechniques() []string {
	return v.supportedTechniques
}

// GetDefaultParameters returns default parameters for this validator
func (v *PrivacyValidator) GetDefaultParameters() models.ValidationParameters {
	return models.ValidationParameters{
		Metrics: []string{constants.MetricPrivacy},
		Timeout: "30s",
	}
}

// Close cleans up resources
func (v *PrivacyValidator) Close() error {
	v.logger.Info("Closing privacy validator")
	return nil
}

// Private methods

func (v *PrivacyValidator) validateDifferentialPrivacy(synthetic, reference *models.TimeSeries, epsilon, delta float64) *models.DifferentialPrivacyResult {
	result := &models.DifferentialPrivacyResult{
		Epsilon:   epsilon,
		Delta:     delta,
		Satisfied: false,
	}

	// Simplified differential privacy check
	// In practice, this would involve more sophisticated analysis
	synValues := extractValues(synthetic)
	refValues := extractValues(reference)

	// Calculate sensitivity
	sensitivity := v.calculateSensitivity(synValues, refValues)
	result.Sensitivity = sensitivity

	// Check if privacy budget is satisfied
	noiseMagnitude := v.estimateNoiseLevel(synValues, refValues)
	expectedNoise := sensitivity / epsilon

	result.Satisfied = noiseMagnitude >= expectedNoise*0.8 // Allow some tolerance
	result.PrivacyLoss = math.Max(0, expectedNoise-noiseMagnitude)

	return result
}

func (v *PrivacyValidator) validateKAnonymity(synthetic, reference *models.TimeSeries, k int) *models.KAnonymityResult {
	result := &models.KAnonymityResult{
		K:         k,
		Satisfied: false,
	}

	// Simplified k-anonymity check for time series
	// This would be more complex for real quasi-identifiers
	synValues := extractValues(synthetic)

	// Count unique value groups (simplified)
	valueGroups := v.groupSimilarValues(synValues, 0.01) // 1% tolerance
	
	minGroupSize := len(synValues)
	for _, group := range valueGroups {
		if len(group) < minGroupSize {
			minGroupSize = len(group)
		}
	}

	result.MinGroupSize = minGroupSize
	result.Satisfied = minGroupSize >= k

	return result
}

func (v *PrivacyValidator) validateLDiversity(synthetic, reference *models.TimeSeries, l int) *models.LDiversityResult {
	result := &models.LDiversityResult{
		L:         l,
		Satisfied: false,
	}

	// Simplified l-diversity check
	synValues := extractValues(synthetic)
	
	// For time series, we consider value ranges as sensitive attributes
	ranges := v.categorizeValueRanges(synValues, 5) // 5 categories
	
	minDiversity := len(ranges)
	for _, rangeCount := range ranges {
		if rangeCount > 0 && rangeCount < minDiversity {
			minDiversity = rangeCount
		}
	}

	result.MinDiversity = minDiversity
	result.Satisfied = minDiversity >= l

	return result
}

func (v *PrivacyValidator) validateTCloseness(synthetic, reference *models.TimeSeries, t float64) *models.TClosenessResult {
	result := &models.TClosenessResult{
		T:         t,
		Satisfied: false,
	}

	// Simplified t-closeness check
	synValues := extractValues(synthetic)
	refValues := extractValues(reference)

	// Calculate distribution distance
	distance := v.calculateDistributionDistance(synValues, refValues)
	result.Distance = distance
	result.Satisfied = distance <= t

	return result
}

func (v *PrivacyValidator) calculateDistanceBasedRisk(synthetic, reference []float64) float64 {
	// Calculate minimum distance between synthetic and reference points
	minDistance := math.Inf(1)
	
	for _, synVal := range synthetic {
		for _, refVal := range reference {
			distance := math.Abs(synVal - refVal)
			if distance < minDistance {
				minDistance = distance
			}
		}
	}

	// Convert distance to risk (closer = higher risk)
	if math.IsInf(minDistance, 1) {
		return 0.0
	}
	
	// Normalize to 0-1 range (simplified)
	return 1.0 / (1.0 + minDistance)
}

func (v *PrivacyValidator) calculateMembershipInferenceRisk(synthetic, reference []float64) float64 {
	// Simplified membership inference risk
	// Count how many synthetic values are very close to reference values
	closeMatches := 0
	threshold := 0.01 // 1% tolerance

	for _, synVal := range synthetic {
		for _, refVal := range reference {
			if math.Abs(synVal-refVal)/math.Max(math.Abs(synVal), math.Abs(refVal)) < threshold {
				closeMatches++
				break
			}
		}
	}

	return float64(closeMatches) / float64(len(synthetic))
}

func (v *PrivacyValidator) calculateAttributeInferenceRisk(synthetic, reference []float64) float64 {
	// Simplified attribute inference risk based on correlation
	if len(synthetic) != len(reference) {
		return 0.5 // Medium risk if lengths differ
	}

	// Calculate correlation as a proxy for attribute inference risk
	correlation := v.calculateCorrelation(synthetic, reference)
	return math.Abs(correlation)
}

func (v *PrivacyValidator) calculateNearestNeighborRisk(synthetic, reference []float64) float64 {
	// Find average distance to nearest neighbors
	totalDistance := 0.0
	
	for _, synVal := range synthetic {
		minDistance := math.Inf(1)
		for _, refVal := range reference {
			distance := math.Abs(synVal - refVal)
			if distance < minDistance {
				minDistance = distance
			}
		}
		if !math.IsInf(minDistance, 1) {
			totalDistance += minDistance
		}
	}

	avgDistance := totalDistance / float64(len(synthetic))
	
	// Convert to risk (smaller distance = higher risk)
	return 1.0 / (1.0 + avgDistance)
}

func (v *PrivacyValidator) calculateStatisticalSimilarityRisk(synthetic, reference []float64) float64 {
	// Risk based on statistical similarity
	synMean := v.calculateMean(synthetic)
	refMean := v.calculateMean(reference)
	synStd := v.calculateStdDev(synthetic)
	refStd := v.calculateStdDev(reference)

	meanDiff := math.Abs(synMean - refMean)
	stdDiff := math.Abs(synStd - refStd)

	// Higher similarity = higher risk
	similarity := 1.0 / (1.0 + meanDiff + stdDiff)
	return similarity
}

func (v *PrivacyValidator) calculatePatternMatchingRisk(synthetic, reference []float64) float64 {
	// Simplified pattern matching risk
	// Check for subsequence matches
	matchingSubsequences := 0
	windowSize := 3

	if len(synthetic) < windowSize || len(reference) < windowSize {
		return 0.0
	}

	for i := 0; i <= len(synthetic)-windowSize; i++ {
		synPattern := synthetic[i : i+windowSize]
		
		for j := 0; j <= len(reference)-windowSize; j++ {
			refPattern := reference[j : j+windowSize]
			
			if v.patternsMatch(synPattern, refPattern, 0.05) { // 5% tolerance
				matchingSubsequences++
				break
			}
		}
	}

	totalPatterns := len(synthetic) - windowSize + 1
	if totalPatterns == 0 {
		return 0.0
	}

	return float64(matchingSubsequences) / float64(totalPatterns)
}

// Helper methods

func (v *PrivacyValidator) calculateSensitivity(synthetic, reference []float64) float64 {
	// Global sensitivity estimation
	synRange := v.calculateRange(synthetic)
	refRange := v.calculateRange(reference)
	return math.Max(synRange, refRange)
}

func (v *PrivacyValidator) estimateNoiseLevel(synthetic, reference []float64) float64 {
	// Estimate noise level by comparing distributions
	synMean := v.calculateMean(synthetic)
	refMean := v.calculateMean(reference)
	return math.Abs(synMean - refMean)
}

func (v *PrivacyValidator) groupSimilarValues(values []float64, tolerance float64) [][]float64 {
	var groups [][]float64
	used := make([]bool, len(values))

	for i, val := range values {
		if used[i] {
			continue
		}

		group := []float64{val}
		used[i] = true

		for j := i + 1; j < len(values); j++ {
			if !used[j] && math.Abs(values[j]-val) <= tolerance {
				group = append(group, values[j])
				used[j] = true
			}
		}

		groups = append(groups, group)
	}

	return groups
}

func (v *PrivacyValidator) categorizeValueRanges(values []float64, numCategories int) []int {
	if len(values) == 0 {
		return make([]int, numCategories)
	}

	min, max := values[0], values[0]
	for _, val := range values {
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}

	rangeSize := (max - min) / float64(numCategories)
	if rangeSize == 0 {
		rangeSize = 1
	}

	categories := make([]int, numCategories)
	for _, val := range values {
		category := int((val - min) / rangeSize)
		if category >= numCategories {
			category = numCategories - 1
		}
		if category < 0 {
			category = 0
		}
		categories[category]++
	}

	return categories
}

func (v *PrivacyValidator) calculateDistributionDistance(values1, values2 []float64) float64 {
	// Simple Wasserstein-like distance
	mean1 := v.calculateMean(values1)
	mean2 := v.calculateMean(values2)
	std1 := v.calculateStdDev(values1)
	std2 := v.calculateStdDev(values2)

	return math.Abs(mean1-mean2) + math.Abs(std1-std2)
}

func (v *PrivacyValidator) calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, val := range values {
		sum += val
	}
	return sum / float64(len(values))
}

func (v *PrivacyValidator) calculateStdDev(values []float64) float64 {
	if len(values) <= 1 {
		return 0
	}
	
	mean := v.calculateMean(values)
	sumSq := 0.0
	for _, val := range values {
		diff := val - mean
		sumSq += diff * diff
	}
	return math.Sqrt(sumSq / float64(len(values)-1))
}

func (v *PrivacyValidator) calculateRange(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	min, max := values[0], values[0]
	for _, val := range values {
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}
	return max - min
}

func (v *PrivacyValidator) calculateCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0
	}

	meanX := v.calculateMean(x)
	meanY := v.calculateMean(y)

	var numSum, denX, denY float64
	for i := 0; i < len(x); i++ {
		diffX := x[i] - meanX
		diffY := y[i] - meanY
		numSum += diffX * diffY
		denX += diffX * diffX
		denY += diffY * diffY
	}

	if denX == 0 || denY == 0 {
		return 0
	}

	return numSum / math.Sqrt(denX*denY)
}

func (v *PrivacyValidator) patternsMatch(pattern1, pattern2 []float64, tolerance float64) bool {
	if len(pattern1) != len(pattern2) {
		return false
	}

	for i := range pattern1 {
		if math.Abs(pattern1[i]-pattern2[i]) > tolerance {
			return false
		}
	}

	return true
}

func (v *PrivacyValidator) calculatePrivacyQualityScore(result *models.PrivacyValidationResult) float64 {
	score := 0.0
	components := 0

	// Privacy risk component (inverted - lower risk = higher score)
	if result.PrivacyRisk != nil {
		score += (1.0 - result.PrivacyRisk.OverallRisk) * 0.4
		components++
	}

	// Reidentification risk component (inverted)
	if result.ReidentificationRisk != nil {
		score += (1.0 - result.ReidentificationRisk.OverallRisk) * 0.3
		components++
	}

	// Technique-specific validation
	if result.DifferentialPrivacy != nil {
		if result.DifferentialPrivacy.Satisfied {
			score += 0.3
		}
		components++
	} else if result.KAnonymity != nil {
		if result.KAnonymity.Satisfied {
			score += 0.3
		}
		components++
	}

	if components == 0 {
		return 0.0
	}

	return score / float64(components)
}

func (v *PrivacyValidator) countPassedPrivacyTests(result *models.PrivacyValidationResult) int {
	passed := 0
	threshold := v.config.QualityThreshold

	if result.PrivacyRisk != nil && (1.0-result.PrivacyRisk.OverallRisk) >= threshold {
		passed++
	}

	if result.ReidentificationRisk != nil && (1.0-result.ReidentificationRisk.OverallRisk) >= threshold {
		passed++
	}

	if result.DifferentialPrivacy != nil && result.DifferentialPrivacy.Satisfied {
		passed++
	}

	if result.KAnonymity != nil && result.KAnonymity.Satisfied {
		passed++
	}

	if result.LDiversity != nil && result.LDiversity.Satisfied {
		passed++
	}

	if result.TCloseness != nil && result.TCloseness.Satisfied {
		passed++
	}

	return passed
}

func (v *PrivacyValidator) convertPrivacyResultToMap(result *models.PrivacyValidationResult) map[string]interface{} {
	metrics := make(map[string]interface{})

	metrics["technique"] = result.Technique
	metrics["overall_score"] = result.OverallScore

	if result.PrivacyRisk != nil {
		metrics["privacy_risk"] = result.PrivacyRisk.OverallRisk
		metrics["risk_level"] = result.PrivacyRisk.RiskLevel
		metrics["distance_based_risk"] = result.PrivacyRisk.DistanceBasedRisk
		metrics["membership_inference_risk"] = result.PrivacyRisk.MembershipInferenceRisk
		metrics["attribute_inference_risk"] = result.PrivacyRisk.AttributeInferenceRisk
	}

	if result.ReidentificationRisk != nil {
		metrics["reidentification_risk"] = result.ReidentificationRisk.OverallRisk
		metrics["risk_assessment"] = result.ReidentificationRisk.RiskAssessment
		metrics["nearest_neighbor_risk"] = result.ReidentificationRisk.NearestNeighborRisk
		metrics["statistical_similarity_risk"] = result.ReidentificationRisk.StatisticalSimilarityRisk
		metrics["pattern_matching_risk"] = result.ReidentificationRisk.PatternMatchingRisk
	}

	if result.DifferentialPrivacy != nil {
		metrics["dp_satisfied"] = result.DifferentialPrivacy.Satisfied
		metrics["dp_epsilon"] = result.DifferentialPrivacy.Epsilon
		metrics["dp_delta"] = result.DifferentialPrivacy.Delta
		metrics["dp_sensitivity"] = result.DifferentialPrivacy.Sensitivity
		metrics["dp_privacy_loss"] = result.DifferentialPrivacy.PrivacyLoss
	}

	if result.KAnonymity != nil {
		metrics["k_anonymity_satisfied"] = result.KAnonymity.Satisfied
		metrics["k_anonymity_k"] = result.KAnonymity.K
		metrics["k_anonymity_min_group_size"] = result.KAnonymity.MinGroupSize
	}

	return metrics
}

func getDefaultPrivacyValidatorConfig() *PrivacyValidatorConfig {
	return &PrivacyValidatorConfig{
		QualityThreshold:           0.7,
		ReidentificationThreshold:  0.3,
		KAnonymityK:               5,
		LDiversityL:               3,
		TClosenessT:               0.2,
		DifferentialPrivacyEpsilon: 1.0,  // Default epsilon for differential privacy
		DifferentialPrivacyDelta:   1e-5, // Default delta for differential privacy
	}
}