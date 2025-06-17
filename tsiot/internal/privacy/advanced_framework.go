package privacy

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/models"
)

// AdvancedPrivacyFramework provides comprehensive privacy-preserving techniques
type AdvancedPrivacyFramework struct {
	logger           *logrus.Logger
	config           *PrivacyFrameworkConfig
	kAnonymizer      *KAnonymizer
	lDiversifier     *LDiversifier
	tClosenessAgent  *TClosenessAgent
	privacyAnalyzer  *PrivacyAnalyzer
	policyEngine     *PrivacyPolicyEngine
	auditLogger      *PrivacyAuditLogger
	metrics          *PrivacyMetrics
	mu               sync.RWMutex
}

// PrivacyFrameworkConfig configures the privacy framework
type PrivacyFrameworkConfig struct {
	Enabled                   bool                      `json:"enabled"`
	DefaultPrivacyLevel       PrivacyLevel              `json:"default_privacy_level"`
	KAnonymityConfig          *KAnonymityConfig         `json:"k_anonymity_config"`
	LDiversityConfig          *LDiversityConfig         `json:"l_diversity_config"`
	TClosenessConfig          *TClosenessConfig         `json:"t_closeness_config"`
	DifferentialPrivacyConfig *DifferentialPrivacyConfig `json:"differential_privacy_config"`
	EnableAuditing            bool                      `json:"enable_auditing"`
	EnableMetrics             bool                      `json:"enable_metrics"`
	PolicyEnforcement         bool                      `json:"policy_enforcement"`
	DataRetentionDays         int                       `json:"data_retention_days"`
}

// PrivacyLevel defines different levels of privacy protection
type PrivacyLevel string

const (
	PrivacyLevelNone   PrivacyLevel = "none"
	PrivacyLevelLow    PrivacyLevel = "low"
	PrivacyLevelMedium PrivacyLevel = "medium"
	PrivacyLevelHigh   PrivacyLevel = "high"
	PrivacyLevelMaximum PrivacyLevel = "maximum"
)

// KAnonymityConfig configures k-anonymity
type KAnonymityConfig struct {
	K                     int      `json:"k"`
	QuasiIdentifiers      []string `json:"quasi_identifiers"`
	SensitiveAttributes   []string `json:"sensitive_attributes"`
	GeneralizationLevels  int      `json:"generalization_levels"`
	SuppressionThreshold  float64  `json:"suppression_threshold"`
	EnableGlobalization   bool     `json:"enable_globalization"`
}

// LDiversityConfig configures l-diversity
type LDiversityConfig struct {
	L                    int      `json:"l"`
	SensitiveAttributes  []string `json:"sensitive_attributes"`
	DiversityMethod      string   `json:"diversity_method"` // distinct, entropy, recursive
	EntropyThreshold     float64  `json:"entropy_threshold"`
	RecursiveFactor      float64  `json:"recursive_factor"`
}

// TClosenessConfig configures t-closeness
type TClosenessConfig struct {
	T                   float64  `json:"t"`
	SensitiveAttributes []string `json:"sensitive_attributes"`
	DistanceMetric      string   `json:"distance_metric"` // emd, kl_divergence, chi_square
	EnableNormalization bool     `json:"enable_normalization"`
}

// DifferentialPrivacyConfig configures differential privacy
type DifferentialPrivacyConfig struct {
	Epsilon         float64 `json:"epsilon"`
	Delta           float64 `json:"delta"`
	NoiseType       string  `json:"noise_type"` // laplace, gaussian, exponential
	ClippingBound   float64 `json:"clipping_bound"`
	SensitivityType string  `json:"sensitivity_type"` // global, local
}

// PrivacyRequest represents a request for privacy-preserving operations
type PrivacyRequest struct {
	ID               string                 `json:"id"`
	Data             []*models.TimeSeries   `json:"data"`
	PrivacyLevel     PrivacyLevel           `json:"privacy_level"`
	Requirements     *PrivacyRequirements   `json:"requirements"`
	Context          map[string]interface{} `json:"context"`
	Timestamp        time.Time              `json:"timestamp"`
}

// PrivacyRequirements specifies privacy requirements
type PrivacyRequirements struct {
	KAnonymity       *KAnonymityRequirement  `json:"k_anonymity,omitempty"`
	LDiversity       *LDiversityRequirement  `json:"l_diversity,omitempty"`
	TCloseness       *TClosenessRequirement  `json:"t_closeness,omitempty"`
	DifferentialPrivacy *DPRequirement       `json:"differential_privacy,omitempty"`
	CustomRequirements  map[string]interface{} `json:"custom_requirements,omitempty"`
}

// KAnonymityRequirement specifies k-anonymity requirements
type KAnonymityRequirement struct {
	K                int      `json:"k"`
	QuasiIdentifiers []string `json:"quasi_identifiers"`
}

// LDiversityRequirement specifies l-diversity requirements
type LDiversityRequirement struct {
	L                   int      `json:"l"`
	SensitiveAttributes []string `json:"sensitive_attributes"`
	Method              string   `json:"method"`
}

// TClosenessRequirement specifies t-closeness requirements
type TClosenessRequirement struct {
	T                   float64  `json:"t"`
	SensitiveAttributes []string `json:"sensitive_attributes"`
}

// DPRequirement specifies differential privacy requirements
type DPRequirement struct {
	Epsilon float64 `json:"epsilon"`
	Delta   float64 `json:"delta"`
}

// PrivacyResponse contains the result of privacy operations
type PrivacyResponse struct {
	RequestID        string               `json:"request_id"`
	ProtectedData    []*models.TimeSeries `json:"protected_data"`
	PrivacyGuarantees *PrivacyGuarantees  `json:"privacy_guarantees"`
	QualityMetrics   *DataQualityMetrics  `json:"quality_metrics"`
	ProcessingTime   time.Duration        `json:"processing_time"`
	Applied          []string             `json:"applied_techniques"`
}

// PrivacyGuarantees specifies what privacy guarantees are provided
type PrivacyGuarantees struct {
	KAnonymity           int     `json:"k_anonymity"`
	LDiversity           int     `json:"l_diversity"`
	TCloseness           float64 `json:"t_closeness"`
	DifferentialPrivacy  *DPGuarantee `json:"differential_privacy,omitempty"`
	OverallPrivacyLevel  PrivacyLevel `json:"overall_privacy_level"`
}

// DPGuarantee specifies differential privacy guarantees
type DPGuarantee struct {
	Epsilon float64 `json:"epsilon"`
	Delta   float64 `json:"delta"`
}

// DataQualityMetrics measures data quality after privacy operations
type DataQualityMetrics struct {
	DataLoss         float64 `json:"data_loss"`
	InformationLoss  float64 `json:"information_loss"`
	UtilityScore     float64 `json:"utility_score"`
	RecordsRemoved   int     `json:"records_removed"`
	RecordsModified  int     `json:"records_modified"`
	AttributesChanged int    `json:"attributes_changed"`
}

// NewAdvancedPrivacyFramework creates a new advanced privacy framework
func NewAdvancedPrivacyFramework(config *PrivacyFrameworkConfig, logger *logrus.Logger) (*AdvancedPrivacyFramework, error) {
	if config == nil {
		config = getDefaultPrivacyFrameworkConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	framework := &AdvancedPrivacyFramework{
		logger:  logger,
		config:  config,
		metrics: NewPrivacyMetrics(),
	}

	// Initialize components
	if config.KAnonymityConfig != nil {
		framework.kAnonymizer = NewKAnonymizer(config.KAnonymityConfig, logger)
	}

	if config.LDiversityConfig != nil {
		framework.lDiversifier = NewLDiversifier(config.LDiversityConfig, logger)
	}

	if config.TClosenessConfig != nil {
		framework.tClosenessAgent = NewTClosenessAgent(config.TClosenessConfig, logger)
	}

	framework.privacyAnalyzer = NewPrivacyAnalyzer(logger)
	framework.policyEngine = NewPrivacyPolicyEngine(logger)

	if config.EnableAuditing {
		framework.auditLogger = NewPrivacyAuditLogger(logger)
	}

	return framework, nil
}

// ApplyPrivacyProtection applies privacy protection to time series data
func (apf *AdvancedPrivacyFramework) ApplyPrivacyProtection(ctx context.Context, request *PrivacyRequest) (*PrivacyResponse, error) {
	if !apf.config.Enabled {
		return &PrivacyResponse{
			RequestID:     request.ID,
			ProtectedData: request.Data,
		}, nil
	}

	startTime := time.Now()

	// Validate request
	if err := apf.validateRequest(request); err != nil {
		return nil, fmt.Errorf("invalid privacy request: %w", err)
	}

	// Analyze privacy risks
	riskAssessment, err := apf.privacyAnalyzer.AnalyzeRisks(request.Data)
	if err != nil {
		return nil, fmt.Errorf("privacy risk analysis failed: %w", err)
	}

	// Determine required privacy techniques
	techniques := apf.determineTechniques(request, riskAssessment)

	// Apply privacy protection
	protectedData := apf.copyData(request.Data)
	appliedTechniques := make([]string, 0)
	guarantees := &PrivacyGuarantees{}
	qualityMetrics := &DataQualityMetrics{}

	// Apply k-anonymity
	if shouldApply(techniques, "k_anonymity") && apf.kAnonymizer != nil {
		result, err := apf.kAnonymizer.Anonymize(ctx, protectedData, request.Requirements.KAnonymity)
		if err != nil {
			return nil, fmt.Errorf("k-anonymity failed: %w", err)
		}
		protectedData = result.Data
		guarantees.KAnonymity = result.K
		qualityMetrics.RecordsRemoved += result.RecordsRemoved
		appliedTechniques = append(appliedTechniques, "k_anonymity")
	}

	// Apply l-diversity
	if shouldApply(techniques, "l_diversity") && apf.lDiversifier != nil {
		result, err := apf.lDiversifier.Diversify(ctx, protectedData, request.Requirements.LDiversity)
		if err != nil {
			return nil, fmt.Errorf("l-diversity failed: %w", err)
		}
		protectedData = result.Data
		guarantees.LDiversity = result.L
		qualityMetrics.RecordsModified += result.RecordsModified
		appliedTechniques = append(appliedTechniques, "l_diversity")
	}

	// Apply t-closeness
	if shouldApply(techniques, "t_closeness") && apf.tClosenessAgent != nil {
		result, err := apf.tClosenessAgent.ApplyTCloseness(ctx, protectedData, request.Requirements.TCloseness)
		if err != nil {
			return nil, fmt.Errorf("t-closeness failed: %w", err)
		}
		protectedData = result.Data
		guarantees.TCloseness = result.T
		qualityMetrics.AttributesChanged += result.AttributesChanged
		appliedTechniques = append(appliedTechniques, "t_closeness")
	}

	// Apply differential privacy
	if shouldApply(techniques, "differential_privacy") && request.Requirements.DifferentialPrivacy != nil {
		result, err := apf.applyDifferentialPrivacy(ctx, protectedData, request.Requirements.DifferentialPrivacy)
		if err != nil {
			return nil, fmt.Errorf("differential privacy failed: %w", err)
		}
		protectedData = result.Data
		guarantees.DifferentialPrivacy = &DPGuarantee{
			Epsilon: result.Epsilon,
			Delta:   result.Delta,
		}
		appliedTechniques = append(appliedTechniques, "differential_privacy")
	}

	// Calculate overall privacy level
	guarantees.OverallPrivacyLevel = apf.calculatePrivacyLevel(guarantees)

	// Calculate quality metrics
	qualityMetrics = apf.calculateQualityMetrics(request.Data, protectedData, qualityMetrics)

	// Update metrics
	apf.updateMetrics(appliedTechniques, qualityMetrics)

	// Audit log
	if apf.auditLogger != nil {
		apf.auditLogger.LogPrivacyOperation(&PrivacyAuditEntry{
			RequestID:         request.ID,
			Timestamp:         time.Now(),
			AppliedTechniques: appliedTechniques,
			PrivacyGuarantees: guarantees,
			QualityMetrics:    qualityMetrics,
		})
	}

	response := &PrivacyResponse{
		RequestID:         request.ID,
		ProtectedData:     protectedData,
		PrivacyGuarantees: guarantees,
		QualityMetrics:    qualityMetrics,
		ProcessingTime:    time.Since(startTime),
		Applied:           appliedTechniques,
	}

	apf.logger.WithFields(logrus.Fields{
		"request_id":        request.ID,
		"applied_techniques": appliedTechniques,
		"processing_time":   response.ProcessingTime,
		"privacy_level":     guarantees.OverallPrivacyLevel,
	}).Info("Privacy protection applied")

	return response, nil
}

// ValidatePrivacyRequirements validates that data meets privacy requirements
func (apf *AdvancedPrivacyFramework) ValidatePrivacyRequirements(ctx context.Context, data []*models.TimeSeries, requirements *PrivacyRequirements) (*PrivacyValidationResult, error) {
	result := &PrivacyValidationResult{
		Valid:      true,
		Violations: make([]PrivacyViolation, 0),
	}

	// Validate k-anonymity
	if requirements.KAnonymity != nil && apf.kAnonymizer != nil {
		kResult := apf.kAnonymizer.Validate(data, requirements.KAnonymity)
		if !kResult.Valid {
			result.Valid = false
			result.Violations = append(result.Violations, PrivacyViolation{
				Type:        "k_anonymity",
				Description: fmt.Sprintf("Data does not satisfy %d-anonymity", requirements.KAnonymity.K),
				Severity:    "high",
			})
		}
	}

	// Validate l-diversity
	if requirements.LDiversity != nil && apf.lDiversifier != nil {
		lResult := apf.lDiversifier.Validate(data, requirements.LDiversity)
		if !lResult.Valid {
			result.Valid = false
			result.Violations = append(result.Violations, PrivacyViolation{
				Type:        "l_diversity",
				Description: fmt.Sprintf("Data does not satisfy %d-diversity", requirements.LDiversity.L),
				Severity:    "high",
			})
		}
	}

	// Validate t-closeness
	if requirements.TCloseness != nil && apf.tClosenessAgent != nil {
		tResult := apf.tClosenessAgent.Validate(data, requirements.TCloseness)
		if !tResult.Valid {
			result.Valid = false
			result.Violations = append(result.Violations, PrivacyViolation{
				Type:        "t_closeness",
				Description: fmt.Sprintf("Data does not satisfy %.2f-closeness", requirements.TCloseness.T),
				Severity:    "medium",
			})
		}
	}

	return result, nil
}

// GetPrivacyMetrics returns privacy framework metrics
func (apf *AdvancedPrivacyFramework) GetPrivacyMetrics() *PrivacyMetrics {
	apf.mu.RLock()
	defer apf.mu.RUnlock()
	return apf.metrics
}

// Helper methods

func (apf *AdvancedPrivacyFramework) validateRequest(request *PrivacyRequest) error {
	if request.ID == "" {
		return fmt.Errorf("request ID is required")
	}

	if len(request.Data) == 0 {
		return fmt.Errorf("data is required")
	}

	if request.Requirements == nil {
		return fmt.Errorf("privacy requirements are required")
	}

	return nil
}

func (apf *AdvancedPrivacyFramework) determineTechniques(request *PrivacyRequest, riskAssessment *PrivacyRiskAssessment) map[string]bool {
	techniques := make(map[string]bool)

	// Determine based on privacy level
	switch request.PrivacyLevel {
	case PrivacyLevelHigh, PrivacyLevelMaximum:
		techniques["k_anonymity"] = true
		techniques["l_diversity"] = true
		techniques["t_closeness"] = true
		techniques["differential_privacy"] = true
	case PrivacyLevelMedium:
		techniques["k_anonymity"] = true
		techniques["l_diversity"] = true
	case PrivacyLevelLow:
		techniques["k_anonymity"] = true
	}

	// Override based on explicit requirements
	if request.Requirements.KAnonymity != nil {
		techniques["k_anonymity"] = true
	}
	if request.Requirements.LDiversity != nil {
		techniques["l_diversity"] = true
	}
	if request.Requirements.TCloseness != nil {
		techniques["t_closeness"] = true
	}
	if request.Requirements.DifferentialPrivacy != nil {
		techniques["differential_privacy"] = true
	}

	// Consider risk assessment
	if riskAssessment.ReidentificationRisk > 0.5 {
		techniques["k_anonymity"] = true
	}
	if riskAssessment.AttributeDisclosureRisk > 0.3 {
		techniques["l_diversity"] = true
	}
	if riskAssessment.MembershipInferenceRisk > 0.4 {
		techniques["differential_privacy"] = true
	}

	return techniques
}

func (apf *AdvancedPrivacyFramework) copyData(original []*models.TimeSeries) []*models.TimeSeries {
	copied := make([]*models.TimeSeries, len(original))
	for i, ts := range original {
		copied[i] = apf.copyTimeSeries(ts)
	}
	return copied
}

func (apf *AdvancedPrivacyFramework) copyTimeSeries(original *models.TimeSeries) *models.TimeSeries {
	copied := &models.TimeSeries{
		ID:          original.ID,
		Name:        original.Name,
		Description: original.Description,
		SensorType:  original.SensorType,
		Frequency:   original.Frequency,
		CreatedAt:   original.CreatedAt,
		UpdatedAt:   original.UpdatedAt,
		Tags:        make(map[string]string),
		Metadata:    make(map[string]interface{}),
		DataPoints:  make([]models.DataPoint, len(original.DataPoints)),
	}

	for k, v := range original.Tags {
		copied.Tags[k] = v
	}

	for k, v := range original.Metadata {
		copied.Metadata[k] = v
	}

	copy(copied.DataPoints, original.DataPoints)

	return copied
}

func (apf *AdvancedPrivacyFramework) calculatePrivacyLevel(guarantees *PrivacyGuarantees) PrivacyLevel {
	score := 0

	if guarantees.KAnonymity >= 10 {
		score += 2
	} else if guarantees.KAnonymity >= 5 {
		score += 1
	}

	if guarantees.LDiversity >= 5 {
		score += 2
	} else if guarantees.LDiversity >= 2 {
		score += 1
	}

	if guarantees.TCloseness <= 0.1 {
		score += 2
	} else if guarantees.TCloseness <= 0.3 {
		score += 1
	}

	if guarantees.DifferentialPrivacy != nil {
		if guarantees.DifferentialPrivacy.Epsilon <= 0.1 {
			score += 3
		} else if guarantees.DifferentialPrivacy.Epsilon <= 1.0 {
			score += 2
		} else {
			score += 1
		}
	}

	switch {
	case score >= 8:
		return PrivacyLevelMaximum
	case score >= 6:
		return PrivacyLevelHigh
	case score >= 4:
		return PrivacyLevelMedium
	case score >= 2:
		return PrivacyLevelLow
	default:
		return PrivacyLevelNone
	}
}

func (apf *AdvancedPrivacyFramework) calculateQualityMetrics(original, protected []*models.TimeSeries, existing *DataQualityMetrics) *DataQualityMetrics {
	if len(original) == 0 {
		return existing
	}

	originalPoints := 0
	protectedPoints := 0

	for _, ts := range original {
		originalPoints += len(ts.DataPoints)
	}

	for _, ts := range protected {
		protectedPoints += len(ts.DataPoints)
	}

	existing.DataLoss = float64(originalPoints-protectedPoints) / float64(originalPoints) * 100

	// Calculate utility score (simplified)
	if existing.DataLoss < 10 {
		existing.UtilityScore = 0.9
	} else if existing.DataLoss < 25 {
		existing.UtilityScore = 0.7
	} else if existing.DataLoss < 50 {
		existing.UtilityScore = 0.5
	} else {
		existing.UtilityScore = 0.3
	}

	// Calculate information loss (simplified)
	existing.InformationLoss = existing.DataLoss * 1.2

	return existing
}

func (apf *AdvancedPrivacyFramework) updateMetrics(techniques []string, quality *DataQualityMetrics) {
	apf.mu.Lock()
	defer apf.mu.Unlock()

	apf.metrics.TotalRequests++

	for _, technique := range techniques {
		switch technique {
		case "k_anonymity":
			apf.metrics.KAnonymityApplications++
		case "l_diversity":
			apf.metrics.LDiversityApplications++
		case "t_closeness":
			apf.metrics.TClosenessApplications++
		case "differential_privacy":
			apf.metrics.DifferentialPrivacyApplications++
		}
	}

	apf.metrics.AverageDataLoss = (apf.metrics.AverageDataLoss + quality.DataLoss) / 2
	apf.metrics.AverageUtilityScore = (apf.metrics.AverageUtilityScore + quality.UtilityScore) / 2
}

func (apf *AdvancedPrivacyFramework) applyDifferentialPrivacy(ctx context.Context, data []*models.TimeSeries, requirement *DPRequirement) (*DPResult, error) {
	// Simplified differential privacy implementation
	noiseGenerator := NewLaplaceNoise(requirement.Epsilon)

	for _, ts := range data {
		for i := range ts.DataPoints {
			noise := noiseGenerator.Generate()
			ts.DataPoints[i].Value += noise
		}
	}

	return &DPResult{
		Data:    data,
		Epsilon: requirement.Epsilon,
		Delta:   requirement.Delta,
	}, nil
}

func shouldApply(techniques map[string]bool, technique string) bool {
	return techniques[technique]
}

func getDefaultPrivacyFrameworkConfig() *PrivacyFrameworkConfig {
	return &PrivacyFrameworkConfig{
		Enabled:             true,
		DefaultPrivacyLevel: PrivacyLevelMedium,
		KAnonymityConfig: &KAnonymityConfig{
			K:                    5,
			QuasiIdentifiers:     []string{"sensor_type", "location"},
			SensitiveAttributes:  []string{"value"},
			GeneralizationLevels: 3,
			SuppressionThreshold: 0.1,
		},
		LDiversityConfig: &LDiversityConfig{
			L:                   2,
			SensitiveAttributes: []string{"value"},
			DiversityMethod:     "distinct",
		},
		TClosenessConfig: &TClosenessConfig{
			T:                   0.2,
			SensitiveAttributes: []string{"value"},
			DistanceMetric:      "emd",
		},
		DifferentialPrivacyConfig: &DifferentialPrivacyConfig{
			Epsilon:         1.0,
			Delta:           1e-5,
			NoiseType:       "laplace",
			ClippingBound:   1.0,
			SensitivityType: "global",
		},
		EnableAuditing:    true,
		EnableMetrics:     true,
		PolicyEnforcement: true,
		DataRetentionDays: 30,
	}
}

// Supporting types and structures

type PrivacyRiskAssessment struct {
	ReidentificationRisk      float64
	AttributeDisclosureRisk   float64
	MembershipInferenceRisk   float64
	OverallRiskScore         float64
}

type PrivacyValidationResult struct {
	Valid      bool
	Violations []PrivacyViolation
}

type PrivacyViolation struct {
	Type        string
	Description string
	Severity    string
}

type PrivacyMetrics struct {
	TotalRequests                    int64   `json:"total_requests"`
	KAnonymityApplications           int64   `json:"k_anonymity_applications"`
	LDiversityApplications           int64   `json:"l_diversity_applications"`
	TClosenessApplications           int64   `json:"t_closeness_applications"`
	DifferentialPrivacyApplications  int64   `json:"differential_privacy_applications"`
	AverageDataLoss                  float64 `json:"average_data_loss"`
	AverageUtilityScore              float64 `json:"average_utility_score"`
	AverageProcessingTime            float64 `json:"average_processing_time"`
}

type PrivacyAuditEntry struct {
	RequestID         string
	Timestamp         time.Time
	AppliedTechniques []string
	PrivacyGuarantees *PrivacyGuarantees
	QualityMetrics    *DataQualityMetrics
}

type DPResult struct {
	Data    []*models.TimeSeries
	Epsilon float64
	Delta   float64
}

func NewPrivacyMetrics() *PrivacyMetrics {
	return &PrivacyMetrics{}
}

// LaplaceNoise generates Laplace noise for differential privacy
type LaplaceNoise struct {
	scale  float64
	random *rand.Rand
}

func NewLaplaceNoise(epsilon float64) *LaplaceNoise {
	return &LaplaceNoise{
		scale:  1.0 / epsilon,
		random: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (ln *LaplaceNoise) Generate() float64 {
	u := ln.random.Float64() - 0.5
	return -ln.scale * math.Copysign(math.Log(1-2*math.Abs(u)), u)
}