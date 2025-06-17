package metrics

import (
	"math"
	"sort"

	"gonum.org/v1/gonum/stat"

	"github.com/inferloop/tsiot/internal/validation/tests"
	"github.com/inferloop/tsiot/pkg/models"
)

// QualityMetrics provides comprehensive quality assessment for synthetic data
type QualityMetrics struct {
	testSuite *tests.StatisticalTestSuite
}

// NewQualityMetrics creates a new quality metrics calculator
func NewQualityMetrics(significanceLevel float64) *QualityMetrics {
	return &QualityMetrics{
		testSuite: tests.NewStatisticalTestSuite(significanceLevel),
	}
}

// QualityReport contains comprehensive quality assessment results
type QualityReport struct {
	OverallScore       float64                    `json:"overall_score"`       // 0-1, higher is better
	StatisticalTests   []*tests.StatisticalTestResult `json:"statistical_tests"`
	DistributionMetrics *DistributionMetrics      `json:"distribution_metrics"`
	CorrelationMetrics  *CorrelationMetrics       `json:"correlation_metrics"`
	TemporalMetrics     *TemporalMetrics          `json:"temporal_metrics"`
	NoveltyMetrics      *NoveltyMetrics           `json:"novelty_metrics"`
	FidelityMetrics     *FidelityMetrics          `json:"fidelity_metrics"`
	PrivacyMetrics      *PrivacyMetrics           `json:"privacy_metrics,omitempty"`
	Summary             *QualitySummary           `json:"summary"`
}

// DistributionMetrics measures distributional similarity
type DistributionMetrics struct {
	MeanAbsoluteError    float64 `json:"mean_absolute_error"`
	RootMeanSquareError  float64 `json:"root_mean_square_error"`
	WassersteinDistance  float64 `json:"wasserstein_distance"`
	KLDivergence         float64 `json:"kl_divergence"`
	JSDistance           float64 `json:"js_distance"`
	MomentComparison     *MomentComparison `json:"moment_comparison"`
	DistributionSimilarity float64 `json:"distribution_similarity"` // 0-1 score
}

// MomentComparison compares statistical moments
type MomentComparison struct {
	MeanDifference     float64 `json:"mean_difference"`
	VarianceDifference float64 `json:"variance_difference"`
	SkewnessDifference float64 `json:"skewness_difference"`
	KurtosisDifference float64 `json:"kurtosis_difference"`
	MomentScore        float64 `json:"moment_score"` // 0-1 score
}

// CorrelationMetrics measures correlation preservation
type CorrelationMetrics struct {
	AutocorrelationSimilarity  float64                `json:"autocorrelation_similarity"`
	CrossCorrelationSimilarity float64                `json:"cross_correlation_similarity"`
	CorrelationMatrixDistance  float64                `json:"correlation_matrix_distance"`
	LaggedCorrelations         map[int]float64        `json:"lagged_correlations"`
	CorrelationScore           float64                `json:"correlation_score"` // 0-1 score
}

// TemporalMetrics measures temporal pattern preservation
type TemporalMetrics struct {
	TrendSimilarity        float64                 `json:"trend_similarity"`
	SeasonalitySimilarity  float64                 `json:"seasonality_similarity"`
	CyclicalSimilarity     float64                 `json:"cyclical_similarity"`
	SpectralSimilarity     float64                 `json:"spectral_similarity"`
	FrequencyDomainMetrics *FrequencyDomainMetrics `json:"frequency_domain_metrics"`
	TemporalScore          float64                 `json:"temporal_score"` // 0-1 score
}

// FrequencyDomainMetrics analyzes frequency domain characteristics
type FrequencyDomainMetrics struct {
	PowerSpectralDensity []float64 `json:"power_spectral_density"`
	DominantFrequencies  []float64 `json:"dominant_frequencies"`
	SpectralEntropy      float64   `json:"spectral_entropy"`
	BandpowerRatios      map[string]float64 `json:"bandpower_ratios"`
}

// NoveltyMetrics measures how novel/diverse the synthetic data is
type NoveltyMetrics struct {
	DiversityIndex     float64 `json:"diversity_index"`
	NoveltyScore       float64 `json:"novelty_score"`
	CoverageScore      float64 `json:"coverage_score"`
	OverfittingRisk    float64 `json:"overfitting_risk"`
	BoundaryPreservation float64 `json:"boundary_preservation"`
}

// FidelityMetrics measures how faithful the synthetic data is to the original
type FidelityMetrics struct {
	GlobalFidelity     float64 `json:"global_fidelity"`
	LocalFidelity      float64 `json:"local_fidelity"`
	StructuralFidelity float64 `json:"structural_fidelity"`
	PatternFidelity    float64 `json:"pattern_fidelity"`
	FidelityScore      float64 `json:"fidelity_score"` // 0-1 score
}

// PrivacyMetrics measures privacy preservation (optional)
type PrivacyMetrics struct {
	MembershipInferenceRisk float64 `json:"membership_inference_risk"`
	AttributeInferenceRisk  float64 `json:"attribute_inference_risk"`
	DifferentialPrivacyScore float64 `json:"differential_privacy_score"`
	PrivacyScore            float64 `json:"privacy_score"` // 0-1 score
}

// QualitySummary provides high-level quality assessment
type QualitySummary struct {
	QualityGrade       string                 `json:"quality_grade"`       // A, B, C, D, F
	StrengthAreas      []string               `json:"strength_areas"`
	ImprovementAreas   []string               `json:"improvement_areas"`
	Recommendations    []string               `json:"recommendations"`
	KeyFindings        []string               `json:"key_findings"`
	QualityBreakdown   map[string]float64     `json:"quality_breakdown"`
}

// EvaluateQuality performs comprehensive quality evaluation
func (qm *QualityMetrics) EvaluateQuality(original, synthetic *models.TimeSeries) (*QualityReport, error) {
	if original == nil || synthetic == nil {
		return nil, NewValidationError("INVALID_INPUT", "Both original and synthetic time series are required")
	}

	if len(original.DataPoints) == 0 || len(synthetic.DataPoints) == 0 {
		return nil, NewValidationError("EMPTY_DATA", "Time series must contain data points")
	}

	// Extract values
	originalValues := extractValues(original.DataPoints)
	syntheticValues := extractValues(synthetic.DataPoints)

	report := &QualityReport{}

	// 1. Statistical Tests
	report.StatisticalTests = qm.runStatisticalTests(originalValues, syntheticValues)

	// 2. Distribution Metrics
	report.DistributionMetrics = qm.calculateDistributionMetrics(originalValues, syntheticValues)

	// 3. Correlation Metrics
	report.CorrelationMetrics = qm.calculateCorrelationMetrics(originalValues, syntheticValues)

	// 4. Temporal Metrics
	report.TemporalMetrics = qm.calculateTemporalMetrics(original, synthetic)

	// 5. Novelty Metrics
	report.NoveltyMetrics = qm.calculateNoveltyMetrics(originalValues, syntheticValues)

	// 6. Fidelity Metrics
	report.FidelityMetrics = qm.calculateFidelityMetrics(originalValues, syntheticValues)

	// 7. Calculate overall score
	report.OverallScore = qm.calculateOverallScore(report)

	// 8. Generate summary
	report.Summary = qm.generateSummary(report)

	return report, nil
}

// runStatisticalTests performs various statistical tests
func (qm *QualityMetrics) runStatisticalTests(original, synthetic []float64) []*tests.StatisticalTestResult {
	var testResults []*tests.StatisticalTestResult

	// Test normality of synthetic data
	testResults = append(testResults, qm.testSuite.AndersonDarlingTest(synthetic))
	testResults = append(testResults, qm.testSuite.ShapiroWilkTest(synthetic))
	testResults = append(testResults, qm.testSuite.JarqueBeraTest(synthetic))

	// Test if original and synthetic come from same distribution
	testResults = append(testResults, qm.testSuite.TwoSampleKSTest(original, synthetic))

	// Test randomness
	testResults = append(testResults, qm.testSuite.RunsTest(synthetic))

	// Test autocorrelation
	testResults = append(testResults, qm.testSuite.LjungBoxTest(synthetic, 10))

	return testResults
}

// calculateDistributionMetrics computes distributional similarity metrics
func (qm *QualityMetrics) calculateDistributionMetrics(original, synthetic []float64) *DistributionMetrics {
	// Basic error metrics
	mae := qm.meanAbsoluteError(original, synthetic)
	rmse := qm.rootMeanSquareError(original, synthetic)

	// Distribution distances
	wasserstein := qm.wassersteinDistance(original, synthetic)
	klDiv := qm.klDivergence(original, synthetic)
	jsDistance := qm.jensenShannonDistance(original, synthetic)

	// Moment comparison
	momentComp := qm.compareMoments(original, synthetic)

	// Overall distribution similarity score
	distSim := qm.calculateDistributionSimilarity(mae, rmse, wasserstein, klDiv)

	return &DistributionMetrics{
		MeanAbsoluteError:      mae,
		RootMeanSquareError:    rmse,
		WassersteinDistance:    wasserstein,
		KLDivergence:          klDiv,
		JSDistance:            jsDistance,
		MomentComparison:      momentComp,
		DistributionSimilarity: distSim,
	}
}

// calculateCorrelationMetrics computes correlation preservation metrics
func (qm *QualityMetrics) calculateCorrelationMetrics(original, synthetic []float64) *CorrelationMetrics {
	// Autocorrelation similarity
	autoSim := qm.autocorrelationSimilarity(original, synthetic)

	// Lagged correlations
	laggedCorr := qm.calculateLaggedCorrelations(original, synthetic, 10)

	// Overall correlation score
	corrScore := autoSim

	return &CorrelationMetrics{
		AutocorrelationSimilarity: autoSim,
		LaggedCorrelations:       laggedCorr,
		CorrelationScore:         corrScore,
	}
}

// calculateTemporalMetrics computes temporal pattern preservation metrics
func (qm *QualityMetrics) calculateTemporalMetrics(original, synthetic *models.TimeSeries) *TemporalMetrics {
	originalValues := extractValues(original.DataPoints)
	syntheticValues := extractValues(synthetic.DataPoints)

	// Trend similarity
	trendSim := qm.trendSimilarity(originalValues, syntheticValues)

	// Spectral similarity
	spectralSim := qm.spectralSimilarity(originalValues, syntheticValues)

	// Frequency domain analysis
	freqMetrics := qm.analyzeFrequencyDomain(originalValues, syntheticValues)

	// Overall temporal score
	temporalScore := (trendSim + spectralSim) / 2.0

	return &TemporalMetrics{
		TrendSimilarity:        trendSim,
		SpectralSimilarity:     spectralSim,
		FrequencyDomainMetrics: freqMetrics,
		TemporalScore:          temporalScore,
	}
}

// calculateNoveltyMetrics computes novelty and diversity metrics
func (qm *QualityMetrics) calculateNoveltyMetrics(original, synthetic []float64) *NoveltyMetrics {
	diversityIndex := qm.calculateDiversityIndex(synthetic)
	noveltyScore := qm.calculateNoveltyScore(original, synthetic)
	coverageScore := qm.calculateCoverageScore(original, synthetic)
	overfittingRisk := qm.assessOverfittingRisk(original, synthetic)
	boundaryPreservation := qm.calculateBoundaryPreservation(original, synthetic)

	return &NoveltyMetrics{
		DiversityIndex:       diversityIndex,
		NoveltyScore:        noveltyScore,
		CoverageScore:       coverageScore,
		OverfittingRisk:     overfittingRisk,
		BoundaryPreservation: boundaryPreservation,
	}
}

// calculateFidelityMetrics computes fidelity metrics
func (qm *QualityMetrics) calculateFidelityMetrics(original, synthetic []float64) *FidelityMetrics {
	globalFidelity := qm.calculateGlobalFidelity(original, synthetic)
	localFidelity := qm.calculateLocalFidelity(original, synthetic)
	structuralFidelity := qm.calculateStructuralFidelity(original, synthetic)
	patternFidelity := qm.calculatePatternFidelity(original, synthetic)

	fidelityScore := (globalFidelity + localFidelity + structuralFidelity + patternFidelity) / 4.0

	return &FidelityMetrics{
		GlobalFidelity:     globalFidelity,
		LocalFidelity:      localFidelity,
		StructuralFidelity: structuralFidelity,
		PatternFidelity:    patternFidelity,
		FidelityScore:      fidelityScore,
	}
}

// Utility methods for metric calculations

func (qm *QualityMetrics) meanAbsoluteError(original, synthetic []float64) float64 {
	if len(original) != len(synthetic) {
		// Handle different lengths by using minimum length
		minLen := min(len(original), len(synthetic))
		original = original[:minLen]
		synthetic = synthetic[:minLen]
	}

	var sum float64
	for i := 0; i < len(original); i++ {
		sum += math.Abs(original[i] - synthetic[i])
	}
	return sum / float64(len(original))
}

func (qm *QualityMetrics) rootMeanSquareError(original, synthetic []float64) float64 {
	if len(original) != len(synthetic) {
		minLen := min(len(original), len(synthetic))
		original = original[:minLen]
		synthetic = synthetic[:minLen]
	}

	var sum float64
	for i := 0; i < len(original); i++ {
		diff := original[i] - synthetic[i]
		sum += diff * diff
	}
	return math.Sqrt(sum / float64(len(original)))
}

func (qm *QualityMetrics) wassersteinDistance(original, synthetic []float64) float64 {
	// Sort both arrays
	sortedOrig := make([]float64, len(original))
	sortedSyn := make([]float64, len(synthetic))
	copy(sortedOrig, original)
	copy(sortedSyn, synthetic)
	sort.Float64s(sortedOrig)
	sort.Float64s(sortedSyn)

	// Resample to same size if needed
	if len(sortedOrig) != len(sortedSyn) {
		minLen := min(len(sortedOrig), len(sortedSyn))
		sortedOrig = qm.resample(sortedOrig, minLen)
		sortedSyn = qm.resample(sortedSyn, minLen)
	}

	// Calculate Wasserstein distance (1st order)
	var sum float64
	for i := 0; i < len(sortedOrig); i++ {
		sum += math.Abs(sortedOrig[i] - sortedSyn[i])
	}

	return sum / float64(len(sortedOrig))
}

func (qm *QualityMetrics) klDivergence(original, synthetic []float64) float64 {
	// Create histograms
	bins := 50
	origHist := qm.createHistogram(original, bins)
	synHist := qm.createHistogram(synthetic, bins)

	// Calculate KL divergence
	var kl float64
	for i := 0; i < bins; i++ {
		if origHist[i] > 0 && synHist[i] > 0 {
			kl += origHist[i] * math.Log(origHist[i]/synHist[i])
		}
	}

	return kl
}

func (qm *QualityMetrics) jensenShannonDistance(original, synthetic []float64) float64 {
	// Create histograms
	bins := 50
	p := qm.createHistogram(original, bins)
	q := qm.createHistogram(synthetic, bins)

	// Calculate JS divergence
	var js float64
	for i := 0; i < bins; i++ {
		if p[i] > 0 || q[i] > 0 {
			m := (p[i] + q[i]) / 2.0
			if p[i] > 0 && m > 0 {
				js += 0.5 * p[i] * math.Log(p[i]/m)
			}
			if q[i] > 0 && m > 0 {
				js += 0.5 * q[i] * math.Log(q[i]/m)
			}
		}
	}

	return math.Sqrt(js) // JS distance is square root of JS divergence
}

func (qm *QualityMetrics) compareMoments(original, synthetic []float64) *MomentComparison {
	// Calculate moments for both series
	origMean := stat.Mean(original, nil)
	synMean := stat.Mean(synthetic, nil)
	origVar := stat.Variance(original, nil)
	synVar := stat.Variance(synthetic, nil)

	// Calculate higher moments (simplified)
	origSkew := qm.calculateSkewness(original)
	synSkew := qm.calculateSkewness(synthetic)
	origKurt := qm.calculateKurtosis(original)
	synKurt := qm.calculateKurtosis(synthetic)

	// Calculate differences
	meanDiff := math.Abs(origMean - synMean)
	varDiff := math.Abs(origVar - synVar)
	skewDiff := math.Abs(origSkew - synSkew)
	kurtDiff := math.Abs(origKurt - synKurt)

	// Calculate moment score (closer to 1 is better)
	momentScore := 1.0 / (1.0 + meanDiff + varDiff + skewDiff + kurtDiff)

	return &MomentComparison{
		MeanDifference:     meanDiff,
		VarianceDifference: varDiff,
		SkewnessDifference: skewDiff,
		KurtosisDifference: kurtDiff,
		MomentScore:        momentScore,
	}
}

func (qm *QualityMetrics) autocorrelationSimilarity(original, synthetic []float64) float64 {
	maxLag := min(20, min(len(original), len(synthetic))/4)
	
	origACF := qm.calculateACF(original, maxLag)
	synACF := qm.calculateACF(synthetic, maxLag)

	// Calculate correlation between ACF functions
	var sumXY, sumX, sumY, sumX2, sumY2 float64
	n := float64(len(origACF))

	for i := 0; i < len(origACF); i++ {
		x := origACF[i]
		y := synACF[i]
		sumXY += x * y
		sumX += x
		sumY += y
		sumX2 += x * x
		sumY2 += y * y
	}

	numerator := n*sumXY - sumX*sumY
	denominator := math.Sqrt((n*sumX2 - sumX*sumX) * (n*sumY2 - sumY*sumY))

	if denominator == 0 {
		return 0
	}

	correlation := numerator / denominator
	return math.Abs(correlation) // Use absolute value for similarity
}

// Additional helper methods...

func (qm *QualityMetrics) calculateOverallScore(report *QualityReport) float64 {
	// Weight different components
	weights := map[string]float64{
		"distribution": 0.25,
		"correlation":  0.20,
		"temporal":     0.20,
		"fidelity":     0.20,
		"novelty":      0.15,
	}

	score := 0.0
	score += weights["distribution"] * report.DistributionMetrics.DistributionSimilarity
	score += weights["correlation"] * report.CorrelationMetrics.CorrelationScore
	score += weights["temporal"] * report.TemporalMetrics.TemporalScore
	score += weights["fidelity"] * report.FidelityMetrics.FidelityScore
	score += weights["novelty"] * (1.0 - report.NoveltyMetrics.OverfittingRisk) // Lower overfitting risk is better

	return math.Max(0, math.Min(1, score))
}

func (qm *QualityMetrics) generateSummary(report *QualityReport) *QualitySummary {
	// Determine quality grade
	grade := qm.determineQualityGrade(report.OverallScore)
	
	// Identify strengths and areas for improvement
	strengths, improvements := qm.analyzeStrengthsAndWeaknesses(report)
	
	// Generate recommendations
	recommendations := qm.generateRecommendations(report)
	
	// Create quality breakdown
	breakdown := map[string]float64{
		"Distribution Quality": report.DistributionMetrics.DistributionSimilarity,
		"Correlation Quality":  report.CorrelationMetrics.CorrelationScore,
		"Temporal Quality":     report.TemporalMetrics.TemporalScore,
		"Fidelity Quality":     report.FidelityMetrics.FidelityScore,
		"Novelty Score":        report.NoveltyMetrics.NoveltyScore,
	}

	return &QualitySummary{
		QualityGrade:     grade,
		StrengthAreas:    strengths,
		ImprovementAreas: improvements,
		Recommendations:  recommendations,
		QualityBreakdown: breakdown,
	}
}

// Implement remaining helper methods...
func (qm *QualityMetrics) calculateACF(data []float64, maxLag int) []float64 {
	n := len(data)
	if maxLag > n/4 {
		maxLag = n / 4
	}
	
	acf := make([]float64, maxLag)
	mean := stat.Mean(data, nil)
	
	// Calculate variance (lag 0)
	var c0 float64
	for i := 0; i < n; i++ {
		c0 += (data[i] - mean) * (data[i] - mean)
	}
	c0 /= float64(n)
	
	// Calculate autocorrelations
	for k := 1; k <= maxLag; k++ {
		var ck float64
		for i := k; i < n; i++ {
			ck += (data[i] - mean) * (data[i-k] - mean)
		}
		ck /= float64(n)
		if c0 > 0 {
			acf[k-1] = ck / c0
		}
	}
	
	return acf
}

func extractValues(dataPoints []models.DataPoint) []float64 {
	values := make([]float64, len(dataPoints))
	for i, dp := range dataPoints {
		values[i] = dp.Value
	}
	return values
}

// Placeholder implementations for remaining methods
func (qm *QualityMetrics) calculateDistributionSimilarity(mae, rmse, wasserstein, klDiv float64) float64 {
	// Combine multiple metrics into a single score
	return 1.0 / (1.0 + mae + rmse + wasserstein + klDiv/10.0)
}

func (qm *QualityMetrics) calculateLaggedCorrelations(original, synthetic []float64, maxLag int) map[int]float64 {
	correlations := make(map[int]float64)
	for lag := 1; lag <= maxLag; lag++ {
		if lag < len(original) && lag < len(synthetic) {
			corr := qm.laggedCorrelation(original, synthetic, lag)
			correlations[lag] = corr
		}
	}
	return correlations
}

func (qm *QualityMetrics) laggedCorrelation(original, synthetic []float64, lag int) float64 {
	// Simplified lagged correlation calculation
	n := min(len(original)-lag, len(synthetic))
	if n <= 0 {
		return 0
	}
	
	var sumXY, sumX, sumY, sumX2, sumY2 float64
	for i := 0; i < n; i++ {
		x := original[i+lag]
		y := synthetic[i]
		sumXY += x * y
		sumX += x
		sumY += y
		sumX2 += x * x
		sumY2 += y * y
	}
	
	nf := float64(n)
	numerator := nf*sumXY - sumX*sumY
	denominator := math.Sqrt((nf*sumX2 - sumX*sumX) * (nf*sumY2 - sumY*sumY))
	
	if denominator == 0 {
		return 0
	}
	
	return numerator / denominator
}

// Additional implementations...
func (qm *QualityMetrics) trendSimilarity(original, synthetic []float64) float64 {
	// Calculate linear trend for both series
	origTrend := qm.calculateLinearTrend(original)
	synthTrend := qm.calculateLinearTrend(synthetic)
	
	// Compare trend slopes
	slopeDiff := math.Abs(origTrend.slope - synthTrend.slope)
	maxSlope := math.Max(math.Abs(origTrend.slope), math.Abs(synthTrend.slope))
	
	if maxSlope == 0 {
		return 1.0
	}
	
	similarity := 1.0 - math.Min(1.0, slopeDiff/maxSlope)
	return math.Max(0.0, similarity)
}
func (qm *QualityMetrics) spectralSimilarity(original, synthetic []float64) float64 {
	// Compute power spectral density for both series
	origPSD := qm.computePowerSpectralDensity(original)
	synthPSD := qm.computePowerSpectralDensity(synthetic)
	
	if len(origPSD) == 0 || len(synthPSD) == 0 {
		return 0.0
	}
	
	// Resample to same length
	minLen := int(math.Min(float64(len(origPSD)), float64(len(synthPSD))))
	origPSD = qm.resample(origPSD, minLen)
	synthPSD = qm.resample(synthPSD, minLen)
	
	// Calculate correlation between PSDs
	correlation := stat.Correlation(origPSD, synthPSD, nil)
	return math.Max(0.0, math.Min(1.0, correlation))
}
func (qm *QualityMetrics) analyzeFrequencyDomain(original, synthetic []float64) *FrequencyDomainMetrics {
	// Compute power spectral densities
	origPSD := qm.computePowerSpectralDensity(original)
	synthPSD := qm.computePowerSpectralDensity(synthetic)
	
	// Find dominant frequencies
	origFreqs := qm.findDominantFrequencies(origPSD, 5)
	synthFreqs := qm.findDominantFrequencies(synthPSD, 5)
	
	// Calculate spectral entropy
	origEntropy := qm.calculateSpectralEntropy(origPSD)
	synthEntropy := qm.calculateSpectralEntropy(synthPSD)
	avgEntropy := (origEntropy + synthEntropy) / 2
	
	// Calculate bandpower ratios
	bandpowerRatios := qm.calculateBandpowerRatios(origPSD, synthPSD)
	
	return &FrequencyDomainMetrics{
		PowerSpectralDensity: synthPSD,
		DominantFrequencies:  synthFreqs,
		SpectralEntropy:      avgEntropy,
		BandpowerRatios:      bandpowerRatios,
	}
}
func (qm *QualityMetrics) calculateDiversityIndex(data []float64) float64 {
	// Shannon diversity index adapted for continuous data
	// Discretize data into bins
	numBins := int(math.Sqrt(float64(len(data))))
	if numBins < 10 {
		numBins = 10
	}
	
	min := data[0]
	max := data[0]
	for _, v := range data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	
	if max == min {
		return 0.0
	}
	
	binWidth := (max - min) / float64(numBins)
	counts := make([]int, numBins)
	
	for _, v := range data {
		binIdx := int((v - min) / binWidth)
		if binIdx >= numBins {
			binIdx = numBins - 1
		}
		counts[binIdx]++
	}
	
	// Calculate Shannon entropy
	entropy := 0.0
	n := float64(len(data))
	for _, count := range counts {
		if count > 0 {
			p := float64(count) / n
			entropy -= p * math.Log(p)
		}
	}
	
	// Normalize by maximum possible entropy
	maxEntropy := math.Log(float64(numBins))
	if maxEntropy == 0 {
		return 0.0
	}
	
	return entropy / maxEntropy
}
func (qm *QualityMetrics) calculateNoveltyScore(original, synthetic []float64) float64 {
	// Measure how much synthetic data explores new regions
	origMin, origMax := qm.getMinMax(original)
	synthMin, synthMax := qm.getMinMax(synthetic)
	
	// Calculate range expansion
	rangeExpansion := 0.0
	origRange := origMax - origMin
	if origRange > 0 {
		synthRange := synthMax - synthMin
		rangeExpansion = math.Abs(synthRange - origRange) / origRange
	}
	
	// Count synthetic points outside original range
	outlierCount := 0
	for _, v := range synthetic {
		if v < origMin || v > origMax {
			outlierCount++
		}
	}
	outlierRatio := float64(outlierCount) / float64(len(synthetic))
	
	// Novelty is balanced between exploration and staying within bounds
	novelty := 0.3*rangeExpansion + 0.7*outlierRatio
	return math.Min(1.0, novelty)
}
func (qm *QualityMetrics) calculateCoverageScore(original, synthetic []float64) float64 {
	// Measure how well synthetic data covers the original data space
	origMin, origMax := qm.getMinMax(original)
	synthMin, synthMax := qm.getMinMax(synthetic)
	
	// Calculate coverage of original range
	origRange := origMax - origMin
	if origRange == 0 {
		return 1.0
	}
	
	// Calculate overlapping range
	overlapMin := math.Max(origMin, synthMin)
	overlapMax := math.Min(origMax, synthMax)
	overlapRange := math.Max(0, overlapMax - overlapMin)
	
	coverage := overlapRange / origRange
	return math.Min(1.0, coverage)
}
func (qm *QualityMetrics) assessOverfittingRisk(original, synthetic []float64) float64 {
	// Assess overfitting by checking if synthetic data is too similar to original
	// Use nearest neighbor distances
	totalDistance := 0.0
	count := 0
	
	// Sample subset for efficiency
	sampleSize := int(math.Min(100, float64(len(synthetic))))
	step := len(synthetic) / sampleSize
	
	for i := 0; i < len(synthetic); i += step {
		if count >= sampleSize {
			break
		}
		
		// Find nearest neighbor in original data
	minDist := math.Inf(1)
		for _, origVal := range original {
			dist := math.Abs(synthetic[i] - origVal)
			if dist < minDist {
				minDist = dist
			}
		}
		
		totalDistance += minDist
		count++
	}
	
	if count == 0 {
		return 0.5
	}
	
	avgDistance := totalDistance / float64(count)
	origStd := math.Sqrt(stat.Variance(original, nil))
	
	// Normalize by original data standard deviation
	if origStd == 0 {
		return 0.0
	}
	
	normalizedDist := avgDistance / origStd
	// Higher distance means lower overfitting risk
	overfittingRisk := math.Max(0.0, 1.0 - normalizedDist)
	return math.Min(1.0, overfittingRisk)
}
func (qm *QualityMetrics) calculateBoundaryPreservation(original, synthetic []float64) float64 {
	// Check how well synthetic data preserves the boundaries of original data
	origMin, origMax := qm.getMinMax(original)
	synthMin, synthMax := qm.getMinMax(synthetic)
	
	origRange := origMax - origMin
	if origRange == 0 {
		return 1.0
	}
	
	// Calculate boundary preservation
	lowerBoundaryError := math.Abs(synthMin - origMin) / origRange
	upperBoundaryError := math.Abs(synthMax - origMax) / origRange
	
	boundaryError := (lowerBoundaryError + upperBoundaryError) / 2
	boundaryPreservation := math.Max(0.0, 1.0 - boundaryError)
	
	return math.Min(1.0, boundaryPreservation)
}
func (qm *QualityMetrics) calculateGlobalFidelity(original, synthetic []float64) float64 {
	// Calculate global statistical fidelity
	origMean := stat.Mean(original, nil)
	origStd := math.Sqrt(stat.Variance(original, nil))
	synthMean := stat.Mean(synthetic, nil)
	synthStd := math.Sqrt(stat.Variance(synthetic, nil))
	
	// Mean fidelity
	meanError := 0.0
	if origStd > 0 {
		meanError = math.Abs(origMean - synthMean) / origStd
	}
	meanFidelity := math.Max(0.0, 1.0 - meanError)
	
	// Standard deviation fidelity
	stdError := 0.0
	if origStd > 0 {
		stdError = math.Abs(origStd - synthStd) / origStd
	}
	stdFidelity := math.Max(0.0, 1.0 - stdError)
	
	// Combined global fidelity
	return (meanFidelity + stdFidelity) / 2
}
func (qm *QualityMetrics) calculateLocalFidelity(original, synthetic []float64) float64 {
	// Calculate local density preservation using KDE estimation
	// Simplified implementation using histogram comparison
	numBins := int(math.Sqrt(float64(len(original))))
	if numBins < 20 {
		numBins = 20
	}
	
	// Get common range
	origMin, origMax := qm.getMinMax(original)
	synthMin, synthMax := qm.getMinMax(synthetic)
	minVal := math.Min(origMin, synthMin)
	maxVal := math.Max(origMax, synthMax)
	
	if maxVal == minVal {
		return 1.0
	}
	
	// Create histograms
	origHist := qm.createHistogram(original, minVal, maxVal, numBins)
	synthHist := qm.createHistogram(synthetic, minVal, maxVal, numBins)
	
	// Calculate histogram intersection
	intersection := 0.0
	origSum := 0.0
	synthSum := 0.0
	
	for i := 0; i < numBins; i++ {
		intersection += math.Min(origHist[i], synthHist[i])
		origSum += origHist[i]
		synthSum += synthHist[i]
	}
	
	// Normalize
	if origSum > 0 && synthSum > 0 {
		return intersection / math.Max(origSum, synthSum)
	}
	return 0.0
}
func (qm *QualityMetrics) calculateStructuralFidelity(original, synthetic []float64) float64 {
	// Calculate structural fidelity based on autocorrelation preservation
	maxLag := int(math.Min(20, float64(len(original))/4))
	if maxLag < 1 {
		return 1.0
	}
	
	origAutocorr := qm.calculateAutocorrelationLags(original, maxLag)
	synthAutocorr := qm.calculateAutocorrelationLags(synthetic, maxLag)
	
	// Calculate correlation between autocorrelation functions
	if len(origAutocorr) == 0 || len(synthAutocorr) == 0 {
		return 0.0
	}
	
	correlation := stat.Correlation(origAutocorr, synthAutocorr, nil)
	return math.Max(0.0, math.Min(1.0, correlation))
}
func (qm *QualityMetrics) calculatePatternFidelity(original, synthetic []float64) float64 {
	// Calculate pattern fidelity using trend and seasonality preservation
	trendSim := qm.trendSimilarity(original, synthetic)
	spectralSim := qm.spectralSimilarity(original, synthetic)
	
	// Weight trend and spectral similarities
	patternFidelity := 0.6*trendSim + 0.4*spectralSim
	return math.Max(0.0, math.Min(1.0, patternFidelity))
}
func (qm *QualityMetrics) calculateSkewness(data []float64) float64 {
	mean := stat.Mean(data, nil)
	variance := stat.Variance(data, nil)
	stdDev := math.Sqrt(variance)
	
	var sum float64
	for _, x := range data {
		sum += math.Pow((x-mean)/stdDev, 3)
	}
	
	return sum / float64(len(data))
}
func (qm *QualityMetrics) calculateKurtosis(data []float64) float64 {
	mean := stat.Mean(data, nil)
	variance := stat.Variance(data, nil)
	stdDev := math.Sqrt(variance)
	
	var sum float64
	for _, x := range data {
		sum += math.Pow((x-mean)/stdDev, 4)
	}
	
	return sum/float64(len(data)) - 3 // Excess kurtosis
}
func (qm *QualityMetrics) resample(data []float64, newSize int) []float64 {
	if len(data) <= newSize {
		return data
	}
	
	resampled := make([]float64, newSize)
	ratio := float64(len(data)) / float64(newSize)
	
	for i := 0; i < newSize; i++ {
		index := int(float64(i) * ratio)
		resampled[i] = data[index]
	}
	
	return resampled
}
func (qm *QualityMetrics) createHistogram(data []float64, bins int) []float64 {
	if len(data) == 0 {
		return make([]float64, bins)
	}
	
	minVal := data[0]
	maxVal := data[0]
	for _, v := range data {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	
	hist := make([]float64, bins)
	binWidth := (maxVal - minVal) / float64(bins)
	
	for _, v := range data {
		binIndex := int((v - minVal) / binWidth)
		if binIndex >= bins {
			binIndex = bins - 1
		}
		hist[binIndex]++
	}
	
	// Normalize
	total := float64(len(data))
	for i := range hist {
		hist[i] /= total
	}
	
	return hist
}
func (qm *QualityMetrics) determineQualityGrade(score float64) string {
	switch {
	case score >= 0.9: return "A"
	case score >= 0.8: return "B"
	case score >= 0.7: return "C"
	case score >= 0.6: return "D"
	default: return "F"
	}
}
func (qm *QualityMetrics) analyzeStrengthsAndWeaknesses(report *QualityReport) ([]string, []string) {
	strengths := []string{}
	improvements := []string{}
	
	if report.DistributionMetrics.DistributionSimilarity > 0.8 {
		strengths = append(strengths, "Good distributional similarity")
	} else {
		improvements = append(improvements, "Improve distributional matching")
	}
	
	if report.CorrelationMetrics.CorrelationScore > 0.8 {
		strengths = append(strengths, "Strong correlation preservation")
	} else {
		improvements = append(improvements, "Enhance correlation structure")
	}
	
	return strengths, improvements
}
func (qm *QualityMetrics) generateRecommendations(report *QualityReport) []string {
	recommendations := []string{}
	
	if report.OverallScore < 0.7 {
		recommendations = append(recommendations, "Consider using a more sophisticated generation model")
	}
	
	if report.DistributionMetrics.DistributionSimilarity < 0.7 {
		recommendations = append(recommendations, "Tune generator parameters for better distributional fit")
	}
	
	return recommendations
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ValidationError represents a validation error
type ValidationError struct {
	Code    string
	Message string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

func NewValidationError(code, message string) *ValidationError {
	return &ValidationError{Code: code, Message: message}
}

// Helper functions for quality metrics

type LinearTrend struct {
	slope     float64
	intercept float64
	r2        float64
}

func (qm *QualityMetrics) calculateLinearTrend(data []float64) LinearTrend {
	n := float64(len(data))
	if n < 2 {
		return LinearTrend{slope: 0, intercept: 0, r2: 0}
	}
	
	// Calculate sums
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumX2 := 0.0
	
	for i, y := range data {
		x := float64(i)
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}
	
	// Calculate slope and intercept
	denominator := n*sumX2 - sumX*sumX
	if denominator == 0 {
		return LinearTrend{slope: 0, intercept: sumY / n, r2: 0}
	}
	
	slope := (n*sumXY - sumX*sumY) / denominator
	intercept := (sumY - slope*sumX) / n
	
	// Calculate R²
	meanY := sumY / n
	ssRes := 0.0
	ssTot := 0.0
	
	for i, y := range data {
		predicted := slope*float64(i) + intercept
		ssRes += (y - predicted) * (y - predicted)
		ssTot += (y - meanY) * (y - meanY)
	}
	
	r2 := 0.0
	if ssTot != 0 {
		r2 = 1 - ssRes/ssTot
	}
	
	return LinearTrend{slope: slope, intercept: intercept, r2: r2}
}

func (qm *QualityMetrics) computePowerSpectralDensity(data []float64) []float64 {
	// Simplified PSD calculation using autocorrelation method
	n := len(data)
	if n < 4 {
		return []float64{}
	}
	
	// Calculate autocorrelation
	maxLag := n / 4
	autocorr := qm.calculateAutocorrelationLags(data, maxLag)
	
	// Apply window function (Hamming)
	for i := range autocorr {
		weight := 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(len(autocorr)-1))
		autocorr[i] *= weight
	}
	
	// Simplified FFT (just return the autocorrelation as PSD approximation)
	return autocorr
}

func (qm *QualityMetrics) findDominantFrequencies(psd []float64, numFreqs int) []float64 {
	if len(psd) == 0 {
		return []float64{}
	}
	
	// Find peaks in PSD
	peaks := make([]struct {
		index int
		value float64
	}, 0)
	
	for i := 1; i < len(psd)-1; i++ {
		if psd[i] > psd[i-1] && psd[i] > psd[i+1] {
			peaks = append(peaks, struct {
				index int
				value float64
			}{i, psd[i]})
		}
	}
	
	// Sort by magnitude
	sort.Slice(peaks, func(i, j int) bool {
		return peaks[i].value > peaks[j].value
	})
	
	// Return top frequencies
	maxFreqs := numFreqs
	if len(peaks) < maxFreqs {
		maxFreqs = len(peaks)
	}
	
	frequencies := make([]float64, maxFreqs)
	for i := 0; i < maxFreqs; i++ {
		frequencies[i] = float64(peaks[i].index) / float64(len(psd))
	}
	
	return frequencies
}

func (qm *QualityMetrics) calculateSpectralEntropy(psd []float64) float64 {
	if len(psd) == 0 {
		return 0.0
	}
	
	// Normalize PSD
	total := 0.0
	for _, p := range psd {
		total += p
	}
	
	if total == 0 {
		return 0.0
	}
	
	// Calculate entropy
	entropy := 0.0
	for _, p := range psd {
		if p > 0 {
			prob := p / total
			entropy -= prob * math.Log(prob)
		}
	}
	
	return entropy
}

func (qm *QualityMetrics) calculateBandpowerRatios(origPSD, synthPSD []float64) map[string]float64 {
	ratios := make(map[string]float64)
	
	if len(origPSD) == 0 || len(synthPSD) == 0 {
		return ratios
	}
	
	// Ensure same length
	minLen := len(origPSD)
	if len(synthPSD) < minLen {
		minLen = len(synthPSD)
	}
	
	// Calculate band powers (simplified frequency bands)
	bands := map[string][2]int{
		"low":    {0, minLen / 4},
		"medium": {minLen / 4, 3 * minLen / 4},
		"high":   {3 * minLen / 4, minLen},
	}
	
	for bandName, bandRange := range bands {
		origPower := 0.0
		synthPower := 0.0
		
		for i := bandRange[0]; i < bandRange[1]; i++ {
			origPower += origPSD[i]
			synthPower += synthPSD[i]
		}
		
		if origPower > 0 {
			ratios[bandName] = synthPower / origPower
		} else {
			ratios[bandName] = 1.0
		}
	}
	
	return ratios
}

func (qm *QualityMetrics) getMinMax(data []float64) (float64, float64) {
	if len(data) == 0 {
		return 0, 0
	}
	
	min := data[0]
	max := data[0]
	
	for _, v := range data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	
	return min, max
}

func (qm *QualityMetrics) createHistogram(data []float64, minVal, maxVal float64, numBins int) []float64 {
	hist := make([]float64, numBins)
	
	if maxVal == minVal {
		return hist
	}
	
	binWidth := (maxVal - minVal) / float64(numBins)
	
	for _, v := range data {
		binIdx := int((v - minVal) / binWidth)
		if binIdx >= numBins {
			binIdx = numBins - 1
		}
		if binIdx < 0 {
			binIdx = 0
		}
		hist[binIdx]++
	}
	
	// Normalize
	total := float64(len(data))
	for i := range hist {
		hist[i] /= total
	}
	
	return hist
}

func (qm *QualityMetrics) calculateAutocorrelationLags(data []float64, maxLag int) []float64 {
	n := len(data)
	if n < 2 || maxLag < 1 {
		return []float64{}
	}
	
	if maxLag >= n {
		maxLag = n - 1
	}
	
	mean := stat.Mean(data, nil)
	autocorr := make([]float64, maxLag)
	
	// Calculate variance (lag 0)
	variance := 0.0
	for _, v := range data {
		diff := v - mean
		variance += diff * diff
	}
	
	if variance == 0 {
		return autocorr
	}
	
	// Calculate autocorrelations for each lag
	for lag := 0; lag < maxLag; lag++ {
		covariance := 0.0
		count := 0
		
		for i := 0; i < n-lag; i++ {
			covariance += (data[i] - mean) * (data[i+lag] - mean)
			count++
		}
		
		if count > 0 {
			autocorr[lag] = covariance / variance
		}
	}
	
	return autocorr
}