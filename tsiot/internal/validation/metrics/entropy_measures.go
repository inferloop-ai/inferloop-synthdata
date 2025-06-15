package metrics

import (
	"errors"
	"math"
	"sort"

	"github.com/inferloop/tsiot/pkg/models"
)

// EntropyMeasures contains various information-theoretic measures
type EntropyMeasures struct {
	ShannonEntropy       float64 `json:"shannon_entropy"`        // Shannon entropy
	DifferentialEntropy  float64 `json:"differential_entropy"`   // Differential entropy (continuous)
	ConditionalEntropy   float64 `json:"conditional_entropy"`    // Conditional entropy H(X|Y)
	MutualInformation    float64 `json:"mutual_information"`     // Mutual information I(X;Y)
	RelativeEntropy      float64 `json:"relative_entropy"`       // KL divergence
	JointEntropy         float64 `json:"joint_entropy"`          // Joint entropy H(X,Y)
	CrossEntropy         float64 `json:"cross_entropy"`          // Cross entropy
	RenyiEntropy         float64 `json:"renyi_entropy"`          // Rényi entropy (alpha=2)
	TsallisEntropy       float64 `json:"tsallis_entropy"`        // Tsallis entropy (q=2)
	PermutationEntropy   float64 `json:"permutation_entropy"`    // Permutation entropy
	ApproximateEntropy   float64 `json:"approximate_entropy"`    // Approximate entropy (ApEn)
	SampleEntropy        float64 `json:"sample_entropy"`         // Sample entropy (SampEn)
}

// EntropyComparison contains comparison results between two datasets
type EntropyComparison struct {
	OriginalEntropy     EntropyMeasures `json:"original_entropy"`      // Entropy of original data
	SyntheticEntropy    EntropyMeasures `json:"synthetic_entropy"`     // Entropy of synthetic data
	EntropyDifferences  EntropyMeasures `json:"entropy_differences"`   // Absolute differences
	RelativeDifferences EntropyMeasures `json:"relative_differences"`  // Relative differences
	SimilarityScore     float64         `json:"similarity_score"`      // Overall similarity (0-1)
	KLDivergence        float64         `json:"kl_divergence"`         // KL divergence between distributions
	JSDistance          float64         `json:"js_distance"`           // Jensen-Shannon distance
}

// HistogramBins represents a histogram with bins and probabilities
type HistogramBins struct {
	Bins         []float64 `json:"bins"`         // Bin edges
	Counts       []int     `json:"counts"`       // Count in each bin
	Probabilities []float64 `json:"probabilities"` // Probability mass in each bin
}

// CalculateEntropyMeasures computes comprehensive entropy measures for a dataset
func CalculateEntropyMeasures(data []float64, config map[string]interface{}) (*EntropyMeasures, error) {
	if len(data) < 10 {
		return nil, errors.New("entropy calculation requires at least 10 data points")
	}

	// Get configuration parameters
	bins := getIntConfig(config, "bins", 50)
	embeddingDim := getIntConfig(config, "embedding_dimension", 3)
	tolerance := getFloatConfig(config, "tolerance", 0.1)
	
	measures := &EntropyMeasures{}

	// Calculate Shannon entropy (discrete approximation)
	hist := createHistogram(data, bins)
	measures.ShannonEntropy = calculateShannonEntropy(hist.Probabilities)

	// Calculate differential entropy (continuous approximation)
	measures.DifferentialEntropy = calculateDifferentialEntropy(data, bins)

	// Calculate Rényi entropy (alpha = 2)
	measures.RenyiEntropy = calculateRenyiEntropy(hist.Probabilities, 2.0)

	// Calculate Tsallis entropy (q = 2)
	measures.TsallisEntropy = calculateTsallisEntropy(hist.Probabilities, 2.0)

	// Calculate permutation entropy
	measures.PermutationEntropy = calculatePermutationEntropy(data, embeddingDim)

	// Calculate approximate entropy
	measures.ApproximateEntropy = calculateApproximateEntropy(data, embeddingDim, tolerance)

	// Calculate sample entropy
	measures.SampleEntropy = calculateSampleEntropy(data, embeddingDim, tolerance)

	return measures, nil
}

// CompareEntropy compares entropy measures between original and synthetic data
func CompareEntropy(original, synthetic []float64, config map[string]interface{}) (*EntropyComparison, error) {
	if len(original) < 10 || len(synthetic) < 10 {
		return nil, errors.New("entropy comparison requires at least 10 data points in each dataset")
	}

	// Calculate entropy measures for both datasets
	originalEntropy, err := CalculateEntropyMeasures(original, config)
	if err != nil {
		return nil, err
	}

	syntheticEntropy, err := CalculateEntropyMeasures(synthetic, config)
	if err != nil {
		return nil, err
	}

	// Calculate differences
	entropyDiffs := calculateEntropyDifferences(originalEntropy, syntheticEntropy, false)
	relativeDiffs := calculateEntropyDifferences(originalEntropy, syntheticEntropy, true)

	// Calculate KL divergence and JS distance
	bins := getIntConfig(config, "bins", 50)
	originalHist := createHistogram(original, bins)
	syntheticHist := createHistogram(synthetic, bins)
	
	klDiv := calculateKLDivergence(originalHist.Probabilities, syntheticHist.Probabilities)
	jsDistance := calculateJSDistance(originalHist.Probabilities, syntheticHist.Probabilities)

	// Calculate overall similarity score
	similarityScore := calculateEntropySimilarityScore(entropyDiffs, relativeDiffs)

	comparison := &EntropyComparison{
		OriginalEntropy:     *originalEntropy,
		SyntheticEntropy:    *syntheticEntropy,
		EntropyDifferences:  *entropyDiffs,
		RelativeDifferences: *relativeDiffs,
		SimilarityScore:     similarityScore,
		KLDivergence:        klDiv,
		JSDistance:          jsDistance,
	}

	return comparison, nil
}

// calculateShannonEntropy computes Shannon entropy from probability distribution
func calculateShannonEntropy(probabilities []float64) float64 {
	entropy := 0.0
	for _, p := range probabilities {
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	return entropy
}

// calculateDifferentialEntropy estimates differential entropy for continuous data
func calculateDifferentialEntropy(data []float64, bins int) float64 {
	hist := createHistogram(data, bins)
	
	// Calculate bin width
	minVal, maxVal := findMinMax(data)
	binWidth := (maxVal - minVal) / float64(bins)
	
	// Differential entropy approximation
	entropy := 0.0
	for _, p := range hist.Probabilities {
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	
	// Add correction for bin width
	return entropy + math.Log2(binWidth)
}

// calculateRenyiEntropy computes Rényi entropy of order alpha
func calculateRenyiEntropy(probabilities []float64, alpha float64) float64 {
	if alpha == 1.0 {
		return calculateShannonEntropy(probabilities)
	}

	sum := 0.0
	for _, p := range probabilities {
		if p > 0 {
			sum += math.Pow(p, alpha)
		}
	}

	if sum <= 0 {
		return 0
	}

	return math.Log2(sum) / (1.0 - alpha)
}

// calculateTsallisEntropy computes Tsallis entropy of order q
func calculateTsallisEntropy(probabilities []float64, q float64) float64 {
	if q == 1.0 {
		return calculateShannonEntropy(probabilities)
	}

	sum := 0.0
	for _, p := range probabilities {
		if p > 0 {
			sum += math.Pow(p, q)
		}
	}

	return (1.0 - sum) / (q - 1.0)
}

// calculatePermutationEntropy computes permutation entropy
func calculatePermutationEntropy(data []float64, embeddingDim int) float64 {
	if len(data) < embeddingDim {
		return 0
	}

	// Create ordinal patterns
	patterns := make(map[string]int)
	
	for i := 0; i <= len(data)-embeddingDim; i++ {
		// Extract embedding vector
		embedding := data[i : i+embeddingDim]
		
		// Create ordinal pattern
		pattern := createOrdinalPattern(embedding)
		patterns[pattern]++
	}

	// Calculate relative frequencies
	total := len(data) - embeddingDim + 1
	probabilities := make([]float64, 0, len(patterns))
	
	for _, count := range patterns {
		prob := float64(count) / float64(total)
		probabilities = append(probabilities, prob)
	}

	return calculateShannonEntropy(probabilities)
}

// calculateApproximateEntropy computes approximate entropy (ApEn)
func calculateApproximateEntropy(data []float64, m int, r float64) float64 {
	n := len(data)
	if n < m+1 {
		return 0
	}

	phi := func(m int) float64 {
		patterns := make([][]float64, n-m+1)
		for i := 0; i <= n-m; i++ {
			patterns[i] = data[i : i+m]
		}

		sum := 0.0
		for i := 0; i <= n-m; i++ {
			matches := 0
			for j := 0; j <= n-m; j++ {
				if maxDistance(patterns[i], patterns[j]) <= r {
					matches++
				}
			}
			if matches > 0 {
				sum += math.Log(float64(matches) / float64(n-m+1))
			}
		}
		return sum / float64(n-m+1)
	}

	return phi(m) - phi(m+1)
}

// calculateSampleEntropy computes sample entropy (SampEn)
func calculateSampleEntropy(data []float64, m int, r float64) float64 {
	n := len(data)
	if n < m+1 {
		return 0
	}

	// Count template matches for length m and m+1
	a := 0 // matches for length m+1
	b := 0 // matches for length m

	for i := 0; i < n-m; i++ {
		for j := i + 1; j < n-m; j++ {
			// Check if templates of length m match
			match := true
			for k := 0; k < m; k++ {
				if math.Abs(data[i+k]-data[j+k]) > r {
					match = false
					break
				}
			}
			
			if match {
				b++
				// Check if templates of length m+1 also match
				if math.Abs(data[i+m]-data[j+m]) <= r {
					a++
				}
			}
		}
	}

	if b == 0 {
		return 0
	}

	return -math.Log(float64(a) / float64(b))
}

// calculateKLDivergence computes Kullback-Leibler divergence
func calculateKLDivergence(p, q []float64) float64 {
	if len(p) != len(q) {
		return math.Inf(1)
	}

	kl := 0.0
	for i := 0; i < len(p); i++ {
		if p[i] > 0 {
			if q[i] > 0 {
				kl += p[i] * math.Log(p[i]/q[i])
			} else {
				return math.Inf(1) // KL divergence is infinite
			}
		}
	}
	return kl
}

// calculateJSDistance computes Jensen-Shannon distance
func calculateJSDistance(p, q []float64) float64 {
	if len(p) != len(q) {
		return math.Inf(1)
	}

	// Calculate average distribution
	m := make([]float64, len(p))
	for i := 0; i < len(p); i++ {
		m[i] = (p[i] + q[i]) / 2.0
	}

	// Calculate JS divergence
	js := 0.0
	for i := 0; i < len(p); i++ {
		if p[i] > 0 && m[i] > 0 {
			js += 0.5 * p[i] * math.Log(p[i]/m[i])
		}
		if q[i] > 0 && m[i] > 0 {
			js += 0.5 * q[i] * math.Log(q[i]/m[i])
		}
	}

	// JS distance is square root of JS divergence
	return math.Sqrt(js)
}

// calculateMutualInformation computes mutual information between two variables
func calculateMutualInformation(x, y []float64, bins int) (float64, error) {
	if len(x) != len(y) {
		return 0, errors.New("x and y must have the same length")
	}

	// Create joint histogram
	jointHist := createJointHistogram(x, y, bins)
	
	// Create marginal histograms
	xHist := createHistogram(x, bins)
	yHist := createHistogram(y, bins)

	mi := 0.0
	for i := 0; i < bins; i++ {
		for j := 0; j < bins; j++ {
			pxy := jointHist[i][j]
			px := xHist.Probabilities[i]
			py := yHist.Probabilities[j]
			
			if pxy > 0 && px > 0 && py > 0 {
				mi += pxy * math.Log(pxy/(px*py))
			}
		}
	}

	return mi, nil
}

// Utility functions

func createHistogram(data []float64, bins int) *HistogramBins {
	if len(data) == 0 {
		return &HistogramBins{
			Bins:          make([]float64, bins+1),
			Counts:        make([]int, bins),
			Probabilities: make([]float64, bins),
		}
	}

	minVal, maxVal := findMinMax(data)
	binWidth := (maxVal - minVal) / float64(bins)
	
	// Create bin edges
	binEdges := make([]float64, bins+1)
	for i := 0; i <= bins; i++ {
		binEdges[i] = minVal + float64(i)*binWidth
	}

	// Count data points in each bin
	counts := make([]int, bins)
	for _, x := range data {
		binIndex := int((x - minVal) / binWidth)
		if binIndex >= bins {
			binIndex = bins - 1
		}
		if binIndex < 0 {
			binIndex = 0
		}
		counts[binIndex]++
	}

	// Convert to probabilities
	total := float64(len(data))
	probabilities := make([]float64, bins)
	for i, count := range counts {
		probabilities[i] = float64(count) / total
	}

	return &HistogramBins{
		Bins:          binEdges,
		Counts:        counts,
		Probabilities: probabilities,
	}
}

func createJointHistogram(x, y []float64, bins int) [][]float64 {
	n := len(x)
	xMin, xMax := findMinMax(x)
	yMin, yMax := findMinMax(y)
	
	xBinWidth := (xMax - xMin) / float64(bins)
	yBinWidth := (yMax - yMin) / float64(bins)

	// Initialize joint histogram
	jointHist := make([][]float64, bins)
	for i := range jointHist {
		jointHist[i] = make([]float64, bins)
	}

	// Fill joint histogram
	for i := 0; i < n; i++ {
		xBin := int((x[i] - xMin) / xBinWidth)
		yBin := int((y[i] - yMin) / yBinWidth)
		
		if xBin >= bins {
			xBin = bins - 1
		}
		if yBin >= bins {
			yBin = bins - 1
		}
		if xBin < 0 {
			xBin = 0
		}
		if yBin < 0 {
			yBin = 0
		}
		
		jointHist[xBin][yBin]++
	}

	// Normalize to probabilities
	total := float64(n)
	for i := 0; i < bins; i++ {
		for j := 0; j < bins; j++ {
			jointHist[i][j] /= total
		}
	}

	return jointHist
}

func createOrdinalPattern(embedding []float64) string {
	// Create index array
	indices := make([]int, len(embedding))
	for i := range indices {
		indices[i] = i
	}

	// Sort indices by corresponding values
	sort.Slice(indices, func(i, j int) bool {
		return embedding[indices[i]] < embedding[indices[j]]
	})

	// Convert to string representation
	pattern := ""
	for _, idx := range indices {
		pattern += string(rune('0' + idx))
	}

	return pattern
}

func maxDistance(a, b []float64) float64 {
	maxDist := 0.0
	for i := 0; i < len(a) && i < len(b); i++ {
		dist := math.Abs(a[i] - b[i])
		if dist > maxDist {
			maxDist = dist
		}
	}
	return maxDist
}

func findMinMax(data []float64) (float64, float64) {
	if len(data) == 0 {
		return 0, 0
	}
	
	min, max := data[0], data[0]
	for _, x := range data[1:] {
		if x < min {
			min = x
		}
		if x > max {
			max = x
		}
	}
	return min, max
}

func calculateEntropyDifferences(original, synthetic *EntropyMeasures, relative bool) *EntropyMeasures {
	diff := &EntropyMeasures{}

	if relative {
		// Relative differences (as percentages)
		diff.ShannonEntropy = calculateRelativeDiff(original.ShannonEntropy, synthetic.ShannonEntropy)
		diff.DifferentialEntropy = calculateRelativeDiff(original.DifferentialEntropy, synthetic.DifferentialEntropy)
		diff.RenyiEntropy = calculateRelativeDiff(original.RenyiEntropy, synthetic.RenyiEntropy)
		diff.TsallisEntropy = calculateRelativeDiff(original.TsallisEntropy, synthetic.TsallisEntropy)
		diff.PermutationEntropy = calculateRelativeDiff(original.PermutationEntropy, synthetic.PermutationEntropy)
		diff.ApproximateEntropy = calculateRelativeDiff(original.ApproximateEntropy, synthetic.ApproximateEntropy)
		diff.SampleEntropy = calculateRelativeDiff(original.SampleEntropy, synthetic.SampleEntropy)
	} else {
		// Absolute differences
		diff.ShannonEntropy = math.Abs(original.ShannonEntropy - synthetic.ShannonEntropy)
		diff.DifferentialEntropy = math.Abs(original.DifferentialEntropy - synthetic.DifferentialEntropy)
		diff.RenyiEntropy = math.Abs(original.RenyiEntropy - synthetic.RenyiEntropy)
		diff.TsallisEntropy = math.Abs(original.TsallisEntropy - synthetic.TsallisEntropy)
		diff.PermutationEntropy = math.Abs(original.PermutationEntropy - synthetic.PermutationEntropy)
		diff.ApproximateEntropy = math.Abs(original.ApproximateEntropy - synthetic.ApproximateEntropy)
		diff.SampleEntropy = math.Abs(original.SampleEntropy - synthetic.SampleEntropy)
	}

	return diff
}

func calculateRelativeDiff(original, synthetic float64) float64 {
	if original == 0 {
		return 0
	}
	return math.Abs(original-synthetic) / math.Abs(original)
}

func calculateEntropySimilarityScore(absoluteDiffs, relativeDiffs *EntropyMeasures) float64 {
	// Weight different entropy measures
	weights := []float64{0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05} // Shannon, Differential, Renyi, etc.
	
	relDiffs := []float64{
		relativeDiffs.ShannonEntropy,
		relativeDiffs.DifferentialEntropy,
		relativeDiffs.RenyiEntropy,
		relativeDiffs.TsallisEntropy,
		relativeDiffs.PermutationEntropy,
		relativeDiffs.ApproximateEntropy,
		relativeDiffs.SampleEntropy,
	}

	// Calculate weighted similarity score
	weightedSum := 0.0
	totalWeight := 0.0
	
	for i, diff := range relDiffs {
		if i < len(weights) && !math.IsInf(diff, 0) && !math.IsNaN(diff) {
			// Convert relative difference to similarity (1 - diff, capped at 0)
			similarity := math.Max(0, 1.0-diff)
			weightedSum += weights[i] * similarity
			totalWeight += weights[i]
		}
	}

	if totalWeight == 0 {
		return 0
	}

	return weightedSum / totalWeight
}

func getIntConfig(config map[string]interface{}, key string, defaultValue int) int {
	if val, ok := config[key]; ok {
		if intVal, ok := val.(int); ok {
			return intVal
		}
	}
	return defaultValue
}

func getFloatConfig(config map[string]interface{}, key string, defaultValue float64) float64 {
	if val, ok := config[key]; ok {
		if floatVal, ok := val.(float64); ok {
			return floatVal
		}
	}
	return defaultValue
}

// EntropyValidationMetric implements validation interface
func EntropyValidationMetric(original, synthetic *models.TimeSeries, config map[string]interface{}) (*models.ValidationMetric, error) {
	// Extract values from time series
	originalValues := make([]float64, len(original.DataPoints))
	syntheticValues := make([]float64, len(synthetic.DataPoints))
	
	for i, dp := range original.DataPoints {
		originalValues[i] = dp.Value
	}
	for i, dp := range synthetic.DataPoints {
		syntheticValues[i] = dp.Value
	}

	// Compare entropy measures
	comparison, err := CompareEntropy(originalValues, syntheticValues, config)
	if err != nil {
		return nil, err
	}

	return &models.ValidationMetric{
		Name:        "Entropy Similarity",
		Value:       comparison.SimilarityScore,
		Score:       comparison.SimilarityScore,
		Passed:      comparison.SimilarityScore > 0.7, // Threshold
		Description: "Measures similarity between information-theoretic properties",
		Details: map[string]interface{}{
			"kl_divergence":         comparison.KLDivergence,
			"js_distance":           comparison.JSDistance,
			"entropy_differences":   comparison.EntropyDifferences,
			"relative_differences":  comparison.RelativeDifferences,
			"original_entropy":      comparison.OriginalEntropy,
			"synthetic_entropy":     comparison.SyntheticEntropy,
		},
	}, nil
}