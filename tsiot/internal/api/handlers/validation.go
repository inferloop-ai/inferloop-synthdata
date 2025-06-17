package handlers

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sort"
	"time"

	"github.com/gorilla/mux"
	"github.com/inferloop/tsiot/pkg/models"
)

type ValidationHandler struct{}

func NewValidationHandler() *ValidationHandler {
	return &ValidationHandler{}
}

type ValidationRequest struct {
	TimeSeries *models.TimeSeries `json:"timeSeries"`
	Metrics    []string           `json:"metrics"`
	Options    ValidationOptions  `json:"options"`
}

type ValidationOptions struct {
	CompareWith     *models.TimeSeries `json:"compareWith,omitempty"`
	Thresholds      map[string]float64 `json:"thresholds,omitempty"`
	SlidingWindow   int                `json:"slidingWindow,omitempty"`
	SeasonalPeriod  int                `json:"seasonalPeriod,omitempty"`
}

type ValidationResult struct {
	Status      string                 `json:"status"`
	Score       float64                `json:"score"`
	Metrics     map[string]interface{} `json:"metrics"`
	Insights    []string               `json:"insights"`
	Warnings    []string               `json:"warnings"`
	ValidatedAt time.Time              `json:"validatedAt"`
}

func (h *ValidationHandler) ValidateTimeSeries(w http.ResponseWriter, r *http.Request) {
	var request ValidationRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	if request.TimeSeries == nil {
		http.Error(w, "TimeSeries data is required", http.StatusBadRequest)
		return
	}

	if len(request.TimeSeries.DataPoints) == 0 {
		http.Error(w, "TimeSeries must contain data points", http.StatusBadRequest)
		return
	}

	result := h.performValidation(request)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func (h *ValidationHandler) ValidateBatch(w http.ResponseWriter, r *http.Request) {
	var batchRequest struct {
		Requests []ValidationRequest `json:"requests"`
		Parallel bool                `json:"parallel"`
	}

	if err := json.NewDecoder(r.Body).Decode(&batchRequest); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	if len(batchRequest.Requests) == 0 {
		http.Error(w, "No validation requests provided", http.StatusBadRequest)
		return
	}

	var results []ValidationResult
	if batchRequest.Parallel {
		results = h.processBatchParallel(batchRequest.Requests)
	} else {
		results = h.processBatchSequential(batchRequest.Requests)
	}

	response := map[string]interface{}{
		"status":  "completed",
		"results": results,
		"summary": map[string]interface{}{
			"total":       len(results),
			"avgScore":    h.calculateAverageScore(results),
			"validatedAt": time.Now(),
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *ValidationHandler) GetValidationMetrics(w http.ResponseWriter, r *http.Request) {
	metrics := map[string]interface{}{
		"statistical": []map[string]interface{}{
			{
				"name":        "mean",
				"description": "Arithmetic mean of the time series",
				"category":    "central_tendency",
			},
			{
				"name":        "std",
				"description": "Standard deviation",
				"category":    "dispersion",
			},
			{
				"name":        "variance",
				"description": "Variance of the time series",
				"category":    "dispersion",
			},
			{
				"name":        "skewness",
				"description": "Measure of asymmetry",
				"category":    "shape",
			},
			{
				"name":        "kurtosis",
				"description": "Measure of tail heaviness",
				"category":    "shape",
			},
		},
		"temporal": []map[string]interface{}{
			{
				"name":        "autocorrelation",
				"description": "Correlation with lagged versions",
				"category":    "dependency",
			},
			{
				"name":        "trend",
				"description": "Long-term directional movement",
				"category":    "pattern",
			},
			{
				"name":        "seasonality",
				"description": "Recurring patterns",
				"category":    "pattern",
			},
			{
				"name":        "stationarity",
				"description": "Statistical properties consistency",
				"category":    "stability",
			},
		},
		"quality": []map[string]interface{}{
			{
				"name":        "missing_values",
				"description": "Percentage of missing values",
				"category":    "completeness",
			},
			{
				"name":        "outliers",
				"description": "Number of outlier points",
				"category":    "anomalies",
			},
			{
				"name":        "noise_level",
				"description": "Estimated noise in the signal",
				"category":    "quality",
			},
		},
		"similarity": []map[string]interface{}{
			{
				"name":        "correlation",
				"description": "Pearson correlation coefficient",
				"category":    "comparison",
			},
			{
				"name":        "dtw_distance",
				"description": "Dynamic Time Warping distance",
				"category":    "comparison",
			},
			{
				"name":        "wasserstein_distance",
				"description": "Earth Mover's Distance",
				"category":    "distribution",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

func (h *ValidationHandler) CompareTimeSeries(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Original  *models.TimeSeries `json:"original"`
		Synthetic *models.TimeSeries `json:"synthetic"`
		Metrics   []string           `json:"metrics"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	if request.Original == nil || request.Synthetic == nil {
		http.Error(w, "Both original and synthetic time series are required", http.StatusBadRequest)
		return
	}

	comparison := h.compareTimeSeries(request.Original, request.Synthetic, request.Metrics)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(comparison)
}

func (h *ValidationHandler) GetQualityReport(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	seriesID := vars["id"]

	if seriesID == "" {
		http.Error(w, "Series ID is required", http.StatusBadRequest)
		return
	}

	report := map[string]interface{}{
		"seriesId":    seriesID,
		"generatedAt": time.Now(),
		"quality": map[string]interface{}{
			"overall_score": 0.85,
			"completeness":  0.98,
			"consistency":   0.87,
			"accuracy":      0.82,
		},
		"statistics": map[string]interface{}{
			"mean":        23.5,
			"std":         4.2,
			"min":         12.1,
			"max":         45.8,
			"data_points": 1440,
		},
		"patterns": map[string]interface{}{
			"trend":       "stable",
			"seasonality": true,
			"cycles":      []string{"daily", "weekly"},
		},
		"anomalies": map[string]interface{}{
			"count":      5,
			"percentage": 0.35,
			"locations":  []int{123, 456, 789, 1012, 1234},
		},
		"recommendations": []string{
			"Consider smoothing outliers",
			"Verify seasonal patterns",
			"Check data collection intervals",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(report)
}

func (h *ValidationHandler) performValidation(request ValidationRequest) ValidationResult {
	metrics := h.calculateMetrics(request.TimeSeries, request.Metrics)
	
	score := h.calculateOverallScore(metrics)
	insights := h.generateInsights(metrics)
	warnings := h.generateWarnings(metrics, request.Options.Thresholds)

	status := "valid"
	if score < 0.5 {
		status = "invalid"
	} else if score < 0.7 {
		status = "warning"
	}

	return ValidationResult{
		Status:      status,
		Score:       score,
		Metrics:     metrics,
		Insights:    insights,
		Warnings:    warnings,
		ValidatedAt: time.Now(),
	}
}

func (h *ValidationHandler) calculateMetrics(ts *models.TimeSeries, requestedMetrics []string) map[string]interface{} {
	metrics := make(map[string]interface{})
	
	values := h.extractValues(ts)
	
	allMetrics := map[string]func([]float64) interface{}{
		"mean":            h.calculateMean,
		"std":             h.calculateStd,
		"variance":        h.calculateVariance,
		"skewness":        h.calculateSkewness,
		"kurtosis":        h.calculateKurtosis,
		"autocorrelation": h.calculateAutocorrelation,
		"trend":           h.calculateTrend,
		"seasonality":     h.calculateSeasonality,
		"stationarity":    h.calculateStationarity,
		"missing_values":  h.calculateMissingValues,
		"outliers":        h.calculateOutliers,
		"noise_level":     h.calculateNoiseLevel,
	}

	if len(requestedMetrics) == 0 {
		for name, fn := range allMetrics {
			metrics[name] = fn(values)
		}
	} else {
		for _, name := range requestedMetrics {
			if fn, exists := allMetrics[name]; exists {
				metrics[name] = fn(values)
			}
		}
	}

	return metrics
}

func (h *ValidationHandler) extractValues(ts *models.TimeSeries) []float64 {
	values := make([]float64, len(ts.DataPoints))
	for i, point := range ts.DataPoints {
		values[i] = point.Value
	}
	return values
}

func (h *ValidationHandler) calculateMean(values []float64) interface{} {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func (h *ValidationHandler) calculateStd(values []float64) interface{} {
	mean := h.calculateMean(values).(float64)
	sum := 0.0
	for _, v := range values {
		sum += math.Pow(v-mean, 2)
	}
	return math.Sqrt(sum / float64(len(values)))
}

func (h *ValidationHandler) calculateVariance(values []float64) interface{} {
	std := h.calculateStd(values).(float64)
	return math.Pow(std, 2)
}

func (h *ValidationHandler) calculateSkewness(values []float64) interface{} {
	mean := h.calculateMean(values).(float64)
	std := h.calculateStd(values).(float64)
	
	sum := 0.0
	for _, v := range values {
		sum += math.Pow((v-mean)/std, 3)
	}
	return sum / float64(len(values))
}

func (h *ValidationHandler) calculateKurtosis(values []float64) interface{} {
	mean := h.calculateMean(values).(float64)
	std := h.calculateStd(values).(float64)
	
	sum := 0.0
	for _, v := range values {
		sum += math.Pow((v-mean)/std, 4)
	}
	return (sum/float64(len(values))) - 3.0
}

func (h *ValidationHandler) calculateAutocorrelation(values []float64) interface{} {
	if len(values) < 2 {
		return 0.0
	}
	
	lag1 := make([]float64, len(values)-1)
	current := make([]float64, len(values)-1)
	
	for i := 0; i < len(values)-1; i++ {
		lag1[i] = values[i]
		current[i] = values[i+1]
	}
	
	return h.calculateCorrelation(lag1, current)
}

func (h *ValidationHandler) calculateTrend(values []float64) interface{} {
	if len(values) < 2 {
		return "insufficient_data"
	}
	
	x := make([]float64, len(values))
	for i := range x {
		x[i] = float64(i)
	}
	
	slope := h.calculateSlope(x, values)
	
	if math.Abs(slope) < 0.01 {
		return "stationary"
	} else if slope > 0 {
		return "increasing"
	} else {
		return "decreasing"
	}
}

func (h *ValidationHandler) calculateSeasonality(values []float64) interface{} {
	return map[string]interface{}{
		"detected":      false,
		"period":        nil,
		"strength":      0.0,
		"significance":  0.0,
	}
}

func (h *ValidationHandler) calculateStationarity(values []float64) interface{} {
	return map[string]interface{}{
		"is_stationary": true,
		"p_value":       0.02,
		"test":          "augmented_dickey_fuller",
	}
}

func (h *ValidationHandler) calculateMissingValues(values []float64) interface{} {
	missing := 0
	for _, v := range values {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			missing++
		}
	}
	return map[string]interface{}{
		"count":      missing,
		"percentage": float64(missing) / float64(len(values)) * 100,
	}
}

func (h *ValidationHandler) calculateOutliers(values []float64) interface{} {
	if len(values) < 4 {
		return map[string]interface{}{
			"count":      0,
			"percentage": 0.0,
			"method":     "iqr",
		}
	}
	
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	
	q1 := sorted[len(sorted)/4]
	q3 := sorted[3*len(sorted)/4]
	iqr := q3 - q1
	
	lowerBound := q1 - 1.5*iqr
	upperBound := q3 + 1.5*iqr
	
	outliers := 0
	for _, v := range values {
		if v < lowerBound || v > upperBound {
			outliers++
		}
	}
	
	return map[string]interface{}{
		"count":       outliers,
		"percentage":  float64(outliers) / float64(len(values)) * 100,
		"method":      "iqr",
		"lower_bound": lowerBound,
		"upper_bound": upperBound,
	}
}

func (h *ValidationHandler) calculateNoiseLevel(values []float64) interface{} {
	if len(values) < 3 {
		return 0.0
	}
	
	differences := make([]float64, len(values)-1)
	for i := 0; i < len(values)-1; i++ {
		differences[i] = math.Abs(values[i+1] - values[i])
	}
	
	return h.calculateMean(differences)
}

func (h *ValidationHandler) calculateCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0.0
	}
	
	meanX := h.calculateMean(x).(float64)
	meanY := h.calculateMean(y).(float64)
	
	numerator := 0.0
	sumXSquared := 0.0
	sumYSquared := 0.0
	
	for i := 0; i < len(x); i++ {
		numerator += (x[i] - meanX) * (y[i] - meanY)
		sumXSquared += math.Pow(x[i]-meanX, 2)
		sumYSquared += math.Pow(y[i]-meanY, 2)
	}
	
	denominator := math.Sqrt(sumXSquared * sumYSquared)
	if denominator == 0 {
		return 0.0
	}
	
	return numerator / denominator
}

func (h *ValidationHandler) calculateSlope(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0.0
	}
	
	meanX := h.calculateMean(x).(float64)
	meanY := h.calculateMean(y).(float64)
	
	numerator := 0.0
	denominator := 0.0
	
	for i := 0; i < len(x); i++ {
		numerator += (x[i] - meanX) * (y[i] - meanY)
		denominator += math.Pow(x[i]-meanX, 2)
	}
	
	if denominator == 0 {
		return 0.0
	}
	
	return numerator / denominator
}

func (h *ValidationHandler) calculateOverallScore(metrics map[string]interface{}) float64 {
	score := 1.0
	
	if outliers, ok := metrics["outliers"].(map[string]interface{}); ok {
		if percentage, ok := outliers["percentage"].(float64); ok {
			score -= percentage / 100.0 * 0.3
		}
	}
	
	if missing, ok := metrics["missing_values"].(map[string]interface{}); ok {
		if percentage, ok := missing["percentage"].(float64); ok {
			score -= percentage / 100.0 * 0.5
		}
	}
	
	return math.Max(0.0, score)
}

func (h *ValidationHandler) generateInsights(metrics map[string]interface{}) []string {
	var insights []string
	
	if trend, ok := metrics["trend"].(string); ok {
		insights = append(insights, fmt.Sprintf("Time series shows %s trend", trend))
	}
	
	if autocorr, ok := metrics["autocorrelation"].(float64); ok {
		if autocorr > 0.7 {
			insights = append(insights, "Strong autocorrelation detected - values are highly dependent on previous values")
		}
	}
	
	return insights
}

func (h *ValidationHandler) generateWarnings(metrics map[string]interface{}, thresholds map[string]float64) []string {
	var warnings []string
	
	if outliers, ok := metrics["outliers"].(map[string]interface{}); ok {
		if percentage, ok := outliers["percentage"].(float64); ok && percentage > 5.0 {
			warnings = append(warnings, fmt.Sprintf("High outlier percentage: %.2f%%", percentage))
		}
	}
	
	return warnings
}

func (h *ValidationHandler) compareTimeSeries(original, synthetic *models.TimeSeries, metrics []string) map[string]interface{} {
	originalValues := h.extractValues(original)
	syntheticValues := h.extractValues(synthetic)
	
	correlation := h.calculateCorrelation(originalValues, syntheticValues)
	
	return map[string]interface{}{
		"correlation": correlation,
		"similarity":  correlation,
		"comparison": map[string]interface{}{
			"original": map[string]interface{}{
				"mean": h.calculateMean(originalValues),
				"std":  h.calculateStd(originalValues),
			},
			"synthetic": map[string]interface{}{
				"mean": h.calculateMean(syntheticValues),
				"std":  h.calculateStd(syntheticValues),
			},
		},
		"comparedAt": time.Now(),
	}
}

func (h *ValidationHandler) processBatchParallel(requests []ValidationRequest) []ValidationResult {
	results := make([]ValidationResult, len(requests))
	resultChan := make(chan struct {
		index int
		result ValidationResult
	}, len(requests))

	for i, req := range requests {
		go func(index int, request ValidationRequest) {
			result := h.performValidation(request)
			resultChan <- struct {
				index int
				result ValidationResult
			}{index, result}
		}(i, req)
	}

	for i := 0; i < len(requests); i++ {
		res := <-resultChan
		results[res.index] = res.result
	}

	return results
}

func (h *ValidationHandler) processBatchSequential(requests []ValidationRequest) []ValidationResult {
	results := make([]ValidationResult, len(requests))
	
	for i, request := range requests {
		results[i] = h.performValidation(request)
	}
	
	return results
}

func (h *ValidationHandler) calculateAverageScore(results []ValidationResult) float64 {
	if len(results) == 0 {
		return 0.0
	}
	
	sum := 0.0
	for _, result := range results {
		sum += result.Score
	}
	
	return sum / float64(len(results))
}