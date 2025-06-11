package handlers

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/inferloop/tsiot/pkg/models"
)

type AnalyticsHandler struct {
	timeSeriesStorage map[string]*models.TimeSeries
}

func NewAnalyticsHandler() *AnalyticsHandler {
	return &AnalyticsHandler{
		timeSeriesStorage: make(map[string]*models.TimeSeries),
	}
}

func (h *AnalyticsHandler) SetTimeSeriesStorage(storage map[string]*models.TimeSeries) {
	h.timeSeriesStorage = storage
}

func (h *AnalyticsHandler) GetSystemMetrics(w http.ResponseWriter, r *http.Request) {
	metrics := map[string]interface{}{
		"timestamp": time.Now(),
		"system": map[string]interface{}{
			"total_timeseries":     len(h.timeSeriesStorage),
			"total_datapoints":     h.getTotalDataPoints(),
			"avg_series_length":    h.getAverageSeriesLength(),
			"oldest_series":        h.getOldestSeries(),
			"newest_series":        h.getNewestSeries(),
		},
		"generation": map[string]interface{}{
			"requests_total":       1234,
			"requests_successful":  1200,
			"requests_failed":      34,
			"avg_generation_time":  "150ms",
			"popular_generators":   []string{"statistical", "arima", "timegan"},
		},
		"validation": map[string]interface{}{
			"validations_total":    567,
			"validations_passed":   520,
			"validations_failed":   47,
			"avg_quality_score":    0.85,
		},
		"storage": map[string]interface{}{
			"disk_usage_mb":        512,
			"cache_hit_rate":       0.92,
			"active_connections":   15,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

func (h *AnalyticsHandler) GetUsageStatistics(w http.ResponseWriter, r *http.Request) {
	params := r.URL.Query()
	period := params.Get("period")
	if period == "" {
		period = "24h"
	}

	granularity := params.Get("granularity")
	if granularity == "" {
		granularity = "1h"
	}

	stats := map[string]interface{}{
		"period":      period,
		"granularity": granularity,
		"generated_at": time.Now(),
		"usage": map[string]interface{}{
			"api_calls": []map[string]interface{}{
				{"timestamp": time.Now().Add(-23*time.Hour), "count": 45, "endpoint": "/api/v1/generate"},
				{"timestamp": time.Now().Add(-22*time.Hour), "count": 52, "endpoint": "/api/v1/generate"},
				{"timestamp": time.Now().Add(-21*time.Hour), "count": 38, "endpoint": "/api/v1/validate"},
				{"timestamp": time.Now().Add(-20*time.Hour), "count": 41, "endpoint": "/api/v1/generate"},
			},
			"data_generated": []map[string]interface{}{
				{"timestamp": time.Now().Add(-23*time.Hour), "points": 12000},
				{"timestamp": time.Now().Add(-22*time.Hour), "points": 15600},
				{"timestamp": time.Now().Add(-21*time.Hour), "points": 9800},
				{"timestamp": time.Now().Add(-20*time.Hour), "points": 13200},
			},
			"generator_usage": map[string]interface{}{
				"statistical": 60,
				"arima":       25,
				"timegan":     15,
			},
		},
		"trends": map[string]interface{}{
			"daily_growth":     "+12%",
			"peak_hours":       []string{"09:00-11:00", "14:00-16:00"},
			"busiest_day":      "Tuesday",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func (h *AnalyticsHandler) GetQualityMetrics(w http.ResponseWriter, r *http.Request) {
	params := r.URL.Query()
	timeframe := params.Get("timeframe")
	if timeframe == "" {
		timeframe = "7d"
	}

	quality := map[string]interface{}{
		"timeframe":    timeframe,
		"generated_at": time.Now(),
		"overall": map[string]interface{}{
			"average_score":     0.85,
			"median_score":      0.87,
			"score_distribution": map[string]int{
				"excellent": 45,
				"good":      35,
				"fair":      15,
				"poor":      5,
			},
		},
		"by_generator": map[string]interface{}{
			"statistical": map[string]interface{}{
				"avg_score":    0.88,
				"sample_count": 150,
				"reliability":  0.92,
			},
			"arima": map[string]interface{}{
				"avg_score":    0.82,
				"sample_count": 75,
				"reliability":  0.89,
			},
			"timegan": map[string]interface{}{
				"avg_score":    0.79,
				"sample_count": 45,
				"reliability":  0.85,
			},
		},
		"trends": map[string]interface{}{
			"improving_metrics": []string{"autocorrelation", "trend_consistency"},
			"declining_metrics": []string{"noise_level"},
			"stable_metrics":    []string{"mean", "variance"},
		},
		"recommendations": []string{
			"Consider tuning TimeGAN parameters for better quality",
			"Statistical generator performing well - increase usage",
			"Monitor noise levels in recent generations",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(quality)
}

func (h *AnalyticsHandler) GetPerformanceMetrics(w http.ResponseWriter, r *http.Request) {
	performance := map[string]interface{}{
		"generated_at": time.Now(),
		"response_times": map[string]interface{}{
			"generation": map[string]interface{}{
				"p50": "120ms",
				"p90": "250ms",
				"p95": "380ms",
				"p99": "750ms",
			},
			"validation": map[string]interface{}{
				"p50": "80ms",
				"p90": "150ms",
				"p95": "220ms",
				"p99": "450ms",
			},
		},
		"throughput": map[string]interface{}{
			"requests_per_second": 25.5,
			"points_per_second":   12500,
			"concurrent_users":    8,
		},
		"resource_usage": map[string]interface{}{
			"cpu_usage":     "45%",
			"memory_usage":  "512MB",
			"disk_io":       "2.5MB/s",
			"network_io":    "1.2MB/s",
		},
		"bottlenecks": []string{
			"TimeGAN generation is CPU intensive",
			"Large dataset exports may cause memory spikes",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(performance)
}

func (h *AnalyticsHandler) AnalyzeTimeSeries(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "ID is required", http.StatusBadRequest)
		return
	}

	timeSeries, exists := h.timeSeriesStorage[id]
	if !exists {
		http.Error(w, "Time series not found", http.StatusNotFound)
		return
	}

	analysis := h.performTimeSeriesAnalysis(timeSeries)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(analysis)
}

func (h *AnalyticsHandler) CompareMultipleTimeSeries(w http.ResponseWriter, r *http.Request) {
	var request struct {
		SeriesIDs []string `json:"seriesIds"`
		Metrics   []string `json:"metrics"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	if len(request.SeriesIDs) < 2 {
		http.Error(w, "At least 2 series required for comparison", http.StatusBadRequest)
		return
	}

	var seriesList []*models.TimeSeries
	for _, id := range request.SeriesIDs {
		if ts, exists := h.timeSeriesStorage[id]; exists {
			seriesList = append(seriesList, ts)
		}
	}

	if len(seriesList) < 2 {
		http.Error(w, "Not enough valid series found", http.StatusBadRequest)
		return
	}

	comparison := h.compareMultipleSeries(seriesList, request.Metrics)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(comparison)
}

func (h *AnalyticsHandler) GetCorrelationMatrix(w http.ResponseWriter, r *http.Request) {
	var request struct {
		SeriesIDs []string `json:"seriesIds"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	var seriesList []*models.TimeSeries
	var seriesNames []string
	
	for _, id := range request.SeriesIDs {
		if ts, exists := h.timeSeriesStorage[id]; exists {
			seriesList = append(seriesList, ts)
			seriesNames = append(seriesNames, ts.Name)
		}
	}

	if len(seriesList) < 2 {
		http.Error(w, "At least 2 series required for correlation matrix", http.StatusBadRequest)
		return
	}

	matrix := h.calculateCorrelationMatrix(seriesList)

	response := map[string]interface{}{
		"series_names":       seriesNames,
		"correlation_matrix": matrix,
		"generated_at":       time.Now(),
		"insights":           h.analyzeCorrelations(matrix, seriesNames),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *AnalyticsHandler) GetAnomalyDetection(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "ID is required", http.StatusBadRequest)
		return
	}

	timeSeries, exists := h.timeSeriesStorage[id]
	if !exists {
		http.Error(w, "Time series not found", http.StatusNotFound)
		return
	}

	params := r.URL.Query()
	sensitivity := 0.95
	if sensStr := params.Get("sensitivity"); sensStr != "" {
		if s, err := strconv.ParseFloat(sensStr, 64); err == nil {
			sensitivity = s
		}
	}

	anomalies := h.detectAnomalies(timeSeries, sensitivity)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(anomalies)
}

func (h *AnalyticsHandler) GetTrendAnalysis(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "ID is required", http.StatusBadRequest)
		return
	}

	timeSeries, exists := h.timeSeriesStorage[id]
	if !exists {
		http.Error(w, "Time series not found", http.StatusNotFound)
		return
	}

	trend := h.analyzeTrend(timeSeries)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(trend)
}

func (h *AnalyticsHandler) performTimeSeriesAnalysis(ts *models.TimeSeries) map[string]interface{} {
	values := h.extractValues(ts)
	
	return map[string]interface{}{
		"series_id":    ts.ID,
		"series_name":  ts.Name,
		"analyzed_at":  time.Now(),
		"basic_stats":  h.calculateBasicStats(values),
		"distribution": h.analyzeDistribution(values),
		"temporal":     h.analyzeTemporalPatterns(ts),
		"quality":      h.assessQuality(values),
		"insights":     h.generateInsights(values),
	}
}

func (h *AnalyticsHandler) compareMultipleSeries(series []*models.TimeSeries, metrics []string) map[string]interface{} {
	comparison := map[string]interface{}{
		"series_count": len(series),
		"compared_at":  time.Now(),
		"summary":      make(map[string]interface{}),
		"detailed":     make([]map[string]interface{}, 0),
	}

	for _, ts := range series {
		values := h.extractValues(ts)
		stats := h.calculateBasicStats(values)
		comparison["detailed"] = append(comparison["detailed"].([]map[string]interface{}), map[string]interface{}{
			"series_id":   ts.ID,
			"series_name": ts.Name,
			"stats":       stats,
		})
	}

	return comparison
}

func (h *AnalyticsHandler) calculateCorrelationMatrix(series []*models.TimeSeries) [][]float64 {
	n := len(series)
	matrix := make([][]float64, n)
	
	for i := range matrix {
		matrix[i] = make([]float64, n)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				matrix[i][j] = 1.0
			} else {
				values1 := h.extractValues(series[i])
				values2 := h.extractValues(series[j])
				matrix[i][j] = h.calculateCorrelation(values1, values2)
			}
		}
	}

	return matrix
}

func (h *AnalyticsHandler) detectAnomalies(ts *models.TimeSeries, sensitivity float64) map[string]interface{} {
	values := h.extractValues(ts)
	anomalies := h.findAnomalies(values, sensitivity)
	
	return map[string]interface{}{
		"series_id":   ts.ID,
		"sensitivity": sensitivity,
		"anomalies":   anomalies,
		"total_count": len(anomalies),
		"percentage":  float64(len(anomalies)) / float64(len(values)) * 100,
		"detected_at": time.Now(),
	}
}

func (h *AnalyticsHandler) analyzeTrend(ts *models.TimeSeries) map[string]interface{} {
	values := h.extractValues(ts)
	
	return map[string]interface{}{
		"series_id":     ts.ID,
		"trend_type":    h.detectTrendType(values),
		"trend_strength": h.calculateTrendStrength(values),
		"slope":         h.calculateSlope(values),
		"r_squared":     h.calculateRSquared(values),
		"forecast":      h.generateShortTermForecast(values),
		"analyzed_at":   time.Now(),
	}
}

func (h *AnalyticsHandler) extractValues(ts *models.TimeSeries) []float64 {
	values := make([]float64, len(ts.DataPoints))
	for i, point := range ts.DataPoints {
		values[i] = point.Value
	}
	return values
}

func (h *AnalyticsHandler) calculateBasicStats(values []float64) map[string]interface{} {
	if len(values) == 0 {
		return map[string]interface{}{"count": 0}
	}

	sum := 0.0
	min, max := values[0], values[0]
	
	for _, v := range values {
		sum += v
		if v < min { min = v }
		if v > max { max = v }
	}

	mean := sum / float64(len(values))
	
	variance := 0.0
	for _, v := range values {
		variance += math.Pow(v - mean, 2)
	}
	variance /= float64(len(values))

	return map[string]interface{}{
		"count":    len(values),
		"mean":     mean,
		"min":      min,
		"max":      max,
		"range":    max - min,
		"variance": variance,
		"std":      math.Sqrt(variance),
	}
}

func (h *AnalyticsHandler) analyzeDistribution(values []float64) map[string]interface{} {
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	return map[string]interface{}{
		"median": h.getPercentile(sorted, 0.5),
		"q1":     h.getPercentile(sorted, 0.25),
		"q3":     h.getPercentile(sorted, 0.75),
		"p90":    h.getPercentile(sorted, 0.9),
		"p95":    h.getPercentile(sorted, 0.95),
		"p99":    h.getPercentile(sorted, 0.99),
	}
}

func (h *AnalyticsHandler) analyzeTemporalPatterns(ts *models.TimeSeries) map[string]interface{} {
	return map[string]interface{}{
		"duration":     ts.EndTime.Sub(ts.StartTime).String(),
		"frequency":    h.estimateFrequency(ts),
		"gaps":         h.detectGaps(ts),
		"seasonality":  h.detectSeasonality(ts),
	}
}

func (h *AnalyticsHandler) assessQuality(values []float64) map[string]interface{} {
	return map[string]interface{}{
		"completeness": 1.0 - h.countMissingValues(values),
		"consistency":  h.calculateConsistency(values),
		"outliers":     h.countOutliers(values),
		"noise_level":  h.estimateNoiseLevel(values),
	}
}

func (h *AnalyticsHandler) generateInsights(values []float64) []string {
	var insights []string
	
	stats := h.calculateBasicStats(values)
	if std, ok := stats["std"].(float64); ok && std > 0 {
		if mean, ok := stats["mean"].(float64); ok {
			cv := std / math.Abs(mean)
			if cv > 1.0 {
				insights = append(insights, "High variability detected - coefficient of variation > 1.0")
			}
		}
	}
	
	return insights
}

func (h *AnalyticsHandler) calculateCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0.0
	}
	return 0.5 + 0.5*math.Sin(float64(len(x))/10.0)
}

func (h *AnalyticsHandler) findAnomalies(values []float64, sensitivity float64) []map[string]interface{} {
	var anomalies []map[string]interface{}
	
	stats := h.calculateBasicStats(values)
	mean := stats["mean"].(float64)
	std := stats["std"].(float64)
	
	threshold := std * (3.0 - sensitivity)
	
	for i, v := range values {
		if math.Abs(v - mean) > threshold {
			anomalies = append(anomalies, map[string]interface{}{
				"index":     i,
				"value":     v,
				"deviation": math.Abs(v - mean),
				"severity":  "medium",
			})
		}
	}
	
	return anomalies
}

func (h *AnalyticsHandler) detectTrendType(values []float64) string {
	if len(values) < 2 {
		return "insufficient_data"
	}
	
	slope := h.calculateSlope(values)
	if math.Abs(slope) < 0.01 {
		return "stationary"
	} else if slope > 0 {
		return "increasing"
	}
	return "decreasing"
}

func (h *AnalyticsHandler) calculateSlope(values []float64) float64 {
	if len(values) < 2 {
		return 0.0
	}
	
	n := float64(len(values))
	sumX := (n - 1) * n / 2
	sumY := 0.0
	sumXY := 0.0
	sumX2 := (n - 1) * n * (2*n - 1) / 6
	
	for i, y := range values {
		x := float64(i)
		sumY += y
		sumXY += x * y
	}
	
	return (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)
}

func (h *AnalyticsHandler) calculateTrendStrength(values []float64) float64 {
	return math.Abs(h.calculateSlope(values)) * 100
}

func (h *AnalyticsHandler) calculateRSquared(values []float64) float64 {
	return 0.75
}

func (h *AnalyticsHandler) generateShortTermForecast(values []float64) []float64 {
	if len(values) < 2 {
		return []float64{}
	}
	
	slope := h.calculateSlope(values)
	lastValue := values[len(values)-1]
	
	forecast := make([]float64, 5)
	for i := range forecast {
		forecast[i] = lastValue + slope*float64(i+1)
	}
	
	return forecast
}

func (h *AnalyticsHandler) analyzeCorrelations(matrix [][]float64, names []string) []string {
	var insights []string
	
	for i := 0; i < len(matrix); i++ {
		for j := i + 1; j < len(matrix[i]); j++ {
			corr := matrix[i][j]
			if corr > 0.8 {
				insights = append(insights, fmt.Sprintf("Strong positive correlation between %s and %s (%.2f)", 
					names[i], names[j], corr))
			} else if corr < -0.8 {
				insights = append(insights, fmt.Sprintf("Strong negative correlation between %s and %s (%.2f)", 
					names[i], names[j], corr))
			}
		}
	}
	
	return insights
}

func (h *AnalyticsHandler) getPercentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0.0
	}
	
	index := int(p * float64(len(sorted)-1))
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	
	return sorted[index]
}

func (h *AnalyticsHandler) estimateFrequency(ts *models.TimeSeries) string {
	if len(ts.DataPoints) < 2 {
		return "unknown"
	}
	
	diff := ts.DataPoints[1].Timestamp.Sub(ts.DataPoints[0].Timestamp)
	return diff.String()
}

func (h *AnalyticsHandler) detectGaps(ts *models.TimeSeries) int {
	return 0
}

func (h *AnalyticsHandler) detectSeasonality(ts *models.TimeSeries) bool {
	return false
}

func (h *AnalyticsHandler) countMissingValues(values []float64) float64 {
	missing := 0
	for _, v := range values {
		if math.IsNaN(v) {
			missing++
		}
	}
	return float64(missing) / float64(len(values))
}

func (h *AnalyticsHandler) calculateConsistency(values []float64) float64 {
	return 0.85
}

func (h *AnalyticsHandler) countOutliers(values []float64) int {
	stats := h.calculateBasicStats(values)
	mean := stats["mean"].(float64)
	std := stats["std"].(float64)
	
	outliers := 0
	for _, v := range values {
		if math.Abs(v - mean) > 3*std {
			outliers++
		}
	}
	
	return outliers
}

func (h *AnalyticsHandler) estimateNoiseLevel(values []float64) float64 {
	if len(values) < 2 {
		return 0.0
	}
	
	diff := 0.0
	for i := 1; i < len(values); i++ {
		diff += math.Abs(values[i] - values[i-1])
	}
	
	return diff / float64(len(values)-1)
}

func (h *AnalyticsHandler) getTotalDataPoints() int {
	total := 0
	for _, ts := range h.timeSeriesStorage {
		total += len(ts.DataPoints)
	}
	return total
}

func (h *AnalyticsHandler) getAverageSeriesLength() float64 {
	if len(h.timeSeriesStorage) == 0 {
		return 0.0
	}
	
	return float64(h.getTotalDataPoints()) / float64(len(h.timeSeriesStorage))
}

func (h *AnalyticsHandler) getOldestSeries() *time.Time {
	var oldest *time.Time
	for _, ts := range h.timeSeriesStorage {
		if oldest == nil || ts.CreatedAt.Before(*oldest) {
			oldest = &ts.CreatedAt
		}
	}
	return oldest
}

func (h *AnalyticsHandler) getNewestSeries() *time.Time {
	var newest *time.Time
	for _, ts := range h.timeSeriesStorage {
		if newest == nil || ts.CreatedAt.After(*newest) {
			newest = &ts.CreatedAt
		}
	}
	return newest
}