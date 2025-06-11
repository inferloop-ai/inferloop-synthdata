package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/inferloop/tsiot/pkg/models"
)

type TimeSeriesHandler struct {
	storage map[string]*models.TimeSeries
}

func NewTimeSeriesHandler() *TimeSeriesHandler {
	return &TimeSeriesHandler{
		storage: make(map[string]*models.TimeSeries),
	}
}

func (h *TimeSeriesHandler) CreateTimeSeries(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Name        string                 `json:"name"`
		Description string                 `json:"description"`
		DataPoints  []models.DataPoint     `json:"dataPoints"`
		Metadata    map[string]interface{} `json:"metadata"`
		Tags        []string               `json:"tags"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	if request.Name == "" {
		http.Error(w, "Name is required", http.StatusBadRequest)
		return
	}

	if len(request.DataPoints) == 0 {
		http.Error(w, "DataPoints are required", http.StatusBadRequest)
		return
	}

	id := h.generateID()
	
	timeSeries := &models.TimeSeries{
		ID:          id,
		Name:        request.Name,
		Description: request.Description,
		DataPoints:  request.DataPoints,
		StartTime:   request.DataPoints[0].Timestamp,
		EndTime:     request.DataPoints[len(request.DataPoints)-1].Timestamp,
		Metadata:    request.Metadata,
		Tags:        request.Tags,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	h.storage[id] = timeSeries

	response := map[string]interface{}{
		"status":     "created",
		"id":         id,
		"timeSeries": timeSeries,
		"message":    "Time series created successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

func (h *TimeSeriesHandler) GetTimeSeries(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "ID is required", http.StatusBadRequest)
		return
	}

	timeSeries, exists := h.storage[id]
	if !exists {
		http.Error(w, "Time series not found", http.StatusNotFound)
		return
	}

	params := r.URL.Query()
	includeData := params.Get("include_data") != "false"
	limit := 0
	if limitStr := params.Get("limit"); limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil {
			limit = l
		}
	}

	response := h.formatTimeSeriesResponse(timeSeries, includeData, limit)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *TimeSeriesHandler) UpdateTimeSeries(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "ID is required", http.StatusBadRequest)
		return
	}

	timeSeries, exists := h.storage[id]
	if !exists {
		http.Error(w, "Time series not found", http.StatusNotFound)
		return
	}

	var updateRequest struct {
		Name        *string                `json:"name,omitempty"`
		Description *string                `json:"description,omitempty"`
		Metadata    map[string]interface{} `json:"metadata,omitempty"`
		Tags        []string               `json:"tags,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&updateRequest); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	if updateRequest.Name != nil {
		timeSeries.Name = *updateRequest.Name
	}
	if updateRequest.Description != nil {
		timeSeries.Description = *updateRequest.Description
	}
	if updateRequest.Metadata != nil {
		timeSeries.Metadata = updateRequest.Metadata
	}
	if updateRequest.Tags != nil {
		timeSeries.Tags = updateRequest.Tags
	}

	timeSeries.UpdatedAt = time.Now()

	response := map[string]interface{}{
		"status":     "updated",
		"timeSeries": timeSeries,
		"message":    "Time series updated successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *TimeSeriesHandler) DeleteTimeSeries(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "ID is required", http.StatusBadRequest)
		return
	}

	_, exists := h.storage[id]
	if !exists {
		http.Error(w, "Time series not found", http.StatusNotFound)
		return
	}

	delete(h.storage, id)

	response := map[string]interface{}{
		"status":  "deleted",
		"id":      id,
		"message": "Time series deleted successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *TimeSeriesHandler) ListTimeSeries(w http.ResponseWriter, r *http.Request) {
	params := r.URL.Query()
	
	limit := 50
	if limitStr := params.Get("limit"); limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 {
			limit = l
		}
	}

	offset := 0
	if offsetStr := params.Get("offset"); offsetStr != "" {
		if o, err := strconv.Atoi(offsetStr); err == nil && o >= 0 {
			offset = o
		}
	}

	tag := params.Get("tag")
	search := params.Get("search")
	includeData := params.Get("include_data") == "true"

	var filtered []*models.TimeSeries
	for _, ts := range h.storage {
		if h.matchesFilters(ts, tag, search) {
			filtered = append(filtered, ts)
		}
	}

	total := len(filtered)
	start := offset
	end := offset + limit
	if start > total {
		start = total
	}
	if end > total {
		end = total
	}

	var results []interface{}
	for i := start; i < end; i++ {
		results = append(results, h.formatTimeSeriesResponse(filtered[i], includeData, 0))
	}

	response := map[string]interface{}{
		"timeSeries": results,
		"pagination": map[string]interface{}{
			"total":  total,
			"limit":  limit,
			"offset": offset,
			"count":  len(results),
		},
		"filters": map[string]interface{}{
			"tag":          tag,
			"search":       search,
			"include_data": includeData,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *TimeSeriesHandler) GetTimeSeriesData(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "ID is required", http.StatusBadRequest)
		return
	}

	timeSeries, exists := h.storage[id]
	if !exists {
		http.Error(w, "Time series not found", http.StatusNotFound)
		return
	}

	params := r.URL.Query()
	
	var startTime, endTime *time.Time
	if startStr := params.Get("start"); startStr != "" {
		if t, err := time.Parse(time.RFC3339, startStr); err == nil {
			startTime = &t
		}
	}
	if endStr := params.Get("end"); endStr != "" {
		if t, err := time.Parse(time.RFC3339, endStr); err == nil {
			endTime = &t
		}
	}

	limit := 0
	if limitStr := params.Get("limit"); limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil {
			limit = l
		}
	}

	format := params.Get("format")
	if format == "" {
		format = "json"
	}

	var filteredPoints []models.DataPoint
	for _, point := range timeSeries.DataPoints {
		if startTime != nil && point.Timestamp.Before(*startTime) {
			continue
		}
		if endTime != nil && point.Timestamp.After(*endTime) {
			continue
		}
		filteredPoints = append(filteredPoints, point)
		
		if limit > 0 && len(filteredPoints) >= limit {
			break
		}
	}

	switch format {
	case "csv":
		h.respondWithCSV(w, filteredPoints)
	case "json":
		fallthrough
	default:
		response := map[string]interface{}{
			"seriesId":   id,
			"dataPoints": filteredPoints,
			"count":      len(filteredPoints),
			"timeRange": map[string]interface{}{
				"start": startTime,
				"end":   endTime,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}
}

func (h *TimeSeriesHandler) AppendData(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "ID is required", http.StatusBadRequest)
		return
	}

	timeSeries, exists := h.storage[id]
	if !exists {
		http.Error(w, "Time series not found", http.StatusNotFound)
		return
	}

	var request struct {
		DataPoints []models.DataPoint `json:"dataPoints"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	if len(request.DataPoints) == 0 {
		http.Error(w, "DataPoints are required", http.StatusBadRequest)
		return
	}

	timeSeries.DataPoints = append(timeSeries.DataPoints, request.DataPoints...)
	
	if len(timeSeries.DataPoints) > 0 {
		timeSeries.StartTime = timeSeries.DataPoints[0].Timestamp
		timeSeries.EndTime = timeSeries.DataPoints[len(timeSeries.DataPoints)-1].Timestamp
	}
	
	timeSeries.UpdatedAt = time.Now()

	response := map[string]interface{}{
		"status":        "appended",
		"pointsAdded":   len(request.DataPoints),
		"totalPoints":   len(timeSeries.DataPoints),
		"updatedRange": map[string]interface{}{
			"start": timeSeries.StartTime,
			"end":   timeSeries.EndTime,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *TimeSeriesHandler) GetTimeSeriesStats(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "ID is required", http.StatusBadRequest)
		return
	}

	timeSeries, exists := h.storage[id]
	if !exists {
		http.Error(w, "Time series not found", http.StatusNotFound)
		return
	}

	stats := h.calculateStats(timeSeries)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func (h *TimeSeriesHandler) SearchTimeSeries(w http.ResponseWriter, r *http.Request) {
	params := r.URL.Query()
	query := params.Get("q")
	
	if query == "" {
		http.Error(w, "Query parameter 'q' is required", http.StatusBadRequest)
		return
	}

	var results []*models.TimeSeries
	for _, ts := range h.storage {
		if h.matchesSearch(ts, query) {
			results = append(results, ts)
		}
	}

	response := map[string]interface{}{
		"query":   query,
		"results": results,
		"count":   len(results),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *TimeSeriesHandler) generateID() string {
	return fmt.Sprintf("ts_%d", time.Now().UnixNano())
}

func (h *TimeSeriesHandler) formatTimeSeriesResponse(ts *models.TimeSeries, includeData bool, limit int) map[string]interface{} {
	response := map[string]interface{}{
		"id":          ts.ID,
		"name":        ts.Name,
		"description": ts.Description,
		"startTime":   ts.StartTime,
		"endTime":     ts.EndTime,
		"metadata":    ts.Metadata,
		"tags":        ts.Tags,
		"createdAt":   ts.CreatedAt,
		"updatedAt":   ts.UpdatedAt,
		"pointCount":  len(ts.DataPoints),
	}

	if includeData {
		dataPoints := ts.DataPoints
		if limit > 0 && len(dataPoints) > limit {
			dataPoints = dataPoints[:limit]
			response["truncated"] = true
			response["displayedPoints"] = limit
		}
		response["dataPoints"] = dataPoints
	}

	return response
}

func (h *TimeSeriesHandler) matchesFilters(ts *models.TimeSeries, tag, search string) bool {
	if tag != "" {
		found := false
		for _, t := range ts.Tags {
			if t == tag {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	if search != "" {
		return h.matchesSearch(ts, search)
	}

	return true
}

func (h *TimeSeriesHandler) matchesSearch(ts *models.TimeSeries, query string) bool {
	searchFields := []string{
		ts.Name,
		ts.Description,
	}

	for _, field := range searchFields {
		if len(field) >= len(query) {
			return true
		}
	}

	for _, tag := range ts.Tags {
		if len(tag) >= len(query) {
			return true
		}
	}

	return false
}

func (h *TimeSeriesHandler) calculateStats(ts *models.TimeSeries) map[string]interface{} {
	if len(ts.DataPoints) == 0 {
		return map[string]interface{}{
			"count": 0,
		}
	}

	values := make([]float64, len(ts.DataPoints))
	for i, point := range ts.DataPoints {
		values[i] = point.Value
	}

	min, max := values[0], values[0]
	sum := 0.0
	
	for _, v := range values {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
		sum += v
	}

	mean := sum / float64(len(values))
	
	sumSquaredDiff := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquaredDiff += diff * diff
	}
	variance := sumSquaredDiff / float64(len(values))

	duration := ts.EndTime.Sub(ts.StartTime)

	return map[string]interface{}{
		"count":    len(ts.DataPoints),
		"min":      min,
		"max":      max,
		"mean":     mean,
		"variance": variance,
		"sum":      sum,
		"range":    max - min,
		"duration": duration.String(),
		"timeRange": map[string]interface{}{
			"start": ts.StartTime,
			"end":   ts.EndTime,
		},
	}
}

func (h *TimeSeriesHandler) respondWithCSV(w http.ResponseWriter, points []models.DataPoint) {
	w.Header().Set("Content-Type", "text/csv")
	w.Header().Set("Content-Disposition", "attachment; filename=timeseries.csv")
	
	w.Write([]byte("timestamp,value\n"))
	
	for _, point := range points {
		line := fmt.Sprintf("%s,%.6f\n", point.Timestamp.Format(time.RFC3339), point.Value)
		w.Write([]byte(line))
	}
}

func (h *TimeSeriesHandler) GetStorage() map[string]*models.TimeSeries {
	return h.storage
}