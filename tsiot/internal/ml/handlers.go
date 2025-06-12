package ml

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// ModelRegistryHandler provides HTTP handlers for the model registry
type ModelRegistryHandler struct {
	registry *ModelRegistry
	logger   *logrus.Logger
}

// NewModelRegistryHandler creates a new model registry handler
func NewModelRegistryHandler(registry *ModelRegistry, logger *logrus.Logger) *ModelRegistryHandler {
	return &ModelRegistryHandler{
		registry: registry,
		logger:   logger,
	}
}

// RegisterRoutes registers HTTP routes for the model registry
func (mrh *ModelRegistryHandler) RegisterRoutes(mux *http.ServeMux) {
	// Model management routes
	mux.HandleFunc("/api/ml/models", mrh.handleModels)
	mux.HandleFunc("/api/ml/models/", mrh.handleModel)
	mux.HandleFunc("/api/ml/models/{id}/versions", mrh.handleModelVersions)
	mux.HandleFunc("/api/ml/models/{id}/versions/", mrh.handleModelVersion)
	
	// Deployment routes
	mux.HandleFunc("/api/ml/deployments", mrh.handleDeployments)
	mux.HandleFunc("/api/ml/deployments/", mrh.handleDeployment)
	
	// A/B testing routes
	mux.HandleFunc("/api/ml/experiments", mrh.handleExperiments)
	mux.HandleFunc("/api/ml/experiments/", mrh.handleExperiment)
	
	// Registry metrics
	mux.HandleFunc("/api/ml/metrics", mrh.handleMetrics)
	
	// Model prediction endpoints
	mux.HandleFunc("/api/ml/predict/", mrh.handlePredict)
}

// handleModels handles requests for model collection
func (mrh *ModelRegistryHandler) handleModels(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		mrh.listModels(w, r)
	case http.MethodPost:
		mrh.createModel(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleModel handles requests for individual models
func (mrh *ModelRegistryHandler) handleModel(w http.ResponseWriter, r *http.Request) {
	// Extract model ID from URL path
	path := strings.TrimPrefix(r.URL.Path, "/api/ml/models/")
	modelID := strings.Split(path, "/")[0]
	
	if modelID == "" {
		http.Error(w, "Model ID required", http.StatusBadRequest)
		return
	}

	switch r.Method {
	case http.MethodGet:
		mrh.getModel(w, r, modelID)
	case http.MethodPut:
		mrh.updateModel(w, r, modelID)
	case http.MethodDelete:
		mrh.deleteModel(w, r, modelID)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleModelVersions handles requests for model versions collection
func (mrh *ModelRegistryHandler) handleModelVersions(w http.ResponseWriter, r *http.Request) {
	// Extract model ID from URL path
	path := strings.TrimPrefix(r.URL.Path, "/api/ml/models/")
	modelID := strings.Split(path, "/")[0]
	
	switch r.Method {
	case http.MethodGet:
		mrh.listModelVersions(w, r, modelID)
	case http.MethodPost:
		mrh.createModelVersion(w, r, modelID)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleModelVersion handles requests for individual model versions
func (mrh *ModelRegistryHandler) handleModelVersion(w http.ResponseWriter, r *http.Request) {
	// Extract model ID and version ID from URL path
	path := strings.TrimPrefix(r.URL.Path, "/api/ml/models/")
	parts := strings.Split(path, "/")
	if len(parts) < 3 {
		http.Error(w, "Invalid URL path", http.StatusBadRequest)
		return
	}
	
	modelID := parts[0]
	versionID := parts[2]

	switch r.Method {
	case http.MethodGet:
		mrh.getModelVersion(w, r, modelID, versionID)
	case http.MethodPut:
		mrh.updateModelVersion(w, r, modelID, versionID)
	case http.MethodDelete:
		mrh.deleteModelVersion(w, r, modelID, versionID)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleDeployments handles requests for deployments collection
func (mrh *ModelRegistryHandler) handleDeployments(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		mrh.listDeployments(w, r)
	case http.MethodPost:
		mrh.createDeployment(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleDeployment handles requests for individual deployments
func (mrh *ModelRegistryHandler) handleDeployment(w http.ResponseWriter, r *http.Request) {
	// Extract deployment ID from URL path
	path := strings.TrimPrefix(r.URL.Path, "/api/ml/deployments/")
	deploymentID := strings.Split(path, "/")[0]
	
	if deploymentID == "" {
		http.Error(w, "Deployment ID required", http.StatusBadRequest)
		return
	}

	switch r.Method {
	case http.MethodGet:
		mrh.getDeployment(w, r, deploymentID)
	case http.MethodPut:
		mrh.updateDeployment(w, r, deploymentID)
	case http.MethodDelete:
		mrh.deleteDeployment(w, r, deploymentID)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleExperiments handles requests for experiments collection
func (mrh *ModelRegistryHandler) handleExperiments(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		mrh.listExperiments(w, r)
	case http.MethodPost:
		mrh.createExperiment(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleExperiment handles requests for individual experiments
func (mrh *ModelRegistryHandler) handleExperiment(w http.ResponseWriter, r *http.Request) {
	// Extract experiment ID from URL path
	path := strings.TrimPrefix(r.URL.Path, "/api/ml/experiments/")
	experimentID := strings.Split(path, "/")[0]
	
	if experimentID == "" {
		http.Error(w, "Experiment ID required", http.StatusBadRequest)
		return
	}

	switch r.Method {
	case http.MethodGet:
		mrh.getExperiment(w, r, experimentID)
	case http.MethodPut:
		mrh.updateExperiment(w, r, experimentID)
	case http.MethodDelete:
		mrh.deleteExperiment(w, r, experimentID)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleMetrics handles requests for registry metrics
func (mrh *ModelRegistryHandler) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	mrh.registry.mu.RLock()
	metrics := mrh.registry.metrics
	mrh.registry.mu.RUnlock()

	mrh.writeJSON(w, metrics)
}

// handlePredict handles model prediction requests
func (mrh *ModelRegistryHandler) handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract model ID from URL path
	path := strings.TrimPrefix(r.URL.Path, "/api/ml/predict/")
	modelID := strings.Split(path, "/")[0]
	
	if modelID == "" {
		http.Error(w, "Model ID required", http.StatusBadRequest)
		return
	}

	mrh.predict(w, r, modelID)
}

// Implementation of specific handlers

func (mrh *ModelRegistryHandler) listModels(w http.ResponseWriter, r *http.Request) {
	// Parse query parameters
	query := r.URL.Query()
	modelType := query.Get("type")
	framework := query.Get("framework")
	status := query.Get("status")
	limit := 50
	offset := 0
	
	if l := query.Get("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil {
			limit = parsed
		}
	}
	
	if o := query.Get("offset"); o != "" {
		if parsed, err := strconv.Atoi(o); err == nil {
			offset = parsed
		}
	}

	mrh.registry.mu.RLock()
	defer mrh.registry.mu.RUnlock()

	var models []*RegisteredModel
	count := 0
	
	for _, model := range mrh.registry.models {
		// Apply filters
		if modelType != "" && string(model.Type) != modelType {
			continue
		}
		if framework != "" && string(model.Framework) != framework {
			continue
		}
		if status != "" && string(model.Status) != status {
			continue
		}
		
		// Apply pagination
		if count < offset {
			count++
			continue
		}
		if len(models) >= limit {
			break
		}
		
		models = append(models, model)
		count++
	}

	response := map[string]interface{}{
		"models": models,
		"total":  len(mrh.registry.models),
		"limit":  limit,
		"offset": offset,
	}

	mrh.writeJSON(w, response)
}

func (mrh *ModelRegistryHandler) createModel(w http.ResponseWriter, r *http.Request) {
	var model RegisteredModel
	if err := json.NewDecoder(r.Body).Decode(&model); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Generate ID if not provided
	if model.ID == "" {
		model.ID = fmt.Sprintf("model_%d", time.Now().UnixNano())
	}

	if err := mrh.registry.RegisterModel(r.Context(), &model); err != nil {
		http.Error(w, "Failed to register model: "+err.Error(), http.StatusBadRequest)
		return
	}

	w.WriteHeader(http.StatusCreated)
	mrh.writeJSON(w, model)
}

func (mrh *ModelRegistryHandler) getModel(w http.ResponseWriter, r *http.Request, modelID string) {
	mrh.registry.mu.RLock()
	model, exists := mrh.registry.models[modelID]
	mrh.registry.mu.RUnlock()

	if !exists {
		http.Error(w, "Model not found", http.StatusNotFound)
		return
	}

	mrh.writeJSON(w, model)
}

func (mrh *ModelRegistryHandler) updateModel(w http.ResponseWriter, r *http.Request, modelID string) {
	var updateData map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&updateData); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	mrh.registry.mu.Lock()
	defer mrh.registry.mu.Unlock()

	model, exists := mrh.registry.models[modelID]
	if !exists {
		http.Error(w, "Model not found", http.StatusNotFound)
		return
	}

	// Update fields
	if name, ok := updateData["name"].(string); ok {
		model.Name = name
	}
	if description, ok := updateData["description"].(string); ok {
		model.Description = description
	}
	if tags, ok := updateData["tags"].([]interface{}); ok {
		model.Tags = make([]string, len(tags))
		for i, tag := range tags {
			if s, ok := tag.(string); ok {
				model.Tags[i] = s
			}
		}
	}

	model.UpdatedAt = time.Now()

	mrh.writeJSON(w, model)
}

func (mrh *ModelRegistryHandler) deleteModel(w http.ResponseWriter, r *http.Request, modelID string) {
	mrh.registry.mu.Lock()
	defer mrh.registry.mu.Unlock()

	if _, exists := mrh.registry.models[modelID]; !exists {
		http.Error(w, "Model not found", http.StatusNotFound)
		return
	}

	delete(mrh.registry.models, modelID)

	w.WriteHeader(http.StatusNoContent)
}

func (mrh *ModelRegistryHandler) listModelVersions(w http.ResponseWriter, r *http.Request, modelID string) {
	mrh.registry.mu.RLock()
	model, exists := mrh.registry.models[modelID]
	mrh.registry.mu.RUnlock()

	if !exists {
		http.Error(w, "Model not found", http.StatusNotFound)
		return
	}

	response := map[string]interface{}{
		"model_id": modelID,
		"versions": model.Versions,
		"total":    len(model.Versions),
	}

	mrh.writeJSON(w, response)
}

func (mrh *ModelRegistryHandler) createModelVersion(w http.ResponseWriter, r *http.Request, modelID string) {
	var version ModelVersion
	if err := json.NewDecoder(r.Body).Decode(&version); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	version.ModelID = modelID
	
	// Generate ID if not provided
	if version.ID == "" {
		version.ID = fmt.Sprintf("version_%d", time.Now().UnixNano())
	}

	// For this mock implementation, we'll create a version without artifact
	if err := mrh.registry.CreateModelVersion(r.Context(), &version, nil); err != nil {
		http.Error(w, "Failed to create version: "+err.Error(), http.StatusBadRequest)
		return
	}

	w.WriteHeader(http.StatusCreated)
	mrh.writeJSON(w, version)
}

func (mrh *ModelRegistryHandler) getModelVersion(w http.ResponseWriter, r *http.Request, modelID, versionID string) {
	mrh.registry.mu.RLock()
	model, exists := mrh.registry.models[modelID]
	mrh.registry.mu.RUnlock()

	if !exists {
		http.Error(w, "Model not found", http.StatusNotFound)
		return
	}

	for _, version := range model.Versions {
		if version.ID == versionID {
			mrh.writeJSON(w, version)
			return
		}
	}

	http.Error(w, "Version not found", http.StatusNotFound)
}

func (mrh *ModelRegistryHandler) updateModelVersion(w http.ResponseWriter, r *http.Request, modelID, versionID string) {
	var updateData map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&updateData); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	mrh.registry.mu.Lock()
	defer mrh.registry.mu.Unlock()

	model, exists := mrh.registry.models[modelID]
	if !exists {
		http.Error(w, "Model not found", http.StatusNotFound)
		return
	}

	for _, version := range model.Versions {
		if version.ID == versionID {
			// Update fields
			if description, ok := updateData["description"].(string); ok {
				version.Description = description
			}
			if isProduction, ok := updateData["is_production"].(bool); ok {
				version.IsProduction = isProduction
			}
			if isStaging, ok := updateData["is_staging"].(bool); ok {
				version.IsStaging = isStaging
			}

			mrh.writeJSON(w, version)
			return
		}
	}

	http.Error(w, "Version not found", http.StatusNotFound)
}

func (mrh *ModelRegistryHandler) deleteModelVersion(w http.ResponseWriter, r *http.Request, modelID, versionID string) {
	mrh.registry.mu.Lock()
	defer mrh.registry.mu.Unlock()

	model, exists := mrh.registry.models[modelID]
	if !exists {
		http.Error(w, "Model not found", http.StatusNotFound)
		return
	}

	for i, version := range model.Versions {
		if version.ID == versionID {
			// Remove version from slice
			model.Versions = append(model.Versions[:i], model.Versions[i+1:]...)
			w.WriteHeader(http.StatusNoContent)
			return
		}
	}

	http.Error(w, "Version not found", http.StatusNotFound)
}

func (mrh *ModelRegistryHandler) listDeployments(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	environment := query.Get("environment")
	status := query.Get("status")

	mrh.registry.mu.RLock()
	defer mrh.registry.mu.RUnlock()

	var deployments []*ModelDeployment
	for _, deployment := range mrh.registry.deployments {
		// Apply filters
		if environment != "" && deployment.Environment != environment {
			continue
		}
		if status != "" && string(deployment.Status) != status {
			continue
		}
		
		deployments = append(deployments, deployment)
	}

	response := map[string]interface{}{
		"deployments": deployments,
		"total":       len(deployments),
	}

	mrh.writeJSON(w, response)
}

func (mrh *ModelRegistryHandler) createDeployment(w http.ResponseWriter, r *http.Request) {
	var deployment ModelDeployment
	if err := json.NewDecoder(r.Body).Decode(&deployment); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Generate ID if not provided
	if deployment.ID == "" {
		deployment.ID = fmt.Sprintf("deployment_%d", time.Now().UnixNano())
	}

	if err := mrh.registry.DeployModel(r.Context(), &deployment); err != nil {
		http.Error(w, "Failed to deploy model: "+err.Error(), http.StatusBadRequest)
		return
	}

	w.WriteHeader(http.StatusCreated)
	mrh.writeJSON(w, deployment)
}

func (mrh *ModelRegistryHandler) getDeployment(w http.ResponseWriter, r *http.Request, deploymentID string) {
	mrh.registry.mu.RLock()
	deployment, exists := mrh.registry.deployments[deploymentID]
	mrh.registry.mu.RUnlock()

	if !exists {
		http.Error(w, "Deployment not found", http.StatusNotFound)
		return
	}

	mrh.writeJSON(w, deployment)
}

func (mrh *ModelRegistryHandler) updateDeployment(w http.ResponseWriter, r *http.Request, deploymentID string) {
	var updateData map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&updateData); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	mrh.registry.mu.Lock()
	defer mrh.registry.mu.Unlock()

	deployment, exists := mrh.registry.deployments[deploymentID]
	if !exists {
		http.Error(w, "Deployment not found", http.StatusNotFound)
		return
	}

	// Update status if provided
	if status, ok := updateData["status"].(string); ok {
		deployment.Status = DeploymentStatus(status)
	}

	mrh.writeJSON(w, deployment)
}

func (mrh *ModelRegistryHandler) deleteDeployment(w http.ResponseWriter, r *http.Request, deploymentID string) {
	mrh.registry.mu.Lock()
	defer mrh.registry.mu.Unlock()

	deployment, exists := mrh.registry.deployments[deploymentID]
	if !exists {
		http.Error(w, "Deployment not found", http.StatusNotFound)
		return
	}

	deployment.Status = DeploymentStatusRetired
	retiredAt := time.Now()
	deployment.RetiredAt = &retiredAt

	w.WriteHeader(http.StatusNoContent)
}

func (mrh *ModelRegistryHandler) listExperiments(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	status := query.Get("status")

	mrh.registry.mu.RLock()
	defer mrh.registry.mu.RUnlock()

	var experiments []*ABTestExperiment
	for _, experiment := range mrh.registry.experiments {
		// Apply filters
		if status != "" && string(experiment.Status) != status {
			continue
		}
		
		experiments = append(experiments, experiment)
	}

	response := map[string]interface{}{
		"experiments": experiments,
		"total":       len(experiments),
	}

	mrh.writeJSON(w, response)
}

func (mrh *ModelRegistryHandler) createExperiment(w http.ResponseWriter, r *http.Request) {
	var experiment ABTestExperiment
	if err := json.NewDecoder(r.Body).Decode(&experiment); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Generate ID if not provided
	if experiment.ID == "" {
		experiment.ID = fmt.Sprintf("experiment_%d", time.Now().UnixNano())
	}

	if err := mrh.registry.CreateABTest(r.Context(), &experiment); err != nil {
		http.Error(w, "Failed to create experiment: "+err.Error(), http.StatusBadRequest)
		return
	}

	w.WriteHeader(http.StatusCreated)
	mrh.writeJSON(w, experiment)
}

func (mrh *ModelRegistryHandler) getExperiment(w http.ResponseWriter, r *http.Request, experimentID string) {
	mrh.registry.mu.RLock()
	experiment, exists := mrh.registry.experiments[experimentID]
	mrh.registry.mu.RUnlock()

	if !exists {
		http.Error(w, "Experiment not found", http.StatusNotFound)
		return
	}

	mrh.writeJSON(w, experiment)
}

func (mrh *ModelRegistryHandler) updateExperiment(w http.ResponseWriter, r *http.Request, experimentID string) {
	var updateData map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&updateData); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	mrh.registry.mu.Lock()
	defer mrh.registry.mu.Unlock()

	experiment, exists := mrh.registry.experiments[experimentID]
	if !exists {
		http.Error(w, "Experiment not found", http.StatusNotFound)
		return
	}

	// Update status if provided
	if status, ok := updateData["status"].(string); ok {
		experiment.Status = ExperimentStatus(status)
		
		if status == string(ExperimentStatusRunning) && experiment.StartedAt == nil {
			startedAt := time.Now()
			experiment.StartedAt = &startedAt
		} else if status == string(ExperimentStatusCompleted) && experiment.EndedAt == nil {
			endedAt := time.Now()
			experiment.EndedAt = &endedAt
		}
	}

	mrh.writeJSON(w, experiment)
}

func (mrh *ModelRegistryHandler) deleteExperiment(w http.ResponseWriter, r *http.Request, experimentID string) {
	mrh.registry.mu.Lock()
	defer mrh.registry.mu.Unlock()

	if _, exists := mrh.registry.experiments[experimentID]; !exists {
		http.Error(w, "Experiment not found", http.StatusNotFound)
		return
	}

	delete(mrh.registry.experiments, experimentID)

	w.WriteHeader(http.StatusNoContent)
}

func (mrh *ModelRegistryHandler) predict(w http.ResponseWriter, r *http.Request, modelID string) {
	var predictionRequest map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&predictionRequest); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Mock prediction logic
	response := map[string]interface{}{
		"model_id":    modelID,
		"prediction":  []float64{0.85, 0.15}, // Mock prediction scores
		"timestamp":   time.Now(),
		"request_id":  fmt.Sprintf("req_%d", time.Now().UnixNano()),
		"metadata": map[string]interface{}{
			"input_features": len(predictionRequest),
			"latency_ms":     42,
		},
	}

	mrh.writeJSON(w, response)
}

// Helper method to write JSON response
func (mrh *ModelRegistryHandler) writeJSON(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(data); err != nil {
		mrh.logger.WithError(err).Error("Failed to encode JSON response")
		http.Error(w, "Internal server error", http.StatusInternalServerError)
	}
}