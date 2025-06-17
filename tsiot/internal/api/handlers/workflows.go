package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/mux"
)

type WorkflowsHandler struct {
	workflows  map[string]*Workflow
	executions map[string]*WorkflowExecution
}

type Workflow struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Version     string                 `json:"version"`
	Status      string                 `json:"status"`
	Steps       []WorkflowStep         `json:"steps"`
	Schedule    *WorkflowSchedule      `json:"schedule,omitempty"`
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedAt   time.Time              `json:"createdAt"`
	UpdatedAt   time.Time              `json:"updatedAt"`
	CreatedBy   string                 `json:"createdBy"`
}

type WorkflowStep struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Type         string                 `json:"type"`
	Action       string                 `json:"action"`
	Parameters   map[string]interface{} `json:"parameters"`
	Dependencies []string               `json:"dependencies"`
	Timeout      time.Duration          `json:"timeout"`
	Retries      int                    `json:"retries"`
	OnFailure    string                 `json:"onFailure"`
}

type WorkflowSchedule struct {
	Type      string    `json:"type"`
	Cron      string    `json:"cron,omitempty"`
	Interval  string    `json:"interval,omitempty"`
	StartTime time.Time `json:"startTime,omitempty"`
	EndTime   time.Time `json:"endTime,omitempty"`
	Enabled   bool      `json:"enabled"`
}

type WorkflowExecution struct {
	ID         string                    `json:"id"`
	WorkflowID string                    `json:"workflowId"`
	Status     string                    `json:"status"`
	StartedAt  time.Time                 `json:"startedAt"`
	FinishedAt *time.Time                `json:"finishedAt,omitempty"`
	Duration   time.Duration             `json:"duration"`
	Steps      []WorkflowStepExecution   `json:"steps"`
	Input      map[string]interface{}    `json:"input"`
	Output     map[string]interface{}    `json:"output"`
	Error      string                    `json:"error,omitempty"`
	Metadata   map[string]interface{}    `json:"metadata"`
}

type WorkflowStepExecution struct {
	StepID     string                 `json:"stepId"`
	Status     string                 `json:"status"`
	StartedAt  time.Time              `json:"startedAt"`
	FinishedAt *time.Time             `json:"finishedAt,omitempty"`
	Duration   time.Duration          `json:"duration"`
	Input      map[string]interface{} `json:"input"`
	Output     map[string]interface{} `json:"output"`
	Error      string                 `json:"error,omitempty"`
	Retries    int                    `json:"retries"`
}

func NewWorkflowsHandler() *WorkflowsHandler {
	handler := &WorkflowsHandler{
		workflows:  make(map[string]*Workflow),
		executions: make(map[string]*WorkflowExecution),
	}
	
	handler.initializeBuiltinWorkflows()
	return handler
}

func (h *WorkflowsHandler) CreateWorkflow(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Name        string                 `json:"name"`
		Description string                 `json:"description"`
		Steps       []WorkflowStep         `json:"steps"`
		Schedule    *WorkflowSchedule      `json:"schedule,omitempty"`
		Metadata    map[string]interface{} `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	if request.Name == "" {
		http.Error(w, "Workflow name is required", http.StatusBadRequest)
		return
	}

	if len(request.Steps) == 0 {
		http.Error(w, "Workflow must have at least one step", http.StatusBadRequest)
		return
	}

	workflow := &Workflow{
		ID:          h.generateID(),
		Name:        request.Name,
		Description: request.Description,
		Version:     "1.0.0",
		Status:      "draft",
		Steps:       request.Steps,
		Schedule:    request.Schedule,
		Metadata:    request.Metadata,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		CreatedBy:   "system",
	}

	h.workflows[workflow.ID] = workflow

	response := map[string]interface{}{
		"status":   "created",
		"workflow": workflow,
		"message":  "Workflow created successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

func (h *WorkflowsHandler) GetWorkflow(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "Workflow ID is required", http.StatusBadRequest)
		return
	}

	workflow, exists := h.workflows[id]
	if !exists {
		http.Error(w, "Workflow not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(workflow)
}

func (h *WorkflowsHandler) UpdateWorkflow(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "Workflow ID is required", http.StatusBadRequest)
		return
	}

	workflow, exists := h.workflows[id]
	if !exists {
		http.Error(w, "Workflow not found", http.StatusNotFound)
		return
	}

	var updateRequest struct {
		Name        *string                `json:"name,omitempty"`
		Description *string                `json:"description,omitempty"`
		Steps       []WorkflowStep         `json:"steps,omitempty"`
		Schedule    *WorkflowSchedule      `json:"schedule,omitempty"`
		Metadata    map[string]interface{} `json:"metadata,omitempty"`
		Status      *string                `json:"status,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&updateRequest); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	if updateRequest.Name != nil {
		workflow.Name = *updateRequest.Name
	}
	if updateRequest.Description != nil {
		workflow.Description = *updateRequest.Description
	}
	if updateRequest.Steps != nil {
		workflow.Steps = updateRequest.Steps
	}
	if updateRequest.Schedule != nil {
		workflow.Schedule = updateRequest.Schedule
	}
	if updateRequest.Metadata != nil {
		workflow.Metadata = updateRequest.Metadata
	}
	if updateRequest.Status != nil {
		workflow.Status = *updateRequest.Status
	}

	workflow.UpdatedAt = time.Now()

	response := map[string]interface{}{
		"status":   "updated",
		"workflow": workflow,
		"message":  "Workflow updated successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *WorkflowsHandler) DeleteWorkflow(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "Workflow ID is required", http.StatusBadRequest)
		return
	}

	_, exists := h.workflows[id]
	if !exists {
		http.Error(w, "Workflow not found", http.StatusNotFound)
		return
	}

	delete(h.workflows, id)

	response := map[string]interface{}{
		"status":  "deleted",
		"id":      id,
		"message": "Workflow deleted successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *WorkflowsHandler) ListWorkflows(w http.ResponseWriter, r *http.Request) {
	params := r.URL.Query()
	status := params.Get("status")
	
	var workflows []*Workflow
	for _, workflow := range h.workflows {
		if status == "" || workflow.Status == status {
			workflows = append(workflows, workflow)
		}
	}

	response := map[string]interface{}{
		"workflows": workflows,
		"count":     len(workflows),
		"filters": map[string]interface{}{
			"status": status,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *WorkflowsHandler) ExecuteWorkflow(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "Workflow ID is required", http.StatusBadRequest)
		return
	}

	workflow, exists := h.workflows[id]
	if !exists {
		http.Error(w, "Workflow not found", http.StatusNotFound)
		return
	}

	var request struct {
		Input    map[string]interface{} `json:"input"`
		Metadata map[string]interface{} `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		request.Input = make(map[string]interface{})
		request.Metadata = make(map[string]interface{})
	}

	execution := h.startWorkflowExecution(workflow, request.Input, request.Metadata)

	response := map[string]interface{}{
		"status":    "started",
		"execution": execution,
		"message":   "Workflow execution started",
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(response)
}

func (h *WorkflowsHandler) GetExecution(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "Execution ID is required", http.StatusBadRequest)
		return
	}

	execution, exists := h.executions[id]
	if !exists {
		http.Error(w, "Execution not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(execution)
}

func (h *WorkflowsHandler) ListExecutions(w http.ResponseWriter, r *http.Request) {
	params := r.URL.Query()
	workflowID := params.Get("workflowId")
	status := params.Get("status")

	var executions []*WorkflowExecution
	for _, execution := range h.executions {
		if (workflowID == "" || execution.WorkflowID == workflowID) &&
		   (status == "" || execution.Status == status) {
			executions = append(executions, execution)
		}
	}

	response := map[string]interface{}{
		"executions": executions,
		"count":      len(executions),
		"filters": map[string]interface{}{
			"workflowId": workflowID,
			"status":     status,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *WorkflowsHandler) StopExecution(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "Execution ID is required", http.StatusBadRequest)
		return
	}

	execution, exists := h.executions[id]
	if !exists {
		http.Error(w, "Execution not found", http.StatusNotFound)
		return
	}

	if execution.Status == "completed" || execution.Status == "failed" || execution.Status == "cancelled" {
		http.Error(w, "Execution cannot be stopped", http.StatusBadRequest)
		return
	}

	execution.Status = "cancelled"
	now := time.Now()
	execution.FinishedAt = &now
	execution.Duration = now.Sub(execution.StartedAt)

	response := map[string]interface{}{
		"status":    "cancelled",
		"execution": execution,
		"message":   "Execution stopped successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *WorkflowsHandler) GetWorkflowTemplates(w http.ResponseWriter, r *http.Request) {
	templates := []map[string]interface{}{
		{
			"id":          "data_generation_pipeline",
			"name":        "Data Generation Pipeline",
			"description": "Generate synthetic time series data with validation",
			"category":    "generation",
		},
		{
			"id":          "quality_assessment",
			"name":        "Quality Assessment Workflow",
			"description": "Comprehensive quality assessment of time series data",
			"category":    "validation",
		},
		{
			"id":          "batch_generation",
			"name":        "Batch Generation Workflow",
			"description": "Generate multiple time series datasets",
			"category":    "generation",
		},
	}

	response := map[string]interface{}{
		"templates": templates,
		"count":     len(templates),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *WorkflowsHandler) CreateFromTemplate(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	templateID := vars["templateId"]

	var request struct {
		Name        string                 `json:"name"`
		Description string                 `json:"description"`
		Parameters  map[string]interface{} `json:"parameters"`
		Schedule    *WorkflowSchedule      `json:"schedule,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	template := h.getTemplate(templateID)
	if template == nil {
		http.Error(w, "Template not found", http.StatusNotFound)
		return
	}

	workflow := h.createWorkflowFromTemplate(template, request)
	h.workflows[workflow.ID] = workflow

	response := map[string]interface{}{
		"status":   "created",
		"workflow": workflow,
		"message":  "Workflow created from template successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

func (h *WorkflowsHandler) initializeBuiltinWorkflows() {
	workflow := &Workflow{
		ID:          "sample-workflow-1",
		Name:        "Sample Data Generation",
		Description: "Sample workflow for generating synthetic time series data",
		Version:     "1.0.0",
		Status:      "active",
		Steps: []WorkflowStep{
			{
				ID:     "step-1",
				Name:   "Generate Data",
				Type:   "generation",
				Action: "generate_timeseries",
				Parameters: map[string]interface{}{
					"generator": "statistical",
					"length":    100,
					"frequency": "1m",
				},
				Timeout: 30 * time.Second,
				Retries: 3,
			},
			{
				ID:           "step-2",
				Name:         "Validate Data",
				Type:         "validation",
				Action:       "validate_timeseries",
				Dependencies: []string{"step-1"},
				Timeout:      15 * time.Second,
				Retries:      2,
			},
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		CreatedBy: "system",
	}

	h.workflows[workflow.ID] = workflow
}

func (h *WorkflowsHandler) startWorkflowExecution(workflow *Workflow, input, metadata map[string]interface{}) *WorkflowExecution {
	execution := &WorkflowExecution{
		ID:         h.generateExecutionID(),
		WorkflowID: workflow.ID,
		Status:     "running",
		StartedAt:  time.Now(),
		Input:      input,
		Output:     make(map[string]interface{}),
		Metadata:   metadata,
		Steps:      make([]WorkflowStepExecution, 0),
	}

	h.executions[execution.ID] = execution

	go h.simulateWorkflowExecution(execution, workflow)

	return execution
}

func (h *WorkflowsHandler) simulateWorkflowExecution(execution *WorkflowExecution, workflow *Workflow) {
	time.Sleep(2 * time.Second)

	for _, step := range workflow.Steps {
		stepExecution := WorkflowStepExecution{
			StepID:    step.ID,
			Status:    "running",
			StartedAt: time.Now(),
			Input:     step.Parameters,
			Output:    make(map[string]interface{}),
		}

		time.Sleep(1 * time.Second)

		stepExecution.Status = "completed"
		now := time.Now()
		stepExecution.FinishedAt = &now
		stepExecution.Duration = now.Sub(stepExecution.StartedAt)
		stepExecution.Output["result"] = "success"

		execution.Steps = append(execution.Steps, stepExecution)
	}

	execution.Status = "completed"
	now := time.Now()
	execution.FinishedAt = &now
	execution.Duration = now.Sub(execution.StartedAt)
	execution.Output["status"] = "success"
	execution.Output["steps_completed"] = len(workflow.Steps)
}

func (h *WorkflowsHandler) getTemplate(templateID string) map[string]interface{} {
	templates := map[string]map[string]interface{}{
		"data_generation_pipeline": {
			"name":        "Data Generation Pipeline",
			"description": "Generate synthetic time series data with validation",
			"steps": []WorkflowStep{
				{
					ID:     "generate",
					Name:   "Generate Data",
					Type:   "generation",
					Action: "generate_timeseries",
					Parameters: map[string]interface{}{
						"generator": "statistical",
						"length":    1000,
					},
				},
				{
					ID:           "validate",
					Name:         "Validate Data",
					Type:         "validation",
					Action:       "validate_timeseries",
					Dependencies: []string{"generate"},
				},
			},
		},
	}

	return templates[templateID]
}

func (h *WorkflowsHandler) createWorkflowFromTemplate(template map[string]interface{}, request struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
	Schedule    *WorkflowSchedule      `json:"schedule,omitempty"`
}) *Workflow {
	workflow := &Workflow{
		ID:          h.generateID(),
		Name:        request.Name,
		Description: request.Description,
		Version:     "1.0.0",
		Status:      "draft",
		Steps:       template["steps"].([]WorkflowStep),
		Schedule:    request.Schedule,
		Metadata:    make(map[string]interface{}),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		CreatedBy:   "system",
	}

	return workflow
}

func (h *WorkflowsHandler) generateID() string {
	return fmt.Sprintf("wf_%d", time.Now().UnixNano())
}

func (h *WorkflowsHandler) generateExecutionID() string {
	return fmt.Sprintf("exec_%d", time.Now().UnixNano())
}