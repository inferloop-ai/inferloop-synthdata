package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/mux"
)

type AgentsHandler struct {
	agents      map[string]*Agent
	coordinators map[string]*Coordinator
	tasks       map[string]*AgentTask
}

type Agent struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Type         string                 `json:"type"`
	Status       string                 `json:"status"`
	Capabilities []string               `json:"capabilities"`
	Config       map[string]interface{} `json:"config"`
	Metadata     map[string]interface{} `json:"metadata"`
	CreatedAt    time.Time              `json:"createdAt"`
	UpdatedAt    time.Time              `json:"updatedAt"`
	LastSeen     time.Time              `json:"lastSeen"`
	Health       AgentHealth            `json:"health"`
	Metrics      AgentMetrics           `json:"metrics"`
}

type AgentHealth struct {
	Status       string    `json:"status"`
	LastCheck    time.Time `json:"lastCheck"`
	ResponseTime int64     `json:"responseTime"`
	ErrorCount   int       `json:"errorCount"`
	Uptime       string    `json:"uptime"`
}

type AgentMetrics struct {
	TasksCompleted int     `json:"tasksCompleted"`
	TasksFailed    int     `json:"tasksFailed"`
	AvgTaskTime    float64 `json:"avgTaskTime"`
	CpuUsage       float64 `json:"cpuUsage"`
	MemoryUsage    float64 `json:"memoryUsage"`
}

type Coordinator struct {
	ID        string                 `json:"id"`
	Name      string                 `json:"name"`
	Status    string                 `json:"status"`
	Agents    []string               `json:"agents"`
	Strategy  string                 `json:"strategy"`
	Config    map[string]interface{} `json:"config"`
	CreatedAt time.Time              `json:"createdAt"`
	UpdatedAt time.Time              `json:"updatedAt"`
}

type AgentTask struct {
	ID           string                 `json:"id"`
	AgentID      string                 `json:"agentId"`
	Type         string                 `json:"type"`
	Status       string                 `json:"status"`
	Priority     int                    `json:"priority"`
	Payload      map[string]interface{} `json:"payload"`
	Result       map[string]interface{} `json:"result"`
	Error        string                 `json:"error,omitempty"`
	CreatedAt    time.Time              `json:"createdAt"`
	StartedAt    *time.Time             `json:"startedAt,omitempty"`
	CompletedAt  *time.Time             `json:"completedAt,omitempty"`
	Duration     time.Duration          `json:"duration"`
	Retries      int                    `json:"retries"`
	MaxRetries   int                    `json:"maxRetries"`
}

func NewAgentsHandler() *AgentsHandler {
	handler := &AgentsHandler{
		agents:      make(map[string]*Agent),
		coordinators: make(map[string]*Coordinator),
		tasks:       make(map[string]*AgentTask),
	}
	
	handler.initializeBuiltinAgents()
	return handler
}

func (h *AgentsHandler) CreateAgent(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Name         string                 `json:"name"`
		Type         string                 `json:"type"`
		Capabilities []string               `json:"capabilities"`
		Config       map[string]interface{} `json:"config"`
		Metadata     map[string]interface{} `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	if request.Name == "" {
		http.Error(w, "Agent name is required", http.StatusBadRequest)
		return
	}

	if request.Type == "" {
		http.Error(w, "Agent type is required", http.StatusBadRequest)
		return
	}

	agent := &Agent{
		ID:           h.generateID(),
		Name:         request.Name,
		Type:         request.Type,
		Status:       "initializing",
		Capabilities: request.Capabilities,
		Config:       request.Config,
		Metadata:     request.Metadata,
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
		LastSeen:     time.Now(),
		Health: AgentHealth{
			Status:    "healthy",
			LastCheck: time.Now(),
			Uptime:    "0s",
		},
		Metrics: AgentMetrics{},
	}

	h.agents[agent.ID] = agent

	response := map[string]interface{}{
		"status":  "created",
		"agent":   agent,
		"message": "Agent created successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

func (h *AgentsHandler) GetAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "Agent ID is required", http.StatusBadRequest)
		return
	}

	agent, exists := h.agents[id]
	if !exists {
		http.Error(w, "Agent not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(agent)
}

func (h *AgentsHandler) UpdateAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "Agent ID is required", http.StatusBadRequest)
		return
	}

	agent, exists := h.agents[id]
	if !exists {
		http.Error(w, "Agent not found", http.StatusNotFound)
		return
	}

	var updateRequest struct {
		Name         *string                `json:"name,omitempty"`
		Status       *string                `json:"status,omitempty"`
		Config       map[string]interface{} `json:"config,omitempty"`
		Metadata     map[string]interface{} `json:"metadata,omitempty"`
		Capabilities []string               `json:"capabilities,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&updateRequest); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	if updateRequest.Name != nil {
		agent.Name = *updateRequest.Name
	}
	if updateRequest.Status != nil {
		agent.Status = *updateRequest.Status
	}
	if updateRequest.Config != nil {
		agent.Config = updateRequest.Config
	}
	if updateRequest.Metadata != nil {
		agent.Metadata = updateRequest.Metadata
	}
	if updateRequest.Capabilities != nil {
		agent.Capabilities = updateRequest.Capabilities
	}

	agent.UpdatedAt = time.Now()

	response := map[string]interface{}{
		"status":  "updated",
		"agent":   agent,
		"message": "Agent updated successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *AgentsHandler) DeleteAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "Agent ID is required", http.StatusBadRequest)
		return
	}

	_, exists := h.agents[id]
	if !exists {
		http.Error(w, "Agent not found", http.StatusNotFound)
		return
	}

	delete(h.agents, id)

	response := map[string]interface{}{
		"status":  "deleted",
		"id":      id,
		"message": "Agent deleted successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *AgentsHandler) ListAgents(w http.ResponseWriter, r *http.Request) {
	params := r.URL.Query()
	agentType := params.Get("type")
	status := params.Get("status")

	var agents []*Agent
	for _, agent := range h.agents {
		if (agentType == "" || agent.Type == agentType) &&
		   (status == "" || agent.Status == status) {
			agents = append(agents, agent)
		}
	}

	response := map[string]interface{}{
		"agents": agents,
		"count":  len(agents),
		"filters": map[string]interface{}{
			"type":   agentType,
			"status": status,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *AgentsHandler) StartAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "Agent ID is required", http.StatusBadRequest)
		return
	}

	agent, exists := h.agents[id]
	if !exists {
		http.Error(w, "Agent not found", http.StatusNotFound)
		return
	}

	agent.Status = "running"
	agent.UpdatedAt = time.Now()
	agent.LastSeen = time.Now()

	response := map[string]interface{}{
		"status":  "started",
		"agent":   agent,
		"message": "Agent started successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *AgentsHandler) StopAgent(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "Agent ID is required", http.StatusBadRequest)
		return
	}

	agent, exists := h.agents[id]
	if !exists {
		http.Error(w, "Agent not found", http.StatusNotFound)
		return
	}

	agent.Status = "stopped"
	agent.UpdatedAt = time.Now()

	response := map[string]interface{}{
		"status":  "stopped",
		"agent":   agent,
		"message": "Agent stopped successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *AgentsHandler) GetAgentHealth(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "Agent ID is required", http.StatusBadRequest)
		return
	}

	agent, exists := h.agents[id]
	if !exists {
		http.Error(w, "Agent not found", http.StatusNotFound)
		return
	}

	health := h.updateAgentHealth(agent)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

func (h *AgentsHandler) GetAgentMetrics(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "Agent ID is required", http.StatusBadRequest)
		return
	}

	agent, exists := h.agents[id]
	if !exists {
		http.Error(w, "Agent not found", http.StatusNotFound)
		return
	}

	metrics := h.updateAgentMetrics(agent)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

func (h *AgentsHandler) CreateCoordinator(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Name     string                 `json:"name"`
		Strategy string                 `json:"strategy"`
		Agents   []string               `json:"agents"`
		Config   map[string]interface{} `json:"config"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	if request.Name == "" {
		http.Error(w, "Coordinator name is required", http.StatusBadRequest)
		return
	}

	coordinator := &Coordinator{
		ID:        h.generateCoordinatorID(),
		Name:      request.Name,
		Status:    "active",
		Strategy:  request.Strategy,
		Agents:    request.Agents,
		Config:    request.Config,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	h.coordinators[coordinator.ID] = coordinator

	response := map[string]interface{}{
		"status":      "created",
		"coordinator": coordinator,
		"message":     "Coordinator created successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

func (h *AgentsHandler) ListCoordinators(w http.ResponseWriter, r *http.Request) {
	var coordinators []*Coordinator
	for _, coordinator := range h.coordinators {
		coordinators = append(coordinators, coordinator)
	}

	response := map[string]interface{}{
		"coordinators": coordinators,
		"count":        len(coordinators),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *AgentsHandler) AssignTask(w http.ResponseWriter, r *http.Request) {
	var request struct {
		AgentID    string                 `json:"agentId"`
		Type       string                 `json:"type"`
		Priority   int                    `json:"priority"`
		Payload    map[string]interface{} `json:"payload"`
		MaxRetries int                    `json:"maxRetries"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	if request.AgentID == "" {
		http.Error(w, "Agent ID is required", http.StatusBadRequest)
		return
	}

	if request.Type == "" {
		http.Error(w, "Task type is required", http.StatusBadRequest)
		return
	}

	agent, exists := h.agents[request.AgentID]
	if !exists {
		http.Error(w, "Agent not found", http.StatusNotFound)
		return
	}

	task := &AgentTask{
		ID:         h.generateTaskID(),
		AgentID:    request.AgentID,
		Type:       request.Type,
		Status:     "pending",
		Priority:   request.Priority,
		Payload:    request.Payload,
		Result:     make(map[string]interface{}),
		CreatedAt:  time.Now(),
		MaxRetries: request.MaxRetries,
	}

	h.tasks[task.ID] = task

	go h.executeTask(task, agent)

	response := map[string]interface{}{
		"status":  "assigned",
		"task":    task,
		"message": "Task assigned successfully",
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(response)
}

func (h *AgentsHandler) GetTask(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "Task ID is required", http.StatusBadRequest)
		return
	}

	task, exists := h.tasks[id]
	if !exists {
		http.Error(w, "Task not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(task)
}

func (h *AgentsHandler) ListTasks(w http.ResponseWriter, r *http.Request) {
	params := r.URL.Query()
	agentID := params.Get("agentId")
	status := params.Get("status")
	taskType := params.Get("type")

	var tasks []*AgentTask
	for _, task := range h.tasks {
		if (agentID == "" || task.AgentID == agentID) &&
		   (status == "" || task.Status == status) &&
		   (taskType == "" || task.Type == taskType) {
			tasks = append(tasks, task)
		}
	}

	response := map[string]interface{}{
		"tasks": tasks,
		"count": len(tasks),
		"filters": map[string]interface{}{
			"agentId": agentID,
			"status":  status,
			"type":    taskType,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *AgentsHandler) GetAgentTypes(w http.ResponseWriter, r *http.Request) {
	types := []map[string]interface{}{
		{
			"id":          "generation_agent",
			"name":        "Generation Agent",
			"description": "Specialized in time series data generation",
			"capabilities": []string{
				"statistical_generation",
				"arima_generation",
				"neural_generation",
			},
		},
		{
			"id":          "validation_agent",
			"name":        "Validation Agent",
			"description": "Focused on data quality validation",
			"capabilities": []string{
				"statistical_validation",
				"anomaly_detection",
				"quality_assessment",
			},
		},
		{
			"id":          "analytics_agent",
			"name":        "Analytics Agent",
			"description": "Advanced analytics and pattern recognition",
			"capabilities": []string{
				"trend_analysis",
				"correlation_analysis",
				"forecasting",
			},
		},
		{
			"id":          "privacy_agent",
			"name":        "Privacy Agent",
			"description": "Privacy-preserving data operations",
			"capabilities": []string{
				"differential_privacy",
				"data_anonymization",
				"secure_aggregation",
			},
		},
	}

	response := map[string]interface{}{
		"types": types,
		"count": len(types),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (h *AgentsHandler) initializeBuiltinAgents() {
	generationAgent := &Agent{
		ID:   "builtin-generation-agent",
		Name: "Default Generation Agent",
		Type: "generation_agent",
		Status: "running",
		Capabilities: []string{
			"statistical_generation",
			"arima_generation",
			"neural_generation",
		},
		Config: map[string]interface{}{
			"max_concurrent_tasks": 5,
			"timeout_seconds":      300,
		},
		Metadata:  make(map[string]interface{}),
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		LastSeen:  time.Now(),
		Health: AgentHealth{
			Status:    "healthy",
			LastCheck: time.Now(),
			Uptime:    "0s",
		},
		Metrics: AgentMetrics{},
	}

	validationAgent := &Agent{
		ID:   "builtin-validation-agent",
		Name: "Default Validation Agent",
		Type: "validation_agent",
		Status: "running",
		Capabilities: []string{
			"statistical_validation",
			"anomaly_detection",
			"quality_assessment",
		},
		Config: map[string]interface{}{
			"max_concurrent_tasks": 3,
			"timeout_seconds":      180,
		},
		Metadata:  make(map[string]interface{}),
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		LastSeen:  time.Now(),
		Health: AgentHealth{
			Status:    "healthy",
			LastCheck: time.Now(),
			Uptime:    "0s",
		},
		Metrics: AgentMetrics{},
	}

	h.agents[generationAgent.ID] = generationAgent
	h.agents[validationAgent.ID] = validationAgent

	coordinator := &Coordinator{
		ID:       "default-coordinator",
		Name:     "Default Coordinator",
		Status:   "active",
		Strategy: "round_robin",
		Agents: []string{
			generationAgent.ID,
			validationAgent.ID,
		},
		Config:    make(map[string]interface{}),
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	h.coordinators[coordinator.ID] = coordinator
}

func (h *AgentsHandler) executeTask(task *AgentTask, agent *Agent) {
	task.Status = "running"
	now := time.Now()
	task.StartedAt = &now

	time.Sleep(2 * time.Second)

	task.Status = "completed"
	completedAt := time.Now()
	task.CompletedAt = &completedAt
	task.Duration = completedAt.Sub(*task.StartedAt)
	task.Result["status"] = "success"
	task.Result["processed_at"] = completedAt

	agent.Metrics.TasksCompleted++
	agent.LastSeen = time.Now()
}

func (h *AgentsHandler) updateAgentHealth(agent *Agent) AgentHealth {
	agent.Health.LastCheck = time.Now()
	agent.Health.Uptime = time.Since(agent.CreatedAt).String()
	agent.Health.ResponseTime = 25

	return agent.Health
}

func (h *AgentsHandler) updateAgentMetrics(agent *Agent) AgentMetrics {
	agent.Metrics.AvgTaskTime = 2.5
	agent.Metrics.CpuUsage = 45.2
	agent.Metrics.MemoryUsage = 128.5

	return agent.Metrics
}

func (h *AgentsHandler) generateID() string {
	return fmt.Sprintf("agent_%d", time.Now().UnixNano())
}

func (h *AgentsHandler) generateCoordinatorID() string {
	return fmt.Sprintf("coord_%d", time.Now().UnixNano())
}

func (h *AgentsHandler) generateTaskID() string {
	return fmt.Sprintf("task_%d", time.Now().UnixNano())
}