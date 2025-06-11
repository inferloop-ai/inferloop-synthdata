package base

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/inferloop/tsiot/pkg/models"
)

// Agent defines the base interface for all agents
type Agent interface {
	// Core lifecycle methods
	Start(ctx context.Context) error
	Stop(ctx context.Context) error
	Restart(ctx context.Context) error
	IsRunning() bool

	// Information methods
	GetID() string
	GetName() string
	GetType() models.AgentType
	GetStatus() models.AgentStatus
	GetInfo() *models.AgentInfo
	GetConfig() *models.AgentConfig

	// Task processing
	ProcessTask(ctx context.Context, task *models.AgentTask) (*models.TaskResult, error)
	CanProcessTask(task *models.AgentTask) bool
	GetCapabilities() []string

	// Health and monitoring
	HealthCheck(ctx context.Context) error
	GetMetrics() map[string]interface{}
	GetLastError() error

	// Configuration
	UpdateConfig(config *models.AgentConfig) error
	Validate() error

	// Resource management
	GetResourceUsage() *ResourceUsage
	SetResourceLimits(limits models.ResourceLimits) error

	// Event handling
	OnTaskReceived(task *models.AgentTask)
	OnTaskCompleted(task *models.AgentTask, result *models.TaskResult)
	OnTaskFailed(task *models.AgentTask, err error)
	OnError(err error)
}

// BaseAgent provides common functionality for all agents
type BaseAgent struct {
	mu       sync.RWMutex
	config   *models.AgentConfig
	status   models.AgentStatus
	info     *models.AgentInfo
	startTime time.Time
	lastActivity time.Time
	currentTasks map[string]*models.AgentTask
	taskHistory  []TaskHistoryEntry
	metrics      AgentMetrics
	lastError    error
	capabilities []string
	resourceUsage *ResourceUsage
	cancelFunc   context.CancelFunc
	eventHandlers EventHandlers
}

// ResourceUsage tracks current resource consumption
type ResourceUsage struct {
	MemoryUsage int64     `json:"memory_usage"`
	CPUUsage    float64   `json:"cpu_usage"`
	DiskUsage   int64     `json:"disk_usage"`
	NetworkIn   int64     `json:"network_in"`
	NetworkOut  int64     `json:"network_out"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// AgentMetrics tracks performance metrics
type AgentMetrics struct {
	TasksProcessed    int64     `json:"tasks_processed"`
	TasksSucceeded    int64     `json:"tasks_succeeded"`
	TasksFailed       int64     `json:"tasks_failed"`
	TasksCancelled    int64     `json:"tasks_cancelled"`
	TotalProcessingTime time.Duration `json:"total_processing_time"`
	AverageProcessingTime time.Duration `json:"average_processing_time"`
	LastResetTime     time.Time `json:"last_reset_time"`
	ErrorRate         float64   `json:"error_rate"`
	Throughput        float64   `json:"throughput"` // tasks per second
}

// TaskHistoryEntry records task execution history
type TaskHistoryEntry struct {
	TaskID      string                 `json:"task_id"`
	TaskType    string                 `json:"task_type"`
	StartTime   time.Time              `json:"start_time"`
	EndTime     time.Time              `json:"end_time"`
	Duration    time.Duration          `json:"duration"`
	Status      models.TaskStatus      `json:"status"`
	Success     bool                   `json:"success"`
	Error       string                 `json:"error,omitempty"`
	Result      *models.TaskResult     `json:"result,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// EventHandlers contains callback functions for various events
type EventHandlers struct {
	OnStart          func(agent Agent) error
	OnStop           func(agent Agent) error
	OnTaskReceived   func(agent Agent, task *models.AgentTask)
	OnTaskCompleted  func(agent Agent, task *models.AgentTask, result *models.TaskResult)
	OnTaskFailed     func(agent Agent, task *models.AgentTask, err error)
	OnError          func(agent Agent, err error)
	OnHealthCheck    func(agent Agent, healthy bool)
	OnConfigUpdate   func(agent Agent, oldConfig, newConfig *models.AgentConfig)
}

// NewBaseAgent creates a new base agent instance
func NewBaseAgent(config *models.AgentConfig) *BaseAgent {
	now := time.Now()
	
	agent := &BaseAgent{
		config:       config,
		status:       models.AgentStatusStopped,
		startTime:    now,
		lastActivity: now,
		currentTasks: make(map[string]*models.AgentTask),
		taskHistory:  make([]TaskHistoryEntry, 0),
		metrics: AgentMetrics{
			LastResetTime: now,
		},
		capabilities: []string{},
		resourceUsage: &ResourceUsage{
			UpdatedAt: now,
		},
		eventHandlers: EventHandlers{},
	}

	agent.info = &models.AgentInfo{
		ID:           config.ID,
		Name:         config.Name,
		Type:         config.Type,
		Status:       agent.status,
		StartTime:    agent.startTime,
		LastActivity: agent.lastActivity,
		Capabilities: agent.capabilities,
	}

	return agent
}

// Core lifecycle methods
func (a *BaseAgent) Start(ctx context.Context) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.status == models.AgentStatusRunning {
		return fmt.Errorf("agent %s is already running", a.config.ID)
	}

	// Create cancellable context for the agent
	agentCtx, cancel := context.WithCancel(ctx)
	a.cancelFunc = cancel

	// Update status and info
	a.status = models.AgentStatusRunning
	a.startTime = time.Now()
	a.lastActivity = a.startTime
	a.updateInfo()

	// Call event handler if set
	if a.eventHandlers.OnStart != nil {
		if err := a.eventHandlers.OnStart(a); err != nil {
			a.status = models.AgentStatusError
			a.lastError = err
			a.updateInfo()
			return fmt.Errorf("failed to start agent %s: %w", a.config.ID, err)
		}
	}

	// Start background routines if needed
	go a.monitorHealth(agentCtx)
	go a.updateResourceUsage(agentCtx)

	return nil
}

func (a *BaseAgent) Stop(ctx context.Context) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.status != models.AgentStatusRunning {
		return fmt.Errorf("agent %s is not running", a.config.ID)
	}

	// Cancel agent context to stop background routines
	if a.cancelFunc != nil {
		a.cancelFunc()
	}

	// Wait for current tasks to complete or timeout
	timeout := time.NewTimer(30 * time.Second)
	defer timeout.Stop()

	for len(a.currentTasks) > 0 {
		select {
		case <-timeout.C:
			// Force stop after timeout
			break
		case <-time.After(100 * time.Millisecond):
			// Check again
		}
	}

	// Update status
	a.status = models.AgentStatusStopped
	a.updateInfo()

	// Call event handler if set
	if a.eventHandlers.OnStop != nil {
		if err := a.eventHandlers.OnStop(a); err != nil {
			a.lastError = err
			return fmt.Errorf("error during agent stop: %w", err)
		}
	}

	return nil
}

func (a *BaseAgent) Restart(ctx context.Context) error {
	if err := a.Stop(ctx); err != nil {
		return fmt.Errorf("failed to stop agent during restart: %w", err)
	}
	
	time.Sleep(1 * time.Second) // Brief pause between stop and start
	
	if err := a.Start(ctx); err != nil {
		return fmt.Errorf("failed to start agent during restart: %w", err)
	}
	
	return nil
}

func (a *BaseAgent) IsRunning() bool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.status == models.AgentStatusRunning
}

// Information methods
func (a *BaseAgent) GetID() string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.config.ID
}

func (a *BaseAgent) GetName() string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.config.Name
}

func (a *BaseAgent) GetType() models.AgentType {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.config.Type
}

func (a *BaseAgent) GetStatus() models.AgentStatus {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.status
}

func (a *BaseAgent) GetInfo() *models.AgentInfo {
	a.mu.RLock()
	defer a.mu.RUnlock()
	a.updateInfo()
	return a.info
}

func (a *BaseAgent) GetConfig() *models.AgentConfig {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.config
}

// Task processing - base implementation (should be overridden by specific agents)
func (a *BaseAgent) ProcessTask(ctx context.Context, task *models.AgentTask) (*models.TaskResult, error) {
	a.mu.Lock()
	a.currentTasks[task.ID] = task
	a.lastActivity = time.Now()
	a.mu.Unlock()

	defer func() {
		a.mu.Lock()
		delete(a.currentTasks, task.ID)
		a.metrics.TasksProcessed++
		a.updateInfo()
		a.mu.Unlock()
	}()

	// Call event handler
	a.OnTaskReceived(task)

	startTime := time.Now()
	
	// Basic validation
	if !a.CanProcessTask(task) {
		err := fmt.Errorf("agent %s cannot process task type %s", a.config.ID, task.Type)
		a.OnTaskFailed(task, err)
		return nil, err
	}

	// Create result
	result := &models.TaskResult{
		Success:   true,
		Data:      make(map[string]interface{}),
		Metrics:   make(map[string]float64),
		Metadata:  make(map[string]interface{}),
		Duration:  time.Since(startTime),
		CreatedAt: time.Now(),
	}

	// This is a base implementation - specific agents should override this method
	result.Data["message"] = fmt.Sprintf("Task %s processed by base agent %s", task.ID, a.config.ID)
	result.Metadata["agent_id"] = a.config.ID
	result.Metadata["agent_type"] = string(a.config.Type)

	a.OnTaskCompleted(task, result)
	return result, nil
}

func (a *BaseAgent) CanProcessTask(task *models.AgentTask) bool {
	// Base implementation - specific agents should override this
	return true
}

func (a *BaseAgent) GetCapabilities() []string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.capabilities
}

// Health and monitoring
func (a *BaseAgent) HealthCheck(ctx context.Context) error {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if a.status != models.AgentStatusRunning {
		return fmt.Errorf("agent %s is not running (status: %s)", a.config.ID, a.status)
	}

	// Check if agent has been inactive for too long
	if time.Since(a.lastActivity) > a.config.HealthCheckInterval*3 {
		return fmt.Errorf("agent %s has been inactive for %v", a.config.ID, time.Since(a.lastActivity))
	}

	return nil
}

func (a *BaseAgent) GetMetrics() map[string]interface{} {
	a.mu.RLock()
	defer a.mu.RUnlock()

	metrics := make(map[string]interface{})
	metrics["tasks_processed"] = a.metrics.TasksProcessed
	metrics["tasks_succeeded"] = a.metrics.TasksSucceeded
	metrics["tasks_failed"] = a.metrics.TasksFailed
	metrics["tasks_cancelled"] = a.metrics.TasksCancelled
	metrics["total_processing_time"] = a.metrics.TotalProcessingTime.Seconds()
	metrics["average_processing_time"] = a.metrics.AverageProcessingTime.Seconds()
	metrics["error_rate"] = a.metrics.ErrorRate
	metrics["throughput"] = a.metrics.Throughput
	metrics["current_tasks"] = len(a.currentTasks)
	metrics["memory_usage"] = a.resourceUsage.MemoryUsage
	metrics["cpu_usage"] = a.resourceUsage.CPUUsage
	
	return metrics
}

func (a *BaseAgent) GetLastError() error {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.lastError
}

// Configuration
func (a *BaseAgent) UpdateConfig(config *models.AgentConfig) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	oldConfig := a.config
	a.config = config
	a.updateInfo()

	// Call event handler if set
	if a.eventHandlers.OnConfigUpdate != nil {
		a.eventHandlers.OnConfigUpdate(a, oldConfig, config)
	}

	return nil
}

func (a *BaseAgent) Validate() error {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if a.config == nil {
		return fmt.Errorf("agent config is nil")
	}

	if a.config.ID == "" {
		return fmt.Errorf("agent ID is required")
	}

	if a.config.Name == "" {
		return fmt.Errorf("agent name is required")
	}

	if a.config.Type == "" {
		return fmt.Errorf("agent type is required")
	}

	return nil
}

// Resource management
func (a *BaseAgent) GetResourceUsage() *ResourceUsage {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.resourceUsage
}

func (a *BaseAgent) SetResourceLimits(limits models.ResourceLimits) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	a.config.Resources = limits
	return nil
}

// Event handling
func (a *BaseAgent) OnTaskReceived(task *models.AgentTask) {
	if a.eventHandlers.OnTaskReceived != nil {
		a.eventHandlers.OnTaskReceived(a, task)
	}
}

func (a *BaseAgent) OnTaskCompleted(task *models.AgentTask, result *models.TaskResult) {
	a.mu.Lock()
	a.metrics.TasksSucceeded++
	a.mu.Unlock()

	if a.eventHandlers.OnTaskCompleted != nil {
		a.eventHandlers.OnTaskCompleted(a, task, result)
	}
}

func (a *BaseAgent) OnTaskFailed(task *models.AgentTask, err error) {
	a.mu.Lock()
	a.metrics.TasksFailed++
	a.lastError = err
	a.mu.Unlock()

	if a.eventHandlers.OnTaskFailed != nil {
		a.eventHandlers.OnTaskFailed(a, task, err)
	}
}

func (a *BaseAgent) OnError(err error) {
	a.mu.Lock()
	a.lastError = err
	a.mu.Unlock()

	if a.eventHandlers.OnError != nil {
		a.eventHandlers.OnError(a, err)
	}
}

// SetEventHandler sets an event handler
func (a *BaseAgent) SetEventHandler(handlerType string, handler interface{}) {
	switch handlerType {
	case "onStart":
		if h, ok := handler.(func(Agent) error); ok {
			a.eventHandlers.OnStart = h
		}
	case "onStop":
		if h, ok := handler.(func(Agent) error); ok {
			a.eventHandlers.OnStop = h
		}
	case "onTaskReceived":
		if h, ok := handler.(func(Agent, *models.AgentTask)); ok {
			a.eventHandlers.OnTaskReceived = h
		}
	case "onTaskCompleted":
		if h, ok := handler.(func(Agent, *models.AgentTask, *models.TaskResult)); ok {
			a.eventHandlers.OnTaskCompleted = h
		}
	case "onTaskFailed":
		if h, ok := handler.(func(Agent, *models.AgentTask, error)); ok {
			a.eventHandlers.OnTaskFailed = h
		}
	case "onError":
		if h, ok := handler.(func(Agent, error)); ok {
			a.eventHandlers.OnError = h
		}
	}
}

// Private helper methods
func (a *BaseAgent) updateInfo() {
	a.info.Status = a.status
	a.info.LastActivity = a.lastActivity
	a.info.TasksProcessed = a.metrics.TasksProcessed
	a.info.TasksSucceeded = a.metrics.TasksSucceeded
	a.info.TasksFailed = a.metrics.TasksFailed
	a.info.CurrentLoad = float64(len(a.currentTasks)) / float64(a.config.MaxConcurrency)
	a.info.MemoryUsage = a.resourceUsage.MemoryUsage
	a.info.CPUUsage = a.resourceUsage.CPUUsage
	a.info.ErrorCount = a.metrics.TasksFailed
	if a.lastError != nil {
		a.info.LastError = a.lastError.Error()
	}
}

func (a *BaseAgent) monitorHealth(ctx context.Context) {
	ticker := time.NewTicker(a.config.HealthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			err := a.HealthCheck(ctx)
			healthy := err == nil
			
			if !healthy && a.status == models.AgentStatusRunning {
				a.mu.Lock()
				a.status = models.AgentStatusError
				a.lastError = err
				a.updateInfo()
				a.mu.Unlock()
			}

			if a.eventHandlers.OnHealthCheck != nil {
				a.eventHandlers.OnHealthCheck(a, healthy)
			}
		}
	}
}

func (a *BaseAgent) updateResourceUsage(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// This is a placeholder implementation
			// In a real implementation, you would collect actual resource metrics
			a.mu.Lock()
			a.resourceUsage.UpdatedAt = time.Now()
			// Update other resource metrics here
			a.mu.Unlock()
		}
	}
}