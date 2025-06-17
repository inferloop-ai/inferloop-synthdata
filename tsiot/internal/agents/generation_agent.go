package agents

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/internal/agents/base"
	"github.com/inferloop/tsiot/internal/generators/factory"
	"github.com/inferloop/tsiot/pkg/constants"
	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/interfaces"
	"github.com/inferloop/tsiot/pkg/models"
)

// GenerationAgent orchestrates synthetic data generation workflows
type GenerationAgent struct {
	*base.BaseAgent
	mu               sync.RWMutex // Add mutex for thread safety (BaseAgent has private mu, so we need our own)
	logger           *logrus.Logger
	generatorFactory interfaces.GeneratorFactory
	activeJobs       map[string]*GenerationJob
	jobQueue         chan *GenerationJob
	workerPool       chan struct{} // Semaphore for controlling concurrency
	jobHistory       []JobHistoryEntry
	maxHistorySize   int
}

// GenerationJob represents a generation job
type GenerationJob struct {
	ID              string                    `json:"id"`
	Request         *models.GenerationRequest `json:"request"`
	Generator       interfaces.Generator      `json:"-"`
	Status          JobStatus                 `json:"status"`
	Progress        float64                   `json:"progress"`
	StartTime       time.Time                 `json:"start_time"`
	EndTime         time.Time                 `json:"end_time"`
	Duration        time.Duration             `json:"duration"`
	Result          *models.GenerationResult  `json:"result,omitempty"`
	Error           error                     `json:"-"`
	ErrorMessage    string                    `json:"error_message,omitempty"`
	CancelCtx       context.Context           `json:"-"`
	CancelFunc      context.CancelFunc        `json:"-"`
	Metadata        map[string]interface{}    `json:"metadata"`
	RetryCount      int                       `json:"retry_count"`
	MaxRetries      int                       `json:"max_retries"`
}

// JobStatus represents the status of a generation job
type JobStatus string

const (
	JobStatusPending    JobStatus = "pending"
	JobStatusRunning    JobStatus = "running"
	JobStatusCompleted  JobStatus = "completed"
	JobStatusFailed     JobStatus = "failed"
	JobStatusCancelled  JobStatus = "cancelled"
	JobStatusRetrying   JobStatus = "retrying"
)

// JobHistoryEntry tracks completed jobs
type JobHistoryEntry struct {
	JobID        string            `json:"job_id"`
	GeneratorType string           `json:"generator_type"`
	StartTime    time.Time         `json:"start_time"`
	EndTime      time.Time         `json:"end_time"`
	Duration     time.Duration     `json:"duration"`
	Status       JobStatus         `json:"status"`
	DataPoints   int64             `json:"data_points"`
	Success      bool              `json:"success"`
	Error        string            `json:"error,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// GenerationAgentConfig contains configuration for the generation agent
type GenerationAgentConfig struct {
	MaxConcurrentJobs int           `json:"max_concurrent_jobs"`
	QueueSize         int           `json:"queue_size"`
	JobTimeout        time.Duration `json:"job_timeout"`
	MaxRetries        int           `json:"max_retries"`
	RetryDelay        time.Duration `json:"retry_delay"`
	HistorySize       int           `json:"history_size"`
	EnableAutoRetry   bool          `json:"enable_auto_retry"`
}

// NewGenerationAgent creates a new generation agent
func NewGenerationAgent(config *models.AgentConfig, logger *logrus.Logger) (*GenerationAgent, error) {
	if logger == nil {
		logger = logrus.New()
	}

	// Create generator factory
	generatorFactory, err := factory.NewGeneratorFactory(logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create generator factory: %w", err)
	}

	// Set default config values
	if config.Type == "" {
		config.Type = models.AgentType(constants.AgentTypeGeneration)
	}
	if config.MaxConcurrency == 0 {
		config.MaxConcurrency = 5
	}
	if config.HealthCheckInterval == 0 {
		config.HealthCheckInterval = 30 * time.Second
	}

	// Create base agent
	baseAgent := base.NewBaseAgent(config)

	// Create generation agent
	agent := &GenerationAgent{
		BaseAgent:        baseAgent,
		logger:           logger,
		generatorFactory: generatorFactory,
		activeJobs:       make(map[string]*GenerationJob),
		jobQueue:         make(chan *GenerationJob, 100), // Default queue size
		workerPool:       make(chan struct{}, config.MaxConcurrency),
		jobHistory:       make([]JobHistoryEntry, 0),
		maxHistorySize:   1000, // Default history size
	}

	// Set capabilities
	agent.SetCapabilities()

	// Set up event handlers
	agent.setupEventHandlers()

	return agent, nil
}

// SetCapabilities sets the agent's capabilities based on available generators
func (ga *GenerationAgent) SetCapabilities() {
	capabilities := []string{
		"data_generation",
		"batch_generation",
		"streaming_generation",
		"job_management",
		"progress_tracking",
		"error_handling",
		"retry_logic",
	}

	// Add generator-specific capabilities
	availableGenerators := ga.generatorFactory.GetAvailableGenerators()
	for _, genType := range availableGenerators {
		capabilities = append(capabilities, fmt.Sprintf("generator_%s", string(genType)))
	}

	// Update base agent capabilities
	ga.BaseAgent.UpdateCapabilities(capabilities)
}

// UpdateCapabilities updates the agent's capabilities (helper method)
func (ba *GenerationAgent) UpdateCapabilities(capabilities []string) {
	// This would be implemented in the base agent
	// For now, we'll store it in metadata
	ba.BaseAgent.GetConfig().Metadata["capabilities"] = capabilities
}

// Start starts the generation agent
func (ga *GenerationAgent) Start(ctx context.Context) error {
	// Start base agent
	if err := ga.BaseAgent.Start(ctx); err != nil {
		return fmt.Errorf("failed to start base agent: %w", err)
	}

	ga.logger.WithFields(logrus.Fields{
		"agent_id":   ga.GetID(),
		"agent_name": ga.GetName(),
	}).Info("Starting generation agent")

	// Start worker goroutines
	for i := 0; i < ga.GetConfig().MaxConcurrency; i++ {
		go ga.worker(ctx)
	}

	// Start job monitor
	go ga.monitorJobs(ctx)

	return nil
}

// Stop stops the generation agent
func (ga *GenerationAgent) Stop(ctx context.Context) error {
	ga.logger.WithFields(logrus.Fields{
		"agent_id": ga.GetID(),
	}).Info("Stopping generation agent")

	// Cancel all active jobs
	for _, job := range ga.activeJobs {
		if job.CancelFunc != nil {
			job.CancelFunc()
		}
	}

	// Stop base agent
	if err := ga.BaseAgent.Stop(ctx); err != nil {
		return fmt.Errorf("failed to stop base agent: %w", err)
	}

	return nil
}

// ProcessTask processes agent tasks (implements base.Agent interface)
func (ga *GenerationAgent) ProcessTask(ctx context.Context, task *models.AgentTask) (*models.TaskResult, error) {
	switch task.Type {
	case "generate":
		return ga.handleGenerateTask(ctx, task)
	case "cancel_job":
		return ga.handleCancelJobTask(ctx, task)
	case "get_job_status":
		return ga.handleGetJobStatusTask(ctx, task)
	case "list_jobs":
		return ga.handleListJobsTask(ctx, task)
	case "get_generator_info":
		return ga.handleGetGeneratorInfoTask(ctx, task)
	default:
		return ga.BaseAgent.ProcessTask(ctx, task)
	}
}

// CanProcessTask checks if the agent can process the given task
func (ga *GenerationAgent) CanProcessTask(task *models.AgentTask) bool {
	supportedTasks := []string{
		"generate",
		"cancel_job",
		"get_job_status",
		"list_jobs",
		"get_generator_info",
	}

	for _, supportedTask := range supportedTasks {
		if task.Type == supportedTask {
			return true
		}
	}

	return false
}

// SubmitGenerationJob submits a new generation job
func (ga *GenerationAgent) SubmitGenerationJob(ctx context.Context, request *models.GenerationRequest) (*GenerationJob, error) {
	if request == nil {
		return nil, errors.NewValidationError("INVALID_REQUEST", "Generation request is required")
	}

	// Validate request
	if err := ga.validateGenerationRequest(request); err != nil {
		return nil, err
	}

	// Get generator
	generator, err := ga.generatorFactory.CreateGenerator(request.GeneratorType)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeValidation, "GENERATOR_ERROR", "Failed to create generator")
	}

	// Create job context with timeout
	jobCtx, cancelFunc := context.WithTimeout(ctx, 30*time.Minute) // Default timeout

	// Create job
	job := &GenerationJob{
		ID:         request.ID,
		Request:    request,
		Generator:  generator,
		Status:     JobStatusPending,
		Progress:   0.0,
		StartTime:  time.Now(),
		CancelCtx:  jobCtx,
		CancelFunc: cancelFunc,
		Metadata:   make(map[string]interface{}),
		MaxRetries: 3, // Default max retries
	}

	// Add job to active jobs
	ga.mu.Lock()
	ga.activeJobs[job.ID] = job
	ga.mu.Unlock()

	// Submit to queue
	select {
	case ga.jobQueue <- job:
		ga.logger.WithFields(logrus.Fields{
			"job_id":         job.ID,
			"generator_type": string(request.GeneratorType),
		}).Info("Generation job submitted")
		return job, nil
	default:
		// Queue is full
		ga.mu.Lock()
		delete(ga.activeJobs, job.ID)
		ga.mu.Unlock()
		cancelFunc()
		return nil, errors.NewProcessingError("QUEUE_FULL", "Generation job queue is full")
	}
}

// GetJobStatus returns the status of a job
func (ga *GenerationAgent) GetJobStatus(jobID string) (*GenerationJob, error) {
	ga.mu.RLock()
	defer ga.mu.RUnlock()

	job, exists := ga.activeJobs[jobID]
	if !exists {
		return nil, errors.NewNotFoundError("JOB_NOT_FOUND", fmt.Sprintf("Job %s not found", jobID))
	}

	return job, nil
}

// CancelJob cancels a running job
func (ga *GenerationAgent) CancelJob(jobID string) error {
	ga.mu.Lock()
	defer ga.mu.Unlock()

	job, exists := ga.activeJobs[jobID]
	if !exists {
		return errors.NewNotFoundError("JOB_NOT_FOUND", fmt.Sprintf("Job %s not found", jobID))
	}

	if job.Status == JobStatusCompleted || job.Status == JobStatusFailed || job.Status == JobStatusCancelled {
		return errors.NewValidationError("INVALID_JOB_STATE", fmt.Sprintf("Cannot cancel job %s in status %s", jobID, job.Status))
	}

	// Cancel the job
	if job.CancelFunc != nil {
		job.CancelFunc()
	}
	job.Status = JobStatusCancelled
	job.EndTime = time.Now()
	job.Duration = job.EndTime.Sub(job.StartTime)

	ga.logger.WithFields(logrus.Fields{
		"job_id": jobID,
	}).Info("Generation job cancelled")

	return nil
}

// ListActiveJobs returns all active jobs
func (ga *GenerationAgent) ListActiveJobs() []*GenerationJob {
	ga.mu.RLock()
	defer ga.mu.RUnlock()

	jobs := make([]*GenerationJob, 0, len(ga.activeJobs))
	for _, job := range ga.activeJobs {
		jobs = append(jobs, job)
	}

	return jobs
}

// GetJobHistory returns job history
func (ga *GenerationAgent) GetJobHistory(limit int) []JobHistoryEntry {
	ga.mu.RLock()
	defer ga.mu.RUnlock()

	if limit <= 0 || limit > len(ga.jobHistory) {
		limit = len(ga.jobHistory)
	}

	history := make([]JobHistoryEntry, limit)
	startIdx := len(ga.jobHistory) - limit
	copy(history, ga.jobHistory[startIdx:])

	return history
}

// Worker methods

func (ga *GenerationAgent) worker(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case job := <-ga.jobQueue:
			ga.processJob(ctx, job)
		}
	}
}

func (ga *GenerationAgent) processJob(ctx context.Context, job *GenerationJob) {
	// Acquire worker slot
	ga.workerPool <- struct{}{}
	defer func() { <-ga.workerPool }()

	ga.logger.WithFields(logrus.Fields{
		"job_id":         job.ID,
		"generator_type": string(job.Request.GeneratorType),
	}).Info("Processing generation job")

	// Update job status
	job.Status = JobStatusRunning
	job.StartTime = time.Now()

	// Process the job
	result, err := ga.executeJob(job.CancelCtx, job)

	// Update job with results
	job.EndTime = time.Now()
	job.Duration = job.EndTime.Sub(job.StartTime)

	if err != nil {
		job.Status = JobStatusFailed
		job.Error = err
		job.ErrorMessage = err.Error()

		// Check if we should retry
		if ga.shouldRetryJob(job) {
			ga.retryJob(ctx, job)
			return
		}
	} else {
		job.Status = JobStatusCompleted
		job.Result = result
		job.Progress = 1.0
	}

	// Move job to history
	ga.moveJobToHistory(job)

	ga.logger.WithFields(logrus.Fields{
		"job_id":   job.ID,
		"status":   job.Status,
		"duration": job.Duration,
	}).Info("Generation job completed")
}

func (ga *GenerationAgent) executeJob(ctx context.Context, job *GenerationJob) (*models.GenerationResult, error) {
	// Validate generator parameters
	if err := job.Generator.ValidateParameters(job.Request.Parameters); err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeValidation, "PARAMETER_VALIDATION", "Invalid generation parameters")
	}

	// Train generator if needed and training data is provided
	if job.Generator.IsTrainable() && job.Request.TrainingData != nil {
		ga.logger.WithFields(logrus.Fields{
			"job_id": job.ID,
		}).Info("Training generator")

		if err := job.Generator.Train(ctx, job.Request.TrainingData, job.Request.Parameters); err != nil {
			return nil, errors.WrapError(err, errors.ErrorTypeProcessing, "TRAINING_ERROR", "Failed to train generator")
		}

		job.Progress = 0.1 // 10% progress after training
	}

	// Generate data
	ga.logger.WithFields(logrus.Fields{
		"job_id": job.ID,
		"length": job.Request.Parameters.Length,
	}).Info("Generating data")

	// Start progress tracking
	go ga.trackJobProgress(ctx, job)

	result, err := job.Generator.Generate(ctx, job.Request)
	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeProcessing, "GENERATION_ERROR", "Failed to generate data")
	}

	// Add job metadata to result
	if result.Metadata == nil {
		result.Metadata = make(map[string]interface{})
	}
	result.Metadata["job_id"] = job.ID
	result.Metadata["agent_id"] = ga.GetID()
	result.Metadata["processed_at"] = time.Now()

	return result, nil
}

func (ga *GenerationAgent) trackJobProgress(ctx context.Context, job *GenerationJob) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if job.Status != JobStatusRunning {
				return
			}

			// Get progress from generator
			progress, err := job.Generator.GetProgress(job.Request.ID)
			if err == nil {
				job.Progress = progress
			}

			// Update metadata
			job.Metadata["last_progress_update"] = time.Now()
		}
	}
}

func (ga *GenerationAgent) shouldRetryJob(job *GenerationJob) bool {
	return job.RetryCount < job.MaxRetries && 
		   job.Error != nil && 
		   !errors.IsValidationError(job.Error)
}

func (ga *GenerationAgent) retryJob(ctx context.Context, job *GenerationJob) {
	job.RetryCount++
	job.Status = JobStatusRetrying

	ga.logger.WithFields(logrus.Fields{
		"job_id":      job.ID,
		"retry_count": job.RetryCount,
		"max_retries": job.MaxRetries,
	}).Info("Retrying generation job")

	// Wait before retry
	time.Sleep(5 * time.Second)

	// Reset job state
	job.Progress = 0.0
	job.Error = nil
	job.ErrorMessage = ""

	// Resubmit to queue
	select {
	case ga.jobQueue <- job:
		// Successfully requeued
	default:
		// Queue full, mark as failed
		job.Status = JobStatusFailed
		job.Error = errors.NewProcessingError("RETRY_QUEUE_FULL", "Failed to requeue job for retry")
		ga.moveJobToHistory(job)
	}
}

func (ga *GenerationAgent) moveJobToHistory(job *GenerationJob) {
	ga.mu.Lock()
	defer ga.mu.Unlock()

	// Remove from active jobs
	delete(ga.activeJobs, job.ID)

	// Add to history
	historyEntry := JobHistoryEntry{
		JobID:        job.ID,
		GeneratorType: string(job.Request.GeneratorType),
		StartTime:    job.StartTime,
		EndTime:      job.EndTime,
		Duration:     job.Duration,
		Status:       job.Status,
		Success:      job.Status == JobStatusCompleted,
		Metadata:     job.Metadata,
	}

	if job.Result != nil && job.Result.TimeSeries != nil {
		historyEntry.DataPoints = int64(len(job.Result.TimeSeries.DataPoints))
	}

	if job.Error != nil {
		historyEntry.Error = job.Error.Error()
	}

	ga.jobHistory = append(ga.jobHistory, historyEntry)

	// Trim history if too large
	if len(ga.jobHistory) > ga.maxHistorySize {
		ga.jobHistory = ga.jobHistory[len(ga.jobHistory)-ga.maxHistorySize:]
	}
}

func (ga *GenerationAgent) monitorJobs(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			ga.cleanupExpiredJobs()
		}
	}
}

func (ga *GenerationAgent) cleanupExpiredJobs() {
	ga.mu.Lock()
	defer ga.mu.Unlock()

	now := time.Now()
	expiredJobs := []string{}

	for jobID, job := range ga.activeJobs {
		// Check for timeout
		if now.Sub(job.StartTime) > 1*time.Hour { // Default timeout
			expiredJobs = append(expiredJobs, jobID)
		}
	}

	// Cancel expired jobs
	for _, jobID := range expiredJobs {
		job := ga.activeJobs[jobID]
		if job.CancelFunc != nil {
			job.CancelFunc()
		}
		job.Status = JobStatusFailed
		job.Error = errors.NewTimeoutError("JOB_TIMEOUT", "Job exceeded maximum execution time")
		job.EndTime = now
		job.Duration = job.EndTime.Sub(job.StartTime)

		ga.logger.WithFields(logrus.Fields{
			"job_id":   jobID,
			"duration": job.Duration,
		}).Warn("Generation job timed out")
	}
}

// Task handlers

func (ga *GenerationAgent) handleGenerateTask(ctx context.Context, task *models.AgentTask) (*models.TaskResult, error) {
	// Extract generation request from task data
	requestData, ok := task.Data["request"]
	if !ok {
		return nil, errors.NewValidationError("INVALID_TASK", "Generation request not found in task data")
	}

	// Convert to generation request (this would need proper type conversion)
	request, ok := requestData.(*models.GenerationRequest)
	if !ok {
		return nil, errors.NewValidationError("INVALID_TASK", "Invalid generation request format")
	}

	// Submit job
	job, err := ga.SubmitGenerationJob(ctx, request)
	if err != nil {
		return nil, err
	}

	// Create task result
	result := &models.TaskResult{
		Success:   true,
		Data:      map[string]interface{}{
			"job_id": job.ID,
			"status": string(job.Status),
		},
		Metrics:   map[string]float64{},
		Metadata:  map[string]interface{}{
			"job_id": job.ID,
			"generator_type": string(request.GeneratorType),
		},
		Duration:  time.Since(time.Now()),
		CreatedAt: time.Now(),
	}

	return result, nil
}

func (ga *GenerationAgent) handleCancelJobTask(ctx context.Context, task *models.AgentTask) (*models.TaskResult, error) {
	jobID, ok := task.Data["job_id"].(string)
	if !ok {
		return nil, errors.NewValidationError("INVALID_TASK", "Job ID not found in task data")
	}

	err := ga.CancelJob(jobID)
	if err != nil {
		return nil, err
	}

	result := &models.TaskResult{
		Success:   true,
		Data:      map[string]interface{}{
			"job_id": jobID,
			"status": "cancelled",
		},
		Duration:  time.Since(time.Now()),
		CreatedAt: time.Now(),
	}

	return result, nil
}

func (ga *GenerationAgent) handleGetJobStatusTask(ctx context.Context, task *models.AgentTask) (*models.TaskResult, error) {
	jobID, ok := task.Data["job_id"].(string)
	if !ok {
		return nil, errors.NewValidationError("INVALID_TASK", "Job ID not found in task data")
	}

	job, err := ga.GetJobStatus(jobID)
	if err != nil {
		return nil, err
	}

	result := &models.TaskResult{
		Success:   true,
		Data:      map[string]interface{}{
			"job_id":   job.ID,
			"status":   string(job.Status),
			"progress": job.Progress,
			"duration": job.Duration.String(),
		},
		Duration:  time.Since(time.Now()),
		CreatedAt: time.Now(),
	}

	return result, nil
}

func (ga *GenerationAgent) handleListJobsTask(ctx context.Context, task *models.AgentTask) (*models.TaskResult, error) {
	jobs := ga.ListActiveJobs()

	jobsData := make([]map[string]interface{}, len(jobs))
	for i, job := range jobs {
		jobsData[i] = map[string]interface{}{
			"job_id":   job.ID,
			"status":   string(job.Status),
			"progress": job.Progress,
			"start_time": job.StartTime,
		}
	}

	result := &models.TaskResult{
		Success:   true,
		Data:      map[string]interface{}{
			"jobs": jobsData,
			"count": len(jobs),
		},
		Duration:  time.Since(time.Now()),
		CreatedAt: time.Now(),
	}

	return result, nil
}

func (ga *GenerationAgent) handleGetGeneratorInfoTask(ctx context.Context, task *models.AgentTask) (*models.TaskResult, error) {
	availableGenerators := ga.generatorFactory.GetAvailableGenerators()

	generatorInfo := make([]map[string]interface{}, len(availableGenerators))
	for i, genType := range availableGenerators {
		generator, err := ga.generatorFactory.CreateGenerator(genType)
		if err != nil {
			continue
		}

		generatorInfo[i] = map[string]interface{}{
			"type":               string(genType),
			"name":               generator.GetName(),
			"description":        generator.GetDescription(),
			"supported_sensors":  generator.GetSupportedSensorTypes(),
			"is_trainable":       generator.IsTrainable(),
		}

		generator.Close()
	}

	result := &models.TaskResult{
		Success:   true,
		Data:      map[string]interface{}{
			"generators": generatorInfo,
			"count":      len(generatorInfo),
		},
		Duration:  time.Since(time.Now()),
		CreatedAt: time.Now(),
	}

	return result, nil
}

// Helper methods

func (ga *GenerationAgent) validateGenerationRequest(request *models.GenerationRequest) error {
	if request.ID == "" {
		return errors.NewValidationError("INVALID_REQUEST", "Request ID is required")
	}

	if request.GeneratorType == "" {
		return errors.NewValidationError("INVALID_REQUEST", "Generator type is required")
	}

	if !ga.generatorFactory.IsSupported(request.GeneratorType) {
		return errors.NewValidationError("UNSUPPORTED_GENERATOR", fmt.Sprintf("Generator type %s is not supported", request.GeneratorType))
	}

	return nil
}

func (ga *GenerationAgent) setupEventHandlers() {
	ga.SetEventHandler("onStart", func(agent base.Agent) error {
		ga.logger.WithFields(logrus.Fields{
			"agent_id": agent.GetID(),
		}).Info("Generation agent started")
		return nil
	})

	ga.SetEventHandler("onStop", func(agent base.Agent) error {
		ga.logger.WithFields(logrus.Fields{
			"agent_id": agent.GetID(),
		}).Info("Generation agent stopped")
		return nil
	})

	ga.SetEventHandler("onError", func(agent base.Agent, err error) {
		ga.logger.WithFields(logrus.Fields{
			"agent_id": agent.GetID(),
			"error":    err.Error(),
		}).Error("Generation agent error")
	})
}

