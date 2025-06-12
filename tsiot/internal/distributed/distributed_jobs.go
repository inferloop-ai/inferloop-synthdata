package distributed

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/models"
)

// DistributedJobManager manages distributed computing jobs
type DistributedJobManager struct {
	logger     *logrus.Logger
	cluster    *SparkCluster
	scheduler  JobScheduler
	executor   JobExecutor
	jobQueue   chan *SparkJob
	workers    chan struct{}
	stopCh     chan struct{}
	mu         sync.RWMutex
}

// JobScheduler interface for job scheduling
type JobScheduler interface {
	ScheduleJob(job *SparkJob) error
	GetNextJob() (*SparkJob, error)
	UpdateJobPriority(jobID string, priority JobPriority) error
	GetQueueStatus() *QueueStatus
}

// JobExecutor interface for job execution
type JobExecutor interface {
	ExecuteJob(ctx context.Context, job *SparkJob) (*JobResult, error)
	GetSupportedJobTypes() []JobType
	EstimateResourceRequirements(job *SparkJob) (*ResourceRequirements, error)
}

// QueueStatus represents the status of the job queue
type QueueStatus struct {
	QueuedJobs    int                        `json:"queued_jobs"`
	RunningJobs   int                        `json:"running_jobs"`
	CompletedJobs int                        `json:"completed_jobs"`
	FailedJobs    int                        `json:"failed_jobs"`
	JobsByType    map[JobType]int            `json:"jobs_by_type"`
	JobsByPriority map[JobPriority]int       `json:"jobs_by_priority"`
	AverageWaitTime time.Duration           `json:"average_wait_time"`
	EstimatedQueueTime time.Duration        `json:"estimated_queue_time"`
}

// JobResult contains the result of a distributed job
type JobResult struct {
	JobID         string                 `json:"job_id"`
	Success       bool                   `json:"success"`
	Output        interface{}            `json:"output"`
	OutputData    *JobOutputData         `json:"output_data"`
	Metrics       *JobMetrics            `json:"metrics"`
	Duration      time.Duration          `json:"duration"`
	Error         error                  `json:"error,omitempty"`
	Artifacts     []JobArtifact          `json:"artifacts"`
	Metadata      map[string]interface{} `json:"metadata"`
}

// JobArtifact represents an output artifact from a job
type JobArtifact struct {
	Type        string    `json:"type"` // model, data, report, log
	Name        string    `json:"name"`
	Location    string    `json:"location"`
	Size        int64     `json:"size"`
	Format      string    `json:"format"`
	Checksum    string    `json:"checksum"`
	CreatedAt   time.Time `json:"created_at"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// NewDistributedJobManager creates a new distributed job manager
func NewDistributedJobManager(cluster *SparkCluster, logger *logrus.Logger) *DistributedJobManager {
	djm := &DistributedJobManager{
		logger:    logger,
		cluster:   cluster,
		jobQueue:  make(chan *SparkJob, 1000),
		workers:   make(chan struct{}, 10),
		stopCh:    make(chan struct{}),
	}

	djm.scheduler = NewFIFOJobScheduler(logger)
	djm.executor = NewDefaultJobExecutor(cluster, logger)

	return djm
}

// Start starts the distributed job manager
func (djm *DistributedJobManager) Start(ctx context.Context) error {
	djm.logger.Info("Starting distributed job manager")

	// Start worker pool
	for i := 0; i < cap(djm.workers); i++ {
		go djm.worker(ctx)
	}

	// Start job scheduler
	go djm.schedulerLoop(ctx)

	return nil
}

// Stop stops the distributed job manager
func (djm *DistributedJobManager) Stop(ctx context.Context) error {
	djm.logger.Info("Stopping distributed job manager")

	close(djm.stopCh)
	close(djm.jobQueue)

	return nil
}

// SubmitGenerationJob submits a distributed data generation job
func (djm *DistributedJobManager) SubmitGenerationJob(ctx context.Context, request *GenerationJobRequest) (*SparkJob, error) {
	job := &SparkJob{
		ID:        fmt.Sprintf("gen_%d", time.Now().UnixNano()),
		Name:      request.Name,
		JobType:   JobTypeGeneration,
		Priority:  request.Priority,
		Configuration: &JobConfiguration{
			Parameters: map[string]interface{}{
				"generator_type": request.GeneratorType,
				"length":         request.Length,
				"frequency":      request.Frequency,
				"parameters":     request.Parameters,
			},
			Resources: request.Resources,
			Timeout:   request.Timeout,
		},
		InputData: &JobInputData{
			Format:   "json",
			Metadata: request.InputMetadata,
		},
	}

	if err := djm.scheduler.ScheduleJob(job); err != nil {
		return nil, fmt.Errorf("failed to schedule generation job: %w", err)
	}

	djm.logger.WithFields(logrus.Fields{
		"job_id":         job.ID,
		"generator_type": request.GeneratorType,
		"length":         request.Length,
	}).Info("Submitted distributed generation job")

	return job, nil
}

// SubmitValidationJob submits a distributed validation job
func (djm *DistributedJobManager) SubmitValidationJob(ctx context.Context, request *ValidationJobRequest) (*SparkJob, error) {
	job := &SparkJob{
		ID:       fmt.Sprintf("val_%d", time.Now().UnixNano()),
		Name:     request.Name,
		JobType:  JobTypeValidation,
		Priority: request.Priority,
		Configuration: &JobConfiguration{
			Parameters: map[string]interface{}{
				"validation_rules": request.ValidationRules,
				"reference_data":   request.ReferenceData,
				"metrics":          request.Metrics,
			},
			Resources: request.Resources,
			Timeout:   request.Timeout,
		},
		InputData: &JobInputData{
			Format:   "parquet",
			Location: request.DataLocation,
			Metadata: request.InputMetadata,
		},
	}

	if err := djm.scheduler.ScheduleJob(job); err != nil {
		return nil, fmt.Errorf("failed to schedule validation job: %w", err)
	}

	djm.logger.WithFields(logrus.Fields{
		"job_id":      job.ID,
		"data_location": request.DataLocation,
		"rules":       len(request.ValidationRules),
	}).Info("Submitted distributed validation job")

	return job, nil
}

// SubmitAnalyticsJob submits a distributed analytics job
func (djm *DistributedJobManager) SubmitAnalyticsJob(ctx context.Context, request *AnalyticsJobRequest) (*SparkJob, error) {
	job := &SparkJob{
		ID:       fmt.Sprintf("analytics_%d", time.Now().UnixNano()),
		Name:     request.Name,
		JobType:  JobTypeAnalytics,
		Priority: request.Priority,
		Configuration: &JobConfiguration{
			Parameters: map[string]interface{}{
				"analysis_type": request.AnalysisType,
				"algorithms":    request.Algorithms,
				"parameters":    request.Parameters,
			},
			Resources: request.Resources,
			Timeout:   request.Timeout,
		},
		InputData: &JobInputData{
			Format:   request.InputFormat,
			Location: request.DataLocation,
			Metadata: request.InputMetadata,
		},
	}

	if err := djm.scheduler.ScheduleJob(job); err != nil {
		return nil, fmt.Errorf("failed to schedule analytics job: %w", err)
	}

	djm.logger.WithFields(logrus.Fields{
		"job_id":       job.ID,
		"analysis_type": request.AnalysisType,
		"algorithms":   request.Algorithms,
	}).Info("Submitted distributed analytics job")

	return job, nil
}

// SubmitTrainingJob submits a distributed model training job
func (djm *DistributedJobManager) SubmitTrainingJob(ctx context.Context, request *TrainingJobRequest) (*SparkJob, error) {
	job := &SparkJob{
		ID:       fmt.Sprintf("train_%d", time.Now().UnixNano()),
		Name:     request.Name,
		JobType:  JobTypeTraining,
		Priority: request.Priority,
		Configuration: &JobConfiguration{
			Parameters: map[string]interface{}{
				"model_type":       request.ModelType,
				"algorithm":        request.Algorithm,
				"hyperparameters":  request.Hyperparameters,
				"training_config":  request.TrainingConfig,
			},
			Resources:           request.Resources,
			Timeout:            request.Timeout,
			EnableCheckpointing: true,
			CheckpointInterval: 10 * time.Minute,
		},
		InputData: &JobInputData{
			Format:   request.InputFormat,
			Location: request.TrainingDataLocation,
			Metadata: request.InputMetadata,
		},
	}

	if err := djm.scheduler.ScheduleJob(job); err != nil {
		return nil, fmt.Errorf("failed to schedule training job: %w", err)
	}

	djm.logger.WithFields(logrus.Fields{
		"job_id":     job.ID,
		"model_type": request.ModelType,
		"algorithm":  request.Algorithm,
	}).Info("Submitted distributed training job")

	return job, nil
}

// GetJobResult returns the result of a completed job
func (djm *DistributedJobManager) GetJobResult(jobID string) (*JobResult, error) {
	job, err := djm.cluster.GetJobStatus(jobID)
	if err != nil {
		return nil, fmt.Errorf("failed to get job status: %w", err)
	}

	if job.State != JobStateCompleted && job.State != JobStateFailed {
		return nil, fmt.Errorf("job not completed: %s", job.State)
	}

	// Execute job to get result
	result, err := djm.executor.ExecuteJob(context.Background(), job)
	if err != nil {
		return nil, fmt.Errorf("failed to get job result: %w", err)
	}

	return result, nil
}

// GetQueueStatus returns the current queue status
func (djm *DistributedJobManager) GetQueueStatus() *QueueStatus {
	return djm.scheduler.GetQueueStatus()
}

// Worker methods

func (djm *DistributedJobManager) worker(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-djm.stopCh:
			return
		case job, ok := <-djm.jobQueue:
			if !ok {
				return
			}

			djm.processJob(ctx, job)
		}
	}
}

func (djm *DistributedJobManager) processJob(ctx context.Context, job *SparkJob) {
	djm.logger.WithField("job_id", job.ID).Info("Processing distributed job")

	// Execute job
	result, err := djm.executor.ExecuteJob(ctx, job)
	if err != nil {
		djm.logger.WithError(err).WithField("job_id", job.ID).Error("Job execution failed")
		job.State = JobStateFailed
		job.Error = err
	} else {
		djm.logger.WithField("job_id", job.ID).Info("Job execution completed")
		job.State = JobStateCompleted
	}

	// Update job completion time
	completedAt := time.Now()
	job.CompletedAt = &completedAt
	job.Duration = completedAt.Sub(*job.StartedAt)

	// Store result if successful
	if result != nil {
		job.OutputData = result.OutputData
		job.Metrics = result.Metrics
	}
}

func (djm *DistributedJobManager) schedulerLoop(ctx context.Context) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-djm.stopCh:
			return
		case <-ticker.C:
			// Get next job from scheduler
			job, err := djm.scheduler.GetNextJob()
			if err != nil || job == nil {
				continue
			}

			// Submit to job queue
			select {
			case djm.jobQueue <- job:
				job.State = JobStateRunning
				startTime := time.Now()
				job.StartedAt = &startTime
			default:
				// Queue full, try again later
				djm.logger.WithField("job_id", job.ID).Warn("Job queue full, retrying")
			}
		}
	}
}

// Job request types

type GenerationJobRequest struct {
	Name           string                 `json:"name"`
	GeneratorType  string                 `json:"generator_type"`
	Length         int                    `json:"length"`
	Frequency      string                 `json:"frequency"`
	Parameters     map[string]interface{} `json:"parameters"`
	Priority       JobPriority            `json:"priority"`
	Resources      *ResourceRequirements  `json:"resources"`
	Timeout        time.Duration          `json:"timeout"`
	InputMetadata  map[string]interface{} `json:"input_metadata"`
}

type ValidationJobRequest struct {
	Name            string                 `json:"name"`
	DataLocation    string                 `json:"data_location"`
	ValidationRules []string               `json:"validation_rules"`
	ReferenceData   string                 `json:"reference_data"`
	Metrics         []string               `json:"metrics"`
	Priority        JobPriority            `json:"priority"`
	Resources       *ResourceRequirements  `json:"resources"`
	Timeout         time.Duration          `json:"timeout"`
	InputMetadata   map[string]interface{} `json:"input_metadata"`
}

type AnalyticsJobRequest struct {
	Name          string                 `json:"name"`
	AnalysisType  string                 `json:"analysis_type"`
	DataLocation  string                 `json:"data_location"`
	Algorithms    []string               `json:"algorithms"`
	Parameters    map[string]interface{} `json:"parameters"`
	InputFormat   string                 `json:"input_format"`
	Priority      JobPriority            `json:"priority"`
	Resources     *ResourceRequirements  `json:"resources"`
	Timeout       time.Duration          `json:"timeout"`
	InputMetadata map[string]interface{} `json:"input_metadata"`
}

type TrainingJobRequest struct {
	Name                 string                 `json:"name"`
	ModelType            string                 `json:"model_type"`
	Algorithm            string                 `json:"algorithm"`
	TrainingDataLocation string                 `json:"training_data_location"`
	Hyperparameters      map[string]interface{} `json:"hyperparameters"`
	TrainingConfig       map[string]interface{} `json:"training_config"`
	InputFormat          string                 `json:"input_format"`
	Priority             JobPriority            `json:"priority"`
	Resources            *ResourceRequirements  `json:"resources"`
	Timeout              time.Duration          `json:"timeout"`
	InputMetadata        map[string]interface{} `json:"input_metadata"`
}

// FIFO Job Scheduler implementation
type FIFOJobScheduler struct {
	logger     *logrus.Logger
	jobQueue   []*SparkJob
	mu         sync.RWMutex
	queueStats *QueueStatus
}

func NewFIFOJobScheduler(logger *logrus.Logger) *FIFOJobScheduler {
	return &FIFOJobScheduler{
		logger:     logger,
		jobQueue:   make([]*SparkJob, 0),
		queueStats: &QueueStatus{
			JobsByType:     make(map[JobType]int),
			JobsByPriority: make(map[JobPriority]int),
		},
	}
}

func (fs *FIFOJobScheduler) ScheduleJob(job *SparkJob) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	job.State = JobStateQueued
	fs.jobQueue = append(fs.jobQueue, job)
	fs.updateQueueStats()

	return nil
}

func (fs *FIFOJobScheduler) GetNextJob() (*SparkJob, error) {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	if len(fs.jobQueue) == 0 {
		return nil, nil
	}

	// Sort by priority (critical > high > normal > low)
	job := fs.jobQueue[0]
	for i, j := range fs.jobQueue {
		if fs.getPriorityValue(j.Priority) > fs.getPriorityValue(job.Priority) {
			job = j
			fs.jobQueue = append(fs.jobQueue[:i], fs.jobQueue[i+1:]...)
			break
		}
	}

	if job == fs.jobQueue[0] {
		fs.jobQueue = fs.jobQueue[1:]
	}

	fs.updateQueueStats()
	return job, nil
}

func (fs *FIFOJobScheduler) UpdateJobPriority(jobID string, priority JobPriority) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	for _, job := range fs.jobQueue {
		if job.ID == jobID {
			job.Priority = priority
			fs.updateQueueStats()
			return nil
		}
	}

	return fmt.Errorf("job not found: %s", jobID)
}

func (fs *FIFOJobScheduler) GetQueueStatus() *QueueStatus {
	fs.mu.RLock()
	defer fs.mu.RUnlock()
	return fs.queueStats
}

func (fs *FIFOJobScheduler) getPriorityValue(priority JobPriority) int {
	switch priority {
	case JobPriorityCritical:
		return 4
	case JobPriorityHigh:
		return 3
	case JobPriorityNormal:
		return 2
	case JobPriorityLow:
		return 1
	default:
		return 0
	}
}

func (fs *FIFOJobScheduler) updateQueueStats() {
	fs.queueStats.QueuedJobs = len(fs.jobQueue)
	
	// Reset counters
	fs.queueStats.JobsByType = make(map[JobType]int)
	fs.queueStats.JobsByPriority = make(map[JobPriority]int)

	for _, job := range fs.jobQueue {
		fs.queueStats.JobsByType[job.JobType]++
		fs.queueStats.JobsByPriority[job.Priority]++
	}
}

// Default Job Executor implementation
type DefaultJobExecutor struct {
	cluster *SparkCluster
	logger  *logrus.Logger
}

func NewDefaultJobExecutor(cluster *SparkCluster, logger *logrus.Logger) *DefaultJobExecutor {
	return &DefaultJobExecutor{
		cluster: cluster,
		logger:  logger,
	}
}

func (de *DefaultJobExecutor) ExecuteJob(ctx context.Context, job *SparkJob) (*JobResult, error) {
	startTime := time.Now()

	// Simulate job execution based on type
	var output interface{}
	var artifacts []JobArtifact
	var err error

	switch job.JobType {
	case JobTypeGeneration:
		output, artifacts, err = de.executeGenerationJob(ctx, job)
	case JobTypeValidation:
		output, artifacts, err = de.executeValidationJob(ctx, job)
	case JobTypeAnalytics:
		output, artifacts, err = de.executeAnalyticsJob(ctx, job)
	case JobTypeTraining:
		output, artifacts, err = de.executeTrainingJob(ctx, job)
	default:
		err = fmt.Errorf("unsupported job type: %s", job.JobType)
	}

	duration := time.Since(startTime)

	result := &JobResult{
		JobID:    job.ID,
		Success:  err == nil,
		Output:   output,
		Duration: duration,
		Error:    err,
		Artifacts: artifacts,
		Metadata: map[string]interface{}{
			"execution_time": duration.String(),
			"executor":       "default",
		},
	}

	if err == nil {
		result.Metrics = &JobMetrics{
			TaskCount:       rand.Intn(100) + 10,
			SuccessfulTasks: rand.Intn(90) + 10,
			TotalTaskTime:   int64(duration.Milliseconds()),
		}
	}

	return result, nil
}

func (de *DefaultJobExecutor) GetSupportedJobTypes() []JobType {
	return []JobType{
		JobTypeGeneration,
		JobTypeValidation,
		JobTypeAnalytics,
		JobTypeTraining,
	}
}

func (de *DefaultJobExecutor) EstimateResourceRequirements(job *SparkJob) (*ResourceRequirements, error) {
	// Basic resource estimation
	requirements := &ResourceRequirements{
		ExecutorInstances: 2,
		ExecutorCores:     2,
		ExecutorMemory:    "2g",
		DriverMemory:      "1g",
		MaxCores:          4,
	}

	// Adjust based on job type
	switch job.JobType {
	case JobTypeTraining:
		requirements.ExecutorInstances = 4
		requirements.ExecutorMemory = "4g"
		requirements.MaxCores = 8
	case JobTypeAnalytics:
		requirements.ExecutorInstances = 3
		requirements.ExecutorMemory = "3g"
		requirements.MaxCores = 6
	}

	return requirements, nil
}

func (de *DefaultJobExecutor) executeGenerationJob(ctx context.Context, job *SparkJob) (interface{}, []JobArtifact, error) {
	// Simulate time series generation
	time.Sleep(time.Duration(rand.Intn(5)+1) * time.Second)

	generatorType := job.Configuration.Parameters["generator_type"].(string)
	length := job.Configuration.Parameters["length"].(int)

	// Create mock time series data
	timeSeries := &models.TimeSeries{
		ID:         fmt.Sprintf("generated_%d", time.Now().UnixNano()),
		Name:       fmt.Sprintf("Generated %s Series", generatorType),
		SensorType: "synthetic",
		DataPoints: make([]models.DataPoint, length),
	}

	// Generate mock data points
	for i := 0; i < length; i++ {
		timeSeries.DataPoints[i] = models.DataPoint{
			Timestamp: time.Now().Add(time.Duration(i) * time.Minute),
			Value:     rand.Float64() * 100,
			Quality:   1.0,
		}
	}

	artifacts := []JobArtifact{
		{
			Type:      "data",
			Name:      "generated_timeseries.json",
			Location:  fmt.Sprintf("/data/generated/%s.json", timeSeries.ID),
			Size:      int64(length * 50), // Estimated size
			Format:    "json",
			CreatedAt: time.Now(),
		},
	}

	return timeSeries, artifacts, nil
}

func (de *DefaultJobExecutor) executeValidationJob(ctx context.Context, job *SparkJob) (interface{}, []JobArtifact, error) {
	// Simulate validation
	time.Sleep(time.Duration(rand.Intn(3)+1) * time.Second)

	rules := job.Configuration.Parameters["validation_rules"].([]string)
	
	result := map[string]interface{}{
		"validation_summary": map[string]interface{}{
			"total_rules":   len(rules),
			"passed_rules":  len(rules) - 1,
			"failed_rules":  1,
			"overall_score": 0.85,
		},
		"rule_results": make([]map[string]interface{}, len(rules)),
	}

	artifacts := []JobArtifact{
		{
			Type:      "report",
			Name:      "validation_report.json",
			Location:  fmt.Sprintf("/reports/validation_%s.json", job.ID),
			Size:      2048,
			Format:    "json",
			CreatedAt: time.Now(),
		},
	}

	return result, artifacts, nil
}

func (de *DefaultJobExecutor) executeAnalyticsJob(ctx context.Context, job *SparkJob) (interface{}, []JobArtifact, error) {
	// Simulate analytics
	time.Sleep(time.Duration(rand.Intn(4)+1) * time.Second)

	analysisType := job.Configuration.Parameters["analysis_type"].(string)
	algorithms := job.Configuration.Parameters["algorithms"].([]string)

	result := map[string]interface{}{
		"analysis_type": analysisType,
		"algorithms":    algorithms,
		"results": map[string]interface{}{
			"patterns_detected": rand.Intn(10) + 1,
			"anomalies_found":   rand.Intn(5),
			"trend_analysis":    "upward",
			"seasonality":       "weekly",
		},
		"metrics": map[string]float64{
			"accuracy":  0.85 + rand.Float64()*0.10,
			"precision": 0.80 + rand.Float64()*0.15,
			"recall":    0.75 + rand.Float64()*0.20,
		},
	}

	artifacts := []JobArtifact{
		{
			Type:      "report",
			Name:      "analytics_report.json",
			Location:  fmt.Sprintf("/reports/analytics_%s.json", job.ID),
			Size:      4096,
			Format:    "json",
			CreatedAt: time.Now(),
		},
	}

	return result, artifacts, nil
}

func (de *DefaultJobExecutor) executeTrainingJob(ctx context.Context, job *SparkJob) (interface{}, []JobArtifact, error) {
	// Simulate model training
	time.Sleep(time.Duration(rand.Intn(10)+5) * time.Second)

	modelType := job.Configuration.Parameters["model_type"].(string)
	algorithm := job.Configuration.Parameters["algorithm"].(string)

	result := map[string]interface{}{
		"model_type": modelType,
		"algorithm":  algorithm,
		"training_metrics": map[string]float64{
			"loss":         0.05 + rand.Float64()*0.10,
			"accuracy":     0.90 + rand.Float64()*0.08,
			"val_loss":     0.08 + rand.Float64()*0.12,
			"val_accuracy": 0.88 + rand.Float64()*0.10,
		},
		"epochs_completed": rand.Intn(50) + 10,
		"model_size":       rand.Intn(100) + 50, // MB
	}

	artifacts := []JobArtifact{
		{
			Type:      "model",
			Name:      "trained_model.pkl",
			Location:  fmt.Sprintf("/models/%s_%s.pkl", modelType, job.ID),
			Size:      int64(rand.Intn(100)+50) * 1024 * 1024, // 50-150 MB
			Format:    "pickle",
			CreatedAt: time.Now(),
		},
		{
			Type:      "report",
			Name:      "training_report.json",
			Location:  fmt.Sprintf("/reports/training_%s.json", job.ID),
			Size:      8192,
			Format:    "json",
			CreatedAt: time.Now(),
		},
	}

	return result, artifacts, nil
}