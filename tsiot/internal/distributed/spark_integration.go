package distributed

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/models"
)

// SparkCluster represents a distributed computing cluster using Apache Spark
type SparkCluster struct {
	logger      *logrus.Logger
	config      *SparkConfig
	client      SparkClient
	sessions    map[string]*SparkSession
	jobs        map[string]*SparkJob
	metrics     *ClusterMetrics
	mu          sync.RWMutex
	stopCh      chan struct{}
	wg          sync.WaitGroup
}

// SparkConfig configures the Spark cluster
type SparkConfig struct {
	Enabled              bool              `json:"enabled"`
	MasterURL            string            `json:"master_url"`
	AppName              string            `json:"app_name"`
	ExecutorInstances    int               `json:"executor_instances"`
	ExecutorCores        int               `json:"executor_cores"`
	ExecutorMemory       string            `json:"executor_memory"`
	DriverMemory         string            `json:"driver_memory"`
	MaxCores             int               `json:"max_cores"`
	SparkHome            string            `json:"spark_home"`
	PySparkPython        string            `json:"pyspark_python"`
	JavaHome             string            `json:"java_home"`
	SparkConf            map[string]string `json:"spark_conf"`
	DeployMode           string            `json:"deploy_mode"` // client, cluster
	Queue                string            `json:"queue"`
	Principal            string            `json:"principal"`
	Keytab               string            `json:"keytab"`
	EnableDynamicAllocation bool           `json:"enable_dynamic_allocation"`
	MinExecutors         int               `json:"min_executors"`
	MaxExecutors         int               `json:"max_executors"`
	InitialExecutors     int               `json:"initial_executors"`
	EnableHivesupport    bool              `json:"enable_hive_support"`
	EnableDeltaLake      bool              `json:"enable_delta_lake"`
	CheckpointDir        string            `json:"checkpoint_dir"`
	TempDir              string            `json:"temp_dir"`
	LogLevel             string            `json:"log_level"`
	EnableEventLog       bool              `json:"enable_event_log"`
	EventLogDir          string            `json:"event_log_dir"`
	SerializerClass      string            `json:"serializer_class"`
	NetworkTimeout       time.Duration     `json:"network_timeout"`
	HeartbeatInterval    time.Duration     `json:"heartbeat_interval"`
}

// SparkSession represents a Spark session
type SparkSession struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	AppID           string                 `json:"app_id"`
	State           SessionState           `json:"state"`
	CreatedAt       time.Time              `json:"created_at"`
	LastActivity    time.Time              `json:"last_activity"`
	Configuration   map[string]string      `json:"configuration"`
	ExecutorSummary []ExecutorSummary      `json:"executor_summary"`
	ActiveJobs      []string               `json:"active_jobs"`
	CompletedJobs   []string               `json:"completed_jobs"`
	FailedJobs      []string               `json:"failed_jobs"`
	Metrics         *SessionMetrics        `json:"metrics"`
}

// SessionState represents the state of a Spark session
type SessionState string

const (
	SessionStateStarting  SessionState = "starting"
	SessionStateIdle      SessionState = "idle"
	SessionStateBusy      SessionState = "busy"
	SessionStateShutdown  SessionState = "shutdown"
	SessionStateError     SessionState = "error"
)

// ExecutorSummary contains information about a Spark executor
type ExecutorSummary struct {
	ID              string    `json:"id"`
	HostPort        string    `json:"host_port"`
	IsActive        bool      `json:"is_active"`
	RDDBlocks       int       `json:"rdd_blocks"`
	MemoryUsed      int64     `json:"memory_used"`
	DiskUsed        int64     `json:"disk_used"`
	TotalCores      int       `json:"total_cores"`
	MaxTasks        int       `json:"max_tasks"`
	ActiveTasks     int       `json:"active_tasks"`
	FailedTasks     int       `json:"failed_tasks"`
	CompletedTasks  int       `json:"completed_tasks"`
	TotalTasks      int       `json:"total_tasks"`
	TaskTime        int64     `json:"task_time"`
	GCTime          int64     `json:"gc_time"`
	TotalInputBytes int64     `json:"total_input_bytes"`
	TotalShuffleRead int64    `json:"total_shuffle_read"`
	TotalShuffleWrite int64   `json:"total_shuffle_write"`
	LastHeartbeat   time.Time `json:"last_heartbeat"`
}

// SparkJob represents a distributed computing job
type SparkJob struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	SessionID       string                 `json:"session_id"`
	JobType         JobType                `json:"job_type"`
	State           JobState               `json:"state"`
	Priority        JobPriority            `json:"priority"`
	SubmittedAt     time.Time              `json:"submitted_at"`
	StartedAt       *time.Time             `json:"started_at,omitempty"`
	CompletedAt     *time.Time             `json:"completed_at,omitempty"`
	Duration        time.Duration          `json:"duration"`
	InputData       *JobInputData          `json:"input_data"`
	OutputData      *JobOutputData         `json:"output_data"`
	Configuration   *JobConfiguration      `json:"configuration"`
	Progress        *JobProgress           `json:"progress"`
	Error           error                  `json:"error,omitempty"`
	Metrics         *JobMetrics            `json:"metrics"`
	Dependencies    []string               `json:"dependencies"`
	Resources       *ResourceRequirements  `json:"resources"`
	Checkpoints     []Checkpoint           `json:"checkpoints"`
}

// JobType defines the type of distributed job
type JobType string

const (
	JobTypeGeneration    JobType = "generation"
	JobTypeValidation    JobType = "validation"
	JobTypeTransformation JobType = "transformation"
	JobTypeAnalytics     JobType = "analytics"
	JobTypeExport        JobType = "export"
	JobTypeTraining      JobType = "training"
	JobTypeInference     JobType = "inference"
	JobTypeAggregation   JobType = "aggregation"
)

// JobState represents the state of a job
type JobState string

const (
	JobStateQueued    JobState = "queued"
	JobStateRunning   JobState = "running"
	JobStateCompleted JobState = "completed"
	JobStateFailed    JobState = "failed"
	JobStateCancelled JobState = "cancelled"
	JobStatePaused    JobState = "paused"
)

// JobPriority defines job priority levels
type JobPriority string

const (
	JobPriorityLow    JobPriority = "low"
	JobPriorityNormal JobPriority = "normal"
	JobPriorityHigh   JobPriority = "high"
	JobPriorityCritical JobPriority = "critical"
)

// JobInputData describes input data for a job
type JobInputData struct {
	Format       string                 `json:"format"`
	Location     string                 `json:"location"`
	Schema       string                 `json:"schema"`
	Partitions   int                    `json:"partitions"`
	Size         int64                  `json:"size"`
	RecordCount  int64                  `json:"record_count"`
	Metadata     map[string]interface{} `json:"metadata"`
	Compression  string                 `json:"compression"`
}

// JobOutputData describes output data for a job
type JobOutputData struct {
	Format      string                 `json:"format"`
	Location    string                 `json:"location"`
	Partitions  int                    `json:"partitions"`
	Size        int64                  `json:"size"`
	RecordCount int64                  `json:"record_count"`
	Metadata    map[string]interface{} `json:"metadata"`
	Compression string                 `json:"compression"`
}

// JobConfiguration contains job-specific configuration
type JobConfiguration struct {
	SparkConf           map[string]string      `json:"spark_conf"`
	Resources           *ResourceRequirements  `json:"resources"`
	CheckpointInterval  time.Duration          `json:"checkpoint_interval"`
	MaxRetries          int                    `json:"max_retries"`
	RetryBackoff        time.Duration          `json:"retry_backoff"`
	Timeout             time.Duration          `json:"timeout"`
	EnableCheckpointing bool                   `json:"enable_checkpointing"`
	EnableCache         bool                   `json:"enable_cache"`
	CacheLevel          string                 `json:"cache_level"`
	Broadcast           []string               `json:"broadcast"`
	Parameters          map[string]interface{} `json:"parameters"`
}

// ResourceRequirements specifies resource requirements for a job
type ResourceRequirements struct {
	ExecutorInstances int    `json:"executor_instances"`
	ExecutorCores     int    `json:"executor_cores"`
	ExecutorMemory    string `json:"executor_memory"`
	DriverMemory      string `json:"driver_memory"`
	MaxCores          int    `json:"max_cores"`
	Queue             string `json:"queue"`
}

// JobProgress tracks job execution progress
type JobProgress struct {
	TotalStages       int     `json:"total_stages"`
	CompletedStages   int     `json:"completed_stages"`
	FailedStages      int     `json:"failed_stages"`
	TotalTasks        int     `json:"total_tasks"`
	CompletedTasks    int     `json:"completed_tasks"`
	FailedTasks       int     `json:"failed_tasks"`
	PercentComplete   float64 `json:"percent_complete"`
	EstimatedTimeLeft time.Duration `json:"estimated_time_left"`
}

// Checkpoint represents a job checkpoint
type Checkpoint struct {
	ID        string                 `json:"id"`
	Timestamp time.Time              `json:"timestamp"`
	State     map[string]interface{} `json:"state"`
	Location  string                 `json:"location"`
}

// SparkClient interface for Spark operations
type SparkClient interface {
	CreateSession(config *SparkConfig) (*SparkSession, error)
	SubmitJob(sessionID string, job *SparkJob) error
	GetJobStatus(sessionID, jobID string) (*SparkJob, error)
	CancelJob(sessionID, jobID string) error
	GetSessionMetrics(sessionID string) (*SessionMetrics, error)
	CloseSession(sessionID string) error
	ListSessions() ([]*SparkSession, error)
	GetClusterInfo() (*ClusterInfo, error)
}

// ClusterMetrics contains cluster-wide metrics
type ClusterMetrics struct {
	TotalCores           int       `json:"total_cores"`
	UsedCores            int       `json:"used_cores"`
	TotalMemory          int64     `json:"total_memory"`
	UsedMemory           int64     `json:"used_memory"`
	TotalExecutors       int       `json:"total_executors"`
	ActiveExecutors      int       `json:"active_executors"`
	TotalApplications    int       `json:"total_applications"`
	RunningApplications  int       `json:"running_applications"`
	CompletedJobs        int64     `json:"completed_jobs"`
	FailedJobs           int64     `json:"failed_jobs"`
	TotalTaskTime        int64     `json:"total_task_time"`
	ThroughputMBPS       float64   `json:"throughput_mbps"`
	LastUpdated          time.Time `json:"last_updated"`
}

// SessionMetrics contains session-specific metrics
type SessionMetrics struct {
	ExecutorCount       int       `json:"executor_count"`
	TotalCores          int       `json:"total_cores"`
	TotalMemory         int64     `json:"total_memory"`
	UsedMemory          int64     `json:"used_memory"`
	ActiveJobs          int       `json:"active_jobs"`
	CompletedJobs       int       `json:"completed_jobs"`
	FailedJobs          int       `json:"failed_jobs"`
	TotalTasks          int       `json:"total_tasks"`
	CompletedTasks      int       `json:"completed_tasks"`
	FailedTasks         int       `json:"failed_tasks"`
	TotalInputBytes     int64     `json:"total_input_bytes"`
	TotalOutputBytes    int64     `json:"total_output_bytes"`
	TotalShuffleRead    int64     `json:"total_shuffle_read"`
	TotalShuffleWrite   int64     `json:"total_shuffle_write"`
	GCTime              int64     `json:"gc_time"`
	LastUpdated         time.Time `json:"last_updated"`
}

// JobMetrics contains job-specific metrics
type JobMetrics struct {
	TaskCount           int       `json:"task_count"`
	SuccessfulTasks     int       `json:"successful_tasks"`
	FailedTasks         int       `json:"failed_tasks"`
	SkippedTasks        int       `json:"skipped_tasks"`
	TotalTaskTime       int64     `json:"total_task_time"`
	InputBytes          int64     `json:"input_bytes"`
	OutputBytes         int64     `json:"output_bytes"`
	ShuffleReadBytes    int64     `json:"shuffle_read_bytes"`
	ShuffleWriteBytes   int64     `json:"shuffle_write_bytes"`
	MemoryBytesSpilled  int64     `json:"memory_bytes_spilled"`
	DiskBytesSpilled    int64     `json:"disk_bytes_spilled"`
	PeakExecutionMemory int64     `json:"peak_execution_memory"`
	GCTime              int64     `json:"gc_time"`
	ResultSize          int64     `json:"result_size"`
}

// ClusterInfo contains information about the cluster
type ClusterInfo struct {
	MasterURL     string            `json:"master_url"`
	SparkVersion  string            `json:"spark_version"`
	ScalaVersion  string            `json:"scala_version"`
	JavaVersion   string            `json:"java_version"`
	Workers       []WorkerInfo      `json:"workers"`
	Applications  []ApplicationInfo `json:"applications"`
	Status        string            `json:"status"`
}

// WorkerInfo contains information about a cluster worker
type WorkerInfo struct {
	ID            string    `json:"id"`
	Host          string    `json:"host"`
	Port          int       `json:"port"`
	WebUIPort     int       `json:"webui_port"`
	Cores         int       `json:"cores"`
	CoresUsed     int       `json:"cores_used"`
	Memory        int64     `json:"memory"`
	MemoryUsed    int64     `json:"memory_used"`
	State         string    `json:"state"`
	LastHeartbeat time.Time `json:"last_heartbeat"`
}

// ApplicationInfo contains information about a Spark application
type ApplicationInfo struct {
	ID              string    `json:"id"`
	Name            string    `json:"name"`
	User            string    `json:"user"`
	State           string    `json:"state"`
	FinalStatus     string    `json:"final_status"`
	StartTime       time.Time `json:"start_time"`
	EndTime         *time.Time `json:"end_time,omitempty"`
	Duration        time.Duration `json:"duration"`
	SparkUser       string    `json:"spark_user"`
	CoresGranted    int       `json:"cores_granted"`
	MaxCores        int       `json:"max_cores"`
	MemoryPerNode   int64     `json:"memory_per_node"`
	ExecutorCount   int       `json:"executor_count"`
}

// NewSparkCluster creates a new Spark cluster instance
func NewSparkCluster(config *SparkConfig, logger *logrus.Logger) (*SparkCluster, error) {
	if config == nil {
		config = getDefaultSparkConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	cluster := &SparkCluster{
		logger:   logger,
		config:   config,
		sessions: make(map[string]*SparkSession),
		jobs:     make(map[string]*SparkJob),
		metrics:  &ClusterMetrics{},
		stopCh:   make(chan struct{}),
	}

	// Initialize Spark client
	client, err := NewMockSparkClient(config, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize Spark client: %w", err)
	}
	cluster.client = client

	return cluster, nil
}

// Start starts the Spark cluster
func (sc *SparkCluster) Start(ctx context.Context) error {
	if !sc.config.Enabled {
		sc.logger.Info("Spark cluster disabled")
		return nil
	}

	sc.logger.Info("Starting Spark cluster")

	// Start metrics collection
	go sc.metricsCollectionLoop(ctx)

	// Start job monitoring
	go sc.jobMonitoringLoop(ctx)

	return nil
}

// Stop stops the Spark cluster
func (sc *SparkCluster) Stop(ctx context.Context) error {
	sc.logger.Info("Stopping Spark cluster")

	close(sc.stopCh)

	// Close all sessions
	sc.mu.Lock()
	for sessionID := range sc.sessions {
		if err := sc.client.CloseSession(sessionID); err != nil {
			sc.logger.WithError(err).WithField("session_id", sessionID).Error("Failed to close session")
		}
	}
	sc.mu.Unlock()

	sc.wg.Wait()

	return nil
}

// CreateSession creates a new Spark session
func (sc *SparkCluster) CreateSession(ctx context.Context, name string) (*SparkSession, error) {
	session, err := sc.client.CreateSession(sc.config)
	if err != nil {
		return nil, fmt.Errorf("failed to create Spark session: %w", err)
	}

	session.Name = name
	session.CreatedAt = time.Now()
	session.LastActivity = time.Now()
	session.State = SessionStateIdle

	sc.mu.Lock()
	sc.sessions[session.ID] = session
	sc.mu.Unlock()

	sc.logger.WithFields(logrus.Fields{
		"session_id": session.ID,
		"name":       name,
	}).Info("Created Spark session")

	return session, nil
}

// SubmitJob submits a distributed computing job
func (sc *SparkCluster) SubmitJob(ctx context.Context, job *SparkJob) error {
	// Validate job
	if err := sc.validateJob(job); err != nil {
		return fmt.Errorf("invalid job: %w", err)
	}

	// Check if session exists
	sc.mu.RLock()
	session, exists := sc.sessions[job.SessionID]
	sc.mu.RUnlock()

	if !exists {
		return fmt.Errorf("session not found: %s", job.SessionID)
	}

	// Update job state
	job.State = JobStateQueued
	job.SubmittedAt = time.Now()
	job.Progress = &JobProgress{}

	// Store job
	sc.mu.Lock()
	sc.jobs[job.ID] = job
	session.ActiveJobs = append(session.ActiveJobs, job.ID)
	session.LastActivity = time.Now()
	session.State = SessionStateBusy
	sc.mu.Unlock()

	// Submit to Spark
	if err := sc.client.SubmitJob(job.SessionID, job); err != nil {
		job.State = JobStateFailed
		job.Error = err
		return fmt.Errorf("failed to submit job to Spark: %w", err)
	}

	job.State = JobStateRunning
	startTime := time.Now()
	job.StartedAt = &startTime

	sc.logger.WithFields(logrus.Fields{
		"job_id":     job.ID,
		"session_id": job.SessionID,
		"job_type":   job.JobType,
	}).Info("Submitted distributed job")

	return nil
}

// GetJobStatus returns the status of a job
func (sc *SparkCluster) GetJobStatus(jobID string) (*SparkJob, error) {
	sc.mu.RLock()
	job, exists := sc.jobs[jobID]
	sc.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("job not found: %s", jobID)
	}

	// Get latest status from Spark
	latestJob, err := sc.client.GetJobStatus(job.SessionID, jobID)
	if err != nil {
		return job, nil // Return cached version if Spark query fails
	}

	// Update cached job
	sc.mu.Lock()
	sc.jobs[jobID] = latestJob
	sc.mu.Unlock()

	return latestJob, nil
}

// CancelJob cancels a running job
func (sc *SparkCluster) CancelJob(jobID string) error {
	sc.mu.RLock()
	job, exists := sc.jobs[jobID]
	sc.mu.RUnlock()

	if !exists {
		return fmt.Errorf("job not found: %s", jobID)
	}

	if err := sc.client.CancelJob(job.SessionID, jobID); err != nil {
		return fmt.Errorf("failed to cancel job: %w", err)
	}

	job.State = JobStateCancelled
	sc.logger.WithField("job_id", jobID).Info("Cancelled distributed job")

	return nil
}

// GetClusterMetrics returns cluster metrics
func (sc *SparkCluster) GetClusterMetrics() *ClusterMetrics {
	sc.mu.RLock()
	defer sc.mu.RUnlock()
	return sc.metrics
}

// ListSessions returns all active sessions
func (sc *SparkCluster) ListSessions() []*SparkSession {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	sessions := make([]*SparkSession, 0, len(sc.sessions))
	for _, session := range sc.sessions {
		sessions = append(sessions, session)
	}

	return sessions
}

// ListJobs returns all jobs
func (sc *SparkCluster) ListJobs() []*SparkJob {
	sc.mu.RLock()
	defer sc.mu.RUnlock()

	jobs := make([]*SparkJob, 0, len(sc.jobs))
	for _, job := range sc.jobs {
		jobs = append(jobs, job)
	}

	return jobs
}

// Helper methods

func (sc *SparkCluster) validateJob(job *SparkJob) error {
	if job.ID == "" {
		return fmt.Errorf("job ID is required")
	}

	if job.SessionID == "" {
		return fmt.Errorf("session ID is required")
	}

	if job.JobType == "" {
		return fmt.Errorf("job type is required")
	}

	return nil
}

func (sc *SparkCluster) metricsCollectionLoop(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-sc.stopCh:
			return
		case <-ticker.C:
			sc.collectMetrics()
		}
	}
}

func (sc *SparkCluster) collectMetrics() {
	// Get cluster info
	clusterInfo, err := sc.client.GetClusterInfo()
	if err != nil {
		sc.logger.WithError(err).Error("Failed to get cluster info")
		return
	}

	// Update cluster metrics
	sc.mu.Lock()
	defer sc.mu.Unlock()

	sc.metrics.TotalExecutors = len(clusterInfo.Workers)
	sc.metrics.RunningApplications = len(clusterInfo.Applications)
	sc.metrics.LastUpdated = time.Now()

	// Calculate totals from workers
	totalCores := 0
	usedCores := 0
	totalMemory := int64(0)
	usedMemory := int64(0)

	for _, worker := range clusterInfo.Workers {
		totalCores += worker.Cores
		usedCores += worker.CoresUsed
		totalMemory += worker.Memory
		usedMemory += worker.MemoryUsed
	}

	sc.metrics.TotalCores = totalCores
	sc.metrics.UsedCores = usedCores
	sc.metrics.TotalMemory = totalMemory
	sc.metrics.UsedMemory = usedMemory
}

func (sc *SparkCluster) jobMonitoringLoop(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-sc.stopCh:
			return
		case <-ticker.C:
			sc.monitorJobs()
		}
	}
}

func (sc *SparkCluster) monitorJobs() {
	sc.mu.RLock()
	activeJobs := make([]*SparkJob, 0)
	for _, job := range sc.jobs {
		if job.State == JobStateRunning {
			activeJobs = append(activeJobs, job)
		}
	}
	sc.mu.RUnlock()

	for _, job := range activeJobs {
		// Update job status
		latestJob, err := sc.client.GetJobStatus(job.SessionID, job.ID)
		if err != nil {
			sc.logger.WithError(err).WithField("job_id", job.ID).Error("Failed to get job status")
			continue
		}

		sc.mu.Lock()
		sc.jobs[job.ID] = latestJob
		sc.mu.Unlock()

		// Update session if job completed
		if latestJob.State == JobStateCompleted || latestJob.State == JobStateFailed {
			sc.updateSessionAfterJobCompletion(latestJob)
		}
	}
}

func (sc *SparkCluster) updateSessionAfterJobCompletion(job *SparkJob) {
	sc.mu.Lock()
	defer sc.mu.Unlock()

	session, exists := sc.sessions[job.SessionID]
	if !exists {
		return
	}

	// Remove from active jobs
	for i, activeJobID := range session.ActiveJobs {
		if activeJobID == job.ID {
			session.ActiveJobs = append(session.ActiveJobs[:i], session.ActiveJobs[i+1:]...)
			break
		}
	}

	// Add to completed or failed jobs
	if job.State == JobStateCompleted {
		session.CompletedJobs = append(session.CompletedJobs, job.ID)
	} else if job.State == JobStateFailed {
		session.FailedJobs = append(session.FailedJobs, job.ID)
	}

	// Update session state
	if len(session.ActiveJobs) == 0 {
		session.State = SessionStateIdle
	}

	session.LastActivity = time.Now()
}

func getDefaultSparkConfig() *SparkConfig {
	return &SparkConfig{
		Enabled:           true,
		MasterURL:         "local[*]",
		AppName:           "tsiot-spark",
		ExecutorInstances: 2,
		ExecutorCores:     2,
		ExecutorMemory:    "2g",
		DriverMemory:      "1g",
		MaxCores:          4,
		SparkConf: map[string]string{
			"spark.sql.adaptive.enabled":               "true",
			"spark.sql.adaptive.coalescePartitions.enabled": "true",
			"spark.serializer":                         "org.apache.spark.serializer.KryoSerializer",
		},
		DeployMode:              "client",
		EnableDynamicAllocation: true,
		MinExecutors:            1,
		MaxExecutors:            10,
		InitialExecutors:        2,
		LogLevel:                "WARN",
		NetworkTimeout:          120 * time.Second,
		HeartbeatInterval:       10 * time.Second,
	}
}