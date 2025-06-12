package distributed

import (
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// MockSparkClient implements SparkClient interface for testing and development
type MockSparkClient struct {
	logger       *logrus.Logger
	config       *SparkConfig
	sessions     map[string]*SparkSession
	jobs         map[string]*SparkJob
	mu           sync.RWMutex
	sessionCounter int
	jobCounter     int
}

// NewMockSparkClient creates a new mock Spark client
func NewMockSparkClient(config *SparkConfig, logger *logrus.Logger) (*MockSparkClient, error) {
	return &MockSparkClient{
		logger:   logger,
		config:   config,
		sessions: make(map[string]*SparkSession),
		jobs:     make(map[string]*SparkJob),
	}, nil
}

// CreateSession creates a new mock Spark session
func (msc *MockSparkClient) CreateSession(config *SparkConfig) (*SparkSession, error) {
	msc.mu.Lock()
	defer msc.mu.Unlock()

	msc.sessionCounter++
	sessionID := fmt.Sprintf("session_%d", msc.sessionCounter)
	appID := fmt.Sprintf("app_%d_%d", time.Now().Unix(), msc.sessionCounter)

	session := &SparkSession{
		ID:              sessionID,
		AppID:           appID,
		State:           SessionStateStarting,
		CreatedAt:       time.Now(),
		LastActivity:    time.Now(),
		Configuration:   config.SparkConf,
		ExecutorSummary: msc.generateMockExecutors(config.ExecutorInstances),
		ActiveJobs:      make([]string, 0),
		CompletedJobs:   make([]string, 0),
		FailedJobs:      make([]string, 0),
		Metrics:         &SessionMetrics{},
	}

	// Simulate session startup time
	go func() {
		time.Sleep(time.Duration(rand.Intn(3)+1) * time.Second)
		msc.mu.Lock()
		session.State = SessionStateIdle
		msc.mu.Unlock()
	}()

	msc.sessions[sessionID] = session

	msc.logger.WithFields(logrus.Fields{
		"session_id": sessionID,
		"app_id":     appID,
	}).Info("Created mock Spark session")

	return session, nil
}

// SubmitJob submits a job to a mock Spark session
func (msc *MockSparkClient) SubmitJob(sessionID string, job *SparkJob) error {
	msc.mu.Lock()
	defer msc.mu.Unlock()

	session, exists := msc.sessions[sessionID]
	if !exists {
		return fmt.Errorf("session not found: %s", sessionID)
	}

	if session.State != SessionStateIdle && session.State != SessionStateBusy {
		return fmt.Errorf("session not ready: %s", session.State)
	}

	// Store job
	job.SessionID = sessionID
	job.State = JobStateRunning
	startTime := time.Now()
	job.StartedAt = &startTime
	job.Progress = &JobProgress{
		TotalStages:     rand.Intn(10) + 1,
		TotalTasks:      rand.Intn(100) + 10,
		PercentComplete: 0,
	}

	msc.jobs[job.ID] = job

	// Update session
	session.ActiveJobs = append(session.ActiveJobs, job.ID)
	session.State = SessionStateBusy
	session.LastActivity = time.Now()

	// Simulate job execution
	go msc.simulateJobExecution(job)

	msc.logger.WithFields(logrus.Fields{
		"job_id":     job.ID,
		"session_id": sessionID,
		"job_type":   job.JobType,
	}).Info("Submitted job to mock Spark session")

	return nil
}

// GetJobStatus returns the status of a mock job
func (msc *MockSparkClient) GetJobStatus(sessionID, jobID string) (*SparkJob, error) {
	msc.mu.RLock()
	defer msc.mu.RUnlock()

	job, exists := msc.jobs[jobID]
	if !exists {
		return nil, fmt.Errorf("job not found: %s", jobID)
	}

	return job, nil
}

// CancelJob cancels a mock job
func (msc *MockSparkClient) CancelJob(sessionID, jobID string) error {
	msc.mu.Lock()
	defer msc.mu.Unlock()

	job, exists := msc.jobs[jobID]
	if !exists {
		return fmt.Errorf("job not found: %s", jobID)
	}

	job.State = JobStateCancelled
	completedAt := time.Now()
	job.CompletedAt = &completedAt
	job.Duration = completedAt.Sub(*job.StartedAt)

	msc.updateSessionAfterJobCompletion(sessionID, jobID, JobStateCancelled)

	msc.logger.WithField("job_id", jobID).Info("Cancelled mock job")

	return nil
}

// GetSessionMetrics returns mock session metrics
func (msc *MockSparkClient) GetSessionMetrics(sessionID string) (*SessionMetrics, error) {
	msc.mu.RLock()
	defer msc.mu.RUnlock()

	session, exists := msc.sessions[sessionID]
	if !exists {
		return nil, fmt.Errorf("session not found: %s", sessionID)
	}

	// Generate mock metrics
	metrics := &SessionMetrics{
		ExecutorCount:        len(session.ExecutorSummary),
		TotalCores:          msc.config.ExecutorInstances * msc.config.ExecutorCores,
		TotalMemory:         int64(msc.config.ExecutorInstances) * 2 * 1024 * 1024 * 1024, // 2GB per executor
		UsedMemory:          int64(rand.Intn(1024)) * 1024 * 1024, // Random used memory
		ActiveJobs:          len(session.ActiveJobs),
		CompletedJobs:       len(session.CompletedJobs),
		FailedJobs:          len(session.FailedJobs),
		TotalTasks:          rand.Intn(1000) + 100,
		CompletedTasks:      rand.Intn(900) + 50,
		FailedTasks:         rand.Intn(10),
		TotalInputBytes:     int64(rand.Intn(1000)) * 1024 * 1024,
		TotalOutputBytes:    int64(rand.Intn(800)) * 1024 * 1024,
		TotalShuffleRead:    int64(rand.Intn(500)) * 1024 * 1024,
		TotalShuffleWrite:   int64(rand.Intn(300)) * 1024 * 1024,
		GCTime:              int64(rand.Intn(10000)),
		LastUpdated:         time.Now(),
	}

	return metrics, nil
}

// CloseSession closes a mock Spark session
func (msc *MockSparkClient) CloseSession(sessionID string) error {
	msc.mu.Lock()
	defer msc.mu.Unlock()

	session, exists := msc.sessions[sessionID]
	if !exists {
		return fmt.Errorf("session not found: %s", sessionID)
	}

	session.State = SessionStateShutdown

	// Cancel any running jobs
	for _, jobID := range session.ActiveJobs {
		if job, exists := msc.jobs[jobID]; exists {
			job.State = JobStateCancelled
			completedAt := time.Now()
			job.CompletedAt = &completedAt
			if job.StartedAt != nil {
				job.Duration = completedAt.Sub(*job.StartedAt)
			}
		}
	}

	delete(msc.sessions, sessionID)

	msc.logger.WithField("session_id", sessionID).Info("Closed mock Spark session")

	return nil
}

// ListSessions returns all mock sessions
func (msc *MockSparkClient) ListSessions() ([]*SparkSession, error) {
	msc.mu.RLock()
	defer msc.mu.RUnlock()

	sessions := make([]*SparkSession, 0, len(msc.sessions))
	for _, session := range msc.sessions {
		sessions = append(sessions, session)
	}

	return sessions, nil
}

// GetClusterInfo returns mock cluster information
func (msc *MockSparkClient) GetClusterInfo() (*ClusterInfo, error) {
	workers := make([]WorkerInfo, msc.config.ExecutorInstances)
	for i := 0; i < msc.config.ExecutorInstances; i++ {
		workers[i] = WorkerInfo{
			ID:            fmt.Sprintf("worker_%d", i+1),
			Host:          fmt.Sprintf("worker%d.cluster.local", i+1),
			Port:          7077,
			WebUIPort:     8081 + i,
			Cores:         msc.config.ExecutorCores,
			CoresUsed:     rand.Intn(msc.config.ExecutorCores),
			Memory:        2 * 1024 * 1024 * 1024, // 2GB
			MemoryUsed:    int64(rand.Intn(1024)) * 1024 * 1024,
			State:         "ALIVE",
			LastHeartbeat: time.Now().Add(-time.Duration(rand.Intn(30)) * time.Second),
		}
	}

	applications := make([]ApplicationInfo, 0)
	for _, session := range msc.sessions {
		app := ApplicationInfo{
			ID:              session.AppID,
			Name:            session.Name,
			User:            "spark",
			State:           string(session.State),
			FinalStatus:     "UNDEFINED",
			StartTime:       session.CreatedAt,
			SparkUser:       "spark",
			CoresGranted:    msc.config.ExecutorCores * msc.config.ExecutorInstances,
			MaxCores:        msc.config.MaxCores,
			MemoryPerNode:   2 * 1024 * 1024 * 1024,
			ExecutorCount:   msc.config.ExecutorInstances,
		}

		if session.State == SessionStateShutdown {
			endTime := time.Now()
			app.EndTime = &endTime
			app.Duration = endTime.Sub(session.CreatedAt)
			app.FinalStatus = "SUCCEEDED"
		}

		applications = append(applications, app)
	}

	clusterInfo := &ClusterInfo{
		MasterURL:     msc.config.MasterURL,
		SparkVersion:  "3.5.0",
		ScalaVersion:  "2.12",
		JavaVersion:   "11.0.20",
		Workers:       workers,
		Applications:  applications,
		Status:        "ALIVE",
	}

	return clusterInfo, nil
}

// Helper methods

func (msc *MockSparkClient) generateMockExecutors(count int) []ExecutorSummary {
	executors := make([]ExecutorSummary, count)
	
	for i := 0; i < count; i++ {
		executors[i] = ExecutorSummary{
			ID:                fmt.Sprintf("executor_%d", i+1),
			HostPort:          fmt.Sprintf("executor%d.cluster.local:7337", i+1),
			IsActive:          true,
			RDDBlocks:         rand.Intn(100),
			MemoryUsed:        int64(rand.Intn(1024)) * 1024 * 1024,
			DiskUsed:          int64(rand.Intn(512)) * 1024 * 1024,
			TotalCores:        msc.config.ExecutorCores,
			MaxTasks:          msc.config.ExecutorCores,
			ActiveTasks:       rand.Intn(msc.config.ExecutorCores),
			FailedTasks:       rand.Intn(5),
			CompletedTasks:    rand.Intn(100) + 10,
			TotalTasks:        rand.Intn(120) + 15,
			TaskTime:          int64(rand.Intn(10000)) + 1000,
			GCTime:            int64(rand.Intn(1000)),
			TotalInputBytes:   int64(rand.Intn(1000)) * 1024 * 1024,
			TotalShuffleRead:  int64(rand.Intn(500)) * 1024 * 1024,
			TotalShuffleWrite: int64(rand.Intn(300)) * 1024 * 1024,
			LastHeartbeat:     time.Now().Add(-time.Duration(rand.Intn(30)) * time.Second),
		}
	}

	return executors
}

func (msc *MockSparkClient) simulateJobExecution(job *SparkJob) {
	// Simulate job execution time based on job type
	var executionTime time.Duration
	switch job.JobType {
	case JobTypeGeneration:
		executionTime = time.Duration(rand.Intn(30)+10) * time.Second
	case JobTypeValidation:
		executionTime = time.Duration(rand.Intn(20)+5) * time.Second
	case JobTypeAnalytics:
		executionTime = time.Duration(rand.Intn(60)+15) * time.Second
	case JobTypeTraining:
		executionTime = time.Duration(rand.Intn(300)+60) * time.Second
	default:
		executionTime = time.Duration(rand.Intn(30)+10) * time.Second
	}

	// Simulate progress updates
	progressTicker := time.NewTicker(executionTime / 10)
	defer progressTicker.Stop()

	stages := job.Progress.TotalStages
	tasks := job.Progress.TotalTasks
	completedStages := 0
	completedTasks := 0

	for i := 0; i < 10; i++ {
		select {
		case <-progressTicker.C:
			msc.mu.Lock()
			
			// Update progress
			if i < 9 { // Don't complete on the last iteration
				completedStages = (i + 1) * stages / 10
				completedTasks = (i + 1) * tasks / 10
				
				job.Progress.CompletedStages = completedStages
				job.Progress.CompletedTasks = completedTasks
				job.Progress.PercentComplete = float64(i+1) * 10
				job.Progress.EstimatedTimeLeft = executionTime * time.Duration(9-i) / 10
			}
			
			msc.mu.Unlock()
		}
	}

	// Complete the job
	msc.mu.Lock()
	defer msc.mu.Unlock()

	// Determine job outcome (90% success rate)
	success := rand.Float32() < 0.9

	if success {
		job.State = JobStateCompleted
		job.Progress.CompletedStages = stages
		job.Progress.CompletedTasks = tasks
		job.Progress.PercentComplete = 100
		job.Progress.EstimatedTimeLeft = 0
	} else {
		job.State = JobStateFailed
		job.Error = fmt.Errorf("mock job execution failed")
	}

	completedAt := time.Now()
	job.CompletedAt = &completedAt
	job.Duration = completedAt.Sub(*job.StartedAt)

	// Generate mock metrics
	job.Metrics = &JobMetrics{
		TaskCount:           tasks,
		SuccessfulTasks:     completedTasks,
		FailedTasks:         tasks - completedTasks,
		TotalTaskTime:       int64(job.Duration.Milliseconds()),
		InputBytes:          int64(rand.Intn(1000)) * 1024 * 1024,
		OutputBytes:         int64(rand.Intn(500)) * 1024 * 1024,
		ShuffleReadBytes:    int64(rand.Intn(200)) * 1024 * 1024,
		ShuffleWriteBytes:   int64(rand.Intn(100)) * 1024 * 1024,
		MemoryBytesSpilled:  int64(rand.Intn(50)) * 1024 * 1024,
		DiskBytesSpilled:    int64(rand.Intn(25)) * 1024 * 1024,
		PeakExecutionMemory: int64(rand.Intn(2048)) * 1024 * 1024,
		GCTime:              int64(rand.Intn(5000)),
		ResultSize:          int64(rand.Intn(10)) * 1024 * 1024,
	}

	// Update session
	msc.updateSessionAfterJobCompletion(job.SessionID, job.ID, job.State)

	msc.logger.WithFields(logrus.Fields{
		"job_id":    job.ID,
		"state":     job.State,
		"duration":  job.Duration,
	}).Info("Mock job execution completed")
}

func (msc *MockSparkClient) updateSessionAfterJobCompletion(sessionID, jobID string, finalState JobState) {
	session, exists := msc.sessions[sessionID]
	if !exists {
		return
	}

	// Remove from active jobs
	for i, activeJobID := range session.ActiveJobs {
		if activeJobID == jobID {
			session.ActiveJobs = append(session.ActiveJobs[:i], session.ActiveJobs[i+1:]...)
			break
		}
	}

	// Add to completed or failed jobs
	if finalState == JobStateCompleted {
		session.CompletedJobs = append(session.CompletedJobs, jobID)
	} else {
		session.FailedJobs = append(session.FailedJobs, jobID)
	}

	// Update session state
	if len(session.ActiveJobs) == 0 {
		session.State = SessionStateIdle
	}

	session.LastActivity = time.Now()
}