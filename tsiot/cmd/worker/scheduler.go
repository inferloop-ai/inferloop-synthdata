package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

type JobType string

const (
	JobTypeGenerate JobType = "generate"
	JobTypeValidate JobType = "validate"
	JobTypeAnalyze  JobType = "analyze"
	JobTypeMigrate  JobType = "migrate"
)

type Job struct {
	ID         string                 `json:"id"`
	Type       JobType                `json:"type"`
	Status     string                 `json:"status"`
	Parameters map[string]interface{} `json:"parameters"`
	CreatedAt  time.Time              `json:"created_at"`
	UpdatedAt  time.Time              `json:"updated_at"`
	Result     interface{}            `json:"result,omitempty"`
	Error      string                 `json:"error,omitempty"`
}

type Scheduler struct {
	config   *WorkerConfig
	logger   *logrus.Logger
	jobQueue chan *Job
	client   *http.Client
	mu       sync.RWMutex
	running  bool
}

func NewScheduler(config *WorkerConfig, logger *logrus.Logger) *Scheduler {
	return &Scheduler{
		config:   config,
		logger:   logger,
		jobQueue: make(chan *Job, config.Concurrency*2),
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

func (s *Scheduler) Start(ctx context.Context) {
	s.mu.Lock()
	s.running = true
	s.mu.Unlock()

	s.logger.Info("Scheduler started")
	
	ticker := time.NewTicker(s.config.PollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			s.logger.Info("Scheduler stopping due to context cancellation")
			return
		case <-ticker.C:
			s.mu.RLock()
			running := s.running
			s.mu.RUnlock()
			
			if !running {
				s.logger.Info("Scheduler stopped")
				return
			}

			s.pollJobs(ctx)
		}
	}
}

func (s *Scheduler) Stop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	
	s.running = false
	close(s.jobQueue)
	s.logger.Info("Scheduler stop requested")
}

func (s *Scheduler) GetJobQueue() <-chan *Job {
	return s.jobQueue
}

func (s *Scheduler) pollJobs(ctx context.Context) {
	// Check if queue is full
	if len(s.jobQueue) >= cap(s.jobQueue)-1 {
		s.logger.Debug("Job queue is full, skipping poll")
		return
	}

	jobs, err := s.fetchPendingJobs(ctx)
	if err != nil {
		s.logger.WithError(err).Error("Failed to fetch pending jobs")
		return
	}

	for _, job := range jobs {
		select {
		case s.jobQueue <- job:
			s.logger.WithFields(logrus.Fields{
				"jobID": job.ID,
				"type":  job.Type,
			}).Debug("Job queued")
		default:
			s.logger.Warn("Job queue is full")
			return
		}
	}

	if len(jobs) > 0 {
		s.logger.WithField("count", len(jobs)).Info("Jobs fetched and queued")
	}
}

func (s *Scheduler) fetchPendingJobs(ctx context.Context) ([]*Job, error) {
	url := fmt.Sprintf("%s/api/v1/worker/jobs?worker_id=%s&limit=%d", 
		s.config.ServerURL, s.config.WorkerID, s.config.Concurrency)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	var jobs []*Job
	if err := json.NewDecoder(resp.Body).Decode(&jobs); err != nil {
		return nil, err
	}

	// For now, return mock jobs for testing
	if len(jobs) == 0 {
		// Simulate some pending jobs
		now := time.Now()
		jobs = []*Job{
			{
				ID:        fmt.Sprintf("job-%d", now.Unix()),
				Type:      JobTypeGenerate,
				Status:    "pending",
				CreatedAt: now,
				UpdatedAt: now,
				Parameters: map[string]interface{}{
					"generator":   "timegan",
					"sensor_type": "temperature",
					"duration":    "24h",
					"frequency":   "1m",
				},
			},
		}
	}

	return jobs, nil
}

func (s *Scheduler) UpdateJobStatus(ctx context.Context, jobID, status string, result interface{}, errorMsg string) error {
	url := fmt.Sprintf("%s/api/v1/worker/jobs/%s/status", s.config.ServerURL, jobID)

	payload := map[string]interface{}{
		"worker_id": s.config.WorkerID,
		"status":    status,
		"updated_at": time.Now(),
	}

	if result != nil {
		payload["result"] = result
	}

	if errorMsg != "" {
		payload["error"] = errorMsg
	}

	_, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, "PUT", url, nil)
	if err != nil {
		return err
	}

	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	s.logger.WithFields(logrus.Fields{
		"jobID":  jobID,
		"status": status,
	}).Debug("Job status updated")

	return nil
}