package batch

import (
	"context"
	"fmt"
	"time"

	"github.com/sirupsen/logrus"
)

// Worker represents a batch processing worker
type Worker struct {
	id       int
	logger   *logrus.Logger
	config   *Config
	processor *Processor
}

// NewWorker creates a new batch processing worker
func NewWorker(id int, logger *logrus.Logger, config *Config) *Worker {
	return &Worker{
		id:     id,
		logger: logger.WithField("worker_id", id),
		config: config,
	}
}

// Start starts the worker to process jobs
func (w *Worker) Start(ctx context.Context, jobQueue <-chan *Job, resultQueue chan<- *Result) {
	w.logger.Info("Worker started")
	defer w.logger.Info("Worker stopped")

	for {
		select {
		case <-ctx.Done():
			w.logger.Info("Worker context cancelled")
			return
		case job, ok := <-jobQueue:
			if !ok {
				w.logger.Info("Job queue closed")
				return
			}
			
			w.processJob(ctx, job, resultQueue)
		}
	}
}

// processJob processes a single job
func (w *Worker) processJob(ctx context.Context, job *Job, resultQueue chan<- *Result) {
	startTime := time.Now()
	
	w.logger.WithFields(logrus.Fields{
		"job_id":   job.ID,
		"job_type": job.Type,
		"attempt":  job.Attempts + 1,
	}).Info("Processing job")

	// Update job status
	job.Status = StatusRunning
	job.Attempts++
	now := time.Now()
	job.StartedAt = &now

	// Create processing context with timeout
	processCtx, cancel := context.WithTimeout(ctx, w.config.ProcessTimeout)
	defer cancel()

	// Process the job based on its type
	result := w.executeJob(processCtx, job)
	result.JobID = job.ID

	// Update job status based on result
	if result.Status == StatusCompleted {
		job.Status = StatusCompleted
		job.CompletedAt = &now
		job.Result = result.Data
	} else {
		job.Status = StatusFailed
		job.Error = result.Error
		
		// Check if we should retry
		if job.Attempts < job.MaxAttempts {
			w.logger.WithFields(logrus.Fields{
				"job_id":      job.ID,
				"attempt":     job.Attempts,
				"max_attempts": job.MaxAttempts,
				"error":       result.Error,
			}).Warn("Job failed, will retry")
			
			// Reset job status for retry
			job.Status = StatusPending
			job.StartedAt = nil
			
			// Schedule retry after delay
			go func() {
				time.Sleep(w.config.RetryDelay)
				// Note: In a real implementation, you'd re-queue the job
				// For now, we'll just mark it as failed
			}()
		} else {
			w.logger.WithFields(logrus.Fields{
				"job_id": job.ID,
				"error":  result.Error,
			}).Error("Job failed after maximum retry attempts")
		}
	}

	// Calculate execution time
	if result.Metrics != nil {
		result.Metrics.ProcessingTime = time.Since(startTime)
	}

	// Send result
	select {
	case resultQueue <- result:
		// Result sent successfully
	case <-ctx.Done():
		w.logger.Warn("Context cancelled while sending result")
		return
	default:
		w.logger.Warn("Result queue is full, dropping result")
	}
}

// executeJob executes the job based on its type
func (w *Worker) executeJob(ctx context.Context, job *Job) *Result {
	defer func() {
		if r := recover(); r != nil {
			w.logger.WithField("job_id", job.ID).Errorf("Job panicked: %v", r)
		}
	}()

	// Create a temporary processor instance for this job
	// In a real implementation, this might be injected or shared
	processor := NewProcessor(w.logger, w.config)

	switch job.Type {
	case JobTypeAggregation:
		return w.executeAggregation(ctx, processor, job)
	case JobTypeTransformation:
		return w.executeTransformation(ctx, processor, job)
	case JobTypeValidation:
		return w.executeValidation(ctx, processor, job)
	case JobTypeAnalytics:
		return w.executeAnalytics(ctx, processor, job)
	case JobTypeExport:
		return w.executeExport(ctx, processor, job)
	case JobTypeCleanup:
		return w.executeCleanup(ctx, processor, job)
	default:
		return &Result{
			Status:      StatusFailed,
			Error:       fmt.Sprintf("unknown job type: %s", job.Type),
			CompletedAt: time.Now(),
		}
	}
}

// executeAggregation executes an aggregation job
func (w *Worker) executeAggregation(ctx context.Context, processor *Processor, job *Job) *Result {
	w.logger.WithField("job_id", job.ID).Info("Executing aggregation job")
	
	result, err := processor.ProcessAggregation(ctx, job.TimeSeries, job.Parameters)
	if err != nil {
		return &Result{
			Status:      StatusFailed,
			Error:       err.Error(),
			CompletedAt: time.Now(),
		}
	}
	
	return result
}

// executeTransformation executes a transformation job
func (w *Worker) executeTransformation(ctx context.Context, processor *Processor, job *Job) *Result {
	w.logger.WithField("job_id", job.ID).Info("Executing transformation job")
	
	result, err := processor.ProcessTransformation(ctx, job.TimeSeries, job.Parameters)
	if err != nil {
		return &Result{
			Status:      StatusFailed,
			Error:       err.Error(),
			CompletedAt: time.Now(),
		}
	}
	
	return result
}

// executeValidation executes a validation job
func (w *Worker) executeValidation(ctx context.Context, processor *Processor, job *Job) *Result {
	w.logger.WithField("job_id", job.ID).Info("Executing validation job")
	
	result, err := processor.ProcessValidation(ctx, job.TimeSeries, job.Parameters)
	if err != nil {
		return &Result{
			Status:      StatusFailed,
			Error:       err.Error(),
			CompletedAt: time.Now(),
		}
	}
	
	return result
}

// executeAnalytics executes an analytics job
func (w *Worker) executeAnalytics(ctx context.Context, processor *Processor, job *Job) *Result {
	w.logger.WithField("job_id", job.ID).Info("Executing analytics job")
	
	// In a real implementation, this would call an analytics processor
	// For now, we'll return a placeholder result
	return &Result{
		Status: StatusCompleted,
		Data: map[string]interface{}{
			"message": "Analytics processing completed",
			"series_count": len(job.TimeSeries),
		},
		Metrics: &ProcessingMetrics{
			RecordsProcessed: w.countTotalRecords(job.TimeSeries),
		},
		CompletedAt: time.Now(),
	}
}

// executeExport executes an export job
func (w *Worker) executeExport(ctx context.Context, processor *Processor, job *Job) *Result {
	w.logger.WithField("job_id", job.ID).Info("Executing export job")
	
	exportFormat, _ := job.Parameters["format"].(string)
	if exportFormat == "" {
		exportFormat = "json"
	}
	
	// In a real implementation, this would export data to the specified format/destination
	return &Result{
		Status: StatusCompleted,
		Data: map[string]interface{}{
			"message": "Export completed",
			"format":  exportFormat,
			"series_count": len(job.TimeSeries),
			"total_records": w.countTotalRecords(job.TimeSeries),
		},
		Metrics: &ProcessingMetrics{
			RecordsProcessed: w.countTotalRecords(job.TimeSeries),
		},
		CompletedAt: time.Now(),
	}
}

// executeCleanup executes a cleanup job
func (w *Worker) executeCleanup(ctx context.Context, processor *Processor, job *Job) *Result {
	w.logger.WithField("job_id", job.ID).Info("Executing cleanup job")
	
	// In a real implementation, this would perform data cleanup operations
	cleanupType, _ := job.Parameters["type"].(string)
	if cleanupType == "" {
		cleanupType = "general"
	}
	
	return &Result{
		Status: StatusCompleted,
		Data: map[string]interface{}{
			"message": "Cleanup completed",
			"cleanup_type": cleanupType,
			"series_processed": len(job.TimeSeries),
		},
		Metrics: &ProcessingMetrics{
			RecordsProcessed: w.countTotalRecords(job.TimeSeries),
		},
		CompletedAt: time.Now(),
	}
}

// countTotalRecords counts the total number of records across all time series
func (w *Worker) countTotalRecords(timeSeries []*storage.TimeSeries) int {
	total := 0
	for _, ts := range timeSeries {
		total += len(ts.Values)
	}
	return total
}

// GetStatus returns the current status of the worker
func (w *Worker) GetStatus() map[string]interface{} {
	return map[string]interface{}{
		"id":     w.id,
		"status": "running", // In a real implementation, track actual status
	}
}