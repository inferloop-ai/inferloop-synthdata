package agents

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/internal/agents/base"
	"github.com/inferloop/tsiot/internal/validation/metrics"
	"github.com/inferloop/tsiot/internal/validation/reports"
	"github.com/inferloop/tsiot/internal/validation/rules"
	"github.com/inferloop/tsiot/internal/validation/tests"
	"github.com/inferloop/tsiot/pkg/errors"
	"github.com/inferloop/tsiot/pkg/models"
)

// ValidationAgent orchestrates automated validation workflows for time series data
type ValidationAgent struct {
	*base.BaseAgent
	mu                 sync.RWMutex
	validationSuite    *tests.StatisticalTestSuite
	qualityMetrics     *metrics.QualityMetricsCalculator
	ruleEngine         *rules.RuleEngine
	reportGenerator    *reports.ValidationReportGenerator
	activeValidations  map[string]*ValidationJob
	validationHistory  []ValidationResult
	config             *ValidationConfig
	workers            chan struct{}
	jobQueue          chan *ValidationJob
	stopCh            chan struct{}
	wg                sync.WaitGroup
}

// ValidationConfig contains configuration for the validation agent
type ValidationConfig struct {
	// Concurrency settings
	MaxConcurrentJobs     int           `json:"max_concurrent_jobs"`
	JobTimeout           time.Duration  `json:"job_timeout"`
	WorkerPoolSize       int           `json:"worker_pool_size"`
	
	// Validation settings
	EnableStatisticalTests bool         `json:"enable_statistical_tests"`
	EnableQualityMetrics   bool         `json:"enable_quality_metrics"`
	EnableCustomRules      bool         `json:"enable_custom_rules"`
	EnableReporting        bool         `json:"enable_reporting"`
	
	// Test configuration
	SignificanceLevel     float64       `json:"significance_level"`
	MinDataPoints         int           `json:"min_data_points"`
	MaxDataPoints         int           `json:"max_data_points"`
	
	// Quality thresholds
	MinQualityScore       float64       `json:"min_quality_score"`
	MaxMissingRate        float64       `json:"max_missing_rate"`
	MaxOutlierRate        float64       `json:"max_outlier_rate"`
	
	// Scheduling
	AutoValidationInterval time.Duration `json:"auto_validation_interval"`
	RetentionPeriod       time.Duration  `json:"retention_period"`
	
	// Notification settings
	NotifyOnFailure       bool          `json:"notify_on_failure"`
	NotifyOnCompletion    bool          `json:"notify_on_completion"`
	AlertThresholds       map[string]float64 `json:"alert_thresholds"`
}

// ValidationJob represents a validation task
type ValidationJob struct {
	ID                string                 `json:"id"`
	Type              ValidationType         `json:"type"`
	Status            ValidationStatus       `json:"status"`
	Priority          int                    `json:"priority"`
	CreatedAt         time.Time              `json:"created_at"`
	StartedAt         *time.Time             `json:"started_at,omitempty"`
	CompletedAt       *time.Time             `json:"completed_at,omitempty"`
	
	// Input data
	OriginalData      *models.TimeSeries     `json:"original_data,omitempty"`
	SyntheticData     *models.TimeSeries     `json:"synthetic_data,omitempty"`
	
	// Configuration
	TestConfig        *ValidationTestConfig  `json:"test_config"`
	
	// Results
	Result            *ValidationResult      `json:"result,omitempty"`
	Error             error                  `json:"error,omitempty"`
	
	// Context
	Context           context.Context        `json:"-"`
	CancelFunc        context.CancelFunc     `json:"-"`
	
	// Metadata
	RequestedBy       string                 `json:"requested_by"`
	Tags              map[string]string      `json:"tags"`
	Metadata          map[string]interface{} `json:"metadata"`
}

// ValidationType defines types of validation
type ValidationType string

const (
	ValidationTypeStatistical ValidationType = "statistical"
	ValidationTypeQuality     ValidationType = "quality"
	ValidationTypeComparison  ValidationType = "comparison"
	ValidationTypeCustom      ValidationType = "custom"
	ValidationTypeComplete    ValidationType = "complete"
)

// ValidationStatus represents the status of a validation job
type ValidationStatus string

const (
	ValidationStatusQueued     ValidationStatus = "queued"
	ValidationStatusRunning    ValidationStatus = "running"
	ValidationStatusCompleted  ValidationStatus = "completed"
	ValidationStatusFailed     ValidationStatus = "failed"
	ValidationStatusCancelled  ValidationStatus = "cancelled"
	ValidationStatusTimeout    ValidationStatus = "timeout"
)

// ValidationTestConfig contains configuration for validation tests
type ValidationTestConfig struct {
	// Statistical tests
	RunKolmogorovSmirnov bool    `json:"run_kolmogorov_smirnov"`
	RunAndersonDarling   bool    `json:"run_anderson_darling"`
	RunShapiroWilk       bool    `json:"run_shapiro_wilk"`
	RunJarqueBera        bool    `json:"run_jarque_bera"`
	RunLjungBox          bool    `json:"run_ljung_box"`
	RunAugmentedDickey   bool    `json:"run_augmented_dickey"`
	RunKPSS              bool    `json:"run_kpss"`
	
	// Quality metrics
	CalculateBasicStats  bool    `json:"calculate_basic_stats"`
	CalculateDistribution bool   `json:"calculate_distribution"`
	CalculateCorrelation bool    `json:"calculate_correlation"`
	CalculateTrends      bool    `json:"calculate_trends"`
	
	// Thresholds
	SignificanceLevel    float64 `json:"significance_level"`
	MinQualityScore      float64 `json:"min_quality_score"`
	
	// Custom rules
	CustomRules          []string `json:"custom_rules"`
}

// ValidationResult contains the result of a validation
type ValidationResult struct {
	JobID                string                      `json:"job_id"`
	Type                 ValidationType              `json:"type"`
	Status               ValidationStatus            `json:"status"`
	OverallScore         float64                     `json:"overall_score"`
	Passed               bool                        `json:"passed"`
	
	// Test results
	StatisticalTests     map[string]*tests.TestResult `json:"statistical_tests,omitempty"`
	QualityMetrics       *metrics.QualityReport       `json:"quality_metrics,omitempty"`
	ComparisonResults    *reports.ComparisonReport    `json:"comparison_results,omitempty"`
	CustomRuleResults    map[string]bool              `json:"custom_rule_results,omitempty"`
	
	// Summary
	TestsPassed          int                         `json:"tests_passed"`
	TestsFailed          int                         `json:"tests_failed"`
	Warnings             []string                    `json:"warnings"`
	Errors               []string                    `json:"errors"`
	
	// Timing
	StartTime            time.Time                   `json:"start_time"`
	EndTime              time.Time                   `json:"end_time"`
	Duration             time.Duration               `json:"duration"`
	
	// Metadata
	DataPoints           int                         `json:"data_points"`
	DataQuality          float64                     `json:"data_quality"`
	Metadata             map[string]interface{}      `json:"metadata"`
}

// NewValidationAgent creates a new validation agent
func NewValidationAgent(config *ValidationConfig, logger *logrus.Logger) (*ValidationAgent, error) {
	if config == nil {
		config = getDefaultValidationConfig()
	}
	
	baseAgent, err := base.NewBaseAgent("validation-agent", logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create base agent: %w", err)
	}
	
	agent := &ValidationAgent{
		BaseAgent:         baseAgent,
		activeValidations: make(map[string]*ValidationJob),
		validationHistory: make([]ValidationResult, 0),
		config:           config,
		workers:          make(chan struct{}, config.WorkerPoolSize),
		jobQueue:         make(chan *ValidationJob, config.MaxConcurrentJobs*2),
		stopCh:           make(chan struct{}),
	}
	
	// Initialize validation components
	agent.validationSuite = tests.NewStatisticalTestSuite()
	agent.qualityMetrics = metrics.NewQualityMetricsCalculator()
	agent.ruleEngine = rules.NewRuleEngine()
	agent.reportGenerator = reports.NewValidationReportGenerator()
	
	// Start worker pool
	agent.startWorkers()
	
	return agent, nil
}

// SubmitValidationJob submits a new validation job
func (va *ValidationAgent) SubmitValidationJob(
	ctx context.Context,
	originalData, syntheticData *models.TimeSeries,
	validationType ValidationType,
	testConfig *ValidationTestConfig,
	metadata map[string]interface{},
) (string, error) {
	
	va.mu.Lock()
	defer va.mu.Unlock()
	
	// Create job
	jobCtx, cancelFunc := context.WithTimeout(ctx, va.config.JobTimeout)
	job := &ValidationJob{
		ID:            va.generateJobID(),
		Type:          validationType,
		Status:        ValidationStatusQueued,
		Priority:      va.calculatePriority(validationType, originalData, syntheticData),
		CreatedAt:     time.Now(),
		OriginalData:  originalData,
		SyntheticData: syntheticData,
		TestConfig:    testConfig,
		Context:       jobCtx,
		CancelFunc:    cancelFunc,
		Tags:          make(map[string]string),
		Metadata:      metadata,
	}
	
	if testConfig == nil {
		job.TestConfig = va.getDefaultTestConfig()
	}
	
	// Validate job
	if err := va.validateJob(job); err != nil {
		cancelFunc()
		return "", err
	}
	
	// Store job
	va.activeValidations[job.ID] = job
	
	// Queue job
	select {
	case va.jobQueue <- job:
		va.Logger.WithFields(logrus.Fields{
			"job_id": job.ID,
			"type":   job.Type,
		}).Info("Validation job queued")
		return job.ID, nil
	default:
		delete(va.activeValidations, job.ID)
		cancelFunc()
		return "", errors.NewProcessingError("QUEUE_FULL", "Validation job queue is full")
	}
}

// GetValidationStatus returns the status of a validation job
func (va *ValidationAgent) GetValidationStatus(jobID string) (*ValidationJob, error) {
	va.mu.RLock()
	defer va.mu.RUnlock()
	
	job, exists := va.activeValidations[jobID]
	if !exists {
		return nil, errors.NewValidationError("JOB_NOT_FOUND", fmt.Sprintf("Validation job %s not found", jobID))
	}
	
	// Return a copy to avoid race conditions
	jobCopy := *job
	return &jobCopy, nil
}

// CancelValidationJob cancels a validation job
func (va *ValidationAgent) CancelValidationJob(jobID string) error {
	va.mu.Lock()
	defer va.mu.Unlock()
	
	job, exists := va.activeValidations[jobID]
	if !exists {
		return errors.NewValidationError("JOB_NOT_FOUND", fmt.Sprintf("Validation job %s not found", jobID))
	}
	
	if job.Status == ValidationStatusCompleted || job.Status == ValidationStatusFailed {
		return errors.NewValidationError("JOB_ALREADY_FINISHED", "Cannot cancel completed job")
	}
	
	job.CancelFunc()
	job.Status = ValidationStatusCancelled
	
	va.Logger.WithFields(logrus.Fields{
		"job_id": jobID,
	}).Info("Validation job cancelled")
	
	return nil
}

// ListActiveJobs returns all active validation jobs
func (va *ValidationAgent) ListActiveJobs() []*ValidationJob {
	va.mu.RLock()
	defer va.mu.RUnlock()
	
	jobs := make([]*ValidationJob, 0, len(va.activeValidations))
	for _, job := range va.activeValidations {
		jobCopy := *job
		jobs = append(jobs, &jobCopy)
	}
	
	return jobs
}

// GetValidationHistory returns validation history
func (va *ValidationAgent) GetValidationHistory(limit int) []ValidationResult {
	va.mu.RLock()
	defer va.mu.RUnlock()
	
	if limit <= 0 || limit > len(va.validationHistory) {
		limit = len(va.validationHistory)
	}
	
	// Return most recent results
	start := len(va.validationHistory) - limit
	history := make([]ValidationResult, limit)
	copy(history, va.validationHistory[start:])
	
	return history
}

// ProcessJob processes a validation job
func (va *ValidationAgent) ProcessJob(ctx context.Context, job *ValidationJob) error {
	va.Logger.WithFields(logrus.Fields{
		"job_id": job.ID,
		"type":   job.Type,
	}).Info("Starting validation job processing")
	
	// Update job status
	va.updateJobStatus(job, ValidationStatusRunning)
	startTime := time.Now()
	job.StartedAt = &startTime
	
	// Create result
	result := &ValidationResult{
		JobID:         job.ID,
		Type:          job.Type,
		StartTime:     startTime,
		Metadata:      make(map[string]interface{}),
	}
	
	var err error
	
	// Process based on validation type
	switch job.Type {
	case ValidationTypeStatistical:
		err = va.runStatisticalValidation(ctx, job, result)
	case ValidationTypeQuality:
		err = va.runQualityValidation(ctx, job, result)
	case ValidationTypeComparison:
		err = va.runComparisonValidation(ctx, job, result)
	case ValidationTypeCustom:
		err = va.runCustomValidation(ctx, job, result)
	case ValidationTypeComplete:
		err = va.runCompleteValidation(ctx, job, result)
	default:
		err = fmt.Errorf("unknown validation type: %s", job.Type)
	}
	
	// Finalize result
	result.EndTime = time.Now()
	result.Duration = result.EndTime.Sub(result.StartTime)
	
	if err != nil {
		result.Status = ValidationStatusFailed
		result.Errors = append(result.Errors, err.Error())
		va.updateJobStatus(job, ValidationStatusFailed)
		job.Error = err
	} else {
		result.Status = ValidationStatusCompleted
		va.updateJobStatus(job, ValidationStatusCompleted)
	}
	
	// Store result
	job.Result = result
	completedTime := time.Now()
	job.CompletedAt = &completedTime
	
	// Add to history
	va.addToHistory(*result)
	
	va.Logger.WithFields(logrus.Fields{
		"job_id":        job.ID,
		"status":        result.Status,
		"duration":      result.Duration,
		"overall_score": result.OverallScore,
		"passed":        result.Passed,
	}).Info("Validation job completed")
	
	return err
}

// Worker methods
func (va *ValidationAgent) startWorkers() {
	for i := 0; i < va.config.WorkerPoolSize; i++ {
		va.wg.Add(1)
		go va.worker()
	}
}

func (va *ValidationAgent) worker() {
	defer va.wg.Done()
	
	for {
		select {
		case job := <-va.jobQueue:
			va.workers <- struct{}{} // Acquire worker slot
			va.ProcessJob(job.Context, job)
			<-va.workers // Release worker slot
		case <-va.stopCh:
			return
		}
	}
}

// Validation methods
func (va *ValidationAgent) runStatisticalValidation(ctx context.Context, job *ValidationJob, result *ValidationResult) error {
	if job.OriginalData == nil {
		return fmt.Errorf("original data required for statistical validation")
	}
	
	// Extract values
	values := make([]float64, len(job.OriginalData.DataPoints))
	for i, dp := range job.OriginalData.DataPoints {
		values[i] = dp.Value
	}
	
	result.DataPoints = len(values)
	result.StatisticalTests = make(map[string]*tests.TestResult)
	
	// Run statistical tests
	if job.TestConfig.RunKolmogorovSmirnov {
		testResult := va.validationSuite.KolmogorovSmirnovTest(values, job.TestConfig.SignificanceLevel)
		result.StatisticalTests["kolmogorov_smirnov"] = testResult
		if testResult.Passed {
			result.TestsPassed++
		} else {
			result.TestsFailed++
		}
	}
	
	if job.TestConfig.RunAndersonDarling {
		testResult := va.validationSuite.AndersonDarlingTest(values, job.TestConfig.SignificanceLevel)
		result.StatisticalTests["anderson_darling"] = testResult
		if testResult.Passed {
			result.TestsPassed++
		} else {
			result.TestsFailed++
		}
	}
	
	if job.TestConfig.RunShapiroWilk {
		testResult := va.validationSuite.ShapiroWilkTest(values, job.TestConfig.SignificanceLevel)
		result.StatisticalTests["shapiro_wilk"] = testResult
		if testResult.Passed {
			result.TestsPassed++
		} else {
			result.TestsFailed++
		}
	}
	
	if job.TestConfig.RunLjungBox {
		testResult := va.validationSuite.LjungBoxTest(values, 10, job.TestConfig.SignificanceLevel)
		result.StatisticalTests["ljung_box"] = testResult
		if testResult.Passed {
			result.TestsPassed++
		} else {
			result.TestsFailed++
		}
	}
	
	// Calculate overall score
	if result.TestsPassed+result.TestsFailed > 0 {
		result.OverallScore = float64(result.TestsPassed) / float64(result.TestsPassed+result.TestsFailed)
	}
	
	result.Passed = result.OverallScore >= job.TestConfig.MinQualityScore
	
	return nil
}

func (va *ValidationAgent) runQualityValidation(ctx context.Context, job *ValidationJob, result *ValidationResult) error {
	if job.OriginalData == nil {
		return fmt.Errorf("original data required for quality validation")
	}
	
	// Calculate quality metrics
	qualityReport := va.qualityMetrics.EvaluateQuality(job.OriginalData)
	result.QualityMetrics = qualityReport
	result.OverallScore = qualityReport.OverallScore
	result.DataQuality = qualityReport.OverallScore
	result.Passed = qualityReport.OverallScore >= job.TestConfig.MinQualityScore
	
	// Check for warnings
	if qualityReport.MissingDataRate > va.config.MaxMissingRate {
		result.Warnings = append(result.Warnings, 
			fmt.Sprintf("High missing data rate: %.2f%%", qualityReport.MissingDataRate*100))
	}
	
	if qualityReport.OutlierRate > va.config.MaxOutlierRate {
		result.Warnings = append(result.Warnings, 
			fmt.Sprintf("High outlier rate: %.2f%%", qualityReport.OutlierRate*100))
	}
	
	return nil
}

func (va *ValidationAgent) runComparisonValidation(ctx context.Context, job *ValidationJob, result *ValidationResult) error {
	if job.OriginalData == nil || job.SyntheticData == nil {
		return fmt.Errorf("both original and synthetic data required for comparison validation")
	}
	
	// Generate comparison report
	comparisonReport := va.reportGenerator.GenerateComparisonReport(job.OriginalData, job.SyntheticData)
	result.ComparisonResults = comparisonReport
	result.OverallScore = comparisonReport.OverallSimilarity
	result.Passed = comparisonReport.OverallSimilarity >= job.TestConfig.MinQualityScore
	
	return nil
}

func (va *ValidationAgent) runCustomValidation(ctx context.Context, job *ValidationJob, result *ValidationResult) error {
	if len(job.TestConfig.CustomRules) == 0 {
		return fmt.Errorf("no custom rules specified for custom validation")
	}
	
	result.CustomRuleResults = make(map[string]bool)
	
	for _, ruleName := range job.TestConfig.CustomRules {
		passed := va.ruleEngine.EvaluateRule(ruleName, job.OriginalData)
		result.CustomRuleResults[ruleName] = passed
		
		if passed {
			result.TestsPassed++
		} else {
			result.TestsFailed++
		}
	}
	
	// Calculate overall score
	if result.TestsPassed+result.TestsFailed > 0 {
		result.OverallScore = float64(result.TestsPassed) / float64(result.TestsPassed+result.TestsFailed)
	}
	
	result.Passed = result.OverallScore >= job.TestConfig.MinQualityScore
	
	return nil
}

func (va *ValidationAgent) runCompleteValidation(ctx context.Context, job *ValidationJob, result *ValidationResult) error {
	// Run all validation types
	var errors []error
	
	// Statistical validation
	if va.config.EnableStatisticalTests && job.OriginalData != nil {
		if err := va.runStatisticalValidation(ctx, job, result); err != nil {
			errors = append(errors, fmt.Errorf("statistical validation failed: %w", err))
		}
	}
	
	// Quality validation  
	if va.config.EnableQualityMetrics && job.OriginalData != nil {
		qualityResult := &ValidationResult{}
		if err := va.runQualityValidation(ctx, job, qualityResult); err != nil {
			errors = append(errors, fmt.Errorf("quality validation failed: %w", err))
		} else {
			result.QualityMetrics = qualityResult.QualityMetrics
			result.DataQuality = qualityResult.DataQuality
		}
	}
	
	// Comparison validation
	if job.OriginalData != nil && job.SyntheticData != nil {
		comparisonResult := &ValidationResult{}
		if err := va.runComparisonValidation(ctx, job, comparisonResult); err != nil {
			errors = append(errors, fmt.Errorf("comparison validation failed: %w", err))
		} else {
			result.ComparisonResults = comparisonResult.ComparisonResults
		}
	}
	
	// Custom rules validation
	if va.config.EnableCustomRules && len(job.TestConfig.CustomRules) > 0 {
		customResult := &ValidationResult{}
		if err := va.runCustomValidation(ctx, job, customResult); err != nil {
			errors = append(errors, fmt.Errorf("custom validation failed: %w", err))
		} else {
			result.CustomRuleResults = customResult.CustomRuleResults
			result.TestsPassed += customResult.TestsPassed
			result.TestsFailed += customResult.TestsFailed
		}
	}
	
	// Calculate combined score
	scores := []float64{}
	if result.QualityMetrics != nil {
		scores = append(scores, result.QualityMetrics.OverallScore)
	}
	if result.ComparisonResults != nil {
		scores = append(scores, result.ComparisonResults.OverallSimilarity)
	}
	if result.TestsPassed+result.TestsFailed > 0 {
		statScore := float64(result.TestsPassed) / float64(result.TestsPassed+result.TestsFailed)
		scores = append(scores, statScore)
	}
	
	if len(scores) > 0 {
		var sum float64
		for _, score := range scores {
			sum += score
		}
		result.OverallScore = sum / float64(len(scores))
	}
	
	result.Passed = result.OverallScore >= job.TestConfig.MinQualityScore
	
	if len(errors) > 0 {
		return fmt.Errorf("validation completed with %d errors", len(errors))
	}
	
	return nil
}

// Helper methods
func (va *ValidationAgent) updateJobStatus(job *ValidationJob, status ValidationStatus) {
	va.mu.Lock()
	defer va.mu.Unlock()
	job.Status = status
}

func (va *ValidationAgent) addToHistory(result ValidationResult) {
	va.mu.Lock()
	defer va.mu.Unlock()
	
	va.validationHistory = append(va.validationHistory, result)
	
	// Trim history if too long
	maxHistory := 1000
	if len(va.validationHistory) > maxHistory {
		va.validationHistory = va.validationHistory[len(va.validationHistory)-maxHistory:]
	}
}

func (va *ValidationAgent) generateJobID() string {
	return fmt.Sprintf("val_%d_%d", time.Now().UnixNano(), len(va.activeValidations))
}

func (va *ValidationAgent) calculatePriority(validationType ValidationType, originalData, syntheticData *models.TimeSeries) int {
	priority := 5 // Default priority
	
	switch validationType {
	case ValidationTypeComplete:
		priority = 1 // Highest priority
	case ValidationTypeComparison:
		priority = 2
	case ValidationTypeQuality:
		priority = 3
	case ValidationTypeStatistical:
		priority = 4
	case ValidationTypeCustom:
		priority = 5
	}
	
	// Adjust based on data size
	if originalData != nil && len(originalData.DataPoints) > 10000 {
		priority++ // Lower priority for large datasets
	}
	
	return priority
}

func (va *ValidationAgent) validateJob(job *ValidationJob) error {
	if job.OriginalData == nil && job.SyntheticData == nil {
		return errors.NewValidationError("NO_DATA", "At least one dataset is required")
	}
	
	if job.OriginalData != nil {
		if len(job.OriginalData.DataPoints) < va.config.MinDataPoints {
			return errors.NewValidationError("INSUFFICIENT_DATA", 
				fmt.Sprintf("Original data has %d points, minimum %d required", 
					len(job.OriginalData.DataPoints), va.config.MinDataPoints))
		}
		
		if len(job.OriginalData.DataPoints) > va.config.MaxDataPoints {
			return errors.NewValidationError("TOO_MUCH_DATA", 
				fmt.Sprintf("Original data has %d points, maximum %d allowed", 
					len(job.OriginalData.DataPoints), va.config.MaxDataPoints))
		}
	}
	
	return nil
}

func (va *ValidationAgent) getDefaultTestConfig() *ValidationTestConfig {
	return &ValidationTestConfig{
		RunKolmogorovSmirnov: true,
		RunAndersonDarling:   true,
		RunShapiroWilk:      true,
		RunLjungBox:         true,
		CalculateBasicStats: true,
		CalculateDistribution: true,
		CalculateCorrelation: true,
		SignificanceLevel:   va.config.SignificanceLevel,
		MinQualityScore:     va.config.MinQualityScore,
		CustomRules:         []string{},
	}
}

// Shutdown stops the validation agent
func (va *ValidationAgent) Shutdown(ctx context.Context) error {
	va.Logger.Info("Shutting down validation agent")
	
	close(va.stopCh)
	
	// Cancel all active jobs
	va.mu.Lock()
	for _, job := range va.activeValidations {
		if job.CancelFunc != nil {
			job.CancelFunc()
		}
	}
	va.mu.Unlock()
	
	// Wait for workers to finish
	done := make(chan struct{})
	go func() {
		va.wg.Wait()
		close(done)
	}()
	
	select {
	case <-done:
		va.Logger.Info("All validation workers stopped")
	case <-ctx.Done():
		va.Logger.Warn("Shutdown timeout, some workers may still be running")
	}
	
	return nil
}

func getDefaultValidationConfig() *ValidationConfig {
	return &ValidationConfig{
		MaxConcurrentJobs:      10,
		JobTimeout:            30 * time.Minute,
		WorkerPoolSize:        4,
		EnableStatisticalTests: true,
		EnableQualityMetrics:   true,
		EnableCustomRules:      true,
		EnableReporting:        true,
		SignificanceLevel:      0.05,
		MinDataPoints:          100,
		MaxDataPoints:          1000000,
		MinQualityScore:        0.7,
		MaxMissingRate:         0.1,
		MaxOutlierRate:         0.05,
		AutoValidationInterval: time.Hour,
		RetentionPeriod:        7 * 24 * time.Hour,
		NotifyOnFailure:        true,
		NotifyOnCompletion:     false,
		AlertThresholds: map[string]float64{
			"quality_score": 0.7,
			"missing_rate":  0.1,
			"outlier_rate":  0.05,
		},
	}
}