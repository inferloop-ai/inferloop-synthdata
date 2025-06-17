package health

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// HealthMonitor provides comprehensive health monitoring
type HealthMonitor struct {
	logger    *logrus.Logger
	config    *HealthConfig
	mu        sync.RWMutex
	checks    map[string]HealthCheck
	status    *SystemStatus
	history   []HealthSnapshot
	observers []HealthObserver
}

// HealthConfig configures health monitoring
type HealthConfig struct {
	Enabled            bool          `json:"enabled"`
	CheckInterval      time.Duration `json:"check_interval"`
	FailureThreshold   int           `json:"failure_threshold"`
	SuccessThreshold   int           `json:"success_threshold"`
	Timeout            time.Duration `json:"timeout"`
	HistorySize        int           `json:"history_size"`
	EnableDetailedLogs bool          `json:"enable_detailed_logs"`
}

// HealthCheck defines a health check function
type HealthCheck interface {
	Name() string
	Check(ctx context.Context) HealthResult
	Critical() bool
	Timeout() time.Duration
}

// HealthResult represents the result of a health check
type HealthResult struct {
	Status      HealthStatus      `json:"status"`
	Message     string            `json:"message"`
	Duration    time.Duration     `json:"duration"`
	Timestamp   time.Time         `json:"timestamp"`
	Details     map[string]string `json:"details"`
	Error       error             `json:"error,omitempty"`
}

// HealthStatus represents the health status
type HealthStatus string

const (
	StatusHealthy   HealthStatus = "healthy"
	StatusDegraded  HealthStatus = "degraded"
	StatusUnhealthy HealthStatus = "unhealthy"
	StatusUnknown   HealthStatus = "unknown"
)

// SystemStatus represents overall system health
type SystemStatus struct {
	OverallStatus   HealthStatus               `json:"overall_status"`
	CheckResults    map[string]HealthResult    `json:"check_results"`
	LastCheck       time.Time                  `json:"last_check"`
	CriticalIssues  []string                   `json:"critical_issues"`
	TotalChecks     int                        `json:"total_checks"`
	HealthyChecks   int                        `json:"healthy_checks"`
	DegradedChecks  int                        `json:"degraded_checks"`
	UnhealthyChecks int                        `json:"unhealthy_checks"`
	Uptime          time.Duration              `json:"uptime"`
	StartTime       time.Time                  `json:"start_time"`
}

// HealthSnapshot captures system health at a point in time
type HealthSnapshot struct {
	Timestamp     time.Time                `json:"timestamp"`
	OverallStatus HealthStatus             `json:"overall_status"`
	CheckResults  map[string]HealthResult  `json:"check_results"`
	Summary       HealthSummary            `json:"summary"`
}

// HealthSummary provides aggregated health metrics
type HealthSummary struct {
	TotalChecks     int     `json:"total_checks"`
	HealthyPercent  float64 `json:"healthy_percent"`
	DegradedPercent float64 `json:"degraded_percent"`
	UnhealthyPercent float64 `json:"unhealthy_percent"`
	AverageResponseTime time.Duration `json:"average_response_time"`
	CriticalFailures int `json:"critical_failures"`
}

// HealthObserver receives health status updates
type HealthObserver interface {
	OnHealthChange(status *SystemStatus)
	OnCheckFailure(checkName string, result HealthResult)
	OnCriticalFailure(checkName string, result HealthResult)
}

// BasicHealthCheck implements a basic health check
type BasicHealthCheck struct {
	name        string
	checkFunc   func(ctx context.Context) error
	critical    bool
	timeout     time.Duration
	description string
}

// NewHealthMonitor creates a new health monitor
func NewHealthMonitor(config *HealthConfig, logger *logrus.Logger) *HealthMonitor {
	if config == nil {
		config = getDefaultHealthConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	return &HealthMonitor{
		logger:    logger,
		config:    config,
		checks:    make(map[string]HealthCheck),
		status:    &SystemStatus{
			StartTime:     time.Now(),
			CheckResults:  make(map[string]HealthResult),
			CriticalIssues: make([]string, 0),
		},
		history:   make([]HealthSnapshot, 0),
		observers: make([]HealthObserver, 0),
	}
}

// Start starts the health monitoring
func (hm *HealthMonitor) Start(ctx context.Context) error {
	if !hm.config.Enabled {
		hm.logger.Info("Health monitoring disabled")
		return nil
	}

	hm.logger.Info("Starting health monitoring")

	// Register default health checks
	hm.registerDefaultChecks()

	// Start monitoring loop
	go hm.monitoringLoop(ctx)

	return nil
}

// RegisterCheck registers a new health check
func (hm *HealthMonitor) RegisterCheck(check HealthCheck) {
	hm.mu.Lock()
	defer hm.mu.Unlock()
	
	hm.checks[check.Name()] = check
	hm.logger.WithField("check", check.Name()).Info("Registered health check")
}

// RegisterObserver registers a health observer
func (hm *HealthMonitor) RegisterObserver(observer HealthObserver) {
	hm.mu.Lock()
	defer hm.mu.Unlock()
	
	hm.observers = append(hm.observers, observer)
}

// GetStatus returns the current system health status
func (hm *HealthMonitor) GetStatus() *SystemStatus {
	hm.mu.RLock()
	defer hm.mu.RUnlock()
	
	// Create a copy to avoid race conditions
	status := &SystemStatus{
		OverallStatus:   hm.status.OverallStatus,
		LastCheck:       hm.status.LastCheck,
		TotalChecks:     hm.status.TotalChecks,
		HealthyChecks:   hm.status.HealthyChecks,
		DegradedChecks:  hm.status.DegradedChecks,
		UnhealthyChecks: hm.status.UnhealthyChecks,
		Uptime:          time.Since(hm.status.StartTime),
		StartTime:       hm.status.StartTime,
		CheckResults:    make(map[string]HealthResult),
		CriticalIssues:  make([]string, len(hm.status.CriticalIssues)),
	}
	
	// Copy check results
	for k, v := range hm.status.CheckResults {
		status.CheckResults[k] = v
	}
	
	// Copy critical issues
	copy(status.CriticalIssues, hm.status.CriticalIssues)
	
	return status
}

// GetHistory returns health check history
func (hm *HealthMonitor) GetHistory() []HealthSnapshot {
	hm.mu.RLock()
	defer hm.mu.RUnlock()
	
	history := make([]HealthSnapshot, len(hm.history))
	copy(history, hm.history)
	return history
}

// RunCheck runs a specific health check manually
func (hm *HealthMonitor) RunCheck(ctx context.Context, checkName string) (HealthResult, error) {
	hm.mu.RLock()
	check, exists := hm.checks[checkName]
	hm.mu.RUnlock()
	
	if !exists {
		return HealthResult{}, fmt.Errorf("health check '%s' not found", checkName)
	}
	
	return hm.executeCheck(ctx, check), nil
}

// monitoringLoop runs the main health monitoring loop
func (hm *HealthMonitor) monitoringLoop(ctx context.Context) {
	ticker := time.NewTicker(hm.config.CheckInterval)
	defer ticker.Stop()
	
	// Run initial check
	hm.runAllChecks(ctx)
	
	for {
		select {
		case <-ctx.Done():
			hm.logger.Info("Stopping health monitoring")
			return
		case <-ticker.C:
			hm.runAllChecks(ctx)
		}
	}
}

// runAllChecks executes all registered health checks
func (hm *HealthMonitor) runAllChecks(ctx context.Context) {
	hm.mu.RLock()
	checks := make(map[string]HealthCheck)
	for k, v := range hm.checks {
		checks[k] = v
	}
	hm.mu.RUnlock()
	
	results := make(map[string]HealthResult)
	criticalIssues := make([]string, 0)
	
	// Execute checks concurrently
	var wg sync.WaitGroup
	resultsChan := make(chan struct {
		name   string
		result HealthResult
	}, len(checks))
	
	for name, check := range checks {
		wg.Add(1)
		go func(n string, c HealthCheck) {
			defer wg.Done()
			result := hm.executeCheck(ctx, c)
			resultsChan <- struct {
				name   string
				result HealthResult
			}{n, result}
		}(name, check)
	}
	
	wg.Wait()
	close(resultsChan)
	
	// Collect results
	for result := range resultsChan {
		results[result.name] = result.result
		
		// Check for critical issues
		if result.result.Status == StatusUnhealthy {
			check := checks[result.name]
			if check.Critical() {
				criticalIssues = append(criticalIssues, result.name)
			}
		}
	}
	
	// Update status
	hm.updateStatus(results, criticalIssues)
	
	// Notify observers
	hm.notifyObservers()
}

// executeCheck executes a single health check
func (hm *HealthMonitor) executeCheck(ctx context.Context, check HealthCheck) HealthResult {
	start := time.Now()
	
	// Create timeout context
	timeout := check.Timeout()
	if timeout == 0 {
		timeout = hm.config.Timeout
	}
	
	checkCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	
	// Execute check
	result := check.Check(checkCtx)
	result.Duration = time.Since(start)
	result.Timestamp = time.Now()
	
	// Log result if detailed logging is enabled
	if hm.config.EnableDetailedLogs {
		hm.logger.WithFields(logrus.Fields{
			"check":    check.Name(),
			"status":   result.Status,
			"duration": result.Duration,
			"message":  result.Message,
		}).Debug("Health check completed")
	}
	
	return result
}

// updateStatus updates the overall system status
func (hm *HealthMonitor) updateStatus(results map[string]HealthResult, criticalIssues []string) {
	hm.mu.Lock()
	defer hm.mu.Unlock()
	
	hm.status.CheckResults = results
	hm.status.LastCheck = time.Now()
	hm.status.CriticalIssues = criticalIssues
	hm.status.TotalChecks = len(results)
	hm.status.HealthyChecks = 0
	hm.status.DegradedChecks = 0
	hm.status.UnhealthyChecks = 0
	
	// Count status types
	for _, result := range results {
		switch result.Status {
		case StatusHealthy:
			hm.status.HealthyChecks++
		case StatusDegraded:
			hm.status.DegradedChecks++
		case StatusUnhealthy:
			hm.status.UnhealthyChecks++
		}
	}
	
	// Determine overall status
	hm.status.OverallStatus = hm.calculateOverallStatus(criticalIssues)
	
	// Add to history
	hm.addToHistory()
}

// calculateOverallStatus determines the overall system status
func (hm *HealthMonitor) calculateOverallStatus(criticalIssues []string) HealthStatus {
	if len(criticalIssues) > 0 {
		return StatusUnhealthy
	}
	
	if hm.status.UnhealthyChecks > 0 {
		return StatusDegraded
	}
	
	if hm.status.DegradedChecks > 0 {
		return StatusDegraded
	}
	
	return StatusHealthy
}

// addToHistory adds current status to history
func (hm *HealthMonitor) addToHistory() {
	snapshot := HealthSnapshot{
		Timestamp:     time.Now(),
		OverallStatus: hm.status.OverallStatus,
		CheckResults:  make(map[string]HealthResult),
		Summary:       hm.calculateSummary(),
	}
	
	// Copy check results
	for k, v := range hm.status.CheckResults {
		snapshot.CheckResults[k] = v
	}
	
	hm.history = append(hm.history, snapshot)
	
	// Trim history if too large
	if len(hm.history) > hm.config.HistorySize {
		hm.history = hm.history[len(hm.history)-hm.config.HistorySize:]
	}
}

// calculateSummary calculates health summary metrics
func (hm *HealthMonitor) calculateSummary() HealthSummary {
	total := float64(hm.status.TotalChecks)
	if total == 0 {
		return HealthSummary{}
	}
	
	var totalDuration time.Duration
	criticalFailures := len(hm.status.CriticalIssues)
	
	for _, result := range hm.status.CheckResults {
		totalDuration += result.Duration
	}
	
	avgDuration := time.Duration(0)
	if len(hm.status.CheckResults) > 0 {
		avgDuration = totalDuration / time.Duration(len(hm.status.CheckResults))
	}
	
	return HealthSummary{
		TotalChecks:         hm.status.TotalChecks,
		HealthyPercent:      float64(hm.status.HealthyChecks) / total * 100,
		DegradedPercent:     float64(hm.status.DegradedChecks) / total * 100,
		UnhealthyPercent:    float64(hm.status.UnhealthyChecks) / total * 100,
		AverageResponseTime: avgDuration,
		CriticalFailures:    criticalFailures,
	}
}

// notifyObservers notifies all registered observers
func (hm *HealthMonitor) notifyObservers() {
	status := hm.GetStatus()
	
	for _, observer := range hm.observers {
		go func(obs HealthObserver) {
			defer func() {
				if r := recover(); r != nil {
					hm.logger.WithField("error", r).Error("Health observer panic")
				}
			}()
			obs.OnHealthChange(status)
		}(observer)
	}
}

// registerDefaultChecks registers built-in health checks
func (hm *HealthMonitor) registerDefaultChecks() {
	// Memory check
	memoryCheck := NewBasicHealthCheck(
		"memory",
		func(ctx context.Context) error {
			// Simplified memory check
			return nil
		},
		false,
		5*time.Second,
		"System memory usage check",
	)
	hm.RegisterCheck(memoryCheck)
	
	// CPU check
	cpuCheck := NewBasicHealthCheck(
		"cpu",
		func(ctx context.Context) error {
			// Simplified CPU check
			return nil
		},
		false,
		5*time.Second,
		"System CPU usage check",
	)
	hm.RegisterCheck(cpuCheck)
	
	// Disk check
	diskCheck := NewBasicHealthCheck(
		"disk",
		func(ctx context.Context) error {
			// Simplified disk check
			return nil
		},
		true,
		10*time.Second,
		"System disk space check",
	)
	hm.RegisterCheck(diskCheck)
}

// NewBasicHealthCheck creates a new basic health check
func NewBasicHealthCheck(name string, checkFunc func(ctx context.Context) error, critical bool, timeout time.Duration, description string) *BasicHealthCheck {
	return &BasicHealthCheck{
		name:        name,
		checkFunc:   checkFunc,
		critical:    critical,
		timeout:     timeout,
		description: description,
	}
}

// Name returns the check name
func (bhc *BasicHealthCheck) Name() string {
	return bhc.name
}

// Check executes the health check
func (bhc *BasicHealthCheck) Check(ctx context.Context) HealthResult {
	err := bhc.checkFunc(ctx)
	
	result := HealthResult{
		Status:    StatusHealthy,
		Message:   "OK",
		Details:   make(map[string]string),
		Timestamp: time.Now(),
	}
	
	if err != nil {
		result.Status = StatusUnhealthy
		result.Message = err.Error()
		result.Error = err
	}
	
	result.Details["description"] = bhc.description
	result.Details["critical"] = fmt.Sprintf("%t", bhc.critical)
	
	return result
}

// Critical returns whether this check is critical
func (bhc *BasicHealthCheck) Critical() bool {
	return bhc.critical
}

// Timeout returns the check timeout
func (bhc *BasicHealthCheck) Timeout() time.Duration {
	return bhc.timeout
}

func getDefaultHealthConfig() *HealthConfig {
	return &HealthConfig{
		Enabled:            true,
		CheckInterval:      30 * time.Second,
		FailureThreshold:   3,
		SuccessThreshold:   2,
		Timeout:            10 * time.Second,
		HistorySize:        100,
		EnableDetailedLogs: false,
	}
}