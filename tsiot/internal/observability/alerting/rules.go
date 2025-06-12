package alerting

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// AlertManager manages alerting rules and notifications
type AlertManager struct {
	logger       *logrus.Logger
	config       *AlertConfig
	mu           sync.RWMutex
	rules        map[string]*AlertRule
	activeAlerts map[string]*Alert
	history      []Alert
	notifiers    []Notifier
	escalations  map[string]*EscalationPolicy
}

// AlertConfig configures the alert manager
type AlertConfig struct {
	Enabled             bool          `json:"enabled"`
	EvaluationInterval  time.Duration `json:"evaluation_interval"`
	DefaultSeverity     string        `json:"default_severity"`
	MaxActiveAlerts     int           `json:"max_active_alerts"`
	HistoryRetention    time.Duration `json:"history_retention"`
	EnableEscalation    bool          `json:"enable_escalation"`
	NotificationTimeout time.Duration `json:"notification_timeout"`
	SuppressRepeats     bool          `json:"suppress_repeats"`
	RepeatInterval      time.Duration `json:"repeat_interval"`
}

// AlertRule defines conditions for triggering alerts
type AlertRule struct {
	ID           string            `json:"id"`
	Name         string            `json:"name"`
	Description  string            `json:"description"`
	Query        string            `json:"query"`
	Condition    AlertCondition    `json:"condition"`
	Threshold    float64           `json:"threshold"`
	Duration     time.Duration     `json:"duration"`
	Severity     AlertSeverity     `json:"severity"`
	Labels       map[string]string `json:"labels"`
	Annotations  map[string]string `json:"annotations"`
	Enabled      bool              `json:"enabled"`
	CreatedAt    time.Time         `json:"created_at"`
	UpdatedAt    time.Time         `json:"updated_at"`
	EvaluationFunc func(context.Context, *AlertRule) (bool, float64, error) `json:"-"`
}

// AlertCondition defines the condition type for alerts
type AlertCondition string

const (
	ConditionGreaterThan    AlertCondition = "gt"
	ConditionLessThan       AlertCondition = "lt"
	ConditionEquals         AlertCondition = "eq"
	ConditionNotEquals      AlertCondition = "ne"
	ConditionGreaterOrEqual AlertCondition = "gte"
	ConditionLessOrEqual    AlertCondition = "lte"
	ConditionContains       AlertCondition = "contains"
	ConditionMissing        AlertCondition = "missing"
	ConditionPresent        AlertCondition = "present"
)

// AlertSeverity defines alert severity levels
type AlertSeverity string

const (
	SeverityInfo     AlertSeverity = "info"
	SeverityWarning  AlertSeverity = "warning"
	SeverityError    AlertSeverity = "error"
	SeverityCritical AlertSeverity = "critical"
)

// AlertStatus defines alert states
type AlertStatus string

const (
	StatusFiring   AlertStatus = "firing"
	StatusResolved AlertStatus = "resolved"
	StatusSuppressed AlertStatus = "suppressed"
)

// Alert represents an active or historical alert
type Alert struct {
	ID           string            `json:"id"`
	RuleID       string            `json:"rule_id"`
	RuleName     string            `json:"rule_name"`
	Status       AlertStatus       `json:"status"`
	Severity     AlertSeverity     `json:"severity"`
	Message      string            `json:"message"`
	Description  string            `json:"description"`
	Value        float64           `json:"value"`
	Threshold    float64           `json:"threshold"`
	Labels       map[string]string `json:"labels"`
	Annotations  map[string]string `json:"annotations"`
	StartsAt     time.Time         `json:"starts_at"`
	EndsAt       *time.Time        `json:"ends_at,omitempty"`
	UpdatedAt    time.Time         `json:"updated_at"`
	NotifiedAt   *time.Time        `json:"notified_at,omitempty"`
	AckBy        string            `json:"ack_by,omitempty"`
	AckAt        *time.Time        `json:"ack_at,omitempty"`
	EscalatedAt  *time.Time        `json:"escalated_at,omitempty"`
	RepeatCount  int               `json:"repeat_count"`
}

// Notifier defines the interface for alert notifications
type Notifier interface {
	Name() string
	Send(ctx context.Context, alert *Alert) error
	SupportsSeverity(severity AlertSeverity) bool
}

// EscalationPolicy defines how alerts should be escalated
type EscalationPolicy struct {
	ID       string              `json:"id"`
	Name     string              `json:"name"`
	Rules    []EscalationRule    `json:"rules"`
	Enabled  bool                `json:"enabled"`
}

// EscalationRule defines a single escalation step
type EscalationRule struct {
	After       time.Duration `json:"after"`
	Notifiers   []string      `json:"notifiers"`
	Severity    AlertSeverity `json:"severity"`
	RepeatEvery time.Duration `json:"repeat_every"`
}

// AlertEvaluation represents the result of evaluating an alert rule
type AlertEvaluation struct {
	RuleID    string    `json:"rule_id"`
	Triggered bool      `json:"triggered"`
	Value     float64   `json:"value"`
	Timestamp time.Time `json:"timestamp"`
	Error     error     `json:"error,omitempty"`
}

// NewAlertManager creates a new alert manager
func NewAlertManager(config *AlertConfig, logger *logrus.Logger) *AlertManager {
	if config == nil {
		config = getDefaultAlertConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	return &AlertManager{
		logger:       logger,
		config:       config,
		rules:        make(map[string]*AlertRule),
		activeAlerts: make(map[string]*Alert),
		history:      make([]Alert, 0),
		notifiers:    make([]Notifier, 0),
		escalations:  make(map[string]*EscalationPolicy),
	}
}

// Start starts the alert manager
func (am *AlertManager) Start(ctx context.Context) error {
	if !am.config.Enabled {
		am.logger.Info("Alert manager disabled")
		return nil
	}

	am.logger.Info("Starting alert manager")

	// Start evaluation loop
	go am.evaluationLoop(ctx)

	// Start escalation loop
	if am.config.EnableEscalation {
		go am.escalationLoop(ctx)
	}

	// Start cleanup loop
	go am.cleanupLoop(ctx)

	return nil
}

// RegisterRule registers a new alert rule
func (am *AlertManager) RegisterRule(rule *AlertRule) error {
	if rule.ID == "" {
		return fmt.Errorf("alert rule ID cannot be empty")
	}

	if rule.EvaluationFunc == nil {
		return fmt.Errorf("alert rule evaluation function cannot be nil")
	}

	am.mu.Lock()
	defer am.mu.Unlock()

	rule.CreatedAt = time.Now()
	rule.UpdatedAt = time.Now()
	am.rules[rule.ID] = rule

	am.logger.WithFields(logrus.Fields{
		"rule_id":   rule.ID,
		"rule_name": rule.Name,
		"severity":  rule.Severity,
	}).Info("Registered alert rule")

	return nil
}

// RegisterNotifier registers a new notifier
func (am *AlertManager) RegisterNotifier(notifier Notifier) {
	am.mu.Lock()
	defer am.mu.Unlock()

	am.notifiers = append(am.notifiers, notifier)
	am.logger.WithField("notifier", notifier.Name()).Info("Registered notifier")
}

// RegisterEscalationPolicy registers an escalation policy
func (am *AlertManager) RegisterEscalationPolicy(policy *EscalationPolicy) {
	am.mu.Lock()
	defer am.mu.Unlock()

	am.escalations[policy.ID] = policy
	am.logger.WithField("policy", policy.Name).Info("Registered escalation policy")
}

// GetActiveAlerts returns all active alerts
func (am *AlertManager) GetActiveAlerts() []*Alert {
	am.mu.RLock()
	defer am.mu.RUnlock()

	alerts := make([]*Alert, 0, len(am.activeAlerts))
	for _, alert := range am.activeAlerts {
		alertCopy := *alert
		alerts = append(alerts, &alertCopy)
	}

	return alerts
}

// GetAlertHistory returns alert history
func (am *AlertManager) GetAlertHistory(limit int) []Alert {
	am.mu.RLock()
	defer am.mu.RUnlock()

	if limit <= 0 || limit > len(am.history) {
		limit = len(am.history)
	}

	history := make([]Alert, limit)
	copy(history, am.history[len(am.history)-limit:])
	return history
}

// AcknowledgeAlert acknowledges an alert
func (am *AlertManager) AcknowledgeAlert(alertID, user string) error {
	am.mu.Lock()
	defer am.mu.Unlock()

	alert, exists := am.activeAlerts[alertID]
	if !exists {
		return fmt.Errorf("alert %s not found or not active", alertID)
	}

	now := time.Now()
	alert.AckBy = user
	alert.AckAt = &now
	alert.UpdatedAt = now

	am.logger.WithFields(logrus.Fields{
		"alert_id": alertID,
		"user":     user,
	}).Info("Alert acknowledged")

	return nil
}

// ResolveAlert manually resolves an alert
func (am *AlertManager) ResolveAlert(alertID, reason string) error {
	am.mu.Lock()
	defer am.mu.Unlock()

	alert, exists := am.activeAlerts[alertID]
	if !exists {
		return fmt.Errorf("alert %s not found or not active", alertID)
	}

	now := time.Now()
	alert.Status = StatusResolved
	alert.EndsAt = &now
	alert.UpdatedAt = now
	alert.Annotations["resolution_reason"] = reason

	// Move to history
	am.history = append(am.history, *alert)
	delete(am.activeAlerts, alertID)

	am.logger.WithFields(logrus.Fields{
		"alert_id": alertID,
		"reason":   reason,
	}).Info("Alert resolved manually")

	return nil
}

// evaluationLoop runs the main alert evaluation loop
func (am *AlertManager) evaluationLoop(ctx context.Context) {
	ticker := time.NewTicker(am.config.EvaluationInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			am.evaluateRules(ctx)
		}
	}
}

// evaluateRules evaluates all registered alert rules
func (am *AlertManager) evaluateRules(ctx context.Context) {
	am.mu.RLock()
	rules := make(map[string]*AlertRule)
	for k, v := range am.rules {
		if v.Enabled {
			rules[k] = v
		}
	}
	am.mu.RUnlock()

	for _, rule := range rules {
		go func(r *AlertRule) {
			am.evaluateRule(ctx, r)
		}(rule)
	}
}

// evaluateRule evaluates a single alert rule
func (am *AlertManager) evaluateRule(ctx context.Context, rule *AlertRule) {
	start := time.Now()

	// Execute evaluation function
	triggered, value, err := rule.EvaluationFunc(ctx, rule)
	
	evaluation := &AlertEvaluation{
		RuleID:    rule.ID,
		Triggered: triggered,
		Value:     value,
		Timestamp: time.Now(),
		Error:     err,
	}

	if err != nil {
		am.logger.WithError(err).WithField("rule_id", rule.ID).Error("Alert rule evaluation failed")
		return
	}

	am.logger.WithFields(logrus.Fields{
		"rule_id":   rule.ID,
		"triggered": triggered,
		"value":     value,
		"duration":  time.Since(start),
	}).Debug("Alert rule evaluated")

	// Handle alert state
	if triggered {
		am.handleAlertTriggered(rule, evaluation)
	} else {
		am.handleAlertResolved(rule)
	}
}

// handleAlertTriggered handles when an alert is triggered
func (am *AlertManager) handleAlertTriggered(rule *AlertRule, evaluation *AlertEvaluation) {
	am.mu.Lock()
	defer am.mu.Unlock()

	alertID := fmt.Sprintf("%s_%d", rule.ID, time.Now().UnixNano())
	
	// Check if alert already exists
	existingAlert := am.findActiveAlertByRule(rule.ID)
	if existingAlert != nil {
		// Update existing alert
		existingAlert.Value = evaluation.Value
		existingAlert.UpdatedAt = evaluation.Timestamp
		existingAlert.RepeatCount++
		
		// Check if we should send repeat notification
		if am.shouldSendRepeatNotification(existingAlert) {
			go am.sendNotification(existingAlert)
		}
		return
	}

	// Create new alert
	alert := &Alert{
		ID:          alertID,
		RuleID:      rule.ID,
		RuleName:    rule.Name,
		Status:      StatusFiring,
		Severity:    rule.Severity,
		Message:     am.generateAlertMessage(rule, evaluation.Value),
		Description: rule.Description,
		Value:       evaluation.Value,
		Threshold:   rule.Threshold,
		Labels:      copyLabels(rule.Labels),
		Annotations: copyLabels(rule.Annotations),
		StartsAt:    evaluation.Timestamp,
		UpdatedAt:   evaluation.Timestamp,
		RepeatCount: 0,
	}

	am.activeAlerts[alertID] = alert

	am.logger.WithFields(logrus.Fields{
		"alert_id":   alertID,
		"rule_name":  rule.Name,
		"severity":   rule.Severity,
		"value":      evaluation.Value,
		"threshold":  rule.Threshold,
	}).Info("Alert triggered")

	// Send notification
	go am.sendNotification(alert)
}

// handleAlertResolved handles when an alert is resolved
func (am *AlertManager) handleAlertResolved(rule *AlertRule) {
	am.mu.Lock()
	defer am.mu.Unlock()

	alert := am.findActiveAlertByRule(rule.ID)
	if alert == nil {
		return
	}

	now := time.Now()
	alert.Status = StatusResolved
	alert.EndsAt = &now
	alert.UpdatedAt = now

	// Move to history
	am.history = append(am.history, *alert)
	delete(am.activeAlerts, alert.ID)

	am.logger.WithFields(logrus.Fields{
		"alert_id":  alert.ID,
		"rule_name": alert.RuleName,
		"duration":  now.Sub(alert.StartsAt),
	}).Info("Alert resolved")

	// Send resolution notification
	go am.sendNotification(alert)
}

// sendNotification sends notifications for an alert
func (am *AlertManager) sendNotification(alert *Alert) {
	ctx, cancel := context.WithTimeout(context.Background(), am.config.NotificationTimeout)
	defer cancel()

	am.mu.RLock()
	notifiers := make([]Notifier, len(am.notifiers))
	copy(notifiers, am.notifiers)
	am.mu.RUnlock()

	for _, notifier := range notifiers {
		if !notifier.SupportsSeverity(alert.Severity) {
			continue
		}

		go func(n Notifier) {
			if err := n.Send(ctx, alert); err != nil {
				am.logger.WithError(err).WithFields(logrus.Fields{
					"notifier":  n.Name(),
					"alert_id":  alert.ID,
				}).Error("Failed to send notification")
			} else {
				am.logger.WithFields(logrus.Fields{
					"notifier": n.Name(),
					"alert_id": alert.ID,
				}).Debug("Notification sent")
			}
		}(notifier)
	}

	// Update notification timestamp
	am.mu.Lock()
	now := time.Now()
	alert.NotifiedAt = &now
	am.mu.Unlock()
}

// escalationLoop handles alert escalations
func (am *AlertManager) escalationLoop(ctx context.Context) {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			am.processEscalations(ctx)
		}
	}
}

// processEscalations processes alert escalations
func (am *AlertManager) processEscalations(ctx context.Context) {
	am.mu.RLock()
	alerts := make([]*Alert, 0, len(am.activeAlerts))
	for _, alert := range am.activeAlerts {
		if alert.Status == StatusFiring && alert.AckAt == nil {
			alerts = append(alerts, alert)
		}
	}
	am.mu.RUnlock()

	for _, alert := range alerts {
		am.checkEscalation(ctx, alert)
	}
}

// checkEscalation checks if an alert should be escalated
func (am *AlertManager) checkEscalation(ctx context.Context, alert *Alert) {
	// Simplified escalation logic
	duration := time.Since(alert.StartsAt)
	
	// Escalate critical alerts after 5 minutes
	if alert.Severity == SeverityCritical && duration > 5*time.Minute && alert.EscalatedAt == nil {
		am.mu.Lock()
		now := time.Now()
		alert.EscalatedAt = &now
		alert.Severity = SeverityCritical
		am.mu.Unlock()

		am.logger.WithFields(logrus.Fields{
			"alert_id": alert.ID,
			"duration": duration,
		}).Warn("Alert escalated")

		go am.sendNotification(alert)
	}
}

// cleanupLoop performs periodic cleanup of old alerts
func (am *AlertManager) cleanupLoop(ctx context.Context) {
	ticker := time.NewTicker(time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			am.cleanup()
		}
	}
}

// cleanup removes old alerts from history
func (am *AlertManager) cleanup() {
	am.mu.Lock()
	defer am.mu.Unlock()

	cutoff := time.Now().Add(-am.config.HistoryRetention)
	newHistory := make([]Alert, 0)

	for _, alert := range am.history {
		if alert.UpdatedAt.After(cutoff) {
			newHistory = append(newHistory, alert)
		}
	}

	removed := len(am.history) - len(newHistory)
	am.history = newHistory

	if removed > 0 {
		am.logger.WithField("removed", removed).Debug("Cleaned up old alerts")
	}
}

// Helper methods

func (am *AlertManager) findActiveAlertByRule(ruleID string) *Alert {
	for _, alert := range am.activeAlerts {
		if alert.RuleID == ruleID && alert.Status == StatusFiring {
			return alert
		}
	}
	return nil
}

func (am *AlertManager) shouldSendRepeatNotification(alert *Alert) bool {
	if !am.config.SuppressRepeats {
		return true
	}

	if alert.NotifiedAt == nil {
		return true
	}

	return time.Since(*alert.NotifiedAt) >= am.config.RepeatInterval
}

func (am *AlertManager) generateAlertMessage(rule *AlertRule, value float64) string {
	return fmt.Sprintf("%s: value %.2f %s threshold %.2f", 
		rule.Name, value, string(rule.Condition), rule.Threshold)
}

func copyLabels(labels map[string]string) map[string]string {
	if labels == nil {
		return make(map[string]string)
	}
	
	copy := make(map[string]string)
	for k, v := range labels {
		copy[k] = v
	}
	return copy
}

func getDefaultAlertConfig() *AlertConfig {
	return &AlertConfig{
		Enabled:             true,
		EvaluationInterval:  30 * time.Second,
		DefaultSeverity:     string(SeverityWarning),
		MaxActiveAlerts:     1000,
		HistoryRetention:    7 * 24 * time.Hour,
		EnableEscalation:    true,
		NotificationTimeout: 30 * time.Second,
		SuppressRepeats:     true,
		RepeatInterval:      4 * time.Hour,
	}
}