package agents

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// LogAlertHandler logs anomaly alerts to a logger
type LogAlertHandler struct {
	logger  *logrus.Logger
	enabled bool
	name    string
}

// NewLogAlertHandler creates a new log alert handler
func NewLogAlertHandler(logger *logrus.Logger) *LogAlertHandler {
	if logger == nil {
		logger = logrus.New()
	}
	
	return &LogAlertHandler{
		logger:  logger,
		enabled: true,
		name:    "log_handler",
	}
}

func (h *LogAlertHandler) HandleAlert(ctx context.Context, event *AnomalyEvent) error {
	if !h.enabled {
		return nil
	}
	
	logLevel := h.getLogLevel(event.Severity)
	
	fields := logrus.Fields{
		"event_id":     event.ID,
		"series_id":    event.SeriesID,
		"detector":     event.DetectorName,
		"severity":     event.Severity.String(),
		"anomaly_type": event.AnomalyType.String(),
		"score":        event.AnomalyScore,
		"description":  event.Description,
		"timestamp":    event.CreatedAt.Format(time.RFC3339),
	}
	
	if len(event.AnomalyPoints) > 0 {
		fields["anomaly_points"] = len(event.AnomalyPoints)
		fields["first_anomaly_value"] = event.AnomalyPoints[0].Value
		fields["first_anomaly_timestamp"] = event.AnomalyPoints[0].Timestamp.Format(time.RFC3339)
	}
	
	if event.Context != nil {
		fields["window_mean"] = event.Context.WindowMean
		fields["window_stddev"] = event.Context.WindowStdDev
		fields["trend_direction"] = event.Context.TrendDirection
	}
	
	entry := h.logger.WithFields(fields)
	
	switch logLevel {
	case logrus.InfoLevel:
		entry.Info("Anomaly detected")
	case logrus.WarnLevel:
		entry.Warn("Anomaly detected")
	case logrus.ErrorLevel:
		entry.Error("Critical anomaly detected")
	default:
		entry.Info("Anomaly detected")
	}
	
	return nil
}

func (h *LogAlertHandler) GetName() string {
	return h.name
}

func (h *LogAlertHandler) IsEnabled() bool {
	return h.enabled
}

func (h *LogAlertHandler) SetEnabled(enabled bool) {
	h.enabled = enabled
}

func (h *LogAlertHandler) getLogLevel(severity AnomalySeverity) logrus.Level {
	switch severity {
	case SeverityInfo, SeverityLow:
		return logrus.InfoLevel
	case SeverityMedium:
		return logrus.WarnLevel
	case SeverityHigh, SeverityCritical:
		return logrus.ErrorLevel
	default:
		return logrus.InfoLevel
	}
}

// ConsoleAlertHandler prints anomaly alerts to console
type ConsoleAlertHandler struct {
	enabled bool
	name    string
}

// NewConsoleAlertHandler creates a new console alert handler
func NewConsoleAlertHandler() *ConsoleAlertHandler {
	return &ConsoleAlertHandler{
		enabled: true,
		name:    "console_handler",
	}
}

func (h *ConsoleAlertHandler) HandleAlert(ctx context.Context, event *AnomalyEvent) error {
	if !h.enabled {
		return nil
	}
	
	prefix := h.getSeverityPrefix(event.Severity)
	timestamp := event.CreatedAt.Format("2006-01-02 15:04:05")
	
	fmt.Printf("%s [%s] ANOMALY DETECTED: %s\n", prefix, timestamp, event.Description)
	fmt.Printf("  Series ID: %s\n", event.SeriesID)
	fmt.Printf("  Detector: %s\n", event.DetectorName)
	fmt.Printf("  Severity: %s\n", event.Severity.String())
	fmt.Printf("  Type: %s\n", event.AnomalyType.String())
	fmt.Printf("  Score: %.3f\n", event.AnomalyScore)
	
	if len(event.AnomalyPoints) > 0 {
		fmt.Printf("  Anomaly Points: %d\n", len(event.AnomalyPoints))
		fmt.Printf("  First Anomaly: %.3f at %s\n", 
			event.AnomalyPoints[0].Value, 
			event.AnomalyPoints[0].Timestamp.Format("15:04:05"))
	}
	
	if event.Context != nil {
		fmt.Printf("  Context: mean=%.3f, stddev=%.3f, trend=%s\n", 
			event.Context.WindowMean, 
			event.Context.WindowStdDev, 
			event.Context.TrendDirection)
	}
	
	fmt.Println("  " + strings.Repeat("-", 50))
	
	return nil
}

func (h *ConsoleAlertHandler) GetName() string {
	return h.name
}

func (h *ConsoleAlertHandler) IsEnabled() bool {
	return h.enabled
}

func (h *ConsoleAlertHandler) SetEnabled(enabled bool) {
	h.enabled = enabled
}

func (h *ConsoleAlertHandler) getSeverityPrefix(severity AnomalySeverity) string {
	switch severity {
	case SeverityInfo:
		return "[INFO]"
	case SeverityLow:
		return "[LOW ]"
	case SeverityMedium:
		return "[MED ]"
	case SeverityHigh:
		return "[HIGH]"
	case SeverityCritical:
		return "[CRIT]"
	default:
		return "[????]"
	}
}

// EmailAlertHandler sends anomaly alerts via email (simplified implementation)
type EmailAlertHandler struct {
	enabled     bool
	name        string
	toAddresses []string
	fromAddress string
	smtpHost    string
	smtpPort    int
	logger      *logrus.Logger
}

// NewEmailAlertHandler creates a new email alert handler
func NewEmailAlertHandler(toAddresses []string, fromAddress, smtpHost string, smtpPort int, logger *logrus.Logger) *EmailAlertHandler {
	if logger == nil {
		logger = logrus.New()
	}
	
	return &EmailAlertHandler{
		enabled:     true,
		name:        "email_handler",
		toAddresses: toAddresses,
		fromAddress: fromAddress,
		smtpHost:    smtpHost,
		smtpPort:    smtpPort,
		logger:      logger,
	}
}

func (h *EmailAlertHandler) HandleAlert(ctx context.Context, event *AnomalyEvent) error {
	if !h.enabled || len(h.toAddresses) == 0 {
		return nil
	}
	
	// Simplified email implementation - in practice would use proper SMTP client
	subject := fmt.Sprintf("[ANOMALY %s] %s - %s", 
		event.Severity.String(), 
		event.SeriesID, 
		event.AnomalyType.String())
	
	body := h.formatEmailBody(event)
	
	h.logger.WithFields(logrus.Fields{
		"to":       h.toAddresses,
		"subject":  subject,
		"event_id": event.ID,
	}).Info("Email alert would be sent (SMTP not implemented)")
	
	// In a real implementation, this would send the email
	// For now, just log that we would send it
	log.Printf("EMAIL ALERT: %s\n%s", subject, body)
	
	return nil
}

func (h *EmailAlertHandler) GetName() string {
	return h.name
}

func (h *EmailAlertHandler) IsEnabled() bool {
	return h.enabled
}

func (h *EmailAlertHandler) SetEnabled(enabled bool) {
	h.enabled = enabled
}

func (h *EmailAlertHandler) formatEmailBody(event *AnomalyEvent) string {
	body := fmt.Sprintf(`
Anomaly Detection Alert

Event Details:
- Event ID: %s
- Series ID: %s
- Detector: %s
- Severity: %s
- Anomaly Type: %s
- Score: %.3f
- Description: %s
- Timestamp: %s

`, event.ID, event.SeriesID, event.DetectorName, 
	event.Severity.String(), event.AnomalyType.String(), 
	event.AnomalyScore, event.Description, 
	event.CreatedAt.Format(time.RFC3339))
	
	if len(event.AnomalyPoints) > 0 {
		body += fmt.Sprintf("\nAnomalous Data Points (%d total):\n", len(event.AnomalyPoints))
		for i, point := range event.AnomalyPoints {
			if i >= 5 { // Limit to first 5 points
				body += fmt.Sprintf("... and %d more\n", len(event.AnomalyPoints)-5)
				break
			}
			body += fmt.Sprintf("- %s: %.3f (expected: %.3f, deviation: %.3f)\n", 
				point.Timestamp.Format("15:04:05"), 
				point.Value, point.Expected, point.Deviation)
		}
	}
	
	if event.Context != nil {
		body += fmt.Sprintf(`
Context Information:
- Window Size: %d
- Window Mean: %.3f
- Window Std Dev: %.3f
- Trend Direction: %s
- Min Value: %.3f
- Max Value: %.3f

`, event.Context.WindowSize, event.Context.WindowMean, 
	event.Context.WindowStdDev, event.Context.TrendDirection,
	event.Context.WindowMin, event.Context.WindowMax)
	}
	
	body += "\nThis is an automated alert from the TSIoT Anomaly Detection System.\n"
	
	return body
}

// WebhookAlertHandler sends anomaly alerts to a webhook endpoint
type WebhookAlertHandler struct {
	enabled    bool
	name       string
	webhookURL string
	timeout    time.Duration
	logger     *logrus.Logger
}

// NewWebhookAlertHandler creates a new webhook alert handler
func NewWebhookAlertHandler(webhookURL string, timeout time.Duration, logger *logrus.Logger) *WebhookAlertHandler {
	if logger == nil {
		logger = logrus.New()
	}
	
	if timeout == 0 {
		timeout = 10 * time.Second
	}
	
	return &WebhookAlertHandler{
		enabled:    true,
		name:       "webhook_handler",
		webhookURL: webhookURL,
		timeout:    timeout,
		logger:     logger,
	}
}

func (h *WebhookAlertHandler) HandleAlert(ctx context.Context, event *AnomalyEvent) error {
	if !h.enabled || h.webhookURL == "" {
		return nil
	}
	
	// Simplified webhook implementation - in practice would use HTTP client
	h.logger.WithFields(logrus.Fields{
		"webhook_url": h.webhookURL,
		"event_id":    event.ID,
		"series_id":   event.SeriesID,
		"severity":    event.Severity.String(),
	}).Info("Webhook alert would be sent (HTTP client not implemented)")
	
	// In a real implementation, this would send an HTTP POST request
	log.Printf("WEBHOOK ALERT to %s: Event %s - %s severity anomaly in series %s", 
		h.webhookURL, event.ID, event.Severity.String(), event.SeriesID)
	
	return nil
}

func (h *WebhookAlertHandler) GetName() string {
	return h.name
}

func (h *WebhookAlertHandler) IsEnabled() bool {
	return h.enabled
}

func (h *WebhookAlertHandler) SetEnabled(enabled bool) {
	h.enabled = enabled
}

