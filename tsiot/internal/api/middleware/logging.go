package middleware

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// LoggingConfig contains logging middleware configuration
type LoggingConfig struct {
	Enabled           bool     `json:"enabled" yaml:"enabled"`
	Level             string   `json:"level" yaml:"level"`
	Format            string   `json:"format" yaml:"format"` // "json", "text"
	IncludeRequestID  bool     `json:"include_request_id" yaml:"include_request_id"`
	IncludeUserAgent  bool     `json:"include_user_agent" yaml:"include_user_agent"`
	IncludeReferer    bool     `json:"include_referer" yaml:"include_referer"`
	IncludeHeaders    bool     `json:"include_headers" yaml:"include_headers"`
	IncludeBody       bool     `json:"include_body" yaml:"include_body"`
	IncludeResponse   bool     `json:"include_response" yaml:"include_response"`
	MaxBodySize       int      `json:"max_body_size" yaml:"max_body_size"`
	ExcludePaths      []string `json:"exclude_paths" yaml:"exclude_paths"`
	ExcludeUserAgents []string `json:"exclude_user_agents" yaml:"exclude_user_agents"`
	SensitiveHeaders  []string `json:"sensitive_headers" yaml:"sensitive_headers"`
	LogSlowRequests   bool     `json:"log_slow_requests" yaml:"log_slow_requests"`
	SlowRequestThreshold time.Duration `json:"slow_request_threshold" yaml:"slow_request_threshold"`
}

// LoggingMiddleware provides comprehensive request/response logging
type LoggingMiddleware struct {
	config *LoggingConfig
	logger *logrus.Logger
}

// responseWriter wraps http.ResponseWriter to capture response data
type responseWriter struct {
	http.ResponseWriter
	statusCode int
	size       int
	body       *bytes.Buffer
	started    time.Time
}

// requestInfo contains information about the HTTP request
type requestInfo struct {
	Method        string            `json:"method"`
	URL           string            `json:"url"`
	Proto         string            `json:"proto"`
	Host          string            `json:"host"`
	RemoteAddr    string            `json:"remote_addr"`
	RequestURI    string            `json:"request_uri"`
	UserAgent     string            `json:"user_agent,omitempty"`
	Referer       string            `json:"referer,omitempty"`
	Headers       map[string]string `json:"headers,omitempty"`
	Body          string            `json:"body,omitempty"`
	RequestID     string            `json:"request_id,omitempty"`
	ContentLength int64             `json:"content_length"`
}

// responseInfo contains information about the HTTP response
type responseInfo struct {
	StatusCode    int               `json:"status_code"`
	Size          int               `json:"size"`
	Headers       map[string]string `json:"headers,omitempty"`
	Body          string            `json:"body,omitempty"`
	Duration      time.Duration     `json:"duration"`
	DurationMs    float64           `json:"duration_ms"`
}

// NewLoggingMiddleware creates a new logging middleware
func NewLoggingMiddleware(config *LoggingConfig, logger *logrus.Logger) *LoggingMiddleware {
	if logger == nil {
		logger = logrus.New()
	}

	if config == nil {
		config = &LoggingConfig{
			Enabled:           true,
			Level:             "info",
			Format:            "json",
			IncludeRequestID:  true,
			IncludeUserAgent:  true,
			IncludeReferer:    false,
			IncludeHeaders:    false,
			IncludeBody:       false,
			IncludeResponse:   false,
			MaxBodySize:       1024, // 1KB
			LogSlowRequests:   true,
			SlowRequestThreshold: 1 * time.Second,
			SensitiveHeaders: []string{
				"authorization",
				"cookie",
				"x-api-key",
				"x-auth-token",
				"authentication",
			},
		}
	}

	// Set logger format
	if config.Format == "json" {
		logger.SetFormatter(&logrus.JSONFormatter{})
	} else {
		logger.SetFormatter(&logrus.TextFormatter{})
	}

	// Set logger level
	level, err := logrus.ParseLevel(config.Level)
	if err != nil {
		level = logrus.InfoLevel
	}
	logger.SetLevel(level)

	return &LoggingMiddleware{
		config: config,
		logger: logger,
	}
}

// Middleware returns the HTTP middleware function
func (lm *LoggingMiddleware) Middleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Skip logging if disabled
			if !lm.config.Enabled {
				next.ServeHTTP(w, r)
				return
			}

			// Check if path should be excluded
			if lm.shouldExclude(r) {
				next.ServeHTTP(w, r)
				return
			}

			start := time.Now()

			// Capture request information
			reqInfo := lm.captureRequestInfo(r)

			// Create response writer wrapper
			rw := &responseWriter{
				ResponseWriter: w,
				statusCode:     200,
				body:           bytes.NewBuffer(nil),
				started:        start,
			}

			// Process request
			next.ServeHTTP(rw, r)

			// Calculate duration
			duration := time.Since(start)

			// Capture response information
			respInfo := &responseInfo{
				StatusCode: rw.statusCode,
				Size:       rw.size,
				Duration:   duration,
				DurationMs: float64(duration.Nanoseconds()) / 1000000,
			}

			if lm.config.IncludeResponse && rw.body.Len() > 0 {
				respInfo.Body = lm.sanitizeBody(rw.body.String())
				if lm.config.IncludeHeaders {
					respInfo.Headers = lm.captureHeaders(w.Header(), false)
				}
			}

			// Log the request
			lm.logRequest(reqInfo, respInfo)
		})
	}
}

// shouldExclude checks if the request should be excluded from logging
func (lm *LoggingMiddleware) shouldExclude(r *http.Request) bool {
	// Check excluded paths
	for _, excludePath := range lm.config.ExcludePaths {
		if strings.HasPrefix(r.URL.Path, excludePath) {
			return true
		}
	}

	// Check excluded user agents
	userAgent := r.UserAgent()
	for _, excludeUA := range lm.config.ExcludeUserAgents {
		if strings.Contains(userAgent, excludeUA) {
			return true
		}
	}

	// Default exclusions for health checks
	healthPaths := []string{"/health", "/ping", "/metrics", "/status"}
	for _, healthPath := range healthPaths {
		if r.URL.Path == healthPath {
			return true
		}
	}

	return false
}

// captureRequestInfo captures relevant request information
func (lm *LoggingMiddleware) captureRequestInfo(r *http.Request) *requestInfo {
	info := &requestInfo{
		Method:        r.Method,
		URL:           r.URL.String(),
		Proto:         r.Proto,
		Host:          r.Host,
		RemoteAddr:    lm.getClientIP(r),
		RequestURI:    r.RequestURI,
		ContentLength: r.ContentLength,
	}

	// Include request ID if available
	if lm.config.IncludeRequestID {
		if requestID := r.Header.Get("X-Request-ID"); requestID != "" {
			info.RequestID = requestID
		} else if requestID := r.Context().Value("request_id"); requestID != nil {
			if id, ok := requestID.(string); ok {
				info.RequestID = id
			}
		}
	}

	// Include user agent
	if lm.config.IncludeUserAgent {
		info.UserAgent = r.UserAgent()
	}

	// Include referer
	if lm.config.IncludeReferer {
		info.Referer = r.Referer()
	}

	// Include headers
	if lm.config.IncludeHeaders {
		info.Headers = lm.captureHeaders(r.Header, true)
	}

	// Include request body
	if lm.config.IncludeBody && r.Body != nil {
		info.Body = lm.captureRequestBody(r)
	}

	return info
}

// captureHeaders captures HTTP headers, filtering sensitive ones
func (lm *LoggingMiddleware) captureHeaders(headers http.Header, isRequest bool) map[string]string {
	captured := make(map[string]string)
	
	for name, values := range headers {
		lowerName := strings.ToLower(name)
		
		// Skip sensitive headers
		if lm.isSensitiveHeader(lowerName) {
			captured[name] = "[REDACTED]"
			continue
		}
		
		// Join multiple values
		captured[name] = strings.Join(values, ", ")
	}
	
	return captured
}

// isSensitiveHeader checks if a header contains sensitive information
func (lm *LoggingMiddleware) isSensitiveHeader(headerName string) bool {
	for _, sensitive := range lm.config.SensitiveHeaders {
		if strings.ToLower(sensitive) == headerName {
			return true
		}
	}
	return false
}

// captureRequestBody captures and sanitizes the request body
func (lm *LoggingMiddleware) captureRequestBody(r *http.Request) string {
	if r.Body == nil {
		return ""
	}

	// Read body
	bodyBytes, err := io.ReadAll(io.LimitReader(r.Body, int64(lm.config.MaxBodySize)))
	if err != nil {
		return "[ERROR_READING_BODY]"
	}

	// Restore body for downstream handlers
	r.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))

	return lm.sanitizeBody(string(bodyBytes))
}

// sanitizeBody removes sensitive information from body content
func (lm *LoggingMiddleware) sanitizeBody(body string) string {
	if body == "" {
		return ""
	}

	// Limit size
	if len(body) > lm.config.MaxBodySize {
		body = body[:lm.config.MaxBodySize] + "...[TRUNCATED]"
	}

	// Try to parse as URL-encoded form and redact sensitive fields
	if values, err := url.ParseQuery(body); err == nil {
		for key := range values {
			lowerKey := strings.ToLower(key)
			if lm.isSensitiveField(lowerKey) {
				values.Set(key, "[REDACTED]")
			}
		}
		return values.Encode()
	}

	// For JSON or other formats, apply basic redaction
	return lm.redactSensitivePatterns(body)
}

// isSensitiveField checks if a field name indicates sensitive data
func (lm *LoggingMiddleware) isSensitiveField(fieldName string) bool {
	sensitiveFields := []string{
		"password", "passwd", "pwd",
		"secret", "token", "key",
		"api_key", "apikey",
		"access_token", "refresh_token",
		"private_key", "certificate",
		"ssn", "social_security",
		"credit_card", "card_number",
	}

	for _, sensitive := range sensitiveFields {
		if strings.Contains(fieldName, sensitive) {
			return true
		}
	}
	return false
}

// redactSensitivePatterns applies basic pattern-based redaction
func (lm *LoggingMiddleware) redactSensitivePatterns(content string) string {
	// Simple patterns for common sensitive data
	patterns := map[string]string{
		`"password"\s*:\s*"[^"]*"`:       `"password":"[REDACTED]"`,
		`"token"\s*:\s*"[^"]*"`:          `"token":"[REDACTED]"`,
		`"api_key"\s*:\s*"[^"]*"`:        `"api_key":"[REDACTED]"`,
		`"secret"\s*:\s*"[^"]*"`:         `"secret":"[REDACTED]"`,
		`"private_key"\s*:\s*"[^"]*"`:    `"private_key":"[REDACTED]"`,
	}

	result := content
	for pattern, replacement := range patterns {
		// In a real implementation, you'd use regexp for better matching
		// For simplicity, we'll just do basic string replacement
		if strings.Contains(strings.ToLower(result), strings.ToLower(pattern[:10])) {
			// Basic replacement - in production use proper regex
			result = "[REDACTED_SENSITIVE_DATA]"
			break
		}
	}

	return result
}

// getClientIP extracts the real client IP address
func (lm *LoggingMiddleware) getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		ips := strings.Split(xff, ",")
		return strings.TrimSpace(ips[0])
	}

	// Check X-Real-IP header
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}

	// Use remote address
	if ip, _, err := net.SplitHostPort(r.RemoteAddr); err == nil {
		return ip
	}

	return r.RemoteAddr
}

// logRequest logs the request and response information
func (lm *LoggingMiddleware) logRequest(reqInfo *requestInfo, respInfo *responseInfo) {
	fields := logrus.Fields{
		"method":       reqInfo.Method,
		"url":          reqInfo.URL,
		"status":       respInfo.StatusCode,
		"size":         respInfo.Size,
		"duration":     respInfo.Duration,
		"duration_ms":  respInfo.DurationMs,
		"remote_addr":  reqInfo.RemoteAddr,
		"proto":        reqInfo.Proto,
	}

	// Add optional fields
	if reqInfo.RequestID != "" {
		fields["request_id"] = reqInfo.RequestID
	}

	if lm.config.IncludeUserAgent && reqInfo.UserAgent != "" {
		fields["user_agent"] = reqInfo.UserAgent
	}

	if lm.config.IncludeReferer && reqInfo.Referer != "" {
		fields["referer"] = reqInfo.Referer
	}

	if lm.config.IncludeHeaders && len(reqInfo.Headers) > 0 {
		fields["request_headers"] = reqInfo.Headers
	}

	if lm.config.IncludeBody && reqInfo.Body != "" {
		fields["request_body"] = reqInfo.Body
	}

	if lm.config.IncludeResponse {
		if len(respInfo.Headers) > 0 {
			fields["response_headers"] = respInfo.Headers
		}
		if respInfo.Body != "" {
			fields["response_body"] = respInfo.Body
		}
	}

	// Determine log level based on status code and duration
	var logLevel logrus.Level
	var message string

	switch {
	case respInfo.StatusCode >= 500:
		logLevel = logrus.ErrorLevel
		message = "HTTP request error"
	case respInfo.StatusCode >= 400:
		logLevel = logrus.WarnLevel
		message = "HTTP request client error"
	case lm.config.LogSlowRequests && respInfo.Duration > lm.config.SlowRequestThreshold:
		logLevel = logrus.WarnLevel
		message = "HTTP request slow"
	default:
		logLevel = logrus.InfoLevel
		message = "HTTP request"
	}

	// Log the entry
	lm.logger.WithFields(fields).Log(logLevel, message)
}

// WriteHeader captures the status code
func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// Write captures the response body and size
func (rw *responseWriter) Write(data []byte) (int, error) {
	size, err := rw.ResponseWriter.Write(data)
	rw.size += size

	// Capture response body if needed
	if rw.body != nil && rw.body.Len() < 1024 { // Limit capture size
		rw.body.Write(data)
	}

	return size, err
}

// Hijack implements http.Hijacker interface
func (rw *responseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	hijacker, ok := rw.ResponseWriter.(http.Hijacker)
	if !ok {
		return nil, nil, fmt.Errorf("ResponseWriter does not implement http.Hijacker")
	}
	return hijacker.Hijack()
}

// GetLoggingStats returns logging middleware statistics
func (lm *LoggingMiddleware) GetLoggingStats() map[string]interface{} {
	return map[string]interface{}{
		"config": lm.config,
		"level":  lm.logger.GetLevel().String(),
	}
}