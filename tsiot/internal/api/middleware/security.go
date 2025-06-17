package middleware

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// SecurityConfig contains security middleware configuration
type SecurityConfig struct {
	Enabled                   bool     `json:"enabled" yaml:"enabled"`
	ContentTypeNosniff        bool     `json:"content_type_nosniff" yaml:"content_type_nosniff"`
	FrameDeny                 bool     `json:"frame_deny" yaml:"frame_deny"`
	ContentSecurityPolicy     string   `json:"content_security_policy" yaml:"content_security_policy"`
	HSTSMaxAge                int      `json:"hsts_max_age" yaml:"hsts_max_age"`
	HSTSIncludeSubdomains     bool     `json:"hsts_include_subdomains" yaml:"hsts_include_subdomains"`
	HSTSPreload               bool     `json:"hsts_preload" yaml:"hsts_preload"`
	ReferrerPolicy            string   `json:"referrer_policy" yaml:"referrer_policy"`
	PermissionsPolicy         string   `json:"permissions_policy" yaml:"permissions_policy"`
	CSRFProtection            bool     `json:"csrf_protection" yaml:"csrf_protection"`
	CSRFTokenName             string   `json:"csrf_token_name" yaml:"csrf_token_name"`
	CSRFCookieName            string   `json:"csrf_cookie_name" yaml:"csrf_cookie_name"`
	CSRFExemptPaths           []string `json:"csrf_exempt_paths" yaml:"csrf_exempt_paths"`
	AllowedHosts              []string `json:"allowed_hosts" yaml:"allowed_hosts"`
	TrustedProxies            []string `json:"trusted_proxies" yaml:"trusted_proxies"`
	ForceHTTPS                bool     `json:"force_https" yaml:"force_https"`
	RemoveServerHeader        bool     `json:"remove_server_header" yaml:"remove_server_header"`
	CustomHeaders             map[string]string `json:"custom_headers" yaml:"custom_headers"`
	SensitiveDataPatterns     []string `json:"sensitive_data_patterns" yaml:"sensitive_data_patterns"`
	MaxRequestSize            int64    `json:"max_request_size" yaml:"max_request_size"`
	RequestTimeout            time.Duration `json:"request_timeout" yaml:"request_timeout"`
}

// SecurityMiddleware provides comprehensive security middleware
type SecurityMiddleware struct {
	config                *SecurityConfig
	logger                *logrus.Logger
	sensitiveDataRegexps  []*regexp.Regexp
	csrfTokenStore        map[string]time.Time
}

// NewSecurityMiddleware creates a new security middleware
func NewSecurityMiddleware(config *SecurityConfig, logger *logrus.Logger) *SecurityMiddleware {
	if logger == nil {
		logger = logrus.New()
	}

	if config == nil {
		config = &SecurityConfig{
			Enabled:               true,
			ContentTypeNosniff:    true,
			FrameDeny:             true,
			ContentSecurityPolicy: "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
			HSTSMaxAge:            31536000, // 1 year
			HSTSIncludeSubdomains: true,
			HSTSPreload:           false,
			ReferrerPolicy:        "strict-origin-when-cross-origin",
			PermissionsPolicy:     "camera=(), microphone=(), geolocation=()",
			CSRFProtection:        true,
			CSRFTokenName:         "X-CSRF-Token",
			CSRFCookieName:        "csrf_token",
			ForceHTTPS:            false,
			RemoveServerHeader:    true,
			MaxRequestSize:        10 * 1024 * 1024, // 10MB
			RequestTimeout:        30 * time.Second,
			SensitiveDataPatterns: []string{
				`(?i)(password|passwd|pwd)[\s]*[:=][\s]*[^\s]+`,
				`(?i)(api[_-]?key|apikey)[\s]*[:=][\s]*[^\s]+`,
				`(?i)(secret|token)[\s]*[:=][\s]*[^\s]+`,
				`(?i)(access[_-]?token)[\s]*[:=][\s]*[^\s]+`,
				`(?i)(private[_-]?key)[\s]*[:=][\s]*[^\s]+`,
			},
		}
	}

	sm := &SecurityMiddleware{
		config:         config,
		logger:         logger,
		csrfTokenStore: make(map[string]time.Time),
	}

	// Compile sensitive data patterns
	sm.compileSensitiveDataPatterns()

	// Start cleanup routine for CSRF tokens
	if config.CSRFProtection {
		go sm.cleanupCSRFTokens()
	}

	return sm
}

// Middleware returns the HTTP middleware function
func (sm *SecurityMiddleware) Middleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Skip security if disabled
			if !sm.config.Enabled {
				next.ServeHTTP(w, r)
				return
			}

			// Apply security checks and headers
			if !sm.applySecurityMeasures(w, r) {
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// applySecurityMeasures applies all security measures and returns false if request should be blocked
func (sm *SecurityMiddleware) applySecurityMeasures(w http.ResponseWriter, r *http.Request) bool {
	// Check host header
	if !sm.validateHost(r) {
		sm.logger.WithFields(logrus.Fields{
			"host":   r.Host,
			"path":   r.URL.Path,
			"method": r.Method,
			"ip":     r.RemoteAddr,
		}).Warn("Request from disallowed host")
		http.Error(w, "Forbidden", http.StatusForbidden)
		return false
	}

	// Force HTTPS redirect
	if sm.config.ForceHTTPS && r.Header.Get("X-Forwarded-Proto") != "https" && r.TLS == nil {
		httpsURL := "https://" + r.Host + r.RequestURI
		http.Redirect(w, r, httpsURL, http.StatusMovedPermanently)
		return false
	}

	// Check request size
	if sm.config.MaxRequestSize > 0 && r.ContentLength > sm.config.MaxRequestSize {
		sm.logger.WithFields(logrus.Fields{
			"content_length": r.ContentLength,
			"max_size":       sm.config.MaxRequestSize,
			"path":           r.URL.Path,
		}).Warn("Request size exceeds limit")
		http.Error(w, "Request Entity Too Large", http.StatusRequestEntityTooLarge)
		return false
	}

	// CSRF protection
	if sm.config.CSRFProtection && !sm.isCSRFExempt(r.URL.Path) {
		if !sm.validateCSRF(w, r) {
			return false
		}
	}

	// Set security headers
	sm.setSecurityHeaders(w, r)

	return true
}

// validateHost checks if the host is allowed
func (sm *SecurityMiddleware) validateHost(r *http.Request) bool {
	if len(sm.config.AllowedHosts) == 0 {
		return true // No restriction if no hosts specified
	}

	host := strings.ToLower(r.Host)
	// Remove port if present
	if colonIndex := strings.Index(host, ":"); colonIndex != -1 {
		host = host[:colonIndex]
	}

	for _, allowedHost := range sm.config.AllowedHosts {
		allowedHost = strings.ToLower(allowedHost)
		if host == allowedHost {
			return true
		}
		// Check wildcard subdomains
		if strings.HasPrefix(allowedHost, "*.") {
			domain := allowedHost[2:]
			if strings.HasSuffix(host, "."+domain) || host == domain {
				return true
			}
		}
	}

	return false
}

// validateCSRF validates CSRF tokens
func (sm *SecurityMiddleware) validateCSRF(w http.ResponseWriter, r *http.Request) bool {
	// Skip CSRF for safe methods
	if r.Method == "GET" || r.Method == "HEAD" || r.Method == "OPTIONS" {
		// Generate and set CSRF token for GET requests
		token := sm.generateCSRFToken()
		sm.setCSRFCookie(w, token)
		return true
	}

	// Get token from header or form
	token := r.Header.Get(sm.config.CSRFTokenName)
	if token == "" {
		token = r.FormValue("csrf_token")
	}

	// Get expected token from cookie
	cookie, err := r.Cookie(sm.config.CSRFCookieName)
	if err != nil || cookie.Value == "" {
		sm.logger.WithFields(logrus.Fields{
			"path":   r.URL.Path,
			"method": r.Method,
			"ip":     r.RemoteAddr,
		}).Warn("CSRF validation failed: no cookie")
		sm.handleCSRFError(w, r)
		return false
	}

	// Validate token
	if !sm.isValidCSRFToken(token, cookie.Value) {
		sm.logger.WithFields(logrus.Fields{
			"path":   r.URL.Path,
			"method": r.Method,
			"ip":     r.RemoteAddr,
		}).Warn("CSRF validation failed: invalid token")
		sm.handleCSRFError(w, r)
		return false
	}

	return true
}

// generateCSRFToken generates a new CSRF token
func (sm *SecurityMiddleware) generateCSRFToken() string {
	bytes := make([]byte, 32)
	if _, err := rand.Read(bytes); err != nil {
		// Fallback to timestamp-based token
		return hex.EncodeToString([]byte(time.Now().String()))
	}
	token := hex.EncodeToString(bytes)
	sm.csrfTokenStore[token] = time.Now().Add(24 * time.Hour) // 24 hour expiry
	return token
}

// isValidCSRFToken validates a CSRF token
func (sm *SecurityMiddleware) isValidCSRFToken(provided, expected string) bool {
	if provided == "" || expected == "" {
		return false
	}

	// Check if token exists and is not expired
	if expiry, exists := sm.csrfTokenStore[expected]; exists {
		if time.Now().Before(expiry) && provided == expected {
			return true
		}
		// Remove expired token
		delete(sm.csrfTokenStore, expected)
	}

	return false
}

// setCSRFCookie sets the CSRF token cookie
func (sm *SecurityMiddleware) setCSRFCookie(w http.ResponseWriter, token string) {
	cookie := &http.Cookie{
		Name:     sm.config.CSRFCookieName,
		Value:    token,
		Path:     "/",
		HttpOnly: true,
		Secure:   sm.config.ForceHTTPS,
		SameSite: http.SameSiteStrictMode,
		MaxAge:   24 * 60 * 60, // 24 hours
	}
	http.SetCookie(w, cookie)
}

// isCSRFExempt checks if a path is exempt from CSRF protection
func (sm *SecurityMiddleware) isCSRFExempt(path string) bool {
	for _, exemptPath := range sm.config.CSRFExemptPaths {
		if strings.HasPrefix(path, exemptPath) {
			return true
		}
	}

	// Default exempt paths
	exemptPaths := []string{
		"/health",
		"/metrics",
		"/ping",
		"/docs",
		"/swagger",
	}

	for _, exemptPath := range exemptPaths {
		if strings.HasPrefix(path, exemptPath) {
			return true
		}
	}

	return false
}

// handleCSRFError handles CSRF validation errors
func (sm *SecurityMiddleware) handleCSRFError(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusForbidden)

	response := map[string]interface{}{
		"error": map[string]interface{}{
			"code":    "CSRF_TOKEN_INVALID",
			"message": "CSRF token validation failed",
			"details": "Invalid or missing CSRF token",
		},
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	}

	if err := writeJSON(w, response); err != nil {
		sm.logger.WithError(err).Error("Failed to write CSRF error response")
	}
}

// setSecurityHeaders sets various security headers
func (sm *SecurityMiddleware) setSecurityHeaders(w http.ResponseWriter, r *http.Request) {
	headers := w.Header()

	// Remove server header if configured
	if sm.config.RemoveServerHeader {
		headers.Del("Server")
	}

	// X-Content-Type-Options
	if sm.config.ContentTypeNosniff {
		headers.Set("X-Content-Type-Options", "nosniff")
	}

	// X-Frame-Options
	if sm.config.FrameDeny {
		headers.Set("X-Frame-Options", "DENY")
	}

	// Content Security Policy
	if sm.config.ContentSecurityPolicy != "" {
		headers.Set("Content-Security-Policy", sm.config.ContentSecurityPolicy)
	}

	// HSTS (only for HTTPS)
	if (r.TLS != nil || r.Header.Get("X-Forwarded-Proto") == "https") && sm.config.HSTSMaxAge > 0 {
		hstsValue := fmt.Sprintf("max-age=%d", sm.config.HSTSMaxAge)
		if sm.config.HSTSIncludeSubdomains {
			hstsValue += "; includeSubDomains"
		}
		if sm.config.HSTSPreload {
			hstsValue += "; preload"
		}
		headers.Set("Strict-Transport-Security", hstsValue)
	}

	// Referrer Policy
	if sm.config.ReferrerPolicy != "" {
		headers.Set("Referrer-Policy", sm.config.ReferrerPolicy)
	}

	// Permissions Policy
	if sm.config.PermissionsPolicy != "" {
		headers.Set("Permissions-Policy", sm.config.PermissionsPolicy)
	}

	// X-XSS-Protection (legacy but still useful)
	headers.Set("X-XSS-Protection", "1; mode=block")

	// Custom headers
	for name, value := range sm.config.CustomHeaders {
		headers.Set(name, value)
	}
}

// compileSensitiveDataPatterns compiles regex patterns for sensitive data detection
func (sm *SecurityMiddleware) compileSensitiveDataPatterns() {
	sm.sensitiveDataRegexps = make([]*regexp.Regexp, 0, len(sm.config.SensitiveDataPatterns))
	
	for _, pattern := range sm.config.SensitiveDataPatterns {
		if compiled, err := regexp.Compile(pattern); err == nil {
			sm.sensitiveDataRegexps = append(sm.sensitiveDataRegexps, compiled)
		} else {
			sm.logger.WithError(err).WithField("pattern", pattern).Warn("Failed to compile sensitive data pattern")
		}
	}
}

// ScanForSensitiveData scans content for sensitive data patterns
func (sm *SecurityMiddleware) ScanForSensitiveData(content string) []string {
	var findings []string
	
	for _, regex := range sm.sensitiveDataRegexps {
		matches := regex.FindAllString(content, -1)
		findings = append(findings, matches...)
	}
	
	return findings
}

// cleanupCSRFTokens periodically removes expired CSRF tokens
func (sm *SecurityMiddleware) cleanupCSRFTokens() {
	ticker := time.NewTicker(time.Hour)
	defer ticker.Stop()

	for range ticker.C {
		sm.cleanupExpiredTokens()
	}
}

// cleanupExpiredTokens removes expired CSRF tokens
func (sm *SecurityMiddleware) cleanupExpiredTokens() {
	now := time.Now()
	for token, expiry := range sm.csrfTokenStore {
		if now.After(expiry) {
			delete(sm.csrfTokenStore, token)
		}
	}
	
	sm.logger.WithField("active_tokens", len(sm.csrfTokenStore)).Debug("CSRF token cleanup completed")
}

// GetSecurityStats returns security middleware statistics
func (sm *SecurityMiddleware) GetSecurityStats() map[string]interface{} {
	return map[string]interface{}{
		"csrf_tokens_active": len(sm.csrfTokenStore),
		"config":             sm.config,
	}
}

