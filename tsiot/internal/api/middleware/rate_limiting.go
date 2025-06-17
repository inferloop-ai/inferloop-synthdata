package middleware

import (
	"fmt"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/errors"
)

// RateLimitConfig contains rate limiting configuration
type RateLimitConfig struct {
	Enabled            bool              `json:"enabled" yaml:"enabled"`
	RequestsPerMinute  int               `json:"requests_per_minute" yaml:"requests_per_minute"`
	RequestsPerHour    int               `json:"requests_per_hour" yaml:"requests_per_hour"`
	RequestsPerDay     int               `json:"requests_per_day" yaml:"requests_per_day"`
	BurstSize          int               `json:"burst_size" yaml:"burst_size"`
	KeyFunc            string            `json:"key_func" yaml:"key_func"` // "ip", "user", "api_key"
	ExemptPaths        []string          `json:"exempt_paths" yaml:"exempt_paths"`
	ExemptIPs          []string          `json:"exempt_ips" yaml:"exempt_ips"`
	PerUserLimits      map[string]int    `json:"per_user_limits" yaml:"per_user_limits"`
	PerPathLimits      map[string]int    `json:"per_path_limits" yaml:"per_path_limits"`
	TrustedProxies     []string          `json:"trusted_proxies" yaml:"trusted_proxies"`
	IncludeHeaders     bool              `json:"include_headers" yaml:"include_headers"`
}

// RateLimiter represents a token bucket rate limiter
type RateLimiter struct {
	tokens    int
	capacity  int
	refillRate float64
	lastRefill time.Time
	mu        sync.Mutex
}

// ClientLimiter tracks rate limits for a specific client
type ClientLimiter struct {
	minuteLimiter *RateLimiter
	hourLimiter   *RateLimiter
	dayLimiter    *RateLimiter
	lastSeen      time.Time
}

// RateLimitMiddleware provides rate limiting middleware
type RateLimitMiddleware struct {
	config   *RateLimitConfig
	logger   *logrus.Logger
	limiters map[string]*ClientLimiter
	mu       sync.RWMutex
}

// NewRateLimitMiddleware creates a new rate limiting middleware
func NewRateLimitMiddleware(config *RateLimitConfig, logger *logrus.Logger) *RateLimitMiddleware {
	if logger == nil {
		logger = logrus.New()
	}

	if config == nil {
		config = &RateLimitConfig{
			Enabled:           true,
			RequestsPerMinute: 60,
			RequestsPerHour:   1000,
			RequestsPerDay:    10000,
			BurstSize:         10,
			KeyFunc:           "ip",
			IncludeHeaders:    true,
		}
	}

	// Set defaults
	if config.BurstSize == 0 {
		config.BurstSize = config.RequestsPerMinute / 6 // 10-second burst
		if config.BurstSize < 1 {
			config.BurstSize = 1
		}
	}

	rlm := &RateLimitMiddleware{
		config:   config,
		logger:   logger,
		limiters: make(map[string]*ClientLimiter),
	}

	// Start cleanup routine
	go rlm.cleanupRoutine()

	return rlm
}

// Middleware returns the HTTP middleware function
func (rlm *RateLimitMiddleware) Middleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Skip rate limiting if disabled
			if !rlm.config.Enabled {
				next.ServeHTTP(w, r)
				return
			}

			// Check if path is exempt
			if rlm.isExemptPath(r.URL.Path) {
				next.ServeHTTP(w, r)
				return
			}

			// Get client key
			key, err := rlm.getClientKey(r)
			if err != nil {
				rlm.handleRateLimitError(w, r, err)
				return
			}

			// Check if IP is exempt
			if rlm.isExemptIP(key) {
				next.ServeHTTP(w, r)
				return
			}

			// Check rate limit
			allowed, resetTime, err := rlm.checkRateLimit(key, r)
			if err != nil {
				rlm.handleRateLimitError(w, r, err)
				return
			}

			// Add rate limit headers
			if rlm.config.IncludeHeaders {
				rlm.addRateLimitHeaders(w, key, resetTime)
			}

			if !allowed {
				rlm.handleRateLimitExceeded(w, r, resetTime)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// getClientKey determines the key to use for rate limiting
func (rlm *RateLimitMiddleware) getClientKey(r *http.Request) (string, error) {
	switch rlm.config.KeyFunc {
	case "ip":
		return rlm.getClientIP(r), nil
	case "user":
		if user, ok := GetUserFromContext(r.Context()); ok {
			return "user:" + user.ID, nil
		}
		// Fallback to IP if no user
		return rlm.getClientIP(r), nil
	case "api_key":
		apiKey := r.Header.Get("X-API-Key")
		if apiKey == "" {
			apiKey = r.URL.Query().Get("api_key")
		}
		if apiKey != "" {
			return "apikey:" + apiKey, nil
		}
		// Fallback to IP if no API key
		return rlm.getClientIP(r), nil
	default:
		return rlm.getClientIP(r), nil
	}
}

// getClientIP extracts the real client IP
func (rlm *RateLimitMiddleware) getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header (for reverse proxies)
	xForwardedFor := r.Header.Get("X-Forwarded-For")
	if xForwardedFor != "" {
		// Take the first IP in the list
		ips := strings.Split(xForwardedFor, ",")
		clientIP := strings.TrimSpace(ips[0])
		if rlm.isTrustedProxy(r.RemoteAddr) {
			return clientIP
		}
	}

	// Check X-Real-IP header
	xRealIP := r.Header.Get("X-Real-IP")
	if xRealIP != "" && rlm.isTrustedProxy(r.RemoteAddr) {
		return xRealIP
	}

	// Use direct connection IP
	ip, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return ip
}

// isTrustedProxy checks if the request comes from a trusted proxy
func (rlm *RateLimitMiddleware) isTrustedProxy(remoteAddr string) bool {
	ip, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		ip = remoteAddr
	}

	for _, trustedProxy := range rlm.config.TrustedProxies {
		if ip == trustedProxy {
			return true
		}
		
		// Check CIDR ranges
		if strings.Contains(trustedProxy, "/") {
			_, network, err := net.ParseCIDR(trustedProxy)
			if err == nil && network.Contains(net.ParseIP(ip)) {
				return true
			}
		}
	}

	return false
}

// checkRateLimit checks if the request is within rate limits
func (rlm *RateLimitMiddleware) checkRateLimit(key string, r *http.Request) (bool, time.Time, error) {
	rlm.mu.Lock()
	defer rlm.mu.Unlock()

	// Get or create client limiter
	limiter, exists := rlm.limiters[key]
	if !exists {
		limiter = rlm.createClientLimiter(key, r)
		rlm.limiters[key] = limiter
	}

	limiter.lastSeen = time.Now()

	// Check each time window
	var resetTime time.Time

	// Check minute limit
	if rlm.config.RequestsPerMinute > 0 {
		if !limiter.minuteLimiter.Allow() {
			resetTime = limiter.minuteLimiter.nextRefill()
			return false, resetTime, nil
		}
	}

	// Check hour limit
	if rlm.config.RequestsPerHour > 0 {
		if !limiter.hourLimiter.Allow() {
			resetTime = limiter.hourLimiter.nextRefill()
			return false, resetTime, nil
		}
	}

	// Check day limit
	if rlm.config.RequestsPerDay > 0 {
		if !limiter.dayLimiter.Allow() {
			resetTime = limiter.dayLimiter.nextRefill()
			return false, resetTime, nil
		}
	}

	return true, time.Time{}, nil
}

// createClientLimiter creates a new client limiter
func (rlm *RateLimitMiddleware) createClientLimiter(key string, r *http.Request) *ClientLimiter {
	// Get custom limits if configured
	minuteLimit := rlm.config.RequestsPerMinute
	hourLimit := rlm.config.RequestsPerHour
	dayLimit := rlm.config.RequestsPerDay

	// Check per-user limits
	if strings.HasPrefix(key, "user:") {
		userID := strings.TrimPrefix(key, "user:")
		if userLimit, exists := rlm.config.PerUserLimits[userID]; exists {
			minuteLimit = userLimit
		}
	}

	// Check per-path limits
	if pathLimit, exists := rlm.config.PerPathLimits[r.URL.Path]; exists {
		minuteLimit = pathLimit
	}

	limiter := &ClientLimiter{
		lastSeen: time.Now(),
	}

	// Create rate limiters for each time window
	if minuteLimit > 0 {
		limiter.minuteLimiter = NewRateLimiter(minuteLimit, rlm.config.BurstSize, time.Minute)
	}
	if hourLimit > 0 {
		limiter.hourLimiter = NewRateLimiter(hourLimit, hourLimit/10, time.Hour)
	}
	if dayLimit > 0 {
		limiter.dayLimiter = NewRateLimiter(dayLimit, dayLimit/100, 24*time.Hour)
	}

	return limiter
}

// NewRateLimiter creates a new token bucket rate limiter
func NewRateLimiter(limit, burst int, window time.Duration) *RateLimiter {
	return &RateLimiter{
		tokens:     burst,
		capacity:   burst,
		refillRate: float64(limit) / window.Seconds(),
		lastRefill: time.Now(),
	}
}

// Allow checks if a request is allowed
func (rl *RateLimiter) Allow() bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(rl.lastRefill).Seconds()
	
	// Add tokens based on elapsed time
	tokensToAdd := int(elapsed * rl.refillRate)
	rl.tokens += tokensToAdd
	
	if rl.tokens > rl.capacity {
		rl.tokens = rl.capacity
	}
	
	rl.lastRefill = now

	// Check if we have tokens available
	if rl.tokens > 0 {
		rl.tokens--
		return true
	}

	return false
}

// nextRefill returns the time when tokens will be available
func (rl *RateLimiter) nextRefill() time.Time {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	if rl.refillRate <= 0 {
		return time.Now().Add(time.Hour) // Default to 1 hour
	}

	secondsUntilToken := 1.0 / rl.refillRate
	return time.Now().Add(time.Duration(secondsUntilToken * float64(time.Second)))
}

// isExemptPath checks if the path is exempt from rate limiting
func (rlm *RateLimitMiddleware) isExemptPath(path string) bool {
	for _, exemptPath := range rlm.config.ExemptPaths {
		if strings.HasPrefix(path, exemptPath) {
			return true
		}
	}

	// Default exempt paths
	exemptPaths := []string{
		"/health",
		"/ping",
		"/metrics",
	}

	for _, exemptPath := range exemptPaths {
		if strings.HasPrefix(path, exemptPath) {
			return true
		}
	}

	return false
}

// isExemptIP checks if the IP is exempt from rate limiting
func (rlm *RateLimitMiddleware) isExemptIP(key string) bool {
	ip := key
	if strings.Contains(key, ":") {
		parts := strings.Split(key, ":")
		if len(parts) > 0 {
			ip = parts[len(parts)-1] // Get the IP part
		}
	}

	for _, exemptIP := range rlm.config.ExemptIPs {
		if ip == exemptIP {
			return true
		}
		
		// Check CIDR ranges
		if strings.Contains(exemptIP, "/") {
			_, network, err := net.ParseCIDR(exemptIP)
			if err == nil && network.Contains(net.ParseIP(ip)) {
				return true
			}
		}
	}

	return false
}

// addRateLimitHeaders adds rate limit headers to the response
func (rlm *RateLimitMiddleware) addRateLimitHeaders(w http.ResponseWriter, key string, resetTime time.Time) {
	rlm.mu.RLock()
	limiter, exists := rlm.limiters[key]
	rlm.mu.RUnlock()

	if !exists {
		return
	}

	headers := w.Header()

	// Add limit headers
	if limiter.minuteLimiter != nil {
		headers.Set("X-RateLimit-Limit", strconv.Itoa(rlm.config.RequestsPerMinute))
		headers.Set("X-RateLimit-Remaining", strconv.Itoa(limiter.minuteLimiter.tokens))
	}

	// Add reset time
	if !resetTime.IsZero() {
		headers.Set("X-RateLimit-Reset", strconv.FormatInt(resetTime.Unix(), 10))
		headers.Set("Retry-After", strconv.Itoa(int(time.Until(resetTime).Seconds())))
	}
}

// handleRateLimitExceeded handles rate limit exceeded errors
func (rlm *RateLimitMiddleware) handleRateLimitExceeded(w http.ResponseWriter, r *http.Request, resetTime time.Time) {
	rlm.logger.WithFields(logrus.Fields{
		"ip":         rlm.getClientIP(r),
		"path":       r.URL.Path,
		"method":     r.Method,
		"user_agent": r.UserAgent(),
		"reset_time": resetTime,
	}).Warn("Rate limit exceeded")

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusTooManyRequests)

	response := map[string]interface{}{
		"error": map[string]interface{}{
			"code":    "RATE_LIMIT_EXCEEDED",
			"message": "Rate limit exceeded",
			"details": fmt.Sprintf("Rate limit exceeded. Try again at %s", resetTime.Format(time.RFC3339)),
		},
		"retry_after": int(time.Until(resetTime).Seconds()),
		"timestamp":   time.Now().UTC().Format(time.RFC3339),
	}

	if err := writeJSON(w, response); err != nil {
		rlm.logger.WithError(err).Error("Failed to write rate limit error response")
	}
}

// handleRateLimitError handles rate limiting errors
func (rlm *RateLimitMiddleware) handleRateLimitError(w http.ResponseWriter, r *http.Request, err error) {
	rlm.logger.WithError(err).Error("Rate limiting error")

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusInternalServerError)

	response := map[string]interface{}{
		"error": map[string]interface{}{
			"code":    "RATE_LIMIT_ERROR",
			"message": "Rate limiting error",
			"details": err.Error(),
		},
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	}

	if err := writeJSON(w, response); err != nil {
		rlm.logger.WithError(err).Error("Failed to write error response")
	}
}

// cleanupRoutine removes old client limiters
func (rlm *RateLimitMiddleware) cleanupRoutine() {
	ticker := time.NewTicker(time.Hour)
	defer ticker.Stop()

	for range ticker.C {
		rlm.cleanup()
	}
}

// cleanup removes limiters for inactive clients
func (rlm *RateLimitMiddleware) cleanup() {
	rlm.mu.Lock()
	defer rlm.mu.Unlock()

	cutoff := time.Now().Add(-2 * time.Hour)
	
	for key, limiter := range rlm.limiters {
		if limiter.lastSeen.Before(cutoff) {
			delete(rlm.limiters, key)
		}
	}

	rlm.logger.WithField("active_limiters", len(rlm.limiters)).Debug("Rate limiter cleanup completed")
}

// GetStats returns rate limiting statistics
func (rlm *RateLimitMiddleware) GetStats() map[string]interface{} {
	rlm.mu.RLock()
	defer rlm.mu.RUnlock()

	return map[string]interface{}{
		"active_limiters": len(rlm.limiters),
		"config":          rlm.config,
	}
}