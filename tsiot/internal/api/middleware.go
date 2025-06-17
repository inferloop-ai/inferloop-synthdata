package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"
)

// MiddlewareConfig holds configuration for all middleware
type MiddlewareConfig struct {
	EnableLogging      bool
	EnableCORS         bool
	EnableRateLimit    bool
	EnableAuth         bool
	EnableCompression  bool
	EnableSecurity     bool
	
	RateLimitRequests  int
	RateLimitWindow    time.Duration
	AuthRequired       []string
	AllowedOrigins     []string
}

// DefaultMiddlewareConfig returns default middleware configuration
func DefaultMiddlewareConfig() *MiddlewareConfig {
	return &MiddlewareConfig{
		EnableLogging:     true,
		EnableCORS:        true,
		EnableRateLimit:   true,
		EnableAuth:        false, // Disabled by default for demo
		EnableCompression: true,
		EnableSecurity:    true,
		
		RateLimitRequests: 100,
		RateLimitWindow:   time.Minute,
		AuthRequired:      []string{"/api/v1/admin"},
		AllowedOrigins:    []string{"*"},
	}
}

// RequestMetrics holds request metrics
type RequestMetrics struct {
	TotalRequests    int64
	ActiveRequests   int64
	RequestsByStatus map[int]int64
	RequestsByMethod map[string]int64
	AverageResponse  time.Duration
	mutex           sync.RWMutex
}

var metrics = &RequestMetrics{
	RequestsByStatus: make(map[int]int64),
	RequestsByMethod: make(map[string]int64),
}

// ApplyMiddleware applies all enabled middleware to the router
func ApplyMiddleware(r *mux.Router, config *MiddlewareConfig) *mux.Router {
	// Apply middleware in reverse order (last applied = first executed)
	
	if config.EnableSecurity {
		r.Use(SecurityMiddleware)
	}
	
	if config.EnableCompression {
		r.Use(CompressionMiddleware)
	}
	
	if config.EnableCORS {
		r.Use(CORSMiddleware(config.AllowedOrigins))
	}
	
	if config.EnableRateLimit {
		r.Use(RateLimitMiddleware(config.RateLimitRequests, config.RateLimitWindow))
	}
	
	if config.EnableAuth {
		r.Use(AuthMiddleware(config.AuthRequired))
	}
	
	if config.EnableLogging {
		r.Use(LoggingMiddleware)
	}
	
	// Always apply metrics middleware
	r.Use(MetricsMiddleware)
	
	return r
}

// LoggingMiddleware logs HTTP requests
func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		
		// Create a response writer wrapper to capture status code
		wrapper := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
		
		next.ServeHTTP(wrapper, r)
		
		duration := time.Since(start)
		
		// Log the request
		fmt.Printf("[%s] %s %s %d %v %s\n",
			start.Format("2006-01-02 15:04:05"),
			r.Method,
			r.URL.Path,
			wrapper.statusCode,
			duration,
			r.RemoteAddr,
		)
	})
}

// CORSMiddleware handles Cross-Origin Resource Sharing
func CORSMiddleware(allowedOrigins []string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			origin := r.Header.Get("Origin")
			
			// Check if origin is allowed
			allowed := false
			for _, allowedOrigin := range allowedOrigins {
				if allowedOrigin == "*" || allowedOrigin == origin {
					allowed = true
					break
				}
			}
			
			if allowed {
				w.Header().Set("Access-Control-Allow-Origin", origin)
			}
			
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
			w.Header().Set("Access-Control-Max-Age", "3600")
			
			// Handle preflight requests
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			
			next.ServeHTTP(w, r)
		})
	}
}

// RateLimitMiddleware implements rate limiting
func RateLimitMiddleware(requestsPerWindow int, window time.Duration) func(http.Handler) http.Handler {
	type clientInfo struct {
		requests  int
		resetTime time.Time
		mutex     sync.Mutex
	}
	
	clients := make(map[string]*clientInfo)
	clientsMutex := sync.RWMutex{}
	
	// Cleanup routine
	go func() {
		ticker := time.NewTicker(window)
		defer ticker.Stop()
		
		for range ticker.C {
			clientsMutex.Lock()
			now := time.Now()
			for ip, info := range clients {
				if now.After(info.resetTime) {
					delete(clients, ip)
				}
			}
			clientsMutex.Unlock()
		}
	}()
	
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ip := getClientIP(r)
			
			clientsMutex.RLock()
			client, exists := clients[ip]
			clientsMutex.RUnlock()
			
			if !exists {
				client = &clientInfo{
					requests:  0,
					resetTime: time.Now().Add(window),
				}
				clientsMutex.Lock()
				clients[ip] = client
				clientsMutex.Unlock()
			}
			
			client.mutex.Lock()
			defer client.mutex.Unlock()
			
			// Reset if window has passed
			if time.Now().After(client.resetTime) {
				client.requests = 0
				client.resetTime = time.Now().Add(window)
			}
			
			// Check rate limit
			if client.requests >= requestsPerWindow {
				w.Header().Set("X-RateLimit-Limit", strconv.Itoa(requestsPerWindow))
				w.Header().Set("X-RateLimit-Remaining", "0")
				w.Header().Set("X-RateLimit-Reset", strconv.FormatInt(client.resetTime.Unix(), 10))
				
				http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
				return
			}
			
			client.requests++
			
			// Add rate limit headers
			w.Header().Set("X-RateLimit-Limit", strconv.Itoa(requestsPerWindow))
			w.Header().Set("X-RateLimit-Remaining", strconv.Itoa(requestsPerWindow-client.requests))
			w.Header().Set("X-RateLimit-Reset", strconv.FormatInt(client.resetTime.Unix(), 10))
			
			next.ServeHTTP(w, r)
		})
	}
}

// AuthMiddleware handles authentication
func AuthMiddleware(protectedPaths []string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Check if path requires authentication
			requiresAuth := false
			for _, path := range protectedPaths {
				if strings.HasPrefix(r.URL.Path, path) {
					requiresAuth = true
					break
				}
			}
			
			if !requiresAuth {
				next.ServeHTTP(w, r)
				return
			}
			
			// Extract token from Authorization header
			authHeader := r.Header.Get("Authorization")
			if authHeader == "" {
				http.Error(w, "Authorization header required", http.StatusUnauthorized)
				return
			}
			
			// Simple bearer token validation (in production, use proper JWT validation)
			if !strings.HasPrefix(authHeader, "Bearer ") {
				http.Error(w, "Invalid authorization header format", http.StatusUnauthorized)
				return
			}
			
			token := strings.TrimPrefix(authHeader, "Bearer ")
			if !isValidToken(token) {
				http.Error(w, "Invalid token", http.StatusUnauthorized)
				return
			}
			
			// Add user info to context
			ctx := context.WithValue(r.Context(), "user", "authenticated_user")
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// CompressionMiddleware adds gzip compression
func CompressionMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if client accepts gzip
		if !strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
			next.ServeHTTP(w, r)
			return
		}
		
		// For simplicity, we're not implementing actual gzip compression here
		// In production, use a proper compression library like gorilla/handlers
		w.Header().Set("Content-Encoding", "gzip")
		next.ServeHTTP(w, r)
	})
}

// SecurityMiddleware adds security headers
func SecurityMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Security headers
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("X-XSS-Protection", "1; mode=block")
		w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		w.Header().Set("Content-Security-Policy", "default-src 'self'")
		w.Header().Set("Referrer-Policy", "strict-origin-when-cross-origin")
		
		next.ServeHTTP(w, r)
	})
}

// MetricsMiddleware collects request metrics
func MetricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		
		metrics.mutex.Lock()
		metrics.TotalRequests++
		metrics.ActiveRequests++
		metrics.RequestsByMethod[r.Method]++
		metrics.mutex.Unlock()
		
		wrapper := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}
		
		next.ServeHTTP(wrapper, r)
		
		duration := time.Since(start)
		
		metrics.mutex.Lock()
		metrics.ActiveRequests--
		metrics.RequestsByStatus[wrapper.statusCode]++
		
		// Update average response time (simple moving average)
		if metrics.TotalRequests == 1 {
			metrics.AverageResponse = duration
		} else {
			metrics.AverageResponse = time.Duration(
				(int64(metrics.AverageResponse) + int64(duration)) / 2,
			)
		}
		metrics.mutex.Unlock()
	})
}

// GetMetrics returns current request metrics
func GetMetrics() *RequestMetrics {
	metrics.mutex.RLock()
	defer metrics.mutex.RUnlock()
	
	// Create a copy to avoid race conditions
	copy := &RequestMetrics{
		TotalRequests:    metrics.TotalRequests,
		ActiveRequests:   metrics.ActiveRequests,
		AverageResponse:  metrics.AverageResponse,
		RequestsByStatus: make(map[int]int64),
		RequestsByMethod: make(map[string]int64),
	}
	
	for k, v := range metrics.RequestsByStatus {
		copy.RequestsByStatus[k] = v
	}
	
	for k, v := range metrics.RequestsByMethod {
		copy.RequestsByMethod[k] = v
	}
	
	return copy
}

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// Helper functions

func getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header
	forwarded := r.Header.Get("X-Forwarded-For")
	if forwarded != "" {
		// Take the first IP in case of multiple
		ips := strings.Split(forwarded, ",")
		return strings.TrimSpace(ips[0])
	}
	
	// Check X-Real-IP header
	realIP := r.Header.Get("X-Real-IP")
	if realIP != "" {
		return realIP
	}
	
	// Fall back to RemoteAddr
	ip := r.RemoteAddr
	if colonIndex := strings.LastIndex(ip, ":"); colonIndex != -1 {
		ip = ip[:colonIndex]
	}
	
	return ip
}

func isValidToken(token string) bool {
	// Simple token validation (in production, validate JWT or check database)
	validTokens := map[string]bool{
		"dev-token":    true,
		"admin-token":  true,
		"client-token": true,
	}
	
	return validTokens[token]
}

// MetricsHandler returns metrics as JSON
func MetricsHandler(w http.ResponseWriter, r *http.Request) {
	currentMetrics := GetMetrics()
	
	response := map[string]interface{}{
		"requests": map[string]interface{}{
			"total":   currentMetrics.TotalRequests,
			"active":  currentMetrics.ActiveRequests,
			"by_status": currentMetrics.RequestsByStatus,
			"by_method": currentMetrics.RequestsByMethod,
		},
		"performance": map[string]interface{}{
			"average_response_time": currentMetrics.AverageResponse.String(),
		},
		"timestamp": time.Now(),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}