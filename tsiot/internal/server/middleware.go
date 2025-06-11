package server

import (
	"context"
	"fmt"
	"net/http"
	"runtime/debug"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/constants"
)

// loggingMiddleware logs HTTP requests
func (s *Server) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Wrap the response writer to capture status code
		wrappedWriter := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		// Process the request
		next.ServeHTTP(wrappedWriter, r)

		// Log the request
		duration := time.Since(start)
		s.logger.WithFields(logrus.Fields{
			"method":        r.Method,
			"path":          r.URL.Path,
			"query":         r.URL.RawQuery,
			"status":        wrappedWriter.statusCode,
			"duration_ms":   duration.Milliseconds(),
			"remote_addr":   getClientIP(r),
			"user_agent":    r.UserAgent(),
			"request_id":    getRequestID(r),
			"content_length": r.ContentLength,
		}).Info("HTTP request")
	})
}

// recoveryMiddleware recovers from panics and returns 500 error
func (s *Server) recoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				// Log the panic
				s.logger.WithFields(logrus.Fields{
					"error":      err,
					"path":       r.URL.Path,
					"method":     r.Method,
					"request_id": getRequestID(r),
					"stack":      string(debug.Stack()),
				}).Error("Panic recovered")

				// Return 500 error
				w.Header().Set(constants.HeaderContentType, constants.ContentTypeJSON)
				w.WriteHeader(http.StatusInternalServerError)
				fmt.Fprintf(w, `{"error": {"code": "INTERNAL_ERROR", "message": "Internal server error"}}`)
			}
		}()

		next.ServeHTTP(w, r)
	})
}

// requestIDMiddleware adds a unique request ID to each request
func (s *Server) requestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if request ID already exists in headers
		requestID := r.Header.Get(constants.HeaderRequestID)
		if requestID == "" {
			// Generate new request ID
			requestID = uuid.New().String()
		}

		// Add request ID to response headers
		w.Header().Set(constants.HeaderRequestID, requestID)

		// Add request ID to context
		ctx := context.WithValue(r.Context(), "request_id", requestID)
		r = r.WithContext(ctx)

		next.ServeHTTP(w, r)
	})
}

// corsMiddleware handles CORS headers
func (s *Server) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		
		// Set CORS headers
		w.Header().Set("Access-Control-Allow-Origin", "*") // In production, be more specific
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Accept, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, X-Request-ID")
		w.Header().Set("Access-Control-Expose-Headers", "X-Request-ID, X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset")
		w.Header().Set("Access-Control-Allow-Credentials", "true")
		w.Header().Set("Access-Control-Max-Age", "3600")

		// Handle preflight request
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// requestSizeLimitMiddleware limits the size of request bodies
func (s *Server) requestSizeLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.ContentLength > s.config.MaxRequestSize {
			s.logger.WithFields(logrus.Fields{
				"content_length": r.ContentLength,
				"max_size":       s.config.MaxRequestSize,
				"path":           r.URL.Path,
				"request_id":     getRequestID(r),
			}).Warn("Request body too large")

			w.Header().Set(constants.HeaderContentType, constants.ContentTypeJSON)
			w.WriteHeader(http.StatusRequestEntityTooLarge)
			fmt.Fprintf(w, `{"error": {"code": "REQUEST_TOO_LARGE", "message": "Request body too large"}}`)
			return
		}

		// Limit the reader to prevent abuse
		r.Body = http.MaxBytesReader(w, r.Body, s.config.MaxRequestSize)

		next.ServeHTTP(w, r)
	})
}

// securityHeadersMiddleware adds security headers
func (s *Server) securityHeadersMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Security headers
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("X-XSS-Protection", "1; mode=block")
		w.Header().Set("Referrer-Policy", "strict-origin-when-cross-origin")
		w.Header().Set("Content-Security-Policy", "default-src 'self'")

		// Remove server identification
		w.Header().Set("Server", "")

		next.ServeHTTP(w, r)
	})
}

// rateLimitMiddleware implements basic rate limiting
func (s *Server) rateLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simple rate limiting implementation
		clientIP := getClientIP(r)
		
		// In a real implementation, you would use a more sophisticated
		// rate limiting mechanism with Redis or in-memory store
		// For now, we'll just add the headers and proceed
		
		w.Header().Set(constants.HeaderRateLimit, strconv.Itoa(constants.DefaultRateLimit))
		w.Header().Set(constants.HeaderRateLimitRemaining, strconv.Itoa(constants.DefaultRateLimit-1))
		w.Header().Set(constants.HeaderRateLimitReset, strconv.FormatInt(time.Now().Add(time.Minute).Unix(), 10))

		s.logger.WithFields(logrus.Fields{
			"client_ip":  clientIP,
			"path":       r.URL.Path,
			"request_id": getRequestID(r),
		}).Debug("Rate limit check")

		next.ServeHTTP(w, r)
	})
}

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
	written    bool
}

// WriteHeader captures the status code
func (rw *responseWriter) WriteHeader(statusCode int) {
	if !rw.written {
		rw.statusCode = statusCode
		rw.written = true
	}
	rw.ResponseWriter.WriteHeader(statusCode)
}

// Write ensures WriteHeader is called
func (rw *responseWriter) Write(data []byte) (int, error) {
	if !rw.written {
		rw.WriteHeader(http.StatusOK)
	}
	return rw.ResponseWriter.Write(data)
}

// getClientIP extracts the client IP address from the request
func getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header
	xff := r.Header.Get(constants.HeaderForwardedFor)
	if xff != "" {
		// X-Forwarded-For can contain multiple IPs, take the first one
		ips := strings.Split(xff, ",")
		if len(ips) > 0 {
			return strings.TrimSpace(ips[0])
		}
	}

	// Check X-Real-IP header
	realIP := r.Header.Get(constants.HeaderRealIP)
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

// getRequestID extracts the request ID from the context
func getRequestID(r *http.Request) string {
	if requestID, ok := r.Context().Value("request_id").(string); ok {
		return requestID
	}
	return ""
}

// timeoutMiddleware adds request timeout
func (s *Server) timeoutMiddleware(timeout time.Duration) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ctx, cancel := context.WithTimeout(r.Context(), timeout)
			defer cancel()

			r = r.WithContext(ctx)

			done := make(chan struct{})
			go func() {
				defer close(done)
				next.ServeHTTP(w, r)
			}()

			select {
			case <-done:
				// Request completed normally
			case <-ctx.Done():
				// Request timed out
				s.logger.WithFields(logrus.Fields{
					"path":       r.URL.Path,
					"timeout":    timeout,
					"request_id": getRequestID(r),
				}).Warn("Request timeout")

				w.Header().Set(constants.HeaderContentType, constants.ContentTypeJSON)
				w.WriteHeader(http.StatusRequestTimeout)
				fmt.Fprintf(w, `{"error": {"code": "REQUEST_TIMEOUT", "message": "Request timeout"}}`)
			}
		})
	}
}

// authMiddleware handles authentication (placeholder implementation)
func (s *Server) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Skip authentication for health checks and public endpoints
		if isPublicEndpoint(r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}

		// Check for Authorization header
		authHeader := r.Header.Get(constants.HeaderAuthorization)
		if authHeader == "" {
			w.Header().Set(constants.HeaderContentType, constants.ContentTypeJSON)
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprintf(w, `{"error": {"code": "UNAUTHORIZED", "message": "Missing authorization header"}}`)
			return
		}

		// In a real implementation, you would validate the token here
		// For now, we'll just check if it starts with "Bearer "
		if !strings.HasPrefix(authHeader, "Bearer ") {
			w.Header().Set(constants.HeaderContentType, constants.ContentTypeJSON)
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprintf(w, `{"error": {"code": "INVALID_TOKEN", "message": "Invalid authorization header format"}}`)
			return
		}

		// Extract and validate token (placeholder)
		token := strings.TrimPrefix(authHeader, "Bearer ")
		if token == "" {
			w.Header().Set(constants.HeaderContentType, constants.ContentTypeJSON)
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprintf(w, `{"error": {"code": "INVALID_TOKEN", "message": "Empty token"}}`)
			return
		}

		// Add user info to context (placeholder)
		ctx := context.WithValue(r.Context(), "user_id", "default-user")
		r = r.WithContext(ctx)

		next.ServeHTTP(w, r)
	})
}

// isPublicEndpoint checks if an endpoint is public (doesn't require authentication)
func isPublicEndpoint(path string) bool {
	publicPaths := []string{
		"/health",
		"/health/ready",
		"/health/live",
		"/version",
		"/metrics",
		"/docs/",
	}

	for _, publicPath := range publicPaths {
		if strings.HasPrefix(path, publicPath) {
			return true
		}
	}

	return false
}