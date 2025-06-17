package middleware

import (
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// CORSConfig contains CORS configuration
type CORSConfig struct {
	Enabled          bool          `json:"enabled" yaml:"enabled"`
	AllowedOrigins   []string      `json:"allowed_origins" yaml:"allowed_origins"`
	AllowedMethods   []string      `json:"allowed_methods" yaml:"allowed_methods"`
	AllowedHeaders   []string      `json:"allowed_headers" yaml:"allowed_headers"`
	ExposedHeaders   []string      `json:"exposed_headers" yaml:"exposed_headers"`
	AllowCredentials bool          `json:"allow_credentials" yaml:"allow_credentials"`
	MaxAge           time.Duration `json:"max_age" yaml:"max_age"`
	AllowAllOrigins  bool          `json:"allow_all_origins" yaml:"allow_all_origins"`
}

// CORSMiddleware provides CORS middleware
type CORSMiddleware struct {
	config *CORSConfig
	logger *logrus.Logger
}

// NewCORSMiddleware creates a new CORS middleware
func NewCORSMiddleware(config *CORSConfig, logger *logrus.Logger) *CORSMiddleware {
	if logger == nil {
		logger = logrus.New()
	}

	if config == nil {
		config = &CORSConfig{
			Enabled: true,
			AllowedOrigins: []string{"http://localhost:3000", "http://localhost:8080"},
			AllowedMethods: []string{"GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"},
			AllowedHeaders: []string{
				"Accept",
				"Authorization",
				"Content-Type",
				"X-CSRF-Token",
				"X-API-Key",
				"X-Request-ID",
				"X-Requested-With",
			},
			ExposedHeaders: []string{
				"X-Request-ID",
				"X-Response-Time",
				"X-Total-Count",
			},
			AllowCredentials: true,
			MaxAge:           24 * time.Hour,
		}
	}

	// Set defaults
	if len(config.AllowedMethods) == 0 {
		config.AllowedMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	}

	if config.MaxAge == 0 {
		config.MaxAge = 24 * time.Hour
	}

	return &CORSMiddleware{
		config: config,
		logger: logger,
	}
}

// Middleware returns the HTTP middleware function
func (cm *CORSMiddleware) Middleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Skip CORS if disabled
			if !cm.config.Enabled {
				next.ServeHTTP(w, r)
				return
			}

			origin := r.Header.Get("Origin")
			
			// Set CORS headers
			cm.setCORSHeaders(w, r, origin)

			// Handle preflight requests
			if r.Method == "OPTIONS" {
				cm.handlePreflight(w, r)
				return
			}

			// Continue with the request
			next.ServeHTTP(w, r)
		})
	}
}

// setCORSHeaders sets the appropriate CORS headers
func (cm *CORSMiddleware) setCORSHeaders(w http.ResponseWriter, r *http.Request, origin string) {
	headers := w.Header()

	// Set Access-Control-Allow-Origin
	if cm.config.AllowAllOrigins {
		headers.Set("Access-Control-Allow-Origin", "*")
	} else if cm.isOriginAllowed(origin) {
		headers.Set("Access-Control-Allow-Origin", origin)
		headers.Set("Vary", "Origin")
	}

	// Set Access-Control-Allow-Credentials
	if cm.config.AllowCredentials && !cm.config.AllowAllOrigins {
		headers.Set("Access-Control-Allow-Credentials", "true")
	}

	// Set Access-Control-Expose-Headers
	if len(cm.config.ExposedHeaders) > 0 {
		headers.Set("Access-Control-Expose-Headers", strings.Join(cm.config.ExposedHeaders, ", "))
	}
}

// handlePreflight handles CORS preflight requests
func (cm *CORSMiddleware) handlePreflight(w http.ResponseWriter, r *http.Request) {
	headers := w.Header()
	origin := r.Header.Get("Origin")

	// Validate origin for preflight
	if !cm.config.AllowAllOrigins && !cm.isOriginAllowed(origin) {
		cm.logger.WithFields(logrus.Fields{
			"origin": origin,
			"path":   r.URL.Path,
		}).Warn("CORS preflight request from disallowed origin")
		w.WriteHeader(http.StatusForbidden)
		return
	}

	// Set Access-Control-Allow-Methods
	if len(cm.config.AllowedMethods) > 0 {
		headers.Set("Access-Control-Allow-Methods", strings.Join(cm.config.AllowedMethods, ", "))
	}

	// Set Access-Control-Allow-Headers
	requestHeaders := r.Header.Get("Access-Control-Request-Headers")
	if requestHeaders != "" {
		if cm.areHeadersAllowed(requestHeaders) {
			headers.Set("Access-Control-Allow-Headers", requestHeaders)
		} else {
			// Fallback to configured headers
			if len(cm.config.AllowedHeaders) > 0 {
				headers.Set("Access-Control-Allow-Headers", strings.Join(cm.config.AllowedHeaders, ", "))
			}
		}
	} else if len(cm.config.AllowedHeaders) > 0 {
		headers.Set("Access-Control-Allow-Headers", strings.Join(cm.config.AllowedHeaders, ", "))
	}

	// Set Access-Control-Max-Age
	if cm.config.MaxAge > 0 {
		headers.Set("Access-Control-Max-Age", strconv.Itoa(int(cm.config.MaxAge.Seconds())))
	}

	// Log preflight request
	cm.logger.WithFields(logrus.Fields{
		"origin":          origin,
		"method":          r.Header.Get("Access-Control-Request-Method"),
		"headers":         requestHeaders,
		"path":            r.URL.Path,
	}).Debug("CORS preflight request handled")

	w.WriteHeader(http.StatusNoContent)
}

// isOriginAllowed checks if the origin is allowed
func (cm *CORSMiddleware) isOriginAllowed(origin string) bool {
	if origin == "" {
		return false
	}

	if cm.config.AllowAllOrigins {
		return true
	}

	for _, allowedOrigin := range cm.config.AllowedOrigins {
		if cm.matchOrigin(allowedOrigin, origin) {
			return true
		}
	}

	return false
}

// matchOrigin checks if origin matches the allowed origin pattern
func (cm *CORSMiddleware) matchOrigin(allowedOrigin, origin string) bool {
	// Exact match
	if allowedOrigin == origin {
		return true
	}

	// Wildcard subdomain match (e.g., *.example.com)
	if strings.HasPrefix(allowedOrigin, "*.") {
		domain := allowedOrigin[2:]
		return strings.HasSuffix(origin, "."+domain) || origin == domain
	}

	return false
}

// areHeadersAllowed checks if all requested headers are allowed
func (cm *CORSMiddleware) areHeadersAllowed(requestHeaders string) bool {
	if len(cm.config.AllowedHeaders) == 0 {
		return true // Allow all if not specified
	}

	// Create map for faster lookup
	allowedMap := make(map[string]bool)
	for _, header := range cm.config.AllowedHeaders {
		allowedMap[strings.ToLower(header)] = true
	}

	// Check each requested header
	headers := strings.Split(requestHeaders, ",")
	for _, header := range headers {
		header = strings.TrimSpace(strings.ToLower(header))
		if !allowedMap[header] {
			return false
		}
	}

	return true
}

// SetOrigin dynamically sets allowed origin (useful for single-page apps)
func (cm *CORSMiddleware) SetOrigin(w http.ResponseWriter, origin string) {
	if cm.isOriginAllowed(origin) {
		w.Header().Set("Access-Control-Allow-Origin", origin)
		if cm.config.AllowCredentials {
			w.Header().Set("Access-Control-Allow-Credentials", "true")
		}
	}
}

// AddAllowedOrigin adds a new allowed origin
func (cm *CORSMiddleware) AddAllowedOrigin(origin string) {
	cm.config.AllowedOrigins = append(cm.config.AllowedOrigins, origin)
}

// RemoveAllowedOrigin removes an allowed origin
func (cm *CORSMiddleware) RemoveAllowedOrigin(origin string) {
	for i, allowedOrigin := range cm.config.AllowedOrigins {
		if allowedOrigin == origin {
			cm.config.AllowedOrigins = append(
				cm.config.AllowedOrigins[:i],
				cm.config.AllowedOrigins[i+1:]...,
			)
			break
		}
	}
}

// GetConfig returns the current CORS configuration
func (cm *CORSMiddleware) GetConfig() *CORSConfig {
	return cm.config
}

// UpdateConfig updates the CORS configuration
func (cm *CORSMiddleware) UpdateConfig(config *CORSConfig) {
	cm.config = config
}