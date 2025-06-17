package middleware

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/sirupsen/logrus"

	"github.com/inferloop/tsiot/pkg/errors"
)

// AuthConfig contains authentication configuration
type AuthConfig struct {
	Enabled          bool          `json:"enabled" yaml:"enabled"`
	JWTSecret        string        `json:"jwt_secret" yaml:"jwt_secret"`
	JWTExpiration    time.Duration `json:"jwt_expiration" yaml:"jwt_expiration"`
	APIKeyEnabled    bool          `json:"api_key_enabled" yaml:"api_key_enabled"`
	APIKeys          []string      `json:"api_keys" yaml:"api_keys"`
	AllowAnonymous   bool          `json:"allow_anonymous" yaml:"allow_anonymous"`
	RequiredScopes   []string      `json:"required_scopes" yaml:"required_scopes"`
	ExemptPaths      []string      `json:"exempt_paths" yaml:"exempt_paths"`
}

// User represents an authenticated user
type User struct {
	ID       string            `json:"id"`
	Username string            `json:"username"`
	Email    string            `json:"email"`
	Roles    []string          `json:"roles"`
	Scopes   []string          `json:"scopes"`
	Metadata map[string]string `json:"metadata"`
}

// Claims represents JWT claims
type Claims struct {
	User   User     `json:"user"`
	Scopes []string `json:"scopes"`
	jwt.RegisteredClaims
}

// AuthMiddleware provides authentication middleware
type AuthMiddleware struct {
	config *AuthConfig
	logger *logrus.Logger
}

// NewAuthMiddleware creates a new authentication middleware
func NewAuthMiddleware(config *AuthConfig, logger *logrus.Logger) *AuthMiddleware {
	if logger == nil {
		logger = logrus.New()
	}

	if config == nil {
		config = &AuthConfig{
			Enabled:        false,
			AllowAnonymous: true,
		}
	}

	// Set defaults
	if config.JWTExpiration == 0 {
		config.JWTExpiration = 24 * time.Hour
	}

	return &AuthMiddleware{
		config: config,
		logger: logger,
	}
}

// Middleware returns the HTTP middleware function
func (am *AuthMiddleware) Middleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Skip authentication if disabled
			if !am.config.Enabled {
				next.ServeHTTP(w, r)
				return
			}

			// Check if path is exempt
			if am.isExemptPath(r.URL.Path) {
				next.ServeHTTP(w, r)
				return
			}

			// Attempt authentication
			user, err := am.authenticate(r)
			if err != nil {
				am.handleAuthError(w, r, err)
				return
			}

			// Check authorization
			if !am.authorize(user, r) {
				am.handleAuthError(w, r, errors.NewValidationError("INSUFFICIENT_PERMISSIONS", "Insufficient permissions"))
				return
			}

			// Add user to context
			ctx := context.WithValue(r.Context(), "user", user)
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

// authenticate attempts to authenticate the request
func (am *AuthMiddleware) authenticate(r *http.Request) (*User, error) {
	// Try JWT authentication first
	if user, err := am.authenticateJWT(r); err == nil {
		return user, nil
	}

	// Try API key authentication
	if am.config.APIKeyEnabled {
		if user, err := am.authenticateAPIKey(r); err == nil {
			return user, nil
		}
	}

	// Allow anonymous if configured
	if am.config.AllowAnonymous {
		return &User{
			ID:       "anonymous",
			Username: "anonymous",
			Roles:    []string{"anonymous"},
			Scopes:   []string{"read"},
		}, nil
	}

	return nil, errors.NewValidationError("AUTHENTICATION_FAILED", "Authentication required")
}

// authenticateJWT authenticates using JWT token
func (am *AuthMiddleware) authenticateJWT(r *http.Request) (*User, error) {
	// Extract token from Authorization header
	authHeader := r.Header.Get("Authorization")
	if authHeader == "" {
		return nil, errors.NewValidationError("NO_TOKEN", "No authorization token provided")
	}

	// Check Bearer prefix
	if !strings.HasPrefix(authHeader, "Bearer ") {
		return nil, errors.NewValidationError("INVALID_TOKEN_FORMAT", "Token must be in Bearer format")
	}

	tokenString := strings.TrimPrefix(authHeader, "Bearer ")

	// Parse and validate token
	token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
		// Validate signing method
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return []byte(am.config.JWTSecret), nil
	})

	if err != nil {
		return nil, errors.WrapError(err, errors.ErrorTypeValidation, "TOKEN_PARSE_FAILED", "Failed to parse JWT token")
	}

	// Extract claims
	if claims, ok := token.Claims.(*Claims); ok && token.Valid {
		// Check expiration
		if claims.ExpiresAt != nil && claims.ExpiresAt.Time.Before(time.Now()) {
			return nil, errors.NewValidationError("TOKEN_EXPIRED", "JWT token has expired")
		}

		return &claims.User, nil
	}

	return nil, errors.NewValidationError("INVALID_TOKEN", "Invalid JWT token")
}

// authenticateAPIKey authenticates using API key
func (am *AuthMiddleware) authenticateAPIKey(r *http.Request) (*User, error) {
	// Try X-API-Key header
	apiKey := r.Header.Get("X-API-Key")
	if apiKey == "" {
		// Try query parameter
		apiKey = r.URL.Query().Get("api_key")
	}

	if apiKey == "" {
		return nil, errors.NewValidationError("NO_API_KEY", "No API key provided")
	}

	// Validate API key
	if !am.isValidAPIKey(apiKey) {
		return nil, errors.NewValidationError("INVALID_API_KEY", "Invalid API key")
	}

	// Return API key user
	return &User{
		ID:       "apikey_user",
		Username: "apikey_user",
		Roles:    []string{"api_user"},
		Scopes:   []string{"read", "write"},
		Metadata: map[string]string{
			"api_key": apiKey[:8] + "...", // Partial key for logging
		},
	}, nil
}

// authorize checks if user has required permissions
func (am *AuthMiddleware) authorize(user *User, r *http.Request) bool {
	// Always allow if no scopes required
	if len(am.config.RequiredScopes) == 0 {
		return true
	}

	// Check if user has any required scope
	userScopes := make(map[string]bool)
	for _, scope := range user.Scopes {
		userScopes[scope] = true
	}

	for _, requiredScope := range am.config.RequiredScopes {
		if userScopes[requiredScope] {
			return true
		}
	}

	// Check method-based permissions
	method := strings.ToUpper(r.Method)
	switch method {
	case "GET", "HEAD", "OPTIONS":
		return userScopes["read"]
	case "POST", "PUT", "PATCH", "DELETE":
		return userScopes["write"]
	default:
		return false
	}
}

// isExemptPath checks if the path is exempt from authentication
func (am *AuthMiddleware) isExemptPath(path string) bool {
	for _, exemptPath := range am.config.ExemptPaths {
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

// isValidAPIKey checks if the API key is valid
func (am *AuthMiddleware) isValidAPIKey(apiKey string) bool {
	for _, validKey := range am.config.APIKeys {
		if apiKey == validKey {
			return true
		}
	}
	return false
}

// handleAuthError handles authentication errors
func (am *AuthMiddleware) handleAuthError(w http.ResponseWriter, r *http.Request, err error) {
	am.logger.WithFields(logrus.Fields{
		"error":  err.Error(),
		"path":   r.URL.Path,
		"method": r.Method,
		"ip":     r.RemoteAddr,
	}).Warn("Authentication failed")

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusUnauthorized)
	
	response := map[string]interface{}{
		"error": map[string]interface{}{
			"code":    "AUTHENTICATION_FAILED",
			"message": "Authentication failed",
			"details": err.Error(),
		},
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	}

	if err := writeJSON(w, response); err != nil {
		am.logger.WithError(err).Error("Failed to write error response")
	}
}

// GenerateJWT generates a JWT token for a user
func (am *AuthMiddleware) GenerateJWT(user *User) (string, error) {
	if am.config.JWTSecret == "" {
		return "", errors.NewValidationError("NO_JWT_SECRET", "JWT secret not configured")
	}

	claims := &Claims{
		User:   *user,
		Scopes: user.Scopes,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(am.config.JWTExpiration)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
			Issuer:    "tsiot",
			Subject:   user.ID,
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(am.config.JWTSecret))
}

// GetUserFromContext extracts user from request context
func GetUserFromContext(ctx context.Context) (*User, bool) {
	user, ok := ctx.Value("user").(*User)
	return user, ok
}

// RequireScope creates middleware that requires specific scopes
func (am *AuthMiddleware) RequireScope(requiredScopes ...string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			user, ok := GetUserFromContext(r.Context())
			if !ok {
				am.handleAuthError(w, r, errors.NewValidationError("NO_USER", "No user in context"))
				return
			}

			// Check if user has required scopes
			userScopes := make(map[string]bool)
			for _, scope := range user.Scopes {
				userScopes[scope] = true
			}

			for _, requiredScope := range requiredScopes {
				if !userScopes[requiredScope] {
					am.handleAuthError(w, r, errors.NewValidationError("INSUFFICIENT_SCOPE", 
						fmt.Sprintf("Scope '%s' required", requiredScope)))
					return
				}
			}

			next.ServeHTTP(w, r)
		})
	}
}

// RequireRole creates middleware that requires specific roles
func (am *AuthMiddleware) RequireRole(requiredRoles ...string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			user, ok := GetUserFromContext(r.Context())
			if !ok {
				am.handleAuthError(w, r, errors.NewValidationError("NO_USER", "No user in context"))
				return
			}

			// Check if user has required roles
			userRoles := make(map[string]bool)
			for _, role := range user.Roles {
				userRoles[role] = true
			}

			for _, requiredRole := range requiredRoles {
				if !userRoles[requiredRole] {
					am.handleAuthError(w, r, errors.NewValidationError("INSUFFICIENT_ROLE", 
						fmt.Sprintf("Role '%s' required", requiredRole)))
					return
				}
			}

			next.ServeHTTP(w, r)
		})
	}
}