package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/gorilla/mux"
	"github.com/rs/cors"
	"golang.org/x/time/rate"
)

// APIGateway manages microservices routing, authentication, and cross-cutting concerns
type APIGateway struct {
	logger           *logrus.Logger
	config           *GatewayConfig
	router           *mux.Router
	serviceRegistry  *ServiceRegistry
	rateLimiter      *RateLimiter
	circuitBreaker   *CircuitBreaker
	loadBalancer     *LoadBalancer
	authManager      *AuthManager
	middleware       *MiddlewareChain
	metricsCollector *GatewayMetrics
	httpServer       *http.Server
	mu               sync.RWMutex
	stopCh           chan struct{}
}

// GatewayConfig configures the API gateway
type GatewayConfig struct {
	Enabled             bool                    `json:"enabled"`
	Port                int                     `json:"port"`
	Host                string                  `json:"host"`
	TLSEnabled          bool                    `json:"tls_enabled"`
	CertFile            string                  `json:"cert_file"`
	KeyFile             string                  `json:"key_file"`
	Services            map[string]*ServiceConfig `json:"services"`
	RateLimiting        RateLimitingConfig      `json:"rate_limiting"`
	Authentication      AuthConfig              `json:"authentication"`
	CORS                CORSConfig              `json:"cors"`
	LoadBalancing       LoadBalancingConfig     `json:"load_balancing"`
	CircuitBreaker      CircuitBreakerConfig    `json:"circuit_breaker"`
	ServiceMesh         ServiceMeshConfig       `json:"service_mesh"`
	APIVersioning       APIVersioningConfig     `json:"api_versioning"`
	GraphQL             GraphQLConfig           `json:"graphql"`
	Metrics             MetricsConfig           `json:"metrics"`
	Logging             LoggingConfig           `json:"logging"`
	Timeouts            TimeoutConfig           `json:"timeouts"`
	Retry               RetryConfig             `json:"retry"`
	Compression         CompressionConfig       `json:"compression"`
	Security            SecurityConfig          `json:"security"`
	HealthCheck         HealthCheckConfig       `json:"health_check"`
}

// ServiceConfig configures individual microservices
type ServiceConfig struct {
	Name            string                 `json:"name"`
	Path            string                 `json:"path"`
	Hosts           []string               `json:"hosts"`
	Protocol        string                 `json:"protocol"` // http, https, grpc
	LoadBalancing   string                 `json:"load_balancing"` // round_robin, least_conn, ip_hash
	HealthCheckPath string                 `json:"health_check_path"`
	CircuitBreaker  bool                   `json:"circuit_breaker"`
	RateLimiting    *ServiceRateLimit      `json:"rate_limiting"`
	Authentication  *ServiceAuth           `json:"authentication"`
	Timeout         time.Duration          `json:"timeout"`
	Retries         int                    `json:"retries"`
	Metadata        map[string]interface{} `json:"metadata"`
	Versioning      ServiceVersioning      `json:"versioning"`
}

// ServiceVersioning configures API versioning for a service
type ServiceVersioning struct {
	Enabled         bool                    `json:"enabled"`
	Strategy        VersioningStrategy      `json:"strategy"` // header, path, query
	HeaderName      string                  `json:"header_name"`
	DefaultVersion  string                  `json:"default_version"`
	SupportedVersions []string              `json:"supported_versions"`
	VersionMapping  map[string]*ServiceConfig `json:"version_mapping"`
}

// VersioningStrategy defines versioning strategies
type VersioningStrategy string

const (
	VersioningHeader VersioningStrategy = "header"
	VersioningPath   VersioningStrategy = "path"
	VersioningQuery  VersioningStrategy = "query"
)

// ServiceRateLimit configures rate limiting for a service
type ServiceRateLimit struct {
	Enabled           bool          `json:"enabled"`
	RequestsPerSecond float64       `json:"requests_per_second"`
	BurstSize         int           `json:"burst_size"`
	KeyExtractor      string        `json:"key_extractor"` // ip, user, api_key
	WindowDuration    time.Duration `json:"window_duration"`
}

// ServiceAuth configures authentication for a service
type ServiceAuth struct {
	Enabled     bool     `json:"enabled"`
	Methods     []string `json:"methods"` // jwt, api_key, oauth2, basic
	Required    bool     `json:"required"`
	Permissions []string `json:"permissions"`
}

// RateLimitingConfig configures global rate limiting
type RateLimitingConfig struct {
	Enabled           bool                       `json:"enabled"`
	GlobalLimits      map[string]RateLimitRule   `json:"global_limits"`
	UserLimits        map[string]RateLimitRule   `json:"user_limits"`
	IPLimits          map[string]RateLimitRule   `json:"ip_limits"`
	HeaderBasedLimits map[string]RateLimitRule   `json:"header_based_limits"`
	CustomLimits      map[string]RateLimitRule   `json:"custom_limits"`
	Storage           string                     `json:"storage"` // memory, redis
	RedisConfig       *RedisConfig               `json:"redis_config"`
}

// RateLimitRule defines a rate limiting rule
type RateLimitRule struct {
	RequestsPerMinute int           `json:"requests_per_minute"`
	RequestsPerHour   int           `json:"requests_per_hour"`
	RequestsPerDay    int           `json:"requests_per_day"`
	BurstSize         int           `json:"burst_size"`
	WindowDuration    time.Duration `json:"window_duration"`
}

// RedisConfig configures Redis for rate limiting
type RedisConfig struct {
	Host     string `json:"host"`
	Port     int    `json:"port"`
	Password string `json:"password"`
	DB       int    `json:"db"`
}

// AuthConfig configures authentication
type AuthConfig struct {
	Enabled         bool                   `json:"enabled"`
	JWT             JWTConfig              `json:"jwt"`
	OAuth2          OAuth2Config           `json:"oauth2"`
	APIKey          APIKeyConfig           `json:"api_key"`
	Basic           BasicAuthConfig        `json:"basic"`
	Providers       []AuthProvider         `json:"providers"`
	SessionConfig   SessionConfig          `json:"session_config"`
	TokenValidation TokenValidationConfig  `json:"token_validation"`
}

// JWTConfig configures JWT authentication
type JWTConfig struct {
	Enabled        bool   `json:"enabled"`
	SecretKey      string `json:"secret_key"`
	Issuer         string `json:"issuer"`
	Audience       string `json:"audience"`
	ExpirationTime int    `json:"expiration_time"`
	Algorithm      string `json:"algorithm"`
}

// OAuth2Config configures OAuth2 authentication
type OAuth2Config struct {
	Enabled      bool   `json:"enabled"`
	ClientID     string `json:"client_id"`
	ClientSecret string `json:"client_secret"`
	RedirectURL  string `json:"redirect_url"`
	Scopes       []string `json:"scopes"`
	AuthURL      string `json:"auth_url"`
	TokenURL     string `json:"token_url"`
}

// APIKeyConfig configures API key authentication
type APIKeyConfig struct {
	Enabled    bool   `json:"enabled"`
	HeaderName string `json:"header_name"`
	QueryParam string `json:"query_param"`
	Validation string `json:"validation"` // database, redis, memory
}

// BasicAuthConfig configures basic authentication
type BasicAuthConfig struct {
	Enabled bool     `json:"enabled"`
	Realm   string   `json:"realm"`
	Users   []string `json:"users"`
}

// AuthProvider defines external auth providers
type AuthProvider struct {
	Name     string                 `json:"name"`
	Type     string                 `json:"type"` // oauth2, oidc, saml
	Config   map[string]interface{} `json:"config"`
	Priority int                    `json:"priority"`
}

// SessionConfig configures session management
type SessionConfig struct {
	Enabled        bool          `json:"enabled"`
	CookieName     string        `json:"cookie_name"`
	CookieDomain   string        `json:"cookie_domain"`
	CookiePath     string        `json:"cookie_path"`
	CookieSecure   bool          `json:"cookie_secure"`
	CookieHttpOnly bool          `json:"cookie_http_only"`
	MaxAge         time.Duration `json:"max_age"`
	Storage        string        `json:"storage"` // memory, redis, database
}

// TokenValidationConfig configures token validation
type TokenValidationConfig struct {
	CacheEnabled   bool          `json:"cache_enabled"`
	CacheTTL       time.Duration `json:"cache_ttl"`
	RemoteValidation bool        `json:"remote_validation"`
	ValidationURL  string        `json:"validation_url"`
}

// CORSConfig configures CORS
type CORSConfig struct {
	Enabled          bool     `json:"enabled"`
	AllowedOrigins   []string `json:"allowed_origins"`
	AllowedMethods   []string `json:"allowed_methods"`
	AllowedHeaders   []string `json:"allowed_headers"`
	ExposedHeaders   []string `json:"exposed_headers"`
	AllowCredentials bool     `json:"allow_credentials"`
	MaxAge           int      `json:"max_age"`
}

// LoadBalancingConfig configures load balancing
type LoadBalancingConfig struct {
	Strategy           LoadBalancingStrategy `json:"strategy"`
	HealthCheckEnabled bool                  `json:"health_check_enabled"`
	HealthCheckInterval time.Duration        `json:"health_check_interval"`
	HealthCheckTimeout time.Duration         `json:"health_check_timeout"`
	UnhealthyThreshold int                   `json:"unhealthy_threshold"`
	HealthyThreshold   int                   `json:"healthy_threshold"`
}

// LoadBalancingStrategy defines load balancing strategies
type LoadBalancingStrategy string

const (
	LoadBalancingRoundRobin    LoadBalancingStrategy = "round_robin"
	LoadBalancingLeastConn     LoadBalancingStrategy = "least_conn"
	LoadBalancingIPHash        LoadBalancingStrategy = "ip_hash"
	LoadBalancingWeightedRoundRobin LoadBalancingStrategy = "weighted_round_robin"
	LoadBalancingRandom        LoadBalancingStrategy = "random"
)

// CircuitBreakerConfig configures circuit breaker
type CircuitBreakerConfig struct {
	Enabled               bool          `json:"enabled"`
	FailureThreshold      int           `json:"failure_threshold"`
	RecoveryTimeout       time.Duration `json:"recovery_timeout"`
	RequestVolumeThreshold int          `json:"request_volume_threshold"`
	SleepWindow           time.Duration `json:"sleep_window"`
	ErrorPercentThreshold int           `json:"error_percent_threshold"`
}

// ServiceMeshConfig configures service mesh integration
type ServiceMeshConfig struct {
	Enabled     bool              `json:"enabled"`
	Type        string            `json:"type"` // istio, linkerd, consul_connect
	Namespace   string            `json:"namespace"`
	TLSMode     string            `json:"tls_mode"` // strict, permissive, disabled
	Observability ObservabilityConfig `json:"observability"`
	Security    MeshSecurityConfig   `json:"security"`
}

// ObservabilityConfig configures service mesh observability
type ObservabilityConfig struct {
	TracingEnabled  bool   `json:"tracing_enabled"`
	MetricsEnabled  bool   `json:"metrics_enabled"`
	LoggingEnabled  bool   `json:"logging_enabled"`
	JaegerEndpoint  string `json:"jaeger_endpoint"`
	PrometheusPort  int    `json:"prometheus_port"`
}

// MeshSecurityConfig configures service mesh security
type MeshSecurityConfig struct {
	mTLSEnabled     bool     `json:"mtls_enabled"`
	TrustDomain     string   `json:"trust_domain"`
	CertificateChain string  `json:"certificate_chain"`
	PrivateKey      string   `json:"private_key"`
	RootCA          string   `json:"root_ca"`
	AllowedServices []string `json:"allowed_services"`
}

// APIVersioningConfig configures API versioning
type APIVersioningConfig struct {
	Enabled          bool               `json:"enabled"`
	DefaultStrategy  VersioningStrategy `json:"default_strategy"`
	HeaderName       string             `json:"header_name"`
	DefaultVersion   string             `json:"default_version"`
	SupportedVersions []string          `json:"supported_versions"`
	DeprecationPolicy DeprecationPolicy `json:"deprecation_policy"`
}

// DeprecationPolicy defines version deprecation policy
type DeprecationPolicy struct {
	WarningPeriod time.Duration `json:"warning_period"`
	GracePeriod   time.Duration `json:"grace_period"`
	NotificationMethods []string `json:"notification_methods"`
}

// GraphQLConfig configures GraphQL support
type GraphQLConfig struct {
	Enabled         bool              `json:"enabled"`
	Endpoint        string            `json:"endpoint"`
	PlaygroundEnabled bool            `json:"playground_enabled"`
	IntrospectionEnabled bool         `json:"introspection_enabled"`
	SchemaLocation  string            `json:"schema_location"`
	Resolvers       map[string]string `json:"resolvers"`
	Directives      []string          `json:"directives"`
}

// MetricsConfig configures metrics collection
type MetricsConfig struct {
	Enabled         bool              `json:"enabled"`
	Endpoint        string            `json:"endpoint"`
	CollectionInterval time.Duration `json:"collection_interval"`
	PrometheusEnabled bool            `json:"prometheus_enabled"`
	CustomMetrics   []CustomMetric    `json:"custom_metrics"`
}

// CustomMetric defines custom metrics
type CustomMetric struct {
	Name        string            `json:"name"`
	Type        string            `json:"type"` // counter, gauge, histogram
	Description string            `json:"description"`
	Labels      []string          `json:"labels"`
}

// LoggingConfig configures request/response logging
type LoggingConfig struct {
	Enabled         bool     `json:"enabled"`
	Level           string   `json:"level"`
	Format          string   `json:"format"` // json, text
	IncludeRequest  bool     `json:"include_request"`
	IncludeResponse bool     `json:"include_response"`
	SensitiveFields []string `json:"sensitive_fields"`
	MaxBodySize     int      `json:"max_body_size"`
}

// TimeoutConfig configures various timeouts
type TimeoutConfig struct {
	ReadTimeout       time.Duration `json:"read_timeout"`
	WriteTimeout      time.Duration `json:"write_timeout"`
	IdleTimeout       time.Duration `json:"idle_timeout"`
	HandlerTimeout    time.Duration `json:"handler_timeout"`
	UpstreamTimeout   time.Duration `json:"upstream_timeout"`
	ConnectionTimeout time.Duration `json:"connection_timeout"`
}

// RetryConfig configures retry policies
type RetryConfig struct {
	Enabled         bool          `json:"enabled"`
	MaxRetries      int           `json:"max_retries"`
	InitialDelay    time.Duration `json:"initial_delay"`
	MaxDelay        time.Duration `json:"max_delay"`
	BackoffStrategy string        `json:"backoff_strategy"` // linear, exponential
	RetryableStatus []int         `json:"retryable_status"`
}

// CompressionConfig configures response compression
type CompressionConfig struct {
	Enabled       bool     `json:"enabled"`
	Algorithm     string   `json:"algorithm"` // gzip, deflate, br
	MinSize       int      `json:"min_size"`
	ContentTypes  []string `json:"content_types"`
	CompressionLevel int   `json:"compression_level"`
}

// SecurityConfig configures security headers and policies
type SecurityConfig struct {
	Headers              SecurityHeaders      `json:"headers"`
	ContentSecurityPolicy string              `json:"content_security_policy"`
	IPWhitelist          []string             `json:"ip_whitelist"`
	IPBlacklist          []string             `json:"ip_blacklist"`
	RequestSizeLimit     int64                `json:"request_size_limit"`
	TLSConfig            TLSSecurityConfig    `json:"tls_config"`
}

// SecurityHeaders configures security headers
type SecurityHeaders struct {
	XFrameOptions        string `json:"x_frame_options"`
	XContentTypeOptions  string `json:"x_content_type_options"`
	XSSProtection        string `json:"xss_protection"`
	StrictTransportSecurity string `json:"strict_transport_security"`
	ReferrerPolicy       string `json:"referrer_policy"`
	PermissionsPolicy    string `json:"permissions_policy"`
}

// TLSSecurityConfig configures TLS security
type TLSSecurityConfig struct {
	MinVersion       string   `json:"min_version"`
	MaxVersion       string   `json:"max_version"`
	CipherSuites     []string `json:"cipher_suites"`
	RequireClientCert bool    `json:"require_client_cert"`
	ClientCAFile     string   `json:"client_ca_file"`
}

// HealthCheckConfig configures health checking
type HealthCheckConfig struct {
	Enabled         bool          `json:"enabled"`
	Endpoint        string        `json:"endpoint"`
	Interval        time.Duration `json:"interval"`
	Timeout         time.Duration `json:"timeout"`
	HealthyThreshold int          `json:"healthy_threshold"`
	UnhealthyThreshold int        `json:"unhealthy_threshold"`
	Checks          []HealthCheck `json:"checks"`
}

// HealthCheck defines individual health checks
type HealthCheck struct {
	Name        string            `json:"name"`
	Type        string            `json:"type"` // http, tcp, grpc, custom
	Target      string            `json:"target"`
	Method      string            `json:"method"`
	Headers     map[string]string `json:"headers"`
	Body        string            `json:"body"`
	ExpectedStatus int            `json:"expected_status"`
	Timeout     time.Duration     `json:"timeout"`
}

// ServiceRegistry manages service discovery and registration
type ServiceRegistry struct {
	logger    *logrus.Logger
	services  map[string]*Service
	discovery ServiceDiscovery
	mu        sync.RWMutex
}

// Service represents a registered service
type Service struct {
	ID            string                 `json:"id"`
	Name          string                 `json:"name"`
	Version       string                 `json:"version"`
	Host          string                 `json:"host"`
	Port          int                    `json:"port"`
	Protocol      string                 `json:"protocol"`
	HealthStatus  HealthStatus           `json:"health_status"`
	LastHeartbeat time.Time              `json:"last_heartbeat"`
	Metadata      map[string]interface{} `json:"metadata"`
	Tags          []string               `json:"tags"`
	Weight        int                    `json:"weight"`
}

// HealthStatus defines service health status
type HealthStatus string

const (
	HealthStatusHealthy   HealthStatus = "healthy"
	HealthStatusUnhealthy HealthStatus = "unhealthy"
	HealthStatusDegraded  HealthStatus = "degraded"
	HealthStatusUnknown   HealthStatus = "unknown"
)

// ServiceDiscovery interface for service discovery
type ServiceDiscovery interface {
	Register(service *Service) error
	Deregister(serviceID string) error
	Discover(serviceName string) ([]*Service, error)
	Watch(serviceName string) (<-chan []*Service, error)
}

// RateLimiter manages rate limiting
type RateLimiter struct {
	logger     *logrus.Logger
	limiters   map[string]*rate.Limiter
	rules      map[string]RateLimitRule
	storage    RateLimitStorage
	mu         sync.RWMutex
}

// RateLimitStorage interface for rate limit storage
type RateLimitStorage interface {
	Get(key string) (*RateLimitEntry, error)
	Set(key string, entry *RateLimitEntry, ttl time.Duration) error
	Increment(key string, window time.Duration) (int, error)
}

// RateLimitEntry represents a rate limit entry
type RateLimitEntry struct {
	Count     int       `json:"count"`
	ResetTime time.Time `json:"reset_time"`
	Window    time.Duration `json:"window"`
}

// CircuitBreaker implements circuit breaker pattern
type CircuitBreaker struct {
	logger    *logrus.Logger
	breakers  map[string]*Breaker
	config    CircuitBreakerConfig
	mu        sync.RWMutex
}

// Breaker represents individual circuit breaker
type Breaker struct {
	State             BreakerState  `json:"state"`
	FailureCount      int           `json:"failure_count"`
	RequestCount      int           `json:"request_count"`
	LastFailureTime   time.Time     `json:"last_failure_time"`
	NextAttemptTime   time.Time     `json:"next_attempt_time"`
	ConsecutiveSuccesses int        `json:"consecutive_successes"`
}

// BreakerState defines circuit breaker states
type BreakerState string

const (
	BreakerStateClosed    BreakerState = "closed"
	BreakerStateOpen      BreakerState = "open"
	BreakerStateHalfOpen  BreakerState = "half_open"
)

// LoadBalancer manages load balancing
type LoadBalancer struct {
	logger    *logrus.Logger
	strategy  LoadBalancingStrategy
	services  map[string][]*Service
	roundRobinCounters map[string]int
	mu        sync.RWMutex
}

// AuthManager manages authentication and authorization
type AuthManager struct {
	logger        *logrus.Logger
	config        AuthConfig
	jwtValidator  *JWTValidator
	apiKeyStore   APIKeyStore
	sessionStore  SessionStore
	userStore     UserStore
}

// JWTValidator validates JWT tokens
type JWTValidator struct {
	secretKey []byte
	issuer    string
	audience  string
}

// APIKeyStore interface for API key storage
type APIKeyStore interface {
	ValidateKey(key string) (*APIKeyInfo, error)
	GetKeyInfo(key string) (*APIKeyInfo, error)
}

// APIKeyInfo contains API key information
type APIKeyInfo struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	UserID      string    `json:"user_id"`
	Permissions []string  `json:"permissions"`
	CreatedAt   time.Time `json:"created_at"`
	ExpiresAt   *time.Time `json:"expires_at,omitempty"`
	IsActive    bool      `json:"is_active"`
}

// SessionStore interface for session storage
type SessionStore interface {
	Get(sessionID string) (*SessionData, error)
	Set(sessionID string, data *SessionData, ttl time.Duration) error
	Delete(sessionID string) error
}

// SessionData contains session information
type SessionData struct {
	UserID      string                 `json:"user_id"`
	Username    string                 `json:"username"`
	Permissions []string               `json:"permissions"`
	CreatedAt   time.Time              `json:"created_at"`
	LastAccess  time.Time              `json:"last_access"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// UserStore interface for user storage
type UserStore interface {
	GetUser(userID string) (*User, error)
	ValidateCredentials(username, password string) (*User, error)
}

// User represents a user
type User struct {
	ID          string    `json:"id"`
	Username    string    `json:"username"`
	Email       string    `json:"email"`
	Roles       []string  `json:"roles"`
	Permissions []string  `json:"permissions"`
	IsActive    bool      `json:"is_active"`
	CreatedAt   time.Time `json:"created_at"`
	LastLogin   *time.Time `json:"last_login,omitempty"`
}

// MiddlewareChain manages middleware execution
type MiddlewareChain struct {
	logger      *logrus.Logger
	middlewares []Middleware
}

// Middleware interface for middleware functions
type Middleware interface {
	ProcessRequest(ctx context.Context, req *http.Request) (*http.Request, error)
	ProcessResponse(ctx context.Context, resp *http.Response) (*http.Response, error)
}

// GatewayMetrics collects gateway metrics
type GatewayMetrics struct {
	TotalRequests       int64                    `json:"total_requests"`
	RequestsByService   map[string]int64         `json:"requests_by_service"`
	RequestsByStatus    map[int]int64            `json:"requests_by_status"`
	AverageResponseTime time.Duration            `json:"average_response_time"`
	ErrorRate           float64                  `json:"error_rate"`
	ThroughputRPS       float64                  `json:"throughput_rps"`
	ActiveConnections   int64                    `json:"active_connections"`
	CircuitBreakerTrips int64                    `json:"circuit_breaker_trips"`
	RateLimitHits       int64                    `json:"rate_limit_hits"`
	LastUpdated         time.Time                `json:"last_updated"`
}

// NewAPIGateway creates a new API gateway
func NewAPIGateway(config *GatewayConfig, logger *logrus.Logger) (*APIGateway, error) {
	if config == nil {
		config = getDefaultGatewayConfig()
	}

	if logger == nil {
		logger = logrus.New()
	}

	gateway := &APIGateway{
		logger:           logger,
		config:           config,
		router:           mux.NewRouter(),
		metricsCollector: &GatewayMetrics{
			RequestsByService: make(map[string]int64),
			RequestsByStatus:  make(map[int]int64),
		},
		stopCh: make(chan struct{}),
	}

	// Initialize components
	gateway.serviceRegistry = NewServiceRegistry(logger)
	gateway.rateLimiter = NewRateLimiter(config.RateLimiting, logger)
	gateway.circuitBreaker = NewCircuitBreaker(config.CircuitBreaker, logger)
	gateway.loadBalancer = NewLoadBalancer(config.LoadBalancing, logger)
	gateway.authManager = NewAuthManager(config.Authentication, logger)
	gateway.middleware = NewMiddlewareChain(logger)

	// Setup routes
	if err := gateway.setupRoutes(); err != nil {
		return nil, fmt.Errorf("failed to setup routes: %w", err)
	}

	return gateway, nil
}

// Start starts the API gateway
func (gw *APIGateway) Start(ctx context.Context) error {
	if !gw.config.Enabled {
		gw.logger.Info("API Gateway disabled")
		return nil
	}

	gw.logger.Info("Starting API Gateway")

	// Setup CORS
	var handler http.Handler = gw.router
	if gw.config.CORS.Enabled {
		c := cors.New(cors.Options{
			AllowedOrigins:   gw.config.CORS.AllowedOrigins,
			AllowedMethods:   gw.config.CORS.AllowedMethods,
			AllowedHeaders:   gw.config.CORS.AllowedHeaders,
			ExposedHeaders:   gw.config.CORS.ExposedHeaders,
			AllowCredentials: gw.config.CORS.AllowCredentials,
			MaxAge:           gw.config.CORS.MaxAge,
		})
		handler = c.Handler(gw.router)
	}

	// Create HTTP server
	addr := fmt.Sprintf("%s:%d", gw.config.Host, gw.config.Port)
	gw.httpServer = &http.Server{
		Addr:         addr,
		Handler:      handler,
		ReadTimeout:  gw.config.Timeouts.ReadTimeout,
		WriteTimeout: gw.config.Timeouts.WriteTimeout,
		IdleTimeout:  gw.config.Timeouts.IdleTimeout,
	}

	// Start metrics collection
	go gw.metricsCollectionLoop(ctx)

	// Start health check monitoring
	if gw.config.HealthCheck.Enabled {
		go gw.healthCheckLoop(ctx)
	}

	// Start server
	go func() {
		if gw.config.TLSEnabled {
			gw.logger.WithField("addr", addr).Info("Starting HTTPS server")
			if err := gw.httpServer.ListenAndServeTLS(gw.config.CertFile, gw.config.KeyFile); err != nil && err != http.ErrServerClosed {
				gw.logger.WithError(err).Error("HTTPS server failed")
			}
		} else {
			gw.logger.WithField("addr", addr).Info("Starting HTTP server")
			if err := gw.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				gw.logger.WithError(err).Error("HTTP server failed")
			}
		}
	}()

	return nil
}

// Stop stops the API gateway
func (gw *APIGateway) Stop(ctx context.Context) error {
	gw.logger.Info("Stopping API Gateway")
	close(gw.stopCh)

	if gw.httpServer != nil {
		return gw.httpServer.Shutdown(ctx)
	}

	return nil
}

// setupRoutes sets up gateway routes
func (gw *APIGateway) setupRoutes() error {
	// Health check endpoint
	gw.router.HandleFunc("/health", gw.handleHealth).Methods("GET")

	// Metrics endpoint
	if gw.config.Metrics.Enabled {
		gw.router.HandleFunc("/metrics", gw.handleMetrics).Methods("GET")
	}

	// GraphQL endpoint
	if gw.config.GraphQL.Enabled {
		gw.router.HandleFunc(gw.config.GraphQL.Endpoint, gw.handleGraphQL).Methods("POST", "GET")
		if gw.config.GraphQL.PlaygroundEnabled {
			gw.router.HandleFunc("/playground", gw.handleGraphQLPlayground).Methods("GET")
		}
	}

	// Service routes
	for _, service := range gw.config.Services {
		gw.setupServiceRoutes(service)
	}

	return nil
}

// setupServiceRoutes sets up routes for a specific service
func (gw *APIGateway) setupServiceRoutes(service *ServiceConfig) {
	// Create route handler
	handler := gw.createServiceHandler(service)

	// Apply middleware
	handler = gw.applyMiddleware(handler, service)

	// Register route
	if service.Versioning.Enabled {
		gw.setupVersionedRoutes(service, handler)
	} else {
		gw.router.PathPrefix(service.Path).Handler(handler)
	}
}

// createServiceHandler creates handler for a service
func (gw *APIGateway) createServiceHandler(service *ServiceConfig) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Circuit breaker check
		if service.CircuitBreaker {
			if !gw.circuitBreaker.AllowRequest(service.Name) {
				gw.writeErrorResponse(w, http.StatusServiceUnavailable, "Service temporarily unavailable")
				return
			}
		}

		// Rate limiting
		if service.RateLimiting != nil && service.RateLimiting.Enabled {
			if !gw.rateLimiter.Allow(r, service.RateLimiting) {
				gw.writeErrorResponse(w, http.StatusTooManyRequests, "Rate limit exceeded")
				return
			}
		}

		// Load balancing
		target := gw.loadBalancer.SelectTarget(service.Name)
		if target == nil {
			gw.writeErrorResponse(w, http.StatusServiceUnavailable, "No healthy service instances")
			return
		}

		// Create reverse proxy
		targetURL, _ := url.Parse(fmt.Sprintf("%s://%s:%d", target.Protocol, target.Host, target.Port))
		proxy := httputil.NewSingleHostReverseProxy(targetURL)

		// Custom transport with timeout
		proxy.Transport = &http.Transport{
			ResponseHeaderTimeout: service.Timeout,
		}

		// Record metrics
		start := time.Now()
		defer func() {
			gw.recordMetrics(service.Name, time.Since(start), 200) // Simplified
		}()

		// Proxy request
		proxy.ServeHTTP(w, r)
	})
}

// applyMiddleware applies middleware to handler
func (gw *APIGateway) applyMiddleware(handler http.Handler, service *ServiceConfig) http.Handler {
	// Authentication middleware
	if service.Authentication != nil && service.Authentication.Enabled {
		handler = gw.authenticationMiddleware(handler, service.Authentication)
	}

	// Logging middleware
	if gw.config.Logging.Enabled {
		handler = gw.loggingMiddleware(handler)
	}

	// Security headers middleware
	handler = gw.securityHeadersMiddleware(handler)

	// Compression middleware
	if gw.config.Compression.Enabled {
		handler = gw.compressionMiddleware(handler)
	}

	return handler
}

// setupVersionedRoutes sets up versioned routes
func (gw *APIGateway) setupVersionedRoutes(service *ServiceConfig, handler http.Handler) {
	for version, versionConfig := range service.Versioning.VersionMapping {
		switch service.Versioning.Strategy {
		case VersioningPath:
			path := fmt.Sprintf("/%s%s", version, service.Path)
			gw.router.PathPrefix(path).Handler(handler)
		case VersioningHeader:
			gw.router.PathPrefix(service.Path).Handler(
				gw.versionHeaderMiddleware(handler, version, service.Versioning.HeaderName),
			)
		case VersioningQuery:
			gw.router.PathPrefix(service.Path).Handler(
				gw.versionQueryMiddleware(handler, version),
			)
		}
	}
}

// Middleware implementations
func (gw *APIGateway) authenticationMiddleware(next http.Handler, auth *ServiceAuth) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authenticated := false

		for _, method := range auth.Methods {
			switch method {
			case "jwt":
				if gw.authManager.ValidateJWT(r) {
					authenticated = true
					break
				}
			case "api_key":
				if gw.authManager.ValidateAPIKey(r) {
					authenticated = true
					break
				}
			case "basic":
				if gw.authManager.ValidateBasicAuth(r) {
					authenticated = true
					break
				}
			}
		}

		if auth.Required && !authenticated {
			gw.writeErrorResponse(w, http.StatusUnauthorized, "Authentication required")
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (gw *APIGateway) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Log request
		gw.logger.WithFields(logrus.Fields{
			"method":     r.Method,
			"path":       r.URL.Path,
			"remote_ip":  r.RemoteAddr,
			"user_agent": r.UserAgent(),
		}).Info("Incoming request")

		// Process request
		next.ServeHTTP(w, r)

		// Log response
		gw.logger.WithFields(logrus.Fields{
			"method":   r.Method,
			"path":     r.URL.Path,
			"duration": time.Since(start),
		}).Info("Request completed")
	})
}

func (gw *APIGateway) securityHeadersMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		headers := gw.config.Security.Headers
		w.Header().Set("X-Frame-Options", headers.XFrameOptions)
		w.Header().Set("X-Content-Type-Options", headers.XContentTypeOptions)
		w.Header().Set("X-XSS-Protection", headers.XSSProtection)
		w.Header().Set("Strict-Transport-Security", headers.StrictTransportSecurity)
		w.Header().Set("Referrer-Policy", headers.ReferrerPolicy)
		w.Header().Set("Permissions-Policy", headers.PermissionsPolicy)

		if gw.config.Security.ContentSecurityPolicy != "" {
			w.Header().Set("Content-Security-Policy", gw.config.Security.ContentSecurityPolicy)
		}

		next.ServeHTTP(w, r)
	})
}

func (gw *APIGateway) compressionMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simple compression implementation
		if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
			w.Header().Set("Content-Encoding", "gzip")
		}
		next.ServeHTTP(w, r)
	})
}

func (gw *APIGateway) versionHeaderMiddleware(next http.Handler, version, headerName string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get(headerName) == version {
			next.ServeHTTP(w, r)
		} else {
			gw.writeErrorResponse(w, http.StatusNotFound, "Version not found")
		}
	})
}

func (gw *APIGateway) versionQueryMiddleware(next http.Handler, version string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("version") == version {
			next.ServeHTTP(w, r)
		} else {
			gw.writeErrorResponse(w, http.StatusNotFound, "Version not found")
		}
	})
}

// Handler implementations
func (gw *APIGateway) handleHealth(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now(),
		"version":   "1.0.0",
		"services":  gw.getServiceHealth(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

func (gw *APIGateway) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(gw.metricsCollector)
}

func (gw *APIGateway) handleGraphQL(w http.ResponseWriter, r *http.Request) {
	// Mock GraphQL handler
	response := map[string]interface{}{
		"data": map[string]interface{}{
			"message": "GraphQL endpoint - implementation would handle queries here",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (gw *APIGateway) handleGraphQLPlayground(w http.ResponseWriter, r *http.Request) {
	playground := `
<!DOCTYPE html>
<html>
<head>
    <title>GraphQL Playground</title>
</head>
<body>
    <div id="root">
        <style>
            body { margin: 0; overflow: hidden; }
            #root { height: 100vh; }
        </style>
        <h1>GraphQL Playground</h1>
        <p>GraphQL endpoint: <code>` + gw.config.GraphQL.Endpoint + `</code></p>
    </div>
</body>
</html>`

	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(playground))
}

// Helper methods
func (gw *APIGateway) writeErrorResponse(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]string{
		"error":   message,
		"status":  fmt.Sprintf("%d", status),
		"timestamp": time.Now().Format(time.RFC3339),
	})
}

func (gw *APIGateway) getServiceHealth() map[string]interface{} {
	// Mock service health
	return map[string]interface{}{
		"generation_service": "healthy",
		"validation_service": "healthy",
		"storage_service":    "healthy",
	}
}

func (gw *APIGateway) recordMetrics(serviceName string, duration time.Duration, status int) {
	gw.mu.Lock()
	defer gw.mu.Unlock()

	gw.metricsCollector.TotalRequests++
	gw.metricsCollector.RequestsByService[serviceName]++
	gw.metricsCollector.RequestsByStatus[status]++
	gw.metricsCollector.LastUpdated = time.Now()
}

func (gw *APIGateway) metricsCollectionLoop(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-gw.stopCh:
			return
		case <-ticker.C:
			gw.updateMetrics()
		}
	}
}

func (gw *APIGateway) updateMetrics() {
	gw.mu.Lock()
	defer gw.mu.Unlock()

	// Calculate throughput and error rate
	if gw.metricsCollector.TotalRequests > 0 {
		errorCount := gw.metricsCollector.RequestsByStatus[500] + 
					 gw.metricsCollector.RequestsByStatus[502] + 
					 gw.metricsCollector.RequestsByStatus[503]
		gw.metricsCollector.ErrorRate = float64(errorCount) / float64(gw.metricsCollector.TotalRequests)
	}

	gw.metricsCollector.LastUpdated = time.Now()
}

func (gw *APIGateway) healthCheckLoop(ctx context.Context) {
	ticker := time.NewTicker(gw.config.HealthCheck.Interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-gw.stopCh:
			return
		case <-ticker.C:
			gw.performHealthChecks()
		}
	}
}

func (gw *APIGateway) performHealthChecks() {
	gw.logger.Info("Performing health checks")
	// Implementation would check service health
}

func getDefaultGatewayConfig() *GatewayConfig {
	return &GatewayConfig{
		Enabled: true,
		Port:    8080,
		Host:    "0.0.0.0",
		Services: map[string]*ServiceConfig{
			"generation": {
				Name:     "generation-service",
				Path:     "/api/v1/generate",
				Hosts:    []string{"localhost:8081"},
				Protocol: "http",
				Timeout:  30 * time.Second,
			},
		},
		RateLimiting: RateLimitingConfig{
			Enabled: true,
			GlobalLimits: map[string]RateLimitRule{
				"default": {
					RequestsPerMinute: 1000,
					BurstSize:         100,
				},
			},
		},
		Authentication: AuthConfig{
			Enabled: false,
		},
		CORS: CORSConfig{
			Enabled:        true,
			AllowedOrigins: []string{"*"},
			AllowedMethods: []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
			AllowedHeaders: []string{"*"},
		},
		LoadBalancing: LoadBalancingConfig{
			Strategy:            LoadBalancingRoundRobin,
			HealthCheckEnabled:  true,
			HealthCheckInterval: 30 * time.Second,
		},
		CircuitBreaker: CircuitBreakerConfig{
			Enabled:          true,
			FailureThreshold: 5,
			RecoveryTimeout:  30 * time.Second,
		},
		Timeouts: TimeoutConfig{
			ReadTimeout:    10 * time.Second,
			WriteTimeout:   10 * time.Second,
			IdleTimeout:    60 * time.Second,
			HandlerTimeout: 30 * time.Second,
		},
		Security: SecurityConfig{
			Headers: SecurityHeaders{
				XFrameOptions:       "DENY",
				XContentTypeOptions: "nosniff",
				XSSProtection:       "1; mode=block",
			},
		},
		Metrics: MetricsConfig{
			Enabled:            true,
			Endpoint:           "/metrics",
			CollectionInterval: 30 * time.Second,
		},
		Logging: LoggingConfig{
			Enabled: true,
			Level:   "info",
			Format:  "json",
		},
		HealthCheck: HealthCheckConfig{
			Enabled:  true,
			Endpoint: "/health",
			Interval: 30 * time.Second,
			Timeout:  5 * time.Second,
		},
	}
}

// Component implementations (simplified for brevity)

func NewServiceRegistry(logger *logrus.Logger) *ServiceRegistry {
	return &ServiceRegistry{
		logger:   logger,
		services: make(map[string]*Service),
	}
}

func NewRateLimiter(config RateLimitingConfig, logger *logrus.Logger) *RateLimiter {
	return &RateLimiter{
		logger:   logger,
		limiters: make(map[string]*rate.Limiter),
		rules:    config.GlobalLimits,
	}
}

func (rl *RateLimiter) Allow(r *http.Request, config *ServiceRateLimit) bool {
	// Mock rate limiting - always allow for now
	return true
}

func NewCircuitBreaker(config CircuitBreakerConfig, logger *logrus.Logger) *CircuitBreaker {
	return &CircuitBreaker{
		logger:   logger,
		breakers: make(map[string]*Breaker),
		config:   config,
	}
}

func (cb *CircuitBreaker) AllowRequest(service string) bool {
	// Mock circuit breaker - always allow for now
	return true
}

func NewLoadBalancer(config LoadBalancingConfig, logger *logrus.Logger) *LoadBalancer {
	return &LoadBalancer{
		logger:   logger,
		strategy: config.Strategy,
		services: make(map[string][]*Service),
		roundRobinCounters: make(map[string]int),
	}
}

func (lb *LoadBalancer) SelectTarget(serviceName string) *Service {
	// Mock load balancer - return default service
	return &Service{
		Host:     "localhost",
		Port:     8081,
		Protocol: "http",
	}
}

func NewAuthManager(config AuthConfig, logger *logrus.Logger) *AuthManager {
	return &AuthManager{
		logger: logger,
		config: config,
	}
}

func (am *AuthManager) ValidateJWT(r *http.Request) bool {
	// Mock JWT validation
	return true
}

func (am *AuthManager) ValidateAPIKey(r *http.Request) bool {
	// Mock API key validation
	return true
}

func (am *AuthManager) ValidateBasicAuth(r *http.Request) bool {
	// Mock basic auth validation
	return true
}

func NewMiddlewareChain(logger *logrus.Logger) *MiddlewareChain {
	return &MiddlewareChain{
		logger:      logger,
		middlewares: make([]Middleware, 0),
	}
}