package server

import (
	"crypto/tls"
	"fmt"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/your-org/tsiot/internal/middleware"
	"github.com/your-org/tsiot/internal/storage"
)

// Config contains the configuration for the TSIOT server
type Config struct {
	// Core server settings
	Server      ServerConfig    `json:"server" yaml:"server"`
	Environment string          `json:"environment" yaml:"environment"`
	Version     string          `json:"version" yaml:"version"`
	BuildTime   string          `json:"build_time" yaml:"build_time"`
	Commit      string          `json:"commit" yaml:"commit"`
	GoVersion   string          `json:"go_version" yaml:"go_version"`
	StartTime   time.Time       `json:"start_time" yaml:"start_time"`
	
	// API settings
	API         APIConfig       `json:"api" yaml:"api"`
	
	// Middleware configurations
	Auth        AuthConfig      `json:"auth" yaml:"auth"`
	CORS        CORSConfig      `json:"cors" yaml:"cors"`
	RateLimit   RateLimitConfig `json:"rate_limit" yaml:"rate_limit"`
	
	// Component configurations  
	Generators  GeneratorConfig `json:"generators" yaml:"generators"`
	Cache       CacheConfig     `json:"cache" yaml:"cache"`
	Workers     WorkerConfig    `json:"workers" yaml:"workers"`
}

// ServerConfig contains server settings
type ServerConfig struct {
	Host           string        `json:"host" yaml:"host"`
	Port           int           `json:"port" yaml:"port"`
	ReadTimeout    time.Duration `json:"read_timeout" yaml:"read_timeout"`
	WriteTimeout   time.Duration `json:"write_timeout" yaml:"write_timeout"`
	MaxHeaderBytes int           `json:"max_header_bytes" yaml:"max_header_bytes"`
}

// APIConfig contains API settings
type APIConfig struct {
	MaxBatchSize   int           `json:"max_batch_size" yaml:"max_batch_size"`
	RequestTimeout time.Duration `json:"request_timeout" yaml:"request_timeout"`
}

// AuthConfig contains authentication settings
type AuthConfig struct {
	JWTSecret    string `json:"jwt_secret" yaml:"jwt_secret"`
	APIKeySecret string `json:"api_key_secret" yaml:"api_key_secret"`
}

// CORSConfig contains CORS settings
type CORSConfig struct {
	Enabled        bool     `json:"enabled" yaml:"enabled"`
	AllowedOrigins []string `json:"allowed_origins" yaml:"allowed_origins"`
}

// RateLimitConfig contains rate limiting settings
type RateLimitConfig struct {
	Enabled           bool   `json:"enabled" yaml:"enabled"`
	RequestsPerMinute int    `json:"requests_per_minute" yaml:"requests_per_minute"`
}

// GeneratorConfig contains generator settings
type GeneratorConfig struct {
	MaxLength int `json:"max_length" yaml:"max_length"`
}

// CacheConfig contains cache settings
type CacheConfig struct {
	Enabled bool `json:"enabled" yaml:"enabled"`
}

// WorkerConfig contains worker settings
type WorkerConfig struct {
	MaxConcurrentJobs int `json:"max_concurrent_jobs" yaml:"max_concurrent_jobs"`
}


// NewDefaultConfig creates a default server configuration
func NewDefaultConfig() *Config {
	return &Config{
		Server: ServerConfig{
			Host:           "0.0.0.0",
			Port:           8080,
			ReadTimeout:    30 * time.Second,
			WriteTimeout:   30 * time.Second,
			MaxHeaderBytes: 1 << 20, // 1MB
		},
		Environment: "development",
		Version:     "1.0.0",
		BuildTime:   time.Now().Format(time.RFC3339),
		Commit:      "unknown",
		GoVersion:   "unknown",
		StartTime:   time.Now(),
		
		API: APIConfig{
			MaxBatchSize:   100,
			RequestTimeout: 30 * time.Second,
		},
		
		Auth: AuthConfig{
			JWTSecret:    "default_jwt_secret",
			APIKeySecret: "default_api_key_secret",
		},
		
		CORS: CORSConfig{
			Enabled:        true,
			AllowedOrigins: []string{"*"},
		},
		
		RateLimit: RateLimitConfig{
			Enabled:           true,
			RequestsPerMinute: 1000,
		},
		
		Generators: GeneratorConfig{
			MaxLength: 1000000,
		},
		
		Cache: CacheConfig{
			Enabled: false,
		},
		
		Workers: WorkerConfig{
			MaxConcurrentJobs: 10,
		},
	}
}

// Validate validates the server configuration
func (c *Config) Validate() error {
	if c.Server.Port < 1 || c.Server.Port > 65535 {
		return fmt.Errorf("invalid port: %d", c.Server.Port)
	}
	
	if c.Server.ReadTimeout <= 0 {
		return fmt.Errorf("read timeout must be positive")
	}
	
	if c.Server.WriteTimeout <= 0 {
		return fmt.Errorf("write timeout must be positive")
	}
	
	if c.API.MaxBatchSize <= 0 {
		return fmt.Errorf("max batch size must be positive")
	}
	
	if c.API.RequestTimeout <= 0 {
		return fmt.Errorf("request timeout must be positive")
	}
	
	if c.Generators.MaxLength <= 0 {
		return fmt.Errorf("max generator length must be positive")
	}
	
	if c.Workers.MaxConcurrentJobs <= 0 {
		return fmt.Errorf("max concurrent jobs must be positive")
	}
	
	return nil
}

// GetAddress returns the server address
func (c *Config) GetAddress() string {
	return fmt.Sprintf("%s:%d", c.Server.Host, c.Server.Port)
}