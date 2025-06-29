# TSIoT Code Style Guide

This document outlines the coding standards and conventions for the TSIoT (Time Series IoT) project. Following these guidelines ensures code consistency, readability, and maintainability across the codebase.

## Table of Contents

- [General Principles](#general-principles)
- [Go Code Standards](#go-code-standards)
- [File Organization](#file-organization)
- [Naming Conventions](#naming-conventions)
- [Comments and Documentation](#comments-and-documentation)
- [Error Handling](#error-handling)
- [Testing Standards](#testing-standards)
- [Import Organization](#import-organization)
- [Configuration Management](#configuration-management)
- [Logging Standards](#logging-standards)

## General Principles

### Clean Code
- Write code that is easy to read and understand
- Prefer explicit over implicit behavior
- Use meaningful names for variables, functions, and types
- Keep functions small and focused on a single responsibility
- Avoid deep nesting levels (maximum 4 levels)

### SOLID Principles
- **Single Responsibility**: Each struct/function should have one reason to change
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Derived types must be substitutable for their base types
- **Interface Segregation**: Clients shouldn't depend on interfaces they don't use
- **Dependency Inversion**: Depend on abstractions, not concretions

## Go Code Standards

### Formatting
Use `gofmt` and `goimports` for consistent formatting:

```bash
# Format all Go files
make fmt

# Import organization
goimports -w .
```

### Line Length
- Keep lines under 120 characters when possible
- Break long function signatures across multiple lines:

```go
// Good
func GenerateTimeSeriesData(
    ctx context.Context,
    generator GeneratorType,
    params *GenerationParams,
    options ...GenerationOption,
) (*TimeSeriesData, error) {
    // implementation
}

// Avoid
func GenerateTimeSeriesData(ctx context.Context, generator GeneratorType, params *GenerationParams, options ...GenerationOption) (*TimeSeriesData, error) {
```

### Variable Declarations

```go
// Prefer short variable declarations
data := &TimeSeriesData{}

// Use var for zero values
var (
    count int
    total float64
    names []string
)

// Group related constants
const (
    DefaultBatchSize = 1000
    MaxRetries      = 3
    TimeoutSeconds  = 30
)
```

## File Organization

### Package Structure
```
internal/
   agents/           # Agent implementations
   generators/       # Data generation algorithms
      arima/       # ARIMA-specific code
      rnn/         # RNN-specific code
      timegan/     # TimeGAN-specific code
   storage/         # Storage backend implementations
   validation/      # Data validation logic
   protocols/       # Communication protocols
   utils/           # Utility functions
```

### File Naming
- Use snake_case for file names: `time_series_generator.go`
- Test files: `time_series_generator_test.go`
- Benchmark files: `time_series_generator_bench_test.go`
- Example files: `time_series_generator_example_test.go`

### File Headers
Every Go file should start with a package comment:

```go
// Package generators provides implementations for various synthetic
// time series data generation algorithms including TimeGAN, ARIMA,
// RNN, and statistical methods.
package generators

import (
    "context"
    "fmt"
    // ... other imports
)
```

## Naming Conventions

### Functions and Methods
```go
// Public functions: PascalCase
func GenerateTimeSeries() {}

// Private functions: camelCase
func validateParameters() {}

// Methods with receiver
func (g *TimeGANGenerator) Generate(ctx context.Context) error {}

// Interface methods
type Generator interface {
    Generate(ctx context.Context, params *Params) (*Data, error)
    Validate(data *Data) error
}
```

### Variables and Constants
```go
// Constants: PascalCase
const (
    DefaultEpochs    = 1000
    MaxBatchSize     = 10000
    ConfigFileName   = "config.yaml"
)

// Variables: camelCase
var (
    defaultConfig *Config
    logger        *slog.Logger
)

// Public variables: PascalCase (avoid when possible)
var DefaultGenerator = &TimeGANGenerator{}
```

### Types and Structs
```go
// Types: PascalCase
type TimeSeriesData struct {
    Timestamps []time.Time `json:"timestamps"`
    Values     []float64   `json:"values"`
    Metadata   Metadata    `json:"metadata"`
}

// Interfaces: PascalCase with "er" suffix when applicable
type Generator interface {
    Generate(context.Context, *Params) (*Data, error)
}

type Validator interface {
    Validate(*Data) (*ValidationResult, error)
}
```

## Comments and Documentation

### Package Documentation
```go
// Package timegan implements the Time-series Generative Adversarial Networks
// algorithm for generating synthetic time series data. It provides both
// training and generation capabilities with support for various data types
// and temporal patterns.
//
// The implementation follows the original TimeGAN paper and includes
// optimizations for time series IoT data generation.
package timegan
```

### Function Documentation
```go
// GenerateTimeSeries generates synthetic time series data using the specified
// generator algorithm and parameters.
//
// Parameters:
//   - ctx: Context for cancellation and timeouts
//   - generator: The generation algorithm to use (timegan, arima, rnn, statistical)
//   - params: Generation parameters including data points, frequency, and features
//   - options: Additional options for customizing generation behavior
//
// Returns:
//   - *TimeSeriesData: The generated synthetic data
//   - error: Any error that occurred during generation
//
// Example:
//   data, err := GenerateTimeSeries(ctx, TimeGAN, &GenerationParams{
//       DataPoints: 1000,
//       Frequency:  time.Minute,
//       Features:   []string{"temperature", "humidity"},
//   })
func GenerateTimeSeries(
    ctx context.Context,
    generator GeneratorType,
    params *GenerationParams,
    options ...GenerationOption,
) (*TimeSeriesData, error) {
    // implementation
}
```

### Inline Comments
```go
func processData(data []float64) []float64 {
    // Apply normalization to ensure values are in [0, 1] range
    normalized := make([]float64, len(data))
    
    // Find min and max values for scaling
    min, max := findMinMax(data)
    
    for i, value := range data {
        // Scale each value: (x - min) / (max - min)
        normalized[i] = (value - min) / (max - min)
    }
    
    return normalized
}
```

## Error Handling

### Error Types
```go
// Define custom error types for different categories
type ValidationError struct {
    Field   string
    Value   interface{}
    Message string
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation failed for field %s: %s", e.Field, e.Message)
}

// Use errors.New for simple errors
var ErrInvalidGenerator = errors.New("invalid generator type")

// Use fmt.Errorf for formatted errors
func validateConfig(config *Config) error {
    if config.BatchSize <= 0 {
        return fmt.Errorf("batch size must be positive, got %d", config.BatchSize)
    }
    return nil
}
```

### Error Wrapping
```go
// Wrap errors to provide context
func loadModel(path string) (*Model, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("failed to load model from %s: %w", path, err)
    }
    
    model, err := parseModel(data)
    if err != nil {
        return nil, fmt.Errorf("failed to parse model: %w", err)
    }
    
    return model, nil
}
```

### Error Handling Patterns
```go
// Early return for error conditions
func processRequest(req *Request) (*Response, error) {
    if req == nil {
        return nil, errors.New("request cannot be nil")
    }
    
    if err := validateRequest(req); err != nil {
        return nil, fmt.Errorf("invalid request: %w", err)
    }
    
    // Process request
    result, err := processData(req.Data)
    if err != nil {
        return nil, fmt.Errorf("processing failed: %w", err)
    }
    
    return &Response{Data: result}, nil
}
```

## Testing Standards

### Test File Organization
```go
// file: generator_test.go
package generators

import (
    "context"
    "testing"
    "time"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)
```

### Test Function Naming
```go
// Test functions: TestFunctionName_Scenario
func TestGenerateTimeSeries_ValidParams(t *testing.T) {}
func TestGenerateTimeSeries_InvalidGenerator(t *testing.T) {}
func TestGenerateTimeSeries_ContextCancellation(t *testing.T) {}

// Benchmark functions: BenchmarkFunctionName_Scenario
func BenchmarkGenerateTimeSeries_LargeDataset(b *testing.B) {}

// Example functions: ExampleFunctionName
func ExampleGenerateTimeSeries() {}
```

### Test Structure
```go
func TestGenerateTimeSeries_ValidParams(t *testing.T) {
    // Arrange
    ctx := context.Background()
    params := &GenerationParams{
        DataPoints: 100,
        Frequency:  time.Minute,
        Features:   []string{"temperature"},
    }
    
    // Act
    data, err := GenerateTimeSeries(ctx, TimeGAN, params)
    
    // Assert
    require.NoError(t, err)
    assert.NotNil(t, data)
    assert.Len(t, data.Values, 100)
    assert.Equal(t, len(data.Timestamps), len(data.Values))
}
```

### Table-Driven Tests
```go
func TestValidateConfig(t *testing.T) {
    tests := []struct {
        name      string
        config    *Config
        wantError bool
        errorMsg  string
    }{
        {
            name: "valid config",
            config: &Config{
                BatchSize: 1000,
                Epochs:    100,
            },
            wantError: false,
        },
        {
            name: "invalid batch size",
            config: &Config{
                BatchSize: 0,
                Epochs:    100,
            },
            wantError: true,
            errorMsg:  "batch size must be positive",
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := validateConfig(tt.config)
            if tt.wantError {
                require.Error(t, err)
                assert.Contains(t, err.Error(), tt.errorMsg)
            } else {
                require.NoError(t, err)
            }
        })
    }
}
```

## Import Organization

### Import Grouping
```go
import (
    // Standard library imports first
    "context"
    "fmt"
    "time"
    
    // Third-party imports second
    "github.com/prometheus/client_golang/prometheus"
    "github.com/stretchr/testify/assert"
    "go.uber.org/zap"
    
    // Local imports last
    "github.com/inferloop/tsiot/internal/storage"
    "github.com/inferloop/tsiot/pkg/models"
)
```

### Import Aliases
```go
import (
    "context"
    
    pb "github.com/inferloop/tsiot/internal/protocols/grpc/protos"
    "github.com/inferloop/tsiot/pkg/models"
)
```

## Configuration Management

### Configuration Structs
```go
type Config struct {
    Server   ServerConfig   `yaml:"server" json:"server"`
    Database DatabaseConfig `yaml:"database" json:"database"`
    Logging  LoggingConfig  `yaml:"logging" json:"logging"`
}

type ServerConfig struct {
    Host         string        `yaml:"host" json:"host"`
    Port         int           `yaml:"port" json:"port"`
    ReadTimeout  time.Duration `yaml:"read_timeout" json:"read_timeout"`
    WriteTimeout time.Duration `yaml:"write_timeout" json:"write_timeout"`
}
```

### Environment Variables
```go
// Use constants for environment variable names
const (
    EnvServerHost = "TSIOT_SERVER_HOST"
    EnvServerPort = "TSIOT_SERVER_PORT"
    EnvLogLevel   = "TSIOT_LOG_LEVEL"
)

// Provide defaults and validation
func loadConfig() (*Config, error) {
    config := &Config{
        Server: ServerConfig{
            Host: getEnvOrDefault(EnvServerHost, "localhost"),
            Port: getEnvIntOrDefault(EnvServerPort, 8080),
        },
    }
    
    if err := validateConfig(config); err != nil {
        return nil, fmt.Errorf("invalid configuration: %w", err)
    }
    
    return config, nil
}
```

## Logging Standards

### Structured Logging
```go
import "log/slog"

// Use structured logging with context
func (g *Generator) Generate(ctx context.Context, params *Params) error {
    logger := slog.With(
        "generator", g.Name(),
        "data_points", params.DataPoints,
        "request_id", getRequestID(ctx),
    )
    
    logger.Info("starting data generation")
    
    start := time.Now()
    defer func() {
        logger.Info("completed data generation",
            "duration", time.Since(start),
        )
    }()
    
    // Generation logic
    return nil
}
```

### Log Levels
```go
// Use appropriate log levels
logger.Debug("detailed debugging information")
logger.Info("general information about operation")
logger.Warn("warning about potential issues")
logger.Error("error occurred", "error", err)
```

## Code Review Checklist

### Before Submitting
- [ ] Code follows the style guidelines
- [ ] All functions have appropriate documentation
- [ ] Error handling is implemented correctly
- [ ] Tests are written and passing
- [ ] No hardcoded values (use constants or configuration)
- [ ] Logging is appropriate and structured
- [ ] Performance considerations are addressed
- [ ] Security implications are considered

### During Review
- [ ] Code is readable and maintainable
- [ ] Business logic is correct
- [ ] Edge cases are handled
- [ ] Resource cleanup is proper
- [ ] Thread safety is considered where applicable
- [ ] Dependencies are justified and minimal

## Tools and Automation

### Required Tools
```bash
# Install development tools
go install golang.org/x/tools/cmd/goimports@latest
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
go install github.com/securecodewarrior/sast-scan/cmd/sast-scan@latest
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
make install-hooks

# Hooks will run:
# - gofmt
# - goimports
# - golangci-lint
# - tests
# - security scan
```

### Makefile Targets
```makefile
# Development commands
make fmt          # Format code
make lint         # Lint code
make test         # Run tests
make security     # Security scan
make doc          # Generate documentation
```