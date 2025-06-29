# Testing Guide

This guide provides comprehensive information about testing in the TSIoT project, including test structure, best practices, and running different types of tests.

## Table of Contents

- [Test Organization](#test-organization)
- [Running Tests](#running-tests)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [End-to-End Testing](#end-to-end-testing)
- [Performance Testing](#performance-testing)
- [Test Coverage](#test-coverage)
- [Mocking and Test Doubles](#mocking-and-test-doubles)
- [Continuous Integration](#continuous-integration)
- [Best Practices](#best-practices)

## Test Organization

```
tests/
   unit/                   # Unit tests
      generators/        # Generator unit tests
      validation/        # Validator unit tests
      storage/          # Storage backend tests
   integration/           # Integration tests
      api/              # API integration tests
      protocols/        # Protocol tests (gRPC, MQTT, Kafka)
      storage/          # Storage integration tests
   e2e/                   # End-to-end tests
      scenarios/        # Test scenarios
      fixtures/         # Test data fixtures
   performance/           # Performance benchmarks
   helpers/              # Test utilities
   testdata/             # Test data files
```

## Running Tests

### All Tests
```bash
# Run all tests
make test

# Run with verbose output
make test-verbose

# Run with race detection
make test-race
```

### Unit Tests
```bash
# Run unit tests only
make test-unit

# Run specific package tests
go test ./internal/generators/...

# Run with coverage
go test -cover ./internal/...

# Run specific test
go test -run TestTimeGANGenerator ./internal/generators/timegan
```

### Integration Tests
```bash
# Run integration tests
make test-integration

# Run with specific tags
go test -tags=integration ./tests/integration/...

# Run against specific backend
STORAGE_BACKEND=influxdb make test-integration
```

### End-to-End Tests
```bash
# Run e2e tests
make test-e2e

# Run specific scenario
./scripts/test/test-e2e.sh --scenario data-generation

# Run with custom configuration
E2E_CONFIG=./tests/e2e/config/prod.yaml make test-e2e
```

## Unit Testing

### Writing Unit Tests

```go
package generators_test

import (
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "github.com/inferloop/tsiot/internal/generators"
)

func TestTimeGANGenerator_Generate(t *testing.T) {
    tests := []struct {
        name    string
        config  generators.Config
        want    int
        wantErr bool
    }{
        {
            name: "valid configuration",
            config: generators.Config{
                SensorType: "temperature",
                Frequency:  "1m",
                Points:     100,
            },
            want:    100,
            wantErr: false,
        },
        {
            name: "invalid frequency",
            config: generators.Config{
                SensorType: "temperature",
                Frequency:  "invalid",
                Points:     100,
            },
            wantErr: true,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            gen := generators.NewTimeGANGenerator()
            data, err := gen.Generate(tt.config)
            
            if tt.wantErr {
                assert.Error(t, err)
                return
            }
            
            require.NoError(t, err)
            assert.Len(t, data.Points, tt.want)
        })
    }
}
```

### Table-Driven Tests

```go
func TestValidateTimeSeries(t *testing.T) {
    testCases := map[string]struct {
        input    *TimeSeries
        expected ValidationResult
    }{
        "valid series": {
            input: &TimeSeries{
                Points: []Point{{Time: time.Now(), Value: 25.5}},
            },
            expected: ValidationResult{Valid: true},
        },
        "empty series": {
            input:    &TimeSeries{},
            expected: ValidationResult{Valid: false, Error: "empty series"},
        },
    }

    for name, tc := range testCases {
        t.Run(name, func(t *testing.T) {
            result := ValidateTimeSeries(tc.input)
            assert.Equal(t, tc.expected, result)
        })
    }
}
```

## Integration Testing

### Database Integration Tests

```go
// +build integration

package storage_test

import (
    "context"
    "testing"
    "github.com/testcontainers/testcontainers-go"
    "github.com/testcontainers/testcontainers-go/wait"
)

func TestInfluxDBStorage_Integration(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping integration test")
    }

    ctx := context.Background()
    
    // Start InfluxDB container
    req := testcontainers.ContainerRequest{
        Image:        "influxdb:2.7",
        ExposedPorts: []string{"8086/tcp"},
        WaitingFor:   wait.ForHTTP("/health").WithPort("8086/tcp"),
        Env: map[string]string{
            "DOCKER_INFLUXDB_INIT_MODE":     "setup",
            "DOCKER_INFLUXDB_INIT_USERNAME": "admin",
            "DOCKER_INFLUXDB_INIT_PASSWORD": "password",
            "DOCKER_INFLUXDB_INIT_ORG":      "test",
            "DOCKER_INFLUXDB_INIT_BUCKET":   "test",
        },
    }

    container, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
        ContainerRequest: req,
        Started:          true,
    })
    require.NoError(t, err)
    defer container.Terminate(ctx)

    // Get container endpoint
    endpoint, err := container.Endpoint(ctx, "http")
    require.NoError(t, err)

    // Test storage operations
    storage := NewInfluxDBStorage(endpoint, "admin", "password")
    
    // Test write
    err = storage.Write(ctx, testData)
    assert.NoError(t, err)
    
    // Test read
    data, err := storage.Read(ctx, query)
    assert.NoError(t, err)
    assert.NotEmpty(t, data)
}
```

### API Integration Tests

```go
func TestAPIEndpoints(t *testing.T) {
    // Setup test server
    server := httptest.NewServer(router)
    defer server.Close()

    client := server.Client()

    t.Run("POST /generate", func(t *testing.T) {
        payload := `{
            "generator": "timegan",
            "sensor_type": "temperature",
            "points": 100
        }`
        
        resp, err := client.Post(
            server.URL+"/api/v1/generate",
            "application/json",
            strings.NewReader(payload),
        )
        require.NoError(t, err)
        assert.Equal(t, http.StatusOK, resp.StatusCode)
        
        var result GenerateResponse
        err = json.NewDecoder(resp.Body).Decode(&result)
        require.NoError(t, err)
        assert.NotEmpty(t, result.JobID)
    })
}
```

## End-to-End Testing

### Writing E2E Tests

```go
package e2e

import (
    "testing"
    "github.com/cucumber/godog"
)

func TestFeatures(t *testing.T) {
    suite := godog.TestSuite{
        ScenarioInitializer: InitializeScenario,
        Options: &godog.Options{
            Format:   "pretty",
            Paths:    []string{"features"},
            TestingT: t,
        },
    }

    if suite.Run() != 0 {
        t.Fatal("non-zero status returned, failed to run feature tests")
    }
}

func InitializeScenario(ctx *godog.ScenarioContext) {
    ctx.Step(`^I generate (\d+) points of "([^"]*)" data$`, iGenerateData)
    ctx.Step(`^I validate the generated data$`, iValidateData)
    ctx.Step(`^the validation score should be above (\d+)$`, validationScoreShouldBe)
}
```

### Feature Files

```gherkin
# features/data_generation.feature
Feature: Synthetic Data Generation
  As a user
  I want to generate synthetic time series data
  So that I can use it for testing and development

  Scenario: Generate temperature data
    Given I have configured the TimeGAN generator
    When I generate 1000 points of "temperature" data
    Then the data should have 1000 points
    And the values should be between -40 and 60
    And the temporal patterns should be preserved

  Scenario: Validate generated data quality
    Given I have generated synthetic temperature data
    When I validate the generated data
    Then the validation score should be above 80
    And the statistical properties should match the reference
```

## Performance Testing

### Benchmarks

```go
func BenchmarkTimeGANGeneration(b *testing.B) {
    gen := NewTimeGANGenerator()
    config := Config{
        SensorType: "temperature",
        Points:     1000,
    }

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, err := gen.Generate(config)
        if err != nil {
            b.Fatal(err)
        }
    }
}

func BenchmarkParallelGeneration(b *testing.B) {
    gen := NewTimeGANGenerator()
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            _, err := gen.Generate(defaultConfig)
            if err != nil {
                b.Fatal(err)
            }
        }
    })
}
```

### Load Testing

```go
func TestHighLoad(t *testing.T) {
    server := StartTestServer()
    defer server.Close()

    // Configure load test
    attack := vegeta.NewAttacker()
    rate := vegeta.Rate{Freq: 100, Per: time.Second}
    duration := 30 * time.Second
    
    targeter := vegeta.NewStaticTargeter(vegeta.Target{
        Method: "POST",
        URL:    server.URL + "/api/v1/generate",
        Body:   []byte(`{"generator":"statistical","points":100}`),
        Header: http.Header{"Content-Type": []string{"application/json"}},
    })

    var metrics vegeta.Metrics
    for res := range attack.Attack(targeter, rate, duration, "Load Test") {
        metrics.Add(res)
    }
    metrics.Close()

    // Assert performance requirements
    assert.Less(t, metrics.Latencies.P95, 500*time.Millisecond)
    assert.Greater(t, metrics.Success, 0.99)
}
```

## Test Coverage

### Generating Coverage Reports

```bash
# Generate coverage report
go test -coverprofile=coverage.out ./...

# View coverage in terminal
go tool cover -func=coverage.out

# Generate HTML report
go tool cover -html=coverage.out -o coverage.html

# Check coverage threshold
go test -cover ./... | grep -E "coverage: [0-9]+\.[0-9]+%" | \
  awk '{print $2}' | sed 's/%//' | \
  awk '{if ($1 < 80) exit 1}'
```

### Coverage Configuration

```yaml
# .codecov.yml
coverage:
  precision: 2
  round: down
  range: "70...100"
  status:
    project:
      default:
        target: 80%
        threshold: 5%
    patch:
      default:
        target: 80%
```

## Mocking and Test Doubles

### Using Mockery

```bash
# Generate mocks
mockery --all --dir internal/storage --output internal/mocks
```

### Writing Tests with Mocks

```go
func TestGeneratorService_Generate(t *testing.T) {
    mockStorage := new(mocks.StorageBackend)
    mockValidator := new(mocks.Validator)
    
    service := NewGeneratorService(mockStorage, mockValidator)

    // Setup expectations
    mockStorage.On("Save", mock.Anything, mock.Anything).Return(nil)
    mockValidator.On("Validate", mock.Anything).Return(&ValidationResult{
        Valid: true,
        Score: 0.95,
    }, nil)

    // Test
    result, err := service.Generate(context.Background(), config)
    
    // Assertions
    assert.NoError(t, err)
    assert.NotNil(t, result)
    mockStorage.AssertExpectations(t)
    mockValidator.AssertExpectations(t)
}
```

### Custom Test Doubles

```go
type fakeTimeSeriesStore struct {
    data map[string]*TimeSeries
    mu   sync.RWMutex
}

func newFakeStore() *fakeTimeSeriesStore {
    return &fakeTimeSeriesStore{
        data: make(map[string]*TimeSeries),
    }
}

func (s *fakeTimeSeriesStore) Save(id string, ts *TimeSeries) error {
    s.mu.Lock()
    defer s.mu.Unlock()
    s.data[id] = ts
    return nil
}
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      influxdb:
        image: influxdb:2.7
        ports:
          - 8086:8086
        env:
          DOCKER_INFLUXDB_INIT_MODE: setup
          DOCKER_INFLUXDB_INIT_USERNAME: admin
          DOCKER_INFLUXDB_INIT_PASSWORD: password
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Go
      uses: actions/setup-go@v4
      with:
        go-version: '1.21'
    
    - name: Install dependencies
      run: go mod download
    
    - name: Run tests
      run: |
        go test -v -race -coverprofile=coverage.out ./...
        go tool cover -html=coverage.out -o coverage.html
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.out
```

## Best Practices

### 1. Test Naming
```go
// Good test names
func TestGenerator_Generate_WithValidConfig_ReturnsData(t *testing.T)
func TestValidator_Validate_WithEmptyData_ReturnsError(t *testing.T)

// Use descriptive sub-test names
t.Run("with valid temperature data", func(t *testing.T) {
    // test code
})
```

### 2. Test Independence
- Each test should be independent and not rely on others
- Use setup and teardown functions
- Clean up resources after tests

```go
func TestWithCleanup(t *testing.T) {
    resource := setupResource(t)
    t.Cleanup(func() {
        resource.Close()
    })
    
    // test code
}
```

### 3. Parallel Testing
```go
func TestParallel(t *testing.T) {
    t.Parallel() // Mark test as safe for parallel execution
    
    // test code
}
```

### 4. Test Helpers
```go
func assertTimeSeriesEqual(t *testing.T, expected, actual *TimeSeries) {
    t.Helper() // Mark as test helper
    
    if !reflect.DeepEqual(expected, actual) {
        t.Errorf("TimeSeries not equal\nexpected: %+v\nactual: %+v", 
            expected, actual)
    }
}
```

### 5. Golden Files
```go
func TestGoldenFile(t *testing.T) {
    actual := generateOutput()
    golden := filepath.Join("testdata", "golden.json")
    
    if *update {
        ioutil.WriteFile(golden, actual, 0644)
    }
    
    expected, _ := ioutil.ReadFile(golden)
    assert.Equal(t, expected, actual)
}
```

### 6. Test Timeouts
```go
func TestWithTimeout(t *testing.T) {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    done := make(chan bool)
    go func() {
        // long running operation
        done <- true
    }()
    
    select {
    case <-done:
        // success
    case <-ctx.Done():
        t.Fatal("test timeout")
    }
}
```

### 7. Environment Variables
```go
func TestWithEnv(t *testing.T) {
    oldVal := os.Getenv("TEST_VAR")
    os.Setenv("TEST_VAR", "test_value")
    t.Cleanup(func() {
        os.Setenv("TEST_VAR", oldVal)
    })
    
    // test code
}
```

## Testing Checklist

Before submitting code:

- [ ] All tests pass locally
- [ ] New features have corresponding tests
- [ ] Test coverage meets minimum threshold (80%)
- [ ] Integration tests pass with real dependencies
- [ ] No flaky tests
- [ ] Performance benchmarks show no regression
- [ ] Tests are documented and easy to understand
- [ ] CI/CD pipeline passes

## Resources

- [Go Testing Documentation](https://golang.org/pkg/testing/)
- [Testify Framework](https://github.com/stretchr/testify)
- [Testcontainers Go](https://golang.testcontainers.org/)
- [GoMock](https://github.com/golang/mock)
- [Godog BDD](https://github.com/cucumber/godog)