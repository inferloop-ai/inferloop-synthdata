# Go SDK Guide

The TSIoT Go SDK provides a native Go interface for generating and validating synthetic time series data.

## Installation

```bash
go get github.com/inferloop/tsiot-go-sdk
```

### Requirements

- Go 1.19+
- No external dependencies (uses only standard library)

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"
    
    "github.com/inferloop/tsiot-go-sdk"
)

func main() {
    // Create client
    client := tsiot.NewClient("http://localhost:8080")
    
    // Generate synthetic data
    req := &tsiot.GenerateRequest{
        Generator:  "statistical",
        SensorType: "temperature",
        Points:     1000,
        Frequency:  "1m",
    }
    
    ctx := context.Background()
    data, err := client.Generate(ctx, req)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Generated %d data points\n", len(data.Points))
    
    // Save to file
    err = data.SaveCSV("temperature_data.csv")
    if err != nil {
        log.Fatal(err)
    }
}
```

## API Reference

### Client

```go
type Client struct {
    baseURL    string
    httpClient *http.Client
    apiKey     string
}

// NewClient creates a new TSIoT client
func NewClient(baseURL string, options ...ClientOption) *Client

// NewClientWithAuth creates a client with API key authentication
func NewClientWithAuth(baseURL, apiKey string, options ...ClientOption) *Client
```

#### Client Options

```go
type ClientOption func(*Client)

// WithTimeout sets the HTTP client timeout
func WithTimeout(timeout time.Duration) ClientOption

// WithHTTPClient sets a custom HTTP client
func WithHTTPClient(client *http.Client) ClientOption

// WithUserAgent sets a custom user agent
func WithUserAgent(userAgent string) ClientOption
```

#### Examples

```go
// Basic client
client := tsiot.NewClient("http://localhost:8080")

// With authentication
client := tsiot.NewClientWithAuth(
    "https://tsiot.example.com",
    "your-api-key",
)

// With custom timeout
client := tsiot.NewClient(
    "http://localhost:8080",
    tsiot.WithTimeout(60*time.Second),
)

// With custom HTTP client
httpClient := &http.Client{
    Transport: &http.Transport{
        MaxIdleConns: 100,
    },
    Timeout: 30 * time.Second,
}
client := tsiot.NewClient(
    "http://localhost:8080",
    tsiot.WithHTTPClient(httpClient),
)
```

### Generate Data

```go
type GenerateRequest struct {
    Generator        string                 `json:"generator"`
    SensorType       string                 `json:"sensor_type"`
    SensorID         string                 `json:"sensor_id,omitempty"`
    Points           int                    `json:"points"`
    StartTime        *time.Time             `json:"start_time,omitempty"`
    EndTime          *time.Time             `json:"end_time,omitempty"`
    Duration         string                 `json:"duration,omitempty"`
    Frequency        string                 `json:"frequency"`
    Anomalies        int                    `json:"anomalies,omitempty"`
    AnomalyMagnitude float64                `json:"anomaly_magnitude,omitempty"`
    NoiseLevel       float64                `json:"noise_level,omitempty"`
    Trend            float64                `json:"trend,omitempty"`
    Seasonality      bool                   `json:"seasonality,omitempty"`
    SeasonalPeriod   string                 `json:"seasonal_period,omitempty"`
    Seed             *int64                 `json:"seed,omitempty"`
    Privacy          bool                   `json:"privacy,omitempty"`
    PrivacyEpsilon   float64                `json:"privacy_epsilon,omitempty"`
    CustomParams     map[string]interface{} `json:"custom_params,omitempty"`
}

func (c *Client) Generate(ctx context.Context, req *GenerateRequest) (*TimeSeriesData, error)
```

#### Examples

```go
// Basic generation
req := &tsiot.GenerateRequest{
    Generator:  "statistical",
    SensorType: "temperature",
    Points:     1000,
    Frequency:  "1m",
}
data, err := client.Generate(ctx, req)

// Time range generation
startTime := time.Now().Add(-7 * 24 * time.Hour)
endTime := time.Now()

req := &tsiot.GenerateRequest{
    Generator:  "arima",
    SensorType: "humidity",
    StartTime:  &startTime,
    EndTime:    &endTime,
    Frequency:  "5m",
}
data, err := client.Generate(ctx, req)

// With anomalies and seasonality
seed := int64(42)
req := &tsiot.GenerateRequest{
    Generator:        "timegan",
    SensorType:       "energy",
    Points:           10000,
    Anomalies:        20,
    AnomalyMagnitude: 2.5,
    Seasonality:      true,
    SeasonalPeriod:   "24h",
    Seed:             &seed,
}
data, err := client.Generate(ctx, req)

// Privacy-preserving
req := &tsiot.GenerateRequest{
    Generator:      "ydata",
    SensorType:     "medical",
    Points:         5000,
    Privacy:        true,
    PrivacyEpsilon: 0.5,
}
data, err := client.Generate(ctx, req)
```

### Validate Data

```go
type ValidateRequest struct {
    Data             *TimeSeriesData `json:"data,omitempty"`
    DataPath         string          `json:"data_path,omitempty"`
    Reference        *TimeSeriesData `json:"reference,omitempty"`
    ReferencePath    string          `json:"reference_path,omitempty"`
    Metrics          []string        `json:"metrics,omitempty"`
    StatisticalTests []string        `json:"statistical_tests,omitempty"`
    QualityThreshold float64         `json:"quality_threshold,omitempty"`
}

type ValidationResult struct {
    QualityScore     float64            `json:"quality_score"`
    Metrics          map[string]float64 `json:"metrics"`
    StatisticalTests map[string]TestResult `json:"statistical_tests"`
    Anomalies        []Anomaly          `json:"anomalies"`
    Recommendations  []string           `json:"recommendations"`
    Passed           bool               `json:"passed"`
}

func (c *Client) Validate(ctx context.Context, req *ValidateRequest) (*ValidationResult, error)
```

#### Examples

```go
// Basic validation
req := &tsiot.ValidateRequest{
    Data: data,
}
result, err := client.Validate(ctx, req)
fmt.Printf("Quality score: %.2f\n", result.QualityScore)

// Compare with reference
referenceData, _ := tsiot.LoadFromCSV("real_data.csv")
req := &tsiot.ValidateRequest{
    Data:      syntheticData,
    Reference: referenceData,
    Metrics:   []string{"distribution", "temporal", "similarity"},
}
result, err := client.Validate(ctx, req)

// Validate from file
req := &tsiot.ValidateRequest{
    DataPath:         "synthetic_data.csv",
    ReferencePath:    "real_data.csv",
    StatisticalTests: []string{"ks", "anderson", "ljung_box"},
}
result, err := client.Validate(ctx, req)
```

### Analyze Data

```go
type AnalyzeRequest struct {
    Data            *TimeSeriesData `json:"data,omitempty"`
    DataPath        string          `json:"data_path,omitempty"`
    DetectAnomalies bool            `json:"detect_anomalies,omitempty"`
    AnomalyMethod   string          `json:"anomaly_method,omitempty"`
    Decompose       bool            `json:"decompose,omitempty"`
    Forecast        bool            `json:"forecast,omitempty"`
    ForecastPeriods int             `json:"forecast_periods,omitempty"`
    Patterns        bool            `json:"patterns,omitempty"`
}

type AnalysisResult struct {
    SummaryStats  map[string]float64 `json:"summary_stats"`
    Anomalies     []Anomaly          `json:"anomalies"`
    Decomposition *Decomposition     `json:"decomposition"`
    Forecast      *Forecast          `json:"forecast"`
    Patterns      []Pattern          `json:"patterns"`
}

func (c *Client) Analyze(ctx context.Context, req *AnalyzeRequest) (*AnalysisResult, error)
```

#### Examples

```go
// Basic analysis
req := &tsiot.AnalyzeRequest{
    Data: data,
}
result, err := client.Analyze(ctx, req)
fmt.Printf("Mean: %.2f, Std: %.2f\n", 
    result.SummaryStats["mean"], 
    result.SummaryStats["std"])

// Anomaly detection
req := &tsiot.AnalyzeRequest{
    Data:            sensorData,
    DetectAnomalies: true,
    AnomalyMethod:   "isolation_forest",
}
result, err := client.Analyze(ctx, req)
fmt.Printf("Found %d anomalies\n", len(result.Anomalies))

// Time series decomposition and forecasting
req := &tsiot.AnalyzeRequest{
    Data:            historicalData,
    Decompose:       true,
    Forecast:        true,
    ForecastPeriods: 24,
}
result, err := client.Analyze(ctx, req)
```

### Stream Data

```go
type StreamRequest struct {
    Generator    string                 `json:"generator"`
    SensorType   string                 `json:"sensor_type"`
    SensorID     string                 `json:"sensor_id,omitempty"`
    Frequency    string                 `json:"frequency"`
    CustomParams map[string]interface{} `json:"custom_params,omitempty"`
}

func (c *Client) Stream(ctx context.Context, req *StreamRequest) (<-chan *DataPoint, <-chan error)
```

#### Examples

```go
// Basic streaming
req := &tsiot.StreamRequest{
    Generator:  "statistical",
    SensorType: "temperature",
    Frequency:  "5s",
}

dataCh, errCh := client.Stream(ctx, req)

for {
    select {
    case point := <-dataCh:
        if point == nil {
            return // Stream closed
        }
        fmt.Printf("%s: %.2f\n", point.Timestamp.Format(time.RFC3339), point.Value)
    case err := <-errCh:
        if err != nil {
            log.Printf("Stream error: %v", err)
        }
    case <-ctx.Done():
        return
    }
}

// With goroutine handler
go func() {
    for point := range dataCh {
        processPoint(point)
    }
}()

go func() {
    for err := range errCh {
        if err != nil {
            log.Printf("Error: %v", err)
        }
    }
}()
```

## Data Types

### TimeSeriesData

```go
type TimeSeriesData struct {
    Points   []DataPoint            `json:"points"`
    Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// Methods
func (ts *TimeSeriesData) Len() int
func (ts *TimeSeriesData) Timestamps() []time.Time
func (ts *TimeSeriesData) Values() []float64
func (ts *TimeSeriesData) SensorIDs() []string
func (ts *TimeSeriesData) Filter(start, end time.Time) *TimeSeriesData
func (ts *TimeSeriesData) Resample(frequency string, method string) (*TimeSeriesData, error)
func (ts *TimeSeriesData) AddNoise(level float64) *TimeSeriesData
func (ts *TimeSeriesData) Normalize(method string) *TimeSeriesData
func (ts *TimeSeriesData) SaveCSV(filename string) error
func (ts *TimeSeriesData) SaveJSON(filename string) error
func (ts *TimeSeriesData) SaveParquet(filename string) error
```

### DataPoint

```go
type DataPoint struct {
    Timestamp  time.Time              `json:"timestamp"`
    SensorID   string                 `json:"sensor_id"`
    SensorType string                 `json:"sensor_type"`
    Value      float64                `json:"value"`
    Unit       string                 `json:"unit,omitempty"`
    Metadata   map[string]interface{} `json:"metadata,omitempty"`
}
```

### ValidationResult Types

```go
type TestResult struct {
    Statistic float64 `json:"statistic"`
    PValue    float64 `json:"p_value"`
    Passed    bool    `json:"passed"`
    Critical  float64 `json:"critical,omitempty"`
}

type Anomaly struct {
    Index     int     `json:"index"`
    Timestamp time.Time `json:"timestamp"`
    Value     float64 `json:"value"`
    Score     float64 `json:"score"`
    Type      string  `json:"type"`
}
```

### Analysis Types

```go
type Decomposition struct {
    Trend      []float64 `json:"trend"`
    Seasonal   []float64 `json:"seasonal"`
    Residual   []float64 `json:"residual"`
    Period     int       `json:"period"`
}

type Forecast struct {
    Values      []float64 `json:"values"`
    Timestamps  []time.Time `json:"timestamps"`
    ConfidenceL []float64 `json:"confidence_lower"`
    ConfidenceU []float64 `json:"confidence_upper"`
    Method      string    `json:"method"`
}

type Pattern struct {
    Type        string    `json:"type"`
    StartIndex  int       `json:"start_index"`
    EndIndex    int       `json:"end_index"`
    Score       float64   `json:"score"`
    Parameters  map[string]interface{} `json:"parameters"`
}
```

## Advanced Usage

### Concurrent Generation

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
    
    "github.com/inferloop/tsiot-go-sdk"
)

func main() {
    client := tsiot.NewClient("http://localhost:8080")
    ctx := context.Background()
    
    // Generate data for multiple sensors concurrently
    sensorIDs := []string{"sensor_001", "sensor_002", "sensor_003"}
    var wg sync.WaitGroup
    results := make([]*tsiot.TimeSeriesData, len(sensorIDs))
    
    for i, sensorID := range sensorIDs {
        wg.Add(1)
        go func(i int, sensorID string) {
            defer wg.Done()
            
            req := &tsiot.GenerateRequest{
                Generator:  "arima",
                SensorType: "temperature",
                SensorID:   sensorID,
                Points:     1440, // 24 hours of minute data
                Frequency:  "1m",
            }
            
            data, err := client.Generate(ctx, req)
            if err != nil {
                fmt.Printf("Error generating data for %s: %v\n", sensorID, err)
                return
            }
            
            results[i] = data
        }(i, sensorID)
    }
    
    wg.Wait()
    
    // Combine all data
    allData := &tsiot.TimeSeriesData{}
    for _, data := range results {
        if data != nil {
            allData.Points = append(allData.Points, data.Points...)
        }
    }
    
    fmt.Printf("Generated %d total data points\n", len(allData.Points))
    allData.SaveParquet("all_sensors.parquet")
}
```

### Custom HTTP Transport

```go
import (
    "crypto/tls"
    "net/http"
    "time"
)

// Custom transport for specific requirements
transport := &http.Transport{
    TLSClientConfig: &tls.Config{
        InsecureSkipVerify: true, // Only for development
    },
    MaxIdleConns:        100,
    MaxIdleConnsPerHost: 10,
    IdleConnTimeout:     90 * time.Second,
}

httpClient := &http.Client{
    Transport: transport,
    Timeout:   60 * time.Second,
}

client := tsiot.NewClient(
    "https://tsiot.example.com",
    tsiot.WithHTTPClient(httpClient),
)
```

### Real-time Processing Pipeline

```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"
    
    "github.com/inferloop/tsiot-go-sdk"
)

type Processor struct {
    client *tsiot.Client
    buffer []float64
    window int
}

func NewProcessor(client *tsiot.Client, windowSize int) *Processor {
    return &Processor{
        client: client,
        buffer: make([]float64, 0, windowSize),
        window: windowSize,
    }
}

func (p *Processor) ProcessStream(ctx context.Context) error {
    req := &tsiot.StreamRequest{
        Generator:  "arima",
        SensorType: "temperature",
        Frequency:  "1s",
    }
    
    dataCh, errCh := p.client.Stream(ctx, req)
    
    for {
        select {
        case point := <-dataCh:
            if point == nil {
                return nil // Stream closed
            }
            
            p.buffer = append(p.buffer, point.Value)
            if len(p.buffer) > p.window {
                p.buffer = p.buffer[1:] // Sliding window
            }
            
            if len(p.buffer) == p.window {
                if err := p.analyzeWindow(ctx); err != nil {
                    log.Printf("Analysis error: %v", err)
                }
            }
            
        case err := <-errCh:
            if err != nil {
                return err
            }
            
        case <-ctx.Done():
            return ctx.Err()
        }
    }
}

func (p *Processor) analyzeWindow(ctx context.Context) error {
    // Convert buffer to TimeSeriesData
    points := make([]tsiot.DataPoint, len(p.buffer))
    baseTime := time.Now().Add(-time.Duration(len(p.buffer)) * time.Second)
    
    for i, value := range p.buffer {
        points[i] = tsiot.DataPoint{
            Timestamp:  baseTime.Add(time.Duration(i) * time.Second),
            SensorID:   "real_time",
            SensorType: "temperature",
            Value:      value,
        }
    }
    
    data := &tsiot.TimeSeriesData{Points: points}
    
    // Analyze for anomalies
    req := &tsiot.AnalyzeRequest{
        Data:            data,
        DetectAnomalies: true,
        AnomalyMethod:   "isolation_forest",
    }
    
    result, err := p.client.Analyze(ctx, req)
    if err != nil {
        return err
    }
    
    if len(result.Anomalies) > 0 {
        fmt.Printf("Detected %d anomalies in current window\n", len(result.Anomalies))
        for _, anomaly := range result.Anomalies {
            fmt.Printf("  Anomaly at %s: value=%.2f, score=%.2f\n",
                anomaly.Timestamp.Format(time.RFC3339),
                anomaly.Value,
                anomaly.Score)
        }
    }
    
    return nil
}

func main() {
    client := tsiot.NewClient("http://localhost:8080")
    processor := NewProcessor(client, 100) // 100-point sliding window
    
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
    defer cancel()
    
    if err := processor.ProcessStream(ctx); err != nil {
        log.Fatal(err)
    }
}
```

### Error Handling

```go
import (
    "errors"
    "net/http"
    
    "github.com/inferloop/tsiot-go-sdk"
)

func handleErrors() {
    client := tsiot.NewClient("http://localhost:8080")
    
    req := &tsiot.GenerateRequest{
        Generator:  "invalid_generator",
        SensorType: "temperature",
        Points:     1000,
    }
    
    data, err := client.Generate(context.Background(), req)
    if err != nil {
        // Check for specific error types
        var apiErr *tsiot.APIError
        if errors.As(err, &apiErr) {
            fmt.Printf("API Error: %s (code: %d)\n", apiErr.Message, apiErr.Code)
            
            switch apiErr.Code {
            case http.StatusBadRequest:
                fmt.Println("Invalid request parameters")
            case http.StatusUnauthorized:
                fmt.Println("Authentication required")
            case http.StatusTooManyRequests:
                fmt.Println("Rate limit exceeded")
            default:
                fmt.Printf("Server error: %s\n", apiErr.Message)
            }
            return
        }
        
        var netErr *tsiot.NetworkError
        if errors.As(err, &netErr) {
            fmt.Printf("Network error: %v\n", netErr)
            // Implement retry logic
            return
        }
        
        fmt.Printf("Unknown error: %v\n", err)
        return
    }
    
    fmt.Printf("Success: generated %d points\n", len(data.Points))
}
```

### Configuration Management

```go
package main

import (
    "os"
    "time"
    
    "github.com/inferloop/tsiot-go-sdk"
)

type Config struct {
    ServerURL   string        `yaml:"server_url"`
    APIKey      string        `yaml:"api_key"`
    Timeout     time.Duration `yaml:"timeout"`
    MaxRetries  int          `yaml:"max_retries"`
    DefaultGen  string       `yaml:"default_generator"`
}

func loadConfig() *Config {
    return &Config{
        ServerURL:  getEnvOrDefault("TSIOT_SERVER_URL", "http://localhost:8080"),
        APIKey:     os.Getenv("TSIOT_API_KEY"),
        Timeout:    30 * time.Second,
        MaxRetries: 3,
        DefaultGen: "statistical",
    }
}

func getEnvOrDefault(key, defaultValue string) string {
    if value := os.Getenv(key); value != "" {
        return value
    }
    return defaultValue
}

func main() {
    config := loadConfig()
    
    var client *tsiot.Client
    if config.APIKey != "" {
        client = tsiot.NewClientWithAuth(
            config.ServerURL,
            config.APIKey,
            tsiot.WithTimeout(config.Timeout),
        )
    } else {
        client = tsiot.NewClient(
            config.ServerURL,
            tsiot.WithTimeout(config.Timeout),
        )
    }
    
    // Use client...
}
```

## Testing

### Mock Client

```go
package main

import (
    "context"
    "testing"
    "time"
    
    "github.com/inferloop/tsiot-go-sdk"
    "github.com/inferloop/tsiot-go-sdk/mock"
)

func TestDataGeneration(t *testing.T) {
    // Create mock client
    mockClient := mock.NewClient()
    
    // Set up mock response
    expectedData := &tsiot.TimeSeriesData{
        Points: []tsiot.DataPoint{
            {
                Timestamp:  time.Now(),
                SensorID:   "test_sensor",
                SensorType: "temperature",
                Value:      25.0,
            },
        },
    }
    
    mockClient.OnGenerate(func(req *tsiot.GenerateRequest) (*tsiot.TimeSeriesData, error) {
        if req.Points != 1 {
            t.Errorf("Expected 1 point, got %d", req.Points)
        }
        return expectedData, nil
    })
    
    // Test your code
    req := &tsiot.GenerateRequest{
        Generator:  "statistical",
        SensorType: "temperature",
        Points:     1,
    }
    
    data, err := mockClient.Generate(context.Background(), req)
    if err != nil {
        t.Fatalf("Unexpected error: %v", err)
    }
    
    if len(data.Points) != 1 {
        t.Errorf("Expected 1 data point, got %d", len(data.Points))
    }
}
```

### Integration Tests

```go
func TestIntegration(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping integration test")
    }
    
    client := tsiot.NewClient("http://localhost:8080")
    ctx := context.Background()
    
    // Test server health
    if err := client.Health(ctx); err != nil {
        t.Fatalf("Server not healthy: %v", err)
    }
    
    // Test generation
    req := &tsiot.GenerateRequest{
        Generator:  "statistical",
        SensorType: "temperature",
        Points:     100,
    }
    
    data, err := client.Generate(ctx, req)
    if err != nil {
        t.Fatalf("Generation failed: %v", err)
    }
    
    if len(data.Points) != 100 {
        t.Errorf("Expected 100 points, got %d", len(data.Points))
    }
    
    // Test validation
    valReq := &tsiot.ValidateRequest{Data: data}
    result, err := client.Validate(ctx, valReq)
    if err != nil {
        t.Fatalf("Validation failed: %v", err)
    }
    
    if result.QualityScore < 0.7 {
        t.Errorf("Quality score too low: %.2f", result.QualityScore)
    }
}
```

## Examples

See the [examples directory](../../../examples/go-sdk/) for complete examples:

- [Basic Usage](../../../examples/go-sdk/basic/main.go)
- [Advanced Features](../../../examples/go-sdk/advanced/main.go)
- [Concurrent Generation](../../../examples/go-sdk/concurrent/main.go)
- [Real-time Streaming](../../../examples/go-sdk/streaming/main.go)

## Support

- **Documentation**: https://docs.inferloop.com/tsiot/go-sdk
- **GitHub Issues**: https://github.com/inferloop/tsiot-go-sdk/issues
- **Go Package**: https://pkg.go.dev/github.com/inferloop/tsiot-go-sdk
- **Examples**: https://github.com/inferloop/tsiot/tree/main/examples/go-sdk