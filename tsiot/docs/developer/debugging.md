# TSIoT Debugging Guide

This comprehensive guide covers debugging techniques, tools, and best practices for the TSIoT (Time Series IoT) synthetic data generation platform. Whether you're troubleshooting performance issues, investigating bugs, or analyzing system behavior, this guide will help you effectively debug the system.

## Table of Contents

- [Debugging Environment Setup](#debugging-environment-setup)
- [Logging and Observability](#logging-and-observability)
- [Interactive Debugging](#interactive-debugging)
- [Performance Debugging](#performance-debugging)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Production Debugging](#production-debugging)
- [Testing and Validation](#testing-and-validation)
- [Tools and Utilities](#tools-and-utilities)

## Debugging Environment Setup

### Development Environment Configuration

```yaml
# configs/environments/debug.yaml
server:
  host: "localhost"
  port: 8080
  enable_pprof: true
  pprof_port: 6060

logging:
  level: "debug"
  format: "pretty"  # Human-readable for development
  enable_caller: true
  enable_stacktrace: true

metrics:
  enabled: true
  port: 9090
  path: "/metrics"

tracing:
  enabled: true
  jaeger_endpoint: "http://localhost:14268/api/traces"
  sample_rate: 1.0  # 100% sampling for debugging

database:
  log_queries: true
  slow_query_threshold: "100ms"
```

### Environment Variables for Debugging

```bash
# Enable debug mode
export TSIOT_DEBUG=true
export TSIOT_LOG_LEVEL=debug
export TSIOT_PROFILE=true

# Enable race detection
export TSIOT_RACE_DETECTOR=true

# Database debugging
export TSIOT_DB_DEBUG=true
export TSIOT_DB_LOG_QUERIES=true

# Memory debugging
export GOMEMLIMIT=4GiB
export GODEBUG=gctrace=1
```

### Build Flags for Debugging

```bash
# Build with debugging symbols
go build -race -ldflags "-X main.buildMode=debug" cmd/server/main.go

# Build with memory sanitizer
go build -msan cmd/server/main.go

# Build with coverage
go build -cover cmd/server/main.go
```

## Logging and Observability

### Structured Logging Setup

```go
// internal/observability/logging/logger.go
package logging

import (
    "context"
    "log/slog"
    "os"
    "runtime"
)

func NewDebugLogger() *slog.Logger {
    opts := &slog.HandlerOptions{
        Level: slog.LevelDebug,
        AddSource: true,
        ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
            // Add custom attributes
            if a.Key == slog.SourceKey {
                source := a.Value.Any().(*slog.Source)
                source.File = filepath.Base(source.File)
            }
            return a
        },
    }
    
    handler := slog.NewJSONHandler(os.Stdout, opts)
    return slog.New(handler)
}

// Context-aware logging
func WithContext(ctx context.Context, logger *slog.Logger) *slog.Logger {
    if requestID := getRequestID(ctx); requestID != "" {
        logger = logger.With("request_id", requestID)
    }
    
    if userID := getUserID(ctx); userID != "" {
        logger = logger.With("user_id", userID)
    }
    
    // Add goroutine ID for debugging
    logger = logger.With("goroutine_id", getGoroutineID())
    
    return logger
}

func getGoroutineID() int {
    var buf [64]byte
    n := runtime.Stack(buf[:], false)
    // Parse goroutine ID from stack trace
    return parseGoroutineID(string(buf[:n]))
}
```

### Request Tracing

```go
// Trace generation requests
func (g *Generator) Generate(ctx context.Context, params *GenerationParams) (*Data, error) {
    span := trace.SpanFromContext(ctx)
    span.SetAttributes(
        attribute.String("generator.type", g.Type()),
        attribute.Int("generator.data_points", params.DataPoints),
        attribute.StringSlice("generator.features", params.Features),
    )
    
    logger := logging.FromContext(ctx)
    logger.Debug("starting data generation",
        "generator_type", g.Type(),
        "params", params,
    )
    
    start := time.Now()
    defer func() {
        duration := time.Since(start)
        span.SetAttributes(attribute.Int64("generator.duration_ms", duration.Milliseconds()))
        
        logger.Debug("completed data generation",
            "duration", duration.String(),
            "success", err == nil,
        )
    }()
    
    // Generation logic with detailed logging
    if err := g.validateParams(params); err != nil {
        span.RecordError(err)
        logger.Error("parameter validation failed", "error", err)
        return nil, fmt.Errorf("invalid parameters: %w", err)
    }
    
    data, err := g.generateInternal(ctx, params)
    if err != nil {
        span.RecordError(err)
        logger.Error("generation failed", "error", err)
        return nil, err
    }
    
    logger.Info("generation completed successfully",
        "generated_points", len(data.Values),
        "memory_used", getMemoryUsage(),
    )
    
    return data, nil
}
```

### Log Analysis Tools

```bash
# Real-time log monitoring
tail -f logs/tsiot.log | jq .

# Filter logs by level
cat logs/tsiot.log | jq 'select(.level == "ERROR")'

# Find logs by request ID
cat logs/tsiot.log | jq 'select(.request_id == "req-123")'

# Performance analysis
cat logs/tsiot.log | jq 'select(.msg == "completed data generation") | .duration' | sort -n

# Error analysis
cat logs/tsiot.log | jq 'select(.level == "ERROR") | .error' | sort | uniq -c
```

## Interactive Debugging

### Delve Debugger Setup

```bash
# Install delve
go install github.com/go-delve/delve/cmd/dlv@latest

# Debug server
dlv debug cmd/server/main.go -- --config configs/environments/debug.yaml

# Debug with arguments
dlv debug cmd/cli/main.go -- generate --generator timegan --output debug.csv

# Attach to running process
dlv attach $(pgrep tsiot-server)

# Remote debugging
dlv debug --headless --listen=:2345 --api-version=2 cmd/server/main.go
```

### Common Delve Commands

```bash
# Set breakpoints
(dlv) break main.main
(dlv) break internal/generators/timegan.(*Generator).Generate
(dlv) break /path/to/file.go:123

# Conditional breakpoints
(dlv) break main.go:50 if requestID == "req-123"

# Execution control
(dlv) continue
(dlv) next
(dlv) step
(dlv) stepout

# Inspect variables
(dlv) print variableName
(dlv) print -v complexStruct
(dlv) locals
(dlv) args

# Stack trace
(dlv) bt
(dlv) up
(dlv) down

# Goroutines
(dlv) goroutines
(dlv) goroutine 15 bt
(dlv) goroutine 15 locals
```

### VS Code Debugging Configuration

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Server",
            "type": "go",
            "request": "launch",
            "mode": "debug",
            "program": "cmd/server/main.go",
            "args": [
                "--config", "configs/environments/debug.yaml"
            ],
            "env": {
                "TSIOT_DEBUG": "true",
                "TSIOT_LOG_LEVEL": "debug"
            },
            "cwd": "${workspaceFolder}",
            "showLog": true
        },
        {
            "name": "Debug CLI",
            "type": "go",
            "request": "launch",
            "mode": "debug",
            "program": "cmd/cli/main.go",
            "args": [
                "generate",
                "--generator", "timegan",
                "--debug"
            ]
        },
        {
            "name": "Debug Tests",
            "type": "go",
            "request": "launch",
            "mode": "test",
            "program": "${workspaceFolder}/internal/generators/timegan",
            "args": [
                "-test.v",
                "-test.run", "TestGenerate"
            ]
        }
    ]
}
```

### GoLand Debugging Configuration

```yaml
# Run/Debug Configurations
Server Debug:
  - Program: cmd/server/main.go
  - Arguments: --config configs/environments/debug.yaml
  - Environment:
      TSIOT_DEBUG: true
      TSIOT_LOG_LEVEL: debug
  - Working directory: $PROJECT_DIR$
  - Build tags: debug
  - Before launch: Build, Generate code
```

## Performance Debugging

### CPU Profiling

```go
// Enable CPU profiling in main.go
func main() {
    if os.Getenv("TSIOT_PROFILE") == "true" {
        f, err := os.Create("cpu.prof")
        if err != nil {
            log.Fatal(err)
        }
        defer f.Close()
        
        if err := pprof.StartCPUProfile(f); err != nil {
            log.Fatal(err)
        }
        defer pprof.StopCPUProfile()
    }
    
    // Application code
    startServer()
}
```

```bash
# Collect CPU profile
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30

# Analyze profile
(pprof) top 10
(pprof) list GenerateTimeSeries
(pprof) web
(pprof) peek GenerateTimeSeries

# Generate flame graph
go tool pprof -http=:8080 cpu.prof
```

### Memory Profiling

```bash
# Heap analysis
go tool pprof http://localhost:6060/debug/pprof/heap

# Memory allocation analysis
go tool pprof http://localhost:6060/debug/pprof/allocs

# Memory leak detection
go tool pprof -base heap1.prof heap2.prof

# Interactive analysis
(pprof) top -cum
(pprof) list problematicFunction
(pprof) peek problematicFunction
```

### Goroutine Analysis

```bash
# Check for goroutine leaks
curl http://localhost:6060/debug/pprof/goroutine?debug=1

# Goroutine profiling
go tool pprof http://localhost:6060/debug/pprof/goroutine

# Blocking profile
go tool pprof http://localhost:6060/debug/pprof/block

# Mutex contention
go tool pprof http://localhost:6060/debug/pprof/mutex
```

### Custom Performance Metrics

```go
// Performance monitoring
type PerformanceMonitor struct {
    startTime   time.Time
    operations  map[string]time.Duration
    memory      map[string]uint64
    mutex       sync.RWMutex
}

func NewPerformanceMonitor() *PerformanceMonitor {
    return &PerformanceMonitor{
        startTime:  time.Now(),
        operations: make(map[string]time.Duration),
        memory:     make(map[string]uint64),
    }
}

func (pm *PerformanceMonitor) StartOperation(name string) func() {
    start := time.Now()
    return func() {
        duration := time.Since(start)
        pm.mutex.Lock()
        pm.operations[name] = duration
        pm.mutex.Unlock()
        
        // Log slow operations
        if duration > 1*time.Second {
            log.Printf("SLOW OPERATION: %s took %v", name, duration)
        }
    }
}

func (pm *PerformanceMonitor) RecordMemory(name string) {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    
    pm.mutex.Lock()
    pm.memory[name] = m.HeapAlloc
    pm.mutex.Unlock()
}

func (pm *PerformanceMonitor) Report() {
    pm.mutex.RLock()
    defer pm.mutex.RUnlock()
    
    log.Printf("Performance Report (uptime: %v):", time.Since(pm.startTime))
    
    for op, duration := range pm.operations {
        log.Printf("  %s: %v", op, duration)
    }
    
    for checkpoint, memory := range pm.memory {
        log.Printf("  Memory at %s: %d KB", checkpoint, memory/1024)
    }
}
```

## Common Issues and Solutions

### Memory Issues

#### Memory Leaks

```go
// Common causes and solutions

// 1. Unclosed resources
func badExample() {
    file, _ := os.Open("data.csv")
    // Missing: defer file.Close()
    // Solution: Always defer cleanup
}

func goodExample() {
    file, err := os.Open("data.csv")
    if err != nil {
        return err
    }
    defer file.Close() // Always clean up
    
    // Use file...
}

// 2. Goroutine leaks
func badGoroutineExample(ctx context.Context) {
    go func() {
        for {
            // This goroutine never exits!
            doWork()
            time.Sleep(1 * time.Second)
        }
    }()
}

func goodGoroutineExample(ctx context.Context) {
    go func() {
        ticker := time.NewTicker(1 * time.Second)
        defer ticker.Stop()
        
        for {
            select {
            case <-ticker.C:
                doWork()
            case <-ctx.Done():
                return // Exit when context is cancelled
            }
        }
    }()
}

// 3. Large slice retention
func badSliceExample(largeSlice []byte) []byte {
    // This retains the entire large slice
    return largeSlice[100:200]
}

func goodSliceExample(largeSlice []byte) []byte {
    // Copy to avoid retaining large slice
    result := make([]byte, 100)
    copy(result, largeSlice[100:200])
    return result
}
```

#### Out of Memory (OOM)

```go
// Memory-efficient data processing
func processLargeDataset(filename string) error {
    file, err := os.Open(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    
    // Process in chunks instead of loading all at once
    scanner := bufio.NewScanner(file)
    scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024) // 1MB buffer
    
    batch := make([]DataPoint, 0, 1000)
    
    for scanner.Scan() {
        point, err := parseDataPoint(scanner.Text())
        if err != nil {
            continue
        }
        
        batch = append(batch, point)
        
        // Process in batches to limit memory usage
        if len(batch) >= 1000 {
            if err := processBatch(batch); err != nil {
                return err
            }
            batch = batch[:0] // Reset slice but keep capacity
        }
    }
    
    // Process remaining items
    if len(batch) > 0 {
        return processBatch(batch)
    }
    
    return scanner.Err()
}
```

### Concurrency Issues

#### Race Conditions

```bash
# Run with race detector
go run -race cmd/server/main.go
go test -race ./...
```

```go
// Example race condition fix
type SafeCounter struct {
    mu    sync.RWMutex
    count int64
}

func (c *SafeCounter) Increment() {
    c.mu.Lock()
    c.count++
    c.mu.Unlock()
}

func (c *SafeCounter) Get() int64 {
    c.mu.RLock()
    count := c.count
    c.mu.RUnlock()
    return count
}
```

#### Deadlocks

```go
// Deadlock detection and prevention
func detectDeadlocks() {
    // Use context with timeout
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    // Always acquire locks in the same order
    mutex1.Lock()
    defer mutex1.Unlock()
    
    mutex2.Lock()
    defer mutex2.Unlock()
    
    // Use select with timeout for channel operations
    select {
    case result := <-resultChan:
        return result
    case <-ctx.Done():
        return ctx.Err()
    }
}
```

### Database Issues

#### Connection Pool Problems

```go
// Debug database connections
func debugDBConnections(db *sql.DB) {
    stats := db.Stats()
    log.Printf("DB Stats: Open=%d, InUse=%d, Idle=%d",
        stats.OpenConnections,
        stats.InUse,
        stats.Idle)
    
    if stats.OpenConnections > 20 {
        log.Warn("High number of database connections")
    }
}

// Connection leak detection
func queryWithDebug(db *sql.DB, query string) error {
    before := db.Stats()
    
    rows, err := db.Query(query)
    if err != nil {
        return err
    }
    defer func() {
        if err := rows.Close(); err != nil {
            log.Error("Failed to close rows", "error", err)
        }
        
        after := db.Stats()
        if after.InUse > before.InUse {
            log.Warn("Potential connection leak detected")
        }
    }()
    
    // Process rows...
    return nil
}
```

#### Slow Queries

```go
// Query performance monitoring
func monitoredQuery(db *sql.DB, query string, args ...interface{}) (*sql.Rows, error) {
    start := time.Now()
    
    rows, err := db.Query(query, args...)
    
    duration := time.Since(start)
    if duration > 100*time.Millisecond {
        log.Warn("Slow query detected",
            "query", query,
            "duration", duration,
            "args", args)
    }
    
    return rows, err
}
```

## Production Debugging

### Health Checks

```go
// Comprehensive health checks
type HealthChecker struct {
    db        *sql.DB
    redis     *redis.Client
    startTime time.Time
}

func (h *HealthChecker) CheckHealth(ctx context.Context) *HealthStatus {
    status := &HealthStatus{
        Status:    "healthy",
        Timestamp: time.Now(),
        Uptime:    time.Since(h.startTime),
        Checks:    make(map[string]CheckResult),
    }
    
    // Database check
    if err := h.checkDatabase(ctx); err != nil {
        status.Checks["database"] = CheckResult{
            Status: "unhealthy",
            Error:  err.Error(),
        }
        status.Status = "unhealthy"
    } else {
        status.Checks["database"] = CheckResult{Status: "healthy"}
    }
    
    // Redis check
    if err := h.checkRedis(ctx); err != nil {
        status.Checks["redis"] = CheckResult{
            Status: "unhealthy",
            Error:  err.Error(),
        }
        status.Status = "degraded"
    } else {
        status.Checks["redis"] = CheckResult{Status: "healthy"}
    }
    
    // Memory check
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    
    memoryMB := m.HeapAlloc / 1024 / 1024
    if memoryMB > 1024 { // More than 1GB
        status.Checks["memory"] = CheckResult{
            Status: "warning",
            Details: map[string]interface{}{
                "heap_alloc_mb": memoryMB,
            },
        }
    }
    
    return status
}
```

### Error Tracking

```go
// Structured error reporting
type ErrorTracker struct {
    errors map[string]*ErrorStats
    mu     sync.RWMutex
}

type ErrorStats struct {
    Count       int64
    FirstSeen   time.Time
    LastSeen    time.Time
    LastError   string
    Occurrences []time.Time
}

func (et *ErrorTracker) RecordError(err error, context map[string]interface{}) {
    et.mu.Lock()
    defer et.mu.Unlock()
    
    key := generateErrorKey(err)
    
    stats, exists := et.errors[key]
    if !exists {
        stats = &ErrorStats{
            FirstSeen:   time.Now(),
            Occurrences: make([]time.Time, 0, 100),
        }
        et.errors[key] = stats
    }
    
    stats.Count++
    stats.LastSeen = time.Now()
    stats.LastError = err.Error()
    
    // Keep last 100 occurrences
    if len(stats.Occurrences) >= 100 {
        stats.Occurrences = stats.Occurrences[1:]
    }
    stats.Occurrences = append(stats.Occurrences, time.Now())
    
    // Log error with context
    log.Error("Error recorded",
        "error", err,
        "error_key", key,
        "count", stats.Count,
        "context", context)
}
```

### Metrics Collection

```go
// Custom metrics for debugging
var (
    generationDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "tsiot_generation_duration_seconds",
            Help: "Time spent generating data",
            Buckets: prometheus.ExponentialBuckets(0.01, 2, 10),
        },
        []string{"generator_type", "status"},
    )
    
    memoryUsage = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "tsiot_memory_usage_bytes",
            Help: "Current memory usage",
        },
        []string{"component"},
    )
    
    errorCount = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "tsiot_errors_total",
            Help: "Total number of errors",
        },
        []string{"error_type", "component"},
    )
)

func recordMetrics() {
    go func() {
        ticker := time.NewTicker(30 * time.Second)
        defer ticker.Stop()
        
        for range ticker.C {
            var m runtime.MemStats
            runtime.ReadMemStats(&m)
            
            memoryUsage.WithLabelValues("heap").Set(float64(m.HeapAlloc))
            memoryUsage.WithLabelValues("stack").Set(float64(m.StackInuse))
            memoryUsage.WithLabelValues("sys").Set(float64(m.Sys))
        }
    }()
}
```

## Testing and Validation

### Debug Test Setup

```go
// Test utilities for debugging
func TestWithDebugLogging(t *testing.T) {
    // Set up debug logging for tests
    logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
        Level: slog.LevelDebug,
    }))
    
    ctx := context.Background()
    ctx = context.WithValue(ctx, "logger", logger)
    
    // Your test code with debug logging
    generator := NewTimeGANGenerator()
    data, err := generator.Generate(ctx, &GenerationParams{
        DataPoints: 100,
    })
    
    require.NoError(t, err)
    assert.NotNil(t, data)
    
    // Debug assertions
    t.Logf("Generated %d data points", len(data.Values))
    t.Logf("Memory usage: %d bytes", getMemoryUsage())
}

// Benchmark with profiling
func BenchmarkGenerationWithProfiling(b *testing.B) {
    b.ReportAllocs()
    
    generator := NewTimeGANGenerator()
    params := &GenerationParams{DataPoints: 1000}
    
    b.ResetTimer()
    
    for i := 0; i < b.N; i++ {
        _, err := generator.Generate(context.Background(), params)
        if err != nil {
            b.Fatal(err)
        }
    }
}
```

### Integration Test Debugging

```go
// Debug integration tests
func TestIntegrationWithDebug(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping integration test in short mode")
    }
    
    // Start test server with debug configuration
    server := startTestServer(t, &Config{
        Debug:    true,
        LogLevel: "debug",
    })
    defer server.Stop()
    
    // Create debug client
    client := &http.Client{
        Transport: &debugTransport{
            base: http.DefaultTransport,
            t:    t,
        },
    }
    
    // Test with detailed logging
    resp, err := client.Post(
        server.URL+"/api/v1/generate",
        "application/json",
        strings.NewReader(`{"generator": "timegan", "dataPoints": 100}`),
    )
    
    require.NoError(t, err)
    defer resp.Body.Close()
    
    assert.Equal(t, http.StatusOK, resp.StatusCode)
    
    // Debug response
    body, err := io.ReadAll(resp.Body)
    require.NoError(t, err)
    t.Logf("Response: %s", string(body))
}

type debugTransport struct {
    base http.RoundTripper
    t    *testing.T
}

func (dt *debugTransport) RoundTrip(req *http.Request) (*http.Response, error) {
    dt.t.Logf("Request: %s %s", req.Method, req.URL)
    
    if req.Body != nil {
        body, _ := io.ReadAll(req.Body)
        req.Body = io.NopCloser(bytes.NewReader(body))
        dt.t.Logf("Request Body: %s", string(body))
    }
    
    start := time.Now()
    resp, err := dt.base.RoundTrip(req)
    duration := time.Since(start)
    
    if err != nil {
        dt.t.Logf("Request failed after %v: %v", duration, err)
        return nil, err
    }
    
    dt.t.Logf("Response: %d in %v", resp.StatusCode, duration)
    return resp, nil
}
```

## Tools and Utilities

### Debug CLI Commands

```bash
# Debug generation
./tsiot-cli generate --debug --verbose \
  --generator timegan \
  --data-points 100 \
  --output debug.csv

# Debug with profiling
./tsiot-cli generate --profile --cpuprofile cpu.prof \
  --memprofile mem.prof \
  --generator timegan

# Validate with debug info
./tsiot-cli validate --debug \
  --input synthetic.csv \
  --reference real.csv \
  --verbose
```

### Debug Server Endpoints

```go
// Add debug endpoints
func setupDebugRoutes(r *mux.Router) {
    r.HandleFunc("/debug/health", debugHealthHandler)
    r.HandleFunc("/debug/metrics", debugMetricsHandler)
    r.HandleFunc("/debug/goroutines", debugGoroutinesHandler)
    r.HandleFunc("/debug/memory", debugMemoryHandler)
    r.HandleFunc("/debug/config", debugConfigHandler)
}

func debugHealthHandler(w http.ResponseWriter, r *http.Request) {
    health := collectHealthInfo()
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(health)
}

func debugGoroutinesHandler(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "text/plain")
    pprof.Lookup("goroutine").WriteTo(w, 1)
}
```

### Automated Debug Scripts

```bash
#!/bin/bash
# debug-collect.sh - Collect debug information

echo "Collecting TSIoT debug information..."

# Create debug directory
mkdir -p debug/$(date +%Y%m%d_%H%M%S)
DEBUG_DIR="debug/$(date +%Y%m%d_%H%M%S)"

# Collect system info
echo "System information:" > $DEBUG_DIR/system.txt
uname -a >> $DEBUG_DIR/system.txt
go version >> $DEBUG_DIR/system.txt
docker version >> $DEBUG_DIR/system.txt 2>/dev/null || echo "Docker not available" >> $DEBUG_DIR/system.txt

# Collect application logs
echo "Collecting logs..."
cp logs/tsiot.log $DEBUG_DIR/
cp logs/error.log $DEBUG_DIR/ 2>/dev/null || echo "No error log found"

# Collect profiles
echo "Collecting profiles..."
curl -s http://localhost:6060/debug/pprof/goroutine > $DEBUG_DIR/goroutines.prof
curl -s http://localhost:6060/debug/pprof/heap > $DEBUG_DIR/heap.prof
curl -s http://localhost:6060/debug/pprof/profile?seconds=10 > $DEBUG_DIR/cpu.prof

# Collect metrics
echo "Collecting metrics..."
curl -s http://localhost:9090/metrics > $DEBUG_DIR/metrics.txt

# Collect health status
echo "Collecting health status..."
curl -s http://localhost:8080/health > $DEBUG_DIR/health.json
curl -s http://localhost:8080/debug/health > $DEBUG_DIR/debug_health.json

# Archive debug info
tar -czf "debug_$(date +%Y%m%d_%H%M%S).tar.gz" $DEBUG_DIR

echo "Debug information collected in debug_$(date +%Y%m%d_%H%M%S).tar.gz"
```

This debugging guide provides comprehensive tools and techniques for troubleshooting issues in the TSIoT platform. Use these debugging strategies to identify and resolve performance bottlenecks, memory issues, concurrency problems, and other system issues effectively.