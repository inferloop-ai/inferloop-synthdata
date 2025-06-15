# TSIoT Performance Tuning Guide

This guide provides comprehensive performance optimization strategies for the TSIoT (Time Series IoT) synthetic data generation platform. It covers profiling, optimization techniques, and best practices for achieving optimal performance.

## Table of Contents

- [Performance Overview](#performance-overview)
- [Profiling and Monitoring](#profiling-and-monitoring)
- [CPU Optimization](#cpu-optimization)
- [Memory Optimization](#memory-optimization)
- [I/O Optimization](#io-optimization)
- [Network Optimization](#network-optimization)
- [Database Optimization](#database-optimization)
- [Generator-Specific Tuning](#generator-specific-tuning)
- [Concurrency and Parallelism](#concurrency-and-parallelism)
- [Caching Strategies](#caching-strategies)
- [Benchmarking](#benchmarking)

## Performance Overview

### Key Performance Metrics

- **Throughput**: Data points generated per second
- **Latency**: Response time for generation requests
- **Memory Usage**: RAM consumption during generation
- **CPU Utilization**: Processor usage patterns
- **I/O Operations**: Disk and network activity
- **Scalability**: Performance under increasing load

### Performance Targets

```yaml
# Performance benchmarks
Targets:
  throughput:
    timegan: 10000 points/second
    arima: 50000 points/second
    statistical: 100000 points/second
  latency:
    p50: < 100ms
    p95: < 500ms
    p99: < 1000ms
  memory:
    peak_usage: < 2GB per generator
    gc_pause: < 10ms
  cpu:
    utilization: 70-80% under load
```

## Profiling and Monitoring

### Go Profiling Tools

#### pprof Integration

```go
// Enable pprof endpoints
package main

import (
    _ "net/http/pprof"
    "net/http"
    "log"
)

func main() {
    // Start pprof server
    go func() {
        log.Println(http.ListenAndServe(":6060", nil))
    }()
    
    // Your application code
    startServer()
}
```

#### CPU Profiling

```bash
# Collect CPU profile
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30

# Analyze profile
(pprof) top 10
(pprof) list functionName
(pprof) web
```

#### Memory Profiling

```bash
# Heap profile
go tool pprof http://localhost:6060/debug/pprof/heap

# Allocation profile
go tool pprof http://localhost:6060/debug/pprof/allocs

# Memory usage over time
go tool pprof http://localhost:6060/debug/pprof/heap?seconds=30
```

#### Goroutine Analysis

```bash
# Goroutine profile
go tool pprof http://localhost:6060/debug/pprof/goroutine

# Check for goroutine leaks
curl http://localhost:6060/debug/pprof/goroutine?debug=1
```

### Continuous Monitoring

```yaml
# Prometheus metrics
metrics:
  - name: tsiot_generation_duration_seconds
    type: histogram
    help: Time spent generating synthetic data
    buckets: [0.1, 0.5, 1, 5, 10, 30]
    
  - name: tsiot_memory_usage_bytes
    type: gauge
    help: Memory usage in bytes
    
  - name: tsiot_goroutines_total
    type: gauge
    help: Number of active goroutines
```

## CPU Optimization

### Efficient Algorithm Selection

```go
// Choose algorithms based on data characteristics
func selectOptimalGenerator(dataSize int, complexity string) GeneratorType {
    switch {
    case dataSize < 1000:
        return StatisticalGenerator
    case complexity == "simple":
        return ARIMAGenerator
    case complexity == "complex" && dataSize < 10000:
        return RNNGenerator
    default:
        return TimeGANGenerator
    }
}
```

### Vectorization and SIMD

```go
// Use optimized math libraries
import "gonum.org/v1/gonum/mat"

// Vectorized operations
func normalizeVector(data []float64) []float64 {
    vec := mat.NewVecDense(len(data), data)
    
    // Compute mean and std efficiently
    mean := stat.Mean(data, nil)
    std := stat.StdDev(data, nil)
    
    // Vectorized normalization
    vec.AddScaledVec(vec, -mean, ones)
    vec.ScaleVec(1/std, vec)
    
    return vec.RawVector().Data
}
```

### CPU-Bound Operations

```go
// Optimize tight loops
func optimizedMatrixMultiply(a, b [][]float64) [][]float64 {
    rows, cols := len(a), len(b[0])
    result := make([][]float64, rows)
    
    for i := 0; i < rows; i++ {
        result[i] = make([]float64, cols)
        for k := 0; k < len(a[i]); k++ {
            // Cache a[i][k] to reduce memory access
            aik := a[i][k]
            for j := 0; j < cols; j++ {
                result[i][j] += aik * b[k][j]
            }
        }
    }
    return result
}
```

## Memory Optimization

### Memory Pool Pattern

```go
// Reuse memory allocations
type DataPool struct {
    pool sync.Pool
}

func NewDataPool() *DataPool {
    return &DataPool{
        pool: sync.Pool{
            New: func() interface{} {
                return make([]float64, 0, 1000) // Pre-allocate capacity
            },
        },
    }
}

func (p *DataPool) Get() []float64 {
    return p.pool.Get().([]float64)
}

func (p *DataPool) Put(data []float64) {
    data = data[:0] // Reset length but keep capacity
    p.pool.Put(data)
}
```

### Efficient Data Structures

```go
// Use appropriate data structures
type TimeSeriesBuffer struct {
    timestamps []int64    // Unix timestamps (8 bytes each)
    values     []float32  // Use float32 when precision allows (4 bytes vs 8)
    capacity   int
}

// Pre-allocate slices with known capacity
func NewTimeSeriesBuffer(size int) *TimeSeriesBuffer {
    return &TimeSeriesBuffer{
        timestamps: make([]int64, 0, size),
        values:     make([]float32, 0, size),
        capacity:   size,
    }
}
```

### Memory Leak Prevention

```go
// Proper cleanup patterns
func (g *Generator) Generate(ctx context.Context) error {
    // Use defer for cleanup
    defer func() {
        g.cleanup()
    }()
    
    // Check for context cancellation
    select {
    case <-ctx.Done():
        return ctx.Err()
    default:
    }
    
    // Force garbage collection in memory-intensive operations
    if g.memoryUsage > threshold {
        runtime.GC()
    }
    
    return nil
}
```

### Garbage Collection Tuning

```go
// Optimize GC settings
func init() {
    // Set GC target percentage
    debug.SetGCPercent(100) // Default is 100
    
    // Set memory limit (Go 1.19+)
    debug.SetMemoryLimit(8 << 30) // 8GB
}

// Monitor GC statistics
func monitorGC() {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    
    log.Printf("GC Stats: NumGC=%d, PauseTotal=%v, HeapAlloc=%d KB",
        m.NumGC, time.Duration(m.PauseTotalNs), m.HeapAlloc/1024)
}
```

## I/O Optimization

### Batch Processing

```go
// Batch database operations
type BatchWriter struct {
    db        *sql.DB
    batchSize int
    buffer    []DataPoint
    mu        sync.Mutex
}

func (w *BatchWriter) Write(point DataPoint) error {
    w.mu.Lock()
    defer w.mu.Unlock()
    
    w.buffer = append(w.buffer, point)
    
    if len(w.buffer) >= w.batchSize {
        return w.flush()
    }
    return nil
}

func (w *BatchWriter) flush() error {
    if len(w.buffer) == 0 {
        return nil
    }
    
    // Batch insert
    tx, err := w.db.Begin()
    if err != nil {
        return err
    }
    defer tx.Rollback()
    
    stmt, err := tx.Prepare("INSERT INTO timeseries (timestamp, value) VALUES (?, ?)")
    if err != nil {
        return err
    }
    defer stmt.Close()
    
    for _, point := range w.buffer {
        if _, err := stmt.Exec(point.Timestamp, point.Value); err != nil {
            return err
        }
    }
    
    w.buffer = w.buffer[:0] // Clear buffer
    return tx.Commit()
}
```

### Streaming I/O

```go
// Stream large datasets
func streamGeneration(ctx context.Context, output io.Writer) error {
    encoder := json.NewEncoder(output)
    
    for i := 0; i < totalPoints; i++ {
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
        }
        
        point := generatePoint(i)
        if err := encoder.Encode(point); err != nil {
            return err
        }
        
        // Periodic flush to avoid buffering too much
        if i%1000 == 0 {
            if flusher, ok := output.(interface{ Flush() error }); ok {
                flusher.Flush()
            }
        }
    }
    
    return nil
}
```

## Network Optimization

### Connection Pooling

```go
// HTTP client optimization
func createOptimizedHTTPClient() *http.Client {
    transport := &http.Transport{
        MaxIdleConns:        100,
        MaxIdleConnsPerHost: 10,
        IdleConnTimeout:     90 * time.Second,
        DisableCompression:  false,
        
        // Connection settings
        DialContext: (&net.Dialer{
            Timeout:   30 * time.Second,
            KeepAlive: 30 * time.Second,
        }).DialContext,
        
        // TLS settings
        TLSHandshakeTimeout: 10 * time.Second,
        ExpectContinueTimeout: 1 * time.Second,
    }
    
    return &http.Client{
        Transport: transport,
        Timeout:   30 * time.Second,
    }
}
```

### gRPC Optimization

```go
// Optimize gRPC client
func createOptimizedGRPCClient() (*grpc.ClientConn, error) {
    return grpc.Dial("localhost:50051",
        grpc.WithTransportCredentials(insecure.NewCredentials()),
        grpc.WithKeepaliveParams(keepalive.ClientParameters{
            Time:                10 * time.Second,
            Timeout:             time.Second,
            PermitWithoutStream: true,
        }),
        grpc.WithDefaultCallOptions(
            grpc.MaxCallRecvMsgSize(4*1024*1024), // 4MB
            grpc.MaxCallSendMsgSize(4*1024*1024), // 4MB
        ),
    )
}
```

## Database Optimization

### Connection Pool Tuning

```go
// Database connection optimization
func setupDatabasePool(dsn string) (*sql.DB, error) {
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        return nil, err
    }
    
    // Connection pool settings
    db.SetMaxOpenConns(25)                 // Maximum connections
    db.SetMaxIdleConns(5)                  // Idle connections
    db.SetConnMaxLifetime(5 * time.Minute) // Connection lifetime
    db.SetConnMaxIdleTime(1 * time.Minute) // Idle timeout
    
    return db, nil
}
```

### Query Optimization

```sql
-- Index optimization for time series queries
CREATE INDEX CONCURRENTLY idx_timeseries_timestamp 
ON timeseries (timestamp DESC);

CREATE INDEX CONCURRENTLY idx_timeseries_sensor_timestamp 
ON timeseries (sensor_id, timestamp DESC);

-- Partitioning for large datasets
CREATE TABLE timeseries_2024 PARTITION OF timeseries
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

### Bulk Operations

```go
// Efficient bulk insert
func bulkInsert(db *sql.DB, data []DataPoint) error {
    txn, err := db.Begin()
    if err != nil {
        return err
    }
    defer txn.Rollback()
    
    stmt, err := txn.Prepare(pq.CopyIn("timeseries", "timestamp", "value", "sensor_id"))
    if err != nil {
        return err
    }
    defer stmt.Close()
    
    for _, point := range data {
        if _, err := stmt.Exec(point.Timestamp, point.Value, point.SensorID); err != nil {
            return err
        }
    }
    
    if _, err := stmt.Exec(); err != nil {
        return err
    }
    
    return txn.Commit()
}
```

## Generator-Specific Tuning

### TimeGAN Optimization

```go
type OptimizedTimeGAN struct {
    batchSize     int
    numWorkers    int
    useGPU        bool
    memoryLimit   int64
    checkpointDir string
}

func (t *OptimizedTimeGAN) Generate(ctx context.Context) error {
    // Enable mixed precision training
    config := &TrainingConfig{
        UseMixedPrecision: true,
        GradientClipping:  1.0,
        BatchSize:         t.batchSize,
    }
    
    // Use data loaders for efficient batching
    dataLoader := NewDataLoader(t.batchSize, t.numWorkers)
    
    // Implement gradient accumulation
    accumulationSteps := 4
    
    for epoch := 0; epoch < config.Epochs; epoch++ {
        for batch := range dataLoader.Batches() {
            loss := t.trainStep(batch)
            
            if (batch.Index+1)%accumulationSteps == 0 {
                t.optimizer.Step()
                t.optimizer.ZeroGrad()
            }
        }
        
        // Periodic checkpointing
        if epoch%100 == 0 {
            t.saveCheckpoint(epoch)
        }
    }
    
    return nil
}
```

### ARIMA Optimization

```go
// Optimize ARIMA parameter estimation
func (a *ARIMAGenerator) optimizeParameters(data []float64) (*ARIMAParams, error) {
    // Use parallel parameter search
    paramChan := make(chan ARIMAParams, 100)
    resultChan := make(chan ARIMAResult, 100)
    
    // Start workers
    numWorkers := runtime.NumCPU()
    for i := 0; i < numWorkers; i++ {
        go a.parameterWorker(paramChan, resultChan, data)
    }
    
    // Generate parameter combinations
    go func() {
        defer close(paramChan)
        for p := 0; p <= 5; p++ {
            for d := 0; d <= 2; d++ {
                for q := 0; q <= 5; q++ {
                    paramChan <- ARIMAParams{P: p, D: d, Q: q}
                }
            }
        }
    }()
    
    // Collect results
    bestParams := ARIMAParams{}
    bestAIC := math.Inf(1)
    
    for i := 0; i < 216; i++ { // 6*3*6 combinations
        result := <-resultChan
        if result.AIC < bestAIC {
            bestAIC = result.AIC
            bestParams = result.Params
        }
    }
    
    return &bestParams, nil
}
```

## Concurrency and Parallelism

### Worker Pool Pattern

```go
type WorkerPool struct {
    workers    int
    jobQueue   chan Job
    resultChan chan Result
    wg         sync.WaitGroup
}

func NewWorkerPool(workers int) *WorkerPool {
    return &WorkerPool{
        workers:    workers,
        jobQueue:   make(chan Job, workers*2),
        resultChan: make(chan Result, workers*2),
    }
}

func (p *WorkerPool) Start(ctx context.Context) {
    for i := 0; i < p.workers; i++ {
        p.wg.Add(1)
        go p.worker(ctx)
    }
}

func (p *WorkerPool) worker(ctx context.Context) {
    defer p.wg.Done()
    
    for {
        select {
        case job := <-p.jobQueue:
            result := processJob(job)
            p.resultChan <- result
        case <-ctx.Done():
            return
        }
    }
}
```

### Pipeline Processing

```go
// Pipeline for data processing
func createProcessingPipeline(ctx context.Context) {
    // Stage 1: Data ingestion
    dataChan := make(chan RawData, 100)
    go dataIngestion(ctx, dataChan)
    
    // Stage 2: Preprocessing
    processedChan := make(chan ProcessedData, 100)
    go preprocessing(ctx, dataChan, processedChan)
    
    // Stage 3: Generation
    generatedChan := make(chan GeneratedData, 100)
    go generation(ctx, processedChan, generatedChan)
    
    // Stage 4: Output
    go output(ctx, generatedChan)
}
```

## Caching Strategies

### In-Memory Caching

```go
type GeneratorCache struct {
    cache map[string]interface{}
    mu    sync.RWMutex
    ttl   time.Duration
}

func (c *GeneratorCache) Get(key string) (interface{}, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()
    
    item, exists := c.cache[key]
    return item, exists
}

func (c *GeneratorCache) Set(key string, value interface{}) {
    c.mu.Lock()
    defer c.mu.Unlock()
    
    c.cache[key] = value
    
    // Start TTL timer
    time.AfterFunc(c.ttl, func() {
        c.mu.Lock()
        delete(c.cache, key)
        c.mu.Unlock()
    })
}
```

### Distributed Caching

```go
// Redis caching for model artifacts
type ModelCache struct {
    client *redis.Client
    ttl    time.Duration
}

func (c *ModelCache) CacheModel(key string, model []byte) error {
    compressed := compress(model)
    return c.client.Set(context.Background(), key, compressed, c.ttl).Err()
}

func (c *ModelCache) GetModel(key string) ([]byte, error) {
    compressed, err := c.client.Get(context.Background(), key).Bytes()
    if err != nil {
        return nil, err
    }
    return decompress(compressed), nil
}
```

## Benchmarking

### Comprehensive Benchmarks

```go
// Benchmark generation performance
func BenchmarkTimeGANGeneration(b *testing.B) {
    generator := NewTimeGANGenerator()
    params := &GenerationParams{
        DataPoints: 1000,
        Features:   []string{"temperature"},
    }
    
    b.ResetTimer()
    b.ReportAllocs()
    
    for i := 0; i < b.N; i++ {
        _, err := generator.Generate(context.Background(), params)
        if err != nil {
            b.Fatal(err)
        }
    }
}

// Memory allocation benchmark
func BenchmarkMemoryAllocation(b *testing.B) {
    b.ReportAllocs()
    
    for i := 0; i < b.N; i++ {
        data := make([]float64, 10000)
        _ = processData(data)
    }
}

// Concurrent generation benchmark
func BenchmarkConcurrentGeneration(b *testing.B) {
    generator := NewARIMAGenerator()
    params := &GenerationParams{DataPoints: 100}
    
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            _, err := generator.Generate(context.Background(), params)
            if err != nil {
                b.Fatal(err)
            }
        }
    })
}
```

### Performance Testing

```bash
#!/bin/bash
# Performance test script

# CPU benchmark
echo "Running CPU benchmarks..."
go test -bench=BenchmarkCPU -benchmem -cpuprofile=cpu.prof

# Memory benchmark
echo "Running memory benchmarks..."
go test -bench=BenchmarkMemory -benchmem -memprofile=mem.prof

# Concurrent benchmark
echo "Running concurrency benchmarks..."
go test -bench=BenchmarkConcurrent -benchmem -cpu=1,2,4,8

# Generate reports
go tool pprof -http=:8080 cpu.prof &
go tool pprof -http=:8081 mem.prof &
```

### Load Testing

```yaml
# k6 load testing script
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 10 },
    { duration: '5m', target: 50 },
    { duration: '2m', target: 100 },
    { duration: '5m', target: 100 },
    { duration: '2m', target: 0 },
  ],
};

export default function() {
  let payload = JSON.stringify({
    generator: 'timegan',
    dataPoints: 1000,
    features: ['temperature', 'humidity']
  });
  
  let response = http.post('http://localhost:8080/api/v1/generate', payload, {
    headers: { 'Content-Type': 'application/json' },
  });
  
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
}
```

## Monitoring and Alerting

### Performance Metrics

```yaml
# Grafana dashboard configuration
dashboard:
  title: "TSIoT Performance Metrics"
  panels:
    - title: "Generation Throughput"
      type: "graph"
      targets:
        - expr: "rate(tsiot_generation_total[5m])"
          legend: "Generations per second"
    
    - title: "Response Time"
      type: "graph"
      targets:
        - expr: "histogram_quantile(0.95, tsiot_generation_duration_seconds)"
          legend: "95th percentile"
    
    - title: "Memory Usage"
      type: "graph"
      targets:
        - expr: "process_resident_memory_bytes"
          legend: "Resident memory"
```

### Alerting Rules

```yaml
# Prometheus alerting rules
groups:
  - name: tsiot_performance
    rules:
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, tsiot_generation_duration_seconds) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes > 2e9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Memory usage exceeds 2GB"
```

This performance tuning guide provides comprehensive strategies for optimizing the TSIoT platform. Regular profiling, monitoring, and benchmarking are essential for maintaining optimal performance as the system scales.