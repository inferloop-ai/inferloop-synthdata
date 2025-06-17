# Profiling Tools

This directory contains profiling utilities for analyzing and optimizing the TSIoT platform's performance.

## Overview

The profiling tools help identify performance bottlenecks, memory leaks, and optimization opportunities in the TSIoT platform. These tools integrate with Go's built-in profiling capabilities and provide enhanced analysis features specific to time series data processing.

## Available Profilers

### 1. CPU Profiler
Analyzes CPU usage patterns and identifies hot paths in the code.

**Features:**
- Function-level CPU usage analysis
- Call graph generation
- Hot path identification
- Flame graph support
- Time-based sampling

### 2. Memory Profiler
Tracks memory allocation and identifies memory leaks.

**Features:**
- Heap allocation tracking
- Memory usage over time
- Garbage collection analysis
- Object allocation profiling
- Memory leak detection

### 3. Goroutine Profiler
Monitors goroutine behavior and concurrency patterns.

**Features:**
- Active goroutine count
- Goroutine stack traces
- Deadlock detection
- Channel blocking analysis
- Concurrency bottleneck identification

### 4. Block Profiler
Identifies blocking operations that impact performance.

**Features:**
- Mutex contention analysis
- Channel blocking profiling
- I/O operation delays
- Synchronization bottlenecks

### 5. Trace Profiler
Provides detailed execution traces for debugging complex issues.

**Features:**
- Request flow tracing
- Latency breakdown
- Event correlation
- Distributed tracing support

## Usage

### Basic Profiling

```bash
# Start profiling server
go run profiling/server.go -port 6060

# CPU profiling for 30 seconds
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30

# Memory profiling
go tool pprof http://localhost:6060/debug/pprof/heap

# Goroutine profiling
go tool pprof http://localhost:6060/debug/pprof/goroutine

# Block profiling
go tool pprof http://localhost:6060/debug/pprof/block

# Mutex profiling
go tool pprof http://localhost:6060/debug/pprof/mutex
```

### Advanced Analysis

```bash
# Generate flame graph
go-torch -u http://localhost:6060/debug/pprof/profile -t 30

# Compare profiles
go tool pprof -base profile1.pprof profile2.pprof

# Web-based visualization
go tool pprof -http=:8080 profile.pprof

# Top functions by CPU usage
go tool pprof -top profile.pprof

# Disassembly of hot functions
go tool pprof -disasm=FunctionName profile.pprof
```

### Time Series Specific Profiling

```bash
# Profile data ingestion pipeline
./profile-ingestion -duration 5m -output ingestion.pprof

# Profile query performance
./profile-queries -workload heavy -duration 10m

# Profile aggregation operations
./profile-aggregations -window 1h -duration 30m

# Profile privacy transformations
./profile-privacy -operations "k-anonymity,differential-privacy" -duration 15m
```

## Integration with TSIoT

### 1. Automatic Profiling

Add to your TSIoT configuration:

```yaml
profiling:
  enabled: true
  cpu:
    enabled: true
    duration: 30s
    interval: 5m
  memory:
    enabled: true
    snapshot_interval: 1m
  output:
    directory: /var/log/tsiot/profiles
    format: pprof
    retention: 7d
```

### 2. Runtime Profiling API

```go
import "github.com/inferloop/tsiot/tools/profiling"

// Start CPU profiling
prof := profiling.StartCPUProfile()
defer prof.Stop()

// Memory snapshot
snapshot := profiling.CaptureMemorySnapshot()
snapshot.Save("memory_snapshot.pprof")

// Custom metrics
profiling.RecordMetric("ingestion_rate", rate)
profiling.RecordLatency("query_latency", duration)
```

### 3. Continuous Profiling

```bash
# Enable continuous profiling
./tsiot-server -enable-continuous-profiling \
  -profiling-interval=5m \
  -profiling-duration=30s \
  -profiling-output=/var/log/tsiot/profiles
```

## Performance Benchmarks

### Running Benchmarks

```bash
# Run all benchmarks
go test -bench=. -benchmem ./...

# Run specific benchmark
go test -bench=BenchmarkDataIngestion -benchmem

# Profile during benchmark
go test -bench=BenchmarkQuery -cpuprofile=cpu.pprof -memprofile=mem.pprof

# Long-running benchmark
go test -bench=. -benchtime=10m
```

### Benchmark Scenarios

1. **Data Ingestion**
   - Single point insertion
   - Batch insertion (1K, 10K, 100K points)
   - Concurrent ingestion
   - Different data types

2. **Query Performance**
   - Time range queries
   - Aggregation queries
   - Tag-based filtering
   - Complex joins

3. **Privacy Operations**
   - K-anonymity transformations
   - Differential privacy noise addition
   - Synthetic data generation

4. **Protocol Performance**
   - gRPC throughput
   - MQTT message processing
   - Kafka consumer/producer rates

## Analysis Tools

### 1. Profile Analyzer

```bash
# Analyze CPU profile
./analyze-profile -type cpu -file cpu.pprof \
  -report summary,top10,callgraph

# Memory leak detection
./analyze-profile -type memory -file mem.pprof \
  -detect-leaks -threshold 10MB

# Goroutine analysis
./analyze-profile -type goroutine -file goroutine.pprof \
  -detect-deadlocks -max-goroutines 10000
```

### 2. Performance Dashboard

Access the performance dashboard at `http://localhost:9090/dashboard`:

- Real-time CPU and memory usage
- Request latency histograms
- Throughput metrics
- Error rates
- Custom TSIoT metrics

### 3. Automated Reports

```bash
# Generate daily performance report
./generate-report -date 2024-01-15 \
  -profiles /var/log/tsiot/profiles \
  -output performance_report.html

# Compare performance between versions
./compare-versions -v1 1.2.0 -v2 1.3.0 \
  -metrics "cpu,memory,latency,throughput"
```

## Best Practices

### 1. Profiling Strategy

- **Development**: Profile frequently, focus on hot paths
- **Testing**: Profile under realistic load conditions
- **Production**: Use sampling to minimize overhead
- **Continuous**: Set up automated profiling pipelines

### 2. Common Bottlenecks

- **CPU**: Look for inefficient algorithms, excessive allocations
- **Memory**: Check for memory leaks, large allocations
- **I/O**: Identify blocking operations, optimize batch sizes
- **Concurrency**: Find lock contention, optimize goroutine count

### 3. Optimization Workflow

1. Establish baseline metrics
2. Profile to identify bottlenecks
3. Optimize the top bottleneck
4. Measure improvement
5. Repeat until performance goals are met

## Troubleshooting

### High CPU Usage

```bash
# Identify top CPU consumers
go tool pprof -top cpu.pprof | head -20

# Check for infinite loops
go tool pprof -traces cpu.pprof | grep -E "runtime|loop"

# Analyze specific function
go tool pprof -focus="FunctionName" cpu.pprof
```

### Memory Issues

```bash
# Find memory allocations
go tool pprof -alloc_space mem.pprof

# Check for growing memory
go tool pprof -inuse_space mem.pprof

# Compare memory snapshots
go tool pprof -base=mem1.pprof mem2.pprof
```

### Goroutine Leaks

```bash
# Count goroutines over time
for i in {1..10}; do
  curl -s http://localhost:6060/debug/pprof/goroutine?debug=1 | grep "goroutine" | wc -l
  sleep 10
done

# Find goroutine creation sites
go tool pprof -traces goroutine.pprof | grep "created by"
```

## Integration Examples

### 1. CI/CD Pipeline

```yaml
performance-test:
  stage: test
  script:
    - go test -bench=. -benchmem -cpuprofile=cpu.pprof
    - ./analyze-profile -file cpu.pprof -assert "max_cpu_per_op < 100ms"
    - ./analyze-profile -file mem.pprof -assert "max_alloc_per_op < 1MB"
  artifacts:
    paths:
      - "*.pprof"
    reports:
      performance: performance_report.json
```

### 2. Monitoring Integration

```yaml
# Prometheus configuration
scrape_configs:
  - job_name: 'tsiot-profiling'
    static_configs:
      - targets: ['localhost:6060']
    metrics_path: '/debug/pprof/metrics'
```

### 3. Alert Rules

```yaml
alerts:
  - name: HighCPUUsage
    condition: cpu_usage_percent > 80
    duration: 5m
    action: capture_cpu_profile

  - name: MemoryLeak
    condition: memory_growth_rate > 100MB/hour
    duration: 30m
    action: capture_memory_profile

  - name: GoroutineLeak
    condition: goroutine_count > 10000
    duration: 10m
    action: capture_goroutine_profile
```

## Advanced Topics

### 1. Custom Profilers

Create application-specific profilers:

```go
type TimeSeriesProfiler struct {
    // Track time series specific metrics
}

func (p *TimeSeriesProfiler) ProfileIngestion() {
    // Profile ingestion pipeline
}

func (p *TimeSeriesProfiler) ProfileQuery() {
    // Profile query execution
}
```

### 2. Distributed Profiling

Profile across multiple nodes:

```bash
# Collect profiles from all nodes
./collect-distributed-profiles -nodes node1,node2,node3 \
  -profile-type cpu,memory \
  -output distributed_profile.tar.gz

# Aggregate distributed profiles
./aggregate-profiles -input distributed_profile.tar.gz \
  -output aggregated_profile.pprof
```

### 3. Production Profiling

Safe profiling in production:

```bash
# Low-overhead CPU profiling
curl http://localhost:6060/debug/pprof/profile?seconds=30&rate=100 > cpu.pprof

# Sampling-based memory profiling
curl http://localhost:6060/debug/pprof/heap?sample_index=alloc_space > mem.pprof

# Non-blocking goroutine dump
curl http://localhost:6060/debug/pprof/goroutine?debug=0 > goroutine.pprof
```

## Resources

- [Go Profiling Documentation](https://golang.org/doc/diagnostics)
- [pprof Tool Guide](https://github.com/google/pprof)
- [Flame Graphs](http://www.brendangregg.com/flamegraphs.html)
- [TSIoT Performance Guide](https://inferloop.com/docs/performance)