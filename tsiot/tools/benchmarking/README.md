# Benchmarking Tool

This tool provides comprehensive performance benchmarking capabilities for the TSIoT platform, focusing on time series data operations, protocol performance, and system scalability.

## Overview

The benchmarking tool helps measure and analyze:
- Data ingestion throughput
- Query performance and latency
- Protocol-specific performance (gRPC, HTTP, MQTT, Kafka)
- Privacy operation overhead
- System resource utilization
- Scalability under various loads

## Features

- **Multi-protocol Support**: Benchmark gRPC, HTTP, MQTT, and Kafka endpoints
- **Configurable Workloads**: Define custom operation mixes and data patterns
- **Real-time Metrics**: Monitor performance during benchmark execution
- **Detailed Reports**: Generate comprehensive reports in multiple formats
- **Comparative Analysis**: Compare results across different configurations
- **System Metrics**: Track CPU, memory, and goroutine usage

## Quick Start

### Basic Usage

```bash
# Run a simple benchmark
./benchmark -target localhost:50051 -duration 60s -concurrency 10

# Run with custom configuration
./benchmark -config benchmark.yaml

# Generate HTML report
./benchmark -target localhost:50051 -duration 5m -format html -output report.html
```

### Configuration File

Create a `benchmark.yaml` file:

```yaml
name: "TSIoT Performance Benchmark"
description: "Comprehensive performance test for v1.0"
duration: 5m
warmup: 30s
concurrency: 50

operations:
  - name: "insert_single"
    type: "insert"
    weight: 40
    parameters:
      batch_size: 1
      
  - name: "insert_batch"
    type: "insert"
    weight: 30
    parameters:
      batch_size: 100
      
  - name: "query_range"
    type: "query"
    weight: 20
    parameters:
      time_range: 1h
      
  - name: "aggregate"
    type: "aggregate"
    weight: 10
    parameters:
      window: 5m
      function: "avg"

data_config:
  num_series: 1000
  points_per_series: 10000
  batch_size: 100
  generator_type: "synthetic"
  generator_config:
    patterns:
      - type: "sine"
        amplitude: 100
        frequency: 0.1
      - type: "random"
        amplitude: 10
    noise:
      enabled: true
      level: 5.0

target_config:
  type: "grpc"
  endpoint: "localhost:50051"
  timeout: 30s
  max_retries: 3
  tls: false

metrics_config:
  collect_interval: 1s
  percentiles: [0.5, 0.9, 0.95, 0.99, 0.999]
  histogram: true
  detailed_errors: true

report_config:
  format: "json"
  output_file: "benchmark_results.json"
  include_raw: false
```

## Benchmark Scenarios

### 1. Ingestion Performance

```bash
# Test single-point ingestion
./benchmark -operation insert -config configs/ingestion_single.yaml

# Test batch ingestion
./benchmark -operation insert -config configs/ingestion_batch.yaml

# Test concurrent ingestion
./benchmark -operation insert -concurrency 100 -duration 10m
```

### 2. Query Performance

```bash
# Time range queries
./benchmark -operation query -config configs/query_timerange.yaml

# Aggregation queries
./benchmark -operation aggregate -config configs/query_aggregation.yaml

# Complex queries
./benchmark -operation query -config configs/query_complex.yaml
```

### 3. Mixed Workload

```bash
# 70% writes, 30% reads
./benchmark -operation mixed -config configs/mixed_write_heavy.yaml

# 30% writes, 70% reads
./benchmark -operation mixed -config configs/mixed_read_heavy.yaml

# Balanced workload
./benchmark -operation mixed -config configs/mixed_balanced.yaml
```

### 4. Protocol Comparison

```bash
# gRPC performance
./benchmark -target grpc://localhost:50051 -config configs/protocol_test.yaml

# HTTP REST performance
./benchmark -target http://localhost:8080 -config configs/protocol_test.yaml

# MQTT performance
./benchmark -target mqtt://localhost:1883 -config configs/protocol_test.yaml

# Kafka performance
./benchmark -target kafka://localhost:9092 -config configs/protocol_test.yaml
```

### 5. Privacy Operations

```bash
# K-anonymity overhead
./benchmark -operation privacy -config configs/privacy_k_anonymity.yaml

# Differential privacy impact
./benchmark -operation privacy -config configs/privacy_differential.yaml

# Synthetic data generation
./benchmark -operation generate -config configs/synthetic_generation.yaml
```

## Metrics and Analysis

### Performance Metrics

1. **Throughput Metrics**
   - Operations per second (ops/sec)
   - Points per second
   - Bytes per second
   - Batches per second

2. **Latency Metrics**
   - Minimum, maximum, mean
   - Standard deviation
   - Percentiles (p50, p90, p95, p99, p99.9)
   - Histogram distribution

3. **Error Metrics**
   - Total errors
   - Error rate
   - Error type breakdown
   - Retry statistics

4. **System Metrics**
   - CPU usage (average, max)
   - Memory usage (average, max)
   - Goroutine count
   - GC pause time and frequency

### Report Formats

#### JSON Report
```json
{
  "name": "TSIoT Performance Benchmark",
  "duration": "5m0s",
  "total_operations": 150000,
  "throughput_ops_sec": 500.0,
  "operations": {
    "insert": {
      "count": 90000,
      "throughput_ops_sec": 300.0,
      "latency": {
        "min": "1ms",
        "max": "100ms",
        "mean": "10ms",
        "percentiles": {
          "p50": "8ms",
          "p99": "50ms"
        }
      }
    }
  }
}
```

#### Markdown Report
Generated report includes:
- Executive summary
- Detailed operation metrics
- Latency distribution charts
- System resource usage
- Error analysis
- Recommendations

#### HTML Report
Interactive report with:
- Real-time charts
- Sortable tables
- Latency heatmaps
- Comparative analysis
- Export capabilities

## Advanced Usage

### Custom Operations

Create custom operations for specific use cases:

```go
type CustomOperation struct {
    name   string
    client interface{}
    params map[string]interface{}
}

func (o *CustomOperation) Execute(ctx context.Context, data interface{}) error {
    // Custom operation logic
    return nil
}
```

### Load Patterns

Define custom load patterns:

```yaml
load_pattern:
  type: "stepped"
  steps:
    - duration: 1m
      concurrency: 10
    - duration: 2m
      concurrency: 50
    - duration: 1m
      concurrency: 100
    - duration: 1m
      concurrency: 50
```

### Data Generation

Configure realistic data patterns:

```yaml
data_patterns:
  - name: "iot_sensor"
    distribution: "normal"
    mean: 25.0
    stddev: 5.0
    
  - name: "network_traffic"
    distribution: "poisson"
    lambda: 100.0
    
  - name: "cpu_usage"
    distribution: "beta"
    alpha: 2.0
    beta: 5.0
```

## Performance Tuning

### Benchmark Configuration

1. **Concurrency Tuning**
   - Start with low concurrency and increase gradually
   - Monitor system resources
   - Find the saturation point

2. **Batch Size Optimization**
   - Test different batch sizes
   - Balance between throughput and latency
   - Consider memory constraints

3. **Connection Pooling**
   - Configure appropriate pool sizes
   - Monitor connection usage
   - Avoid connection exhaustion

### System Preparation

1. **Before Benchmarking**
   - Disable unnecessary services
   - Ensure sufficient resources
   - Clear caches if needed
   - Set appropriate ulimits

2. **During Benchmarking**
   - Monitor system metrics
   - Watch for bottlenecks
   - Check error logs
   - Verify data consistency

3. **After Benchmarking**
   - Analyze results thoroughly
   - Compare with baselines
   - Document findings
   - Plan optimizations

## Troubleshooting

### Common Issues

1. **Connection Errors**
   ```bash
   # Check endpoint availability
   nc -zv localhost 50051
   
   # Verify TLS configuration
   openssl s_client -connect localhost:50051
   ```

2. **Resource Exhaustion**
   ```bash
   # Increase file descriptors
   ulimit -n 65536
   
   # Monitor resource usage
   top -p $(pgrep benchmark)
   ```

3. **Inconsistent Results**
   - Use longer warmup periods
   - Increase benchmark duration
   - Control background processes
   - Use dedicated hardware

### Debug Mode

Enable detailed logging:

```bash
./benchmark -config benchmark.yaml -verbose -log-level debug
```

## Integration

### CI/CD Pipeline

```yaml
benchmark-job:
  stage: performance
  script:
    - ./benchmark -config ci/benchmark.yaml -format json -output results.json
    - ./analyze-results results.json --assert "throughput > 1000"
    - ./compare-results results.json baseline.json --threshold 10%
  artifacts:
    paths:
      - results.json
    reports:
      performance: results.json
```

### Monitoring Integration

Export metrics to monitoring systems:

```yaml
export:
  prometheus:
    enabled: true
    endpoint: "localhost:9090"
  
  grafana:
    enabled: true
    dashboard: "tsiot-benchmark"
  
  elasticsearch:
    enabled: true
    index: "benchmark-results"
```

### Automated Testing

Schedule regular benchmarks:

```bash
# Cron job for nightly benchmarks
0 2 * * * /opt/tsiot/tools/benchmark -config /etc/tsiot/nightly-benchmark.yaml
```

## Best Practices

1. **Reproducibility**
   - Use configuration files
   - Document environment details
   - Control variables
   - Multiple runs for consistency

2. **Realistic Workloads**
   - Model actual usage patterns
   - Include think time
   - Vary data characteristics
   - Test error scenarios

3. **Progressive Testing**
   - Start with simple scenarios
   - Gradually increase complexity
   - Isolate variables
   - Build performance baselines

4. **Result Analysis**
   - Look beyond averages
   - Analyze percentiles
   - Consider outliers
   - Correlate with system metrics

## Examples

### Example 1: Baseline Performance Test

```bash
./benchmark \
  -config configs/baseline.yaml \
  -duration 10m \
  -warmup 2m \
  -format markdown \
  -output baseline_$(date +%Y%m%d).md
```

### Example 2: Stress Test

```bash
./benchmark \
  -config configs/stress.yaml \
  -concurrency 1000 \
  -duration 1h \
  -format html \
  -output stress_test.html
```

### Example 3: Comparative Analysis

```bash
# Test v1.0
./benchmark -config test.yaml -output v1.0.json

# Test v1.1
./benchmark -config test.yaml -output v1.1.json

# Compare results
./compare-results v1.0.json v1.1.json -output comparison.html
```