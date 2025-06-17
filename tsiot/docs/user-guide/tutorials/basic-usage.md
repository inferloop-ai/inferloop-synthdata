# Basic Usage Tutorial

This tutorial covers the fundamental operations of the TSIOT synthetic data generation system.

## Prerequisites

- Go 1.24 or later
- Docker (for storage backends)
- Basic understanding of time series data

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/inferloop/tsiot
cd tsiot

# Install dependencies
make deps

# Build the CLI
make build-cli
```

### 2. Start Required Services

```bash
# Start InfluxDB and Redis using Docker Compose
docker-compose -f deployments/docker/docker-compose.dev.yml up -d influxdb redis
```

### 3. Generate Your First Time Series

```bash
# Generate a simple time series using statistical generator
./bin/cli generate \
  --generator statistical \
  --duration 24h \
  --frequency 1m \
  --output ./data/sample.json

# Generate using TimeGAN (deep learning approach)
./bin/cli generate \
  --generator timegan \
  --duration 7d \
  --frequency 5m \
  --features 3 \
  --output ./data/timegan_sample.json
```

### 4. Validate Generated Data

```bash
# Run quality validation
./bin/cli validate \
  --input ./data/sample.json \
  --tests statistical,quality \
  --output ./reports/validation.json

# Compare with reference data
./bin/cli validate \
  --input ./data/sample.json \
  --reference ./data/reference.json \
  --tests correlation,distribution
```

## Common Use Cases

### IoT Sensor Data Generation

```bash
# Generate temperature sensor data
./bin/cli generate \
  --generator arima \
  --pattern seasonal \
  --seasonality 24h \
  --trend increasing \
  --noise 0.1 \
  --duration 30d \
  --output ./data/temperature.json
```

### Multi-variate Time Series

```bash
# Generate correlated sensor readings
./bin/cli generate \
  --generator rnn \
  --features temperature,humidity,pressure \
  --correlation 0.7 \
  --duration 7d \
  --output ./data/multivariate.json
```

### Privacy-Preserving Generation

```bash
# Generate with differential privacy
./bin/cli generate \
  --generator ydata \
  --privacy-level high \
  --epsilon 1.0 \
  --delta 1e-5 \
  --input ./data/sensitive.json \
  --output ./data/private.json
```

## Understanding Output Formats

### JSON Format
```json
{
  "metadata": {
    "generator": "statistical",
    "duration": "24h",
    "frequency": "1m",
    "features": ["value"],
    "generated_at": "2024-06-14T10:00:00Z"
  },
  "data": [
    {
      "timestamp": "2024-06-14T10:00:00Z",
      "value": 23.5
    },
    {
      "timestamp": "2024-06-14T10:01:00Z", 
      "value": 23.7
    }
  ]
}
```

### CSV Format
```csv
timestamp,value
2024-06-14T10:00:00Z,23.5
2024-06-14T10:01:00Z,23.7
```

## Validation Reports

Validation reports include:
- **Statistical Tests**: Kolmogorov-Smirnov, Anderson-Darling
- **Quality Metrics**: Autocorrelation, entropy measures
- **Privacy Metrics**: Differential privacy compliance
- **Temporal Analysis**: Trend, seasonality, stationarity

Example validation output:
```json
{
  "overall_score": 0.85,
  "tests": {
    "kolmogorov_smirnov": {
      "p_value": 0.45,
      "passed": true
    },
    "autocorrelation": {
      "score": 0.78,
      "passed": true
    }
  }
}
```

## Configuration

### Environment Variables
```bash
export TSIOT_STORAGE_TYPE=influxdb
export TSIOT_INFLUXDB_URL=http://localhost:8086
export TSIOT_INFLUXDB_TOKEN=your-token
export TSIOT_LOG_LEVEL=info
```

### Configuration File
Create `config.yaml`:
```yaml
server:
  host: localhost
  port: 8080

storage:
  type: influxdb
  influxdb:
    url: http://localhost:8086
    token: your-token
    org: your-org
    bucket: tsiot

generators:
  statistical:
    default_noise: 0.1
  timegan:
    epochs: 100
    batch_size: 32
```

## Error Handling

Common errors and solutions:

### Storage Connection Failed
```bash
Error: failed to connect to InfluxDB: connection refused
```
**Solution**: Ensure InfluxDB is running: `docker-compose up -d influxdb`

### Generator Not Found
```bash
Error: generator 'xyz' not found
```
**Solution**: Use `./bin/cli generate --list` to see available generators

### Invalid Time Format
```bash
Error: invalid duration format
```
**Solution**: Use valid duration formats: `1h`, `30m`, `7d`, `1w`

## Next Steps

- Read [Best Practices](./best-practices.md) for production usage
- Explore [Advanced Features](./advanced-features.md) for complex scenarios
- Check the [API Reference](../../api/openapi.yaml) for programmatic access