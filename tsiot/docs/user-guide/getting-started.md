# Getting Started with TSIoT

Welcome to TSIoT! This guide will help you get up and running with synthetic time series generation in minutes.

## Prerequisites

Before you begin, ensure you have:

- **Go 1.21+** installed ([Download Go](https://golang.org/dl/))
- **Git** for cloning the repository
- **Docker** (optional, for containerized deployment)
- **Make** (optional, for using Makefile commands)

## Quick Start

### 1. Installation

#### Option A: Install from Source

```bash
# Clone the repository
git clone https://github.com/inferloop/tsiot.git
cd tsiot

# Install dependencies and build
make install

# Verify installation
tsiot-cli version
```

#### Option B: Use Docker

```bash
# Pull the latest image
docker pull inferloop/tsiot:latest

# Run the server
docker run -d -p 8080:8080 --name tsiot-server inferloop/tsiot:latest

# Run CLI commands
docker run --rm inferloop/tsiot:latest tsiot-cli --help
```

#### Option C: Download Pre-built Binaries

```bash
# Download for your platform from releases
curl -L https://github.com/inferloop/tsiot/releases/latest/download/tsiot-linux-amd64.tar.gz | tar xz

# Move to PATH
sudo mv tsiot-cli tsiot-server /usr/local/bin/
```

### 2. Basic Configuration

Create a configuration file:

```bash
# Create config directory
mkdir -p ~/.tsiot

# Create basic config
cat > ~/.tsiot/config.yaml << EOF
server:
  host: localhost
  port: 8080

storage:
  backend: file
  path: ~/.tsiot/data

generators:
  default: statistical
  
logging:
  level: info
EOF
```

### 3. Start the Server

```bash
# Start with default configuration
tsiot-server

# Or with custom config
tsiot-server --config ~/.tsiot/config.yaml

# Run in background
tsiot-server --daemon
```

Verify the server is running:

```bash
curl http://localhost:8080/health
# Should return: {"status":"healthy"}
```

### 4. Generate Your First Dataset

#### Using the CLI

```bash
# Generate simple temperature data
tsiot-cli generate \
  --generator statistical \
  --sensor-type temperature \
  --points 100 \
  --output temperature_data.csv

# View the generated data
head temperature_data.csv
```

#### Using the API

```bash
# Generate data via REST API
curl -X POST http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "generator": "statistical",
    "sensor_type": "temperature",
    "points": 100,
    "frequency": "1m"
  }' \
  -o api_temperature_data.json
```

#### Using Python SDK

```python
# Install the SDK
pip install tsiot-sdk

# Generate data
from tsiot import Client

client = Client("http://localhost:8080")
data = client.generate(
    generator="statistical",
    sensor_type="temperature",
    points=100,
    frequency="1m"
)

# Save to file
data.to_csv("python_temperature_data.csv")
print(f"Generated {len(data)} data points")
```

## Core Concepts

### Generators

TSIoT provides multiple generation algorithms:

1. **Statistical** - Fast, simple patterns
2. **ARIMA** - Time series with trends and seasonality
3. **TimeGAN** - Deep learning for complex patterns
4. **RNN/LSTM** - Sequential patterns
5. **YData** - Privacy-preserving generation

### Sensor Types

Pre-configured sensor types include:

- `temperature` - Range: -40°C to 60°C
- `humidity` - Range: 0% to 100%
- `pressure` - Range: 900 to 1100 hPa
- `motion` - 3-axis accelerometer data
- `energy` - Power consumption in kWh
- `custom` - Define your own parameters

### Time Frequencies

Supported time intervals:

- `1s`, `5s`, `10s`, `30s` - Seconds
- `1m`, `5m`, `15m`, `30m` - Minutes  
- `1h`, `6h`, `12h` - Hours
- `1d`, `7d` - Days

## Common Use Cases

### 1. Generate Training Data for ML

```bash
# Generate 10,000 points with anomalies
tsiot-cli generate \
  --generator timegan \
  --sensor-type temperature \
  --points 10000 \
  --anomalies 50 \
  --anomaly-magnitude 3.0 \
  --output training_data.csv
```

### 2. Simulate Multiple Sensors

```bash
# Generate data for 10 sensors
for i in {1..10}; do
  tsiot-cli generate \
    --generator statistical \
    --sensor-type temperature \
    --sensor-id "sensor_$i" \
    --points 1440 \
    --frequency 1m \
    --output "sensor_${i}_data.csv" &
done
wait
```

### 3. Create Continuous Stream

```bash
# Stream data in real-time
tsiot-cli stream \
  --generator arima \
  --sensor-type humidity \
  --frequency 5s \
  --output - | while read line; do
    echo "New reading: $line"
    # Process each data point
done
```

### 4. Generate Privacy-Safe Data

```bash
# Generate with differential privacy
tsiot-cli generate \
  --generator ydata \
  --sensor-type energy \
  --points 5000 \
  --privacy-epsilon 1.0 \
  --privacy-delta 1e-5 \
  --output private_energy_data.csv
```

## Validation and Quality

### Validate Generated Data

```bash
# Basic validation
tsiot-cli validate --input temperature_data.csv

# Compare with reference data
tsiot-cli validate \
  --input synthetic_data.csv \
  --reference real_data.csv \
  --metrics all
```

### Quality Metrics

- **Statistical Similarity**: KS test, Chi-square test
- **Temporal Consistency**: Autocorrelation, trend preservation
- **Distribution Matching**: Mean, variance, skewness
- **Anomaly Detection**: Outlier identification

## Working with Storage Backends

### Local File Storage

```yaml
storage:
  backend: file
  path: ./data
  format: parquet  # csv, json, parquet
```

### InfluxDB

```yaml
storage:
  backend: influxdb
  connection_string: http://localhost:8086
  token: your-token
  org: your-org
  bucket: tsiot
```

### TimescaleDB

```yaml
storage:
  backend: timescale
  connection_string: postgres://user:pass@localhost:5432/tsiot
  compression: true
```

## Next Steps

### Explore Advanced Features

1. **[CLI Reference](cli-reference.md)** - Complete command documentation
2. **[SDK Guides](sdk-guides/)** - Language-specific integration
3. **[Tutorials](tutorials/)** - Step-by-step guides
4. **[API Documentation](../api/)** - REST API reference

### Learn Best Practices

- **[Data Generation Best Practices](tutorials/best-practices.md)**
- **[Performance Optimization](../developer/performance-tuning.md)**
- **[Security Guidelines](../architecture/security.md)**

### Join the Community

- **GitHub**: [github.com/inferloop/tsiot](https://github.com/inferloop/tsiot)
- **Discord**: [discord.gg/inferloop](https://discord.gg/inferloop)
- **Forum**: [forum.inferloop.com](https://forum.inferloop.com)

## Example: End-to-End Workflow

Here's a complete example generating and validating temperature sensor data:

```bash
#!/bin/bash

# 1. Start the server
tsiot-server --daemon

# 2. Wait for server to be ready
until curl -s http://localhost:8080/health > /dev/null; do
  sleep 1
done

# 3. Generate training data
echo "Generating training data..."
tsiot-cli generate \
  --generator timegan \
  --sensor-type temperature \
  --points 10000 \
  --start-time "2024-01-01T00:00:00Z" \
  --frequency 5m \
  --output training_data.csv

# 4. Generate test data with anomalies
echo "Generating test data..."
tsiot-cli generate \
  --generator timegan \
  --sensor-type temperature \
  --points 1000 \
  --anomalies 10 \
  --start-time "2024-01-08T00:00:00Z" \
  --frequency 5m \
  --output test_data.csv

# 5. Validate the generated data
echo "Validating data quality..."
tsiot-cli validate \
  --input test_data.csv \
  --reference training_data.csv \
  --report-format json \
  --output validation_report.json

# 6. Check validation results
if jq -e '.quality_score > 0.8' validation_report.json > /dev/null; then
  echo " Data quality is good (score > 0.8)"
else
  echo " Data quality needs improvement"
fi

# 7. Export to different formats
echo "Exporting data..."
tsiot-cli convert \
  --input test_data.csv \
  --output test_data.parquet \
  --format parquet

# 8. Stop the server
tsiot-cli server stop

echo "Workflow complete!"
```

## Troubleshooting

### Common Issues

**Server won't start**
- Check if port 8080 is already in use: `lsof -i :8080`
- Verify config file syntax: `tsiot-server --config-check`
- Check logs: `tail -f ~/.tsiot/logs/server.log`

**Generation fails**
- Ensure sufficient memory for large datasets
- Check generator-specific requirements
- Verify storage backend connectivity

**Poor data quality**
- Increase training epochs for ML generators
- Use appropriate generator for your use case
- Tune algorithm-specific parameters

### Getting Help

```bash
# Built-in help
tsiot-cli --help
tsiot-cli generate --help

# Check version and compatibility
tsiot-cli version --verbose

# Run diagnostics
tsiot-cli diagnose
```

For more help, see our [FAQ](faq.md) or [file an issue](https://github.com/inferloop/tsiot/issues).