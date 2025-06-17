# Frequently Asked Questions (FAQ)

## General Questions

### What is TSIoT?
TSIoT (Time Series IoT) is a comprehensive platform for generating, validating, and managing synthetic time series data for IoT applications. It provides multiple generation algorithms, quality validation, and integrates with Model Context Protocol (MCP) for seamless AI-powered workflows.

### What are the main use cases?
- **Testing and Development**: Generate realistic test data without using production data
- **Machine Learning**: Create training datasets for time series models
- **System Testing**: Simulate IoT sensor data for load testing
- **Privacy Compliance**: Generate synthetic data that preserves patterns without exposing sensitive information
- **Proof of Concepts**: Quickly prototype with realistic data

### What types of time series can be generated?
- Temperature sensors
- Humidity sensors
- Pressure readings
- Motion/accelerometer data
- Energy consumption
- Network traffic
- Custom sensor types (configurable)

### Which generation algorithms are available?
- **TimeGAN**: Deep learning-based generation preserving temporal dynamics
- **ARIMA**: Statistical modeling for trends and seasonality
- **RNN/LSTM**: Neural networks for sequential patterns
- **Statistical**: Simple distributions and patterns
- **YData**: Privacy-preserving synthetic data

## Installation & Setup

### What are the system requirements?
- **OS**: Linux, macOS, Windows (with WSL)
- **Go**: 1.21 or later
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 1GB for application + data storage needs
- **Docker**: Optional but recommended

### How do I install TSIoT?
```bash
# From source
git clone https://github.com/inferloop/tsiot.git
cd tsiot
make install

# Using Docker
docker pull inferloop/tsiot:latest

# Using Helm
helm repo add inferloop https://charts.inferloop.com
helm install tsiot inferloop/tsiot
```

### How do I configure storage backends?
TSIoT supports multiple storage backends:
- **InfluxDB**: Best for time series data
- **TimescaleDB**: PostgreSQL-based time series
- **S3**: Object storage for large datasets
- **Local Files**: For development and testing

Configure in `config.yaml`:
```yaml
storage:
  backend: "influxdb"
  connection_string: "http://localhost:8086"
```

## Usage Questions

### How do I generate basic time series data?
```bash
# Generate temperature data
tsiot-cli generate --generator timegan --sensor-type temperature --points 1000

# Generate with specific frequency
tsiot-cli generate --generator arima --frequency 5m --duration 24h

# Generate with anomalies
tsiot-cli generate --generator statistical --anomalies 5 --anomaly-magnitude 2.0
```

### How do I validate synthetic data quality?
```bash
# Basic validation
tsiot-cli validate --input synthetic.csv

# Compare with reference data
tsiot-cli validate --input synthetic.csv --reference real.csv --metrics all

# Generate quality report
tsiot-cli validate --input data.csv --report-format html --output report.html
```

### Can I use TSIoT programmatically?
Yes! TSIoT provides SDKs for multiple languages:

**Python**:
```python
from tsiot import Client

client = Client("http://localhost:8080")
data = client.generate(
    generator="timegan",
    sensor_type="temperature",
    points=1000
)
```

**Go**:
```go
client := tsiot.NewClient("http://localhost:8080")
data, err := client.Generate(ctx, tsiot.GenerateRequest{
    Generator:  "timegan",
    SensorType: "temperature",
    Points:     1000,
})
```

### How do I integrate with MCP?
TSIoT natively supports Model Context Protocol:

1. Start the MCP server:
```bash
tsiot-server --mcp-enabled --transport stdio
```

2. Connect from an MCP client:
```python
import mcp

client = mcp.Client("stdio://tsiot")
result = client.call("generate_timeseries", {
    "generator": "timegan",
    "sensor_type": "temperature"
})
```

## Data & Privacy

### Is the generated data truly synthetic?
Yes, the generated data is completely synthetic and does not contain any real sensor readings. The algorithms learn patterns and distributions from training data (if provided) but generate entirely new data points.

### How does privacy-preserving generation work?
The YData generator implements differential privacy techniques:
- Adds calibrated noise to protect individual data points
- Preserves aggregate statistics and patterns
- Configurable privacy budget (epsilon parameter)
- Suitable for compliance with GDPR, HIPAA, etc.

### Can I control the randomness/reproducibility?
Yes, you can set seeds for reproducible generation:
```bash
tsiot-cli generate --seed 42 --generator timegan
```

### What formats are supported for import/export?
- **CSV**: Standard comma-separated values
- **JSON**: Structured JSON with metadata
- **Parquet**: Efficient columnar format
- **InfluxDB Line Protocol**: Native InfluxDB format
- **Custom formats**: Via plugins

## Performance & Scaling

### How much data can TSIoT generate?
TSIoT can generate millions of data points. Performance depends on:
- Generation algorithm (Statistical > ARIMA > RNN > TimeGAN)
- Hardware resources
- Storage backend
- Data complexity

Typical performance:
- Statistical: 1M+ points/second
- ARIMA: 100K points/second  
- TimeGAN: 10K points/second

### Can TSIoT scale horizontally?
Yes, TSIoT supports distributed generation:
```yaml
# Enable distributed mode
workers:
  enabled: true
  count: 4
  queue: "redis://localhost:6379"
```

### How do I optimize for large datasets?
1. Use batch generation
2. Enable parallel processing
3. Choose appropriate storage backend
4. Use streaming for continuous generation
5. Implement data partitioning

## Troubleshooting

### Common Issues

**Q: Generation is slow**
A: Try:
- Using a simpler algorithm (statistical/ARIMA)
- Reducing batch size
- Enabling parallel workers
- Checking resource constraints

**Q: Out of memory errors**
A: 
- Reduce batch size
- Enable streaming mode
- Increase system memory
- Use disk-based storage

**Q: Validation scores are low**
A:
- Increase training epochs for ML models
- Tune algorithm parameters
- Ensure reference data quality
- Check for data preprocessing issues

**Q: Connection errors to storage**
A:
- Verify storage backend is running
- Check connection string
- Ensure network connectivity
- Review authentication credentials

### Getting Help

- **Documentation**: https://docs.inferloop.com/tsiot
- **GitHub Issues**: https://github.com/inferloop/tsiot/issues
- **Community Forum**: https://forum.inferloop.com
- **Email Support**: support@inferloop.com

## Advanced Topics

### Can I create custom generators?
Yes, implement the Generator interface:
```go
type Generator interface {
    Generate(config Config) (*TimeSeries, error)
    Validate() error
    GetInfo() GeneratorInfo
}
```

### How do I add custom validation metrics?
Implement the Validator interface and register it:
```go
type Validator interface {
    Validate(data *TimeSeries) (*ValidationResult, error)
    GetMetrics() []Metric
}
```

### Can I extend TSIoT with plugins?
Yes, TSIoT supports plugins for:
- Custom generators
- Storage backends  
- Validation metrics
- Export formats
- Preprocessing pipelines

### Is there a web UI?
Yes, a web dashboard is available:
```bash
# Start with UI enabled
tsiot-server --enable-ui --ui-port 3000
```

Access at http://localhost:3000

### How do I contribute?
We welcome contributions! See our [Contributing Guide](contributing.md) for:
- Code style guidelines
- Development setup
- Testing requirements
- PR process