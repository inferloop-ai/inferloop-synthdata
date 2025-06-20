# Time Series Synthetic Data MCP

A Model Context Protocol (MCP) server for generating, validating, and managing synthetic time series data for IoT applications.

## =� Features

- **Multiple Generation Algorithms**: TimeGAN, ARIMA, RNN, Statistical, and Privacy-preserving methods
- **Quality Validation**: Statistical tests, similarity metrics, and quality reports
- **Data Analysis**: Pattern detection, anomaly identification, and forecasting
- **Storage Backends**: InfluxDB, TimescaleDB, S3, and more
- **Privacy-Preserving**: Differential privacy and secure aggregation
- **Scalable Architecture**: Distributed workers and horizontal scaling
- **MCP Integration**: Native Model Context Protocol support
- **Multiple Protocols**: HTTP REST, gRPC, MQTT, Kafka
- **Observability**: Prometheus metrics, distributed tracing, structured logging

## <� Architecture

```
                                                           
   MCP Client           CLI Tools           Web Dashboard  
         ,                    ,                    ,       
                                                      
                                <                      
                                 
                                 �               
                            MCP Server           
                      (HTTP/gRPC/WebSocket)      
                                 ,               
                                  
                                 <                   
                                                    
             �                 �                 �        
       Generator           Validator         Analyzer     
         Agents              Agents           Agents      
                                                          
                                                    
                                 <                   
                                  
                                 �               
                          Storage Layer          
                     (InfluxDB/TimescaleDB/S3)   
                                                 
```

## =' Installation

### Prerequisites

- Go 1.21 or later
- Docker (optional)
- Make

### Build from Source

```bash
# Clone the repository
git clone https://github.com/inferloop/timeseries-synthetic-mcp.git
cd timeseries-synthetic-mcp

# Install dependencies
make deps

# Build all binaries
make build

# Or build individual components
make build-server    # MCP server
make build-cli       # CLI tool
make build-worker    # Background worker
```

### Docker

```bash
# Build Docker images
make docker-build

# Run with docker-compose
make docker-compose-up
```

## =� Quick Start

### 1. Start the MCP Server

```bash
# Development mode
make dev-server

# Or with binary
./bin/tsiot-server --port 8080 --log-level info
```

### 2. Generate Synthetic Data

```bash
# Using CLI
./bin/tsiot-cli generate \
  --generator timegan \
  --sensor-type temperature \
  --start-time "2024-01-01" \
  --frequency 1m \
  --output data.csv

# With anomalies
./bin/tsiot-cli generate \
  --generator arima \
  --anomalies 10 \
  --privacy \
  --output synthetic_data.json
```

### 3. Validate Data Quality

```bash
# Compare with reference data
./bin/tsiot-cli validate \
  --input synthetic_data.csv \
  --reference real_data.csv \
  --metrics all \
  --statistical-tests

# Quality report
./bin/tsiot-cli validate \
  --input data.csv \
  --report-format html \
  --output quality_report.html
```

### 4. Analyze Time Series

```bash
# Pattern analysis
./bin/tsiot-cli analyze \
  --input data.csv \
  --detect-anomalies \
  --seasonality \
  --forecast \
  --forecast-periods 24
```

### 5. Data Migration

```bash
# Migrate between storage backends
./bin/tsiot-cli migrate \
  --source "file://old_data.csv" \
  --dest "influx://localhost:8086/newdb" \
  --batch-size 1000
```

## =' Configuration

### Server Configuration

```yaml
# configs/environments/development.yaml
server:
  host: "0.0.0.0"
  port: 8080
  enable_tls: false

mcp:
  enabled: true
  transport: "stdio"

storage:
  backend: "influxdb"
  connection_string: "http://localhost:8086"

generators:
  timegan:
    enabled: true
    model_path: "./models/timegan"
  arima:
    enabled: true
    default_order: [2, 1, 2]

logging:
  level: "info"
  format: "json"
```

### CLI Configuration

```yaml
# ~/.tsiot/config.yaml
server_url: "http://localhost:8080"
default_output: "-"
default_format: "csv"

generators:
  timegan:
    enabled: true
    parameters:
      epochs: 1000
      batch_size: 32

preferences:
  color_output: true
  progress_bars: true
  timezone: "UTC"
```

## =� Generators

### TimeGAN
- **Use Case**: Complex temporal patterns
- **Features**: Deep learning-based, preserves temporal dynamics
- **Parameters**: epochs, batch_size, hidden_dim, num_layers

### ARIMA
- **Use Case**: Linear time series with trends/seasonality
- **Features**: Statistical modeling, interpretable parameters
- **Parameters**: order (p,d,q), seasonal_order

### RNN/LSTM
- **Use Case**: Sequential patterns, long-term dependencies
- **Features**: Recurrent neural networks, attention mechanisms
- **Parameters**: hidden_size, num_layers, dropout

### Statistical
- **Use Case**: Simple patterns, baseline generation
- **Features**: Gaussian, Markov chains, Fourier synthesis
- **Parameters**: distribution, noise_level, periodicity

### YData (Privacy-Preserving)
- **Use Case**: Sensitive data, privacy compliance
- **Features**: Differential privacy, federated learning
- **Parameters**: epsilon, delta, noise_multiplier

## = Validation Metrics

- **Statistical Similarity**: KS test, Anderson-Darling test
- **Trend Preservation**: Correlation analysis, slope comparison
- **Distribution Matching**: Wasserstein distance, Jensen-Shannon divergence
- **Temporal Properties**: Autocorrelation, cross-correlation
- **Quality Score**: Composite metric (0-100)

## =3 Docker Deployment

```bash
# Quick start with docker-compose
docker-compose -f deployments/docker/docker-compose.yml up -d

# Or build and run individually
docker build -f deployments/docker/Dockerfile.server -t tsiot-server .
docker run -p 8080:8080 -p 9090:9090 tsiot-server
```

## 8 Kubernetes Deployment

```bash
# Deploy with kubectl
kubectl apply -f deployments/kubernetes/

# Or with Helm
helm install tsiot deployments/helm/ --namespace tsiot --create-namespace
```

## =� Monitoring

- **Metrics**: Prometheus endpoint at `:9090/metrics`
- **Tracing**: Jaeger integration
- **Logging**: Structured JSON logs
- **Health Checks**: `/health` and `/ready` endpoints

## >� Development

### Project Structure

```
.
   cmd/                    # Entry points
      server/            # MCP server
      cli/               # CLI tool
      worker/            # Background worker
   internal/              # Private application code
      agents/            # MCP agents
      generators/        # Synthetic data generators
      validation/        # Quality validation
      storage/           # Storage backends
      protocols/         # Communication protocols
   pkg/                   # Public packages
   deployments/           # Deployment configs
   docs/                  # Documentation
   tests/                 # Test suites
   examples/              # Usage examples
```

### Testing

```bash
# Run all tests
make test

# Unit tests only
make test-unit

# Integration tests
make test-integration

# Benchmarks
make benchmark
```

### Code Quality

```bash
# Lint code
make lint

# Format code
make fmt

# Security scan
make security-scan
```

## > Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

See [CONTRIBUTING.md](docs/developer/contributing.md) for detailed guidelines.

## =� License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## = Links

- [Documentation](docs/)
- [API Reference](docs/api/)
- [Examples](examples/)
- [Architecture Guide](docs/architecture/)
- [Deployment Guide](docs/deployment/)

## <� Support

- [GitHub Issues](https://github.com/inferloop/timeseries-synthetic-mcp/issues)
- [Discussions](https://github.com/inferloop/timeseries-synthetic-mcp/discussions)
- [Wiki](https://github.com/inferloop/timeseries-synthetic-mcp/wiki)