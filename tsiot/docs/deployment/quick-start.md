# TSIOT Quick Start Guide

## Overview

This guide will help you get TSIOT (Time Series IoT Synthetic Data) platform up and running quickly in various environments. Choose the deployment method that best fits your needs.

## Prerequisites

### System Requirements
- **CPU**: 4+ cores (8+ recommended for production)
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 50GB+ available space
- **Network**: Internet connection for downloading dependencies

### Software Requirements
- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.24+ (for K8s deployment)
- Go 1.21+ (for development)

## Quick Start Options

### Option 1: Docker Compose (Recommended for Testing)

#### 1. Clone the Repository
```bash
git clone https://github.com/your-org/tsiot.git
cd tsiot
```

#### 2. Start with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

#### 3. Verify Installation
```bash
# Health check
curl http://localhost:8080/health

# API info
curl http://localhost:8080/api/v1/info
```

#### 4. Generate Your First Time Series
```bash
curl -X POST http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "type": "arima",
    "length": 100,
    "parameters": {
      "ar_params": [0.5, -0.3],
      "ma_params": [0.2]
    }
  }'
```

### Option 2: Kubernetes (Production Ready)

#### 1. Install Dependencies
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://get.helm.sh/helm-v3.12.0-linux-amd64.tar.gz | tar xz
sudo mv linux-amd64/helm /usr/local/bin/
```

#### 2. Deploy with Helm
```bash
# Add TSIOT Helm repository
helm repo add tsiot https://charts.tsiot.io
helm repo update

# Install with default values
helm install tsiot tsiot/tsiot

# Or with custom values
helm install tsiot tsiot/tsiot -f values-production.yaml
```

#### 3. Access the Service
```bash
# Get service URL
kubectl get svc tsiot-api

# Port forward for local access
kubectl port-forward svc/tsiot-api 8080:80

# Test the service
curl http://localhost:8080/health
```

### Option 3: Binary Installation (Development)

#### 1. Build from Source
```bash
# Clone repository
git clone https://github.com/your-org/tsiot.git
cd tsiot

# Build the application
make build

# Run locally
./bin/tsiot-server --config config/local.yaml
```

#### 2. Quick Configuration
```yaml
# config/local.yaml
server:
  host: "localhost"
  port: 8080
  
database:
  type: "sqlite"
  connection: "file:tsiot.db"
  
generators:
  default_type: "arima"
  max_length: 100000
  
logging:
  level: "info"
  format: "json"
```

## Configuration

### Environment Variables
```bash
# Core settings
export TSIOT_HOST=localhost
export TSIOT_PORT=8080
export TSIOT_LOG_LEVEL=info

# Database
export TSIOT_DB_TYPE=postgres
export TSIOT_DB_HOST=localhost
export TSIOT_DB_PORT=5432
export TSIOT_DB_NAME=tsiot
export TSIOT_DB_USER=tsiot
export TSIOT_DB_PASSWORD=your_password

# Security
export TSIOT_JWT_SECRET=your_jwt_secret
export TSIOT_API_KEY_SECRET=your_api_key_secret

# Optional: External services
export TSIOT_REDIS_URL=redis://localhost:6379
export TSIOT_KAFKA_BROKERS=localhost:9092
```

### Configuration File
```yaml
# config/production.yaml
server:
  host: "0.0.0.0"
  port: 8080
  read_timeout: "30s"
  write_timeout: "30s"
  max_header_bytes: 1048576

database:
  type: "postgres"
  host: "postgres"
  port: 5432
  name: "tsiot"
  user: "tsiot"
  password: "${TSIOT_DB_PASSWORD}"
  ssl_mode: "require"
  max_open_conns: 25
  max_idle_conns: 5

cache:
  type: "redis"
  url: "redis://redis:6379"
  ttl: "1h"

generators:
  default_type: "arima"
  max_length: 1000000
  timeout: "5m"
  workers: 10

storage:
  backends:
    - name: "timescaledb"
      type: "timescaledb"
      primary: true
    - name: "s3"
      type: "s3"
      bucket: "tsiot-data"
      region: "us-west-2"

security:
  jwt:
    secret: "${TSIOT_JWT_SECRET}"
    expiry: "24h"
  rate_limiting:
    enabled: true
    requests_per_minute: 1000
  cors:
    enabled: true
    allowed_origins: ["*"]

monitoring:
  metrics:
    enabled: true
    path: "/metrics"
  health:
    enabled: true
    path: "/health"
  tracing:
    enabled: true
    endpoint: "http://jaeger:14268/api/traces"

logging:
  level: "info"
  format: "json"
  output: "stdout"
```

## Service Architecture

### Default Services (Docker Compose)
```
                                      
   TSIOT API            PostgreSQL    
   Port: 8080    �   $   Port: 5432    
                                      
         
         �
                                      
     Redis                Grafana     
   Port: 6379           Port: 3000    
                                      
```

### Available Endpoints

#### Health and Info
```bash
# Health check
GET /health
GET /ready

# Service information
GET /api/v1/info
GET /metrics
```

#### Core API
```bash
# Generation
POST /api/v1/generate
POST /api/v1/batch/generate
GET  /api/v1/generators

# Validation
POST /api/v1/validate
GET  /api/v1/validators

# Analytics
POST /api/v1/analyze
GET  /api/v1/analytics/capabilities

# Data export
POST /api/v1/export
```

## Testing Your Installation

### 1. Basic Functionality Test
```bash
#!/bin/bash

BASE_URL="http://localhost:8080"

echo "Testing TSIOT installation..."

# Health check
echo "1. Health check..."
curl -f $BASE_URL/health || exit 1

# Get info
echo "2. Service info..."
curl -f $BASE_URL/api/v1/info

# Generate ARIMA series
echo "3. Generate ARIMA time series..."
curl -X POST $BASE_URL/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "type": "arima",
    "length": 50,
    "parameters": {
      "ar_params": [0.7, -0.2],
      "ma_params": [0.3]
    }
  }' > arima_output.json

# Validate generated data
echo "4. Validate generated data..."
curl -X POST $BASE_URL/api/v1/validate \
  -H "Content-Type: application/json" \
  -d @arima_output.json

echo " All tests passed!"
```

### 2. Load Testing
```bash
# Install tools
go install github.com/tsenart/vegeta@latest

# Simple load test
echo "GET http://localhost:8080/health" | vegeta attack -duration=30s -rate=100 | vegeta report

# Generation load test
echo 'POST http://localhost:8080/api/v1/generate
Content-Type: application/json

{
  "type": "statistical",
  "length": 100,
  "parameters": {
    "distribution": "normal",
    "mean": 0,
    "std": 1
  }
}' | vegeta attack -duration=60s -rate=10 | vegeta report
```

## Common Issues and Solutions

### Issue 1: Port Already in Use
```bash
# Check what's using port 8080
sudo lsof -i :8080

# Kill the process or change port
export TSIOT_PORT=8081
```

### Issue 2: Database Connection Failed
```bash
# Check database logs
docker-compose logs postgres

# Verify database is ready
docker-compose exec postgres pg_isready -U tsiot

# Reset database
docker-compose down -v
docker-compose up -d
```

### Issue 3: Out of Memory
```bash
# Increase Docker memory limits
# Edit docker-compose.yml:
services:
  tsiot-api:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

### Issue 4: Permission Denied
```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod +x scripts/*.sh

# Fix Docker socket permissions
sudo chmod 666 /var/run/docker.sock
```

## Next Steps

### 1. Explore the Web Dashboard
```bash
# Access dashboard (if enabled)
open http://localhost:3000

# Default credentials
Username: admin
Password: admin
```

### 2. Try Different Generators
```bash
# LSTM Generator
curl -X POST http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "type": "lstm",
    "length": 200,
    "parameters": {
      "hidden_size": 64,
      "sequence_length": 20
    }
  }'

# Statistical Generator
curl -X POST http://localhost:8080/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "type": "statistical",
    "length": 1000,
    "parameters": {
      "distribution": "exponential",
      "lambda": 1.5
    }
  }'
```

### 3. Use the SDKs
```python
# Python SDK example
from tsiot import TSIOTClient

client = TSIOTClient("http://localhost:8080")
ts = client.generate({
    "type": "arima",
    "length": 100,
    "parameters": {
        "ar_params": [0.5, -0.3],
        "ma_params": [0.2]
    }
})

print(f"Generated {len(ts)} data points")
```

### 4. Configure Production Settings
- Set up TLS certificates
- Configure authentication
- Set up monitoring and alerting
- Configure backup strategies
- Set up log aggregation

## Support

### Documentation
- [Architecture Guide](../architecture/overview.md)
- [API Reference](../api/openapi.yaml)
- [SDK Guides](../user-guide/sdk-guides/)

### Community
- GitHub Issues: [Report bugs or request features](https://github.com/your-org/tsiot/issues)
- Discord: [Join our community](https://discord.gg/tsiot)
- Documentation: [Full documentation](https://docs.tsiot.io)

### Enterprise Support
For production deployments and enterprise support:
- Email: support@tsiot.io
- Enterprise docs: [Enterprise deployment guide](./enterprise-setup.md)

## Security Note

� **Important**: This quick start guide uses default configurations suitable for development and testing. For production deployments, please refer to the [Security Guide](../architecture/security.md) and implement proper security measures including:

- Strong authentication and authorization
- TLS encryption
- Firewall configuration
- Regular security updates
- Monitoring and alerting

Happy generating! =�