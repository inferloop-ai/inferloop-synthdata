# Getting Started Guide

Welcome to the Inferloop Synthetic Data (Video)! This guide will help you get up and running quickly.

## Prerequisites

Before you begin, ensure you have:

- **Docker** (version 20.0 or later)
- **Docker Compose** (version 2.0 or later)
- **Git** for cloning the repository
- **8GB+ RAM** for local development
- **20GB+ storage** for video processing

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd inferloop-synthdata/video

# Setup development environment
./scripts/setup/dev-environment.sh
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional for quick start)
# nano .env
```

### 3. Deploy Local Stack

```bash
# Build and deploy all services
make deploy

# Check deployment status
make status
```

### 4. Verify Installation

```bash
# Run health checks
./scripts/deployment/health-check.sh

# Access the API documentation
open http://localhost:8080/docs
```

## First Pipeline Run

### Option 1: Web Interface

1. Visit http://localhost:8080
2. Navigate to the pipeline creation interface
3. Select a vertical (e.g., "autonomous_vehicles")
4. Configure your data sources and quality requirements
5. Start the pipeline and monitor progress

### Option 2: API Call

```bash
curl -X POST "http://localhost:8080/api/v1/pipeline/start" \
  -H "Content-Type: application/json" \
  -d '{
    "vertical": "autonomous_vehicles",
    "data_sources": [
      {
        "source_type": "web",
        "url": "https://example.com/traffic-videos"
      }
    ],
    "generation_config": {
      "engine": "unreal",
      "duration_seconds": 60,
      "resolution": "1920x1080"
    },
    "quality_requirements": {
      "min_label_accuracy": 0.92,
      "max_frame_lag_ms": 300
    },
    "delivery_config": {
      "format": "mp4",
      "delivery_method": "download"
    }
  }'
```

### Option 3: SDK Usage

```python
from video_synth_sdk import VideoClient

# Initialize client
client = VideoClient(api_url="http://localhost:8080")

# Start pipeline
pipeline = client.start_pipeline(
    vertical="robotics",
    data_sources=[{"source_type": "upload", "file": "robot_task.mp4"}],
    quality_requirements={"min_precision": 0.98}
)

# Monitor progress
status = client.get_pipeline_status(pipeline.id)
print(f"Status: {status.current_stage} - {status.progress}%")
```

## Understanding the Pipeline

### Stage 1: Data Ingestion
- Scrapes or receives real-world video data
- Validates format and quality
- Stores in object storage (MinIO)

### Stage 2: Metrics Extraction
- Analyzes video quality (PSNR, SSIM, LPIPS)
- Detects objects and motion patterns
- Establishes quality benchmarks

### Stage 3: Synthetic Generation
- Generates synthetic video using selected engine
- Applies real-world parameters
- Optimizes for target vertical

### Stage 4: Validation
- Compares synthetic video against real-world metrics
- Performs compliance checks
- Generates quality report

### Stage 5: Delivery
- Provides access via multiple channels
- Implements authentication and rate limiting
- Delivers final content

## Monitoring and Debugging

### View Service Logs
```bash
# All services
make logs

# Specific service
docker-compose logs -f orchestration-service
```

### Check Service Health
```bash
# Quick health check
make status

# Detailed health check
./scripts/deployment/health-check.sh
```

### Access Monitoring Dashboards
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)

## Troubleshooting

### Common Issues

1. **Services Won't Start**
   ```bash
   # Check Docker daemon
   docker info
   
   # Restart services
   make restart
   ```

2. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8080
   
   # Modify ports in docker-compose.yml if needed
   ```

3. **Out of Memory**
   ```bash
   # Check Docker memory
   docker system df
   
   # Increase Docker memory limit to 8GB+
   ```

4. **Permission Errors**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER .
   chmod +x scripts/**/*.sh
   ```

## Next Steps

- Explore [API Documentation](../api-specifications/)
- Try [Example Use Cases](../../examples/)
- Learn about [Vertical Configurations](../../verticals/)
- Set up [Production Deployment](../deployment-guides/)

## Support

- Documentation: `/docs/`
- Examples: `/examples/`
- Issues: GitHub Issues
- Community: Discord/Slack
