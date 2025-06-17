# TSIOT Docker Deployment Guide

## Overview

This guide covers deploying TSIOT using Docker and Docker Compose, from simple single-container setups to full production environments with all dependencies.

## Prerequisites

### Required Software
- **Docker** 20.10+ 
- **Docker Compose** 2.0+
- **Git** (for cloning repository)

### System Requirements
- **Minimum**: 2 CPU cores, 4GB RAM, 20GB storage
- **Recommended**: 4+ CPU cores, 8GB+ RAM, 50GB+ storage

## Quick Start

### 1. Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/tsiot.git
cd tsiot

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f tsiot-api
```

### 2. Single Container (Simple Setup)

```bash
# Run TSIOT with SQLite (no external dependencies)
docker run -d \
  --name tsiot \
  -p 8080:8080 \
  -e TSIOT_DB_TYPE=sqlite \
  -e TSIOT_DB_CONNECTION="file:/data/tsiot.db" \
  -v tsiot-data:/data \
  tsiot/tsiot:latest

# Test the service
curl http://localhost:8080/health
```

## Docker Compose Configurations

### Development Setup

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  tsiot-api:
    image: tsiot/tsiot:latest
    ports:
      - "8080:8080"
    environment:
      - TSIOT_HOST=0.0.0.0
      - TSIOT_PORT=8080
      - TSIOT_LOG_LEVEL=debug
      - TSIOT_DB_TYPE=postgres
      - TSIOT_DB_HOST=postgres
      - TSIOT_DB_PORT=5432
      - TSIOT_DB_NAME=tsiot
      - TSIOT_DB_USER=tsiot
      - TSIOT_DB_PASSWORD=password
      - TSIOT_REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=tsiot
      - POSTGRES_USER=tsiot
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  tsiot-api:
    image: tsiot/tsiot:1.0.0
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '0.5'
          memory: 1G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    ports:
      - "8080:8080"
    environment:
      - TSIOT_HOST=0.0.0.0
      - TSIOT_PORT=8080
      - TSIOT_LOG_LEVEL=info
      - TSIOT_LOG_FORMAT=json
      - TSIOT_DB_TYPE=postgres
      - TSIOT_DB_HOST=postgres
      - TSIOT_DB_SSL_MODE=require
      - TSIOT_REDIS_URL=redis://redis:6379
      - TSIOT_KAFKA_BROKERS=kafka:9092
    env_file:
      - .env.prod
    depends_on:
      - postgres
      - redis
      - kafka
    volumes:
      - tsiot_data:/app/data
      - tsiot_logs:/app/logs
    networks:
      - tsiot-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=tsiot
      - POSTGRES_USER=tsiot
      - POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
    secrets:
      - postgres_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    networks:
      - tsiot-network
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.25'
          memory: 512M
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tsiot"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - tsiot-network
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 1G
        reservations:
          cpus: '0.1'
          memory: 256M

  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    depends_on:
      - zookeeper
    volumes:
      - kafka_data:/var/lib/kafka/data
    networks:
      - tsiot-network

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    volumes:
      - zookeeper_data:/var/lib/zookeeper/data
    networks:
      - tsiot-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - tsiot-api
    networks:
      - tsiot-network

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    networks:
      - tsiot-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - tsiot-network

secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt

volumes:
  tsiot_data:
  tsiot_logs:
  postgres_data:
  redis_data:
  kafka_data:
  zookeeper_data:
  prometheus_data:
  grafana_data:

networks:
  tsiot-network:
    driver: bridge
```

## Configuration Files

### Environment Variables (.env)

```bash
# .env.prod
# Database Configuration
TSIOT_DB_PASSWORD=your_secure_password
TSIOT_JWT_SECRET=your_jwt_secret_key_here
TSIOT_API_KEY_SECRET=your_api_key_secret_here

# Redis Configuration
REDIS_PASSWORD=your_redis_password

# External Services
TSIOT_INFLUXDB_URL=http://influxdb:8086
TSIOT_INFLUXDB_TOKEN=your_influxdb_token

# Monitoring
TSIOT_METRICS_ENABLED=true
TSIOT_TRACING_ENABLED=true
TSIOT_TRACING_ENDPOINT=http://jaeger:14268/api/traces

# Performance
TSIOT_WORKERS=4
TSIOT_MAX_CONNECTIONS=25
TSIOT_CACHE_TTL=3600
```

### Nginx Configuration

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream tsiot-backend {
        server tsiot-api:8080;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/ssl/cert.pem;
        ssl_certificate_key /etc/ssl/key.pem;

        location / {
            proxy_pass http://tsiot-backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
            
            # Buffer settings
            proxy_buffering on;
            proxy_buffer_size 128k;
            proxy_buffers 4 256k;
            proxy_busy_buffers_size 256k;
        }

        location /health {
            proxy_pass http://tsiot-backend/health;
            access_log off;
        }

        location /metrics {
            proxy_pass http://tsiot-backend/metrics;
            allow 127.0.0.1;
            allow 10.0.0.0/8;
            deny all;
        }
    }
}
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "tsiot_rules.yml"

scrape_configs:
  - job_name: 'tsiot'
    static_configs:
      - targets: ['tsiot-api:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

## Building Custom Images

### Dockerfile

```dockerfile
# Dockerfile
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o tsiot ./cmd/server

FROM alpine:latest

RUN apk --no-cache add ca-certificates curl
WORKDIR /root/

COPY --from=builder /app/tsiot .
COPY --from=builder /app/config ./config

# Create non-root user
RUN addgroup -S tsiot && adduser -S tsiot -G tsiot
USER tsiot

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["./tsiot"]
```

### Multi-stage Build with Dependencies

```dockerfile
# Dockerfile.full
FROM node:18-alpine AS web-builder

WORKDIR /app/web
COPY web/package*.json ./
RUN npm ci --only=production

COPY web/ ./
RUN npm run build

FROM golang:1.21-alpine AS go-builder

RUN apk add --no-cache git

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
COPY --from=web-builder /app/web/dist ./web/dist

RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags '-w -s' -o tsiot ./cmd/server

FROM alpine:latest

RUN apk --no-cache add ca-certificates curl tzdata
WORKDIR /app

COPY --from=go-builder /app/tsiot .
COPY --from=go-builder /app/config ./config
COPY --from=go-builder /app/web/dist ./web/dist

RUN addgroup -S tsiot && adduser -S tsiot -G tsiot
RUN chown -R tsiot:tsiot /app
USER tsiot

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["./tsiot"]
```

### Build Script

```bash
#!/bin/bash
# build.sh

set -e

VERSION=${1:-latest}
REGISTRY=${REGISTRY:-docker.io/tsiot}

echo "Building TSIOT version: $VERSION"

# Build the image
docker build -t $REGISTRY/tsiot:$VERSION .

# Tag as latest if this is a release
if [[ $VERSION != "latest" ]]; then
    docker tag $REGISTRY/tsiot:$VERSION $REGISTRY/tsiot:latest
fi

# Push to registry
if [[ "${PUSH}" == "true" ]]; then
    docker push $REGISTRY/tsiot:$VERSION
    if [[ $VERSION != "latest" ]]; then
        docker push $REGISTRY/tsiot:latest
    fi
fi

echo "Build complete: $REGISTRY/tsiot:$VERSION"
```

## Deployment Scripts

### Deployment Script

```bash
#!/bin/bash
# deploy.sh

set -e

ENVIRONMENT=${1:-dev}
VERSION=${2:-latest}

echo "Deploying TSIOT $VERSION to $ENVIRONMENT environment"

# Create necessary directories
mkdir -p data logs backups

# Generate secrets if they don't exist
if [ ! -f secrets/postgres_password.txt ]; then
    mkdir -p secrets
    openssl rand -hex 32 > secrets/postgres_password.txt
fi

# Set environment-specific configuration
if [ "$ENVIRONMENT" = "prod" ]; then
    COMPOSE_FILE="docker-compose.prod.yml"
    ENV_FILE=".env.prod"
else
    COMPOSE_FILE="docker-compose.dev.yml"
    ENV_FILE=".env.dev"
fi

# Pull latest images
docker-compose -f $COMPOSE_FILE pull

# Deploy with zero downtime
docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up -d

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 30

# Verify deployment
docker-compose -f $COMPOSE_FILE ps
curl -f http://localhost:8080/health

echo "Deployment complete!"
```

### Backup Script

```bash
#!/bin/bash
# backup.sh

set -e

BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

echo "Creating backup at $TIMESTAMP"

# Database backup
docker-compose exec -T postgres pg_dump -U tsiot tsiot | gzip > "$BACKUP_DIR/postgres_$TIMESTAMP.sql.gz"

# Redis backup
docker-compose exec -T redis redis-cli SAVE
docker cp $(docker-compose ps -q redis):/data/dump.rdb "$BACKUP_DIR/redis_$TIMESTAMP.rdb"

# Application data backup
tar -czf "$BACKUP_DIR/data_$TIMESTAMP.tar.gz" data/

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -type f -mtime +7 -delete

echo "Backup complete: $BACKUP_DIR/*_$TIMESTAMP.*"
```

## Monitoring and Logging

### Docker Compose Override for Monitoring

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  fluentd:
    image: fluent/fluentd:v1.16
    volumes:
      - ./fluentd.conf:/fluentd/etc/fluent.conf
      - tsiot_logs:/var/log/tsiot
    ports:
      - "24224:24224"
    networks:
      - tsiot-network

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    networks:
      - tsiot-network

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
    networks:
      - tsiot-network

volumes:
  elasticsearch_data:
```

### Log Configuration

```xml
<!-- fluentd.conf -->
<source>
  @type forward
  port 24224
</source>

<filter tsiot.**>
  @type parser
  key_name log
  <parse>
    @type json
  </parse>
</filter>

<match tsiot.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name tsiot-logs
  <buffer>
    flush_interval 10s
  </buffer>
</match>
```

## Security Hardening

### Security-focused Compose

```yaml
# docker-compose.secure.yml
version: '3.8'

services:
  tsiot-api:
    image: tsiot/tsiot:latest
    read_only: true
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    user: "1000:1000"
    tmpfs:
      - /tmp:noexec,nosuid,size=1g
    volumes:
      - tsiot_data:/app/data:rw
      - tsiot_logs:/app/logs:rw
    networks:
      - tsiot-internal

  postgres:
    image: postgres:15
    read_only: true
    security_opt:
      - no-new-privileges:true
    user: "999:999"
    tmpfs:
      - /tmp:noexec,nosuid,size=512m
      - /var/run/postgresql:noexec,nosuid,size=128m
    volumes:
      - postgres_data:/var/lib/postgresql/data:rw
    networks:
      - tsiot-internal

networks:
  tsiot-internal:
    driver: bridge
    internal: true
```

## Troubleshooting

### Common Issues

**Container Won't Start**
```bash
# Check logs
docker-compose logs tsiot-api

# Check resource usage
docker stats

# Verify configuration
docker-compose config
```

**Database Connection Issues**
```bash
# Test database connectivity
docker-compose exec tsiot-api nc -zv postgres 5432

# Check database logs
docker-compose logs postgres

# Connect to database manually
docker-compose exec postgres psql -U tsiot -d tsiot
```

**Performance Issues**
```bash
# Monitor resource usage
docker stats --no-stream

# Check container limits
docker inspect tsiot_tsiot-api_1 | grep -A 10 "Memory"

# Profile application
docker-compose exec tsiot-api curl localhost:6060/debug/pprof/profile?seconds=30 > profile.out
```

### Health Checks

```bash
# Check all services
docker-compose ps

# Test API health
curl http://localhost:8080/health

# Check metrics
curl http://localhost:8080/metrics

# Database health
docker-compose exec postgres pg_isready -U tsiot
```

For more troubleshooting information, see the [Troubleshooting Guide](./troubleshooting.md).