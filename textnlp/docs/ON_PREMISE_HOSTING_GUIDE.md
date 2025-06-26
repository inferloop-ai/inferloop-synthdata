# TextNLP On-Premise Hosting Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture Overview](#architecture-overview)
3. [Deployment Scenarios](#deployment-scenarios)
4. [System Requirements](#system-requirements)
5. [Installation Guide](#installation-guide)
6. [Configuration](#configuration)
7. [Security Considerations](#security-considerations)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Overview

This guide provides comprehensive instructions for deploying TextNLP on-premise, supporting both air-gapped (offline) and internet-connected environments. The deployment can scale from single localhost installations to multi-node clusters.

### Key Features
- **Air-Gapped Support**: Full offline operation capability
- **Scalable Architecture**: From single node to multi-node clusters
- **Security-First Design**: Enterprise-grade security controls
- **High Availability**: Built-in redundancy and failover
- **Resource Optimization**: Efficient use of on-premise hardware

## Architecture Overview

### Component Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (HAProxy/Nginx)            │
└─────────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┴──────────────────────┐
        │                                             │
┌───────▼─────────┐                          ┌───────▼─────────┐
│   API Gateway   │                          │   API Gateway   │
│   (Primary)     │                          │   (Secondary)   │
└───────┬─────────┘                          └───────┬─────────┘
        │                                             │
┌───────▼─────────────────────────────────────────────▼─────────┐
│                      Application Layer                         │
├────────────────┬───────────────┬──────────────┬──────────────┤
│ Text Generation│ Validation    │ Safety       │ Metrics      │
│ Service        │ Service       │ Service      │ Service      │
└────────────────┴───────────────┴──────────────┴──────────────┘
                               │
┌───────────────────────────────▼────────────────────────────────┐
│                         Data Layer                              │
├─────────────┬──────────────┬─────────────┬────────────────────┤
│ PostgreSQL  │ Redis Cache  │ Model Store │ Metrics DB         │
│ (Primary)   │ (Cluster)    │ (NFS/S3)   │ (TimescaleDB)      │
└─────────────┴──────────────┴─────────────┴────────────────────┘
```

### Network Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                   External Network (Optional)                │
└─────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Firewall/DMZ      │
                    └──────────┬──────────┘
                               │
┌─────────────────────────────▼─────────────────────────────┐
│                   Application Network                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
│  │ App Node│  │ App Node│  │ App Node│  │ App Node│     │
│  │    1    │  │    2    │  │    3    │  │    N    │     │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │
└────────────────────────────┬──────────────────────────────┘
                             │
┌────────────────────────────▼──────────────────────────────┐
│                    Backend Network                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
│  │Database │  │  Redis  │  │  Model  │  │ Metrics │     │
│  │ Cluster │  │ Cluster │  │ Storage │  │   DB    │     │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │
└────────────────────────────────────────────────────────────┘
```

## Deployment Scenarios

### 1. Localhost Development
Single machine deployment for development and testing.

**Characteristics:**
- All services on one machine
- Docker Compose orchestration
- Minimal resource requirements
- Easy setup and teardown

### 2. Single Server Production
Production deployment on a single powerful server.

**Characteristics:**
- All services on one machine with resource isolation
- Systemd service management
- Local storage for models
- Suitable for small to medium workloads

### 3. Small Cluster (3-5 nodes)
High availability deployment across multiple servers.

**Characteristics:**
- Service distribution across nodes
- Database replication
- Shared storage for models
- Load balancing and failover

### 4. Large Cluster (10+ nodes)
Enterprise-scale deployment with full redundancy.

**Characteristics:**
- Dedicated nodes for each service type
- Multi-master database setup
- Distributed storage system
- Auto-scaling capabilities

### 5. Air-Gapped Deployment
Completely offline deployment with no internet access.

**Characteristics:**
- Pre-downloaded models and dependencies
- Internal package repository
- Offline documentation
- Manual update process

## System Requirements

### Minimum Requirements (Development)
```yaml
Hardware:
  CPU: 4 cores (x86_64 or ARM64)
  RAM: 8 GB
  Storage: 50 GB SSD
  Network: 1 Gbps (internal)

Software:
  OS: Ubuntu 20.04+ / RHEL 8+ / Rocky Linux 8+
  Docker: 20.10+
  Docker Compose: 2.0+
  Python: 3.8+
```

### Recommended Requirements (Production)
```yaml
Hardware:
  CPU: 16+ cores (x86_64)
  RAM: 32+ GB
  Storage: 500+ GB NVMe SSD
  Network: 10 Gbps (internal)
  GPU: Optional (NVIDIA Tesla T4+ for acceleration)

Software:
  OS: Ubuntu 22.04 LTS / RHEL 9
  Container Runtime: Docker 24+ or Podman 4+
  Orchestration: Kubernetes 1.28+ (optional)
  Database: PostgreSQL 15+
  Cache: Redis 7+
```

### Cluster Requirements
```yaml
Per Node:
  CPU: 8+ cores
  RAM: 16+ GB
  Storage: 200+ GB SSD
  Network: 10 Gbps interconnect

Shared Storage:
  Type: NFS v4 / GlusterFS / Ceph
  Capacity: 2+ TB
  Performance: 1000+ IOPS

Network:
  Internal: 10 Gbps minimum
  External: 1 Gbps minimum (if connected)
  Latency: <1ms between nodes
```

## Installation Guide

### Prerequisites Setup

#### 1. System Preparation
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y  # Ubuntu/Debian
# OR
sudo yum update -y  # RHEL/CentOS

# Install required packages
sudo apt install -y \
    curl wget git vim \
    build-essential python3-dev \
    postgresql-client redis-tools \
    nfs-common cifs-utils \
    htop iotop sysstat \
    ufw fail2ban

# Configure firewall (Ubuntu)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8000/tcp  # API
sudo ufw allow 5432/tcp  # PostgreSQL (internal only)
sudo ufw allow 6379/tcp  # Redis (internal only)
sudo ufw enable

# Set up system limits
cat << EOF | sudo tee /etc/security/limits.d/textnlp.conf
* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768
EOF

# Configure sysctl for performance
cat << EOF | sudo tee /etc/sysctl.d/99-textnlp.conf
# Network optimizations
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.ip_local_port_range = 1024 65535
net.ipv4.tcp_tw_reuse = 1

# Memory optimizations
vm.overcommit_memory = 1
vm.swappiness = 10
EOF

sudo sysctl -p /etc/sysctl.d/99-textnlp.conf
```

#### 2. Docker Installation
```bash
# Install Docker (Ubuntu)
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Configure Docker daemon
sudo mkdir -p /etc/docker
cat << EOF | sudo tee /etc/docker/daemon.json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "10"
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  }
}
EOF

sudo systemctl restart docker
```

#### 3. For Air-Gapped Environments
```bash
# On internet-connected machine, download all dependencies
mkdir -p /tmp/textnlp-offline
cd /tmp/textnlp-offline

# Download Docker images
docker pull python:3.9-slim
docker pull postgres:15
docker pull redis:7-alpine
docker pull nginx:alpine
docker pull grafana/grafana:latest
docker pull prom/prometheus:latest

# Save images
docker save -o textnlp-images.tar \
    python:3.9-slim \
    postgres:15 \
    redis:7-alpine \
    nginx:alpine \
    grafana/grafana:latest \
    prom/prometheus:latest

# Download Python packages
pip download -d ./python-packages \
    -r https://raw.githubusercontent.com/inferloop/textnlp/main/requirements.txt

# Download models (example for GPT-2)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model.save_pretrained('./models/gpt2')
tokenizer.save_pretrained('./models/gpt2')
"

# Create offline bundle
tar -czf textnlp-offline-bundle.tar.gz *

# Transfer to air-gapped system and extract
# On air-gapped system:
tar -xzf textnlp-offline-bundle.tar.gz
docker load -i textnlp-images.tar
```

### Localhost Deployment

#### 1. Clone Repository
```bash
# For connected environments
git clone https://github.com/inferloop/inferloop-synthdata.git
cd inferloop-synthdata/textnlp

# For air-gapped environments
# Copy the repository manually
```

#### 2. Configure Environment
```bash
# Create environment file
cat << EOF > .env
# Application settings
ENVIRONMENT=development
DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000

# Database settings
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=textnlp
POSTGRES_USER=textnlp
POSTGRES_PASSWORD=$(openssl rand -base64 32)

# Redis settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=$(openssl rand -base64 32)

# Model settings
MODEL_PATH=/app/models
DEFAULT_MODEL=gpt2
MODEL_CACHE_SIZE=4

# Security settings
JWT_SECRET=$(openssl rand -base64 32)
API_KEY=$(openssl rand -base64 32)

# Resource limits
MAX_WORKERS=4
MAX_MEMORY=4G
MAX_REQUESTS_PER_MINUTE=100
EOF

# Secure the environment file
chmod 600 .env
```

#### 3. Docker Compose Setup
```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    container_name: textnlp-postgres
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "127.0.0.1:5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: textnlp-redis
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "127.0.0.1:6379:6379"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: textnlp-api
    environment:
      - ENVIRONMENT=${ENVIRONMENT}
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - MODEL_PATH=${MODEL_PATH}
      - JWT_SECRET=${JWT_SECRET}
      - API_KEY=${API_KEY}
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    ports:
      - "${API_HOST}:${API_PORT}:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '${MAX_WORKERS}'
          memory: ${MAX_MEMORY}

  nginx:
    image: nginx:alpine
    container_name: textnlp-nginx
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - api
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

#### 4. Start Services
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Run database migrations
docker-compose exec api python manage.py migrate

# Create admin user
docker-compose exec api python manage.py createsuperuser
```

### Production Cluster Deployment

#### 1. Cluster Initialization

##### Master Node Setup
```bash
# Install Kubernetes (using kubeadm)
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list

sudo apt update
sudo apt install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

# Initialize cluster
sudo kubeadm init --pod-network-cidr=10.244.0.0/16

# Configure kubectl
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Install network plugin (Flannel)
kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
```

##### Worker Node Setup
```bash
# Join cluster (use token from master node)
sudo kubeadm join <master-ip>:6443 --token <token> --discovery-token-ca-cert-hash <hash>
```

#### 2. Storage Setup

##### NFS Server Configuration
```bash
# On storage node
sudo apt install -y nfs-kernel-server

# Create shared directories
sudo mkdir -p /srv/nfs/textnlp/{models,data,logs}
sudo chown -R nobody:nogroup /srv/nfs/textnlp

# Configure exports
cat << EOF | sudo tee /etc/exports
/srv/nfs/textnlp/models *(rw,sync,no_subtree_check,no_root_squash)
/srv/nfs/textnlp/data *(rw,sync,no_subtree_check,no_root_squash)
/srv/nfs/textnlp/logs *(rw,sync,no_subtree_check,no_root_squash)
EOF

sudo exportfs -a
sudo systemctl restart nfs-kernel-server
```

##### Persistent Volume Configuration
```yaml
# pv-models.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: textnlp-models-pv
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadOnlyMany
  nfs:
    server: <nfs-server-ip>
    path: /srv/nfs/textnlp/models
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: textnlp-models-pvc
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 100Gi
```

#### 3. Kubernetes Deployment

##### Namespace and ConfigMap
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: textnlp
---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: textnlp-config
  namespace: textnlp
data:
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  ENVIRONMENT: "production"
  MODEL_PATH: "/app/models"
  DEFAULT_MODEL: "gpt2"
  MAX_WORKERS: "8"
  LOG_LEVEL: "INFO"
```

##### Database Deployment
```yaml
# postgres-deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: textnlp
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: textnlp
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 50Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: textnlp
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

##### Redis Deployment
```yaml
# redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: textnlp
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command: ["redis-server", "--requirepass", "$(REDIS_PASSWORD)"]
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: password
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "1Gi"
            cpu: "0.5"
          limits:
            memory: "2Gi"
            cpu: "1"
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: textnlp
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
```

##### Application Deployment
```yaml
# app-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: textnlp-api
  namespace: textnlp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: textnlp-api
  template:
    metadata:
      labels:
        app: textnlp-api
    spec:
      initContainers:
      - name: wait-for-db
        image: busybox:1.35
        command: ['sh', '-c', 'until nc -z postgres 5432; do echo waiting for postgres; sleep 2; done']
      containers:
      - name: api
        image: textnlp:latest
        imagePullPolicy: Always
        env:
        - name: DATABASE_URL
          value: "postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@postgres:5432/textnlp"
        - name: REDIS_URL
          value: "redis://:$(REDIS_PASSWORD)@redis:6379/0"
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: username
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: password
        envFrom:
        - configMapRef:
            name: textnlp-config
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
        - name: logs
          mountPath: /app/logs
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: textnlp-models-pvc
      - name: logs
        persistentVolumeClaim:
          claimName: textnlp-logs-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: textnlp-api
  namespace: textnlp
spec:
  selector:
    app: textnlp-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

##### Ingress Configuration
```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: textnlp-ingress
  namespace: textnlp
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - textnlp.example.com
    secretName: textnlp-tls
  rules:
  - host: textnlp.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: textnlp-api
            port:
              number: 80
```

##### Horizontal Pod Autoscaler
```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: textnlp-api-hpa
  namespace: textnlp
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: textnlp-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### 4. Deploy to Kubernetes
```bash
# Create secrets
kubectl create secret generic postgres-secret \
  --from-literal=username=textnlp \
  --from-literal=password=$(openssl rand -base64 32) \
  -n textnlp

kubectl create secret generic redis-secret \
  --from-literal=password=$(openssl rand -base64 32) \
  -n textnlp

# Apply configurations
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f pv-models.yaml
kubectl apply -f postgres-deployment.yaml
kubectl apply -f redis-deployment.yaml
kubectl apply -f app-deployment.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# Check deployment status
kubectl get all -n textnlp
```

## Configuration

### Application Configuration

#### Core Settings
```yaml
# config/production.yaml
application:
  name: "TextNLP"
  environment: "production"
  debug: false
  timezone: "UTC"
  
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  max_request_size: "100MB"
  timeout: 300
  cors:
    enabled: true
    origins: ["https://example.com"]
    
security:
  jwt_algorithm: "HS256"
  jwt_expiry: 3600
  api_key_header: "X-API-Key"
  rate_limit:
    enabled: true
    requests_per_minute: 100
    burst: 20
    
database:
  engine: "postgresql"
  pool_size: 20
  max_overflow: 40
  pool_timeout: 30
  pool_recycle: 3600
  
cache:
  backend: "redis"
  ttl: 3600
  max_entries: 10000
  
models:
  cache_size: 4
  default_model: "gpt2"
  available_models:
    - "gpt2"
    - "gpt2-medium"
    - "distilgpt2"
  model_timeout: 60
  
monitoring:
  metrics_enabled: true
  logging_level: "INFO"
  sentry_dsn: ""
  prometheus_port: 9090
```

#### Model-Specific Configuration
```yaml
# config/models.yaml
gpt2:
  max_length: 1024
  temperature: 1.0
  top_p: 0.9
  top_k: 50
  batch_size: 8
  optimization:
    quantization: "int8"
    use_cache: true
    compile: true
    
gpt2-medium:
  max_length: 1024
  temperature: 1.0
  top_p: 0.9
  top_k: 50
  batch_size: 4
  optimization:
    quantization: "int8"
    use_cache: true
    compile: false
```

### Infrastructure Configuration

#### Load Balancer (HAProxy)
```
# /etc/haproxy/haproxy.cfg
global
    maxconn 4096
    log /dev/log local0
    log /dev/log local1 notice
    chroot /var/lib/haproxy
    stats socket /run/haproxy/admin.sock mode 660 level admin
    stats timeout 30s
    user haproxy
    group haproxy
    daemon

defaults
    log     global
    mode    http
    option  httplog
    option  dontlognull
    timeout connect 5000
    timeout client  50000
    timeout server  50000
    errorfile 400 /etc/haproxy/errors/400.http
    errorfile 403 /etc/haproxy/errors/403.http
    errorfile 408 /etc/haproxy/errors/408.http
    errorfile 500 /etc/haproxy/errors/500.http
    errorfile 502 /etc/haproxy/errors/502.http
    errorfile 503 /etc/haproxy/errors/503.http
    errorfile 504 /etc/haproxy/errors/504.http

frontend textnlp_frontend
    bind *:80
    bind *:443 ssl crt /etc/haproxy/certs/textnlp.pem
    redirect scheme https if !{ ssl_fc }
    mode http
    default_backend textnlp_backend

backend textnlp_backend
    mode http
    balance roundrobin
    option httpchk GET /health
    server api1 10.0.1.10:8000 check
    server api2 10.0.1.11:8000 check
    server api3 10.0.1.12:8000 check
```

#### Monitoring Stack (Prometheus)
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'textnlp-api'
    static_configs:
      - targets: ['api1:9090', 'api2:9090', 'api3:9090']
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:9187']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node1:9100', 'node2:9100', 'node3:9100']
```

## Security Considerations

### Network Security

#### Firewall Rules
```bash
# External facing (Load Balancer)
sudo ufw allow 80/tcp   # HTTP (redirects to HTTPS)
sudo ufw allow 443/tcp  # HTTPS

# Internal communication (Application Network)
sudo ufw allow from 10.0.1.0/24 to any port 8000  # API
sudo ufw allow from 10.0.1.0/24 to any port 9090  # Metrics

# Backend communication (Backend Network)
sudo ufw allow from 10.0.2.0/24 to any port 5432  # PostgreSQL
sudo ufw allow from 10.0.2.0/24 to any port 6379  # Redis
sudo ufw allow from 10.0.2.0/24 to any port 2049  # NFS
```

#### SSL/TLS Configuration
```bash
# Generate self-signed certificate (for testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/textnlp.key \
    -out /etc/ssl/certs/textnlp.crt

# For production, use Let's Encrypt
sudo apt install certbot
sudo certbot certonly --standalone -d textnlp.example.com
```

### Application Security

#### Authentication & Authorization
```python
# config/security.py
SECURITY_CONFIG = {
    "authentication": {
        "type": "jwt",
        "algorithm": "RS256",
        "public_key_path": "/etc/textnlp/keys/public.pem",
        "private_key_path": "/etc/textnlp/keys/private.pem",
        "expiry": 3600,
        "refresh_expiry": 86400
    },
    "authorization": {
        "rbac_enabled": True,
        "default_role": "user",
        "roles": {
            "admin": ["*"],
            "developer": ["read", "write", "generate"],
            "user": ["read", "generate"]
        }
    },
    "api_keys": {
        "enabled": True,
        "header": "X-API-Key",
        "rotation_days": 90
    }
}
```

#### Data Encryption
```yaml
# Encryption at rest
database:
  encryption:
    enabled: true
    algorithm: "AES-256-GCM"
    key_rotation_days: 30
    
storage:
  encryption:
    enabled: true
    type: "server-side"
    kms_key_id: "arn:aws:kms:region:account:key/id"
    
# Encryption in transit
tls:
  min_version: "1.2"
  ciphers:
    - "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"
    - "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
```

### Compliance & Auditing

#### Audit Logging
```python
# config/audit.py
AUDIT_CONFIG = {
    "enabled": True,
    "log_level": "INFO",
    "events": [
        "authentication",
        "authorization",
        "data_access",
        "configuration_change",
        "model_inference"
    ],
    "storage": {
        "type": "database",
        "retention_days": 365,
        "encryption": True
    },
    "export": {
        "format": "json",
        "destination": "s3://audit-logs/textnlp/"
    }
}
```

#### GDPR Compliance
```yaml
privacy:
  data_retention:
    user_data: 365  # days
    inference_logs: 30
    metrics: 90
    
  anonymization:
    enabled: true
    fields: ["ip_address", "user_agent", "email"]
    
  right_to_erasure:
    enabled: true
    api_endpoint: "/api/v1/privacy/delete"
    
  data_portability:
    enabled: true
    api_endpoint: "/api/v1/privacy/export"
    formats: ["json", "csv"]
```

## Monitoring and Maintenance

### Health Checks

#### Application Health
```python
# health/checks.py
from typing import Dict, List
import asyncio
import aiohttp
import asyncpg
import redis.asyncio as redis

class HealthChecker:
    def __init__(self, config):
        self.config = config
        self.checks = {
            "database": self.check_database,
            "redis": self.check_redis,
            "models": self.check_models,
            "storage": self.check_storage,
            "api": self.check_api
        }
    
    async def check_all(self) -> Dict[str, Dict]:
        results = {}
        tasks = []
        
        for name, check_func in self.checks.items():
            task = asyncio.create_task(self.run_check(name, check_func))
            tasks.append(task)
        
        completed = await asyncio.gather(*tasks)
        
        for name, result in completed:
            results[name] = result
        
        return results
    
    async def run_check(self, name: str, check_func) -> tuple:
        try:
            result = await check_func()
            return name, {"status": "healthy", **result}
        except Exception as e:
            return name, {"status": "unhealthy", "error": str(e)}
    
    async def check_database(self) -> Dict:
        conn = await asyncpg.connect(self.config.database_url)
        try:
            version = await conn.fetchval("SELECT version()")
            return {"version": version}
        finally:
            await conn.close()
    
    async def check_redis(self) -> Dict:
        r = redis.from_url(self.config.redis_url)
        try:
            info = await r.info()
            return {
                "version": info["redis_version"],
                "used_memory": info["used_memory_human"]
            }
        finally:
            await r.close()
    
    async def check_models(self) -> Dict:
        # Check if models are loaded and accessible
        loaded_models = []
        for model_name in self.config.available_models:
            if await self.is_model_loaded(model_name):
                loaded_models.append(model_name)
        
        return {
            "loaded": loaded_models,
            "total": len(self.config.available_models)
        }
    
    async def check_storage(self) -> Dict:
        import shutil
        storage_path = self.config.model_path
        
        total, used, free = shutil.disk_usage(storage_path)
        
        return {
            "total_gb": total // (2**30),
            "used_gb": used // (2**30),
            "free_gb": free // (2**30),
            "usage_percent": (used / total) * 100
        }
    
    async def check_api(self) -> Dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://localhost:{self.config.api_port}/health") as resp:
                return {
                    "status_code": resp.status,
                    "response_time_ms": resp.headers.get("X-Response-Time", "N/A")
                }
```

### Monitoring Dashboard

#### Grafana Configuration
```json
{
  "dashboard": {
    "title": "TextNLP Production Monitoring",
    "panels": [
      {
        "title": "API Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Model Inference Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, model_inference_duration_seconds_bucket)",
            "legendFormat": "{{model}}"
          }
        ]
      },
      {
        "title": "System Resources",
        "targets": [
          {
            "expr": "node_cpu_usage",
            "legendFormat": "CPU {{instance}}"
          },
          {
            "expr": "node_memory_usage",
            "legendFormat": "Memory {{instance}}"
          }
        ]
      }
    ]
  }
}
```

### Backup and Recovery

#### Backup Strategy
```bash
#!/bin/bash
# backup.sh

# Configuration
BACKUP_DIR="/backup/textnlp"
RETENTION_DAYS=30
S3_BUCKET="s3://backups/textnlp"

# Create backup directory
mkdir -p $BACKUP_DIR/$(date +%Y%m%d)

# Backup database
echo "Backing up database..."
PGPASSWORD=$POSTGRES_PASSWORD pg_dump \
    -h $POSTGRES_HOST \
    -U $POSTGRES_USER \
    -d $POSTGRES_DB \
    -f $BACKUP_DIR/$(date +%Y%m%d)/database.sql

# Backup Redis
echo "Backing up Redis..."
redis-cli -h $REDIS_HOST -a $REDIS_PASSWORD --rdb $BACKUP_DIR/$(date +%Y%m%d)/redis.rdb

# Backup configuration
echo "Backing up configuration..."
tar -czf $BACKUP_DIR/$(date +%Y%m%d)/config.tar.gz /etc/textnlp

# Backup models (incremental)
echo "Backing up models..."
rsync -av --delete /app/models/ $BACKUP_DIR/$(date +%Y%m%d)/models/

# Upload to S3 (if configured)
if [ ! -z "$S3_BUCKET" ]; then
    echo "Uploading to S3..."
    aws s3 sync $BACKUP_DIR/$(date +%Y%m%d)/ $S3_BUCKET/$(date +%Y%m%d)/
fi

# Clean old backups
find $BACKUP_DIR -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \;

echo "Backup completed successfully"
```

#### Recovery Procedure
```bash
#!/bin/bash
# restore.sh

# Configuration
BACKUP_DATE=$1
BACKUP_DIR="/backup/textnlp/$BACKUP_DATE"

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: ./restore.sh YYYYMMDD"
    exit 1
fi

if [ ! -d "$BACKUP_DIR" ]; then
    echo "Backup not found: $BACKUP_DIR"
    exit 1
fi

# Stop services
echo "Stopping services..."
docker-compose down

# Restore database
echo "Restoring database..."
PGPASSWORD=$POSTGRES_PASSWORD psql \
    -h $POSTGRES_HOST \
    -U $POSTGRES_USER \
    -d postgres \
    -c "DROP DATABASE IF EXISTS $POSTGRES_DB; CREATE DATABASE $POSTGRES_DB;"

PGPASSWORD=$POSTGRES_PASSWORD psql \
    -h $POSTGRES_HOST \
    -U $POSTGRES_USER \
    -d $POSTGRES_DB \
    -f $BACKUP_DIR/database.sql

# Restore Redis
echo "Restoring Redis..."
redis-cli -h $REDIS_HOST -a $REDIS_PASSWORD FLUSHALL
redis-cli -h $REDIS_HOST -a $REDIS_PASSWORD --rdb $BACKUP_DIR/redis.rdb

# Restore configuration
echo "Restoring configuration..."
tar -xzf $BACKUP_DIR/config.tar.gz -C /

# Restore models
echo "Restoring models..."
rsync -av --delete $BACKUP_DIR/models/ /app/models/

# Start services
echo "Starting services..."
docker-compose up -d

echo "Restore completed successfully"
```

### Maintenance Tasks

#### Regular Maintenance Schedule
```yaml
daily:
  - name: "Database vacuum"
    command: "vacuumdb -z -d textnlp"
    time: "02:00"
    
  - name: "Log rotation"
    command: "logrotate /etc/logrotate.d/textnlp"
    time: "03:00"
    
  - name: "Cache cleanup"
    command: "redis-cli -a $REDIS_PASSWORD BGREWRITEAOF"
    time: "04:00"

weekly:
  - name: "Security updates"
    command: "apt update && apt upgrade -y"
    time: "Sunday 01:00"
    
  - name: "Model optimization"
    command: "python /app/scripts/optimize_models.py"
    time: "Sunday 05:00"

monthly:
  - name: "Full backup"
    command: "/backup/scripts/full_backup.sh"
    time: "1st Sunday 00:00"
    
  - name: "Performance analysis"
    command: "python /app/scripts/analyze_performance.py"
    time: "1st Monday 09:00"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. High Memory Usage
```bash
# Check memory usage by process
ps aux --sort=-%mem | head -20

# Check Python memory usage
python -c "import psutil; print(psutil.Process().memory_info())"

# Solutions:
# 1. Reduce model cache size
# 2. Enable model quantization
# 3. Implement memory limits in Docker/Kubernetes
```

#### 2. Slow Response Times
```bash
# Check database query performance
psql -U textnlp -d textnlp -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"

# Check Redis latency
redis-cli -a $REDIS_PASSWORD --latency

# Solutions:
# 1. Add database indexes
# 2. Increase Redis memory
# 3. Enable query caching
# 4. Scale horizontally
```

#### 3. Model Loading Failures
```python
# Debug model loading
import logging
logging.basicConfig(level=logging.DEBUG)

from transformers import AutoModelForCausalLM

try:
    model = AutoModelForCausalLM.from_pretrained("gpt2")
except Exception as e:
    print(f"Model loading failed: {e}")
    # Check disk space
    # Verify model files
    # Check permissions
```

#### 4. Network Connectivity Issues
```bash
# Test internal connectivity
ping -c 4 postgres
ping -c 4 redis

# Test external connectivity (if not air-gapped)
curl -I https://huggingface.co

# Check firewall rules
sudo ufw status verbose

# Check network interfaces
ip addr show
```

### Debug Mode

#### Enable Debug Logging
```python
# config/logging.py
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "/app/logs/debug.log",
            "maxBytes": 104857600,  # 100MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": "DEBUG"
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["file"]
    }
}
```

#### Performance Profiling
```python
# profile_api.py
import cProfile
import pstats
from io import StringIO

def profile_endpoint(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        
        result = func(*args, **kwargs)
        
        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        with open('/app/logs/profile.log', 'a') as f:
            f.write(s.getvalue())
        
        return result
    return wrapper
```

## Best Practices

### Deployment Best Practices

1. **Use Infrastructure as Code**
   - Version control all configurations
   - Use Terraform/Ansible for provisioning
   - Automate deployment processes

2. **Implement Blue-Green Deployments**
   - Maintain two identical environments
   - Switch traffic with zero downtime
   - Easy rollback capability

3. **Monitor Everything**
   - Application metrics
   - System metrics
   - Business metrics
   - User experience metrics

4. **Security First**
   - Regular security audits
   - Automated vulnerability scanning
   - Principle of least privilege
   - Defense in depth

5. **Plan for Failure**
   - Regular disaster recovery drills
   - Automated failover testing
   - Comprehensive backup strategy
   - Clear incident response procedures

### Performance Optimization

1. **Model Optimization**
   ```python
   # Use quantized models for CPU inference
   from transformers import AutoModelForCausalLM
   import torch
   
   model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)
   model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
   ```

2. **Caching Strategy**
   ```python
   # Implement multi-level caching
   from functools import lru_cache
   import redis
   
   redis_client = redis.Redis(host='localhost', port=6379, db=0)
   
   @lru_cache(maxsize=1000)
   def get_from_memory_cache(key):
       return redis_client.get(key)
   ```

3. **Connection Pooling**
   ```python
   # Database connection pooling
   from sqlalchemy import create_engine
   
   engine = create_engine(
       DATABASE_URL,
       pool_size=20,
       max_overflow=40,
       pool_pre_ping=True,
       pool_recycle=3600
   )
   ```

### Scaling Guidelines

#### Vertical Scaling
- Start with vertical scaling for simplicity
- Monitor resource utilization
- Identify bottlenecks (CPU, memory, I/O)
- Upgrade hardware incrementally

#### Horizontal Scaling
- Implement when vertical scaling limits reached
- Use load balancers for distribution
- Ensure stateless application design
- Implement distributed caching

#### Auto-Scaling Rules
```yaml
scaling_rules:
  scale_up:
    cpu_threshold: 80
    memory_threshold: 85
    response_time_threshold: 2000  # ms
    evaluation_periods: 2
    scale_increment: 2  # nodes
    
  scale_down:
    cpu_threshold: 30
    memory_threshold: 40
    response_time_threshold: 500  # ms
    evaluation_periods: 5
    scale_decrement: 1  # node
    min_nodes: 3
```

## Conclusion

This comprehensive guide provides everything needed to deploy TextNLP on-premise, from single-node development environments to large-scale production clusters. The architecture supports both air-gapped and internet-connected deployments with enterprise-grade security, monitoring, and maintenance capabilities.

For additional support or custom deployment scenarios, consult the TextNLP team or refer to the source code documentation.