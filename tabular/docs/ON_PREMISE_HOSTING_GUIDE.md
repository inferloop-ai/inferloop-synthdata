# Tabular Data On-Premise Hosting Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture Design](#architecture-design)
3. [Deployment Scenarios](#deployment-scenarios)
4. [System Requirements](#system-requirements)
5. [Installation and Setup](#installation-and-setup)
6. [Configuration Management](#configuration-management)
7. [Security Implementation](#security-implementation)
8. [Performance Optimization](#performance-optimization)
9. [High Availability Setup](#high-availability-setup)
10. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Overview

The Tabular synthetic data generation system is designed for enterprise deployment with support for:
- High-volume data generation
- Multiple data formats (CSV, Parquet, JSON, SQL)
- Statistical validation and quality assurance
- Privacy-preserving techniques
- Distributed processing capabilities

### Key Components
- **Data Generation Engine**: Core synthetic data generation with multiple algorithms
- **Validation Framework**: Statistical tests and data quality checks
- **Privacy Engine**: Differential privacy and anonymization
- **Processing Pipeline**: Distributed data processing with Spark/Dask
- **API Layer**: RESTful API and SDK interfaces
- **Storage Layer**: Multi-format data storage and retrieval

## Architecture Design

### System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   External Access Layer                      │
├────────────────────────────────────────────────────────────┤
│                 Load Balancer (HAProxy/Nginx)               │
└────────────────────────────────────────────────────────────┘
                              │
     ┌────────────────────────┴────────────────────────┐
     │                                                 │
┌────▼──────────┐                            ┌────────▼────────┐
│  API Gateway  │                            │  API Gateway    │
│  (Primary)    │                            │  (Secondary)    │
└────┬──────────┘                            └────────┬────────┘
     │                                                 │
┌────▼─────────────────────────────────────────────────▼────────┐
│                      Application Services                      │
├──────────────┬──────────────┬──────────────┬─────────────────┤
│ Generation   │ Validation   │ Privacy      │ Processing      │
│ Service      │ Service      │ Service      │ Service         │
└──────────────┴──────────────┴──────────────┴─────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│                      Data Processing Layer                     │
├─────────────────┬─────────────────┬───────────────────────────┤
│ Apache Spark    │ Dask Cluster    │ Ray Cluster             │
│ (Batch)         │ (Parallel)      │ (ML Workloads)          │
└─────────────────┴─────────────────┴───────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│                        Storage Layer                           │
├──────────┬──────────┬──────────┬──────────┬──────────────────┤
│PostgreSQL│ Redis    │ MinIO/S3 │ HDFS     │ Time Series DB   │
│(Metadata)│ (Cache)  │ (Objects)│ (Big Data)│ (Metrics)       │
└──────────┴──────────┴──────────┴──────────┴──────────────────┘
```

### Component Details

#### 1. Generation Service
- Synthetic data generation algorithms
- Schema management and versioning
- Template-based generation
- Real-time and batch generation modes

#### 2. Validation Service
- Statistical distribution validation
- Data quality metrics
- Constraint verification
- Anomaly detection

#### 3. Privacy Service
- Differential privacy implementation
- K-anonymity and l-diversity
- Data masking and encryption
- Audit logging

#### 4. Processing Service
- Distributed data processing
- ETL pipeline management
- Stream processing capabilities
- Job scheduling and monitoring

## Deployment Scenarios

### 1. Single Server Deployment
Suitable for development and small-scale production (< 1TB daily).

```yaml
# Single server with all components
Server Specifications:
  CPU: 16+ cores
  RAM: 64+ GB
  Storage: 2+ TB NVMe SSD
  Network: 10 Gbps

Components:
  - All services in Docker containers
  - Local PostgreSQL and Redis
  - MinIO for object storage
  - Single-node Spark
```

### 2. Small Cluster (3-5 nodes)
For medium-scale production (1-10TB daily).

```yaml
# Distributed across multiple nodes
Master Nodes (2):
  - API Gateway
  - Service orchestration
  - PostgreSQL primary/replica
  
Worker Nodes (3):
  - Data generation services
  - Spark executors
  - Distributed storage
```

### 3. Large Cluster (10+ nodes)
Enterprise-scale deployment (10TB+ daily).

```yaml
# Full distributed architecture
Control Plane (3 nodes):
  - Kubernetes masters
  - Service mesh
  - Monitoring stack

Data Plane (10+ nodes):
  - Dedicated Spark cluster
  - Distributed storage (HDFS/Ceph)
  - GPU nodes for ML workloads
```

### 4. Hybrid Cloud Deployment
On-premise with cloud burst capabilities.

```yaml
On-Premise Core:
  - Sensitive data processing
  - Primary storage
  - Core services

Cloud Extensions:
  - Burst compute capacity
  - Backup and archival
  - Development environments
```

## System Requirements

### Hardware Requirements

#### Minimum Requirements (Development)
```yaml
CPU: 8 cores (x86_64)
RAM: 16 GB
Storage: 
  - System: 100 GB SSD
  - Data: 500 GB SSD
Network: 1 Gbps
GPU: Optional (for ML models)
```

#### Recommended Requirements (Production)
```yaml
CPU: 32+ cores (x86_64, AVX2 support)
RAM: 128+ GB ECC
Storage:
  - System: 500 GB NVMe SSD
  - Data: 10+ TB NVMe SSD array
  - Archive: 100+ TB HDD
Network: 10+ Gbps
GPU: NVIDIA T4/V100 (for ML acceleration)
```

### Software Requirements
```yaml
Operating System:
  - Ubuntu 20.04/22.04 LTS
  - RHEL 8/9
  - Rocky Linux 8/9

Container Runtime:
  - Docker 24.0+
  - Containerd 1.7+
  - Podman 4.0+ (RHEL)

Orchestration:
  - Kubernetes 1.28+
  - Docker Swarm (alternative)

Data Processing:
  - Apache Spark 3.5+
  - Dask 2023.10+
  - Apache Arrow 14.0+

Databases:
  - PostgreSQL 15+
  - Redis 7+
  - TimescaleDB 2.13+

Storage:
  - MinIO RELEASE.2024+
  - HDFS 3.3+ (optional)
  - Ceph Pacific+ (optional)
```

## Installation and Setup

### 1. Base System Preparation

```bash
#!/bin/bash
# prepare_system.sh

# Update system
sudo apt update && sudo apt upgrade -y

# Install base packages
sudo apt install -y \
    build-essential \
    python3-dev \
    python3-pip \
    git \
    curl \
    wget \
    htop \
    iotop \
    sysstat \
    jq

# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Configure system limits
cat << EOF | sudo tee /etc/security/limits.d/tabular.conf
* soft nofile 65536
* hard nofile 65536
* soft nproc 65536
* hard nproc 65536
EOF

# Configure sysctl
cat << EOF | sudo tee /etc/sysctl.d/99-tabular.conf
# Network optimizations
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.ip_local_port_range = 1024 65535

# Memory optimizations
vm.max_map_count = 262144
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
EOF

sudo sysctl -p /etc/sysctl.d/99-tabular.conf
```

### 2. Storage Setup

```bash
#!/bin/bash
# setup_storage.sh

# Create directory structure
sudo mkdir -p /data/tabular/{postgres,redis,minio,spark,hdfs}
sudo mkdir -p /var/log/tabular
sudo mkdir -p /etc/tabular/config

# Set up RAID for data volumes (if multiple disks)
# Example for RAID 10 with 4 disks
sudo mdadm --create /dev/md0 --level=10 --raid-devices=4 /dev/sd[bcde]
sudo mkfs.ext4 /dev/md0
sudo mount /dev/md0 /data

# Configure MinIO storage
sudo mkdir -p /data/minio/{data,config}
sudo chown -R 1000:1000 /data/minio

# Set up NFS for shared storage (if needed)
sudo apt install -y nfs-kernel-server
echo "/data/tabular/shared *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo exportfs -a
```

### 3. Container Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: tabular-postgres
    environment:
      POSTGRES_DB: tabular
      POSTGRES_USER: tabular
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_INITDB_ARGS: "--encoding=UTF8 --data-checksums"
    volumes:
      - /data/tabular/postgres:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    ports:
      - "127.0.0.1:5432:5432"
    restart: unless-stopped
    command: >
      postgres
      -c shared_buffers=2GB
      -c effective_cache_size=6GB
      -c maintenance_work_mem=512MB
      -c checkpoint_completion_target=0.9
      -c wal_buffers=16MB
      -c default_statistics_target=100
      -c random_page_cost=1.1
      -c effective_io_concurrency=200
      -c work_mem=32MB
      -c min_wal_size=1GB
      -c max_wal_size=4GB

  redis:
    image: redis:7-alpine
    container_name: tabular-redis
    command: >
      redis-server
      --requirepass ${REDIS_PASSWORD}
      --maxmemory 4gb
      --maxmemory-policy allkeys-lru
      --save 900 1
      --save 300 10
      --save 60 10000
    volumes:
      - /data/tabular/redis:/data
    ports:
      - "127.0.0.1:6379:6379"
    restart: unless-stopped

  minio:
    image: minio/minio:latest
    container_name: tabular-minio
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    volumes:
      - /data/minio/data:/data
      - /data/minio/config:/root/.minio
    ports:
      - "9000:9000"
      - "9001:9001"
    command: server /data --console-address ":9001"
    restart: unless-stopped

  spark-master:
    image: bitnami/spark:3.5
    container_name: tabular-spark-master
    environment:
      - SPARK_MODE=master
      - SPARK_RPC_AUTHENTICATION_ENABLED=no
      - SPARK_RPC_ENCRYPTION_ENABLED=no
      - SPARK_LOCAL_STORAGE_ENCRYPTION_ENABLED=no
      - SPARK_SSL_ENABLED=no
    ports:
      - "8080:8080"
      - "7077:7077"
    volumes:
      - /data/tabular/spark:/opt/spark/work
    restart: unless-stopped

  spark-worker:
    image: bitnami/spark:3.5
    container_name: tabular-spark-worker
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
      - SPARK_WORKER_MEMORY=8G
      - SPARK_WORKER_CORES=4
    volumes:
      - /data/tabular/spark:/opt/spark/work
    depends_on:
      - spark-master
    restart: unless-stopped
    deploy:
      replicas: 3

  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: tabular-api
    environment:
      - DATABASE_URL=postgresql://tabular:${POSTGRES_PASSWORD}@postgres:5432/tabular
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=${MINIO_ROOT_USER}
      - MINIO_SECRET_KEY=${MINIO_ROOT_PASSWORD}
      - SPARK_MASTER=spark://spark-master:7077
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
      - minio
      - spark-master
    restart: unless-stopped
    deploy:
      replicas: 2

  nginx:
    image: nginx:alpine
    container_name: tabular-nginx
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - api
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: tabular-prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - /data/tabular/prometheus:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: tabular-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
    volumes:
      - /data/tabular/grafana:/var/lib/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    restart: unless-stopped

networks:
  default:
    name: tabular-network
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### 4. Kubernetes Deployment

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: tabular
---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: tabular-config
  namespace: tabular
data:
  app.yaml: |
    server:
      host: 0.0.0.0
      port: 8000
      workers: 4
    
    database:
      host: postgres-service
      port: 5432
      name: tabular
      pool_size: 20
    
    redis:
      host: redis-service
      port: 6379
      db: 0
    
    storage:
      type: minio
      endpoint: minio-service:9000
      bucket: tabular-data
    
    spark:
      master: spark://spark-master:7077
      executor_memory: 4g
      executor_cores: 2
---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tabular-api
  namespace: tabular
spec:
  replicas: 3
  selector:
    matchLabels:
      app: tabular-api
  template:
    metadata:
      labels:
        app: tabular-api
    spec:
      containers:
      - name: api
        image: tabular:latest
        ports:
        - containerPort: 8000
        env:
        - name: CONFIG_PATH
          value: /etc/tabular/app.yaml
        volumeMounts:
        - name: config
          mountPath: /etc/tabular
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
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
      - name: config
        configMap:
          name: tabular-config
---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: tabular-api-service
  namespace: tabular
spec:
  selector:
    app: tabular-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: tabular-ingress
  namespace: tabular
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - tabular.example.com
    secretName: tabular-tls
  rules:
  - host: tabular.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: tabular-api-service
            port:
              number: 80
```

## Configuration Management

### 1. Application Configuration

```yaml
# config/production.yaml
application:
  name: "Tabular Synthetic Data"
  version: "${APP_VERSION}"
  environment: "production"
  debug: false

server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  max_request_size: "1GB"
  timeout: 600
  cors:
    enabled: true
    origins: ["https://app.example.com"]

database:
  host: "${DB_HOST}"
  port: 5432
  name: "tabular"
  user: "${DB_USER}"
  password: "${DB_PASSWORD}"
  pool_size: 20
  max_overflow: 40
  pool_timeout: 30
  pool_recycle: 3600
  echo: false

redis:
  host: "${REDIS_HOST}"
  port: 6379
  password: "${REDIS_PASSWORD}"
  db: 0
  pool_size: 10
  socket_timeout: 5
  socket_connect_timeout: 5

storage:
  type: "s3"  # s3, minio, filesystem
  endpoint: "${S3_ENDPOINT}"
  access_key: "${S3_ACCESS_KEY}"
  secret_key: "${S3_SECRET_KEY}"
  bucket: "tabular-data"
  region: "us-east-1"
  use_ssl: true
  
  # Storage tiers
  hot_storage:
    path: "/data/hot"
    max_size: "1TB"
    ttl: 7  # days
  
  warm_storage:
    path: "/data/warm"
    max_size: "10TB"
    ttl: 30  # days
  
  cold_storage:
    path: "/data/cold"
    max_size: "100TB"
    ttl: 365  # days

generation:
  max_rows_per_request: 10000000
  max_columns_per_table: 1000
  chunk_size: 50000
  parallel_workers: 4
  memory_limit: "8GB"
  
  algorithms:
    - name: "gaussian_copula"
      enabled: true
      params:
        default_distribution: "beta"
    
    - name: "ctgan"
      enabled: true
      params:
        epochs: 300
        batch_size: 500
    
    - name: "tvae"
      enabled: true
      params:
        epochs: 300
        batch_size: 500

validation:
  enabled: true
  statistical_tests:
    - "ks_test"
    - "chi_square"
    - "jensen_shannon"
  
  quality_thresholds:
    distribution_similarity: 0.95
    correlation_difference: 0.1
    privacy_risk: 0.01

privacy:
  differential_privacy:
    enabled: true
    epsilon: 1.0
    delta: 1e-5
  
  anonymization:
    k_anonymity: 5
    l_diversity: 3
    t_closeness: 0.2
  
  encryption:
    at_rest: true
    algorithm: "AES-256-GCM"
    key_rotation_days: 90

processing:
  spark:
    master: "${SPARK_MASTER}"
    app_name: "tabular-synthetic"
    executor_memory: "4g"
    executor_cores: 2
    max_executors: 10
    shuffle_partitions: 200
  
  dask:
    scheduler: "${DASK_SCHEDULER}"
    n_workers: 4
    threads_per_worker: 2
    memory_limit: "4GB"
  
  batch_processing:
    enabled: true
    schedule: "0 2 * * *"  # 2 AM daily
    retention_days: 30

monitoring:
  metrics:
    enabled: true
    export_interval: 60
    retention: "30d"
  
  logging:
    level: "INFO"
    format: "json"
    output: "file"
    file_path: "/var/log/tabular/app.log"
    max_size: "1GB"
    max_files: 10
  
  alerts:
    enabled: true
    channels: ["email", "slack"]
    rules:
      - name: "high_memory_usage"
        condition: "memory_usage > 90"
        severity: "warning"
      
      - name: "generation_failure"
        condition: "error_rate > 0.05"
        severity: "critical"

security:
  authentication:
    type: "jwt"
    secret_key: "${JWT_SECRET}"
    algorithm: "HS256"
    expiry: 3600
  
  authorization:
    rbac_enabled: true
    default_role: "viewer"
    roles:
      admin: ["*"]
      developer: ["generate", "validate", "read"]
      viewer: ["read"]
  
  rate_limiting:
    enabled: true
    default_limit: "100/hour"
    burst: 20
  
  audit_logging:
    enabled: true
    log_file: "/var/log/tabular/audit.log"
    events: ["authentication", "generation", "data_access"]
```

### 2. Infrastructure Configuration

```yaml
# infrastructure/ansible/playbook.yml
---
- name: Deploy Tabular Infrastructure
  hosts: all
  become: yes
  vars:
    tabular_version: "{{ lookup('env', 'TABULAR_VERSION') | default('latest') }}"
  
  tasks:
    - name: Install system dependencies
      package:
        name:
          - docker.io
          - docker-compose
          - python3-pip
          - nfs-common
          - monitoring-plugins
        state: present
    
    - name: Configure Docker daemon
      copy:
        content: |
          {
            "log-driver": "json-file",
            "log-opts": {
              "max-size": "100m",
              "max-file": "10"
            },
            "storage-driver": "overlay2",
            "default-ulimits": {
              "nofile": {
                "Name": "nofile",
                "Hard": 64000,
                "Soft": 64000
              }
            }
          }
        dest: /etc/docker/daemon.json
      notify: restart docker
    
    - name: Create tabular user
      user:
        name: tabular
        groups: docker
        shell: /bin/bash
        home: /opt/tabular
    
    - name: Create directory structure
      file:
        path: "{{ item }}"
        state: directory
        owner: tabular
        group: tabular
        mode: '0755'
      loop:
        - /opt/tabular
        - /data/tabular
        - /var/log/tabular
        - /etc/tabular/config
    
    - name: Deploy application
      docker_compose:
        project_src: /opt/tabular
        state: present
      become_user: tabular
  
  handlers:
    - name: restart docker
      service:
        name: docker
        state: restarted
```

## Security Implementation

### 1. Network Security

```bash
#!/bin/bash
# network_security.sh

# Configure firewall rules
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (restrict source IPs in production)
sudo ufw allow from 10.0.0.0/8 to any port 22

# Allow application ports
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Internal services (restrict to internal network)
sudo ufw allow from 172.20.0.0/16 to any port 5432  # PostgreSQL
sudo ufw allow from 172.20.0.0/16 to any port 6379  # Redis
sudo ufw allow from 172.20.0.0/16 to any port 9000  # MinIO
sudo ufw allow from 172.20.0.0/16 to any port 7077  # Spark

# Enable firewall
sudo ufw --force enable

# Configure iptables for Docker
sudo iptables -I DOCKER-USER -i ext_if ! -s 10.0.0.0/8 -j DROP

# Set up VPN for remote access (WireGuard)
sudo apt install -y wireguard

# Generate VPN keys
wg genkey | tee server_private.key | wg pubkey > server_public.key

# Configure WireGuard
cat << EOF | sudo tee /etc/wireguard/wg0.conf
[Interface]
Address = 10.100.0.1/24
ListenPort = 51820
PrivateKey = $(cat server_private.key)

# PostUp/PostDown for NAT
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

# Peer configurations added here
EOF

# Enable IP forwarding
echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### 2. Application Security

```python
# security/authentication.py
"""Authentication and authorization for Tabular API."""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict
import secrets
from functools import wraps
from flask import request, jsonify

class SecurityManager:
    """Manage security for Tabular application."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.secret_key = config.get('secret_key', secrets.token_urlsafe(32))
        self.algorithm = config.get('algorithm', 'HS256')
        self.token_expiry = config.get('token_expiry', 3600)
    
    def generate_token(self, user_id: str, role: str = 'user') -> str:
        """Generate JWT token."""
        payload = {
            'user_id': user_id,
            'role': role,
            'exp': datetime.utcnow() + timedelta(seconds=self.token_expiry),
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(
            password.encode('utf-8'), 
            hashed.encode('utf-8')
        )
    
    def require_auth(self, required_role: str = None):
        """Decorator for requiring authentication."""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                token = None
                
                # Check for token in headers
                if 'Authorization' in request.headers:
                    auth_header = request.headers['Authorization']
                    try:
                        token = auth_header.split(' ')[1]  # Bearer <token>
                    except IndexError:
                        return jsonify({'error': 'Invalid token format'}), 401
                
                if not token:
                    return jsonify({'error': 'Token missing'}), 401
                
                # Verify token
                payload = self.verify_token(token)
                if not payload:
                    return jsonify({'error': 'Invalid or expired token'}), 401
                
                # Check role if required
                if required_role and payload.get('role') != required_role:
                    return jsonify({'error': 'Insufficient permissions'}), 403
                
                # Add user info to request
                request.current_user = {
                    'user_id': payload['user_id'],
                    'role': payload['role']
                }
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
    
    def require_api_key(self):
        """Decorator for requiring API key."""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                api_key = request.headers.get('X-API-Key')
                
                if not api_key:
                    return jsonify({'error': 'API key missing'}), 401
                
                # Verify API key (implement your logic)
                if not self.verify_api_key(api_key):
                    return jsonify({'error': 'Invalid API key'}), 401
                
                return f(*args, **kwargs)
            
            return decorated_function
        return decorator
```

### 3. Data Security

```python
# security/encryption.py
"""Data encryption and privacy for Tabular."""

import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import hashlib
from typing import Dict, Any, List

class DataEncryption:
    """Handle data encryption and decryption."""
    
    def __init__(self, master_key: str = None):
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = os.environ.get('MASTER_KEY', '').encode()
        
        if not self.master_key:
            raise ValueError("Master key not provided")
        
        self.fernet = self._create_fernet()
    
    def _create_fernet(self) -> Fernet:
        """Create Fernet instance from master key."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'stable_salt',  # Use proper salt in production
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        return Fernet(key)
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data."""
        return self.fernet.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        return self.fernet.decrypt(encrypted_data)
    
    def encrypt_field(self, value: Any) -> str:
        """Encrypt a single field value."""
        if value is None:
            return None
        
        str_value = str(value)
        encrypted = self.encrypt_data(str_value.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_field(self, encrypted_value: str) -> str:
        """Decrypt a single field value."""
        if encrypted_value is None:
            return None
        
        encrypted_bytes = base64.b64decode(encrypted_value.encode())
        decrypted = self.decrypt_data(encrypted_bytes)
        return decrypted.decode()
    
    def hash_identifier(self, identifier: str) -> str:
        """Create one-way hash of identifier."""
        return hashlib.sha256(
            (identifier + self.master_key.decode()).encode()
        ).hexdigest()


class PrivacyEngine:
    """Implement privacy-preserving techniques."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.k_anonymity = config.get('k_anonymity', 5)
        self.l_diversity = config.get('l_diversity', 3)
        self.epsilon = config.get('epsilon', 1.0)
        self.delta = config.get('delta', 1e-5)
    
    def apply_k_anonymity(self, df, quasi_identifiers: List[str]) -> Any:
        """Apply k-anonymity to dataframe."""
        # Group by quasi-identifiers
        grouped = df.groupby(quasi_identifiers)
        
        # Filter groups with less than k records
        k_anonymous = grouped.filter(lambda x: len(x) >= self.k_anonymity)
        
        # Generalize remaining records
        small_groups = grouped.filter(lambda x: len(x) < self.k_anonymity)
        if not small_groups.empty:
            # Apply generalization techniques
            for col in quasi_identifiers:
                if df[col].dtype in ['int64', 'float64']:
                    # Numerical: use ranges
                    small_groups[col] = pd.cut(
                        small_groups[col], 
                        bins=5, 
                        labels=False
                    )
                else:
                    # Categorical: use hierarchy
                    small_groups[col] = self._generalize_categorical(
                        small_groups[col]
                    )
        
        return pd.concat([k_anonymous, small_groups])
    
    def add_differential_privacy(self, value: float, sensitivity: float) -> float:
        """Add Laplace noise for differential privacy."""
        import numpy as np
        
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        
        return value + noise
    
    def _generalize_categorical(self, series):
        """Generalize categorical values."""
        # Implement hierarchy-based generalization
        # This is a simplified example
        value_counts = series.value_counts()
        
        # Group rare values
        threshold = len(series) * 0.05
        rare_values = value_counts[value_counts < threshold].index
        
        return series.replace(rare_values, '*')
```

## Performance Optimization

### 1. Database Optimization

```sql
-- Performance optimization for PostgreSQL

-- Create indexes for common queries
CREATE INDEX idx_generations_created_at ON generations(created_at DESC);
CREATE INDEX idx_generations_user_status ON generations(user_id, status);
CREATE INDEX idx_datasets_schema_version ON datasets(schema_id, version);

-- Partitioning for large tables
CREATE TABLE generations_2024_01 PARTITION OF generations
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Optimize configuration
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET work_mem = '32MB';
ALTER SYSTEM SET min_wal_size = '2GB';
ALTER SYSTEM SET max_wal_size = '8GB';

-- Create materialized views for analytics
CREATE MATERIALIZED VIEW generation_stats AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    COUNT(*) as total_generations,
    SUM(row_count) as total_rows,
    AVG(generation_time) as avg_time,
    COUNT(DISTINCT user_id) as unique_users
FROM generations
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY 1;

-- Refresh materialized view periodically
CREATE OR REPLACE FUNCTION refresh_generation_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY generation_stats;
END;
$$ LANGUAGE plpgsql;

-- Schedule refresh
SELECT cron.schedule('refresh-stats', '*/15 * * * *', 'SELECT refresh_generation_stats()');
```

### 2. Caching Strategy

```python
# caching/cache_manager.py
"""Intelligent caching for Tabular system."""

import redis
import pickle
import hashlib
import json
from typing import Any, Optional, Dict, List
from functools import wraps
import pandas as pd

class CacheManager:
    """Manage multi-level caching."""
    
    def __init__(self, redis_config: Dict):
        self.redis_client = redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'],
            password=redis_config.get('password'),
            db=redis_config.get('db', 0),
            decode_responses=False
        )
        
        # Cache configuration
        self.default_ttl = 3600  # 1 hour
        self.max_cache_size = 1024 * 1024 * 100  # 100MB per key
    
    def _generate_key(self, prefix: str, params: Dict) -> str:
        """Generate cache key from parameters."""
        # Sort params for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(sorted_params.encode()).hexdigest()
        
        return f"{prefix}:{param_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return pickle.loads(cached_data)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache."""
        try:
            # Check size limit
            serialized = pickle.dumps(value)
            if len(serialized) > self.max_cache_size:
                logger.warning(f"Cache value too large: {len(serialized)} bytes")
                return False
            
            ttl = ttl or self.default_ttl
            self.redis_client.setex(key, ttl, serialized)
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        count = 0
        for key in self.redis_client.scan_iter(match=pattern):
            self.redis_client.delete(key)
            count += 1
        return count
    
    def cache_dataframe(self, df: pd.DataFrame, key: str, ttl: int = None) -> bool:
        """Cache pandas DataFrame efficiently."""
        try:
            # Use parquet for efficient storage
            buffer = io.BytesIO()
            df.to_parquet(buffer, compression='snappy')
            buffer.seek(0)
            
            return self.set(key, buffer.getvalue(), ttl)
            
        except Exception as e:
            logger.error(f"DataFrame cache error: {e}")
            return False
    
    def get_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """Get DataFrame from cache."""
        try:
            data = self.get(key)
            if data:
                buffer = io.BytesIO(data)
                return pd.read_parquet(buffer)
            return None
            
        except Exception as e:
            logger.error(f"DataFrame retrieval error: {e}")
            return None
    
    def cached(self, prefix: str, ttl: int = None):
        """Decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_params = {
                    'args': args,
                    'kwargs': kwargs
                }
                cache_key = self._generate_key(prefix, cache_params)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Cache result
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator


class QueryCache:
    """Cache for database queries."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.query_prefix = "query"
    
    def cache_query(self, query: str, params: tuple, result: List[Dict], 
                   ttl: int = 300) -> bool:
        """Cache query results."""
        key = self._query_key(query, params)
        return self.cache.set(key, result, ttl)
    
    def get_query(self, query: str, params: tuple) -> Optional[List[Dict]]:
        """Get cached query results."""
        key = self._query_key(query, params)
        return self.cache.get(key)
    
    def _query_key(self, query: str, params: tuple) -> str:
        """Generate key for query."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        params_hash = hashlib.md5(str(params).encode()).hexdigest()
        
        return f"{self.query_prefix}:{query_hash}:{params_hash}"
    
    def invalidate_table(self, table_name: str):
        """Invalidate all queries for a table."""
        # Clear all cached queries that might involve this table
        pattern = f"{self.query_prefix}:*"
        self.cache.clear_pattern(pattern)
```

### 3. Distributed Processing

```python
# processing/spark_manager.py
"""Spark processing for large-scale operations."""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.sql.functions as F
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class SparkManager:
    """Manage Spark operations for Tabular."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.spark = self._create_spark_session()
    
    def _create_spark_session(self) -> SparkSession:
        """Create optimized Spark session."""
        builder = SparkSession.builder \
            .appName(self.config.get('app_name', 'tabular-synthetic')) \
            .master(self.config.get('master', 'local[*]'))
        
        # Optimization settings
        optimizations = {
            'spark.sql.adaptive.enabled': 'true',
            'spark.sql.adaptive.coalescePartitions.enabled': 'true',
            'spark.sql.adaptive.skewJoin.enabled': 'true',
            'spark.sql.adaptive.localShuffleReader.enabled': 'true',
            'spark.sql.shuffle.partitions': '200',
            'spark.serializer': 'org.apache.spark.serializer.KryoSerializer',
            'spark.sql.execution.arrow.pyspark.enabled': 'true',
            'spark.sql.execution.arrow.maxRecordsPerBatch': '10000',
            'spark.dynamicAllocation.enabled': 'true',
            'spark.dynamicAllocation.minExecutors': '1',
            'spark.dynamicAllocation.maxExecutors': '10',
        }
        
        for key, value in optimizations.items():
            builder = builder.config(key, value)
        
        # Memory settings
        builder = builder.config('spark.executor.memory', 
                               self.config.get('executor_memory', '4g'))
        builder = builder.config('spark.executor.cores', 
                               self.config.get('executor_cores', '2'))
        
        return builder.getOrCreate()
    
    def process_large_dataset(self, input_path: str, output_path: str, 
                            transformations: List[Dict]) -> Dict:
        """Process large dataset with transformations."""
        try:
            # Read data
            df = self.spark.read.parquet(input_path)
            
            # Apply transformations
            for transform in transformations:
                df = self._apply_transformation(df, transform)
            
            # Optimize before writing
            df = df.coalesce(self._optimal_partitions(df))
            
            # Write results
            df.write.mode('overwrite').parquet(output_path)
            
            # Return statistics
            return {
                'rows_processed': df.count(),
                'partitions': df.rdd.getNumPartitions(),
                'columns': len(df.columns)
            }
            
        except Exception as e:
            logger.error(f"Spark processing failed: {e}")
            raise
    
    def generate_synthetic_batch(self, schema: Dict, num_rows: int,
                               output_path: str) -> bool:
        """Generate synthetic data in batch mode."""
        try:
            # Create schema
            spark_schema = self._create_spark_schema(schema)
            
            # Generate data in partitions
            num_partitions = max(1, num_rows // 1000000)  # 1M rows per partition
            
            # Use RDD to generate data in parallel
            rdd = self.spark.sparkContext.parallelize(
                range(num_rows), 
                num_partitions
            )
            
            # Map to generate rows
            def generate_row(idx):
                # Implement your generation logic
                return Row(**generate_synthetic_row(schema, idx))
            
            row_rdd = rdd.map(generate_row)
            
            # Create DataFrame
            df = self.spark.createDataFrame(row_rdd, spark_schema)
            
            # Write to output
            df.write.mode('overwrite').parquet(output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return False
    
    def _optimal_partitions(self, df) -> int:
        """Calculate optimal number of partitions."""
        # Rule of thumb: 128MB per partition
        total_size = df.rdd.map(lambda x: len(str(x))).sum()
        optimal = max(1, int(total_size / (128 * 1024 * 1024)))
        
        return min(optimal, 1000)  # Cap at 1000 partitions
    
    def _create_spark_schema(self, schema: Dict) -> StructType:
        """Convert schema dict to Spark StructType."""
        fields = []
        
        for col_name, col_type in schema.items():
            spark_type = self._map_type(col_type)
            fields.append(StructField(col_name, spark_type, True))
        
        return StructType(fields)
    
    def _map_type(self, type_str: str) -> DataType:
        """Map string type to Spark DataType."""
        type_mapping = {
            'string': StringType(),
            'integer': IntegerType(),
            'long': LongType(),
            'float': FloatType(),
            'double': DoubleType(),
            'boolean': BooleanType(),
            'date': DateType(),
            'timestamp': TimestampType(),
        }
        
        return type_mapping.get(type_str, StringType())
```

## High Availability Setup

### 1. Database HA

```bash
#!/bin/bash
# setup_postgres_ha.sh

# PostgreSQL High Availability with Patroni

# Install Patroni
pip3 install patroni[etcd] psycopg2-binary

# Create Patroni configuration
cat << EOF > /etc/patroni/patroni.yml
scope: tabular-postgres
namespace: /db/
name: $(hostname)

restapi:
  listen: 0.0.0.0:8008
  connect_address: $(hostname -I | awk '{print $1}'):8008

etcd:
  hosts:
    - etcd1:2379
    - etcd2:2379
    - etcd3:2379

bootstrap:
  dcs:
    ttl: 30
    loop_wait: 10
    retry_timeout: 10
    maximum_lag_on_failover: 1048576
    master_start_timeout: 300
    synchronous_mode: false
    
  postgresql:
    use_pg_rewind: true
    use_slots: true
    parameters:
      max_connections: 200
      shared_buffers: 2GB
      effective_cache_size: 6GB
      maintenance_work_mem: 512MB
      checkpoint_completion_target: 0.9
      wal_buffers: 16MB
      default_statistics_target: 100
      random_page_cost: 1.1
      effective_io_concurrency: 200
      work_mem: 32MB
      min_wal_size: 1GB
      max_wal_size: 4GB
      wal_level: replica
      hot_standby: "on"
      wal_log_hints: "on"
      archive_mode: "on"
      archive_command: 'test ! -f /archive/%f && cp %p /archive/%f'
      max_wal_senders: 10
      max_replication_slots: 10
      hot_standby_feedback: "on"

postgresql:
  listen: 0.0.0.0:5432
  connect_address: $(hostname -I | awk '{print $1}'):5432
  data_dir: /data/postgresql/13/main
  bin_dir: /usr/lib/postgresql/13/bin
  authentication:
    replication:
      username: replicator
      password: replication_password
    superuser:
      username: postgres
      password: postgres_password
  parameters:
    unix_socket_directories: '/var/run/postgresql'

watchdog:
  mode: automatic
  device: /dev/watchdog
  safety_margin: 5

tags:
  nofailover: false
  noloadbalance: false
  clonefrom: false
  nosync: false
EOF

# Create systemd service
cat << EOF > /etc/systemd/system/patroni.service
[Unit]
Description=Patroni PostgreSQL
After=syslog.target network.target etcd.service
Wants=network-online.target

[Service]
Type=simple
User=postgres
Group=postgres
ExecStart=/usr/local/bin/patroni /etc/patroni/patroni.yml
ExecReload=/bin/kill -s HUP \$MAINPID
KillMode=mixed
KillSignal=SIGINT
TimeoutSec=30
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Start Patroni
sudo systemctl daemon-reload
sudo systemctl enable patroni
sudo systemctl start patroni

# Set up HAProxy for connection pooling
cat << EOF > /etc/haproxy/haproxy.cfg
global
    maxconn 100
    log 127.0.0.1 local0

defaults
    log global
    mode tcp
    retries 3
    timeout queue 1m
    timeout connect 10s
    timeout client 1m
    timeout server 1m
    timeout check 10s
    maxconn 100

listen postgres
    bind *:5000
    option httpchk OPTIONS /master
    http-check expect status 200
    default-server inter 3s fall 3 rise 2 on-marked-down shutdown-sessions
    server postgresql_node1 node1:5432 maxconn 100 check port 8008
    server postgresql_node2 node2:5432 maxconn 100 check port 8008
    server postgresql_node3 node3:5432 maxconn 100 check port 8008
EOF
```

### 2. Redis HA

```bash
#!/bin/bash
# setup_redis_ha.sh

# Redis Sentinel setup for HA

# Install Redis Sentinel
sudo apt install -y redis-sentinel

# Configure Redis master
cat << EOF > /etc/redis/redis.conf
bind 0.0.0.0
port 6379
requirepass ${REDIS_PASSWORD}
masterauth ${REDIS_PASSWORD}
maxmemory 4gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
dir /data/redis
dbfilename dump.rdb
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
EOF

# Configure Sentinel
cat << EOF > /etc/redis/sentinel.conf
bind 0.0.0.0
port 26379
sentinel monitor mymaster redis-master 6379 2
sentinel auth-pass mymaster ${REDIS_PASSWORD}
sentinel down-after-milliseconds mymaster 5000
sentinel parallel-syncs mymaster 1
sentinel failover-timeout mymaster 60000
sentinel notification-script mymaster /opt/scripts/redis-notify.sh
EOF

# Create notification script
cat << 'EOF' > /opt/scripts/redis-notify.sh
#!/bin/bash
# Redis Sentinel notification script

EVENT_TYPE=$1
EVENT_DESCRIPTION=$2

# Log the event
echo "$(date) - $EVENT_TYPE: $EVENT_DESCRIPTION" >> /var/log/redis/sentinel-events.log

# Send alert (implement your alerting logic)
curl -X POST https://alerts.example.com/webhook \
  -H "Content-Type: application/json" \
  -d "{\"event\": \"$EVENT_TYPE\", \"description\": \"$EVENT_DESCRIPTION\"}"
EOF

chmod +x /opt/scripts/redis-notify.sh

# Start services
sudo systemctl enable redis-server redis-sentinel
sudo systemctl start redis-server redis-sentinel
```

### 3. Application HA

```python
# ha/health_checker.py
"""Health checking and failover for Tabular services."""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional
from datetime import datetime
import consul

logger = logging.getLogger(__name__)

class HealthChecker:
    """Monitor service health and manage failover."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.consul = consul.Consul(
            host=config.get('consul_host', 'localhost'),
            port=config.get('consul_port', 8500)
        )
        self.services = {}
        self.check_interval = config.get('check_interval', 10)
    
    async def start(self):
        """Start health checking loop."""
        while True:
            try:
                await self.check_all_services()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def check_all_services(self):
        """Check health of all registered services."""
        services = self.consul.agent.services()
        
        for service_id, service_info in services.items():
            health = await self.check_service_health(service_info)
            
            if health != self.services.get(service_id, {}).get('health'):
                # Health status changed
                await self.handle_health_change(
                    service_id, 
                    service_info, 
                    health
                )
            
            self.services[service_id] = {
                'info': service_info,
                'health': health,
                'last_check': datetime.utcnow()
            }
    
    async def check_service_health(self, service_info: Dict) -> str:
        """Check individual service health."""
        address = service_info['Address']
        port = service_info['Port']
        
        health_endpoint = f"http://{address}:{port}/health"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    health_endpoint, 
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('status', 'healthy')
                    else:
                        return 'unhealthy'
                        
        except Exception as e:
            logger.error(f"Health check failed for {service_info['Service']}: {e}")
            return 'unhealthy'
    
    async def handle_health_change(self, service_id: str, 
                                 service_info: Dict, new_health: str):
        """Handle service health status change."""
        service_name = service_info['Service']
        
        if new_health == 'unhealthy':
            logger.warning(f"Service {service_name} ({service_id}) is unhealthy")
            
            # Update Consul health check
            self.consul.agent.check.update(
                f"service:{service_id}",
                consul.Check.Status.CRITICAL,
                output=f"Service unhealthy at {datetime.utcnow()}"
            )
            
            # Trigger failover if needed
            if service_name in ['tabular-api', 'tabular-postgres']:
                await self.trigger_failover(service_name, service_id)
        
        else:
            logger.info(f"Service {service_name} ({service_id}) is healthy")
            
            # Update Consul health check
            self.consul.agent.check.update(
                f"service:{service_id}",
                consul.Check.Status.PASSING,
                output=f"Service healthy at {datetime.utcnow()}"
            )
    
    async def trigger_failover(self, service_name: str, failed_service_id: str):
        """Trigger failover for critical services."""
        logger.warning(f"Triggering failover for {service_name}")
        
        # Get healthy instances
        healthy_instances = []
        for service_id, service_data in self.services.items():
            if (service_data['info']['Service'] == service_name and 
                service_data['health'] == 'healthy' and
                service_id != failed_service_id):
                healthy_instances.append(service_data['info'])
        
        if not healthy_instances:
            logger.error(f"No healthy instances available for {service_name}")
            # Send critical alert
            await self.send_alert(
                'critical',
                f"All {service_name} instances are down"
            )
            return
        
        # Update load balancer configuration
        await self.update_load_balancer(service_name, healthy_instances)
        
        # Send alert
        await self.send_alert(
            'warning',
            f"Failover triggered for {service_name}. "
            f"Failed instance: {failed_service_id}"
        )
    
    async def update_load_balancer(self, service_name: str, 
                                  healthy_instances: List[Dict]):
        """Update load balancer with healthy instances."""
        # Implementation depends on your load balancer
        # Example for HAProxy dynamic configuration
        
        backend_config = []
        for idx, instance in enumerate(healthy_instances):
            backend_config.append(
                f"server {service_name}_{idx} "
                f"{instance['Address']}:{instance['Port']} check"
            )
        
        # Update HAProxy configuration
        # This is a simplified example
        config_data = "\n".join(backend_config)
        
        # Write to HAProxy socket or use API
        # ...
    
    async def send_alert(self, severity: str, message: str):
        """Send alert notification."""
        alert_config = self.config.get('alerting', {})
        
        if alert_config.get('enabled'):
            webhook_url = alert_config.get('webhook_url')
            
            async with aiohttp.ClientSession() as session:
                await session.post(
                    webhook_url,
                    json={
                        'severity': severity,
                        'message': message,
                        'timestamp': datetime.utcnow().isoformat(),
                        'service': 'tabular-ha'
                    }
                )
```

## Monitoring and Maintenance

### 1. Monitoring Stack

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'tabular-prod'
    
rule_files:
  - 'alerts.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

scrape_configs:
  - job_name: 'tabular-api'
    static_configs:
      - targets: ['api1:8000', 'api2:8000', 'api3:8000']
    metrics_path: '/metrics'
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
      
  - job_name: 'node'
    static_configs:
      - targets: ['node1:9100', 'node2:9100', 'node3:9100']
      
  - job_name: 'spark'
    static_configs:
      - targets: ['spark-master:8080']
    metrics_path: '/metrics/prometheus'
```

```yaml
# monitoring/alerts.yml
groups:
  - name: tabular
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"
          
      - alert: HighMemoryUsage
        expr: (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) < 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Available memory is below 10%"
          
      - alert: DatabaseConnectionsHigh
        expr: pg_stat_database_numbackends{datname="tabular"} > 180
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Database connections near limit"
          description: "{{ $value }} connections to database"
          
      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes) < 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Disk space critically low"
          description: "Less than 10% disk space remaining"
```

### 2. Grafana Dashboards

```json
{
  "dashboard": {
    "title": "Tabular System Overview",
    "panels": [
      {
        "title": "API Request Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (method)",
            "legendFormat": "{{method}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Generation Performance",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(generation_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(generation_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Data Volume",
        "targets": [
          {
            "expr": "sum(increase(generated_rows_total[24h]))",
            "legendFormat": "Rows generated (24h)"
          }
        ],
        "type": "stat"
      },
      {
        "title": "System Resources",
        "targets": [
          {
            "expr": "100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU Usage %"
          },
          {
            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
            "legendFormat": "Memory Usage %"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

### 3. Maintenance Scripts

```python
# maintenance/cleanup.py
"""Maintenance and cleanup tasks for Tabular."""

import os
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
import psycopg2
from typing import Dict, List

logger = logging.getLogger(__name__)

class MaintenanceManager:
    """Handle system maintenance tasks."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.db_config = config['database']
        self.storage_config = config['storage']
    
    def run_daily_maintenance(self):
        """Run daily maintenance tasks."""
        logger.info("Starting daily maintenance")
        
        tasks = [
            self.cleanup_old_data,
            self.vacuum_database,
            self.archive_logs,
            self.cleanup_temp_files,
            self.update_statistics
        ]
        
        for task in tasks:
            try:
                task()
            except Exception as e:
                logger.error(f"Maintenance task {task.__name__} failed: {e}")
    
    def cleanup_old_data(self):
        """Clean up old generated data based on retention policy."""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        # Get retention settings
        hot_retention = self.storage_config['hot_storage']['ttl']
        warm_retention = self.storage_config['warm_storage']['ttl']
        
        # Move data from hot to warm storage
        cutoff_date = datetime.now() - timedelta(days=hot_retention)
        
        cur.execute("""
            UPDATE datasets 
            SET storage_tier = 'warm'
            WHERE storage_tier = 'hot' 
            AND created_at < %s
        """, (cutoff_date,))
        
        moved_count = cur.rowcount
        logger.info(f"Moved {moved_count} datasets from hot to warm storage")
        
        # Delete old data from warm storage
        delete_date = datetime.now() - timedelta(days=warm_retention)
        
        cur.execute("""
            DELETE FROM datasets 
            WHERE storage_tier = 'warm' 
            AND created_at < %s
        """, (delete_date,))
        
        deleted_count = cur.rowcount
        logger.info(f"Deleted {deleted_count} old datasets")
        
        conn.commit()
        cur.close()
        conn.close()
    
    def vacuum_database(self):
        """Run VACUUM on database tables."""
        conn = psycopg2.connect(**self.db_config)
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        tables = ['datasets', 'generations', 'users', 'audit_logs']
        
        for table in tables:
            logger.info(f"Vacuuming table: {table}")
            cur.execute(f"VACUUM ANALYZE {table}")
        
        cur.close()
        conn.close()
    
    def archive_logs(self):
        """Archive old log files."""
        log_dir = Path(self.config['logging']['file_path']).parent
        archive_dir = log_dir / 'archive'
        archive_dir.mkdir(exist_ok=True)
        
        # Archive logs older than 7 days
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for log_file in log_dir.glob('*.log'):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                # Compress and move
                archive_path = archive_dir / f"{log_file.name}.gz"
                
                import gzip
                with open(log_file, 'rb') as f_in:
                    with gzip.open(archive_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                log_file.unlink()
                logger.info(f"Archived log file: {log_file.name}")
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        temp_dirs = [
            '/tmp/tabular',
            '/data/tabular/temp',
            '/data/spark/work'
        ]
        
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                # Remove files older than 1 day
                cutoff_time = time.time() - 86400
                
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.getmtime(file_path) < cutoff_time:
                            os.remove(file_path)
    
    def update_statistics(self):
        """Update system statistics."""
        conn = psycopg2.connect(**self.db_config)
        cur = conn.cursor()
        
        # Update materialized views
        views = ['generation_stats', 'user_activity', 'data_quality_metrics']
        
        for view in views:
            logger.info(f"Refreshing materialized view: {view}")
            cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}")
        
        conn.commit()
        cur.close()
        conn.close()


def main():
    """Run maintenance tasks."""
    import yaml
    
    with open('/etc/tabular/config/production.yaml') as f:
        config = yaml.safe_load(f)
    
    manager = MaintenanceManager(config)
    manager.run_daily_maintenance()

if __name__ == "__main__":
    main()
```

## Best Practices

### 1. Deployment Checklist

- [ ] System requirements verified
- [ ] Storage properly configured (RAID, partitions)
- [ ] Network security implemented
- [ ] SSL/TLS certificates installed
- [ ] Database optimized and secured
- [ ] Backup strategy implemented
- [ ] Monitoring and alerting configured
- [ ] High availability setup completed
- [ ] Documentation updated
- [ ] Runbooks created

### 2. Security Best Practices

1. **Network Isolation**
   - Use private networks for internal communication
   - Implement VPN for remote access
   - Configure strict firewall rules

2. **Access Control**
   - Implement RBAC
   - Use strong authentication
   - Regular access audits

3. **Data Protection**
   - Encrypt data at rest and in transit
   - Implement data retention policies
   - Regular security scans

### 3. Performance Best Practices

1. **Resource Allocation**
   - Right-size instances
   - Monitor resource usage
   - Plan for peak loads

2. **Optimization**
   - Regular performance tuning
   - Query optimization
   - Caching strategy

3. **Scaling**
   - Horizontal scaling for stateless components
   - Vertical scaling for databases
   - Auto-scaling policies

### 4. Operational Best Practices

1. **Monitoring**
   - Comprehensive metrics collection
   - Proactive alerting
   - Regular health checks

2. **Maintenance**
   - Regular patching schedule
   - Automated cleanup tasks
   - Performance reviews

3. **Documentation**
   - Keep documentation current
   - Document all procedures
   - Maintain runbooks

This comprehensive guide provides everything needed to deploy and operate the Tabular synthetic data system on-premise, from single-server setups to large-scale enterprise deployments.