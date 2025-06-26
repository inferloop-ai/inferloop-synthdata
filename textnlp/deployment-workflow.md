# TextNLP Deployment Workflow Guide

## Overview

This document provides comprehensive deployment workflows for the TextNLP Synthetic Data Generation platform across multiple environments and cloud providers. It covers everything from local development to production deployment with detailed step-by-step instructions.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud-Specific Deployments](#cloud-specific-deployments)
6. [CI/CD Pipeline](#cicd-pipeline)
7. [Monitoring Setup](#monitoring-setup)
8. [Rollback Procedures](#rollback-procedures)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools

```bash
# Core tools
- Python 3.11+
- Docker 24.0+
- Kubernetes 1.28+
- Helm 3.12+
- Terraform 1.5+

# Cloud CLIs
- AWS CLI 2.13+
- Azure CLI 2.53+
- gcloud SDK 450+

# Development tools
- Git 2.40+
- Make 4.3+
- jq 1.6+
- yq 4.35+
```

### Tool Installation

```bash
# Install Python and pip
curl -sSL https://install.python-poetry.org | python3 -

# Install Docker
curl -fsSL https://get.docker.com | sh

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install Terraform
wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform
```

## Local Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/inferloop/textnlp.git
cd textnlp
```

### 2. Setup Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
cat > .env << EOF
# Application settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=postgresql://textnlp:password@localhost:5432/textnlp
REDIS_URL=redis://localhost:6379/0

# API Keys (for local testing)
JWT_SECRET_KEY=$(openssl rand -hex 32)
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here

# Model settings
DEFAULT_MODEL=gpt2
MODEL_CACHE_DIR=./models
MAX_TOKENS=2048
EOF
```

### 4. Start Local Services

```bash
# Start PostgreSQL and Redis using Docker Compose
docker-compose -f docker-compose.dev.yml up -d

# Run database migrations
alembic upgrade head

# Seed initial data
python scripts/seed_data.py
```

### 5. Run Application

```bash
# Start API server
uvicorn textnlp.api.app:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start background workers
celery -A textnlp.workers worker --loglevel=info

# In another terminal, start the scheduler
celery -A textnlp.workers beat --loglevel=info
```

## Docker Deployment

### 1. Build Images

```bash
# Build all images
make docker-build

# Or build individually
docker build -t textnlp-api:latest -f docker/Dockerfile.api .
docker build -t textnlp-worker:latest -f docker/Dockerfile.worker .
docker build -t textnlp-nginx:latest -f docker/Dockerfile.nginx .
```

### 2. Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

### 3. Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    image: textnlp-api:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://textnlp:password@postgres:5432/textnlp
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - model-cache:/app/models
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G

  worker:
    image: textnlp-worker:latest
    environment:
      - DATABASE_URL=postgresql://textnlp:password@postgres:5432/textnlp
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - model-cache:/app/models
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '4'
          memory: 8G

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=textnlp
      - POSTGRES_USER=textnlp
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"

  nginx:
    image: textnlp-nginx:latest
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - api
    volumes:
      - ./nginx/ssl:/etc/nginx/ssl:ro

volumes:
  postgres-data:
  redis-data:
  model-cache:
```

## Kubernetes Deployment

### 1. Prepare Kubernetes Cluster

```bash
# For local testing with kind
kind create cluster --name textnlp --config kind-config.yaml

# Or with minikube
minikube start --cpus=4 --memory=8192 --disk-size=50g

# Verify cluster access
kubectl cluster-info
kubectl get nodes
```

### 2. Create Namespace and Secrets

```bash
# Create namespace
kubectl create namespace textnlp-prod

# Create secrets
kubectl create secret generic textnlp-secrets \
  --from-literal=database-url='postgresql://user:pass@postgres:5432/textnlp' \
  --from-literal=redis-url='redis://redis:6379/0' \
  --from-literal=jwt-secret='your-secret-key' \
  -n textnlp-prod

# Create image pull secret (if using private registry)
kubectl create secret docker-registry regcred \
  --docker-server=your-registry.com \
  --docker-username=your-username \
  --docker-password=your-password \
  -n textnlp-prod
```

### 3. Deploy with Helm

```bash
# Add Helm repository
helm repo add textnlp https://charts.textnlp.io
helm repo update

# Install with custom values
helm install textnlp textnlp/textnlp \
  --namespace textnlp-prod \
  --values helm/values.prod.yaml \
  --set image.tag=latest \
  --set ingress.hosts[0].host=api.textnlp.io \
  --set postgresql.enabled=true \
  --set redis.enabled=true
```

### 4. Helm Values Configuration

```yaml
# helm/values.prod.yaml
replicaCount: 3

image:
  repository: textnlp/api
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
  hosts:
    - host: api.textnlp.io
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: textnlp-tls
      hosts:
        - api.textnlp.io

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 100
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

postgresql:
  enabled: true
  auth:
    database: textnlp
    username: textnlp
  primary:
    persistence:
      size: 100Gi
    resources:
      limits:
        memory: 2Gi
        cpu: 1000m

redis:
  enabled: true
  auth:
    enabled: false
  master:
    persistence:
      size: 10Gi
```

### 5. Deploy Application Components

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Or use kustomize
kubectl apply -k k8s/overlays/production
```

## Cloud-Specific Deployments

### AWS Deployment

#### 1. Setup Infrastructure with Terraform

```bash
cd terraform/aws

# Initialize Terraform
terraform init

# Create workspace
terraform workspace new production

# Plan deployment
terraform plan -var-file=production.tfvars

# Apply configuration
terraform apply -var-file=production.tfvars -auto-approve
```

#### 2. Deploy to ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REGISTRY
docker build -t textnlp-api .
docker tag textnlp-api:latest $ECR_REGISTRY/textnlp-api:latest
docker push $ECR_REGISTRY/textnlp-api:latest

# Update ECS service
aws ecs update-service \
  --cluster textnlp-prod \
  --service textnlp-api \
  --force-new-deployment
```

#### 3. Deploy to EKS

```bash
# Update kubeconfig
aws eks update-kubeconfig --region us-east-1 --name textnlp-eks

# Deploy using Helm
helm upgrade --install textnlp ./helm/textnlp \
  --namespace textnlp-prod \
  --values helm/values.eks.yaml
```

### Azure Deployment

#### 1. Setup Infrastructure

```bash
cd terraform/azure

# Login to Azure
az login
az account set --subscription "Your Subscription"

# Initialize Terraform
terraform init

# Deploy infrastructure
terraform apply -var-file=production.tfvars
```

#### 2. Deploy to AKS

```bash
# Get AKS credentials
az aks get-credentials --resource-group textnlp-rg --name textnlp-aks

# Create Azure Container Registry secret
kubectl create secret docker-registry acr-secret \
  --docker-server=textnlpacr.azurecr.io \
  --docker-username=$ACR_USERNAME \
  --docker-password=$ACR_PASSWORD \
  -n textnlp-prod

# Deploy application
helm upgrade --install textnlp ./helm/textnlp \
  --namespace textnlp-prod \
  --values helm/values.aks.yaml
```

### GCP Deployment

#### 1. Setup Infrastructure

```bash
cd terraform/gcp

# Authenticate with GCP
gcloud auth application-default login
gcloud config set project textnlp-prod

# Initialize Terraform
terraform init

# Deploy infrastructure
terraform apply -var-file=production.tfvars
```

#### 2. Deploy to GKE

```bash
# Get GKE credentials
gcloud container clusters get-credentials textnlp-gke --region us-central1

# Deploy application
helm upgrade --install textnlp ./helm/textnlp \
  --namespace textnlp-prod \
  --values helm/values.gke.yaml
```

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
  workflow_dispatch:

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
          
      - name: Run tests
        run: |
          poetry run pytest --cov=textnlp --cov-report=xml
          
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v3
      
      - name: Log in to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
          
      - name: Deploy to EKS
        run: |
          aws eks update-kubeconfig --name textnlp-eks
          kubectl set image deployment/textnlp-api \
            api=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n textnlp-prod
          kubectl rollout status deployment/textnlp-api -n textnlp-prod
```

### GitLab CI/CD Pipeline

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: ""
  REGISTRY: registry.gitlab.com
  IMAGE_NAME: $CI_PROJECT_PATH

test:
  stage: test
  image: python:3.11
  script:
    - pip install poetry
    - poetry install
    - poetry run pytest --cov=textnlp
  coverage: '/TOTAL.*\s+(\d+%)$/'

build:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - docker tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA $CI_REGISTRY_IMAGE:latest
    - docker push $CI_REGISTRY_IMAGE:latest

deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  only:
    - main
  script:
    - kubectl config set-cluster k8s --server="$KUBE_URL" --insecure-skip-tls-verify=true
    - kubectl config set-credentials admin --token="$KUBE_TOKEN"
    - kubectl config set-context default --cluster=k8s --user=admin
    - kubectl config use-context default
    - kubectl set image deployment/textnlp-api api=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA -n textnlp-prod
    - kubectl rollout status deployment/textnlp-api -n textnlp-prod
```

## Monitoring Setup

### 1. Prometheus and Grafana

```bash
# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values monitoring/prometheus-values.yaml

# Access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
# Default login: admin/prom-operator
```

### 2. Application Metrics

```python
# textnlp/api/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
request_count = Counter('textnlp_requests_total', 
                       'Total requests', 
                       ['method', 'endpoint', 'status'])

request_duration = Histogram('textnlp_request_duration_seconds',
                           'Request duration',
                           ['method', 'endpoint'])

active_connections = Gauge('textnlp_active_connections',
                         'Active connections')

tokens_generated = Counter('textnlp_tokens_generated_total',
                         'Total tokens generated',
                         ['model'])

# Middleware to track metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response
```

### 3. Logging Configuration

```yaml
# logging-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
  namespace: textnlp-prod
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush         1
        Log_Level     info
        Daemon        off
        
    [INPUT]
        Name              tail
        Path              /var/log/containers/*textnlp*.log
        Parser            docker
        Tag               kube.*
        Refresh_Interval  5
        
    [FILTER]
        Name                kubernetes
        Match               kube.*
        Kube_URL            https://kubernetes.default.svc:443
        Kube_CA_File        /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        Kube_Token_File     /var/run/secrets/kubernetes.io/serviceaccount/token
        Merge_Log           On
        K8S-Logging.Parser  On
        K8S-Logging.Exclude On
        
    [OUTPUT]
        Name            es
        Match           *
        Host            elasticsearch.monitoring.svc.cluster.local
        Port            9200
        Index           textnlp
        Type            _doc
```

## Rollback Procedures

### 1. Kubernetes Rollback

```bash
# Check rollout history
kubectl rollout history deployment/textnlp-api -n textnlp-prod

# Rollback to previous version
kubectl rollout undo deployment/textnlp-api -n textnlp-prod

# Rollback to specific revision
kubectl rollout undo deployment/textnlp-api --to-revision=3 -n textnlp-prod

# Monitor rollback status
kubectl rollout status deployment/textnlp-api -n textnlp-prod
```

### 2. Helm Rollback

```bash
# List release history
helm history textnlp -n textnlp-prod

# Rollback to previous release
helm rollback textnlp -n textnlp-prod

# Rollback to specific revision
helm rollback textnlp 3 -n textnlp-prod
```

### 3. Database Rollback

```bash
# Connect to database
kubectl exec -it postgres-0 -n textnlp-prod -- psql -U textnlp

# Check migration history
SELECT * FROM alembic_version;

# Rollback migration
kubectl exec -it api-pod -n textnlp-prod -- alembic downgrade -1
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Pod Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n textnlp-prod

# Check logs
kubectl logs <pod-name> -n textnlp-prod --previous

# Common fixes:
# - Check image pull secrets
# - Verify resource limits
# - Check liveness/readiness probes
# - Verify environment variables
```

#### 2. Database Connection Issues

```bash
# Test database connectivity
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- \
  psql -h postgres-service -U textnlp -d textnlp

# Check database logs
kubectl logs postgres-0 -n textnlp-prod

# Common fixes:
# - Verify connection string
# - Check network policies
# - Ensure database is running
# - Check credentials
```

#### 3. Performance Issues

```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n textnlp-prod

# Check HPA status
kubectl get hpa -n textnlp-prod

# Profile application
kubectl exec -it <pod-name> -n textnlp-prod -- python -m cProfile -o profile.out app.py
```

#### 4. SSL/TLS Issues

```bash
# Check certificate status
kubectl describe certificate textnlp-tls -n textnlp-prod

# Verify ingress configuration
kubectl describe ingress textnlp-ingress -n textnlp-prod

# Test SSL connection
openssl s_client -connect api.textnlp.io:443 -servername api.textnlp.io
```

### Debug Commands Cheatsheet

```bash
# Namespace issues
kubectl get all -n textnlp-prod
kubectl get events -n textnlp-prod --sort-by='.lastTimestamp'

# Service discovery
kubectl run -it --rm debug --image=nicolaka/netshoot --restart=Never -- sh
nslookup textnlp-api-service.textnlp-prod.svc.cluster.local

# Port forwarding for debugging
kubectl port-forward -n textnlp-prod svc/textnlp-api 8080:80

# Execute commands in pod
kubectl exec -it <pod-name> -n textnlp-prod -- /bin/bash

# Copy files from pod
kubectl cp textnlp-prod/<pod-name>:/app/logs/app.log ./app.log

# Resource usage analysis
kubectl describe nodes
kubectl describe pod <pod-name> -n textnlp-prod | grep -A 5 "Limits\|Requests"
```

## Post-Deployment Checklist

### Verification Steps

```bash
# 1. Health check endpoints
curl https://api.textnlp.io/health
curl https://api.textnlp.io/ready

# 2. API functionality
curl -X POST https://api.textnlp.io/v1/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test generation", "model": "gpt2"}'

# 3. Metrics endpoint
curl https://api.textnlp.io/metrics

# 4. Check logs
kubectl logs -f deployment/textnlp-api -n textnlp-prod

# 5. Monitor resource usage
watch kubectl top pods -n textnlp-prod

# 6. Verify autoscaling
kubectl get hpa -n textnlp-prod -w
```

### Performance Testing

```bash
# Load testing with k6
k6 run --vus 100 --duration 30s load-test.js

# Stress testing
siege -c 100 -t 60s https://api.textnlp.io/v1/generate

# Monitor during testing
kubectl get pods -n textnlp-prod -w
```

## Security Checklist

- [ ] All secrets are stored in secret management system
- [ ] Network policies are configured
- [ ] RBAC is properly set up
- [ ] Images are scanned for vulnerabilities
- [ ] SSL/TLS certificates are valid
- [ ] API authentication is enabled
- [ ] Rate limiting is configured
- [ ] Audit logging is enabled
- [ ] Backup procedures are tested
- [ ] Disaster recovery plan is documented

## Conclusion

This deployment workflow guide provides comprehensive instructions for deploying the TextNLP platform across various environments. Always test deployments in staging before production, maintain proper backups, and follow the rollback procedures when needed. For additional support, consult the architecture documentation and monitoring dashboards.