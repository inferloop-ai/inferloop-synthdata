# Unified Cloud Deployment Implementation Guide

## Overview

This document provides step-by-step implementation instructions for deploying the unified cloud infrastructure that serves all Inferloop synthetic data services. It covers the complete process from initial setup to production deployment across AWS, Azure, and GCP.

## Prerequisites

### Required Tools
```bash
# Infrastructure tools
terraform >= 1.5.0
kubectl >= 1.28.0
helm >= 3.12.0
kustomize >= 5.0.0

# Cloud CLIs
aws-cli >= 2.13.0
az >= 2.53.0
gcloud >= 450.0.0

# Service mesh
istioctl >= 1.19.0

# GitOps
argocd >= 2.8.0
flux >= 2.1.0

# Monitoring
prometheus-operator >= 0.68.0
grafana >= 10.0.0
```

### Access Requirements
- Admin access to cloud provider accounts
- Domain name registered and DNS control
- SSL certificates (or ability to use Let's Encrypt)
- Container registry access

## Directory Structure Setup

```bash
# Create unified deployment structure
mkdir -p unified-cloud-deployment/{infrastructure,kubernetes,monitoring,scripts,docs}

# Infrastructure modules
mkdir -p infrastructure/{terraform,ansible,packer}
mkdir -p infrastructure/terraform/{modules,providers,environments}
mkdir -p infrastructure/terraform/modules/{networking,compute,storage,security,database,monitoring}

# Kubernetes configurations
mkdir -p kubernetes/{base,overlays,services,operators}
mkdir -p kubernetes/base/{namespace,rbac,network-policies,storage}
mkdir -p kubernetes/overlays/{dev,staging,production}
mkdir -p kubernetes/services/{tabular,textnlp,syndoc,shared}

# Helm charts
mkdir -p helm/{charts,values}
mkdir -p helm/charts/{inferloop-base,inferloop-service}

# Monitoring stack
mkdir -p monitoring/{prometheus,grafana,alertmanager,loki}

# CI/CD configurations
mkdir -p cicd/{github-actions,gitlab-ci,jenkins}
```

## Phase 1: Base Infrastructure Setup

### 1.1 Create Terraform Base Modules

```hcl
# infrastructure/terraform/modules/base/variables.tf
variable "project_name" {
  description = "Project name for all resources"
  type        = string
  default     = "inferloop"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
}

variable "region" {
  description = "Primary region for deployment"
  type        = string
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
}

variable "services" {
  description = "Map of services to deploy"
  type = map(object({
    enabled = bool
    config  = string
  }))
}

# infrastructure/terraform/modules/base/main.tf
locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    CreatedAt   = timestamp()
  }
}

# Networking module
module "networking" {
  source = "../networking"
  
  project_name       = var.project_name
  environment        = var.environment
  vpc_cidr           = "10.0.0.0/16"
  availability_zones = var.availability_zones
  
  tags = local.common_tags
}

# Kubernetes cluster
module "kubernetes" {
  source = "../compute/kubernetes"
  
  project_name = var.project_name
  environment  = var.environment
  
  vpc_id     = module.networking.vpc_id
  subnet_ids = module.networking.private_subnet_ids
  
  cluster_version = "1.28"
  
  node_groups = {
    system = {
      instance_types = ["t3.large"]
      min_size       = 3
      max_size       = 6
      desired_size   = 3
    }
    
    general = {
      instance_types = ["t3.xlarge"]
      min_size       = 3
      max_size       = 20
      desired_size   = 5
    }
    
    gpu = {
      instance_types = ["g4dn.xlarge"]
      min_size       = 0
      max_size       = 10
      desired_size   = 2
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "present"
        effect = "NO_SCHEDULE"
      }]
    }
  }
  
  tags = local.common_tags
}

# Database cluster
module "database" {
  source = "../database"
  
  project_name = var.project_name
  environment  = var.environment
  
  vpc_id     = module.networking.vpc_id
  subnet_ids = module.networking.database_subnet_ids
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.environment == "production" ? "db.r6g.xlarge" : "db.t3.medium"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  
  backup_retention_period = var.environment == "production" ? 30 : 7
  
  tags = local.common_tags
}

# Redis cluster
module "cache" {
  source = "../cache"
  
  project_name = var.project_name
  environment  = var.environment
  
  vpc_id     = module.networking.vpc_id
  subnet_ids = module.networking.private_subnet_ids
  
  node_type               = var.environment == "production" ? "cache.r6g.large" : "cache.t3.micro"
  number_cache_clusters   = var.environment == "production" ? 3 : 1
  automatic_failover      = var.environment == "production" ? true : false
  multi_az                = var.environment == "production" ? true : false
  
  tags = local.common_tags
}

# Object storage
module "storage" {
  source = "../storage"
  
  project_name = var.project_name
  environment  = var.environment
  
  buckets = {
    data = {
      versioning = true
      lifecycle_rules = [{
        id      = "archive-old-data"
        enabled = true
        transitions = [{
          days          = 30
          storage_class = "STANDARD_IA"
        }, {
          days          = 90
          storage_class = "GLACIER"
        }]
      }]
    }
    
    models = {
      versioning = true
      cors_rules = [{
        allowed_methods = ["GET", "HEAD"]
        allowed_origins = ["*"]
        allowed_headers = ["*"]
        max_age_seconds = 3000
      }]
    }
    
    backups = {
      versioning = true
      lifecycle_rules = [{
        id      = "expire-old-backups"
        enabled = true
        expiration = {
          days = 90
        }
      }]
    }
  }
  
  tags = local.common_tags
}
```

### 1.2 Cloud Provider Implementations

#### AWS Implementation
```hcl
# infrastructure/terraform/providers/aws/main.tf
terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket = "inferloop-terraform-state"
    key    = "unified-deployment/terraform.tfstate"
    region = "us-east-1"
    
    dynamodb_table = "terraform-state-lock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "inferloop"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# Use base module with AWS-specific overrides
module "inferloop_platform" {
  source = "../../modules/base"
  
  project_name = "inferloop"
  environment  = var.environment
  region       = var.aws_region
  
  availability_zones = data.aws_availability_zones.available.names
  
  services = var.services
}

# AWS-specific resources
resource "aws_eks_addon" "vpc_cni" {
  cluster_name = module.inferloop_platform.kubernetes_cluster_name
  addon_name   = "vpc-cni"
  addon_version = "v1.14.1-eksbuild.1"
}

resource "aws_eks_addon" "ebs_csi_driver" {
  cluster_name = module.inferloop_platform.kubernetes_cluster_name
  addon_name   = "aws-ebs-csi-driver"
  addon_version = "v1.23.1-eksbuild.1"
}
```

#### Azure Implementation
```hcl
# infrastructure/terraform/providers/azure/main.tf
terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
  
  backend "azurerm" {
    resource_group_name  = "inferloop-terraform"
    storage_account_name = "inferlooptr"
    container_name       = "tfstate"
    key                  = "unified-deployment.tfstate"
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = true
    }
  }
}

# Resource group
resource "azurerm_resource_group" "main" {
  name     = "rg-inferloop-${var.environment}"
  location = var.azure_region
  
  tags = {
    Project     = "inferloop"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# Use base module with Azure-specific configurations
module "inferloop_platform" {
  source = "../../modules/base"
  
  project_name = "inferloop"
  environment  = var.environment
  region       = var.azure_region
  
  availability_zones = ["1", "2", "3"]
  
  services = var.services
  
  # Azure-specific overrides
  providers = {
    cloud = azurerm
  }
}
```

#### GCP Implementation
```hcl
# infrastructure/terraform/providers/gcp/main.tf
terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  
  backend "gcs" {
    bucket = "inferloop-terraform-state"
    prefix = "unified-deployment"
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# Use base module with GCP-specific configurations
module "inferloop_platform" {
  source = "../../modules/base"
  
  project_name = "inferloop"
  environment  = var.environment
  region       = var.gcp_region
  
  availability_zones = data.google_compute_zones.available.names
  
  services = var.services
  
  # GCP-specific overrides
  providers = {
    cloud = google
  }
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "container.googleapis.com",
    "servicemesh.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudtrace.googleapis.com",
  ])
  
  service = each.key
  disable_on_destroy = false
}
```

## Phase 2: Kubernetes Setup

### 2.1 Base Kubernetes Configuration

```yaml
# kubernetes/base/namespace/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: inferloop
  labels:
    name: inferloop
    istio-injection: enabled
---
apiVersion: v1
kind: Namespace
metadata:
  name: inferloop-monitoring
  labels:
    name: inferloop-monitoring
---
apiVersion: v1
kind: Namespace
metadata:
  name: inferloop-ingress
  labels:
    name: inferloop-ingress
```

```yaml
# kubernetes/base/rbac/service-accounts.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: inferloop-base
  namespace: inferloop
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT:role/inferloop-base-role
    azure.workload.identity/client-id: "CLIENT_ID"
    iam.gke.io/gcp-service-account: inferloop-base@PROJECT.iam.gserviceaccount.com
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: inferloop-base
rules:
- apiGroups: [""]
  resources: ["secrets", "configmaps"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: inferloop-base
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: inferloop-base
subjects:
- kind: ServiceAccount
  name: inferloop-base
  namespace: inferloop
```

### 2.2 Service Mesh Installation

```bash
# Install Istio
curl -L https://istio.io/downloadIstio | sh -
cd istio-1.19.0
export PATH=$PWD/bin:$PATH

# Install Istio with custom configuration
istioctl install --set values.pilot.env.PILOT_ENABLE_WORKLOAD_ENTRY_AUTOREGISTRATION=true \
  --set values.global.proxy.resources.requests.cpu=100m \
  --set values.global.proxy.resources.requests.memory=128Mi \
  --set values.global.proxy.resources.limits.cpu=200m \
  --set values.global.proxy.resources.limits.memory=256Mi
```

```yaml
# kubernetes/base/istio/gateway.yaml
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: inferloop-gateway
  namespace: inferloop
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "api.inferloop.io"
    tls:
      httpsRedirect: true
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: SIMPLE
      credentialName: inferloop-tls
    hosts:
    - "api.inferloop.io"
```

### 2.3 Shared Services Deployment

```yaml
# kubernetes/services/shared/auth-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: auth-service
  namespace: inferloop
spec:
  replicas: 3
  selector:
    matchLabels:
      app: auth-service
  template:
    metadata:
      labels:
        app: auth-service
        version: v1
    spec:
      serviceAccountName: auth-service-sa
      containers:
      - name: auth
        image: inferloop/auth-service:latest
        ports:
        - containerPort: 8080
        env:
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: auth-secrets
              key: jwt-secret
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: shared-secrets
              key: database-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: auth-service
  namespace: inferloop
spec:
  selector:
    app: auth-service
  ports:
  - port: 80
    targetPort: 8080
    name: http
```

## Phase 3: Service Migration

### 3.1 Service Configuration Template

```yaml
# kubernetes/services/textnlp/deployment-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: textnlp-config
  namespace: inferloop
data:
  config.yaml: |
    service:
      name: textnlp
      version: v1.0.0
      
    server:
      port: 8000
      workers: 4
      
    database:
      schema: textnlp
      pool_size: 20
      
    cache:
      prefix: textnlp:
      ttl: 3600
      
    models:
      default: gpt2
      available:
        - gpt2
        - gpt-j
        - llama
      
    monitoring:
      metrics_path: /metrics
      metrics_port: 9090
```

### 3.2 Service Deployment

```yaml
# kubernetes/services/textnlp/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: textnlp
  namespace: inferloop
  labels:
    app: textnlp
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: textnlp
  template:
    metadata:
      labels:
        app: textnlp
        version: v1.0.0
      annotations:
        sidecar.istio.io/inject: "true"
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: textnlp-sa
      containers:
      - name: textnlp
        image: inferloop/textnlp:v1.0.0
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: SERVICE_NAME
          value: textnlp
        - name: ENVIRONMENT
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: shared-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: shared-secrets
              key: redis-url
        - name: CONFIG_PATH
          value: /etc/config/config.yaml
        volumeMounts:
        - name: config
          mountPath: /etc/config
        - name: models
          mountPath: /models
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
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
          name: textnlp-config
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

### 3.3 Service Mesh Configuration

```yaml
# kubernetes/services/textnlp/istio-config.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: textnlp
  namespace: inferloop
spec:
  hosts:
  - textnlp
  - api.inferloop.io
  gateways:
  - inferloop-gateway
  - mesh
  http:
  - match:
    - uri:
        prefix: /api/textnlp
    - headers:
        x-service:
          exact: textnlp
    route:
    - destination:
        host: textnlp
        port:
          number: 80
      weight: 100
    timeout: 30s
    retries:
      attempts: 3
      perTryTimeout: 10s
      retryOn: 5xx,reset,connect-failure,refused-stream
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: textnlp
  namespace: inferloop
spec:
  host: textnlp
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 100
        http2MaxRequests: 100
        maxRequestsPerConnection: 2
    loadBalancer:
      simple: LEAST_REQUEST
    outlierDetection:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
      minHealthPercent: 50
```

## Phase 4: Monitoring Stack

### 4.1 Prometheus Configuration

```yaml
# monitoring/prometheus/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: inferloop-monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
      external_labels:
        cluster: 'inferloop-prod'
        
    rule_files:
      - /etc/prometheus/rules/*.yml
      
    alerting:
      alertmanagers:
      - static_configs:
        - targets:
          - alertmanager:9093
          
    scrape_configs:
      - job_name: 'kubernetes-apiservers'
        kubernetes_sd_configs:
        - role: endpoints
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
        - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
          action: keep
          regex: default;kubernetes;https
          
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
        - role: pod
        relabel_configs:
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
          action: keep
          regex: true
        - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
          action: replace
          target_label: __metrics_path__
          regex: (.+)
        - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
          action: replace
          regex: ([^:]+)(?::\d+)?;(\d+)
          replacement: $1:$2
          target_label: __address__
        - action: labelmap
          regex: __meta_kubernetes_pod_label_(.+)
        - source_labels: [__meta_kubernetes_namespace]
          action: replace
          target_label: kubernetes_namespace
        - source_labels: [__meta_kubernetes_pod_name]
          action: replace
          target_label: kubernetes_pod_name
```

### 4.2 Grafana Dashboards

```json
{
  "dashboard": {
    "id": null,
    "uid": "inferloop-overview",
    "title": "Inferloop Platform Overview",
    "tags": ["inferloop", "platform"],
    "timezone": "browser",
    "panels": [
      {
        "datasource": "Prometheus",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "id": 1,
        "title": "Service Request Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (service)",
            "legendFormat": "{{service}}"
          }
        ],
        "type": "graph"
      },
      {
        "datasource": "Prometheus",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "id": 2,
        "title": "Service Latency (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le))",
            "legendFormat": "{{service}}"
          }
        ],
        "type": "graph"
      },
      {
        "datasource": "Prometheus",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "id": 3,
        "title": "Error Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) by (service) / sum(rate(http_requests_total[5m])) by (service)",
            "legendFormat": "{{service}}"
          }
        ],
        "type": "graph"
      },
      {
        "datasource": "Prometheus",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "id": 4,
        "title": "Resource Usage",
        "targets": [
          {
            "expr": "sum(container_memory_usage_bytes{namespace=\"inferloop\"}) by (pod)",
            "legendFormat": "{{pod}} - Memory"
          },
          {
            "expr": "sum(rate(container_cpu_usage_seconds_total{namespace=\"inferloop\"}[5m])) by (pod) * 1000",
            "legendFormat": "{{pod}} - CPU (millicores)"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

### 4.3 Alerting Rules

```yaml
# monitoring/prometheus/alerts.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-alerts
  namespace: inferloop-monitoring
data:
  alerts.yml: |
    groups:
    - name: inferloop
      interval: 30s
      rules:
      - alert: ServiceDown
        expr: up{job="kubernetes-pods",namespace="inferloop"} == 0
        for: 2m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "Service {{ $labels.pod }} is down"
          description: "{{ $labels.pod }} in namespace {{ $labels.namespace }} has been down for more than 2 minutes."
          
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5..",namespace="inferloop"}[5m])) by (service)
          /
          sum(rate(http_requests_total{namespace="inferloop"}[5m])) by (service)
          > 0.05
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High error rate for {{ $labels.service }}"
          description: "{{ $labels.service }} has error rate above 5% (current: {{ $value | humanizePercentage }})"
          
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket{namespace="inferloop"}[5m])) by (service, le)
          ) > 1
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High latency for {{ $labels.service }}"
          description: "{{ $labels.service }} p95 latency is above 1s (current: {{ $value }}s)"
          
      - alert: PodMemoryUsage
        expr: |
          container_memory_usage_bytes{namespace="inferloop"}
          /
          container_spec_memory_limit_bytes{namespace="inferloop"}
          > 0.8
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High memory usage for {{ $labels.pod }}"
          description: "{{ $labels.pod }} memory usage is above 80% (current: {{ $value | humanizePercentage }})"
```

## Phase 5: CI/CD Pipeline

### 5.1 GitHub Actions Workflow

```yaml
# .github/workflows/unified-deploy.yml
name: Unified Deployment

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  CLUSTER_NAME: inferloop-prod

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.set-matrix.outputs.matrix }}
      infrastructure: ${{ steps.changes.outputs.infrastructure }}
    steps:
    - uses: actions/checkout@v3
    
    - uses: dorny/paths-filter@v2
      id: changes
      with:
        filters: |
          infrastructure:
            - 'unified-cloud-deployment/infrastructure/**'
          tabular:
            - 'tabular/**'
          textnlp:
            - 'textnlp/**'
          syndoc:
            - 'syndoc/**'
            
    - id: set-matrix
      run: |
        SERVICES=()
        if [[ "${{ steps.changes.outputs.tabular }}" == "true" ]]; then
          SERVICES+=("tabular")
        fi
        if [[ "${{ steps.changes.outputs.textnlp }}" == "true" ]]; then
          SERVICES+=("textnlp")
        fi
        if [[ "${{ steps.changes.outputs.syndoc }}" == "true" ]]; then
          SERVICES+=("syndoc")
        fi
        
        MATRIX=$(printf '%s\n' "${SERVICES[@]}" | jq -R . | jq -s -c .)
        echo "matrix={\"service\":$MATRIX}" >> $GITHUB_OUTPUT

  test:
    needs: detect-changes
    if: needs.detect-changes.outputs.matrix != '{"service":[]}'
    runs-on: ubuntu-latest
    strategy:
      matrix: ${{ fromJson(needs.detect-changes.outputs.matrix) }}
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Test ${{ matrix.service }}
      run: |
        cd ${{ matrix.service }}
        pip install -e ".[test]"
        pytest --cov --cov-report=xml
        
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./${{ matrix.service }}/coverage.xml
        flags: ${{ matrix.service }}

  build:
    needs: [detect-changes, test]
    if: needs.detect-changes.outputs.matrix != '{"service":[]}'
    runs-on: ubuntu-latest
    strategy:
      matrix: ${{ fromJson(needs.detect-changes.outputs.matrix) }}
    permissions:
      contents: read
      packages: write
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
      
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Build and push ${{ matrix.service }}
      uses: docker/build-push-action@v4
      with:
        context: ./${{ matrix.service }}
        push: ${{ github.event_name != 'pull_request' }}
        tags: |
          ${{ env.REGISTRY }}/${{ github.repository }}/${{ matrix.service }}:latest
          ${{ env.REGISTRY }}/${{ github.repository }}/${{ matrix.service }}:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-infrastructure:
    needs: detect-changes
    if: needs.detect-changes.outputs.infrastructure == 'true' && github.event_name != 'pull_request'
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
        
    - name: Setup Terraform
      uses: hashicorp/setup-terraform@v2
      
    - name: Terraform Apply
      run: |
        cd unified-cloud-deployment/infrastructure/terraform/providers/aws
        terraform init
        terraform apply -auto-approve

  deploy-services:
    needs: [detect-changes, build]
    if: needs.detect-changes.outputs.matrix != '{"service":[]}'
    runs-on: ubuntu-latest
    strategy:
      matrix: ${{ fromJson(needs.detect-changes.outputs.matrix) }}
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure kubectl
      run: |
        aws eks update-kubeconfig --name ${{ env.CLUSTER_NAME }} --region us-east-1
        
    - name: Deploy ${{ matrix.service }}
      run: |
        kubectl set image deployment/${{ matrix.service }} \
          ${{ matrix.service }}=${{ env.REGISTRY }}/${{ github.repository }}/${{ matrix.service }}:${{ github.sha }} \
          -n inferloop
          
        kubectl rollout status deployment/${{ matrix.service }} -n inferloop
        
    - name: Run smoke tests
      run: |
        ./unified-cloud-deployment/scripts/smoke-test.sh ${{ matrix.service }}
```

## Phase 6: Migration Execution

### 6.1 Migration Script

```bash
#!/bin/bash
# unified-cloud-deployment/scripts/migrate-service.sh

set -euo pipefail

SERVICE_NAME=$1
ENVIRONMENT=${2:-production}

echo "Starting migration for service: $SERVICE_NAME"

# Step 1: Validate service directory exists
if [ ! -d "$SERVICE_NAME" ]; then
    echo "Error: Service directory $SERVICE_NAME not found"
    exit 1
fi

# Step 2: Generate deployment config
echo "Generating deployment configuration..."
cat > "$SERVICE_NAME/deployment-config.yaml" <<EOF
apiVersion: v1
kind: ServiceConfig
metadata:
  name: $SERVICE_NAME
  namespace: inferloop
spec:
  service:
    type: api
    replicas: 3
    resources:
      cpu: "2"
      memory: "4Gi"
  dependencies:
    - name: postgres
      type: database
      shared: true
    - name: redis
      type: cache
      shared: true
  networking:
    ingress:
      enabled: true
      path: /api/$SERVICE_NAME
EOF

# Step 3: Build and push Docker image
echo "Building Docker image..."
docker build -t inferloop/$SERVICE_NAME:migration ./$SERVICE_NAME
docker tag inferloop/$SERVICE_NAME:migration $REGISTRY/inferloop/$SERVICE_NAME:migration
docker push $REGISTRY/inferloop/$SERVICE_NAME:migration

# Step 4: Deploy to unified infrastructure
echo "Deploying to unified infrastructure..."
kubectl apply -f unified-cloud-deployment/kubernetes/services/$SERVICE_NAME/

# Step 5: Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/$SERVICE_NAME -n inferloop

# Step 6: Run health checks
echo "Running health checks..."
HEALTH_URL="https://api.inferloop.io/api/$SERVICE_NAME/health"
for i in {1..30}; do
    if curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL | grep -q "200"; then
        echo "Health check passed"
        break
    fi
    echo "Waiting for service to be healthy... ($i/30)"
    sleep 10
done

# Step 7: Switch traffic
echo "Switching traffic to new deployment..."
kubectl patch virtualservice $SERVICE_NAME -n inferloop --type merge -p '
{
  "spec": {
    "http": [{
      "route": [{
        "destination": {
          "host": "'$SERVICE_NAME'",
          "subset": "migration"
        },
        "weight": 100
      }]
    }]
  }
}'

echo "Migration completed successfully for $SERVICE_NAME"
```

### 6.2 Rollback Procedure

```bash
#!/bin/bash
# unified-cloud-deployment/scripts/rollback.sh

set -euo pipefail

SERVICE_NAME=$1
PREVIOUS_VERSION=$2

echo "Starting rollback for service: $SERVICE_NAME to version: $PREVIOUS_VERSION"

# Step 1: Update deployment
kubectl set image deployment/$SERVICE_NAME \
    $SERVICE_NAME=$REGISTRY/inferloop/$SERVICE_NAME:$PREVIOUS_VERSION \
    -n inferloop

# Step 2: Wait for rollout
kubectl rollout status deployment/$SERVICE_NAME -n inferloop

# Step 3: Verify health
./unified-cloud-deployment/scripts/health-check.sh $SERVICE_NAME

echo "Rollback completed successfully"
```

## Production Checklist

### Pre-deployment
- [ ] All tests passing in CI/CD pipeline
- [ ] Security scan completed
- [ ] Performance testing completed
- [ ] Documentation updated
- [ ] Runbooks prepared
- [ ] Backup verified

### Deployment
- [ ] Infrastructure provisioned
- [ ] Kubernetes cluster ready
- [ ] Service mesh configured
- [ ] Monitoring stack deployed
- [ ] Services migrated
- [ ] Health checks passing

### Post-deployment
- [ ] Traffic switched to new infrastructure
- [ ] Old infrastructure decommissioned
- [ ] Costs optimized
- [ ] Alerts configured
- [ ] Team trained
- [ ] Post-mortem completed

## Conclusion

This implementation guide provides a complete path from initial setup to production deployment of the unified cloud infrastructure. Follow the phases sequentially, validate each step, and maintain proper documentation throughout the process.