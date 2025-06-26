# Unified Cloud Deployment Plan for Inferloop Synthetic Data Platform

## Executive Summary

This document outlines a comprehensive plan to unify cloud deployment infrastructure across all Inferloop synthetic data services (tabular, textnlp, syndoc, etc.) into a common, reusable deployment framework. This approach will eliminate duplication, reduce maintenance overhead, ensure consistency, and enable efficient scaling across all services.

## Current State Analysis

### Existing Structure
```
inferloop-synthdata/
├── tabular/
│   ├── aws-infrastructure-design.md
│   ├── azure-infrastructure-design.md
│   ├── gcp-infrastructure-design.md
│   └── deploy/
├── textnlp/
│   ├── aws-infrastructure-design.md
│   ├── azure-infrastructure-design.md
│   ├── gcp-infrastructure-design.md
│   └── deploy/
└── syndoc/
    └── (deployment files)
```

### Problems with Current Approach
1. **Duplication**: Each service has its own cloud deployment configuration
2. **Inconsistency**: Different services may have different deployment patterns
3. **Maintenance Overhead**: Updates must be applied to multiple locations
4. **Resource Inefficiency**: Cannot share common infrastructure components
5. **Cost Inefficiency**: Duplicate resources across services

## Proposed Unified Architecture

### New Structure
```
inferloop-synthdata/
├── unified-cloud-deployment/
│   ├── docs/
│   │   ├── ARCHITECTURE.md
│   │   ├── DEPLOYMENT_GUIDE.md
│   │   └── MIGRATION_GUIDE.md
│   ├── infrastructure/
│   │   ├── terraform/
│   │   │   ├── modules/
│   │   │   ├── environments/
│   │   │   └── providers/
│   │   ├── kubernetes/
│   │   │   ├── base/
│   │   │   ├── services/
│   │   │   └── overlays/
│   │   └── helm/
│   │       ├── charts/
│   │       └── values/
│   ├── cicd/
│   │   ├── github-actions/
│   │   ├── gitlab-ci/
│   │   └── azure-devops/
│   ├── monitoring/
│   │   ├── prometheus/
│   │   ├── grafana/
│   │   └── alerts/
│   └── scripts/
│       ├── deploy.sh
│       ├── rollback.sh
│       └── health-check.sh
├── tabular/
│   └── deployment-config.yaml
├── textnlp/
│   └── deployment-config.yaml
└── syndoc/
    └── deployment-config.yaml
```

## Key Design Principles

### 1. Service Mesh Architecture
- **Istio/Linkerd** for inter-service communication
- Automatic service discovery
- Built-in load balancing and circuit breaking
- End-to-end encryption

### 2. Microservices Pattern
- Each service (tabular, textnlp, etc.) as independent microservice
- Shared infrastructure layer
- Service-specific configuration only

### 3. Infrastructure as Code (IaC)
- Terraform modules for cloud resources
- Kubernetes manifests for container orchestration
- Helm charts for application deployment
- GitOps for configuration management

### 4. Multi-Cloud Abstraction
- Provider-agnostic base modules
- Cloud-specific implementations
- Easy switching between clouds
- Hybrid cloud support

### 5. Shared Services
- Centralized authentication/authorization
- Common API gateway
- Shared monitoring and logging
- Unified secret management

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
1. Create unified repository structure
2. Design base Terraform modules
3. Set up Kubernetes base configurations
4. Create shared Helm chart templates

### Phase 2: Core Infrastructure (Weeks 3-4)
1. Implement cloud provider modules
2. Create service mesh configuration
3. Set up shared services (API Gateway, Auth)
4. Implement monitoring stack

### Phase 3: Service Migration (Weeks 5-6)
1. Migrate tabular service
2. Migrate textnlp service
3. Migrate other services
4. Validate inter-service communication

### Phase 4: Optimization (Weeks 7-8)
1. Performance tuning
2. Cost optimization
3. Security hardening
4. Documentation completion

## Technical Architecture

### 1. Infrastructure Layer

#### Shared Components
```yaml
# Common infrastructure for all services
SharedInfrastructure:
  Networking:
    - VPC/VNet with multiple subnets
    - Load balancers (Application/Network)
    - CDN for static content
    - Private endpoints for PaaS services
    
  Security:
    - WAF for API protection
    - Network security groups
    - Identity and access management
    - Key/Secret management service
    
  Data:
    - Shared database cluster (PostgreSQL)
    - Distributed cache (Redis cluster)
    - Object storage for large files
    - Message queue for async processing
    
  Compute:
    - Kubernetes cluster (EKS/AKS/GKE)
    - Serverless functions for event processing
    - GPU node pools for ML workloads
```

### 2. Application Layer

#### Service Configuration
```yaml
# Example: textnlp/deployment-config.yaml
apiVersion: v1
kind: ServiceConfig
metadata:
  name: textnlp
  namespace: inferloop
spec:
  service:
    type: api
    replicas: 3
    resources:
      cpu: "2"
      memory: "4Gi"
      gpu: "1"  # Only for ML services
    
  dependencies:
    - name: postgres
      type: database
      shared: true
    - name: redis
      type: cache
      shared: true
    - name: model-storage
      type: storage
      path: /models/textnlp
    
  networking:
    ingress:
      enabled: true
      path: /api/textnlp
      rateLimit: 1000
    serviceMesh:
      enabled: true
      retries: 3
      timeout: 30s
    
  monitoring:
    metrics:
      enabled: true
      path: /metrics
    tracing:
      enabled: true
      samplingRate: 0.1
```

### 3. Service Mesh Design

```yaml
# Istio service mesh configuration
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: inferloop-services
spec:
  hosts:
  - api.inferloop.io
  http:
  - match:
    - uri:
        prefix: /api/tabular
    route:
    - destination:
        host: tabular-service
        port:
          number: 8000
      weight: 100
  - match:
    - uri:
        prefix: /api/textnlp
    route:
    - destination:
        host: textnlp-service
        port:
          number: 8000
      weight: 100
```

### 4. Shared API Gateway

```yaml
# Kong/AWS API Gateway configuration
Routes:
  - name: tabular-route
    paths: ["/api/tabular/*"]
    service: tabular-service
    plugins:
      - rate-limiting:
          minute: 1000
      - jwt-auth:
          enabled: true
      - cors:
          origins: ["https://app.inferloop.io"]
          
  - name: textnlp-route
    paths: ["/api/textnlp/*"]
    service: textnlp-service
    plugins:
      - rate-limiting:
          minute: 500
      - jwt-auth:
          enabled: true
      - request-transformer:
          add:
            headers:
              X-Model-Type: "nlp"
```

## Cloud-Specific Implementations

### AWS Implementation
```hcl
# terraform/providers/aws/main.tf
module "inferloop_platform" {
  source = "../../modules/base"
  
  providers = {
    cloud = aws
  }
  
  project_name = "inferloop"
  environment  = var.environment
  
  services = {
    tabular = {
      enabled = true
      config  = file("${path.root}/../../../tabular/deployment-config.yaml")
    }
    textnlp = {
      enabled = true
      config  = file("${path.root}/../../../textnlp/deployment-config.yaml")
    }
  }
  
  # AWS-specific configurations
  eks_config = {
    cluster_version = "1.28"
    node_groups = {
      general = {
        instance_types = ["m5.xlarge"]
        min_size      = 3
        max_size      = 10
      }
      gpu = {
        instance_types = ["g4dn.xlarge"]
        min_size      = 1
        max_size      = 5
        taints = [{
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }]
      }
    }
  }
}
```

### Azure Implementation
```hcl
# terraform/providers/azure/main.tf
module "inferloop_platform" {
  source = "../../modules/base"
  
  providers = {
    cloud = azurerm
  }
  
  project_name = "inferloop"
  environment  = var.environment
  
  services = {
    tabular = {
      enabled = true
      config  = file("${path.root}/../../../tabular/deployment-config.yaml")
    }
    textnlp = {
      enabled = true
      config  = file("${path.root}/../../../textnlp/deployment-config.yaml")
    }
  }
  
  # Azure-specific configurations
  aks_config = {
    kubernetes_version = "1.28"
    default_node_pool = {
      vm_size    = "Standard_D4s_v3"
      node_count = 3
      enable_auto_scaling = true
      min_count  = 3
      max_count  = 10
    }
    additional_node_pools = [{
      name       = "gpu"
      vm_size    = "Standard_NC6s_v3"
      node_count = 1
      node_taints = ["nvidia.com/gpu=true:NoSchedule"]
    }]
  }
}
```

### GCP Implementation
```hcl
# terraform/providers/gcp/main.tf
module "inferloop_platform" {
  source = "../../modules/base"
  
  providers = {
    cloud = google
  }
  
  project_name = "inferloop"
  environment  = var.environment
  
  services = {
    tabular = {
      enabled = true
      config  = file("${path.root}/../../../tabular/deployment-config.yaml")
    }
    textnlp = {
      enabled = true
      config  = file("${path.root}/../../../textnlp/deployment-config.yaml")
    }
  }
  
  # GCP-specific configurations
  gke_config = {
    release_channel = "STABLE"
    cluster_autoscaling = {
      enabled = true
      resource_limits = {
        cpu = {
          minimum = 10
          maximum = 100
        }
        memory = {
          minimum = 40
          maximum = 400
        }
      }
    }
    node_pools = [{
      name         = "default-pool"
      machine_type = "n1-standard-4"
      autoscaling = {
        min_node_count = 3
        max_node_count = 10
      }
    }, {
      name         = "gpu-pool"
      machine_type = "n1-standard-4"
      accelerator = {
        type  = "nvidia-tesla-t4"
        count = 1
      }
    }]
  }
}
```

## Monitoring and Observability

### Unified Monitoring Stack
```yaml
# monitoring/prometheus/values.yaml
prometheus:
  serverFiles:
    prometheus.yml:
      global:
        scrape_interval: 15s
      scrape_configs:
        - job_name: 'inferloop-services'
          kubernetes_sd_configs:
            - role: pod
          relabel_configs:
            - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
              action: keep
              regex: true
            - source_labels: [__meta_kubernetes_pod_label_app]
              action: replace
              target_label: service
              
# monitoring/grafana/dashboards/unified-dashboard.json
{
  "dashboard": {
    "title": "Inferloop Platform Overview",
    "panels": [
      {
        "title": "Service Health",
        "targets": [
          {
            "expr": "up{job='inferloop-services'}"
          }
        ]
      },
      {
        "title": "Request Rate by Service",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (service)"
          }
        ]
      },
      {
        "title": "Resource Usage",
        "targets": [
          {
            "expr": "sum(container_memory_usage_bytes) by (pod_name)"
          }
        ]
      }
    ]
  }
}
```

## CI/CD Pipeline

### Unified Deployment Pipeline
```yaml
# .github/workflows/unified-deploy.yml
name: Unified Deployment Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      services: ${{ steps.detect.outputs.services }}
      infrastructure: ${{ steps.detect.outputs.infrastructure }}
    steps:
      - uses: actions/checkout@v3
      - id: detect
        run: |
          # Detect which services changed
          SERVICES=$(./scripts/detect-changes.sh services)
          INFRA=$(./scripts/detect-changes.sh infrastructure)
          echo "services=$SERVICES" >> $GITHUB_OUTPUT
          echo "infrastructure=$INFRA" >> $GITHUB_OUTPUT

  test-services:
    needs: detect-changes
    if: needs.detect-changes.outputs.services != '[]'
    strategy:
      matrix:
        service: ${{ fromJson(needs.detect-changes.outputs.services) }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Test ${{ matrix.service }}
        run: |
          cd ${{ matrix.service }}
          make test

  deploy-infrastructure:
    needs: [detect-changes, test-services]
    if: needs.detect-changes.outputs.infrastructure == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy Infrastructure
        run: |
          cd unified-cloud-deployment/infrastructure/terraform
          terraform init
          terraform apply -auto-approve

  deploy-services:
    needs: [deploy-infrastructure, test-services]
    strategy:
      matrix:
        service: ${{ fromJson(needs.detect-changes.outputs.services) }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy ${{ matrix.service }}
        run: |
          ./unified-cloud-deployment/scripts/deploy.sh ${{ matrix.service }}
```

## Migration Strategy

### Step 1: Prepare Unified Infrastructure
```bash
# Initialize unified deployment
cd unified-cloud-deployment
terraform init
terraform plan -out=tfplan

# Review and apply
terraform apply tfplan
```

### Step 2: Migrate Services One by One
```bash
# Example: Migrate tabular service
./scripts/migrate-service.sh tabular

# The script will:
# 1. Extract service-specific config
# 2. Create deployment-config.yaml
# 3. Deploy to unified infrastructure
# 4. Validate service health
# 5. Switch traffic to new deployment
```

### Step 3: Decommission Old Infrastructure
```bash
# After all services migrated and validated
cd tabular/deploy
terraform destroy

cd textnlp/deploy
terraform destroy
```

## Cost Optimization Strategies

### 1. Resource Sharing
- Shared Kubernetes cluster reduces overhead
- Common monitoring stack
- Shared databases with logical separation
- Centralized logging and metrics storage

### 2. Auto-scaling
- Cluster autoscaler for nodes
- HPA for pods based on metrics
- Scheduled scaling for predictable loads
- Spot/Preemptible instances for non-critical workloads

### 3. Multi-tenancy
- Namespace isolation for services
- Resource quotas per service
- Priority classes for critical services
- Pod disruption budgets

### 4. Cost Monitoring
```yaml
# Cost allocation tags
CommonTags:
  Project: "inferloop"
  Environment: "${environment}"
  ManagedBy: "terraform"
  CostCenter: "engineering"
  
ServiceTags:
  Service: "${service_name}"
  Team: "${team_name}"
  Owner: "${owner_email}"
```

## Security Considerations

### 1. Network Security
- Service mesh for zero-trust networking
- Network policies for pod-to-pod communication
- Private endpoints for PaaS services
- WAF for external APIs

### 2. Identity and Access
- Workload identity for cloud resources
- RBAC for Kubernetes access
- Service accounts with minimal permissions
- Regular credential rotation

### 3. Data Security
- Encryption at rest and in transit
- Separate encryption keys per service
- Data residency compliance
- Regular security scanning

## Benefits of Unified Approach

### 1. Operational Benefits
- **Single pane of glass** for monitoring
- **Unified logging** across all services
- **Consistent deployment** process
- **Simplified troubleshooting**

### 2. Development Benefits
- **Shared libraries** and utilities
- **Consistent patterns** across services
- **Faster onboarding** for new services
- **Reduced boilerplate** code

### 3. Cost Benefits
- **30-40% reduction** in infrastructure costs
- **Shared resources** utilization
- **Bulk pricing** for cloud services
- **Reduced operational overhead**

### 4. Scalability Benefits
- **Horizontal scaling** across services
- **Shared capacity** planning
- **Efficient resource** utilization
- **Global deployment** capabilities

## Success Metrics

1. **Infrastructure Cost**: 35% reduction within 3 months
2. **Deployment Time**: From hours to minutes
3. **Service Uptime**: 99.95% SLA across all services
4. **Development Velocity**: 2x faster service deployment
5. **Operational Overhead**: 50% reduction in maintenance time

## Conclusion

The unified cloud deployment architecture provides a robust, scalable, and cost-effective foundation for all Inferloop synthetic data services. By consolidating infrastructure while maintaining service independence, we achieve the best of both worlds: operational efficiency and development flexibility.