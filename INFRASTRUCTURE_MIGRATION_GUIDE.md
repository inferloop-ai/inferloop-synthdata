# Infrastructure Migration Guide - From Module-Specific to Unified

## Overview

This guide outlines the migration process for refactoring module-specific infrastructure code (tabular, textnlp, etc.) to use the unified cloud deployment infrastructure. The migration will eliminate code duplication, improve maintainability, and enable consistent deployment across all services.

## Current State Analysis

### What Needs to Change in Tabular (and other modules)

1. **Remove Cloud-Specific Code**
   - `tabular/deploy/` directory → Move to `unified-cloud-deployment/`
   - `tabular/inferloop-infra/` → Consolidate with unified infrastructure
   - Direct cloud SDK calls → Use unified provider interfaces

2. **Centralize Common Services**
   - `tabular/api/auth/` → Use unified auth service
   - `tabular/api/middleware/` → Use unified middleware
   - Module-specific monitoring → Use unified monitoring stack

3. **Standardize Configuration**
   - Hardcoded values → Environment-based configuration
   - Service-specific configs → Unified ConfigMaps
   - Local secrets → Centralized secret management

## Migration Steps

### Step 1: Create Service Adapter Layer

First, create an adapter layer in each module to interface with the unified infrastructure:

```python
# tabular/infrastructure/adapter.py
from typing import Dict, Any, Optional
from unified_cloud_deployment.core import ServiceConfig, ServiceAdapter

class TabularServiceAdapter(ServiceAdapter):
    """Adapter to connect tabular service with unified infrastructure"""
    
    SERVICE_NAME = "tabular"
    SERVICE_TYPE = "api"
    
    def get_service_config(self) -> ServiceConfig:
        """Return service-specific configuration"""
        return ServiceConfig(
            name=self.SERVICE_NAME,
            type=self.SERVICE_TYPE,
            image="inferloop/tabular:latest",
            port=8000,
            health_check_path="/health",
            ready_check_path="/ready",
            resources={
                "requests": {"cpu": "1", "memory": "2Gi"},
                "limits": {"cpu": "2", "memory": "4Gi"}
            },
            environment_vars={
                "SERVICE_NAME": self.SERVICE_NAME,
                "LOG_LEVEL": "INFO"
            },
            dependencies=["postgres", "redis", "auth-service"]
        )
    
    def get_scaling_config(self) -> Dict[str, Any]:
        """Return auto-scaling configuration"""
        return {
            "min_replicas": 2,
            "max_replicas": 100,
            "target_cpu_utilization": 70,
            "target_memory_utilization": 80,
            "scale_down_stabilization": 300
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Return monitoring configuration"""
        return {
            "metrics_path": "/metrics",
            "metrics_port": 9090,
            "custom_metrics": [
                "tabular_rows_generated_total",
                "tabular_generation_duration_seconds",
                "tabular_algorithm_usage"
            ],
            "alerts": [
                {
                    "name": "TabularHighErrorRate",
                    "condition": "rate(tabular_errors_total[5m]) > 0.05",
                    "severity": "warning"
                }
            ]
        }
```

### Step 2: Update Service Code

#### Remove Direct Infrastructure Dependencies

**Before:**
```python
# tabular/api/app.py
import boto3
from tabular.deploy.aws import AWSProvider
from tabular.api.auth import APIKeyAuth

s3_client = boto3.client('s3')
auth = APIKeyAuth()
```

**After:**
```python
# tabular/api/app.py
from unified_cloud_deployment.storage import StorageClient
from unified_cloud_deployment.auth import AuthClient

storage_client = StorageClient()  # Cloud-agnostic
auth_client = AuthClient()  # Unified auth
```

#### Update Authentication Integration

**Before:**
```python
# tabular/api/auth/auth_handler.py
class APIKeyAuth:
    def __init__(self):
        self.db = LocalDatabase()
    
    def validate_api_key(self, key: str) -> bool:
        # Local validation logic
        pass
```

**After:**
```python
# tabular/api/middleware/auth.py
from unified_cloud_deployment.auth import AuthMiddleware

app = FastAPI()
app.add_middleware(AuthMiddleware, service_name="tabular")

# Auth is now handled by unified auth service
```

#### Update Logging and Monitoring

**Before:**
```python
# tabular/api/middleware/logging_middleware.py
class RequestLogger:
    def __init__(self, service_name: str = "inferloop-synthetic-api"):
        self.logger = logging.getLogger(service_name)
```

**After:**
```python
# tabular/api/middleware/telemetry.py
from unified_cloud_deployment.monitoring import TelemetryMiddleware

app = FastAPI()
app.add_middleware(
    TelemetryMiddleware,
    service_name="tabular",
    enable_tracing=True,
    enable_metrics=True
)
```

### Step 3: Create Service Deployment Configuration

Create a deployment configuration file for each service:

```yaml
# tabular/deployment-config.yaml
apiVersion: inferloop.io/v1
kind: ServiceDeployment
metadata:
  name: tabular
  namespace: inferloop
spec:
  service:
    type: api
    tier: business  # starter, professional, business, enterprise
    
  deployment:
    replicas:
      min: 2
      max: 50
    strategy: RollingUpdate
    
  resources:
    requests:
      cpu: "1"
      memory: "2Gi"
    limits:
      cpu: "2"
      memory: "4Gi"
      
  features:
    algorithms:
      - name: sdv
        enabled: true
        tier: starter
      - name: ctgan
        enabled: true
        tier: professional
      - name: ydata
        enabled: true
        tier: professional
      - name: custom
        enabled: true
        tier: enterprise
        
  integrations:
    database:
      type: postgres
      schema: tabular
      pool_size: 20
    cache:
      type: redis
      prefix: "tabular:"
      ttl: 3600
    storage:
      type: object
      bucket: "${STORAGE_BUCKET}/tabular"
      
  api:
    rate_limits:
      starter: 100/hour
      professional: 1000/hour
      business: 10000/hour
      enterprise: unlimited
    endpoints:
      - path: /generate
        method: POST
        billing_metric: rows_generated
      - path: /validate
        method: POST
        billing_metric: validations_performed
        
  monitoring:
    metrics:
      enabled: true
      port: 9090
      path: /metrics
    logging:
      level: INFO
      format: json
    tracing:
      enabled: true
      sample_rate: 0.1
```

### Step 4: Remove Module-Specific Infrastructure Code

#### Move to Unified Infrastructure:

1. **Delete these directories:**
   ```bash
   rm -rf tabular/deploy/
   rm -rf tabular/inferloop-infra/
   ```

2. **Update imports in existing code:**
   ```python
   # Replace all imports like:
   from tabular.deploy.aws import SomeClass
   from tabular.inferloop-infra import SomeInfraClass
   
   # With unified imports:
   from unified_cloud_deployment.providers import CloudProvider
   from unified_cloud_deployment.core import InfrastructureManager
   ```

3. **Update configuration files:**
   ```yaml
   # Remove from tabular/pyproject.toml:
   [tool.poetry.dependencies]
   boto3 = "^1.26.0"  # Remove cloud-specific SDKs
   azure-mgmt = "^4.0.0"
   google-cloud = "^3.0.0"
   
   # Add unified dependency:
   inferloop-unified-infra = "^1.0.0"
   ```

### Step 5: Update CI/CD Pipeline

Update the CI/CD pipeline to use unified deployment:

```yaml
# .github/workflows/tabular-deploy.yml
name: Deploy Tabular Service

on:
  push:
    paths:
      - 'tabular/**'
      - '!tabular/deploy/**'  # Ignore old deploy directory

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker Image
        run: |
          docker build -t inferloop/tabular:${{ github.sha }} ./tabular
          
      - name: Deploy using Unified Infrastructure
        run: |
          cd unified-cloud-deployment
          ./scripts/deploy-service.sh tabular ${{ github.sha }}
```

### Step 6: Create Migration Script

Create an automated migration script:

```bash
#!/bin/bash
# migrate-to-unified-infra.sh

SERVICE_NAME=$1
ENVIRONMENT=${2:-development}

echo "Migrating $SERVICE_NAME to unified infrastructure..."

# Step 1: Backup current configuration
cp -r $SERVICE_NAME/deploy $SERVICE_NAME/deploy.backup
cp $SERVICE_NAME/pyproject.toml $SERVICE_NAME/pyproject.toml.backup

# Step 2: Create adapter
mkdir -p $SERVICE_NAME/infrastructure
cat > $SERVICE_NAME/infrastructure/adapter.py << EOF
from unified_cloud_deployment.core import ServiceAdapter
# ... adapter implementation
EOF

# Step 3: Update dependencies
cd $SERVICE_NAME
poetry remove boto3 azure-mgmt google-cloud
poetry add inferloop-unified-infra

# Step 4: Create deployment config
cat > $SERVICE_NAME/deployment-config.yaml << EOF
apiVersion: inferloop.io/v1
kind: ServiceDeployment
metadata:
  name: $SERVICE_NAME
# ... configuration
EOF

# Step 5: Update imports
find . -name "*.py" -type f -exec sed -i \
  's/from.*deploy.*import/from unified_cloud_deployment import/g' {} \;

# Step 6: Test the migration
pytest tests/

echo "Migration completed for $SERVICE_NAME"
```

## Service-Specific Migration Notes

### Tabular Service
- Move SDV, CTGAN, YData configurations to unified model registry
- Update batch processing to use unified job queue
- Migrate validation metrics to unified monitoring

### TextNLP Service
- Consolidate LLM model management with unified model service
- Move prompt templates to unified storage
- Update streaming to use unified WebSocket gateway

### SynDoc Service
- Migrate document templates to unified template service
- Update PDF generation to use unified job workers
- Move compliance templates to unified compliance service

## Benefits After Migration

1. **Simplified Deployment**
   - Single command to deploy any service
   - Consistent deployment across all environments
   - Automated infrastructure provisioning

2. **Unified Operations**
   - Single monitoring dashboard for all services
   - Centralized logging and tracing
   - Consistent alerting rules

3. **Cost Optimization**
   - Shared infrastructure resources
   - Better resource utilization
   - Centralized cost monitoring

4. **Enhanced Security**
   - Single authentication service
   - Consistent security policies
   - Centralized secret management

5. **Improved Developer Experience**
   - Less infrastructure code to maintain
   - Focus on business logic
   - Standardized patterns

## Migration Timeline

### Phase 1: Foundation (Week 1)
- Set up unified infrastructure
- Create base adapter interfaces
- Deploy shared services (auth, monitoring)

### Phase 2: Pilot Migration (Week 2)
- Migrate one service (e.g., SynDoc) as pilot
- Validate all functionality
- Document lessons learned

### Phase 3: Full Migration (Weeks 3-4)
- Migrate remaining services in parallel
- Update all CI/CD pipelines
- Comprehensive testing

### Phase 4: Cleanup (Week 5)
- Remove old infrastructure code
- Update all documentation
- Training for development team

## Validation Checklist

After migration, verify:

- [ ] Service starts successfully
- [ ] All API endpoints respond correctly
- [ ] Authentication works through unified auth
- [ ] Metrics appear in unified monitoring
- [ ] Logs are centralized
- [ ] Auto-scaling functions properly
- [ ] Multi-tenancy isolation works
- [ ] Billing metrics are tracked
- [ ] All tests pass
- [ ] Performance meets SLA

## Rollback Plan

If issues arise during migration:

1. **Immediate Rollback**
   ```bash
   # Restore backup
   mv $SERVICE_NAME/deploy.backup $SERVICE_NAME/deploy
   mv $SERVICE_NAME/pyproject.toml.backup $SERVICE_NAME/pyproject.toml
   
   # Redeploy old version
   cd $SERVICE_NAME/deploy
   ./deploy.sh $ENVIRONMENT
   ```

2. **Gradual Rollback**
   - Use feature flags to toggle between old and new infrastructure
   - Route percentage of traffic to new infrastructure
   - Monitor and adjust based on metrics

## Conclusion

This migration will transform Inferloop from module-specific infrastructure to a unified, scalable platform. While the initial migration requires effort, the long-term benefits in maintainability, scalability, and operational efficiency make it worthwhile.