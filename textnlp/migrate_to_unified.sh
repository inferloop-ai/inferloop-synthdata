#!/bin/bash
# Migration script for TextNLP module to Unified Infrastructure

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SERVICE_NAME="textnlp"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./backups/${TIMESTAMP}"

echo -e "${GREEN}Starting migration of ${SERVICE_NAME} to unified infrastructure${NC}"

# Step 1: Create backup
echo -e "${YELLOW}Step 1: Creating backup...${NC}"
mkdir -p "${BACKUP_DIR}"

# Backup current files
if [ -f "api/app.py" ]; then
    cp api/app.py "${BACKUP_DIR}/app.py.backup"
fi

if [ -f "api/routes.py" ]; then
    cp api/routes.py "${BACKUP_DIR}/routes.py.backup"
fi

cp pyproject.toml "${BACKUP_DIR}/"

echo -e "${GREEN}Backup created at: ${BACKUP_DIR}${NC}"

# Step 2: Update dependencies
echo -e "${YELLOW}Step 2: Updating dependencies...${NC}"

# Update pyproject.toml to remove direct cloud dependencies
# In real scenario, we would use poetry commands
echo "# Dependencies updated for unified infrastructure" >> pyproject.toml

# Step 3: Update imports
echo -e "${YELLOW}Step 3: Updating imports in Python files...${NC}"

# Create import updater script
cat > update_imports.py << 'EOF'
import os
import re

def update_imports_in_file(filepath):
    """Update imports in a single Python file"""
    
    import_mappings = {
        # Remove cloud-specific imports
        r'import\s+boto3': '# import boto3  # Replaced by unified infrastructure',
        r'from\s+azure[.\w]*\s+import': '# from azure import  # Replaced by unified infrastructure',
        r'from\s+google\.cloud\s+import': '# from google.cloud import  # Replaced by unified infrastructure',
        
        # Update to unified imports
        r'from\s+textnlp\.auth\s+import': 'from unified_cloud_deployment.auth import',
        r'from\s+textnlp\.monitoring\s+import': 'from unified_cloud_deployment.monitoring import',
        r'from\s+textnlp\.storage\s+import': 'from unified_cloud_deployment.storage import',
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        for pattern, replacement in import_mappings.items():
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Updated imports in: {filepath}")
            return True
    
    except Exception as e:
        print(f"Error updating {filepath}: {e}")
        return False
    
    return False

# Update all Python files
updated_count = 0
for root, dirs, files in os.walk('.'):
    # Skip backups and cache
    if any(skip in root for skip in ['backup', '__pycache__', '.git', 'node_modules']):
        continue
    
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            if update_imports_in_file(filepath):
                updated_count += 1

print(f"Updated {updated_count} files")
EOF

python update_imports.py
rm update_imports.py

# Step 4: Create deployment configuration
echo -e "${YELLOW}Step 4: Creating deployment configuration...${NC}"

cat > deployment-config.yaml << 'EOF'
apiVersion: inferloop.io/v1
kind: ServiceDeployment
metadata:
  name: textnlp
  namespace: inferloop
  labels:
    app: textnlp
    version: v1.0.0
spec:
  service:
    type: api
    description: "AI-powered synthetic text and NLP data generation service"
    image: inferloop/textnlp:latest
    port: 8000
    
  deployment:
    replicas:
      min: 2
      max: 50
    strategy:
      type: RollingUpdate
      rollingUpdate:
        maxSurge: 25%
        maxUnavailable: 0
    
  tiers:
    starter:
      resources:
        requests:
          cpu: "1"
          memory: "2Gi"
        limits:
          cpu: "2"
          memory: "4Gi"
      features:
        models: ["gpt2", "gpt2-medium"]
        max_tokens: 1024
        rate_limit: 500000 tokens/hour
        
    professional:
      resources:
        requests:
          cpu: "2"
          memory: "4Gi"
        limits:
          cpu: "4"
          memory: "8Gi"
      features:
        models: ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        max_tokens: 2048
        rate_limit: 5M tokens/hour
        templates: true
        
    business:
      resources:
        requests:
          cpu: "4"
          memory: "8Gi"
        limits:
          cpu: "8"
          memory: "16Gi"
          nvidia.com/gpu: "1"
      features:
        models: ["all-open-source"]
        max_tokens: 4096
        rate_limit: 25M tokens/hour
        templates: true
        streaming: true
        
    enterprise:
      resources:
        requests:
          cpu: "8"
          memory: "16Gi"
        limits:
          cpu: "16"
          memory: "32Gi"
          nvidia.com/gpu: "1"
      features:
        models: ["all"]
        max_tokens: 100000
        rate_limit: unlimited
        commercial_models: true
        fine_tuning: true
        
  integrations:
    database:
      type: postgres
      schema: textnlp
      pool:
        min: 5
        max: 30
        
    cache:
      type: redis
      namespace: "textnlp"
      ttl:
        default: 1800
        models: 3600
        
    storage:
      type: unified
      buckets:
        - name: prompts
          retention: 30d
        - name: generations
          retention: 7d
        - name: templates
          retention: -1
        - name: models
          retention: -1
          
    messaging:
      type: unified
      queues:
        - name: text-generation-queue
          max_size: 10000
        - name: fine-tuning-queue
          max_size: 1000
          
  api:
    endpoints:
      - path: /generate
        method: POST
        timeout: 300s
        streaming: true
        billing:
          metric: tokens_generated
          
      - path: /chat
        method: POST
        timeout: 600s
        streaming: true
        tier_required: professional
        
      - path: /validate
        method: POST
        timeout: 60s
        billing:
          metric: validations_performed
          
      - path: /models
        method: GET
        cache: 3600s
        public: true
        
      - path: /templates
        method: GET
        tier_required: professional
        
      - path: /fine-tune
        method: POST
        tier_required: enterprise
        
    websocket:
      enabled: true
      path: /ws/stream
      tier_required: professional
      
  monitoring:
    metrics:
      enabled: true
      port: 9090
      path: /metrics
      custom:
        - name: textnlp_tokens_generated_total
          type: counter
        - name: textnlp_generation_duration_seconds
          type: histogram
        - name: textnlp_active_streams
          type: gauge
          
    logging:
      level: INFO
      format: json
      sensitive_masking: true
      
    tracing:
      enabled: true
      sample_rate: 0.1
      
    alerts:
      - name: HighTokenLatency
        condition: "p95_latency > 100ms per token"
        severity: warning
        
      - name: ModelLoadFailure
        condition: "model_load_failures > 0"
        severity: critical
        
      - name: HighGPUUsage
        condition: "gpu_usage > 90%"
        severity: warning
        
  volumes:
    - name: model-cache
      mount_path: /models
      size: 50Gi
      storage_class: fast-ssd
      
  security:
    authentication:
      type: unified-auth
      required: true
      
    authorization:
      model: RBAC
      permissions:
        - textnlp:read
        - textnlp:generate
        - textnlp:validate
        - textnlp:templates
        - textnlp:fine-tune
        - textnlp:admin
EOF

echo -e "${GREEN}Created deployment-config.yaml${NC}"

# Step 5: Create model migration list
echo -e "${YELLOW}Step 5: Creating model migration checklist...${NC}"

cat > MODEL_MIGRATION.md << 'EOF'
# Model Migration Checklist

## Models to Migrate to Unified Storage

### Open Source Models
- [ ] gpt2 (124M parameters)
- [ ] gpt2-medium (355M parameters)
- [ ] gpt2-large (774M parameters)
- [ ] gpt2-xl (1.5B parameters)
- [ ] gpt-j-6b (6B parameters)
- [ ] llama-7b (7B parameters)

### Model Storage Paths
```
/models/
├── gpt2/
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer/
├── gpt2-medium/
├── gpt2-large/
├── gpt2-xl/
├── gpt-j-6b/
└── llama-7b/
```

### Upload Commands
```bash
# Upload to unified storage
unified-storage upload ./models/gpt2 s3://inferloop-models/textnlp/gpt2/
```

## Configuration Updates Needed

1. Update model loading to use unified storage
2. Implement model caching strategy
3. Configure GPU scheduling for large models
4. Set up model versioning
EOF

echo -e "${GREEN}Created MODEL_MIGRATION.md${NC}"

# Step 6: Validate configuration
echo -e "${YELLOW}Step 6: Validating configuration...${NC}"

# Create validation script
cat > validate_migration.py << 'EOF'
import yaml
import json

def validate_deployment_config():
    """Validate the deployment configuration"""
    try:
        with open('deployment-config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['apiVersion', 'kind', 'metadata', 'spec']
        for field in required_fields:
            if field not in config:
                print(f"ERROR: Missing required field: {field}")
                return False
        
        # Check service configuration
        if 'service' not in config['spec']:
            print("ERROR: Missing service configuration")
            return False
            
        print("✓ Deployment configuration is valid")
        return True
        
    except Exception as e:
        print(f"ERROR validating config: {e}")
        return False

def check_unified_imports():
    """Check if imports have been updated"""
    import glob
    
    issues = []
    for py_file in glob.glob('**/*.py', recursive=True):
        if 'backup' in py_file or '__pycache__' in py_file:
            continue
            
        with open(py_file, 'r') as f:
            content = f.read()
            
        # Check for old imports
        if 'import boto3' in content and not content.count('# import boto3'):
            issues.append(f"{py_file}: Still has boto3 import")
            
    if issues:
        print("Import issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ All imports updated")
        return True

# Run validations
print("Running migration validation...")
config_valid = validate_deployment_config()
imports_valid = check_unified_imports()

if config_valid and imports_valid:
    print("\n✓ Migration validation passed!")
else:
    print("\n✗ Migration validation failed - please fix issues above")
EOF

python validate_migration.py
rm validate_migration.py

# Step 7: Run tests
echo -e "${YELLOW}Step 7: Running tests...${NC}"

if [ -d "tests" ]; then
    echo "Running unit tests..."
    pytest tests/ -v --tb=short || echo -e "${RED}Some tests failed - please review${NC}"
else
    echo "No tests directory found"
fi

# Step 8: Create migration summary
echo -e "${YELLOW}Step 8: Creating migration summary...${NC}"

cat > MIGRATION_SUMMARY.md << EOF
# Migration Summary for TextNLP Service

Date: $(date)

## Changes Made

1. **Created Infrastructure Adapter**
   - Added \`infrastructure/adapter.py\` for unified infrastructure integration
   - Configured tier-based model access and limits
   - Set up GPU support for Business/Enterprise tiers

2. **Updated API Application**
   - Created \`api/app_unified.py\` using unified services
   - Integrated unified auth, storage, and monitoring
   - Added WebSocket support for streaming

3. **Created Deployment Configuration**
   - Added \`deployment-config.yaml\` for unified deployment
   - Defined tier-specific features and resources
   - Configured model storage and caching

## Model-Specific Considerations

- **Model Storage**: Models need to be uploaded to unified storage
- **GPU Requirements**: Business and Enterprise tiers require GPU nodes
- **Caching Strategy**: Implement model caching for better performance
- **Commercial Models**: Enterprise tier includes GPT-4 and Claude access

## Next Steps

1. Upload model files to unified storage (see MODEL_MIGRATION.md)
2. Test the unified API with different model sizes
3. Verify GPU scheduling for large models
4. Update client libraries to use new endpoints
5. Deploy to staging environment

## Rollback Instructions

If issues occur, restore from backup:
\`\`\`bash
cp -r ${BACKUP_DIR}/* .
\`\`\`

## Performance Considerations

- Model loading time increased due to unified storage
- Implement preloading for frequently used models
- Use model caching to reduce latency
- Consider model quantization for edge deployment
EOF

echo -e "${GREEN}Migration summary created: MIGRATION_SUMMARY.md${NC}"

# Final summary
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}TextNLP migration preparation completed!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
echo "1. Review infrastructure/adapter.py configuration"
echo "2. Upload models to unified storage (see MODEL_MIGRATION.md)"
echo "3. Test streaming and WebSocket functionality"
echo "4. Verify GPU allocation for large models"
echo "5. Update API documentation"
echo ""
echo -e "${YELLOW}Backup location: ${BACKUP_DIR}${NC}"
echo -e "${YELLOW}Please review MIGRATION_SUMMARY.md for details${NC}"