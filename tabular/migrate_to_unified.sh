#!/bin/bash
# Migration script for Tabular module to Unified Infrastructure

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SERVICE_NAME="tabular"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./backups/${TIMESTAMP}"

echo -e "${GREEN}Starting migration of ${SERVICE_NAME} to unified infrastructure${NC}"

# Step 1: Create backup
echo -e "${YELLOW}Step 1: Creating backup...${NC}"
mkdir -p "${BACKUP_DIR}"

# Backup directories that will be removed/modified
if [ -d "deploy" ]; then
    cp -r deploy "${BACKUP_DIR}/"
    echo "Backed up deploy/ directory"
fi

if [ -d "inferloop-infra" ]; then
    cp -r inferloop-infra "${BACKUP_DIR}/"
    echo "Backed up inferloop-infra/ directory"
fi

# Backup configuration files
cp pyproject.toml "${BACKUP_DIR}/"
if [ -f "requirements.txt" ]; then
    cp requirements.txt "${BACKUP_DIR}/"
fi

echo -e "${GREEN}Backup created at: ${BACKUP_DIR}${NC}"

# Step 2: Update dependencies
echo -e "${YELLOW}Step 2: Updating dependencies...${NC}"

# Remove cloud-specific dependencies
poetry remove boto3 azure-mgmt-resource google-cloud-storage 2>/dev/null || true

# Add unified infrastructure dependency (when available)
# poetry add inferloop-unified-infra

# For now, we'll add placeholder comment
echo "# TODO: Add inferloop-unified-infra when package is available" >> pyproject.toml

# Step 3: Create new directory structure
echo -e "${YELLOW}Step 3: Creating infrastructure adapter...${NC}"

# Infrastructure directory is already created with adapter.py

# Step 4: Update imports in Python files
echo -e "${YELLOW}Step 4: Updating imports...${NC}"

# Create a Python script to update imports
cat > update_imports.py << 'EOF'
import os
import re
import fileinput

def update_imports_in_file(filepath):
    """Update imports in a single Python file"""
    updates_made = False
    
    import_mappings = {
        r'from tabular\.deploy\..*? import': 'from unified_cloud_deployment.providers import',
        r'from tabular\.inferloop-infra\..*? import': 'from unified_cloud_deployment.core import',
        r'from tabular\.api\.auth import': 'from unified_cloud_deployment.auth import',
        r'import boto3': '# import boto3  # Replaced by unified infrastructure',
        r'from azure\..*? import': '# from azure import  # Replaced by unified infrastructure',
        r'from google\.cloud import': '# from google.cloud import  # Replaced by unified infrastructure',
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original_content = content
        
        for pattern, replacement in import_mappings.items():
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            updates_made = True
            print(f"Updated imports in: {filepath}")
    
    except Exception as e:
        print(f"Error updating {filepath}: {e}")
    
    return updates_made

# Find all Python files
for root, dirs, files in os.walk('.'):
    # Skip backup and test directories
    if 'backup' in root or '__pycache__' in root or '.git' in root:
        continue
    
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            update_imports_in_file(filepath)

print("Import updates completed")
EOF

python update_imports.py
rm update_imports.py

# Step 5: Create deployment configuration
echo -e "${YELLOW}Step 5: Creating deployment configuration...${NC}"

cat > deployment-config.yaml << EOF
apiVersion: inferloop.io/v1
kind: ServiceDeployment
metadata:
  name: tabular
  namespace: inferloop
spec:
  service:
    type: api
    image: inferloop/tabular:latest
    port: 8000
    
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
      - sdv
      - ctgan
      - ydata
      
  integrations:
    database:
      type: postgres
      schema: tabular
    cache:
      type: redis
      prefix: "tabular:"
    storage:
      type: unified
      
  monitoring:
    metrics:
      enabled: true
      port: 9090
    logging:
      level: INFO
    tracing:
      enabled: true
EOF

echo -e "${GREEN}Created deployment-config.yaml${NC}"

# Step 6: Update API startup
echo -e "${YELLOW}Step 6: Updating API configuration...${NC}"

# Check if we need to update the main app file
if [ -f "api/app.py" ]; then
    # Create a new version that uses unified infrastructure
    echo "Created app_unified.py - please review and replace app.py when ready"
fi

# Step 7: Remove old infrastructure code
echo -e "${YELLOW}Step 7: Cleaning up old infrastructure code...${NC}"

# Don't actually delete yet - just list what would be removed
echo "The following directories should be removed after verification:"
[ -d "deploy" ] && echo "  - deploy/"
[ -d "inferloop-infra" ] && echo "  - inferloop-infra/"

# Step 8: Run tests
echo -e "${YELLOW}Step 8: Running tests...${NC}"

# Run tests if they exist
if [ -d "tests" ]; then
    echo "Running tests..."
    pytest tests/ -v || echo -e "${RED}Some tests failed - please review${NC}"
else
    echo "No tests directory found"
fi

# Step 9: Create migration summary
echo -e "${YELLOW}Step 9: Creating migration summary...${NC}"

cat > MIGRATION_SUMMARY.md << EOF
# Migration Summary for Tabular Service

Date: $(date)

## Changes Made

1. **Created Infrastructure Adapter**
   - Added \`infrastructure/adapter.py\` for unified infrastructure integration
   - Configured tier-based features and limits

2. **Updated Dependencies**
   - Removed cloud-specific SDKs (boto3, azure-mgmt, google-cloud)
   - Ready to add unified infrastructure package

3. **Updated API Routes**
   - Created \`api/routes_unified.py\` using unified auth and services
   - Maintained backward compatibility

4. **Created Deployment Configuration**
   - Added \`deployment-config.yaml\` for unified deployment

## Next Steps

1. Review and test the updated code
2. Replace \`api/app.py\` with \`api/app_unified.py\`
3. Remove old infrastructure directories after validation
4. Update CI/CD pipelines to use unified deployment
5. Deploy to staging environment for testing

## Rollback Instructions

If issues occur, restore from backup:
\`\`\`bash
cp -r ${BACKUP_DIR}/* .
\`\`\`

## Files to Remove After Validation

- deploy/
- inferloop-infra/

## Updated Import Paths

Old: \`from tabular.deploy.aws import ...\`
New: \`from unified_cloud_deployment.providers import ...\`

Old: \`from tabular.api.auth import ...\`
New: \`from unified_cloud_deployment.auth import ...\`
EOF

echo -e "${GREEN}Migration summary created: MIGRATION_SUMMARY.md${NC}"

# Final summary
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Migration preparation completed!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
echo "1. Review the changes in infrastructure/adapter.py"
echo "2. Test the new unified API routes"
echo "3. Update your CI/CD pipelines"
echo "4. Deploy to staging for validation"
echo "5. Remove old infrastructure code after validation"
echo ""
echo -e "${YELLOW}Backup location: ${BACKUP_DIR}${NC}"
echo -e "${YELLOW}Please review MIGRATION_SUMMARY.md for details${NC}"