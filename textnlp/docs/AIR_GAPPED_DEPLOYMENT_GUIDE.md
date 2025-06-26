# TextNLP Air-Gapped Deployment Guide

## Table of Contents
1. [Overview](#overview)
2. [Pre-Deployment Preparation](#pre-deployment-preparation)
3. [Offline Bundle Creation](#offline-bundle-creation)
4. [Transfer and Installation](#transfer-and-installation)
5. [Configuration for Offline Environments](#configuration-for-offline-environments)
6. [Model Management](#model-management)
7. [Updates and Maintenance](#updates-and-maintenance)
8. [Troubleshooting Offline Issues](#troubleshooting-offline-issues)
9. [Security Considerations](#security-considerations)
10. [Compliance and Auditing](#compliance-and-auditing)

## Overview

Air-gapped deployment refers to running TextNLP in environments with no internet connectivity. This is common in:
- High-security government facilities
- Financial institutions
- Healthcare organizations
- Industrial control systems
- Military installations

### Key Challenges
- No access to external package repositories
- No automatic model downloads
- No external API calls
- Manual update processes
- Limited debugging capabilities

### Solution Architecture
```
┌─────────────────────────────────────────┐
│        Internet-Connected Zone          │
│  ┌─────────────────────────────────┐   │
│  │   Preparation Environment        │   │
│  │   - Download dependencies       │   │
│  │   - Create offline bundles      │   │
│  │   - Security scanning           │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
                    │
                    │ Physical Transfer
                    │ (USB, DVD, etc.)
                    ▼
┌─────────────────────────────────────────┐
│          Air-Gapped Zone                │
│  ┌─────────────────────────────────┐   │
│  │   Production Environment         │   │
│  │   - Local repositories          │   │
│  │   - Offline model serving       │   │
│  │   - Internal monitoring         │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## Pre-Deployment Preparation

### 1. System Requirements Verification

Create a requirements checklist:

```bash
#!/bin/bash
# check_requirements.sh

echo "TextNLP Air-Gapped Deployment Requirements Check"
echo "=============================================="

# Check OS
echo -n "Operating System: "
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "$NAME $VERSION"
else
    echo "Unknown"
fi

# Check CPU
echo -n "CPU Cores: "
nproc

# Check Memory
echo -n "Total Memory: "
free -h | grep "^Mem:" | awk '{print $2}'

# Check Disk Space
echo -n "Available Disk Space: "
df -h / | tail -1 | awk '{print $4}'

# Check Python
echo -n "Python Version: "
python3 --version 2>/dev/null || echo "Not installed"

# Check Docker
echo -n "Docker Version: "
docker --version 2>/dev/null || echo "Not installed"

# Check Architecture
echo -n "Architecture: "
uname -m

# Generate report
cat > requirements_report.txt << EOF
TextNLP Air-Gapped Deployment Requirements Report
Generated: $(date)

Minimum Requirements:
- OS: Ubuntu 20.04+ or RHEL 8+
- CPU: 8+ cores
- Memory: 16+ GB
- Storage: 100+ GB
- Python: 3.8+
- Docker: 20.10+

Current System:
$(uname -a)
$(cat /proc/cpuinfo | grep "model name" | head -1)
$(free -h)
$(df -h /)
EOF

echo ""
echo "Report saved to requirements_report.txt"
```

### 2. Dependency Analysis

```python
# analyze_dependencies.py
"""Analyze all dependencies for air-gapped deployment."""

import subprocess
import json
import os
from pathlib import Path
import pkg_resources
import requests
from packaging import version

def get_all_dependencies():
    """Get all Python dependencies including transitive ones."""
    # Get direct dependencies
    with open('requirements.txt') as f:
        direct_deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Get all dependencies using pip-tools
    subprocess.run(['pip-compile', '--generate-hashes', 'requirements.txt', '-o', 'requirements-locked.txt'])
    
    # Parse locked requirements
    dependencies = {}
    with open('requirements-locked.txt') as f:
        for line in f:
            if '==' in line and not line.startswith('#'):
                parts = line.strip().split('==')
                if len(parts) == 2:
                    name = parts[0].strip()
                    version_hash = parts[1].strip()
                    version_num = version_hash.split(' ')[0]
                    dependencies[name] = {
                        'version': version_num,
                        'direct': name in [d.split('==')[0] for d in direct_deps]
                    }
    
    return dependencies

def analyze_security_vulnerabilities():
    """Check for known vulnerabilities."""
    subprocess.run(['pip', 'install', 'safety'])
    result = subprocess.run(['safety', 'check', '--json'], capture_output=True, text=True)
    
    vulnerabilities = json.loads(result.stdout) if result.returncode == 0 else []
    return vulnerabilities

def estimate_download_size():
    """Estimate total download size."""
    total_size = 0
    
    # Python packages
    packages = subprocess.run(
        ['pip', 'download', '-r', 'requirements.txt', '--no-deps', '--dry-run'],
        capture_output=True, text=True
    )
    
    # Docker images
    docker_images = [
        'python:3.9-slim',
        'postgres:15',
        'redis:7-alpine',
        'nginx:alpine'
    ]
    
    # Models (approximate sizes)
    model_sizes = {
        'gpt2': 500,  # MB
        'gpt2-medium': 1500,
        'gpt2-large': 3000,
        'distilgpt2': 350
    }
    
    return {
        'python_packages': '~500 MB',
        'docker_images': '~2 GB',
        'models': sum(model_sizes.values()),
        'total_estimated': f'~{(500 + 2000 + sum(model_sizes.values())) / 1000:.1f} GB'
    }

def generate_dependency_report():
    """Generate comprehensive dependency report."""
    print("Analyzing dependencies...")
    
    dependencies = get_all_dependencies()
    vulnerabilities = analyze_security_vulnerabilities()
    sizes = estimate_download_size()
    
    report = {
        'total_dependencies': len(dependencies),
        'direct_dependencies': sum(1 for d in dependencies.values() if d['direct']),
        'transitive_dependencies': sum(1 for d in dependencies.values() if not d['direct']),
        'known_vulnerabilities': len(vulnerabilities),
        'estimated_sizes': sizes,
        'dependencies': dependencies,
        'vulnerabilities': vulnerabilities
    }
    
    with open('dependency_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Total dependencies: {report['total_dependencies']}")
    print(f"Known vulnerabilities: {report['known_vulnerabilities']}")
    print(f"Estimated total size: {sizes['total_estimated']}")
    print("\nReport saved to dependency_report.json")

if __name__ == "__main__":
    generate_dependency_report()
```

## Offline Bundle Creation

### 1. Complete Offline Bundle Script

```bash
#!/bin/bash
# create_offline_bundle.sh

set -e

BUNDLE_DIR="textnlp-offline-bundle"
BUNDLE_VERSION=$(date +%Y%m%d-%H%M%S)
BUNDLE_NAME="textnlp-offline-${BUNDLE_VERSION}.tar.gz"

echo "Creating TextNLP Offline Bundle v${BUNDLE_VERSION}"
echo "================================================"

# Create bundle directory structure
mkdir -p ${BUNDLE_DIR}/{packages,docker,models,scripts,config,docs}

# 1. Download Python packages
echo "1. Downloading Python packages..."
pip download -r requirements.txt -d ${BUNDLE_DIR}/packages/
pip download pip setuptools wheel -d ${BUNDLE_DIR}/packages/

# Create package index
cd ${BUNDLE_DIR}/packages/
pip install pip2pi
dir2pi .
cd ../..

# 2. Save Docker images
echo "2. Saving Docker images..."
DOCKER_IMAGES=(
    "python:3.9-slim"
    "postgres:15"
    "redis:7-alpine"
    "nginx:alpine"
    "busybox:latest"
    "alpine:latest"
)

for image in "${DOCKER_IMAGES[@]}"; do
    echo "Pulling ${image}..."
    docker pull ${image}
done

docker save ${DOCKER_IMAGES[@]} -o ${BUNDLE_DIR}/docker/images.tar

# 3. Download models
echo "3. Downloading models..."
python scripts/download_models_offline.py --output-dir ${BUNDLE_DIR}/models/

# 4. Copy source code
echo "4. Copying source code..."
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='venv' \
    --exclude='*.pyc' --exclude='.pytest_cache' \
    . ${BUNDLE_DIR}/source/

# 5. Create offline installation scripts
echo "5. Creating installation scripts..."

# Main installation script
cat > ${BUNDLE_DIR}/install.sh << 'INSTALL_SCRIPT'
#!/bin/bash
set -e

echo "TextNLP Offline Installation"
echo "==========================="

# Check prerequisites
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi

# Set up Python environment
echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install packages from local index
echo "Installing Python packages..."
pip install --upgrade pip setuptools wheel --no-index --find-links packages/
pip install -r source/requirements.txt --no-index --find-links packages/

# Load Docker images
echo "Loading Docker images..."
docker load -i docker/images.tar

# Copy models
echo "Installing models..."
mkdir -p /opt/textnlp/models
cp -r models/* /opt/textnlp/models/

# Set up configuration
echo "Setting up configuration..."
cp -r config/* /etc/textnlp/

echo "Installation complete!"
echo "To start TextNLP, run: ./start.sh"
INSTALL_SCRIPT

chmod +x ${BUNDLE_DIR}/install.sh

# Startup script
cat > ${BUNDLE_DIR}/start.sh << 'START_SCRIPT'
#!/bin/bash
set -e

# Start services
docker-compose -f docker-compose.offline.yml up -d

# Wait for services
echo "Waiting for services to start..."
sleep 10

# Run migrations
docker-compose exec -T api python manage.py migrate

echo "TextNLP is running!"
echo "API available at: http://localhost:8000"
echo "Documentation at: http://localhost:8000/docs"
START_SCRIPT

chmod +x ${BUNDLE_DIR}/start.sh

# 6. Create offline Docker Compose
cat > ${BUNDLE_DIR}/docker-compose.offline.yml << 'COMPOSE_FILE'
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: textnlp
      POSTGRES_PASSWORD: ${DB_PASSWORD:-changeme}
      POSTGRES_DB: textnlp
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - textnlp-network
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD:-changeme}
    volumes:
      - redis_data:/data
    networks:
      - textnlp-network
    restart: unless-stopped

  api:
    build:
      context: ./source
      dockerfile: Dockerfile.offline
    environment:
      - OFFLINE_MODE=true
      - MODEL_PATH=/models
      - DATABASE_URL=postgresql://textnlp:${DB_PASSWORD:-changeme}@postgres:5432/textnlp
      - REDIS_URL=redis://:${REDIS_PASSWORD:-changeme}@redis:6379/0
    volumes:
      - /opt/textnlp/models:/models:ro
      - ./logs:/app/logs
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    networks:
      - textnlp-network
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf:ro
    ports:
      - "80:80"
    depends_on:
      - api
    networks:
      - textnlp-network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:

networks:
  textnlp-network:
    driver: bridge
COMPOSE_FILE

# 7. Create offline Dockerfile
cat > ${BUNDLE_DIR}/source/Dockerfile.offline << 'DOCKERFILE'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages from offline bundle
COPY packages /packages
RUN pip install --no-index --find-links /packages -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 textnlp && chown -R textnlp:textnlp /app
USER textnlp

# Set environment for offline mode
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV OFFLINE_MODE=true

EXPOSE 8000

CMD ["uvicorn", "textnlp.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
DOCKERFILE

# 8. Documentation
echo "8. Preparing documentation..."
cp -r docs/* ${BUNDLE_DIR}/docs/

# Create offline README
cat > ${BUNDLE_DIR}/README.md << 'README'
# TextNLP Offline Bundle

This bundle contains everything needed to run TextNLP in an air-gapped environment.

## Contents
- `/packages/` - Python packages and local PyPI index
- `/docker/` - Docker images
- `/models/` - Pre-downloaded language models
- `/source/` - Application source code
- `/config/` - Configuration files
- `/docs/` - Documentation
- `/scripts/` - Utility scripts

## Installation
1. Extract this bundle on the target system
2. Run: `./install.sh`
3. Configure environment variables in `.env`
4. Start services: `./start.sh`

## Updating
To update the system, create a new bundle on an internet-connected system
and transfer it to the air-gapped environment.

## Support
Refer to the documentation in the `/docs/` directory.
README

# 9. Create checksum
echo "9. Creating checksums..."
cd ${BUNDLE_DIR}
find . -type f -exec sha256sum {} \; > checksums.txt
cd ..

# 10. Compress bundle
echo "10. Creating compressed bundle..."
tar -czf ${BUNDLE_NAME} ${BUNDLE_DIR}

# Calculate final size
BUNDLE_SIZE=$(du -h ${BUNDLE_NAME} | cut -f1)

echo ""
echo "Bundle creation complete!"
echo "========================"
echo "Bundle: ${BUNDLE_NAME}"
echo "Size: ${BUNDLE_SIZE}"
echo "Version: ${BUNDLE_VERSION}"
echo ""
echo "To deploy:"
echo "1. Transfer ${BUNDLE_NAME} to the air-gapped system"
echo "2. Extract: tar -xzf ${BUNDLE_NAME}"
echo "3. Install: cd textnlp-offline-bundle && ./install.sh"
```

### 2. Model Download Script

```python
# scripts/download_models_offline.py
"""Download models for offline deployment."""

import os
import argparse
import json
from pathlib import Path
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig
)
import torch
import shutil
from tqdm import tqdm

def download_model(model_name: str, output_dir: Path, quantize: bool = False):
    """Download a model and all its dependencies."""
    print(f"Downloading {model_name}...")
    
    model_path = output_dir / model_name
    model_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download configuration first
        config = AutoConfig.from_pretrained(model_name)
        config.save_pretrained(model_path)
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_path)
        
        # Download model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if quantize else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Optionally quantize for smaller size
        if quantize:
            print(f"Quantizing {model_name}...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        
        # Save model
        model.save_pretrained(model_path)
        
        # Create metadata file
        metadata = {
            'model_name': model_name,
            'downloaded_at': str(Path.ctime(model_path)),
            'quantized': quantize,
            'size_mb': sum(f.stat().st_size for f in model_path.rglob('*')) / 1024 / 1024
        }
        
        with open(model_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ {model_name} downloaded successfully ({metadata['size_mb']:.1f} MB)")
        
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        if model_path.exists():
            shutil.rmtree(model_path)

def main():
    parser = argparse.ArgumentParser(description='Download models for offline deployment')
    parser.add_argument('--models', nargs='+', 
                       default=['gpt2', 'distilgpt2', 'gpt2-medium'],
                       help='Models to download')
    parser.add_argument('--output-dir', type=Path, default='models',
                       help='Output directory')
    parser.add_argument('--quantize', action='store_true',
                       help='Quantize models for smaller size')
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download each model
    for model_name in args.models:
        download_model(model_name, args.output_dir, args.quantize)
    
    # Create model registry
    registry = {}
    for model_dir in args.output_dir.iterdir():
        if model_dir.is_dir() and (model_dir / 'config.json').exists():
            with open(model_dir / 'metadata.json') as f:
                registry[model_dir.name] = json.load(f)
    
    with open(args.output_dir / 'registry.json', 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"\nTotal models downloaded: {len(registry)}")
    total_size = sum(m['size_mb'] for m in registry.values())
    print(f"Total size: {total_size:.1f} MB")

if __name__ == "__main__":
    main()
```

### 3. Security Scanning Script

```bash
#!/bin/bash
# security_scan.sh

echo "Security Scan for Offline Bundle"
echo "================================"

SCAN_DIR=$1
REPORT_FILE="security_scan_report_$(date +%Y%m%d_%H%M%S).txt"

if [ -z "$SCAN_DIR" ]; then
    echo "Usage: $0 <bundle_directory>"
    exit 1
fi

# Initialize report
cat > $REPORT_FILE << EOF
Security Scan Report
Generated: $(date)
Bundle: $SCAN_DIR

EOF

# 1. Check for known vulnerabilities in Python packages
echo "1. Scanning Python packages for vulnerabilities..."
echo "## Python Package Vulnerabilities" >> $REPORT_FILE
pip install safety
safety check -r $SCAN_DIR/source/requirements.txt >> $REPORT_FILE 2>&1

# 2. Scan for secrets and sensitive data
echo "2. Scanning for secrets and sensitive data..."
echo -e "\n## Secrets Scan" >> $REPORT_FILE
pip install truffleHog
truffleHog --regex --entropy=False $SCAN_DIR >> $REPORT_FILE 2>&1

# 3. Check file permissions
echo "3. Checking file permissions..."
echo -e "\n## File Permissions" >> $REPORT_FILE
find $SCAN_DIR -type f -perm /111 -ls >> $REPORT_FILE

# 4. Verify checksums
echo "4. Verifying checksums..."
echo -e "\n## Checksum Verification" >> $REPORT_FILE
cd $SCAN_DIR
sha256sum -c checksums.txt >> $REPORT_FILE 2>&1
cd ..

# 5. Scan Docker images
echo "5. Scanning Docker images..."
echo -e "\n## Docker Image Scan" >> $REPORT_FILE
if command -v trivy &> /dev/null; then
    trivy image --input $SCAN_DIR/docker/images.tar >> $REPORT_FILE 2>&1
else
    echo "Trivy not installed - skipping Docker scan" >> $REPORT_FILE
fi

# 6. Check for unauthorized executables
echo "6. Checking for unauthorized executables..."
echo -e "\n## Executable Files" >> $REPORT_FILE
find $SCAN_DIR -type f -executable -exec file {} \; | grep -E 'ELF|script' >> $REPORT_FILE

# Summary
echo -e "\n## Summary" >> $REPORT_FILE
echo "Scan completed at $(date)" >> $REPORT_FILE

echo ""
echo "Security scan complete. Report saved to: $REPORT_FILE"
echo ""
echo "Summary:"
grep -A 5 "## Summary" $REPORT_FILE
```

## Transfer and Installation

### 1. Transfer Methods

#### Secure USB Transfer
```bash
#!/bin/bash
# secure_usb_transfer.sh

# Format USB with encryption
USB_DEVICE="/dev/sdb"
USB_MOUNT="/mnt/secure-usb"

# Create encrypted partition
echo "Creating encrypted USB..."
sudo cryptsetup luksFormat $USB_DEVICE
sudo cryptsetup luksOpen $USB_DEVICE secure-usb
sudo mkfs.ext4 /dev/mapper/secure-usb

# Mount encrypted USB
sudo mkdir -p $USB_MOUNT
sudo mount /dev/mapper/secure-usb $USB_MOUNT

# Copy bundle with verification
echo "Copying bundle..."
cp textnlp-offline-*.tar.gz $USB_MOUNT/
cp textnlp-offline-*.tar.gz.sig $USB_MOUNT/  # Digital signature

# Verify copy
cd $USB_MOUNT
sha256sum -c ../checksums.txt

# Unmount and close
sudo umount $USB_MOUNT
sudo cryptsetup luksClose secure-usb

echo "Transfer complete. USB can be safely removed."
```

#### Network Transfer (Internal Network Only)
```bash
#!/bin/bash
# internal_network_transfer.sh

# On source machine
BUNDLE_FILE="textnlp-offline-20240115.tar.gz"
TARGET_HOST="airgapped-server.internal"
TARGET_USER="admin"

# Create SSH tunnel with compression
echo "Transferring bundle via secure internal network..."
scp -C -o Compression=yes \
    -o CompressionLevel=9 \
    $BUNDLE_FILE \
    $TARGET_USER@$TARGET_HOST:/tmp/

# Verify transfer
ssh $TARGET_USER@$TARGET_HOST "cd /tmp && sha256sum $BUNDLE_FILE"
```

### 2. Installation Process

#### Automated Installation
```bash
#!/bin/bash
# install_airgapped.sh

set -e

BUNDLE_FILE=$1
INSTALL_DIR="/opt/textnlp"

if [ -z "$BUNDLE_FILE" ]; then
    echo "Usage: $0 <bundle_file>"
    exit 1
fi

echo "TextNLP Air-Gapped Installation"
echo "==============================="

# 1. Verify bundle integrity
echo "1. Verifying bundle integrity..."
sha256sum -c ${BUNDLE_FILE}.sha256

# 2. Extract bundle
echo "2. Extracting bundle..."
tar -xzf $BUNDLE_FILE
cd textnlp-offline-bundle

# 3. Run pre-installation checks
echo "3. Running pre-installation checks..."
./scripts/pre_install_check.sh

# 4. Create installation directory
echo "4. Creating installation directory..."
sudo mkdir -p $INSTALL_DIR
sudo chown $USER:$USER $INSTALL_DIR

# 5. Install Python packages
echo "5. Installing Python packages..."
python3 -m venv $INSTALL_DIR/venv
source $INSTALL_DIR/venv/bin/activate
pip install --no-index --find-links packages/ -r source/requirements.txt

# 6. Load Docker images
echo "6. Loading Docker images..."
docker load -i docker/images.tar

# 7. Copy application files
echo "7. Installing application..."
cp -r source/* $INSTALL_DIR/
cp -r models $INSTALL_DIR/
cp -r config $INSTALL_DIR/

# 8. Set up systemd services
echo "8. Setting up services..."
sudo cp scripts/textnlp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable textnlp

# 9. Initialize database
echo "9. Initializing database..."
cd $INSTALL_DIR
docker-compose -f docker-compose.offline.yml up -d postgres redis
sleep 10
source venv/bin/activate
python manage.py migrate

# 10. Start services
echo "10. Starting services..."
sudo systemctl start textnlp

echo ""
echo "Installation complete!"
echo "TextNLP is available at: http://localhost:8000"
echo "To check status: sudo systemctl status textnlp"
```

## Configuration for Offline Environments

### 1. Environment Configuration

```bash
# .env.offline
# TextNLP Offline Configuration

# Application Settings
ENVIRONMENT=production
OFFLINE_MODE=true
DEBUG=false
SECRET_KEY=<generate-strong-key>

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database (Local)
DATABASE_URL=postgresql://textnlp:secure-password@localhost:5432/textnlp
DATABASE_POOL_SIZE=20

# Redis (Local)
REDIS_URL=redis://:secure-password@localhost:6379/0

# Model Configuration
MODEL_PATH=/opt/textnlp/models
MODEL_CACHE_SIZE=4
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1

# Disable External Services
DISABLE_TELEMETRY=true
DISABLE_EXTERNAL_APIS=true
DISABLE_AUTO_UPDATE=true

# Local Services Only
USE_LOCAL_MODELS=true
USE_LOCAL_CACHE=true

# Security
ENABLE_AUTHENTICATION=true
ENABLE_AUDIT_LOGGING=true
SESSION_TIMEOUT=1800

# Resource Limits
MAX_REQUEST_SIZE=10MB
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=300
```

### 2. Application Configuration

```python
# config/offline_config.py
"""Configuration for offline deployment."""

import os
from pathlib import Path

class OfflineConfig:
    """Configuration for air-gapped environments."""
    
    # Offline mode flag
    OFFLINE_MODE = True
    
    # Disable all external connections
    DISABLE_EXTERNAL_APIS = True
    DISABLE_TELEMETRY = True
    DISABLE_AUTO_UPDATE = True
    
    # Local model configuration
    MODEL_PATH = Path("/opt/textnlp/models")
    USE_LOCAL_MODELS_ONLY = True
    ALLOWED_MODELS = ["gpt2", "distilgpt2", "gpt2-medium"]
    
    # Cache configuration
    CACHE_TYPE = "filesystem"
    CACHE_DIR = Path("/var/cache/textnlp")
    CACHE_DEFAULT_TIMEOUT = 86400  # 24 hours
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'postgresql://textnlp:changeme@localhost:5432/textnlp'
    )
    
    # Security settings
    REQUIRE_AUTHENTICATION = True
    AUDIT_ALL_REQUESTS = True
    AUDIT_LOG_PATH = Path("/var/log/textnlp/audit.log")
    
    # Resource limits
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
    MAX_TOKENS_PER_REQUEST = 1024
    RATE_LIMIT_PER_HOUR = 1000
    
    # Monitoring
    METRICS_ENABLED = True
    METRICS_EXPORT_PATH = Path("/var/log/textnlp/metrics")
    
    @classmethod
    def validate(cls):
        """Validate offline configuration."""
        errors = []
        
        # Check model directory
        if not cls.MODEL_PATH.exists():
            errors.append(f"Model directory not found: {cls.MODEL_PATH}")
        
        # Check available models
        for model in cls.ALLOWED_MODELS:
            model_path = cls.MODEL_PATH / model
            if not model_path.exists():
                errors.append(f"Model not found: {model}")
        
        # Check cache directory
        if not cls.CACHE_DIR.exists():
            cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Check audit log directory
        if not cls.AUDIT_LOG_PATH.parent.exists():
            cls.AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
```

### 3. Offline Model Loading

```python
# models/offline_loader.py
"""Offline model loading for air-gapped environments."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

logger = logging.getLogger(__name__)

class OfflineModelLoader:
    """Load models from local storage only."""
    
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.loaded_models: Dict[str, any] = {}
        self.model_registry = self._load_registry()
        
        # Ensure offline mode
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
    
    def _load_registry(self) -> Dict:
        """Load model registry from local storage."""
        registry_path = self.model_path / 'registry.json'
        if registry_path.exists():
            with open(registry_path) as f:
                return json.load(f)
        return {}
    
    def list_available_models(self) -> List[str]:
        """List all available offline models."""
        models = []
        for model_dir in self.model_path.iterdir():
            if model_dir.is_dir() and (model_dir / 'config.json').exists():
                models.append(model_dir.name)
        return models
    
    def load_model(self, model_name: str) -> tuple:
        """Load model from local storage."""
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded")
            return self.loaded_models[model_name]
        
        model_dir = self.model_path / model_name
        if not model_dir.exists():
            raise ValueError(f"Model {model_name} not found in offline storage")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                local_files_only=True
            )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                local_files_only=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Cache loaded model
            self.loaded_models[model_name] = (model, tokenizer)
            
            logger.info(f"Successfully loaded offline model: {model_name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a model."""
        if model_name in self.model_registry:
            return self.model_registry[model_name]
        
        model_dir = self.model_path / model_name
        if not model_dir.exists():
            return {}
        
        info = {
            'name': model_name,
            'path': str(model_dir),
            'size_mb': sum(f.stat().st_size for f in model_dir.rglob('*')) / 1024 / 1024
        }
        
        # Load config
        config_path = model_dir / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                info['architecture'] = config.get('architectures', ['unknown'])[0]
                info['parameters'] = config.get('n_params', 'unknown')
        
        return info
```

## Model Management

### 1. Model Version Control

```python
# scripts/model_version_manager.py
"""Manage model versions in offline environment."""

import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class ModelVersionManager:
    """Manage model versions and updates."""
    
    def __init__(self, model_base_path: Path):
        self.model_base_path = Path(model_base_path)
        self.versions_file = self.model_base_path / 'versions.json'
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict:
        """Load version information."""
        if self.versions_file.exists():
            with open(self.versions_file) as f:
                return json.load(f)
        return {}
    
    def _save_versions(self):
        """Save version information."""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def _calculate_checksum(self, path: Path) -> str:
        """Calculate checksum for a directory."""
        hasher = hashlib.sha256()
        for file_path in sorted(path.rglob('*')):
            if file_path.is_file():
                hasher.update(file_path.name.encode())
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b''):
                        hasher.update(chunk)
        return hasher.hexdigest()
    
    def register_model(self, model_name: str, version: str, metadata: Dict = None):
        """Register a new model version."""
        model_path = self.model_base_path / model_name
        if not model_path.exists():
            raise ValueError(f"Model {model_name} not found")
        
        checksum = self._calculate_checksum(model_path)
        
        if model_name not in self.versions:
            self.versions[model_name] = {}
        
        self.versions[model_name][version] = {
            'checksum': checksum,
            'registered_at': datetime.now().isoformat(),
            'metadata': metadata or {},
            'active': True
        }
        
        self._save_versions()
        print(f"Registered {model_name} version {version}")
    
    def list_versions(self, model_name: str) -> List[str]:
        """List all versions of a model."""
        if model_name in self.versions:
            return list(self.versions[model_name].keys())
        return []
    
    def get_active_version(self, model_name: str) -> Optional[str]:
        """Get the active version of a model."""
        if model_name in self.versions:
            for version, info in self.versions[model_name].items():
                if info.get('active', False):
                    return version
        return None
    
    def backup_model(self, model_name: str, backup_name: str = None):
        """Create a backup of a model."""
        model_path = self.model_base_path / model_name
        if not model_path.exists():
            raise ValueError(f"Model {model_name} not found")
        
        if not backup_name:
            backup_name = f"{model_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.model_base_path / 'backups' / backup_name
        backup_path.parent.mkdir(exist_ok=True)
        
        shutil.copytree(model_path, backup_path)
        
        # Record backup
        backup_info = {
            'original_model': model_name,
            'backup_path': str(backup_path),
            'created_at': datetime.now().isoformat(),
            'size_mb': sum(f.stat().st_size for f in backup_path.rglob('*')) / 1024 / 1024
        }
        
        backups_file = self.model_base_path / 'backups' / 'backups.json'
        backups = {}
        if backups_file.exists():
            with open(backups_file) as f:
                backups = json.load(f)
        
        backups[backup_name] = backup_info
        
        with open(backups_file, 'w') as f:
            json.dump(backups, f, indent=2)
        
        print(f"Created backup: {backup_name}")
        return backup_path
    
    def restore_model(self, model_name: str, backup_name: str):
        """Restore a model from backup."""
        backup_path = self.model_base_path / 'backups' / backup_name
        if not backup_path.exists():
            raise ValueError(f"Backup {backup_name} not found")
        
        model_path = self.model_base_path / model_name
        
        # Backup current version before restore
        if model_path.exists():
            temp_backup = self.backup_model(model_name, f"{model_name}_pre_restore")
        
        # Remove current version
        if model_path.exists():
            shutil.rmtree(model_path)
        
        # Restore from backup
        shutil.copytree(backup_path, model_path)
        
        print(f"Restored {model_name} from backup {backup_name}")
    
    def verify_integrity(self, model_name: str) -> bool:
        """Verify model integrity."""
        model_path = self.model_base_path / model_name
        if not model_path.exists():
            return False
        
        current_checksum = self._calculate_checksum(model_path)
        
        # Check against registered versions
        if model_name in self.versions:
            for version, info in self.versions[model_name].items():
                if info['checksum'] == current_checksum:
                    print(f"Model {model_name} matches version {version}")
                    return True
        
        print(f"Model {model_name} integrity check failed")
        return False
```

### 2. Model Update Process

```bash
#!/bin/bash
# update_models_offline.sh

set -e

UPDATE_BUNDLE=$1
MODEL_PATH="/opt/textnlp/models"
BACKUP_PATH="/opt/textnlp/model_backups"

if [ -z "$UPDATE_BUNDLE" ]; then
    echo "Usage: $0 <update_bundle.tar.gz>"
    exit 1
fi

echo "TextNLP Model Update Process"
echo "============================"

# 1. Verify update bundle
echo "1. Verifying update bundle..."
sha256sum -c ${UPDATE_BUNDLE}.sha256

# 2. Create backup
echo "2. Creating backup of current models..."
mkdir -p $BACKUP_PATH
BACKUP_NAME="models_backup_$(date +%Y%m%d_%H%M%S)"
cp -r $MODEL_PATH $BACKUP_PATH/$BACKUP_NAME

# 3. Extract update bundle
echo "3. Extracting update bundle..."
TEMP_DIR=$(mktemp -d)
tar -xzf $UPDATE_BUNDLE -C $TEMP_DIR

# 4. Verify new models
echo "4. Verifying new models..."
cd $TEMP_DIR/models
for model in */; do
    if [ -f "${model}config.json" ]; then
        echo "  ✓ Found model: $model"
    else
        echo "  ✗ Invalid model: $model"
        exit 1
    fi
done

# 5. Stop services
echo "5. Stopping services..."
sudo systemctl stop textnlp

# 6. Apply updates
echo "6. Applying model updates..."
for model in */; do
    model_name=${model%/}
    if [ -d "$MODEL_PATH/$model_name" ]; then
        echo "  Updating $model_name..."
        rm -rf "$MODEL_PATH/$model_name"
    else
        echo "  Adding new model $model_name..."
    fi
    cp -r "$model" "$MODEL_PATH/"
done

# 7. Update registry
echo "7. Updating model registry..."
cp registry.json $MODEL_PATH/

# 8. Verify installation
echo "8. Verifying updated models..."
python3 << EOF
import sys
sys.path.append('/opt/textnlp')
from models.offline_loader import OfflineModelLoader

loader = OfflineModelLoader('$MODEL_PATH')
models = loader.list_available_models()
print(f"Available models: {models}")

for model in models:
    try:
        loader.load_model(model)
        print(f"  ✓ {model} loads successfully")
    except Exception as e:
        print(f"  ✗ {model} failed to load: {e}")
        sys.exit(1)
EOF

# 9. Start services
echo "9. Starting services..."
sudo systemctl start textnlp

# 10. Cleanup
rm -rf $TEMP_DIR

echo ""
echo "Model update complete!"
echo "Backup saved to: $BACKUP_PATH/$BACKUP_NAME"
echo ""
echo "To rollback if needed:"
echo "  sudo systemctl stop textnlp"
echo "  rm -rf $MODEL_PATH/*"
echo "  cp -r $BACKUP_PATH/$BACKUP_NAME/* $MODEL_PATH/"
echo "  sudo systemctl start textnlp"
```

## Updates and Maintenance

### 1. Offline Update Bundle Creation

```bash
#!/bin/bash
# create_update_bundle.sh

set -e

CURRENT_VERSION=$1
NEW_VERSION=$2
UPDATE_TYPE=$3  # patch, minor, major

if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <current_version> <new_version> <update_type>"
    echo "Example: $0 1.0.0 1.0.1 patch"
    exit 1
fi

BUNDLE_DIR="textnlp-update-${NEW_VERSION}"
BUNDLE_NAME="textnlp-update-${CURRENT_VERSION}-to-${NEW_VERSION}.tar.gz"

echo "Creating Update Bundle: ${CURRENT_VERSION} → ${NEW_VERSION}"
echo "================================================"

mkdir -p ${BUNDLE_DIR}/{source,packages,models,scripts,migrations}

# 1. Determine what needs updating
echo "1. Analyzing changes..."

# Git diff for source code changes
if [ -d .git ]; then
    git diff --name-only v${CURRENT_VERSION}..v${NEW_VERSION} > ${BUNDLE_DIR}/changed_files.txt
fi

# 2. Package source code updates
echo "2. Packaging source code updates..."
if [ "$UPDATE_TYPE" != "models_only" ]; then
    rsync -av --files-from=${BUNDLE_DIR}/changed_files.txt . ${BUNDLE_DIR}/source/
fi

# 3. Package dependency updates
echo "3. Checking for dependency updates..."
pip install pip-tools
pip-compile requirements.txt -o requirements-new.txt
diff requirements-locked.txt requirements-new.txt > ${BUNDLE_DIR}/dependency_changes.txt || true

if [ -s ${BUNDLE_DIR}/dependency_changes.txt ]; then
    echo "  Downloading updated packages..."
    pip download -r requirements-new.txt -d ${BUNDLE_DIR}/packages/
fi

# 4. Package model updates (if any)
echo "4. Checking for model updates..."
# Compare model checksums
python3 << EOF
import json
import hashlib
from pathlib import Path

def calculate_checksum(path):
    hasher = hashlib.sha256()
    for file in sorted(Path(path).rglob('*')):
        if file.is_file():
            with open(file, 'rb') as f:
                hasher.update(f.read())
    return hasher.hexdigest()

old_models = json.load(open('models/registry-${CURRENT_VERSION}.json'))
new_models = json.load(open('models/registry.json'))

updated_models = []
for model, info in new_models.items():
    if model not in old_models or info['checksum'] != old_models[model]['checksum']:
        updated_models.append(model)
        print(f"  Model updated: {model}")

with open('${BUNDLE_DIR}/updated_models.json', 'w') as f:
    json.dump(updated_models, f)
EOF

# Copy updated models
while IFS= read -r model; do
    cp -r models/$model ${BUNDLE_DIR}/models/
done < <(python3 -c "import json; print('\n'.join(json.load(open('${BUNDLE_DIR}/updated_models.json'))))")

# 5. Create migration scripts
echo "5. Creating migration scripts..."

cat > ${BUNDLE_DIR}/migrations/migrate_${CURRENT_VERSION}_to_${NEW_VERSION}.sql << EOF
-- Database migrations for ${CURRENT_VERSION} to ${NEW_VERSION}
BEGIN;

-- Add your migrations here

COMMIT;
EOF

# 6. Create update script
cat > ${BUNDLE_DIR}/apply_update.sh << 'UPDATE_SCRIPT'
#!/bin/bash
set -e

echo "Applying TextNLP Update"
echo "======================"

# Verify current version
CURRENT_VERSION=$(cat /opt/textnlp/VERSION)
if [ "$CURRENT_VERSION" != "${CURRENT_VERSION}" ]; then
    echo "ERROR: Current version mismatch. Expected ${CURRENT_VERSION}, found $CURRENT_VERSION"
    exit 1
fi

# Create backup
echo "Creating backup..."
/opt/textnlp/scripts/backup_all.sh

# Stop services
echo "Stopping services..."
sudo systemctl stop textnlp

# Apply source updates
if [ -d source ]; then
    echo "Applying source code updates..."
    cp -r source/* /opt/textnlp/
fi

# Apply package updates
if [ -d packages ] && [ "$(ls -A packages)" ]; then
    echo "Updating packages..."
    source /opt/textnlp/venv/bin/activate
    pip install --no-index --find-links packages --upgrade -r source/requirements.txt
fi

# Apply model updates
if [ -d models ] && [ "$(ls -A models)" ]; then
    echo "Updating models..."
    cp -r models/* /opt/textnlp/models/
fi

# Run migrations
if [ -f migrations/*.sql ]; then
    echo "Running database migrations..."
    psql -U textnlp -d textnlp -f migrations/*.sql
fi

# Update version file
echo "${NEW_VERSION}" > /opt/textnlp/VERSION

# Start services
echo "Starting services..."
sudo systemctl start textnlp

echo "Update complete! New version: ${NEW_VERSION}"
UPDATE_SCRIPT

chmod +x ${BUNDLE_DIR}/apply_update.sh

# 7. Create update notes
cat > ${BUNDLE_DIR}/RELEASE_NOTES.md << EOF
# TextNLP Update: ${CURRENT_VERSION} to ${NEW_VERSION}

## Update Type: ${UPDATE_TYPE}

## Changes
$(git log --oneline v${CURRENT_VERSION}..v${NEW_VERSION} 2>/dev/null || echo "See changed_files.txt")

## Update Instructions
1. Transfer this bundle to the air-gapped system
2. Verify checksums: sha256sum -c update_bundle.sha256
3. Extract: tar -xzf ${BUNDLE_NAME}
4. Apply: cd ${BUNDLE_DIR} && sudo ./apply_update.sh

## Rollback Instructions
If the update fails, rollback using:
sudo /opt/textnlp/scripts/restore_backup.sh

## Verification
After update, verify:
- Service status: sudo systemctl status textnlp
- API health: curl http://localhost:8000/health
- Version: curl http://localhost:8000/api/v1/version
EOF

# 8. Create checksums
cd ${BUNDLE_DIR}
find . -type f -exec sha256sum {} \; > ../update_bundle.sha256
cd ..

# 9. Create bundle
tar -czf ${BUNDLE_NAME} ${BUNDLE_DIR}

echo ""
echo "Update bundle created: ${BUNDLE_NAME}"
echo "Size: $(du -h ${BUNDLE_NAME} | cut -f1)"
```

### 2. Maintenance Scripts

```python
# scripts/maintenance.py
"""Maintenance utilities for air-gapped TextNLP."""

import click
import psutil
import shutil
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

@click.group()
def cli():
    """TextNLP maintenance utilities."""
    pass

@cli.command()
def health_check():
    """Perform comprehensive health check."""
    print("TextNLP Health Check")
    print("===================")
    
    checks = {
        "System Resources": check_system_resources(),
        "Database": check_database(),
        "Models": check_models(),
        "Services": check_services(),
        "Storage": check_storage(),
        "Logs": check_logs()
    }
    
    all_healthy = True
    for component, result in checks.items():
        status = "✓" if result['healthy'] else "✗"
        print(f"{status} {component}: {result['message']}")
        if not result['healthy']:
            all_healthy = False
            for detail in result.get('details', []):
                print(f"  - {detail}")
    
    return 0 if all_healthy else 1

def check_system_resources() -> Dict:
    """Check system resource usage."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    issues = []
    if cpu_percent > 90:
        issues.append(f"High CPU usage: {cpu_percent}%")
    if memory.percent > 90:
        issues.append(f"High memory usage: {memory.percent}%")
    if disk.percent > 90:
        issues.append(f"Low disk space: {disk.percent}% used")
    
    return {
        'healthy': len(issues) == 0,
        'message': 'OK' if not issues else f"{len(issues)} issues found",
        'details': issues
    }

def check_database() -> Dict:
    """Check database health."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            database="textnlp",
            user="textnlp",
            password="changeme"  # Load from env
        )
        
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
        table_count = cur.fetchone()[0]
        
        cur.execute("SELECT pg_database_size('textnlp')")
        db_size = cur.fetchone()[0]
        
        conn.close()
        
        return {
            'healthy': True,
            'message': f'{table_count} tables, {db_size / 1024 / 1024:.1f} MB'
        }
    except Exception as e:
        return {
            'healthy': False,
            'message': 'Database connection failed',
            'details': [str(e)]
        }

def check_models() -> Dict:
    """Check model availability."""
    model_path = Path('/opt/textnlp/models')
    if not model_path.exists():
        return {
            'healthy': False,
            'message': 'Model directory not found'
        }
    
    models = []
    issues = []
    
    for model_dir in model_path.iterdir():
        if model_dir.is_dir() and (model_dir / 'config.json').exists():
            models.append(model_dir.name)
            
            # Check model integrity
            required_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json']
            for req_file in required_files:
                if not (model_dir / req_file).exists():
                    issues.append(f"{model_dir.name}: missing {req_file}")
    
    return {
        'healthy': len(issues) == 0,
        'message': f'{len(models)} models available',
        'details': issues
    }

def check_services() -> Dict:
    """Check service status."""
    services = ['textnlp', 'postgresql', 'redis', 'nginx']
    issues = []
    
    for service in services:
        result = subprocess.run(
            ['systemctl', 'is-active', service],
            capture_output=True,
            text=True
        )
        if result.stdout.strip() != 'active':
            issues.append(f"{service} is not active")
    
    return {
        'healthy': len(issues) == 0,
        'message': 'All services running' if not issues else f'{len(issues)} services down',
        'details': issues
    }

def check_storage() -> Dict:
    """Check storage usage and cleanup opportunities."""
    paths = {
        '/opt/textnlp/logs': 'Logs',
        '/opt/textnlp/cache': 'Cache',
        '/var/lib/postgresql': 'Database',
        '/tmp': 'Temp files'
    }
    
    total_size = 0
    details = []
    
    for path, name in paths.items():
        if Path(path).exists():
            size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
            total_size += size
            size_mb = size / 1024 / 1024
            if size_mb > 1000:  # More than 1GB
                details.append(f"{name}: {size_mb:.1f} MB (consider cleanup)")
    
    return {
        'healthy': len(details) == 0,
        'message': f'Total usage: {total_size / 1024 / 1024:.1f} MB',
        'details': details
    }

def check_logs() -> Dict:
    """Check for errors in logs."""
    log_path = Path('/opt/textnlp/logs')
    if not log_path.exists():
        return {
            'healthy': True,
            'message': 'No logs found'
        }
    
    errors = []
    for log_file in log_path.glob('*.log'):
        if log_file.stat().st_mtime > time.time() - 86400:  # Last 24 hours
            with open(log_file) as f:
                for line in f:
                    if 'ERROR' in line or 'CRITICAL' in line:
                        errors.append(f"{log_file.name}: {line.strip()[:100]}")
    
    return {
        'healthy': len(errors) == 0,
        'message': 'No recent errors' if not errors else f'{len(errors)} errors found',
        'details': errors[:10]  # Show first 10 errors
    }

@cli.command()
@click.option('--days', default=30, help='Keep logs for N days')
def cleanup(days):
    """Clean up old files and logs."""
    print(f"Cleaning up files older than {days} days...")
    
    cleanup_paths = {
        '/opt/textnlp/logs': '*.log',
        '/opt/textnlp/cache': '*',
        '/tmp': 'textnlp-*'
    }
    
    cutoff_time = time.time() - (days * 86400)
    total_freed = 0
    
    for base_path, pattern in cleanup_paths.items():
        path = Path(base_path)
        if not path.exists():
            continue
        
        freed = 0
        for file in path.glob(pattern):
            if file.is_file() and file.stat().st_mtime < cutoff_time:
                size = file.stat().st_size
                file.unlink()
                freed += size
        
        if freed > 0:
            print(f"  {base_path}: freed {freed / 1024 / 1024:.1f} MB")
            total_freed += freed
    
    # Database cleanup
    print("  Running database vacuum...")
    subprocess.run(['psql', '-U', 'textnlp', '-d', 'textnlp', '-c', 'VACUUM ANALYZE;'])
    
    print(f"\nTotal freed: {total_freed / 1024 / 1024:.1f} MB")

@cli.command()
def backup():
    """Create full system backup."""
    backup_dir = Path('/backup/textnlp')
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = backup_dir / f'backup_{timestamp}'
    backup_path.mkdir()
    
    print(f"Creating backup in {backup_path}...")
    
    # Backup components
    components = {
        'database': backup_database,
        'models': backup_models,
        'config': backup_config,
        'logs': backup_logs
    }
    
    for name, func in components.items():
        print(f"  Backing up {name}...")
        func(backup_path / name)
    
    # Create archive
    archive_path = backup_dir / f'textnlp_backup_{timestamp}.tar.gz'
    shutil.make_archive(
        str(archive_path).replace('.tar.gz', ''),
        'gztar',
        backup_path
    )
    
    # Cleanup temp directory
    shutil.rmtree(backup_path)
    
    print(f"\nBackup complete: {archive_path}")
    print(f"Size: {archive_path.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    cli()
```

## Troubleshooting Offline Issues

### 1. Common Issues and Solutions

#### Model Loading Failures
```python
# troubleshoot_models.py
"""Troubleshoot model loading issues."""

import sys
import os
from pathlib import Path
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def diagnose_model(model_path: str):
    """Diagnose issues with a model."""
    model_path = Path(model_path)
    
    print(f"Diagnosing model: {model_path}")
    print("=" * 50)
    
    # Check if path exists
    if not model_path.exists():
        print("✗ Model directory does not exist")
        return False
    
    # Check required files
    required_files = {
        'config.json': 'Model configuration',
        'pytorch_model.bin': 'Model weights (PyTorch)',
        'tokenizer_config.json': 'Tokenizer configuration',
        'tokenizer.json': 'Tokenizer data',
        'vocab.json': 'Vocabulary'
    }
    
    missing_files = []
    for file, description in required_files.items():
        file_path = model_path / file
        if file_path.exists():
            print(f"✓ {file}: {file_path.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print(f"✗ {file}: Missing - {description}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nMissing {len(missing_files)} required files")
        return False
    
    # Try to load model
    print("\nAttempting to load model...")
    try:
        # Set offline mode
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True
        )
        print("✓ Tokenizer loaded successfully")
        
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=torch.float32
        )
        print("✓ Model loaded successfully")
        
        # Test inference
        inputs = tokenizer("Test", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=10)
        print("✓ Inference test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Loading failed: {e}")
        
        # Common fixes
        print("\nPossible solutions:")
        print("1. Ensure all model files are from the same version")
        print("2. Check file permissions (chmod -R 755 model_directory)")
        print("3. Verify sufficient memory available")
        print("4. Try loading with low_cpu_mem_usage=True")
        
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python troubleshoot_models.py <model_path>")
        sys.exit(1)
    
    success = diagnose_model(sys.argv[1])
    sys.exit(0 if success else 1)
```

#### Network Connectivity Issues
```bash
#!/bin/bash
# diagnose_network.sh

echo "Network Diagnostics for Air-Gapped Environment"
echo "============================================="

# Check network interfaces
echo "1. Network Interfaces:"
ip addr show

# Check routing
echo -e "\n2. Routing Table:"
ip route show

# Check listening ports
echo -e "\n3. Listening Ports:"
ss -tlnp 2>/dev/null | grep -E ":(5432|6379|8000|80|443)"

# Check firewall
echo -e "\n4. Firewall Status:"
if command -v ufw &> /dev/null; then
    sudo ufw status
elif command -v firewall-cmd &> /dev/null; then
    sudo firewall-cmd --list-all
else
    echo "No firewall detected"
fi

# Check service connectivity
echo -e "\n5. Service Connectivity:"
services=(
    "localhost:5432:PostgreSQL"
    "localhost:6379:Redis"
    "localhost:8000:API"
    "localhost:80:Nginx"
)

for service in "${services[@]}"; do
    IFS=':' read -r host port name <<< "$service"
    if nc -z $host $port 2>/dev/null; then
        echo "✓ $name ($host:$port) - Connected"
    else
        echo "✗ $name ($host:$port) - Failed"
    fi
done

# Check DNS (should fail in air-gapped)
echo -e "\n6. External Connectivity (should fail):"
if ping -c 1 -W 1 8.8.8.8 &> /dev/null; then
    echo "⚠ WARNING: External network accessible (not air-gapped!)"
else
    echo "✓ No external network (properly air-gapped)"
fi
```

#### Performance Issues
```python
# diagnose_performance.py
"""Diagnose performance issues."""

import time
import psutil
import torch
from pathlib import Path
import numpy as np

def benchmark_system():
    """Run system benchmarks."""
    print("System Performance Benchmark")
    print("===========================")
    
    # CPU benchmark
    print("\n1. CPU Performance:")
    start = time.time()
    # Matrix multiplication benchmark
    a = np.random.rand(1000, 1000)
    b = np.random.rand(1000, 1000)
    c = np.dot(a, b)
    cpu_time = time.time() - start
    print(f"  Matrix multiplication (1000x1000): {cpu_time:.3f}s")
    print(f"  Estimated GFLOPS: {(2 * 1000**3) / (cpu_time * 1e9):.2f}")
    
    # Memory bandwidth
    print("\n2. Memory Bandwidth:")
    data_size = 100 * 1024 * 1024  # 100MB
    data = np.random.rand(data_size // 8)
    start = time.time()
    data_copy = data.copy()
    mem_time = time.time() - start
    bandwidth = (data_size * 2) / mem_time / 1024 / 1024 / 1024
    print(f"  Copy bandwidth: {bandwidth:.2f} GB/s")
    
    # Disk I/O
    print("\n3. Disk I/O Performance:")
    test_file = Path("/tmp/textnlp_io_test")
    data = np.random.bytes(10 * 1024 * 1024)  # 10MB
    
    # Write test
    start = time.time()
    with open(test_file, 'wb') as f:
        f.write(data)
    write_time = time.time() - start
    write_speed = 10 / write_time
    
    # Read test
    start = time.time()
    with open(test_file, 'rb') as f:
        _ = f.read()
    read_time = time.time() - start
    read_speed = 10 / read_time
    
    test_file.unlink()
    
    print(f"  Write speed: {write_speed:.1f} MB/s")
    print(f"  Read speed: {read_speed:.1f} MB/s")
    
    # Model loading benchmark
    print("\n4. Model Loading Performance:")
    model_path = Path("/opt/textnlp/models/gpt2")
    if model_path.exists():
        start = time.time()
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True
        )
        load_time = time.time() - start
        print(f"  GPT-2 load time: {load_time:.2f}s")
        
        # Inference benchmark
        start = time.time()
        inputs = torch.randint(0, 1000, (1, 50))
        with torch.no_grad():
            outputs = model(inputs)
        inference_time = time.time() - start
        print(f"  Inference time (50 tokens): {inference_time*1000:.1f}ms")
    
    # System resources
    print("\n5. Current System Resources:")
    print(f"  CPU Usage: {psutil.cpu_percent()}%")
    print(f"  Memory Usage: {psutil.virtual_memory().percent}%")
    print(f"  Disk Usage: {psutil.disk_usage('/').percent}%")
    
    # Recommendations
    print("\n6. Performance Recommendations:")
    if cpu_time > 1.0:
        print("  - CPU performance is below expected")
    if bandwidth < 10:
        print("  - Memory bandwidth is limited")
    if write_speed < 100 or read_speed < 100:
        print("  - Disk I/O is slow, consider SSD")
    if psutil.virtual_memory().percent > 80:
        print("  - High memory usage, consider adding RAM")

if __name__ == "__main__":
    benchmark_system()
```

## Security Considerations

### 1. Security Hardening Script

```bash
#!/bin/bash
# security_hardening.sh

echo "TextNLP Security Hardening for Air-Gapped Environment"
echo "===================================================="

# 1. File permissions
echo "1. Setting secure file permissions..."
chmod 700 /opt/textnlp
chmod 600 /opt/textnlp/.env
chmod 600 /opt/textnlp/config/*
find /opt/textnlp -type f -name "*.key" -exec chmod 600 {} \;
find /opt/textnlp -type f -name "*.pem" -exec chmod 600 {} \;

# 2. Remove unnecessary packages
echo "2. Removing unnecessary packages..."
apt-get remove -y telnet ftp
apt-get autoremove -y

# 3. Disable unnecessary services
echo "3. Disabling unnecessary services..."
systemctl disable avahi-daemon 2>/dev/null || true
systemctl disable cups 2>/dev/null || true

# 4. Configure auditd
echo "4. Configuring audit logging..."
cat >> /etc/audit/rules.d/textnlp.rules << EOF
# Monitor TextNLP configuration changes
-w /opt/textnlp/config -p wa -k textnlp_config
-w /opt/textnlp/.env -p wa -k textnlp_env

# Monitor model access
-w /opt/textnlp/models -p r -k textnlp_models

# Monitor authentication
-w /var/log/textnlp/auth.log -p wa -k textnlp_auth
EOF

systemctl restart auditd

# 5. Set up log rotation
echo "5. Configuring log rotation..."
cat > /etc/logrotate.d/textnlp << EOF
/opt/textnlp/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 textnlp textnlp
    sharedscripts
    postrotate
        systemctl reload textnlp >/dev/null 2>&1 || true
    endscript
}
EOF

# 6. Create security report
echo "6. Generating security report..."
cat > /opt/textnlp/security_report.txt << EOF
TextNLP Security Configuration Report
Generated: $(date)

1. File Permissions:
$(ls -la /opt/textnlp/.env)
$(ls -la /opt/textnlp/config/)

2. Service Status:
$(systemctl is-active textnlp)
$(systemctl is-active postgresql)
$(systemctl is-active redis)

3. Open Ports:
$(ss -tlnp | grep LISTEN)

4. User Accounts:
$(grep textnlp /etc/passwd)

5. Sudo Access:
$(grep textnlp /etc/sudoers.d/* 2>/dev/null || echo "No sudo access configured")

6. Audit Rules:
$(auditctl -l | grep textnlp)
EOF

echo ""
echo "Security hardening complete!"
echo "Review security report: /opt/textnlp/security_report.txt"
```

### 2. Access Control Configuration

```python
# security/access_control.py
"""Access control for air-gapped TextNLP."""

import jwt
import bcrypt
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import json
from pathlib import Path

class AccessController:
    """Manage user access in offline environment."""
    
    def __init__(self, db_path: str = "/opt/textnlp/auth.db"):
        self.db_path = db_path
        self.secret_key = self._load_secret_key()
        self._init_database()
    
    def _load_secret_key(self) -> str:
        """Load secret key from secure location."""
        key_file = Path("/opt/textnlp/config/jwt_secret.key")
        if not key_file.exists():
            # Generate new key
            import secrets
            key = secrets.token_urlsafe(32)
            key_file.write_text(key)
            key_file.chmod(0o600)
            return key
        return key_file.read_text().strip()
    
    def _init_database(self):
        """Initialize authentication database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                active BOOLEAN DEFAULT 1
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key_hash TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP,
                active BOOLEAN DEFAULT 1,
                FOREIGN KEY (username) REFERENCES users(username)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                username TEXT,
                action TEXT,
                resource TEXT,
                success BOOLEAN,
                ip_address TEXT,
                details TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_user(self, username: str, password: str, role: str = "user") -> bool:
        """Create a new user."""
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, password_hash.decode(), role)
            )
            conn.commit()
            conn.close()
            
            self.audit_log(username, "user_created", username, True)
            return True
            
        except sqlite3.IntegrityError:
            return False
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return JWT token."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT password_hash, role, active FROM users WHERE username = ?",
            (username,)
        )
        result = cur.fetchone()
        
        if not result or not result[2]:  # User not found or inactive
            self.audit_log(username, "login_failed", "auth", False)
            return None
        
        password_hash, role, _ = result
        
        if bcrypt.checkpw(password.encode(), password_hash.encode()):
            # Update last login
            conn.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = ?",
                (username,)
            )
            conn.commit()
            conn.close()
            
            # Generate JWT token
            payload = {
                'username': username,
                'role': role,
                'exp': datetime.utcnow() + timedelta(hours=8),
                'iat': datetime.utcnow()
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm='HS256')
            
            self.audit_log(username, "login_success", "auth", True)
            return token
        
        self.audit_log(username, "login_failed", "auth", False)
        return None
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def create_api_key(self, username: str, key_name: str) -> str:
        """Create API key for user."""
        import secrets
        api_key = secrets.token_urlsafe(32)
        key_hash = bcrypt.hashpw(api_key.encode(), bcrypt.gensalt())
        
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO api_keys (key_hash, username, name) VALUES (?, ?, ?)",
            (key_hash.decode(), username, key_name)
        )
        conn.commit()
        conn.close()
        
        self.audit_log(username, "api_key_created", key_name, True)
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[str]:
        """Verify API key and return username."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT key_hash, username, active FROM api_keys WHERE active = 1"
        )
        
        for key_hash, username, active in cur.fetchall():
            if bcrypt.checkpw(api_key.encode(), key_hash.encode()):
                # Update last used
                conn.execute(
                    "UPDATE api_keys SET last_used = CURRENT_TIMESTAMP WHERE key_hash = ?",
                    (key_hash,)
                )
                conn.commit()
                conn.close()
                return username
        
        conn.close()
        return None
    
    def audit_log(self, username: Optional[str], action: str, 
                  resource: str, success: bool, details: Dict = None):
        """Log audit event."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO audit_log 
            (username, action, resource, success, details) 
            VALUES (?, ?, ?, ?, ?)""",
            (
                username,
                action,
                resource,
                success,
                json.dumps(details) if details else None
            )
        )
        conn.commit()
        conn.close()
```

## Compliance and Auditing

### 1. Compliance Reporting

```python
# compliance/reporter.py
"""Generate compliance reports for air-gapped deployment."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import hashlib

class ComplianceReporter:
    """Generate compliance reports for various standards."""
    
    def __init__(self, config_path: str = "/opt/textnlp/config"):
        self.config_path = Path(config_path)
        self.audit_db = "/opt/textnlp/auth.db"
    
    def generate_report(self, standard: str = "all") -> Dict:
        """Generate compliance report."""
        reports = {
            'generated_at': datetime.now().isoformat(),
            'deployment_type': 'air-gapped',
            'standards': {}
        }
        
        if standard == "all" or standard == "security":
            reports['standards']['security'] = self._security_compliance()
        
        if standard == "all" or standard == "data_protection":
            reports['standards']['data_protection'] = self._data_protection_compliance()
        
        if standard == "all" or standard == "audit":
            reports['standards']['audit'] = self._audit_compliance()
        
        if standard == "all" or standard == "configuration":
            reports['standards']['configuration'] = self._configuration_compliance()
        
        return reports
    
    def _security_compliance(self) -> Dict:
        """Check security compliance."""
        checks = {}
        
        # File permissions
        checks['file_permissions'] = self._check_file_permissions()
        
        # Encryption
        checks['encryption'] = {
            'at_rest': self._check_encryption_at_rest(),
            'in_transit': self._check_encryption_in_transit()
        }
        
        # Access control
        checks['access_control'] = self._check_access_control()
        
        # Network isolation
        checks['network_isolation'] = self._check_network_isolation()
        
        return {
            'compliant': all(c.get('compliant', False) for c in checks.values()),
            'checks': checks
        }
    
    def _data_protection_compliance(self) -> Dict:
        """Check data protection compliance."""
        checks = {}
        
        # Data retention
        checks['data_retention'] = self._check_data_retention()
        
        # Data anonymization
        checks['anonymization'] = self._check_anonymization()
        
        # Backup procedures
        checks['backups'] = self._check_backup_compliance()
        
        return {
            'compliant': all(c.get('compliant', False) for c in checks.values()),
            'checks': checks
        }
    
    def _audit_compliance(self) -> Dict:
        """Check audit compliance."""
        checks = {}
        
        # Audit logging
        checks['audit_logging'] = self._check_audit_logging()
        
        # Log retention
        checks['log_retention'] = self._check_log_retention()
        
        # Log integrity
        checks['log_integrity'] = self._check_log_integrity()
        
        return {
            'compliant': all(c.get('compliant', False) for c in checks.values()),
            'checks': checks
        }
    
    def _check_file_permissions(self) -> Dict:
        """Check critical file permissions."""
        critical_files = {
            '/opt/textnlp/.env': 0o600,
            '/opt/textnlp/config/jwt_secret.key': 0o600,
            '/opt/textnlp/auth.db': 0o640
        }
        
        issues = []
        for file_path, expected_mode in critical_files.items():
            path = Path(file_path)
            if path.exists():
                actual_mode = path.stat().st_mode & 0o777
                if actual_mode != expected_mode:
                    issues.append(f"{file_path}: expected {oct(expected_mode)}, got {oct(actual_mode)}")
            else:
                issues.append(f"{file_path}: file not found")
        
        return {
            'compliant': len(issues) == 0,
            'issues': issues
        }
    
    def _check_network_isolation(self) -> Dict:
        """Verify network isolation."""
        import subprocess
        
        # Check for external connectivity
        result = subprocess.run(
            ['ping', '-c', '1', '-W', '1', '8.8.8.8'],
            capture_output=True
        )
        
        external_access = result.returncode == 0
        
        return {
            'compliant': not external_access,
            'external_access': external_access,
            'message': 'System is properly air-gapped' if not external_access else 'WARNING: External network access detected'
        }
    
    def _check_audit_logging(self) -> Dict:
        """Check audit logging status."""
        conn = sqlite3.connect(self.audit_db)
        cur = conn.cursor()
        
        # Check recent audit entries
        cur.execute("""
            SELECT COUNT(*) FROM audit_log 
            WHERE timestamp > datetime('now', '-1 day')
        """)
        recent_logs = cur.fetchone()[0]
        
        # Check log categories
        cur.execute("""
            SELECT DISTINCT action FROM audit_log 
            WHERE timestamp > datetime('now', '-7 days')
        """)
        log_types = [row[0] for row in cur.fetchall()]
        
        conn.close()
        
        required_types = ['login_success', 'login_failed', 'api_access', 'model_inference']
        missing_types = [t for t in required_types if t not in log_types]
        
        return {
            'compliant': recent_logs > 0 and len(missing_types) == 0,
            'recent_logs': recent_logs,
            'missing_log_types': missing_types
        }
    
    def generate_audit_report(self, start_date: datetime, end_date: datetime) -> Dict:
        """Generate detailed audit report."""
        conn = sqlite3.connect(self.audit_db)
        cur = conn.cursor()
        
        # Get audit events
        cur.execute("""
            SELECT timestamp, username, action, resource, success, details
            FROM audit_log
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        """, (start_date.isoformat(), end_date.isoformat()))
        
        events = []
        for row in cur.fetchall():
            events.append({
                'timestamp': row[0],
                'username': row[1],
                'action': row[2],
                'resource': row[3],
                'success': bool(row[4]),
                'details': json.loads(row[5]) if row[5] else None
            })
        
        # Summary statistics
        cur.execute("""
            SELECT 
                COUNT(*) as total_events,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_events,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_events,
                COUNT(DISTINCT username) as unique_users,
                COUNT(DISTINCT action) as unique_actions
            FROM audit_log
            WHERE timestamp BETWEEN ? AND ?
        """, (start_date.isoformat(), end_date.isoformat()))
        
        stats = cur.fetchone()
        
        conn.close()
        
        return {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_events': stats[0],
                'successful_events': stats[1],
                'failed_events': stats[2],
                'unique_users': stats[3],
                'unique_actions': stats[4]
            },
            'events': events[:1000]  # Limit to recent 1000
        }
```

### 2. Automated Compliance Checks

```bash
#!/bin/bash
# compliance_check.sh

set -e

echo "TextNLP Compliance Check"
echo "======================="
echo "Date: $(date)"
echo ""

REPORT_DIR="/opt/textnlp/compliance_reports"
mkdir -p $REPORT_DIR

# Run Python compliance checker
python3 << 'EOF' > $REPORT_DIR/compliance_$(date +%Y%m%d).json
import sys
sys.path.append('/opt/textnlp')
from compliance.reporter import ComplianceReporter

reporter = ComplianceReporter()
report = reporter.generate_report("all")

import json
print(json.dumps(report, indent=2))
EOF

# Check system compliance
echo "System Compliance Checks:"
echo "========================"

# 1. Check for unauthorized software
echo -n "1. Unauthorized software: "
if dpkg -l | grep -E "(telnet|ftp|nc|nmap)" > /dev/null; then
    echo "FAIL - Found unauthorized packages"
else
    echo "PASS"
fi

# 2. Check for weak passwords
echo -n "2. Password policy: "
if grep -E "^PASS_MIN_LEN\s+12" /etc/login.defs > /dev/null; then
    echo "PASS"
else
    echo "FAIL - Minimum password length < 12"
fi

# 3. Check audit daemon
echo -n "3. Audit daemon: "
if systemctl is-active auditd > /dev/null; then
    echo "PASS"
else
    echo "FAIL - Audit daemon not running"
fi

# 4. Check log encryption
echo -n "4. Log encryption: "
if [ -f "/opt/textnlp/logs/.encrypted" ]; then
    echo "PASS"
else
    echo "WARNING - Logs not encrypted"
fi

# 5. Check backup encryption
echo -n "5. Backup encryption: "
if [ -f "/backup/textnlp/.encrypted" ]; then
    echo "PASS"
else
    echo "WARNING - Backups not encrypted"
fi

# Generate HTML report
python3 << 'EOF'
import json
from pathlib import Path
from datetime import datetime

report_file = Path(f"/opt/textnlp/compliance_reports/compliance_{datetime.now().strftime('%Y%m%d')}.json")
with open(report_file) as f:
    report = json.load(f)

html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>TextNLP Compliance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .compliant {{ color: green; }}
        .non-compliant {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>TextNLP Compliance Report</h1>
    <p>Generated: {report['generated_at']}</p>
    <p>Deployment Type: {report['deployment_type']}</p>
    
    <h2>Compliance Summary</h2>
    <table>
        <tr>
            <th>Standard</th>
            <th>Status</th>
            <th>Details</th>
        </tr>
"""

for standard, details in report['standards'].items():
    status = "compliant" if details['compliant'] else "non-compliant"
    html += f"""
        <tr>
            <td>{standard.replace('_', ' ').title()}</td>
            <td class="{status}">{'✓ Compliant' if details['compliant'] else '✗ Non-Compliant'}</td>
            <td>{len(details.get('checks', {}))} checks performed</td>
        </tr>
    """

html += """
    </table>
</body>
</html>
"""

output_file = Path(f"/opt/textnlp/compliance_reports/compliance_{datetime.now().strftime('%Y%m%d')}.html")
output_file.write_text(html)
print(f"HTML report generated: {output_file}")
EOF

echo ""
echo "Compliance check complete!"
echo "Reports saved to: $REPORT_DIR"
```

## Best Practices Summary

1. **Pre-Deployment Planning**
   - Thoroughly analyze all dependencies
   - Test deployment process in similar environment
   - Create comprehensive documentation
   - Plan for update procedures

2. **Security**
   - Use encrypted transfer methods
   - Verify all file checksums
   - Implement strong access controls
   - Regular security audits

3. **Maintenance**
   - Regular health checks
   - Automated backup procedures
   - Clear rollback procedures
   - Documented troubleshooting guides

4. **Documentation**
   - Keep offline copies of all documentation
   - Include troubleshooting guides
   - Document all configuration changes
   - Maintain change logs

5. **Testing**
   - Test all functionality after deployment
   - Verify performance meets requirements
   - Test backup and restore procedures
   - Regular disaster recovery drills

This comprehensive guide ensures successful deployment and operation of TextNLP in air-gapped environments while maintaining security, compliance, and operational efficiency.