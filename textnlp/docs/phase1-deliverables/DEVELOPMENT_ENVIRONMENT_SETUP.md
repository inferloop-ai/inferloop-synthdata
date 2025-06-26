# Development Environment Setup - TextNLP Platform

## Overview

This document provides comprehensive setup instructions for the TextNLP development environment, including local development, cloud SDKs, and NLP-specific tools.

## System Requirements

### Hardware Requirements
- **CPU**: Minimum 8 cores (16 recommended)
- **RAM**: Minimum 32GB (64GB recommended for large models)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional for local testing)
- **Storage**: 500GB SSD (1TB recommended)
- **Network**: Stable broadband connection

### Operating System Support
- **Linux**: Ubuntu 20.04/22.04 LTS (recommended)
- **macOS**: 12.0+ (Monterey or later)
- **Windows**: WSL2 with Ubuntu 20.04/22.04
- **Container**: Docker 20.10+

## Base Development Tools

### 1. Python Environment Setup
```bash
# Install Python 3.9+ (3.10 recommended)
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# Install pip and essential tools
curl https://bootstrap.pypa.io/get-pip.py | python3.10
pip install --upgrade pip setuptools wheel

# Install pyenv for Python version management
curl https://pyenv.run | bash

# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Install Python versions
pyenv install 3.9.16
pyenv install 3.10.11
pyenv install 3.11.4
pyenv global 3.10.11
```

### 2. Virtual Environment Setup
```bash
# Create project directory
mkdir -p ~/projects/textnlp
cd ~/projects/textnlp

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Upgrade pip in virtual environment
pip install --upgrade pip
```

### 3. Git Configuration
```bash
# Install Git
sudo apt install git

# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@company.com"
git config --global init.defaultBranch main

# Configure Git credentials
git config --global credential.helper store

# SSH key setup for GitHub/GitLab
ssh-keygen -t ed25519 -C "your.email@company.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

## TextNLP Development Setup

### 1. Clone Repository
```bash
# Clone TextNLP repository
git clone git@github.com:company/inferloop-synthdata.git
cd inferloop-synthdata/textnlp

# Install in development mode
pip install -e ".[dev,all]"
```

### 2. Install Dependencies
```bash
# Core dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# NLP-specific dependencies
pip install transformers==4.35.0
pip install torch==2.1.0
pip install accelerate==0.24.1
pip install datasets==2.14.6
pip install sentencepiece==0.1.99
pip install tokenizers==0.14.1

# Validation tools
pip install nltk==3.8.1
pip install rouge-score==0.1.2
pip install sacrebleu==2.3.1
pip install bert-score==0.3.13
```

### 3. Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run against all files
pre-commit run --all-files

# .pre-commit-config.yaml
repos:
  - repo: https://github.com/ambv/black
    rev: 23.10.1
    hooks:
      - id: black
        language_version: python3.10
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.1
    hooks:
      - id: mypy
```

## Cloud SDK Installation

### 1. AWS CLI
```bash
# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify installation
aws --version

# Install additional tools
pip install boto3 awscli-local

# Install AWS SAM CLI for serverless
pip install aws-sam-cli

# Install eksctl for EKS
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin
```

### 2. Google Cloud SDK
```bash
# Add Cloud SDK repository
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install Cloud SDK
sudo apt update
sudo apt install google-cloud-sdk

# Install additional components
gcloud components install kubectl
gcloud components install gke-gcloud-auth-plugin

# Python libraries
pip install google-cloud-storage google-cloud-aiplatform
```

### 3. Azure CLI
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Verify installation
az --version

# Install Azure Python SDK
pip install azure-mgmt-compute azure-mgmt-storage azure-identity

# Install Azure Functions Core Tools
curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg
sudo mv microsoft.gpg /etc/apt/trusted.gpg.d/microsoft.gpg
sudo sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/microsoft-ubuntu-$(lsb_release -cs)-prod $(lsb_release -cs) main" > /etc/apt/sources.list.d/dotnetdev.list'
sudo apt update
sudo apt install azure-functions-core-tools-4
```

### 4. Kubernetes Tools
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install k9s for cluster management
curl -sS https://webinstall.dev/k9s | bash

# Install kubectx and kubens
sudo git clone https://github.com/ahmetb/kubectx /opt/kubectx
sudo ln -s /opt/kubectx/kubectx /usr/local/bin/kubectx
sudo ln -s /opt/kubectx/kubens /usr/local/bin/kubens
```

## Container Development

### 1. Docker Installation
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. Container Registry Setup
```bash
# Docker Hub login
docker login

# AWS ECR login
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# GCP Container Registry login
gcloud auth configure-docker

# Azure Container Registry login
az acr login --name textnlpregistry
```

## GPU Development Setup

### 1. NVIDIA Driver Installation
```bash
# Add NVIDIA repository
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install NVIDIA driver
sudo apt install nvidia-driver-525

# Verify installation
nvidia-smi
```

### 2. CUDA Toolkit
```bash
# Install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-11-8

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. PyTorch with CUDA
```bash
# Install PyTorch with CUDA support
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"
```

## IDE Setup

### 1. VS Code Configuration
```bash
# Install VS Code
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt update
sudo apt install code

# Install extensions
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.black-formatter
code --install-extension ms-azuretools.vscode-docker
code --install-extension ms-kubernetes-tools.vscode-kubernetes-tools
code --install-extension amazonwebservices.aws-toolkit-vscode
code --install-extension googlecloudtools.cloudcode
```

### 2. VS Code Settings
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "tests"
    ]
}
```

### 3. PyCharm Configuration
```bash
# Install PyCharm Professional
sudo snap install pycharm-professional --classic

# Or download from JetBrains
wget https://download.jetbrains.com/python/pycharm-professional-2023.2.4.tar.gz
tar -xzf pycharm-professional-2023.2.4.tar.gz
sudo mv pycharm-2023.2.4 /opt/
/opt/pycharm-2023.2.4/bin/pycharm.sh
```

## Local Testing Environment

### 1. Local Kubernetes
```bash
# Install Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Start Minikube with GPU support
minikube start --cpus=4 --memory=8192 --gpu

# Or install Kind
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Create Kind cluster
kind create cluster --name textnlp-dev
```

### 2. Local Model Storage
```bash
# Install MinIO for S3-compatible storage
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
sudo mv minio /usr/local/bin/

# Run MinIO
minio server ~/minio-data --console-address ":9001"

# Create bucket for models
mc alias set local http://localhost:9000 minioadmin minioadmin
mc mb local/textnlp-models
mc mb local/textnlp-datasets
```

### 3. Local Database
```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE textnlp_dev;
CREATE USER textnlp_user WITH ENCRYPTED PASSWORD 'dev_password';
GRANT ALL PRIVILEGES ON DATABASE textnlp_dev TO textnlp_user;
\q

# Install Redis
sudo apt install redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

## Environment Variables

### 1. Create .env file
```bash
# .env.development
# Application
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=postgresql://textnlp_user:dev_password@localhost:5432/textnlp_dev
REDIS_URL=redis://localhost:6379/0

# Storage
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=textnlp-models

# ML/NLP
TRANSFORMERS_CACHE=/home/user/.cache/huggingface
TORCH_HOME=/home/user/.cache/torch
MODEL_CACHE_DIR=/home/user/models

# Cloud (for testing)
USE_LOCAL_STORAGE=true
USE_LOCAL_COMPUTE=true
```

### 2. Load environment
```bash
# Install python-dotenv
pip install python-dotenv

# Create env loader script
cat > load_env.py << 'EOF'
from dotenv import load_dotenv
import os

env = os.getenv('ENVIRONMENT', 'development')
load_dotenv(f'.env.{env}')
EOF
```

## Testing Setup

### 1. Install Testing Tools
```bash
# Testing frameworks
pip install pytest pytest-cov pytest-asyncio pytest-mock
pip install pytest-benchmark pytest-timeout pytest-xdist

# Code quality
pip install black isort flake8 mypy
pip install pylint bandit safety

# Documentation
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
```

### 2. Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=textnlp --cov-report=html

# Run specific test file
pytest tests/test_generators.py

# Run in parallel
pytest -n auto

# Run with markers
pytest -m "not slow"
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Set memory fraction
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

2. **Model Download Issues**
```bash
# Use local cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Change cache directory
export HF_HOME=/path/to/large/storage
```

3. **Docker Permission Denied**
```bash
# Fix docker permissions
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

## Development Workflow

### 1. Daily Workflow
```bash
# Start development environment
cd ~/projects/textnlp
source venv/bin/activate
export $(cat .env.development | xargs)

# Pull latest changes
git pull origin main

# Create feature branch
git checkout -b feature/new-nlp-model

# Run tests before committing
pytest
black .
isort .
mypy textnlp

# Commit changes
git add .
git commit -m "feat: add new NLP model"
git push origin feature/new-nlp-model
```

### 2. Model Development
```bash
# Download model for local development
python -c "from transformers import AutoModel, AutoTokenizer; \
          model = AutoModel.from_pretrained('gpt2'); \
          tokenizer = AutoTokenizer.from_pretrained('gpt2')"

# Test model locally
python scripts/test_model.py --model gpt2 --prompt "Test prompt"

# Profile model performance
python -m cProfile -o profile.stats scripts/benchmark_model.py
```

## Completion Checklist

- [ ] Python 3.10+ installed
- [ ] Virtual environment created
- [ ] Git configured with SSH keys
- [ ] TextNLP repository cloned
- [ ] Dependencies installed
- [ ] Pre-commit hooks configured
- [ ] AWS CLI installed and configured
- [ ] GCP SDK installed and configured
- [ ] Azure CLI installed and configured
- [ ] Kubernetes tools installed
- [ ] Docker installed and running
- [ ] GPU drivers installed (if applicable)
- [ ] IDE configured with extensions
- [ ] Local testing environment ready
- [ ] Environment variables configured
- [ ] Tests passing locally