#!/bin/bash
# TextNLP Development Environment Setup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}TextNLP Development Setup${NC}"
echo -e "${BLUE}================================${NC}"

# Function to check command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        required_version="3.8"
        if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
            echo -e "${GREEN}✓ Python $python_version detected${NC}"
            return 0
        else
            echo -e "${RED}✗ Python $python_version is below required version $required_version${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ Python 3 not found${NC}"
        return 1
    fi
}

# Function to setup Python virtual environment
setup_venv() {
    echo -e "\n${YELLOW}Setting up Python virtual environment...${NC}"
    
    if [ -d "$PROJECT_ROOT/venv" ]; then
        echo -e "${YELLOW}Virtual environment already exists. Activating...${NC}"
    else
        python3 -m venv "$PROJECT_ROOT/venv"
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    fi
    
    # Activate virtual environment
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    echo -e "${GREEN}✓ pip upgraded${NC}"
}

# Function to install Python dependencies
install_python_deps() {
    echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
    
    # Install base requirements
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements.txt"
        echo -e "${GREEN}✓ Base requirements installed${NC}"
    fi
    
    # Install development requirements
    if [ -f "$PROJECT_ROOT/requirements-dev.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements-dev.txt"
        echo -e "${GREEN}✓ Development requirements installed${NC}"
    fi
    
    # Install TextNLP in development mode
    pip install -e "$PROJECT_ROOT"
    echo -e "${GREEN}✓ TextNLP installed in development mode${NC}"
}

# Function to check GPU availability
check_gpu() {
    echo -e "\n${YELLOW}Checking GPU availability...${NC}"
    
    if command_exists nvidia-smi; then
        gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        echo -e "${GREEN}✓ $gpu_count GPU(s) detected${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv
        
        # Check CUDA
        if command_exists nvcc; then
            cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
            echo -e "${GREEN}✓ CUDA $cuda_version detected${NC}"
        else
            echo -e "${YELLOW}⚠ CUDA not found in PATH${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ No GPU detected. CPU mode will be used${NC}"
        
        # Create mock GPU config
        cat > "$PROJECT_ROOT/config/no_gpu.yaml" << EOF
gpu:
  enabled: false
  devices: []
development:
  mock_gpu: true
EOF
    fi
}

# Function to setup configuration files
setup_config() {
    echo -e "\n${YELLOW}Setting up configuration files...${NC}"
    
    # Create config directory if not exists
    mkdir -p "$PROJECT_ROOT/config"
    
    # Copy .env.example to .env if not exists
    if [ ! -f "$PROJECT_ROOT/config/.env" ]; then
        if [ -f "$PROJECT_ROOT/config/.env.example" ]; then
            cp "$PROJECT_ROOT/config/.env.example" "$PROJECT_ROOT/config/.env"
            echo -e "${GREEN}✓ Created .env from template${NC}"
            echo -e "${YELLOW}  Please update .env with your settings${NC}"
        fi
    else
        echo -e "${GREEN}✓ .env already exists${NC}"
    fi
    
    # Create necessary directories
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/models"
    mkdir -p "$PROJECT_ROOT/storage"/{models,outputs,temp}
    mkdir -p "$PROJECT_ROOT/tensorboard_logs"
    echo -e "${GREEN}✓ Created necessary directories${NC}"
}

# Function to setup development services
setup_services() {
    echo -e "\n${YELLOW}Checking development services...${NC}"
    
    # Check Docker
    if command_exists docker; then
        echo -e "${GREEN}✓ Docker detected${NC}"
        
        # Create docker-compose for dev services
        cat > "$PROJECT_ROOT/docker-compose.dev.yml" << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_USER: textnlp
      POSTGRES_PASSWORD: password
      POSTGRES_DB: textnlp
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
EOF
        echo -e "${GREEN}✓ Created docker-compose.dev.yml${NC}"
        
        # Create Prometheus config
        cat > "$PROJECT_ROOT/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'textnlp'
    static_configs:
      - targets: ['host.docker.internal:9090']
EOF
        echo -e "${GREEN}✓ Created prometheus.yml${NC}"
        
    else
        echo -e "${YELLOW}⚠ Docker not found. Install Docker for PostgreSQL and Redis${NC}"
    fi
}

# Function to install pre-commit hooks
setup_precommit() {
    echo -e "\n${YELLOW}Setting up pre-commit hooks...${NC}"
    
    if [ -f "$PROJECT_ROOT/.pre-commit-config.yaml" ]; then
        pre-commit install
        echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"
    else
        # Create pre-commit config
        cat > "$PROJECT_ROOT/.pre-commit-config.yaml" << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=88", "--extend-ignore=E203,W503"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
EOF
        pre-commit install
        echo -e "${GREEN}✓ Created .pre-commit-config.yaml and installed hooks${NC}"
    fi
}

# Function to download sample models
download_models() {
    echo -e "\n${YELLOW}Downloading sample models...${NC}"
    
    # Create models directory
    mkdir -p "$PROJECT_ROOT/models"
    
    # Create a Python script to download models
    cat > "$PROJECT_ROOT/download_models.py" << 'EOF'
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

models_dir = "./models"
models = ["gpt2", "gpt2-medium"]

for model_name in models:
    print(f"Downloading {model_name}...")
    model_path = os.path.join(models_dir, model_name)
    
    if not os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        print(f"✓ {model_name} downloaded")
    else:
        print(f"✓ {model_name} already exists")
EOF
    
    python "$PROJECT_ROOT/download_models.py"
    rm "$PROJECT_ROOT/download_models.py"
    echo -e "${GREEN}✓ Sample models downloaded${NC}"
}

# Function to create development scripts
create_dev_scripts() {
    echo -e "\n${YELLOW}Creating development scripts...${NC}"
    
    # Create run script
    cat > "$PROJECT_ROOT/run_dev.sh" << 'EOF'
#!/bin/bash
# Run TextNLP in development mode

source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
export ENVIRONMENT=development

# Start services if using Docker
if command -v docker >/dev/null 2>&1; then
    echo "Starting development services..."
    docker-compose -f docker-compose.dev.yml up -d
    sleep 5
fi

# Run the application
echo "Starting TextNLP API..."
python -m textnlp.api.app
EOF
    chmod +x "$PROJECT_ROOT/run_dev.sh"
    
    # Create test script
    cat > "$PROJECT_ROOT/run_tests.sh" << 'EOF'
#!/bin/bash
# Run TextNLP tests

source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
export ENVIRONMENT=testing

# Run tests with coverage
pytest --cov=textnlp --cov-report=html --cov-report=term -v
EOF
    chmod +x "$PROJECT_ROOT/run_tests.sh"
    
    echo -e "${GREEN}✓ Development scripts created${NC}"
}

# Function to setup VSCode configuration
setup_vscode() {
    echo -e "\n${YELLOW}Setting up VSCode configuration...${NC}"
    
    mkdir -p "$PROJECT_ROOT/.vscode"
    
    # Create settings.json
    cat > "$PROJECT_ROOT/.vscode/settings.json" << 'EOF'
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": ["--max-line-length=88", "--extend-ignore=E203,W503"],
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "python.sortImports.args": ["--profile", "black"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true,
        "**/venv": true
    }
}
EOF
    
    # Create launch.json for debugging
    cat > "$PROJECT_ROOT/.vscode/launch.json" << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "TextNLP API",
            "type": "python",
            "request": "launch",
            "module": "textnlp.api.app",
            "env": {
                "ENVIRONMENT": "development",
                "PYTHONPATH": "${workspaceFolder}"
            },
            "console": "integratedTerminal"
        },
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        }
    ]
}
EOF
    
    echo -e "${GREEN}✓ VSCode configuration created${NC}"
}

# Function to display final instructions
show_instructions() {
    echo -e "\n${GREEN}================================${NC}"
    echo -e "${GREEN}Setup Complete!${NC}"
    echo -e "${GREEN}================================${NC}"
    
    echo -e "\n${BLUE}Next steps:${NC}"
    echo -e "1. Activate virtual environment:"
    echo -e "   ${YELLOW}source venv/bin/activate${NC}"
    
    echo -e "\n2. Update configuration:"
    echo -e "   ${YELLOW}edit config/.env${NC}"
    
    echo -e "\n3. Start development services (optional):"
    echo -e "   ${YELLOW}docker-compose -f docker-compose.dev.yml up -d${NC}"
    
    echo -e "\n4. Run the application:"
    echo -e "   ${YELLOW}./run_dev.sh${NC}"
    
    echo -e "\n5. Run tests:"
    echo -e "   ${YELLOW}./run_tests.sh${NC}"
    
    echo -e "\n${BLUE}Available endpoints:${NC}"
    echo -e "   API: http://localhost:8000"
    echo -e "   Docs: http://localhost:8000/docs"
    echo -e "   Metrics: http://localhost:9090/metrics"
    
    if [ -f "$PROJECT_ROOT/config/no_gpu.yaml" ]; then
        echo -e "\n${YELLOW}Note: No GPU detected. Running in CPU mode.${NC}"
    fi
}

# Main setup flow
main() {
    cd "$PROJECT_ROOT"
    
    # Check Python version
    if ! check_python_version; then
        echo -e "${RED}Please install Python 3.8 or higher${NC}"
        exit 1
    fi
    
    # Setup steps
    setup_venv
    install_python_deps
    check_gpu
    setup_config
    setup_services
    setup_precommit
    download_models
    create_dev_scripts
    setup_vscode
    
    # Show completion message
    show_instructions
}

# Run main function
main