# TextNLP Localhost Development Setup Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Detailed Setup Instructions](#detailed-setup-instructions)
3. [Development Configurations](#development-configurations)
4. [IDE Setup](#ide-setup)
5. [Testing Environment](#testing-environment)
6. [Debugging](#debugging)
7. [Performance Tuning](#performance-tuning)
8. [Common Development Tasks](#common-development-tasks)

## Quick Start

### Prerequisites Check
```bash
# Check Python version (need 3.8+)
python3 --version

# Check available memory (need 8GB+)
free -h  # Linux
# or
sysctl hw.memsize  # macOS

# Check disk space (need 50GB+)
df -h
```

### One-Line Setup (Linux/macOS)
```bash
curl -sSL https://raw.githubusercontent.com/inferloop/textnlp/main/scripts/setup-dev.sh | bash
```

### Manual Quick Setup
```bash
# 1. Clone repository
git clone https://github.com/inferloop/inferloop-synthdata.git
cd inferloop-synthdata/textnlp

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Set up environment
cp .env.example .env
# Edit .env with your settings

# 5. Start services
docker-compose -f docker-compose.dev.yml up -d

# 6. Run application
python -m textnlp.api.app
```

## Detailed Setup Instructions

### 1. System Dependencies

#### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install Python and development tools
sudo apt install -y \
    python3.9 python3.9-dev python3.9-venv \
    python3-pip python3-setuptools \
    build-essential gcc g++ make \
    git curl wget vim

# Install system libraries
sudo apt install -y \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and dependencies
brew install python@3.9 git wget

# Install PostgreSQL client and Redis
brew install postgresql redis

# Install Docker Desktop
brew install --cask docker

# Start Docker Desktop
open /Applications/Docker.app
```

#### Windows (WSL2)
```powershell
# Enable WSL2
wsl --install

# Install Ubuntu from Microsoft Store
# Then follow Ubuntu instructions above inside WSL2
```

### 2. Repository Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/inferloop/inferloop-synthdata.git
cd inferloop-synthdata/textnlp

# If already cloned, init submodules
git submodule update --init --recursive

# Create necessary directories
mkdir -p logs models cache data/uploads data/exports
```

### 3. Python Environment

#### Virtual Environment Setup
```bash
# Create virtual environment
python3.9 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install package in development mode
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

#### Conda Environment (Alternative)
```bash
# Create conda environment
conda create -n textnlp python=3.9
conda activate textnlp

# Install dependencies
conda install -c conda-forge \
    numpy scipy pandas scikit-learn \
    pytorch torchvision torchaudio \
    transformers datasets

# Install remaining pip packages
pip install -e ".[dev,test,docs]"
```

### 4. Database Setup

#### PostgreSQL
```bash
# Start PostgreSQL container
docker run -d \
    --name textnlp-postgres \
    -e POSTGRES_USER=textnlp \
    -e POSTGRES_PASSWORD=devpassword \
    -e POSTGRES_DB=textnlp_dev \
    -p 5432:5432 \
    -v textnlp_postgres_data:/var/lib/postgresql/data \
    postgres:15

# Wait for PostgreSQL to start
sleep 5

# Create development database
docker exec -it textnlp-postgres psql -U textnlp -c "CREATE DATABASE textnlp_test;"

# Run migrations
alembic upgrade head

# Seed development data (optional)
python scripts/seed_dev_data.py
```

#### Redis
```bash
# Start Redis container
docker run -d \
    --name textnlp-redis \
    -p 6379:6379 \
    -v textnlp_redis_data:/data \
    redis:7-alpine \
    redis-server --appendonly yes --requirepass devpassword

# Test Redis connection
docker exec -it textnlp-redis redis-cli -a devpassword ping
```

### 5. Model Setup

#### Download Models
```bash
# Create model directory
mkdir -p models

# Download models using Python script
python scripts/download_models.py --models gpt2,distilgpt2

# Or manually download
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
models = ['gpt2', 'distilgpt2', 'gpt2-medium']
for model_name in models:
    print(f'Downloading {model_name}...')
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(f'models/{model_name}')
    tokenizer.save_pretrained(f'models/{model_name}')
"
```

#### Model Optimization for Development
```python
# optimize_dev_models.py
import torch
from transformers import AutoModelForCausalLM

# Load and optimize models for faster development
models = ['gpt2', 'distilgpt2']
for model_name in models:
    print(f"Optimizing {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(f"models/{model_name}")
    
    # Convert to half precision for faster inference
    if torch.cuda.is_available():
        model = model.half().cuda()
    
    # Enable torch.compile for PyTorch 2.0+
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode="reduce-overhead")
    
    # Save optimized model
    model.save_pretrained(f"models/{model_name}-optimized")
```

### 6. Environment Configuration

#### Development Environment File
```bash
# .env.development
# Application
APP_NAME=TextNLP-Dev
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=2
API_RELOAD=true
API_DOCS_ENABLED=true

# Database
DATABASE_URL=postgresql://textnlp:devpassword@localhost:5432/textnlp_dev
DATABASE_POOL_SIZE=5
DATABASE_ECHO=true

# Redis
REDIS_URL=redis://:devpassword@localhost:6379/0
REDIS_POOL_SIZE=5

# Models
MODEL_PATH=./models
MODEL_CACHE_SIZE=2
DEFAULT_MODEL=gpt2
MODEL_DEVICE=cpu  # or cuda if available

# Security (development only)
JWT_SECRET=dev-secret-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRY=3600
API_KEY=dev-api-key-change-in-production
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Features
ENABLE_SAFETY_CHECKS=true
ENABLE_METRICS=true
ENABLE_CACHING=true
ENABLE_RATE_LIMITING=false  # Disabled for development

# External Services (optional)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
HUGGINGFACE_TOKEN=

# Development Tools
PROFILING_ENABLED=true
SQL_ECHO=true
EXPLAIN_QUERIES=true
```

## Development Configurations

### 1. Docker Compose for Development

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    container_name: textnlp-postgres-dev
    environment:
      POSTGRES_USER: textnlp
      POSTGRES_PASSWORD: devpassword
      POSTGRES_DB: textnlp_dev
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-dev-db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U textnlp"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: textnlp-redis-dev
    command: redis-server --requirepass devpassword --loglevel debug
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "devpassword", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  mailhog:
    image: mailhog/mailhog
    container_name: textnlp-mailhog
    ports:
      - "1025:1025"  # SMTP
      - "8025:8025"  # Web UI
    
  adminer:
    image: adminer
    container_name: textnlp-adminer
    ports:
      - "8090:8080"
    environment:
      ADMINER_DEFAULT_SERVER: postgres
    depends_on:
      - postgres

  redis-commander:
    image: rediscommander/redis-commander:latest
    container_name: textnlp-redis-commander
    environment:
      REDIS_HOSTS: local:redis:6379:0:devpassword
    ports:
      - "8091:8081"
    depends_on:
      - redis

volumes:
  postgres_data:
  redis_data:
```

### 2. Development Server Configuration

```python
# config/development.py
from pathlib import Path

class DevelopmentConfig:
    # Debug settings
    DEBUG = True
    TESTING = False
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = 8000
    RELOAD = True
    WORKERS = 1  # Single worker for debugging
    
    # Database
    SQLALCHEMY_DATABASE_URI = "postgresql://textnlp:devpassword@localhost:5432/textnlp_dev"
    SQLALCHEMY_ECHO = True
    SQLALCHEMY_RECORD_QUERIES = True
    
    # Redis
    REDIS_URL = "redis://:devpassword@localhost:6379/0"
    
    # Logging
    LOG_LEVEL = "DEBUG"
    LOG_TO_FILE = True
    LOG_FILE = Path("logs/development.log")
    
    # Model configuration
    MODEL_PATH = Path("models")
    MODEL_CACHE_SIZE = 2
    MODEL_DEVICE = "cpu"  # Use "cuda" if GPU available
    
    # Development features
    PROFILING_ENABLED = True
    EXPLAIN_TEMPLATES_ENABLED = True
    DEBUG_TB_ENABLED = True
    DEBUG_TB_INTERCEPT_REDIRECTS = False
    
    # API Documentation
    API_DOCS_ENABLED = True
    API_TITLE = "TextNLP Development API"
    API_VERSION = "dev"
    
    # CORS (allow all in development)
    CORS_ORIGINS = ["*"]
    CORS_ALLOW_CREDENTIALS = True
    
    # Rate limiting (disabled in development)
    RATELIMIT_ENABLED = False
    
    # Hot reload for templates and static files
    TEMPLATES_AUTO_RELOAD = True
    SEND_FILE_MAX_AGE_DEFAULT = 0
```

### 3. Logging Configuration

```python
# config/logging_dev.py
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s.%(funcName)s (%(pathname)s:%(lineno)d): %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '%(levelname)s: %(message)s'
        },
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'logs/development.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': 'logs/errors.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        'textnlp': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        },
        'sqlalchemy.engine': {
            'level': 'INFO',  # Set to DEBUG to see SQL queries
            'handlers': ['console'],
            'propagate': False
        },
        'transformers': {
            'level': 'INFO',
            'handlers': ['console'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file', 'error_file']
    }
}
```

## IDE Setup

### Visual Studio Code

#### Extensions
```json
// .vscode/extensions.json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.black-formatter",
    "ms-python.isort",
    "ms-python.flake8",
    "ms-toolsai.jupyter",
    "visualstudioexptteam.vscodeintellicode",
    "redhat.vscode-yaml",
    "ms-azuretools.vscode-docker",
    "rangav.vscode-thunder-client",
    "gruntfuggly.todo-tree",
    "aaron-bond.better-comments",
    "streetsidesoftware.code-spell-checker"
  ]
}
```

#### Settings
```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Args": ["--max-line-length=88", "--extend-ignore=E203,W503"],
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "88"],
  "python.sortImports.provider": "isort",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["-v", "--tb=short"],
  "python.terminal.activateEnvironment": true,
  "python.analysis.typeCheckingMode": "strict",
  "python.analysis.autoImportCompletions": true,
  "python.analysis.autoSearchPaths": true,
  "python.analysis.extraPaths": ["${workspaceFolder}"],
  
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "editor.rulers": [88],
  
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/.mypy_cache": true,
    "**/venv": true,
    "**/.venv": true
  },
  
  "files.watcherExclude": {
    "**/__pycache__/**": true,
    "**/venv/**": true,
    "**/models/**": true,
    "**/data/**": true
  }
}
```

#### Launch Configurations
```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "TextNLP API",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "textnlp.api.app:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "debug"
      ],
      "jinja": true,
      "justMyCode": false,
      "env": {
        "PYTHONPATH": "${workspaceFolder}",
        "ENVIRONMENT": "development"
      }
    },
    {
      "name": "TextNLP CLI",
      "type": "python",
      "request": "launch",
      "module": "textnlp.cli",
      "args": ["generate", "--prompt", "Test prompt"],
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    {
      "name": "Debug Current Test",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": [
        "${file}",
        "-v",
        "--tb=short",
        "--no-cov"
      ],
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Python: Attach to Remote",
      "type": "python",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}",
          "remoteRoot": "/app"
        }
      ]
    }
  ]
}
```

### PyCharm Setup

#### Run Configurations
```xml
<!-- .idea/runConfigurations/TextNLP_API.xml -->
<component name="ProjectRunConfigurationManager">
  <configuration default="false" name="TextNLP API" type="PythonConfigurationType" factoryName="Python">
    <module name="textnlp" />
    <option name="INTERPRETER_OPTIONS" value="" />
    <option name="PARENT_ENVS" value="true" />
    <envs>
      <env name="PYTHONUNBUFFERED" value="1" />
      <env name="ENVIRONMENT" value="development" />
    </envs>
    <option name="SDK_HOME" value="$PROJECT_DIR$/venv/bin/python" />
    <option name="WORKING_DIRECTORY" value="$PROJECT_DIR$" />
    <option name="IS_MODULE_SDK" value="true" />
    <option name="ADD_CONTENT_ROOTS" value="true" />
    <option name="ADD_SOURCE_ROOTS" value="true" />
    <option name="SCRIPT_NAME" value="uvicorn" />
    <option name="PARAMETERS" value="textnlp.api.app:app --reload --host 0.0.0.0 --port 8000" />
    <option name="SHOW_COMMAND_LINE" value="false" />
    <option name="EMULATE_TERMINAL" value="false" />
    <option name="MODULE_MODE" value="true" />
  </configuration>
</component>
```

## Testing Environment

### Unit Testing Setup

```python
# conftest.py
import pytest
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import Mock, patch

from textnlp.database import Base
from textnlp.config import TestConfig

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_db():
    """Create a test database."""
    engine = create_engine(TestConfig.DATABASE_URL)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)

@pytest.fixture
def db_session(test_db):
    """Create a database session for tests."""
    SessionLocal = sessionmaker(bind=test_db)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()

@pytest.fixture
def mock_model():
    """Mock language model for testing."""
    model = Mock()
    model.generate.return_value = "Generated text"
    return model

@pytest.fixture
def client():
    """Create a test client."""
    from textnlp.api.app import app
    from fastapi.testclient import TestClient
    
    with TestClient(app) as client:
        yield client
```

### Integration Testing

```python
# tests/integration/test_api_integration.py
import pytest
from httpx import AsyncClient
from textnlp.api.app import app

@pytest.mark.asyncio
async def test_full_generation_flow():
    """Test complete text generation flow."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # 1. Health check
        response = await client.get("/health")
        assert response.status_code == 200
        
        # 2. Generate text
        response = await client.post(
            "/api/v1/generate",
            json={
                "prompt": "Write a story about AI",
                "max_length": 100,
                "model": "gpt2"
            },
            headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 200
        assert "generated_text" in response.json()
        
        # 3. Check metrics
        response = await client.get("/api/v1/metrics")
        assert response.status_code == 200
```

### Load Testing

```python
# tests/load/test_load.py
import asyncio
import aiohttp
import time
from statistics import mean, stdev

async def make_request(session, url, headers, data):
    """Make a single request and measure time."""
    start = time.time()
    try:
        async with session.post(url, json=data, headers=headers) as response:
            await response.text()
            return time.time() - start, response.status
    except Exception as e:
        return time.time() - start, 0

async def load_test(num_requests=100, concurrent=10):
    """Run load test."""
    url = "http://localhost:8000/api/v1/generate"
    headers = {"X-API-Key": "test-key"}
    data = {
        "prompt": "Test prompt",
        "max_length": 50,
        "model": "gpt2"
    }
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            if len(tasks) >= concurrent:
                done, tasks = await asyncio.wait(
                    tasks, 
                    return_when=asyncio.FIRST_COMPLETED
                )
            
            task = asyncio.create_task(
                make_request(session, url, headers, data)
            )
            tasks.append(task)
        
        # Wait for remaining tasks
        results = await asyncio.gather(*tasks)
    
    # Analyze results
    times = [r[0] for r in results if r[1] == 200]
    errors = [r for r in results if r[1] != 200]
    
    print(f"Successful requests: {len(times)}/{num_requests}")
    print(f"Failed requests: {len(errors)}")
    if times:
        print(f"Average response time: {mean(times):.3f}s")
        print(f"Std deviation: {stdev(times):.3f}s")
        print(f"Min time: {min(times):.3f}s")
        print(f"Max time: {max(times):.3f}s")

if __name__ == "__main__":
    asyncio.run(load_test(num_requests=1000, concurrent=50))
```

## Debugging

### Remote Debugging

```python
# Enable remote debugging in your code
import debugpy

# Listen for debugger connection
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger to attach...")
debugpy.wait_for_client()
print("Debugger attached!")
```

### Memory Profiling

```python
# memory_profile.py
from memory_profiler import profile
import tracemalloc

# Start tracing
tracemalloc.start()

@profile
def memory_intensive_function():
    # Your code here
    large_list = [i for i in range(1000000)]
    return large_list

# Get memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 10**6:.1f} MB")
print(f"Peak memory usage: {peak / 10**6:.1f} MB")
tracemalloc.stop()
```

### Performance Profiling

```python
# profile_api.py
import cProfile
import pstats
from pstats import SortKey

def profile_endpoint():
    """Profile API endpoint."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Your code here
    import requests
    response = requests.post(
        "http://localhost:8000/api/v1/generate",
        json={"prompt": "Test", "max_length": 100}
    )
    
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)  # Top 20 functions
    
    # Save to file
    stats.dump_stats("profile_results.prof")
```

### SQL Query Debugging

```python
# Enable SQL logging
import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Or use Flask-SQLAlchemy's EXPLAIN
from sqlalchemy_utils import explain

query = db.session.query(User).filter(User.active == True)
print(explain(query))
```

## Performance Tuning

### 1. Model Optimization

```python
# scripts/optimize_models_dev.py
import torch
from transformers import AutoModelForCausalLM
import os

def optimize_for_development():
    """Optimize models for faster development."""
    models = ['gpt2', 'distilgpt2']
    
    for model_name in models:
        print(f"Optimizing {model_name}...")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(f"models/{model_name}")
        
        # CPU optimizations
        if not torch.cuda.is_available():
            # Use smaller precision
            model = model.to(torch.float16)
            
            # Enable MKL optimizations
            torch.backends.mkl.enabled = True
            
            # Use torch.jit.script for faster inference
            scripted_model = torch.jit.script(model)
            scripted_model.save(f"models/{model_name}-scripted.pt")
        
        # GPU optimizations
        else:
            model = model.cuda()
            model = model.half()  # FP16
            
            # Enable TF32 on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Compile with torch.compile (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                compiled_model = torch.compile(
                    model, 
                    mode="reduce-overhead",
                    backend="inductor"
                )
                torch.save(compiled_model, f"models/{model_name}-compiled.pt")

if __name__ == "__main__":
    optimize_for_development()
```

### 2. Database Performance

```sql
-- Create indexes for common queries
CREATE INDEX idx_generations_user_created 
ON generations(user_id, created_at DESC);

CREATE INDEX idx_generations_model_status 
ON generations(model_name, status);

-- Analyze tables
ANALYZE generations;
ANALYZE users;

-- Check slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    max_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 20;
```

### 3. Caching Configuration

```python
# config/cache_dev.py
CACHE_CONFIG = {
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://:devpassword@localhost:6379/1',
    'CACHE_DEFAULT_TIMEOUT': 300,
    'CACHE_KEY_PREFIX': 'textnlp_dev_',
    
    # Cache specific endpoints
    'CACHED_ENDPOINTS': {
        '/api/v1/models': 3600,  # 1 hour
        '/api/v1/health': 60,    # 1 minute
    },
    
    # Model inference caching
    'MODEL_CACHE': {
        'enabled': True,
        'ttl': 3600,
        'max_size': 1000,
        'key_func': lambda prompt, model: f"{model}:{hash(prompt)}"
    }
}
```

## Common Development Tasks

### 1. Database Migrations

```bash
# Create a new migration
alembic revision -m "Add user preferences table"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# Show migration history
alembic history

# Generate migration from model changes
alembic revision --autogenerate -m "Auto migration"
```

### 2. Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=textnlp --cov-report=html

# Run specific test file
pytest tests/test_api.py

# Run tests matching pattern
pytest -k "test_generation"

# Run with verbose output
pytest -v

# Run in parallel
pytest -n auto

# Run with debugging
pytest --pdb
```

### 3. Code Quality Checks

```bash
# Format code
black .
isort .

# Lint code
flake8 textnlp/
pylint textnlp/
mypy textnlp/

# Security checks
bandit -r textnlp/
safety check

# Run all checks
pre-commit run --all-files
```

### 4. API Testing with HTTPie

```bash
# Install HTTPie
pip install httpie

# Test health endpoint
http GET localhost:8000/health

# Test generation endpoint
http POST localhost:8000/api/v1/generate \
    X-API-Key:dev-api-key \
    prompt="Write a story" \
    max_length=100 \
    model=gpt2

# Test with custom headers
http POST localhost:8000/api/v1/generate \
    X-API-Key:dev-api-key \
    X-Request-ID:test-123 \
    prompt="Test prompt"

# Save response
http POST localhost:8000/api/v1/generate \
    X-API-Key:dev-api-key \
    prompt="Test" \
    > response.json
```

### 5. Model Management

```python
# scripts/manage_models.py
import click
from pathlib import Path
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@click.group()
def cli():
    """Model management CLI."""
    pass

@cli.command()
@click.argument('model_name')
def download(model_name):
    """Download a model."""
    click.echo(f"Downloading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    output_path = Path(f"models/{model_name}")
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    click.echo(f"Model saved to {output_path}")

@cli.command()
@click.argument('model_name')
def info(model_name):
    """Show model information."""
    model_path = Path(f"models/{model_name}")
    if not model_path.exists():
        click.echo(f"Model {model_name} not found")
        return
    
    config_path = model_path / "config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
        
        click.echo(f"Model: {model_name}")
        click.echo(f"Architecture: {config.get('architectures', ['Unknown'])[0]}")
        click.echo(f"Parameters: {config.get('n_params', 'Unknown')}")
        click.echo(f"Layers: {config.get('n_layer', 'Unknown')}")
        click.echo(f"Hidden size: {config.get('n_embd', 'Unknown')}")
    
    # Check size
    size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
    click.echo(f"Size: {size / 1024 / 1024:.1f} MB")

@cli.command()
def list():
    """List available models."""
    models_dir = Path("models")
    if not models_dir.exists():
        click.echo("No models found")
        return
    
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir() and (model_dir / "config.json").exists():
            click.echo(f"- {model_dir.name}")

if __name__ == "__main__":
    cli()
```

### 6. Development Utilities

```python
# scripts/dev_utils.py
"""Development utilities for TextNLP."""

import os
import time
import psutil
import GPUtil
from contextlib import contextmanager
from typing import Dict, Any

@contextmanager
def timer(name: str = "Operation"):
    """Time a code block."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name} took {elapsed:.3f} seconds")

@contextmanager
def memory_monitor(name: str = "Operation"):
    """Monitor memory usage."""
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    print(f"{name} memory: {mem_before:.1f} MB -> {mem_after:.1f} MB "
          f"(+{mem_after - mem_before:.1f} MB)")

def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        "memory_total": psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
        "memory_available": psutil.virtual_memory().available / 1024 / 1024 / 1024,
        "disk_usage": psutil.disk_usage('/').percent,
    }
    
    # GPU info
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        info["gpu_name"] = gpu.name
        info["gpu_memory_total"] = gpu.memoryTotal
        info["gpu_memory_used"] = gpu.memoryUsed
        info["gpu_utilization"] = gpu.load * 100
    
    return info

def clear_cache():
    """Clear all caches."""
    import shutil
    
    cache_dirs = [
        "cache",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Cleared {cache_dir}")
    
    # Clear Redis cache
    import redis
    try:
        r = redis.Redis(host='localhost', port=6379, password='devpassword')
        r.flushall()
        print("Cleared Redis cache")
    except:
        print("Could not clear Redis cache")

if __name__ == "__main__":
    # Example usage
    with timer("System info"):
        info = get_system_info()
        for key, value in info.items():
            print(f"{key}: {value}")
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows
```

2. **Model Loading Errors**
```python
# Clear model cache
import shutil
shutil.rmtree(os.path.expanduser("~/.cache/huggingface"))

# Re-download model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2", force_download=True)
```

3. **Database Connection Issues**
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Check connection
psql -h localhost -U textnlp -d textnlp_dev

# Reset database
docker-compose down -v
docker-compose up -d postgres
alembic upgrade head
```

4. **Memory Issues**
```python
# Reduce model size
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use smaller batch sizes
config.batch_size = 1
```

## Best Practices

1. **Use Virtual Environments**
   - Always work in a virtual environment
   - Keep requirements.txt updated
   - Use pip-tools for dependency management

2. **Version Control**
   - Commit early and often
   - Write meaningful commit messages
   - Use feature branches

3. **Testing**
   - Write tests for new features
   - Run tests before committing
   - Maintain high test coverage

4. **Documentation**
   - Document your code
   - Update README for new features
   - Keep API documentation current

5. **Performance**
   - Profile before optimizing
   - Cache expensive operations
   - Use async where appropriate

6. **Security**
   - Never commit secrets
   - Use environment variables
   - Validate all inputs

This comprehensive guide covers everything needed for local development of TextNLP, from initial setup to advanced debugging and optimization techniques.