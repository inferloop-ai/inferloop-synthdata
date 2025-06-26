# Tabular Data Localhost Development Setup Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Development Environment Setup](#development-environment-setup)
4. [Database Configuration](#database-configuration)
5. [Service Setup](#service-setup)
6. [Development Workflow](#development-workflow)
7. [Testing Environment](#testing-environment)
8. [Debugging Tools](#debugging-tools)
9. [Performance Profiling](#performance-profiling)
10. [Common Tasks](#common-tasks)

## Quick Start

### One-Command Setup (Linux/macOS)
```bash
# Clone and setup in one command
git clone https://github.com/inferloop/inferloop-synthdata.git && \
cd inferloop-synthdata/tabular && \
./scripts/dev-setup.sh
```

### Manual Quick Setup
```bash
# 1. Clone repository
git clone https://github.com/inferloop/inferloop-synthdata.git
cd inferloop-synthdata/tabular

# 2. Create Python environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Start services
docker-compose -f docker-compose.dev.yml up -d

# 5. Initialize database
python scripts/init_db.py

# 6. Run development server
python -m tabular.api.app --reload
```

## Prerequisites

### System Requirements
```yaml
Minimum:
  OS: Ubuntu 20.04+ / macOS 11+ / Windows 10 with WSL2
  CPU: 4 cores
  RAM: 8 GB
  Storage: 20 GB free
  Python: 3.8+
  
Recommended:
  CPU: 8+ cores
  RAM: 16+ GB
  Storage: 50+ GB SSD
  GPU: Optional (for ML model acceleration)
```

### Software Dependencies
```bash
# Check prerequisites
python3 --version  # Should be 3.8+
docker --version   # Should be 20.10+
docker-compose --version  # Should be 2.0+
git --version      # Should be 2.25+

# Install missing dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install -y python3.9 python3.9-dev python3.9-venv
sudo apt install -y docker.io docker-compose
sudo apt install -y git build-essential

# macOS (using Homebrew)
brew install python@3.9 git
brew install --cask docker
```

## Development Environment Setup

### 1. Repository Setup

```bash
# Clone with all submodules
git clone --recursive https://github.com/inferloop/inferloop-synthdata.git
cd inferloop-synthdata/tabular

# Configure git
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Set up git hooks
cp scripts/pre-commit .git/hooks/
chmod +x .git/hooks/pre-commit
```

### 2. Python Environment

#### Using venv (Recommended)
```bash
# Create virtual environment
python3.9 -m venv venv

# Activate environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install development dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

#### Using conda
```bash
# Create conda environment
conda create -n tabular python=3.9
conda activate tabular

# Install dependencies
pip install -e ".[dev,test,docs]"
```

### 3. Environment Configuration

```bash
# Create .env file from template
cp .env.example .env.development

# Edit configuration
cat > .env.development << EOF
# Application Settings
APP_NAME=Tabular-Dev
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# API Configuration  
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
API_RELOAD=true

# Database
DATABASE_URL=postgresql://tabular:devpassword@localhost:5432/tabular_dev
DATABASE_ECHO=true
DATABASE_POOL_SIZE=5

# Redis
REDIS_URL=redis://:devpassword@localhost:6379/0

# Storage
STORAGE_TYPE=filesystem
STORAGE_PATH=./data/storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Spark (Local Mode)
SPARK_MASTER=local[*]
SPARK_EXECUTOR_MEMORY=2g

# Security (Development Only)
JWT_SECRET=dev-secret-key-change-in-production
API_KEY=dev-api-key
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Features
ENABLE_PROFILING=true
ENABLE_DEBUG_TOOLBAR=true
ENABLE_SQL_LOGGING=true
EOF
```

### 4. IDE Configuration

#### VS Code Setup
```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.pylintEnabled": false,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["-v", "--tb=short"],
  
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    ".mypy_cache": true,
    "venv": true
  }
}
```

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Tabular API",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "tabular.api.app:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000"
      ],
      "env": {
        "ENVIRONMENT": "development"
      },
      "jinja": true
    },
    {
      "name": "Tabular CLI",
      "type": "python",
      "request": "launch",
      "module": "tabular.cli",
      "args": ["generate", "--schema", "sample.json"],
      "console": "integratedTerminal"
    },
    {
      "name": "Debug Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["-v", "--no-cov", "${file}"],
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

#### PyCharm Setup
```xml
<!-- .idea/runConfigurations/Tabular_API.xml -->
<component name="ProjectRunConfigurationManager">
  <configuration default="false" name="Tabular API" type="PythonConfigurationType">
    <module name="tabular" />
    <option name="INTERPRETER_OPTIONS" value="" />
    <option name="PARENT_ENVS" value="true" />
    <envs>
      <env name="PYTHONUNBUFFERED" value="1" />
      <env name="ENVIRONMENT" value="development" />
    </envs>
    <option name="SDK_HOME" value="$PROJECT_DIR$/venv/bin/python" />
    <option name="SCRIPT_NAME" value="uvicorn" />
    <option name="PARAMETERS" value="tabular.api.app:app --reload" />
  </configuration>
</component>
```

## Database Configuration

### 1. PostgreSQL Setup

```bash
# Using Docker
docker run -d \
  --name tabular-postgres \
  -e POSTGRES_USER=tabular \
  -e POSTGRES_PASSWORD=devpassword \
  -e POSTGRES_DB=tabular_dev \
  -p 5432:5432 \
  -v tabular_postgres_data:/var/lib/postgresql/data \
  postgres:15-alpine

# Create test database
docker exec tabular-postgres psql -U tabular -c "CREATE DATABASE tabular_test;"

# Install extensions
docker exec tabular-postgres psql -U tabular -d tabular_dev -c "
CREATE EXTENSION IF NOT EXISTS 'uuid-ossp';
CREATE EXTENSION IF NOT EXISTS 'pgcrypto';
CREATE EXTENSION IF NOT EXISTS 'pg_stat_statements';
"
```

### 2. Redis Setup

```bash
# Using Docker
docker run -d \
  --name tabular-redis \
  -p 6379:6379 \
  redis:7-alpine redis-server --requirepass devpassword

# Test connection
docker exec -it tabular-redis redis-cli -a devpassword ping
```

### 3. MinIO Setup (S3-compatible storage)

```bash
# Using Docker
docker run -d \
  --name tabular-minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  -v tabular_minio_data:/data \
  minio/minio server /data --console-address ":9001"

# Create bucket
docker exec tabular-minio mc alias set local http://localhost:9000 minioadmin minioadmin
docker exec tabular-minio mc mb local/tabular-dev
```

### 4. Database Migrations

```bash
# Initialize Alembic
alembic init alembic

# Create initial migration
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head

# Create sample data
python scripts/seed_data.py --samples 1000
```

## Service Setup

### 1. Docker Compose Development Stack

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: tabular-postgres-dev
    environment:
      POSTGRES_USER: tabular
      POSTGRES_PASSWORD: devpassword
      POSTGRES_DB: tabular_dev
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tabular"]
      interval: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: tabular-redis-dev
    command: redis-server --requirepass devpassword --loglevel debug
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  minio:
    image: minio/minio:latest
    container_name: tabular-minio-dev
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data

  adminer:
    image: adminer
    container_name: tabular-adminer
    ports:
      - "8080:8080"
    environment:
      ADMINER_DEFAULT_SERVER: postgres

  redis-commander:
    image: rediscommander/redis-commander
    container_name: tabular-redis-commander
    environment:
      REDIS_HOSTS: local:redis:6379:0:devpassword
    ports:
      - "8081:8081"
    depends_on:
      - redis

  mailhog:
    image: mailhog/mailhog
    container_name: tabular-mailhog
    ports:
      - "1025:1025"  # SMTP
      - "8025:8025"  # Web UI

volumes:
  postgres_data:
  redis_data:
  minio_data:
```

### 2. Local Spark Setup

```bash
# Download Spark (if not using Docker)
wget https://dlcdn.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
tar -xzf spark-3.5.0-bin-hadoop3.tgz
mv spark-3.5.0-bin-hadoop3 ~/spark

# Set environment variables
echo 'export SPARK_HOME=~/spark' >> ~/.bashrc
echo 'export PATH=$SPARK_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Or use Docker
docker run -d \
  --name tabular-spark \
  -p 8082:8080 \
  -p 7077:7077 \
  bitnami/spark:3.5 \
  /opt/bitnami/spark/bin/spark-class org.apache.spark.deploy.master.Master
```

### 3. Development API Server

```python
# scripts/run_dev_server.py
"""Development server with hot reload and debugging."""

import uvicorn
from tabular.api.app import app

if __name__ == "__main__":
    uvicorn.run(
        "tabular.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",
        access_log=True,
        use_colors=True,
        reload_dirs=["tabular"],
        reload_excludes=["*.pyc", "__pycache__"]
    )
```

## Development Workflow

### 1. Code Organization

```
tabular/
├── api/                 # REST API endpoints
│   ├── routes/         # API routes
│   ├── models/         # Pydantic models
│   └── dependencies/   # FastAPI dependencies
├── core/               # Core business logic
│   ├── generators/     # Data generation algorithms
│   ├── validators/     # Data validation
│   └── privacy/        # Privacy features
├── db/                 # Database layer
│   ├── models/         # SQLAlchemy models
│   ├── repositories/   # Data access layer
│   └── migrations/     # Alembic migrations
├── services/           # Business services
├── utils/              # Utilities
└── tests/              # Test suite
```

### 2. Development Commands

```bash
# Run development server
make dev

# Run tests
make test

# Run specific test
pytest tests/test_generators.py::test_gaussian_copula -v

# Format code
make format

# Lint code
make lint

# Type checking
make typecheck

# Generate test coverage
make coverage

# Build documentation
make docs
```

### 3. Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-generator

# Make changes and commit
git add .
git commit -m "feat: Add new synthetic data generator"

# Run pre-commit checks
pre-commit run --all-files

# Push to remote
git push origin feature/new-generator
```

## Testing Environment

### 1. Test Configuration

```python
# tests/conftest.py
import pytest
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from tabular.api.app import app
from tabular.db.base import Base
from tabular.core.config import settings

# Override settings for testing
settings.database_url = "postgresql://tabular:devpassword@localhost:5432/tabular_test"
settings.redis_url = "redis://:devpassword@localhost:6379/1"

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_db():
    """Create test database."""
    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)

@pytest.fixture
def db_session(test_db):
    """Create database session for tests."""
    SessionLocal = sessionmaker(bind=test_db)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()

@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c

@pytest.fixture
def sample_schema():
    """Sample schema for testing."""
    return {
        "name": "test_table",
        "columns": [
            {"name": "id", "type": "integer", "primary_key": True},
            {"name": "name", "type": "string", "max_length": 100},
            {"name": "age", "type": "integer", "min": 0, "max": 120},
            {"name": "salary", "type": "float", "min": 0},
            {"name": "is_active", "type": "boolean"}
        ]
    }
```

### 2. Unit Tests

```python
# tests/test_generators.py
import pytest
import pandas as pd
from tabular.core.generators import GaussianCopulaGenerator

class TestGaussianCopulaGenerator:
    """Test Gaussian Copula synthetic data generator."""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return GaussianCopulaGenerator()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000],
            'department': ['IT', 'HR', 'IT', 'Finance', 'HR']
        })
    
    def test_fit(self, generator, sample_data):
        """Test fitting the generator."""
        generator.fit(sample_data)
        
        assert generator.is_fitted
        assert generator.metadata is not None
        assert len(generator.metadata['columns']) == 3
    
    def test_generate(self, generator, sample_data):
        """Test generating synthetic data."""
        generator.fit(sample_data)
        synthetic_data = generator.generate(n_samples=100)
        
        assert len(synthetic_data) == 100
        assert list(synthetic_data.columns) == list(sample_data.columns)
        assert synthetic_data['age'].min() >= 0
        assert synthetic_data['department'].isin(['IT', 'HR', 'Finance']).all()
    
    @pytest.mark.parametrize("n_samples", [10, 100, 1000])
    def test_generate_various_sizes(self, generator, sample_data, n_samples):
        """Test generating various data sizes."""
        generator.fit(sample_data)
        synthetic_data = generator.generate(n_samples=n_samples)
        
        assert len(synthetic_data) == n_samples
```

### 3. Integration Tests

```python
# tests/test_api_integration.py
import pytest
from httpx import AsyncClient
from tabular.api.app import app

@pytest.mark.asyncio
class TestAPIIntegration:
    """Test API integration."""
    
    async def test_health_check(self):
        """Test health check endpoint."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"
    
    async def test_generate_data_flow(self, sample_schema):
        """Test complete data generation flow."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # 1. Create schema
            response = await client.post("/api/v1/schemas", json=sample_schema)
            assert response.status_code == 201
            schema_id = response.json()["id"]
            
            # 2. Generate data
            generation_request = {
                "schema_id": schema_id,
                "num_rows": 1000,
                "generator_type": "gaussian_copula"
            }
            response = await client.post("/api/v1/generate", json=generation_request)
            assert response.status_code == 202
            job_id = response.json()["job_id"]
            
            # 3. Check job status
            response = await client.get(f"/api/v1/jobs/{job_id}")
            assert response.status_code == 200
            assert response.json()["status"] in ["pending", "running", "completed"]
            
            # 4. Download results (would wait for completion in real test)
            # response = await client.get(f"/api/v1/results/{job_id}")
            # assert response.status_code == 200
```

### 4. Performance Tests

```python
# tests/test_performance.py
import pytest
import time
import pandas as pd
from tabular.core.generators import GaussianCopulaGenerator

class TestPerformance:
    """Performance tests for generators."""
    
    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing."""
        n_rows = 10000
        return pd.DataFrame({
            'id': range(n_rows),
            'numeric_1': np.random.normal(100, 15, n_rows),
            'numeric_2': np.random.exponential(50, n_rows),
            'categorical_1': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
            'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_rows),
            'date': pd.date_range('2020-01-01', periods=n_rows, freq='H')
        })
    
    @pytest.mark.benchmark
    def test_fit_performance(self, benchmark, large_dataset):
        """Benchmark fitting performance."""
        generator = GaussianCopulaGenerator()
        
        result = benchmark(generator.fit, large_dataset)
        assert generator.is_fitted
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("n_samples", [1000, 10000, 100000])
    def test_generate_performance(self, benchmark, large_dataset, n_samples):
        """Benchmark generation performance."""
        generator = GaussianCopulaGenerator()
        generator.fit(large_dataset)
        
        result = benchmark(generator.generate, n_samples=n_samples)
        assert len(result) == n_samples
```

## Debugging Tools

### 1. Interactive Debugging

```python
# debug_utils.py
"""Debugging utilities for development."""

import pdb
import logging
from functools import wraps
from typing import Any, Callable
import traceback

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

def debug_on_error(func: Callable) -> Callable:
    """Decorator to drop into debugger on error."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\nError in {func.__name__}: {e}")
            print("Traceback:")
            traceback.print_exc()
            print("\nDropping into debugger...")
            pdb.post_mortem()
    return wrapper

def log_execution(func: Callable) -> Callable:
    """Decorator to log function execution."""
    logger = logging.getLogger(func.__module__)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {e}")
            raise
    
    return wrapper

# Usage example
@debug_on_error
@log_execution
def problematic_function(data):
    """Function that might fail."""
    return process_data(data)
```

### 2. API Debugging

```python
# debug_api.py
"""API debugging tools."""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import json
import time

class DebugMiddleware(BaseHTTPMiddleware):
    """Middleware for debugging API requests."""
    
    async def dispatch(self, request: Request, call_next):
        # Log request
        print(f"\n{'='*60}")
        print(f"REQUEST: {request.method} {request.url}")
        print(f"Headers: {dict(request.headers)}")
        
        # Log body for POST/PUT
        if request.method in ["POST", "PUT", "PATCH"]:
            body = await request.body()
            request._body = body  # Store for later use
            try:
                json_body = json.loads(body)
                print(f"Body: {json.dumps(json_body, indent=2)}")
            except:
                print(f"Body: {body}")
        
        # Time the request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        print(f"\nRESPONSE: {response.status_code}")
        print(f"Process Time: {process_time:.3f}s")
        print(f"{'='*60}\n")
        
        # Add debug headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Debug-Mode"] = "true"
        
        return response

# Add to app in development
if settings.debug:
    app.add_middleware(DebugMiddleware)
```

### 3. Database Query Debugging

```python
# debug_db.py
"""Database debugging utilities."""

import logging
from sqlalchemy import event
from sqlalchemy.engine import Engine
import time

# Enable SQL logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Add query timing
@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault('query_start_time', []).append(time.time())
    print("\n" + "="*60)
    print("SQL Query:")
    print(statement)
    print(f"Parameters: {parameters}")

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - conn.info['query_start_time'].pop(-1)
    print(f"Query Time: {total:.3f}s")
    print("="*60 + "\n")

# Query profiler
class QueryProfiler:
    """Profile database queries."""
    
    def __init__(self):
        self.queries = []
    
    def __enter__(self):
        event.listen(Engine, "before_cursor_execute", self._before_execute)
        event.listen(Engine, "after_cursor_execute", self._after_execute)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        event.remove(Engine, "before_cursor_execute", self._before_execute)
        event.remove(Engine, "after_cursor_execute", self._after_execute)
    
    def _before_execute(self, conn, cursor, statement, parameters, context, executemany):
        self.current_query = {
            'statement': statement,
            'parameters': parameters,
            'start_time': time.time()
        }
    
    def _after_execute(self, conn, cursor, statement, parameters, context, executemany):
        self.current_query['duration'] = time.time() - self.current_query['start_time']
        self.queries.append(self.current_query)
    
    def get_slow_queries(self, threshold=0.1):
        """Get queries slower than threshold."""
        return [q for q in self.queries if q['duration'] > threshold]
    
    def print_summary(self):
        """Print query summary."""
        print(f"\nTotal queries: {len(self.queries)}")
        print(f"Total time: {sum(q['duration'] for q in self.queries):.3f}s")
        print("\nSlowest queries:")
        for q in sorted(self.queries, key=lambda x: x['duration'], reverse=True)[:5]:
            print(f"  {q['duration']:.3f}s: {q['statement'][:50]}...")
```

## Performance Profiling

### 1. Code Profiling

```python
# profile_code.py
"""Performance profiling utilities."""

import cProfile
import pstats
from memory_profiler import profile
import line_profiler
from functools import wraps

def profile_function(func):
    """Profile function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        return result
    return wrapper

# Memory profiling
@profile
def memory_intensive_function():
    # Your code here
    large_list = [i for i in range(1000000)]
    return large_list

# Line profiling
def line_profile_this(func):
    """Line-by-line profiling."""
    lp = line_profiler.LineProfiler()
    lp.add_function(func)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        return lp.runcall(func, *args, **kwargs)
    
    wrapper.print_stats = lp.print_stats
    return wrapper

@line_profile_this
def slow_function():
    total = 0
    for i in range(1000000):
        total += i ** 2
    return total

# Usage
if __name__ == "__main__":
    slow_function()
    slow_function.print_stats()
```

### 2. API Performance Testing

```python
# test_api_performance.py
"""API performance testing."""

import asyncio
import aiohttp
import time
from statistics import mean, stdev

async def benchmark_endpoint(url: str, num_requests: int = 100):
    """Benchmark API endpoint."""
    times = []
    errors = 0
    
    async with aiohttp.ClientSession() as session:
        for i in range(num_requests):
            start = time.time()
            try:
                async with session.get(url) as response:
                    await response.text()
                    if response.status == 200:
                        times.append(time.time() - start)
                    else:
                        errors += 1
            except Exception as e:
                errors += 1
                print(f"Request {i} failed: {e}")
    
    if times:
        print(f"\nBenchmark Results for {url}")
        print(f"Requests: {num_requests}")
        print(f"Successful: {len(times)}")
        print(f"Failed: {errors}")
        print(f"Average: {mean(times):.3f}s")
        print(f"Min: {min(times):.3f}s")
        print(f"Max: {max(times):.3f}s")
        print(f"Std Dev: {stdev(times):.3f}s")
        print(f"Requests/second: {len(times) / sum(times):.2f}")

async def load_test():
    """Run load test."""
    endpoints = [
        "http://localhost:8000/health",
        "http://localhost:8000/api/v1/schemas",
        "http://localhost:8000/api/v1/generators"
    ]
    
    tasks = [benchmark_endpoint(url) for url in endpoints]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(load_test())
```

### 3. Database Performance

```sql
-- Enable query performance tracking
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- View slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    stddev_time,
    rows
FROM pg_stat_statements
WHERE mean_time > 100  -- Queries taking more than 100ms
ORDER BY mean_time DESC
LIMIT 20;

-- Analyze table statistics
ANALYZE;

-- Check for missing indexes
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    most_common_vals
FROM pg_stats
WHERE schemaname = 'public'
    AND n_distinct > 100
    AND most_common_vals IS NULL;
```

## Common Tasks

### 1. Generate Test Data

```python
# scripts/generate_test_data.py
"""Generate test data for development."""

import pandas as pd
import numpy as np
from faker import Faker
from tabular.core.generators import GaussianCopulaGenerator

fake = Faker()

def generate_sample_dataset(num_rows: int = 1000) -> pd.DataFrame:
    """Generate sample dataset."""
    data = {
        'customer_id': range(1, num_rows + 1),
        'name': [fake.name() for _ in range(num_rows)],
        'email': [fake.email() for _ in range(num_rows)],
        'age': np.random.normal(35, 10, num_rows).astype(int).clip(18, 80),
        'income': np.random.lognormal(10.5, 0.5, num_rows),
        'city': [fake.city() for _ in range(num_rows)],
        'registration_date': pd.date_range('2020-01-01', periods=num_rows, freq='H'),
        'is_active': np.random.choice([True, False], num_rows, p=[0.8, 0.2])
    }
    
    return pd.DataFrame(data)

def generate_synthetic_version(original_df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic version of dataset."""
    generator = GaussianCopulaGenerator()
    generator.fit(original_df)
    
    return generator.generate(n_samples=len(original_df))

if __name__ == "__main__":
    # Generate original data
    original = generate_sample_dataset(1000)
    original.to_csv('data/original_sample.csv', index=False)
    
    # Generate synthetic version
    synthetic = generate_synthetic_version(original)
    synthetic.to_csv('data/synthetic_sample.csv', index=False)
    
    print("Sample data generated successfully!")
```

### 2. Database Operations

```python
# scripts/db_operations.py
"""Common database operations for development."""

from sqlalchemy import create_engine, text
from tabular.db.session import SessionLocal
from tabular.db.models import Schema, Generation

def reset_database():
    """Reset database to clean state."""
    engine = create_engine(settings.database_url)
    
    # Drop all tables
    with engine.connect() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
        conn.commit()
    
    # Recreate tables
    from tabular.db.base import Base
    Base.metadata.create_all(engine)
    
    print("Database reset complete!")

def create_sample_schemas():
    """Create sample schemas."""
    db = SessionLocal()
    
    schemas = [
        {
            "name": "customers",
            "description": "Customer data schema",
            "columns": {
                "customer_id": {"type": "integer", "primary_key": True},
                "name": {"type": "string", "max_length": 100},
                "email": {"type": "string", "format": "email"},
                "age": {"type": "integer", "min": 18, "max": 100},
                "income": {"type": "float", "min": 0}
            }
        },
        {
            "name": "transactions",
            "description": "Transaction data schema",
            "columns": {
                "transaction_id": {"type": "uuid"},
                "customer_id": {"type": "integer", "foreign_key": "customers.customer_id"},
                "amount": {"type": "float", "min": 0},
                "timestamp": {"type": "datetime"},
                "status": {"type": "enum", "values": ["pending", "completed", "failed"]}
            }
        }
    ]
    
    for schema_data in schemas:
        schema = Schema(**schema_data)
        db.add(schema)
    
    db.commit()
    db.close()
    
    print(f"Created {len(schemas)} sample schemas")
```

### 3. Development Utilities

```bash
# Makefile for common tasks
.PHONY: help dev test format lint clean

help:
	@echo "Available commands:"
	@echo "  make dev        - Run development server"
	@echo "  make test       - Run tests"
	@echo "  make format     - Format code"
	@echo "  make lint       - Lint code"
	@echo "  make clean      - Clean up"

dev:
	@echo "Starting development server..."
	docker-compose -f docker-compose.dev.yml up -d
	python scripts/wait_for_db.py
	python scripts/run_dev_server.py

test:
	@echo "Running tests..."
	pytest -v --cov=tabular --cov-report=html

format:
	@echo "Formatting code..."
	black tabular tests
	isort tabular tests

lint:
	@echo "Linting code..."
	flake8 tabular tests
	mypy tabular

clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
	docker-compose -f docker-compose.dev.yml down -v
```

### 4. Shell Shortcuts

```bash
# .env.dev.sh - Development shortcuts
alias td='cd ~/projects/tabular'
alias tdv='cd ~/projects/tabular && source venv/bin/activate'
alias tdd='tdv && docker-compose -f docker-compose.dev.yml up -d'
alias tds='tdv && python scripts/run_dev_server.py'
alias tdt='tdv && pytest -v'
alias tdf='tdv && make format'
alias tdl='tdv && make lint'
alias tdc='docker-compose -f docker-compose.dev.yml'

# Database shortcuts
alias tddb='docker exec -it tabular-postgres-dev psql -U tabular -d tabular_dev'
alias tdredis='docker exec -it tabular-redis-dev redis-cli -a devpassword'

# Log viewing
alias tdlogs='docker-compose -f docker-compose.dev.yml logs -f'
alias tdapi='tail -f logs/api.log | jq .'
```

## Troubleshooting

### Common Issues

1. **Port conflicts**
```bash
# Check what's using a port
lsof -i :8000
# Kill process
kill -9 <PID>
```

2. **Database connection issues**
```bash
# Check PostgreSQL is running
docker ps | grep postgres
# Check logs
docker logs tabular-postgres-dev
# Test connection
psql -h localhost -U tabular -d tabular_dev
```

3. **Module import errors**
```bash
# Ensure virtual environment is activated
which python  # Should show venv path
# Reinstall in development mode
pip install -e ".[dev]"
```

4. **Docker issues**
```bash
# Clean up Docker
docker system prune -a
# Rebuild containers
docker-compose -f docker-compose.dev.yml build --no-cache
```

## Best Practices

1. **Always work in a virtual environment**
2. **Run tests before committing**
3. **Use type hints and docstrings**
4. **Follow the project's code style**
5. **Keep dependencies up to date**
6. **Document your changes**
7. **Use meaningful commit messages**

This comprehensive guide covers everything needed for local development of the Tabular synthetic data system.