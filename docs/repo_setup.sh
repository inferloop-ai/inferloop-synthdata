#!/bin/bash

# Structured Documents Synthetic Data Repository Setup Script
# This script creates the complete repository structure with placeholder files

set -e  # Exit on any error

PROJECT_NAME="structured-documents-synthetic-data"
REPO_DIR="${1:-$PROJECT_NAME}"

echo "🏗️  Creating Structured Documents Synthetic Data Repository: $REPO_DIR"
echo "=================================================================="

# Create main repository directory
mkdir -p "$REPO_DIR"
cd "$REPO_DIR"

# Function to create file with basic content
create_file() {
    local filepath="$1"
    local content="$2"
    
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$filepath")"
    
    # Create file with content
    if [[ -n "$content" ]]; then
        echo "$content" > "$filepath"
    else
        touch "$filepath"
    fi
    echo "📄 Created: $filepath"
}

# Function to create directory
create_dir() {
    local dirpath="$1"
    mkdir -p "$dirpath"
    echo "📁 Created: $dirpath/"
}

echo "🚀 Creating root files..."

# Root files
create_file "README.md" "# Structured Documents Synthetic Data Platform

## Overview
Enterprise-grade synthetic data generation platform for structured documents across Legal, Banking, Healthcare, Insurance, and Government verticals.

## Features
- Multi-modal document generation (PDF, DOCX, JSON)
- Advanced OCR and NLP processing
- Comprehensive privacy and compliance framework
- Real-time data integration
- AI/ML security testing
- Enterprise deployment ready

## Quick Start
\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Setup database
python scripts/setup/setup_database.py

# Generate documents
synth doc generate --type legal_contract --count 100
\`\`\`

## Documentation
See \`docs/\` directory for comprehensive documentation.

## License
See LICENSE file for details.
"

create_file "LICENSE" "MIT License

Copyright (c) 2025 Structured Documents Synthetic Data Platform

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"

create_file "requirements.txt" "# Core dependencies
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
sqlalchemy>=2.0.0
alembic>=1.13.0
redis>=5.0.0
celery>=5.3.0

# Document processing
PyPDF2>=3.0.0
python-docx>=1.1.0
reportlab>=4.0.0
Pillow>=10.1.0
pytesseract>=0.3.10
transformers>=4.36.0

# ML/AI dependencies
torch>=2.1.0
torchvision>=0.16.0
sentence-transformers>=2.2.0
spacy>=3.7.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.1.0

# Privacy and security
opacus>=1.4.0
cryptography>=41.0.0
faker>=20.1.0
presidio-analyzer>=2.2.0
presidio-anonymizer>=2.2.0

# Cloud and storage
boto3>=1.34.0
google-cloud-storage>=2.10.0
azure-storage-blob>=12.19.0
minio>=7.2.0

# Monitoring and observability
prometheus-client>=0.19.0
structlog>=23.2.0
sentry-sdk>=1.38.0
opentelemetry-api>=1.21.0

# Testing and security
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
bandit>=1.7.0
safety>=2.3.0

# Development dependencies
black>=23.11.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.7.0
pre-commit>=3.6.0
"

create_file "setup.py" "from setuptools import setup, find_packages

with open(\"README.md\", \"r\", encoding=\"utf-8\") as fh:
    long_description = fh.read()

with open(\"requirements.txt\", \"r\", encoding=\"utf-8\") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith(\"#\")]

setup(
    name=\"structured-docs-synth\",
    version=\"0.1.0\",
    author=\"Structured Documents Team\",
    author_email=\"team@structured-docs.com\",
    description=\"Enterprise synthetic data generation platform for structured documents\",
    long_description=long_description,
    long_description_content_type=\"text/markdown\",
    url=\"https://github.com/your-org/structured-documents-synthetic-data\",
    packages=find_packages(where=\"src\"),
    package_dir={\"\": \"src\"},
    classifiers=[
        \"Development Status :: 4 - Beta\",
        \"Intended Audience :: Developers\",
        \"License :: OSI Approved :: MIT License\",
        \"Operating System :: OS Independent\",
        \"Programming Language :: Python :: 3\",
        \"Programming Language :: Python :: 3.9\",
        \"Programming Language :: Python :: 3.10\",
        \"Programming Language :: Python :: 3.11\",
    ],
    python_requires=\">=3.9\",
    install_requires=requirements,
    extras_require={
        \"dev\": [
            \"pytest>=7.4.0\",
            \"black>=23.11.0\",
            \"isort>=5.12.0\",
            \"flake8>=6.1.0\",
            \"mypy>=1.7.0\",
        ],
        \"security\": [
            \"bandit>=1.7.0\",
            \"safety>=2.3.0\",
        ],
    },
    entry_points={
        \"console_scripts\": [
            \"synth=cli.main:main\",
        ],
    },
)
"

create_file "pyproject.toml" "[build-system]
requires = [\"setuptools>=61.0\", \"wheel\"]
build-backend = \"setuptools.build_meta\"

[project]
name = \"structured-docs-synth\"
version = \"0.1.0\"
description = \"Enterprise synthetic data generation platform for structured documents\"
authors = [{name = \"Structured Documents Team\", email = \"team@structured-docs.com\"}]
license = {text = \"MIT\"}
readme = \"README.md\"
requires-python = \">=3.9\"

[tool.black]
line-length = 88
target-version = ['py39']
include = '\\.pyi?$'
extend-exclude = '''
/(
  # directories
  \\.eggs
  | \\.git
  | \\.hg
  | \\.mypy_cache
  | \\.tox
  | \\.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = \"black\"
multi_line_output = 3
line_length = 88

[tool.mypy]
python_version = \"3.9\"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = [\"tests\"]
python_files = [\"test_*.py\"]
python_classes = [\"Test*\"]
python_functions = [\"test_*\"]
addopts = \"-v --cov=src --cov-report=html --cov-report=term-missing\"

[tool.bandit]
exclude_dirs = [\"tests\", \"venv\", \".venv\"]
skips = [\"B101\", \"B601\"]
"

echo "📁 Creating .github directory structure..."

# GitHub workflows
create_file ".github/workflows/ci.yml" "name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python \${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: \${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run tests
      run: |
        pytest tests/unit tests/integration
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
"

create_file ".github/workflows/security-scan.yml" "name: Security Scanning

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Bandit Security Scan
      run: |
        pip install bandit
        bandit -r src/ -f json -o bandit-report.json
    
    - name: Run Safety Check
      run: |
        pip install safety
        safety check --json --output safety-report.json
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
"

create_file ".github/workflows/ai-security-testing.yml" "name: AI Security Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  ai-security-tests:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run LLM Security Tests
      run: |
        pytest tests/security/ai_security/llm_security_tests/ -v
    
    - name: Run Agent Security Tests
      run: |
        pytest tests/security/ai_security/agent_security_tests/ -v
    
    - name: Run RAG Security Tests
      run: |
        pytest tests/security/ai_security/rag_security_tests/ -v
    
    - name: Run Synthetic Data Security Tests
      run: |
        pytest tests/security/ai_security/synthetic_data_security/ -v
"

create_file ".github/workflows/synthetic-data-validation.yml" "name: Synthetic Data Validation

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  data-validation:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run Privacy Tests
      run: |
        pytest tests/security/ai_security/synthetic_data_security/ -v
    
    - name: Run Quality Validation
      run: |
        pytest tests/unit/test_quality/ -v
    
    - name: Generate Quality Report
      run: |
        python scripts/security/ai_red_team_testing.py --report-only
"

create_file ".github/workflows/llm-safety-testing.yml" "name: LLM Safety Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  llm-safety:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run Bias Detection Tests
      run: |
        pytest tests/security/ai_security/llm_security_tests/bias_fairness_tests.py -v
    
    - name: Run Prompt Injection Tests
      run: |
        pytest tests/security/ai_security/llm_security_tests/prompt_injection_tests.py -v
    
    - name: Run Jailbreak Resistance Tests
      run: |
        pytest tests/security/ai_security/llm_security_tests/jailbreak_resistance_tests.py -v
"

create_file ".github/workflows/agent-security-testing.yml" "name: Agent Security Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  agent-security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run Agent Isolation Tests
      run: |
        pytest tests/security/ai_security/agent_security_tests/agent_isolation_tests.py -v
    
    - name: Run Privilege Escalation Tests
      run: |
        pytest tests/security/ai_security/agent_security_tests/privilege_escalation_tests.py -v
    
    - name: Run Agent Communication Security Tests
      run: |
        pytest tests/security/ai_security/agent_security_tests/agent_communication_security.py -v
"

create_file ".github/workflows/rag-security-testing.yml" "name: RAG Security Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  rag-security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run Vector Store Security Tests
      run: |
        pytest tests/security/ai_security/rag_security_tests/vector_store_poisoning_tests.py -v
    
    - name: Run Context Injection Tests
      run: |
        pytest tests/security/ai_security/rag_security_tests/context_injection_tests.py -v
    
    - name: Run DeepSeek RAG Security Tests
      run: |
        pytest tests/security/ai_security/rag_security_tests/deepseek_rag_security_tests.py -v
"

create_file ".github/workflows/dependency-security.yml" "name: Dependency Security

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM

jobs:
  dependency-check:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Snyk Security Check
      uses: snyk/actions/python@master
      env:
        SNYK_TOKEN: \${{ secrets.SNYK_TOKEN }}
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
"

create_file ".github/workflows/release.yml" "name: Release Pipeline

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Build package
      run: |
        pip install build
        python -m build
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: \${{ secrets.PYPI_API_TOKEN }}
    
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: \${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: \${{ github.ref }}
        release_name: Release \${{ github.ref }}
        draft: false
        prerelease: false
"

create_file ".github/ISSUE_TEMPLATE/bug_report.md" "---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: 'bug'
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment (please complete the following information):**
 - OS: [e.g. Ubuntu 20.04]
 - Python Version: [e.g. 3.11]
 - Package Version: [e.g. 0.1.0]

**Additional context**
Add any other context about the problem here.
"

create_file ".github/ISSUE_TEMPLATE/feature_request.md" "---
name: Feature request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: 'enhancement'
assignees: ''

---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
"

create_file ".github/PULL_REQUEST_TEMPLATE.md" "## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Security fix

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Security tests pass
- [ ] Manual testing completed

## Security Checklist
- [ ] No sensitive data exposed
- [ ] Security tests added/updated
- [ ] Dependencies reviewed for vulnerabilities
- [ ] AI safety tests pass (if applicable)

## Documentation
- [ ] Code is self-documenting
- [ ] README updated (if needed)
- [ ] API documentation updated (if needed)

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published
"

create_file ".github/SECURITY.md" "# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it to us privately.

### How to Report

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. Email us at security@structured-docs.com
3. Include as much detail as possible about the vulnerability
4. If possible, include steps to reproduce the issue

### What to Expect

1. **Acknowledgment**: We'll acknowledge receipt within 24 hours
2. **Assessment**: We'll assess the vulnerability within 72 hours
3. **Updates**: We'll provide regular updates on our progress
4. **Resolution**: We'll work to resolve the issue as quickly as possible

### Security Features

This project includes:
- AI/ML security testing (LLM, Agent, RAG security)
- Synthetic data privacy protection
- Comprehensive compliance frameworks (GDPR, HIPAA, PCI-DSS)
- Regular security scanning and monitoring
- Penetration testing and red team validation

### Responsible Disclosure

We follow responsible disclosure practices and ask that you do the same:
- Give us reasonable time to investigate and fix the issue
- Don't access or modify user data
- Don't perform actions that could harm our users or services

Thank you for helping keep our project secure!
"

echo "📚 Creating docs directory structure..."

# Documentation structure
create_dir "docs/api"
create_dir "docs/examples"
create_dir "docs/deployment"
create_dir "docs/compliance"

create_file "docs/README.md" "# Documentation

## Structure

- `api/` - API documentation and OpenAPI specs
- `examples/` - Usage examples and tutorials
- `deployment/` - Deployment guides and configurations
- `compliance/` - Compliance and security documentation

## Getting Started

1. [Installation Guide](installation.md)
2. [Quick Start](quickstart.md)
3. [API Reference](api/README.md)
4. [Security Guide](security.md)
"

create_file "docs/api/README.md" "# API Documentation

## REST API

The platform provides a comprehensive REST API for document generation and management.

### Base URL
\`https://api.structured-docs.com/v1\`

### Authentication
API key authentication required for all endpoints.

### Endpoints

- \`POST /generate/document\` - Generate single document
- \`POST /generate/batch\` - Generate documents in batch
- \`GET /status/{job_id}\` - Check generation job status
- \`POST /validate/quality\` - Validate document quality
- \`GET /templates\` - List available templates

See individual endpoint documentation for details.
"

echo "⚙️  Creating configs directory structure..."

# Configs structure with comprehensive security and CI/CD
create_dir "configs/schema_bank/legal"
create_dir "configs/schema_bank/healthcare"
create_dir "configs/schema_bank/banking"
create_dir "configs/schema_bank/government"
create_dir "configs/schema_bank/insurance"

create_file "configs/schema_bank/legal/contract_template.yaml" "# Legal Contract Template
template_name: \"service_agreement\"
template_type: \"legal_contract\"
document_format: \"pdf\"

fields:
  - name: \"party_1\"
    type: \"text\"
    required: true
    privacy_level: \"medium\"
  
  - name: \"party_2\"
    type: \"text\"
    required: true
    privacy_level: \"medium\"
  
  - name: \"effective_date\"
    type: \"date\"
    required: true
    privacy_level: \"low\"
  
  - name: \"jurisdiction\"
    type: \"text\"
    required: true
    privacy_level: \"low\"

layout:
  sections:
    - header
    - parties
    - terms
    - signatures
  
  styling:
    font: \"Times New Roman\"
    font_size: 12
    margin: \"1 inch\"

compliance:
  - gdpr
  - sox
"

create_file "configs/compliance/gdpr_rules.yaml" "# GDPR Compliance Rules
gdpr_compliance:
  data_minimization:
    enabled: true
    max_retention_days: 365
  
  consent_management:
    explicit_consent_required: true
    consent_tracking: true
  
  right_to_erasure:
    enabled: true
    deletion_policy: \"immediate\"
  
  privacy_by_design:
    default_privacy_level: \"high\"
    anonymization_required: true
  
  audit_requirements:
    log_all_access: true
    retention_period_days: 2555  # 7 years
"

create_dir "configs/security/ai_security/llm_security_configs"
create_dir "configs/security/ai_security/agent_security_configs"
create_dir "configs/security/ai_security/rag_security_configs"
create_dir "configs/security/ai_security/mcp_security_configs"
create_dir "configs/security/ai_security/synthetic_data_security"

create_file "configs/security/ai_security/llm_security_configs/prompt_injection_filters.yaml" "# Prompt Injection Detection and Prevention
prompt_injection_filters:
  detection_rules:
    - pattern: \"ignore previous instructions\"
      severity: \"high\"
      action: \"block\"
    
    - pattern: \"system prompt\"
      severity: \"medium\"
      action: \"sanitize\"
    
    - pattern: \"jailbreak\"
      severity: \"high\"
      action: \"block\"
  
  sanitization:
    enabled: true
    methods:
      - \"input_validation\"
      - \"output_filtering\"
      - \"content_moderation\"
  
  monitoring:
    log_attempts: true
    alert_threshold: 5
    reporting: \"real_time\"
"

create_file "configs/security/ai_security/agent_security_configs/agent_isolation_policies.yaml" "# Agent Isolation Security Policies
agent_isolation:
  sandbox_config:
    enabled: true
    isolation_level: \"strict\"
    resource_limits:
      memory_mb: 1024
      cpu_percent: 50
      network_access: \"restricted\"
  
  privilege_restrictions:
    file_system_access: \"read_only\"
    network_permissions: \"api_only\"
    system_calls: \"blocked\"
  
  communication_security:
    inter_agent_encryption: true
    message_validation: true
    authentication_required: true
  
  monitoring:
    behavior_tracking: true
    anomaly_detection: true
    security_logging: true
"

create_dir "configs/ci_cd/pipeline_configs"
create_dir "configs/ci_cd/quality_gates"
create_dir "configs/ci_cd/scanning_configs"
create_dir "configs/ci_cd/deployment_policies"

create_file "configs/ci_cd/quality_gates/security_quality_gates.yaml" "# Security Quality Gates
security_quality_gates:
  vulnerability_thresholds:
    critical: 0
    high: 5
    medium: 20
    low: 50
  
  ai_safety_requirements:
    bias_score_max: 0.05
    fairness_score_min: 0.95
    safety_compliance_min: 0.99
  
  compliance_requirements:
    gdpr_compliance: 100
    hipaa_compliance: 100
    pci_dss_compliance: 100
  
  code_quality:
    coverage_min: 80
    complexity_max: 10
    duplication_max: 3
"

echo "💻 Creating src directory structure..."

# Main source code structure
create_dir "src/structured_docs_synth"
create_file "src/structured_docs_synth/__init__.py" "\"\"\"
Structured Documents Synthetic Data Platform

Enterprise-grade synthetic data generation for structured documents.
\"\"\""

create_dir "src/structured_docs_synth/core"
create_file "src/structured_docs_synth/core/__init__.py" ""
create_file "src/structured_docs_synth/core/config.py" "\"\"\"Core configuration management.\"\"\""
create_file "src/structured_docs_synth/core/exceptions.py" "\"\"\"Custom exceptions for the platform.\"\"\""
create_file "src/structured_docs_synth/core/logging.py" "\"\"\"Logging configuration and utilities.\"\"\""

# Ingestion modules
create_dir "src/structured_docs_synth/ingestion/external_datasets"
create_file "src/structured_docs_synth/ingestion/__init__.py" ""
create_file "src/structured_docs_synth/ingestion/external_datasets/__init__.py" ""
create_file "src/structured_docs_synth/ingestion/external_datasets/legal_data_adapter.py" "\"\"\"Legal data source adapters (CourtListener, SEC EDGAR, etc.).\"\"\""
create_file "src/structured_docs_synth/ingestion/external_datasets/healthcare_data_adapter.py" "\"\"\"Healthcare data source adapters.\"\"\""
create_file "src/structured_docs_synth/ingestion/external_datasets/banking_data_adapter.py" "\"\"\"Banking data source adapters.\"\"\""
create_file "src/structured_docs_synth/ingestion/external_datasets/government_data_adapter.py" "\"\"\"Government data source adapters.\"\"\""
create_file "src/structured_docs_synth/ingestion/external_datasets/document_datasets_adapter.py" "\"\"\"Document processing dataset adapters (FUNSD, DocBank, SROIE).\"\"\""

create_dir "src/structured_docs_synth/ingestion/streaming"
create_file "src/structured_docs_synth/ingestion/streaming/__init__.py" ""
create_file "src/structured_docs_synth/ingestion/streaming/kafka_consumer.py" "\"\"\"Kafka consumer for real-time data ingestion.\"\"\""
create_file "src/structured_docs_synth/ingestion/streaming/webhook_handler.py" "\"\"\"Webhook handlers for external data sources.\"\"\""
create_file "src/structured_docs_synth/ingestion/streaming/api_poller.py" "\"\"\"API polling for external data sources.\"\"\""

# Generation modules
create_dir "src/structured_docs_synth/generation/engines"
create_file "src/structured_docs_synth/generation/__init__.py" ""
create_file "src/structured_docs_synth/generation/engines/__init__.py" ""
create_file "src/structured_docs_synth/generation/engines/latex_generator.py" "\"\"\"LaTeX document generation engine.\"\"\""
create_file "src/structured_docs_synth/generation/engines/docx_generator.py" "\"\"\"DOCX document generation engine.\"\"\""
create_file "src/structured_docs_synth/generation/engines/pdf_generator.py" "\"\"\"PDF document generation engine.\"\"\""
create_file "src/structured_docs_synth/generation/engines/template_engine.py" "\"\"\"Template processing engine.\"\"\""

# Processing modules
create_dir "src/structured_docs_synth/processing/ocr"
create_file "src/structured_docs_synth/processing/__init__.py" ""
create_file "src/structured_docs_synth/processing/ocr/__init__.py" ""
create_file "src/structured_docs_synth/processing/ocr/tesseract_engine.py" "\"\"\"Tesseract OCR engine integration.\"\"\""
create_file "src/structured_docs_synth/processing/ocr/trocr_engine.py" "\"\"\"TrOCR (Transformer-based OCR) engine.\"\"\""
create_file "src/structured_docs_synth/processing/ocr/custom_ocr_models.py" "\"\"\"Custom OCR models for specialized documents.\"\"\""

# Privacy and security modules
create_dir "src/structured_docs_synth/privacy/differential_privacy"
create_file "src/structured_docs_synth/privacy/__init__.py" ""
create_file "src/structured_docs_synth/privacy/differential_privacy/__init__.py" ""
create_file "src/structured_docs_synth/privacy/differential_privacy/laplace_mechanism.py" "\"\"\"Laplace mechanism for differential privacy.\"\"\""
create_file "src/structured_docs_synth/privacy/differential_privacy/exponential_mechanism.py" "\"\"\"Exponential mechanism for differential privacy.\"\"\""
create_file "src/structured_docs_synth/privacy/differential_privacy/composition_analyzer.py" "\"\"\"Privacy budget composition analysis.\"\"\""

create_dir "src/structured_docs_synth/privacy/pii_protection"
create_file "src/structured_docs_synth/privacy/pii_protection/__init__.py" ""
create_file "src/structured_docs_synth/privacy/pii_protection/pii_detector.py" "\"\"\"PII detection using multiple methods.\"\"\""
create_file "src/structured_docs_synth/privacy/pii_protection/masking_strategies.py" "\"\"\"PII masking and anonymization strategies.\"\"\""

# Quality and validation modules
create_dir "src/structured_docs_synth/quality/metrics"
create_file "src/structured_docs_synth/quality/__init__.py" ""
create_file "src/structured_docs_synth/quality/metrics/__init__.py" ""
create_file "src/structured_docs_synth/quality/metrics/ocr_metrics.py" "\"\"\"OCR quality metrics (TEDS, Levenshtein, etc.).\"\"\""
create_file "src/structured_docs_synth/quality/metrics/layout_metrics.py" "\"\"\"Layout and structure quality metrics.\"\"\""
create_file "src/structured_docs_synth/quality/metrics/content_metrics.py" "\"\"\"Content quality and consistency metrics.\"\"\""

# Delivery modules
create_dir "src/structured_docs_synth/delivery/api"
create_file "src/structured_docs_synth/delivery/__init__.py" ""
create_file "src/structured_docs_synth/delivery/api/__init__.py" ""
create_file "src/structured_docs_synth/delivery/api/rest_api.py" "\"\"\"FastAPI REST API implementation.\"\"\""
create_file "src/structured_docs_synth/delivery/api/graphql_api.py" "\"\"\"GraphQL API implementation.\"\"\""

echo "🧪 Creating comprehensive test structure..."

# Test structure with comprehensive security testing
create_dir "tests/unit/test_generation"
create_dir "tests/unit/test_processing"
create_dir "tests/unit/test_privacy"
create_dir "tests/unit/test_quality"
create_dir "tests/unit/test_delivery"

create_file "tests/__init__.py" ""
create_file "tests/conftest.py" "\"\"\"Pytest configuration and fixtures.\"\"\""

# AI Security Tests
create_dir "tests/security/ai_security/llm_security_tests"
create_file "tests/security/__init__.py" ""
create_file "tests/security/ai_security/__init__.py" ""
create_file "tests/security/ai_security/llm_security_tests/__init__.py" ""
create_file "tests/security/ai_security/llm_security_tests/prompt_injection_tests.py" "\"\"\"Prompt injection attack tests.\"\"\""
create_file "tests/security/ai_security/llm_security_tests/jailbreak_resistance_tests.py" "\"\"\"Jailbreak resistance tests.\"\"\""
create_file "tests/security/ai_security/llm_security_tests/data_leakage_tests.py" "\"\"\"Data leakage prevention tests.\"\"\""
create_file "tests/security/ai_security/llm_security_tests/bias_fairness_tests.py" "\"\"\"Bias and fairness testing.\"\"\""
create_file "tests/security/ai_security/llm_security_tests/adversarial_input_tests.py" "\"\"\"Adversarial input resistance tests.\"\"\""
create_file "tests/security/ai_security/llm_security_tests/model_inversion_tests.py" "\"\"\"Model inversion attack tests.\"\"\""

create_dir "tests/security/ai_security/agent_security_tests"
create_file "tests/security/ai_security/agent_security_tests/__init__.py" ""
create_file "tests/security/ai_security/agent_security_tests/agent_isolation_tests.py" "\"\"\"Agent isolation and sandbox tests.\"\"\""
create_file "tests/security/ai_security/agent_security_tests/privilege_escalation_tests.py" "\"\"\"Privilege escalation prevention tests.\"\"\""
create_file "tests/security/ai_security/agent_security_tests/agent_communication_security.py" "\"\"\"Agent communication security tests.\"\"\""
create_file "tests/security/ai_security/agent_security_tests/resource_abuse_tests.py" "\"\"\"Resource abuse prevention tests.\"\"\""
create_file "tests/security/ai_security/agent_security_tests/malicious_tool_usage_tests.py" "\"\"\"Malicious tool usage detection tests.\"\"\""
create_file "tests/security/ai_security/agent_security_tests/agent_orchestration_security.py" "\"\"\"Agent orchestration security tests.\"\"\""

create_dir "tests/security/ai_security/rag_security_tests"
create_file "tests/security/ai_security/rag_security_tests/__init__.py" ""
create_file "tests/security/ai_security/rag_security_tests/vector_store_poisoning_tests.py" "\"\"\"Vector store poisoning attack tests.\"\"\""
create_file "tests/security/ai_security/rag_security_tests/retrieval_manipulation_tests.py" "\"\"\"Retrieval manipulation tests.\"\"\""
create_file "tests/security/ai_security/rag_security_tests/context_injection_tests.py" "\"\"\"Context injection attack tests.\"\"\""
create_file "tests/security/ai_security/rag_security_tests/knowledge_extraction_tests.py" "\"\"\"Knowledge extraction attack tests.\"\"\""
create_file "tests/security/ai_security/rag_security_tests/deepseek_rag_security_tests.py" "\"\"\"DeepSeek RAG specific security tests.\"\"\""
create_file "tests/security/ai_security/rag_security_tests/embedding_security_tests.py" "\"\"\"Embedding security tests.\"\"\""

create_dir "tests/security/ai_security/synthetic_data_security"
create_file "tests/security/ai_security/synthetic_data_security/__init__.py" ""
create_file "tests/security/ai_security/synthetic_data_security/data_reconstruction_tests.py" "\"\"\"Data reconstruction attack tests.\"\"\""
create_file "tests/security/ai_security/synthetic_data_security/membership_inference_tests.py" "\"\"\"Membership inference attack tests.\"\"\""
create_file "tests/security/ai_security/synthetic_data_security/model_inversion_attacks.py" "\"\"\"Model inversion attack tests.\"\"\""
create_file "tests/security/ai_security/synthetic_data_security/property_inference_tests.py" "\"\"\"Property inference attack tests.\"\"\""
create_file "tests/security/ai_security/synthetic_data_security/differential_privacy_tests.py" "\"\"\"Differential privacy validation tests.\"\"\""
create_file "tests/security/ai_security/synthetic_data_security/anonymization_robustness_tests.py" "\"\"\"Anonymization robustness tests.\"\"\""

create_dir "tests/security/ai_security/mcp_security_tests"
create_file "tests/security/ai_security/mcp_security_tests/__init__.py" ""
create_file "tests/security/ai_security/mcp_security_tests/protocol_validation_tests.py" "\"\"\"MCP protocol validation tests.\"\"\""
create_file "tests/security/ai_security/mcp_security_tests/context_isolation_tests.py" "\"\"\"MCP context isolation tests.\"\"\""
create_file "tests/security/ai_security/mcp_security_tests/capability_boundary_tests.py" "\"\"\"MCP capability boundary tests.\"\"\""
create_file "tests/security/ai_security/mcp_security_tests/resource_access_control_tests.py" "\"\"\"MCP resource access control tests.\"\"\""
create_file "tests/security/ai_security/mcp_security_tests/mcp_communication_security.py" "\"\"\"MCP communication security tests.\"\"\""

# Red Team Tests
create_dir "tests/security/ai_security/red_team_tests/adversarial_scenarios"
create_dir "tests/security/ai_security/red_team_tests/attack_simulations"
create_dir "tests/security/ai_security/red_team_tests/social_engineering_tests"
create_dir "tests/security/ai_security/red_team_tests/multi_vector_attacks"

create_file "tests/security/ai_security/red_team_tests/__init__.py" ""
create_file "tests/security/ai_security/red_team_tests/adversarial_scenarios/__init__.py" ""
create_file "tests/security/ai_security/red_team_tests/attack_simulations/__init__.py" ""

# Infrastructure Security Tests
create_dir "tests/security/infrastructure_security"
create_file "tests/security/infrastructure_security/__init__.py" ""
create_file "tests/security/infrastructure_security/container_security_tests.py" "\"\"\"Container security tests.\"\"\""
create_file "tests/security/infrastructure_security/kubernetes_security_tests.py" "\"\"\"Kubernetes security tests.\"\"\""
create_file "tests/security/infrastructure_security/network_security_tests.py" "\"\"\"Network security tests.\"\"\""
create_file "tests/security/infrastructure_security/api_security_tests.py" "\"\"\"API security tests.\"\"\""

# Test fixtures
create_dir "tests/fixtures/sample_documents"
create_dir "tests/fixtures/test_templates"
create_dir "tests/fixtures/mock_data"
create_dir "tests/fixtures/attack_vectors"
create_dir "tests/fixtures/malicious_prompts"
create_dir "tests/fixtures/security_test_data"

echo "📝 Creating scripts directory..."

# Scripts structure
create_dir "scripts/setup"
create_file "scripts/setup/install_dependencies.sh" "#!/bin/bash
# Install system dependencies
echo \"Installing system dependencies...\"
sudo apt-get update
sudo apt-get install -y tesseract-ocr ghostscript libgl1-mesa-glx

# Install Python dependencies
pip install -r requirements.txt
pip install -e .

echo \"Dependencies installed successfully!\"
"

create_file "scripts/setup/setup_database.py" "#!/usr/bin/env python3
\"\"\"Database setup script.\"\"\"

def setup_database():
    \"\"\"Initialize database with required tables and data.\"\"\"
    print(\"Setting up database...\")
    # Database setup logic here
    print(\"Database setup complete!\")

if __name__ == \"__main__\":
    setup_database()
"

create_file "scripts/setup/setup_security_tools.sh" "#!/bin/bash
# Setup security tools and scanning utilities
echo \"Setting up security tools...\"

# Install security scanning tools
pip install bandit safety snyk

# Setup SAST tools
pip install semgrep

# Setup container security tools
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

echo \"Security tools setup complete!\"
"

create_dir "scripts/security"
create_file "scripts/security/run_security_scans.sh" "#!/bin/bash
# Run comprehensive security scans
echo \"Running security scans...\"

# Static analysis
bandit -r src/ -f json -o reports/bandit-report.json

# Dependency scanning
safety check --json --output reports/safety-report.json

# Container scanning
trivy fs . --format json --output reports/trivy-report.json

echo \"Security scans complete! Check reports/ directory.\"
"

create_file "scripts/security/ai_red_team_testing.py" "#!/usr/bin/env python3
\"\"\"AI Red Team Testing Script.\"\"\"

import argparse

def run_red_team_tests(test_type=\"all\"):
    \"\"\"Run red team security tests.\"\"\"
    print(f\"Running red team tests: {test_type}\")
    # Red team testing logic here
    print(\"Red team testing complete!\")

if __name__ == \"__main__\":
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--type\", default=\"all\", help=\"Test type to run\")
    parser.add_argument(\"--report-only\", action=\"store_true\", help=\"Generate report only\")
    args = parser.parse_args()
    
    run_red_team_tests(args.type)
"

create_dir "scripts/ci_cd"
create_file "scripts/ci_cd/build_pipeline.sh" "#!/bin/bash
# Build pipeline script
echo \"Starting build pipeline...\"

# Build package
python -m build

# Run tests
pytest tests/unit tests/integration

# Security scans
bash scripts/security/run_security_scans.sh

echo \"Build pipeline complete!\"
"

create_file "scripts/ci_cd/security_pipeline.sh" "#!/bin/bash
# Security pipeline script
echo \"Starting security pipeline...\"

# AI security tests
pytest tests/security/ai_security/ -v

# Infrastructure security tests
pytest tests/security/infrastructure_security/ -v

# Compliance tests
pytest tests/security/compliance_security/ -v

echo \"Security pipeline complete!\"
"

echo "📊 Creating data directory structure..."

# Data structure
create_dir "data/external/legal"
create_dir "data/external/healthcare"
create_dir "data/external/banking"
create_dir "data/external/government"
create_dir "data/external/reference_datasets/funsd"
create_dir "data/external/reference_datasets/docbank"
create_dir "data/external/reference_datasets/sroie"

create_dir "data/templates/legal"
create_dir "data/templates/healthcare"
create_dir "data/templates/banking"
create_dir "data/templates/government"
create_dir "data/templates/insurance"

create_dir "data/synthetic/generated/pdf"
create_dir "data/synthetic/generated/docx"
create_dir "data/synthetic/generated/json"
create_dir "data/synthetic/generated/images"

create_dir "data/synthetic/annotated/ocr_results"
create_dir "data/synthetic/annotated/ner_labels"
create_dir "data/synthetic/annotated/layout_tokens"
create_dir "data/synthetic/annotated/ground_truth"

create_dir "data/output/exports"
create_dir "data/output/benchmarks"
create_dir "data/output/reports"

echo "📱 Creating CLI and SDK structure..."

# CLI structure
create_dir "cli/commands"
create_file "cli/__init__.py" ""
create_file "cli/main.py" "#!/usr/bin/env python3
\"\"\"Main CLI entry point.\"\"\"

import click

@click.group()
def main():
    \"\"\"Structured Documents Synthetic Data CLI.\"\"\"
    pass

if __name__ == \"__main__\":
    main()
"

create_file "cli/commands/__init__.py" ""
create_file "cli/commands/generate.py" "\"\"\"Document generation commands.\"\"\""
create_file "cli/commands/validate.py" "\"\"\"Validation commands.\"\"\""
create_file "cli/commands/export.py" "\"\"\"Export commands.\"\"\""
create_file "cli/commands/benchmark.py" "\"\"\"Benchmark commands.\"\"\""

# SDK structure
create_dir "sdk/models"
create_dir "sdk/examples"
create_file "sdk/__init__.py" ""
create_file "sdk/client.py" "\"\"\"SDK client implementation.\"\"\""
create_file "sdk/async_client.py" "\"\"\"Async SDK client implementation.\"\"\""
create_file "sdk/models/__init__.py" ""
create_file "sdk/models/document_types.py" "\"\"\"Document type models.\"\"\""
create_file "sdk/examples/basic_usage.py" "\"\"\"Basic SDK usage examples.\"\"\""

echo "🐳 Creating deployment structure..."

# Deployment structure
create_dir "deployment/docker"
create_file "deployment/docker/Dockerfile" "FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    tesseract-ocr \\
    ghostscript \\
    libgl1-mesa-glx \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

EXPOSE 8000

CMD [\"uvicorn\", \"src.structured_docs_synth.delivery.api.rest_api:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]
"

create_file "deployment/docker/docker-compose.yml" "version: '3.8'

services:
  api:
    build: .
    ports:
      - \"8000:8000\"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/structured_docs
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=structured_docs
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    ports:
      - \"6379:6379\"

volumes:
  postgres_data:
"

create_dir "deployment/kubernetes"
create_file "deployment/kubernetes/deployment.yaml" "apiVersion: apps/v1
kind: Deployment
metadata:
  name: structured-docs-api
  namespace: synthetic-data
spec:
  replicas: 3
  selector:
    matchLabels:
      app: structured-docs-api
  template:
    metadata:
      labels:
        app: structured-docs-api
    spec:
      containers:
      - name: api
        image: structured-docs-synth:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        resources:
          requests:
            memory: \"2Gi\"
            cpu: \"1000m\"
          limits:
            memory: \"4Gi\"
            cpu: \"2000m\"
"

echo "📊 Creating comprehensive monitoring structure..."

# Monitoring structure with security and AI-specific monitoring
create_dir "monitoring/prometheus/rules"
create_dir "monitoring/prometheus/exporters"

create_file "monitoring/prometheus/prometheus.yml" "global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - \"rules/*.yml\"

scrape_configs:
  - job_name: 'structured-docs-api'
    static_configs:
      - targets: ['localhost:8000']
  
  - job_name: 'ai-metrics'
    static_configs:
      - targets: ['localhost:9090']
"

create_file "monitoring/prometheus/rules/ai_model_alerts.yml" "groups:
- name: ai_model_alerts
  rules:
  - alert: ModelDriftDetected
    expr: model_drift_score > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: \"AI model drift detected\"
  
  - alert: BiasThresholdExceeded
    expr: bias_score > 0.05
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: \"AI model bias threshold exceeded\"
"

create_file "monitoring/prometheus/rules/security_alerts.yml" "groups:
- name: security_alerts
  rules:
  - alert: PromptInjectionAttempt
    expr: prompt_injection_attempts > 5
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: \"Multiple prompt injection attempts detected\"
  
  - alert: UnauthorizedDataAccess
    expr: unauthorized_access_attempts > 3
    for: 2m
    labels:
      severity: high
    annotations:
      summary: \"Unauthorized data access attempts detected\"
"

create_dir "monitoring/security_monitoring/siem/security_rules"
create_file "monitoring/security_monitoring/siem/security_rules/ai_attack_detection.yml" "# AI Attack Detection Rules
ai_attack_rules:
  - name: \"Prompt Injection Detection\"
    pattern: \".*ignore previous instructions.*\"
    severity: \"high\"
    action: \"alert\"
  
  - name: \"Jailbreak Attempt\"
    pattern: \".*jailbreak.*|.*system prompt.*\"
    severity: \"critical\"
    action: \"block\"
  
  - name: \"Data Extraction Attempt\"
    pattern: \".*training data.*|.*model weights.*\"
    severity: \"high\"
    action: \"alert\"
"

create_dir "monitoring/ai_observability/model_monitoring"
create_dir "monitoring/ai_observability/agent_monitoring"
create_dir "monitoring/ai_observability/rag_monitoring"
create_dir "monitoring/ai_observability/synthetic_data_monitoring"

create_file "monitoring/ai_observability/model_monitoring/drift_detection.py" "\"\"\"Model drift detection utilities.\"\"\"

def detect_model_drift():
    \"\"\"Detect AI model drift.\"\"\"
    pass
"

create_dir "monitoring/chaos_engineering/ai_chaos_experiments"
create_file "monitoring/chaos_engineering/ai_chaos_experiments/model_failure_scenarios.py" "\"\"\"AI model failure chaos experiments.\"\"\"

def run_model_failure_experiment():
    \"\"\"Run AI model failure chaos experiment.\"\"\"
    pass
"

echo "📓 Creating notebooks structure..."

# Notebooks structure
create_dir "notebooks/01_data_exploration"
create_dir "notebooks/02_generation_examples"
create_dir "notebooks/03_quality_analysis"
create_dir "notebooks/04_privacy_compliance"
create_dir "notebooks/05_integration_demos"

create_file "notebooks/01_data_exploration/explore_legal_documents.ipynb" "{
 \"cells\": [],
 \"metadata\": {},
 \"nbformat\": 4,
 \"nbformat_minor\": 4
}"

create_file "notebooks/02_generation_examples/generate_contracts.ipynb" "{
 \"cells\": [],
 \"metadata\": {},
 \"nbformat\": 4,
 \"nbformat_minor\": 4
}"

echo "🔧 Creating additional utility files..."

# Additional utility files
create_file ".gitignore" "# Python
__pycache__/
*.py[cod]
*\$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data
data/synthetic/generated/
data/output/
*.pdf
*.docx

# Secrets
.env
*.key
*.pem
secrets/

# Logs
logs/
*.log

# Test coverage
htmlcov/
.coverage
.pytest_cache/

# Security reports
bandit-report.json
safety-report.json
trivy-report.json
"

create_file ".env.example" "# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/structured_docs
REDIS_URL=redis://localhost:6379

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=your-secret-key-here

# External API Keys
SEC_EDGAR_API_KEY=your-sec-api-key
COURTLISTENER_API_KEY=your-court-api-key

# AI/ML Configuration
OPENAI_API_KEY=your-openai-key
HUGGINGFACE_API_KEY=your-hf-key

# Security Configuration
ENCRYPTION_KEY=your-encryption-key
JWT_SECRET=your-jwt-secret

# Monitoring
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_ENABLED=true

# Privacy Settings
DIFFERENTIAL_PRIVACY_EPSILON=0.1
PII_DETECTION_ENABLED=true
"

create_file "Makefile" "# Makefile for Structured Documents Synthetic Data Platform

.PHONY: help install test security-test lint format clean build deploy

help:
	@echo \"Available commands:\"
	@echo \"  install         Install dependencies\"
	@echo \"  test           Run all tests\"
	@echo \"  security-test  Run security tests\"
	@echo \"  lint           Run linting\"
	@echo \"  format         Format code\"
	@echo \"  clean          Clean build artifacts\"
	@echo \"  build          Build package\"
	@echo \"  deploy         Deploy to production\"

install:
	pip install -r requirements.txt
	pip install -e .
	bash scripts/setup/setup_security_tools.sh

test:
	pytest tests/unit tests/integration -v --cov=src

security-test:
	pytest tests/security/ -v
	bash scripts/security/run_security_scans.sh

lint:
	flake8 src/ tests/
	mypy src/
	bandit -r src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name \"*.pyc\" -delete

build:
	python -m build

deploy:
	bash scripts/deployment/deploy_kubernetes.sh
"

create_file "CHANGELOG.md" "# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Initial repository structure
- Comprehensive AI security testing framework
- Multi-modal document generation pipeline
- Privacy-preserving synthetic data generation
- Real-time data integration capabilities
- Enterprise compliance framework (GDPR, HIPAA, PCI-DSS, SOX)
- CI/CD pipeline with security gates
- Monitoring and observability infrastructure

### Security
- AI/ML security testing (LLM, Agent, RAG security)
- Differential privacy implementation
- PII detection and masking
- Comprehensive audit logging
- Red team testing framework

## [0.1.0] - 2025-01-XX

### Added
- Initial release
- Basic document generation capabilities
- Template system for structured documents
- OCR and NLP processing pipeline
- REST API and CLI interface
"

echo ""
echo "✅ Repository structure created successfully!"
echo ""
echo "📁 Repository: $REPO_DIR"
echo "📊 Total directories created: $(find $REPO_DIR -type d | wc -l)"
echo "📄 Total files created: $(find $REPO_DIR -type f | wc -l)"
echo ""
echo "🚀 Next steps:"
echo "1. cd $REPO_DIR"
echo "2. python -m venv venv"
echo "3. source venv/bin/activate"
echo "4. make install"
echo "5. make test"
echo ""
echo "🔐 Security features included:"
echo "- AI/ML security testing (LLM, Agent, RAG, MCP)"
echo "- Comprehensive CI/CD security pipeline"
echo "- Privacy and compliance frameworks"
echo "- Red team testing capabilities"
echo "- Real-time security monitoring"
echo ""
echo "🏗️  Repository structure is ready for development!"
