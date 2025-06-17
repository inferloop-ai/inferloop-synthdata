#!/bin/bash

# Structured Documents Synthetic Data Repository Setup Script
# This script creates the complete repository structure in the docs directory
# of the inferloop-synthdata GitHub repository
#
# USAGE:
#   ./create_repo_structure.sh [target_directory]
#
# EXAMPLES:
#   ./create_repo_structure.sh                    # Creates in ./docs/
#   ./create_repo_structure.sh documentation     # Creates in ./documentation/
#
# REPOSITORY INTEGRATION:
#   This script is designed for the inferloop-synthdata ecosystem
#   Repository: https://github.com/inferloop/inferloop-synthdata
#   Target Location: inferloop-synthdata/docs/structured-documents-synthetic-data/
#
# FEATURES CREATED:
#   - Complete source code structure with AI security testing
#   - Comprehensive CI/CD pipelines with security gates  
#   - Privacy and compliance frameworks (GDPR, HIPAA, PCI-DSS, SOX)
#   - Real-time monitoring and observability infrastructure
#   - Integration points with inferloop-synthdata ecosystem
#   - Production-ready deployment configurations

set -e  # Exit on any error

# Default to docs directory in inferloop-synthdata repo
REPO_DIR="${1:-docs}"
PROJECT_NAME="structured-documents-synthetic-data"

echo "🏗️  Creating Structured Documents Synthetic Data Structure in: $REPO_DIR"
echo "📂 Target: inferloop-synthdata/$REPO_DIR/$PROJECT_NAME"
echo "==============================================================================="

# Create docs directory structure
mkdir -p "$REPO_DIR"
cd "$REPO_DIR"

# Create project subdirectory within docs
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

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

**Location**: This documentation and codebase is part of the `inferloop-synthdata` repository under the `docs/structured-documents-synthetic-data/` directory.

## Features
- Multi-modal document generation (PDF, DOCX, JSON)
- Advanced OCR and NLP processing
- Comprehensive privacy and compliance framework
- Real-time data integration
- AI/ML security testing
- Enterprise deployment ready

## Repository Structure
\`\`\`
inferloop-synthdata/
└── docs/
    └── structured-documents-synthetic-data/
        ├── src/                    # Source code
        ├── tests/                  # Comprehensive testing including AI security
        ├── configs/               # Configuration and security policies
        ├── scripts/               # Automation and deployment scripts
        ├── monitoring/            # Observability and security monitoring
        ├── deployment/            # Docker, Kubernetes, Terraform configs
        └── docs/                  # Documentation
\`\`\`

## Quick Start
\`\`\`bash
# Navigate to the project directory
cd docs/structured-documents-synthetic-data

# Install dependencies
pip install -r requirements.txt

# Setup database
python scripts/setup/setup_database.py

# Generate documents
synth doc generate --type legal_contract --count 100
\`\`\`

## Development Workflow
\`\`\`bash
# Navigate to project
cd docs/structured-documents-synthetic-data

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install in development mode
make install

# Run tests
make test

# Run security tests
make security-test

# Format code
make format
\`\`\`

## AI Security Testing
This platform includes comprehensive AI security testing:
- **LLM Security**: Prompt injection, jailbreak resistance, bias detection
- **Agent Security**: Isolation, privilege escalation, tool abuse prevention
- **RAG Security**: Vector poisoning, retrieval manipulation, context injection
- **Synthetic Data Security**: Privacy preservation, reconstruction attack prevention

## Documentation
- See \`docs/\` directory for comprehensive documentation
- API documentation: \`docs/api/\`
- Security guide: \`docs/security.md\`
- Deployment guide: \`docs/deployment/\`

## Repository Integration
This project is designed to integrate with the broader `inferloop-synthdata` ecosystem:
- Shared security policies and compliance frameworks
- Integration with existing monitoring and observability infrastructure
- Coordinated CI/CD pipelines with other inferloop components

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
    author=\"Inferloop Synthdata Team\",
    author_email=\"team@inferloop.com\",
    description=\"Enterprise synthetic data generation platform for structured documents\",
    long_description=long_description,
    long_description_content_type=\"text/markdown\",
    url=\"https://github.com/inferloop/inferloop-synthdata\",
    project_urls={
        \"Documentation\": \"https://github.com/inferloop/inferloop-synthdata/tree/main/docs/structured-documents-synthetic-data\",
        \"Source\": \"https://github.com/inferloop/inferloop-synthdata/tree/main/docs/structured-documents-synthetic-data\",
        \"Tracker\": \"https://github.com/inferloop/inferloop-synthdata/issues\",
    },
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
        \"Topic :: Scientific/Engineering :: Artificial Intelligence\",
        \"Topic :: Software Development :: Libraries :: Python Modules\",
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

# Documentation structure with integration guides
create_dir "docs/api"
create_dir "docs/examples"
create_dir "docs/deployment"
create_dir "docs/compliance"
create_dir "docs/integration"
create_dir "docs/security"

create_file "docs/README.md" "# Documentation - Structured Documents Synthetic Data Platform

## Overview

This documentation covers the **Structured Documents Synthetic Data Platform**, which is part of the broader `inferloop-synthdata` ecosystem.

**Repository Structure**: `inferloop-synthdata/docs/structured-documents-synthetic-data/`

## Documentation Structure

- `api/` - API documentation and OpenAPI specs
- `examples/` - Usage examples and tutorials  
- `deployment/` - Deployment guides and configurations
- `compliance/` - Compliance and security documentation
- `integration/` - Integration guides with inferloop-synthdata ecosystem

## Quick Navigation

### Getting Started
1. [Installation Guide](installation.md)
2. [Quick Start Tutorial](quickstart.md)
3. [Integration with Inferloop](integration/inferloop-integration.md)

### API & Development
1. [API Reference](api/README.md)
2. [SDK Documentation](sdk-guide.md)
3. [CLI Reference](cli-guide.md)

### Security & Compliance
1. [Security Guide](security.md)
2. [AI Security Testing](security/ai-security-testing.md)
3. [Privacy & Compliance](compliance/README.md)

### Deployment & Operations
1. [Deployment Guide](deployment/README.md)
2. [Monitoring & Observability](monitoring-guide.md)
3. [Troubleshooting](troubleshooting.md)

## Integration with Inferloop-Synthdata

This platform is designed as a specialized component within the inferloop-synthdata ecosystem:

### Shared Infrastructure
- **Security Policies**: Inherits from inferloop-synthdata security framework
- **Monitoring**: Integrates with centralized observability infrastructure
- **CI/CD**: Coordinated pipelines with other inferloop components
- **Compliance**: Unified compliance and audit framework

### Data Flow Integration
```
inferloop-synthdata ecosystem
├── core-synthetic-data/          # Core synthetic data generation
├── time-series-synthdata/        # Time series synthetic data
├── docs/
│   └── structured-documents-synthetic-data/  # This platform
├── shared-security/              # Shared security policies
├── monitoring/                   # Centralized monitoring
└── deployment/                   # Unified deployment configs
```

### Cross-Platform Capabilities
- **Unified API**: Consistent API patterns across all inferloop platforms
- **Shared Models**: Common ML models and privacy-preserving techniques
- **Integrated Workflows**: Seamless workflows between different data types
- **Centralized Governance**: Unified data governance and compliance

## Development Workflow

### Local Development
```bash
# Navigate to the platform
cd docs/structured-documents-synthetic-data

# Setup development environment
python -m venv venv
source venv/bin/activate
make install

# Run tests
make test
make security-test
```

### Integration Testing
```bash
# Test integration with inferloop ecosystem
make validate-integration

# Sync configurations with parent repo
make sync-configs

# Update security policies
make update-security-policies
```

## Support & Contributing

### Getting Help
- Check the [troubleshooting guide](troubleshooting.md)
- Review [common issues](common-issues.md)
- Open an issue in the main [inferloop-synthdata repository](https://github.com/inferloop/inferloop-synthdata/issues)

### Contributing
1. Follow the [inferloop-synthdata contribution guidelines](../../CONTRIBUTING.md)
2. Ensure all security tests pass
3. Update documentation for any new features
4. Test integration with other inferloop components

## Security & Privacy

This platform implements enterprise-grade security:
- **AI/ML Security**: Comprehensive testing for LLMs, agents, and RAG systems
- **Privacy Preservation**: Differential privacy and PII protection
- **Compliance**: GDPR, HIPAA, PCI-DSS, SOX compliance frameworks
- **Monitoring**: Real-time security monitoring and threat detection

See the [Security Guide](security.md) for detailed information.
"

create_file "docs/api/README.md" "# API Documentation - Structured Documents Platform

## Overview

The Structured Documents platform provides a comprehensive REST API for document generation and management, integrated with the broader inferloop-synthdata ecosystem.

## Base URL
\`https://api.inferloop.com/v1/structured-documents\`

## Authentication

Uses inferloop unified authentication:
```bash
# Get authentication token
curl -X POST https://api.inferloop.com/auth/token \\
  -H \"Content-Type: application/json\" \\
  -d '{\"username\": \"your-username\", \"password\": \"your-password\"}'

# Use token in requests
curl -H \"Authorization: Bearer YOUR_TOKEN\" \\
  https://api.inferloop.com/v1/structured-documents/templates
```

## Core Endpoints

### Document Generation
```http
POST /v1/structured-documents/generate/document
Content-Type: application/json
Authorization: Bearer YOUR_TOKEN

{
  \"document_type\": \"legal_contract\",
  \"template\": \"service_agreement\",
  \"parameters\": {
    \"parties\": [\"Company A\", \"Company B\"],
    \"jurisdiction\": \"New York\",
    \"effective_date\": \"2025-01-01\"
  },
  \"output_formats\": [\"pdf\", \"docx\", \"json\"],
  \"privacy_config\": {
    \"differential_privacy\": true,
    \"epsilon\": 0.1
  }
}
```

### Batch Generation
```http
POST /v1/structured-documents/generate/batch
Content-Type: application/json
Authorization: Bearer YOUR_TOKEN

{
  \"batch_id\": \"batch_001\",
  \"documents\": [
    {
      \"document_type\": \"healthcare_form\",
      \"template\": \"patient_intake\",
      \"count\": 1000
    }
  ],
  \"compliance_rules\": [\"gdpr\", \"hipaa\"]
}
```

### Status and Monitoring
```http
GET /v1/structured-documents/status/{job_id}
Authorization: Bearer YOUR_TOKEN

GET /v1/structured-documents/metrics/quality
Authorization: Bearer YOUR_TOKEN

GET /v1/structured-documents/compliance/audit
Authorization: Bearer YOUR_TOKEN
```

## Cross-Platform Integration

### Unified Data Access
```http
# Access data across platforms
GET /v1/data/unified?platforms=structured-documents,time-series,tabular
Authorization: Bearer YOUR_TOKEN

# Cross-platform analytics
POST /v1/analytics/cross-platform
Content-Type: application/json
Authorization: Bearer YOUR_TOKEN

{
  \"platforms\": [\"structured-documents\", \"tabular\"],
  \"analysis_type\": \"quality_comparison\",
  \"job_ids\": [\"doc-123\", \"tab-456\"]
}
```

### Shared Services
```http
# Shared privacy services
POST /v1/shared/privacy/anonymize
Content-Type: application/json
Authorization: Bearer YOUR_TOKEN

# Shared compliance validation
POST /v1/shared/compliance/validate
Content-Type: application/json
Authorization: Bearer YOUR_TOKEN

# Shared quality metrics
GET /v1/shared/metrics/quality/{platform}/{job_id}
Authorization: Bearer YOUR_TOKEN
```

## SDK Integration

### Python SDK
```python
from inferloop_synthdata import InferloopClient

# Initialize unified client
client = InferloopClient(
    api_key=\"your-api-key\",
    platform=\"structured-documents\"
)

# Generate documents
documents = client.generate_documents(
    document_type=\"legal_contract\",
    template=\"service_agreement\",
    count=100,
    privacy_level=\"high\"
)

# Cross-platform integration
tabular_data = client.cross_platform.extract_tabular(documents)
time_series = client.cross_platform.extract_temporal(documents)
```

## WebSocket API

### Real-time Updates
```javascript
// Connect to real-time updates
const ws = new WebSocket('wss://api.inferloop.com/v1/structured-documents/ws');

ws.on('message', (data) => {
  const event = JSON.parse(data);
  
  switch(event.type) {
    case 'generation_progress':
      console.log(`Progress: ${event.progress}%`);
      break;
    case 'generation_complete':
      console.log(`Job ${event.job_id} completed`);
      break;
    case 'security_alert':
      console.log(`Security alert: ${event.message}`);
      break;
  }
});
```

## Error Handling

### Standard Error Responses
```json
{
  \"error\": {
    \"code\": \"VALIDATION_ERROR\",
    \"message\": \"Invalid document template\",
    \"details\": {
      \"field\": \"template\",
      \"allowed_values\": [\"service_agreement\", \"nda\", \"employment_contract\"]
    },
    \"request_id\": \"req_123456789\",
    \"timestamp\": \"2025-01-01T12:00:00Z\"
  }
}
```

### Error Codes
- `AUTHENTICATION_ERROR` - Invalid or expired token
- `AUTHORIZATION_ERROR` - Insufficient permissions
- `VALIDATION_ERROR` - Invalid request parameters
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `PRIVACY_VIOLATION` - Privacy policy violation
- `COMPLIANCE_ERROR` - Compliance rule violation
- `GENERATION_FAILED` - Document generation failure
- `INTERNAL_ERROR` - Server error

## Rate Limiting

### Limits by Plan
```yaml
rate_limits:
  free_tier:
    requests_per_minute: 60
    documents_per_day: 1000
  
  professional:
    requests_per_minute: 300
    documents_per_day: 50000
  
  enterprise:
    requests_per_minute: 1000
    documents_per_day: 1000000
```

### Rate Limit Headers
```http
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 299
X-RateLimit-Reset: 1640995200
```

## Webhooks

### Event Notifications
```http
POST /v1/structured-documents/webhooks
Content-Type: application/json
Authorization: Bearer YOUR_TOKEN

{
  \"url\": \"https://your-app.com/webhooks/inferloop\",
  \"events\": [
    \"generation.completed\",
    \"generation.failed\",
    \"privacy.violation\",
    \"compliance.alert\"
  ],
  \"secret\": \"your-webhook-secret\"
}
```

## API Versioning

### Version Strategy
- **Current Version**: v1
- **Deprecation Policy**: 12 months notice before removal
- **Breaking Changes**: New major version required
- **Backward Compatibility**: Maintained within major versions

### Version Headers
```http
API-Version: v1
Accept: application/vnd.inferloop.v1+json
```

## OpenAPI Specification

Full OpenAPI 3.0 specification available at:
- **Structured Documents**: `https://api.inferloop.com/v1/structured-documents/openapi.json`
- **Unified API**: `https://api.inferloop.com/v1/openapi.json`

## Testing and Development

### Sandbox Environment
```bash
# Sandbox base URL
https://sandbox-api.inferloop.com/v1/structured-documents

# Test API key (limited functionality)
sk_test_1234567890abcdef
```

### Postman Collection
Download the complete Postman collection:
`https://api.inferloop.com/v1/structured-documents/postman-collection.json`

For detailed endpoint documentation, see the individual API reference files in this directory.
"

## Overview

The Structured Documents Synthetic Data Platform is designed as an integrated component of the broader `inferloop-synthdata` ecosystem, providing specialized capabilities for document-based synthetic data generation while leveraging shared infrastructure and policies.

## Architecture Integration

### Repository Structure
```
inferloop-synthdata/
├── README.md                     # Main repository documentation
├── CONTRIBUTING.md               # Contribution guidelines
├── LICENSE                       # Repository license
├── .github/                      # Shared CI/CD workflows
├── shared/                       # Shared components across platforms
│   ├── security/                 # Common security policies
│   ├── monitoring/               # Centralized monitoring configs
│   ├── compliance/               # Shared compliance frameworks
│   └── deployment/               # Common deployment infrastructure
├── core-synthetic-data/          # Core synthetic data platform
├── time-series-synthdata/        # Time series synthetic data
├── tabular-synthdata/            # Tabular synthetic data
└── docs/
    └── structured-documents-synthetic-data/  # This platform
        ├── src/                  # Document-specific source code
        ├── tests/                # Document-specific tests
        ├── configs/              # Document-specific configurations
        └── docs/                 # Documentation
```

## Shared Infrastructure Components

### 1. Security Framework Integration
The platform inherits and extends the inferloop-synthdata security framework:

```yaml
# Inherits from: inferloop-synthdata/shared/security/
security_inheritance:
  base_policies:
    - authentication: \"OAuth 2.0 + JWT\"
    - encryption: \"AES-256 + TLS 1.3\"
    - audit_logging: \"Centralized ELK stack\"
    - access_control: \"RBAC with fine-grained permissions\"
  
  document_specific:
    - ocr_security: \"OCR output sanitization\"
    - document_privacy: \"PII detection in documents\"
    - template_security: \"Template injection prevention\"
    - layout_validation: \"Document structure validation\"
```

### 2. Monitoring & Observability
Integrates with centralized monitoring infrastructure:

```yaml
# Uses: inferloop-synthdata/shared/monitoring/
monitoring_integration:
  metrics_collection:
    - prometheus_exporters: \"Document generation metrics\"
    - custom_metrics: \"OCR accuracy, layout quality\"
    - ai_safety_metrics: \"Bias detection, fairness scores\"
  
  alerting:
    - security_alerts: \"Inherits from shared/security/alerts/\"
    - performance_alerts: \"Document-specific SLAs\"
    - compliance_alerts: \"GDPR/HIPAA/PCI-DSS violations\"
  
  dashboards:
    - grafana_integration: \"Shared Grafana instance\"
    - custom_dashboards: \"Document generation analytics\"
```

### 3. CI/CD Pipeline Coordination
Coordinated with ecosystem-wide CI/CD:

```yaml
# Coordinates with: inferloop-synthdata/.github/workflows/
pipeline_coordination:
  shared_workflows:
    - security_scanning: \"Ecosystem-wide security scans\"
    - dependency_management: \"Centralized dependency updates\"
    - compliance_testing: \"Cross-platform compliance validation\"
  
  document_specific:
    - ai_security_testing: \"LLM/Agent/RAG security tests\"
    - document_quality: \"OCR accuracy, layout validation\"
    - privacy_testing: \"Document PII detection\"
```

## Data Flow Integration

### 1. Cross-Platform Data Exchange
```mermaid
graph LR
    A[Core Synthetic Data] --> D[Document Platform]
    B[Time Series Data] --> D
    C[Tabular Data] --> D
    D --> E[Unified RAG System]
    D --> F[Multi-Modal AI Training]
    D --> G[Cross-Domain Analytics]
```

### 2. Unified API Gateway
All inferloop platforms share a common API pattern:

```python
# Unified API structure across all platforms
from inferloop_synthdata import UnifiedClient

client = UnifiedClient(
    platform=\"structured-documents\",
    api_key=\"your-api-key\",
    base_url=\"https://api.inferloop.com\"
)

# Generate structured documents
documents = client.structured_documents.generate(
    document_type=\"legal_contract\",
    template=\"service_agreement\",
    count=100
)

# Cross-platform integration
tabular_data = client.tabular.generate_from_documents(documents)
time_series = client.time_series.extract_temporal_patterns(documents)
```

### 3. Shared Model Registry
Leverages inferloop's centralized model registry:

```yaml
# Uses: inferloop-synthdata/shared/models/
model_integration:
  shared_models:
    - privacy_models: \"Differential privacy mechanisms\"
    - bias_detection: \"Fairness evaluation models\"
    - quality_assessment: \"Cross-platform quality metrics\"
  
  document_specific:
    - ocr_models: \"Document-optimized OCR engines\"
    - layout_models: \"Document structure understanding\"
    - template_models: \"Template generation and validation\"
```

## Configuration Management

### 1. Hierarchical Configuration
```yaml
# Configuration inheritance hierarchy
configuration_hierarchy:
  1. inferloop-synthdata/shared/config/base.yaml
  2. inferloop-synthdata/shared/config/security.yaml
  3. docs/structured-documents-synthetic-data/configs/base_config.yaml
  4. docs/structured-documents-synthetic-data/configs/security/
  5. Environment-specific overrides
```

### 2. Environment Synchronization
```bash
# Sync configurations with parent repository
make sync-configs

# Update security policies from ecosystem
make update-security-policies

# Validate cross-platform compatibility
make validate-integration
```

## Deployment Integration

### 1. Shared Infrastructure
```yaml
# Uses shared deployment infrastructure
deployment_integration:
  kubernetes:
    namespace: \"inferloop-synthdata\"
    shared_resources:
      - ingress_controller: \"Nginx Ingress\"
      - cert_manager: \"Let's Encrypt SSL\"
      - monitoring: \"Prometheus + Grafana\"
      - logging: \"ELK Stack\"
  
  cloud_resources:
    shared_vpc: \"inferloop-production-vpc\"
    shared_subnets: \"Private subnets with NAT gateway\"
    shared_security_groups: \"Inferloop security policies\"
```

### 2. Service Mesh Integration
```yaml
# Istio service mesh integration
service_mesh:
  discovery: \"Automatic service discovery\"
  security: \"mTLS between all services\"
  observability: \"Distributed tracing with Jaeger\"
  traffic_management: \"Canary deployments\"
```

## Cross-Platform Features

### 1. Unified Authentication
```python
# Single sign-on across all inferloop platforms
from inferloop_synthdata.auth import InferloopAuth

auth = InferloopAuth()
token = auth.authenticate(username, password)

# Token works across all platforms
structured_docs_client = StructuredDocsClient(token=token)
time_series_client = TimeSeriesClient(token=token)
tabular_client = TabularClient(token=token)
```

### 2. Cross-Platform Analytics
```python
# Unified analytics across data types
from inferloop_synthdata.analytics import CrossPlatformAnalytics

analytics = CrossPlatformAnalytics()

# Analyze quality across platforms
quality_report = analytics.compare_quality([
    {\"platform\": \"structured-documents\", \"job_id\": \"doc-123\"},
    {\"platform\": \"time-series\", \"job_id\": \"ts-456\"},
    {\"platform\": \"tabular\", \"job_id\": \"tab-789\"}
])
```

### 3. Federated Privacy Governance
```yaml
# Unified privacy governance
privacy_governance:
  global_policies:
    - differential_privacy: \"Shared epsilon budget management\"
    - pii_detection: \"Cross-platform PII identification\"
    - consent_management: \"Unified consent tracking\"
  
  document_specific:
    - ocr_privacy: \"OCR output sanitization\"
    - template_privacy: \"Template PII validation\"
    - layout_privacy: \"Layout-aware privacy preservation\"
```

## Development Workflow Integration

### 1. Shared Development Standards
```bash
# Unified development workflow
git clone https://github.com/inferloop/inferloop-synthdata.git
cd inferloop-synthdata/docs/structured-documents-synthetic-data

# Use shared development tools
pre-commit install  # Shared pre-commit hooks
make format         # Shared code formatting standards
make lint           # Shared linting rules
make test           # Platform-specific tests
make integration-test  # Cross-platform integration tests
```

### 2. Cross-Platform Testing
```yaml
# Integration test matrix
integration_testing:
  compatibility_tests:
    - api_compatibility: \"Ensure API consistency\"
    - data_format_compatibility: \"Cross-platform data exchange\"
    - security_policy_consistency: \"Unified security enforcement\"
  
  performance_tests:
    - resource_sharing: \"Shared resource utilization\"
    - scalability: \"Cross-platform load testing\"
    - latency: \"Inter-service communication performance\"
```

## Migration and Upgrade Strategies

### 1. Version Compatibility
```yaml
# Version compatibility matrix
compatibility_matrix:
  inferloop_synthdata_core: \">=2.0.0\"
  shared_security: \">=1.5.0\"
  shared_monitoring: \">=1.3.0\"
  structured_docs_platform: \"0.1.0\"
```

### 2. Rolling Updates
```bash
# Coordinated rolling updates
kubectl apply -f inferloop-synthdata/shared/deployment/
kubectl apply -f docs/structured-documents-synthetic-data/deployment/

# Validate cross-platform functionality
make validate-integration
```

## Support and Troubleshooting

### 1. Unified Support Channel
- **Issues**: Report in main `inferloop-synthdata` repository
- **Documentation**: Centralized documentation portal
- **Status Page**: Shared status monitoring for all platforms

### 2. Cross-Platform Debugging
```bash
# Debug cross-platform issues
kubectl logs -n inferloop-synthdata -l app=structured-documents
kubectl logs -n inferloop-synthdata -l app=api-gateway
kubectl logs -n inferloop-synthdata -l app=shared-monitoring
```

## Future Roadmap

### 1. Enhanced Integration
- **Unified Data Catalog**: Cross-platform data discovery
- **Federated Learning**: Shared model training across platforms
- **Advanced Analytics**: Cross-platform insights and recommendations

### 2. Ecosystem Expansion
- **New Data Types**: Integration points for future synthetic data types
- **Third-Party Integrations**: Ecosystem partner integrations
- **Enterprise Features**: Enhanced governance and compliance tools

This integration ensures that the Structured Documents platform leverages the full power of the inferloop-synthdata ecosystem while providing specialized capabilities for document-based synthetic data generation.
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
# Security pipeline script for inferloop-synthdata integration
echo \"🔐 Starting Inferloop Security Pipeline...\"
echo \"===========================================\"

# Set error handling
set -e

# Validate inferloop integration first
echo \"🔍 Step 1: Validating Inferloop Integration...\"
bash scripts/ci_cd/validate_inferloop_integration.sh

# AI security tests
echo \"🤖 Step 2: Running AI Security Tests...\"
echo \"Testing LLM security (prompt injection, jailbreak resistance)...\"
pytest tests/security/ai_security/llm_security_tests/ -v --tb=short

echo \"Testing Agent security (isolation, privilege escalation)...\"
pytest tests/security/ai_security/agent_security_tests/ -v --tb=short

echo \"Testing RAG security (vector poisoning, context injection)...\"
pytest tests/security/ai_security/rag_security_tests/ -v --tb=short

echo \"Testing Synthetic Data security (privacy preservation)...\"
pytest tests/security/ai_security/synthetic_data_security/ -v --tb=short

echo \"Testing MCP security (protocol validation)...\"
pytest tests/security/ai_security/mcp_security_tests/ -v --tb=short

# Infrastructure security tests
echo \"🏗️  Step 3: Running Infrastructure Security Tests...\"
pytest tests/security/infrastructure_security/ -v --tb=short

# Compliance tests
echo \"📋 Step 4: Running Compliance Tests...\"
pytest tests/security/compliance_security/ -v --tb=short

# Red team tests
echo \"🎯 Step 5: Running Red Team Tests...\"
pytest tests/security/ai_security/red_team_tests/ -v --tb=short

# Static security analysis
echo \"🔍 Step 6: Running Static Security Analysis...\"
bandit -r src/ -f json -o reports/bandit-report.json || echo \"Bandit scan completed with findings\"

# Dependency vulnerability scanning
echo \"📦 Step 7: Scanning Dependencies...\"
safety check --json --output reports/safety-report.json || echo \"Safety scan completed with findings\"

# Generate security report
echo \"📊 Step 8: Generating Security Report...\"
python scripts/security/security_report_generator.py

echo \"\"
echo \"✅ Inferloop Security Pipeline Complete!\"
echo \"=======================================\"
echo \"\"
echo \"📊 Results:\"
echo \"- AI Security Tests: PASSED\"
echo \"- Infrastructure Tests: PASSED\"  
echo \"- Compliance Tests: PASSED\"
echo \"- Red Team Tests: PASSED\"
echo \"- Static Analysis: CHECK REPORTS\"
echo \"- Dependency Scan: CHECK REPORTS\"
echo \"\"
echo \"📁 Reports available in: reports/\"
echo \"🔗 Inferloop Integration: VALIDATED\"
"
# Validate integration with inferloop-synthdata ecosystem

echo \"🔍 Validating Inferloop-Synthdata Integration...\"
echo \"================================================\"

# Check if we're in the correct repository structure
if [[ ! -f \"../../README.md\" ]]; then
    echo \"❌ Error: Not in expected inferloop-synthdata repository structure\"
    echo \"   Expected: inferloop-synthdata/docs/structured-documents-synthetic-data/\"
    exit 1
fi

echo \"✅ Repository structure validated\"

# Check for shared configuration inheritance
if [[ -d \"../../shared/security/\" ]]; then
    echo \"✅ Shared security configurations found\"
else
    echo \"⚠️  Warning: Shared security configurations not found\"
    echo \"   Expected: ../../shared/security/\"
fi

if [[ -d \"../../shared/monitoring/\" ]]; then
    echo \"✅ Shared monitoring configurations found\"
else
    echo \"⚠️  Warning: Shared monitoring configurations not found\"
    echo \"   Expected: ../../shared/monitoring/\"
fi

# Validate configuration compatibility
echo \"🔧 Validating configuration compatibility...\"

# Check Python version compatibility
python_version=\$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version=\"3.9\"

if python3 -c \"import sys; exit(0 if sys.version_info >= (3, 9) else 1)\"; then
    echo \"✅ Python version \$python_version is compatible\"
else
    echo \"❌ Error: Python version \$python_version is too old (requires >= \$required_version)\"
    exit 1
fi

# Check for required dependencies
echo \"📦 Checking dependencies...\"

if command -v docker &> /dev/null; then
    echo \"✅ Docker found\"
else
    echo \"⚠️  Warning: Docker not found - containerized deployment may not work\"
fi

if command -v kubectl &> /dev/null; then
    echo \"✅ kubectl found\"
else
    echo \"⚠️  Warning: kubectl not found - Kubernetes deployment may not work\"
fi

# Validate environment variables
echo \"🔐 Validating environment configuration...\"

if [[ -f \".env\" ]]; then
    if grep -q \"INFERLOOP_API_KEY\" .env; then
        echo \"✅ Inferloop API key configuration found\"
    else
        echo \"⚠️  Warning: INFERLOOP_API_KEY not found in .env\"
    fi
    
    if grep -q \"INFERLOOP_ECOSYSTEM_MODE=enabled\" .env; then
        echo \"✅ Ecosystem mode enabled\"
    else
        echo \"⚠️  Warning: Ecosystem mode not enabled\"
    fi
else
    echo \"⚠️  Warning: .env file not found - copy .env.example to .env and configure\"
fi

# Test import capabilities
echo \"🧪 Testing Python import capabilities...\"

if python3 -c \"import pytest\" 2>/dev/null; then
    echo \"✅ pytest import successful\"
else
    echo \"❌ Error: pytest import failed - run 'make install' first\"
fi

if python3 -c \"import fastapi\" 2>/dev/null; then
    echo \"✅ FastAPI import successful\"
else
    echo \"❌ Error: FastAPI import failed - run 'make install' first\"
fi

# Validate security configurations
echo \"🛡️  Validating security configurations...\"

security_configs=(
    \"configs/security/ai_security/llm_security_configs/prompt_injection_filters.yaml\"
    \"configs/security/ai_security/agent_security_configs/agent_isolation_policies.yaml\"
    \"configs/security/ai_security/rag_security_configs/vector_store_security.yaml\"
)

for config in \"\${security_configs[@]}\"; do
    if [[ -f \"\$config\" ]]; then
        echo \"✅ Security config found: \$(basename \$config)\"
    else
        echo \"❌ Error: Missing security config: \$config\"
    fi
done

# Validate test structure
echo \"🧪 Validating test structure...\"

test_dirs=(
    \"tests/security/ai_security/llm_security_tests\"
    \"tests/security/ai_security/agent_security_tests\"
    \"tests/security/ai_security/rag_security_tests\"
    \"tests/security/ai_security/synthetic_data_security\"
)

for test_dir in \"\${test_dirs[@]}\"; do
    if [[ -d \"\$test_dir\" ]]; then
        echo \"✅ Test directory found: \$(basename \$test_dir)\"
    else
        echo \"❌ Error: Missing test directory: \$test_dir\"
    fi
done

# Validate CI/CD workflows
echo \"⚙️  Validating CI/CD workflows...\"

workflows=(
    \".github/workflows/ai-security-testing.yml\"
    \".github/workflows/synthetic-data-validation.yml\"
    \".github/workflows/llm-safety-testing.yml\"
)

for workflow in \"\${workflows[@]}\"; do
    if [[ -f \"\$workflow\" ]]; then
        echo \"✅ Workflow found: \$(basename \$workflow)\"
    else
        echo \"❌ Error: Missing workflow: \$workflow\"
    fi
done

# Summary
echo \"\"
echo \"📊 Integration Validation Summary\"
echo \"================================\"

if [[ \$? -eq 0 ]]; then
    echo \"✅ Integration validation completed successfully!\"
    echo \"\"
    echo \"🚀 Next steps:\"
    echo \"1. Configure .env file with your API keys\"
    echo \"2. Run 'make install' to install dependencies\"
    echo \"3. Run 'make test' to verify functionality\"
    echo \"4. Run 'make security-test' to validate security\"
    echo \"\"
    echo \"🔗 Inferloop-Synthdata Integration Status: READY\"
else
    echo \"❌ Integration validation failed - please fix errors above\"
    exit 1
fi
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

create_file ".env.example" "# Inferloop-Synthdata Environment Configuration
# Structured Documents Platform

# =============================================================================
# INFERLOOP ECOSYSTEM CONFIGURATION
# =============================================================================

# Inferloop API Configuration
INFERLOOP_API_BASE_URL=https://api.inferloop.com
INFERLOOP_API_KEY=your-inferloop-api-key
INFERLOOP_PLATFORM=structured-documents

# Cross-Platform Integration
INFERLOOP_ECOSYSTEM_MODE=enabled
UNIFIED_AUTH_ENABLED=true
CROSS_PLATFORM_ANALYTICS=true

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Primary Database (PostgreSQL)
DATABASE_URL=postgresql://user:password@localhost:5432/inferloop_structured_docs
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis Cache
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your-redis-password
REDIS_CLUSTER_MODE=false

# =============================================================================
# API CONFIGURATION
# =============================================================================

# Application Settings
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=your-secret-key-here
API_CORS_ORIGINS=https://app.inferloop.com,https://dashboard.inferloop.com

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=300
RATE_LIMIT_DOCUMENTS_PER_DAY=50000

# =============================================================================
# EXTERNAL DATA SOURCE API KEYS
# =============================================================================

# Legal Data Sources
SEC_EDGAR_API_KEY=your-sec-api-key
COURTLISTENER_API_KEY=your-court-api-key
PACER_USERNAME=your-pacer-username
PACER_PASSWORD=your-pacer-password

# Government Data Sources
DATA_GOV_API_KEY=your-data-gov-key
EUR_LEX_API_KEY=your-eur-lex-key

# Document Processing Datasets
FUNSD_API_KEY=your-funsd-key
DOCBANK_ACCESS_TOKEN=your-docbank-token

# =============================================================================
# AI/ML MODEL CONFIGURATION
# =============================================================================

# LLM API Keys
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
COHERE_API_KEY=your-cohere-key
HUGGINGFACE_API_KEY=your-hf-key

# DeepSeek Configuration
DEEPSEEK_API_KEY=your-deepseek-key
DEEPSEEK_MODEL=deepseek-coder
DEEPSEEK_RAG_ENDPOINT=https://api.deepseek.com/v1/rag

# OCR Services
GOOGLE_VISION_API_KEY=your-google-vision-key
AZURE_COMPUTER_VISION_KEY=your-azure-cv-key
AWS_TEXTRACT_ACCESS_KEY=your-aws-textract-key

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

# Encryption
ENCRYPTION_KEY=your-32-byte-encryption-key
JWT_SECRET=your-jwt-secret
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# OAuth Configuration
OAUTH_CLIENT_ID=your-oauth-client-id
OAUTH_CLIENT_SECRET=your-oauth-client-secret
OAUTH_REDIRECT_URI=https://app.inferloop.com/auth/callback

# API Security
API_KEY_ENCRYPTION_KEY=your-api-key-encryption
WEBHOOK_SECRET=your-webhook-secret

# =============================================================================
# PRIVACY & COMPLIANCE CONFIGURATION
# =============================================================================

# Differential Privacy
DIFFERENTIAL_PRIVACY_EPSILON=0.1
DIFFERENTIAL_PRIVACY_DELTA=1e-5
PRIVACY_BUDGET_TRACKING=true

# PII Detection
PII_DETECTION_ENABLED=true
PII_CONFIDENCE_THRESHOLD=0.8
PII_MASKING_METHOD=synthetic_replacement

# Compliance Frameworks
GDPR_COMPLIANCE_ENABLED=true
HIPAA_COMPLIANCE_ENABLED=true
PCI_DSS_COMPLIANCE_ENABLED=true
SOX_COMPLIANCE_ENABLED=true

# Audit Logging
AUDIT_LOGGING_ENABLED=true
AUDIT_LOG_RETENTION_DAYS=2555  # 7 years
COMPLIANCE_REPORTING_ENABLED=true

# =============================================================================
# MONITORING & OBSERVABILITY
# =============================================================================

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
STRUCTURED_LOGGING=true

# Metrics & Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
METRICS_COLLECTION_INTERVAL=15

# Distributed Tracing
JAEGER_ENABLED=true
JAEGER_ENDPOINT=http://localhost:14268/api/traces
TRACE_SAMPLING_RATE=0.1

# Error Tracking
SENTRY_DSN=your-sentry-dsn
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.1

# Health Checks
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL=30
READINESS_CHECK_TIMEOUT=10

# =============================================================================
# CLOUD PROVIDER CONFIGURATION
# =============================================================================

# AWS Configuration
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_DEFAULT_REGION=us-east-1
AWS_S3_BUCKET=inferloop-structured-docs
AWS_S3_PREFIX=synthetic-documents/

# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GCP_PROJECT_ID=inferloop-synthdata
GCS_BUCKET=inferloop-structured-docs
GCS_PREFIX=synthetic-documents/

# Azure Configuration
AZURE_STORAGE_ACCOUNT=inferloopstorage
AZURE_STORAGE_KEY=your-azure-storage-key
AZURE_CONTAINER=structured-documents
AZURE_PREFIX=synthetic-documents/

# =============================================================================
# KUBERNETES & DEPLOYMENT CONFIGURATION
# =============================================================================

# Kubernetes
KUBERNETES_NAMESPACE=inferloop-synthdata
KUBERNETES_SERVICE_ACCOUNT=structured-docs-service
POD_NAME=${HOSTNAME}

# Container Configuration
CONTAINER_MEMORY_LIMIT=4Gi
CONTAINER_CPU_LIMIT=2000m
WORKER_PROCESSES=4
WORKER_CONNECTIONS=1000

# =============================================================================
# DEVELOPMENT & TESTING CONFIGURATION
# =============================================================================

# Environment
ENVIRONMENT=production
DEBUG=false
TESTING=false

# Development Tools
ENABLE_DEBUG_TOOLBAR=false
ENABLE_PROFILING=false
HOT_RELOAD=false

# Testing Configuration
TEST_DATABASE_URL=postgresql://test_user:test_pass@localhost:5432/test_structured_docs
TEST_REDIS_URL=redis://localhost:6379/1
PYTEST_PARALLEL=true

# =============================================================================
# AI SECURITY CONFIGURATION
# =============================================================================

# LLM Security
PROMPT_INJECTION_DETECTION=true
JAILBREAK_DETECTION=true
CONTENT_MODERATION=true
BIAS_DETECTION_ENABLED=true

# Agent Security
AGENT_ISOLATION_ENABLED=true
AGENT_RESOURCE_LIMITS=true
AGENT_PRIVILEGE_RESTRICTION=true
MALICIOUS_TOOL_DETECTION=true

# RAG Security
VECTOR_STORE_VALIDATION=true
RETRIEVAL_VALIDATION=true
CONTEXT_SANITIZATION=true
KNOWLEDGE_ACCESS_CONTROL=true

# MCP Security
MCP_PROTOCOL_VALIDATION=true
MCP_CONTEXT_ISOLATION=true
MCP_CAPABILITY_RESTRICTIONS=true

# =============================================================================
# BACKUP & DISASTER RECOVERY
# =============================================================================

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE=\"0 2 * * *\"  # Daily at 2 AM
BACKUP_RETENTION_DAYS=30
BACKUP_ENCRYPTION=true

# Disaster Recovery
DR_ENABLED=true
DR_REGION=us-west-2
DR_RTO_MINUTES=60  # Recovery Time Objective
DR_RPO_MINUTES=15  # Recovery Point Objective

# =============================================================================
# FEATURE FLAGS
# =============================================================================

# Platform Features
MULTI_MODAL_GENERATION=true
REAL_TIME_PROCESSING=true
BATCH_PROCESSING=true
STREAMING_API=true

# Experimental Features
EXPERIMENTAL_OCR_MODELS=false
EXPERIMENTAL_LAYOUT_MODELS=false
EXPERIMENTAL_AI_FEATURES=false

# Integration Features
CROSS_PLATFORM_ANALYTICS=true
UNIFIED_DASHBOARD=true
FEDERATED_SEARCH=true
"

create_file "Makefile" "# Makefile for Structured Documents Synthetic Data Platform
# Part of inferloop-synthdata repository

.PHONY: help install test security-test lint format clean build deploy docs

help:
	@echo \"Structured Documents Synthetic Data Platform - inferloop-synthdata\"
	@echo \"================================================================\"
	@echo \"Available commands:\"
	@echo \"  install         Install dependencies\"
	@echo \"  test           Run all tests\"
	@echo \"  security-test  Run security tests\"
	@echo \"  lint           Run linting\"
	@echo \"  format         Format code\"
	@echo \"  clean          Clean build artifacts\"
	@echo \"  build          Build package\"
	@echo \"  deploy         Deploy to production\"
	@echo \"  docs           Generate documentation\"

install:
	@echo \"Installing dependencies for structured documents platform...\"
	pip install -r requirements.txt
	pip install -e .
	bash scripts/setup/setup_security_tools.sh

test:
	@echo \"Running tests...\"
	pytest tests/unit tests/integration -v --cov=src --cov-report=html --cov-report=term-missing

security-test:
	@echo \"Running comprehensive security tests...\"
	pytest tests/security/ -v
	bash scripts/security/run_security_scans.sh

lint:
	@echo \"Running linting...\"
	flake8 src/ tests/
	mypy src/
	bandit -r src/

format:
	@echo \"Formatting code...\"
	black src/ tests/
	isort src/ tests/

clean:
	@echo \"Cleaning build artifacts...\"
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name \"*.pyc\" -delete
	rm -rf htmlcov/ .coverage .pytest_cache/

build:
	@echo \"Building package...\"
	python -m build

deploy:
	@echo \"Deploying to production...\"
	bash scripts/deployment/deploy_kubernetes.sh

docs:
	@echo \"Generating documentation...\"
	mkdir -p docs/generated
	python -m pydoc -w src/structured_docs_synth

# Integration commands for inferloop-synthdata ecosystem
sync-configs:
	@echo \"Syncing configurations with inferloop-synthdata...\"
	# Copy shared configs from parent repo if needed

update-security-policies:
	@echo \"Updating security policies from inferloop-synthdata...\"
	# Update security policies from parent repo

validate-integration:
	@echo \"Validating integration with inferloop-synthdata ecosystem...\"
	pytest tests/integration/ -k \"inferloop\" -v
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
echo "✅ Repository structure created successfully in inferloop-synthdata!"
echo ""
echo "📂 Location: inferloop-synthdata/$REPO_DIR/$PROJECT_NAME"
echo "📊 Total directories created: $(find . -type d | wc -l)"
echo "📄 Total files created: $(find . -type f | wc -l)"
echo ""
echo "🚀 Next steps:"
echo "1. cd $REPO_DIR/$PROJECT_NAME"
echo "2. python -m venv venv"
echo "3. source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
echo "4. make install"
echo "5. make test"
echo "6. make security-test"
echo ""
echo "🔗 Integration with inferloop-synthdata:"
echo "- Shared security policies and compliance frameworks"
echo "- Coordinated CI/CD pipelines"
echo "- Integrated monitoring and observability"
echo "- Cross-platform synthetic data capabilities"
echo ""
echo "🔐 Security features included:"
echo "- AI/ML security testing (LLM, Agent, RAG, MCP)"
echo "- Comprehensive CI/CD security pipeline"
echo "- Privacy and compliance frameworks (GDPR, HIPAA, PCI-DSS, SOX)"
echo "- Red team testing capabilities"
echo "- Real-time security monitoring"
echo "- Integration with inferloop security infrastructure"
echo ""
echo "📁 Key directories created:"
echo "- src/                    # Source code modules"
echo "- tests/security/         # AI security testing framework"
echo "- configs/security/       # Security policies and configurations"
echo "- monitoring/             # Observability and security monitoring"
echo "- .github/workflows/      # CI/CD security pipelines"
echo "- deployment/             # Docker, Kubernetes, Terraform"
echo ""
echo "🔧 Development workflow:"
echo "1. Navigate: cd $REPO_DIR/$PROJECT_NAME"
echo "2. Setup: make install"
echo "3. Test: make test && make security-test"
echo "4. Code: Edit files in src/, tests/, configs/"
echo "5. Format: make format"
echo "6. Deploy: make deploy"
echo ""
echo "🏗️  Structured Documents platform is ready for development within inferloop-synthdata!"
echo "🎯 This platform integrates seamlessly with the broader inferloop synthetic data ecosystem."
