#!/bin/bash

# Test Runner Script

set -e

echo "🧪 Running Enterprise Video Synthesis Pipeline tests..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Activated virtual environment"
fi

# Install test dependencies if needed
pip install pytest pytest-asyncio pytest-cov pytest-mock

# Run unit tests
echo "🔬 Running unit tests..."
pytest qa/test-suites/unit-tests/ -v --cov=services --cov-report=html --cov-report=term

# Run integration tests (if services are running)
if curl -f -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "🔗 Running integration tests..."
    pytest qa/test-suites/integration-tests/ -v
else
    echo "⚠️  Skipping integration tests (services not running)"
    echo "   Start services with 'make deploy' to run integration tests"
fi

# Run code quality checks
echo "📏 Running code quality checks..."
if command -v black &> /dev/null; then
    black --check services/
fi

if command -v flake8 &> /dev/null; then
    flake8 services/
fi

if command -v mypy &> /dev/null; then
    mypy services/
fi

echo "✅ All tests completed!"
