"""
Pytest configuration and fixtures
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import shutil
from unittest.mock import Mock
import coverage

# Start coverage when pytest starts
cov = None

def pytest_configure(config):
    """Configure pytest with coverage"""
    global cov
    
    # Start coverage collection
    cov = coverage.Coverage(config_file='.coveragerc')
    cov.start()


def pytest_unconfigure(config):
    """Generate coverage report after tests"""
    global cov
    
    if cov:
        cov.stop()
        cov.save()
        
        # Generate reports
        print("\n\nCoverage Report:")
        print("=" * 80)
        cov.report()
        
        # Generate HTML report
        cov.html_report()
        print(f"\nHTML coverage report generated in: htmlcov/index.html")
        
        # Generate XML report for CI/CD
        cov.xml_report()
        
        # Generate JSON report for further processing
        cov.json_report()


# Fixtures for testing
@pytest.fixture
def sample_data():
    """Generate sample dataset for testing"""
    np.random.seed(42)
    n_rows = 1000
    
    data = pd.DataFrame({
        'numeric_1': np.random.randn(n_rows),
        'numeric_2': np.random.rand(n_rows) * 100,
        'categorical_1': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_rows),
        'datetime': pd.date_range('2020-01-01', periods=n_rows, freq='H'),
        'integer': np.random.randint(0, 100, n_rows)
    })
    
    return data


@pytest.fixture
def small_data():
    """Small dataset for quick tests"""
    return pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['A', 'B', 'A', 'B', 'C'],
        'col3': [0.1, 0.2, 0.3, 0.4, 0.5]
    })


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    # Cleanup
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_file():
    """Create temporary file for tests"""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    yield Path(path)
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def mock_generator():
    """Mock generator for testing"""
    generator = Mock()
    generator.fit = Mock()
    generator.generate = Mock(return_value=Mock(data=pd.DataFrame({'col1': [1, 2, 3]})))
    generator.is_fitted = False
    generator.config = Mock()
    return generator


@pytest.fixture
def api_client():
    """Create test client for API"""
    from fastapi.testclient import TestClient
    from api.app import app
    
    return TestClient(app)


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )
    config.addinivalue_line(
        "markers", "cli: marks tests as CLI tests"
    )


# Coverage fixtures
@pytest.fixture(scope='session')
def coverage_report():
    """Generate coverage report at end of session"""
    yield
    
    # This will be called after all tests
    import subprocess
    
    # Generate coverage badge
    try:
        subprocess.run(['coverage-badge', '-o', 'coverage.svg'], check=True)
        print("\nCoverage badge generated: coverage.svg")
    except:
        pass