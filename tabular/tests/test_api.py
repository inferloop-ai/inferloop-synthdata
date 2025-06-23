# tests/test_api.py
"""
Test suite for REST API endpoints
"""

import pytest
import json
import io
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from fastapi.testclient import TestClient
from datetime import datetime

from api.app import app
from sdk import SyntheticDataConfig, GenerationResult


client = TestClient(app)


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing"""
    return """age,income,gender,city
25,50000,M,New York
30,60000,F,San Francisco
35,70000,M,Chicago
40,80000,F,Boston
45,90000,M,Seattle"""


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "generator_type": "sdv",
        "model_type": "gaussian_copula",
        "num_samples": 100,
        "categorical_columns": ["gender", "city"],
        "continuous_columns": ["age", "income"]
    }


@pytest.fixture
def mock_generator():
    """Mock generator for testing"""
    generator = Mock()
    generator.fit = Mock()
    generator.generate = Mock(return_value=GenerationResult(
        synthetic_data=pd.DataFrame({
            'age': [26, 31, 36],
            'income': [55000, 65000, 75000],
            'gender': ['F', 'M', 'F'],
            'city': ['New York', 'Chicago', 'Boston']
        }),
        config=SyntheticDataConfig(generator_type="sdv", model_type="gaussian_copula"),
        generation_time=1.5,
        model_info={'status': 'fitted'}
    ))
    generator.get_model_info = Mock(return_value={'status': 'fitted', 'model_type': 'gaussian_copula'})
    return generator


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data


class TestGeneratorEndpoints:
    """Test generator-related endpoints"""
    
    def test_list_generators(self):
        response = client.get("/generators")
        assert response.status_code == 200
        data = response.json()
        assert "generators" in data
        assert "sdv" in data["generators"]
        assert "ctgan" in data["generators"]
        assert "ydata" in data["generators"]
    
    def test_generator_info(self):
        response = client.get("/generators/sdv")
        assert response.status_code == 200
        data = response.json()
        assert data["generator"] == "sdv"
        assert "models" in data
        assert "description" in data
    
    def test_generator_info_not_found(self):
        response = client.get("/generators/invalid")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestGenerateEndpoint:
    """Test synthetic data generation endpoint"""
    
    @patch('sdk.factory.GeneratorFactory.create_generator')
    def test_generate_success(self, mock_factory, mock_generator, sample_csv_data, sample_config):
        mock_factory.return_value = mock_generator
        
        files = {'file': ('data.csv', sample_csv_data, 'text/csv')}
        response = client.post(
            "/generate",
            files=files,
            data={'config': json.dumps(sample_config)}
        )
        
        assert response.status_code == 200
        assert response.headers['content-type'] == 'text/csv; charset=utf-8'
        assert 'age,income,gender,city' in response.text
        
        mock_factory.assert_called_once()
        mock_generator.fit_generate.assert_called_once()
    
    def test_generate_missing_file(self, sample_config):
        response = client.post(
            "/generate",
            data={'config': json.dumps(sample_config)}
        )
        assert response.status_code == 422
    
    def test_generate_invalid_config(self, sample_csv_data):
        files = {'file': ('data.csv', sample_csv_data, 'text/csv')}
        response = client.post(
            "/generate",
            files=files,
            data={'config': json.dumps({"invalid": "config"})}
        )
        assert response.status_code == 422
    
    @patch('sdk.factory.GeneratorFactory.create_generator')
    def test_generate_with_format_json(self, mock_factory, mock_generator, sample_csv_data, sample_config):
        mock_factory.return_value = mock_generator
        
        files = {'file': ('data.csv', sample_csv_data, 'text/csv')}
        response = client.post(
            "/generate?output_format=json",
            files=files,
            data={'config': json.dumps(sample_config)}
        )
        
        assert response.status_code == 200
        assert response.headers['content-type'] == 'application/json'
        data = response.json()
        assert "data" in data
        assert "metadata" in data
        assert data["metadata"]["num_samples"] == 3


class TestAsyncGenerateEndpoint:
    """Test asynchronous generation endpoint"""
    
    @patch('sdk.factory.GeneratorFactory.create_generator')
    def test_async_generate_start(self, mock_factory, mock_generator, sample_csv_data, sample_config):
        mock_factory.return_value = mock_generator
        
        files = {'file': ('data.csv', sample_csv_data, 'text/csv')}
        response = client.post(
            "/generate/async",
            files=files,
            data={'config': json.dumps(sample_config)}
        )
        
        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "accepted"
    
    def test_get_task_status(self):
        # First create a task
        task_id = "test-task-123"
        
        # Mock the task status
        with patch('api.app.generation_tasks', {task_id: {"status": "completed", "result": "data.csv"}}):
            response = client.get(f"/tasks/{task_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["task_id"] == task_id
            assert data["status"] == "completed"
    
    def test_get_task_not_found(self):
        response = client.get("/tasks/non-existent-task")
        assert response.status_code == 404


class TestValidateEndpoint:
    """Test validation endpoint"""
    
    @patch('sdk.validator.SyntheticDataValidator')
    def test_validate_success(self, mock_validator_class, sample_csv_data):
        mock_validator = Mock()
        mock_validator.validate_all = Mock(return_value={
            "statistical_similarity": {"mean_difference": 0.05, "std_difference": 0.03},
            "privacy_metrics": {"unique_ratio": 0.95},
            "data_utility": {"correlation_difference": 0.02}
        })
        mock_validator_class.return_value = mock_validator
        
        files = {
            'original': ('original.csv', sample_csv_data, 'text/csv'),
            'synthetic': ('synthetic.csv', sample_csv_data, 'text/csv')
        }
        
        response = client.post("/validate", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "statistical_similarity" in data["metrics"]
        assert "summary" in data
    
    def test_validate_missing_files(self):
        response = client.post("/validate")
        assert response.status_code == 422


class TestConfigEndpoints:
    """Test configuration endpoints"""
    
    def test_create_config(self):
        config_data = {
            "generator_type": "sdv",
            "model_type": "gaussian_copula",
            "num_samples": 1000
        }
        
        response = client.post("/config/create", json=config_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["generator_type"] == "sdv"
        assert data["model_type"] == "gaussian_copula"
        assert data["num_samples"] == 1000
        assert "created_at" in data
    
    def test_validate_config(self):
        config_data = {
            "generator_type": "sdv",
            "model_type": "gaussian_copula",
            "num_samples": 1000
        }
        
        response = client.post("/config/validate", json=config_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] == True
        assert "config" in data
    
    def test_validate_config_invalid(self):
        config_data = {
            "generator_type": "invalid_generator",
            "model_type": "gaussian_copula"
        }
        
        response = client.post("/config/validate", json=config_data)
        
        assert response.status_code == 400
        data = response.json()
        assert data["valid"] == False
        assert "errors" in data


class TestTemplateEndpoints:
    """Test template endpoints"""
    
    def test_list_templates(self):
        response = client.get("/templates")
        assert response.status_code == 200
        data = response.json()
        assert "templates" in data
        assert isinstance(data["templates"], list)
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.read_text')
    def test_get_template(self, mock_read, mock_exists):
        mock_exists.return_value = True
        mock_read.return_value = """
generator_type: sdv
model_type: gaussian_copula
num_samples: 1000
"""
        
        response = client.get("/templates/sdv_config")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "sdv_config"
        assert "content" in data
    
    def test_get_template_not_found(self):
        response = client.get("/templates/non_existent")
        assert response.status_code == 404


class TestErrorHandling:
    """Test error handling in API"""
    
    @patch('sdk.factory.GeneratorFactory.create_generator')
    def test_generator_error_handling(self, mock_factory, sample_csv_data, sample_config):
        mock_factory.side_effect = ValueError("Generator error")
        
        files = {'file': ('data.csv', sample_csv_data, 'text/csv')}
        response = client.post(
            "/generate",
            files=files,
            data={'config': json.dumps(sample_config)}
        )
        
        assert response.status_code == 400
        assert "Generator error" in response.json()["detail"]
    
    def test_invalid_csv_data(self, sample_config):
        files = {'file': ('data.csv', 'invalid csv data', 'text/csv')}
        response = client.post(
            "/generate",
            files=files,
            data={'config': json.dumps(sample_config)}
        )
        
        # Should handle gracefully
        assert response.status_code in [400, 422]


class TestCORSAndMiddleware:
    """Test CORS and middleware functionality"""
    
    def test_cors_headers(self):
        response = client.options("/health")
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
    
    def test_request_id_header(self):
        response = client.get("/health")
        assert "x-request-id" in response.headers


if __name__ == "__main__":
    pytest.main([__file__])