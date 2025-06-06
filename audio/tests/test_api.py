# tests/test_api.py
"""
Test suite for API endpoints
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from audio_synth.api.server import app
from audio_synth.sdk.client import AudioSynthSDK

class TestAPIEndpoints:
    """Test suite for API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def mock_sdk(self):
        """Mock SDK for testing"""
        with patch('audio_synth.api.server.sdk') as mock:
            mock_sdk = Mock(spec=AudioSynthSDK)
            mock_sdk.generate.return_value = [torch.randn(22050)]
            mock_sdk.validate.return_value = {
                "quality": [{"snr": 15.0, "realism_score": 0.8}],
                "privacy": [{"speaker_anonymity": 0.9}]
            }
            mock = mock_sdk
            yield mock
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "Audio Synthetic Data API"
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_generate_endpoint(self, client, mock_sdk):
        """Test audio generation endpoint"""
        request_data = {
            "method": "diffusion",
            "prompt": "Test audio generation",
            "num_samples": 2,
            "privacy_level": "medium"
        }
        
        response = client.post("/api/v1/generate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["num_samples"] == 2
    
    def test_generate_batch_endpoint(self, client, mock_sdk):
        """Test batch generation endpoint"""
        request_data = {
            "requests": [
                {
                    "method": "diffusion",
                    "prompt": "First sample",
                    "num_samples": 1
                },
                {
                    "method": "tts",
                    "prompt": "Second sample", 
                    "num_samples": 2
                }
            ]
        }
        
        response = client.post("/api/v1/generate/batch", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["num_samples"] == 3  # 1 + 2
    
    def test_validate_endpoint(self, client, mock_sdk):
        """Test validation endpoint"""
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            import torchaudio
            audio = torch.randn(1, 22050)
            torchaudio.save(tmp.name, audio, 22050)
            
            with open(tmp.name, "rb") as audio_file:
                files = {"files": ("test.wav", audio_file, "audio/wav")}
                data = {
                    "validators": ["quality", "privacy"]
                }
                
                response = client.post(
                    "/api/v1/validate",
                    files=files,
                    data={"request": json.dumps(data)}
                )
        
        assert response.status_code == 200
        result = response.json()
        assert "job_id" in result
    
    def test_job_status_endpoint(self, client):
        """Test job status endpoint"""
        # First create a job
        request_data = {
            "method": "diffusion",
            "prompt": "Test job",
            "num_samples": 1
        }
        
        response = client.post("/api/v1/generate", json=request_data)
        job_id = response.json()["job_id"]
        
        # Then check its status
        status_response = client.get(f"/api/v1/jobs/{job_id}")
        
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["job_id"] == job_id
        assert "status" in status_data
    
    def test_list_jobs_endpoint(self, client):
        """Test list jobs endpoint"""
        response = client.get("/api/v1/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_models_endpoint(self, client):
        """Test models listing endpoint"""
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "generation_methods" in data
        assert "validators" in data
    
    def test_config_endpoint(self, client, mock_sdk):
        """Test configuration endpoint"""
        response = client.get("/api/v1/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "supported_formats" in data
        assert "max_duration" in data

class TestAPIValidation:
    """Test API input validation"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_invalid_generation_method(self, client):
        """Test validation of generation method"""
        request_data = {
            "method": "invalid_method",
            "prompt": "Test"
        }
        
        response = client.post("/api/v1/generate", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_num_samples(self, client):
        """Test validation of number of samples"""
        request_data = {
            "method": "diffusion",
            "num_samples": 0  # Invalid
        }
        
        response = client.post("/api/v1/generate", json=request_data)
        assert response.status_code == 422
    
    def test_invalid_privacy_level(self, client):
        """Test validation of privacy level"""
        request_data = {
            "method": "diffusion",
            "privacy_level": "invalid"
        }
        
        response = client.post("/api/v1/generate", json=request_data)
        assert response.status_code == 422
    
    def test_invalid_audio_config(self, client):
        """Test validation of audio configuration"""
        request_data = {
            "method": "diffusion",
            "audio_config": {
                "sample_rate": 1000  # Too low
            }
        }
        
        response = client.post("/api/v1/generate", json=request_data)
        assert response.status_code == 422

# ============================================================================
