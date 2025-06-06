"""Unit tests for the API endpoints"""

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.api.routes import router
from src.api.models import GenerateCodeRequest, GenerateCodeResponse

# Create a test client using the FastAPI router
from fastapi import FastAPI
app = FastAPI()
app.include_router(router)
client = TestClient(app)


class TestAPIEndpoints(unittest.TestCase):
    """Test cases for API endpoints"""
    
    def test_health_check(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
    
    @patch("src.api.routes.CodeLlamaGenerator")
    def test_generate_code(self, mock_generator_class):
        """Test the generate code endpoint"""
        # Mock the generator
        mock_generator = MagicMock()
        mock_generator.generate_batch.return_value = [
            {
                "id": "sample_0",
                "prompt": "fibonacci function",
                "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "language": "python",
                "metadata": {
                    "lines_of_code": 4,
                    "estimated_complexity": "low",
                    "dependencies": []
                }
            }
        ]
        mock_generator_class.return_value = mock_generator
        
        request_data = {
            "prompts": ["fibonacci function"],
            "language": "python",
            "count": 1,
        }
        
        response = client.post("/generate", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("samples", data)
        self.assertEqual(len(data["samples"]), 1)
        self.assertEqual(data["samples"][0]["prompt"], "fibonacci function")
        
    @patch("src.api.routes.SyntaxValidator")
    def test_validate_code(self, mock_validator_class):
        """Test the validate code endpoint"""
        # Mock the validator
        mock_validator = MagicMock()
        mock_validator.validate.return_value = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        mock_validator_class.return_value = mock_validator
        
        request_data = {
            "code": "def test(): return 42",
            "language": "python"
        }
        
        response = client.post("/validate", json=request_data)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("valid", data)
        self.assertTrue(data["valid"])
        self.assertEqual(len(data["errors"]), 0)


class TestAPIModels(unittest.TestCase):
    """Test cases for API models"""
    
    def test_generate_code_request_model(self):
        """Test the GenerateCodeRequest model"""
        request = GenerateCodeRequest(
            prompts=["binary search", "quick sort"],
            language="python",
            framework="fastapi",
            count=2
        )
        
        self.assertEqual(request.language, "python")
        self.assertEqual(request.framework, "fastapi")
        self.assertEqual(request.count, 2)
        self.assertEqual(len(request.prompts), 2)
    
    def test_generate_code_response_model(self):
        """Test the GenerateCodeResponse model"""
        samples = [
            {
                "id": "sample_0",
                "prompt": "binary search",
                "code": "def binary_search(arr, target): pass",
                "language": "python",
                "metadata": {
                    "lines_of_code": 1,
                    "estimated_complexity": "low",
                    "dependencies": []
                }
            }
        ]
        
        response = GenerateCodeResponse(
            samples=samples,
            metadata={
                "total_samples": 1,
                "generation_time": 0.5,
                "model": "codellama"
            }
        )
        
        self.assertEqual(len(response.samples), 1)
        self.assertEqual(response.metadata["total_samples"], 1)
        self.assertEqual(response.metadata["model"], "codellama")


if __name__ == "__main__":
    unittest.main()
