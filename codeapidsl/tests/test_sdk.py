"""Unit tests for the SDK client"""

import unittest
from unittest.mock import patch, MagicMock
import requests

from src.sdk.client import CodeAPIClient
from src.sdk.exceptions import APIError, ValidationError, AuthenticationError


class TestCodeAPIClient(unittest.TestCase):
    """Test cases for the CodeAPIClient class"""
    
    def setUp(self):
        self.client = CodeAPIClient(
            api_key="test_api_key",
            base_url="http://localhost:8000"
        )
    
    @patch("src.sdk.client.requests.get")
    def test_health_check(self, mock_get):
        """Test the health check method"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_get.return_value = mock_response
        
        result = self.client.health_check()
        
        self.assertEqual(result["status"], "ok")
        mock_get.assert_called_once_with(
            "http://localhost:8000/health",
            headers={"Authorization": "Bearer test_api_key"}
        )
    
    @patch("src.sdk.client.requests.post")
    def test_generate_code(self, mock_post):
        """Test the generate code method"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "samples": [
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
            ],
            "metadata": {
                "total_samples": 1,
                "generation_time": 0.5,
                "model": "codellama"
            }
        }
        mock_post.return_value = mock_response
        
        result = self.client.generate_code(
            prompts=["fibonacci function"],
            language="python",
            count=1
        )
        
        self.assertEqual(len(result["samples"]), 1)
        self.assertEqual(result["samples"][0]["prompt"], "fibonacci function")
        mock_post.assert_called_once_with(
            "http://localhost:8000/generate",
            headers={"Authorization": "Bearer test_api_key"},
            json={
                "prompts": ["fibonacci function"],
                "language": "python",
                "count": 1,
                "framework": None
            }
        )
    
    @patch("src.sdk.client.requests.post")
    def test_validate_code(self, mock_post):
        """Test the validate code method"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        mock_post.return_value = mock_response
        
        result = self.client.validate_code(
            code="def test(): return 42",
            language="python"
        )
        
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)
        mock_post.assert_called_once_with(
            "http://localhost:8000/validate",
            headers={"Authorization": "Bearer test_api_key"},
            json={
                "code": "def test(): return 42",
                "language": "python"
            }
        )
    
    @patch("src.sdk.client.requests.post")
    def test_api_error_handling(self, mock_post):
        """Test error handling in API calls"""
        # Test API error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal server error"}
        mock_post.return_value = mock_response
        
        with self.assertRaises(APIError):
            self.client.generate_code(
                prompts=["fibonacci function"],
                language="python"
            )
        
        # Test validation error
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Invalid request parameters"}
        mock_post.return_value = mock_response
        
        with self.assertRaises(ValidationError):
            self.client.generate_code(
                prompts=["fibonacci function"],
                language="python"
            )
        
        # Test authentication error
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Invalid API key"}
        mock_post.return_value = mock_response
        
        with self.assertRaises(AuthenticationError):
            self.client.generate_code(
                prompts=["fibonacci function"],
                language="python"
            )


class TestExceptions(unittest.TestCase):
    """Test cases for SDK exceptions"""
    
    def test_api_error(self):
        """Test APIError exception"""
        error = APIError("API error message", status_code=500)
        self.assertEqual(str(error), "API error message (status code: 500)")
    
    def test_validation_error(self):
        """Test ValidationError exception"""
        error = ValidationError("Validation error message")
        self.assertEqual(str(error), "Validation error message")
    
    def test_authentication_error(self):
        """Test AuthenticationError exception"""
        error = AuthenticationError("Authentication error message")
        self.assertEqual(str(error), "Authentication error message")


if __name__ == "__main__":
    unittest.main()
