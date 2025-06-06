"""Unit tests for code generators"""

import unittest
from unittest.mock import patch, MagicMock

from src.generators import BaseCodeGenerator, GenerationConfig
from src.generators import CodeLlamaGenerator, StarCoderGenerator


class TestBaseGenerator(unittest.TestCase):
    """Test cases for the BaseCodeGenerator class"""
    
    def setUp(self):
        self.config = GenerationConfig(
            language="python",
            framework=None,
            complexity="medium",
            count=10,
        )
    
    def test_extract_metadata(self):
        """Test metadata extraction from code"""
        # Create a mock subclass of the abstract base class for testing
        class MockGenerator(BaseCodeGenerator):
            def generate_function(self, prompt, **kwargs):
                return "def test(): pass"
                
            def generate_class(self, prompt, **kwargs):
                return "class Test: pass"
                
            def generate_module(self, prompt, **kwargs):
                return "# Module\ndef test(): pass"
                
            def generate_api_endpoint(self, schema):
                return "def endpoint(): return {'status': 'ok'}"
        
        generator = MockGenerator(self.config)
        code = "def test():\n    pass\n"
        metadata = generator._extract_metadata(code)
        
        self.assertIn("lines_of_code", metadata)
        self.assertIn("estimated_complexity", metadata)
        self.assertIn("dependencies", metadata)
        self.assertEqual(metadata["lines_of_code"], 3)
        self.assertEqual(metadata["estimated_complexity"], "low")
    
    def test_extract_dependencies(self):
        """Test extraction of dependencies from code"""
        # Create a mock subclass for testing
        class MockGenerator(BaseCodeGenerator):
            def generate_function(self, prompt, **kwargs):
                return "def test(): pass"
                
            def generate_class(self, prompt, **kwargs):
                return "class Test: pass"
                
            def generate_module(self, prompt, **kwargs):
                return "# Module\ndef test(): pass"
                
            def generate_api_endpoint(self, schema):
                return "def endpoint(): return {'status': 'ok'}"
        
        generator = MockGenerator(self.config)
        code = "import os\nfrom typing import List\n\ndef test(items: List[int]):\n    pass\n"
        dependencies = generator._extract_dependencies(code)
        
        self.assertEqual(len(dependencies), 2)
        self.assertIn("import os", dependencies)
        self.assertIn("from typing import List", dependencies)


class TestCodeLlamaGenerator(unittest.TestCase):
    """Test cases for the CodeLlamaGenerator class"""
    
    def setUp(self):
        self.config = GenerationConfig(
            language="python",
            framework=None,
            complexity="medium",
        )
    
    @patch("src.generators.code_llama_generator.requests")
    def test_generate_function(self, mock_requests):
        """Test function generation with CodeLlama"""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "code": "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        }
        mock_requests.post.return_value = mock_response
        
        generator = CodeLlamaGenerator(self.config, model_endpoint="http://localhost:8080")
        code = generator.generate_function("fibonacci function")
        
        self.assertIn("def fibonacci", code)
        mock_requests.post.assert_called_once()


class TestStarCoderGenerator(unittest.TestCase):
    """Test cases for the StarCoderGenerator class"""
    
    def setUp(self):
        self.config = GenerationConfig(
            language="python",
            framework=None,
            complexity="medium",
        )
    
    @patch("src.generators.starcoder_generator.requests")
    def test_generate_function(self, mock_requests):
        """Test function generation with StarCoder"""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "code": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
        }
        mock_requests.post.return_value = mock_response
        
        generator = StarCoderGenerator(self.config, model_endpoint="http://localhost:8080")
        code = generator.generate_function("binary search function")
        
        self.assertIn("def binary_search", code)
        mock_requests.post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
