"""Unit tests for the delivery module"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import tempfile
import os

from src.delivery import JSONFormatter, JSONLFormatter, ProtobufFormatter
from src.delivery import FileExporter, S3Exporter, APIExporter


class TestFormatters(unittest.TestCase):
    """Test cases for the code formatters"""
    
    def setUp(self):
        self.sample = {
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
        self.samples = [self.sample]
    
    def test_json_formatter(self):
        """Test the JSON formatter"""
        formatter = JSONFormatter()
        result = formatter.format(self.samples)
        
        # Verify the result is valid JSON
        parsed = json.loads(result)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["id"], "sample_0")
        self.assertEqual(parsed[0]["prompt"], "fibonacci function")
    
    def test_jsonl_formatter(self):
        """Test the JSONL formatter"""
        formatter = JSONLFormatter()
        result = formatter.format(self.samples)
        
        # Split by newline and parse each line
        lines = result.strip().split('\n')
        self.assertEqual(len(lines), 1)  # One sample, one line
        
        parsed = json.loads(lines[0])
        self.assertEqual(parsed["id"], "sample_0")
        self.assertEqual(parsed["prompt"], "fibonacci function")
    
    @patch("src.delivery.formatters.ProtobufMessage")
    def test_protobuf_formatter(self, mock_proto):
        """Test the Protobuf formatter"""
        # Mock the protobuf message serialization
        mock_instance = MagicMock()
        mock_instance.SerializeToString.return_value = b"serialized_data"
        mock_proto.return_value = mock_instance
        
        formatter = ProtobufFormatter()
        result = formatter.format(self.samples)
        
        # Check that the protobuf message was created and serialized
        self.assertEqual(result, b"serialized_data")
        mock_proto.assert_called_once()
        mock_instance.SerializeToString.assert_called_once()


class TestExporters(unittest.TestCase):
    """Test cases for the code exporters"""
    
    def setUp(self):
        self.formatted_data = json.dumps([{
            "id": "sample_0",
            "prompt": "fibonacci function",
            "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "language": "python"
        }])
    
    @patch("builtins.open", new_callable=mock_open)
    def test_file_exporter(self, mock_file):
        """Test the file exporter"""
        exporter = FileExporter()
        result = exporter.export(
            data=self.formatted_data,
            destination="output.json",
            format_type="json"
        )
        
        # Check that the file was opened and written to
        mock_file.assert_called_once_with("output.json", "w")
        mock_file().write.assert_called_once_with(self.formatted_data)
        self.assertTrue(result)
    
    @patch("src.delivery.exporters.boto3")
    def test_s3_exporter(self, mock_boto3):
        """Test the S3 exporter"""
        # Mock the S3 client
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        
        exporter = S3Exporter()
        result = exporter.export(
            data=self.formatted_data,
            destination="s3://bucket/path/output.json",
            format_type="json"
        )
        
        # Check that the S3 client was used correctly
        mock_boto3.client.assert_called_once_with("s3")
        mock_s3.put_object.assert_called_once()
        self.assertTrue(result)
    
    @patch("src.delivery.exporters.requests")
    def test_api_exporter(self, mock_requests):
        """Test the API exporter"""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_requests.post.return_value = mock_response
        
        exporter = APIExporter()
        result = exporter.export(
            data=self.formatted_data,
            destination="https://api.example.com/endpoint",
            format_type="json"
        )
        
        # Check that the API was called correctly
        mock_requests.post.assert_called_once()
        self.assertTrue(result)
    
    @patch("src.delivery.exporters.requests")
    def test_api_exporter_failure(self, mock_requests):
        """Test the API exporter with a failure response"""
        # Mock the API response with an error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_requests.post.return_value = mock_response
        
        exporter = APIExporter()
        result = exporter.export(
            data=self.formatted_data,
            destination="https://api.example.com/endpoint",
            format_type="json"
        )
        
        # Check that the result is False due to the failure
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
