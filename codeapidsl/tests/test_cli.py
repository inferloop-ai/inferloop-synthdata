"""Unit tests for the CLI commands"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import json
from click.testing import CliRunner

from src.cli.commands import cli
from src.cli.utils import load_config


class TestCLICommands(unittest.TestCase):
    """Test cases for CLI commands"""
    
    def setUp(self):
        self.runner = CliRunner()
    
    def test_version_command(self):
        """Test the version command"""
        with patch("src.cli.commands.__version__", "1.0.0"):
            result = self.runner.invoke(cli, ["version"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("1.0.0", result.output)
    
    @patch("src.cli.commands.CodeLlamaGenerator")
    def test_generate_command(self, mock_generator_class):
        """Test the generate command"""
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
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "output.jsonl")
            
            # Test generate command with single prompt
            result = self.runner.invoke(cli, [
                "generate",
                "--prompt", "fibonacci function",
                "--language", "python",
                "--output", output_file
            ])
            
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(output_file))
            
            with open(output_file) as f:
                data = f.readline()
                sample = json.loads(data)
                self.assertEqual(sample["prompt"], "fibonacci function")
                self.assertEqual(sample["language"], "python")
    
    @patch("src.cli.commands.SyntaxValidator")
    def test_validate_command(self, mock_validator_class):
        """Test the validate command"""
        # Mock the validator
        mock_validator = MagicMock()
        mock_validator.validate.return_value = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        mock_validator_class.return_value = mock_validator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = os.path.join(tmpdir, "test.py")
            with open(code_file, "w") as f:
                f.write("def test(): return 42")
            
            result = self.runner.invoke(cli, [
                "validate",
                "--file", code_file,
                "--language", "python"
            ])
            
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Valid", result.output)
    
    @patch("src.cli.commands.start_api_server")
    def test_serve_command(self, mock_serve):
        """Test the serve command"""
        mock_serve.return_value = None
        
        result = self.runner.invoke(cli, [
            "serve",
            "--port", "8000",
            "--host", "127.0.0.1"
        ])
        
        self.assertEqual(result.exit_code, 0)
        mock_serve.assert_called_once_with(host="127.0.0.1", port=8000)


class TestCLIUtils(unittest.TestCase):
    """Test cases for CLI utilities"""
    
    def test_load_config(self):
        """Test loading configuration from a file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "config.yaml")
            with open(config_file, "w") as f:
                f.write("""model:
  name: codellama
  endpoint: http://localhost:8080

language: python
framework: flask
""")
            
            config = load_config(config_file)
            
            self.assertEqual(config["model"]["name"], "codellama")
            self.assertEqual(config["language"], "python")
            self.assertEqual(config["framework"], "flask")
    
    def test_load_config_default(self):
        """Test loading default configuration when file doesn't exist"""
        with patch("os.path.exists", return_value=False):
            with patch("src.cli.utils.DEFAULT_CONFIG", {"model": {"name": "default"}}):  
                config = load_config("nonexistent.yaml")
                
                self.assertEqual(config["model"]["name"], "default")


if __name__ == "__main__":
    unittest.main()
