# tests/test_cli.py
"""
Test suite for CLI commands
"""

import pytest
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import pandas as pd
from typer.testing import CliRunner
from io import StringIO

from cli.main import app
from sdk import SyntheticDataConfig, GenerationResult, GeneratorFactory


runner = CliRunner()


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a sample CSV file for testing"""
    csv_content = """age,income,gender,city
25,50000,M,New York
30,60000,F,San Francisco
35,70000,M,Chicago
40,80000,F,Boston
45,90000,M,Seattle"""
    
    csv_file = tmp_path / "sample_data.csv"
    csv_file.write_text(csv_content)
    return str(csv_file)


@pytest.fixture
def sample_config_file(tmp_path):
    """Create a sample config file for testing"""
    config = {
        "generator_type": "sdv",
        "model_type": "gaussian_copula",
        "num_samples": 100,
        "categorical_columns": ["gender", "city"],
        "continuous_columns": ["age", "income"]
    }
    
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    return str(config_file)


@pytest.fixture
def mock_generator():
    """Mock generator for testing"""
    generator = Mock()
    generator.fit_generate = Mock(return_value=GenerationResult(
        synthetic_data=pd.DataFrame({
            'age': [26, 31, 36, 41, 46],
            'income': [55000, 65000, 75000, 85000, 95000],
            'gender': ['F', 'M', 'F', 'M', 'F'],
            'city': ['New York', 'Chicago', 'Boston', 'Seattle', 'San Francisco']
        }),
        config=SyntheticDataConfig(generator_type="sdv", model_type="gaussian_copula"),
        generation_time=1.5,
        model_info={'status': 'fitted', 'model_type': 'gaussian_copula'}
    ))
    return generator


class TestGenerateCommand:
    """Test the generate command"""
    
    @patch('sdk.factory.GeneratorFactory.create_generator')
    def test_generate_basic(self, mock_factory, mock_generator, sample_csv_file, tmp_path):
        mock_factory.return_value = mock_generator
        output_file = tmp_path / "output.csv"
        
        result = runner.invoke(app, [
            "generate",
            sample_csv_file,
            str(output_file),
            "--generator-type", "sdv",
            "--model-type", "gaussian_copula",
            "--num-samples", "100"
        ])
        
        assert result.exit_code == 0
        assert "Generated 5 synthetic samples" in result.stdout
        assert output_file.exists()
        
        # Verify the generator was called correctly
        mock_factory.assert_called_once()
        mock_generator.fit_generate.assert_called_once()
    
    @patch('sdk.factory.GeneratorFactory.create_generator')
    def test_generate_with_config_file(self, mock_factory, mock_generator, sample_csv_file, sample_config_file, tmp_path):
        mock_factory.return_value = mock_generator
        output_file = tmp_path / "output.csv"
        
        result = runner.invoke(app, [
            "generate",
            sample_csv_file,
            str(output_file),
            "--config", sample_config_file
        ])
        
        assert result.exit_code == 0
        assert "Loading configuration from" in result.stdout
        assert "Generated 5 synthetic samples" in result.stdout
    
    def test_generate_invalid_input_file(self, tmp_path):
        output_file = tmp_path / "output.csv"
        
        result = runner.invoke(app, [
            "generate",
            "non_existent_file.csv",
            str(output_file),
            "--generator-type", "sdv",
            "--model-type", "gaussian_copula"
        ])
        
        assert result.exit_code != 0
        assert "Error" in result.stdout or "does not exist" in result.stdout
    
    @patch('sdk.factory.GeneratorFactory.create_generator')
    def test_generate_with_categorical_columns(self, mock_factory, mock_generator, sample_csv_file, tmp_path):
        mock_factory.return_value = mock_generator
        output_file = tmp_path / "output.csv"
        
        result = runner.invoke(app, [
            "generate",
            sample_csv_file,
            str(output_file),
            "--generator-type", "sdv",
            "--model-type", "gaussian_copula",
            "--categorical", "gender",
            "--categorical", "city"
        ])
        
        assert result.exit_code == 0
        # Verify categorical columns were passed to config
        config = mock_factory.call_args[0][0]
        assert config.categorical_columns == ["gender", "city"]
    
    @patch('sdk.factory.GeneratorFactory.create_generator')
    def test_generate_verbose_mode(self, mock_factory, mock_generator, sample_csv_file, tmp_path):
        mock_factory.return_value = mock_generator
        output_file = tmp_path / "output.csv"
        
        result = runner.invoke(app, [
            "generate",
            sample_csv_file,
            str(output_file),
            "--generator-type", "sdv",
            "--model-type", "gaussian_copula",
            "--verbose"
        ])
        
        assert result.exit_code == 0
        assert "Configuration:" in result.stdout or "verbose" in result.stdout.lower()


class TestValidateCommand:
    """Test the validate command"""
    
    @patch('sdk.validator.SyntheticDataValidator')
    def test_validate_basic(self, mock_validator_class, sample_csv_file, tmp_path):
        mock_validator = Mock()
        mock_validator.validate_all = Mock(return_value={
            "statistical_similarity": {
                "mean_difference": 0.05,
                "std_difference": 0.03,
                "correlation_difference": 0.02
            },
            "privacy_metrics": {
                "unique_ratio": 0.95,
                "exact_match_ratio": 0.0
            },
            "data_utility": {
                "column_correlation": 0.98,
                "mutual_information": 0.85
            }
        })
        mock_validator_class.return_value = mock_validator
        
        synthetic_file = tmp_path / "synthetic.csv"
        synthetic_file.write_text(Path(sample_csv_file).read_text())
        
        result = runner.invoke(app, [
            "validate",
            sample_csv_file,
            str(synthetic_file)
        ])
        
        assert result.exit_code == 0
        assert "Validation Results" in result.stdout
        assert "Statistical Similarity" in result.stdout
        assert "Privacy Metrics" in result.stdout
        assert "Data Utility" in result.stdout
    
    def test_validate_missing_files(self):
        result = runner.invoke(app, [
            "validate",
            "non_existent1.csv",
            "non_existent2.csv"
        ])
        
        assert result.exit_code != 0
        assert "Error" in result.stdout or "does not exist" in result.stdout
    
    @patch('sdk.validator.SyntheticDataValidator')
    def test_validate_with_output_file(self, mock_validator_class, sample_csv_file, tmp_path):
        mock_validator = Mock()
        mock_validator.validate_all = Mock(return_value={
            "statistical_similarity": {"mean_difference": 0.05}
        })
        mock_validator_class.return_value = mock_validator
        
        synthetic_file = tmp_path / "synthetic.csv"
        synthetic_file.write_text(Path(sample_csv_file).read_text())
        output_file = tmp_path / "validation_results.json"
        
        result = runner.invoke(app, [
            "validate",
            sample_csv_file,
            str(synthetic_file),
            "--output", str(output_file)
        ])
        
        assert result.exit_code == 0
        assert output_file.exists()
        assert "Results saved to" in result.stdout


class TestInfoCommand:
    """Test the info command"""
    
    def test_info_list_generators(self):
        result = runner.invoke(app, ["info", "--list-generators"])
        
        assert result.exit_code == 0
        assert "Available Generators" in result.stdout
        assert "sdv" in result.stdout
        assert "ctgan" in result.stdout
        assert "ydata" in result.stdout
    
    def test_info_generator_details(self):
        result = runner.invoke(app, ["info", "--generator", "sdv"])
        
        assert result.exit_code == 0
        assert "SDV Generator Information" in result.stdout or "sdv" in result.stdout.lower()
        assert "Models" in result.stdout or "model" in result.stdout.lower()
    
    def test_info_invalid_generator(self):
        result = runner.invoke(app, ["info", "--generator", "invalid_generator"])
        
        assert result.exit_code != 0 or "Unknown generator" in result.stdout
    
    def test_info_list_models(self):
        result = runner.invoke(app, ["info", "--list-models", "sdv"])
        
        assert result.exit_code == 0
        assert "gaussian_copula" in result.stdout or "Gaussian Copula" in result.stdout
        assert "ctgan" in result.stdout or "CTGAN" in result.stdout


class TestCreateConfigCommand:
    """Test the create-config command"""
    
    def test_create_config_interactive(self, tmp_path):
        output_file = tmp_path / "config.yaml"
        
        # Simulate interactive input
        result = runner.invoke(app, [
            "create-config",
            str(output_file)
        ], input="sdv\ngaussian_copula\n1000\n\n\n\n")
        
        assert result.exit_code == 0
        assert output_file.exists()
        assert "Configuration saved to" in result.stdout
        
        # Verify config content
        with open(output_file) as f:
            config = yaml.safe_load(f)
            assert config["generator_type"] == "sdv"
            assert config["model_type"] == "gaussian_copula"
            assert config["num_samples"] == 1000
    
    def test_create_config_from_template(self, tmp_path):
        output_file = tmp_path / "config.yaml"
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="generator_type: sdv\nmodel_type: gaussian_copula")):
                result = runner.invoke(app, [
                    "create-config",
                    str(output_file),
                    "--template", "sdv_config"
                ])
        
        assert result.exit_code == 0
        assert "Loading template" in result.stdout
    
    def test_create_config_json_format(self, tmp_path):
        output_file = tmp_path / "config.json"
        
        result = runner.invoke(app, [
            "create-config",
            str(output_file),
            "--format", "json"
        ], input="sdv\ngaussian_copula\n1000\n\n\n\n")
        
        assert result.exit_code == 0
        assert output_file.exists()
        
        # Verify JSON format
        with open(output_file) as f:
            config = json.load(f)
            assert config["generator_type"] == "sdv"


class TestCLIErrorHandling:
    """Test error handling in CLI"""
    
    def test_keyboard_interrupt(self, sample_csv_file, tmp_path):
        output_file = tmp_path / "output.csv"
        
        with patch('sdk.factory.GeneratorFactory.create_generator') as mock_factory:
            mock_factory.side_effect = KeyboardInterrupt()
            
            result = runner.invoke(app, [
                "generate",
                sample_csv_file,
                str(output_file),
                "--generator-type", "sdv",
                "--model-type", "gaussian_copula"
            ])
            
            assert "Operation cancelled by user" in result.stdout or result.exit_code != 0
    
    @patch('sdk.factory.GeneratorFactory.create_generator')
    def test_generation_error(self, mock_factory, sample_csv_file, tmp_path):
        mock_factory.side_effect = ValueError("Invalid configuration")
        output_file = tmp_path / "output.csv"
        
        result = runner.invoke(app, [
            "generate",
            sample_csv_file,
            str(output_file),
            "--generator-type", "sdv",
            "--model-type", "gaussian_copula"
        ])
        
        assert result.exit_code != 0
        assert "Error" in result.stdout
        assert "Invalid configuration" in result.stdout


class TestCLIHelpers:
    """Test CLI helper functions and utilities"""
    
    def test_version_command(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.stdout.lower()
    
    def test_help_command(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.stdout
        assert "generate" in result.stdout
        assert "validate" in result.stdout
    
    def test_command_help(self):
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "Generate synthetic data" in result.stdout
        assert "--generator-type" in result.stdout
        assert "--model-type" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__])