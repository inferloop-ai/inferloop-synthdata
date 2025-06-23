# tests/test_sdk.py
"""
Test suite for Inferloop Synthetic Data SDK
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sdk import (
    SyntheticDataConfig,
    GeneratorFactory,
    SyntheticDataValidator,
    GenerationResult
)


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    data = {
        'age': np.random.normal(40, 15, 1000).astype(int),
        'income': np.random.lognormal(10, 0.5, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'score': np.random.uniform(0, 100, 1000)
    }
    return pd.DataFrame(data)


@pytest.fixture
def basic_config():
    """Basic configuration for testing"""
    return SyntheticDataConfig(
        generator_type="sdv",
        model_type="gaussian_copula",
        num_samples=100,
        categorical_columns=['category'],
        continuous_columns=['age', 'income', 'score']
    )


class TestSyntheticDataConfig:
    """Test SyntheticDataConfig functionality"""
    
    def test_config_creation(self):
        """Test basic config creation"""
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula",
            num_samples=1000
        )
        
        assert config.generator_type == "sdv"
        assert config.model_type == "gaussian_copula"
        assert config.num_samples == 1000
        assert config.epochs == 300  # default value
    
    def test_config_to_dict(self, basic_config):
        """Test config serialization"""
        config_dict = basic_config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['generator_type'] == "sdv"
        assert config_dict['model_type'] == "gaussian_copula"
        assert config_dict['num_samples'] == 100


class TestGeneratorFactory:
    """Test GeneratorFactory functionality"""
    
    def test_list_generators(self):
        """Test listing available generators"""
        generators = GeneratorFactory.list_generators()
        
        assert isinstance(generators, list)
        assert 'sdv' in generators
        assert 'ctgan' in generators
        assert 'ydata' in generators
    
    def test_create_generator(self, basic_config):
        """Test generator creation"""
        with patch('sdk.sdv_generator.SDV_AVAILABLE', True):
            generator = GeneratorFactory.create_generator(basic_config)
            assert generator is not None
            assert generator.config == basic_config
    
    def test_invalid_generator_type(self):
        """Test handling of invalid generator type"""
        config = SyntheticDataConfig(
            generator_type="invalid",
            model_type="test"
        )
        
        with pytest.raises(ValueError, match="Unknown generator type"):
            GeneratorFactory.create_generator(config)
    
    def test_create_from_dict(self):
        """Test generator creation from dictionary"""
        config_dict = {
            'generator_type': 'sdv',
            'model_type': 'gaussian_copula',
            'num_samples': 500
        }
        
        with patch('sdk.sdv_generator.SDV_AVAILABLE', True):
            generator = GeneratorFactory.create_from_dict(config_dict)
            assert generator.config.generator_type == 'sdv'
            assert generator.config.num_samples == 500


class TestSyntheticDataValidator:
    """Test SyntheticDataValidator functionality"""
    
    def test_validator_creation(self, sample_data):
        """Test validator creation"""
        synthetic_data = sample_data.copy()
        validator = SyntheticDataValidator(sample_data, synthetic_data)
        
        assert validator.real_data is not None
        assert validator.synthetic_data is not None
    
    def test_basic_statistics_validation(self, sample_data):
        """Test basic statistics validation"""
        # Create slightly modified synthetic data
        synthetic_data = sample_data.copy()
        synthetic_data['age'] = synthetic_data['age'] * 1.1  # slight modification
        
        validator = SyntheticDataValidator(sample_data, synthetic_data)
        results = validator.validate_basic_statistics()
        
        assert 'score' in results
        assert 'differences' in results
        assert 0 <= results['score'] <= 1
    
    def test_distribution_validation(self, sample_data):
        """Test distribution validation"""
        synthetic_data = sample_data.copy()
        
        validator = SyntheticDataValidator(sample_data, synthetic_data)
        results = validator.validate_distributions()
        
        assert 'score' in results
        assert 'ks_tests' in results
        assert 0 <= results['score'] <= 1
    
    def test_correlation_validation(self, sample_data):
        """Test correlation validation"""
        synthetic_data = sample_data.copy()
        
        validator = SyntheticDataValidator(sample_data, synthetic_data)
        results = validator.validate_correlations()
        
        assert 'score' in results
        assert 'average_correlation_difference' in results
        assert 0 <= results['score'] <= 1
    
    def test_comprehensive_validation(self, sample_data):
        """Test comprehensive validation"""
        synthetic_data = sample_data.copy()
        
        validator = SyntheticDataValidator(sample_data, synthetic_data)
        results = validator.validate_all()
        
        assert 'basic_stats' in results
        assert 'distribution_similarity' in results
        assert 'correlation_preservation' in results
        assert 'privacy_metrics' in results
        assert 'utility_metrics' in results
        assert 'overall_quality' in results
        assert 0 <= results['overall_quality'] <= 1
    
    def test_generate_report(self, sample_data):
        """Test report generation"""
        synthetic_data = sample_data.copy()
        
        validator = SyntheticDataValidator(sample_data, synthetic_data)
        validator.validate_all()
        report = validator.generate_report()
        
        assert isinstance(report, str)
        assert "Validation Report" in report
        assert "Overall Quality Score" in report


class TestGenerationResult:
    """Test GenerationResult functionality"""
    
    def test_result_creation(self, sample_data, basic_config):
        """Test generation result creation"""
        result = GenerationResult(
            synthetic_data=sample_data,
            config=basic_config,
            generation_time=1.5,
            model_info={'library': 'test'}
        )
        
        assert result.synthetic_data is not None
        assert result.config == basic_config
        assert result.generation_time == 1.5
        assert result.validation_passed is True
    
    def test_result_save(self, sample_data, basic_config, tmp_path):
        """Test saving generation results"""
        result = GenerationResult(
            synthetic_data=sample_data,
            config=basic_config,
            generation_time=1.5,
            model_info={'library': 'test'}
        )
        
        output_path = tmp_path / "test_output.csv"
        result.save(output_path)
        
        assert output_path.exists()
        
        # Check metadata file
        metadata_path = output_path.with_suffix('.metadata.json')
        assert metadata_path.exists()


class TestIntegration:
    """Integration tests"""
    
    @patch('sdk.sdv_generator.SDV_AVAILABLE', True)
    @patch('sdk.sdv_generator.SingleTableMetadata')
    @patch('sdk.sdv_generator.GaussianCopulaSynthesizer')
    def test_end_to_end_workflow(self, mock_synthesizer, mock_metadata_class, sample_data):
        """Test complete end-to-end workflow"""
        # Mock the SDV synthesizer
        mock_synth_instance = Mock()
        mock_synth_instance.sample.return_value = sample_data.head(50)
        mock_synthesizer.return_value = mock_synth_instance
        
        # Mock metadata
        mock_metadata = Mock()
        mock_metadata.to_dict.return_value = {}
        mock_metadata_class.return_value = mock_metadata
        
        # Create configuration
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula",
            num_samples=50,
            categorical_columns=['category']
        )
        
        # Generate synthetic data
        generator = GeneratorFactory.create_generator(config)
        result = generator.fit_generate(sample_data)
        
        # Validate results
        assert result.synthetic_data is not None
        assert len(result.synthetic_data) == 50
        assert result.generation_time >= 0
        
        # Run validation
        validator = SyntheticDataValidator(sample_data, result.synthetic_data)
        validation_results = validator.validate_all()
        
        assert validation_results['overall_quality'] >= 0


if __name__ == "__main__":
    pytest.main([__file__])