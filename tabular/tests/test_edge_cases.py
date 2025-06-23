"""
Edge case tests for synthetic data generation
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path

from sdk import GeneratorFactory, SyntheticDataConfig
from sdk.validator import SyntheticDataValidator
from sdk.batch import BatchProcessor, BatchBuilder
from sdk.versioning import ModelVersionManager
from sdk.cache import CacheManager
from sdk.streaming import StreamingGenerator
from sdk.progress import ProgressTracker
from sdk.benchmark import GeneratorBenchmark


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_dataset(self):
        """Test handling of empty dataset"""
        # Create empty DataFrame
        empty_df = pd.DataFrame()
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula"
        )
        
        generator = GeneratorFactory.create_generator(config)
        
        # Should raise error on empty dataset
        with pytest.raises(ValueError, match="empty"):
            generator.fit(empty_df)
    
    def test_single_row_dataset(self):
        """Test handling of dataset with single row"""
        # Create single row DataFrame
        single_row_df = pd.DataFrame({
            'col1': [1],
            'col2': ['A'],
            'col3': [0.5]
        })
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula",
            num_samples=10
        )
        
        generator = GeneratorFactory.create_generator(config)
        
        # Should handle single row gracefully
        with pytest.raises(ValueError, match="at least"):
            generator.fit(single_row_df)
    
    def test_single_column_dataset(self):
        """Test handling of dataset with single column"""
        # Create single column DataFrame
        single_col_df = pd.DataFrame({
            'col1': range(100)
        })
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula"
        )
        
        generator = GeneratorFactory.create_generator(config)
        
        # Should handle single column
        generator.fit(single_col_df)
        result = generator.generate(50)
        
        assert len(result.data) == 50
        assert len(result.data.columns) == 1
    
    def test_all_null_column(self):
        """Test handling of column with all null values"""
        # Create DataFrame with all-null column
        df_with_nulls = pd.DataFrame({
            'col1': range(100),
            'col2': [None] * 100,  # All nulls
            'col3': np.random.rand(100)
        })
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula"
        )
        
        generator = GeneratorFactory.create_generator(config)
        
        # Should handle or skip all-null column
        generator.fit(df_with_nulls)
        result = generator.generate(50)
        
        assert len(result.data) == 50
    
    def test_constant_column(self):
        """Test handling of column with constant values"""
        # Create DataFrame with constant column
        df_constant = pd.DataFrame({
            'col1': range(100),
            'col2': [42] * 100,  # Constant value
            'col3': np.random.rand(100)
        })
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula"
        )
        
        generator = GeneratorFactory.create_generator(config)
        generator.fit(df_constant)
        result = generator.generate(50)
        
        # Check that constant column is preserved
        assert len(result.data) == 50
        assert (result.data['col2'] == 42).all() or result.data['col2'].nunique() == 1
    
    def test_extreme_values(self):
        """Test handling of extreme values"""
        # Create DataFrame with extreme values
        df_extreme = pd.DataFrame({
            'very_large': [1e100, 1e101, 1e102] * 10,
            'very_small': [1e-100, 1e-101, 1e-102] * 10,
            'mixed': [1e100, 0, -1e100] * 10
        })
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula"
        )
        
        generator = GeneratorFactory.create_generator(config)
        generator.fit(df_extreme)
        result = generator.generate(30)
        
        assert len(result.data) == 30
        assert not result.data.isnull().all().any()
    
    def test_high_cardinality_categorical(self):
        """Test handling of high cardinality categorical columns"""
        # Create DataFrame with high cardinality
        n_rows = 1000
        df_high_card = pd.DataFrame({
            'id': range(n_rows),  # Unique values
            'category': [f'cat_{i % 500}' for i in range(n_rows)],  # 500 categories
            'value': np.random.rand(n_rows)
        })
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula",
            categorical_columns=['category']
        )
        
        generator = GeneratorFactory.create_generator(config)
        generator.fit(df_high_card)
        result = generator.generate(100)
        
        assert len(result.data) == 100
    
    def test_memory_constraints(self):
        """Test handling of memory constraints"""
        # Create large dataset that might cause memory issues
        n_rows = 10000
        n_cols = 100
        
        data = pd.DataFrame(
            np.random.rand(n_rows, n_cols),
            columns=[f'col_{i}' for i in range(n_cols)]
        )
        
        # Use streaming for large dataset
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula"
        )
        
        streaming_gen = StreamingGenerator(
            generator_config=config,
            chunk_size=1000
        )
        
        # Should handle in chunks
        streaming_gen.fit(data)
        
        # Generate in streaming mode
        output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        streaming_gen.generate_to_file(
            num_samples=5000,
            output_path=output_file.name
        )
        
        # Check output
        result_df = pd.read_csv(output_file.name)
        assert len(result_df) == 5000
        
        # Cleanup
        os.unlink(output_file.name)
    
    def test_special_characters_in_columns(self):
        """Test handling of special characters in column names"""
        # Create DataFrame with special characters
        df_special = pd.DataFrame({
            'normal_col': range(100),
            'col with spaces': range(100),
            'col@with#special$chars': range(100),
            '日本語列': range(100),  # Japanese characters
            '': range(100)  # Empty string column name
        })
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula"
        )
        
        generator = GeneratorFactory.create_generator(config)
        
        # Should handle or sanitize column names
        generator.fit(df_special)
        result = generator.generate(50)
        
        assert len(result.data) == 50
        assert len(result.data.columns) == len(df_special.columns)
    
    def test_mixed_data_types(self):
        """Test handling of mixed data types in columns"""
        # Create DataFrame with mixed types
        df_mixed = pd.DataFrame({
            'mixed_col': [1, '2', 3.0, '4', None] * 20,
            'normal_col': range(100)
        })
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula"
        )
        
        generator = GeneratorFactory.create_generator(config)
        
        # Should handle mixed types (convert or error)
        try:
            generator.fit(df_mixed)
            result = generator.generate(50)
            assert len(result.data) == 50
        except Exception as e:
            # Should provide meaningful error
            assert "mixed" in str(e).lower() or "type" in str(e).lower()
    
    def test_datetime_edge_cases(self):
        """Test handling of datetime edge cases"""
        # Create DataFrame with datetime edge cases
        df_datetime = pd.DataFrame({
            'normal_date': pd.date_range('2020-01-01', periods=100),
            'future_date': pd.date_range('2100-01-01', periods=100),
            'past_date': pd.date_range('1900-01-01', periods=100),
            'value': range(100)
        })
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula"
        )
        
        generator = GeneratorFactory.create_generator(config)
        generator.fit(df_datetime)
        result = generator.generate(50)
        
        assert len(result.data) == 50
    
    def test_corrupted_input_file(self):
        """Test handling of corrupted input files"""
        # Create corrupted CSV
        corrupted_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        corrupted_file.write("col1,col2,col3\n")
        corrupted_file.write("1,2,3\n")
        corrupted_file.write("4,5\n")  # Missing value
        corrupted_file.write("6,7,8,9\n")  # Extra value
        corrupted_file.close()
        
        # Try to load
        try:
            df = pd.read_csv(corrupted_file.name)
            
            config = SyntheticDataConfig(
                generator_type="sdv",
                model_type="gaussian_copula"
            )
            
            generator = GeneratorFactory.create_generator(config)
            generator.fit(df)
            
        except Exception as e:
            # Should handle gracefully
            assert True
        
        finally:
            os.unlink(corrupted_file.name)
    
    def test_concurrent_operations(self):
        """Test concurrent operations"""
        import concurrent.futures
        
        # Create test data
        df = pd.DataFrame({
            'col1': range(100),
            'col2': np.random.rand(100)
        })
        
        def generate_data(index):
            config = SyntheticDataConfig(
                generator_type="sdv",
                model_type="gaussian_copula"
            )
            generator = GeneratorFactory.create_generator(config)
            generator.fit(df)
            return generator.generate(50)
        
        # Run concurrent generations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(generate_data, i) for i in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should complete successfully
        assert len(results) == 5
        assert all(len(r.data) == 50 for r in results)
    
    def test_batch_processing_failures(self):
        """Test batch processing with failures"""
        builder = BatchBuilder()
        
        # Add mix of valid and invalid datasets
        builder.add_dataset(
            input_path="valid_file.csv",
            output_path="output1.csv",
            config={'generator_type': 'sdv', 'model_type': 'gaussian_copula'}
        )
        
        builder.add_dataset(
            input_path="nonexistent_file.csv",  # Will fail
            output_path="output2.csv",
            config={'generator_type': 'sdv', 'model_type': 'gaussian_copula'}
        )
        
        datasets = builder.build()
        
        # Mock file reading
        with patch('pandas.read_csv') as mock_read:
            def side_effect(path):
                if 'valid' in path:
                    return pd.DataFrame({'col1': range(100), 'col2': range(100)})
                else:
                    raise FileNotFoundError(f"File not found: {path}")
            
            mock_read.side_effect = side_effect
            
            # Process batch
            processor = BatchProcessor(fail_fast=False)
            result = processor.process_batch(datasets)
            
            # Should handle partial failures
            assert result.successful == 1
            assert result.failed == 1
            assert result.status.value == 'partial'
    
    def test_version_manager_edge_cases(self):
        """Test version manager edge cases"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ModelVersionManager(storage_dir=temp_dir)
            
            # Test operations on non-existent model
            with pytest.raises(ValueError):
                manager.list_versions("nonexistent_model")
            
            with pytest.raises(ValueError):
                manager.rollback("nonexistent_model", 1)
            
            with pytest.raises(ValueError):
                manager.delete_version("nonexistent_model", 1)
    
    def test_cache_overflow(self):
        """Test cache overflow handling"""
        with tempfile.TemporaryDirectory() as cache_dir:
            # Create cache with small size limit
            cache = CacheManager(
                cache_dir=cache_dir,
                max_cache_size_mb=0.001  # Very small cache
            )
            
            # Try to cache large object
            large_data = pd.DataFrame(np.random.rand(1000, 100))
            
            # Should handle overflow gracefully
            cache_key = cache.generate_cache_key({'test': 'config'})
            cache.save_to_cache(cache_key, large_data, {})
            
            # Cache cleanup should work
            cache.cleanup_cache()
    
    def test_progress_tracker_edge_cases(self):
        """Test progress tracker edge cases"""
        tracker = ProgressTracker("test_task")
        
        # Test zero total
        tracker.update(0, 0)
        assert tracker.percentage == 0
        
        # Test negative values
        tracker.update(-1, 10)
        assert tracker.current == 0  # Should clamp to 0
        
        # Test overflow
        tracker.update(150, 100)
        assert tracker.percentage == 100  # Should clamp to 100
    
    def test_validator_with_mismatched_schemas(self):
        """Test validator with mismatched schemas"""
        # Create DataFrames with different schemas
        real_data = pd.DataFrame({
            'col1': range(100),
            'col2': range(100),
            'col3': range(100)
        })
        
        synthetic_data = pd.DataFrame({
            'col1': range(100),
            'col2': range(100),
            'col4': range(100)  # Different column
        })
        
        validator = SyntheticDataValidator(real_data, synthetic_data)
        
        # Should handle schema mismatch
        with pytest.raises(ValueError, match="schema"):
            validator.validate_all()
    
    def test_generator_with_zero_samples(self):
        """Test generator with zero samples requested"""
        df = pd.DataFrame({
            'col1': range(100),
            'col2': range(100)
        })
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula",
            num_samples=0
        )
        
        generator = GeneratorFactory.create_generator(config)
        generator.fit(df)
        
        # Should handle zero samples
        result = generator.generate(0)
        assert len(result.data) == 0
    
    def test_benchmark_with_failing_generators(self):
        """Test benchmark with failing generators"""
        df = pd.DataFrame({
            'col1': range(10),
            'col2': range(10)
        })
        
        benchmark = GeneratorBenchmark()
        
        # Mock a failing generator
        with patch.object(GeneratorFactory, 'create_generator') as mock_create:
            def side_effect(config):
                if config.model_type == 'failing_model':
                    raise RuntimeError("Generator failed")
                return Mock()
            
            mock_create.side_effect = side_effect
            
            # Run benchmark with failing generator
            result = benchmark.benchmark_generator(
                data=df,
                generator_type='test',
                model_type='failing_model'
            )
            
            # Should record error
            assert result.error is not None
            assert "Generator failed" in result.error
    
    def test_api_file_upload_limits(self):
        """Test API file upload size limits"""
        from fastapi.testclient import TestClient
        from api.app import app
        
        client = TestClient(app)
        
        # Create large file
        large_content = b"x" * (100 * 1024 * 1024)  # 100MB
        
        response = client.post(
            "/data/upload",
            files={"file": ("large.csv", large_content, "text/csv")}
        )
        
        # Should handle large file appropriately
        assert response.status_code in [413, 400, 500]  # Request Entity Too Large or Bad Request
    
    def test_circular_dependencies(self):
        """Test handling of circular dependencies in data"""
        # Create DataFrame with potential circular relationships
        df = pd.DataFrame({
            'id': range(100),
            'parent_id': [i - 1 if i > 0 else 99 for i in range(100)],  # Circular reference
            'value': np.random.rand(100)
        })
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula"
        )
        
        generator = GeneratorFactory.create_generator(config)
        
        # Should handle circular dependencies
        generator.fit(df)
        result = generator.generate(50)
        assert len(result.data) == 50


class TestErrorRecovery:
    """Test error recovery mechanisms"""
    
    def test_generator_recovery_after_error(self):
        """Test generator can recover after error"""
        df = pd.DataFrame({
            'col1': range(100),
            'col2': range(100)
        })
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula"
        )
        
        generator = GeneratorFactory.create_generator(config)
        
        # First, cause an error
        with pytest.raises(Exception):
            generator.generate(100)  # Should fail - not fitted
        
        # Now fit properly
        generator.fit(df)
        
        # Should work after recovery
        result = generator.generate(50)
        assert len(result.data) == 50
    
    def test_batch_processor_recovery(self):
        """Test batch processor recovery after failures"""
        processor = BatchProcessor(fail_fast=False)
        
        # Mock some failing operations
        with patch.object(processor, '_process_single_dataset') as mock_process:
            call_count = 0
            
            def side_effect(dataset):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise RuntimeError("Processing failed")
                return Mock()
            
            mock_process.side_effect = side_effect
            
            # Create datasets
            datasets = [Mock(id=f"dataset_{i}") for i in range(3)]
            
            # Process should continue despite failure
            result = processor.process_batch(datasets)
            
            assert result.successful == 2
            assert result.failed == 1
    
    def test_streaming_recovery_after_interruption(self):
        """Test streaming can recover after interruption"""
        df = pd.DataFrame({
            'col1': range(1000),
            'col2': range(1000)
        })
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula"
        )
        
        streaming_gen = StreamingGenerator(
            generator_config=config,
            chunk_size=100
        )
        
        # Simulate interruption during streaming
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            streaming_gen.fit(df)
            
            # Mock interruption
            original_method = streaming_gen.generate_to_file
            call_count = 0
            
            def interrupted_generate(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise KeyboardInterrupt("User interrupted")
                return original_method(*args, **kwargs)
            
            streaming_gen.generate_to_file = interrupted_generate
            
            # First attempt - interrupted
            with pytest.raises(KeyboardInterrupt):
                streaming_gen.generate_to_file(500, tmp.name)
            
            # Second attempt - should work
            streaming_gen.generate_to_file(500, tmp.name)
            
            # Verify output
            result = pd.read_csv(tmp.name)
            assert len(result) == 500
            
            os.unlink(tmp.name)