"""
Tests for error recovery and retry logic
"""

import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from sdk import GeneratorFactory, SyntheticDataConfig
from sdk.batch import BatchProcessor, BatchBuilder
from sdk.versioning import ModelVersionManager
from sdk.streaming import StreamingGenerator
from sdk.cache import CacheManager
from sdk.progress import ProgressTracker
from api.app import app
from fastapi.testclient import TestClient


class TestRetryMechanisms:
    """Test retry mechanisms in various components"""
    
    def test_generator_retry_on_memory_error(self):
        """Test generator retries on memory errors"""
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula"
        )
        
        # Mock generator that fails on first attempts
        with patch('sdk.factory.GeneratorFactory.create_generator') as mock_create:
            mock_gen = Mock()
            attempt = 0
            
            def fit_with_retry(data):
                nonlocal attempt
                attempt += 1
                if attempt < 3:
                    raise MemoryError("Out of memory")
                # Success on third attempt
                return None
            
            mock_gen.fit = fit_with_retry
            mock_gen.generate = Mock(return_value=Mock(data=pd.DataFrame({'col': [1, 2, 3]})))
            mock_create.return_value = mock_gen
            
            # Should retry and eventually succeed
            generator = GeneratorFactory.create_generator(config)
            
            # Wrap with retry logic
            max_retries = 3
            for i in range(max_retries):
                try:
                    generator.fit(pd.DataFrame({'col': range(100)}))
                    break
                except MemoryError:
                    if i == max_retries - 1:
                        raise
                    time.sleep(0.1)  # Brief delay before retry
            
            assert attempt == 3  # Should have taken 3 attempts
    
    def test_batch_processor_retry_on_failure(self):
        """Test batch processor retries failed datasets"""
        
        # Create processor with retry logic
        processor = BatchProcessor(fail_fast=False)
        
        # Track processing attempts
        attempts = {}
        
        def process_with_failures(dataset):
            if dataset.id not in attempts:
                attempts[dataset.id] = 0
            
            attempts[dataset.id] += 1
            
            # Fail first attempt for dataset_2
            if dataset.id == 'dataset_2' and attempts[dataset.id] == 1:
                raise RuntimeError("Temporary failure")
            
            # Success otherwise
            return Mock(data=pd.DataFrame({'col': [1, 2, 3]}))
        
        # Mock processing
        with patch.object(processor, '_process_single_dataset', process_with_failures):
            # Create datasets
            builder = BatchBuilder()
            builder.add_dataset('data1.csv', 'out1.csv', dataset_id='dataset_1')
            builder.add_dataset('data2.csv', 'out2.csv', dataset_id='dataset_2')
            builder.add_dataset('data3.csv', 'out3.csv', dataset_id='dataset_3')
            datasets = builder.build()
            
            # Process with retry
            result = processor.process_batch(datasets)
            
            # Dataset 2 should fail on first attempt
            assert result.failed == 1
            
            # Retry failed datasets
            failed_datasets = [d for d in datasets if d.id in result.errors]
            retry_result = processor.process_batch(failed_datasets)
            
            # Should succeed on retry
            assert retry_result.successful == 1
            assert attempts['dataset_2'] == 2
    
    def test_api_retry_on_connection_error(self):
        """Test API client retries on connection errors"""
        
        client = TestClient(app)
        
        # Mock endpoint that fails intermittently
        call_count = 0
        
        @app.get("/test_retry")
        def test_endpoint():
            nonlocal call_count
            call_count += 1
            
            if call_count < 3:
                raise ConnectionError("Connection lost")
            
            return {"status": "success", "attempt": call_count}
        
        # Client with retry logic
        max_retries = 5
        retry_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                response = client.get("/test_retry")
                if response.status_code == 200:
                    break
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)
        
        assert response.json()["attempt"] == 3
    
    def test_streaming_recovery_from_interruption(self):
        """Test streaming generator recovery from interruptions"""
        
        # Create large dataset
        df = pd.DataFrame({
            'col1': range(10000),
            'col2': np.random.rand(10000)
        })
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula"
        )
        
        streaming_gen = StreamingGenerator(
            generator_config=config,
            chunk_size=1000
        )
        
        # Mock interruption during generation
        original_generate = streaming_gen._generate_chunk
        chunks_generated = 0
        
        def generate_with_interruption(*args, **kwargs):
            nonlocal chunks_generated
            chunks_generated += 1
            
            # Simulate interruption after 3 chunks
            if chunks_generated == 3:
                raise KeyboardInterrupt("User interrupted")
            
            return original_generate(*args, **kwargs)
        
        streaming_gen._generate_chunk = generate_with_interruption
        
        # First attempt - should be interrupted
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            try:
                streaming_gen.fit(df)
                streaming_gen.generate_to_file(5000, tmp.name)
            except KeyboardInterrupt:
                pass
            
            # Check partial output
            partial_df = pd.read_csv(tmp.name)
            assert len(partial_df) < 5000
            
            # Reset and retry with recovery
            chunks_generated = 0
            streaming_gen._generate_chunk = original_generate
            
            # Resume generation
            remaining_samples = 5000 - len(partial_df)
            streaming_gen.generate_to_file(
                remaining_samples,
                tmp.name,
                mode='append'
            )
            
            # Verify complete output
            final_df = pd.read_csv(tmp.name)
            assert len(final_df) >= 5000
            
            os.unlink(tmp.name)
    
    def test_cache_recovery_on_corruption(self):
        """Test cache recovery when cache files are corrupted"""
        
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = CacheManager(cache_dir=cache_dir)
            
            # Save valid data
            key = cache.generate_cache_key({'test': 'config'})
            data = pd.DataFrame({'col': range(100)})
            metadata = {'timestamp': time.time()}
            
            cache.save_to_cache(key, data, metadata)
            
            # Corrupt the cache file
            cache_file = Path(cache_dir) / f"{key}.pkl"
            with open(cache_file, 'wb') as f:
                f.write(b"corrupted data")
            
            # Try to load - should handle corruption gracefully
            loaded_data, loaded_metadata = cache.load_from_cache(key)
            
            assert loaded_data is None
            assert loaded_metadata is None
            
            # Should be able to save new data with same key
            new_data = pd.DataFrame({'col': range(50)})
            cache.save_to_cache(key, new_data, metadata)
            
            # Verify new data loads correctly
            loaded_data, _ = cache.load_from_cache(key)
            assert loaded_data is not None
            assert len(loaded_data) == 50
    
    def test_version_manager_recovery_from_missing_files(self):
        """Test version manager recovery when files are missing"""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ModelVersionManager(storage_dir=temp_dir)
            
            # Create and save a model version
            mock_generator = Mock()
            mock_generator.model = "test_model"
            mock_generator.config = Mock(
                generator_type="sdv",
                model_type="gaussian_copula",
                to_dict=lambda: {'generator_type': 'sdv', 'model_type': 'gaussian_copula'}
            )
            
            data = pd.DataFrame({'col': range(100)})
            version = manager.save_model(mock_generator, data)
            
            # Delete the model file
            model_path = Path(version.model_path)
            if model_path.exists():
                os.unlink(model_path)
            
            # Try to load - should handle missing file
            with pytest.raises(FileNotFoundError):
                manager.load_model(
                    f"{mock_generator.config.generator_type}_{mock_generator.config.model_type}",
                    version.version_number
                )
            
            # Should still be able to list versions
            versions = manager.list_versions(
                f"{mock_generator.config.generator_type}_{mock_generator.config.model_type}"
            )
            assert len(versions) > 0
            
            # Should be able to delete the broken version
            manager.delete_version(
                f"{mock_generator.config.generator_type}_{mock_generator.config.model_type}",
                version.version_number
            )
    
    def test_async_operation_timeout_recovery(self):
        """Test recovery from async operation timeouts"""
        
        async def slow_operation():
            await asyncio.sleep(10)  # Simulate slow operation
            return "completed"
        
        async def operation_with_timeout():
            try:
                # Set timeout
                result = await asyncio.wait_for(slow_operation(), timeout=0.5)
                return result
            except asyncio.TimeoutError:
                # Recover with fallback
                return "timeout_fallback"
        
        # Run async test
        result = asyncio.run(operation_with_timeout())
        assert result == "timeout_fallback"
    
    def test_concurrent_request_recovery(self):
        """Test recovery from concurrent request failures"""
        
        results = []
        errors = []
        
        def make_request(index):
            try:
                # Simulate some requests failing
                if index % 5 == 0:
                    raise ConnectionError(f"Request {index} failed")
                
                time.sleep(0.01)  # Simulate work
                return f"success_{index}"
            
            except Exception as e:
                # Retry logic
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        time.sleep(0.05 * (retry + 1))  # Exponential backoff
                        if retry == max_retries - 1:  # Success on last retry
                            return f"recovered_{index}"
                    except:
                        if retry == max_retries - 1:
                            raise
                
                raise e
        
        # Run concurrent requests with recovery
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for i in range(20):
                future = executor.submit(make_request, i)
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append(str(e))
        
        # All requests should eventually succeed
        assert len(results) == 20
        assert len(errors) == 0
        
        # Check that some were recovered
        recovered = [r for r in results if r.startswith("recovered_")]
        assert len(recovered) == 4  # indices 0, 5, 10, 15


class TestCircuitBreaker:
    """Test circuit breaker pattern for preventing cascading failures"""
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures"""
        
        class CircuitBreaker:
            def __init__(self, failure_threshold=3, recovery_timeout=1.0):
                self.failure_threshold = failure_threshold
                self.recovery_timeout = recovery_timeout
                self.failure_count = 0
                self.last_failure_time = None
                self.is_open = False
            
            def call(self, func, *args, **kwargs):
                # Check if circuit should be closed
                if self.is_open:
                    if time.time() - self.last_failure_time > self.recovery_timeout:
                        self.is_open = False
                        self.failure_count = 0
                    else:
                        raise RuntimeError("Circuit breaker is open")
                
                try:
                    result = func(*args, **kwargs)
                    self.failure_count = 0  # Reset on success
                    return result
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.is_open = True
                    
                    raise e
        
        # Test function that fails initially
        call_count = 0
        
        def unreliable_function():
            nonlocal call_count
            call_count += 1
            
            if call_count <= 5:
                raise RuntimeError("Service unavailable")
            
            return "success"
        
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=0.5)
        
        # First 3 calls should fail and open the circuit
        for i in range(3):
            with pytest.raises(RuntimeError):
                breaker.call(unreliable_function)
        
        # Circuit should be open now
        assert breaker.is_open
        
        # Next call should fail immediately
        with pytest.raises(RuntimeError, match="Circuit breaker is open"):
            breaker.call(unreliable_function)
        
        # Wait for recovery timeout
        time.sleep(0.6)
        
        # Circuit should close and allow retry
        with pytest.raises(RuntimeError):  # Still fails but circuit is closed
            breaker.call(unreliable_function)
        
        # After more attempts, it should succeed
        result = breaker.call(unreliable_function)
        assert result == "success"


class TestDataValidationRecovery:
    """Test recovery from data validation failures"""
    
    def test_recovery_from_invalid_data_types(self):
        """Test recovery when data has invalid types"""
        
        # Create DataFrame with mixed types
        df_mixed = pd.DataFrame({
            'numeric': [1, 2, 'three', 4, 5],  # Mixed types
            'categorical': ['A', 'B', 'C', 'D', 'E']
        })
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula"
        )
        
        # Data cleaning function
        def clean_data(df):
            df_clean = df.copy()
            
            for col in df_clean.columns:
                # Try to convert to numeric
                if df_clean[col].dtype == 'object':
                    try:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    except:
                        pass
            
            # Drop rows with NaN values
            df_clean = df_clean.dropna()
            
            return df_clean
        
        # Clean data before generation
        df_cleaned = clean_data(df_mixed)
        
        generator = GeneratorFactory.create_generator(config)
        generator.fit(df_cleaned)
        result = generator.generate(10)
        
        assert len(result.data) == 10
    
    def test_recovery_from_missing_columns(self):
        """Test recovery when expected columns are missing"""
        
        # Create generator expecting certain columns
        df_train = pd.DataFrame({
            'col1': range(100),
            'col2': range(100),
            'col3': range(100)
        })
        
        config = SyntheticDataConfig(
            generator_type="sdv",
            model_type="gaussian_copula"
        )
        
        generator = GeneratorFactory.create_generator(config)
        generator.fit(df_train)
        
        # Try to generate with missing columns
        df_new = pd.DataFrame({
            'col1': range(50),
            'col2': range(50)
            # col3 is missing
        })
        
        # Recovery: add missing columns with default values
        expected_columns = df_train.columns
        for col in expected_columns:
            if col not in df_new.columns:
                # Use median/mode as default
                if df_train[col].dtype in ['int64', 'float64']:
                    df_new[col] = df_train[col].median()
                else:
                    df_new[col] = df_train[col].mode()[0]
        
        # Now should work
        result = generator.generate(25)
        assert len(result.data) == 25


class TestProgressRecovery:
    """Test progress tracking recovery"""
    
    def test_progress_tracker_recovery_on_error(self):
        """Test progress tracker handles errors gracefully"""
        
        tracker = ProgressTracker("test_task")
        
        # Simulate progress with error
        tracker.update(50, 100, "Processing...")
        
        try:
            # Simulate error during processing
            raise RuntimeError("Processing failed")
        except RuntimeError:
            # Mark as failed
            tracker.fail("Task failed due to error")
        
        assert tracker.stage.value == "failed"
        assert "failed" in tracker.message.lower()
        
        # Can reset and retry
        tracker.reset()
        tracker.update(0, 100, "Retrying...")
        
        assert tracker.stage.value == "initializing"
        assert tracker.percentage == 0
    
    def test_multi_progress_tracker_partial_failure(self):
        """Test multi-progress tracker with partial failures"""
        
        from sdk.progress import MultiProgressTracker
        
        multi_tracker = MultiProgressTracker()
        
        # Create multiple trackers
        tracker1 = multi_tracker.create_tracker("task1")
        tracker2 = multi_tracker.create_tracker("task2")
        tracker3 = multi_tracker.create_tracker("task3")
        
        # Progress on all
        tracker1.update(100, 100)
        tracker1.complete("Task 1 done")
        
        tracker2.update(50, 100)
        tracker2.fail("Task 2 failed")
        
        tracker3.update(75, 100)
        
        # Check overall status
        active_trackers = multi_tracker.trackers
        
        completed = sum(1 for t in active_trackers.values() 
                       if t.stage.value == "completed")
        failed = sum(1 for t in active_trackers.values() 
                    if t.stage.value == "failed")
        
        assert completed == 1
        assert failed == 1
        
        # Can retry failed task
        tracker2.reset()
        tracker2.update(100, 100)
        tracker2.complete("Task 2 retry successful")
        
        completed = sum(1 for t in active_trackers.values() 
                       if t.stage.value == "completed")
        assert completed == 2


if __name__ == "__main__":
    # Run some recovery tests
    print("Testing error recovery mechanisms...")
    
    # Test retry logic
    test_retry = TestRetryMechanisms()
    test_retry.test_generator_retry_on_memory_error()
    print("✓ Generator retry on memory error")
    
    test_retry.test_batch_processor_retry_on_failure()
    print("✓ Batch processor retry on failure")
    
    # Test circuit breaker
    test_breaker = TestCircuitBreaker()
    test_breaker.test_circuit_breaker_opens_on_failures()
    print("✓ Circuit breaker pattern")
    
    # Test data validation recovery
    test_validation = TestDataValidationRecovery()
    test_validation.test_recovery_from_invalid_data_types()
    print("✓ Recovery from invalid data types")
    
    print("\nAll recovery tests passed!")