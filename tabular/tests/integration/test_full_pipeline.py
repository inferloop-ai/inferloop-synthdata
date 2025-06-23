"""
Integration tests for the complete synthetic data generation pipeline
"""

import os
import sys
import tempfile
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List

import pytest
import pandas as pd
import numpy as np
import requests
from fastapi.testclient import TestClient

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.app import app
from sdk.factory import GeneratorFactory
from sdk.base import SyntheticDataConfig, GeneratorType, ModelType
from cli.main import app as cli_app
from typer.testing import CliRunner


class TestFullPipeline:
    """Test complete end-to-end workflows"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    @pytest.fixture
    def cli_runner(self):
        """Typer CLI test runner"""
        return CliRunner()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset"""
        np.random.seed(42)
        return pd.DataFrame({
            'age': np.random.randint(18, 80, 100),
            'income': np.random.normal(50000, 20000, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'score': np.random.uniform(0, 100, 100)
        })
    
    @pytest.fixture
    def config_templates(self):
        """Sample configuration templates"""
        return {
            'sdv': {
                'generator_type': 'sdv',
                'model_type': 'gaussian_copula',
                'num_samples': 50,
                'model_params': {
                    'default_distribution': 'gaussian'
                }
            },
            'ctgan': {
                'generator_type': 'ctgan',
                'model_type': 'ctgan',
                'num_samples': 50,
                'model_params': {
                    'epochs': 10,
                    'batch_size': 100
                }
            },
            'ydata': {
                'generator_type': 'ydata',
                'model_type': 'gaussian_copula',
                'num_samples': 50,
                'model_params': {}
            }
        }
    
    def test_sdk_full_workflow(self, sample_data, config_templates):
        """Test SDK workflow from data loading to validation"""
        for generator_name, config_dict in config_templates.items():
            # Skip if dependencies not available
            try:
                if generator_name == 'sdv':
                    import sdv
                elif generator_name == 'ctgan':
                    import ctgan
                elif generator_name == 'ydata':
                    import ydata_synthetic
            except ImportError:
                pytest.skip(f"{generator_name} not installed")
            
            # Create generator
            config = SyntheticDataConfig(**config_dict)
            generator = GeneratorFactory.create_generator(config)
            
            # Generate synthetic data
            result = generator.fit_generate(sample_data)
            
            # Validate results
            assert result.synthetic_data is not None
            assert len(result.synthetic_data) == config.num_samples
            assert set(result.synthetic_data.columns) == set(sample_data.columns)
            
            # Check metadata
            assert result.metadata['generator_type'] == generator_name
            assert result.metadata['model_type'] == config_dict['model_type']
            assert result.metadata['num_samples'] == config.num_samples
            assert 'generation_time' in result.metadata
            
            # Check validation metrics if available
            if result.validation_metrics:
                assert 'statistical_similarity' in result.validation_metrics
                assert 'column_shapes' in result.validation_metrics
    
    def test_api_full_workflow(self, client, sample_data, config_templates):
        """Test API workflow with all endpoints"""
        # Test health check
        response = client.get("/health")
        assert response.status_code == 200
        
        # Test generator listing
        response = client.get("/generators")
        assert response.status_code == 200
        generators = response.json()
        assert len(generators) >= 1
        
        # Test generation for each generator type
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            for generator_name, config_dict in config_templates.items():
                # Skip if dependencies not available
                response = client.get(f"/generators/{generator_name}")
                if response.status_code == 404:
                    continue
                
                # Test synchronous generation
                with open(temp_file, 'rb') as f:
                    files = {'file': ('data.csv', f, 'text/csv')}
                    response = client.post(
                        "/generate",
                        files=files,
                        data={'config': json.dumps(config_dict)}
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    assert 'synthetic_data' in result
                    assert 'metadata' in result
                    assert 'validation_metrics' in result
                    
                    # Download generated data
                    if 'download_url' in result:
                        download_response = client.get(result['download_url'])
                        assert download_response.status_code == 200
        finally:
            os.unlink(temp_file)
    
    def test_cli_full_workflow(self, cli_runner, sample_data, config_templates):
        """Test CLI workflow with all commands"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save sample data
            input_file = os.path.join(tmpdir, 'input.csv')
            sample_data.to_csv(input_file, index=False)
            
            # Test list generators command
            result = cli_runner.invoke(cli_app, ['list-generators'])
            assert result.exit_code == 0
            assert 'Available generators:' in result.stdout
            
            # Test generation for each generator type
            for generator_name, config_dict in config_templates.items():
                output_file = os.path.join(tmpdir, f'output_{generator_name}.csv')
                config_file = os.path.join(tmpdir, f'config_{generator_name}.json')
                
                # Save config
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f)
                
                # Run generation
                result = cli_runner.invoke(cli_app, [
                    'generate',
                    input_file,
                    output_file,
                    '--config-file', config_file
                ])
                
                # Check result (may fail if dependencies not installed)
                if result.exit_code == 0:
                    assert os.path.exists(output_file)
                    generated_data = pd.read_csv(output_file)
                    assert len(generated_data) == config_dict['num_samples']
                    assert set(generated_data.columns) == set(sample_data.columns)
    
    @pytest.mark.asyncio
    async def test_async_generation_workflow(self, client, sample_data, config_templates):
        """Test asynchronous generation workflow"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            for generator_name, config_dict in config_templates.items():
                # Check if generator available
                response = client.get(f"/generators/{generator_name}")
                if response.status_code == 404:
                    continue
                
                # Submit async generation job
                with open(temp_file, 'rb') as f:
                    files = {'file': ('data.csv', f, 'text/csv')}
                    response = client.post(
                        "/generate/async",
                        files=files,
                        data={'config': json.dumps(config_dict)}
                    )
                
                if response.status_code == 202:
                    result = response.json()
                    job_id = result['job_id']
                    assert 'status_url' in result
                    
                    # Poll for completion
                    max_attempts = 30
                    for _ in range(max_attempts):
                        status_response = client.get(f"/jobs/{job_id}")
                        if status_response.status_code == 200:
                            job_status = status_response.json()
                            if job_status['status'] in ['completed', 'failed']:
                                break
                        await asyncio.sleep(1)
                    
                    # Check final status
                    if job_status['status'] == 'completed':
                        assert 'result' in job_status
                        assert 'synthetic_data' in job_status['result']
        finally:
            os.unlink(temp_file)
    
    def test_validation_workflow(self, client, sample_data):
        """Test data validation workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create original and synthetic data files
            original_file = os.path.join(tmpdir, 'original.csv')
            synthetic_file = os.path.join(tmpdir, 'synthetic.csv')
            
            sample_data.to_csv(original_file, index=False)
            
            # Create synthetic data with some modifications
            synthetic_data = sample_data.copy()
            synthetic_data['age'] = synthetic_data['age'] + np.random.normal(0, 5, len(synthetic_data))
            synthetic_data['income'] = synthetic_data['income'] * 1.1
            synthetic_data.to_csv(synthetic_file, index=False)
            
            # Test validation endpoint
            with open(original_file, 'rb') as f1, open(synthetic_file, 'rb') as f2:
                files = {
                    'original_file': ('original.csv', f1, 'text/csv'),
                    'synthetic_file': ('synthetic.csv', f2, 'text/csv')
                }
                response = client.post("/validate", files=files)
            
            assert response.status_code == 200
            validation_result = response.json()
            
            # Check validation metrics
            assert 'statistical_similarity' in validation_result
            assert 'privacy_metrics' in validation_result
            assert 'utility_metrics' in validation_result
            assert 'column_metrics' in validation_result
    
    def test_error_handling_workflow(self, client):
        """Test error handling across the pipeline"""
        # Test invalid file format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is not CSV data")
            temp_file = f.name
        
        try:
            # Test generation with invalid file
            with open(temp_file, 'rb') as f:
                files = {'file': ('data.txt', f, 'text/plain')}
                response = client.post(
                    "/generate",
                    files=files,
                    data={'config': json.dumps({'generator_type': 'sdv', 'model_type': 'gaussian_copula'})}
                )
            
            # Should return error (either from file validation or parsing)
            assert response.status_code >= 400
            
            # Test with invalid configuration
            valid_csv = "age,income\n25,50000\n30,60000"
            files = {'file': ('data.csv', valid_csv.encode(), 'text/csv')}
            response = client.post(
                "/generate",
                files=files,
                data={'config': json.dumps({'generator_type': 'invalid_generator'})}
            )
            assert response.status_code >= 400
            
        finally:
            os.unlink(temp_file)
    
    def test_performance_workflow(self, sample_data, config_templates):
        """Test performance with larger datasets"""
        # Create larger dataset
        large_data = pd.concat([sample_data] * 10, ignore_index=True)
        
        for generator_name, config_dict in config_templates.items():
            # Skip if dependencies not available
            try:
                if generator_name == 'sdv':
                    import sdv
                elif generator_name == 'ctgan':
                    import ctgan
                elif generator_name == 'ydata':
                    import ydata_synthetic
            except ImportError:
                continue
            
            # Time the generation
            config = SyntheticDataConfig(**config_dict)
            generator = GeneratorFactory.create_generator(config)
            
            start_time = time.time()
            result = generator.fit_generate(large_data)
            end_time = time.time()
            
            generation_time = end_time - start_time
            
            # Check performance
            assert result.synthetic_data is not None
            assert generation_time < 60  # Should complete within 1 minute
            
            # Check memory usage stays reasonable
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            assert memory_mb < 2048  # Less than 2GB


class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios"""
    
    @pytest.fixture
    def healthcare_data(self):
        """Create sample healthcare dataset"""
        np.random.seed(42)
        return pd.DataFrame({
            'patient_id': range(1000),
            'age': np.random.normal(45, 15, 1000).clip(18, 90).astype(int),
            'gender': np.random.choice(['M', 'F'], 1000),
            'blood_pressure': np.random.normal(120, 20, 1000),
            'cholesterol': np.random.normal(200, 40, 1000),
            'diagnosis': np.random.choice(['Healthy', 'Hypertension', 'Diabetes'], 1000, p=[0.6, 0.25, 0.15])
        })
    
    @pytest.fixture
    def financial_data(self):
        """Create sample financial dataset"""
        np.random.seed(42)
        return pd.DataFrame({
            'customer_id': range(500),
            'age': np.random.normal(40, 12, 500).clip(18, 75).astype(int),
            'income': np.random.lognormal(10.5, 0.7, 500),
            'credit_score': np.random.normal(700, 100, 500).clip(300, 850).astype(int),
            'loan_amount': np.random.lognormal(9.5, 1.2, 500),
            'default': np.random.choice([0, 1], 500, p=[0.9, 0.1])
        })
    
    def test_healthcare_scenario(self, healthcare_data):
        """Test synthetic healthcare data generation"""
        # Configure for healthcare data
        config = SyntheticDataConfig(
            generator_type=GeneratorType.SDV,
            model_type=ModelType.GAUSSIAN_COPULA,
            num_samples=500,
            privacy_settings={
                'differential_privacy': True,
                'epsilon': 1.0
            }
        )
        
        # Generate synthetic data
        generator = GeneratorFactory.create_generator(config)
        result = generator.fit_generate(healthcare_data)
        
        # Validate healthcare-specific constraints
        synthetic = result.synthetic_data
        
        # Age should be in reasonable range
        assert synthetic['age'].min() >= 18
        assert synthetic['age'].max() <= 90
        
        # Gender should only have valid values
        assert set(synthetic['gender'].unique()).issubset({'M', 'F'})
        
        # Medical values should be in reasonable ranges
        assert synthetic['blood_pressure'].min() > 0
        assert synthetic['cholesterol'].min() > 0
        
        # Diagnosis distribution should be similar
        original_dist = healthcare_data['diagnosis'].value_counts(normalize=True)
        synthetic_dist = synthetic['diagnosis'].value_counts(normalize=True)
        
        for diagnosis in original_dist.index:
            if diagnosis in synthetic_dist.index:
                assert abs(original_dist[diagnosis] - synthetic_dist[diagnosis]) < 0.1
    
    def test_financial_scenario(self, financial_data):
        """Test synthetic financial data generation"""
        # Configure for financial data with strict privacy
        config = SyntheticDataConfig(
            generator_type=GeneratorType.CTGAN,
            model_type=ModelType.CTGAN,
            num_samples=300,
            model_params={
                'epochs': 50,
                'batch_size': 50
            }
        )
        
        try:
            import ctgan
        except ImportError:
            pytest.skip("CTGAN not installed")
        
        # Generate synthetic data
        generator = GeneratorFactory.create_generator(config)
        result = generator.fit_generate(financial_data)
        
        # Validate financial-specific constraints
        synthetic = result.synthetic_data
        
        # Credit scores should be in valid range
        assert synthetic['credit_score'].min() >= 300
        assert synthetic['credit_score'].max() <= 850
        
        # Income and loan amounts should be positive
        assert (synthetic['income'] > 0).all()
        assert (synthetic['loan_amount'] > 0).all()
        
        # Default rate should be similar
        original_default_rate = financial_data['default'].mean()
        synthetic_default_rate = synthetic['default'].mean()
        assert abs(original_default_rate - synthetic_default_rate) < 0.05
        
        # Check correlations are preserved
        original_corr = financial_data[['income', 'credit_score', 'loan_amount']].corr()
        synthetic_corr = synthetic[['income', 'credit_score', 'loan_amount']].corr()
        
        # Correlations should be similar (within 0.2)
        corr_diff = abs(original_corr - synthetic_corr)
        assert (corr_diff < 0.2).all().all()
    
    def test_multi_table_scenario(self):
        """Test synthetic data generation for related tables"""
        # Create related datasets
        customers = pd.DataFrame({
            'customer_id': range(100),
            'age': np.random.randint(18, 70, 100),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
        })
        
        transactions = pd.DataFrame({
            'transaction_id': range(500),
            'customer_id': np.random.choice(range(100), 500),
            'amount': np.random.lognormal(3, 1.5, 500),
            'category': np.random.choice(['Food', 'Transport', 'Shopping', 'Bills'], 500)
        })
        
        # Generate synthetic data for each table
        config = SyntheticDataConfig(
            generator_type=GeneratorType.SDV,
            model_type=ModelType.GAUSSIAN_COPULA,
            num_samples=100
        )
        
        generator = GeneratorFactory.create_generator(config)
        
        # Generate customers
        customers_result = generator.fit_generate(customers)
        synthetic_customers = customers_result.synthetic_data
        
        # Generate transactions with matching customer IDs
        config.num_samples = 500
        generator = GeneratorFactory.create_generator(config)
        transactions_result = generator.fit_generate(transactions)
        synthetic_transactions = transactions_result.synthetic_data
        
        # Ensure referential integrity
        # All transaction customer_ids should exist in customers
        valid_customer_ids = set(synthetic_customers['customer_id'])
        transaction_customer_ids = set(synthetic_transactions['customer_id'])
        
        # Most transaction customer IDs should be valid (allow some flexibility due to generation)
        valid_ratio = len(transaction_customer_ids.intersection(valid_customer_ids)) / len(transaction_customer_ids)
        assert valid_ratio > 0.8  # At least 80% should be valid


class TestConcurrentOperations:
    """Test concurrent operations and thread safety"""
    
    @pytest.mark.asyncio
    async def test_concurrent_generation(self, sample_data):
        """Test multiple concurrent generation requests"""
        configs = [
            SyntheticDataConfig(
                generator_type=GeneratorType.SDV,
                model_type=ModelType.GAUSSIAN_COPULA,
                num_samples=50
            )
            for _ in range(5)
        ]
        
        async def generate_data(config, data):
            """Generate data asynchronously"""
            generator = GeneratorFactory.create_generator(config)
            return generator.fit_generate(data)
        
        # Run multiple generations concurrently
        tasks = [generate_data(config, sample_data) for config in configs]
        results = await asyncio.gather(*tasks)
        
        # Verify all completed successfully
        assert len(results) == 5
        for result in results:
            assert result.synthetic_data is not None
            assert len(result.synthetic_data) == 50
    
    def test_thread_safe_api_calls(self, sample_data):
        """Test thread-safe API calls"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker(worker_id, data, config_dict):
            """Worker thread for API calls"""
            try:
                config = SyntheticDataConfig(**config_dict)
                generator = GeneratorFactory.create_generator(config)
                result = generator.fit_generate(data)
                results_queue.put((worker_id, result))
            except Exception as e:
                errors_queue.put((worker_id, str(e)))
        
        # Create configuration
        config_dict = {
            'generator_type': 'sdv',
            'model_type': 'gaussian_copula',
            'num_samples': 25
        }
        
        # Start multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i, sample_data, config_dict))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check results
        assert errors_queue.empty(), f"Errors occurred: {list(errors_queue.queue)}"
        assert results_queue.qsize() == 10
        
        # Verify all results are valid
        while not results_queue.empty():
            worker_id, result = results_queue.get()
            assert result.synthetic_data is not None
            assert len(result.synthetic_data) == 25