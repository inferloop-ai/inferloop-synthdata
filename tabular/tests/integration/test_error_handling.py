"""
Integration tests for error handling and recovery
"""

import os
import sys
import json
import tempfile
from typing import Dict, Any

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.app import app
from sdk.factory import GeneratorFactory
from sdk.base import SyntheticDataConfig


class TestErrorHandlingFlow:
    """Test error handling and recovery workflows"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    @pytest.fixture
    def invalid_csv_data(self):
        """Create various invalid CSV data"""
        return {
            'malformed': "age,income\n25,50000\n30",  # Missing value
            'empty': "",  # Empty file
            'no_headers': "25,50000\n30,60000",  # No headers
            'special_chars': "age,income\n25,50000\n=cmd|'/c calc'!A1,60000",  # Formula injection
            'huge_columns': ",".join([f"col{i}" for i in range(2000)]),  # Too many columns
            'binary': b"\x00\x01\x02\x03\x04\x05",  # Binary data
            'mixed_encoding': "age,income\n25,50000\n30,€60000".encode('latin-1')  # Encoding issues
        }
    
    def test_file_validation_errors(self, client, invalid_csv_data):
        """Test handling of various file validation errors"""
        for error_type, data in invalid_csv_data.items():
            # Convert to bytes if string
            if isinstance(data, str):
                data = data.encode()
            
            files = {'file': (f'{error_type}.csv', data, 'text/csv')}
            config = {'generator_type': 'sdv', 'model_type': 'gaussian_copula'}
            
            response = client.post(
                '/generate',
                files=files,
                data={'config': json.dumps(config)}
            )
            
            # Should return error
            assert response.status_code >= 400
            error_detail = response.json()
            assert 'detail' in error_detail or 'error' in error_detail
    
    def test_configuration_errors(self, client):
        """Test handling of configuration errors"""
        # Valid CSV data
        csv_data = "age,income\n25,50000\n30,60000"
        files = {'file': ('data.csv', csv_data.encode(), 'text/csv')}
        
        # Test various configuration errors
        invalid_configs = [
            {'generator_type': 'invalid_generator'},  # Invalid generator
            {'generator_type': 'sdv'},  # Missing model_type
            {'generator_type': 'sdv', 'model_type': 'invalid_model'},  # Invalid model
            {'generator_type': 'sdv', 'model_type': 'gaussian_copula', 'num_samples': -10},  # Invalid num_samples
            {'generator_type': 'sdv', 'model_type': 'gaussian_copula', 'num_samples': 'not_a_number'},  # Wrong type
            'not_json_at_all',  # Invalid JSON
            None,  # None config
        ]
        
        for config in invalid_configs:
            if config is None:
                response = client.post('/generate', files=files)
            elif isinstance(config, str):
                response = client.post(
                    '/generate',
                    files=files,
                    data={'config': config}
                )
            else:
                response = client.post(
                    '/generate',
                    files=files,
                    data={'config': json.dumps(config)}
                )
            
            # Should return error
            assert response.status_code >= 400
    
    def test_large_file_handling(self, client):
        """Test handling of large files"""
        # Create large CSV (but within limits)
        rows = 10000
        large_data = pd.DataFrame({
            'id': range(rows),
            'value1': np.random.randn(rows),
            'value2': np.random.randn(rows),
            'category': np.random.choice(['A', 'B', 'C'], rows)
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Test with large file
            with open(temp_file, 'rb') as f:
                files = {'file': ('large.csv', f, 'text/csv')}
                config = {
                    'generator_type': 'sdv',
                    'model_type': 'gaussian_copula',
                    'num_samples': 100
                }
                
                response = client.post(
                    '/generate',
                    files=files,
                    data={'config': json.dumps(config)},
                    timeout=60  # Increase timeout for large file
                )
            
            # Should either succeed or fail gracefully
            assert response.status_code in [200, 413, 422, 500]
            
            if response.status_code != 200:
                error_data = response.json()
                assert 'detail' in error_data or 'error' in error_data
        
        finally:
            os.unlink(temp_file)
    
    def test_timeout_handling(self, client):
        """Test handling of request timeouts"""
        # Create data that might take long to process
        complex_data = pd.DataFrame({
            f'col_{i}': np.random.randn(1000) for i in range(50)
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            complex_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            with open(temp_file, 'rb') as f:
                files = {'file': ('complex.csv', f, 'text/csv')}
                config = {
                    'generator_type': 'ctgan',
                    'model_type': 'ctgan',
                    'num_samples': 1000,
                    'model_params': {
                        'epochs': 300,  # High epochs for long processing
                        'batch_size': 500
                    }
                }
                
                # Use very short timeout to trigger timeout
                try:
                    response = client.post(
                        '/generate',
                        files=files,
                        data={'config': json.dumps(config)},
                        timeout=0.001  # Unreasonably short timeout
                    )
                except Exception as e:
                    # Should get timeout error
                    assert 'timeout' in str(e).lower() or 'timed out' in str(e).lower()
        
        finally:
            os.unlink(temp_file)
    
    def test_memory_error_handling(self, client):
        """Test handling of memory errors"""
        # Request generation of extremely large synthetic dataset
        csv_data = "id,value\n1,100\n2,200"
        files = {'file': ('data.csv', csv_data.encode(), 'text/csv')}
        
        config = {
            'generator_type': 'sdv',
            'model_type': 'gaussian_copula',
            'num_samples': 10000000  # Request 10 million samples
        }
        
        response = client.post(
            '/generate',
            files=files,
            data={'config': json.dumps(config)}
        )
        
        # Should either handle gracefully or return appropriate error
        if response.status_code != 200:
            assert response.status_code >= 400
            error_data = response.json()
            assert 'detail' in error_data or 'error' in error_data
    
    def test_concurrent_error_scenarios(self, client):
        """Test error handling under concurrent load"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_error_request(request_id: int, error_type: str):
            """Make request that should trigger error"""
            if error_type == 'invalid_file':
                files = {'file': ('bad.csv', b'invalid csv data', 'text/csv')}
                config = {'generator_type': 'sdv', 'model_type': 'gaussian_copula'}
            elif error_type == 'invalid_config':
                files = {'file': ('data.csv', b'age,income\n25,50000', 'text/csv')}
                config = {'invalid': 'config'}
            else:
                files = {'file': ('empty.csv', b'', 'text/csv')}
                config = {'generator_type': 'sdv', 'model_type': 'gaussian_copula'}
            
            response = client.post(
                '/generate',
                files=files,
                data={'config': json.dumps(config)}
            )
            
            results.put((request_id, error_type, response.status_code))
        
        # Make concurrent error requests
        threads = []
        error_types = ['invalid_file', 'invalid_config', 'empty_file']
        
        for i in range(15):
            error_type = error_types[i % len(error_types)]
            t = threading.Thread(target=make_error_request, args=(i, error_type))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # All should complete with errors
        assert results.qsize() == 15
        
        while not results.empty():
            request_id, error_type, status_code = results.get()
            assert status_code >= 400  # All should be errors
    
    def test_error_recovery(self, client):
        """Test system recovery after errors"""
        # First, trigger an error
        files = {'file': ('bad.csv', b'malformed,csv\ndata', 'text/csv')}
        config = {'generator_type': 'sdv', 'model_type': 'gaussian_copula'}
        
        error_response = client.post(
            '/generate',
            files=files,
            data={'config': json.dumps(config)}
        )
        assert error_response.status_code >= 400
        
        # Now make a valid request
        valid_csv = "age,income\n25,50000\n30,60000"
        files = {'file': ('good.csv', valid_csv.encode(), 'text/csv')}
        
        success_response = client.post(
            '/generate',
            files=files,
            data={'config': json.dumps(config)}
        )
        
        # Should work normally after error
        # May fail due to missing dependencies, but shouldn't be affected by previous error
        assert success_response.status_code != error_response.status_code
    
    def test_graceful_degradation(self, client):
        """Test graceful degradation when optional features fail"""
        # Valid data
        csv_data = "age,income\n25,50000\n30,60000"
        files = {'file': ('data.csv', csv_data.encode(), 'text/csv')}
        
        # Config that might trigger optional feature failures
        config = {
            'generator_type': 'sdv',
            'model_type': 'gaussian_copula',
            'num_samples': 50,
            'validation_settings': {
                'enable_advanced_metrics': True,  # Might fail if dependencies missing
                'calculate_privacy_metrics': True
            }
        }
        
        response = client.post(
            '/generate',
            files=files,
            data={'config': json.dumps(config)}
        )
        
        # Should either succeed or fail gracefully
        if response.status_code == 200:
            result = response.json()
            # Check if it degraded gracefully (skipped some metrics)
            if 'validation_metrics' in result:
                metrics = result['validation_metrics']
                # Some metrics might be missing due to graceful degradation
                assert isinstance(metrics, dict)
    
    def test_error_message_quality(self, client):
        """Test quality and usefulness of error messages"""
        test_cases = [
            {
                'files': {'file': ('data.txt', b'not csv data', 'text/plain')},
                'config': {'generator_type': 'sdv', 'model_type': 'gaussian_copula'},
                'expected_keywords': ['file', 'format', 'csv']
            },
            {
                'files': {'file': ('data.csv', b'age,income\n25,50000', 'text/csv')},
                'config': {'generator_type': 'nonexistent', 'model_type': 'model'},
                'expected_keywords': ['generator', 'type', 'valid']
            },
            {
                'files': {'file': ('data.csv', b'', 'text/csv')},
                'config': {'generator_type': 'sdv', 'model_type': 'gaussian_copula'},
                'expected_keywords': ['empty', 'file', 'data']
            }
        ]
        
        for test_case in test_cases:
            response = client.post(
                '/generate',
                files=test_case['files'],
                data={'config': json.dumps(test_case['config'])}
            )
            
            assert response.status_code >= 400
            error_data = response.json()
            
            # Error message should be present
            error_msg = str(error_data.get('detail', error_data.get('error', ''))).lower()
            assert error_msg  # Should have error message
            
            # Should contain relevant keywords
            found_keyword = any(
                keyword in error_msg 
                for keyword in test_case['expected_keywords']
            )
            assert found_keyword, f"Error message '{error_msg}' missing expected keywords"


class TestErrorTracking:
    """Test error tracking and monitoring integration"""
    
    def test_error_tracking_integration(self, client):
        """Test that errors are properly tracked"""
        # Trigger various errors
        error_requests = [
            {
                'files': {'file': ('bad.csv', b'malformed csv', 'text/csv')},
                'config': {'generator_type': 'sdv', 'model_type': 'gaussian_copula'}
            },
            {
                'files': {'file': ('data.csv', b'age\n25\n30', 'text/csv')},
                'config': {'generator_type': 'invalid', 'model_type': 'model'}
            },
            {
                'files': {'file': ('empty.csv', b'', 'text/csv')},
                'config': {'generator_type': 'sdv', 'model_type': 'gaussian_copula'}
            }
        ]
        
        # Make error requests
        for request_data in error_requests:
            response = client.post(
                '/generate',
                files=request_data['files'],
                data={'config': json.dumps(request_data['config'])}
            )
            assert response.status_code >= 400
            
            # Check error response includes tracking ID
            error_data = response.json()
            # Error responses should include some identifier for tracking
            assert any(key in error_data for key in ['error_id', 'request_id', 'trace_id'])
    
    def test_error_patterns(self, client):
        """Test detection of error patterns"""
        # Make multiple similar errors
        for i in range(5):
            files = {'file': (f'bad{i}.csv', b'malformed,csv\ndata', 'text/csv')}
            config = {'generator_type': 'sdv', 'model_type': 'gaussian_copula'}
            
            response = client.post(
                '/generate',
                files=files,
                data={'config': json.dumps(config)}
            )
            assert response.status_code >= 400
        
        # System should still be responsive
        health_response = client.get('/health')
        assert health_response.status_code == 200