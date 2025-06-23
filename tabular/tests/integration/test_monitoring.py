"""
Integration tests for monitoring and observability
"""

import os
import sys
import time
import json
from typing import Dict, Any, List

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.app import app
from api.middleware.logging_middleware import request_logger, performance_monitor
from api.middleware.error_tracker import error_tracker


class TestMonitoringIntegration:
    """Test monitoring and observability features"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    def test_request_logging(self, client):
        """Test comprehensive request logging"""
        # Make various types of requests
        requests_to_test = [
            ('GET', '/health', None, None),
            ('GET', '/generators', None, None),
            ('POST', '/generate', 
             {'file': ('test.csv', b'age,income\n25,50000', 'text/csv')},
             {'config': json.dumps({'generator_type': 'sdv', 'model_type': 'gaussian_copula'})}),
        ]
        
        initial_count = request_logger.metrics['request_count']
        
        for method, path, files, data in requests_to_test:
            if method == 'GET':
                response = client.get(path)
            else:
                response = client.post(path, files=files, data=data)
            
            # Check request ID header
            assert 'X-Request-ID' in response.headers
            request_id = response.headers['X-Request-ID']
            assert len(request_id) == 36  # UUID format
        
        # Verify metrics were updated
        assert request_logger.metrics['request_count'] > initial_count
        
        # Check metrics summary
        summary = request_logger.get_metrics_summary()
        assert summary['total_requests'] > 0
        assert 'average_duration_ms' in summary
        assert 'status_code_distribution' in summary
        assert 'endpoint_stats' in summary
    
    def test_performance_monitoring(self, client):
        """Test performance monitoring capabilities"""
        # Make requests to different endpoints
        endpoints = ['/health', '/generators', '/generators/sdv']
        
        for endpoint in endpoints:
            # Track performance
            with performance_monitor.track_endpoint(endpoint):
                response = client.get(endpoint)
        
        # Get performance report
        report = performance_monitor.get_performance_report()
        
        # Check report structure
        assert 'endpoint_performance' in report
        assert 'slow_requests' in report
        assert 'database_performance' in report
        assert 'cache_performance' in report
        
        # Verify endpoint metrics
        for endpoint in endpoints:
            if endpoint in report['endpoint_performance']:
                metrics = report['endpoint_performance'][endpoint]
                assert 'count' in metrics
                assert 'avg_latency_ms' in metrics
                assert 'min_latency_ms' in metrics
                assert 'max_latency_ms' in metrics
                assert metrics['avg_latency_ms'] > 0
    
    def test_error_tracking(self, client):
        """Test error tracking and reporting"""
        # Trigger various errors
        error_scenarios = [
            ('POST', '/generate', {'file': ('bad.csv', b'invalid', 'text/csv')}, 
             {'config': json.dumps({'generator_type': 'sdv', 'model_type': 'gaussian_copula'})}),
            ('GET', '/generators/nonexistent', None, None),
            ('POST', '/generate', {'file': ('test.csv', b'age\n25', 'text/csv')},
             {'config': 'invalid json'}),
        ]
        
        initial_error_count = error_tracker.error_stats['total_errors']
        
        for method, path, files, data in error_scenarios:
            if method == 'GET':
                response = client.get(path)
            else:
                response = client.post(path, files=files, data=data)
            
            # Should be error responses
            assert response.status_code >= 400
        
        # Get error summary
        error_summary = error_tracker.get_error_summary()
        
        # Verify error tracking
        assert error_summary['total_errors'] > initial_error_count
        assert 'error_rate_per_minute' in error_summary
        assert 'error_types' in error_summary
        assert 'error_endpoints' in error_summary
        assert 'recent_errors' in error_summary
        
        # Check recent errors structure
        if error_summary['recent_errors']:
            recent_error = error_summary['recent_errors'][0]
            assert 'error_id' in recent_error
            assert 'timestamp' in recent_error
            assert 'error_type' in recent_error
            assert 'endpoint' in recent_error
    
    def test_request_tracing(self, client):
        """Test request tracing through the system"""
        # Make a request and track its journey
        response = client.get('/generators')
        
        assert response.status_code == 200
        request_id = response.headers.get('X-Request-ID')
        assert request_id
        
        # Make request that will fail
        files = {'file': ('test.csv', b'invalid csv data', 'text/csv')}
        config = {'generator_type': 'sdv', 'model_type': 'gaussian_copula'}
        
        error_response = client.post(
            '/generate',
            files=files,
            data={'config': json.dumps(config)}
        )
        
        error_request_id = error_response.headers.get('X-Request-ID')
        assert error_request_id
        assert error_request_id != request_id  # Different requests
    
    def test_metrics_aggregation(self, client):
        """Test metrics aggregation over time"""
        # Make requests over time
        for i in range(5):
            client.get('/health')
            client.get('/generators')
            time.sleep(0.2)
        
        # Check aggregated metrics
        metrics = request_logger.get_metrics_summary()
        
        # Should have meaningful aggregations
        assert metrics['total_requests'] >= 10
        assert metrics['average_duration_ms'] > 0
        
        # Check endpoint-specific stats
        endpoint_stats = metrics['endpoint_stats']
        if 'GET /health' in endpoint_stats:
            health_stats = endpoint_stats['GET /health']
            assert health_stats['count'] >= 5
            assert health_stats['total_duration'] > 0
            assert health_stats['errors'] == 0
    
    def test_monitoring_under_load(self, client):
        """Test monitoring system under load"""
        import concurrent.futures
        
        def make_request(i):
            """Make a single request"""
            if i % 3 == 0:
                # GET request
                return client.get('/generators')
            elif i % 3 == 1:
                # Valid POST request
                files = {'file': ('test.csv', b'age,income\n25,50000\n30,60000', 'text/csv')}
                config = {'generator_type': 'sdv', 'model_type': 'gaussian_copula', 'num_samples': 10}
                return client.post('/generate', files=files, data={'config': json.dumps(config)})
            else:
                # Error request
                return client.get('/nonexistent/endpoint')
        
        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(30)]
            responses = [f.result() for f in futures]
        
        # Check monitoring still works
        metrics = request_logger.get_metrics_summary()
        assert metrics['total_requests'] >= 30
        
        # Should have mix of status codes
        status_distribution = metrics['status_code_distribution']
        assert len(status_distribution) > 1  # Multiple status codes
        
        # Performance should still be tracked
        perf_report = performance_monitor.get_performance_report()
        assert len(perf_report['endpoint_performance']) > 0
    
    def test_monitoring_persistence(self, client):
        """Test that monitoring data persists across requests"""
        # Get initial state
        initial_metrics = request_logger.get_metrics_summary()
        initial_total = initial_metrics['total_requests']
        
        # Make some requests
        for _ in range(5):
            client.get('/health')
        
        # Check metrics increased
        new_metrics = request_logger.get_metrics_summary()
        assert new_metrics['total_requests'] == initial_total + 5
        
        # Make error requests
        for _ in range(3):
            client.get('/nonexistent')
        
        # Check error tracking
        final_metrics = request_logger.get_metrics_summary()
        assert final_metrics['total_requests'] == initial_total + 8
        
        # Status code distribution should include 404s
        assert 404 in final_metrics['status_code_distribution']
    
    def test_custom_metrics(self, client):
        """Test custom application metrics"""
        # Test cache hit/miss tracking
        initial_hits = performance_monitor.metrics['cache_hits']
        initial_misses = performance_monitor.metrics['cache_misses']
        
        # Simulate cache operations
        performance_monitor.record_cache_hit()
        performance_monitor.record_cache_hit()
        performance_monitor.record_cache_miss()
        
        # Verify tracking
        assert performance_monitor.metrics['cache_hits'] == initial_hits + 2
        assert performance_monitor.metrics['cache_misses'] == initial_misses + 1
        
        # Check cache performance report
        report = performance_monitor.get_performance_report()
        cache_perf = report['cache_performance']
        assert cache_perf['hits'] == initial_hits + 2
        assert cache_perf['misses'] == initial_misses + 1
        if cache_perf['hits'] + cache_perf['misses'] > 0:
            assert 0 <= cache_perf['hit_rate'] <= 1
    
    def test_monitoring_edge_cases(self, client):
        """Test monitoring in edge cases"""
        # Very large request
        large_data = 'x' * 10000
        files = {'file': ('large.csv', large_data.encode(), 'text/csv')}
        response = client.post(
            '/generate',
            files=files,
            data={'config': json.dumps({'generator_type': 'sdv', 'model_type': 'gaussian_copula'})}
        )
        
        # Should still track
        metrics = request_logger.get_metrics_summary()
        assert metrics['total_requests'] > 0
        
        # Request with special characters
        special_path = '/generators/%3Cscript%3E'
        response = client.get(special_path)
        
        # Should handle safely
        assert response.status_code in [400, 404]
        
        # Empty request
        response = client.post('/generate')
        assert response.status_code >= 400
        
        # All should be tracked
        final_metrics = request_logger.get_metrics_summary()
        assert final_metrics['total_requests'] >= 3


class TestObservability:
    """Test observability and debugging features"""
    
    def test_health_check_details(self, client):
        """Test detailed health check information"""
        response = client.get('/health')
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data['status'] == 'healthy'
        assert 'timestamp' in health_data
        assert 'version' in health_data
        
        # Could include more details like:
        # - Database connectivity
        # - External service status
        # - Resource usage
    
    def test_debug_endpoints(self, client):
        """Test debug endpoints for troubleshooting"""
        from api.auth.auth_handler import AuthHandler
        from api.auth.models import UserRole
        
        # Create admin token for debug access
        auth_handler = AuthHandler()
        admin_token = auth_handler.create_access_token({
            'sub': 'admin',
            'role': UserRole.ADMIN.value
        })
        headers = {'Authorization': f'Bearer {admin_token}'}
        
        # Test metrics endpoint
        response = client.get('/admin/metrics', headers=headers)
        if response.status_code == 200:
            metrics = response.json()
            assert 'request_metrics' in metrics or 'performance_metrics' in metrics
        
        # Test error summary endpoint
        response = client.get('/admin/errors', headers=headers)
        if response.status_code == 200:
            errors = response.json()
            assert isinstance(errors, dict)
    
    def test_correlation_ids(self, client):
        """Test request correlation across services"""
        # Make request with correlation ID
        correlation_id = 'test-correlation-123'
        headers = {'X-Correlation-ID': correlation_id}
        
        response = client.get('/generators', headers=headers)
        assert response.status_code == 200
        
        # Should propagate correlation ID
        request_id = response.headers.get('X-Request-ID')
        assert request_id  # Should have request ID
        
        # In real system, would verify correlation ID is logged
        # and propagated to downstream services