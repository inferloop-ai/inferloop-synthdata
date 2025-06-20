"""
Unit tests for REST API implementation.

Tests the RESTful API endpoints for document generation, export, and management.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path

from structured_docs_synth.delivery.api.rest_api import (
    RestAPI,
    APIConfig,
    RequestHandler,
    ResponseBuilder,
    APIError,
    RateLimiter,
    AuthenticationMiddleware
)


class TestAPIConfig:
    """Test API configuration."""
    
    def test_default_config(self):
        """Test default API configuration."""
        config = APIConfig()
        
        assert config.host == '0.0.0.0'
        assert config.port == 8080
        assert config.base_path == '/api/v1'
        assert config.enable_cors is True
        assert config.enable_auth is True
        assert config.rate_limit == 100
        assert config.request_timeout == 30
        assert config.max_request_size == 10 * 1024 * 1024  # 10MB
    
    def test_custom_config(self):
        """Test custom API configuration."""
        config = APIConfig(
            host='localhost',
            port=3000,
            enable_auth=False,
            rate_limit=50
        )
        
        assert config.host == 'localhost'
        assert config.port == 3000
        assert config.enable_auth is False
        assert config.rate_limit == 50


class TestResponseBuilder:
    """Test API response builder."""
    
    def test_success_response(self):
        """Test building success response."""
        data = {'document_id': '123', 'status': 'completed'}
        response = ResponseBuilder.success(data, message="Document generated")
        
        assert response['success'] is True
        assert response['data'] == data
        assert response['message'] == "Document generated"
        assert 'timestamp' in response
        assert response['error'] is None
    
    def test_error_response(self):
        """Test building error response."""
        response = ResponseBuilder.error(
            message="Invalid request",
            code="INVALID_REQUEST",
            status_code=400
        )
        
        assert response['success'] is False
        assert response['message'] == "Invalid request"
        assert response['error']['code'] == "INVALID_REQUEST"
        assert response['error']['status_code'] == 400
        assert response['data'] is None
    
    def test_paginated_response(self):
        """Test building paginated response."""
        items = [{'id': i} for i in range(25)]
        response = ResponseBuilder.paginated(
            items=items[:10],
            page=1,
            per_page=10,
            total=25
        )
        
        assert response['success'] is True
        assert len(response['data']['items']) == 10
        assert response['data']['pagination']['page'] == 1
        assert response['data']['pagination']['per_page'] == 10
        assert response['data']['pagination']['total'] == 25
        assert response['data']['pagination']['pages'] == 3


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_rate_limit_allows_requests(self):
        """Test rate limiter allows requests within limit."""
        limiter = RateLimiter(limit=5, window=60)
        client_id = "test-client"
        
        for _ in range(5):
            assert limiter.check_limit(client_id) is True
    
    def test_rate_limit_blocks_excess_requests(self):
        """Test rate limiter blocks requests over limit."""
        limiter = RateLimiter(limit=5, window=60)
        client_id = "test-client"
        
        # Use up the limit
        for _ in range(5):
            limiter.check_limit(client_id)
        
        # Next request should be blocked
        assert limiter.check_limit(client_id) is False
    
    def test_rate_limit_reset(self):
        """Test rate limit reset after window."""
        limiter = RateLimiter(limit=5, window=1)  # 1 second window
        client_id = "test-client"
        
        # Use up the limit
        for _ in range(5):
            limiter.check_limit(client_id)
        
        # Should be blocked
        assert limiter.check_limit(client_id) is False
        
        # Wait for window to pass
        import time
        time.sleep(1.1)
        
        # Should be allowed again
        assert limiter.check_limit(client_id) is True


class TestAuthenticationMiddleware:
    """Test authentication middleware."""
    
    def test_valid_api_key(self):
        """Test authentication with valid API key."""
        auth = AuthenticationMiddleware(api_keys=['valid-key-123'])
        
        headers = {'Authorization': 'Bearer valid-key-123'}
        assert auth.authenticate(headers) is True
    
    def test_invalid_api_key(self):
        """Test authentication with invalid API key."""
        auth = AuthenticationMiddleware(api_keys=['valid-key-123'])
        
        headers = {'Authorization': 'Bearer invalid-key'}
        assert auth.authenticate(headers) is False
    
    def test_missing_auth_header(self):
        """Test authentication with missing header."""
        auth = AuthenticationMiddleware(api_keys=['valid-key-123'])
        
        headers = {}
        assert auth.authenticate(headers) is False
    
    def test_jwt_validation(self):
        """Test JWT token validation."""
        auth = AuthenticationMiddleware(jwt_secret='secret-key')
        
        # Mock JWT validation
        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {'user_id': '123', 'exp': 9999999999}
            
            headers = {'Authorization': 'Bearer jwt-token'}
            assert auth.authenticate(headers) is True
            
            mock_decode.assert_called_once()


class TestRequestHandler:
    """Test request handling."""
    
    @pytest.fixture
    def handler(self):
        """Provide request handler instance."""
        return RequestHandler()
    
    def test_parse_json_request(self, handler):
        """Test parsing JSON request."""
        request_data = '{"document_type": "invoice", "count": 5}'
        
        parsed = handler.parse_request(request_data, content_type='application/json')
        
        assert parsed['document_type'] == 'invoice'
        assert parsed['count'] == 5
    
    def test_parse_invalid_json(self, handler):
        """Test parsing invalid JSON."""
        request_data = 'invalid json'
        
        with pytest.raises(APIError, match="Invalid JSON"):
            handler.parse_request(request_data, content_type='application/json')
    
    def test_validate_required_fields(self, handler):
        """Test required field validation."""
        data = {'document_type': 'invoice'}
        required = ['document_type', 'count']
        
        with pytest.raises(APIError, match="Missing required field: count"):
            handler.validate_request(data, required_fields=required)
    
    def test_validate_field_types(self, handler):
        """Test field type validation."""
        data = {'count': 'five'}  # Should be int
        
        with pytest.raises(APIError, match="Invalid type for field 'count'"):
            handler.validate_request(
                data,
                field_types={'count': int}
            )


class TestRestAPI:
    """Test REST API implementation."""
    
    @pytest.fixture
    def api_config(self):
        """Provide API configuration."""
        return APIConfig(
            host='localhost',
            port=8080,
            enable_auth=False,
            rate_limit=100
        )
    
    @pytest.fixture
    def mock_generator(self):
        """Provide mock document generator."""
        mock = AsyncMock()
        mock.generate = AsyncMock(return_value={
            'document_id': 'doc-123',
            'status': 'completed',
            'output_path': '/tmp/doc.pdf'
        })
        return mock
    
    @pytest.fixture
    def mock_exporter(self):
        """Provide mock exporter."""
        mock = AsyncMock()
        mock.export = AsyncMock(return_value={
            'job_id': 'job-123',
            'status': 'running'
        })
        return mock
    
    @pytest.fixture
    def rest_api(self, api_config, mock_generator, mock_exporter):
        """Provide REST API instance."""
        api = RestAPI(api_config)
        api.generator = mock_generator
        api.exporter = mock_exporter
        return api
    
    @pytest.mark.asyncio
    async def test_generate_document_endpoint(self, rest_api):
        """Test document generation endpoint."""
        request_data = {
            'document_type': 'invoice',
            'format': 'pdf',
            'metadata': {'project': 'test'}
        }
        
        response = await rest_api.generate_document(request_data)
        
        assert response['success'] is True
        assert response['data']['document_id'] == 'doc-123'
        assert response['data']['status'] == 'completed'
        
        # Verify generator was called
        rest_api.generator.generate.assert_called_once_with(
            document_type='invoice',
            format='pdf',
            metadata={'project': 'test'}
        )
    
    @pytest.mark.asyncio
    async def test_batch_generate_endpoint(self, rest_api):
        """Test batch generation endpoint."""
        request_data = {
            'document_type': 'invoice',
            'count': 10,
            'format': 'pdf'
        }
        
        # Mock batch results
        rest_api.generator.generate = AsyncMock(side_effect=[
            {'document_id': f'doc-{i}', 'status': 'completed'}
            for i in range(10)
        ])
        
        response = await rest_api.batch_generate(request_data)
        
        assert response['success'] is True
        assert response['data']['total_requested'] == 10
        assert response['data']['total_generated'] == 10
        assert len(response['data']['documents']) == 10
    
    @pytest.mark.asyncio
    async def test_export_documents_endpoint(self, rest_api):
        """Test document export endpoint."""
        request_data = {
            'document_ids': ['doc-1', 'doc-2', 'doc-3'],
            'format': 'zip',
            'include_metadata': True
        }
        
        response = await rest_api.export_documents(request_data)
        
        assert response['success'] is True
        assert response['data']['job_id'] == 'job-123'
        assert response['data']['status'] == 'running'
    
    @pytest.mark.asyncio
    async def test_get_document_status_endpoint(self, rest_api):
        """Test document status endpoint."""
        document_id = 'doc-123'
        
        # Mock status response
        rest_api.generator.get_status = AsyncMock(return_value={
            'document_id': document_id,
            'status': 'completed',
            'created_at': datetime.now().isoformat(),
            'output_path': '/tmp/doc.pdf'
        })
        
        response = await rest_api.get_document_status(document_id)
        
        assert response['success'] is True
        assert response['data']['document_id'] == document_id
        assert response['data']['status'] == 'completed'
    
    @pytest.mark.asyncio
    async def test_list_documents_endpoint(self, rest_api):
        """Test list documents endpoint."""
        # Mock document list
        rest_api.generator.list_documents = AsyncMock(return_value={
            'documents': [
                {'document_id': f'doc-{i}', 'created_at': datetime.now().isoformat()}
                for i in range(25)
            ],
            'total': 25
        })
        
        response = await rest_api.list_documents(page=1, per_page=10)
        
        assert response['success'] is True
        assert len(response['data']['items']) == 10
        assert response['data']['pagination']['total'] == 25
        assert response['data']['pagination']['pages'] == 3
    
    @pytest.mark.asyncio
    async def test_error_handling(self, rest_api):
        """Test API error handling."""
        # Mock generator error
        rest_api.generator.generate = AsyncMock(
            side_effect=Exception("Generation failed")
        )
        
        request_data = {'document_type': 'invoice'}
        
        response = await rest_api.generate_document(request_data)
        
        assert response['success'] is False
        assert 'Generation failed' in response['message']
        assert response['error']['code'] == 'GENERATION_ERROR'
    
    @pytest.mark.asyncio
    async def test_request_validation(self, rest_api):
        """Test request validation."""
        # Missing required field
        request_data = {'format': 'pdf'}  # Missing document_type
        
        response = await rest_api.generate_document(request_data)
        
        assert response['success'] is False
        assert 'Missing required field' in response['message']
        assert response['error']['code'] == 'VALIDATION_ERROR'
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, rest_api):
        """Test rate limiting."""
        rest_api.config.rate_limit = 5
        rest_api.rate_limiter = RateLimiter(limit=5, window=60)
        
        client_id = 'test-client'
        request_data = {'document_type': 'invoice'}
        
        # Make requests up to limit
        for _ in range(5):
            response = await rest_api.generate_document(
                request_data,
                client_id=client_id
            )
            assert response['success'] is True
        
        # Next request should be rate limited
        response = await rest_api.generate_document(
            request_data,
            client_id=client_id
        )
        
        assert response['success'] is False
        assert response['error']['code'] == 'RATE_LIMIT_EXCEEDED'
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self, rest_api):
        """Test health check endpoint."""
        response = await rest_api.health_check()
        
        assert response['success'] is True
        assert response['data']['status'] == 'healthy'
        assert 'version' in response['data']
        assert 'uptime' in response['data']
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, rest_api):
        """Test metrics endpoint."""
        # Mock metrics
        rest_api.metrics = {
            'requests_total': 1000,
            'requests_success': 950,
            'requests_failed': 50,
            'average_response_time': 0.150
        }
        
        response = await rest_api.get_metrics()
        
        assert response['success'] is True
        assert response['data']['requests_total'] == 1000
        assert response['data']['success_rate'] == 0.95
        assert response['data']['average_response_time'] == 0.150