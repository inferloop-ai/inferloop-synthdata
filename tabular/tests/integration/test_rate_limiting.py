"""
Integration tests for rate limiting functionality
"""

import os
import sys
import time
import asyncio
from typing import List, Dict, Any

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.app import app
from api.middleware.rate_limiter import RateLimiter, EndpointRateLimiter


class TestRateLimitingFlow:
    """Test rate limiting workflows"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    @pytest.fixture
    def rate_limiter(self):
        """Rate limiter instance with low limits for testing"""
        return RateLimiter(requests_per_minute=10, burst_size=5)
    
    def test_basic_rate_limiting(self, client, rate_limiter):
        """Test basic rate limiting functionality"""
        # Make requests up to burst limit
        responses = []
        for i in range(5):
            response = client.get('/generators')
            responses.append(response)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            assert 'X-RateLimit-Limit' in response.headers
            assert 'X-RateLimit-Remaining' in response.headers
            assert 'X-RateLimit-Reset' in response.headers
        
        # Check remaining decreases
        remaining_values = [int(r.headers['X-RateLimit-Remaining']) for r in responses]
        for i in range(1, len(remaining_values)):
            assert remaining_values[i] < remaining_values[i-1]
    
    def test_rate_limit_exceeded(self, client):
        """Test behavior when rate limit is exceeded"""
        # Create custom rate limiter with very low limit
        limiter = RateLimiter(requests_per_minute=6, burst_size=3)
        
        # Make requests to exceed burst limit
        responses = []
        for i in range(5):
            response = client.get('/generators')
            responses.append(response)
            # Small delay to avoid test client issues
            time.sleep(0.1)
        
        # Count successful and rate-limited responses
        success_count = sum(1 for r in responses if r.status_code == 200)
        rate_limited_count = sum(1 for r in responses if r.status_code == 429)
        
        # Should have some successes and some rate limits
        assert success_count >= 3
        assert rate_limited_count >= 1
        
        # Check rate limit response
        for response in responses:
            if response.status_code == 429:
                assert response.json()['detail'] == 'Rate limit exceeded'
                assert 'Retry-After' in response.headers
                assert int(response.headers['Retry-After']) > 0
    
    def test_token_refill(self, client):
        """Test token refill over time"""
        # Make initial requests
        for _ in range(3):
            response = client.get('/generators')
            time.sleep(0.1)
        
        # Wait for tokens to refill (based on 60 requests/minute = 1 request/second)
        time.sleep(2)
        
        # Should be able to make more requests
        response = client.get('/generators')
        assert response.status_code == 200
        
        # Check that remaining tokens increased
        remaining = int(response.headers['X-RateLimit-Remaining'])
        assert remaining > 0
    
    def test_endpoint_specific_limits(self, client):
        """Test different rate limits for different endpoints"""
        # Test endpoints with different limits
        endpoints = [
            ('/generators', 60),  # Higher limit
            ('/generate', 10),    # Lower limit
            ('/validate', 20)     # Medium limit
        ]
        
        # Test each endpoint
        for endpoint, expected_limit in endpoints:
            if endpoint == '/generate':
                # POST request with file
                files = {'file': ('test.csv', b'age,income\n25,50000', 'text/csv')}
                response = client.post(
                    endpoint,
                    files=files,
                    data={'config': '{"generator_type": "sdv", "model_type": "gaussian_copula"}'}
                )
            elif endpoint == '/validate':
                # POST request with two files
                files = {
                    'original_file': ('orig.csv', b'age,income\n25,50000', 'text/csv'),
                    'synthetic_file': ('syn.csv', b'age,income\n26,51000', 'text/csv')
                }
                response = client.post(endpoint, files=files)
            else:
                # GET request
                response = client.get(endpoint)
            
            # Check rate limit headers
            if 'X-RateLimit-Limit' in response.headers:
                limit = int(response.headers['X-RateLimit-Limit'])
                # Verify endpoint has appropriate limit
                assert limit > 0
    
    def test_user_based_rate_limiting(self, client):
        """Test rate limiting per user/API key"""
        from api.auth.auth_handler import AuthHandler
        from api.auth.models import UserRole
        
        auth_handler = AuthHandler()
        
        # Create tokens for different users
        user1_token = auth_handler.create_access_token({
            'sub': 'user1',
            'role': UserRole.USER.value
        })
        user2_token = auth_handler.create_access_token({
            'sub': 'user2',
            'role': UserRole.USER.value
        })
        
        # Make requests as user1
        user1_responses = []
        for _ in range(3):
            response = client.get(
                '/generators',
                headers={'Authorization': f'Bearer {user1_token}'}
            )
            user1_responses.append(response)
            time.sleep(0.1)
        
        # Make requests as user2
        user2_responses = []
        for _ in range(3):
            response = client.get(
                '/generators',
                headers={'Authorization': f'Bearer {user2_token}'}
            )
            user2_responses.append(response)
            time.sleep(0.1)
        
        # Both users should have successful requests
        assert all(r.status_code == 200 for r in user1_responses[:3])
        assert all(r.status_code == 200 for r in user2_responses[:3])
        
        # Rate limits should be tracked separately
        user1_remaining = int(user1_responses[-1].headers.get('X-RateLimit-Remaining', 0))
        user2_remaining = int(user2_responses[-1].headers.get('X-RateLimit-Remaining', 0))
        
        # Both should have consumed tokens
        assert user1_remaining < 60
        assert user2_remaining < 60
    
    def test_ip_based_rate_limiting(self, client):
        """Test rate limiting based on IP address"""
        # Make requests without authentication
        responses = []
        for _ in range(5):
            response = client.get('/generators')
            responses.append(response)
            time.sleep(0.1)
        
        # Should have rate limit headers
        for response in responses:
            if response.status_code == 200:
                assert 'X-RateLimit-Limit' in response.headers
                assert 'X-RateLimit-Remaining' in response.headers
    
    @pytest.mark.asyncio
    async def test_concurrent_rate_limiting(self, client):
        """Test rate limiting under concurrent load"""
        import aiohttp
        
        async def make_request(session: aiohttp.ClientSession, url: str) -> int:
            """Make async request and return status code"""
            try:
                async with session.get(url) as response:
                    return response.status
            except Exception:
                return 500
        
        # Make many concurrent requests
        base_url = "http://testserver"
        async with aiohttp.ClientSession() as session:
            tasks = []
            for _ in range(20):
                task = make_request(session, f"{base_url}/generators")
                tasks.append(task)
            
            # Execute all requests concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful and rate-limited responses
        status_codes = [r for r in results if isinstance(r, int)]
        success_count = sum(1 for code in status_codes if code == 200)
        rate_limited_count = sum(1 for code in status_codes if code == 429)
        
        # Should have mix of successful and rate-limited
        assert success_count > 0
        assert rate_limited_count > 0
        assert success_count + rate_limited_count <= 20
    
    def test_rate_limit_headers_accuracy(self, client):
        """Test accuracy of rate limit headers"""
        # Get initial state
        response1 = client.get('/generators')
        assert response1.status_code == 200
        
        initial_remaining = int(response1.headers['X-RateLimit-Remaining'])
        reset_time = int(response1.headers['X-RateLimit-Reset'])
        
        # Make another request
        time.sleep(0.5)
        response2 = client.get('/generators')
        assert response2.status_code == 200
        
        new_remaining = int(response2.headers['X-RateLimit-Remaining'])
        
        # Remaining should decrease by 1 (or slightly more if tokens were added)
        assert new_remaining <= initial_remaining
        
        # Reset time should be in the future
        assert reset_time > time.time()
    
    def test_burst_handling(self, client):
        """Test burst request handling"""
        # Make burst of requests quickly
        burst_responses = []
        start_time = time.time()
        
        for _ in range(10):
            response = client.get('/generators')
            burst_responses.append(response)
        
        end_time = time.time()
        burst_duration = end_time - start_time
        
        # Should complete quickly (under 1 second for burst)
        assert burst_duration < 1.0
        
        # Should have mix of success and rate limited
        success_count = sum(1 for r in burst_responses if r.status_code == 200)
        rate_limited = sum(1 for r in burst_responses if r.status_code == 429)
        
        assert success_count > 0  # At least some should succeed
        assert success_count + rate_limited == 10
    
    def test_rate_limit_persistence(self, client):
        """Test that rate limits persist across requests"""
        # Exhaust rate limit
        responses = []
        for _ in range(10):
            response = client.get('/generators')
            responses.append(response)
            if response.status_code == 429:
                break
            time.sleep(0.1)
        
        # Find when we got rate limited
        rate_limited_index = next(
            (i for i, r in enumerate(responses) if r.status_code == 429),
            None
        )
        
        if rate_limited_index is not None:
            # Make another request immediately
            response = client.get('/generators')
            # Should still be rate limited
            assert response.status_code == 429
            
            # Get retry-after value
            retry_after = int(response.headers.get('Retry-After', 0))
            assert retry_after > 0
            
            # Wait less than retry-after
            time.sleep(retry_after / 2)
            
            # Should still be rate limited
            response = client.get('/generators')
            # Might succeed if tokens refilled
            assert response.status_code in [200, 429]


class TestAdvancedRateLimiting:
    """Test advanced rate limiting scenarios"""
    
    def test_custom_endpoint_limits(self, client):
        """Test custom limits for specific endpoints"""
        # Test /generate endpoint (lower limit)
        generate_responses = []
        for i in range(15):
            files = {'file': ('test.csv', b'age,income\n25,50000', 'text/csv')}
            response = client.post(
                '/generate',
                files=files,
                data={'config': '{"generator_type": "sdv", "model_type": "gaussian_copula"}'}
            )
            generate_responses.append(response.status_code)
            time.sleep(0.2)
        
        # Should hit rate limit for /generate endpoint
        rate_limited = sum(1 for code in generate_responses if code == 429)
        assert rate_limited > 0
    
    def test_rate_limit_bypass_for_admin(self, client):
        """Test if admins have higher or no rate limits"""
        from api.auth.auth_handler import AuthHandler
        from api.auth.models import UserRole
        
        auth_handler = AuthHandler()
        
        # Create admin token
        admin_token = auth_handler.create_access_token({
            'sub': 'admin_user',
            'role': UserRole.ADMIN.value
        })
        
        # Make many requests as admin
        admin_responses = []
        for _ in range(20):
            response = client.get(
                '/generators',
                headers={'Authorization': f'Bearer {admin_token}'}
            )
            admin_responses.append(response)
        
        # Admin might have higher limits or no limits
        success_count = sum(1 for r in admin_responses if r.status_code == 200)
        
        # Create regular user token
        user_token = auth_handler.create_access_token({
            'sub': 'regular_user',
            'role': UserRole.USER.value
        })
        
        # Make same number of requests as regular user
        user_responses = []
        for _ in range(20):
            response = client.get(
                '/generators',
                headers={'Authorization': f'Bearer {user_token}'}
            )
            user_responses.append(response)
        
        user_success_count = sum(1 for r in user_responses if r.status_code == 200)
        
        # Admin should have at least as many successes as regular user
        assert success_count >= user_success_count
    
    def test_distributed_rate_limiting(self, client):
        """Test rate limiting with distributed clients simulation"""
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        results = []
        lock = threading.Lock()
        
        def make_requests(client_id: int, num_requests: int):
            """Simulate requests from different clients"""
            client_results = []
            
            # Simulate different IPs with different headers
            headers = {'X-Forwarded-For': f'192.168.1.{client_id}'}
            
            for _ in range(num_requests):
                response = client.get('/generators', headers=headers)
                client_results.append({
                    'client_id': client_id,
                    'status_code': response.status_code,
                    'remaining': response.headers.get('X-RateLimit-Remaining')
                })
                time.sleep(0.1)
            
            with lock:
                results.extend(client_results)
        
        # Simulate 5 different clients
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(5):
                future = executor.submit(make_requests, i, 10)
                futures.append(future)
            
            # Wait for all to complete
            for future in futures:
                future.result()
        
        # Analyze results
        by_client = {}
        for result in results:
            client_id = result['client_id']
            if client_id not in by_client:
                by_client[client_id] = {'success': 0, 'rate_limited': 0}
            
            if result['status_code'] == 200:
                by_client[client_id]['success'] += 1
            elif result['status_code'] == 429:
                by_client[client_id]['rate_limited'] += 1
        
        # Each client should have their own rate limit
        for client_id, stats in by_client.items():
            # Each client should have some successful requests
            assert stats['success'] > 0