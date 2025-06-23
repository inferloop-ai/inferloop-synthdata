"""
Integration tests for authentication and authorization flows
"""

import os
import sys
import time
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.app import app
from api.auth.auth_handler import AuthHandler
from api.auth.models import User, UserRole, APIKey


class TestAuthenticationFlow:
    """Test authentication and authorization workflows"""
    
    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)
    
    @pytest.fixture
    def auth_handler(self):
        """Auth handler instance"""
        return AuthHandler()
    
    @pytest.fixture
    def test_users(self):
        """Test user data"""
        return {
            'admin': {
                'username': 'admin_user',
                'password': 'admin_pass123',
                'role': UserRole.ADMIN
            },
            'regular': {
                'username': 'regular_user',
                'password': 'user_pass123',
                'role': UserRole.USER
            },
            'readonly': {
                'username': 'readonly_user',
                'password': 'readonly_pass123',
                'role': UserRole.READ_ONLY
            }
        }
    
    def test_jwt_authentication_flow(self, client, auth_handler, test_users):
        """Test JWT-based authentication flow"""
        # Test login for each user type
        for user_type, user_data in test_users.items():
            # Create user (in real app, this would be in database)
            user = User(
                id=f"{user_type}_id",
                username=user_data['username'],
                email=f"{user_data['username']}@example.com",
                role=user_data['role'],
                hashed_password=auth_handler.password_hasher.hash(user_data['password'])
            )
            
            # Mock authenticate_user to return our test user
            original_authenticate = auth_handler.authenticate_user
            auth_handler.authenticate_user = lambda u, p: user if u == user_data['username'] and p == user_data['password'] else None
            
            # Test login
            response = client.post("/auth/login", data={
                'username': user_data['username'],
                'password': user_data['password']
            })
            
            assert response.status_code == 200
            token_data = response.json()
            assert 'access_token' in token_data
            assert 'token_type' in token_data
            assert token_data['token_type'] == 'bearer'
            
            # Decode and verify token
            decoded = jwt.decode(
                token_data['access_token'],
                auth_handler.secret_key,
                algorithms=[auth_handler.algorithm]
            )
            assert decoded['sub'] == user_data['username']
            assert decoded['role'] == user_data['role'].value
            
            # Test accessing protected endpoint with token
            headers = {'Authorization': f"Bearer {token_data['access_token']}"}
            response = client.get("/auth/me", headers=headers)
            
            # Should work for all authenticated users
            assert response.status_code == 200
            user_info = response.json()
            assert user_info['username'] == user_data['username']
            assert user_info['role'] == user_data['role'].value
            
            # Restore original authenticate method
            auth_handler.authenticate_user = original_authenticate
    
    def test_api_key_authentication_flow(self, client, auth_handler):
        """Test API key authentication flow"""
        # Create test API keys
        api_keys = [
            APIKey(
                id='key1',
                key='test_api_key_1234567890',
                name='Test API Key',
                user_id='test_user',
                permissions=['read', 'write'],
                rate_limit=100
            ),
            APIKey(
                id='key2',
                key='readonly_api_key_0987654321',
                name='Read-only API Key',
                user_id='test_user',
                permissions=['read'],
                rate_limit=50
            )
        ]
        
        for api_key in api_keys:
            # Mock validate_api_key
            original_validate = auth_handler.validate_api_key
            auth_handler.validate_api_key = lambda k: api_key if k == api_key.key else None
            
            # Test with API key in header
            headers = {'X-API-Key': api_key.key}
            response = client.get("/generators", headers=headers)
            
            # Should work for valid API keys
            assert response.status_code == 200
            
            # Test with invalid API key
            headers = {'X-API-Key': 'invalid_key'}
            response = client.get("/generators", headers=headers)
            assert response.status_code == 401
            
            # Restore original validate method
            auth_handler.validate_api_key = original_validate
    
    def test_role_based_access_control(self, client, auth_handler):
        """Test role-based access control"""
        # Create tokens for different roles
        tokens = {}
        for role in UserRole:
            token_data = {
                'sub': f'{role.value}_user',
                'role': role.value,
                'exp': datetime.utcnow() + timedelta(hours=1)
            }
            tokens[role] = auth_handler.create_access_token(token_data)
        
        # Define role-based endpoints
        role_endpoints = {
            '/admin/users': [UserRole.ADMIN],  # Admin only
            '/generate': [UserRole.ADMIN, UserRole.USER],  # Admin and User
            '/generators': [UserRole.ADMIN, UserRole.USER, UserRole.READ_ONLY]  # All roles
        }
        
        # Test access for each role
        for endpoint, allowed_roles in role_endpoints.items():
            for role, token in tokens.items():
                headers = {'Authorization': f'Bearer {token}'}
                
                # Make request based on endpoint
                if endpoint == '/generate':
                    # POST request with file
                    files = {'file': ('test.csv', b'age,income\n25,50000', 'text/csv')}
                    response = client.post(
                        endpoint,
                        headers=headers,
                        files=files,
                        data={'config': '{"generator_type": "sdv", "model_type": "gaussian_copula"}'}
                    )
                else:
                    # GET request
                    response = client.get(endpoint, headers=headers)
                
                # Check if access is allowed
                if role in allowed_roles:
                    # Should not get 403 Forbidden
                    assert response.status_code != 403
                else:
                    # Should get 403 Forbidden
                    assert response.status_code == 403
    
    def test_token_expiration_flow(self, client, auth_handler):
        """Test token expiration handling"""
        # Create token with short expiration
        token_data = {
            'sub': 'test_user',
            'role': UserRole.USER.value,
            'exp': datetime.utcnow() + timedelta(seconds=2)
        }
        token = auth_handler.create_access_token(token_data)
        
        # Test immediate access - should work
        headers = {'Authorization': f'Bearer {token}'}
        response = client.get('/generators', headers=headers)
        assert response.status_code == 200
        
        # Wait for token to expire
        time.sleep(3)
        
        # Test after expiration - should fail
        response = client.get('/generators', headers=headers)
        assert response.status_code == 401
        assert 'Token has expired' in response.json()['detail']
    
    def test_multiple_auth_methods(self, client, auth_handler):
        """Test using multiple authentication methods"""
        # Create valid JWT token
        jwt_token = auth_handler.create_access_token({
            'sub': 'jwt_user',
            'role': UserRole.USER.value
        })
        
        # Create valid API key
        api_key = APIKey(
            id='test_key',
            key='multi_auth_test_key',
            name='Multi-auth Test',
            user_id='api_user',
            permissions=['read', 'write']
        )
        
        # Mock validate_api_key
        original_validate = auth_handler.validate_api_key
        auth_handler.validate_api_key = lambda k: api_key if k == api_key.key else None
        
        # Test with both JWT and API key (JWT should take precedence)
        headers = {
            'Authorization': f'Bearer {jwt_token}',
            'X-API-Key': api_key.key
        }
        response = client.get('/generators', headers=headers)
        assert response.status_code == 200
        
        # Test with only API key
        headers = {'X-API-Key': api_key.key}
        response = client.get('/generators', headers=headers)
        assert response.status_code == 200
        
        # Test with invalid JWT but valid API key
        headers = {
            'Authorization': 'Bearer invalid_token',
            'X-API-Key': api_key.key
        }
        response = client.get('/generators', headers=headers)
        # Should fail on JWT validation first
        assert response.status_code == 401
        
        # Restore original validate method
        auth_handler.validate_api_key = original_validate
    
    def test_auth_error_scenarios(self, client, auth_handler):
        """Test various authentication error scenarios"""
        # Test missing credentials
        response = client.get('/auth/me')
        assert response.status_code == 401
        assert 'Not authenticated' in response.json()['detail']
        
        # Test malformed token
        headers = {'Authorization': 'Bearer malformed.token.here'}
        response = client.get('/auth/me', headers=headers)
        assert response.status_code == 401
        
        # Test wrong token type
        headers = {'Authorization': 'Basic dGVzdDp0ZXN0'}  # Basic auth instead of Bearer
        response = client.get('/auth/me', headers=headers)
        assert response.status_code == 401
        
        # Test login with wrong credentials
        response = client.post('/auth/login', data={
            'username': 'wrong_user',
            'password': 'wrong_pass'
        })
        assert response.status_code == 401
        assert 'Incorrect username or password' in response.json()['detail']
    
    def test_concurrent_auth_requests(self, client, auth_handler):
        """Test handling concurrent authentication requests"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_auth_request(token, request_id):
            """Make authenticated request"""
            headers = {'Authorization': f'Bearer {token}'}
            response = client.get('/generators', headers=headers)
            results.put((request_id, response.status_code))
        
        # Create token
        token = auth_handler.create_access_token({
            'sub': 'concurrent_user',
            'role': UserRole.USER.value
        })
        
        # Make multiple concurrent requests
        threads = []
        for i in range(20):
            t = threading.Thread(target=make_auth_request, args=(token, i))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check all requests succeeded
        assert results.qsize() == 20
        while not results.empty():
            request_id, status_code = results.get()
            assert status_code == 200


class TestAPIKeyManagement:
    """Test API key management workflows"""
    
    @pytest.fixture
    def admin_token(self, auth_handler):
        """Create admin token for API key management"""
        return auth_handler.create_access_token({
            'sub': 'admin_user',
            'role': UserRole.ADMIN.value
        })
    
    def test_api_key_lifecycle(self, client, admin_token):
        """Test complete API key lifecycle"""
        headers = {'Authorization': f'Bearer {admin_token}'}
        
        # Create API key
        create_data = {
            'name': 'Test Integration Key',
            'permissions': ['read', 'write'],
            'expires_in_days': 30,
            'rate_limit': 100
        }
        response = client.post('/admin/api-keys', json=create_data, headers=headers)
        assert response.status_code == 201
        
        key_data = response.json()
        assert 'key' in key_data
        assert 'id' in key_data
        assert key_data['name'] == create_data['name']
        
        api_key = key_data['key']
        key_id = key_data['id']
        
        # List API keys
        response = client.get('/admin/api-keys', headers=headers)
        assert response.status_code == 200
        keys = response.json()
        assert any(k['id'] == key_id for k in keys)
        
        # Use the API key
        response = client.get('/generators', headers={'X-API-Key': api_key})
        assert response.status_code == 200
        
        # Update API key
        update_data = {
            'rate_limit': 200,
            'is_active': True
        }
        response = client.put(f'/admin/api-keys/{key_id}', json=update_data, headers=headers)
        assert response.status_code == 200
        
        # Revoke API key
        response = client.delete(f'/admin/api-keys/{key_id}', headers=headers)
        assert response.status_code == 200
        
        # Try to use revoked key
        response = client.get('/generators', headers={'X-API-Key': api_key})
        assert response.status_code == 401
    
    def test_api_key_permissions(self, client, admin_token):
        """Test API key permission enforcement"""
        headers = {'Authorization': f'Bearer {admin_token}'}
        
        # Create read-only key
        response = client.post('/admin/api-keys', json={
            'name': 'Read-only Key',
            'permissions': ['read'],
            'rate_limit': 50
        }, headers=headers)
        assert response.status_code == 201
        readonly_key = response.json()['key']
        
        # Create read-write key
        response = client.post('/admin/api-keys', json={
            'name': 'Read-Write Key',
            'permissions': ['read', 'write'],
            'rate_limit': 100
        }, headers=headers)
        assert response.status_code == 201
        readwrite_key = response.json()['key']
        
        # Test read operation with both keys
        for key in [readonly_key, readwrite_key]:
            response = client.get('/generators', headers={'X-API-Key': key})
            assert response.status_code == 200
        
        # Test write operation
        files = {'file': ('test.csv', b'age,income\n25,50000', 'text/csv')}
        config = '{"generator_type": "sdv", "model_type": "gaussian_copula"}'
        
        # Should fail with read-only key
        response = client.post(
            '/generate',
            headers={'X-API-Key': readonly_key},
            files=files,
            data={'config': config}
        )
        assert response.status_code == 403
        
        # Should work with read-write key
        files = {'file': ('test.csv', b'age,income\n25,50000', 'text/csv')}
        response = client.post(
            '/generate',
            headers={'X-API-Key': readwrite_key},
            files=files,
            data={'config': config}
        )
        # May fail due to missing dependencies, but shouldn't be 403
        assert response.status_code != 403