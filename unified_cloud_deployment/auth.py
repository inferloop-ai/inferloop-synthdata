"""
Authentication and Authorization Module

Provides unified authentication and authorization services for all Inferloop services.
Supports JWT tokens, API keys, and role-based access control (RBAC).
"""

import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import jwt
from fastapi import HTTPException, Request, Depends
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import httpx


class UserTier(Enum):
    STARTER = "starter"
    PROFESSIONAL = "professional" 
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


@dataclass
class User:
    """User data model"""
    id: str
    email: str
    tier: UserTier
    permissions: List[str]
    organization_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def is_enterprise(self) -> bool:
        return self.tier == UserTier.ENTERPRISE
    
    @property
    def is_business_or_higher(self) -> bool:
        return self.tier in [UserTier.BUSINESS, UserTier.ENTERPRISE]


class AuthConfig:
    """Authentication configuration"""
    
    def __init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET", "dev-secret-key")
        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.jwt_expiration_hours = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
        self.auth_service_url = os.getenv("AUTH_SERVICE_URL", "http://auth-service:8000")
        self.enable_api_key_auth = os.getenv("ENABLE_API_KEY_AUTH", "true").lower() == "true"
        self.enable_jwt_auth = os.getenv("ENABLE_JWT_AUTH", "true").lower() == "true"


auth_config = AuthConfig()
security = HTTPBearer(auto_error=False)


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for FastAPI applications"""
    
    def __init__(self, app, service_name: str, required_permissions: List[str] = None):
        super().__init__(app)
        self.service_name = service_name
        self.required_permissions = required_permissions or []
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for health checks and public endpoints
        if request.url.path in ["/health", "/ready", "/metrics", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        # Extract token from request
        authorization = request.headers.get("Authorization")
        api_key = request.headers.get("X-API-Key")
        
        user = None
        
        # Try JWT authentication first
        if authorization and auth_config.enable_jwt_auth:
            try:
                user = await self._validate_jwt_token(authorization)
            except HTTPException:
                pass
        
        # Try API key authentication
        if not user and api_key and auth_config.enable_api_key_auth:
            try:
                user = await self._validate_api_key(api_key)
            except HTTPException:
                pass
        
        # If no valid authentication found, return 401
        if not user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Check service-specific permissions
        if self.required_permissions:
            missing_permissions = [
                perm for perm in self.required_permissions 
                if perm not in user.permissions
            ]
            if missing_permissions:
                raise HTTPException(
                    status_code=403, 
                    detail=f"Missing required permissions: {missing_permissions}"
                )
        
        # Add user to request state
        request.state.user = user
        
        return await call_next(request)
    
    async def _validate_jwt_token(self, authorization: str) -> User:
        """Validate JWT token and return user"""
        try:
            # Extract token from "Bearer <token>"
            token = authorization.split(" ")[1] if " " in authorization else authorization
            
            # Decode JWT
            payload = jwt.decode(
                token, 
                auth_config.jwt_secret, 
                algorithms=[auth_config.jwt_algorithm]
            )
            
            # Extract user information
            user_id = payload.get("sub")
            email = payload.get("email")
            tier = payload.get("tier", "starter")
            permissions = payload.get("permissions", [])
            org_id = payload.get("organization_id")
            
            if not user_id or not email:
                raise HTTPException(status_code=401, detail="Invalid token payload")
            
            return User(
                id=user_id,
                email=email,
                tier=UserTier(tier),
                permissions=permissions,
                organization_id=org_id
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Token validation failed: {str(e)}")
    
    async def _validate_api_key(self, api_key: str) -> User:
        """Validate API key and return user"""
        try:
            # Call auth service to validate API key
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{auth_config.auth_service_url}/api/auth/validate-api-key",
                    json={"api_key": api_key},
                    timeout=5.0
                )
                
                if response.status_code != 200:
                    raise HTTPException(status_code=401, detail="Invalid API key")
                
                data = response.json()
                
                return User(
                    id=data["user_id"],
                    email=data["email"],
                    tier=UserTier(data["tier"]),
                    permissions=data["permissions"],
                    organization_id=data.get("organization_id")
                )
                
        except httpx.TimeoutException:
            raise HTTPException(status_code=401, detail="Auth service timeout")
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"API key validation failed: {str(e)}")


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> User:
    """Dependency to get current authenticated user"""
    
    # Check if user was set by middleware
    if hasattr(request.state, 'user'):
        return request.state.user
    
    # If no middleware, validate manually
    if not credentials and not request.headers.get("X-API-Key"):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    user = None
    
    # Try JWT first
    if credentials and auth_config.enable_jwt_auth:
        try:
            token = credentials.credentials
            payload = jwt.decode(
                token,
                auth_config.jwt_secret,
                algorithms=[auth_config.jwt_algorithm]
            )
            
            user = User(
                id=payload["sub"],
                email=payload["email"],
                tier=UserTier(payload.get("tier", "starter")),
                permissions=payload.get("permissions", []),
                organization_id=payload.get("organization_id")
            )
        except Exception:
            pass
    
    # Try API key
    if not user and auth_config.enable_api_key_auth:
        api_key = request.headers.get("X-API-Key")
        if api_key:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{auth_config.auth_service_url}/api/auth/validate-api-key",
                        json={"api_key": api_key},
                        timeout=5.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        user = User(
                            id=data["user_id"],
                            email=data["email"],
                            tier=UserTier(data["tier"]),
                            permissions=data["permissions"],
                            organization_id=data.get("organization_id")
                        )
            except Exception:
                pass
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication")
    
    return user


def require_permissions(user: User, required_permissions: List[str]):
    """Check if user has required permissions"""
    missing_permissions = [
        perm for perm in required_permissions 
        if perm not in user.permissions
    ]
    
    if missing_permissions:
        raise HTTPException(
            status_code=403,
            detail=f"Missing required permissions: {missing_permissions}"
        )


def require_tier(user: User, min_tier: UserTier):
    """Check if user has minimum required tier"""
    tier_hierarchy = {
        UserTier.STARTER: 1,
        UserTier.PROFESSIONAL: 2,
        UserTier.BUSINESS: 3,
        UserTier.ENTERPRISE: 4
    }
    
    if tier_hierarchy[user.tier] < tier_hierarchy[min_tier]:
        raise HTTPException(
            status_code=403,
            detail=f"Minimum tier {min_tier.value} required, user has {user.tier.value}"
        )


def create_jwt_token(user: User) -> str:
    """Create JWT token for user"""
    payload = {
        "sub": user.id,
        "email": user.email,
        "tier": user.tier.value,
        "permissions": user.permissions,
        "organization_id": user.organization_id,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=auth_config.jwt_expiration_hours)
    }
    
    return jwt.encode(payload, auth_config.jwt_secret, algorithm=auth_config.jwt_algorithm)


class AuthService:
    """Service for managing authentication operations"""
    
    def __init__(self):
        self.config = auth_config
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email/password"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.config.auth_service_url}/api/auth/login",
                    json={"email": email, "password": password}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return User(
                        id=data["user_id"],
                        email=data["email"],
                        tier=UserTier(data["tier"]),
                        permissions=data["permissions"],
                        organization_id=data.get("organization_id")
                    )
        except Exception:
            pass
        
        return None
    
    async def create_api_key(self, user: User, name: str) -> str:
        """Create new API key for user"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.config.auth_service_url}/api/auth/api-keys",
                json={
                    "user_id": user.id,
                    "name": name,
                    "permissions": user.permissions
                }
            )
            
            if response.status_code == 201:
                return response.json()["api_key"]
            else:
                raise HTTPException(status_code=400, detail="Failed to create API key")
    
    async def revoke_api_key(self, user: User, api_key: str):
        """Revoke API key"""
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{self.config.auth_service_url}/api/auth/api-keys/{api_key}",
                headers={"X-User-ID": user.id}
            )
            
            if response.status_code != 204:
                raise HTTPException(status_code=400, detail="Failed to revoke API key")


# Global auth service instance
auth_service = AuthService()


# Utility functions for common permission checks
def can_access_service(user: User, service_name: str) -> bool:
    """Check if user can access a specific service"""
    return f"{service_name}:read" in user.permissions


def can_generate(user: User, service_name: str) -> bool:
    """Check if user can generate data in a service"""
    return f"{service_name}:generate" in user.permissions


def can_validate(user: User, service_name: str) -> bool:
    """Check if user can validate data in a service"""
    return f"{service_name}:validate" in user.permissions


def can_admin(user: User, service_name: str) -> bool:
    """Check if user has admin access to a service"""
    return f"{service_name}:admin" in user.permissions