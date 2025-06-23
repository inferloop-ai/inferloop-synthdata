"""
Authentication and authorization handler
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from functools import wraps
import secrets
import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

from .models import User, UserRole, TokenData, APIKey


# Configuration
SECRET_KEY = secrets.token_urlsafe(32)  # In production, load from environment
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
bearer_scheme = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthHandler:
    """Handle authentication and authorization"""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or SECRET_KEY
        self.pwd_context = pwd_context
        
        # In-memory storage (in production, use database)
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.refresh_tokens: Dict[str, str] = {}
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin = User(
            username="admin",
            email="admin@inferloop.ai",
            full_name="Admin User",
            role=UserRole.ADMIN,
            hashed_password=self.hash_password("admin123")  # Change in production
        )
        self.users[admin.username] = admin
    
    def hash_password(self, password: str) -> str:
        """Hash a password"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create refresh token"""
        token = secrets.token_urlsafe(32)
        self.refresh_tokens[token] = user_id
        return token
    
    def decode_token(self, token: str) -> TokenData:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            scopes: List[str] = payload.get("scopes", [])
            
            if username is None and user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return TokenData(username=username, user_id=user_id, scopes=scopes)
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password"""
        user = self.users.get(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        return user
    
    def get_current_user(self, token: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> User:
        """Get current user from JWT token"""
        token_data = self.decode_token(token.credentials)
        
        user = self.users.get(token_data.username)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Inactive user"
            )
        
        return user
    
    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate API key"""
        # Hash the key to find it
        hashed = hashlib.sha256(api_key.encode()).hexdigest()
        
        for key in self.api_keys.values():
            if key.hash_key() == hashed:
                if not key.is_active:
                    return None
                if key.is_expired():
                    return None
                
                # Update last used
                key.last_used = datetime.utcnow()
                return key
        
        return None
    
    def get_current_api_key(self, api_key: Optional[str] = Depends(api_key_header)) -> APIKey:
        """Get current API key"""
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"X-API-Key": "API key required"},
            )
        
        key = self.validate_api_key(api_key)
        if not key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"X-API-Key": "Invalid API key"},
            )
        
        return key
    
    def create_user(self, username: str, email: str, password: str, 
                   role: UserRole = UserRole.USER, full_name: Optional[str] = None) -> User:
        """Create a new user"""
        if username in self.users:
            raise ValueError(f"User {username} already exists")
        
        user = User(
            username=username,
            email=email,
            full_name=full_name,
            role=role,
            hashed_password=self.hash_password(password)
        )
        
        self.users[username] = user
        return user
    
    def create_api_key(self, user: User, name: str, expires_in_days: Optional[int] = None,
                      scopes: List[str] = None, rate_limit: Optional[int] = None) -> APIKey:
        """Create a new API key"""
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        api_key = APIKey(
            name=name,
            user_id=user.id,
            role=user.role,
            expires_at=expires_at,
            scopes=scopes or [],
            rate_limit=rate_limit
        )
        
        self.api_keys[api_key.id] = api_key
        return api_key


# Global auth handler instance
auth_handler = AuthHandler()


def require_auth(required_role: Optional[UserRole] = None):
    """Decorator to require authentication and optionally a specific role"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current user from request
            request = kwargs.get('request')
            if not request:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            if not hasattr(request.state, 'user'):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            user = request.state.user
            
            # Check role if required
            if required_role and user.role != required_role:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role {required_role} required"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_api_key(scopes: Optional[List[str]] = None):
    """Decorator to require API key authentication"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get API key from request
            request = kwargs.get('request')
            if not request:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            if not hasattr(request.state, 'api_key'):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required"
                )
            
            api_key = request.state.api_key
            
            # Check scopes if required
            if scopes:
                for scope in scopes:
                    if scope not in api_key.scopes:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"Scope {scope} required"
                        )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator