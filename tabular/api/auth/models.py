"""
Authentication models
"""

from typing import Optional, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, EmailStr
import secrets
import hashlib


class UserRole(str, Enum):
    """User roles for authorization"""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    SERVICE_ACCOUNT = "service_account"


class User(BaseModel):
    """User model"""
    id: str = Field(default_factory=lambda: secrets.token_urlsafe(16))
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    hashed_password: Optional[str] = None
    
    class Config:
        use_enum_values = True


class APIKey(BaseModel):
    """API Key model"""
    id: str = Field(default_factory=lambda: secrets.token_urlsafe(16))
    key: str = Field(default_factory=lambda: f"sk_{secrets.token_urlsafe(32)}")
    name: str
    user_id: str
    role: UserRole = UserRole.SERVICE_ACCOUNT
    is_active: bool = True
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    scopes: List[str] = Field(default_factory=list)
    rate_limit: Optional[int] = None  # requests per minute
    
    def hash_key(self) -> str:
        """Hash the API key for storage"""
        return hashlib.sha256(self.key.encode()).hexdigest()
    
    def is_expired(self) -> bool:
        """Check if the API key is expired"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    class Config:
        use_enum_values = True


class TokenData(BaseModel):
    """Token data model"""
    username: Optional[str] = None
    user_id: Optional[str] = None
    scopes: List[str] = []


class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    refresh_token: Optional[str] = None
    user: User


class APIKeyRequest(BaseModel):
    """API key creation request"""
    name: str
    expires_in_days: Optional[int] = None
    scopes: List[str] = Field(default_factory=list)
    rate_limit: Optional[int] = None


class APIKeyResponse(BaseModel):
    """API key response"""
    id: str
    key: str  # Only returned on creation
    name: str
    expires_at: Optional[datetime] = None
    created_at: datetime
    scopes: List[str]
    rate_limit: Optional[int] = None