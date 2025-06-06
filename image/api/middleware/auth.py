from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
import os
from typing import Optional, Dict
import redis
import hashlib

security = HTTPBearer()

class AuthManager:
    """Advanced authentication and authorization manager."""
    
    def __init__(self):
        self.secret_key = os.getenv('JWT_SECRET_KEY', 'your-secret-key')
        self.algorithm = 'HS256'
        self.access_token_expire_minutes = 30
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Verify JWT token and extract user info."""
        try:
            payload = jwt.decode(credentials.credentials, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Check if token is blacklisted
            token_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
            if self.redis_client.get(f"blacklist:{token_hash}"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return {"username": username, "permissions": payload.get("permissions", [])}
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def check_permission(self, required_permission: str):
        """Check if user has required permission."""
        def permission_checker(current_user: dict = Depends(self.verify_token)):
            if required_permission not in current_user.get("permissions", []):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {required_permission} required"
                )
            return current_user
        return permission_checker
    
    def revoke_token(self, token: str):
        """Add token to blacklist."""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        # Set expiration to match token expiration
        self.redis_client.setex(f"blacklist:{token_hash}", 
                               timedelta(minutes=self.access_token_expire_minutes), 
                               "revoked")

auth_manager = AuthManager()

# Usage in API endpoints:
# @app.post("/generate/diffusion")
# async def generate_diffusion(
#     request: GenerationRequest,
#     current_user: dict = Depends(auth_manager.check_permission("generate:images"))
# ):
#     # Your endpoint logic here
'''

# ==================== Advanced Monitoring & Observability ====================
monitoring_config = '''
