"""
API Key management system
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio
import aiofiles

from .models import APIKey, User, UserRole


class APIKeyManager:
    """Manage API keys with persistence"""
    
    def __init__(self, storage_path: str = "./api_keys.json"):
        self.storage_path = Path(storage_path)
        self.api_keys: Dict[str, APIKey] = {}
        self.key_index: Dict[str, str] = {}  # hashed_key -> key_id
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize the API key manager"""
        if self.storage_path.exists():
            await self.load_keys()
    
    async def load_keys(self):
        """Load API keys from storage"""
        async with self._lock:
            try:
                async with aiofiles.open(self.storage_path, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
                    
                    for key_data in data.get('api_keys', []):
                        api_key = APIKey(**key_data)
                        self.api_keys[api_key.id] = api_key
                        self.key_index[api_key.hash_key()] = api_key.id
                        
            except Exception as e:
                print(f"Error loading API keys: {e}")
    
    async def save_keys(self):
        """Save API keys to storage"""
        async with self._lock:
            try:
                data = {
                    'api_keys': [
                        key.dict(exclude={'key'})  # Don't save raw key
                        for key in self.api_keys.values()
                    ]
                }
                
                async with aiofiles.open(self.storage_path, 'w') as f:
                    await f.write(json.dumps(data, default=str, indent=2))
                    
            except Exception as e:
                print(f"Error saving API keys: {e}")
    
    async def create_key(self, user: User, name: str, 
                        expires_in_days: Optional[int] = None,
                        scopes: Optional[List[str]] = None,
                        rate_limit: Optional[int] = None) -> APIKey:
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
        
        # Store the key
        self.api_keys[api_key.id] = api_key
        self.key_index[api_key.hash_key()] = api_key.id
        
        # Save to storage
        await self.save_keys()
        
        return api_key
    
    async def validate_key(self, api_key: str) -> Optional[APIKey]:
        """Validate an API key"""
        import hashlib
        hashed = hashlib.sha256(api_key.encode()).hexdigest()
        
        key_id = self.key_index.get(hashed)
        if not key_id:
            return None
        
        key = self.api_keys.get(key_id)
        if not key:
            return None
        
        if not key.is_active:
            return None
        
        if key.is_expired():
            return None
        
        # Update last used
        key.last_used = datetime.utcnow()
        await self.save_keys()
        
        return key
    
    async def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key"""
        if key_id not in self.api_keys:
            return False
        
        key = self.api_keys[key_id]
        key.is_active = False
        
        await self.save_keys()
        return True
    
    async def list_user_keys(self, user_id: str) -> List[APIKey]:
        """List all API keys for a user"""
        return [
            key for key in self.api_keys.values()
            if key.user_id == user_id
        ]
    
    async def rotate_key(self, key_id: str) -> Optional[APIKey]:
        """Rotate an API key (revoke old, create new)"""
        old_key = self.api_keys.get(key_id)
        if not old_key:
            return None
        
        # Revoke old key
        old_key.is_active = False
        
        # Create new key with same settings
        new_key = APIKey(
            name=f"{old_key.name} (rotated)",
            user_id=old_key.user_id,
            role=old_key.role,
            expires_at=old_key.expires_at,
            scopes=old_key.scopes,
            rate_limit=old_key.rate_limit
        )
        
        self.api_keys[new_key.id] = new_key
        self.key_index[new_key.hash_key()] = new_key.id
        
        await self.save_keys()
        return new_key
    
    async def cleanup_expired(self):
        """Clean up expired API keys"""
        now = datetime.utcnow()
        expired_keys = []
        
        for key_id, key in self.api_keys.items():
            if key.expires_at and key.expires_at < now:
                expired_keys.append(key_id)
        
        for key_id in expired_keys:
            key = self.api_keys[key_id]
            del self.key_index[key.hash_key()]
            del self.api_keys[key_id]
        
        if expired_keys:
            await self.save_keys()
        
        return len(expired_keys)
    
    async def get_usage_stats(self, key_id: str) -> Dict[str, Any]:
        """Get usage statistics for an API key"""
        key = self.api_keys.get(key_id)
        if not key:
            return {}
        
        return {
            'key_id': key_id,
            'name': key.name,
            'created_at': key.created_at,
            'last_used': key.last_used,
            'is_active': key.is_active,
            'expires_at': key.expires_at,
            'is_expired': key.is_expired(),
            'rate_limit': key.rate_limit,
            'scopes': key.scopes
        }