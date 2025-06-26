"""
Unified Cache Module

Provides unified caching abstraction for all Inferloop services.
Supports Redis, Memcached, and in-memory caching with a consistent interface.
"""

import os
import asyncio
import json
import pickle
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib

import redis.asyncio as aioredis
import aiomemcache
from fastapi import Depends


@dataclass
class CacheConfig:
    """Cache configuration"""
    provider: str  # redis, memcached, memory
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    database: int = 0
    namespace: str = ""
    default_ttl: int = 3600
    max_memory: str = "256MB"
    eviction_policy: str = "allkeys-lru"


class CacheBackend(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[bytes]:
        """Get value by key"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    @abstractmethod
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter"""
        pass
    
    @abstractmethod
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key"""
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        pass
    
    @abstractmethod
    async def flush(self) -> bool:
        """Flush all keys"""
        pass
    
    @abstractmethod
    async def ping(self) -> bool:
        """Check connection"""
        pass


class RedisBackend(CacheBackend):
    """Redis cache backend"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis = None
        self._connection_pool = None
    
    async def _ensure_connection(self):
        """Ensure Redis connection is established"""
        if self.redis is None:
            self._connection_pool = aioredis.ConnectionPool.from_url(
                f"redis://:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}",
                encoding="utf-8",
                decode_responses=False,
                max_connections=20
            )
            self.redis = aioredis.Redis(connection_pool=self._connection_pool)
    
    async def get(self, key: str) -> Optional[bytes]:
        """Get value by key from Redis"""
        await self._ensure_connection()
        try:
            return await self.redis.get(key)
        except Exception as e:
            raise CacheError(f"Failed to get key {key}: {str(e)}")
    
    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Set value in Redis with optional TTL"""
        await self._ensure_connection()
        try:
            if ttl is not None:
                return await self.redis.setex(key, ttl, value)
            else:
                return await self.redis.set(key, value)
        except Exception as e:
            raise CacheError(f"Failed to set key {key}: {str(e)}")
    
    async def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        await self._ensure_connection()
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception as e:
            raise CacheError(f"Failed to delete key {key}: {str(e)}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        await self._ensure_connection()
        try:
            result = await self.redis.exists(key)
            return result > 0
        except Exception as e:
            raise CacheError(f"Failed to check existence of key {key}: {str(e)}")
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter in Redis"""
        await self._ensure_connection()
        try:
            return await self.redis.incr(key, amount)
        except Exception as e:
            raise CacheError(f"Failed to increment key {key}: {str(e)}")
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key in Redis"""
        await self._ensure_connection()
        try:
            return await self.redis.expire(key, ttl)
        except Exception as e:
            raise CacheError(f"Failed to set expiration for key {key}: {str(e)}")
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern from Redis"""
        await self._ensure_connection()
        try:
            keys = await self.redis.keys(pattern)
            return [key.decode('utf-8') if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            raise CacheError(f"Failed to get keys with pattern {pattern}: {str(e)}")
    
    async def flush(self) -> bool:
        """Flush all keys from current Redis database"""
        await self._ensure_connection()
        try:
            await self.redis.flushdb()
            return True
        except Exception as e:
            raise CacheError(f"Failed to flush database: {str(e)}")
    
    async def ping(self) -> bool:
        """Check Redis connection"""
        await self._ensure_connection()
        try:
            await self.redis.ping()
            return True
        except:
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()


class MemcachedBackend(CacheBackend):
    """Memcached cache backend"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memcached = None
    
    async def _ensure_connection(self):
        """Ensure Memcached connection is established"""
        if self.memcached is None:
            self.memcached = aiomemcache.Client(
                f"{self.config.host}:{self.config.port}",
                pool_size=10
            )
    
    async def get(self, key: str) -> Optional[bytes]:
        """Get value by key from Memcached"""
        await self._ensure_connection()
        try:
            value = await self.memcached.get(key.encode())
            return value
        except Exception as e:
            raise CacheError(f"Failed to get key {key}: {str(e)}")
    
    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Set value in Memcached with optional TTL"""
        await self._ensure_connection()
        try:
            ttl = ttl or self.config.default_ttl
            result = await self.memcached.set(key.encode(), value, exptime=ttl)
            return result
        except Exception as e:
            raise CacheError(f"Failed to set key {key}: {str(e)}")
    
    async def delete(self, key: str) -> bool:
        """Delete key from Memcached"""
        await self._ensure_connection()
        try:
            result = await self.memcached.delete(key.encode())
            return result
        except Exception as e:
            raise CacheError(f"Failed to delete key {key}: {str(e)}")
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Memcached"""
        value = await self.get(key)
        return value is not None
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter in Memcached"""
        await self._ensure_connection()
        try:
            result = await self.memcached.incr(key.encode(), amount)
            return result if result is not None else 0
        except Exception as e:
            raise CacheError(f"Failed to increment key {key}: {str(e)}")
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key in Memcached (touch operation)"""
        await self._ensure_connection()
        try:
            result = await self.memcached.touch(key.encode(), ttl)
            return result
        except Exception as e:
            raise CacheError(f"Failed to set expiration for key {key}: {str(e)}")
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern (limited support in Memcached)"""
        # Memcached doesn't support pattern matching
        # This is a placeholder implementation
        raise NotImplementedError("Pattern matching not supported in Memcached")
    
    async def flush(self) -> bool:
        """Flush all keys from Memcached"""
        await self._ensure_connection()
        try:
            await self.memcached.flush_all()
            return True
        except Exception as e:
            raise CacheError(f"Failed to flush cache: {str(e)}")
    
    async def ping(self) -> bool:
        """Check Memcached connection"""
        try:
            await self._ensure_connection()
            # Try a simple operation to check connection
            await self.memcached.get(b"ping_test")
            return True
        except:
            return False
    
    async def close(self):
        """Close Memcached connection"""
        if self.memcached:
            await self.memcached.close()


class MemoryBackend(CacheBackend):
    """In-memory cache backend (for development/testing)"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._data: Dict[str, Any] = {}
        self._expiry: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
    
    async def _cleanup_expired(self):
        """Remove expired keys"""
        now = datetime.utcnow()
        expired_keys = [
            key for key, expiry in self._expiry.items()
            if expiry <= now
        ]
        
        for key in expired_keys:
            self._data.pop(key, None)
            self._expiry.pop(key, None)
    
    async def get(self, key: str) -> Optional[bytes]:
        """Get value by key from memory"""
        async with self._lock:
            await self._cleanup_expired()
            return self._data.get(key)
    
    async def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Set value in memory with optional TTL"""
        async with self._lock:
            self._data[key] = value
            
            if ttl is not None:
                self._expiry[key] = datetime.utcnow() + timedelta(seconds=ttl)
            elif key in self._expiry:
                # Remove expiry if no TTL specified
                del self._expiry[key]
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from memory"""
        async with self._lock:
            existed = key in self._data
            self._data.pop(key, None)
            self._expiry.pop(key, None)
            return existed
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory"""
        async with self._lock:
            await self._cleanup_expired()
            return key in self._data
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter in memory"""
        async with self._lock:
            await self._cleanup_expired()
            
            if key not in self._data:
                self._data[key] = b"0"
            
            try:
                current_value = int(self._data[key].decode())
                new_value = current_value + amount
                self._data[key] = str(new_value).encode()
                return new_value
            except (ValueError, UnicodeDecodeError):
                raise CacheError(f"Key {key} is not a valid integer")
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key in memory"""
        async with self._lock:
            if key in self._data:
                self._expiry[key] = datetime.utcnow() + timedelta(seconds=ttl)
                return True
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern from memory"""
        async with self._lock:
            await self._cleanup_expired()
            
            if pattern == "*":
                return list(self._data.keys())
            
            # Simple pattern matching (only supports * wildcard)
            import fnmatch
            return [key for key in self._data.keys() if fnmatch.fnmatch(key, pattern)]
    
    async def flush(self) -> bool:
        """Flush all keys from memory"""
        async with self._lock:
            self._data.clear()
            self._expiry.clear()
            return True
    
    async def ping(self) -> bool:
        """Check memory backend (always available)"""
        return True
    
    async def close(self):
        """Close memory backend (no-op)"""
        pass


class CacheError(Exception):
    """Cache operation error"""
    pass


class CacheClient:
    """Unified cache client"""
    
    def __init__(self, namespace: str = ""):
        self.namespace = namespace
        self.config = self._load_config()
        self.backend = self._create_backend()
    
    def _load_config(self) -> CacheConfig:
        """Load cache configuration"""
        provider = os.getenv('CACHE_PROVIDER', 'redis').lower()
        
        return CacheConfig(
            provider=provider,
            host=os.getenv('CACHE_HOST', 'localhost'),
            port=int(os.getenv('CACHE_PORT', '6379')),
            password=os.getenv('CACHE_PASSWORD'),
            database=int(os.getenv('CACHE_DATABASE', '0')),
            namespace=self.namespace,
            default_ttl=int(os.getenv('CACHE_DEFAULT_TTL', '3600')),
            max_memory=os.getenv('CACHE_MAX_MEMORY', '256MB'),
            eviction_policy=os.getenv('CACHE_EVICTION_POLICY', 'allkeys-lru')
        )
    
    def _create_backend(self) -> CacheBackend:
        """Create cache backend based on configuration"""
        if self.config.provider == 'redis':
            return RedisBackend(self.config)
        elif self.config.provider == 'memcached':
            return MemcachedBackend(self.config)
        elif self.config.provider == 'memory':
            return MemoryBackend(self.config)
        else:
            raise ValueError(f"Unsupported cache provider: {self.config.provider}")
    
    def _make_key(self, key: str) -> str:
        """Create namespaced key"""
        if self.config.namespace:
            return f"{self.config.namespace}:{key}"
        return key
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        namespaced_key = self._make_key(key)
        data = await self.backend.get(namespaced_key)
        
        if data is None:
            return None
        
        try:
            # Try to unpickle first (for complex objects)
            return pickle.loads(data)
        except:
            try:
                # Try JSON decode
                return json.loads(data.decode('utf-8'))
            except:
                # Return as string
                return data.decode('utf-8')
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL"""
        namespaced_key = self._make_key(key)
        
        # Serialize value
        if isinstance(value, (dict, list)):
            data = json.dumps(value).encode('utf-8')
        elif isinstance(value, str):
            data = value.encode('utf-8')
        elif isinstance(value, bytes):
            data = value
        else:
            # Use pickle for complex objects
            data = pickle.dumps(value)
        
        ttl = ttl or self.config.default_ttl
        return await self.backend.set(namespaced_key, data, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete key"""
        namespaced_key = self._make_key(key)
        return await self.backend.delete(namespaced_key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        namespaced_key = self._make_key(key)
        return await self.backend.exists(namespaced_key)
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter"""
        namespaced_key = self._make_key(key)
        return await self.backend.increment(namespaced_key, amount)
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key"""
        namespaced_key = self._make_key(key)
        return await self.backend.expire(namespaced_key, ttl)
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern (without namespace prefix)"""
        namespaced_pattern = self._make_key(pattern)
        keys = await self.backend.keys(namespaced_pattern)
        
        # Remove namespace prefix from results
        if self.config.namespace:
            prefix = f"{self.config.namespace}:"
            keys = [key[len(prefix):] for key in keys if key.startswith(prefix)]
        
        return keys
    
    async def flush(self) -> bool:
        """Flush all keys in namespace"""
        if self.config.namespace:
            # Delete only keys in our namespace
            keys = await self.keys("*")
            for key in keys:
                await self.delete(key)
            return True
        else:
            # Flush entire cache
            return await self.backend.flush()
    
    async def ping(self) -> bool:
        """Check cache connection"""
        return await self.backend.ping()
    
    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get and parse JSON value"""
        value = await self.get(key)
        if isinstance(value, dict):
            return value
        return None
    
    async def set_json(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set JSON value"""
        return await self.set(key, value, ttl)
    
    async def get_string(self, key: str) -> Optional[str]:
        """Get string value"""
        value = await self.get(key)
        if isinstance(value, str):
            return value
        return None
    
    async def set_string(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set string value"""
        return await self.set(key, value, ttl)
    
    def cached(self, key_func: Callable = None, ttl: Optional[int] = None):
        """Decorator for caching function results"""
        def decorator(func: Callable) -> Callable:
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    key_parts = [func.__name__]
                    key_parts.extend(str(arg) for arg in args)
                    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                    cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                await self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    async def close(self):
        """Close cache connection"""
        await self.backend.close()


# Dependency for FastAPI
async def get_cache_client() -> CacheClient:
    """Dependency to get cache client"""
    return CacheClient("default")


# Global cache clients cache
_cache_clients: Dict[str, CacheClient] = {}


def get_cache_client_sync(namespace: str = "") -> CacheClient:
    """Get cache client synchronously (cached)"""
    if namespace not in _cache_clients:
        _cache_clients[namespace] = CacheClient(namespace)
    return _cache_clients[namespace]


# Rate limiting support using cache
class RateLimiter:
    """Rate limiter using cache backend"""
    
    def __init__(self, cache_client: CacheClient):
        self.cache = cache_client
    
    async def is_allowed(self, identifier: str, limit: int, window: int) -> bool:
        """Check if request is allowed under rate limit"""
        key = f"rate_limit:{identifier}"
        
        current_count = await self.cache.get(key)
        if current_count is None:
            # First request in window
            await self.cache.set(key, 1, window)
            return True
        
        if isinstance(current_count, str):
            current_count = int(current_count)
        elif not isinstance(current_count, int):
            current_count = 0
        
        if current_count >= limit:
            return False
        
        # Increment counter
        await self.cache.increment(key)
        return True
    
    async def get_remaining(self, identifier: str, limit: int) -> int:
        """Get remaining requests in current window"""
        key = f"rate_limit:{identifier}"
        current_count = await self.cache.get(key)
        
        if current_count is None:
            return limit
        
        if isinstance(current_count, str):
            current_count = int(current_count)
        elif not isinstance(current_count, int):
            current_count = 0
        
        return max(0, limit - current_count)


# Cache warming utilities
async def warm_cache(cache_client: CacheClient, data_loader: Callable, keys: List[str], ttl: Optional[int] = None):
    """Warm cache with data"""
    for key in keys:
        if not await cache_client.exists(key):
            data = await data_loader(key)
            if data is not None:
                await cache_client.set(key, data, ttl)