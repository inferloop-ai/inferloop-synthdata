"""
Cache manager for structured document data.
Provides in-memory and distributed caching for improved performance.
"""

from __future__ import annotations

import json
import pickle
import time
import hashlib
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

import redis
from redis.exceptions import RedisError

from ...core.config import BaseConfig
from ...core.logging import get_logger

logger = get_logger(__name__)


class CacheManagerConfig(BaseConfig):
    """Cache manager configuration"""
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    redis_timeout: int = 5
    
    # Cache settings
    default_ttl: int = 3600  # 1 hour
    max_memory_cache_size: int = 1000
    compression_enabled: bool = True
    
    # Cache prefixes
    document_prefix: str = "doc:"
    content_prefix: str = "content:"
    validation_prefix: str = "validation:"
    generation_prefix: str = "generation:"


class CacheManager:
    """
    Cache manager for document data and generation results.
    Supports both in-memory and Redis-based distributed caching.
    """
    
    def __init__(self, config: Optional[CacheManagerConfig] = None):
        """Initialize cache manager"""
        self.config = config or CacheManagerConfig()
        
        # In-memory cache
        self.memory_cache: Dict[str, Tuple[Any, float]] = {}
        
        # Redis connection
        self.redis_client = None
        self._connect_redis()
        
        logger.info("Initialized CacheManager")
    
    def _connect_redis(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                password=self.config.redis_password,
                socket_timeout=self.config.redis_timeout,
                decode_responses=False  # We handle encoding manually
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis cache")
        except RedisError as e:
            logger.warning(f"Redis connection failed, using memory cache only: {str(e)}")
            self.redis_client = None
    
    def _generate_key(self, prefix: str, identifier: str) -> str:
        """Generate cache key"""
        return f"{prefix}{identifier}"
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for caching"""
        try:
            if self.config.compression_enabled:
                import gzip
                return gzip.compress(pickle.dumps(data))
            else:
                return pickle.dumps(data)
        except Exception as e:
            logger.error(f"Failed to serialize data: {str(e)}")
            return b""
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize cached data"""
        try:
            if self.config.compression_enabled:
                import gzip
                return pickle.loads(gzip.decompress(data))
            else:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize data: {str(e)}")
            return None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        # Try Redis first
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    return self._deserialize_data(data)
            except RedisError as e:
                logger.warning(f"Redis get failed: {str(e)}")
        
        # Try memory cache
        if key in self.memory_cache:
            data, expiry = self.memory_cache[key]
            if time.time() < expiry:
                return data
            else:
                del self.memory_cache[key]
        
        return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        use_memory: bool = True
    ) -> bool:
        """Set value in cache"""
        ttl = ttl or self.config.default_ttl
        success = False
        
        # Store in Redis
        if self.redis_client:
            try:
                serialized = self._serialize_data(value)
                if serialized:
                    self.redis_client.setex(key, ttl, serialized)
                    success = True
            except RedisError as e:
                logger.warning(f"Redis set failed: {str(e)}")
        
        # Store in memory cache
        if use_memory:
            expiry = time.time() + ttl
            self.memory_cache[key] = (value, expiry)
            
            # Cleanup memory cache if too large
            if len(self.memory_cache) > self.config.max_memory_cache_size:
                self._cleanup_memory_cache()
            
            success = True
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        success = False
        
        # Delete from Redis
        if self.redis_client:
            try:
                self.redis_client.delete(key)
                success = True
            except RedisError as e:
                logger.warning(f"Redis delete failed: {str(e)}")
        
        # Delete from memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
            success = True
        
        return success
    
    def _cleanup_memory_cache(self):
        """Cleanup expired entries from memory cache"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expiry) in self.memory_cache.items()
            if current_time >= expiry
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        # If still too large, remove oldest entries
        if len(self.memory_cache) > self.config.max_memory_cache_size:
            # Sort by expiry time and remove oldest
            sorted_items = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1][1]
            )
            
            excess_count = len(self.memory_cache) - self.config.max_memory_cache_size
            for key, _ in sorted_items[:excess_count]:
                del self.memory_cache[key]
    
    # Document-specific cache methods
    
    def cache_document(
        self,
        document_id: str,
        document_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache document data"""
        key = self._generate_key(self.config.document_prefix, document_id)
        return self.set(key, document_data, ttl)
    
    def get_cached_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get cached document data"""
        key = self._generate_key(self.config.document_prefix, document_id)
        return self.get(key)
    
    def cache_document_content(
        self,
        document_id: str,
        content_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache document content"""
        key = self._generate_key(self.config.content_prefix, document_id)
        return self.set(key, content_data, ttl)
    
    def get_cached_content(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get cached document content"""
        key = self._generate_key(self.config.content_prefix, document_id)
        return self.get(key)
    
    def cache_validation_result(
        self,
        document_id: str,
        validation_type: str,
        result_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache validation result"""
        key = self._generate_key(
            self.config.validation_prefix,
            f"{document_id}:{validation_type}"
        )
        return self.set(key, result_data, ttl)
    
    def get_cached_validation(
        self,
        document_id: str,
        validation_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached validation result"""
        key = self._generate_key(
            self.config.validation_prefix,
            f"{document_id}:{validation_type}"
        )
        return self.get(key)
    
    def cache_generation_result(
        self,
        config_hash: str,
        result_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache generation result by configuration hash"""
        key = self._generate_key(self.config.generation_prefix, config_hash)
        return self.set(key, result_data, ttl)
    
    def get_cached_generation(self, config_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached generation result"""
        key = self._generate_key(self.config.generation_prefix, config_hash)
        return self.get(key)
    
    def hash_generation_config(self, config: Dict[str, Any]) -> str:
        """Generate hash for generation configuration"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def invalidate_document(self, document_id: str) -> bool:
        """Invalidate all cache entries for a document"""
        success = True
        
        # Document cache
        doc_key = self._generate_key(self.config.document_prefix, document_id)
        success &= self.delete(doc_key)
        
        # Content cache
        content_key = self._generate_key(self.config.content_prefix, document_id)
        success &= self.delete(content_key)
        
        # Validation caches (need to find all validation types)
        if self.redis_client:
            try:
                pattern = self._generate_key(self.config.validation_prefix, f"{document_id}:*")
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            except RedisError as e:
                logger.warning(f"Failed to invalidate validation cache: {str(e)}")
                success = False
        
        return success
    
    def clear_all(self) -> bool:
        """Clear all cache entries"""
        success = True
        
        # Clear Redis
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except RedisError as e:
                logger.warning(f"Failed to clear Redis cache: {str(e)}")
                success = False
        
        # Clear memory cache
        self.memory_cache.clear()
        
        logger.info("Cleared all cache entries")
        return success
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "memory_cache_size": len(self.memory_cache),
            "memory_cache_max_size": self.config.max_memory_cache_size,
            "redis_connected": self.redis_client is not None
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats.update({
                    "redis_used_memory": info.get("used_memory_human"),
                    "redis_connected_clients": info.get("connected_clients"),
                    "redis_total_commands": info.get("total_commands_processed")
                })
            except RedisError as e:
                logger.warning(f"Failed to get Redis stats: {str(e)}")
        
        return stats


def create_cache_manager(
    config: Optional[Union[Dict[str, Any], CacheManagerConfig]] = None
) -> CacheManager:
    """Factory function to create cache manager"""
    if isinstance(config, dict):
        config = CacheManagerConfig(**config)
    return CacheManager(config)