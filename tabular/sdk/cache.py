"""
Caching mechanism for synthetic data generation
"""

import os
import json
import hashlib
import pickle
import tempfile
import shutil
from typing import Optional, Dict, Any, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
from dataclasses import dataclass, field
from functools import wraps

import pandas as pd


@dataclass
class CacheEntry:
    """Cache entry metadata"""
    key: str
    file_path: str
    created_at: datetime
    expires_at: Optional[datetime]
    size_bytes: int
    hits: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'key': self.key,
            'file_path': self.file_path,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'size_bytes': self.size_bytes,
            'hits': self.hits,
            'metadata': self.metadata
        }


class CacheBackend:
    """Abstract cache backend interface"""
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        raise NotImplementedError
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        raise NotImplementedError
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        raise NotImplementedError
    
    def clear(self) -> int:
        """Clear all cache entries"""
        raise NotImplementedError


class FileSystemCache(CacheBackend):
    """File system based cache backend"""
    
    def __init__(self, cache_dir: Optional[str] = None, max_size_mb: int = 1024):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "inferloop_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.index_file = self.cache_dir / "cache_index.json"
        self.lock = threading.Lock()
        self._load_index()
    
    def _load_index(self):
        """Load cache index from disk"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                index_data = json.load(f)
                self.index = {
                    k: CacheEntry(**{**v, 'created_at': datetime.fromisoformat(v['created_at']),
                                   'expires_at': datetime.fromisoformat(v['expires_at']) if v['expires_at'] else None})
                    for k, v in index_data.items()
                }
        else:
            self.index = {}
    
    def _save_index(self):
        """Save cache index to disk"""
        with open(self.index_file, 'w') as f:
            json.dump({k: v.to_dict() for k, v in self.index.items()}, f)
    
    def _generate_key(self, key: str) -> str:
        """Generate cache key hash"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key_hash: str) -> Path:
        """Get file path for cache key"""
        return self.cache_dir / f"{key_hash}.cache"
    
    def _evict_if_needed(self, required_bytes: int):
        """Evict old entries if cache size exceeds limit"""
        current_size = sum(entry.size_bytes for entry in self.index.values())
        
        if current_size + required_bytes > self.max_size_bytes:
            # Sort by last access time (created_at + hits as proxy)
            sorted_entries = sorted(
                self.index.items(),
                key=lambda x: (x[1].created_at, x[1].hits)
            )
            
            # Evict until we have enough space
            for key, entry in sorted_entries:
                if current_size + required_bytes <= self.max_size_bytes:
                    break
                
                self.delete(key)
                current_size -= entry.size_bytes
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            key_hash = self._generate_key(key)
            
            if key_hash not in self.index:
                return None
            
            entry = self.index[key_hash]
            
            # Check expiration
            if entry.is_expired():
                self.delete(key)
                return None
            
            # Load from file
            cache_path = Path(entry.file_path)
            if not cache_path.exists():
                del self.index[key_hash]
                self._save_index()
                return None
            
            try:
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)
                
                # Update hits
                entry.hits += 1
                self._save_index()
                
                return value
            except Exception:
                # Corrupted cache entry
                self.delete(key)
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        with self.lock:
            key_hash = self._generate_key(key)
            cache_path = self._get_cache_path(key_hash)
            
            try:
                # Serialize value
                serialized = pickle.dumps(value)
                size_bytes = len(serialized)
                
                # Check if we need to evict
                self._evict_if_needed(size_bytes)
                
                # Write to file
                with open(cache_path, 'wb') as f:
                    f.write(serialized)
                
                # Update index
                expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None
                
                self.index[key_hash] = CacheEntry(
                    key=key,
                    file_path=str(cache_path),
                    created_at=datetime.now(),
                    expires_at=expires_at,
                    size_bytes=size_bytes
                )
                
                self._save_index()
                return True
                
            except Exception:
                return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            key_hash = self._generate_key(key)
            
            if key_hash not in self.index:
                return False
            
            entry = self.index[key_hash]
            cache_path = Path(entry.file_path)
            
            # Delete file
            if cache_path.exists():
                cache_path.unlink()
            
            # Update index
            del self.index[key_hash]
            self._save_index()
            
            return True
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        with self.lock:
            key_hash = self._generate_key(key)
            
            if key_hash not in self.index:
                return False
            
            entry = self.index[key_hash]
            return not entry.is_expired() and Path(entry.file_path).exists()
    
    def clear(self) -> int:
        """Clear all cache entries"""
        with self.lock:
            count = len(self.index)
            
            # Delete all cache files
            for entry in self.index.values():
                cache_path = Path(entry.file_path)
                if cache_path.exists():
                    cache_path.unlink()
            
            # Clear index
            self.index = {}
            self._save_index()
            
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_size = sum(entry.size_bytes for entry in self.index.values())
            total_hits = sum(entry.hits for entry in self.index.values())
            
            return {
                'entries': len(self.index),
                'total_size_mb': total_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization': (total_size / self.max_size_bytes) * 100,
                'total_hits': total_hits,
                'avg_hits_per_entry': total_hits / len(self.index) if self.index else 0
            }


class MemoryCache(CacheBackend):
    """In-memory cache backend"""
    
    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.cache: Dict[str, tuple[Any, Optional[datetime]]] = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                return None
            
            value, expires_at = self.cache[key]
            
            # Check expiration
            if expires_at and datetime.now() > expires_at:
                del self.cache[key]
                return None
            
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        with self.lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_entries and key not in self.cache:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None
            self.cache[key] = (value, expires_at)
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        return self.get(key) is not None
    
    def clear(self) -> int:
        """Clear all cache entries"""
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            return count


class SyntheticDataCache:
    """High-level caching for synthetic data generation"""
    
    def __init__(self, backend: Optional[CacheBackend] = None, 
                 default_ttl: int = 3600,
                 cache_generated_data: bool = True,
                 cache_models: bool = True):
        self.backend = backend or FileSystemCache()
        self.default_ttl = default_ttl
        self.cache_generated_data = cache_generated_data
        self.cache_models = cache_models
    
    def _create_data_key(self, config: Dict[str, Any], data_hash: str) -> str:
        """Create cache key for synthetic data"""
        config_str = json.dumps(config, sort_keys=True)
        return f"data:{data_hash}:{hashlib.md5(config_str.encode()).hexdigest()}"
    
    def _create_model_key(self, generator_type: str, model_type: str, data_hash: str) -> str:
        """Create cache key for trained model"""
        return f"model:{generator_type}:{model_type}:{data_hash}"
    
    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Create hash of dataframe"""
        # Use shape and sample of data for hashing
        shape_str = f"{df.shape[0]}x{df.shape[1]}"
        columns_str = ",".join(sorted(df.columns))
        
        # Sample some data for hashing (first, middle, last rows)
        sample_indices = [0, len(df)//2, -1]
        sample_data = df.iloc[sample_indices].to_json()
        
        combined = f"{shape_str}:{columns_str}:{sample_data}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_synthetic_data(self, config: Dict[str, Any], 
                          original_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get cached synthetic data if available"""
        if not self.cache_generated_data:
            return None
        
        data_hash = self._hash_dataframe(original_data)
        cache_key = self._create_data_key(config, data_hash)
        
        return self.backend.get(cache_key)
    
    def set_synthetic_data(self, config: Dict[str, Any], 
                          original_data: pd.DataFrame,
                          synthetic_data: pd.DataFrame,
                          ttl: Optional[int] = None) -> bool:
        """Cache synthetic data"""
        if not self.cache_generated_data:
            return False
        
        data_hash = self._hash_dataframe(original_data)
        cache_key = self._create_data_key(config, data_hash)
        
        return self.backend.set(cache_key, synthetic_data, ttl or self.default_ttl)
    
    def get_model(self, generator_type: str, model_type: str,
                  original_data: pd.DataFrame) -> Optional[Any]:
        """Get cached trained model"""
        if not self.cache_models:
            return None
        
        data_hash = self._hash_dataframe(original_data)
        cache_key = self._create_model_key(generator_type, model_type, data_hash)
        
        return self.backend.get(cache_key)
    
    def set_model(self, generator_type: str, model_type: str,
                  original_data: pd.DataFrame, model: Any,
                  ttl: Optional[int] = None) -> bool:
        """Cache trained model"""
        if not self.cache_models:
            return False
        
        data_hash = self._hash_dataframe(original_data)
        cache_key = self._create_model_key(generator_type, model_type, data_hash)
        
        # Models typically have longer TTL
        model_ttl = ttl or (self.default_ttl * 24)  # 24 hours by default
        
        return self.backend.set(cache_key, model, model_ttl)
    
    def invalidate_data_cache(self, config: Dict[str, Any], 
                             original_data: pd.DataFrame) -> bool:
        """Invalidate cached synthetic data"""
        data_hash = self._hash_dataframe(original_data)
        cache_key = self._create_data_key(config, data_hash)
        
        return self.backend.delete(cache_key)
    
    def invalidate_model_cache(self, generator_type: str, model_type: str,
                              original_data: pd.DataFrame) -> bool:
        """Invalidate cached model"""
        data_hash = self._hash_dataframe(original_data)
        cache_key = self._create_model_key(generator_type, model_type, data_hash)
        
        return self.backend.delete(cache_key)
    
    def clear_all(self) -> int:
        """Clear all cache entries"""
        return self.backend.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if hasattr(self.backend, 'get_stats'):
            return self.backend.get_stats()
        return {}


def cached_generation(cache: Optional[SyntheticDataCache] = None, 
                     ttl: Optional[int] = None):
    """Decorator for caching synthetic data generation results"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, data: pd.DataFrame, *args, **kwargs):
            # Use provided cache or create default
            _cache = cache or getattr(self, '_cache', None)
            
            if _cache and hasattr(self, 'config'):
                # Try to get from cache
                config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config
                cached_result = _cache.get_synthetic_data(config_dict, data)
                
                if cached_result is not None:
                    # Return cached result wrapped in GenerationResult
                    from .base import GenerationResult
                    return GenerationResult(
                        synthetic_data=cached_result,
                        metadata={'cached': True},
                        validation_metrics={}
                    )
            
            # Generate new data
            result = func(self, data, *args, **kwargs)
            
            # Cache the result
            if _cache and hasattr(self, 'config'):
                config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else self.config
                _cache.set_synthetic_data(config_dict, data, result.synthetic_data, ttl)
            
            return result
        
        return wrapper
    return decorator


def cached_model_training(cache: Optional[SyntheticDataCache] = None,
                         ttl: Optional[int] = None):
    """Decorator for caching model training"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, data: pd.DataFrame, *args, **kwargs):
            # Use provided cache or create default
            _cache = cache or getattr(self, '_cache', None)
            
            if _cache and hasattr(self, 'generator_type') and hasattr(self, 'model_type'):
                # Try to get cached model
                cached_model = _cache.get_model(
                    self.generator_type, 
                    self.model_type, 
                    data
                )
                
                if cached_model is not None:
                    # Restore model state
                    self.model = cached_model
                    self.is_fitted = True
                    return self
            
            # Train new model
            result = func(self, data, *args, **kwargs)
            
            # Cache the trained model
            if _cache and hasattr(self, 'model'):
                _cache.set_model(
                    self.generator_type,
                    self.model_type,
                    data,
                    self.model,
                    ttl
                )
            
            return result
        
        return wrapper
    return decorator


# Global cache instance
_global_cache = None

def get_cache() -> SyntheticDataCache:
    """Get global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = SyntheticDataCache()
    return _global_cache

def set_cache(cache: SyntheticDataCache):
    """Set global cache instance"""
    global _global_cache
    _global_cache = cache