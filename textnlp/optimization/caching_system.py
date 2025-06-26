"""
Advanced Caching System for TextNLP
Multi-level caching with intelligent cache management, prefetching, and optimization
"""

import asyncio
import hashlib
import logging
import pickle
import time
import threading
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
from pathlib import Path
import json
import sqlite3
import redis
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torch
import psutil

try:
    import memcache
    HAS_MEMCACHED = True
except ImportError:
    HAS_MEMCACHED = False

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels in the hierarchy"""
    L1_MEMORY = "l1_memory"  # In-process memory cache
    L2_REDIS = "l2_redis"  # Redis cache
    L3_DISK = "l3_disk"  # Disk-based cache
    L4_DISTRIBUTED = "l4_distributed"  # Distributed cache


class CacheType(Enum):
    """Types of cached content"""
    GENERATED_TEXT = "generated_text"
    MODEL_OUTPUTS = "model_outputs"
    EMBEDDINGS = "embeddings"
    PREPROCESSED_INPUTS = "preprocessed_inputs"
    SAFETY_RESULTS = "safety_results"
    QUALITY_METRICS = "quality_metrics"
    MODEL_WEIGHTS = "model_weights"
    KV_CACHE = "kv_cache"


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns


@dataclass
class CacheConfig:
    """Configuration for caching system"""
    # Cache levels to enable
    enabled_levels: List[CacheLevel] = field(default_factory=lambda: [
        CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DISK
    ])
    
    # Cache sizes (in MB or number of items)
    l1_max_size: int = 512  # MB
    l2_max_size: int = 2048  # MB
    l3_max_size: int = 10240  # MB
    
    # Cache TTL (seconds)
    default_ttl: int = 3600  # 1 hour
    
    # Eviction policies
    l1_eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    l2_eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    l3_eviction_policy: EvictionPolicy = EvictionPolicy.TTL
    
    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Disk cache configuration
    disk_cache_dir: str = "cache"
    
    # Performance settings
    enable_compression: bool = True
    enable_prefetching: bool = True
    prefetch_workers: int = 2
    enable_warming: bool = True
    
    # Advanced features
    enable_cache_analytics: bool = True
    enable_adaptive_caching: bool = True
    cache_hit_threshold: float = 0.8  # Threshold for cache effectiveness


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    cache_type: CacheType
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get age in seconds"""
        return time.time() - self.created_at


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class InMemoryCache:
    """L1 in-memory cache with configurable eviction"""
    
    def __init__(self, max_size_mb: int, eviction_policy: EvictionPolicy):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.data = OrderedDict() if eviction_policy == EvictionPolicy.LRU else {}
        self.access_counts = defaultdict(int)
        self.lock = threading.RLock()
        self.stats = CacheStats()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from cache"""
        with self.lock:
            if key in self.data:
                entry = self.data[key]
                
                # Check expiration
                if entry.is_expired:
                    self._remove_entry(key)
                    self.stats.misses += 1
                    return None
                
                # Update access info
                entry.last_accessed = time.time()
                entry.access_count += 1
                self.access_counts[key] += 1
                
                # Move to end for LRU
                if self.eviction_policy == EvictionPolicy.LRU:
                    self.data.move_to_end(key)
                
                self.stats.hits += 1
                return entry
            else:
                self.stats.misses += 1
                return None
    
    def put(self, entry: CacheEntry):
        """Put item in cache"""
        with self.lock:
            # Check if we need to evict
            entry_size = entry.size_bytes
            
            while self._get_total_size() + entry_size > self.max_size_bytes:
                if not self._evict_one():
                    break  # Can't evict more
            
            # Add entry
            self.data[entry.key] = entry
            self.stats.entry_count += 1
            self.stats.size_bytes += entry_size
    
    def _evict_one(self) -> bool:
        """Evict one entry based on policy"""
        if not self.data:
            return False
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used (first in OrderedDict)
            key = next(iter(self.data))
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Remove expired entries first, then oldest
            expired_keys = [k for k, v in self.data.items() if v.is_expired]
            if expired_keys:
                key = expired_keys[0]
            else:
                key = min(self.data.keys(), key=lambda k: self.data[k].created_at)
        else:  # FIFO
            key = next(iter(self.data))
        
        self._remove_entry(key)
        return True
    
    def _remove_entry(self, key: str):
        """Remove entry from cache"""
        if key in self.data:
            entry = self.data.pop(key)
            self.stats.size_bytes -= entry.size_bytes
            self.stats.entry_count -= 1
            self.stats.evictions += 1
            if key in self.access_counts:
                del self.access_counts[key]
    
    def _get_total_size(self) -> int:
        """Get total cache size"""
        return self.stats.size_bytes
    
    def clear(self):
        """Clear all entries"""
        with self.lock:
            self.data.clear()
            self.access_counts.clear()
            self.stats = CacheStats()


class RedisCache:
    """L2 Redis cache"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.stats = CacheStats()
        
        try:
            self.redis_client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                db=config.redis_db,
                password=config.redis_password,
                decode_responses=False  # We handle encoding ourselves
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache connected")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from Redis"""
        if not self.redis_client:
            return None
        
        try:
            data = self.redis_client.get(key)
            if data:
                entry = pickle.loads(data)
                
                # Check expiration
                if entry.is_expired:
                    self.redis_client.delete(key)
                    self.stats.misses += 1
                    return None
                
                # Update access info
                entry.last_accessed = time.time()
                entry.access_count += 1
                
                # Update in Redis
                self.redis_client.set(key, pickle.dumps(entry))
                
                self.stats.hits += 1
                return entry
            else:
                self.stats.misses += 1
                return None
                
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.stats.misses += 1
            return None
    
    def put(self, entry: CacheEntry):
        """Put item in Redis"""
        if not self.redis_client:
            return
        
        try:
            data = pickle.dumps(entry)
            
            if entry.ttl:
                self.redis_client.setex(entry.key, entry.ttl, data)
            else:
                self.redis_client.set(entry.key, data)
            
            self.stats.entry_count += 1
            self.stats.size_bytes += len(data)
            
        except Exception as e:
            logger.error(f"Redis put error: {e}")
    
    def clear(self):
        """Clear Redis cache"""
        if self.redis_client:
            self.redis_client.flushdb()
            self.stats = CacheStats()


class DiskCache:
    """L3 disk-based cache"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = Path(config.disk_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache_metadata.db"
        self.stats = CacheStats()
        
        # Initialize SQLite database for metadata
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for cache metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    cache_type TEXT,
                    file_path TEXT,
                    size_bytes INTEGER,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER,
                    ttl INTEGER,
                    metadata TEXT
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)")
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from disk cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if not row:
                    self.stats.misses += 1
                    return None
                
                # Parse row
                (key, cache_type, file_path, size_bytes, created_at, 
                 last_accessed, access_count, ttl, metadata_json) = row
                
                # Check expiration
                if ttl and time.time() > created_at + ttl:
                    self._remove_entry(key)
                    self.stats.misses += 1
                    return None
                
                # Load data from file
                file_full_path = self.cache_dir / file_path
                if not file_full_path.exists():
                    self._remove_entry(key)
                    self.stats.misses += 1
                    return None
                
                with open(file_full_path, 'rb') as f:
                    value = pickle.load(f)
                
                # Create entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    cache_type=CacheType(cache_type),
                    size_bytes=size_bytes,
                    created_at=created_at,
                    last_accessed=time.time(),
                    access_count=access_count + 1,
                    ttl=ttl,
                    metadata=json.loads(metadata_json) if metadata_json else {}
                )
                
                # Update access info in database
                conn.execute(
                    "UPDATE cache_entries SET last_accessed = ?, access_count = ? WHERE key = ?",
                    (entry.last_accessed, entry.access_count, key)
                )
                
                self.stats.hits += 1
                return entry
                
        except Exception as e:
            logger.error(f"Disk cache get error: {e}")
            self.stats.misses += 1
            return None
    
    def put(self, entry: CacheEntry):
        """Put item in disk cache"""
        try:
            # Generate file path
            file_name = hashlib.md5(entry.key.encode()).hexdigest() + ".pkl"
            file_path = self.cache_dir / file_name
            
            # Save data to file
            with open(file_path, 'wb') as f:
                pickle.dump(entry.value, f)
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, cache_type, file_path, size_bytes, created_at, last_accessed, 
                     access_count, ttl, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.key,
                    entry.cache_type.value,
                    file_name,
                    entry.size_bytes,
                    entry.created_at,
                    entry.last_accessed,
                    entry.access_count,
                    entry.ttl,
                    json.dumps(entry.metadata)
                ))
            
            self.stats.entry_count += 1
            self.stats.size_bytes += entry.size_bytes
            
            # Check if we need to evict
            self._evict_if_needed()
            
        except Exception as e:
            logger.error(f"Disk cache put error: {e}")
    
    def _remove_entry(self, key: str):
        """Remove entry from disk cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get file path
                cursor = conn.execute(
                    "SELECT file_path, size_bytes FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    file_path, size_bytes = row
                    
                    # Remove file
                    file_full_path = self.cache_dir / file_path
                    if file_full_path.exists():
                        file_full_path.unlink()
                    
                    # Remove from database
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    
                    self.stats.entry_count -= 1
                    self.stats.size_bytes -= size_bytes
                    self.stats.evictions += 1
                    
        except Exception as e:
            logger.error(f"Disk cache remove error: {e}")
    
    def _evict_if_needed(self):
        """Evict entries if cache is too large"""
        max_size_bytes = self.config.l3_max_size * 1024 * 1024
        
        if self.stats.size_bytes <= max_size_bytes:
            return
        
        # Evict oldest entries
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT key FROM cache_entries ORDER BY created_at ASC"
                )
                
                for (key,) in cursor:
                    self._remove_entry(key)
                    if self.stats.size_bytes <= max_size_bytes:
                        break
                        
        except Exception as e:
            logger.error(f"Disk cache eviction error: {e}")
    
    def clear(self):
        """Clear disk cache"""
        try:
            # Remove all files
            for file_path in self.cache_dir.glob("*.pkl"):
                file_path.unlink()
            
            # Clear database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cache_entries")
            
            self.stats = CacheStats()
            
        except Exception as e:
            logger.error(f"Disk cache clear error: {e}")


class MultiLevelCache:
    """Multi-level cache system"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.caches = {}
        
        # Initialize cache levels
        if CacheLevel.L1_MEMORY in config.enabled_levels:
            self.caches[CacheLevel.L1_MEMORY] = InMemoryCache(
                config.l1_max_size, config.l1_eviction_policy
            )
            logger.info("L1 memory cache initialized")
        
        if CacheLevel.L2_REDIS in config.enabled_levels:
            self.caches[CacheLevel.L2_REDIS] = RedisCache(config)
            if self.caches[CacheLevel.L2_REDIS].redis_client:
                logger.info("L2 Redis cache initialized")
            else:
                del self.caches[CacheLevel.L2_REDIS]
        
        if CacheLevel.L3_DISK in config.enabled_levels:
            self.caches[CacheLevel.L3_DISK] = DiskCache(config)
            logger.info("L3 disk cache initialized")
        
        # Prefetching and warming
        self.prefetch_executor = ThreadPoolExecutor(max_workers=config.prefetch_workers)
        self.access_patterns = defaultdict(list)
        
        logger.info(f"Multi-level cache initialized with {len(self.caches)} levels")
    
    def get(self, key: str, cache_type: CacheType = CacheType.GENERATED_TEXT) -> Optional[Any]:
        """Get item from cache hierarchy"""
        
        # Try each cache level in order
        for level in [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DISK]:
            if level not in self.caches:
                continue
            
            cache = self.caches[level]
            entry = cache.get(key)
            
            if entry:
                # Promote to higher cache levels
                self._promote_entry(entry, level)
                
                # Record access pattern
                if self.config.enable_adaptive_caching:
                    self.access_patterns[key].append(time.time())
                
                return entry.value
        
        return None
    
    def put(self, key: str, value: Any, cache_type: CacheType = CacheType.GENERATED_TEXT,
            ttl: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None):
        """Put item in cache hierarchy"""
        
        # Calculate size
        try:
            if isinstance(value, torch.Tensor):
                size_bytes = value.numel() * value.element_size()
            elif isinstance(value, str):
                size_bytes = len(value.encode('utf-8'))
            else:
                size_bytes = len(pickle.dumps(value))
        except:
            size_bytes = 1024  # Default estimate
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            cache_type=cache_type,
            size_bytes=size_bytes,
            created_at=time.time(),
            last_accessed=time.time(),
            ttl=ttl or self.config.default_ttl,
            metadata=metadata or {}
        )
        
        # Store in all cache levels
        for level, cache in self.caches.items():
            try:
                cache.put(entry)
            except Exception as e:
                logger.error(f"Failed to store in {level.value}: {e}")
    
    def _promote_entry(self, entry: CacheEntry, found_level: CacheLevel):
        """Promote entry to higher cache levels"""
        
        higher_levels = []
        if found_level == CacheLevel.L3_DISK:
            higher_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
        elif found_level == CacheLevel.L2_REDIS:
            higher_levels = [CacheLevel.L1_MEMORY]
        
        for level in higher_levels:
            if level in self.caches:
                try:
                    self.caches[level].put(entry)
                except Exception as e:
                    logger.error(f"Failed to promote to {level.value}: {e}")
    
    def invalidate(self, key: str):
        """Invalidate key from all cache levels"""
        for cache in self.caches.values():
            if hasattr(cache, '_remove_entry'):
                cache._remove_entry(key)
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all cache levels"""
        stats = {}
        for level, cache in self.caches.items():
            stats[level.value] = cache.stats
        return stats
    
    def clear_all(self):
        """Clear all cache levels"""
        for cache in self.caches.values():
            cache.clear()
    
    def warm_cache(self, warm_data: List[Tuple[str, Any, CacheType]]):
        """Warm cache with frequently accessed data"""
        if not self.config.enable_warming:
            return
        
        logger.info(f"Warming cache with {len(warm_data)} entries")
        
        for key, value, cache_type in warm_data:
            self.put(key, value, cache_type)
    
    async def prefetch(self, keys: List[str], fetch_func: Callable[[str], Any]):
        """Prefetch data asynchronously"""
        if not self.config.enable_prefetching:
            return
        
        async def prefetch_key(key: str):
            if self.get(key) is None:  # Not in cache
                try:
                    value = await fetch_func(key)
                    if value is not None:
                        self.put(key, value)
                except Exception as e:
                    logger.error(f"Prefetch failed for {key}: {e}")
        
        tasks = [prefetch_key(key) for key in keys]
        await asyncio.gather(*tasks, return_exceptions=True)


class CacheManager:
    """High-level cache manager with intelligent features"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = MultiLevelCache(config)
        self.analytics = CacheAnalytics() if config.enable_cache_analytics else None
        
        # Smart caching features
        self.semantic_cache = SemanticCache() if config.enable_adaptive_caching else None
        
    def get_or_compute(self, key: str, compute_func: Callable[[], Any],
                      cache_type: CacheType = CacheType.GENERATED_TEXT,
                      ttl: Optional[int] = None) -> Any:
        """Get from cache or compute and store"""
        
        # Try cache first
        cached_value = self.cache.get(key, cache_type)
        if cached_value is not None:
            if self.analytics:
                self.analytics.record_hit(key, cache_type)
            return cached_value
        
        # Compute value
        try:
            value = compute_func()
            
            # Store in cache
            self.cache.put(key, value, cache_type, ttl)
            
            if self.analytics:
                self.analytics.record_miss(key, cache_type)
            
            return value
            
        except Exception as e:
            logger.error(f"Compute function failed for {key}: {e}")
            if self.analytics:
                self.analytics.record_error(key, cache_type, str(e))
            raise
    
    async def get_or_compute_async(self, key: str, compute_func: Callable[[], Any],
                                  cache_type: CacheType = CacheType.GENERATED_TEXT,
                                  ttl: Optional[int] = None) -> Any:
        """Async version of get_or_compute"""
        
        # Try cache first
        cached_value = self.cache.get(key, cache_type)
        if cached_value is not None:
            if self.analytics:
                self.analytics.record_hit(key, cache_type)
            return cached_value
        
        # Compute value
        try:
            if asyncio.iscoroutinefunction(compute_func):
                value = await compute_func()
            else:
                value = compute_func()
            
            # Store in cache
            self.cache.put(key, value, cache_type, ttl)
            
            if self.analytics:
                self.analytics.record_miss(key, cache_type)
            
            return value
            
        except Exception as e:
            logger.error(f"Async compute function failed for {key}: {e}")
            if self.analytics:
                self.analytics.record_error(key, cache_type, str(e))
            raise
    
    def generate_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def semantic_get(self, prompt: str, similarity_threshold: float = 0.9) -> Optional[Any]:
        """Get semantically similar cached result"""
        if not self.semantic_cache:
            return None
        
        return self.semantic_cache.get_similar(prompt, similarity_threshold)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive cache performance report"""
        stats = self.cache.get_stats()
        
        report = {
            "cache_levels": {},
            "overall_performance": {},
            "recommendations": []
        }
        
        total_hits = 0
        total_misses = 0
        
        for level, level_stats in stats.items():
            total_hits += level_stats.hits
            total_misses += level_stats.misses
            
            report["cache_levels"][level] = {
                "hit_rate": level_stats.hit_rate,
                "hits": level_stats.hits,
                "misses": level_stats.misses,
                "size_mb": level_stats.size_bytes / 1024 / 1024,
                "entry_count": level_stats.entry_count,
                "evictions": level_stats.evictions
            }
        
        overall_hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
        
        report["overall_performance"] = {
            "total_hit_rate": overall_hit_rate,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
        
        # Generate recommendations
        if overall_hit_rate < self.config.cache_hit_threshold:
            report["recommendations"].append("Consider increasing cache sizes")
        
        if "l1_memory" in stats and stats["l1_memory"].evictions > stats["l1_memory"].hits * 0.1:
            report["recommendations"].append("L1 cache experiencing high eviction rate")
        
        return report


class CacheAnalytics:
    """Cache analytics for performance monitoring"""
    
    def __init__(self):
        self.access_log = []
        self.hit_counts = defaultdict(int)
        self.miss_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.access_times = defaultdict(list)
    
    def record_hit(self, key: str, cache_type: CacheType):
        """Record cache hit"""
        self.hit_counts[cache_type.value] += 1
        self.access_times[key].append(time.time())
    
    def record_miss(self, key: str, cache_type: CacheType):
        """Record cache miss"""
        self.miss_counts[cache_type.value] += 1
    
    def record_error(self, key: str, cache_type: CacheType, error: str):
        """Record cache error"""
        self.error_counts[cache_type.value] += 1
    
    def get_analytics_report(self) -> Dict[str, Any]:
        """Get analytics report"""
        return {
            "hit_counts": dict(self.hit_counts),
            "miss_counts": dict(self.miss_counts),
            "error_counts": dict(self.error_counts),
            "total_accesses": sum(self.hit_counts.values()) + sum(self.miss_counts.values())
        }


class SemanticCache:
    """Semantic cache for similar prompts"""
    
    def __init__(self):
        self.embeddings = {}
        self.values = {}
        
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            logger.warning("Sentence transformers not available, semantic cache disabled")
            self.encoder = None
    
    def put(self, prompt: str, value: Any):
        """Store prompt and value with embedding"""
        if not self.encoder:
            return
        
        embedding = self.encoder.encode(prompt)
        key = hashlib.md5(prompt.encode()).hexdigest()
        self.embeddings[key] = embedding
        self.values[key] = value
    
    def get_similar(self, prompt: str, threshold: float = 0.9) -> Optional[Any]:
        """Get value for similar prompt"""
        if not self.encoder or not self.embeddings:
            return None
        
        query_embedding = self.encoder.encode(prompt)
        
        best_similarity = 0
        best_key = None
        
        for key, embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_key = key
        
        return self.values.get(best_key) if best_key else None


# Example usage
if __name__ == "__main__":
    # Configuration
    config = CacheConfig(
        enabled_levels=[CacheLevel.L1_MEMORY, CacheLevel.L3_DISK],
        l1_max_size=256,  # MB
        l3_max_size=1024,  # MB
        enable_prefetching=True,
        enable_cache_analytics=True
    )
    
    # Create cache manager
    cache_manager = CacheManager(config)
    
    # Example usage
    def expensive_computation(x):
        time.sleep(0.1)  # Simulate expensive operation
        return f"Result for {x}"
    
    # Test cache
    key = cache_manager.generate_cache_key("test", param=123)
    
    # First call - cache miss
    start_time = time.time()
    result1 = cache_manager.get_or_compute(
        key, 
        lambda: expensive_computation("test"),
        CacheType.GENERATED_TEXT
    )
    time1 = time.time() - start_time
    
    # Second call - cache hit
    start_time = time.time()
    result2 = cache_manager.get_or_compute(
        key,
        lambda: expensive_computation("test"),
        CacheType.GENERATED_TEXT
    )
    time2 = time.time() - start_time
    
    print(f"First call (miss): {time1:.3f}s - {result1}")
    print(f"Second call (hit): {time2:.3f}s - {result2}")
    print(f"Speedup: {time1/time2:.1f}x")
    
    # Performance report
    report = cache_manager.get_performance_report()
    print(f"\nCache Performance:")
    print(f"Overall hit rate: {report['overall_performance']['total_hit_rate']:.2%}")
    print(f"Total hits: {report['overall_performance']['total_hits']}")
    print(f"Total misses: {report['overall_performance']['total_misses']}")