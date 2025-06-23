"""
Cache management API endpoints
"""

from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from sdk.cache import get_cache, FileSystemCache, MemoryCache, SyntheticDataCache
from api.deps import get_current_user
from api.auth.models import UserRole

router = APIRouter(prefix="/cache", tags=["cache"])


class CacheConfig(BaseModel):
    """Cache configuration model"""
    backend: str = Field(default="filesystem", regex="^(filesystem|memory)$")
    cache_dir: Optional[str] = None
    max_size_mb: int = Field(default=1024, ge=1)
    default_ttl: int = Field(default=3600, ge=60)
    cache_generated_data: bool = True
    cache_models: bool = True


class CacheStats(BaseModel):
    """Cache statistics model"""
    backend_type: str
    entries: int
    total_size_mb: float
    max_size_mb: float
    utilization_percentage: float
    total_hits: int
    avg_hits_per_entry: float


class CacheEntry(BaseModel):
    """Cache entry information"""
    key: str
    created_at: datetime
    expires_at: Optional[datetime]
    size_bytes: int
    hits: int


@router.get("/stats", response_model=CacheStats)
async def get_cache_stats(current_user = Depends(get_current_user)):
    """Get cache statistics"""
    cache = get_cache()
    
    stats = cache.get_stats()
    
    # Get backend type
    backend_type = "unknown"
    if isinstance(cache.backend, FileSystemCache):
        backend_type = "filesystem"
    elif isinstance(cache.backend, MemoryCache):
        backend_type = "memory"
    
    return CacheStats(
        backend_type=backend_type,
        entries=stats.get('entries', 0),
        total_size_mb=stats.get('total_size_mb', 0),
        max_size_mb=stats.get('max_size_mb', 0),
        utilization_percentage=stats.get('utilization', 0),
        total_hits=stats.get('total_hits', 0),
        avg_hits_per_entry=stats.get('avg_hits_per_entry', 0)
    )


@router.post("/clear")
async def clear_cache(
    pattern: Optional[str] = Query(None, description="Clear only keys matching pattern"),
    current_user = Depends(get_current_user)
):
    """Clear cache entries"""
    # Only admins can clear cache
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    cache = get_cache()
    
    if pattern:
        # Clear specific pattern (not implemented in basic backend)
        cleared = 0
        if hasattr(cache.backend, 'index'):
            # FileSystemCache specific
            keys_to_delete = []
            for key_hash, entry in cache.backend.index.items():
                if pattern in entry.key:
                    keys_to_delete.append(entry.key)
            
            for key in keys_to_delete:
                if cache.backend.delete(key):
                    cleared += 1
        
        return {"message": f"Cleared {cleared} cache entries matching pattern '{pattern}'"}
    else:
        # Clear all
        cleared = cache.clear_all()
        return {"message": f"Cleared {cleared} cache entries"}


@router.post("/configure")
async def configure_cache(
    config: CacheConfig,
    current_user = Depends(get_current_user)
):
    """Configure cache settings"""
    # Only admins can configure cache
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Create new backend based on config
    if config.backend == "filesystem":
        backend = FileSystemCache(
            cache_dir=config.cache_dir,
            max_size_mb=config.max_size_mb
        )
    elif config.backend == "memory":
        # Estimate max entries based on average entry size
        max_entries = (config.max_size_mb * 1024) // 100  # Assume 100KB average
        backend = MemoryCache(max_entries=max_entries)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown backend: {config.backend}")
    
    # Create new cache with backend
    new_cache = SyntheticDataCache(
        backend=backend,
        default_ttl=config.default_ttl,
        cache_generated_data=config.cache_generated_data,
        cache_models=config.cache_models
    )
    
    # Set as global cache
    from sdk.cache import set_cache
    set_cache(new_cache)
    
    return {"message": "Cache configured successfully", "config": config}


@router.get("/entries")
async def list_cache_entries(
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    current_user = Depends(get_current_user)
):
    """List cache entries"""
    cache = get_cache()
    
    entries = []
    
    if hasattr(cache.backend, 'index'):
        # FileSystemCache specific
        all_entries = list(cache.backend.index.values())
        
        # Sort by creation time (newest first)
        all_entries.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        paginated = all_entries[offset:offset + limit]
        
        for entry in paginated:
            entries.append(CacheEntry(
                key=entry.key,
                created_at=entry.created_at,
                expires_at=entry.expires_at,
                size_bytes=entry.size_bytes,
                hits=entry.hits
            ))
        
        return {
            "entries": entries,
            "total": len(all_entries),
            "limit": limit,
            "offset": offset
        }
    else:
        # Backend doesn't support listing
        return {
            "entries": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "message": "Backend doesn't support entry listing"
        }


@router.delete("/entry/{key}")
async def delete_cache_entry(
    key: str,
    current_user = Depends(get_current_user)
):
    """Delete specific cache entry"""
    # Only admins can delete entries
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    cache = get_cache()
    
    # Try different key formats
    deleted = False
    
    # Try as data key
    if cache.backend.delete(key):
        deleted = True
    
    # Try as model key
    if not deleted and cache.backend.exists(f"model:{key}"):
        deleted = cache.backend.delete(f"model:{key}")
    
    # Try as data key with prefix
    if not deleted and cache.backend.exists(f"data:{key}"):
        deleted = cache.backend.delete(f"data:{key}")
    
    if deleted:
        return {"message": f"Cache entry '{key}' deleted"}
    else:
        raise HTTPException(status_code=404, detail="Cache entry not found")


@router.post("/warmup")
async def warmup_cache(
    generator_type: str,
    model_type: str,
    sample_data_path: Optional[str] = None,
    current_user = Depends(get_current_user)
):
    """Warm up cache with pre-trained models"""
    # This would be implemented to pre-load commonly used models
    return {
        "message": "Cache warmup initiated",
        "generator_type": generator_type,
        "model_type": model_type,
        "status": "not_implemented"
    }


@router.get("/config")
async def get_cache_config(current_user = Depends(get_current_user)):
    """Get current cache configuration"""
    cache = get_cache()
    
    config = {
        "backend_type": type(cache.backend).__name__,
        "default_ttl": cache.default_ttl,
        "cache_generated_data": cache.cache_generated_data,
        "cache_models": cache.cache_models
    }
    
    if hasattr(cache.backend, 'cache_dir'):
        config["cache_dir"] = str(cache.backend.cache_dir)
    
    if hasattr(cache.backend, 'max_size_bytes'):
        config["max_size_mb"] = cache.backend.max_size_bytes / (1024 * 1024)
    
    return config