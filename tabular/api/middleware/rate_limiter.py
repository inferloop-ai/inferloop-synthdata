"""
Rate limiting middleware for API endpoints
"""

from typing import Dict, Optional, Callable, Any
from datetime import datetime, timedelta
from functools import wraps
import asyncio
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse


class RateLimiter:
    """Token bucket rate limiter implementation"""
    
    def __init__(self, requests_per_minute: int = 60, 
                 burst_size: Optional[int] = None,
                 storage_backend: Optional[Any] = None):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or requests_per_minute
        self.storage_backend = storage_backend
        
        # In-memory storage (for development)
        self.buckets: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'tokens': self.burst_size,
            'last_update': time.time(),
            'request_history': deque(maxlen=100)
        })
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request"""
        # Try to get from authenticated user
        if hasattr(request.state, 'user') and request.state.user:
            return f"user:{request.state.user.id}"
        
        # Try to get from API key
        if hasattr(request.state, 'api_key') and request.state.api_key:
            return f"api_key:{request.state.api_key.id}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    async def _refill_tokens(self, bucket: Dict[str, Any]) -> None:
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - bucket['last_update']
        
        # Calculate tokens to add
        tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
        bucket['tokens'] = min(self.burst_size, bucket['tokens'] + tokens_to_add)
        bucket['last_update'] = now
    
    async def check_rate_limit(self, request: Request) -> bool:
        """Check if request is within rate limits"""
        client_id = self._get_client_id(request)
        
        async with self._lock:
            bucket = self.buckets[client_id]
            
            # Refill tokens
            await self._refill_tokens(bucket)
            
            # Check if tokens available
            if bucket['tokens'] >= 1:
                bucket['tokens'] -= 1
                bucket['request_history'].append({
                    'timestamp': datetime.utcnow(),
                    'endpoint': str(request.url),
                    'method': request.method
                })
                return True
            
            return False
    
    async def get_rate_limit_info(self, request: Request) -> Dict[str, Any]:
        """Get current rate limit information for client"""
        client_id = self._get_client_id(request)
        
        async with self._lock:
            bucket = self.buckets[client_id]
            await self._refill_tokens(bucket)
            
            return {
                'limit': self.requests_per_minute,
                'remaining': int(bucket['tokens']),
                'reset': int(time.time() + (60 - (time.time() - bucket['last_update']))),
                'burst_size': self.burst_size
            }
    
    async def __call__(self, request: Request, call_next: Callable) -> Any:
        """Middleware to enforce rate limits"""
        # Skip rate limiting for certain paths
        skip_paths = ['/health', '/docs', '/openapi.json', '/favicon.ico']
        if request.url.path in skip_paths:
            return await call_next(request)
        
        # Check rate limit
        if not await self.check_rate_limit(request):
            # Get rate limit info for headers
            info = await self.get_rate_limit_info(request)
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "limit": info['limit'],
                    "reset": info['reset']
                },
                headers={
                    "X-RateLimit-Limit": str(info['limit']),
                    "X-RateLimit-Remaining": str(info['remaining']),
                    "X-RateLimit-Reset": str(info['reset']),
                    "Retry-After": str(info['reset'] - int(time.time()))
                }
            )
        
        # Process request and add rate limit headers
        response = await call_next(request)
        
        # Add rate limit headers
        info = await self.get_rate_limit_info(request)
        response.headers["X-RateLimit-Limit"] = str(info['limit'])
        response.headers["X-RateLimit-Remaining"] = str(info['remaining'])
        response.headers["X-RateLimit-Reset"] = str(info['reset'])
        
        return response


class EndpointRateLimiter:
    """Rate limiter for specific endpoints with custom limits"""
    
    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
    
    def add_endpoint(self, path: str, requests_per_minute: int, 
                    burst_size: Optional[int] = None):
        """Add rate limit for specific endpoint"""
        self.limiters[path] = RateLimiter(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size
        )
    
    async def check_endpoint_limit(self, request: Request) -> bool:
        """Check rate limit for specific endpoint"""
        path = request.url.path
        
        # Check if endpoint has specific limit
        if path in self.limiters:
            return await self.limiters[path].check_rate_limit(request)
        
        # Check pattern matching (e.g., /api/v1/*)
        for pattern, limiter in self.limiters.items():
            if path.startswith(pattern):
                return await limiter.check_rate_limit(request)
        
        return True


def rate_limit(requests_per_minute: int = 60, burst_size: Optional[int] = None):
    """Decorator for rate limiting specific endpoints"""
    limiter = RateLimiter(requests_per_minute=requests_per_minute, burst_size=burst_size)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            if not await limiter.check_rate_limit(request):
                info = await limiter.get_rate_limit_info(request)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Limit": str(info['limit']),
                        "X-RateLimit-Remaining": str(info['remaining']),
                        "X-RateLimit-Reset": str(info['reset']),
                        "Retry-After": str(info['reset'] - int(time.time()))
                    }
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator


# Global rate limiter instances
default_rate_limiter = RateLimiter(requests_per_minute=60)
endpoint_rate_limiter = EndpointRateLimiter()

# Configure endpoint-specific limits
endpoint_rate_limiter.add_endpoint("/generate", requests_per_minute=10, burst_size=5)
endpoint_rate_limiter.add_endpoint("/generate/async", requests_per_minute=5, burst_size=3)
endpoint_rate_limiter.add_endpoint("/validate", requests_per_minute=20, burst_size=10)