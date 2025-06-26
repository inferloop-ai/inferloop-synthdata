"""
Rate Limiting Module

Provides unified rate limiting for all Inferloop services.
Supports token bucket, sliding window, and fixed window algorithms.
"""

import os
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import hashlib

from fastapi import Request, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from .cache import CacheClient, get_cache_client_sync
from .auth import User


class RateLimitAlgorithm(Enum):
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""
    limit: int
    window: int  # seconds
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    burst_limit: Optional[int] = None
    identifier: str = "ip"  # ip, user, api_key


@dataclass
class RateLimitResult:
    """Rate limit check result"""
    allowed: bool
    limit: int
    remaining: int
    reset_time: int
    retry_after: Optional[int] = None


class RateLimiter:
    """Rate limiter implementation"""
    
    def __init__(self, cache_client: CacheClient):
        self.cache = cache_client
    
    async def check_rate_limit(self, identifier: str, rule: RateLimitRule) -> RateLimitResult:
        """Check if request is allowed under rate limit"""
        if rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return await self._token_bucket(identifier, rule)
        elif rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return await self._sliding_window(identifier, rule)
        elif rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return await self._fixed_window(identifier, rule)
        else:
            raise ValueError(f"Unsupported rate limit algorithm: {rule.algorithm}")
    
    async def _token_bucket(self, identifier: str, rule: RateLimitRule) -> RateLimitResult:
        """Token bucket rate limiting algorithm"""
        key = f"rate_limit:token_bucket:{identifier}"
        now = time.time()
        
        # Get current bucket state
        bucket_data = await self.cache.get(key)
        
        if bucket_data is None:
            # Initialize bucket
            tokens = rule.limit
            last_refill = now
        else:
            tokens = bucket_data.get('tokens', 0)
            last_refill = bucket_data.get('last_refill', now)
        
        # Refill tokens based on time passed
        time_passed = now - last_refill
        tokens_to_add = int(time_passed * (rule.limit / rule.window))
        tokens = min(rule.limit, tokens + tokens_to_add)
        
        # Check if request can be allowed
        if tokens >= 1:
            tokens -= 1
            allowed = True
        else:
            allowed = False
        
        # Update bucket state
        bucket_state = {
            'tokens': tokens,
            'last_refill': now
        }
        await self.cache.set(key, bucket_state, rule.window * 2)
        
        # Calculate reset time
        reset_time = int(now + (rule.window / rule.limit) * (rule.limit - tokens))
        
        return RateLimitResult(
            allowed=allowed,
            limit=rule.limit,
            remaining=int(tokens),
            reset_time=reset_time,
            retry_after=None if allowed else int((1 - tokens) * (rule.window / rule.limit))
        )
    
    async def _sliding_window(self, identifier: str, rule: RateLimitRule) -> RateLimitResult:
        """Sliding window rate limiting algorithm"""
        key = f"rate_limit:sliding_window:{identifier}"
        now = time.time()
        window_start = now - rule.window
        
        # Get request timestamps
        timestamps = await self.cache.get(key) or []
        
        # Remove old timestamps
        timestamps = [ts for ts in timestamps if ts > window_start]
        
        # Check if request can be allowed
        if len(timestamps) < rule.limit:
            timestamps.append(now)
            allowed = True
        else:
            allowed = False
        
        # Update timestamps
        await self.cache.set(key, timestamps, rule.window)
        
        # Calculate reset time (when oldest request expires)
        reset_time = int(timestamps[0] + rule.window) if timestamps else int(now + rule.window)
        
        return RateLimitResult(
            allowed=allowed,
            limit=rule.limit,
            remaining=max(0, rule.limit - len(timestamps)),
            reset_time=reset_time,
            retry_after=None if allowed else int(reset_time - now)
        )
    
    async def _fixed_window(self, identifier: str, rule: RateLimitRule) -> RateLimitResult:
        """Fixed window rate limiting algorithm"""
        now = time.time()
        window_start = int(now // rule.window) * rule.window
        key = f"rate_limit:fixed_window:{identifier}:{window_start}"
        
        # Get current count
        current_count = await self.cache.get(key) or 0
        
        # Check if request can be allowed
        if current_count < rule.limit:
            # Increment counter
            await self.cache.increment(key)
            await self.cache.expire(key, rule.window)
            allowed = True
            current_count += 1
        else:
            allowed = False
        
        # Calculate reset time
        reset_time = int(window_start + rule.window)
        
        return RateLimitResult(
            allowed=allowed,
            limit=rule.limit,
            remaining=max(0, rule.limit - current_count),
            reset_time=reset_time,
            retry_after=None if allowed else int(reset_time - now)
        )


class RateLimitConfig:
    """Rate limiting configuration"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.enabled = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
        
        # Default rate limits per tier
        self.default_rules = {
            'starter': RateLimitRule(limit=100, window=3600),  # 100 requests per hour
            'professional': RateLimitRule(limit=1000, window=3600),  # 1000 requests per hour  
            'business': RateLimitRule(limit=10000, window=3600),  # 10k requests per hour
            'enterprise': RateLimitRule(limit=100000, window=3600)  # 100k requests per hour
        }
        
        # Service-specific overrides
        self.service_rules = self._load_service_rules()
    
    def _load_service_rules(self) -> Dict[str, Dict[str, RateLimitRule]]:
        """Load service-specific rate limit rules"""
        # This could be loaded from configuration files or environment variables
        # For now, return empty dict - services will use default rules
        return {}
    
    def get_rule_for_user(self, user: User, endpoint: str = None) -> RateLimitRule:
        """Get rate limit rule for user"""
        # Check for service-specific rules first
        if self.service_name in self.service_rules:
            service_rules = self.service_rules[self.service_name]
            if endpoint and endpoint in service_rules:
                return service_rules[endpoint]
            if user.tier.value in service_rules:
                return service_rules[user.tier.value]
        
        # Fall back to default rules
        return self.default_rules.get(user.tier.value, self.default_rules['starter'])
    
    def get_rule_for_ip(self) -> RateLimitRule:
        """Get rate limit rule for IP-based limiting"""
        # More restrictive for unauthenticated requests
        return RateLimitRule(limit=50, window=3600, identifier="ip")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for FastAPI"""
    
    def __init__(self, app, service_name: str, tier_config: Dict[str, Any] = None):
        super().__init__(app)
        self.service_name = service_name
        self.config = RateLimitConfig(service_name)
        self.cache = get_cache_client_sync(f"rate_limit:{service_name}")
        self.rate_limiter = RateLimiter(self.cache)
        
        # Override with provided tier config
        if tier_config:
            self._update_config_from_tier(tier_config)
    
    def _update_config_from_tier(self, tier_config: Dict[str, Any]):
        """Update configuration from tier config"""
        for tier, limits in tier_config.items():
            if isinstance(limits, dict):
                rule = RateLimitRule(
                    limit=limits.get('requests_per_hour', 100),
                    window=3600,
                    algorithm=RateLimitAlgorithm(limits.get('algorithm', 'token_bucket'))
                )
                self.config.default_rules[tier] = rule
    
    async def dispatch(self, request: Request, call_next):
        if not self.config.enabled:
            return await call_next(request)
        
        # Skip rate limiting for health checks and internal endpoints
        if request.url.path in ["/health", "/ready", "/metrics"]:
            return await call_next(request)
        
        # Determine identifier and rule
        identifier, rule = await self._get_identifier_and_rule(request)
        
        # Check rate limit
        result = await self.rate_limiter.check_rate_limit(identifier, rule)
        
        if not result.allowed:
            # Rate limit exceeded
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": str(result.remaining),
                    "X-RateLimit-Reset": str(result.reset_time),
                    "Retry-After": str(result.retry_after) if result.retry_after else "60"
                }
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(result.reset_time)
        
        return response
    
    async def _get_identifier_and_rule(self, request: Request) -> Tuple[str, RateLimitRule]:
        """Get rate limit identifier and rule for request"""
        # Try to get user from request state (set by auth middleware)
        user = getattr(request.state, 'user', None)
        
        if user:
            # User-based rate limiting
            identifier = f"user:{user.id}"
            rule = self.config.get_rule_for_user(user, request.url.path)
        else:
            # IP-based rate limiting for unauthenticated requests
            client_ip = self._get_client_ip(request)
            identifier = f"ip:{client_ip}"
            rule = self.config.get_rule_for_ip()
        
        return identifier, rule
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        # Check X-Forwarded-For header first (for load balancers/proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to client host
        if hasattr(request.client, 'host'):
            return request.client.host
        
        return "unknown"


class ServiceRateLimiter:
    """Service-specific rate limiter for API endpoints"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.cache = get_cache_client_sync(f"service_rate_limit:{service_name}")
        self.rate_limiter = RateLimiter(self.cache)
    
    async def check_api_rate_limit(self, user: User, endpoint: str, custom_rule: RateLimitRule = None) -> RateLimitResult:
        """Check rate limit for specific API endpoint"""
        if custom_rule:
            rule = custom_rule
        else:
            # Use default rules based on user tier
            config = RateLimitConfig(self.service_name)
            rule = config.get_rule_for_user(user, endpoint)
        
        identifier = f"api:{user.id}:{endpoint}"
        return await self.rate_limiter.check_rate_limit(identifier, rule)
    
    async def check_resource_rate_limit(self, user: User, resource_type: str, amount: int = 1) -> RateLimitResult:
        """Check rate limit for resource consumption (tokens, generations, etc.)"""
        # Different limits for different resource types
        resource_limits = {
            'tokens': {
                'starter': RateLimitRule(limit=500000, window=3600),  # 500k tokens per hour
                'professional': RateLimitRule(limit=5000000, window=3600),  # 5M tokens per hour
                'business': RateLimitRule(limit=25000000, window=3600),  # 25M tokens per hour
                'enterprise': RateLimitRule(limit=100000000, window=3600)  # 100M tokens per hour
            },
            'generations': {
                'starter': RateLimitRule(limit=100, window=3600),
                'professional': RateLimitRule(limit=1000, window=3600),
                'business': RateLimitRule(limit=10000, window=3600),
                'enterprise': RateLimitRule(limit=50000, window=3600)
            },
            'validations': {
                'starter': RateLimitRule(limit=50, window=3600),
                'professional': RateLimitRule(limit=500, window=3600),
                'business': RateLimitRule(limit=5000, window=3600),
                'enterprise': RateLimitRule(limit=25000, window=3600)
            }
        }
        
        if resource_type not in resource_limits:
            raise ValueError(f"Unknown resource type: {resource_type}")
        
        rule = resource_limits[resource_type].get(user.tier.value)
        if not rule:
            rule = resource_limits[resource_type]['starter']
        
        identifier = f"resource:{user.id}:{resource_type}"
        
        # For resource consumption, we need to check if we can consume 'amount' resources
        # This is a simplified implementation - in production you'd want to handle this more carefully
        result = await self.rate_limiter.check_rate_limit(identifier, rule)
        
        if result.allowed and amount > 1:
            # Consume additional resources
            for _ in range(amount - 1):
                additional_result = await self.rate_limiter.check_rate_limit(identifier, rule)
                if not additional_result.allowed:
                    result.allowed = False
                    result.remaining = additional_result.remaining
                    break
        
        return result


# Decorator for rate limiting functions
def rate_limit(rule: RateLimitRule, identifier_func=None):
    """Decorator for rate limiting functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get cache client
            cache = get_cache_client_sync("function_rate_limit")
            rate_limiter = RateLimiter(cache)
            
            # Generate identifier
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                # Default identifier based on function name and arguments
                func_name = func.__name__
                arg_str = ":".join(str(arg) for arg in args)
                kwarg_str = ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                identifier = hashlib.md5(f"{func_name}:{arg_str}:{kwarg_str}".encode()).hexdigest()
            
            # Check rate limit
            result = await rate_limiter.check_rate_limit(identifier, rule)
            
            if not result.allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded for {func.__name__}",
                    headers={"Retry-After": str(result.retry_after) if result.retry_after else "60"}
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Usage tracking for billing
class UsageTracker:
    """Track usage for billing purposes"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.cache = get_cache_client_sync(f"usage:{service_name}")
    
    async def track_usage(self, user_id: str, resource_type: str, amount: int, metadata: Dict[str, Any] = None):
        """Track resource usage"""
        key = f"usage:{user_id}:{resource_type}:{datetime.utcnow().strftime('%Y-%m-%d-%H')}"
        
        usage_data = await self.cache.get(key) or {
            'total': 0,
            'events': []
        }
        
        usage_data['total'] += amount
        usage_data['events'].append({
            'amount': amount,
            'timestamp': time.time(),
            'metadata': metadata or {}
        })
        
        # Keep usage data for 48 hours
        await self.cache.set(key, usage_data, 48 * 3600)
    
    async def get_usage(self, user_id: str, resource_type: str, hours: int = 24) -> Dict[str, Any]:
        """Get usage statistics"""
        total_usage = 0
        events = []
        
        current_time = datetime.utcnow()
        
        for i in range(hours):
            hour_time = current_time - timedelta(hours=i)
            key = f"usage:{user_id}:{resource_type}:{hour_time.strftime('%Y-%m-%d-%H')}"
            
            usage_data = await self.cache.get(key)
            if usage_data:
                total_usage += usage_data['total']
                events.extend(usage_data['events'])
        
        return {
            'total': total_usage,
            'events': sorted(events, key=lambda x: x['timestamp'], reverse=True),
            'period_hours': hours
        }