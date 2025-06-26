"""
Unified Cloud Deployment Infrastructure Package

This package provides unified infrastructure services for all Inferloop synthetic data
generation services including tabular, textnlp, syndocs, image, video, audio, etc.

The package provides abstractions for:
- Authentication and authorization
- Monitoring and telemetry 
- Storage (object, database, cache)
- Rate limiting and billing
- Configuration management
- Service discovery
"""

__version__ = "1.0.0"

from .auth import AuthMiddleware, get_current_user, User, require_permissions
from .monitoring import TelemetryMiddleware, MetricsCollector, create_counter, create_histogram, create_gauge
from .storage import StorageClient, get_storage_client
from .cache import CacheClient, get_cache_client
from .database import DatabaseClient, get_db_session
from .config import get_service_config
from .ratelimit import RateLimitMiddleware
from .websocket import WebSocketManager

__all__ = [
    "AuthMiddleware",
    "get_current_user", 
    "User",
    "require_permissions",
    "TelemetryMiddleware",
    "MetricsCollector",
    "create_counter",
    "create_histogram", 
    "create_gauge",
    "StorageClient",
    "get_storage_client",
    "CacheClient", 
    "get_cache_client",
    "DatabaseClient",
    "get_db_session",
    "get_service_config",
    "RateLimitMiddleware",
    "WebSocketManager"
]