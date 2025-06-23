"""
Request/Response logging and monitoring middleware
"""

import time
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager
import uuid

from fastapi import Request, Response
from fastapi.routing import APIRoute
import structlog


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)


class RequestLogger:
    """Structured request logger"""
    
    def __init__(self, service_name: str = "inferloop-synthetic-api"):
        self.service_name = service_name
        self.logger = structlog.get_logger(service_name)
        
        # Metrics storage (in production, use Prometheus/StatsD)
        self.metrics = {
            'request_count': 0,
            'request_duration': [],
            'status_codes': {},
            'endpoints': {},
            'errors': []
        }
    
    def log_request(self, request: Request, request_id: str) -> Dict[str, Any]:
        """Log incoming request"""
        log_data = {
            'event': 'request_started',
            'request_id': request_id,
            'method': request.method,
            'path': request.url.path,
            'query_params': dict(request.query_params),
            'client_host': request.client.host if request.client else None,
            'user_agent': request.headers.get('user-agent'),
            'content_type': request.headers.get('content-type'),
            'content_length': request.headers.get('content-length'),
        }
        
        # Add authentication info if available
        if hasattr(request.state, 'user') and request.state.user:
            log_data['user_id'] = request.state.user.id
            log_data['user_role'] = request.state.user.role
        elif hasattr(request.state, 'api_key') and request.state.api_key:
            log_data['api_key_id'] = request.state.api_key.id
            log_data['api_key_name'] = request.state.api_key.name
        
        self.logger.info(**log_data)
        return log_data
    
    def log_response(self, request: Request, response: Response, 
                    request_id: str, duration: float, 
                    request_data: Dict[str, Any]) -> None:
        """Log response details"""
        log_data = {
            'event': 'request_completed',
            'request_id': request_id,
            'method': request.method,
            'path': request.url.path,
            'status_code': response.status_code,
            'duration_ms': round(duration * 1000, 2),
            'response_size': response.headers.get('content-length', 0)
        }
        
        # Add request context
        log_data.update({
            'user_id': request_data.get('user_id'),
            'api_key_id': request_data.get('api_key_id')
        })
        
        # Update metrics
        self.metrics['request_count'] += 1
        self.metrics['request_duration'].append(duration)
        self.metrics['status_codes'][response.status_code] = \
            self.metrics['status_codes'].get(response.status_code, 0) + 1
        
        endpoint = f"{request.method} {request.url.path}"
        if endpoint not in self.metrics['endpoints']:
            self.metrics['endpoints'][endpoint] = {
                'count': 0,
                'total_duration': 0,
                'errors': 0
            }
        
        self.metrics['endpoints'][endpoint]['count'] += 1
        self.metrics['endpoints'][endpoint]['total_duration'] += duration
        
        if response.status_code >= 400:
            self.metrics['endpoints'][endpoint]['errors'] += 1
        
        # Log level based on status code
        if response.status_code >= 500:
            self.logger.error(**log_data)
        elif response.status_code >= 400:
            self.logger.warning(**log_data)
        else:
            self.logger.info(**log_data)
    
    def log_error(self, request: Request, error: Exception, 
                 request_id: str, duration: float) -> None:
        """Log error details"""
        log_data = {
            'event': 'request_error',
            'request_id': request_id,
            'method': request.method,
            'path': request.url.path,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'duration_ms': round(duration * 1000, 2)
        }
        
        self.logger.error(**log_data, exc_info=True)
        
        # Track error
        self.metrics['errors'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'endpoint': f"{request.method} {request.url.path}",
            'error_type': type(error).__name__,
            'error_message': str(error)
        })
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if not self.metrics['request_duration']:
            avg_duration = 0
        else:
            avg_duration = sum(self.metrics['request_duration']) / len(self.metrics['request_duration'])
        
        return {
            'total_requests': self.metrics['request_count'],
            'average_duration_ms': round(avg_duration * 1000, 2),
            'status_code_distribution': self.metrics['status_codes'],
            'endpoint_stats': self.metrics['endpoints'],
            'recent_errors': self.metrics['errors'][-10:]  # Last 10 errors
        }


class LoggingMiddleware:
    """Middleware for comprehensive request/response logging"""
    
    def __init__(self, logger: Optional[RequestLogger] = None):
        self.logger = logger or RequestLogger()
    
    async def __call__(self, request: Request, call_next):
        """Log request and response"""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        request_data = self.logger.log_request(request, request_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            self.logger.log_response(request, response, request_id, duration, request_data)
            
            # Add request ID header
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as error:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            self.logger.log_error(request, error, request_id, duration)
            
            # Re-raise error
            raise


class PerformanceMonitor:
    """Monitor API performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'endpoint_latencies': {},
            'slow_requests': [],
            'database_queries': {},
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.slow_request_threshold = 5.0  # seconds
    
    @contextmanager
    def track_endpoint(self, endpoint: str):
        """Track endpoint performance"""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            
            if endpoint not in self.metrics['endpoint_latencies']:
                self.metrics['endpoint_latencies'][endpoint] = []
            
            self.metrics['endpoint_latencies'][endpoint].append(duration)
            
            # Track slow requests
            if duration > self.slow_request_threshold:
                self.metrics['slow_requests'].append({
                    'endpoint': endpoint,
                    'duration': duration,
                    'timestamp': datetime.utcnow().isoformat()
                })
    
    @contextmanager
    def track_database_query(self, query_type: str):
        """Track database query performance"""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            
            if query_type not in self.metrics['database_queries']:
                self.metrics['database_queries'][query_type] = {
                    'count': 0,
                    'total_duration': 0
                }
            
            self.metrics['database_queries'][query_type]['count'] += 1
            self.metrics['database_queries'][query_type]['total_duration'] += duration
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.metrics['cache_misses'] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        report = {
            'endpoint_performance': {},
            'slow_requests': self.metrics['slow_requests'][-20:],  # Last 20
            'database_performance': {},
            'cache_performance': {
                'hits': self.metrics['cache_hits'],
                'misses': self.metrics['cache_misses'],
                'hit_rate': self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses'])
                if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0
            }
        }
        
        # Calculate endpoint statistics
        for endpoint, latencies in self.metrics['endpoint_latencies'].items():
            if latencies:
                report['endpoint_performance'][endpoint] = {
                    'count': len(latencies),
                    'avg_latency_ms': round(sum(latencies) / len(latencies) * 1000, 2),
                    'min_latency_ms': round(min(latencies) * 1000, 2),
                    'max_latency_ms': round(max(latencies) * 1000, 2),
                    'p95_latency_ms': round(sorted(latencies)[int(len(latencies) * 0.95)] * 1000, 2)
                    if len(latencies) > 1 else round(latencies[0] * 1000, 2)
                }
        
        # Calculate database statistics
        for query_type, stats in self.metrics['database_queries'].items():
            if stats['count'] > 0:
                report['database_performance'][query_type] = {
                    'count': stats['count'],
                    'avg_duration_ms': round(stats['total_duration'] / stats['count'] * 1000, 2)
                }
        
        return report


# Global instances
request_logger = RequestLogger()
performance_monitor = PerformanceMonitor()