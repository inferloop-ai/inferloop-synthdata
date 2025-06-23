"""
Error tracking integration for monitoring and alerting
"""

import traceback
import sys
import os
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import asyncio
import json
from collections import deque
import hashlib

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import structlog


class ErrorTracker:
    """Error tracking and reporting system"""
    
    def __init__(self, 
                 service_name: str = "inferloop-synthetic-api",
                 environment: str = None,
                 sentry_dsn: Optional[str] = None,
                 custom_handlers: Optional[List[Callable]] = None):
        
        self.service_name = service_name
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self.logger = structlog.get_logger(service_name)
        self.custom_handlers = custom_handlers or []
        
        # Error storage (in production, use external service)
        self.error_queue = deque(maxlen=1000)
        self.error_stats = {
            'total_errors': 0,
            'error_types': {},
            'error_endpoints': {},
            'error_rate': []
        }
        
        # Initialize Sentry if DSN provided
        self.sentry_enabled = False
        if sentry_dsn:
            try:
                import sentry_sdk
                from sentry_sdk.integrations.fastapi import FastAPIIntegration
                
                sentry_sdk.init(
                    dsn=sentry_dsn,
                    environment=self.environment,
                    integrations=[FastAPIIntegration()],
                    traces_sample_rate=0.1,
                    profiles_sample_rate=0.1,
                )
                self.sentry_enabled = True
                self.logger.info("Sentry error tracking initialized")
            except ImportError:
                self.logger.warning("Sentry SDK not installed. Install with: pip install sentry-sdk")
    
    def create_error_context(self, request: Request, error: Exception) -> Dict[str, Any]:
        """Create comprehensive error context"""
        context = {
            'error_id': self._generate_error_id(error),
            'timestamp': datetime.utcnow().isoformat(),
            'environment': self.environment,
            'service': self.service_name,
            'error': {
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc()
            },
            'request': {
                'method': request.method,
                'path': request.url.path,
                'query_params': dict(request.query_params),
                'headers': dict(request.headers),
                'client_host': request.client.host if request.client else None
            }
        }
        
        # Add user context
        if hasattr(request.state, 'user') and request.state.user:
            context['user'] = {
                'id': request.state.user.id,
                'username': request.state.user.username,
                'role': request.state.user.role
            }
        elif hasattr(request.state, 'api_key') and request.state.api_key:
            context['api_key'] = {
                'id': request.state.api_key.id,
                'name': request.state.api_key.name
            }
        
        # Add request ID if available
        if hasattr(request.state, 'request_id'):
            context['request_id'] = request.state.request_id
        
        return context
    
    def _generate_error_id(self, error: Exception) -> str:
        """Generate unique error ID based on error characteristics"""
        error_signature = f"{type(error).__name__}:{str(error)}:{traceback.format_exc()[:200]}"
        return hashlib.md5(error_signature.encode()).hexdigest()[:12]
    
    async def track_error(self, request: Request, error: Exception) -> Dict[str, Any]:
        """Track error occurrence"""
        context = self.create_error_context(request, error)
        
        # Update statistics
        self.error_stats['total_errors'] += 1
        
        error_type = type(error).__name__
        self.error_stats['error_types'][error_type] = \
            self.error_stats['error_types'].get(error_type, 0) + 1
        
        endpoint = f"{request.method} {request.url.path}"
        self.error_stats['error_endpoints'][endpoint] = \
            self.error_stats['error_endpoints'].get(endpoint, 0) + 1
        
        # Add to error queue
        self.error_queue.append(context)
        
        # Log error
        self.logger.error(
            "error_tracked",
            error_id=context['error_id'],
            error_type=error_type,
            endpoint=endpoint,
            **context
        )
        
        # Send to Sentry if enabled
        if self.sentry_enabled:
            import sentry_sdk
            sentry_sdk.capture_exception(error)
        
        # Call custom handlers
        for handler in self.custom_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(context)
                else:
                    handler(context)
            except Exception as e:
                self.logger.error(f"Error in custom handler: {str(e)}")
        
        return context
    
    async def __call__(self, request: Request, call_next):
        """Middleware to catch and track errors"""
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # Don't track client errors (4xx)
            if e.status_code < 500:
                raise
            
            # Track server errors
            await self.track_error(request, e)
            raise
            
        except Exception as e:
            # Track unexpected errors
            context = await self.track_error(request, e)
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "error_id": context['error_id'],
                    "message": "An unexpected error occurred. Please try again later.",
                    "timestamp": context['timestamp']
                }
            )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error statistics summary"""
        recent_errors = list(self.error_queue)[-10:]  # Last 10 errors
        
        # Calculate error rate (errors per minute)
        now = datetime.utcnow()
        recent_error_times = [
            e['timestamp'] for e in recent_errors
            if (now - datetime.fromisoformat(e['timestamp'])).total_seconds() < 300  # Last 5 minutes
        ]
        error_rate = len(recent_error_times) / 5.0  # Per minute
        
        return {
            'total_errors': self.error_stats['total_errors'],
            'error_rate_per_minute': round(error_rate, 2),
            'error_types': dict(sorted(
                self.error_stats['error_types'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),  # Top 10 error types
            'error_endpoints': dict(sorted(
                self.error_stats['error_endpoints'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),  # Top 10 error endpoints
            'recent_errors': [
                {
                    'error_id': e['error_id'],
                    'timestamp': e['timestamp'],
                    'error_type': e['error']['type'],
                    'endpoint': f"{e['request']['method']} {e['request']['path']}",
                    'message': e['error']['message'][:100]
                }
                for e in recent_errors
            ]
        }
    
    def get_error_details(self, error_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed error information by ID"""
        for error in self.error_queue:
            if error['error_id'] == error_id:
                return error
        return None


class AlertManager:
    """Manage error alerts and notifications"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.alert_thresholds = {
            'error_rate': 10,  # Errors per minute
            'error_count': 50,  # Total errors in 5 minutes
            'endpoint_errors': 10  # Errors per endpoint in 5 minutes
        }
        self.alerts_sent = {}
    
    async def check_alert_conditions(self, error_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if alert conditions are met"""
        alerts = []
        
        # Check error rate
        if error_summary['error_rate_per_minute'] > self.alert_thresholds['error_rate']:
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'critical',
                'message': f"High error rate: {error_summary['error_rate_per_minute']} errors/minute",
                'threshold': self.alert_thresholds['error_rate'],
                'current_value': error_summary['error_rate_per_minute']
            })
        
        # Check total errors
        if error_summary['total_errors'] > self.alert_thresholds['error_count']:
            alerts.append({
                'type': 'high_error_count',
                'severity': 'warning',
                'message': f"High error count: {error_summary['total_errors']} total errors",
                'threshold': self.alert_thresholds['error_count'],
                'current_value': error_summary['total_errors']
            })
        
        # Check endpoint errors
        for endpoint, count in error_summary['error_endpoints'].items():
            if count > self.alert_thresholds['endpoint_errors']:
                alerts.append({
                    'type': 'endpoint_errors',
                    'severity': 'warning',
                    'message': f"High errors on endpoint {endpoint}: {count} errors",
                    'endpoint': endpoint,
                    'threshold': self.alert_thresholds['endpoint_errors'],
                    'current_value': count
                })
        
        return alerts
    
    async def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert notification"""
        # Implement rate limiting for alerts
        alert_key = f"{alert['type']}:{alert.get('endpoint', 'global')}"
        last_sent = self.alerts_sent.get(alert_key)
        
        if last_sent:
            if (datetime.utcnow() - last_sent).total_seconds() < 300:  # 5 minutes
                return False  # Don't spam alerts
        
        # Send webhook notification
        if self.webhook_url:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.webhook_url, json={
                        'service': 'inferloop-synthetic-api',
                        'alert': alert,
                        'timestamp': datetime.utcnow().isoformat()
                    }) as response:
                        if response.status == 200:
                            self.alerts_sent[alert_key] = datetime.utcnow()
                            return True
            except Exception as e:
                print(f"Failed to send alert: {str(e)}")
        
        return False


# Global error tracker instance
error_tracker = ErrorTracker(
    sentry_dsn=os.getenv('SENTRY_DSN'),
    environment=os.getenv('ENVIRONMENT', 'development')
)

# Alert manager instance
alert_manager = AlertManager(
    webhook_url=os.getenv('ALERT_WEBHOOK_URL')
)