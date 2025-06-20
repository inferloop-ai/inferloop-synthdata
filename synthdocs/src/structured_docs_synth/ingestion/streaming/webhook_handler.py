#!/usr/bin/env python3
"""
Webhook Handler for receiving streaming data via HTTP webhooks
"""

import json
import time
import hashlib
import threading
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from flask import Flask, request, jsonify, abort
from werkzeug.serving import make_server

from pydantic import BaseModel, Field

from ...core.config import get_config
from ...core.logging import get_logger
from ...core.exceptions import ProcessingError, ValidationError


class WebhookSecurity(Enum):
    """Webhook security methods"""
    NONE = "none"
    API_KEY = "api_key"
    HMAC_SHA256 = "hmac_sha256"
    JWT_TOKEN = "jwt_token"
    BASIC_AUTH = "basic_auth"


class WebhookMethod(Enum):
    """Supported HTTP methods"""
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"


@dataclass
class WebhookEvent:
    """Webhook event data"""
    event_id: str
    timestamp: datetime
    method: str
    path: str
    headers: Dict[str, str]
    data: Dict[str, Any]
    source_ip: str
    user_agent: str = ""
    content_type: str = ""
    content_length: int = 0
    event_hash: str = ""
    validation_status: str = "pending"
    processing_status: str = "pending"


class WebhookHandlerConfig(BaseModel):
    """Webhook handler configuration"""
    
    # Server settings
    host: str = Field("0.0.0.0", description="Server host")
    port: int = Field(8080, description="Server port")
    debug: bool = Field(False, description="Enable debug mode")
    threaded: bool = Field(True, description="Enable threaded mode")
    
    # Webhook endpoints
    webhook_paths: List[str] = Field(
        default=["/webhook", "/api/webhook"],
        description="Webhook endpoint paths"
    )
    allowed_methods: List[WebhookMethod] = Field(
        default=[WebhookMethod.POST],
        description="Allowed HTTP methods"
    )
    
    # Security
    security_method: WebhookSecurity = Field(
        WebhookSecurity.NONE,
        description="Security method"
    )
    api_key: Optional[str] = Field(None, description="API key for authentication")
    api_key_header: str = Field("X-API-Key", description="API key header name")
    hmac_secret: Optional[str] = Field(None, description="HMAC secret for signature validation")
    hmac_header: str = Field("X-Hub-Signature-256", description="HMAC signature header")
    
    # Request validation
    max_content_length: int = Field(1048576, description="Maximum content length in bytes")
    required_headers: List[str] = Field(default=[], description="Required headers")
    allowed_content_types: List[str] = Field(
        default=["application/json", "application/x-www-form-urlencoded"],
        description="Allowed content types"
    )
    
    # Rate limiting
    enable_rate_limiting: bool = Field(False, description="Enable rate limiting")
    rate_limit_requests: int = Field(100, description="Requests per rate limit window")
    rate_limit_window: int = Field(3600, description="Rate limit window in seconds")
    
    # Processing
    enable_deduplication: bool = Field(True, description="Enable event deduplication")
    store_events: bool = Field(True, description="Store received events")
    max_stored_events: int = Field(10000, description="Maximum stored events")
    
    # Response settings
    response_timeout: float = Field(30.0, description="Response timeout in seconds")
    success_response: Dict[str, Any] = Field(
        default={"status": "success", "message": "Webhook received"},
        description="Success response payload"
    )
    
    # Logging
    log_requests: bool = Field(True, description="Log incoming requests")
    log_headers: bool = Field(False, description="Log request headers")
    log_body: bool = Field(False, description="Log request body")


class WebhookHandler:
    """
    Webhook Handler for receiving streaming data via HTTP webhooks
    
    Provides secure webhook endpoints with authentication, validation,
    deduplication, rate limiting, and callback-based processing.
    """
    
    def __init__(self, config: Optional[WebhookHandlerConfig] = None):
        self.logger = get_logger(__name__)
        self.config = config or WebhookHandlerConfig()
        
        # Flask app setup
        self.app = Flask(__name__)
        self.app.config['MAX_CONTENT_LENGTH'] = self.config.max_content_length
        
        # State management
        self.received_events: List[WebhookEvent] = []
        self.event_hashes: set = set()
        self.rate_limit_tracker: Dict[str, List[float]] = {}
        
        # Server state
        self.server: Optional[Any] = None
        self.server_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Callbacks
        self.event_callbacks: List[Callable[[WebhookEvent], None]] = []
        self.error_callbacks: List[Callable[[str, Exception], None]] = []
        self.validation_callbacks: List[Callable[[WebhookEvent], bool]] = []
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "unauthorized_requests": 0,
            "rate_limited_requests": 0,
            "start_time": None
        }
        
        # Setup webhook routes
        self._setup_routes()
        
        self.logger.info(f"Webhook Handler initialized on {self.config.host}:{self.config.port}")
    
    def _setup_routes(self):
        """Setup webhook routes"""
        for path in self.config.webhook_paths:
            # Add route for each allowed method
            methods = [method.value for method in self.config.allowed_methods]
            self.app.add_url_rule(
                path,
                f"webhook_{path.replace('/', '_')}",
                self._handle_webhook,
                methods=methods
            )
        
        # Health check endpoint
        self.app.add_url_rule(
            "/health",
            "health_check",
            self._health_check,
            methods=["GET"]
        )
        
        # Metrics endpoint
        self.app.add_url_rule(
            "/metrics",
            "metrics",
            self._get_metrics_endpoint,
            methods=["GET"]
        )
    
    def _handle_webhook(self):
        """Handle incoming webhook requests"""
        self.metrics["total_requests"] += 1
        
        try:
            # Rate limiting check
            if self.config.enable_rate_limiting:
                if not self._check_rate_limit(request.remote_addr):
                    self.metrics["rate_limited_requests"] += 1
                    abort(429, "Rate limit exceeded")
            
            # Security validation
            if not self._validate_security():
                self.metrics["unauthorized_requests"] += 1
                abort(401, "Unauthorized")
            
            # Create webhook event
            event = self._create_webhook_event()
            
            # Validate event
            if not self._validate_event(event):
                self.metrics["failed_requests"] += 1
                abort(400, "Invalid webhook data")
            
            # Check for duplicates
            if self.config.enable_deduplication:
                if event.event_hash in self.event_hashes:
                    self.logger.debug(f"Duplicate event {event.event_hash}, skipping")
                    return jsonify(self.config.success_response)
                
                self.event_hashes.add(event.event_hash)
            
            # Process event
            self._process_event(event)
            
            # Store event
            if self.config.store_events:
                self._store_event(event)
            
            # Update rate limit tracker
            if self.config.enable_rate_limiting:
                self._update_rate_limit(request.remote_addr)
            
            self.metrics["successful_requests"] += 1
            
            # Log request if enabled
            if self.config.log_requests:
                self.logger.info(f"Webhook received from {request.remote_addr}: {request.path}")
            
            return jsonify(self.config.success_response)
            
        except Exception as e:
            self.metrics["failed_requests"] += 1
            self._handle_error("webhook_processing", e)
            abort(500, f"Internal server error: {str(e)}")
    
    def _validate_security(self) -> bool:
        """Validate webhook security"""
        if self.config.security_method == WebhookSecurity.NONE:
            return True
        
        elif self.config.security_method == WebhookSecurity.API_KEY:
            api_key = request.headers.get(self.config.api_key_header)
            return api_key == self.config.api_key
        
        elif self.config.security_method == WebhookSecurity.HMAC_SHA256:
            return self._validate_hmac_signature()
        
        elif self.config.security_method == WebhookSecurity.BASIC_AUTH:
            # Implement basic auth validation
            return True  # Placeholder
        
        elif self.config.security_method == WebhookSecurity.JWT_TOKEN:
            # Implement JWT validation
            return True  # Placeholder
        
        return False
    
    def _validate_hmac_signature(self) -> bool:
        """Validate HMAC SHA256 signature"""
        if not self.config.hmac_secret:
            return False
        
        signature = request.headers.get(self.config.hmac_header)
        if not signature:
            return False
        
        # Calculate expected signature
        import hmac
        payload = request.get_data()
        expected_signature = hmac.new(
            self.config.hmac_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures (constant time comparison)
        expected = f"sha256={expected_signature}"
        return hmac.compare_digest(signature, expected)
    
    def _create_webhook_event(self) -> WebhookEvent:
        """Create webhook event from request"""
        # Generate event ID
        event_id = f"webhook_{int(time.time() * 1000)}_{hash(request.remote_addr) % 10000}"
        
        # Extract request data
        try:
            if request.is_json:
                data = request.get_json() or {}
            elif request.form:
                data = dict(request.form)
            else:
                data = {"raw_data": request.get_data(as_text=True)}
        except Exception:
            data = {"error": "Failed to parse request data"}
        
        # Create event
        event = WebhookEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            method=request.method,
            path=request.path,
            headers=dict(request.headers),
            data=data,
            source_ip=request.remote_addr or "unknown",
            user_agent=request.headers.get("User-Agent", ""),
            content_type=request.headers.get("Content-Type", ""),
            content_length=request.content_length or 0
        )
        
        # Calculate event hash for deduplication
        if self.config.enable_deduplication:
            event_str = json.dumps({
                "path": event.path,
                "data": event.data,
                "timestamp": event.timestamp.isoformat()[:19]  # Truncate microseconds
            }, sort_keys=True)
            event.event_hash = hashlib.md5(event_str.encode()).hexdigest()
        
        return event
    
    def _validate_event(self, event: WebhookEvent) -> bool:
        """Validate webhook event"""
        try:
            # Check content type
            if (self.config.allowed_content_types and 
                event.content_type not in self.config.allowed_content_types):
                return False
            
            # Check required headers
            for header in self.config.required_headers:
                if header not in event.headers:
                    return False
            
            # Check content length
            if event.content_length > self.config.max_content_length:
                return False
            
            # Run custom validation callbacks
            for callback in self.validation_callbacks:
                try:
                    if not callback(event):
                        return False
                except Exception as e:
                    self.logger.warning(f"Validation callback failed: {e}")
                    return False
            
            event.validation_status = "valid"
            return True
            
        except Exception as e:
            event.validation_status = "invalid"
            self.logger.warning(f"Event validation failed: {e}")
            return False
    
    def _process_event(self, event: WebhookEvent):
        """Process webhook event"""
        try:
            # Log event details if enabled
            if self.config.log_body:
                self.logger.debug(f"Webhook event data: {event.data}")
            
            if self.config.log_headers:
                self.logger.debug(f"Webhook event headers: {event.headers}")
            
            # Trigger event callbacks
            for callback in self.event_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.warning(f"Event callback failed: {e}")
            
            event.processing_status = "processed"
            
        except Exception as e:
            event.processing_status = "failed"
            raise ProcessingError(f"Event processing failed: {e}")
    
    def _store_event(self, event: WebhookEvent):
        """Store webhook event"""
        self.received_events.append(event)
        
        # Maintain maximum stored events
        if len(self.received_events) > self.config.max_stored_events:
            old_event = self.received_events.pop(0)
            if old_event.event_hash in self.event_hashes:
                self.event_hashes.remove(old_event.event_hash)
    
    def _check_rate_limit(self, ip_address: str) -> bool:
        """Check rate limit for IP address"""
        now = time.time()
        window_start = now - self.config.rate_limit_window
        
        # Initialize IP tracker if not exists
        if ip_address not in self.rate_limit_tracker:
            self.rate_limit_tracker[ip_address] = []
        
        # Clean old requests
        self.rate_limit_tracker[ip_address] = [
            req_time for req_time in self.rate_limit_tracker[ip_address]
            if req_time > window_start
        ]
        
        # Check if under limit
        return len(self.rate_limit_tracker[ip_address]) < self.config.rate_limit_requests
    
    def _update_rate_limit(self, ip_address: str):
        """Update rate limit tracker"""
        self.rate_limit_tracker[ip_address].append(time.time())
    
    def _health_check(self):
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.metrics["start_time"] if self.metrics["start_time"] else 0
        })
    
    def _get_metrics_endpoint(self):
        """Metrics endpoint"""
        return jsonify(self.get_webhook_metrics())
    
    def _handle_error(self, context: str, error: Exception):
        """Handle webhook errors"""
        self.logger.error(f"Error in {context}: {error}")
        
        for callback in self.error_callbacks:
            try:
                callback(context, error)
            except Exception as e:
                self.logger.warning(f"Error callback failed: {e}")
    
    def add_event_callback(self, callback: Callable[[WebhookEvent], None]):
        """Add callback for webhook events"""
        self.event_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str, Exception], None]):
        """Add callback for errors"""
        self.error_callbacks.append(callback)
    
    def add_validation_callback(self, callback: Callable[[WebhookEvent], bool]):
        """Add custom validation callback"""
        self.validation_callbacks.append(callback)
    
    def start_server(self, blocking: bool = False):
        """Start webhook server"""
        if self.is_running:
            raise ValueError("Webhook server is already running")
        
        self.metrics["start_time"] = time.time()
        
        try:
            # Create server
            self.server = make_server(
                self.config.host,
                self.config.port,
                self.app,
                threaded=self.config.threaded
            )
            
            if blocking:
                # Run in current thread
                self.is_running = True
                self.logger.info(f"Starting webhook server on {self.config.host}:{self.config.port}")
                self.server.serve_forever()
            else:
                # Run in background thread
                def serve():
                    self.is_running = True
                    self.server.serve_forever()
                
                self.server_thread = threading.Thread(target=serve, daemon=True)
                self.server_thread.start()
                self.logger.info(f"Started webhook server on {self.config.host}:{self.config.port}")
            
        except Exception as e:
            self.is_running = False
            self.logger.error(f"Failed to start webhook server: {e}")
            raise ProcessingError(f"Webhook server error: {e}")
    
    def stop_server(self):
        """Stop webhook server"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.server:
            self.server.shutdown()
        
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5.0)
        
        self.logger.info("Webhook server stopped")
    
    def get_webhook_metrics(self) -> Dict[str, Any]:
        """Get webhook metrics"""
        current_time = time.time()
        uptime = current_time - self.metrics["start_time"] if self.metrics["start_time"] else 0
        
        return {
            "total_requests": self.metrics["total_requests"],
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
            "unauthorized_requests": self.metrics["unauthorized_requests"],
            "rate_limited_requests": self.metrics["rate_limited_requests"],
            "success_rate": (
                self.metrics["successful_requests"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0 else 0
            ),
            "stored_events": len(self.received_events),
            "unique_events": len(self.event_hashes),
            "uptime": uptime,
            "requests_per_second": (
                self.metrics["total_requests"] / uptime
                if uptime > 0 else 0
            ),
            "is_running": self.is_running
        }
    
    def export_webhook_events(self, format: str = "json") -> List[Dict[str, Any]]:
        """Export webhook events"""
        exported_data = []
        
        for event in self.received_events:
            exported_item = {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "method": event.method,
                "path": event.path,
                "source_ip": event.source_ip,
                "data": event.data,
                "validation_status": event.validation_status,
                "processing_status": event.processing_status
            }
            
            if event.event_hash:
                exported_item["event_hash"] = event.event_hash
            
            if event.user_agent:
                exported_item["user_agent"] = event.user_agent
            
            exported_data.append(exported_item)
        
        return exported_data
    
    def clear_events(self):
        """Clear stored events"""
        self.received_events.clear()
        self.event_hashes.clear()
        self.logger.info("Webhook events cleared")


# Factory function
def create_webhook_handler(**config_kwargs) -> WebhookHandler:
    """Factory function to create webhook handler"""
    config = WebhookHandlerConfig(**config_kwargs)
    return WebhookHandler(config)