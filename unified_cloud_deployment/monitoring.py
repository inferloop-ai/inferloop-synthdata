"""
Monitoring and Telemetry Module

Provides unified monitoring, metrics collection, logging, and distributed tracing
for all Inferloop services.
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import opentelemetry.trace as trace
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


class MetricsConfig:
    """Configuration for metrics collection"""
    
    def __init__(self):
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.metrics_port = int(os.getenv("METRICS_PORT", "9090"))
        self.metrics_path = os.getenv("METRICS_PATH", "/metrics")
        self.registry = CollectorRegistry()


class TracingConfig:
    """Configuration for distributed tracing"""
    
    def __init__(self):
        self.enable_tracing = os.getenv("ENABLE_TRACING", "true").lower() == "true"
        self.jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", "http://jaeger:14268/api/traces")
        self.service_name = os.getenv("SERVICE_NAME", "inferloop-service")
        self.sample_rate = float(os.getenv("TRACE_SAMPLE_RATE", "0.1"))


# Global configurations
metrics_config = MetricsConfig()
tracing_config = TracingConfig()

# Global registry for metrics
METRICS_REGISTRY = {}


@dataclass
class MetricDefinition:
    """Definition for a metric"""
    name: str
    description: str
    metric_type: str  # counter, histogram, gauge
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None
    instance: Optional[Any] = None


class MetricsCollector:
    """Centralized metrics collector for services"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.registry = metrics_config.registry
        self.metrics: Dict[str, Any] = {}
        
        # Default service metrics
        self._create_default_metrics()
    
    def _create_default_metrics(self):
        """Create default metrics for all services"""
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code', 'service'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint', 'service'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'http_active_connections',
            'Active HTTP connections',
            ['service'],
            registry=self.registry
        )
        
        self.errors_total = Counter(
            'errors_total',
            'Total errors by type',
            ['error_type', 'service'],
            registry=self.registry
        )
    
    def create_counter(self, name: str, description: str, labels: List[str] = None) -> Counter:
        """Create a counter metric"""
        labels = labels or []
        labels.append('service')  # Always include service label
        
        counter = Counter(
            name,
            description,
            labels,
            registry=self.registry
        )
        
        self.metrics[name] = counter
        METRICS_REGISTRY[name] = MetricDefinition(
            name=name,
            description=description,
            metric_type="counter",
            labels=labels,
            instance=counter
        )
        
        return counter
    
    def create_histogram(self, name: str, description: str, labels: List[str] = None, 
                        buckets: List[float] = None) -> Histogram:
        """Create a histogram metric"""
        labels = labels or []
        labels.append('service')
        buckets = buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
        
        histogram = Histogram(
            name,
            description,
            labels,
            buckets=buckets,
            registry=self.registry
        )
        
        self.metrics[name] = histogram
        METRICS_REGISTRY[name] = MetricDefinition(
            name=name,
            description=description,
            metric_type="histogram", 
            labels=labels,
            buckets=buckets,
            instance=histogram
        )
        
        return histogram
    
    def create_gauge(self, name: str, description: str, labels: List[str] = None) -> Gauge:
        """Create a gauge metric"""
        labels = labels or []
        labels.append('service')
        
        gauge = Gauge(
            name,
            description,
            labels,
            registry=self.registry
        )
        
        self.metrics[name] = gauge
        METRICS_REGISTRY[name] = MetricDefinition(
            name=name,
            description=description,
            metric_type="gauge",
            labels=labels,
            instance=gauge
        )
        
        return gauge
    
    def track_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Track HTTP request metrics"""
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            service=self.service_name
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint,
            service=self.service_name
        ).observe(duration)
    
    def track_error(self, error_type: str):
        """Track error occurrence"""
        self.errors_total.labels(
            error_type=error_type,
            service=self.service_name
        ).inc()
    
    def set_active_connections(self, count: int):
        """Set active connections gauge"""
        self.active_connections.labels(service=self.service_name).set(count)
    
    async def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')


class TelemetryMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting telemetry data"""
    
    def __init__(self, app, service_name: str, enable_tracing: bool = True, enable_metrics: bool = True):
        super().__init__(app)
        self.service_name = service_name
        self.enable_tracing = enable_tracing and tracing_config.enable_tracing
        self.enable_metrics = enable_metrics and metrics_config.enable_metrics
        
        if self.enable_metrics:
            self.metrics_collector = MetricsCollector(service_name)
        
        if self.enable_tracing:
            self._setup_tracing()
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)
        
        jaeger_exporter = JaegerExporter(
            agent_host_name="jaeger",
            agent_port=6831,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Start tracing span if enabled
        span = None
        if self.enable_tracing:
            tracer = trace.get_tracer(__name__)
            span = tracer.start_span(f"{request.method} {request.url.path}")
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("service.name", self.service_name)
        
        try:
            # Increment active connections
            if self.enable_metrics:
                self.metrics_collector.set_active_connections(
                    self.metrics_collector.active_connections._value.get() + 1
                )
            
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Track metrics
            if self.enable_metrics:
                self.metrics_collector.track_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=response.status_code,
                    duration=duration
                )
            
            # Update span
            if span:
                span.set_attribute("http.status_code", response.status_code)
                span.set_status(otel_trace.Status(otel_trace.StatusCode.OK))
            
            return response
            
        except Exception as e:
            # Track error
            if self.enable_metrics:
                self.metrics_collector.track_error(type(e).__name__)
            
            # Update span with error
            if span:
                span.set_status(
                    otel_trace.Status(
                        otel_trace.StatusCode.ERROR,
                        str(e)
                    )
                )
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
            
            raise
        
        finally:
            # Decrement active connections
            if self.enable_metrics:
                self.metrics_collector.set_active_connections(
                    max(0, self.metrics_collector.active_connections._value.get() - 1)
                )
            
            # End span
            if span:
                span.end()


def track_request(endpoint: str):
    """Decorator to track request metrics"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                # Track error in metrics if collector available
                if hasattr(wrapper, '_metrics_collector'):
                    wrapper._metrics_collector.track_error(type(e).__name__)
                raise
            finally:
                duration = time.time() - start_time
                
                # Log request completion
                logger = logging.getLogger(__name__)
                logger.info(f"Request {endpoint} completed in {duration:.3f}s")
        
        return wrapper
    return decorator


@contextmanager
def trace_operation(operation_name: str, **attributes):
    """Context manager for tracing operations"""
    if not tracing_config.enable_tracing:
        yield
        return
    
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(operation_name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        
        try:
            yield span
        except Exception as e:
            span.set_status(
                otel_trace.Status(
                    otel_trace.StatusCode.ERROR,
                    str(e)
                )
            )
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            raise


class StructuredLogger:
    """Structured logger for consistent logging across services"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        
        # Configure structured logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup structured logging configuration"""
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        log_format = os.getenv("LOG_FORMAT", "json")  # json or text
        
        self.logger.setLevel(getattr(logging, log_level))
        
        # Create formatter
        if log_format == "json":
            formatter = StructuredFormatter(self.service_name)
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Create handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        
        # Add handler to logger
        if not self.logger.handlers:
            self.logger.addHandler(handler)
    
    def info(self, message: str, **extra):
        """Log info message with extra context"""
        self.logger.info(message, extra=self._prepare_extra(extra))
    
    def warning(self, message: str, **extra):
        """Log warning message with extra context"""
        self.logger.warning(message, extra=self._prepare_extra(extra))
    
    def error(self, message: str, **extra):
        """Log error message with extra context"""
        self.logger.error(message, extra=self._prepare_extra(extra))
    
    def debug(self, message: str, **extra):
        """Log debug message with extra context"""
        self.logger.debug(message, extra=self._prepare_extra(extra))
    
    def _prepare_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare extra context for logging"""
        extra.update({
            'service': self.service_name,
            'timestamp': datetime.utcnow().isoformat(),
        })
        return extra


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def __init__(self, service_name: str):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'service': self.service_name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


# Convenience functions for creating metrics
def create_counter(name: str, description: str, labels: List[str] = None) -> Counter:
    """Create a counter metric (convenience function)"""
    # This will be used by services that don't have their own MetricsCollector
    registry = metrics_config.registry
    labels = labels or []
    
    counter = Counter(name, description, labels, registry=registry)
    METRICS_REGISTRY[name] = MetricDefinition(
        name=name,
        description=description,
        metric_type="counter",
        labels=labels,
        instance=counter
    )
    
    return counter


def create_histogram(name: str, description: str, labels: List[str] = None,
                    buckets: List[float] = None) -> Histogram:
    """Create a histogram metric (convenience function)"""
    registry = metrics_config.registry
    labels = labels or []
    buckets = buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
    
    histogram = Histogram(name, description, labels, buckets=buckets, registry=registry)
    METRICS_REGISTRY[name] = MetricDefinition(
        name=name,
        description=description,
        metric_type="histogram",
        labels=labels,
        buckets=buckets,
        instance=histogram
    )
    
    return histogram


def create_gauge(name: str, description: str, labels: List[str] = None) -> Gauge:
    """Create a gauge metric (convenience function)"""
    registry = metrics_config.registry
    labels = labels or []
    
    gauge = Gauge(name, description, labels, registry=registry)
    METRICS_REGISTRY[name] = MetricDefinition(
        name=name,
        description=description,
        metric_type="gauge",
        labels=labels,
        instance=gauge
    )
    
    return gauge


# Utility functions
def get_service_logger(service_name: str) -> StructuredLogger:
    """Get a structured logger for a service"""
    return StructuredLogger(service_name)


def setup_monitoring(service_name: str, enable_tracing: bool = True) -> MetricsCollector:
    """Setup monitoring for a service"""
    collector = MetricsCollector(service_name)
    
    if enable_tracing and tracing_config.enable_tracing:
        # Setup FastAPI instrumentation
        FastAPIInstrumentor.instrument()
    
    return collector