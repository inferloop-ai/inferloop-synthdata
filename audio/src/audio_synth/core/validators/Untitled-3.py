
# audio_synth/monitoring/metrics_collector.py
"""
Advanced metrics collection and observability for audio synthesis
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import threading
from collections import defaultdict, deque
import asyncio

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, push_to_gateway
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Prometheus client not available. Install with: pip install prometheus-client")

# OpenTelemetry tracing
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    print("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk")

logger = logging.getLogger(__name__)

@dataclass
class GenerationMetrics:
    """Metrics for a single generation request"""
    request_id: str
    method: str
    start_time: float
    end_time: float
    duration: float
    num_samples: int
    success: bool
    error: Optional[str] = None
    
    # Resource usage
    memory_used: float = 0.0  # MB
    gpu_memory_used: float = 0.0  # MB
    cpu_utilization: float = 0.0  # %
    
    # Quality metrics
    quality_scores: Dict[str, float] = field(default_factory=dict)
    privacy_scores: Dict[str, float] = field(default_factory=dict)
    fairness_scores: Dict[str, float] = field(default_factory=dict)
    
    # Audio characteristics
    audio_length: float = 0.0  # seconds
    sample_rate: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "method": self.method,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "num_samples": self.num_samples,
            "success": self.success,
            "error": self.error,
            "memory_used": self.memory_used,
            "gpu_memory_used": self.gpu_memory_used,
            "cpu_utilization": self.cpu_utilization,
            "quality_scores": self.quality_scores,
            "privacy_scores": self.privacy_scores,
            "fairness_scores": self.fairness_scores,
            "audio_length": self.audio_length,
            "sample_rate": self.sample_rate
        }

class MetricsCollector:
    """Comprehensive metrics collection system"""
    
    def __init__(self, 
                 enable_prometheus: bool = True,
                 enable_tracing: bool = True,
                 retention_hours: int = 24):
        
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.enable_tracing = enable_tracing and OTEL_AVAILABLE
        self.retention_hours = retention_hours
        
        # Storage for metrics
        self.generation_metrics: deque = deque(maxlen=10000)
        self.system_metrics: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.error_log: deque = deque(maxlen=1000)
        
        # Real-time aggregations
        self.current_requests = {}
        self.hourly_stats = defaultdict(lambda: {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_duration": 0.0,
            "total_samples": 0
        })
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
        
        # OpenTelemetry tracing
        if self.enable_tracing:
            self._setup_tracing()
        
        # Start background monitoring
        self._start_system_monitoring()
        
        logger.info("Metrics collector initialized")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        
        # Counters
        self.requests_total = Counter(
            'audio_generation_requests_total',
            'Total number of audio generation requests',
            ['method', 'status'],
            registry=self.registry
        )
        
        self.samples_generated_total = Counter(
            'audio_samples_generated_total',
            'Total number of audio samples generated',
            ['method'],
            registry=self.registry
        )
        
        # Histograms
        self.generation_duration = Histogram(
            'audio_generation_duration_seconds',
            'Time spent generating audio',
            ['method'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.quality_scores = Histogram(
            'audio_quality_scores',
            'Audio quality scores',
            ['method', 'metric'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # Gauges
        self.active_requests = Gauge(
            'audio_generation_active_requests',
            'Number of currently active generation requests',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        if torch.cuda.is_available():
            self.gpu_memory_usage = Gauge(
                'gpu_memory_usage_percent',
                'GPU memory usage percentage',
                ['device'],
                registry=self.registry
            )
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        
        trace.set_tracer_provider(TracerProvider())
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        self.tracer = trace.get_tracer("audio_synthesis")
    
    def _start_system_monitoring(self):
        """Start background system monitoring"""
        
        def monitor_system():
            while True:
                try:
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    system_metric = {
                        "timestamp": time.time(),
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_used_gb": memory.used / (1024**3),
                        "memory_available_gb": memory.available / (1024**3)
                    }
                    
                    # GPU metrics
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            gpu_memory_used = torch.cuda.memory_allocated(i) / (1024**3)
                            gpu_memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                            gpu_utilization = (gpu_memory_used / gpu_memory_total) * 100
                            
                            system_metric[f"gpu_{i}_memory_used_gb"] = gpu_memory_used
                            system_metric[f"gpu_{i}_memory_total_gb"] = gpu_memory_total
                            system_metric[f"gpu_{i}_utilization_percent"] = gpu_utilization
                    
                    self.system_metrics.append(system_metric)
                    
                    # Update Prometheus gauges
                    if self.enable_prometheus:
                        self.system_memory_usage.set(memory.percent)
                        self.system_cpu_usage.set(cpu_percent)
                        
                        if torch.cuda.is_available():
                            for i in range(torch.cuda.device_count()):
                                gpu_memory_used = torch.cuda.memory_allocated(i)
                                gpu_memory_total = torch.cuda.get_device_properties(i).total_memory
                                gpu_percent = (gpu_memory_used / gpu_memory_total) * 100
                                self.gpu_memory_usage.labels(device=f"gpu_{i}").set(gpu_percent)
                    
                    # Cleanup old metrics
                    self._cleanup_old_metrics()
                    
                except Exception as e:
                    logger.error(f"Error in system monitoring: {e}")
                
                time.sleep(60)  # Monitor every minute
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    def start_generation(self, 
                        request_id: str, 
                        method: str, 
                        num_samples: int) -> Optional[Any]:
        """Start tracking a generation request"""
        
        start_time = time.time()
        
        # Create metrics object
        metrics = GenerationMetrics(
            request_id=request_id,
            method=method,
            start_time=start_time,
            end_time=0.0,
            duration=0.0,
            num_samples=num_samples,
            success=False
        )
        
        self.current_requests[request_id] = metrics
        
        # Update Prometheus
        if self.enable_prometheus:
            self.active_requests.inc()
        
        # Start tracing span
        span = None
        if self.enable_tracing:
            span = self.tracer.start_span(f"audio_generation_{method}")
            span.set_attribute("request_id", request_id)
            span.set_attribute("method", method)
            span.set_attribute("num_samples", num_samples)
        
        logger.info(f"Started tracking generation request {request_id} ({method})")
        return span
    
    def end_generation(self, 
                      request_id: str, 
                      success: bool = True,
                      error: Optional[str] = None,
                      quality_scores: Optional[Dict[str, float]] = None,
                      privacy_scores: Optional[Dict[str, float]] = None,
                      fairness_scores: Optional[Dict[str, float]] = None,
                      audio_length: float = 0.0,
                      sample_rate: int = 0,
                      span: Optional[Any] = None):
        """End tracking a generation request"""
        
        if request_id not in self.current_requests:
            logger.warning(f"Request {request_id} not found in current requests")
            return
        
        metrics = self.current_requests[request_id]
        end_time = time.time()
        
        # Update metrics
        metrics.end_time = end_time
        metrics.duration = end_time - metrics.start_time
        metrics.success = success
        metrics.error = error
        metrics.quality_scores = quality_scores or {}
        metrics.privacy_scores = privacy_scores or {}
        metrics.fairness_scores = fairness_scores or {}
        metrics.audio_length = audio_length
        metrics.sample_rate = sample_rate
        
        # Collect resource usage
        metrics.memory_used = psutil.virtual_memory().used / (1024**2)  # MB
        metrics.cpu_utilization = psutil.cpu_percent()
        
        if torch.cuda.is_available():
            metrics.gpu_memory_used = torch.cuda.memory_allocated() / (1024**2)  # MB
        
        # Store metrics
        self.generation_metrics.append(metrics)
        
        # Update hourly stats
        hour_key = datetime.fromtimestamp(metrics.start_time).strftime("%Y%m%d_%H")
        hourly = self.hourly_stats[hour_key]
        hourly["total_requests"] += 1
        
        if success:
            hourly["successful_requests"] += 1
        else:
            hourly["failed_requests"] += 1
            self.error_log.append({
                "timestamp": end_time,
                "request_id": request_id,
                "method": metrics.method,
                "error": error
            })
        
        hourly["total_samples"] += metrics.num_samples
        
        # Update average duration (incremental)
        prev_avg = hourly["avg_duration"]
        total_reqs = hourly["total_requests"]
        hourly["avg_duration"] = (prev_avg * (total_reqs - 1) + metrics.duration) / total_reqs
        
        # Update Prometheus metrics
        if self.enable_prometheus:
            status = "success" if success else "error"
            self.requests_total.labels(method=metrics.method, status=status).inc()
            self.samples_generated_total.labels(method=metrics.method).inc(metrics.num_samples)
            self.generation_duration.labels(method=metrics.method).observe(metrics.duration)
            self.active_requests.dec()
            
            # Quality scores
            for metric, score in metrics.quality_scores.items():
                self.quality_scores.labels(method=metrics.method, metric=metric).observe(score)
        
        # End tracing span
        if span and self.enable_tracing:
            span.set_attribute("success", success)
            span.set_attribute("duration", metrics.duration)
            if error:
                span.set_attribute("error", error)
            span.end()
        
        # Remove from current requests
        del self.current_requests[request_id]
        
        logger.info(f"Completed tracking generation request {request_id} "
                   f"(success={success}, duration={metrics.duration:.2f}s)")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        
        now = time.time()
        recent_metrics = [m for m in self.generation_metrics 
                         if now - m.end_time < 3600]  # Last hour
        
        if recent_metrics:
            successful = [m for m in recent_metrics if m.success]
            
            stats = {
                "requests_last_hour": len(recent_metrics),
                "successful_requests": len(successful),
                "failed_requests": len(recent_metrics) - len(successful),
                "success_rate": len(successful) / len(recent_metrics),
                "avg_duration": np.mean([m.duration for m in recent_metrics]),
                "avg_samples_per_request": np.mean([m.num_samples for m in recent_metrics]),
                "total_samples": sum(m.num_samples for m in recent_metrics),
                "active_requests": len(self.current_requests)
            }
            
            # Method breakdown
            method_stats = defaultdict(lambda: {"count": 0, "avg_duration": 0.0})
            for m in recent_metrics:
                method_stats[m.method]["count"] += 1
                
            for method, data in method_stats.items():
                method_metrics = [m for m in recent_metrics if m.method == method]
                data["avg_duration"] = np.mean([m.duration for m in method_metrics])
            
            stats["method_breakdown"] = dict(method_stats)
        else:
            stats = {
                "requests_last_hour": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "avg_samples_per_request": 0.0,
                "total_samples": 0,
                "active_requests": len(self.current_requests),
                "method_breakdown": {}
            }
        
        # System stats
        if self.system_metrics:
            latest_system = self.system_metrics[-1]
            stats["system"] = {
                "cpu_percent": latest_system["cpu_percent"],
                "memory_percent": latest_system["memory_percent"],
                "memory_used_gb": latest_system["memory_used_gb"],
                "memory_available_gb": latest_system["memory_available_gb"]
            }
            
            # Add GPU stats if available
            for key, value in latest_system.items():
                if key.startswith("gpu_"):
                    stats["system"][key] = value
        
        return stats
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over time"""
        
        now = time.time()
        cutoff = now - (hours * 3600)
        
        recent_metrics = [m for m in self.generation_metrics if m.start_time >= cutoff]
        
        if not recent_metrics:
            return {"error": "No data available for the specified time period"}
        
        # Group by hour
        hourly_data = defaultdict(lambda: {
            "requests": 0,
            "successful": 0,
            "avg_duration": 0.0,
            "durations": []
        })
        
        for metric in recent_metrics:
            hour_key = int(metric.start_time // 3600) * 3600  # Round to hour
            hourly_data[hour_key]["requests"] += 1
            hourly_data[hour_key]["durations"].append(metric.duration)
            
            if metric.success:
                hourly_data[hour_key]["successful"] += 1
        
        # Calculate averages
        trends = []
        for hour_timestamp in sorted(hourly_data.keys()):
            data = hourly_data[hour_timestamp]
            
            trends.append({
                "timestamp": hour_timestamp,
                "hour": datetime.fromtimestamp(hour_timestamp).strftime("%Y-%m-%d %H:00"),
                "requests": data["requests"],
                "successful": data["successful"],
                "success_rate": data["successful"] / data["requests"] if data["requests"] > 0 else 0,
                "avg_duration": np.mean(data["durations"]) if data["durations"] else 0,
                "p95_duration": np.percentile(data["durations"], 95) if data["durations"] else 0,
                "p99_duration": np.percentile(data["durations"], 99) if data["durations"] else 0
            })
        
        return {
            "period_hours": hours,
            "total_requests": len(recent_metrics),
            "total_successful": sum(1 for m in recent_metrics if m.success),
            "hourly_trends": trends
        }
    
    def get_quality_analysis(self) -> Dict[str, Any]:
        """Analyze quality metrics across generations"""
        
        recent_metrics = [m for m in self.generation_metrics 
                         if m.quality_scores and time.time() - m.end_time < 86400]  # Last 24 hours
        
        if not recent_metrics:
            return {"error": "No quality data available"}
        
        # Aggregate quality scores by method and metric
        quality_analysis = {}
        
        # Get all unique quality metrics
        all_metrics = set()
        for m in recent_metrics:
            all_metrics.update(m.quality_scores.keys())
        
        for method in ["diffusion", "tts", "gan", "vae"]:
            method_metrics = [m for m in recent_metrics if m.method == method]
            if not method_metrics:
                continue
            
            method_analysis = {}
            
            for metric_name in all_metrics:
                scores = [m.quality_scores.get(metric_name, 0) for m in method_metrics 
                         if metric_name in m.quality_scores]
                
                if scores:
                    method_analysis[metric_name] = {
                        "mean": np.mean(scores),
                        "std": np.std(scores),
                        "min": np.min(scores),
                        "max": np.max(scores),
                        "p50": np.percentile(scores, 50),
                        "p95": np.percentile(scores, 95),
                        "count": len(scores)
                    }
            
            if method_analysis:
                quality_analysis[method] = method_analysis
        
        return quality_analysis
    
    def export_metrics(self, format: str = "json") -> str:
        """Export collected metrics"""
        
        if format == "json":
            export_data = {
                "export_timestamp": time.time(),
                "current_stats": self.get_current_stats(),
                "performance_trends": self.get_performance_trends(),
                "quality_analysis": self.get_quality_analysis(),
                "recent_errors": list(self.error_log)[-50:],  # Last 50 errors
                "system_metrics": list(self.system_metrics)[-60:]  # Last hour
            }
            
            return json.dumps(export_data, indent=2, default=str)
        
        elif format == "prometheus":
            # Export Prometheus metrics
            if self.enable_prometheus:
                from prometheus_client import generate_latest
                return generate_latest(self.registry).decode('utf-8')
            else:
                return "Prometheus not enabled"
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics beyond retention period"""
        
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        # Clean generation metrics
        while (self.generation_metrics and 
               self.generation_metrics[0].end_time < cutoff_time):
            self.generation_metrics.popleft()
        
        # Clean system metrics
        while (self.system_metrics and 
               self.system_metrics[0]["timestamp"] < cutoff_time):
            self.system_metrics.popleft()
        
        # Clean error log
        while (self.error_log and 
               self.error_log[0]["timestamp"] < cutoff_time):
            self.error_log.popleft()
        
        # Clean hourly stats
        cutoff_hour = datetime.fromtimestamp(cutoff_time).strftime("%Y%m%d_%H")
        old_hours = [h for h in self.hourly_stats.keys() if h < cutoff_hour]
        for hour in old_hours:
            del self.hourly_stats[hour]

# ============================================================================

# audio_synth/monitoring/alerting.py
"""
Alerting system for audio synthesis monitoring
"""

import smtplib
import time
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
import requests
import json
import logging

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    metadata: Dict[str, Any]
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "resolved": self.resolved
        }

class AlertRule:
    """Base class for alert rules"""
    
    def __init__(self, 
                 name: str, 
                 severity: AlertSeverity,
                 cooldown_minutes: int = 15):
        self.name = name
        self.severity = severity
        self.cooldown_minutes = cooldown_minutes
        self.last_triggered = 0
        
    def should_trigger(self, metrics: Dict[str, Any]) -> bool:
        """Check if this rule should trigger an alert"""
        raise NotImplementedError
    
    def get_message(self, metrics: Dict[str, Any]) -> str:
        """Get alert message"""
        raise NotImplementedError
    
    def can_trigger(self) -> bool:
        """Check if enough time has passed since last trigger"""
        return time.time() - self.last_triggered > (self.cooldown_minutes * 60)

class HighErrorRateRule(AlertRule):
    """Alert when error rate exceeds threshold"""
    
    def __init__(self, threshold: float = 0.1, **kwargs):
        super().__init__("high_error_rate", AlertSeverity.ERROR, **kwargs)
        self.threshold = threshold
    
    def should_trigger(self, metrics: Dict[str, Any]) -> bool:
        success_rate = metrics.get("success_rate", 1.0)
        error_rate = 1.0 - success_rate
        return error_rate > self.threshold and metrics.get("requests_last_hour", 0) > 10
    
    def get_message(self, metrics: Dict[str, Any]) -> str:
        error_rate = 1.0 - metrics.get("success_rate", 1.0)
        return f"High error rate detected: {error_rate:.1%} (threshold: {self.threshold:.1%})"

class SlowResponseRule(AlertRule):
    """Alert when response time is too slow"""
    
    def __init__(self, threshold_seconds: float = 30.0, **kwargs):
        super().__init__("slow_response", AlertSeverity.WARNING, **kwargs)
        self.threshold_seconds = threshold_seconds
    
    def should_trigger(self, metrics: Dict[str, Any]) -> bool:
        avg_duration = metrics.get("avg_duration", 0.0)
        return avg_duration > self.threshold_seconds and metrics.get("requests_last_hour", 0) > 5
    
    def get_message(self, metrics: Dict[str, Any]) -> str:
        avg_duration = metrics.get("avg_duration", 0.0)
        return f"Slow response time: {avg_duration:.1f}s (threshold: {self.threshold_seconds}s)"

class HighMemoryUsageRule(AlertRule):
    """Alert when memory usage is high"""
    
    def __init__(self, threshold_percent: float = 85.0, **kwargs):
        super().__init__("high_memory_usage", AlertSeverity.WARNING, **kwargs)
        self.threshold_percent = threshold_percent
    
    def should_trigger(self, metrics: Dict[str, Any]) -> bool:
        system = metrics.get("system", {})
        memory_percent = system.get("memory_percent", 0.0)
        return memory_percent > self.threshold_percent
    
    def get_message(self, metrics: Dict[str, Any]) -> str:
        system = metrics.get("system", {})
        memory_percent = system.get("memory_percent", 0.0)
        return f"High memory usage: {memory_percent:.1f}% (threshold: {self.threshold_percent}%)"

class LowQualityRule(AlertRule):
    """Alert when quality scores are consistently low"""
    
    def __init__(self, threshold: float = 0.7, **kwargs):
        super().__init__("low_quality", AlertSeverity.WARNING, **kwargs)
        self.threshold = threshold
    
    def should_trigger(self, metrics: Dict[str, Any]) -> bool:
        # This would need access to quality analysis data
        # For now, just a placeholder
        return False
    
    def get_message(self, metrics: Dict[str, Any]) -> str:
        return f"Quality scores below threshold: {self.threshold}"

class AlertManager:
    """Manages alert rules and notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels = []
        
        # Setup notification channels
        self._setup_notification_channels()
        
        # Setup default rules
        self._setup_default_rules()
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Alert manager initialized")
    
    def _setup_notification_channels(self):
        """Setup notification channels"""
        
        channels_config = self.config.get("notification_channels", {})
        
        # Email notifications
        if "email" in channels_config:
            email_config = channels_config["email"]
            self.notification_channels.append(
                EmailNotifier(
                    smtp_server=email_config["smtp_server"],
                    smtp_port=email_config["smtp_port"],
                    username=email_config["username"],
                    password=email_config["password"],
                    recipients=email_config["recipients"]
                )
            )
        
        # Slack notifications
        if "slack" in channels_config:
            slack_config = channels_config["slack"]
            self.notification_channels.append(
                SlackNotifier(webhook_url=slack_config["webhook_url"])
            )
        
        # PagerDuty notifications
        if "pagerduty" in channels_config:
            pd_config = channels_config["pagerduty"]
            self.notification_channels.append(
                PagerDutyNotifier(integration_key=pd_config["integration_key"])
            )
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        
        rules_config = self.config.get("alert_rules", {})
        
        # High error rate
        if rules_config.get("high_error_rate", {}).get("enabled", True):
            threshold = rules_config.get("high_error_rate", {}).get("threshold", 0.1)
            self.add_rule(HighErrorRateRule(threshold=threshold))
        
        # Slow response
        if rules_config.get("slow_response", {}).get("enabled", True):
            threshold = rules_config.get("slow_response", {}).get("threshold_seconds", 30.0)
            self.add_rule(SlowResponseRule(threshold_seconds=threshold))
        
        # High memory usage
        if rules_config.get("high_memory_usage", {}).get("enabled", True):
            threshold = rules_config.get("high_memory_usage", {}).get("threshold_percent", 85.0)
            self.add_rule(HighMemoryUsageRule(threshold_percent=threshold))
        
        # Low quality
        if rules_config.get("low_quality", {}).get("enabled", False):
            threshold = rules_config.get("low_quality", {}).get("threshold", 0.7)
            self.add_rule(LowQualityRule(threshold=threshold))
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all rules against current metrics"""
        
        for rule in self.rules:
            if rule.should_trigger(metrics) and rule.can_trigger():
                # Create alert
                alert = Alert(
                    name=rule.name,
                    severity=rule.severity,
                    message=rule.get_message(metrics),
                    timestamp=time.time(),
                    metadata={"metrics": metrics}
                )
                
                # Add to active alerts
                self.active_alerts[rule.name] = alert
                self.alert_history.append(alert)
                
                # Update rule trigger time
                rule.last_triggered = time.time()
                
                # Send notifications
                self._send_alert(alert)
                
                logger.warning(f"Alert triggered: {rule.name} - {alert.message}")
    
    def resolve_alert(self, alert_name: str):
        """Manually resolve an alert"""
        if alert_name in self.active_alerts:
            alert = self.active_alerts[alert_name]
            alert.resolved = True
            del self.active_alerts[alert_name]
            
            # Send resolution notification
            self._send_resolution(alert)
            
            logger.info(f"Alert resolved: {alert_name}")
    
    def _send_alert(self, alert: Alert):
        """Send alert to all notification channels"""
        for channel in self.notification_channels:
            try:
                channel.send_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {type(channel).__name__}: {e}")
    
    def _send_resolution(self, alert: Alert):
        """Send resolution notification"""
        for channel in self.notification_channels:
            try:
                if hasattr(channel, 'send_resolution'):
                    channel.send_resolution(alert)
            except Exception as e:
                logger.error(f"Failed to send resolution via {type(channel).__name__}: {e}")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        # This would be connected to the MetricsCollector
        # For now, just a placeholder
        pass
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history"""
        cutoff_time = time.time() - (hours * 3600)
        recent_alerts = [alert for alert in self.alert_history 
                        if alert.timestamp >= cutoff_time]
        return [alert.to_dict() for alert in recent_alerts]

class EmailNotifier:
    """Email notification channel"""
    
    def __init__(self, smtp_server: str, smtp_port: int, 
                 username: str, password: str, recipients: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients
    
    def send_alert(self, alert: Alert):
        """Send alert via email"""
        
        subject = f"[{alert.severity.value.upper()}] Audio Synthesis Alert: {alert.name}"
        
        body = f"""
Alert: {alert.name}
Severity: {alert.severity.value.upper()}
Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}

Message: {alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}

---
Audio Synthesis Monitoring System
        """
        
        msg = MIMEMultipart()
        msg['From'] = self.username
        msg['To'] = ', '.join(self.recipients)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.username, self.recipients, msg.as_string())
            
            logger.info(f"Email alert sent: {alert.name}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

class SlackNotifier:
    """Slack notification channel"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_alert(self, alert: Alert):
        """Send alert via Slack"""
        
        color_map = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "danger",
            AlertSeverity.CRITICAL: "danger"
        }
        
        payload = {
            "text": f"Audio Synthesis Alert: {alert.name}",
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "danger"),
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": time.strftime('%Y-%m-%d %H:%M:%S', 
                                               time.localtime(alert.timestamp)),
                            "short": True
                        },
                        {
                            "title": "Message",
                            "value": alert.message,
                            "short": False
                        }
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(self.webhook_url, 
                                   data=json.dumps(payload),
                                   headers={'Content-Type': 'application/json'})
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {alert.name}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

class PagerDutyNotifier:
    """PagerDuty notification channel"""
    
    def __init__(self, integration_key: str):
        self.integration_key = integration_key
        self.api_url = "https://events.pagerduty.com/v2/enqueue"
    
    def send_alert(self, alert: Alert):
        """Send alert via PagerDuty"""
        
        severity_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning", 
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical"
        }
        
        payload = {
            "routing_key": self.integration_key,
            "event_action": "trigger",
            "dedup_key": f"audio_synthesis_{alert.name}",
            "payload": {
                "summary": f"Audio Synthesis Alert: {alert.name}",
                "severity": severity_map.get(alert.severity, "error"),
                "source": "audio_synthesis_monitoring",
                "component": "audio_generator",
                "group": "production",
                "class": "alert",
                "custom_details": {
                    "message": alert.message,
                    "metadata": alert.metadata
                }
            }
        }
        
        try:
            response = requests.post(self.api_url,
                                   data=json.dumps(payload),
                                   headers={'Content-Type': 'application/json'})
            response.raise_for_status()
            
            logger.info(f"PagerDuty alert sent: {alert.name}")
            
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")

# ============================================================================

# audio_synth/monitoring/dashboard.py
"""
Real-time monitoring dashboard
"""

from flask import Flask, render_template, jsonify, request
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class MonitoringDashboard:
    """Real-time monitoring dashboard for audio synthesis"""
    
    def __init__(self, metrics_collector, alert_manager, port: int = 5000):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.port = port
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self._setup_routes()
        
        logger.info(f"Monitoring dashboard initialized on port {port}")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/stats')
        def get_stats():
            """Get current statistics"""
            return jsonify(self.metrics_collector.get_current_stats())
        
        @self.app.route('/api/trends')
        def get_trends():
            """Get performance trends"""
            hours = request.args.get('hours', 24, type=int)
            return jsonify(self.metrics_collector.get_performance_trends(hours))
        
        @self.app.route('/api/quality')
        def get_quality():
            """Get quality analysis"""
            return jsonify(self.metrics_collector.get_quality_analysis())
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """Get active alerts"""
            return jsonify({
                "active": self.alert_manager.get_active_alerts(),
                "history": self.alert_manager.get_alert_history()
            })
        
        @self.app.route('/api/alerts/<alert_name>/resolve', methods=['POST'])
        def resolve_alert(alert_name):
            """Resolve an alert"""
            self.alert_manager.resolve_alert(alert_name)
            return jsonify({"status": "resolved"})
        
        @self.app.route('/api/export')
        def export_metrics():
            """Export metrics"""
            format_type = request.args.get('format', 'json')
            
            try:
                data = self.metrics_collector.export_metrics(format_type)
                
                if format_type == 'json':
                    return jsonify(json.loads(data))
                else:
                    return data, 200, {'Content-Type': 'text/plain'}
                    
            except Exception as e:
                return jsonify({"error": str(e)}), 400
        
        @self.app.route('/health')
        def health():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "timestamp": time.time(),
                "metrics_collector": "active",
                "alert_manager": "active"
            })
    
    def run(self, debug: bool = False):
        """Run the dashboard"""
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)

# Dashboard HTML template would be saved as templates/dashboard.html
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Synthesis Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .chart-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        .alerts-section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .alert {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid;
        }
        .alert.error {
            background-color: #ffeaea;
            border-color: #e74c3c;
        }
        .alert.warning {
            background-color: #fff3cd;
            border-color: #f39c12;
        }
        .alert.info {
            background-color: #d1ecf1;
            border-color: #17a2b8;
        }
        .resolve-btn {
            background-color: #28a745;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
            float: right;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-good { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-error { background-color: #dc3545; }
        .refresh-info {
            text-align: center;
            color: #666;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎵 Audio Synthesis Monitoring Dashboard</h1>
        <p>Real-time performance and quality monitoring</p>
        <div id="lastUpdate" class="refresh-info">Loading...</div>
    </div>

    <div class="stats-grid" id="statsGrid">
        <!-- Stats cards will be populated here -->
    </div>

    <div class="chart-container">
        <div class="chart-title">Request Volume & Success Rate</div>
        <canvas id="requestsChart" height="100"></canvas>
    </div>

    <div class="chart-container">
        <div class="chart-title">Response Time Trends</div>
        <canvas id="responseTimeChart" height="100"></canvas>
    </div>

    <div class="chart-container">
        <div class="chart-title">Method Performance Comparison</div>
        <canvas id="methodsChart" height="100"></canvas>
    </div>

    <div class="alerts-section">
        <h2>🚨 Active Alerts</h2>
        <div id="alertsList">
            <!-- Alerts will be populated here -->
        </div>
    </div>

    <script>
        // Initialize charts
        let requestsChart, responseTimeChart, methodsChart;
        
        function initializeCharts() {
            // Requests chart
            const ctx1 = document.getElementById('requestsChart').getContext('2d');
            requestsChart = new Chart(ctx1, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Requests',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        yAxisID: 'y'
                    }, {
                        label: 'Success Rate (%)',
                        data: [],
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            min: 0,
                            max: 100,
                            grid: {
                                drawOnChartArea: false,
                            },
                        }
                    }
                }
            });

            // Response time chart
            const ctx2 = document.getElementById('responseTimeChart').getContext('2d');
            responseTimeChart = new Chart(ctx2, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Avg Response Time (s)',
                        data: [],
                        borderColor: '#f39c12',
                        backgroundColor: 'rgba(243, 156, 18, 0.1)'
                    }, {
                        label: 'P95 Response Time (s)',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });

            // Methods chart
            const ctx3 = document.getElementById('methodsChart').getContext('2d');
            methodsChart = new Chart(ctx3, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Requests',
                        data: [],
                        backgroundColor: ['#667eea', '#764ba2', '#f093fb', '#f5576c']
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }

        function updateStats(stats) {
            const statsGrid = document.getElementById('statsGrid');
            const successRate = (stats.success_rate * 100).toFixed(1);
            const statusClass = stats.success_rate > 0.95 ? 'status-good' : 
                              stats.success_rate > 0.8 ? 'status-warning' : 'status-error';
            
            statsGrid.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${stats.requests_last_hour}</div>
                    <div class="stat-label">Requests (Last Hour)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">
                        <span class="status-indicator ${statusClass}"></span>
                        ${successRate}%
                    </div>
                    <div class="stat-label">Success Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.avg_duration.toFixed(2)}s</div>
                    <div class="stat-label">Avg Response Time</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.total_samples}</div>
                    <div class="stat-label">Samples Generated</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.active_requests}</div>
                    <div class="stat-label">Active Requests</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.system?.memory_percent?.toFixed(1) || 'N/A'}%</div>
                    <div class="stat-label">Memory Usage</div>
                </div>
            `;
        }

        function updateTrends(trends) {
            if (!trends.hourly_trends || trends.hourly_trends.length === 0) return;

            const labels = trends.hourly_trends.map(t => new Date(t.timestamp * 1000).toLocaleTimeString());
            const requests = trends.hourly_trends.map(t => t.requests);
            const successRates = trends.hourly_trends.map(t => t.success_rate * 100);
            const avgTimes = trends.hourly_trends.map(t => t.avg_duration);
            const p95Times = trends.hourly_trends.map(t => t.p95_duration);

            // Update requests chart
            requestsChart.data.labels = labels;
            requestsChart.data.datasets[0].data = requests;
            requestsChart.data.datasets[1].data = successRates;
            requestsChart.update();

            // Update response time chart
            responseTimeChart.data.labels = labels;
            responseTimeChart.data.datasets[0].data = avgTimes;
            responseTimeChart.data.datasets[1].data = p95Times;
            responseTimeChart.update();
        }

        function updateMethods(stats) {
            if (!stats.method_breakdown) return;

            const methods = Object.keys(stats.method_breakdown);
            const counts = methods.map(m => stats.method_breakdown[m].count);

            methodsChart.data.labels = methods.map(m => m.toUpperCase());
            methodsChart.data.datasets[0].data = counts;
            methodsChart.update();
        }

        function updateAlerts(alertsData) {
            const alertsList = document.getElementById('alertsList');
            const activeAlerts = alertsData.active || [];

            if (activeAlerts.length === 0) {
                alertsList.innerHTML = '<p style="color: #28a745;">✅ No active alerts</p>';
                return;
            }

            alertsList.innerHTML = activeAlerts.map(alert => `
                <div class="alert ${alert.severity}">
                    <strong>${alert.name}</strong> - ${alert.message}
                    <button class="resolve-btn" onclick="resolveAlert('${alert.name}')">Resolve</button>
                    <br>
                    <small>${new Date(alert.timestamp * 1000).toLocaleString()}</small>
                </div>
            `).join('');
        }

        function resolveAlert(alertName) {
            axios.post(`/api/alerts/${alertName}/resolve`)
                .then(() => {
                    loadAlerts();
                })
                .catch(err => {
                    console.error('Failed to resolve alert:', err);
                });
        }

        async function loadData() {
            try {
                const [stats, trends, alerts] = await Promise.all([
                    axios.get('/api/stats'),
                    axios.get('/api/trends?hours=24'),
                    axios.get('/api/alerts')
                ]);

                updateStats(stats.data);
                updateTrends(trends.data);
                updateMethods(stats.data);
                updateAlerts(alerts.data);

                document.getElementById('lastUpdate').textContent = 
                    `Last updated: ${new Date().toLocaleTimeString()}`;

            } catch (error) {
                console.error('Failed to load data:', error);
                document.getElementById('lastUpdate').textContent = 
                    `⚠️ Error loading data: ${error.message}`;
            }
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initializeCharts();
            loadData();
            
            // Refresh every 30 seconds
            setInterval(loadData, 30000);
        });
    </script>
</body>
</html>
'''

