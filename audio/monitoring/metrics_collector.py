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


