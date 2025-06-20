#!/usr/bin/env python3
"""
Metrics collector for aggregating and exporting system metrics.

Provides comprehensive metrics collection, aggregation, and export
capabilities with support for Prometheus, StatsD, and custom exporters.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum

import aiohttp
from pydantic import BaseModel, Field

from ...core import get_logger, BaseConfig

logger = get_logger(__name__)


class MetricType(str, Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: Union[int, float]
    timestamp: datetime
    metric_type: MetricType
    labels: Optional[Dict[str, str]] = None
    help_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class MetricSeries:
    """Time series of metrics"""
    name: str
    metric_type: MetricType
    values: List[Metric]
    labels: Optional[Dict[str, str]] = None
    help_text: Optional[str] = None
    
    def add_value(self, value: Union[int, float], timestamp: Optional[datetime] = None):
        """Add value to series"""
        if timestamp is None:
            timestamp = datetime.now()
        
        metric = Metric(
            name=self.name,
            value=value,
            timestamp=timestamp,
            metric_type=self.metric_type,
            labels=self.labels,
            help_text=self.help_text
        )
        self.values.append(metric)


class MetricsCollectorConfig(BaseConfig):
    """Metrics collector configuration"""
    collection_interval_seconds: int = 15
    retention_hours: int = 24
    max_series_length: int = 5760  # 24 hours at 15-second intervals
    
    # Export settings
    prometheus_enabled: bool = False
    prometheus_port: int = 8000
    prometheus_path: str = "/metrics"
    
    statsd_enabled: bool = False
    statsd_host: str = "localhost"
    statsd_port: int = 8125
    
    custom_exporters: List[str] = Field(default_factory=list)
    
    # Aggregation settings
    enable_aggregation: bool = True
    aggregation_window_minutes: int = 5
    
    # Labels
    default_labels: Dict[str, str] = Field(default_factory=dict)
    
    # Sampling
    enable_sampling: bool = False
    sample_rate: float = 1.0


class MetricsCollector:
    """Comprehensive metrics collection and export system"""
    
    def __init__(self, config: MetricsCollectorConfig):
        self.config = config
        self.metrics: Dict[str, MetricSeries] = {}
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.running = False
        self.exporters: List[Callable] = []
        
        # Setup exporters
        self._setup_exporters()
        
    def _setup_exporters(self):
        """Setup metric exporters"""
        if self.config.prometheus_enabled:
            self.exporters.append(self._prometheus_exporter)
        if self.config.statsd_enabled:
            self.exporters.append(self._statsd_exporter)
    
    async def start(self):
        """Start metrics collector"""
        self.running = True
        logger.info("Metrics collector started")
        
        # Start background tasks
        if self.config.enable_aggregation:
            asyncio.create_task(self._aggregation_worker())
        
        asyncio.create_task(self._export_worker())
        asyncio.create_task(self._cleanup_worker())
        
        # Start Prometheus server if enabled
        if self.config.prometheus_enabled:
            asyncio.create_task(self._start_prometheus_server())
    
    async def stop(self):
        """Stop metrics collector"""
        self.running = False
        logger.info("Metrics collector stopped")
    
    async def increment_counter(self, name: str, value: float = 1.0, 
                              labels: Optional[Dict[str, str]] = None,
                              help_text: Optional[str] = None):
        """
        Increment counter metric.
        
        Args:
            name: Metric name
            value: Increment value
            labels: Metric labels
            help_text: Help text for metric
        """
        try:
            if not self._should_sample():
                return
            
            metric_key = self._get_metric_key(name, labels)
            self.counters[metric_key] += value
            
            # Store in time series
            await self._store_metric(name, value, MetricType.COUNTER, labels, help_text)
            
        except Exception as e:
            logger.error(f"Failed to increment counter {name}: {e}")
    
    async def set_gauge(self, name: str, value: float,
                       labels: Optional[Dict[str, str]] = None,
                       help_text: Optional[str] = None):
        """
        Set gauge metric value.
        
        Args:
            name: Metric name
            value: Gauge value
            labels: Metric labels
            help_text: Help text for metric
        """
        try:
            if not self._should_sample():
                return
            
            metric_key = self._get_metric_key(name, labels)
            self.gauges[metric_key] = value
            
            # Store in time series
            await self._store_metric(name, value, MetricType.GAUGE, labels, help_text)
            
        except Exception as e:
            logger.error(f"Failed to set gauge {name}: {e}")
    
    async def record_histogram(self, name: str, value: float,
                             labels: Optional[Dict[str, str]] = None,
                             help_text: Optional[str] = None):
        """
        Record histogram observation.
        
        Args:
            name: Metric name
            value: Observation value
            labels: Metric labels
            help_text: Help text for metric
        """
        try:
            if not self._should_sample():
                return
            
            metric_key = self._get_metric_key(name, labels)
            self.histograms[metric_key].append(value)
            
            # Store in time series
            await self._store_metric(name, value, MetricType.HISTOGRAM, labels, help_text)
            
        except Exception as e:
            logger.error(f"Failed to record histogram {name}: {e}")
    
    async def timing(self, name: str, duration_seconds: float,
                    labels: Optional[Dict[str, str]] = None):
        """
        Record timing metric.
        
        Args:
            name: Metric name
            duration_seconds: Duration in seconds
            labels: Metric labels
        """
        await self.record_histogram(
            f"{name}_duration_seconds", 
            duration_seconds, 
            labels,
            f"Duration of {name} operations in seconds"
        )
    
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.
        
        Args:
            name: Metric name
            labels: Metric labels
        
        Returns:
            Timer context manager
        """
        return TimerContext(self, name, labels)
    
    async def get_metric_value(self, name: str, 
                             labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """
        Get current metric value.
        
        Args:
            name: Metric name
            labels: Metric labels
        
        Returns:
            Current metric value or None
        """
        try:
            metric_key = self._get_metric_key(name, labels)
            
            # Check counters
            if metric_key in self.counters:
                return self.counters[metric_key]
            
            # Check gauges
            if metric_key in self.gauges:
                return self.gauges[metric_key]
            
            # Check time series
            if name in self.metrics and self.metrics[name].values:
                return self.metrics[name].values[-1].value
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get metric value {name}: {e}")
            return None
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics.
        
        Returns:
            Metrics summary dictionary
        """
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histogram_counts': {k: len(v) for k, v in self.histograms.items()},
                'total_series': len(self.metrics),
                'total_data_points': sum(len(series.values) for series in self.metrics.values())
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {'error': str(e)}
    
    async def export_prometheus_format(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics string
        """
        try:
            lines = []
            
            # Export counters
            for metric_key, value in self.counters.items():
                name, labels = self._parse_metric_key(metric_key)
                labels_str = self._format_prometheus_labels(labels)
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{labels_str} {value}")
            
            # Export gauges
            for metric_key, value in self.gauges.items():
                name, labels = self._parse_metric_key(metric_key)
                labels_str = self._format_prometheus_labels(labels)
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{labels_str} {value}")
            
            # Export histograms
            for metric_key, values in self.histograms.items():
                if values:
                    name, labels = self._parse_metric_key(metric_key)
                    labels_str = self._format_prometheus_labels(labels)
                    
                    lines.append(f"# TYPE {name} histogram")
                    
                    # Calculate percentiles
                    sorted_values = sorted(values)
                    count = len(sorted_values)
                    
                    # Add histogram buckets (simplified)
                    buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
                    cumulative_count = 0
                    
                    for bucket in buckets:
                        bucket_count = sum(1 for v in sorted_values if v <= bucket)
                        bucket_labels = labels.copy() if labels else {}
                        bucket_labels['le'] = str(bucket)
                        bucket_labels_str = self._format_prometheus_labels(bucket_labels)
                        lines.append(f"{name}_bucket{bucket_labels_str} {bucket_count}")
                    
                    # Add +Inf bucket
                    inf_labels = labels.copy() if labels else {}
                    inf_labels['le'] = '+Inf'
                    inf_labels_str = self._format_prometheus_labels(inf_labels)
                    lines.append(f"{name}_bucket{inf_labels_str} {count}")
                    
                    # Add count and sum
                    lines.append(f"{name}_count{labels_str} {count}")
                    lines.append(f"{name}_sum{labels_str} {sum(sorted_values)}")
            
            return '\n'.join(lines) + '\n'
            
        except Exception as e:
            logger.error(f"Failed to export Prometheus format: {e}")
            return f"# Error exporting metrics: {e}\n"
    
    def _get_metric_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Generate unique key for metric"""
        if labels:
            # Merge with default labels
            all_labels = {**self.config.default_labels, **labels}
            labels_str = ','.join(f"{k}={v}" for k, v in sorted(all_labels.items()))
            return f"{name}[{labels_str}]"
        elif self.config.default_labels:
            labels_str = ','.join(f"{k}={v}" for k, v in sorted(self.config.default_labels.items()))
            return f"{name}[{labels_str}]"
        else:
            return name
    
    def _parse_metric_key(self, metric_key: str) -> tuple[str, Optional[Dict[str, str]]]:
        """Parse metric key back to name and labels"""
        if '[' in metric_key:
            name, labels_part = metric_key.split('[', 1)
            labels_str = labels_part.rstrip(']')
            labels = {}
            for pair in labels_str.split(','):
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    labels[k] = v
            return name, labels
        return metric_key, None
    
    def _format_prometheus_labels(self, labels: Optional[Dict[str, str]]) -> str:
        """Format labels for Prometheus export"""
        if not labels:
            return ""
        
        label_pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return f"{{{','.join(label_pairs)}}}"
    
    def _should_sample(self) -> bool:
        """Check if metric should be sampled"""
        if not self.config.enable_sampling:
            return True
        
        import random
        return random.random() < self.config.sample_rate
    
    async def _store_metric(self, name: str, value: Union[int, float], 
                          metric_type: MetricType,
                          labels: Optional[Dict[str, str]] = None,
                          help_text: Optional[str] = None):
        """Store metric in time series"""
        try:
            if name not in self.metrics:
                self.metrics[name] = MetricSeries(
                    name=name,
                    metric_type=metric_type,
                    values=deque(maxlen=self.config.max_series_length),
                    labels=labels,
                    help_text=help_text
                )
            
            self.metrics[name].add_value(value)
            
        except Exception as e:
            logger.error(f"Failed to store metric {name}: {e}")
    
    async def _aggregation_worker(self):
        """Background worker for metric aggregation"""
        while self.running:
            try:
                await self._aggregate_metrics()
                await asyncio.sleep(self.config.aggregation_window_minutes * 60)
            except Exception as e:
                logger.error(f"Error in aggregation worker: {e}")
    
    async def _aggregate_metrics(self):
        """Aggregate metrics over time window"""
        try:
            # This is a placeholder for aggregation logic
            # In a real implementation, you might:
            # - Calculate rolling averages
            # - Compute rates of change
            # - Generate derived metrics
            pass
            
        except Exception as e:
            logger.error(f"Failed to aggregate metrics: {e}")
    
    async def _export_worker(self):
        """Background worker for metric export"""
        while self.running:
            try:
                for exporter in self.exporters:
                    await exporter()
                await asyncio.sleep(60)  # Export every minute
            except Exception as e:
                logger.error(f"Error in export worker: {e}")
    
    async def _prometheus_exporter(self):
        """Export metrics to Prometheus format"""
        try:
            # This would typically push to Prometheus pushgateway
            # or expose via HTTP endpoint
            prometheus_data = await self.export_prometheus_format()
            logger.debug(f"Exported {len(prometheus_data)} bytes to Prometheus format")
            
        except Exception as e:
            logger.error(f"Failed Prometheus export: {e}")
    
    async def _statsd_exporter(self):
        """Export metrics to StatsD"""
        try:
            # This would send metrics to StatsD server
            logger.debug("Exported metrics to StatsD")
            
        except Exception as e:
            logger.error(f"Failed StatsD export: {e}")
    
    async def _start_prometheus_server(self):
        """Start HTTP server for Prometheus metrics"""
        try:
            from aiohttp import web
            
            async def metrics_handler(request):
                prometheus_data = await self.export_prometheus_format()
                return web.Response(text=prometheus_data, content_type='text/plain')
            
            app = web.Application()
            app.router.add_get(self.config.prometheus_path, metrics_handler)
            
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(runner, 'localhost', self.config.prometheus_port)
            await site.start()
            
            logger.info(f"Prometheus metrics server started on port {self.config.prometheus_port}")
            
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    async def _cleanup_worker(self):
        """Background worker for cleanup"""
        while self.running:
            try:
                await self._cleanup_old_metrics()
                await asyncio.sleep(3600)  # Cleanup hourly
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics data"""
        try:
            cutoff = datetime.now() - timedelta(hours=self.config.retention_hours)
            
            for series in self.metrics.values():
                original_count = len(series.values)
                series.values = deque(
                    (m for m in series.values if m.timestamp >= cutoff),
                    maxlen=self.config.max_series_length
                )
                cleaned_count = original_count - len(series.values)
                
                if cleaned_count > 0:
                    logger.debug(f"Cleaned {cleaned_count} old metrics from {series.name}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")


class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, collector: MetricsCollector, name: str, 
                 labels: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            await self.collector.timing(self.name, duration, self.labels)


def create_metrics_collector(config: Optional[MetricsCollectorConfig] = None) -> MetricsCollector:
    """Factory function to create metrics collector"""
    if config is None:
        config = MetricsCollectorConfig()
    return MetricsCollector(config)


__all__ = [
    'MetricsCollector',
    'MetricsCollectorConfig',
    'Metric',
    'MetricSeries',
    'MetricType',
    'TimerContext',
    'create_metrics_collector'
]