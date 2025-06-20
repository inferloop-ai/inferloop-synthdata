#!/usr/bin/env python3
"""
Performance monitor for tracking system performance metrics.

Provides comprehensive performance monitoring including CPU, memory,
disk usage, network metrics, and application-specific performance tracking.
"""

import asyncio
import psutil
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque

from pydantic import BaseModel, Field

from ...core import get_logger, BaseConfig

logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class SystemMetrics:
    """System performance metrics snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class PerformanceMonitorConfig(BaseConfig):
    """Performance monitor configuration"""
    collection_interval_seconds: int = 60
    retention_hours: int = 24
    max_metrics_per_type: int = 1440  # 24 hours at 1-minute intervals
    
    # Thresholds for alerting
    cpu_warning_threshold: float = 80.0
    cpu_critical_threshold: float = 95.0
    memory_warning_threshold: float = 80.0
    memory_critical_threshold: float = 95.0
    disk_warning_threshold: float = 85.0
    disk_critical_threshold: float = 95.0
    
    # Network monitoring
    enable_network_monitoring: bool = True
    network_interface: Optional[str] = None
    
    # Process monitoring
    enable_process_monitoring: bool = True
    monitored_processes: List[str] = Field(default_factory=list)
    
    # Custom metrics
    enable_custom_metrics: bool = True


class PerformanceMonitor:
    """Comprehensive system performance monitoring"""
    
    def __init__(self, config: PerformanceMonitorConfig):
        self.config = config
        self.metrics_history: Dict[str, deque] = {}
        self.system_metrics_history: deque = deque(
            maxlen=config.max_metrics_per_type
        )
        self.custom_metrics: Dict[str, deque] = {}
        self.running = False
        self.alert_callbacks: List[Callable] = []
        
        # Initialize metrics storage
        self._init_metrics_storage()
        
    def _init_metrics_storage(self):
        """Initialize metrics storage"""
        metric_types = [
            'cpu_percent', 'memory_percent', 'memory_used_gb', 'memory_available_gb',
            'disk_percent', 'disk_used_gb', 'disk_free_gb',
            'network_bytes_sent', 'network_bytes_recv', 'process_count'
        ]
        
        for metric_type in metric_types:
            self.metrics_history[metric_type] = deque(
                maxlen=self.config.max_metrics_per_type
            )
    
    async def start(self):
        """Start performance monitoring"""
        self.running = True
        logger.info("Performance monitor started")
        
        # Start collection task
        asyncio.create_task(self._collection_worker())
        
    async def stop(self):
        """Stop performance monitoring"""
        self.running = False
        logger.info("Performance monitor stopped")
    
    def add_alert_callback(self, callback: Callable[[str, float, float], None]):
        """
        Add callback for performance alerts.
        
        Args:
            callback: Function called when threshold exceeded
        """
        self.alert_callbacks.append(callback)
    
    async def collect_system_metrics(self) -> SystemMetrics:
        """
        Collect current system metrics.
        
        Returns:
            SystemMetrics object with current values
        """
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_used_gb = disk.used / (1024**3)
            disk_free_gb = disk.free / (1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix-like systems)
            try:
                load_average = list(psutil.getloadavg())
            except (AttributeError, OSError):
                load_average = [0.0, 0.0, 0.0]
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_available_gb=memory_available_gb,
                disk_percent=disk_percent,
                disk_used_gb=disk_used_gb,
                disk_free_gb=disk_free_gb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                process_count=process_count,
                load_average=load_average
            )
            
            # Store metrics
            self.system_metrics_history.append(metrics)
            
            # Store individual metrics
            self.metrics_history['cpu_percent'].append(
                PerformanceMetric(metrics.timestamp, 'cpu_percent', cpu_percent, '%')
            )
            self.metrics_history['memory_percent'].append(
                PerformanceMetric(metrics.timestamp, 'memory_percent', memory_percent, '%')
            )
            self.metrics_history['disk_percent'].append(
                PerformanceMetric(metrics.timestamp, 'disk_percent', disk_percent, '%')
            )
            
            # Check thresholds
            await self._check_thresholds(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            raise
    
    async def record_custom_metric(self, name: str, value: float, 
                                 unit: str = '', 
                                 metadata: Optional[Dict[str, Any]] = None):
        """
        Record custom application metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            metadata: Additional metadata
        """
        try:
            if not self.config.enable_custom_metrics:
                return
            
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_name=name,
                value=value,
                unit=unit,
                metadata=metadata
            )
            
            if name not in self.custom_metrics:
                self.custom_metrics[name] = deque(
                    maxlen=self.config.max_metrics_per_type
                )
            
            self.custom_metrics[name].append(metric)
            logger.debug(f"Recorded custom metric: {name} = {value} {unit}")
            
        except Exception as e:
            logger.error(f"Failed to record custom metric {name}: {e}")
    
    async def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """
        Get performance metrics summary.
        
        Args:
            hours: Number of hours to include in summary
        
        Returns:
            Metrics summary dictionary
        """
        try:
            cutoff = datetime.now() - timedelta(hours=hours)
            
            # Filter recent metrics
            recent_metrics = [
                m for m in self.system_metrics_history 
                if m.timestamp >= cutoff
            ]
            
            if not recent_metrics:
                return {'error': 'No metrics available for the specified period'}
            
            # Calculate statistics
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_percent for m in recent_metrics]
            disk_values = [m.disk_percent for m in recent_metrics]
            
            summary = {
                'period_hours': hours,
                'samples_count': len(recent_metrics),
                'cpu': {
                    'current': recent_metrics[-1].cpu_percent,
                    'average': sum(cpu_values) / len(cpu_values),
                    'min': min(cpu_values),
                    'max': max(cpu_values)
                },
                'memory': {
                    'current': recent_metrics[-1].memory_percent,
                    'average': sum(memory_values) / len(memory_values),
                    'min': min(memory_values),
                    'max': max(memory_values),
                    'used_gb': recent_metrics[-1].memory_used_gb,
                    'available_gb': recent_metrics[-1].memory_available_gb
                },
                'disk': {
                    'current': recent_metrics[-1].disk_percent,
                    'average': sum(disk_values) / len(disk_values),
                    'min': min(disk_values),
                    'max': max(disk_values),
                    'used_gb': recent_metrics[-1].disk_used_gb,
                    'free_gb': recent_metrics[-1].disk_free_gb
                },
                'network': {
                    'bytes_sent': recent_metrics[-1].network_bytes_sent,
                    'bytes_recv': recent_metrics[-1].network_bytes_recv
                },
                'system': {
                    'process_count': recent_metrics[-1].process_count,
                    'load_average': recent_metrics[-1].load_average
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {'error': str(e)}
    
    async def get_custom_metrics_summary(self, metric_name: str, 
                                       hours: int = 1) -> Dict[str, Any]:
        """
        Get custom metrics summary.
        
        Args:
            metric_name: Name of custom metric
            hours: Number of hours to include
        
        Returns:
            Custom metrics summary
        """
        try:
            if metric_name not in self.custom_metrics:
                return {'error': f'Metric {metric_name} not found'}
            
            cutoff = datetime.now() - timedelta(hours=hours)
            
            # Filter recent metrics
            recent_metrics = [
                m for m in self.custom_metrics[metric_name] 
                if m.timestamp >= cutoff
            ]
            
            if not recent_metrics:
                return {'error': 'No data available for the specified period'}
            
            values = [m.value for m in recent_metrics]
            
            summary = {
                'metric_name': metric_name,
                'period_hours': hours,
                'samples_count': len(recent_metrics),
                'unit': recent_metrics[-1].unit,
                'current': recent_metrics[-1].value,
                'average': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'total': sum(values) if recent_metrics[-1].unit in ['count', 'requests'] else None
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get custom metrics summary: {e}")
            return {'error': str(e)}
    
    async def get_process_metrics(self, process_name: str) -> Dict[str, Any]:
        """
        Get metrics for specific process.
        
        Args:
            process_name: Name of process to monitor
        
        Returns:
            Process metrics dictionary
        """
        try:
            process_metrics = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if process_name.lower() in proc.info['name'].lower():
                        process_metrics.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_percent': proc.info['memory_percent']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return {
                'process_name': process_name,
                'instances_found': len(process_metrics),
                'processes': process_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get process metrics for {process_name}: {e}")
            return {'error': str(e)}
    
    async def _collection_worker(self):
        """Background worker for metrics collection"""
        while self.running:
            try:
                await self.collect_system_metrics()
                await asyncio.sleep(self.config.collection_interval_seconds)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.config.collection_interval_seconds)
    
    async def _check_thresholds(self, metrics: SystemMetrics):
        """Check performance thresholds and trigger alerts"""
        try:
            alerts = []
            
            # CPU thresholds
            if metrics.cpu_percent >= self.config.cpu_critical_threshold:
                alerts.append(('cpu_critical', metrics.cpu_percent, self.config.cpu_critical_threshold))
            elif metrics.cpu_percent >= self.config.cpu_warning_threshold:
                alerts.append(('cpu_warning', metrics.cpu_percent, self.config.cpu_warning_threshold))
            
            # Memory thresholds
            if metrics.memory_percent >= self.config.memory_critical_threshold:
                alerts.append(('memory_critical', metrics.memory_percent, self.config.memory_critical_threshold))
            elif metrics.memory_percent >= self.config.memory_warning_threshold:
                alerts.append(('memory_warning', metrics.memory_percent, self.config.memory_warning_threshold))
            
            # Disk thresholds
            if metrics.disk_percent >= self.config.disk_critical_threshold:
                alerts.append(('disk_critical', metrics.disk_percent, self.config.disk_critical_threshold))
            elif metrics.disk_percent >= self.config.disk_warning_threshold:
                alerts.append(('disk_warning', metrics.disk_percent, self.config.disk_warning_threshold))
            
            # Trigger alert callbacks
            for alert_type, value, threshold in alerts:
                for callback in self.alert_callbacks:
                    try:
                        await callback(alert_type, value, threshold)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {e}")
        
        except Exception as e:
            logger.error(f"Failed to check thresholds: {e}")


def create_performance_monitor(config: Optional[PerformanceMonitorConfig] = None) -> PerformanceMonitor:
    """Factory function to create performance monitor"""
    if config is None:
        config = PerformanceMonitorConfig()
    return PerformanceMonitor(config)


__all__ = [
    'PerformanceMonitor', 
    'PerformanceMonitorConfig', 
    'PerformanceMetric', 
    'SystemMetrics',
    'create_performance_monitor'
]