#!/usr/bin/env python3
"""
Resource Manager for monitoring and managing system resources.

Provides comprehensive resource management including CPU, memory, GPU,
and disk usage monitoring with automatic scaling and throttling capabilities.
"""

import os
import psutil
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import platform

from pydantic import BaseModel, Field

from ..core.config import get_config
from ..core.logging import get_logger
from ..core.exceptions import ProcessingError, ResourceError


logger = get_logger(__name__)


class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"


class ResourceStatus(Enum):
    """Resource availability status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EXHAUSTED = "exhausted"


@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    timestamp: datetime
    resource_type: ResourceType
    usage_percent: float
    available: float
    total: float
    status: ResourceStatus
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceLimit:
    """Resource usage limits"""
    soft_limit: float  # Warning threshold
    hard_limit: float  # Critical threshold
    max_limit: float   # Absolute maximum
    reserve_percent: float = 10.0  # Reserve percentage


class ResourceManagerConfig(BaseModel):
    """Resource manager configuration"""
    # Monitoring settings
    monitor_interval: float = Field(5.0, description="Resource monitoring interval in seconds")
    history_size: int = Field(1000, description="Number of historical metrics to keep")
    
    # Resource limits (percentages)
    cpu_soft_limit: float = Field(70.0, description="CPU soft limit %")
    cpu_hard_limit: float = Field(85.0, description="CPU hard limit %")
    memory_soft_limit: float = Field(70.0, description="Memory soft limit %")
    memory_hard_limit: float = Field(85.0, description="Memory hard limit %")
    disk_soft_limit: float = Field(80.0, description="Disk soft limit %")
    disk_hard_limit: float = Field(90.0, description="Disk hard limit %")
    
    # GPU settings
    enable_gpu_monitoring: bool = Field(True, description="Enable GPU monitoring")
    gpu_soft_limit: float = Field(75.0, description="GPU soft limit %")
    gpu_hard_limit: float = Field(90.0, description="GPU hard limit %")
    
    # Throttling settings
    enable_throttling: bool = Field(True, description="Enable automatic throttling")
    throttle_factor: float = Field(0.5, description="Throttling reduction factor")
    recovery_interval: float = Field(60.0, description="Recovery check interval")
    
    # Alerting
    enable_alerts: bool = Field(True, description="Enable resource alerts")
    alert_cooldown: float = Field(300.0, description="Alert cooldown period in seconds")


class ResourceManager:
    """
    Comprehensive resource manager for monitoring and controlling system resources.
    
    Features:
    - Real-time resource monitoring
    - Automatic throttling
    - Resource allocation
    - Historical metrics
    - Alert generation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize resource manager"""
        self.config = ResourceManagerConfig(**(config or {}))
        self.logger = get_logger(__name__)
        
        # Resource limits
        self.limits = {
            ResourceType.CPU: ResourceLimit(
                soft_limit=self.config.cpu_soft_limit,
                hard_limit=self.config.cpu_hard_limit,
                max_limit=100.0
            ),
            ResourceType.MEMORY: ResourceLimit(
                soft_limit=self.config.memory_soft_limit,
                hard_limit=self.config.memory_hard_limit,
                max_limit=100.0
            ),
            ResourceType.DISK: ResourceLimit(
                soft_limit=self.config.disk_soft_limit,
                hard_limit=self.config.disk_hard_limit,
                max_limit=100.0
            ),
            ResourceType.GPU: ResourceLimit(
                soft_limit=self.config.gpu_soft_limit,
                hard_limit=self.config.gpu_hard_limit,
                max_limit=100.0
            )
        }
        
        # Metrics history
        self.metrics_history: Dict[ResourceType, deque] = {
            resource_type: deque(maxlen=self.config.history_size)
            for resource_type in ResourceType
        }
        
        # Current resource status
        self.current_status: Dict[ResourceType, ResourceStatus] = {
            resource_type: ResourceStatus.HEALTHY
            for resource_type in ResourceType
        }
        
        # Throttling state
        self.throttled_resources: Dict[ResourceType, float] = {}
        self.throttle_callbacks: List[Callable] = []
        
        # Alert tracking
        self.last_alert_time: Dict[ResourceType, datetime] = {}
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring thread
        self.monitor_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        self.monitor_lock = threading.Lock()
        
        # GPU availability
        self.has_gpu = self._check_gpu_availability()
        
        self.logger.info("Resource manager initialized")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except ImportError:
            self.logger.warning("GPUtil not available, GPU monitoring disabled")
            return False
        except Exception as e:
            self.logger.warning(f"GPU detection failed: {e}")
            return False
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="ResourceMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10.0)
        
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Update history and status
                with self.monitor_lock:
                    for metric in metrics:
                        self.metrics_history[metric.resource_type].append(metric)
                        self.current_status[metric.resource_type] = metric.status
                
                # Check for alerts
                if self.config.enable_alerts:
                    self._check_alerts(metrics)
                
                # Apply throttling if needed
                if self.config.enable_throttling:
                    self._apply_throttling(metrics)
                
                # Wait for next interval
                time.sleep(self.config.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(self.config.monitor_interval)
    
    def _collect_metrics(self) -> List[ResourceMetrics]:
        """Collect current resource metrics"""
        metrics = []
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        metrics.append(ResourceMetrics(
            timestamp=timestamp,
            resource_type=ResourceType.CPU,
            usage_percent=cpu_percent,
            available=100.0 - cpu_percent,
            total=100.0,
            status=self._get_resource_status(ResourceType.CPU, cpu_percent),
            details={
                "cpu_count": cpu_count,
                "per_cpu": psutil.cpu_percent(interval=0, percpu=True),
                "load_average": os.getloadavg() if platform.system() != "Windows" else None
            }
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        metrics.append(ResourceMetrics(
            timestamp=timestamp,
            resource_type=ResourceType.MEMORY,
            usage_percent=memory_percent,
            available=memory.available / (1024 ** 3),  # GB
            total=memory.total / (1024 ** 3),  # GB
            status=self._get_resource_status(ResourceType.MEMORY, memory_percent),
            details={
                "used_gb": memory.used / (1024 ** 3),
                "free_gb": memory.free / (1024 ** 3),
                "cached_gb": getattr(memory, 'cached', 0) / (1024 ** 3)
            }
        ))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        metrics.append(ResourceMetrics(
            timestamp=timestamp,
            resource_type=ResourceType.DISK,
            usage_percent=disk_percent,
            available=disk.free / (1024 ** 3),  # GB
            total=disk.total / (1024 ** 3),  # GB
            status=self._get_resource_status(ResourceType.DISK, disk_percent),
            details={
                "used_gb": disk.used / (1024 ** 3),
                "free_gb": disk.free / (1024 ** 3)
            }
        ))
        
        # GPU metrics (if available)
        if self.has_gpu and self.config.enable_gpu_monitoring:
            gpu_metrics = self._collect_gpu_metrics(timestamp)
            if gpu_metrics:
                metrics.extend(gpu_metrics)
        
        return metrics
    
    def _collect_gpu_metrics(self, timestamp: datetime) -> List[ResourceMetrics]:
        """Collect GPU metrics"""
        metrics = []
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            total_memory_used = 0
            total_memory = 0
            total_load = 0
            
            gpu_details = []
            
            for i, gpu in enumerate(gpus):
                memory_used = gpu.memoryUsed / 1024  # GB
                memory_total = gpu.memoryTotal / 1024  # GB
                
                total_memory_used += memory_used
                total_memory += memory_total
                total_load += gpu.load * 100
                
                gpu_details.append({
                    "gpu_id": i,
                    "name": gpu.name,
                    "load_percent": gpu.load * 100,
                    "memory_used_gb": memory_used,
                    "memory_total_gb": memory_total,
                    "temperature": gpu.temperature
                })
            
            if gpus:
                avg_load = total_load / len(gpus)
                memory_percent = (total_memory_used / total_memory * 100) if total_memory > 0 else 0
                
                metrics.append(ResourceMetrics(
                    timestamp=timestamp,
                    resource_type=ResourceType.GPU,
                    usage_percent=avg_load,
                    available=total_memory - total_memory_used,
                    total=total_memory,
                    status=self._get_resource_status(ResourceType.GPU, avg_load),
                    details={
                        "gpu_count": len(gpus),
                        "memory_percent": memory_percent,
                        "gpus": gpu_details
                    }
                ))
        
        except Exception as e:
            self.logger.debug(f"GPU metrics collection failed: {e}")
        
        return metrics
    
    def _get_resource_status(self, resource_type: ResourceType, usage_percent: float) -> ResourceStatus:
        """Determine resource status based on usage"""
        limits = self.limits[resource_type]
        
        if usage_percent >= limits.max_limit:
            return ResourceStatus.EXHAUSTED
        elif usage_percent >= limits.hard_limit:
            return ResourceStatus.CRITICAL
        elif usage_percent >= limits.soft_limit:
            return ResourceStatus.WARNING
        else:
            return ResourceStatus.HEALTHY
    
    def _check_alerts(self, metrics: List[ResourceMetrics]):
        """Check and trigger alerts for resource issues"""
        for metric in metrics:
            if metric.status in [ResourceStatus.CRITICAL, ResourceStatus.EXHAUSTED]:
                # Check cooldown
                last_alert = self.last_alert_time.get(metric.resource_type)
                if last_alert and (datetime.now() - last_alert).total_seconds() < self.config.alert_cooldown:
                    continue
                
                # Trigger alert
                self._trigger_alert(metric)
                self.last_alert_time[metric.resource_type] = datetime.now()
    
    def _trigger_alert(self, metric: ResourceMetrics):
        """Trigger resource alert"""
        alert_message = f"Resource {metric.resource_type.value} is {metric.status.value}: {metric.usage_percent:.1f}% used"
        self.logger.warning(alert_message)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(metric)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def _apply_throttling(self, metrics: List[ResourceMetrics]):
        """Apply resource throttling if needed"""
        for metric in metrics:
            if metric.status in [ResourceStatus.CRITICAL, ResourceStatus.EXHAUSTED]:
                # Apply throttling
                if metric.resource_type not in self.throttled_resources:
                    throttle_level = self.config.throttle_factor
                    self.throttled_resources[metric.resource_type] = throttle_level
                    
                    self.logger.info(f"Throttling {metric.resource_type.value} to {throttle_level * 100}%")
                    
                    # Notify callbacks
                    for callback in self.throttle_callbacks:
                        try:
                            callback(metric.resource_type, throttle_level)
                        except Exception as e:
                            self.logger.error(f"Throttle callback failed: {e}")
            
            elif metric.status == ResourceStatus.HEALTHY:
                # Remove throttling
                if metric.resource_type in self.throttled_resources:
                    del self.throttled_resources[metric.resource_type]
                    
                    self.logger.info(f"Removing throttling for {metric.resource_type.value}")
                    
                    # Notify callbacks
                    for callback in self.throttle_callbacks:
                        try:
                            callback(metric.resource_type, 1.0)
                        except Exception as e:
                            self.logger.error(f"Throttle callback failed: {e}")
    
    def get_current_metrics(self) -> Dict[ResourceType, ResourceMetrics]:
        """Get current resource metrics"""
        with self.monitor_lock:
            current_metrics = {}
            for resource_type, history in self.metrics_history.items():
                if history:
                    current_metrics[resource_type] = history[-1]
            return current_metrics
    
    def get_resource_status(self, resource_type: ResourceType) -> ResourceStatus:
        """Get current status for a specific resource"""
        with self.monitor_lock:
            return self.current_status.get(resource_type, ResourceStatus.HEALTHY)
    
    def get_resource_history(
        self,
        resource_type: ResourceType,
        duration_minutes: int = 60
    ) -> List[ResourceMetrics]:
        """Get resource history for the specified duration"""
        with self.monitor_lock:
            history = list(self.metrics_history[resource_type])
            
            if not history:
                return []
            
            # Filter by time
            cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
            return [m for m in history if m.timestamp >= cutoff_time]
    
    def is_resource_available(
        self,
        resource_type: ResourceType,
        required_percent: float = 10.0
    ) -> bool:
        """Check if resource has required availability"""
        metrics = self.get_current_metrics()
        if resource_type not in metrics:
            return True  # Assume available if no metrics
        
        metric = metrics[resource_type]
        available_percent = 100.0 - metric.usage_percent
        
        return available_percent >= required_percent
    
    def wait_for_resources(
        self,
        required_resources: Dict[ResourceType, float],
        timeout: float = 300.0
    ) -> bool:
        """
        Wait for required resources to become available.
        
        Args:
            required_resources: Dict of resource type to required percentage
            timeout: Maximum wait time in seconds
            
        Returns:
            True if resources became available, False if timeout
        """
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            all_available = True
            
            for resource_type, required_percent in required_resources.items():
                if not self.is_resource_available(resource_type, required_percent):
                    all_available = False
                    break
            
            if all_available:
                return True
            
            time.sleep(self.config.monitor_interval)
        
        return False
    
    def allocate_resources(
        self,
        task_id: str,
        required_resources: Dict[ResourceType, float]
    ) -> bool:
        """
        Allocate resources for a task.
        
        Args:
            task_id: Unique task identifier
            required_resources: Dict of resource type to required amount
            
        Returns:
            True if allocation successful
        """
        # Check availability
        for resource_type, required_amount in required_resources.items():
            if not self.is_resource_available(resource_type, required_amount):
                self.logger.warning(f"Insufficient {resource_type.value} for task {task_id}")
                return False
        
        # Record allocation (in real implementation, would track actual usage)
        self.logger.info(f"Resources allocated for task {task_id}: {required_resources}")
        return True
    
    def release_resources(self, task_id: str):
        """Release resources allocated to a task"""
        # In real implementation, would track and release actual resources
        self.logger.info(f"Resources released for task {task_id}")
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource summary"""
        metrics = self.get_current_metrics()
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "status": {
                resource_type.value: status.value
                for resource_type, status in self.current_status.items()
            },
            "usage": {},
            "throttled": [rt.value for rt in self.throttled_resources.keys()]
        }
        
        for resource_type, metric in metrics.items():
            summary["usage"][resource_type.value] = {
                "percent": metric.usage_percent,
                "available": metric.available,
                "total": metric.total,
                "details": metric.details
            }
        
        return summary
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for resource alerts"""
        self.alert_callbacks.append(callback)
    
    def register_throttle_callback(self, callback: Callable):
        """Register callback for throttling events"""
        self.throttle_callbacks.append(callback)
    
    def set_resource_limit(
        self,
        resource_type: ResourceType,
        soft_limit: float,
        hard_limit: float
    ):
        """Update resource limits"""
        if resource_type in self.limits:
            self.limits[resource_type].soft_limit = soft_limit
            self.limits[resource_type].hard_limit = hard_limit
            
            self.logger.info(f"Updated {resource_type.value} limits: soft={soft_limit}%, hard={hard_limit}%")
    
    def cleanup(self):
        """Clean up resource manager"""
        self.stop_monitoring()
        self.alert_callbacks.clear()
        self.throttle_callbacks.clear()
        
        self.logger.info("Resource manager cleaned up")


# Factory function
def create_resource_manager(config: Optional[Dict[str, Any]] = None) -> ResourceManager:
    """Create and return a resource manager instance"""
    return ResourceManager(config)