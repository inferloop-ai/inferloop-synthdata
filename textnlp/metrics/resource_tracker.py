"""
Resource Utilization Tracking for TextNLP
Comprehensive monitoring of system resources, costs, and performance
"""

import time
import psutil
import threading
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from collections import deque, defaultdict
import statistics
import json
import platform
import subprocess
import shutil
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources to track"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"
    PROCESS = "process"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ResourceSample:
    """Single resource measurement sample"""
    timestamp: datetime
    resource_type: ResourceType
    metric_name: str
    value: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceAlert:
    """Resource utilization alert"""
    timestamp: datetime
    resource_type: ResourceType
    metric_name: str
    current_value: float
    threshold_value: float
    severity: AlertSeverity
    message: str
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class ResourceUtilization:
    """Resource utilization snapshot"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_usage: float
    disk_free: float
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    network_sent: float = 0.0
    network_recv: float = 0.0
    process_count: int = 0
    load_average: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "memory_available": self.memory_available,
            "disk_usage": self.disk_usage,
            "disk_free": self.disk_free,
            "gpu_usage": self.gpu_usage,
            "gpu_memory": self.gpu_memory,
            "network_sent": self.network_sent,
            "network_recv": self.network_recv,
            "process_count": self.process_count,
            "load_average": self.load_average
        }


class ResourceTracker:
    """Comprehensive resource utilization tracker"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled_resources = set(self.config.get("enabled_resources", [
            ResourceType.CPU, ResourceType.MEMORY, ResourceType.GPU, ResourceType.DISK
        ]))
        
        # Monitoring configuration
        self.sample_interval = self.config.get("sample_interval", 10)  # seconds
        self.history_duration = self.config.get("history_duration", 3600)  # seconds
        self.max_samples = self.history_duration // self.sample_interval
        
        # Storage for samples
        self.samples: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_samples))
        self.current_utilization: Optional[ResourceUtilization] = None
        
        # Alerting
        self.alert_thresholds = self.config.get("alert_thresholds", {
            "cpu_warning": 80.0,
            "cpu_critical": 95.0,
            "memory_warning": 80.0,
            "memory_critical": 95.0,
            "gpu_warning": 85.0,
            "gpu_critical": 95.0,
            "disk_warning": 80.0,
            "disk_critical": 95.0
        })
        self.alerts: List[ResourceAlert] = []
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring control
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Cost tracking
        self.cost_config = self.config.get("cost_config", {})
        self.cost_samples: deque = deque(maxlen=self.max_samples)
        
        # Initialize GPU monitoring
        self.gpu_available = self._check_gpu_availability()
        
        # Network tracking
        self.last_network_stats = None
        
        logger.info("Resource tracker initialized")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available"""
        if not HAS_TORCH and not HAS_GPUTIL:
            logger.info("No GPU monitoring libraries available")
            return False
        
        if HAS_TORCH and torch.cuda.is_available():
            logger.info(f"CUDA available with {torch.cuda.device_count()} GPUs")
            return True
        
        if HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    logger.info(f"GPUtil detected {len(gpus)} GPUs")
                    return True
            except Exception as e:
                logger.debug(f"GPUtil check failed: {e}")
        
        logger.info("No GPUs detected")
        return False
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self.stop_event.wait(self.sample_interval):
            try:
                utilization = self._collect_resource_data()
                self.current_utilization = utilization
                
                # Store samples
                timestamp = utilization.timestamp
                self._store_sample(timestamp, ResourceType.CPU, "usage", utilization.cpu_usage, "%")
                self._store_sample(timestamp, ResourceType.MEMORY, "usage", utilization.memory_usage, "%")
                self._store_sample(timestamp, ResourceType.MEMORY, "available", utilization.memory_available, "GB")
                self._store_sample(timestamp, ResourceType.DISK, "usage", utilization.disk_usage, "%")
                self._store_sample(timestamp, ResourceType.DISK, "free", utilization.disk_free, "GB")
                
                if utilization.gpu_usage is not None:
                    self._store_sample(timestamp, ResourceType.GPU, "usage", utilization.gpu_usage, "%")
                
                if utilization.gpu_memory is not None:
                    self._store_sample(timestamp, ResourceType.GPU, "memory", utilization.gpu_memory, "GB")
                
                self._store_sample(timestamp, ResourceType.NETWORK, "sent", utilization.network_sent, "MB/s")
                self._store_sample(timestamp, ResourceType.NETWORK, "recv", utilization.network_recv, "MB/s")
                
                # Check for alerts
                self._check_alerts(utilization)
                
                # Calculate costs
                self._calculate_costs(utilization)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_resource_data(self) -> ResourceUtilization:
        """Collect current resource utilization data"""
        timestamp = datetime.now(timezone.utc)
        
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_available = memory.available / (1024**3)  # GB
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        disk_free = disk.free / (1024**3)  # GB
        
        # GPU usage
        gpu_usage = None
        gpu_memory = None
        if self.gpu_available:
            gpu_usage, gpu_memory = self._get_gpu_stats()
        
        # Network statistics
        network_sent, network_recv = self._get_network_stats()
        
        # Process count
        process_count = len(psutil.pids())
        
        # Load average (Unix-like systems)
        load_average = None
        try:
            if platform.system() != "Windows":
                load_average = psutil.getloadavg()[0]  # 1-minute load average
        except AttributeError:
            pass
        
        return ResourceUtilization(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_available=memory_available,
            disk_usage=disk_usage,
            disk_free=disk_free,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            network_sent=network_sent,
            network_recv=network_recv,
            process_count=process_count,
            load_average=load_average
        )
    
    def _get_gpu_stats(self) -> Tuple[Optional[float], Optional[float]]:
        """Get GPU utilization statistics"""
        try:
            if HAS_TORCH and torch.cuda.is_available():
                # PyTorch CUDA stats
                gpu_usage = 0.0
                gpu_memory = 0.0
                
                for i in range(torch.cuda.device_count()):
                    # Memory usage
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                    total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                    gpu_memory += allocated
                    
                    # Utilization (approximated by memory usage)
                    gpu_usage += (allocated / total_memory) * 100
                
                gpu_usage /= torch.cuda.device_count()
                return gpu_usage, gpu_memory
            
            elif HAS_GPUTIL:
                # GPUtil stats
                gpus = GPUtil.getGPUs()
                if gpus:
                    total_load = sum(gpu.load * 100 for gpu in gpus)
                    total_memory = sum(gpu.memoryUsed / 1024 for gpu in gpus)  # GB
                    avg_load = total_load / len(gpus)
                    return avg_load, total_memory
            
        except Exception as e:
            logger.debug(f"GPU stats collection failed: {e}")
        
        return None, None
    
    def _get_network_stats(self) -> Tuple[float, float]:
        """Get network I/O statistics"""
        try:
            current_stats = psutil.net_io_counters()
            
            if self.last_network_stats is None:
                self.last_network_stats = current_stats
                return 0.0, 0.0
            
            # Calculate rates (MB/s)
            time_delta = self.sample_interval
            sent_rate = (current_stats.bytes_sent - self.last_network_stats.bytes_sent) / time_delta / (1024**2)
            recv_rate = (current_stats.bytes_recv - self.last_network_stats.bytes_recv) / time_delta / (1024**2)
            
            self.last_network_stats = current_stats
            return max(0, sent_rate), max(0, recv_rate)
            
        except Exception as e:
            logger.debug(f"Network stats collection failed: {e}")
            return 0.0, 0.0
    
    def _store_sample(self, timestamp: datetime, resource_type: ResourceType, 
                     metric_name: str, value: float, unit: str):
        """Store a resource sample"""
        sample = ResourceSample(
            timestamp=timestamp,
            resource_type=resource_type,
            metric_name=metric_name,
            value=value,
            unit=unit
        )
        
        key = f"{resource_type.value}.{metric_name}"
        self.samples[key].append(sample)
    
    def _check_alerts(self, utilization: ResourceUtilization):
        """Check for resource utilization alerts"""
        alerts_to_check = [
            ("cpu", utilization.cpu_usage, ResourceType.CPU, "usage"),
            ("memory", utilization.memory_usage, ResourceType.MEMORY, "usage"),
            ("disk", utilization.disk_usage, ResourceType.DISK, "usage")
        ]
        
        if utilization.gpu_usage is not None:
            alerts_to_check.append(("gpu", utilization.gpu_usage, ResourceType.GPU, "usage"))
        
        for resource_name, current_value, resource_type, metric_name in alerts_to_check:
            warning_threshold = self.alert_thresholds.get(f"{resource_name}_warning")
            critical_threshold = self.alert_thresholds.get(f"{resource_name}_critical")
            
            if critical_threshold and current_value >= critical_threshold:
                self._create_alert(
                    resource_type, metric_name, current_value, critical_threshold,
                    AlertSeverity.CRITICAL,
                    f"{resource_name.upper()} usage critical: {current_value:.1f}%"
                )
            elif warning_threshold and current_value >= warning_threshold:
                self._create_alert(
                    resource_type, metric_name, current_value, warning_threshold,
                    AlertSeverity.WARNING,
                    f"{resource_name.upper()} usage high: {current_value:.1f}%"
                )
    
    def _create_alert(self, resource_type: ResourceType, metric_name: str,
                     current_value: float, threshold_value: float,
                     severity: AlertSeverity, message: str):
        """Create a new alert if not already active"""
        # Check if similar alert is already active
        for alert in self.alerts:
            if (alert.resource_type == resource_type and 
                alert.metric_name == metric_name and 
                alert.severity == severity and
                not alert.resolved):
                return  # Alert already exists
        
        alert = ResourceAlert(
            timestamp=datetime.now(timezone.utc),
            resource_type=resource_type,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            severity=severity,
            message=message
        )
        
        self.alerts.append(alert)
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        logger.warning(f"Resource alert: {message}")
    
    def _calculate_costs(self, utilization: ResourceUtilization):
        """Calculate resource costs based on usage"""
        if not self.cost_config:
            return
        
        timestamp = utilization.timestamp
        total_cost = 0.0
        cost_breakdown = {}
        
        # CPU cost
        if "cpu_cost_per_hour" in self.cost_config:
            cpu_cost = (utilization.cpu_usage / 100) * self.cost_config["cpu_cost_per_hour"] / 3600 * self.sample_interval
            total_cost += cpu_cost
            cost_breakdown["cpu"] = cpu_cost
        
        # Memory cost
        if "memory_cost_per_gb_hour" in self.cost_config:
            memory_used_gb = (100 - utilization.memory_usage) / 100 * utilization.memory_available
            memory_cost = memory_used_gb * self.cost_config["memory_cost_per_gb_hour"] / 3600 * self.sample_interval
            total_cost += memory_cost
            cost_breakdown["memory"] = memory_cost
        
        # GPU cost
        if utilization.gpu_usage is not None and "gpu_cost_per_hour" in self.cost_config:
            gpu_cost = (utilization.gpu_usage / 100) * self.cost_config["gpu_cost_per_hour"] / 3600 * self.sample_interval
            total_cost += gpu_cost
            cost_breakdown["gpu"] = gpu_cost
        
        # Storage cost
        if "storage_cost_per_gb_hour" in self.cost_config:
            storage_used_gb = utilization.disk_free  # Simplified
            storage_cost = storage_used_gb * self.cost_config["storage_cost_per_gb_hour"] / 3600 * self.sample_interval
            total_cost += storage_cost
            cost_breakdown["storage"] = storage_cost
        
        cost_sample = {
            "timestamp": timestamp,
            "total_cost": total_cost,
            "breakdown": cost_breakdown
        }
        
        self.cost_samples.append(cost_sample)
    
    def get_current_utilization(self) -> Optional[ResourceUtilization]:
        """Get current resource utilization"""
        return self.current_utilization
    
    def get_resource_history(self, resource_type: ResourceType, metric_name: str,
                           duration_minutes: int = 60) -> List[ResourceSample]:
        """Get resource history for specified duration"""
        key = f"{resource_type.value}.{metric_name}"
        samples = list(self.samples[key])
        
        if not samples:
            return []
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=duration_minutes)
        return [s for s in samples if s.timestamp >= cutoff_time]
    
    def get_resource_statistics(self, resource_type: ResourceType, metric_name: str,
                              duration_minutes: int = 60) -> Dict[str, float]:
        """Get statistical summary of resource usage"""
        history = self.get_resource_history(resource_type, metric_name, duration_minutes)
        
        if not history:
            return {}
        
        values = [sample.value for sample in history]
        
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
            "current": values[-1] if values else 0.0,
            "samples": len(values)
        }
    
    def get_alerts(self, severity: Optional[AlertSeverity] = None,
                  resolved: Optional[bool] = None) -> List[ResourceAlert]:
        """Get alerts with optional filtering"""
        alerts = self.alerts
        
        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def resolve_alert(self, alert: ResourceAlert):
        """Mark an alert as resolved"""
        alert.resolved = True
        alert.resolution_time = datetime.now(timezone.utc)
    
    def add_alert_callback(self, callback: Callable[[ResourceAlert], None]):
        """Add callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_cost_summary(self, duration_hours: int = 24) -> Dict[str, Any]:
        """Get cost summary for specified duration"""
        if not self.cost_samples:
            return {"message": "No cost data available"}
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=duration_hours)
        recent_costs = [
            sample for sample in self.cost_samples
            if sample["timestamp"] >= cutoff_time
        ]
        
        if not recent_costs:
            return {"message": "No cost data in time range"}
        
        total_cost = sum(sample["total_cost"] for sample in recent_costs)
        
        # Breakdown by resource type
        cost_breakdown = defaultdict(float)
        for sample in recent_costs:
            for resource, cost in sample["breakdown"].items():
                cost_breakdown[resource] += cost
        
        # Projected costs
        hourly_rate = total_cost / duration_hours if duration_hours > 0 else 0
        
        return {
            "period_hours": duration_hours,
            "total_cost": total_cost,
            "hourly_rate": hourly_rate,
            "daily_projection": hourly_rate * 24,
            "monthly_projection": hourly_rate * 24 * 30,
            "cost_breakdown": dict(cost_breakdown),
            "samples": len(recent_costs)
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        current = self.current_utilization
        if not current:
            return {"message": "No current utilization data"}
        
        report = {
            "timestamp": current.timestamp.isoformat(),
            "current_utilization": current.to_dict(),
            "statistics": {},
            "alerts": {
                "active": len(self.get_alerts(resolved=False)),
                "critical": len(self.get_alerts(AlertSeverity.CRITICAL, False)),
                "warning": len(self.get_alerts(AlertSeverity.WARNING, False))
            }
        }
        
        # Add statistics for key metrics
        key_metrics = [
            (ResourceType.CPU, "usage"),
            (ResourceType.MEMORY, "usage"),
            (ResourceType.DISK, "usage")
        ]
        
        if self.gpu_available:
            key_metrics.append((ResourceType.GPU, "usage"))
        
        for resource_type, metric_name in key_metrics:
            stats = self.get_resource_statistics(resource_type, metric_name, 60)
            if stats:
                report["statistics"][f"{resource_type.value}_{metric_name}"] = stats
        
        # Add cost information
        cost_summary = self.get_cost_summary(24)
        if "message" not in cost_summary:
            report["cost_summary"] = cost_summary
        
        return report
    
    def export_data(self, format: str = "json", duration_hours: int = 24) -> str:
        """Export resource data in specified format"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=duration_hours)
        
        # Collect all samples
        all_samples = []
        for key, samples in self.samples.items():
            for sample in samples:
                if sample.timestamp >= cutoff_time:
                    all_samples.append({
                        "timestamp": sample.timestamp.isoformat(),
                        "resource_type": sample.resource_type.value,
                        "metric_name": sample.metric_name,
                        "value": sample.value,
                        "unit": sample.unit,
                        "metadata": sample.metadata
                    })
        
        # Collect alerts
        alerts_data = []
        for alert in self.get_alerts():
            if alert.timestamp >= cutoff_time:
                alerts_data.append({
                    "timestamp": alert.timestamp.isoformat(),
                    "resource_type": alert.resource_type.value,
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "resolved": alert.resolved,
                    "resolution_time": alert.resolution_time.isoformat() if alert.resolution_time else None
                })
        
        data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_hours": duration_hours,
            "samples": all_samples,
            "alerts": alerts_data,
            "cost_data": list(self.cost_samples),
            "performance_report": self.get_performance_report()
        }
        
        if format.lower() == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def cleanup_old_data(self, max_age_hours: int = 72):
        """Clean up old samples and alerts"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        
        # Clean up samples
        for key in list(self.samples.keys()):
            samples = self.samples[key]
            while samples and samples[0].timestamp < cutoff_time:
                samples.popleft()
            
            if not samples:
                del self.samples[key]
        
        # Clean up alerts
        self.alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
        
        # Clean up cost samples
        while self.cost_samples and self.cost_samples[0]["timestamp"] < cutoff_time:
            self.cost_samples.popleft()
        
        logger.info(f"Cleaned up data older than {max_age_hours} hours")


# Example usage
if __name__ == "__main__":
    async def example():
        # Configure resource tracker
        config = {
            "enabled_resources": [
                ResourceType.CPU, ResourceType.MEMORY, ResourceType.GPU,
                ResourceType.DISK, ResourceType.NETWORK
            ],
            "sample_interval": 5,
            "alert_thresholds": {
                "cpu_warning": 70.0,
                "cpu_critical": 90.0,
                "memory_warning": 75.0,
                "memory_critical": 90.0
            },
            "cost_config": {
                "cpu_cost_per_hour": 0.096,
                "memory_cost_per_gb_hour": 0.012,
                "gpu_cost_per_hour": 2.40
            }
        }
        
        tracker = ResourceTracker(config)
        
        # Alert callback
        def alert_handler(alert: ResourceAlert):
            print(f"ALERT: {alert.message}")
        
        tracker.add_alert_callback(alert_handler)
        
        # Start monitoring
        tracker.start_monitoring()
        
        # Let it run for a bit
        await asyncio.sleep(30)
        
        # Get current utilization
        current = tracker.get_current_utilization()
        if current:
            print("Current Utilization:")
            print(f"CPU: {current.cpu_usage:.1f}%")
            print(f"Memory: {current.memory_usage:.1f}%")
            if current.gpu_usage is not None:
                print(f"GPU: {current.gpu_usage:.1f}%")
        
        # Get performance report
        report = tracker.get_performance_report()
        print("\nPerformance Report:")
        print(f"Active alerts: {report['alerts']['active']}")
        
        # Get statistics
        cpu_stats = tracker.get_resource_statistics(ResourceType.CPU, "usage", 30)
        if cpu_stats:
            print(f"\nCPU Statistics (30 min):")
            print(f"Mean: {cpu_stats['mean']:.1f}%")
            print(f"Max: {cpu_stats['max']:.1f}%")
        
        # Get cost summary
        cost_summary = tracker.get_cost_summary(1)
        if "message" not in cost_summary:
            print(f"\nCost Summary:")
            print(f"Total cost (1h): ${cost_summary['total_cost']:.4f}")
            print(f"Hourly rate: ${cost_summary['hourly_rate']:.4f}")
        
        # Stop monitoring
        tracker.stop_monitoring()
        
        print("\nResource tracking completed")
    
    # Run example
    # asyncio.run(example())