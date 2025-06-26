"""
GPU Health Monitoring Implementation for TextNLP
Monitors GPU health metrics across cloud providers
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """GPU health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class GPUMetrics:
    """GPU metrics data structure"""
    timestamp: datetime
    gpu_id: str
    utilization_percent: float
    memory_used_gb: float
    memory_total_gb: float
    temperature_celsius: float
    power_usage_watts: Optional[float] = None
    clock_speed_mhz: Optional[int] = None
    pcie_throughput_mb: Optional[float] = None
    
    @property
    def memory_utilization_percent(self) -> float:
        """Calculate memory utilization percentage"""
        if self.memory_total_gb > 0:
            return (self.memory_used_gb / self.memory_total_gb) * 100
        return 0.0


@dataclass
class HealthCheck:
    """Health check result"""
    status: HealthStatus
    message: str
    metrics: GPUMetrics
    alerts: List[str] = field(default_factory=list)


@dataclass
class HealthThresholds:
    """Configurable health thresholds"""
    utilization_warning: float = 80.0
    utilization_critical: float = 95.0
    memory_warning: float = 85.0
    memory_critical: float = 95.0
    temperature_warning: float = 80.0
    temperature_critical: float = 85.0
    power_warning_percent: float = 90.0  # % of TDP


class BaseHealthMonitor(ABC):
    """Base class for GPU health monitoring"""
    
    def __init__(self, thresholds: Optional[HealthThresholds] = None):
        self.thresholds = thresholds or HealthThresholds()
        self.alert_callbacks: List[Callable] = []
        self.metrics_history: List[GPUMetrics] = []
        self.max_history_size = 1000
    
    @abstractmethod
    async def collect_metrics(self, gpu_id: str) -> GPUMetrics:
        """Collect current GPU metrics"""
        pass
    
    @abstractmethod
    async def setup_monitoring(self) -> None:
        """Setup provider-specific monitoring"""
        pass
    
    def evaluate_health(self, metrics: GPUMetrics) -> HealthCheck:
        """Evaluate GPU health based on metrics"""
        status = HealthStatus.HEALTHY
        alerts = []
        
        # Check utilization
        if metrics.utilization_percent >= self.thresholds.utilization_critical:
            status = HealthStatus.CRITICAL
            alerts.append(f"Critical GPU utilization: {metrics.utilization_percent:.1f}%")
        elif metrics.utilization_percent >= self.thresholds.utilization_warning:
            status = HealthStatus.WARNING if status == HealthStatus.HEALTHY else status
            alerts.append(f"High GPU utilization: {metrics.utilization_percent:.1f}%")
        
        # Check memory
        memory_percent = metrics.memory_utilization_percent
        if memory_percent >= self.thresholds.memory_critical:
            status = HealthStatus.CRITICAL
            alerts.append(f"Critical GPU memory usage: {memory_percent:.1f}%")
        elif memory_percent >= self.thresholds.memory_warning:
            status = HealthStatus.WARNING if status == HealthStatus.HEALTHY else status
            alerts.append(f"High GPU memory usage: {memory_percent:.1f}%")
        
        # Check temperature
        if metrics.temperature_celsius >= self.thresholds.temperature_critical:
            status = HealthStatus.CRITICAL
            alerts.append(f"Critical GPU temperature: {metrics.temperature_celsius}°C")
        elif metrics.temperature_celsius >= self.thresholds.temperature_warning:
            status = HealthStatus.WARNING if status == HealthStatus.HEALTHY else status
            alerts.append(f"High GPU temperature: {metrics.temperature_celsius}°C")
        
        message = "GPU operating normally" if status == HealthStatus.HEALTHY else "; ".join(alerts)
        
        return HealthCheck(
            status=status,
            message=message,
            metrics=metrics,
            alerts=alerts
        )
    
    async def monitor_gpu(self, gpu_id: str) -> HealthCheck:
        """Monitor a single GPU and return health check"""
        try:
            metrics = await self.collect_metrics(gpu_id)
            self._store_metrics(metrics)
            health_check = self.evaluate_health(metrics)
            
            # Trigger alerts if needed
            if health_check.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                await self._trigger_alerts(health_check)
            
            return health_check
        except Exception as e:
            logger.error(f"Error monitoring GPU {gpu_id}: {e}")
            return HealthCheck(
                status=HealthStatus.UNKNOWN,
                message=f"Failed to collect metrics: {str(e)}",
                metrics=GPUMetrics(
                    timestamp=datetime.utcnow(),
                    gpu_id=gpu_id,
                    utilization_percent=0,
                    memory_used_gb=0,
                    memory_total_gb=0,
                    temperature_celsius=0
                )
            )
    
    def _store_metrics(self, metrics: GPUMetrics) -> None:
        """Store metrics in history"""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)
    
    async def _trigger_alerts(self, health_check: HealthCheck) -> None:
        """Trigger registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                await callback(health_check)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def register_alert_callback(self, callback: Callable) -> None:
        """Register a callback for health alerts"""
        self.alert_callbacks.append(callback)
    
    def get_metrics_summary(self, gpu_id: str, minutes: int = 5) -> Dict[str, Any]:
        """Get summary statistics for recent metrics"""
        cutoff_time = datetime.utcnow()
        recent_metrics = [
            m for m in self.metrics_history
            if m.gpu_id == gpu_id and (cutoff_time - m.timestamp).seconds <= minutes * 60
        ]
        
        if not recent_metrics:
            return {"error": "No recent metrics available"}
        
        utilizations = [m.utilization_percent for m in recent_metrics]
        memories = [m.memory_utilization_percent for m in recent_metrics]
        temperatures = [m.temperature_celsius for m in recent_metrics]
        
        return {
            "gpu_id": gpu_id,
            "period_minutes": minutes,
            "samples": len(recent_metrics),
            "utilization": {
                "avg": sum(utilizations) / len(utilizations),
                "max": max(utilizations),
                "min": min(utilizations)
            },
            "memory": {
                "avg": sum(memories) / len(memories),
                "max": max(memories),
                "min": min(memories)
            },
            "temperature": {
                "avg": sum(temperatures) / len(temperatures),
                "max": max(temperatures),
                "min": min(temperatures)
            }
        }


class AWSGPUHealthMonitor(BaseHealthMonitor):
    """AWS-specific GPU health monitoring using CloudWatch"""
    
    def __init__(self, cloudwatch_client, ec2_client, thresholds: Optional[HealthThresholds] = None):
        super().__init__(thresholds)
        self.cloudwatch = cloudwatch_client
        self.ec2 = ec2_client
        self.namespace = "TextNLP/GPU"
    
    async def setup_monitoring(self) -> None:
        """Setup CloudWatch monitoring for GPU instances"""
        # Create custom CloudWatch dashboard
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["AWS/EC2", "GPUUtilization", {"stat": "Average"}],
                            ["AWS/EC2", "GPUMemoryUtilization", {"stat": "Average"}],
                            ["AWS/EC2", "GPUTemperature", {"stat": "Average"}]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": "us-east-1",
                        "title": "GPU Health Metrics"
                    }
                }
            ]
        }
        
        try:
            self.cloudwatch.put_dashboard(
                DashboardName="TextNLP-GPU-Health",
                DashboardBody=json.dumps(dashboard_body)
            )
            logger.info("Created CloudWatch dashboard for GPU monitoring")
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
    
    async def collect_metrics(self, instance_id: str) -> GPUMetrics:
        """Collect GPU metrics from CloudWatch"""
        # Get instance details
        response = self.ec2.describe_instances(InstanceIds=[instance_id])
        instance = response['Reservations'][0]['Instances'][0]
        
        # Collect CloudWatch metrics
        end_time = datetime.utcnow()
        start_time = end_time.replace(minute=end_time.minute - 1)
        
        metric_queries = [
            {
                'Id': 'gpu_util',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/EC2',
                        'MetricName': 'GPUUtilization',
                        'Dimensions': [{'Name': 'InstanceId', 'Value': instance_id}]
                    },
                    'Period': 60,
                    'Stat': 'Average'
                }
            },
            {
                'Id': 'gpu_mem',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/EC2',
                        'MetricName': 'GPUMemoryUtilization',
                        'Dimensions': [{'Name': 'InstanceId', 'Value': instance_id}]
                    },
                    'Period': 60,
                    'Stat': 'Average'
                }
            },
            {
                'Id': 'gpu_temp',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/EC2',
                        'MetricName': 'GPUTemperature',
                        'Dimensions': [{'Name': 'InstanceId', 'Value': instance_id}]
                    },
                    'Period': 60,
                    'Stat': 'Average'
                }
            }
        ]
        
        response = self.cloudwatch.get_metric_data(
            MetricDataQueries=metric_queries,
            StartTime=start_time,
            EndTime=end_time
        )
        
        # Parse results
        results = {r['Id']: r['Values'][0] if r['Values'] else 0 for r in response['MetricDataResults']}
        
        # Get GPU memory info (this would come from instance metadata in real implementation)
        gpu_memory_gb = self._get_gpu_memory_size(instance['InstanceType'])
        
        return GPUMetrics(
            timestamp=datetime.utcnow(),
            gpu_id=instance_id,
            utilization_percent=results.get('gpu_util', 0),
            memory_used_gb=gpu_memory_gb * results.get('gpu_mem', 0) / 100,
            memory_total_gb=gpu_memory_gb,
            temperature_celsius=results.get('gpu_temp', 0)
        )
    
    def _get_gpu_memory_size(self, instance_type: str) -> float:
        """Get GPU memory size based on instance type"""
        gpu_memory_map = {
            "g4dn.xlarge": 16,
            "g4dn.2xlarge": 16,
            "g4dn.4xlarge": 16,
            "g4dn.8xlarge": 16,
            "g4dn.12xlarge": 64,  # 4x16GB
            "p3.2xlarge": 16,
            "p3.8xlarge": 64,  # 4x16GB
            "p3.16xlarge": 128,  # 8x16GB
            "p4d.24xlarge": 320,  # 8x40GB
            "g5.xlarge": 24,
            "g5.2xlarge": 24,
            "g5.4xlarge": 24,
            "g5.8xlarge": 24,
            "g5.12xlarge": 96  # 4x24GB
        }
        return gpu_memory_map.get(instance_type, 16)


class GCPGPUHealthMonitor(BaseHealthMonitor):
    """GCP-specific GPU health monitoring using Stackdriver"""
    
    def __init__(self, monitoring_client, compute_client, project_id: str, thresholds: Optional[HealthThresholds] = None):
        super().__init__(thresholds)
        self.monitoring = monitoring_client
        self.compute = compute_client
        self.project_id = project_id
    
    async def setup_monitoring(self) -> None:
        """Setup Stackdriver monitoring for GPU instances"""
        # Create alert policy for GPU health
        alert_policy = {
            "display_name": "TextNLP GPU Health Alert",
            "conditions": [
                {
                    "display_name": "GPU Utilization High",
                    "condition_threshold": {
                        "filter": 'metric.type="compute.googleapis.com/instance/gpu/utilization"',
                        "comparison": "COMPARISON_GT",
                        "threshold_value": self.thresholds.utilization_critical,
                        "duration": "60s"
                    }
                }
            ],
            "notification_channels": [],
            "alert_strategy": {
                "notification_rate_limit": {
                    "period": "300s"
                }
            }
        }
        
        try:
            self.monitoring.create_alert_policy(
                name=f"projects/{self.project_id}",
                alert_policy=alert_policy
            )
            logger.info("Created Stackdriver alert policy for GPU monitoring")
        except Exception as e:
            logger.error(f"Failed to create alert policy: {e}")
    
    async def collect_metrics(self, instance_name: str) -> GPUMetrics:
        """Collect GPU metrics from Stackdriver"""
        # Build metric filters
        interval = {
            "end_time": {"seconds": int(datetime.utcnow().timestamp())},
            "start_time": {"seconds": int((datetime.utcnow().timestamp()) - 60)}
        }
        
        # GPU utilization
        utilization_filter = (
            f'metric.type="compute.googleapis.com/instance/gpu/utilization" '
            f'AND resource.labels.instance_id="{instance_name}"'
        )
        
        utilization_request = {
            "name": f"projects/{self.project_id}",
            "filter": utilization_filter,
            "interval": interval,
            "view": "FULL"
        }
        
        util_response = self.monitoring.list_time_series(utilization_request)
        utilization = self._extract_metric_value(util_response)
        
        # Similar queries for memory and temperature
        memory_filter = (
            f'metric.type="compute.googleapis.com/instance/gpu/memory_utilization" '
            f'AND resource.labels.instance_id="{instance_name}"'
        )
        
        memory_response = self.monitoring.list_time_series({
            "name": f"projects/{self.project_id}",
            "filter": memory_filter,
            "interval": interval,
            "view": "FULL"
        })
        memory_util = self._extract_metric_value(memory_response)
        
        temperature_filter = (
            f'metric.type="compute.googleapis.com/instance/gpu/temperature" '
            f'AND resource.labels.instance_id="{instance_name}"'
        )
        
        temp_response = self.monitoring.list_time_series({
            "name": f"projects/{self.project_id}",
            "filter": temperature_filter,
            "interval": interval,
            "view": "FULL"
        })
        temperature = self._extract_metric_value(temp_response)
        
        # Get instance details for GPU memory size
        instance = self.compute.instances().get(
            project=self.project_id,
            zone=self._extract_zone(instance_name),
            instance=instance_name
        ).execute()
        
        gpu_memory_gb = self._get_gpu_memory_from_instance(instance)
        
        return GPUMetrics(
            timestamp=datetime.utcnow(),
            gpu_id=instance_name,
            utilization_percent=utilization,
            memory_used_gb=gpu_memory_gb * memory_util / 100,
            memory_total_gb=gpu_memory_gb,
            temperature_celsius=temperature
        )
    
    def _extract_metric_value(self, time_series_response) -> float:
        """Extract metric value from Stackdriver response"""
        if time_series_response and len(time_series_response) > 0:
            points = time_series_response[0].get("points", [])
            if points:
                return points[0]["value"]["double_value"]
        return 0.0
    
    def _extract_zone(self, instance_name: str) -> str:
        """Extract zone from instance name (simplified)"""
        # In real implementation, this would query the instance details
        return "us-central1-a"
    
    def _get_gpu_memory_from_instance(self, instance: Dict) -> float:
        """Get GPU memory size from instance metadata"""
        # Parse guest accelerators
        accelerators = instance.get("guestAccelerators", [])
        if accelerators:
            accelerator_type = accelerators[0]["acceleratorType"]
            if "nvidia-tesla-t4" in accelerator_type:
                return 16
            elif "nvidia-tesla-v100" in accelerator_type:
                return 16
            elif "nvidia-tesla-a100" in accelerator_type:
                return 40
        return 16  # Default


class AzureGPUHealthMonitor(BaseHealthMonitor):
    """Azure-specific GPU health monitoring using Azure Monitor"""
    
    def __init__(self, monitor_client, compute_client, resource_group: str, thresholds: Optional[HealthThresholds] = None):
        super().__init__(thresholds)
        self.monitor = monitor_client
        self.compute = compute_client
        self.resource_group = resource_group
    
    async def setup_monitoring(self) -> None:
        """Setup Azure Monitor for GPU instances"""
        # Create action group for alerts
        action_group = {
            "location": "Global",
            "enabled": True,
            "email_receivers": [
                {
                    "name": "TextNLP GPU Alerts",
                    "email_address": "gpu-alerts@textnlp.ai",
                    "use_common_alert_schema": True
                }
            ]
        }
        
        try:
            self.monitor.action_groups.create_or_update(
                resource_group_name=self.resource_group,
                action_group_name="TextNLP-GPU-Alerts",
                action_group=action_group
            )
            
            # Create metric alert for GPU utilization
            alert_rule = {
                "location": "global",
                "enabled": True,
                "scopes": [f"/subscriptions/{{subscription_id}}/resourceGroups/{self.resource_group}"],
                "evaluation_frequency": "PT1M",
                "window_size": "PT5M",
                "criteria": {
                    "odata.type": "Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria",
                    "all_of": [
                        {
                            "name": "GPU Utilization Critical",
                            "metric_name": "GPU Utilization Percentage",
                            "metric_namespace": "Microsoft.Compute/virtualMachines",
                            "operator": "GreaterThan",
                            "threshold": self.thresholds.utilization_critical,
                            "aggregation": "Average"
                        }
                    ]
                },
                "actions": [
                    {
                        "action_group_id": f"/subscriptions/{{subscription_id}}/resourceGroups/{self.resource_group}/providers/Microsoft.Insights/actionGroups/TextNLP-GPU-Alerts"
                    }
                ]
            }
            
            self.monitor.metric_alerts.create_or_update(
                resource_group_name=self.resource_group,
                rule_name="TextNLP-GPU-Utilization-Alert",
                parameters=alert_rule
            )
            
            logger.info("Created Azure Monitor alerts for GPU monitoring")
        except Exception as e:
            logger.error(f"Failed to create alerts: {e}")
    
    async def collect_metrics(self, vm_name: str) -> GPUMetrics:
        """Collect GPU metrics from Azure Monitor"""
        # Get VM resource ID
        vm = self.compute.virtual_machines.get(
            resource_group_name=self.resource_group,
            vm_name=vm_name
        )
        resource_id = vm.id
        
        # Query metrics
        end_time = datetime.utcnow()
        start_time = end_time.replace(minute=end_time.minute - 1)
        
        # GPU Utilization
        utilization_result = self.monitor.metrics.list(
            resource_uri=resource_id,
            metricnames="GPU Utilization Percentage",
            timespan=f"{start_time.isoformat()}/{end_time.isoformat()}",
            aggregation="Average"
        )
        
        utilization = self._extract_azure_metric(utilization_result)
        
        # GPU Memory
        memory_result = self.monitor.metrics.list(
            resource_uri=resource_id,
            metricnames="GPU Memory Utilization Percentage",
            timespan=f"{start_time.isoformat()}/{end_time.isoformat()}",
            aggregation="Average"
        )
        
        memory_util = self._extract_azure_metric(memory_result)
        
        # GPU Temperature
        temp_result = self.monitor.metrics.list(
            resource_uri=resource_id,
            metricnames="GPU Temperature",
            timespan=f"{start_time.isoformat()}/{end_time.isoformat()}",
            aggregation="Average"
        )
        
        temperature = self._extract_azure_metric(temp_result)
        
        # Get GPU memory size from VM size
        gpu_memory_gb = self._get_gpu_memory_from_vm_size(vm.hardware_profile.vm_size)
        
        return GPUMetrics(
            timestamp=datetime.utcnow(),
            gpu_id=vm_name,
            utilization_percent=utilization,
            memory_used_gb=gpu_memory_gb * memory_util / 100,
            memory_total_gb=gpu_memory_gb,
            temperature_celsius=temperature
        )
    
    def _extract_azure_metric(self, metric_result) -> float:
        """Extract metric value from Azure Monitor response"""
        if metric_result.value and len(metric_result.value) > 0:
            timeseries = metric_result.value[0].timeseries
            if timeseries and len(timeseries) > 0:
                data = timeseries[0].data
                if data and len(data) > 0:
                    return data[-1].average or 0.0
        return 0.0
    
    def _get_gpu_memory_from_vm_size(self, vm_size: str) -> float:
        """Get GPU memory based on VM size"""
        gpu_memory_map = {
            "Standard_NC4as_T4_v3": 16,
            "Standard_NC8as_T4_v3": 16,
            "Standard_NC16as_T4_v3": 64,  # 4x16GB
            "Standard_NC6s_v3": 16,
            "Standard_NC12s_v3": 32,  # 2x16GB
            "Standard_NC24s_v3": 64,  # 4x16GB
            "Standard_NC24ads_A100_v4": 80,
            "Standard_NC48ads_A100_v4": 160  # 2x80GB
        }
        return gpu_memory_map.get(vm_size, 16)


class GPUHealthMonitorManager:
    """Manager for GPU health monitoring across providers"""
    
    def __init__(self):
        self.monitors: Dict[str, BaseHealthMonitor] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.monitoring_interval = 60  # seconds
    
    def register_monitor(self, provider: str, monitor: BaseHealthMonitor) -> None:
        """Register a health monitor for a provider"""
        self.monitors[provider] = monitor
        logger.info(f"Registered health monitor for {provider}")
    
    async def start_monitoring(self, provider: str, gpu_ids: List[str]) -> None:
        """Start monitoring GPUs for a provider"""
        if provider not in self.monitors:
            raise ValueError(f"No monitor registered for provider {provider}")
        
        monitor = self.monitors[provider]
        await monitor.setup_monitoring()
        
        # Start monitoring tasks for each GPU
        for gpu_id in gpu_ids:
            task_key = f"{provider}:{gpu_id}"
            if task_key not in self.monitoring_tasks:
                task = asyncio.create_task(
                    self._monitor_loop(provider, gpu_id)
                )
                self.monitoring_tasks[task_key] = task
                logger.info(f"Started monitoring for {task_key}")
    
    async def _monitor_loop(self, provider: str, gpu_id: str) -> None:
        """Continuous monitoring loop for a GPU"""
        monitor = self.monitors[provider]
        
        while True:
            try:
                health_check = await monitor.monitor_gpu(gpu_id)
                logger.info(
                    f"GPU {gpu_id} health: {health_check.status.value} - {health_check.message}"
                )
                
                # Store health check result (could be sent to a time-series DB)
                await self._store_health_check(provider, gpu_id, health_check)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop for {gpu_id}: {e}")
            
            await asyncio.sleep(self.monitoring_interval)
    
    async def _store_health_check(self, provider: str, gpu_id: str, health_check: HealthCheck) -> None:
        """Store health check results (placeholder for actual storage)"""
        # In production, this would write to a time-series database
        # like InfluxDB, Prometheus, or CloudWatch
        pass
    
    async def stop_monitoring(self, provider: str, gpu_id: str) -> None:
        """Stop monitoring a specific GPU"""
        task_key = f"{provider}:{gpu_id}"
        if task_key in self.monitoring_tasks:
            self.monitoring_tasks[task_key].cancel()
            del self.monitoring_tasks[task_key]
            logger.info(f"Stopped monitoring for {task_key}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary across all monitored GPUs"""
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "providers": {}
        }
        
        for provider, monitor in self.monitors.items():
            gpu_summaries = {}
            
            # Get unique GPU IDs from monitoring tasks
            gpu_ids = [
                task_key.split(":")[1]
                for task_key in self.monitoring_tasks
                if task_key.startswith(f"{provider}:")
            ]
            
            for gpu_id in gpu_ids:
                gpu_summaries[gpu_id] = monitor.get_metrics_summary(gpu_id, minutes=5)
            
            summary["providers"][provider] = gpu_summaries
        
        return summary


# Example alert callback
async def email_alert_callback(health_check: HealthCheck):
    """Example callback to send email alerts"""
    if health_check.status == HealthStatus.CRITICAL:
        # In production, this would send an actual email
        logger.critical(f"CRITICAL ALERT: {health_check.gpu_id} - {health_check.message}")


# Example usage
if __name__ == "__main__":
    async def main():
        # Create health monitor manager
        manager = GPUHealthMonitorManager()
        
        # Create AWS monitor (mock clients for example)
        aws_monitor = AWSGPUHealthMonitor(
            cloudwatch_client=None,  # Would be actual client
            ec2_client=None,
            thresholds=HealthThresholds(
                utilization_warning=75,
                utilization_critical=90
            )
        )
        aws_monitor.register_alert_callback(email_alert_callback)
        
        # Register monitor
        manager.register_monitor("aws", aws_monitor)
        
        # Start monitoring
        await manager.start_monitoring("aws", ["i-1234567890abcdef0"])
        
        # Let it run for a bit
        await asyncio.sleep(300)
        
        # Get health summary
        summary = manager.get_health_summary()
        print(json.dumps(summary, indent=2))
    
    # Run the example
    asyncio.run(main())