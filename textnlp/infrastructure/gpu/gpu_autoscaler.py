"""
GPU Autoscaling Policies for TextNLP
Implements intelligent autoscaling based on workload patterns
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json

from .gpu_resource_manager import GPUType, GPUResourceManager
from .gpu_health_monitor import GPUMetrics, HealthStatus

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Possible scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


class ScalingMetric(Enum):
    """Metrics used for scaling decisions"""
    GPU_UTILIZATION = "gpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_LENGTH = "queue_length"
    INFERENCE_LATENCY = "inference_latency"
    COST_EFFICIENCY = "cost_efficiency"


@dataclass
class ScalingPolicy:
    """Configuration for autoscaling behavior"""
    min_instances: int = 1
    max_instances: int = 10
    target_utilization: float = 70.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_period: int = 300  # seconds
    scale_down_period: int = 600  # seconds
    cooldown_period: int = 300  # seconds
    metrics: List[ScalingMetric] = field(default_factory=lambda: [ScalingMetric.GPU_UTILIZATION])
    
    # Advanced policies
    predictive_scaling: bool = False
    cost_aware: bool = True
    spot_instance_ratio: float = 0.7  # Use up to 70% spot instances
    business_hours_min: int = 3  # Minimum during business hours
    off_hours_min: int = 1  # Minimum during off hours


@dataclass
class WorkloadMetrics:
    """Current workload metrics for scaling decisions"""
    timestamp: datetime
    gpu_utilization: float
    memory_utilization: float
    queue_length: int
    avg_inference_latency_ms: float
    active_models: List[str]
    requests_per_second: float
    

@dataclass
class ScalingDecision:
    """Result of scaling decision"""
    action: ScalingAction
    current_instances: int
    target_instances: int
    reason: str
    metrics: WorkloadMetrics
    estimated_cost_change: Optional[float] = None


class BaseAutoscaler(ABC):
    """Base class for GPU autoscaling"""
    
    def __init__(self, policy: ScalingPolicy):
        self.policy = policy
        self.current_instances = policy.min_instances
        self.last_scaling_time = datetime.utcnow()
        self.scaling_history: List[ScalingDecision] = []
        self.metrics_history: List[WorkloadMetrics] = []
        self.max_history_size = 1000
    
    @abstractmethod
    async def get_current_metrics(self) -> WorkloadMetrics:
        """Get current workload metrics"""
        pass
    
    @abstractmethod
    async def scale_instances(self, target_count: int) -> bool:
        """Scale to target number of instances"""
        pass
    
    @abstractmethod
    def calculate_cost(self, instance_count: int, gpu_type: GPUType) -> float:
        """Calculate hourly cost for given instance count"""
        pass
    
    def should_scale(self, metrics: WorkloadMetrics) -> ScalingDecision:
        """Determine if scaling is needed based on metrics"""
        # Check cooldown period
        if not self._is_cooldown_expired():
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                current_instances=self.current_instances,
                target_instances=self.current_instances,
                reason="In cooldown period",
                metrics=metrics
            )
        
        # Store metrics
        self._store_metrics(metrics)
        
        # Evaluate each metric
        scale_up_votes = 0
        scale_down_votes = 0
        reasons = []
        
        if ScalingMetric.GPU_UTILIZATION in self.policy.metrics:
            action, reason = self._evaluate_gpu_utilization(metrics)
            if action == ScalingAction.SCALE_UP:
                scale_up_votes += 1
            elif action == ScalingAction.SCALE_DOWN:
                scale_down_votes += 1
            reasons.append(reason)
        
        if ScalingMetric.MEMORY_UTILIZATION in self.policy.metrics:
            action, reason = self._evaluate_memory_utilization(metrics)
            if action == ScalingAction.SCALE_UP:
                scale_up_votes += 1
            elif action == ScalingAction.SCALE_DOWN:
                scale_down_votes += 1
            reasons.append(reason)
        
        if ScalingMetric.QUEUE_LENGTH in self.policy.metrics:
            action, reason = self._evaluate_queue_length(metrics)
            if action == ScalingAction.SCALE_UP:
                scale_up_votes += 1
            elif action == ScalingAction.SCALE_DOWN:
                scale_down_votes += 1
            reasons.append(reason)
        
        if ScalingMetric.INFERENCE_LATENCY in self.policy.metrics:
            action, reason = self._evaluate_latency(metrics)
            if action == ScalingAction.SCALE_UP:
                scale_up_votes += 1
            elif action == ScalingAction.SCALE_DOWN:
                scale_down_votes += 1
            reasons.append(reason)
        
        # Determine final action
        if scale_up_votes > scale_down_votes and scale_up_votes > 0:
            target = min(self.current_instances + 1, self.policy.max_instances)
            if self.policy.predictive_scaling:
                # Look ahead and potentially scale more aggressively
                predicted_load = self._predict_future_load()
                if predicted_load > 0.9:
                    target = min(self.current_instances + 2, self.policy.max_instances)
            
            return ScalingDecision(
                action=ScalingAction.SCALE_UP if target > self.current_instances else ScalingAction.NO_ACTION,
                current_instances=self.current_instances,
                target_instances=target,
                reason="; ".join(reasons),
                metrics=metrics
            )
        elif scale_down_votes > scale_up_votes and scale_down_votes > 0:
            # Check if we're in business hours
            min_instances = self._get_minimum_instances()
            target = max(self.current_instances - 1, min_instances)
            
            return ScalingDecision(
                action=ScalingAction.SCALE_DOWN if target < self.current_instances else ScalingAction.NO_ACTION,
                current_instances=self.current_instances,
                target_instances=target,
                reason="; ".join(reasons),
                metrics=metrics
            )
        
        return ScalingDecision(
            action=ScalingAction.NO_ACTION,
            current_instances=self.current_instances,
            target_instances=self.current_instances,
            reason="Metrics within normal range",
            metrics=metrics
        )
    
    def _evaluate_gpu_utilization(self, metrics: WorkloadMetrics) -> Tuple[ScalingAction, str]:
        """Evaluate GPU utilization for scaling"""
        recent_utilization = self._get_average_metric(
            lambda m: m.gpu_utilization,
            self.policy.scale_up_period
        )
        
        if recent_utilization > self.policy.scale_up_threshold:
            return ScalingAction.SCALE_UP, f"GPU utilization {recent_utilization:.1f}% > {self.policy.scale_up_threshold}%"
        elif recent_utilization < self.policy.scale_down_threshold:
            return ScalingAction.SCALE_DOWN, f"GPU utilization {recent_utilization:.1f}% < {self.policy.scale_down_threshold}%"
        
        return ScalingAction.NO_ACTION, f"GPU utilization {recent_utilization:.1f}% within range"
    
    def _evaluate_memory_utilization(self, metrics: WorkloadMetrics) -> Tuple[ScalingAction, str]:
        """Evaluate memory utilization for scaling"""
        if metrics.memory_utilization > 85:
            return ScalingAction.SCALE_UP, f"Memory utilization critical: {metrics.memory_utilization:.1f}%"
        elif metrics.memory_utilization < 20 and self.current_instances > self._get_minimum_instances():
            return ScalingAction.SCALE_DOWN, f"Memory underutilized: {metrics.memory_utilization:.1f}%"
        
        return ScalingAction.NO_ACTION, f"Memory utilization normal: {metrics.memory_utilization:.1f}%"
    
    def _evaluate_queue_length(self, metrics: WorkloadMetrics) -> Tuple[ScalingAction, str]:
        """Evaluate request queue length for scaling"""
        # Scale up if queue is building up
        queue_per_instance = metrics.queue_length / self.current_instances
        if queue_per_instance > 10:
            return ScalingAction.SCALE_UP, f"Queue length high: {metrics.queue_length} requests"
        elif queue_per_instance < 1 and self.current_instances > self._get_minimum_instances():
            return ScalingAction.SCALE_DOWN, f"Queue length low: {metrics.queue_length} requests"
        
        return ScalingAction.NO_ACTION, f"Queue length normal: {metrics.queue_length} requests"
    
    def _evaluate_latency(self, metrics: WorkloadMetrics) -> Tuple[ScalingAction, str]:
        """Evaluate inference latency for scaling"""
        # Target latency depends on model type
        target_latency = 500  # ms for most models
        if any(model in metrics.active_models for model in ["gpt-j", "llama"]):
            target_latency = 1000  # ms for large models
        
        if metrics.avg_inference_latency_ms > target_latency * 1.5:
            return ScalingAction.SCALE_UP, f"Latency high: {metrics.avg_inference_latency_ms:.0f}ms"
        elif metrics.avg_inference_latency_ms < target_latency * 0.5:
            return ScalingAction.SCALE_DOWN, f"Latency low: {metrics.avg_inference_latency_ms:.0f}ms"
        
        return ScalingAction.NO_ACTION, f"Latency normal: {metrics.avg_inference_latency_ms:.0f}ms"
    
    def _is_cooldown_expired(self) -> bool:
        """Check if cooldown period has expired"""
        elapsed = (datetime.utcnow() - self.last_scaling_time).seconds
        return elapsed >= self.policy.cooldown_period
    
    def _get_minimum_instances(self) -> int:
        """Get minimum instances based on time of day"""
        current_hour = datetime.utcnow().hour
        is_business_hours = 8 <= current_hour <= 18  # UTC
        
        if is_business_hours:
            return max(self.policy.business_hours_min, self.policy.min_instances)
        else:
            return max(self.policy.off_hours_min, self.policy.min_instances)
    
    def _store_metrics(self, metrics: WorkloadMetrics) -> None:
        """Store metrics in history"""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history.pop(0)
    
    def _get_average_metric(self, metric_func, period_seconds: int) -> float:
        """Get average of a metric over a time period"""
        cutoff_time = datetime.utcnow() - timedelta(seconds=period_seconds)
        recent_metrics = [
            metric_func(m) for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return metric_func(self.metrics_history[-1]) if self.metrics_history else 0
        
        return statistics.mean(recent_metrics)
    
    def _predict_future_load(self) -> float:
        """Predict future load based on historical patterns"""
        if len(self.metrics_history) < 10:
            return 0.5  # Not enough data
        
        # Simple prediction based on trend
        recent_utilizations = [m.gpu_utilization for m in self.metrics_history[-10:]]
        if len(recent_utilizations) < 2:
            return recent_utilizations[-1] / 100
        
        # Calculate trend
        trend = (recent_utilizations[-1] - recent_utilizations[0]) / len(recent_utilizations)
        predicted = recent_utilizations[-1] + (trend * 5)  # Predict 5 steps ahead
        
        return max(0, min(1, predicted / 100))
    
    async def execute_scaling(self, decision: ScalingDecision) -> bool:
        """Execute a scaling decision"""
        if decision.action == ScalingAction.NO_ACTION:
            return True
        
        logger.info(
            f"Executing scaling: {decision.action.value} from {decision.current_instances} "
            f"to {decision.target_instances} instances. Reason: {decision.reason}"
        )
        
        try:
            success = await self.scale_instances(decision.target_instances)
            if success:
                self.current_instances = decision.target_instances
                self.last_scaling_time = datetime.utcnow()
                self.scaling_history.append(decision)
                
                # Trim history
                if len(self.scaling_history) > self.max_history_size:
                    self.scaling_history.pop(0)
                
                logger.info(f"Scaling completed successfully")
                return True
            else:
                logger.error(f"Scaling failed")
                return False
        except Exception as e:
            logger.error(f"Error during scaling: {e}")
            return False
    
    def get_scaling_report(self) -> Dict[str, Any]:
        """Generate a scaling report"""
        if not self.scaling_history:
            return {"message": "No scaling events recorded"}
        
        recent_events = self.scaling_history[-10:]
        scale_up_count = sum(1 for e in recent_events if e.action == ScalingAction.SCALE_UP)
        scale_down_count = sum(1 for e in recent_events if e.action == ScalingAction.SCALE_DOWN)
        
        return {
            "current_instances": self.current_instances,
            "last_scaling_time": self.last_scaling_time.isoformat(),
            "recent_scaling_events": scale_up_count + scale_down_count,
            "scale_up_events": scale_up_count,
            "scale_down_events": scale_down_count,
            "recent_events": [
                {
                    "timestamp": e.metrics.timestamp.isoformat(),
                    "action": e.action.value,
                    "from_instances": e.current_instances,
                    "to_instances": e.target_instances,
                    "reason": e.reason
                }
                for e in recent_events
            ]
        }


class AWSGPUAutoscaler(BaseAutoscaler):
    """AWS-specific GPU autoscaling implementation"""
    
    def __init__(self, policy: ScalingPolicy, auto_scaling_client, cloudwatch_client, instance_prefix: str):
        super().__init__(policy)
        self.auto_scaling = auto_scaling_client
        self.cloudwatch = cloudwatch_client
        self.instance_prefix = instance_prefix
        self.auto_scaling_group_name = f"{instance_prefix}-gpu-asg"
    
    async def get_current_metrics(self) -> WorkloadMetrics:
        """Get current metrics from CloudWatch"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)
        
        # Get GPU utilization
        gpu_response = self.cloudwatch.get_metric_statistics(
            Namespace='TextNLP/GPU',
            MetricName='AverageGPUUtilization',
            Dimensions=[{'Name': 'AutoScalingGroup', 'Value': self.auto_scaling_group_name}],
            StartTime=start_time,
            EndTime=end_time,
            Period=60,
            Statistics=['Average']
        )
        
        gpu_utilization = self._extract_latest_value(gpu_response, 0)
        
        # Get memory utilization
        memory_response = self.cloudwatch.get_metric_statistics(
            Namespace='TextNLP/GPU',
            MetricName='AverageMemoryUtilization',
            Dimensions=[{'Name': 'AutoScalingGroup', 'Value': self.auto_scaling_group_name}],
            StartTime=start_time,
            EndTime=end_time,
            Period=60,
            Statistics=['Average']
        )
        
        memory_utilization = self._extract_latest_value(memory_response, 0)
        
        # Get queue length from SQS or custom metric
        queue_response = self.cloudwatch.get_metric_statistics(
            Namespace='TextNLP/Application',
            MetricName='InferenceQueueLength',
            StartTime=start_time,
            EndTime=end_time,
            Period=60,
            Statistics=['Average']
        )
        
        queue_length = int(self._extract_latest_value(queue_response, 0))
        
        # Get inference latency
        latency_response = self.cloudwatch.get_metric_statistics(
            Namespace='TextNLP/Application',
            MetricName='InferenceLatency',
            StartTime=start_time,
            EndTime=end_time,
            Period=60,
            Statistics=['Average'],
            Unit='Milliseconds'
        )
        
        latency = self._extract_latest_value(latency_response, 0)
        
        # Get request rate
        request_response = self.cloudwatch.get_metric_statistics(
            Namespace='TextNLP/Application',
            MetricName='RequestsPerSecond',
            StartTime=start_time,
            EndTime=end_time,
            Period=60,
            Statistics=['Average']
        )
        
        requests_per_second = self._extract_latest_value(request_response, 0)
        
        return WorkloadMetrics(
            timestamp=datetime.utcnow(),
            gpu_utilization=gpu_utilization,
            memory_utilization=memory_utilization,
            queue_length=queue_length,
            avg_inference_latency_ms=latency,
            active_models=["gpt2", "bert"],  # Would query from application
            requests_per_second=requests_per_second
        )
    
    def _extract_latest_value(self, response: Dict, default: float) -> float:
        """Extract the latest value from CloudWatch response"""
        datapoints = response.get('Datapoints', [])
        if not datapoints:
            return default
        
        # Sort by timestamp and get the latest
        sorted_points = sorted(datapoints, key=lambda x: x['Timestamp'])
        return sorted_points[-1].get('Average', default)
    
    async def scale_instances(self, target_count: int) -> bool:
        """Scale Auto Scaling Group to target count"""
        try:
            # Update desired capacity
            response = self.auto_scaling.set_desired_capacity(
                AutoScalingGroupName=self.auto_scaling_group_name,
                DesiredCapacity=target_count,
                HonorCooldown=False  # Override cooldown for immediate scaling
            )
            
            # Create scaling activity alarm
            self.cloudwatch.put_metric_alarm(
                AlarmName=f"{self.instance_prefix}-scaling-in-progress",
                ComparisonOperator='LessThan',
                EvaluationPeriods=1,
                MetricName='GroupDesiredCapacity',
                Namespace='AWS/AutoScaling',
                Period=60,
                Statistic='Average',
                Threshold=target_count,
                ActionsEnabled=False
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to scale ASG: {e}")
            return False
    
    def calculate_cost(self, instance_count: int, gpu_type: GPUType) -> float:
        """Calculate hourly cost for AWS GPU instances"""
        # Base costs per instance type
        costs = {
            GPUType.NVIDIA_T4: 0.526,  # g4dn.xlarge
            GPUType.NVIDIA_V100: 3.06,  # p3.2xlarge
            GPUType.NVIDIA_A100: 4.10,  # p4d.24xlarge divided by 8 GPUs
            GPUType.NVIDIA_A10G: 1.006  # g5.xlarge
        }
        
        base_cost = costs.get(gpu_type, 1.0)
        
        # Apply spot instance discount if enabled
        if self.policy.cost_aware and self.policy.spot_instance_ratio > 0:
            spot_discount = 0.7  # Assume 30% discount for spot
            spot_instances = int(instance_count * self.policy.spot_instance_ratio)
            on_demand_instances = instance_count - spot_instances
            
            total_cost = (on_demand_instances * base_cost) + (spot_instances * base_cost * spot_discount)
            return total_cost
        
        return instance_count * base_cost


class GCPGPUAutoscaler(BaseAutoscaler):
    """GCP-specific GPU autoscaling implementation"""
    
    def __init__(self, policy: ScalingPolicy, compute_client, monitoring_client, project_id: str, zone: str):
        super().__init__(policy)
        self.compute = compute_client
        self.monitoring = monitoring_client
        self.project_id = project_id
        self.zone = zone
        self.instance_group_name = "textnlp-gpu-mig"
    
    async def get_current_metrics(self) -> WorkloadMetrics:
        """Get current metrics from Stackdriver"""
        # Implementation similar to AWS but using GCP APIs
        # This is a simplified version
        return WorkloadMetrics(
            timestamp=datetime.utcnow(),
            gpu_utilization=75.0,
            memory_utilization=60.0,
            queue_length=5,
            avg_inference_latency_ms=450,
            active_models=["gpt2"],
            requests_per_second=10.5
        )
    
    async def scale_instances(self, target_count: int) -> bool:
        """Scale GCP Managed Instance Group"""
        try:
            operation = self.compute.instanceGroupManagers().resize(
                project=self.project_id,
                zone=self.zone,
                instanceGroupManager=self.instance_group_name,
                size=target_count
            ).execute()
            
            # Wait for operation to complete (in production, this would be async)
            return True
        except Exception as e:
            logger.error(f"Failed to scale MIG: {e}")
            return False
    
    def calculate_cost(self, instance_count: int, gpu_type: GPUType) -> float:
        """Calculate hourly cost for GCP GPU instances"""
        costs = {
            GPUType.NVIDIA_T4: 0.35,
            GPUType.NVIDIA_V100: 2.48,
            GPUType.NVIDIA_A100: 3.67,
            GPUType.NVIDIA_K80: 0.45
        }
        
        base_cost = costs.get(gpu_type, 1.0)
        
        # GCP preemptible instance discount
        if self.policy.cost_aware and self.policy.spot_instance_ratio > 0:
            preemptible_discount = 0.8  # 80% discount
            preemptible_instances = int(instance_count * self.policy.spot_instance_ratio)
            standard_instances = instance_count - preemptible_instances
            
            total_cost = (standard_instances * base_cost) + (preemptible_instances * base_cost * (1 - preemptible_discount))
            return total_cost
        
        return instance_count * base_cost


class AzureGPUAutoscaler(BaseAutoscaler):
    """Azure-specific GPU autoscaling implementation"""
    
    def __init__(self, policy: ScalingPolicy, monitor_client, compute_client, resource_group: str):
        super().__init__(policy)
        self.monitor = monitor_client
        self.compute = compute_client
        self.resource_group = resource_group
        self.scale_set_name = "textnlp-gpu-vmss"
    
    async def get_current_metrics(self) -> WorkloadMetrics:
        """Get current metrics from Azure Monitor"""
        # Implementation similar to AWS but using Azure APIs
        return WorkloadMetrics(
            timestamp=datetime.utcnow(),
            gpu_utilization=70.0,
            memory_utilization=55.0,
            queue_length=3,
            avg_inference_latency_ms=400,
            active_models=["gpt2", "bert"],
            requests_per_second=8.0
        )
    
    async def scale_instances(self, target_count: int) -> bool:
        """Scale Azure VM Scale Set"""
        try:
            scale_set = self.compute.virtual_machine_scale_sets.get(
                resource_group_name=self.resource_group,
                vm_scale_set_name=self.scale_set_name
            )
            
            scale_set.sku.capacity = target_count
            
            async_update = self.compute.virtual_machine_scale_sets.create_or_update(
                resource_group_name=self.resource_group,
                vm_scale_set_name=self.scale_set_name,
                parameters=scale_set
            )
            
            # In production, would wait for completion
            return True
        except Exception as e:
            logger.error(f"Failed to scale VMSS: {e}")
            return False
    
    def calculate_cost(self, instance_count: int, gpu_type: GPUType) -> float:
        """Calculate hourly cost for Azure GPU instances"""
        costs = {
            GPUType.NVIDIA_T4: 0.526,
            GPUType.NVIDIA_V100: 3.06,
            GPUType.NVIDIA_A100: 3.67
        }
        
        base_cost = costs.get(gpu_type, 1.0)
        
        # Azure spot instance discount
        if self.policy.cost_aware and self.policy.spot_instance_ratio > 0:
            spot_discount = 0.6  # 60% discount typical
            spot_instances = int(instance_count * self.policy.spot_instance_ratio)
            standard_instances = instance_count - spot_instances
            
            total_cost = (standard_instances * base_cost) + (spot_instances * base_cost * (1 - spot_discount))
            return total_cost
        
        return instance_count * base_cost


class GPUAutoscalerManager:
    """Manager for GPU autoscaling across providers"""
    
    def __init__(self):
        self.autoscalers: Dict[str, BaseAutoscaler] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.check_interval = 60  # seconds
    
    def register_autoscaler(self, provider: str, autoscaler: BaseAutoscaler) -> None:
        """Register an autoscaler for a provider"""
        self.autoscalers[provider] = autoscaler
        logger.info(f"Registered autoscaler for {provider}")
    
    async def start_autoscaling(self, provider: str) -> None:
        """Start autoscaling for a provider"""
        if provider not in self.autoscalers:
            raise ValueError(f"No autoscaler registered for {provider}")
        
        if provider not in self.monitoring_tasks:
            task = asyncio.create_task(self._autoscaling_loop(provider))
            self.monitoring_tasks[provider] = task
            logger.info(f"Started autoscaling for {provider}")
    
    async def _autoscaling_loop(self, provider: str) -> None:
        """Continuous autoscaling loop"""
        autoscaler = self.autoscalers[provider]
        
        while True:
            try:
                # Get current metrics
                metrics = await autoscaler.get_current_metrics()
                
                # Make scaling decision
                decision = autoscaler.should_scale(metrics)
                
                # Execute if needed
                if decision.action != ScalingAction.NO_ACTION:
                    await autoscaler.execute_scaling(decision)
                    
                    # Log cost impact if available
                    if autoscaler.policy.cost_aware:
                        current_cost = autoscaler.calculate_cost(
                            decision.current_instances,
                            GPUType.NVIDIA_T4  # Default for example
                        )
                        new_cost = autoscaler.calculate_cost(
                            decision.target_instances,
                            GPUType.NVIDIA_T4
                        )
                        logger.info(
                            f"Cost impact: ${current_cost:.2f}/hr -> ${new_cost:.2f}/hr "
                            f"(${(new_cost - current_cost):.2f}/hr change)"
                        )
                
            except Exception as e:
                logger.error(f"Error in autoscaling loop for {provider}: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    async def stop_autoscaling(self, provider: str) -> None:
        """Stop autoscaling for a provider"""
        if provider in self.monitoring_tasks:
            self.monitoring_tasks[provider].cancel()
            del self.monitoring_tasks[provider]
            logger.info(f"Stopped autoscaling for {provider}")
    
    def get_autoscaling_status(self) -> Dict[str, Any]:
        """Get status of all autoscalers"""
        status = {}
        
        for provider, autoscaler in self.autoscalers.items():
            status[provider] = {
                "active": provider in self.monitoring_tasks,
                "current_instances": autoscaler.current_instances,
                "policy": {
                    "min_instances": autoscaler.policy.min_instances,
                    "max_instances": autoscaler.policy.max_instances,
                    "target_utilization": autoscaler.policy.target_utilization
                },
                "report": autoscaler.get_scaling_report()
            }
        
        return status


# Example usage
if __name__ == "__main__":
    async def main():
        # Create autoscaler manager
        manager = GPUAutoscalerManager()
        
        # Create scaling policy
        policy = ScalingPolicy(
            min_instances=1,
            max_instances=10,
            target_utilization=70.0,
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            metrics=[
                ScalingMetric.GPU_UTILIZATION,
                ScalingMetric.QUEUE_LENGTH,
                ScalingMetric.INFERENCE_LATENCY
            ],
            predictive_scaling=True,
            cost_aware=True,
            spot_instance_ratio=0.7
        )
        
        # Create AWS autoscaler (with mock clients)
        aws_autoscaler = AWSGPUAutoscaler(
            policy=policy,
            auto_scaling_client=None,  # Would be actual client
            cloudwatch_client=None,
            instance_prefix="textnlp-prod"
        )
        
        # Register and start
        manager.register_autoscaler("aws", aws_autoscaler)
        await manager.start_autoscaling("aws")
        
        # Let it run
        await asyncio.sleep(300)
        
        # Get status
        status = manager.get_autoscaling_status()
        print(json.dumps(status, indent=2))
    
    asyncio.run(main())