"""
GPU Resource Manager for TextNLP
Manages GPU resources across cloud providers (AWS, GCP, Azure)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class GPUType(Enum):
    """Supported GPU types across cloud providers"""
    NVIDIA_T4 = "nvidia-t4"
    NVIDIA_V100 = "nvidia-v100"
    NVIDIA_A100 = "nvidia-a100"
    NVIDIA_A10G = "nvidia-a10g"
    NVIDIA_K80 = "nvidia-k80"


@dataclass
class GPUConfig:
    """GPU configuration details"""
    gpu_type: GPUType
    count: int
    memory_gb: int
    compute_capability: str
    cuda_cores: int


@dataclass
class GPUInstance:
    """GPU instance configuration for cloud providers"""
    instance_type: str
    gpu_config: GPUConfig
    vcpus: int
    memory_gb: int
    network_performance: str
    cost_per_hour: float


class BaseGPUProvider(ABC):
    """Base class for GPU providers"""
    
    def __init__(self, region: str):
        self.region = region
        self.gpu_instances: Dict[GPUType, GPUInstance] = {}
        self._initialize_gpu_mappings()
    
    @abstractmethod
    def _initialize_gpu_mappings(self):
        """Initialize GPU instance mappings for the provider"""
        pass
    
    @abstractmethod
    def get_instance_type(self, gpu_type: GPUType, count: int = 1) -> str:
        """Get instance type for requested GPU configuration"""
        pass
    
    @abstractmethod
    def validate_gpu_availability(self, gpu_type: GPUType) -> bool:
        """Check if GPU type is available in the region"""
        pass
    
    @abstractmethod
    def get_gpu_health_checks(self) -> Dict[str, Any]:
        """Get GPU health monitoring configuration"""
        pass


class AWSGPUProvider(BaseGPUProvider):
    """AWS GPU provider implementation"""
    
    def _initialize_gpu_mappings(self):
        """Initialize AWS GPU instance mappings"""
        self.gpu_instances = {
            GPUType.NVIDIA_T4: GPUInstance(
                instance_type="g4dn.xlarge",
                gpu_config=GPUConfig(
                    gpu_type=GPUType.NVIDIA_T4,
                    count=1,
                    memory_gb=16,
                    compute_capability="7.5",
                    cuda_cores=2560
                ),
                vcpus=4,
                memory_gb=16,
                network_performance="Up to 25 Gbps",
                cost_per_hour=0.526
            ),
            GPUType.NVIDIA_V100: GPUInstance(
                instance_type="p3.2xlarge",
                gpu_config=GPUConfig(
                    gpu_type=GPUType.NVIDIA_V100,
                    count=1,
                    memory_gb=16,
                    compute_capability="7.0",
                    cuda_cores=5120
                ),
                vcpus=8,
                memory_gb=61,
                network_performance="Up to 10 Gbps",
                cost_per_hour=3.06
            ),
            GPUType.NVIDIA_A100: GPUInstance(
                instance_type="p4d.24xlarge",
                gpu_config=GPUConfig(
                    gpu_type=GPUType.NVIDIA_A100,
                    count=8,
                    memory_gb=40,
                    compute_capability="8.0",
                    cuda_cores=6912
                ),
                vcpus=96,
                memory_gb=1152,
                network_performance="400 Gbps",
                cost_per_hour=32.77
            ),
            GPUType.NVIDIA_A10G: GPUInstance(
                instance_type="g5.xlarge",
                gpu_config=GPUConfig(
                    gpu_type=GPUType.NVIDIA_A10G,
                    count=1,
                    memory_gb=24,
                    compute_capability="8.6",
                    cuda_cores=9216
                ),
                vcpus=4,
                memory_gb=16,
                network_performance="Up to 10 Gbps",
                cost_per_hour=1.006
            )
        }
        
        # Multi-GPU instance mappings
        self.multi_gpu_instances = {
            (GPUType.NVIDIA_T4, 4): "g4dn.12xlarge",
            (GPUType.NVIDIA_V100, 4): "p3.8xlarge",
            (GPUType.NVIDIA_V100, 8): "p3.16xlarge",
            (GPUType.NVIDIA_A10G, 4): "g5.12xlarge"
        }
    
    def get_instance_type(self, gpu_type: GPUType, count: int = 1) -> str:
        """Get AWS instance type for requested GPU configuration"""
        if count == 1:
            if gpu_type in self.gpu_instances:
                return self.gpu_instances[gpu_type].instance_type
        else:
            key = (gpu_type, count)
            if key in self.multi_gpu_instances:
                return self.multi_gpu_instances[key]
        
        raise ValueError(f"No AWS instance found for {count}x {gpu_type.value}")
    
    def validate_gpu_availability(self, gpu_type: GPUType) -> bool:
        """Check GPU availability in AWS region"""
        # Region-specific availability
        regional_availability = {
            "us-east-1": [GPUType.NVIDIA_T4, GPUType.NVIDIA_V100, GPUType.NVIDIA_A100, GPUType.NVIDIA_A10G],
            "us-west-2": [GPUType.NVIDIA_T4, GPUType.NVIDIA_V100, GPUType.NVIDIA_A10G],
            "eu-west-1": [GPUType.NVIDIA_T4, GPUType.NVIDIA_V100, GPUType.NVIDIA_A10G],
            "ap-southeast-1": [GPUType.NVIDIA_T4, GPUType.NVIDIA_V100]
        }
        
        available_gpus = regional_availability.get(self.region, [])
        return gpu_type in available_gpus
    
    def get_gpu_health_checks(self) -> Dict[str, Any]:
        """AWS GPU health monitoring configuration"""
        return {
            "cloudwatch_metrics": [
                "GPUUtilization",
                "GPUMemoryUtilization",
                "GPUTemperature"
            ],
            "health_check_interval": 60,
            "alarm_thresholds": {
                "gpu_utilization_high": 90,
                "gpu_memory_high": 95,
                "gpu_temperature_high": 85
            },
            "custom_metrics_namespace": "TextNLP/GPU"
        }


class GCPGPUProvider(BaseGPUProvider):
    """GCP GPU provider implementation"""
    
    def _initialize_gpu_mappings(self):
        """Initialize GCP GPU instance mappings"""
        self.gpu_instances = {
            GPUType.NVIDIA_T4: GPUInstance(
                instance_type="n1-standard-4",
                gpu_config=GPUConfig(
                    gpu_type=GPUType.NVIDIA_T4,
                    count=1,
                    memory_gb=16,
                    compute_capability="7.5",
                    cuda_cores=2560
                ),
                vcpus=4,
                memory_gb=15,
                network_performance="10 Gbps",
                cost_per_hour=0.35
            ),
            GPUType.NVIDIA_V100: GPUInstance(
                instance_type="n1-standard-8",
                gpu_config=GPUConfig(
                    gpu_type=GPUType.NVIDIA_V100,
                    count=1,
                    memory_gb=16,
                    compute_capability="7.0",
                    cuda_cores=5120
                ),
                vcpus=8,
                memory_gb=30,
                network_performance="16 Gbps",
                cost_per_hour=2.48
            ),
            GPUType.NVIDIA_A100: GPUInstance(
                instance_type="a2-highgpu-1g",
                gpu_config=GPUConfig(
                    gpu_type=GPUType.NVIDIA_A100,
                    count=1,
                    memory_gb=40,
                    compute_capability="8.0",
                    cuda_cores=6912
                ),
                vcpus=12,
                memory_gb=85,
                network_performance="24 Gbps",
                cost_per_hour=3.67
            ),
            GPUType.NVIDIA_K80: GPUInstance(
                instance_type="n1-standard-2",
                gpu_config=GPUConfig(
                    gpu_type=GPUType.NVIDIA_K80,
                    count=1,
                    memory_gb=12,
                    compute_capability="3.7",
                    cuda_cores=2496
                ),
                vcpus=2,
                memory_gb=7.5,
                network_performance="10 Gbps",
                cost_per_hour=0.45
            )
        }
        
        # GCP accelerator attachments
        self.accelerator_types = {
            GPUType.NVIDIA_T4: "nvidia-tesla-t4",
            GPUType.NVIDIA_V100: "nvidia-tesla-v100",
            GPUType.NVIDIA_A100: "nvidia-tesla-a100",
            GPUType.NVIDIA_K80: "nvidia-tesla-k80"
        }
    
    def get_instance_type(self, gpu_type: GPUType, count: int = 1) -> str:
        """Get GCP instance configuration for requested GPU"""
        if gpu_type == GPUType.NVIDIA_A100:
            # A100 uses specific machine types
            if count == 1:
                return "a2-highgpu-1g"
            elif count == 2:
                return "a2-highgpu-2g"
            elif count == 4:
                return "a2-highgpu-4g"
            elif count == 8:
                return "a2-highgpu-8g"
        else:
            # Other GPUs use n1 instances with attached accelerators
            base_instance = self.gpu_instances.get(gpu_type)
            if base_instance:
                return f"{base_instance.instance_type}+{count}x{self.accelerator_types[gpu_type]}"
        
        raise ValueError(f"No GCP instance found for {count}x {gpu_type.value}")
    
    def validate_gpu_availability(self, gpu_type: GPUType) -> bool:
        """Check GPU availability in GCP region"""
        regional_availability = {
            "us-central1": [GPUType.NVIDIA_T4, GPUType.NVIDIA_V100, GPUType.NVIDIA_K80, GPUType.NVIDIA_A100],
            "us-west1": [GPUType.NVIDIA_T4, GPUType.NVIDIA_V100, GPUType.NVIDIA_K80],
            "europe-west4": [GPUType.NVIDIA_T4, GPUType.NVIDIA_V100, GPUType.NVIDIA_A100],
            "asia-east1": [GPUType.NVIDIA_T4, GPUType.NVIDIA_V100, GPUType.NVIDIA_K80]
        }
        
        available_gpus = regional_availability.get(self.region, [])
        return gpu_type in available_gpus
    
    def get_gpu_health_checks(self) -> Dict[str, Any]:
        """GCP GPU health monitoring configuration"""
        return {
            "stackdriver_metrics": [
                "compute.googleapis.com/instance/gpu/utilization",
                "compute.googleapis.com/instance/gpu/memory_utilization",
                "compute.googleapis.com/instance/gpu/temperature"
            ],
            "monitoring_interval": 60,
            "alert_policies": {
                "gpu_utilization_threshold": 90,
                "gpu_memory_threshold": 95,
                "gpu_temperature_threshold": 85
            },
            "custom_metrics_prefix": "custom.googleapis.com/textnlp/gpu/"
        }


class AzureGPUProvider(BaseGPUProvider):
    """Azure GPU provider implementation"""
    
    def _initialize_gpu_mappings(self):
        """Initialize Azure GPU instance mappings"""
        self.gpu_instances = {
            GPUType.NVIDIA_T4: GPUInstance(
                instance_type="Standard_NC4as_T4_v3",
                gpu_config=GPUConfig(
                    gpu_type=GPUType.NVIDIA_T4,
                    count=1,
                    memory_gb=16,
                    compute_capability="7.5",
                    cuda_cores=2560
                ),
                vcpus=4,
                memory_gb=28,
                network_performance="8 Gbps",
                cost_per_hour=0.526
            ),
            GPUType.NVIDIA_V100: GPUInstance(
                instance_type="Standard_NC6s_v3",
                gpu_config=GPUConfig(
                    gpu_type=GPUType.NVIDIA_V100,
                    count=1,
                    memory_gb=16,
                    compute_capability="7.0",
                    cuda_cores=5120
                ),
                vcpus=6,
                memory_gb=112,
                network_performance="24 Gbps",
                cost_per_hour=3.06
            ),
            GPUType.NVIDIA_A100: GPUInstance(
                instance_type="Standard_NC24ads_A100_v4",
                gpu_config=GPUConfig(
                    gpu_type=GPUType.NVIDIA_A100,
                    count=1,
                    memory_gb=80,
                    compute_capability="8.0",
                    cuda_cores=6912
                ),
                vcpus=24,
                memory_gb=220,
                network_performance="40 Gbps",
                cost_per_hour=3.67
            )
        }
        
        # Multi-GPU Azure instances
        self.multi_gpu_instances = {
            (GPUType.NVIDIA_T4, 4): "Standard_NC16as_T4_v3",
            (GPUType.NVIDIA_V100, 2): "Standard_NC12s_v3",
            (GPUType.NVIDIA_V100, 4): "Standard_NC24s_v3",
            (GPUType.NVIDIA_A100, 2): "Standard_NC48ads_A100_v4"
        }
    
    def get_instance_type(self, gpu_type: GPUType, count: int = 1) -> str:
        """Get Azure instance type for requested GPU configuration"""
        if count == 1:
            if gpu_type in self.gpu_instances:
                return self.gpu_instances[gpu_type].instance_type
        else:
            key = (gpu_type, count)
            if key in self.multi_gpu_instances:
                return self.multi_gpu_instances[key]
        
        raise ValueError(f"No Azure instance found for {count}x {gpu_type.value}")
    
    def validate_gpu_availability(self, gpu_type: GPUType) -> bool:
        """Check GPU availability in Azure region"""
        regional_availability = {
            "eastus": [GPUType.NVIDIA_T4, GPUType.NVIDIA_V100, GPUType.NVIDIA_A100],
            "westus2": [GPUType.NVIDIA_T4, GPUType.NVIDIA_V100],
            "westeurope": [GPUType.NVIDIA_T4, GPUType.NVIDIA_V100, GPUType.NVIDIA_A100],
            "southeastasia": [GPUType.NVIDIA_T4, GPUType.NVIDIA_V100]
        }
        
        available_gpus = regional_availability.get(self.region, [])
        return gpu_type in available_gpus
    
    def get_gpu_health_checks(self) -> Dict[str, Any]:
        """Azure GPU health monitoring configuration"""
        return {
            "azure_monitor_metrics": [
                "GPU Utilization Percentage",
                "GPU Memory Utilization Percentage",
                "GPU Temperature"
            ],
            "diagnostic_settings": {
                "logs": ["GuestOSUpdates", "GPUHealthEvents"],
                "metrics": ["AllMetrics"]
            },
            "alert_rules": {
                "gpu_high_utilization": {
                    "threshold": 90,
                    "time_aggregation": "Average",
                    "frequency": "PT1M"
                },
                "gpu_memory_pressure": {
                    "threshold": 95,
                    "time_aggregation": "Average",
                    "frequency": "PT1M"
                }
            }
        }


class GPUResourceManager:
    """Main GPU Resource Manager coordinating across cloud providers"""
    
    def __init__(self):
        self.providers: Dict[str, BaseGPUProvider] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_provider(self, provider_name: str, region: str) -> None:
        """Register a cloud provider for GPU management"""
        provider_map = {
            "aws": AWSGPUProvider,
            "gcp": GCPGPUProvider,
            "azure": AzureGPUProvider
        }
        
        if provider_name.lower() not in provider_map:
            raise ValueError(f"Unsupported provider: {provider_name}")
        
        provider_class = provider_map[provider_name.lower()]
        self.providers[provider_name] = provider_class(region)
        self.logger.info(f"Registered {provider_name} GPU provider for region {region}")
    
    def get_gpu_instance(self, provider: str, gpu_type: GPUType, count: int = 1) -> Dict[str, Any]:
        """Get GPU instance configuration for specified provider"""
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not registered")
        
        gpu_provider = self.providers[provider]
        
        # Validate availability
        if not gpu_provider.validate_gpu_availability(gpu_type):
            raise ValueError(f"{gpu_type.value} not available in {gpu_provider.region}")
        
        try:
            instance_type = gpu_provider.get_instance_type(gpu_type, count)
            instance_info = gpu_provider.gpu_instances.get(gpu_type)
            
            return {
                "provider": provider,
                "instance_type": instance_type,
                "gpu_type": gpu_type.value,
                "gpu_count": count,
                "region": gpu_provider.region,
                "instance_details": instance_info.__dict__ if instance_info else None,
                "health_monitoring": gpu_provider.get_gpu_health_checks()
            }
        except ValueError as e:
            self.logger.error(f"Failed to get GPU instance: {e}")
            raise
    
    def get_cost_comparison(self, gpu_type: GPUType, count: int = 1) -> Dict[str, float]:
        """Compare costs across providers for given GPU configuration"""
        costs = {}
        
        for provider_name, provider in self.providers.items():
            try:
                if provider.validate_gpu_availability(gpu_type):
                    instance = provider.gpu_instances.get(gpu_type)
                    if instance:
                        # Calculate cost based on count
                        base_cost = instance.cost_per_hour
                        total_cost = base_cost * count
                        costs[provider_name] = total_cost
            except Exception as e:
                self.logger.warning(f"Could not get cost for {provider_name}: {e}")
        
        return costs
    
    def get_gpu_recommendations(self, workload_type: str) -> Dict[str, Any]:
        """Get GPU recommendations based on workload type"""
        recommendations = {
            "inference_small": {
                "gpu_type": GPUType.NVIDIA_T4,
                "count": 1,
                "reason": "Cost-effective for small model inference"
            },
            "inference_large": {
                "gpu_type": GPUType.NVIDIA_A10G,
                "count": 1,
                "reason": "Good balance of memory and compute for large models"
            },
            "training_small": {
                "gpu_type": GPUType.NVIDIA_V100,
                "count": 1,
                "reason": "Proven performance for model training"
            },
            "training_large": {
                "gpu_type": GPUType.NVIDIA_A100,
                "count": 4,
                "reason": "Best performance for large model training"
            },
            "batch_processing": {
                "gpu_type": GPUType.NVIDIA_T4,
                "count": 4,
                "reason": "Cost-effective for parallel batch processing"
            }
        }
        
        return recommendations.get(workload_type, recommendations["inference_small"])


# Example usage
if __name__ == "__main__":
    # Initialize GPU Resource Manager
    manager = GPUResourceManager()
    
    # Register providers
    manager.register_provider("aws", "us-east-1")
    manager.register_provider("gcp", "us-central1")
    manager.register_provider("azure", "eastus")
    
    # Get GPU instance for inference
    instance = manager.get_gpu_instance("aws", GPUType.NVIDIA_T4, count=1)
    print(f"Recommended instance: {instance}")
    
    # Compare costs
    costs = manager.get_cost_comparison(GPUType.NVIDIA_V100, count=2)
    print(f"Cost comparison: {costs}")
    
    # Get workload recommendations
    recommendation = manager.get_gpu_recommendations("inference_large")
    print(f"Recommendation: {recommendation}")