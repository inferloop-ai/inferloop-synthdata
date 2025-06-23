"""
Common abstractions for cloud and on-premise infrastructure
"""

from .compute import (
    BaseCompute,
    BaseContainer,
    BaseServerless,
    ComputeResource,
    ContainerConfig,
    ServerlessConfig
)
from .storage import (
    BaseStorage,
    BaseObjectStorage,
    BaseFileStorage,
    BaseBlockStorage,
    StorageResource,
    StorageConfig
)
from .networking import (
    BaseNetwork,
    BaseLoadBalancer,
    BaseFirewall,
    NetworkResource,
    NetworkConfig,
    LoadBalancerConfig
)
from .security import (
    BaseIAM,
    BaseSecrets,
    BaseCertificates,
    SecurityResource,
    IAMConfig,
    SecretConfig
)
from .monitoring import (
    BaseMonitoring,
    BaseLogging,
    BaseMetrics,
    MonitoringResource,
    MonitoringConfig
)
from .database import (
    BaseDatabase,
    BaseCache,
    DatabaseResource,
    DatabaseConfig
)

__all__ = [
    # Compute
    'BaseCompute',
    'BaseContainer',
    'BaseServerless',
    'ComputeResource',
    'ContainerConfig',
    'ServerlessConfig',
    
    # Storage
    'BaseStorage',
    'BaseObjectStorage',
    'BaseFileStorage',
    'BaseBlockStorage',
    'StorageResource',
    'StorageConfig',
    
    # Networking
    'BaseNetwork',
    'BaseLoadBalancer',
    'BaseFirewall',
    'NetworkResource',
    'NetworkConfig',
    'LoadBalancerConfig',
    
    # Security
    'BaseIAM',
    'BaseSecrets',
    'BaseCertificates',
    'SecurityResource',
    'IAMConfig',
    'SecretConfig',
    
    # Monitoring
    'BaseMonitoring',
    'BaseLogging',
    'BaseMetrics',
    'MonitoringResource',
    'MonitoringConfig',
    
    # Database
    'BaseDatabase',
    'BaseCache',
    'DatabaseResource',
    'DatabaseConfig'
]