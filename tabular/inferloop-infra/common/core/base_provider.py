"""Base infrastructure provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum


class ResourceStatus(Enum):
    """Resource status enumeration."""
    
    CREATING = "creating"
    RUNNING = "running"
    STOPPED = "stopped"
    DELETING = "deleting"
    DELETED = "deleted"
    ERROR = "error"


@dataclass
class ResourceConfig:
    """Configuration for infrastructure resources."""
    
    name: str
    resource_type: str
    region: str
    environment: str = "dev"
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Add default tags."""
        default_tags = {
            "Project": "inferloop-synthdata",
            "Environment": self.environment,
            "ManagedBy": "inferloop-infra",
            "CreatedAt": datetime.utcnow().isoformat(),
        }
        self.tags = {**default_tags, **self.tags}


@dataclass
class ResourceInfo:
    """Information about a deployed resource."""
    
    resource_id: str
    resource_type: str
    name: str
    status: ResourceStatus
    region: str
    created_at: datetime
    endpoint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComputeConfig(ResourceConfig):
    """Configuration for compute resources."""
    
    instance_type: str = "medium"
    cpu: Optional[int] = None
    memory_gb: Optional[int] = None
    disk_size_gb: int = 100
    min_instances: int = 1
    max_instances: int = 10
    enable_gpu: bool = False
    gpu_type: Optional[str] = None
    user_data: Optional[str] = None
    ssh_key_name: Optional[str] = None


@dataclass
class StorageConfig(ResourceConfig):
    """Configuration for storage resources."""
    
    storage_class: str = "standard"
    encryption_enabled: bool = True
    versioning_enabled: bool = False
    lifecycle_rules: List[Dict[str, Any]] = field(default_factory=list)
    access_control: str = "private"
    retention_days: Optional[int] = None


@dataclass
class NetworkConfig(ResourceConfig):
    """Configuration for network resources."""
    
    cidr_block: str = "10.0.0.0/16"
    enable_dns: bool = True
    enable_firewall: bool = True
    public_subnets: List[str] = field(default_factory=list)
    private_subnets: List[str] = field(default_factory=list)
    allowed_ports: List[int] = field(default_factory=lambda: [80, 443])
    allowed_ip_ranges: List[str] = field(default_factory=lambda: ["0.0.0.0/0"])


@dataclass
class ApplicationConfig:
    """Configuration for application deployment."""
    
    name: str
    version: str
    image: str
    port: int = 8000
    environment_variables: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)
    health_check_path: str = "/health"
    cpu_limit: str = "1"
    memory_limit: str = "2Gi"
    replicas: int = 2
    autoscaling_enabled: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70


class BaseInfrastructureProvider(ABC):
    """Abstract base class for infrastructure providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider with configuration."""
        self.config = config
        self.resources: Dict[str, ResourceInfo] = {}
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider connection and authentication."""
        pass
    
    @abstractmethod
    def create_compute_instance(self, config: ComputeConfig) -> ResourceInfo:
        """Create a compute instance."""
        pass
    
    @abstractmethod
    def create_storage_bucket(self, config: StorageConfig) -> ResourceInfo:
        """Create a storage bucket."""
        pass
    
    @abstractmethod
    def create_network(self, config: NetworkConfig) -> ResourceInfo:
        """Create a network with subnets."""
        pass
    
    @abstractmethod
    def deploy_application(self, app_config: ApplicationConfig) -> ResourceInfo:
        """Deploy the synthetic data application."""
        pass
    
    @abstractmethod
    def get_resource(self, resource_id: str) -> Optional[ResourceInfo]:
        """Get information about a specific resource."""
        pass
    
    @abstractmethod
    def list_resources(self, resource_type: Optional[str] = None) -> List[ResourceInfo]:
        """List all resources, optionally filtered by type."""
        pass
    
    @abstractmethod
    def delete_resource(self, resource_id: str) -> bool:
        """Delete a specific resource."""
        pass
    
    @abstractmethod
    def update_resource(self, resource_id: str, updates: Dict[str, Any]) -> ResourceInfo:
        """Update a resource configuration."""
        pass
    
    @abstractmethod
    def get_resource_metrics(
        self, resource_id: str, metric_names: List[str], start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Get metrics for a resource."""
        pass
    
    @abstractmethod
    def estimate_cost(self, resources: List[ResourceConfig]) -> Dict[str, float]:
        """Estimate the cost of resources."""
        pass
    
    def validate_config(self, config: ResourceConfig) -> bool:
        """Validate resource configuration."""
        # Basic validation - providers can override for specific checks
        if not config.name:
            raise ValueError("Resource name is required")
        if not config.region:
            raise ValueError("Region is required")
        return True
    
    def tag_resource(self, resource_id: str, tags: Dict[str, str]) -> None:
        """Add or update tags on a resource."""
        resource = self.get_resource(resource_id)
        if resource:
            resource.metadata["tags"] = {**resource.metadata.get("tags", {}), **tags}