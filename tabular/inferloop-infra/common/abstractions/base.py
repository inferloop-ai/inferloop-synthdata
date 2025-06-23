"""
Base classes for all infrastructure abstractions
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


class ResourceState(Enum):
    """Common resource states across all providers"""
    PENDING = "pending"
    CREATING = "creating"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    DELETING = "deleting"
    DELETED = "deleted"
    ERROR = "error"
    UNKNOWN = "unknown"


class ResourceType(Enum):
    """Types of infrastructure resources"""
    COMPUTE = "compute"
    CONTAINER = "container"
    SERVERLESS = "serverless"
    STORAGE = "storage"
    NETWORK = "network"
    LOADBALANCER = "loadbalancer"
    DATABASE = "database"
    CACHE = "cache"
    SECURITY = "security"
    MONITORING = "monitoring"


@dataclass
class ResourceMetadata:
    """Common metadata for all resources"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    provider: str = ""
    region: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    state: ResourceState = ResourceState.UNKNOWN
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "region": self.region,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "state": self.state.value
        }


@dataclass
class ResourceConfig:
    """Base configuration for all resources"""
    name: str
    region: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        if not self.name:
            errors.append("Resource name is required")
        if not self.region:
            errors.append("Region is required")
        return errors


T = TypeVar('T', bound=ResourceConfig)
R = TypeVar('R')


class BaseResource(ABC, Generic[T, R]):
    """Base class for all infrastructure resources"""
    
    def __init__(self, provider: str):
        self.provider = provider
        self.metadata = ResourceMetadata(provider=provider)
    
    @abstractmethod
    async def create(self, config: T) -> R:
        """Create the resource"""
        pass
    
    @abstractmethod
    async def delete(self, resource_id: str) -> bool:
        """Delete the resource"""
        pass
    
    @abstractmethod
    async def get(self, resource_id: str) -> Optional[R]:
        """Get resource details"""
        pass
    
    @abstractmethod
    async def list(self, filters: Optional[Dict[str, Any]] = None) -> List[R]:
        """List resources with optional filters"""
        pass
    
    @abstractmethod
    async def update(self, resource_id: str, config: T) -> R:
        """Update the resource"""
        pass
    
    @abstractmethod
    async def get_state(self, resource_id: str) -> ResourceState:
        """Get current resource state"""
        pass
    
    @abstractmethod
    async def wait_for_state(self, resource_id: str, target_state: ResourceState, timeout: int = 300) -> bool:
        """Wait for resource to reach target state"""
        pass
    
    @abstractmethod
    def estimate_cost(self, config: T) -> Dict[str, float]:
        """Estimate cost for the resource configuration"""
        pass
    
    def validate_config(self, config: T) -> List[str]:
        """Validate resource configuration"""
        return config.validate()
    
    def get_resource_type(self) -> ResourceType:
        """Get the resource type"""
        return ResourceType.COMPUTE  # Override in subclasses


class BaseProvider(ABC):
    """Base class for all cloud/infrastructure providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.get_provider_name()
        
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name"""
        pass
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the provider"""
        pass
    
    @abstractmethod
    async def validate_credentials(self) -> bool:
        """Validate provider credentials"""
        pass
    
    @abstractmethod
    def get_regions(self) -> List[str]:
        """Get available regions"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get provider capabilities"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform provider health check"""
        pass