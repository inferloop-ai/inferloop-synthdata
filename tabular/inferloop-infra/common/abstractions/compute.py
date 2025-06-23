"""
Compute resource abstractions for VMs, containers, and serverless
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum

from .base import BaseResource, ResourceConfig, ResourceType


class ComputeType(Enum):
    """Types of compute resources"""
    VM = "vm"
    CONTAINER = "container"
    SERVERLESS = "serverless"
    KUBERNETES = "kubernetes"


class ContainerRuntime(Enum):
    """Container runtime engines"""
    DOCKER = "docker"
    CONTAINERD = "containerd"
    CRIO = "cri-o"
    PODMAN = "podman"


@dataclass
class ComputeConfig(ResourceConfig):
    """Base configuration for compute resources"""
    compute_type: ComputeType = ComputeType.VM
    cpu: int = 2
    memory: int = 4096  # MB
    disk_size: int = 30  # GB
    network_config: Dict[str, Any] = field(default_factory=dict)
    security_groups: List[str] = field(default_factory=list)
    ssh_key: Optional[str] = None
    user_data: Optional[str] = None
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if self.cpu < 1:
            errors.append("CPU count must be at least 1")
        if self.memory < 512:
            errors.append("Memory must be at least 512 MB")
        if self.disk_size < 10:
            errors.append("Disk size must be at least 10 GB")
        return errors


@dataclass
class ContainerConfig(ComputeConfig):
    """Configuration for container-based compute"""
    image: str = ""
    runtime: ContainerRuntime = ContainerRuntime.DOCKER
    environment: Dict[str, str] = field(default_factory=dict)
    ports: List[Dict[str, Any]] = field(default_factory=list)
    volumes: List[Dict[str, str]] = field(default_factory=list)
    command: Optional[List[str]] = None
    entrypoint: Optional[List[str]] = None
    replicas: int = 1
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if not self.image:
            errors.append("Container image is required")
        if self.replicas < 1:
            errors.append("Replicas must be at least 1")
        return errors


@dataclass
class ServerlessConfig(ComputeConfig):
    """Configuration for serverless compute"""
    runtime: str = "python3.9"
    handler: str = ""
    code_uri: str = ""
    timeout: int = 300  # seconds
    memory: int = 512  # MB
    environment: Dict[str, str] = field(default_factory=dict)
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    layers: List[str] = field(default_factory=list)
    vpc_config: Optional[Dict[str, Any]] = None
    
    def validate(self) -> List[str]:
        errors = []
        if not self.handler:
            errors.append("Handler function is required")
        if not self.code_uri:
            errors.append("Code URI is required")
        if self.timeout < 1:
            errors.append("Timeout must be at least 1 second")
        if self.memory < 128:
            errors.append("Memory must be at least 128 MB")
        return errors


@dataclass
class ComputeResource:
    """Representation of a compute resource"""
    id: str
    name: str
    type: ComputeType
    state: str
    config: Union[ComputeConfig, ContainerConfig, ServerlessConfig]
    ip_addresses: Dict[str, str] = field(default_factory=dict)
    dns_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseCompute(BaseResource[ComputeConfig, ComputeResource]):
    """Base class for compute resources (VMs)"""
    
    def get_resource_type(self) -> ResourceType:
        return ResourceType.COMPUTE
    
    @abstractmethod
    async def start(self, resource_id: str) -> bool:
        """Start a stopped compute resource"""
        pass
    
    @abstractmethod
    async def stop(self, resource_id: str) -> bool:
        """Stop a running compute resource"""
        pass
    
    @abstractmethod
    async def restart(self, resource_id: str) -> bool:
        """Restart a compute resource"""
        pass
    
    @abstractmethod
    async def resize(self, resource_id: str, cpu: int, memory: int) -> bool:
        """Resize compute resource"""
        pass
    
    @abstractmethod
    async def get_console_output(self, resource_id: str) -> str:
        """Get console output from compute resource"""
        pass
    
    @abstractmethod
    async def execute_command(self, resource_id: str, command: str) -> Dict[str, Any]:
        """Execute command on compute resource"""
        pass


class BaseContainer(BaseResource[ContainerConfig, ComputeResource]):
    """Base class for container-based compute"""
    
    def get_resource_type(self) -> ResourceType:
        return ResourceType.CONTAINER
    
    @abstractmethod
    async def scale(self, resource_id: str, replicas: int) -> bool:
        """Scale container replicas"""
        pass
    
    @abstractmethod
    async def update_image(self, resource_id: str, image: str) -> bool:
        """Update container image"""
        pass
    
    @abstractmethod
    async def get_logs(self, resource_id: str, tail: int = 100) -> List[str]:
        """Get container logs"""
        pass
    
    @abstractmethod
    async def exec(self, resource_id: str, command: List[str]) -> Dict[str, Any]:
        """Execute command in container"""
        pass
    
    @abstractmethod
    async def port_forward(self, resource_id: str, local_port: int, container_port: int) -> bool:
        """Set up port forwarding"""
        pass


class BaseServerless(BaseResource[ServerlessConfig, ComputeResource]):
    """Base class for serverless compute"""
    
    def get_resource_type(self) -> ResourceType:
        return ResourceType.SERVERLESS
    
    @abstractmethod
    async def invoke(self, resource_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke serverless function"""
        pass
    
    @abstractmethod
    async def update_code(self, resource_id: str, code_uri: str) -> bool:
        """Update function code"""
        pass
    
    @abstractmethod
    async def update_config(self, resource_id: str, config: ServerlessConfig) -> bool:
        """Update function configuration"""
        pass
    
    @abstractmethod
    async def get_logs(self, resource_id: str, start_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get function execution logs"""
        pass
    
    @abstractmethod
    async def add_trigger(self, resource_id: str, trigger: Dict[str, Any]) -> bool:
        """Add trigger to function"""
        pass
    
    @abstractmethod
    async def remove_trigger(self, resource_id: str, trigger_id: str) -> bool:
        """Remove trigger from function"""
        pass