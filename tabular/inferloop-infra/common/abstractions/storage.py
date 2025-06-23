"""
Storage resource abstractions for object, file, and block storage
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, List, IO, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import asyncio

from .base import BaseResource, ResourceConfig, ResourceType


class StorageType(Enum):
    """Types of storage resources"""
    OBJECT = "object"
    FILE = "file"
    BLOCK = "block"
    ARCHIVE = "archive"


class StorageClass(Enum):
    """Storage classes for cost optimization"""
    STANDARD = "standard"
    INFREQUENT_ACCESS = "infrequent_access"
    ARCHIVE = "archive"
    GLACIER = "glacier"
    COLD = "cold"


@dataclass
class StorageConfig(ResourceConfig):
    """Base configuration for storage resources"""
    storage_type: StorageType = StorageType.OBJECT
    storage_class: StorageClass = StorageClass.STANDARD
    encryption: bool = True
    versioning: bool = False
    lifecycle_rules: List[Dict[str, Any]] = field(default_factory=list)
    access_control: Dict[str, Any] = field(default_factory=dict)
    replication: Optional[Dict[str, Any]] = None
    
    def validate(self) -> List[str]:
        errors = super().validate()
        return errors


@dataclass
class ObjectStorageConfig(StorageConfig):
    """Configuration for object storage (S3-like)"""
    bucket_name: str = ""
    cors_rules: List[Dict[str, Any]] = field(default_factory=list)
    website_config: Optional[Dict[str, Any]] = None
    notification_config: Optional[Dict[str, Any]] = None
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if not self.bucket_name:
            errors.append("Bucket name is required")
        return errors


@dataclass
class FileStorageConfig(StorageConfig):
    """Configuration for file storage (NFS-like)"""
    size_gb: int = 100
    performance_mode: str = "general_purpose"
    throughput_mode: str = "bursting"
    mount_targets: List[Dict[str, Any]] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if self.size_gb < 1:
            errors.append("Storage size must be at least 1 GB")
        return errors


@dataclass
class BlockStorageConfig(StorageConfig):
    """Configuration for block storage (EBS-like)"""
    size_gb: int = 30
    iops: Optional[int] = None
    volume_type: str = "gp3"
    multi_attach: bool = False
    availability_zone: Optional[str] = None
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if self.size_gb < 1:
            errors.append("Volume size must be at least 1 GB")
        return errors


@dataclass
class StorageResource:
    """Representation of a storage resource"""
    id: str
    name: str
    type: StorageType
    state: str
    config: StorageConfig
    endpoint: Optional[str] = None
    size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageObject:
    """Representation of an object in storage"""
    key: str
    size: int
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    storage_class: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


class BaseStorage(BaseResource[StorageConfig, StorageResource]):
    """Base class for all storage types"""
    
    @abstractmethod
    async def get_usage(self, resource_id: str) -> Dict[str, Any]:
        """Get storage usage statistics"""
        pass
    
    @abstractmethod
    async def set_lifecycle_policy(self, resource_id: str, rules: List[Dict[str, Any]]) -> bool:
        """Set lifecycle policy for storage"""
        pass
    
    @abstractmethod
    async def set_access_policy(self, resource_id: str, policy: Dict[str, Any]) -> bool:
        """Set access policy for storage"""
        pass


class BaseObjectStorage(BaseStorage):
    """Base class for object storage (S3-like)"""
    
    def get_resource_type(self) -> ResourceType:
        return ResourceType.STORAGE
    
    @abstractmethod
    async def put_object(self, bucket: str, key: str, data: IO[bytes], metadata: Optional[Dict[str, str]] = None) -> bool:
        """Upload an object to storage"""
        pass
    
    @abstractmethod
    async def get_object(self, bucket: str, key: str) -> IO[bytes]:
        """Download an object from storage"""
        pass
    
    @abstractmethod
    async def delete_object(self, bucket: str, key: str) -> bool:
        """Delete an object from storage"""
        pass
    
    @abstractmethod
    async def list_objects(self, bucket: str, prefix: Optional[str] = None, max_keys: int = 1000) -> List[StorageObject]:
        """List objects in bucket"""
        pass
    
    @abstractmethod
    async def copy_object(self, source_bucket: str, source_key: str, dest_bucket: str, dest_key: str) -> bool:
        """Copy object within or between buckets"""
        pass
    
    @abstractmethod
    async def generate_presigned_url(self, bucket: str, key: str, expiration: int = 3600, method: str = "GET") -> str:
        """Generate presigned URL for object access"""
        pass
    
    @abstractmethod
    async def multipart_upload(self, bucket: str, key: str, data: AsyncIterator[bytes], part_size: int = 5 * 1024 * 1024) -> bool:
        """Upload large objects using multipart upload"""
        pass
    
    @abstractmethod
    async def enable_versioning(self, bucket: str) -> bool:
        """Enable versioning for bucket"""
        pass
    
    @abstractmethod
    async def set_cors(self, bucket: str, cors_rules: List[Dict[str, Any]]) -> bool:
        """Set CORS rules for bucket"""
        pass


class BaseFileStorage(BaseStorage):
    """Base class for file storage (NFS-like)"""
    
    def get_resource_type(self) -> ResourceType:
        return ResourceType.STORAGE
    
    @abstractmethod
    async def create_mount_target(self, resource_id: str, subnet_id: str, security_groups: List[str]) -> str:
        """Create mount target for file system"""
        pass
    
    @abstractmethod
    async def delete_mount_target(self, mount_target_id: str) -> bool:
        """Delete mount target"""
        pass
    
    @abstractmethod
    async def get_mount_instructions(self, resource_id: str) -> Dict[str, str]:
        """Get instructions for mounting file system"""
        pass
    
    @abstractmethod
    async def create_snapshot(self, resource_id: str, description: str) -> str:
        """Create snapshot of file system"""
        pass
    
    @abstractmethod
    async def restore_snapshot(self, snapshot_id: str, resource_id: str) -> bool:
        """Restore file system from snapshot"""
        pass


class BaseBlockStorage(BaseStorage):
    """Base class for block storage (EBS-like)"""
    
    def get_resource_type(self) -> ResourceType:
        return ResourceType.STORAGE
    
    @abstractmethod
    async def attach_to_instance(self, volume_id: str, instance_id: str, device: str) -> bool:
        """Attach volume to compute instance"""
        pass
    
    @abstractmethod
    async def detach_from_instance(self, volume_id: str) -> bool:
        """Detach volume from compute instance"""
        pass
    
    @abstractmethod
    async def create_snapshot(self, volume_id: str, description: str) -> str:
        """Create snapshot of volume"""
        pass
    
    @abstractmethod
    async def create_from_snapshot(self, snapshot_id: str, config: BlockStorageConfig) -> str:
        """Create volume from snapshot"""
        pass
    
    @abstractmethod
    async def resize(self, volume_id: str, new_size_gb: int) -> bool:
        """Resize volume"""
        pass
    
    @abstractmethod
    async def modify_type(self, volume_id: str, volume_type: str, iops: Optional[int] = None) -> bool:
        """Modify volume type and performance"""
        pass