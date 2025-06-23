"""
Database resource abstractions for relational and NoSQL databases
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum

from .base import BaseResource, ResourceConfig, ResourceType


class DatabaseEngine(Enum):
    """Database engines"""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    MARIADB = "mariadb"
    ORACLE = "oracle"
    SQLSERVER = "sqlserver"
    MONGODB = "mongodb"
    DYNAMODB = "dynamodb"
    CASSANDRA = "cassandra"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"


class DatabaseType(Enum):
    """Types of databases"""
    RELATIONAL = "relational"
    DOCUMENT = "document"
    KEY_VALUE = "key_value"
    WIDE_COLUMN = "wide_column"
    GRAPH = "graph"
    TIME_SERIES = "time_series"
    CACHE = "cache"


class BackupType(Enum):
    """Types of backups"""
    AUTOMATED = "automated"
    MANUAL = "manual"
    SNAPSHOT = "snapshot"
    CONTINUOUS = "continuous"


@dataclass
class DatabaseConfig(ResourceConfig):
    """Base configuration for database resources"""
    engine: DatabaseEngine = DatabaseEngine.POSTGRESQL
    engine_version: Optional[str] = None
    instance_class: str = "db.t3.micro"
    storage_type: str = "gp3"
    storage_size_gb: int = 20
    iops: Optional[int] = None
    username: str = "admin"
    password: Optional[str] = None
    port: Optional[int] = None
    vpc_id: Optional[str] = None
    subnet_group: Optional[str] = None
    security_groups: List[str] = field(default_factory=list)
    publicly_accessible: bool = False
    multi_az: bool = False
    backup_retention_days: int = 7
    backup_window: Optional[str] = None
    maintenance_window: Optional[str] = None
    encryption_enabled: bool = True
    kms_key_id: Optional[str] = None
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if self.storage_size_gb < 20:
            errors.append("Storage size must be at least 20 GB")
        if self.backup_retention_days < 0 or self.backup_retention_days > 35:
            errors.append("Backup retention must be between 0 and 35 days")
        if not self.username:
            errors.append("Database username is required")
        return errors


@dataclass
class NoSQLDatabaseConfig(DatabaseConfig):
    """Configuration for NoSQL databases"""
    consistency_level: str = "eventual"
    read_capacity_units: Optional[int] = None
    write_capacity_units: Optional[int] = None
    global_secondary_indexes: List[Dict[str, Any]] = field(default_factory=list)
    stream_enabled: bool = False
    ttl_attribute: Optional[str] = None
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if self.consistency_level not in ["eventual", "strong"]:
            errors.append("Consistency level must be 'eventual' or 'strong'")
        return errors


@dataclass
class CacheConfig(ResourceConfig):
    """Configuration for cache resources"""
    engine: str = "redis"
    engine_version: str = "7.0"
    node_type: str = "cache.t3.micro"
    num_nodes: int = 1
    parameter_group: Optional[str] = None
    subnet_group: Optional[str] = None
    security_groups: List[str] = field(default_factory=list)
    snapshot_retention_days: int = 5
    automatic_failover: bool = True
    multi_az: bool = False
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if self.num_nodes < 1:
            errors.append("Number of nodes must be at least 1")
        if self.engine not in ["redis", "memcached"]:
            errors.append("Cache engine must be 'redis' or 'memcached'")
        return errors


@dataclass
class DatabaseResource:
    """Representation of a database resource"""
    id: str
    name: str
    engine: DatabaseEngine
    type: DatabaseType
    state: str
    endpoint: Dict[str, Any]
    config: Union[DatabaseConfig, NoSQLDatabaseConfig, CacheConfig]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatabaseBackup:
    """Representation of a database backup"""
    id: str
    database_id: str
    type: BackupType
    created_at: str
    size_bytes: int
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDatabase(BaseResource[DatabaseConfig, DatabaseResource]):
    """Base class for database resources"""
    
    def get_resource_type(self) -> ResourceType:
        return ResourceType.DATABASE
    
    @abstractmethod
    async def create_database(self, db_name: str, charset: str = "utf8mb4") -> bool:
        """Create database within instance"""
        pass
    
    @abstractmethod
    async def create_user(self, username: str, password: str, privileges: List[str]) -> bool:
        """Create database user"""
        pass
    
    @abstractmethod
    async def grant_privileges(self, username: str, database: str, privileges: List[str]) -> bool:
        """Grant privileges to user"""
        pass
    
    @abstractmethod
    async def create_backup(self, resource_id: str, backup_name: str) -> str:
        """Create manual backup"""
        pass
    
    @abstractmethod
    async def restore_backup(self, backup_id: str, target_instance: str) -> bool:
        """Restore from backup"""
        pass
    
    @abstractmethod
    async def list_backups(self, resource_id: str) -> List[DatabaseBackup]:
        """List available backups"""
        pass
    
    @abstractmethod
    async def modify_instance(self, resource_id: str, config: DatabaseConfig) -> bool:
        """Modify database instance configuration"""
        pass
    
    @abstractmethod
    async def create_read_replica(self, source_id: str, replica_config: DatabaseConfig) -> str:
        """Create read replica"""
        pass
    
    @abstractmethod
    async def promote_read_replica(self, replica_id: str) -> bool:
        """Promote read replica to standalone"""
        pass
    
    @abstractmethod
    async def enable_logging(self, resource_id: str, log_types: List[str]) -> bool:
        """Enable database logging"""
        pass
    
    @abstractmethod
    async def get_metrics(self, resource_id: str, metric_name: str, 
                         period: int = 300) -> List[Dict[str, Any]]:
        """Get database metrics"""
        pass
    
    @abstractmethod
    async def create_parameter_group(self, name: str, family: str, 
                                   parameters: Dict[str, Any]) -> str:
        """Create parameter group"""
        pass
    
    @abstractmethod
    async def execute_query(self, resource_id: str, query: str, 
                          database: Optional[str] = None) -> List[Dict[str, Any]]:
        """Execute SQL query"""
        pass


class BaseCache(BaseResource[CacheConfig, DatabaseResource]):
    """Base class for cache resources"""
    
    def get_resource_type(self) -> ResourceType:
        return ResourceType.CACHE
    
    @abstractmethod
    async def flush_cache(self, resource_id: str) -> bool:
        """Flush all cache data"""
        pass
    
    @abstractmethod
    async def get_cache_metrics(self, resource_id: str) -> Dict[str, Any]:
        """Get cache performance metrics"""
        pass
    
    @abstractmethod
    async def create_snapshot(self, resource_id: str, snapshot_name: str) -> str:
        """Create cache snapshot"""
        pass
    
    @abstractmethod
    async def restore_snapshot(self, snapshot_id: str, config: CacheConfig) -> str:
        """Restore cache from snapshot"""
        pass
    
    @abstractmethod
    async def add_node(self, resource_id: str) -> bool:
        """Add node to cache cluster"""
        pass
    
    @abstractmethod
    async def remove_node(self, resource_id: str, node_id: str) -> bool:
        """Remove node from cache cluster"""
        pass
    
    @abstractmethod
    async def reboot_node(self, resource_id: str, node_id: str) -> bool:
        """Reboot cache node"""
        pass
    
    @abstractmethod
    async def modify_cache_cluster(self, resource_id: str, config: CacheConfig) -> bool:
        """Modify cache cluster configuration"""
        pass
    
    @abstractmethod
    async def set_parameter_group(self, resource_id: str, parameter_group: str) -> bool:
        """Apply parameter group to cache"""
        pass