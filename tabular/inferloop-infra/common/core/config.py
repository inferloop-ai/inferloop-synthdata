"""Common infrastructure configuration management."""

from pydantic import BaseSettings, Field, validator
from typing import Dict, List, Optional, Any
from enum import Enum
import os
from pathlib import Path


class Environment(str, Enum):
    """Deployment environment types."""
    
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"


class CloudProvider(str, Enum):
    """Supported cloud providers."""
    
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    ONPREM = "onprem"


class InstanceSize(str, Enum):
    """Standard instance sizes across providers."""
    
    SMALL = "small"     # 2 vCPU, 4GB RAM
    MEDIUM = "medium"   # 4 vCPU, 8GB RAM
    LARGE = "large"     # 8 vCPU, 16GB RAM
    XLARGE = "xlarge"   # 16 vCPU, 32GB RAM
    XXLARGE = "xxlarge" # 32 vCPU, 64GB RAM


class InfrastructureConfig(BaseSettings):
    """Common infrastructure configuration."""
    
    # General settings
    provider: CloudProvider = Field(..., env="INFRA_PROVIDER")
    environment: Environment = Field(Environment.DEVELOPMENT, env="INFRA_ENV")
    region: str = Field(..., env="INFRA_REGION")
    availability_zones: List[str] = Field(default_factory=list, env="INFRA_AZS")
    project_name: str = Field("inferloop-synthdata", env="PROJECT_NAME")
    
    # Resource naming
    resource_prefix: str = Field("", env="RESOURCE_PREFIX")
    resource_suffix: str = Field("", env="RESOURCE_SUFFIX")
    
    # Compute settings
    compute_instance_size: InstanceSize = Field(
        InstanceSize.MEDIUM, env="COMPUTE_INSTANCE_SIZE"
    )
    compute_instance_type: Optional[str] = Field(None, env="COMPUTE_INSTANCE_TYPE")
    min_instances: int = Field(1, env="MIN_INSTANCES", ge=0)
    max_instances: int = Field(10, env="MAX_INSTANCES", ge=1)
    enable_spot_instances: bool = Field(False, env="ENABLE_SPOT_INSTANCES")
    spot_max_price: Optional[float] = Field(None, env="SPOT_MAX_PRICE")
    
    # Container settings
    container_registry: Optional[str] = Field(None, env="CONTAINER_REGISTRY")
    container_image: str = Field(
        "inferloop/synthdata:latest", env="CONTAINER_IMAGE"
    )
    container_cpu: str = Field("1", env="CONTAINER_CPU")
    container_memory: str = Field("2Gi", env="CONTAINER_MEMORY")
    
    # Storage settings
    storage_encryption: bool = Field(True, env="STORAGE_ENCRYPTION")
    storage_retention_days: int = Field(30, env="STORAGE_RETENTION_DAYS", ge=1)
    storage_class: str = Field("standard", env="STORAGE_CLASS")
    enable_versioning: bool = Field(False, env="ENABLE_VERSIONING")
    enable_lifecycle_rules: bool = Field(True, env="ENABLE_LIFECYCLE_RULES")
    
    # Network settings
    vpc_cidr: str = Field("10.0.0.0/16", env="VPC_CIDR")
    enable_vpc: bool = Field(True, env="ENABLE_VPC")
    enable_private_subnets: bool = Field(True, env="ENABLE_PRIVATE_SUBNETS")
    enable_nat_gateway: bool = Field(True, env="ENABLE_NAT_GATEWAY")
    enable_vpc_endpoints: bool = Field(True, env="ENABLE_VPC_ENDPOINTS")
    
    # Security settings
    enable_firewall: bool = Field(True, env="ENABLE_FIREWALL")
    allowed_ip_ranges: List[str] = Field(
        default_factory=lambda: ["0.0.0.0/0"], env="ALLOWED_IP_RANGES"
    )
    enable_waf: bool = Field(False, env="ENABLE_WAF")
    enable_ddos_protection: bool = Field(False, env="ENABLE_DDOS_PROTECTION")
    ssl_certificate_arn: Optional[str] = Field(None, env="SSL_CERTIFICATE_ARN")
    
    # Database settings
    enable_database: bool = Field(False, env="ENABLE_DATABASE")
    database_engine: str = Field("postgres", env="DATABASE_ENGINE")
    database_version: str = Field("14", env="DATABASE_VERSION")
    database_instance_class: str = Field("small", env="DATABASE_INSTANCE_CLASS")
    database_storage_gb: int = Field(100, env="DATABASE_STORAGE_GB", ge=20)
    database_backup_retention_days: int = Field(7, env="DATABASE_BACKUP_RETENTION", ge=1)
    
    # Monitoring settings
    enable_monitoring: bool = Field(True, env="ENABLE_MONITORING")
    log_retention_days: int = Field(7, env="LOG_RETENTION_DAYS", ge=1)
    enable_detailed_monitoring: bool = Field(False, env="ENABLE_DETAILED_MONITORING")
    enable_tracing: bool = Field(True, env="ENABLE_TRACING")
    enable_profiling: bool = Field(False, env="ENABLE_PROFILING")
    alert_email: Optional[str] = Field(None, env="ALERT_EMAIL")
    
    # Cost optimization
    enable_auto_shutdown: bool = Field(False, env="ENABLE_AUTO_SHUTDOWN")
    auto_shutdown_schedule: str = Field("0 20 * * *", env="AUTO_SHUTDOWN_SCHEDULE")
    enable_cost_alerts: bool = Field(True, env="ENABLE_COST_ALERTS")
    monthly_budget_usd: float = Field(1000.0, env="MONTHLY_BUDGET_USD", gt=0)
    
    # Backup settings
    enable_backups: bool = Field(True, env="ENABLE_BACKUPS")
    backup_schedule: str = Field("0 2 * * *", env="BACKUP_SCHEDULE")
    backup_retention_days: int = Field(7, env="BACKUP_RETENTION_DAYS", ge=1)
    
    # Tags
    default_tags: Dict[str, str] = Field(default_factory=dict, env="DEFAULT_TAGS")
    
    class Config:
        """Pydantic configuration."""
        
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        @staticmethod
        def json_loads(v):
            """Custom JSON loader for list/dict fields from env vars."""
            import json
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    # Try comma-separated for lists
                    if "," in v:
                        return [item.strip() for item in v.split(",")]
            return v
    
    @validator("availability_zones", "allowed_ip_ranges", pre=True)
    def parse_list_from_string(cls, v):
        """Parse list from comma-separated string."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v
    
    @validator("default_tags", pre=True)
    def parse_dict_from_string(cls, v):
        """Parse dict from JSON string or key=value pairs."""
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # Try key=value format
                result = {}
                for pair in v.split(","):
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        result[key.strip()] = value.strip()
                return result
        return v
    
    @validator("resource_prefix", "resource_suffix")
    def validate_naming(cls, v, values):
        """Validate resource naming conventions."""
        if v and not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Resource prefix/suffix must be alphanumeric with - or _")
        return v
    
    @property
    def resource_name(self) -> str:
        """Generate a base resource name."""
        parts = []
        if self.resource_prefix:
            parts.append(self.resource_prefix)
        parts.append(self.project_name)
        if self.resource_suffix:
            parts.append(self.resource_suffix)
        parts.append(self.environment.value)
        return "-".join(parts)
    
    def get_instance_type_mapping(self) -> Dict[CloudProvider, Dict[InstanceSize, str]]:
        """Get instance type mappings for each provider."""
        return {
            CloudProvider.AWS: {
                InstanceSize.SMALL: "t3.medium",
                InstanceSize.MEDIUM: "t3.large",
                InstanceSize.LARGE: "t3.xlarge",
                InstanceSize.XLARGE: "t3.2xlarge",
                InstanceSize.XXLARGE: "m5.4xlarge",
            },
            CloudProvider.GCP: {
                InstanceSize.SMALL: "n2-standard-2",
                InstanceSize.MEDIUM: "n2-standard-4",
                InstanceSize.LARGE: "n2-standard-8",
                InstanceSize.XLARGE: "n2-standard-16",
                InstanceSize.XXLARGE: "n2-standard-32",
            },
            CloudProvider.AZURE: {
                InstanceSize.SMALL: "Standard_B2s",
                InstanceSize.MEDIUM: "Standard_B4ms",
                InstanceSize.LARGE: "Standard_D8s_v3",
                InstanceSize.XLARGE: "Standard_D16s_v3",
                InstanceSize.XXLARGE: "Standard_D32s_v3",
            },
        }
    
    def get_provider_instance_type(self) -> str:
        """Get the provider-specific instance type."""
        if self.compute_instance_type:
            return self.compute_instance_type
        
        mapping = self.get_instance_type_mapping()
        if self.provider in mapping:
            return mapping[self.provider].get(
                self.compute_instance_size, 
                mapping[self.provider][InstanceSize.MEDIUM]
            )
        return "medium"
    
    def to_provider_config(self) -> Dict[str, Any]:
        """Convert to provider-specific configuration."""
        config = self.dict()
        config["instance_type"] = self.get_provider_instance_type()
        config["tags"] = {
            **self.default_tags,
            "Environment": self.environment.value,
            "Project": self.project_name,
            "Provider": self.provider.value,
        }
        return config