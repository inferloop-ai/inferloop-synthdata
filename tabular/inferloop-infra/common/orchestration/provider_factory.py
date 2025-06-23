"""
Factory for creating and managing cloud providers
"""

from typing import Dict, Any, Type, Optional, List
import importlib
import logging
from dataclasses import dataclass, field

from ..abstractions.base import BaseProvider


logger = logging.getLogger(__name__)


@dataclass
class ProviderInfo:
    """Information about a registered provider"""
    name: str
    class_path: str
    description: str
    supported_regions: List[str] = field(default_factory=list)
    capabilities: Dict[str, List[str]] = field(default_factory=dict)
    required_config: List[str] = field(default_factory=list)


class ProviderRegistry:
    """Registry for cloud providers"""
    
    def __init__(self):
        self._providers: Dict[str, ProviderInfo] = {}
        self._provider_instances: Dict[str, BaseProvider] = {}
        self._register_default_providers()
    
    def _register_default_providers(self):
        """Register default cloud providers"""
        # AWS Provider
        self.register(
            ProviderInfo(
                name="aws",
                class_path="inferloop_infra.providers.aws.provider.AWSProvider",
                description="Amazon Web Services",
                supported_regions=[
                    "us-east-1", "us-east-2", "us-west-1", "us-west-2",
                    "eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1",
                    "ap-southeast-1", "ap-southeast-2", "ap-northeast-1",
                    "ap-northeast-2", "ap-south-1", "sa-east-1"
                ],
                capabilities={
                    "compute": ["ec2", "ecs", "lambda", "eks"],
                    "storage": ["s3", "ebs", "efs"],
                    "database": ["rds", "dynamodb", "elasticache"],
                    "networking": ["vpc", "elb", "cloudfront"],
                    "security": ["iam", "kms", "secrets-manager"]
                },
                required_config=["access_key_id", "secret_access_key", "region"]
            )
        )
        
        # GCP Provider
        self.register(
            ProviderInfo(
                name="gcp",
                class_path="inferloop_infra.providers.gcp.provider.GCPProvider",
                description="Google Cloud Platform",
                supported_regions=[
                    "us-central1", "us-east1", "us-east4", "us-west1",
                    "us-west2", "us-west3", "us-west4", "europe-west1",
                    "europe-west2", "europe-west3", "europe-west4",
                    "asia-east1", "asia-east2", "asia-northeast1"
                ],
                capabilities={
                    "compute": ["compute-engine", "cloud-run", "cloud-functions", "gke"],
                    "storage": ["cloud-storage", "persistent-disk", "filestore"],
                    "database": ["cloud-sql", "firestore", "memorystore"],
                    "networking": ["vpc", "load-balancing", "cdn"],
                    "security": ["iam", "kms", "secret-manager"]
                },
                required_config=["project_id", "credentials_path", "region"]
            )
        )
        
        # Azure Provider
        self.register(
            ProviderInfo(
                name="azure",
                class_path="inferloop_infra.providers.azure.provider.AzureProvider",
                description="Microsoft Azure",
                supported_regions=[
                    "eastus", "eastus2", "westus", "westus2", "westus3",
                    "centralus", "northeurope", "westeurope", "uksouth",
                    "ukwest", "eastasia", "southeastasia", "japaneast",
                    "japanwest", "australiaeast", "australiasoutheast"
                ],
                capabilities={
                    "compute": ["virtual-machines", "container-instances", "functions", "aks"],
                    "storage": ["blob-storage", "managed-disks", "files"],
                    "database": ["sql-database", "cosmos-db", "cache-for-redis"],
                    "networking": ["virtual-network", "load-balancer", "cdn"],
                    "security": ["active-directory", "key-vault", "managed-identity"]
                },
                required_config=["subscription_id", "tenant_id", "client_id", "client_secret", "region"]
            )
        )
        
        # On-Premise Provider
        self.register(
            ProviderInfo(
                name="onprem",
                class_path="inferloop_infra.providers.onprem.provider.OnPremProvider",
                description="On-Premise Infrastructure",
                supported_regions=["default"],
                capabilities={
                    "compute": ["kubernetes", "docker", "openshift"],
                    "storage": ["nfs", "minio", "ceph"],
                    "database": ["postgresql", "mysql", "mongodb", "redis"],
                    "networking": ["haproxy", "nginx", "istio"],
                    "security": ["ldap", "vault", "cert-manager"]
                },
                required_config=["kubernetes_config", "storage_endpoint"]
            )
        )
    
    def register(self, provider_info: ProviderInfo) -> None:
        """Register a provider"""
        self._providers[provider_info.name] = provider_info
        logger.info(f"Registered provider: {provider_info.name}")
    
    def get_provider_info(self, name: str) -> Optional[ProviderInfo]:
        """Get provider information"""
        return self._providers.get(name)
    
    def list_providers(self) -> List[str]:
        """List registered provider names"""
        return list(self._providers.keys())
    
    def get_provider_capabilities(self, name: str) -> Dict[str, List[str]]:
        """Get provider capabilities"""
        info = self.get_provider_info(name)
        return info.capabilities if info else {}
    
    def validate_provider_config(self, name: str, config: Dict[str, Any]) -> List[str]:
        """Validate provider configuration"""
        info = self.get_provider_info(name)
        if not info:
            return [f"Unknown provider: {name}"]
        
        errors = []
        for required in info.required_config:
            if required not in config:
                errors.append(f"Missing required configuration: {required}")
        
        if 'region' in config and config['region'] not in info.supported_regions:
            errors.append(f"Unsupported region: {config['region']}")
        
        return errors


class ProviderFactory:
    """Factory for creating provider instances"""
    
    def __init__(self):
        self.registry = ProviderRegistry()
        self._provider_cache: Dict[str, BaseProvider] = {}
    
    async def get_provider(self, name: str, config: Dict[str, Any]) -> BaseProvider:
        """Get or create provider instance"""
        # Validate configuration
        errors = self.registry.validate_provider_config(name, config)
        if errors:
            raise ValueError(f"Invalid provider configuration: {errors}")
        
        # Check cache
        cache_key = f"{name}:{config.get('region', 'default')}"
        if cache_key in self._provider_cache:
            return self._provider_cache[cache_key]
        
        # Create new instance
        provider = await self._create_provider(name, config)
        
        # Cache instance
        self._provider_cache[cache_key] = provider
        
        return provider
    
    async def _create_provider(self, name: str, config: Dict[str, Any]) -> BaseProvider:
        """Create provider instance"""
        info = self.registry.get_provider_info(name)
        if not info:
            raise ValueError(f"Unknown provider: {name}")
        
        # Dynamic import
        module_path, class_name = info.class_path.rsplit('.', 1)
        try:
            module = importlib.import_module(module_path)
            provider_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import provider {name}: {str(e)}")
        
        # Create instance
        provider = provider_class(config)
        
        # Authenticate
        if not await provider.authenticate():
            raise RuntimeError(f"Failed to authenticate with provider {name}")
        
        # Validate credentials
        if not await provider.validate_credentials():
            raise RuntimeError(f"Invalid credentials for provider {name}")
        
        # Health check
        health = await provider.health_check()
        if not health.get('healthy', False):
            raise RuntimeError(f"Provider {name} health check failed: {health}")
        
        logger.info(f"Successfully created provider instance: {name}")
        return provider
    
    def list_providers(self) -> List[str]:
        """List available providers"""
        return self.registry.list_providers()
    
    def get_provider_info(self, name: str) -> Optional[ProviderInfo]:
        """Get provider information"""
        return self.registry.get_provider_info(name)
    
    def register_custom_provider(self, provider_info: ProviderInfo) -> None:
        """Register a custom provider"""
        self.registry.register(provider_info)
    
    async def test_provider_connection(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test provider connection and return diagnostics"""
        try:
            provider = await self.get_provider(name, config)
            health = await provider.health_check()
            
            return {
                'success': True,
                'provider': name,
                'region': config.get('region', 'default'),
                'health': health,
                'capabilities': provider.get_capabilities()
            }
        except Exception as e:
            return {
                'success': False,
                'provider': name,
                'error': str(e)
            }
    
    def clear_cache(self, provider: Optional[str] = None) -> None:
        """Clear provider cache"""
        if provider:
            keys_to_remove = [k for k in self._provider_cache if k.startswith(f"{provider}:")]
            for key in keys_to_remove:
                del self._provider_cache[key]
        else:
            self._provider_cache.clear()
        
        logger.info(f"Cleared provider cache for: {provider or 'all providers'}")