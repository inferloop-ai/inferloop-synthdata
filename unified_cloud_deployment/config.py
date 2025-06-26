"""
Configuration Management Module

Provides unified configuration management for all Inferloop services.
Supports environment variables, configuration files, and service discovery.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging


@dataclass
class ServiceConfig:
    """Service configuration data class"""
    name: str
    version: str = "1.0.0"
    tier: str = "starter"
    environment: str = "development"
    debug: bool = False
    
    # Service-specific settings
    settings: Dict[str, Any] = field(default_factory=dict)
    
    # Infrastructure settings
    database: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)
    storage: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)
    auth: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Configuration manager for unified services"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.config = self._load_config()
    
    def _load_config(self) -> ServiceConfig:
        """Load configuration from multiple sources"""
        # Start with defaults
        config = ServiceConfig(name=self.service_name)
        
        # Load from environment variables
        self._load_from_env(config)
        
        # Load from configuration file
        config_file = self._find_config_file()
        if config_file:
            self._load_from_file(config, config_file)
        
        # Load service-specific overrides
        self._load_service_overrides(config)
        
        return config
    
    def _load_from_env(self, config: ServiceConfig):
        """Load configuration from environment variables"""
        # Service basics
        config.version = os.getenv('SERVICE_VERSION', config.version)
        config.tier = os.getenv('SERVICE_TIER', config.tier)
        config.environment = os.getenv('ENVIRONMENT', config.environment)
        config.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        
        # Database configuration
        config.database = {
            'provider': os.getenv('DATABASE_PROVIDER', 'postgresql'),
            'host': os.getenv('DATABASE_HOST', 'localhost'),
            'port': int(os.getenv('DATABASE_PORT', '5432')),
            'username': os.getenv('DATABASE_USERNAME', 'postgres'),
            'password': os.getenv('DATABASE_PASSWORD', 'password'),
            'database': os.getenv('DATABASE_NAME', 'inferloop'),
            'pool_size': int(os.getenv('DATABASE_POOL_SIZE', '10')),
            'echo': os.getenv('DATABASE_ECHO', 'false').lower() == 'true'
        }
        
        # Cache configuration
        config.cache = {
            'provider': os.getenv('CACHE_PROVIDER', 'redis'),
            'host': os.getenv('CACHE_HOST', 'localhost'),
            'port': int(os.getenv('CACHE_PORT', '6379')),
            'password': os.getenv('CACHE_PASSWORD'),
            'database': int(os.getenv('CACHE_DATABASE', '0')),
            'default_ttl': int(os.getenv('CACHE_DEFAULT_TTL', '3600'))
        }
        
        # Storage configuration
        config.storage = {
            'provider': os.getenv('STORAGE_PROVIDER', 's3'),
            'bucket_name': os.getenv('STORAGE_BUCKET_NAME', f'inferloop-{self.service_name}'),
            'region': os.getenv('STORAGE_REGION', 'us-east-1'),
            'endpoint_url': os.getenv('STORAGE_ENDPOINT_URL'),
            'access_key': os.getenv('STORAGE_ACCESS_KEY'),
            'secret_key': os.getenv('STORAGE_SECRET_KEY')
        }
        
        # Monitoring configuration
        config.monitoring = {
            'enable_metrics': os.getenv('ENABLE_METRICS', 'true').lower() == 'true',
            'enable_tracing': os.getenv('ENABLE_TRACING', 'true').lower() == 'true',
            'metrics_port': int(os.getenv('METRICS_PORT', '9090')),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'jaeger_endpoint': os.getenv('JAEGER_ENDPOINT', 'http://jaeger:14268/api/traces')
        }
        
        # Auth configuration
        config.auth = {
            'jwt_secret': os.getenv('JWT_SECRET', 'dev-secret-key'),
            'jwt_algorithm': os.getenv('JWT_ALGORITHM', 'HS256'),
            'jwt_expiration_hours': int(os.getenv('JWT_EXPIRATION_HOURS', '24')),
            'auth_service_url': os.getenv('AUTH_SERVICE_URL', 'http://auth-service:8000'),
            'enable_api_key_auth': os.getenv('ENABLE_API_KEY_AUTH', 'true').lower() == 'true'
        }
    
    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file"""
        possible_paths = [
            Path(f'config/{self.service_name}.yaml'),
            Path(f'config/{self.service_name}.yml'),
            Path(f'config/{self.service_name}.json'),
            Path(f'{self.service_name}.yaml'),
            Path(f'{self.service_name}.yml'),
            Path(f'{self.service_name}.json'),
            Path('config/default.yaml'),
            Path('config/default.yml'),
            Path('config/default.json')
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def _load_from_file(self, config: ServiceConfig, config_file: Path):
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                elif config_file.suffix == '.json':
                    file_config = json.load(f)
                else:
                    return
            
            # Merge file configuration
            if file_config:
                self._merge_config(config, file_config)
                
        except Exception as e:
            logging.warning(f"Failed to load config file {config_file}: {e}")
    
    def _merge_config(self, config: ServiceConfig, file_config: Dict[str, Any]):
        """Merge file configuration into config object"""
        for key, value in file_config.items():
            if hasattr(config, key):
                if isinstance(getattr(config, key), dict) and isinstance(value, dict):
                    # Merge dictionaries
                    getattr(config, key).update(value)
                else:
                    # Override simple values
                    setattr(config, key, value)
            else:
                # Add to settings
                config.settings[key] = value
    
    def _load_service_overrides(self, config: ServiceConfig):
        """Load service-specific configuration overrides"""
        # Service-specific environment variables
        service_env_prefix = f"{self.service_name.upper()}_"
        
        for key, value in os.environ.items():
            if key.startswith(service_env_prefix):
                # Remove service prefix and convert to lowercase
                setting_key = key[len(service_env_prefix):].lower()
                
                # Try to parse as JSON, fall back to string
                try:
                    parsed_value = json.loads(value)
                except:
                    parsed_value = value
                
                config.settings[setting_key] = parsed_value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        # Check service settings first
        if key in self.config.settings:
            return self.config.settings[key]
        
        # Check config attributes
        if hasattr(self.config, key):
            return getattr(self.config, key)
        
        return default
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.config.database
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration"""
        return self.config.cache
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration"""
        return self.config.storage
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.config.monitoring
    
    def get_auth_config(self) -> Dict[str, Any]:
        """Get auth configuration"""
        return self.config.auth
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.config.environment.lower() in ['development', 'dev', 'local']
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.config.environment.lower() in ['production', 'prod']
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled"""
        return self.config.debug


# Global configuration managers
_config_managers: Dict[str, ConfigManager] = {}


def get_config_manager(service_name: str) -> ConfigManager:
    """Get configuration manager for service"""
    if service_name not in _config_managers:
        _config_managers[service_name] = ConfigManager(service_name)
    return _config_managers[service_name]


def get_service_config(service_name: str = None) -> Dict[str, Any]:
    """Get service configuration (for backwards compatibility)"""
    if service_name is None:
        service_name = os.getenv('SERVICE_NAME', 'default')
    
    manager = get_config_manager(service_name)
    return {
        'name': manager.config.name,
        'version': manager.config.version,
        'tier': manager.config.tier,
        'environment': manager.config.environment,
        'debug': manager.config.debug,
        'database': manager.config.database,
        'cache': manager.config.cache,
        'storage': manager.config.storage,
        'monitoring': manager.config.monitoring,
        'auth': manager.config.auth,
        'settings': manager.config.settings
    }


class FeatureFlags:
    """Feature flags management"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.config_manager = get_config_manager(service_name)
    
    def is_enabled(self, feature_name: str) -> bool:
        """Check if feature is enabled"""
        # Check environment variable first
        env_var = f"FEATURE_{feature_name.upper()}_ENABLED"
        env_value = os.getenv(env_var)
        if env_value is not None:
            return env_value.lower() == 'true'
        
        # Check configuration
        features = self.config_manager.get('features', {})
        return features.get(feature_name, False)
    
    def get_feature_config(self, feature_name: str) -> Dict[str, Any]:
        """Get feature configuration"""
        features = self.config_manager.get('features', {})
        return features.get(feature_name, {})


class SecretsManager:
    """Secrets management (simplified version)"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """Get secret value"""
        # In production, this would integrate with AWS Secrets Manager,
        # Azure Key Vault, or Kubernetes secrets
        
        # For now, use environment variables
        return os.getenv(f"SECRET_{secret_name.upper()}")
    
    def get_database_password(self) -> Optional[str]:
        """Get database password from secrets"""
        return self.get_secret('database_password')
    
    def get_jwt_secret(self) -> Optional[str]:
        """Get JWT secret from secrets"""
        return self.get_secret('jwt_secret')
    
    def get_storage_credentials(self) -> Dict[str, str]:
        """Get storage credentials from secrets"""
        return {
            'access_key': self.get_secret('storage_access_key') or '',
            'secret_key': self.get_secret('storage_secret_key') or ''
        }


# Configuration validation
def validate_config(config: ServiceConfig) -> List[str]:
    """Validate configuration and return list of errors"""
    errors = []
    
    # Validate required fields
    if not config.name:
        errors.append("Service name is required")
    
    # Validate database config
    db_config = config.database
    if not db_config.get('host'):
        errors.append("Database host is required")
    if not db_config.get('username'):
        errors.append("Database username is required")
    if not db_config.get('password'):
        errors.append("Database password is required")
    
    # Validate cache config
    cache_config = config.cache
    if not cache_config.get('host'):
        errors.append("Cache host is required")
    
    # Validate storage config
    storage_config = config.storage
    if not storage_config.get('bucket_name'):
        errors.append("Storage bucket name is required")
    
    return errors


# Environment-specific configurations
class EnvironmentConfig:
    """Environment-specific configuration helpers"""
    
    @staticmethod
    def get_development_overrides() -> Dict[str, Any]:
        """Get development environment overrides"""
        return {
            'database': {
                'echo': True,
                'pool_size': 2
            },
            'cache': {
                'default_ttl': 60  # Shorter TTL for development
            },
            'monitoring': {
                'log_level': 'DEBUG',
                'enable_tracing': True
            }
        }
    
    @staticmethod
    def get_production_overrides() -> Dict[str, Any]:
        """Get production environment overrides"""
        return {
            'database': {
                'echo': False,
                'pool_size': 20
            },
            'cache': {
                'default_ttl': 3600
            },
            'monitoring': {
                'log_level': 'INFO',
                'enable_tracing': True
            }
        }
    
    @staticmethod
    def get_test_overrides() -> Dict[str, Any]:
        """Get test environment overrides"""
        return {
            'database': {
                'provider': 'sqlite',
                'database': ':memory:'
            },
            'cache': {
                'provider': 'memory'
            },
            'storage': {
                'provider': 'memory'
            },
            'monitoring': {
                'enable_metrics': False,
                'enable_tracing': False
            }
        }