"""
Tabular Service Adapter for Unified Infrastructure

This adapter connects the tabular service with the unified cloud deployment
infrastructure, providing service-specific configuration while leveraging
shared infrastructure components.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class ServiceTier(Enum):
    STARTER = "starter"
    PROFESSIONAL = "professional"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


@dataclass
class AlgorithmConfig:
    name: str
    enabled: bool
    min_tier: ServiceTier
    resources_multiplier: float = 1.0


class TabularServiceAdapter:
    """Adapter to connect tabular service with unified infrastructure"""
    
    SERVICE_NAME = "tabular"
    SERVICE_TYPE = "api"
    SERVICE_VERSION = "1.0.0"
    
    # Algorithm configurations with tier requirements
    ALGORITHMS = [
        AlgorithmConfig("sdv", True, ServiceTier.STARTER),
        AlgorithmConfig("ctgan", True, ServiceTier.PROFESSIONAL, 1.5),
        AlgorithmConfig("ydata", True, ServiceTier.PROFESSIONAL, 1.3),
        AlgorithmConfig("custom", True, ServiceTier.ENTERPRISE, 2.0),
    ]
    
    def __init__(self, tier: ServiceTier = ServiceTier.STARTER):
        self.tier = tier
    
    def get_service_config(self) -> Dict[str, Any]:
        """Return service-specific configuration for unified infrastructure"""
        return {
            "name": self.SERVICE_NAME,
            "type": self.SERVICE_TYPE,
            "version": self.SERVICE_VERSION,
            "image": f"inferloop/{self.SERVICE_NAME}:{self.SERVICE_VERSION}",
            "port": 8000,
            "health_check": {
                "path": "/health",
                "interval": "30s",
                "timeout": "5s",
                "retries": 3
            },
            "readiness_check": {
                "path": "/ready",
                "initial_delay": "10s",
                "period": "5s"
            },
            "resources": self._get_resource_config(),
            "environment": self._get_environment_vars(),
            "dependencies": self._get_dependencies(),
            "features": self._get_feature_flags()
        }
    
    def get_scaling_config(self) -> Dict[str, Any]:
        """Return auto-scaling configuration based on tier"""
        scaling_configs = {
            ServiceTier.STARTER: {
                "min_replicas": 1,
                "max_replicas": 3,
                "target_cpu_utilization": 80,
                "target_memory_utilization": 85,
                "scale_down_stabilization": 300
            },
            ServiceTier.PROFESSIONAL: {
                "min_replicas": 2,
                "max_replicas": 10,
                "target_cpu_utilization": 70,
                "target_memory_utilization": 80,
                "scale_down_stabilization": 300
            },
            ServiceTier.BUSINESS: {
                "min_replicas": 3,
                "max_replicas": 50,
                "target_cpu_utilization": 70,
                "target_memory_utilization": 75,
                "scale_down_stabilization": 180
            },
            ServiceTier.ENTERPRISE: {
                "min_replicas": 5,
                "max_replicas": 100,
                "target_cpu_utilization": 60,
                "target_memory_utilization": 70,
                "scale_down_stabilization": 120,
                "custom_metrics": [
                    {
                        "name": "rows_generation_rate",
                        "target_value": 10000,
                        "type": "AverageValue"
                    }
                ]
            }
        }
        return scaling_configs[self.tier]
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Return monitoring configuration"""
        return {
            "metrics": {
                "enabled": True,
                "path": "/metrics",
                "port": 9090,
                "custom_metrics": [
                    {
                        "name": "tabular_rows_generated_total",
                        "type": "counter",
                        "labels": ["algorithm", "tier", "user_id"]
                    },
                    {
                        "name": "tabular_generation_duration_seconds",
                        "type": "histogram",
                        "labels": ["algorithm", "tier"],
                        "buckets": [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
                    },
                    {
                        "name": "tabular_active_generations",
                        "type": "gauge",
                        "labels": ["algorithm"]
                    }
                ]
            },
            "logging": {
                "level": "INFO" if self.tier != ServiceTier.ENTERPRISE else "DEBUG",
                "format": "json",
                "additional_fields": {
                    "service": self.SERVICE_NAME,
                    "version": self.SERVICE_VERSION,
                    "tier": self.tier.value
                }
            },
            "tracing": {
                "enabled": self.tier != ServiceTier.STARTER,
                "sample_rate": 0.01 if self.tier == ServiceTier.PROFESSIONAL else 0.1,
                "tags": {
                    "service.tier": self.tier.value,
                    "service.type": self.SERVICE_TYPE
                }
            },
            "alerts": self._get_alert_rules()
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """Return API configuration including rate limits and billing"""
        rate_limits = {
            ServiceTier.STARTER: {
                "requests_per_hour": 100,
                "concurrent_requests": 2,
                "burst_size": 10
            },
            ServiceTier.PROFESSIONAL: {
                "requests_per_hour": 1000,
                "concurrent_requests": 10,
                "burst_size": 50
            },
            ServiceTier.BUSINESS: {
                "requests_per_hour": 10000,
                "concurrent_requests": 50,
                "burst_size": 200
            },
            ServiceTier.ENTERPRISE: {
                "requests_per_hour": -1,  # Unlimited
                "concurrent_requests": -1,
                "burst_size": -1
            }
        }
        
        return {
            "rate_limiting": rate_limits[self.tier],
            "endpoints": [
                {
                    "path": "/api/tabular/generate",
                    "method": "POST",
                    "billing": {
                        "metric": "rows_generated",
                        "rate": self._get_billing_rate("rows")
                    },
                    "timeout": "300s" if self.tier == ServiceTier.ENTERPRISE else "60s"
                },
                {
                    "path": "/api/tabular/validate",
                    "method": "POST",
                    "billing": {
                        "metric": "validations_performed",
                        "rate": self._get_billing_rate("validations")
                    },
                    "timeout": "60s"
                },
                {
                    "path": "/api/tabular/algorithms",
                    "method": "GET",
                    "cache": {
                        "enabled": True,
                        "ttl": "3600s"
                    }
                }
            ],
            "cors": {
                "enabled": True,
                "origins": ["https://app.inferloop.io"],
                "methods": ["GET", "POST", "PUT", "DELETE"],
                "headers": ["Content-Type", "Authorization"]
            }
        }
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Return storage configuration"""
        storage_limits = {
            ServiceTier.STARTER: 10 * 1024 * 1024 * 1024,  # 10GB
            ServiceTier.PROFESSIONAL: 100 * 1024 * 1024 * 1024,  # 100GB
            ServiceTier.BUSINESS: 1024 * 1024 * 1024 * 1024,  # 1TB
            ServiceTier.ENTERPRISE: -1  # Unlimited
        }
        
        return {
            "type": "object",
            "provider": "unified",  # Uses unified storage abstraction
            "bucket_prefix": f"inferloop-{self.SERVICE_NAME}",
            "paths": {
                "input_data": "/input",
                "output_data": "/output",
                "models": "/models",
                "temp": "/temp"
            },
            "lifecycle": {
                "temp_retention_days": 1,
                "output_retention_days": 30 if self.tier == ServiceTier.STARTER else 90
            },
            "quota": storage_limits[self.tier],
            "encryption": {
                "enabled": True,
                "key_rotation": self.tier == ServiceTier.ENTERPRISE
            }
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """Return database configuration"""
        return {
            "type": "postgres",
            "schema": f"tabular_{self.tier.value}",
            "pool": {
                "min_size": 5,
                "max_size": 20 if self.tier != ServiceTier.ENTERPRISE else 100,
                "timeout": 30,
                "max_lifetime": 3600
            },
            "migrations": {
                "auto_migrate": True,
                "migration_path": "/migrations"
            }
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Return cache configuration"""
        cache_sizes = {
            ServiceTier.STARTER: "1GB",
            ServiceTier.PROFESSIONAL: "4GB",
            ServiceTier.BUSINESS: "16GB",
            ServiceTier.ENTERPRISE: "64GB"
        }
        
        return {
            "type": "redis",
            "namespace": f"tabular:{self.tier.value}",
            "ttl": {
                "default": 3600,
                "generation_results": 86400,  # 24 hours
                "validation_results": 3600,   # 1 hour
                "api_responses": 300          # 5 minutes
            },
            "max_memory": cache_sizes[self.tier],
            "eviction_policy": "allkeys-lru"
        }
    
    def _get_resource_config(self) -> Dict[str, Any]:
        """Get resource configuration based on tier"""
        resource_configs = {
            ServiceTier.STARTER: {
                "requests": {"cpu": "500m", "memory": "1Gi"},
                "limits": {"cpu": "1", "memory": "2Gi"}
            },
            ServiceTier.PROFESSIONAL: {
                "requests": {"cpu": "1", "memory": "2Gi"},
                "limits": {"cpu": "2", "memory": "4Gi"}
            },
            ServiceTier.BUSINESS: {
                "requests": {"cpu": "2", "memory": "4Gi"},
                "limits": {"cpu": "4", "memory": "8Gi"}
            },
            ServiceTier.ENTERPRISE: {
                "requests": {"cpu": "4", "memory": "8Gi"},
                "limits": {"cpu": "8", "memory": "16Gi"}
            }
        }
        return resource_configs[self.tier]
    
    def _get_environment_vars(self) -> Dict[str, str]:
        """Get environment variables"""
        return {
            "SERVICE_NAME": self.SERVICE_NAME,
            "SERVICE_TIER": self.tier.value,
            "LOG_LEVEL": "INFO",
            "ENABLE_PROFILING": str(self.tier == ServiceTier.ENTERPRISE),
            "MAX_WORKERS": "4" if self.tier != ServiceTier.ENTERPRISE else "8",
            "ENABLE_METRICS": "true",
            "METRICS_PORT": "9090"
        }
    
    def _get_dependencies(self) -> List[str]:
        """Get service dependencies"""
        deps = ["postgres", "redis", "auth-service"]
        if self.tier in [ServiceTier.BUSINESS, ServiceTier.ENTERPRISE]:
            deps.extend(["monitoring-service", "billing-service"])
        return deps
    
    def _get_feature_flags(self) -> Dict[str, bool]:
        """Get feature flags based on tier"""
        return {
            "enable_sdv": True,
            "enable_ctgan": self.tier != ServiceTier.STARTER,
            "enable_ydata": self.tier != ServiceTier.STARTER,
            "enable_custom_models": self.tier == ServiceTier.ENTERPRISE,
            "enable_batch_processing": self.tier != ServiceTier.STARTER,
            "enable_api_key_auth": True,
            "enable_oauth": self.tier in [ServiceTier.BUSINESS, ServiceTier.ENTERPRISE],
            "enable_audit_logging": self.tier == ServiceTier.ENTERPRISE,
            "enable_data_encryption": self.tier != ServiceTier.STARTER
        }
    
    def _get_alert_rules(self) -> List[Dict[str, Any]]:
        """Get monitoring alert rules"""
        base_alerts = [
            {
                "name": "TabularHighErrorRate",
                "condition": 'rate(tabular_errors_total[5m]) > 0.05',
                "severity": "warning",
                "annotations": {
                    "summary": "High error rate in Tabular service",
                    "description": "Error rate is above 5% for 5 minutes"
                }
            },
            {
                "name": "TabularHighLatency",
                "condition": 'histogram_quantile(0.95, tabular_generation_duration_seconds) > 30',
                "severity": "warning",
                "annotations": {
                    "summary": "High generation latency",
                    "description": "95th percentile latency is above 30 seconds"
                }
            }
        ]
        
        if self.tier in [ServiceTier.BUSINESS, ServiceTier.ENTERPRISE]:
            base_alerts.extend([
                {
                    "name": "TabularHighMemoryUsage",
                    "condition": 'container_memory_usage_bytes{pod=~"tabular-.*"} / container_spec_memory_limit_bytes > 0.8',
                    "severity": "warning",
                    "for": "5m"
                },
                {
                    "name": "TabularPodCrashLooping",
                    "condition": 'rate(kube_pod_container_status_restarts_total{pod=~"tabular-.*"}[15m]) > 0',
                    "severity": "critical"
                }
            ])
        
        return base_alerts
    
    def _get_billing_rate(self, metric_type: str) -> float:
        """Get billing rate based on tier and metric type"""
        billing_rates = {
            ServiceTier.STARTER: {"rows": 0.003, "validations": 0.001},
            ServiceTier.PROFESSIONAL: {"rows": 0.001, "validations": 0.0005},
            ServiceTier.BUSINESS: {"rows": 0.0005, "validations": 0.0001},
            ServiceTier.ENTERPRISE: {"rows": 0.0001, "validations": 0.00005}
        }
        return billing_rates[self.tier].get(metric_type, 0.0)
    
    def validate_migration_readiness(self) -> Dict[str, Any]:
        """Validate if the service is ready for migration to unified infrastructure"""
        checks = {
            "configuration_valid": True,
            "dependencies_available": True,
            "resources_adequate": True,
            "api_endpoints_defined": True,
            "monitoring_configured": True,
            "storage_configured": True,
            "database_configured": True,
            "cache_configured": True
        }
        
        issues = []
        
        # Validate configuration
        config = self.get_service_config()
        if not config.get("image"):
            checks["configuration_valid"] = False
            issues.append("Docker image not specified")
        
        # Validate dependencies
        deps = self._get_dependencies()
        # In real implementation, check if these services exist
        
        # Validate resources
        resources = self._get_resource_config()
        if not resources.get("requests"):
            checks["resources_adequate"] = False
            issues.append("Resource requests not defined")
        
        return {
            "ready": all(checks.values()),
            "checks": checks,
            "issues": issues
        }