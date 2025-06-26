"""
TextNLP Service Adapter for Unified Infrastructure

This adapter connects the TextNLP service with the unified cloud deployment
infrastructure, providing service-specific configuration for text and NLP
synthetic data generation.
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
class ModelConfig:
    name: str
    enabled: bool
    min_tier: ServiceTier
    is_commercial: bool = False
    max_tokens: int = 2048
    resources_multiplier: float = 1.0


class TextNLPServiceAdapter:
    """Adapter to connect TextNLP service with unified infrastructure"""
    
    SERVICE_NAME = "textnlp"
    SERVICE_TYPE = "api"
    SERVICE_VERSION = "1.0.0"
    
    # Model configurations with tier requirements
    MODELS = [
        ModelConfig("gpt2", True, ServiceTier.STARTER, max_tokens=1024),
        ModelConfig("gpt2-medium", True, ServiceTier.STARTER, max_tokens=1024, resources_multiplier=1.5),
        ModelConfig("gpt2-large", True, ServiceTier.PROFESSIONAL, max_tokens=2048, resources_multiplier=2.0),
        ModelConfig("gpt2-xl", True, ServiceTier.PROFESSIONAL, max_tokens=2048, resources_multiplier=3.0),
        ModelConfig("gpt-j-6b", True, ServiceTier.BUSINESS, max_tokens=4096, resources_multiplier=4.0),
        ModelConfig("llama-7b", True, ServiceTier.BUSINESS, max_tokens=4096, resources_multiplier=5.0),
        ModelConfig("gpt-4", True, ServiceTier.ENTERPRISE, is_commercial=True, max_tokens=8192),
        ModelConfig("claude-2", True, ServiceTier.ENTERPRISE, is_commercial=True, max_tokens=100000),
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
                "timeout": "10s",
                "retries": 3
            },
            "readiness_check": {
                "path": "/ready",
                "initial_delay": "20s",  # Longer for model loading
                "period": "10s"
            },
            "resources": self._get_resource_config(),
            "environment": self._get_environment_vars(),
            "dependencies": self._get_dependencies(),
            "features": self._get_feature_flags(),
            "volumes": self._get_volume_config()
        }
    
    def get_scaling_config(self) -> Dict[str, Any]:
        """Return auto-scaling configuration based on tier"""
        scaling_configs = {
            ServiceTier.STARTER: {
                "min_replicas": 1,
                "max_replicas": 3,
                "target_cpu_utilization": 80,
                "target_memory_utilization": 85,
                "scale_down_stabilization": 300,
                "scale_up_stabilization": 60
            },
            ServiceTier.PROFESSIONAL: {
                "min_replicas": 2,
                "max_replicas": 10,
                "target_cpu_utilization": 70,
                "target_memory_utilization": 80,
                "scale_down_stabilization": 300,
                "scale_up_stabilization": 30
            },
            ServiceTier.BUSINESS: {
                "min_replicas": 3,
                "max_replicas": 25,
                "target_cpu_utilization": 65,
                "target_memory_utilization": 75,
                "scale_down_stabilization": 180,
                "scale_up_stabilization": 30,
                "custom_metrics": [
                    {
                        "name": "tokens_generation_rate",
                        "target_value": 100000,
                        "type": "AverageValue"
                    }
                ]
            },
            ServiceTier.ENTERPRISE: {
                "min_replicas": 5,
                "max_replicas": 50,
                "target_cpu_utilization": 60,
                "target_memory_utilization": 70,
                "scale_down_stabilization": 120,
                "scale_up_stabilization": 15,
                "custom_metrics": [
                    {
                        "name": "tokens_generation_rate",
                        "target_value": 1000000,
                        "type": "AverageValue"
                    },
                    {
                        "name": "concurrent_streams",
                        "target_value": 100,
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
                        "name": "textnlp_tokens_generated_total",
                        "type": "counter",
                        "labels": ["model", "tier", "user_id"]
                    },
                    {
                        "name": "textnlp_generation_duration_seconds",
                        "type": "histogram",
                        "labels": ["model", "tier", "prompt_length"],
                        "buckets": [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
                    },
                    {
                        "name": "textnlp_active_streams",
                        "type": "gauge",
                        "labels": ["model"]
                    },
                    {
                        "name": "textnlp_validation_scores",
                        "type": "histogram",
                        "labels": ["metric_type", "model"],
                        "buckets": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
                },
                "sensitive_data_masking": True
            },
            "tracing": {
                "enabled": self.tier != ServiceTier.STARTER,
                "sample_rate": 0.01 if self.tier == ServiceTier.PROFESSIONAL else 0.1,
                "tags": {
                    "service.tier": self.tier.value,
                    "service.type": "nlp-generation"
                }
            },
            "alerts": self._get_alert_rules()
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """Return API configuration including rate limits and billing"""
        rate_limits = {
            ServiceTier.STARTER: {
                "tokens_per_hour": 500_000,
                "requests_per_hour": 100,
                "concurrent_requests": 2,
                "streaming_connections": 1
            },
            ServiceTier.PROFESSIONAL: {
                "tokens_per_hour": 5_000_000,
                "requests_per_hour": 1000,
                "concurrent_requests": 10,
                "streaming_connections": 5
            },
            ServiceTier.BUSINESS: {
                "tokens_per_hour": 25_000_000,
                "requests_per_hour": 10000,
                "concurrent_requests": 50,
                "streaming_connections": 25
            },
            ServiceTier.ENTERPRISE: {
                "tokens_per_hour": -1,  # Unlimited
                "requests_per_hour": -1,
                "concurrent_requests": -1,
                "streaming_connections": -1
            }
        }
        
        return {
            "rate_limiting": rate_limits[self.tier],
            "endpoints": [
                {
                    "path": "/api/textnlp/generate",
                    "method": "POST",
                    "billing": {
                        "metric": "tokens_generated",
                        "rate": self._get_billing_rate("tokens")
                    },
                    "timeout": "300s",
                    "streaming": True
                },
                {
                    "path": "/api/textnlp/chat",
                    "method": "POST",
                    "billing": {
                        "metric": "tokens_generated",
                        "rate": self._get_billing_rate("tokens")
                    },
                    "timeout": "600s",
                    "streaming": True
                },
                {
                    "path": "/api/textnlp/validate",
                    "method": "POST",
                    "billing": {
                        "metric": "validations_performed",
                        "rate": self._get_billing_rate("validations")
                    },
                    "timeout": "60s"
                },
                {
                    "path": "/api/textnlp/models",
                    "method": "GET",
                    "cache": {
                        "enabled": True,
                        "ttl": "3600s"
                    }
                },
                {
                    "path": "/api/textnlp/templates",
                    "method": "GET",
                    "tier_required": ServiceTier.PROFESSIONAL
                },
                {
                    "path": "/api/textnlp/fine-tune",
                    "method": "POST",
                    "tier_required": ServiceTier.ENTERPRISE
                }
            ],
            "websocket": {
                "enabled": self.tier != ServiceTier.STARTER,
                "path": "/ws/textnlp/stream",
                "max_connections": rate_limits[self.tier].get("streaming_connections", 1)
            }
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Return model-specific configuration"""
        available_models = {}
        
        for model in self.MODELS:
            if model.enabled and model.min_tier.value <= self.tier.value:
                available_models[model.name] = {
                    "enabled": True,
                    "max_tokens": model.max_tokens,
                    "is_commercial": model.is_commercial,
                    "resources_multiplier": model.resources_multiplier,
                    "cache_config": {
                        "enabled": not model.is_commercial,  # Don't cache commercial model outputs
                        "ttl": 3600 if model.name.startswith("gpt2") else 1800
                    }
                }
        
        return {
            "available_models": available_models,
            "default_model": "gpt2" if self.tier == ServiceTier.STARTER else "gpt2-large",
            "model_loading": {
                "preload": self.tier != ServiceTier.STARTER,
                "lazy_load": self.tier == ServiceTier.STARTER,
                "cache_models": True
            }
        }
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Return storage configuration"""
        storage_limits = {
            ServiceTier.STARTER: 5 * 1024 * 1024 * 1024,  # 5GB
            ServiceTier.PROFESSIONAL: 50 * 1024 * 1024 * 1024,  # 50GB
            ServiceTier.BUSINESS: 500 * 1024 * 1024 * 1024,  # 500GB
            ServiceTier.ENTERPRISE: -1  # Unlimited
        }
        
        return {
            "type": "object",
            "provider": "unified",
            "bucket_prefix": f"inferloop-{self.SERVICE_NAME}",
            "paths": {
                "prompts": "/prompts",
                "generations": "/generations",
                "templates": "/templates",
                "fine_tuned_models": "/models/fine-tuned",
                "validation_results": "/validations"
            },
            "lifecycle": {
                "temp_retention_days": 1,
                "generation_retention_days": 7 if self.tier == ServiceTier.STARTER else 30,
                "validation_retention_days": 30
            },
            "quota": storage_limits[self.tier],
            "encryption": {
                "enabled": True,
                "key_rotation": self.tier == ServiceTier.ENTERPRISE
            }
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Return cache configuration"""
        cache_sizes = {
            ServiceTier.STARTER: "512MB",
            ServiceTier.PROFESSIONAL: "2GB",
            ServiceTier.BUSINESS: "8GB",
            ServiceTier.ENTERPRISE: "32GB"
        }
        
        return {
            "type": "redis",
            "namespace": f"textnlp:{self.tier.value}",
            "ttl": {
                "default": 3600,
                "generation_results": 1800,  # 30 minutes
                "model_outputs": 900,        # 15 minutes
                "templates": 86400,          # 24 hours
                "api_responses": 300         # 5 minutes
            },
            "max_memory": cache_sizes[self.tier],
            "eviction_policy": "allkeys-lru",
            "features": {
                "result_caching": self.tier != ServiceTier.STARTER,
                "model_output_caching": self.tier == ServiceTier.ENTERPRISE
            }
        }
    
    def _get_resource_config(self) -> Dict[str, Any]:
        """Get resource configuration based on tier"""
        resource_configs = {
            ServiceTier.STARTER: {
                "requests": {"cpu": "1", "memory": "2Gi"},
                "limits": {"cpu": "2", "memory": "4Gi"}
            },
            ServiceTier.PROFESSIONAL: {
                "requests": {"cpu": "2", "memory": "4Gi"},
                "limits": {"cpu": "4", "memory": "8Gi"}
            },
            ServiceTier.BUSINESS: {
                "requests": {"cpu": "4", "memory": "8Gi"},
                "limits": {"cpu": "8", "memory": "16Gi"}
            },
            ServiceTier.ENTERPRISE: {
                "requests": {"cpu": "8", "memory": "16Gi"},
                "limits": {"cpu": "16", "memory": "32Gi"}
            }
        }
        
        # Add GPU resources for business and enterprise tiers
        if self.tier in [ServiceTier.BUSINESS, ServiceTier.ENTERPRISE]:
            resource_configs[self.tier]["limits"]["nvidia.com/gpu"] = "1"
            
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
            "METRICS_PORT": "9090",
            "MODEL_CACHE_DIR": "/models",
            "ENABLE_GPU": str(self.tier in [ServiceTier.BUSINESS, ServiceTier.ENTERPRISE]),
            "MAX_BATCH_SIZE": "8" if self.tier != ServiceTier.ENTERPRISE else "16"
        }
    
    def _get_dependencies(self) -> List[str]:
        """Get service dependencies"""
        deps = ["postgres", "redis", "auth-service", "storage-service"]
        
        if self.tier in [ServiceTier.BUSINESS, ServiceTier.ENTERPRISE]:
            deps.extend(["monitoring-service", "billing-service", "gpu-scheduler"])
            
        if self.tier == ServiceTier.ENTERPRISE:
            deps.extend(["model-registry", "fine-tuning-service"])
            
        return deps
    
    def _get_feature_flags(self) -> Dict[str, bool]:
        """Get feature flags based on tier"""
        return {
            "enable_gpt2_models": True,
            "enable_large_models": self.tier != ServiceTier.STARTER,
            "enable_commercial_models": self.tier == ServiceTier.ENTERPRISE,
            "enable_streaming": True,
            "enable_batch_processing": self.tier != ServiceTier.STARTER,
            "enable_templates": self.tier != ServiceTier.STARTER,
            "enable_fine_tuning": self.tier == ServiceTier.ENTERPRISE,
            "enable_validation": True,
            "enable_caching": self.tier != ServiceTier.STARTER,
            "enable_websocket": self.tier != ServiceTier.STARTER,
            "enable_langchain": self.tier in [ServiceTier.PROFESSIONAL, ServiceTier.BUSINESS, ServiceTier.ENTERPRISE]
        }
    
    def _get_volume_config(self) -> List[Dict[str, Any]]:
        """Get volume configuration for model storage"""
        volumes = [
            {
                "name": "model-cache",
                "mount_path": "/models",
                "size": "10Gi" if self.tier == ServiceTier.STARTER else "50Gi",
                "storage_class": "fast-ssd"
            }
        ]
        
        if self.tier == ServiceTier.ENTERPRISE:
            volumes.append({
                "name": "fine-tuned-models",
                "mount_path": "/models/fine-tuned",
                "size": "100Gi",
                "storage_class": "fast-ssd"
            })
            
        return volumes
    
    def _get_alert_rules(self) -> List[Dict[str, Any]]:
        """Get monitoring alert rules"""
        base_alerts = [
            {
                "name": "TextNLPHighErrorRate",
                "condition": 'rate(textnlp_errors_total[5m]) > 0.05',
                "severity": "warning",
                "annotations": {
                    "summary": "High error rate in TextNLP service",
                    "description": "Error rate is above 5% for 5 minutes"
                }
            },
            {
                "name": "TextNLPHighLatency",
                "condition": 'histogram_quantile(0.95, textnlp_generation_duration_seconds) > 60',
                "severity": "warning",
                "annotations": {
                    "summary": "High text generation latency",
                    "description": "95th percentile latency is above 60 seconds"
                }
            },
            {
                "name": "TextNLPLowTokenThroughput",
                "condition": 'rate(textnlp_tokens_generated_total[5m]) < 1000',
                "severity": "info",
                "for": "10m"
            }
        ]
        
        if self.tier in [ServiceTier.BUSINESS, ServiceTier.ENTERPRISE]:
            base_alerts.extend([
                {
                    "name": "TextNLPHighGPUUsage",
                    "condition": 'nvidia_gpu_usage_percent{pod=~"textnlp-.*"} > 90',
                    "severity": "warning",
                    "for": "5m"
                },
                {
                    "name": "TextNLPModelLoadFailure",
                    "condition": 'increase(textnlp_model_load_failures_total[5m]) > 0',
                    "severity": "critical"
                }
            ])
        
        return base_alerts
    
    def _get_billing_rate(self, metric_type: str) -> float:
        """Get billing rate based on tier and metric type"""
        billing_rates = {
            ServiceTier.STARTER: {
                "tokens": 0.0004,  # per 1K tokens
                "validations": 0.01
            },
            ServiceTier.PROFESSIONAL: {
                "tokens": 0.0002,  # per 1K tokens
                "validations": 0.005
            },
            ServiceTier.BUSINESS: {
                "tokens": 0.0001,  # per 1K tokens
                "validations": 0.002
            },
            ServiceTier.ENTERPRISE: {
                "tokens": 0.00005,  # per 1K tokens
                "validations": 0.001
            }
        }
        return billing_rates[self.tier].get(metric_type, 0.0)
    
    def validate_migration_readiness(self) -> Dict[str, Any]:
        """Validate if the service is ready for migration to unified infrastructure"""
        checks = {
            "configuration_valid": True,
            "models_available": True,
            "dependencies_ready": True,
            "api_endpoints_defined": True,
            "monitoring_configured": True,
            "storage_configured": True,
            "cache_configured": True,
            "scaling_configured": True
        }
        
        issues = []
        
        # Validate models
        available_models = [m for m in self.MODELS if m.min_tier.value <= self.tier.value]
        if not available_models:
            checks["models_available"] = False
            issues.append("No models available for current tier")
        
        # Validate GPU requirements
        if self.tier in [ServiceTier.BUSINESS, ServiceTier.ENTERPRISE]:
            # Check if GPU nodes are available in the cluster
            # This would be a real check in production
            pass
        
        return {
            "ready": all(checks.values()),
            "checks": checks,
            "issues": issues,
            "recommendations": [
                "Ensure model files are uploaded to unified storage",
                "Verify GPU nodes are available for Business/Enterprise tiers",
                "Update API clients to use new endpoints"
            ]
        }