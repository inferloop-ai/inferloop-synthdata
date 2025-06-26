# Inferloop Product Architecture - Modular Services & Commercialization

## Overview

This document details the technical architecture for productizing Inferloop's synthetic data services as standalone commercial offerings while leveraging the unified infrastructure. It covers multi-tenancy, billing integration, API design, and deployment models.

## Multi-Product Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Customer Portal                           │
│              (Billing, User Management, Analytics)               │
└───────────────┬─────────────────────────────┬───────────────────┘
                │                             │
┌───────────────▼─────────────┐ ┌────────────▼────────────────────┐
│   Product Marketplace       │ │    Customer Admin Panel         │
│  • Tabular Analytics        │ │  • Usage Dashboard              │
│  • TextNLP Generation       │ │  • Team Management              │
│  • SynDoc Creator          │ │  • Billing & Invoices           │
│  • Model Store             │ │  • API Keys                     │
└───────────────┬─────────────┘ └────────────┬────────────────────┘
                │                             │
┌───────────────▼─────────────────────────────▼───────────────────┐
│                    Unified API Gateway                           │
│  ┌──────────────┬──────────────┬──────────────┬─────────────┐ │
│  │   Auth &     │   Usage      │   Rate       │   Billing   │ │
│  │   Tenant     │   Tracking   │   Limiting   │   Metering  │ │
│  └──────────────┴──────────────┴──────────────┴─────────────┘ │
└───────────────┬─────────────┬──────────────┬───────────────────┘
                │             │              │
┌───────────────▼────┐ ┌──────▼──────┐ ┌────▼───────────────────┐
│  Tabular Service   │ │  TextNLP    │ │   SynDoc Service       │
│  ┌───────────────┐ │ │  Service    │ │  ┌─────────────────┐  │
│  │ SDV Engine    │ │ │ ┌─────────┐ │ │  │ Template Engine │  │
│  │ CTGAN Engine  │ │ │ │ GPT-2   │ │ │  │ PDF Generator   │  │
│  │ YData Engine  │ │ │ │ LLaMA   │ │ │  │ Form Builder    │  │
│  └───────────────┘ │ │ │ Claude  │ │ │  └─────────────────┘  │
└────────────────────┘ │ └─────────┘ │ └────────────────────────┘
                       └─────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼──────────┐ ┌───────▼──────────┐ ┌───────▼──────────┐
│ Shared Storage   │ │ Shared Database  │ │   Message Queue  │
│ (Model Artifacts)│ │ (Tenant Isolated)│ │ (Job Processing) │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

## Core Components

### 1. API Gateway & Routing

```python
# api_gateway/router.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx

class ProductRouter:
    def __init__(self):
        self.app = FastAPI(title="Inferloop API Gateway")
        self.services = {
            "tabular": "http://tabular-service:8000",
            "textnlp": "http://textnlp-service:8000",
            "syndoc": "http://syndoc-service:8000"
        }
        self.setup_middleware()
        self.setup_routes()
    
    def setup_middleware(self):
        # CORS for web clients
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://app.inferloop.io"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Authentication middleware
        self.app.add_middleware(AuthenticationMiddleware)
        
        # Usage tracking middleware
        self.app.add_middleware(UsageTrackingMiddleware)
        
        # Rate limiting middleware
        self.app.add_middleware(RateLimitingMiddleware)
    
    def setup_routes(self):
        @self.app.post("/api/{service}/{path:path}")
        async def route_request(service: str, path: str, request: Request):
            if service not in self.services:
                raise HTTPException(404, f"Service {service} not found")
            
            # Check user permissions for service
            if not await self.check_permissions(request.state.user, service):
                raise HTTPException(403, "Access denied to this service")
            
            # Forward request to service
            async with httpx.AsyncClient() as client:
                service_url = f"{self.services[service]}/{path}"
                response = await client.request(
                    method=request.method,
                    url=service_url,
                    headers=dict(request.headers),
                    content=await request.body()
                )
                
            return response.json()
```

### 2. Multi-Tenancy Implementation

```python
# core/multitenancy.py
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import contextvars

# Context variable for current tenant
current_tenant = contextvars.ContextVar('current_tenant', default=None)

class TenantManager:
    def __init__(self, base_db_url: str):
        self.base_db_url = base_db_url
        self.engines = {}
        self.sessions = {}
        
    def get_tenant_engine(self, tenant_id: str):
        if tenant_id not in self.engines:
            # Create schema-specific connection
            db_url = f"{self.base_db_url}?options=-csearch_path=tenant_{tenant_id}"
            self.engines[tenant_id] = create_engine(db_url)
            self.sessions[tenant_id] = sessionmaker(bind=self.engines[tenant_id])
        return self.engines[tenant_id]
    
    def get_session(self):
        tenant_id = current_tenant.get()
        if not tenant_id:
            raise ValueError("No tenant context set")
        return self.sessions[tenant_id]()

# Middleware to set tenant context
class TenantMiddleware:
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Extract tenant from JWT claims
            tenant_id = scope.get("auth", {}).get("tenant_id")
            if tenant_id:
                token = current_tenant.set(tenant_id)
                try:
                    await self.app(scope, receive, send)
                finally:
                    current_tenant.reset(token)
            else:
                await self.app(scope, receive, send)
```

### 3. Billing & Usage Tracking

```python
# billing/usage_tracker.py
from datetime import datetime
from typing import Dict, Any
import asyncio
from redis import asyncio as aioredis
import json

class UsageTracker:
    def __init__(self, redis_url: str, billing_service_url: str):
        self.redis = None
        self.redis_url = redis_url
        self.billing_service_url = billing_service_url
        self.batch_size = 1000
        self.flush_interval = 60  # seconds
        
    async def initialize(self):
        self.redis = await aioredis.from_url(self.redis_url)
        asyncio.create_task(self._flush_loop())
    
    async def track_usage(
        self,
        tenant_id: str,
        service: str,
        operation: str,
        metrics: Dict[str, Any],
        request_id: str
    ):
        """Track usage for billing purposes"""
        usage_record = {
            "tenant_id": tenant_id,
            "service": service,
            "operation": operation,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id
        }
        
        # Add to Redis queue
        await self.redis.lpush(
            f"usage_queue:{service}",
            json.dumps(usage_record)
        )
        
        # Update real-time counters
        for metric, value in metrics.items():
            key = f"usage:{tenant_id}:{service}:{metric}:{datetime.now().strftime('%Y-%m')}"
            await self.redis.incrbyfloat(key, value)
    
    async def check_limits(
        self,
        tenant_id: str,
        service: str,
        metric: str
    ) -> bool:
        """Check if tenant has exceeded usage limits"""
        # Get current usage
        key = f"usage:{tenant_id}:{service}:{metric}:{datetime.now().strftime('%Y-%m')}"
        current_usage = float(await self.redis.get(key) or 0)
        
        # Get tenant limits
        limit_key = f"limits:{tenant_id}:{service}:{metric}"
        limit = float(await self.redis.get(limit_key) or float('inf'))
        
        return current_usage < limit
    
    async def _flush_loop(self):
        """Periodically flush usage data to billing service"""
        while True:
            await asyncio.sleep(self.flush_interval)
            await self._flush_usage_data()
    
    async def _flush_usage_data(self):
        """Send batched usage data to billing service"""
        services = ["tabular", "textnlp", "syndoc"]
        
        for service in services:
            queue_key = f"usage_queue:{service}"
            batch = []
            
            # Get batch of records
            for _ in range(self.batch_size):
                record = await self.redis.rpop(queue_key)
                if not record:
                    break
                batch.append(json.loads(record))
            
            if batch:
                # Send to billing service
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{self.billing_service_url}/usage/batch",
                        json={"records": batch}
                    )
```

### 4. Product-Specific APIs

#### Tabular Service API
```python
# tabular/api.py
from fastapi import FastAPI, HTTPException, Depends
from typing import Optional, Dict, Any
import pandas as pd

app = FastAPI(title="Inferloop Tabular API")

class TabularService:
    def __init__(self):
        self.generators = {
            "sdv": SDVGenerator(),
            "ctgan": CTGANGenerator(),
            "ydata": YDataGenerator()
        }
    
    async def generate(
        self,
        request: GenerateRequest,
        tenant_id: str
    ) -> GenerateResponse:
        # Validate request
        if request.rows > 1000000:
            # Check if tenant has enterprise plan
            if not await self.check_enterprise_features(tenant_id):
                raise HTTPException(
                    403,
                    "Generation of >1M rows requires Enterprise plan"
                )
        
        # Select generator
        generator = self.generators.get(
            request.algorithm,
            self.generators["sdv"]
        )
        
        # Generate synthetic data
        synthetic_data = await generator.generate(
            source_data=request.source_data,
            rows=request.rows,
            config=request.config
        )
        
        # Track usage
        await usage_tracker.track_usage(
            tenant_id=tenant_id,
            service="tabular",
            operation="generate",
            metrics={
                "rows_generated": request.rows,
                "algorithm": request.algorithm
            },
            request_id=request.request_id
        )
        
        return GenerateResponse(
            data=synthetic_data,
            metadata={
                "rows": len(synthetic_data),
                "algorithm": request.algorithm,
                "quality_score": await self.calculate_quality_score(
                    request.source_data,
                    synthetic_data
                )
            }
        )

@app.post("/generate")
async def generate_synthetic_data(
    request: GenerateRequest,
    tenant_id: str = Depends(get_current_tenant)
):
    service = TabularService()
    return await service.generate(request, tenant_id)

@app.post("/validate")
async def validate_synthetic_data(
    request: ValidateRequest,
    tenant_id: str = Depends(get_current_tenant)
):
    # Validation logic
    pass

@app.get("/algorithms")
async def list_algorithms(
    tenant_id: str = Depends(get_current_tenant)
):
    # Return available algorithms based on plan
    pass
```

#### TextNLP Service API
```python
# textnlp/api.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI(title="Inferloop TextNLP API")

class TextNLPService:
    def __init__(self):
        self.models = {
            "gpt2": GPT2Generator(),
            "gpt-j": GPTJGenerator(),
            "llama": LLaMAGenerator(),
            "claude": ClaudeAPIGenerator()
        }
    
    async def generate_stream(
        self,
        request: TextGenerateRequest,
        tenant_id: str
    ):
        # Check model access
        if request.model in ["claude", "gpt-4"]:
            if not await self.check_premium_models(tenant_id):
                raise HTTPException(
                    403,
                    f"Model {request.model} requires Pro or Enterprise plan"
                )
        
        # Get model
        model = self.models.get(request.model)
        if not model:
            raise HTTPException(404, f"Model {request.model} not found")
        
        # Stream generation
        token_count = 0
        async for token in model.generate_stream(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        ):
            token_count += 1
            yield f"data: {json.dumps({'token': token})}\n\n"
        
        # Track usage after completion
        await usage_tracker.track_usage(
            tenant_id=tenant_id,
            service="textnlp",
            operation="generate",
            metrics={
                "tokens_generated": token_count,
                "model": request.model
            },
            request_id=request.request_id
        )

@app.post("/generate")
async def generate_text(
    request: TextGenerateRequest,
    tenant_id: str = Depends(get_current_tenant)
):
    service = TextNLPService()
    
    if request.stream:
        return StreamingResponse(
            service.generate_stream(request, tenant_id),
            media_type="text/event-stream"
        )
    else:
        # Non-streaming generation
        pass

@app.post("/fine-tune")
async def fine_tune_model(
    request: FineTuneRequest,
    tenant_id: str = Depends(get_current_tenant)
):
    # Fine-tuning logic (Enterprise only)
    pass
```

### 5. Licensing & Deployment Models

```python
# licensing/license_manager.py
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import json
import base64

class LicenseManager:
    def __init__(self, master_key: str):
        self.cipher = Fernet(master_key.encode())
    
    def generate_license(
        self,
        customer_id: str,
        product: str,
        tier: str,
        features: Dict[str, Any],
        valid_until: datetime
    ) -> str:
        """Generate a license key for a customer"""
        license_data = {
            "customer_id": customer_id,
            "product": product,
            "tier": tier,
            "features": features,
            "issued_at": datetime.utcnow().isoformat(),
            "valid_until": valid_until.isoformat(),
            "license_id": str(uuid.uuid4())
        }
        
        # Encrypt license data
        encrypted = self.cipher.encrypt(
            json.dumps(license_data).encode()
        )
        
        # Create readable license key
        license_key = base64.urlsafe_b64encode(encrypted).decode()
        
        # Format as XXXX-XXXX-XXXX-XXXX
        formatted = '-'.join(
            license_key[i:i+4] 
            for i in range(0, min(16, len(license_key)), 4)
        )
        
        return formatted
    
    def validate_license(self, license_key: str) -> Dict[str, Any]:
        """Validate and decode a license key"""
        try:
            # Remove formatting
            clean_key = license_key.replace('-', '')
            
            # Decode and decrypt
            encrypted = base64.urlsafe_b64decode(clean_key)
            decrypted = self.cipher.decrypt(encrypted)
            license_data = json.loads(decrypted)
            
            # Check validity
            valid_until = datetime.fromisoformat(license_data['valid_until'])
            if datetime.utcnow() > valid_until:
                raise ValueError("License expired")
            
            return license_data
            
        except Exception as e:
            raise ValueError(f"Invalid license: {str(e)}")

# Deployment configuration based on license
class DeploymentConfigurator:
    def __init__(self, license_data: Dict[str, Any]):
        self.license_data = license_data
    
    def get_deployment_config(self) -> Dict[str, Any]:
        """Generate deployment configuration based on license"""
        tier = self.license_data['tier']
        
        if tier == 'enterprise':
            return {
                "deployment_mode": "dedicated",
                "features": {
                    "multi_region": True,
                    "custom_domain": True,
                    "sso": True,
                    "audit_logs": True,
                    "compliance": ["SOC2", "HIPAA"],
                    "sla": "99.95%",
                    "support": "24/7"
                },
                "resources": {
                    "cpu": "unlimited",
                    "memory": "unlimited",
                    "storage": "unlimited"
                }
            }
        elif tier == 'professional':
            return {
                "deployment_mode": "shared-dedicated",
                "features": {
                    "multi_region": False,
                    "custom_domain": True,
                    "sso": False,
                    "audit_logs": True,
                    "compliance": ["SOC2"],
                    "sla": "99.9%",
                    "support": "business-hours"
                },
                "resources": {
                    "cpu": "16 cores",
                    "memory": "64GB",
                    "storage": "1TB"
                }
            }
        else:  # starter
            return {
                "deployment_mode": "shared",
                "features": {
                    "multi_region": False,
                    "custom_domain": False,
                    "sso": False,
                    "audit_logs": False,
                    "compliance": [],
                    "sla": "99.5%",
                    "support": "community"
                },
                "resources": {
                    "cpu": "2 cores",
                    "memory": "8GB",
                    "storage": "100GB"
                }
            }
```

### 6. White-Label Configuration

```python
# whitelabel/configurator.py
class WhiteLabelConfigurator:
    def __init__(self, partner_config: Dict[str, Any]):
        self.partner_config = partner_config
    
    def generate_kubernetes_config(self) -> Dict[str, Any]:
        """Generate Kubernetes configuration for white-label deployment"""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{self.partner_config['partner_id']}-config",
                "namespace": self.partner_config['namespace']
            },
            "data": {
                "branding.json": json.dumps({
                    "company_name": self.partner_config['company_name'],
                    "logo_url": self.partner_config['logo_url'],
                    "primary_color": self.partner_config['primary_color'],
                    "support_email": self.partner_config['support_email'],
                    "custom_domain": self.partner_config['custom_domain']
                }),
                "features.json": json.dumps({
                    "enabled_services": self.partner_config['enabled_services'],
                    "custom_features": self.partner_config.get('custom_features', {}),
                    "api_endpoints": self._generate_api_endpoints()
                })
            }
        }
    
    def _generate_api_endpoints(self) -> Dict[str, str]:
        """Generate white-labeled API endpoints"""
        domain = self.partner_config['custom_domain']
        return {
            "base_url": f"https://api.{domain}",
            "auth_url": f"https://auth.{domain}",
            "portal_url": f"https://app.{domain}",
            "docs_url": f"https://docs.{domain}"
        }
```

### 7. Marketplace Integration

```python
# marketplace/connectors.py
class MarketplaceConnector:
    """Base class for cloud marketplace integrations"""
    
    async def publish_product(
        self,
        product_id: str,
        product_config: Dict[str, Any]
    ):
        raise NotImplementedError
    
    async def handle_subscription(
        self,
        subscription_event: Dict[str, Any]
    ):
        raise NotImplementedError

class AWSMarketplaceConnector(MarketplaceConnector):
    def __init__(self, aws_config: Dict[str, Any]):
        self.aws_config = aws_config
        self.sns_client = boto3.client('sns')
        self.marketplace_client = boto3.client('aws-marketplace')
    
    async def handle_subscription(
        self,
        subscription_event: Dict[str, Any]
    ):
        """Handle AWS Marketplace subscription events"""
        event_type = subscription_event['action']
        
        if event_type == 'subscribe-success':
            # Create tenant
            tenant_id = await self.create_tenant(
                customer_id=subscription_event['customer_identifier'],
                product_code=subscription_event['product_code'],
                plan=subscription_event['dimension']
            )
            
            # Send welcome email
            await self.send_welcome_email(tenant_id)
            
        elif event_type == 'unsubscribe-complete':
            # Deactivate tenant
            await self.deactivate_tenant(
                customer_id=subscription_event['customer_identifier']
            )

class AzureMarketplaceConnector(MarketplaceConnector):
    def __init__(self, azure_config: Dict[str, Any]):
        self.azure_config = azure_config
        self.webhook_secret = azure_config['webhook_secret']
    
    async def validate_webhook(
        self,
        request_body: bytes,
        signature: str
    ) -> bool:
        """Validate Azure Marketplace webhook signature"""
        expected_signature = hmac.new(
            self.webhook_secret.encode(),
            request_body,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
```

## Security & Compliance

### 1. Data Isolation

```python
# security/data_isolation.py
class DataIsolationManager:
    def __init__(self):
        self.encryption_keys = {}
    
    def get_tenant_key(self, tenant_id: str) -> bytes:
        """Get or create tenant-specific encryption key"""
        if tenant_id not in self.encryption_keys:
            # Generate new key for tenant
            key = Fernet.generate_key()
            
            # Store in secure key management service
            self._store_key_in_kms(tenant_id, key)
            
            self.encryption_keys[tenant_id] = key
        
        return self.encryption_keys[tenant_id]
    
    def encrypt_tenant_data(
        self,
        tenant_id: str,
        data: bytes
    ) -> bytes:
        """Encrypt data with tenant-specific key"""
        key = self.get_tenant_key(tenant_id)
        cipher = Fernet(key)
        return cipher.encrypt(data)
    
    def decrypt_tenant_data(
        self,
        tenant_id: str,
        encrypted_data: bytes
    ) -> bytes:
        """Decrypt data with tenant-specific key"""
        key = self.get_tenant_key(tenant_id)
        cipher = Fernet(key)
        return cipher.decrypt(encrypted_data)
```

### 2. Audit Logging

```python
# security/audit_logger.py
class AuditLogger:
    def __init__(self, log_storage: str):
        self.log_storage = log_storage
    
    async def log_event(
        self,
        tenant_id: str,
        user_id: str,
        event_type: str,
        resource_type: str,
        resource_id: str,
        details: Dict[str, Any],
        ip_address: str
    ):
        """Log audit event for compliance"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "tenant_id": tenant_id,
            "user_id": user_id,
            "event_type": event_type,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": details,
            "ip_address": ip_address,
            "event_id": str(uuid.uuid4())
        }
        
        # Store in immutable log storage
        await self._store_event(event)
        
        # Send to SIEM if configured
        if self.siem_configured():
            await self._send_to_siem(event)
```

## Monitoring & Analytics

### 1. Product Analytics

```python
# analytics/product_analytics.py
class ProductAnalytics:
    def __init__(self, analytics_db: str):
        self.analytics_db = analytics_db
    
    async def track_feature_usage(
        self,
        tenant_id: str,
        feature: str,
        metadata: Dict[str, Any]
    ):
        """Track feature usage for product insights"""
        event = {
            "tenant_id": tenant_id,
            "feature": feature,
            "timestamp": datetime.utcnow(),
            "metadata": metadata
        }
        
        # Store in analytics database
        await self._store_analytics_event(event)
    
    async def get_usage_insights(
        self,
        tenant_id: str,
        period: str = "30d"
    ) -> Dict[str, Any]:
        """Get usage insights for a tenant"""
        return {
            "most_used_features": await self._get_top_features(tenant_id, period),
            "usage_trend": await self._get_usage_trend(tenant_id, period),
            "cost_breakdown": await self._get_cost_breakdown(tenant_id, period),
            "recommendations": await self._get_recommendations(tenant_id)
        }
```

## Deployment Automation

### 1. Tenant Provisioning

```python
# deployment/tenant_provisioner.py
class TenantProvisioner:
    def __init__(self, k8s_client, terraform_client):
        self.k8s = k8s_client
        self.terraform = terraform_client
    
    async def provision_tenant(
        self,
        tenant_id: str,
        plan: str,
        config: Dict[str, Any]
    ):
        """Provision resources for a new tenant"""
        # Create namespace
        await self.k8s.create_namespace(f"tenant-{tenant_id}")
        
        # Deploy tenant-specific resources
        if plan == "enterprise":
            # Dedicated resources
            await self._provision_dedicated_resources(tenant_id, config)
        else:
            # Shared resources with quotas
            await self._provision_shared_resources(tenant_id, config)
        
        # Configure networking
        await self._configure_networking(tenant_id, config)
        
        # Set up monitoring
        await self._configure_monitoring(tenant_id)
        
        # Create initial API keys
        api_keys = await self._generate_api_keys(tenant_id)
        
        return {
            "tenant_id": tenant_id,
            "api_keys": api_keys,
            "endpoints": await self._get_tenant_endpoints(tenant_id),
            "status": "active"
        }
```

## Conclusion

This product architecture enables Inferloop to offer its synthetic data services as:

1. **Individual Products**: Each service can be sold separately with its own pricing
2. **Integrated Platform**: All services available under one subscription
3. **White-Label Solutions**: Partners can rebrand and resell
4. **Infrastructure Licensing**: The platform itself can be licensed

The architecture supports multiple deployment models (SaaS, on-premise, hybrid) and business models (subscription, usage-based, licensing) while maintaining security, scalability, and multi-tenancy.