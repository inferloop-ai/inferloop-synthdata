"""
Tabular API Application - Unified Infrastructure Version

This is the refactored version of the tabular API that uses the unified
cloud deployment infrastructure instead of module-specific implementations.
"""

import os
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import from unified infrastructure instead of local modules
from unified_cloud_deployment.auth import AuthMiddleware, get_current_user, User
from unified_cloud_deployment.monitoring import (
    TelemetryMiddleware,
    MetricsCollector,
    create_counter,
    create_histogram
)
from unified_cloud_deployment.storage import StorageClient, StorageError
from unified_cloud_deployment.cache import CacheClient
from unified_cloud_deployment.database import DatabaseClient, get_db_session
from unified_cloud_deployment.config import get_service_config
from unified_cloud_deployment.ratelimit import RateLimitMiddleware

# Import service-specific logic (these remain in tabular module)
from tabular.sdk.factory import GeneratorFactory
from tabular.sdk.base import SyntheticDataConfig, GenerationResult
from tabular.infrastructure.adapter import TabularServiceAdapter, ServiceTier


# Initialize metrics
rows_generated_counter = create_counter(
    "tabular_rows_generated_total",
    "Total number of synthetic rows generated",
    labels=["algorithm", "tier", "user_id"]
)

generation_duration_histogram = create_histogram(
    "tabular_generation_duration_seconds",
    "Duration of synthetic data generation",
    labels=["algorithm", "tier"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

# Service configuration
service_config = get_service_config()
service_tier = ServiceTier(service_config.get("tier", "starter"))
adapter = TabularServiceAdapter(service_tier)

# Initialize clients
storage_client = StorageClient(service_name="tabular")
cache_client = CacheClient(namespace="tabular")
metrics_collector = MetricsCollector(service_name="tabular")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print(f"Starting Tabular API Service (Tier: {service_tier.value})")
    
    # Initialize storage buckets
    await storage_client.ensure_bucket("input")
    await storage_client.ensure_bucket("output")
    await storage_client.ensure_bucket("models")
    
    # Warm up cache
    await cache_client.ping()
    
    # Register with service discovery
    # await service_registry.register("tabular", adapter.get_service_config())
    
    yield
    
    # Shutdown
    print("Shutting down Tabular API Service")
    await storage_client.close()
    await cache_client.close()


# Create FastAPI app
app = FastAPI(
    title="Inferloop Tabular API",
    description="Enterprise-grade synthetic tabular data generation",
    version=adapter.SERVICE_VERSION,
    lifespan=lifespan
)

# Add unified middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.inferloop.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    AuthMiddleware,
    service_name="tabular",
    required_permissions=["tabular:read"]
)

app.add_middleware(
    RateLimitMiddleware,
    service_name="tabular",
    tier_config=adapter.get_api_config()["rate_limiting"]
)

app.add_middleware(
    TelemetryMiddleware,
    service_name="tabular",
    enable_tracing=service_tier != ServiceTier.STARTER,
    enable_metrics=True
)


# Request/Response models
class GenerateRequest(BaseModel):
    """Request model for synthetic data generation"""
    source_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Source data for generation (if not using stored data)"
    )
    source_data_path: Optional[str] = Field(
        None,
        description="Path to source data in storage"
    )
    algorithm: str = Field(
        "sdv",
        description="Algorithm to use for generation"
    )
    num_rows: int = Field(
        100,
        description="Number of synthetic rows to generate",
        ge=1,
        le=1000000
    )
    config: Optional[Dict[str, Any]] = Field(
        None,
        description="Algorithm-specific configuration"
    )
    output_format: str = Field(
        "json",
        description="Output format (json, csv, parquet)"
    )


class GenerateResponse(BaseModel):
    """Response model for synthetic data generation"""
    job_id: str
    status: str
    data_path: Optional[str] = None
    data_preview: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any]
    billing: Dict[str, Any]


class ValidationRequest(BaseModel):
    """Request model for synthetic data validation"""
    real_data_path: str
    synthetic_data_path: str
    metrics: List[str] = ["statistical", "privacy", "utility"]


class ValidationResponse(BaseModel):
    """Response model for validation results"""
    job_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    report_path: Optional[str] = None


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "tabular",
        "version": adapter.SERVICE_VERSION,
        "tier": service_tier.value
    }


@app.get("/ready")
async def readiness_check(db=Depends(get_db_session)):
    """Readiness check endpoint"""
    try:
        # Check database connection
        await db.execute("SELECT 1")
        
        # Check cache connection
        await cache_client.ping()
        
        # Check storage connection
        await storage_client.check_connection()
        
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")


@app.post("/api/tabular/generate", response_model=GenerateResponse)
async def generate_synthetic_data(
    request: GenerateRequest,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db_session)
):
    """Generate synthetic tabular data"""
    import time
    start_time = time.time()
    
    try:
        # Check algorithm availability for tier
        available_algorithms = [
            algo.name for algo in adapter.ALGORITHMS
            if algo.enabled and algo.min_tier.value <= service_tier.value
        ]
        
        if request.algorithm not in available_algorithms:
            raise HTTPException(
                status_code=403,
                detail=f"Algorithm '{request.algorithm}' not available in {service_tier.value} tier"
            )
        
        # Check row limit for tier
        row_limits = {
            ServiceTier.STARTER: 100000,
            ServiceTier.PROFESSIONAL: 1000000,
            ServiceTier.BUSINESS: 5000000,
            ServiceTier.ENTERPRISE: -1  # Unlimited
        }
        
        limit = row_limits.get(service_tier, 100000)
        if limit > 0 and request.num_rows > limit:
            raise HTTPException(
                status_code=403,
                detail=f"Row limit exceeded. {service_tier.value} tier allows up to {limit} rows"
            )
        
        # Generate unique job ID
        import uuid
        job_id = f"tabular-{uuid.uuid4()}"
        
        # Check cache for similar request
        cache_key = f"generation:{current_user.tenant_id}:{request.algorithm}:{request.num_rows}"
        cached_result = await cache_client.get(cache_key)
        
        if cached_result and service_tier != ServiceTier.STARTER:
            return GenerateResponse(
                job_id=job_id,
                status="completed",
                data_path=cached_result["data_path"],
                metadata=cached_result["metadata"],
                billing={"cached": True, "cost": 0}
            )
        
        # Load source data
        if request.source_data:
            source_data = request.source_data
        elif request.source_data_path:
            source_data = await storage_client.read_json(request.source_data_path)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either source_data or source_data_path must be provided"
            )
        
        # Create generator configuration
        generator_config = SyntheticDataConfig(
            generator_type=request.algorithm,
            num_samples=request.num_rows,
            **request.config or {}
        )
        
        # Generate synthetic data
        generator = GeneratorFactory.create_generator(generator_config)
        result = await generator.generate_async(source_data)
        
        # Store output
        output_path = f"output/{current_user.tenant_id}/{job_id}/synthetic_data.{request.output_format}"
        await storage_client.write(output_path, result.synthetic_data, format=request.output_format)
        
        # Calculate generation duration
        duration = time.time() - start_time
        
        # Update metrics
        rows_generated_counter.labels(
            algorithm=request.algorithm,
            tier=service_tier.value,
            user_id=str(current_user.id)
        ).inc(request.num_rows)
        
        generation_duration_histogram.labels(
            algorithm=request.algorithm,
            tier=service_tier.value
        ).observe(duration)
        
        # Cache result
        cache_data = {
            "data_path": output_path,
            "metadata": result.metadata
        }
        await cache_client.set(cache_key, cache_data, ttl=3600)
        
        # Calculate billing
        billing_rate = adapter._get_billing_rate("rows")
        cost = request.num_rows * billing_rate
        
        # Store job information in database
        await db.execute(
            """
            INSERT INTO tabular.generation_jobs 
            (id, user_id, tenant_id, algorithm, num_rows, duration, cost, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            job_id, current_user.id, current_user.tenant_id,
            request.algorithm, request.num_rows, duration, cost, "completed"
        )
        
        # Return response with preview for small datasets
        preview = None
        if request.num_rows <= 100 and service_tier != ServiceTier.STARTER:
            preview = result.synthetic_data[:10] if isinstance(result.synthetic_data, list) else None
        
        return GenerateResponse(
            job_id=job_id,
            status="completed",
            data_path=output_path,
            data_preview=preview,
            metadata={
                "rows_generated": request.num_rows,
                "algorithm": request.algorithm,
                "duration_seconds": round(duration, 2),
                "quality_metrics": result.quality_metrics
            },
            billing={
                "rows": request.num_rows,
                "rate": billing_rate,
                "cost": round(cost, 4),
                "currency": "USD"
            }
        )
        
    except Exception as e:
        # Log error
        await metrics_collector.increment("tabular_errors_total", labels={
            "error_type": type(e).__name__,
            "algorithm": request.algorithm
        })
        
        # Re-raise HTTP exceptions
        if isinstance(e, HTTPException):
            raise e
        
        # Wrap other exceptions
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


@app.post("/api/tabular/validate", response_model=ValidationResponse)
async def validate_synthetic_data(
    request: ValidationRequest,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db_session)
):
    """Validate synthetic data quality"""
    try:
        # Generate job ID
        import uuid
        job_id = f"validation-{uuid.uuid4()}"
        
        # Load data from storage
        real_data = await storage_client.read(request.real_data_path)
        synthetic_data = await storage_client.read(request.synthetic_data_path)
        
        # Perform validation
        from tabular.sdk.validator import SyntheticDataValidator
        validator = SyntheticDataValidator()
        
        results = {}
        for metric in request.metrics:
            if metric == "statistical":
                results["statistical"] = await validator.validate_statistical(
                    real_data, synthetic_data
                )
            elif metric == "privacy":
                results["privacy"] = await validator.validate_privacy(
                    real_data, synthetic_data
                )
            elif metric == "utility":
                results["utility"] = await validator.validate_utility(
                    real_data, synthetic_data
                )
        
        # Generate report
        report_path = f"output/{current_user.tenant_id}/{job_id}/validation_report.json"
        await storage_client.write_json(report_path, results)
        
        # Calculate billing
        billing_rate = adapter._get_billing_rate("validations")
        cost = len(request.metrics) * billing_rate
        
        # Store job information
        await db.execute(
            """
            INSERT INTO tabular.validation_jobs 
            (id, user_id, tenant_id, metrics, cost, status)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            job_id, current_user.id, current_user.tenant_id,
            request.metrics, cost, "completed"
        )
        
        return ValidationResponse(
            job_id=job_id,
            status="completed",
            results=results if service_tier != ServiceTier.STARTER else None,
            report_path=report_path
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Validation failed: {str(e)}"
        )


@app.get("/api/tabular/algorithms")
async def list_algorithms(
    current_user: User = Depends(get_current_user)
):
    """List available algorithms for the user's tier"""
    user_tier = ServiceTier(current_user.tier)
    
    available_algorithms = []
    for algo in adapter.ALGORITHMS:
        if algo.enabled and algo.min_tier.value <= user_tier.value:
            available_algorithms.append({
                "name": algo.name,
                "available": True,
                "min_tier": algo.min_tier.value,
                "resources_multiplier": algo.resources_multiplier
            })
        else:
            available_algorithms.append({
                "name": algo.name,
                "available": False,
                "min_tier": algo.min_tier.value,
                "upgrade_required": algo.min_tier.value
            })
    
    return {
        "user_tier": user_tier.value,
        "algorithms": available_algorithms
    }


@app.get("/api/tabular/usage")
async def get_usage_stats(
    current_user: User = Depends(get_current_user),
    db=Depends(get_db_session)
):
    """Get usage statistics for the current user"""
    # Only available for professional tier and above
    if ServiceTier(current_user.tier) == ServiceTier.STARTER:
        raise HTTPException(
            status_code=403,
            detail="Usage statistics not available in Starter tier"
        )
    
    # Query usage from database
    result = await db.fetch_one(
        """
        SELECT 
            COUNT(*) as total_jobs,
            SUM(num_rows) as total_rows,
            SUM(cost) as total_cost,
            AVG(duration) as avg_duration
        FROM tabular.generation_jobs
        WHERE user_id = $1 
        AND created_at >= NOW() - INTERVAL '30 days'
        """,
        current_user.id
    )
    
    return {
        "period": "last_30_days",
        "statistics": {
            "total_jobs": result["total_jobs"] or 0,
            "total_rows_generated": result["total_rows"] or 0,
            "total_cost": round(result["total_cost"] or 0, 2),
            "average_duration_seconds": round(result["avg_duration"] or 0, 2)
        },
        "tier_limits": adapter.get_api_config()["rate_limiting"]
    }


# Metrics endpoint (Prometheus format)
@app.get("/metrics")
async def metrics():
    """Expose metrics in Prometheus format"""
    return await metrics_collector.get_metrics()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_unified:app",
        host="0.0.0.0",
        port=8000,
        reload=bool(os.getenv("DEBUG", False))
    )