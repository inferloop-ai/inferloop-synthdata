"""
Unified API Routes for Tabular Service

This module defines all the API routes using the unified infrastructure
components instead of module-specific implementations.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field, validator

# Import from unified infrastructure
from unified_cloud_deployment.auth import get_current_user, User, require_permissions
from unified_cloud_deployment.database import get_db_session
from unified_cloud_deployment.storage import get_storage_client, StorageClient
from unified_cloud_deployment.cache import get_cache_client, CacheClient
from unified_cloud_deployment.monitoring import track_request, log_event
from unified_cloud_deployment.billing import track_usage, check_quota

# Import service-specific components
from tabular.sdk.factory import GeneratorFactory
from tabular.sdk.base import SyntheticDataConfig
from tabular.sdk.validator import SyntheticDataValidator
from tabular.infrastructure import TabularServiceAdapter, ServiceTier

# Create router
router = APIRouter(
    prefix="/api/tabular",
    tags=["tabular"],
    dependencies=[Depends(get_current_user)]
)

# Initialize service adapter
adapter = TabularServiceAdapter()


# Request/Response Models
class SourceDataConfig(BaseModel):
    """Configuration for source data"""
    type: str = Field(..., description="Type of source data: 'inline', 'storage', 'database'")
    data: Optional[Dict[str, Any]] = Field(None, description="Inline data (for type='inline')")
    path: Optional[str] = Field(None, description="Storage path (for type='storage')")
    query: Optional[str] = Field(None, description="SQL query (for type='database')")
    
    @validator('data')
    def validate_inline_data(cls, v, values):
        if values.get('type') == 'inline' and not v:
            raise ValueError("Data is required when type is 'inline'")
        return v


class GenerationRequest(BaseModel):
    """Request for synthetic data generation"""
    source: SourceDataConfig
    algorithm: str = Field("sdv", description="Generation algorithm")
    num_rows: int = Field(100, ge=1, le=10000000)
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    config: Dict[str, Any] = Field(default_factory=dict, description="Algorithm-specific config")
    output_format: str = Field("json", description="Output format: json, csv, parquet")
    async_mode: bool = Field(False, description="Run generation asynchronously")


class GenerationResponse(BaseModel):
    """Response for generation request"""
    job_id: str
    status: str
    created_at: datetime
    data_url: Optional[str] = None
    preview: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    estimated_completion: Optional[datetime] = None


class ValidationRequest(BaseModel):
    """Request for data validation"""
    real_data_path: str
    synthetic_data_path: str
    metrics: List[str] = Field(
        default=["statistical", "privacy"],
        description="Validation metrics to compute"
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class ValidationResponse(BaseModel):
    """Response for validation request"""
    job_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    report_url: Optional[str] = None
    summary: Optional[Dict[str, float]] = None


class JobStatusResponse(BaseModel):
    """Response for job status check"""
    job_id: str
    status: str
    progress: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    result_url: Optional[str] = None
    error: Optional[str] = None


# Endpoints
@router.post("/generate", response_model=GenerationResponse)
@track_request("generate")
async def generate_synthetic_data(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db_session),
    storage: StorageClient = Depends(get_storage_client),
    cache: CacheClient = Depends(get_cache_client)
):
    """
    Generate synthetic tabular data.
    
    Requires permissions: tabular:generate
    """
    # Check permissions
    require_permissions(current_user, ["tabular:generate"])
    
    # Check user tier and limits
    user_tier = ServiceTier(current_user.tier)
    tier_config = adapter.get_api_config()["rate_limiting"]
    
    # Validate algorithm availability
    available_algos = [
        algo.name for algo in adapter.ALGORITHMS
        if algo.enabled and algo.min_tier.value <= user_tier.value
    ]
    
    if request.algorithm not in available_algos:
        raise HTTPException(
            status_code=403,
            detail=f"Algorithm '{request.algorithm}' requires {request.algorithm} tier or higher"
        )
    
    # Check row limits
    row_limits = {
        ServiceTier.STARTER: 100_000,
        ServiceTier.PROFESSIONAL: 1_000_000,
        ServiceTier.BUSINESS: 5_000_000,
        ServiceTier.ENTERPRISE: -1
    }
    
    limit = row_limits.get(user_tier, 100_000)
    if limit > 0 and request.num_rows > limit:
        raise HTTPException(
            status_code=403,
            detail=f"Row limit exceeded. {user_tier.value} tier allows up to {limit:,} rows"
        )
    
    # Check quota
    quota_ok = await check_quota(
        user_id=current_user.id,
        service="tabular",
        metric="rows_generated",
        amount=request.num_rows
    )
    
    if not quota_ok:
        raise HTTPException(
            status_code=429,
            detail="Monthly quota exceeded. Please upgrade your plan."
        )
    
    # Generate job ID
    job_id = f"tab-gen-{uuid.uuid4()}"
    
    # Create job record
    await db.execute(
        """
        INSERT INTO tabular.generation_jobs 
        (id, user_id, tenant_id, algorithm, num_rows, status, config, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        job_id, current_user.id, current_user.tenant_id,
        request.algorithm, request.num_rows, "pending",
        request.config, datetime.utcnow()
    )
    
    # Log event
    await log_event(
        user_id=current_user.id,
        event_type="generation_started",
        metadata={
            "job_id": job_id,
            "algorithm": request.algorithm,
            "num_rows": request.num_rows
        }
    )
    
    if request.async_mode:
        # Queue for async processing
        background_tasks.add_task(
            _process_generation_async,
            job_id, request, current_user, db, storage, cache
        )
        
        return GenerationResponse(
            job_id=job_id,
            status="queued",
            created_at=datetime.utcnow(),
            metadata={"algorithm": request.algorithm, "num_rows": request.num_rows},
            estimated_completion=datetime.utcnow().replace(
                minute=datetime.utcnow().minute + 5
            )
        )
    else:
        # Process synchronously
        result = await _process_generation_sync(
            job_id, request, current_user, db, storage, cache
        )
        return result


@router.post("/validate", response_model=ValidationResponse)
@track_request("validate")
async def validate_synthetic_data(
    request: ValidationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db_session),
    storage: StorageClient = Depends(get_storage_client)
):
    """
    Validate synthetic data quality.
    
    Requires permissions: tabular:validate
    """
    require_permissions(current_user, ["tabular:validate"])
    
    # Check if validation is available for user tier
    user_tier = ServiceTier(current_user.tier)
    if user_tier == ServiceTier.STARTER and "privacy" in request.metrics:
        raise HTTPException(
            status_code=403,
            detail="Privacy validation requires Professional tier or higher"
        )
    
    # Generate job ID
    job_id = f"tab-val-{uuid.uuid4()}"
    
    # Create job record
    await db.execute(
        """
        INSERT INTO tabular.validation_jobs
        (id, user_id, real_data_path, synthetic_data_path, metrics, status, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        job_id, current_user.id, request.real_data_path,
        request.synthetic_data_path, request.metrics, "pending",
        datetime.utcnow()
    )
    
    # Queue for async processing
    background_tasks.add_task(
        _process_validation_async,
        job_id, request, current_user, db, storage
    )
    
    return ValidationResponse(
        job_id=job_id,
        status="queued",
        summary={"metrics_requested": len(request.metrics)}
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db_session)
):
    """Get status of a generation or validation job"""
    # Check if job belongs to user
    result = await db.fetch_one(
        """
        SELECT id, user_id, status, created_at, updated_at, result_url, error
        FROM (
            SELECT id, user_id, status, created_at, updated_at, output_path as result_url, error
            FROM tabular.generation_jobs
            WHERE id = $1
            UNION ALL
            SELECT id, user_id, status, created_at, updated_at, report_path as result_url, error
            FROM tabular.validation_jobs
            WHERE id = $1
        ) combined
        """,
        job_id
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if result["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Calculate progress for running jobs
    progress = None
    if result["status"] == "running":
        # Simple progress estimation based on time
        elapsed = (datetime.utcnow() - result["created_at"]).total_seconds()
        progress = min(elapsed / 60, 0.9) * 100  # Estimate 1 minute completion
    
    return JobStatusResponse(
        job_id=job_id,
        status=result["status"],
        progress=progress,
        created_at=result["created_at"],
        updated_at=result["updated_at"],
        result_url=result["result_url"],
        error=result["error"]
    )


@router.get("/algorithms")
async def list_algorithms(
    current_user: User = Depends(get_current_user)
):
    """List available algorithms for user's tier"""
    user_tier = ServiceTier(current_user.tier)
    
    algorithms = []
    for algo in adapter.ALGORITHMS:
        available = algo.enabled and algo.min_tier.value <= user_tier.value
        algorithms.append({
            "name": algo.name,
            "available": available,
            "min_tier": algo.min_tier.value,
            "description": f"{algo.name.upper()} synthetic data generation",
            "resource_multiplier": algo.resources_multiplier
        })
    
    return {
        "user_tier": user_tier.value,
        "algorithms": algorithms
    }


@router.get("/usage/summary")
@track_request("usage")
async def get_usage_summary(
    period: str = Query("30d", description="Time period: 24h, 7d, 30d"),
    current_user: User = Depends(get_current_user),
    db=Depends(get_db_session)
):
    """Get usage summary for current user"""
    # Starter tier doesn't have access to usage stats
    if ServiceTier(current_user.tier) == ServiceTier.STARTER:
        raise HTTPException(
            status_code=403,
            detail="Usage statistics require Professional tier or higher"
        )
    
    # Parse period
    period_map = {
        "24h": "1 day",
        "7d": "7 days", 
        "30d": "30 days"
    }
    
    interval = period_map.get(period, "30 days")
    
    # Get generation stats
    gen_stats = await db.fetch_one(
        f"""
        SELECT 
            COUNT(*) as total_jobs,
            SUM(num_rows) as total_rows,
            AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_duration,
            COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_jobs
        FROM tabular.generation_jobs
        WHERE user_id = $1 
        AND created_at >= NOW() - INTERVAL '{interval}'
        """,
        current_user.id
    )
    
    # Get validation stats
    val_stats = await db.fetch_one(
        f"""
        SELECT 
            COUNT(*) as total_validations,
            array_agg(DISTINCT unnest(metrics)) as metrics_used
        FROM tabular.validation_jobs
        WHERE user_id = $1
        AND created_at >= NOW() - INTERVAL '{interval}'
        """,
        current_user.id
    )
    
    # Get billing info
    billing = await db.fetch_one(
        """
        SELECT 
            SUM(amount) as total_cost,
            SUM(CASE WHEN metric = 'rows_generated' THEN amount END) as rows_cost
        FROM billing.usage_records
        WHERE user_id = $1 
        AND service = 'tabular'
        AND created_at >= NOW() - INTERVAL $2
        """,
        current_user.id, interval
    )
    
    return {
        "period": period,
        "generation": {
            "total_jobs": gen_stats["total_jobs"] or 0,
            "successful_jobs": gen_stats["successful_jobs"] or 0,
            "total_rows": gen_stats["total_rows"] or 0,
            "avg_duration_seconds": round(gen_stats["avg_duration"] or 0, 2)
        },
        "validation": {
            "total_validations": val_stats["total_validations"] or 0,
            "metrics_used": val_stats["metrics_used"] or []
        },
        "billing": {
            "total_cost": round(billing["total_cost"] or 0, 2),
            "currency": "USD"
        },
        "limits": adapter.get_api_config()["rate_limiting"]
    }


# Helper functions
async def _process_generation_sync(
    job_id: str,
    request: GenerationRequest,
    user: User,
    db,
    storage: StorageClient,
    cache: CacheClient
) -> GenerationResponse:
    """Process generation synchronously"""
    try:
        # Update job status
        await db.execute(
            "UPDATE tabular.generation_jobs SET status = 'running' WHERE id = $1",
            job_id
        )
        
        # Load source data
        if request.source.type == "inline":
            source_data = request.source.data
        elif request.source.type == "storage":
            source_data = await storage.read_json(request.source.path)
        else:
            raise ValueError(f"Unsupported source type: {request.source.type}")
        
        # Create generator
        config = SyntheticDataConfig(
            generator_type=request.algorithm,
            num_samples=request.num_rows,
            random_seed=request.seed,
            **request.config
        )
        
        generator = GeneratorFactory.create_generator(config)
        
        # Generate data
        result = generator.fit_generate(source_data)
        
        # Save output
        output_path = f"output/{user.tenant_id}/{job_id}/synthetic_data.{request.output_format}"
        await storage.write(
            output_path,
            result.synthetic_data,
            format=request.output_format
        )
        
        # Update job
        await db.execute(
            """
            UPDATE tabular.generation_jobs 
            SET status = 'completed', output_path = $2, updated_at = $3
            WHERE id = $1
            """,
            job_id, output_path, datetime.utcnow()
        )
        
        # Track usage for billing
        await track_usage(
            user_id=user.id,
            service="tabular",
            metric="rows_generated",
            amount=request.num_rows,
            metadata={"job_id": job_id, "algorithm": request.algorithm}
        )
        
        # Get preview (first 10 rows)
        preview = None
        if request.num_rows <= 1000:
            preview = result.synthetic_data[:10]
        
        return GenerationResponse(
            job_id=job_id,
            status="completed",
            created_at=datetime.utcnow(),
            data_url=output_path,
            preview=preview,
            metadata={
                "rows_generated": len(result.synthetic_data),
                "quality_metrics": result.quality_metrics
            }
        )
        
    except Exception as e:
        # Update job with error
        await db.execute(
            """
            UPDATE tabular.generation_jobs 
            SET status = 'failed', error = $2, updated_at = $3
            WHERE id = $1
            """,
            job_id, str(e), datetime.utcnow()
        )
        raise


async def _process_generation_async(
    job_id: str,
    request: GenerationRequest,
    user: User,
    db,
    storage: StorageClient,
    cache: CacheClient
):
    """Process generation asynchronously"""
    try:
        await _process_generation_sync(job_id, request, user, db, storage, cache)
    except Exception as e:
        # Error handling is done in sync function
        pass


async def _process_validation_async(
    job_id: str,
    request: ValidationRequest,
    user: User,
    db,
    storage: StorageClient
):
    """Process validation asynchronously"""
    try:
        # Update status
        await db.execute(
            "UPDATE tabular.validation_jobs SET status = 'running' WHERE id = $1",
            job_id
        )
        
        # Load data
        real_data = await storage.read(request.real_data_path)
        synthetic_data = await storage.read(request.synthetic_data_path)
        
        # Run validation
        validator = SyntheticDataValidator()
        results = {}
        
        for metric in request.metrics:
            if metric == "statistical":
                results[metric] = validator.calculate_statistical_similarity(
                    real_data, synthetic_data
                )
            elif metric == "privacy":
                results[metric] = validator.calculate_privacy_metrics(
                    real_data, synthetic_data
                )
            elif metric == "utility":
                results[metric] = validator.calculate_utility_metrics(
                    real_data, synthetic_data
                )
        
        # Save report
        report_path = f"output/{user.tenant_id}/{job_id}/validation_report.json"
        await storage.write_json(report_path, {
            "job_id": job_id,
            "metrics": results,
            "summary": {
                metric: results[metric].get("overall_score", 0)
                for metric in results
            },
            "created_at": datetime.utcnow().isoformat()
        })
        
        # Update job
        await db.execute(
            """
            UPDATE tabular.validation_jobs
            SET status = 'completed', report_path = $2, results = $3, updated_at = $4
            WHERE id = $1
            """,
            job_id, report_path, results, datetime.utcnow()
        )
        
        # Track usage
        await track_usage(
            user_id=user.id,
            service="tabular",
            metric="validations_performed",
            amount=len(request.metrics),
            metadata={"job_id": job_id}
        )
        
    except Exception as e:
        await db.execute(
            """
            UPDATE tabular.validation_jobs
            SET status = 'failed', error = $2, updated_at = $3
            WHERE id = $1
            """,
            job_id, str(e), datetime.utcnow()
        )