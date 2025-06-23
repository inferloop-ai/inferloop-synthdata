"""
Batch processing API endpoints
"""

import os
import uuid
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from sdk.batch import BatchProcessor, BatchBuilder, BatchDataset, BatchStatus, load_batch_from_config
from sdk.base import SyntheticDataConfig
from api.deps import get_current_user
from api.middleware.rate_limiter import rate_limit

router = APIRouter(prefix="/batch", tags=["batch"])

# Global storage for batch jobs
batch_jobs = {}


class BatchJobRequest(BaseModel):
    """Request model for batch job"""
    datasets: List[Dict[str, Any]]
    default_config: Optional[Dict[str, Any]] = None
    max_workers: int = Field(default=4, ge=1, le=16)
    fail_fast: bool = False
    cache_models: bool = True


class BatchJobResponse(BaseModel):
    """Response model for batch job"""
    job_id: str
    status: str
    message: str
    total_datasets: int


class BatchJobStatus(BaseModel):
    """Status of batch job"""
    job_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_datasets: int
    processed: int
    successful: int
    failed: int
    progress: Dict[str, Any]


@router.post("/create", response_model=BatchJobResponse)
@rate_limit(requests_per_minute=10)
async def create_batch_job(
    request: BatchJobRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """Create a new batch processing job"""
    # Validate datasets
    if not request.datasets:
        raise HTTPException(status_code=400, detail="No datasets provided")
    
    # Create batch builder
    builder = BatchBuilder()
    
    # Set default config if provided
    if request.default_config:
        builder.set_default_config(request.default_config)
    
    # Add datasets
    for dataset_info in request.datasets:
        try:
            builder.add_dataset(
                input_path=dataset_info['input_path'],
                output_path=dataset_info['output_path'],
                config=dataset_info.get('config'),
                dataset_id=dataset_info.get('id'),
                priority=dataset_info.get('priority', 0)
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid dataset configuration: {str(e)}"
            )
    
    # Build dataset list
    datasets = builder.build()
    
    # Create job ID
    job_id = f"batch_{uuid.uuid4().hex[:8]}"
    
    # Initialize job storage
    batch_jobs[job_id] = {
        'status': BatchStatus.PENDING,
        'started_at': datetime.now(),
        'total_datasets': len(datasets),
        'processed': 0,
        'successful': 0,
        'failed': 0,
        'processor': None,
        'result': None,
        'progress': {}
    }
    
    # Start background processing
    background_tasks.add_task(
        process_batch_job,
        job_id,
        datasets,
        request.max_workers,
        request.fail_fast,
        request.cache_models
    )
    
    return BatchJobResponse(
        job_id=job_id,
        status=BatchStatus.PENDING.value,
        message=f"Batch job created with {len(datasets)} datasets",
        total_datasets=len(datasets)
    )


async def process_batch_job(
    job_id: str,
    datasets: List[BatchDataset],
    max_workers: int,
    fail_fast: bool,
    cache_models: bool
):
    """Process batch job in background"""
    try:
        # Update status
        batch_jobs[job_id]['status'] = BatchStatus.RUNNING
        
        # Create processor
        processor = BatchProcessor(
            max_workers=max_workers,
            fail_fast=fail_fast,
            cache_models=cache_models
        )
        
        batch_jobs[job_id]['processor'] = processor
        
        # Progress callback
        def update_progress(dataset_id: str, progress_info):
            batch_jobs[job_id]['progress'][dataset_id] = progress_info.to_dict()
            
            # Update counters
            completed = sum(
                1 for p in batch_jobs[job_id]['progress'].values()
                if p['stage'] in ['completed', 'failed']
            )
            successful = sum(
                1 for p in batch_jobs[job_id]['progress'].values()
                if p['stage'] == 'completed'
            )
            failed = sum(
                1 for p in batch_jobs[job_id]['progress'].values()
                if p['stage'] == 'failed'
            )
            
            batch_jobs[job_id]['processed'] = completed
            batch_jobs[job_id]['successful'] = successful
            batch_jobs[job_id]['failed'] = failed
        
        # Process batch
        result = processor.process_batch(datasets, progress_callback=update_progress)
        
        # Update final status
        batch_jobs[job_id]['status'] = result.status
        batch_jobs[job_id]['completed_at'] = result.completed_at
        batch_jobs[job_id]['result'] = result
        
    except Exception as e:
        batch_jobs[job_id]['status'] = BatchStatus.FAILED
        batch_jobs[job_id]['error'] = str(e)


@router.get("/jobs/{job_id}", response_model=BatchJobStatus)
async def get_job_status(job_id: str, current_user = Depends(get_current_user)):
    """Get status of batch job"""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = batch_jobs[job_id]
    
    return BatchJobStatus(
        job_id=job_id,
        status=job['status'].value if hasattr(job['status'], 'value') else job['status'],
        started_at=job['started_at'],
        completed_at=job.get('completed_at'),
        total_datasets=job['total_datasets'],
        processed=job['processed'],
        successful=job['successful'],
        failed=job['failed'],
        progress=job['progress']
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, current_user = Depends(get_current_user)):
    """Cancel a running batch job"""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = batch_jobs[job_id]
    
    if job['status'] not in [BatchStatus.PENDING, BatchStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in status: {job['status']}"
        )
    
    # Cancel processor if running
    if job['processor']:
        job['processor'].cancel()
    
    job['status'] = BatchStatus.CANCELLED
    job['completed_at'] = datetime.now()
    
    return {"message": "Job cancelled"}


@router.get("/jobs/{job_id}/results")
async def get_job_results(job_id: str, current_user = Depends(get_current_user)):
    """Get detailed results of completed batch job"""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = batch_jobs[job_id]
    
    if job['status'] not in [BatchStatus.COMPLETED, BatchStatus.PARTIAL, BatchStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job['status']}"
        )
    
    result = job.get('result')
    if not result:
        raise HTTPException(status_code=404, detail="Results not found")
    
    return {
        'job_id': job_id,
        'status': result.status.value,
        'summary': result.to_dict(),
        'errors': result.errors,
        'datasets': [
            {
                'id': dataset_id,
                'success': dataset_id not in result.errors,
                'error': result.errors.get(dataset_id)
            }
            for dataset_id in job['progress'].keys()
        ]
    }


@router.post("/upload")
@rate_limit(requests_per_minute=5)
async def upload_batch_config(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    max_workers: int = Form(default=4),
    current_user = Depends(get_current_user)
):
    """Upload batch configuration file"""
    # Validate file type
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
        content = await file.read()
        tmp.write(content)
        config_path = tmp.name
    
    try:
        # Load batch configuration
        datasets = load_batch_from_config(config_path)
        
        # Create job
        job_id = f"batch_{uuid.uuid4().hex[:8]}"
        
        batch_jobs[job_id] = {
            'status': BatchStatus.PENDING,
            'started_at': datetime.now(),
            'total_datasets': len(datasets),
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'processor': None,
            'result': None,
            'progress': {}
        }
        
        # Start processing
        background_tasks.add_task(
            process_batch_job,
            job_id,
            datasets,
            max_workers,
            fail_fast=False,
            cache_models=True
        )
        
        return BatchJobResponse(
            job_id=job_id,
            status=BatchStatus.PENDING.value,
            message=f"Batch job created from config with {len(datasets)} datasets",
            total_datasets=len(datasets)
        )
        
    finally:
        # Clean up temp file
        if os.path.exists(config_path):
            os.unlink(config_path)


@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 100,
    current_user = Depends(get_current_user)
):
    """List all batch jobs"""
    jobs = []
    
    for job_id, job_info in batch_jobs.items():
        job_status = job_info['status']
        
        # Filter by status if provided
        if status and job_status.value != status:
            continue
        
        jobs.append({
            'job_id': job_id,
            'status': job_status.value if hasattr(job_status, 'value') else str(job_status),
            'started_at': job_info['started_at'],
            'completed_at': job_info.get('completed_at'),
            'total_datasets': job_info['total_datasets'],
            'processed': job_info['processed'],
            'successful': job_info['successful'],
            'failed': job_info['failed']
        })
    
    # Sort by start time (newest first)
    jobs.sort(key=lambda x: x['started_at'], reverse=True)
    
    return jobs[:limit]


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, current_user = Depends(get_current_user)):
    """Delete a completed batch job"""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = batch_jobs[job_id]
    
    if job['status'] == BatchStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete running job. Cancel it first."
        )
    
    del batch_jobs[job_id]
    
    return {"message": "Job deleted"}


@router.get("/templates/config")
async def get_batch_config_template(current_user = Depends(get_current_user)):
    """Get batch configuration template"""
    template = {
        "version": "1.0",
        "description": "Batch processing configuration template",
        "default_config": {
            "generator_type": "sdv",
            "model_type": "gaussian_copula",
            "num_samples": 1000,
            "epochs": 300,
            "batch_size": 500
        },
        "datasets": [
            {
                "id": "dataset_001",
                "input_path": "/path/to/input1.csv",
                "output_path": "/path/to/output1.csv",
                "priority": 1,
                "config": {
                    "num_samples": 2000
                }
            },
            {
                "id": "dataset_002",
                "input_path": "/path/to/input2.csv",
                "output_path": "/path/to/output2.csv",
                "priority": 0,
                "config": {}
            }
        ]
    }
    
    return template