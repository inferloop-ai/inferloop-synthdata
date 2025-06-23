"""
Streaming endpoints for large dataset processing
"""

import os
import asyncio
import tempfile
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

from sdk.streaming import StreamingSyntheticGenerator, StreamingValidator
from sdk.base import SyntheticDataConfig
from api.deps import get_current_user
from api.middleware.rate_limiter import rate_limit
from api.middleware.security_middleware import SecurityMiddleware

router = APIRouter(prefix="/streaming", tags=["streaming"])

# Global storage for streaming jobs
streaming_jobs = {}


class StreamingJobRequest(BaseModel):
    """Request model for streaming job"""
    generator_type: str
    model_type: str
    chunk_size: int = Field(default=10000, ge=1000, le=1000000)
    sample_ratio: float = Field(default=1.0, gt=0, le=1.0)
    output_format: str = Field(default='csv', regex='^(csv|parquet)$')
    model_params: Optional[Dict[str, Any]] = {}


class StreamingJobResponse(BaseModel):
    """Response model for streaming job"""
    job_id: str
    status: str
    message: str
    stream_url: Optional[str] = None
    download_url: Optional[str] = None


class StreamingJobStatus(BaseModel):
    """Status of streaming job"""
    job_id: str
    status: str
    progress: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/generate", response_model=StreamingJobResponse)
@rate_limit(requests_per_minute=5)
async def generate_streaming(
    file: UploadFile = File(...),
    config: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user = Depends(get_current_user)
):
    """
    Generate synthetic data for large files using streaming processing
    """
    # Parse configuration
    try:
        config_dict = json.loads(config)
        job_request = StreamingJobRequest(**config_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")
    
    # Validate file
    security_middleware = SecurityMiddleware(max_file_size=5 * 1024 * 1024 * 1024)  # 5GB limit
    validation_result = await security_middleware.validate_file_upload(file)
    
    if not validation_result['valid']:
        raise HTTPException(
            status_code=400,
            detail=f"File validation failed: {', '.join(validation_result['errors'])}"
        )
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        content = await file.read()
        tmp.write(content)
        input_path = tmp.name
    
    # Initialize job
    streaming_jobs[job_id] = {
        'status': 'initializing',
        'progress': {
            'chunks_processed': 0,
            'total_chunks': 0,
            'rows_generated': 0
        },
        'input_path': input_path,
        'output_path': None,
        'config': job_request
    }
    
    # Start background processing
    background_tasks.add_task(
        process_streaming_job,
        job_id,
        input_path,
        job_request
    )
    
    return StreamingJobResponse(
        job_id=job_id,
        status='accepted',
        message='Streaming job started',
        stream_url=f'/streaming/jobs/{job_id}/stream'
    )


async def process_streaming_job(job_id: str, input_path: str, config: StreamingJobRequest):
    """Process streaming job in background"""
    try:
        # Update status
        streaming_jobs[job_id]['status'] = 'processing'
        
        # Create output path
        output_dir = tempfile.mkdtemp(prefix='inferloop_output_')
        output_path = os.path.join(
            output_dir, 
            f'synthetic_data.{config.output_format}'
        )
        
        # Create streaming generator
        generator_config = SyntheticDataConfig(
            generator_type=config.generator_type,
            model_type=config.model_type,
            num_samples=1000,  # Per chunk
            model_params=config.model_params
        )
        
        streaming_generator = StreamingSyntheticGenerator(
            config=generator_config,
            chunk_size=config.chunk_size
        )
        
        # Process with progress updates
        def update_progress(chunks_done, total_chunks):
            streaming_jobs[job_id]['progress'] = {
                'chunks_processed': chunks_done,
                'total_chunks': total_chunks,
                'percentage': round((chunks_done / total_chunks) * 100, 2)
            }
        
        # Generate synthetic data
        result = streaming_generator.generate_streaming(
            input_path=input_path,
            output_path=output_path,
            sample_ratio=config.sample_ratio,
            output_format=config.output_format,
            progress_callback=update_progress
        )
        
        # Update job completion
        streaming_jobs[job_id].update({
            'status': 'completed',
            'output_path': output_path,
            'result': {
                'output_path': output_path,
                'total_rows': result.metadata['total_rows'],
                'chunks_processed': result.metadata['chunks_processed'],
                'download_url': f'/streaming/jobs/{job_id}/download'
            }
        })
        
    except Exception as e:
        streaming_jobs[job_id].update({
            'status': 'failed',
            'error': str(e)
        })
    
    finally:
        # Clean up input file
        if os.path.exists(input_path):
            os.unlink(input_path)


@router.get("/jobs/{job_id}", response_model=StreamingJobStatus)
async def get_job_status(job_id: str, current_user = Depends(get_current_user)):
    """Get status of streaming job"""
    if job_id not in streaming_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = streaming_jobs[job_id]
    return StreamingJobStatus(
        job_id=job_id,
        status=job['status'],
        progress=job['progress'],
        result=job.get('result'),
        error=job.get('error')
    )


@router.get("/jobs/{job_id}/stream")
async def stream_results(job_id: str, current_user = Depends(get_current_user)):
    """Stream results as they are generated"""
    if job_id not in streaming_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def generate():
        """Generate streaming response"""
        while True:
            job = streaming_jobs.get(job_id)
            if not job:
                break
            
            # Send current status
            status_data = {
                'status': job['status'],
                'progress': job['progress']
            }
            
            yield f"data: {json.dumps(status_data)}\n\n"
            
            # Check if completed
            if job['status'] in ['completed', 'failed']:
                if job['status'] == 'completed':
                    yield f"data: {json.dumps({'result': job.get('result')})}\n\n"
                else:
                    yield f"data: {json.dumps({'error': job.get('error')})}\n\n"
                break
            
            # Wait before next update
            await asyncio.sleep(1)
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/jobs/{job_id}/download")
async def download_results(job_id: str, current_user = Depends(get_current_user)):
    """Download generated synthetic data"""
    if job_id not in streaming_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = streaming_jobs[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed. Current status: {job['status']}"
        )
    
    output_path = job.get('output_path')
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    # Determine media type
    media_type = 'text/csv' if output_path.endswith('.csv') else 'application/octet-stream'
    filename = os.path.basename(output_path)
    
    return FileResponse(
        path=output_path,
        media_type=media_type,
        filename=filename
    )


@router.post("/validate", response_model=Dict[str, Any])
@rate_limit(requests_per_minute=10)
async def validate_streaming(
    original_file: UploadFile = File(...),
    synthetic_file: UploadFile = File(...),
    sample_size: int = Form(default=10000, ge=1000, le=1000000),
    current_user = Depends(get_current_user)
):
    """
    Validate synthetic data against original using streaming sampling
    """
    # Save uploaded files temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp1:
        content1 = await original_file.read()
        tmp1.write(content1)
        original_path = tmp1.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp2:
        content2 = await synthetic_file.read()
        tmp2.write(content2)
        synthetic_path = tmp2.name
    
    try:
        # Create streaming validator
        validator = StreamingValidator()
        
        # Run validation
        validation_results = validator.validate_streaming(
            original_path=original_path,
            synthetic_path=synthetic_path,
            sample_size=sample_size
        )
        
        return validation_results
        
    finally:
        # Clean up temporary files
        for path in [original_path, synthetic_path]:
            if os.path.exists(path):
                os.unlink(path)


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, current_user = Depends(get_current_user)):
    """Cancel or clean up streaming job"""
    if job_id not in streaming_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = streaming_jobs[job_id]
    
    # Update status
    if job['status'] == 'processing':
        job['status'] = 'cancelled'
    
    # Clean up files
    for path_key in ['input_path', 'output_path']:
        path = job.get(path_key)
        if path and os.path.exists(path):
            try:
                if os.path.isdir(path):
                    import shutil
                    shutil.rmtree(path)
                else:
                    os.unlink(path)
            except Exception:
                pass
    
    # Remove from jobs
    del streaming_jobs[job_id]
    
    return {"message": "Job cancelled and cleaned up"}


@router.get("/jobs", response_model=list)
async def list_jobs(
    status: Optional[str] = None,
    current_user = Depends(get_current_user)
):
    """List all streaming jobs for current user"""
    jobs = []
    
    for job_id, job in streaming_jobs.items():
        if status and job['status'] != status:
            continue
            
        jobs.append({
            'job_id': job_id,
            'status': job['status'],
            'progress': job['progress'],
            'created_at': job.get('created_at', 'unknown')
        })
    
    return jobs


# Cleanup old jobs periodically
async def cleanup_old_jobs():
    """Clean up completed jobs older than 1 hour"""
    import time
    current_time = time.time()
    
    jobs_to_remove = []
    for job_id, job in streaming_jobs.items():
        if job['status'] in ['completed', 'failed', 'cancelled']:
            # Check if older than 1 hour
            created_at = job.get('created_at', current_time)
            if current_time - created_at > 3600:
                jobs_to_remove.append(job_id)
    
    # Remove old jobs
    for job_id in jobs_to_remove:
        # Clean up files
        job = streaming_jobs[job_id]
        for path_key in ['input_path', 'output_path']:
            path = job.get(path_key)
            if path and os.path.exists(path):
                try:
                    if os.path.isdir(path):
                        import shutil
                        shutil.rmtree(path)
                    else:
                        os.unlink(path)
                except Exception:
                    pass
        
        # Remove from jobs
        del streaming_jobs[job_id]


# Schedule cleanup task
import json
import uuid
from datetime import datetime