# inferloop-synthetic/api/app.py
"""
Inferloop Synthetic Data REST API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import io
import json
import uuid
from pathlib import Path
import tempfile
import asyncio
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sdk import GeneratorFactory, SyntheticDataConfig, SyntheticDataValidator, GenerationResult
from api.middleware import RateLimiter, LoggingMiddleware, SecurityMiddleware, ErrorTracker
from api.auth.auth_handler import AuthHandler
from api.endpoints import streaming, profiling, cache, batch, versioning, benchmark, privacy

app = FastAPI(
    title="Inferloop Synthetic Data API",
    description="REST API for synthetic data generation using multiple libraries",
    version="0.1.0"
)

# Add middleware
app.add_middleware(ErrorTracker)
app.add_middleware(SecurityMiddleware)
app.add_middleware(RateLimiter)
app.add_middleware(LoggingMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(streaming.router)
app.include_router(profiling.router)
app.include_router(cache.router)
app.include_router(batch.router)
app.include_router(versioning.router)
app.include_router(benchmark.router)
app.include_router(privacy.router)

# In-memory storage for jobs (in production, use Redis or database)
jobs_storage = {}


# Pydantic models
class GenerationRequest(BaseModel):
    config: Dict[str, Any]
    num_samples: Optional[int] = None
    validate_output: bool = True


class GenerationJob(BaseModel):
    job_id: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    config: Dict[str, Any]
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_path: Optional[str] = None


class ValidationRequest(BaseModel):
    real_data_url: Optional[str] = None
    synthetic_data_url: Optional[str] = None


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}


# Generator information
@app.get("/generators")
async def list_generators():
    """List available generators and their capabilities"""
    return {
        "generators": GeneratorFactory.list_generators(),
        "models": {
            "sdv": ["gaussian_copula", "ctgan", "copula_gan", "tvae"],
            "ctgan": ["ctgan", "tvae"],
            "ydata": ["wgan_gp", "cramer_gan", "dragan"]
        }
    }


# Data upload and preview
@app.post("/data/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload dataset for synthetic data generation"""
    
    if not file.filename.endswith(('.csv', '.json', '.parquet')):
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    # Save uploaded file temporarily
    temp_dir = Path(tempfile.gettempdir()) / "inferloop_uploads"
    temp_dir.mkdir(exist_ok=True)
    
    file_id = str(uuid.uuid4())
    file_path = temp_dir / f"{file_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Load and analyze data
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file_path)
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        
        # Data summary
        summary = {
            "file_id": file_id,
            "filename": file.filename,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "sample_data": df.head().to_dict('records')
        }
        
        return summary
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


@app.get("/data/{file_id}/preview")
async def preview_data(file_id: str, rows: int = 10):
    """Preview uploaded dataset"""
    
    temp_dir = Path(tempfile.gettempdir()) / "inferloop_uploads"
    file_paths = list(temp_dir.glob(f"{file_id}_*"))
    
    if not file_paths:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = file_paths[0]
    
    try:
        # Load data based on file extension
        if file_path.name.endswith('.csv'):
            df = pd.read_csv(file_path, nrows=rows)
        elif file_path.name.endswith('.json'):
            df = pd.read_json(file_path)
            df = df.head(rows)
        elif file_path.name.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            df = df.head(rows)
        
        return {
            "preview": df.to_dict('records'),
            "shape": list(df.shape),
            "columns": list(df.columns)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading file: {str(e)}")


# Synchronous generation
@app.post("/generate/sync")
async def generate_sync(request: GenerationRequest, file_id: str):
    """Generate synthetic data synchronously"""
    
    # Load data
    temp_dir = Path(tempfile.gettempdir()) / "inferloop_uploads"
    file_paths = list(temp_dir.glob(f"{file_id}_*"))
    
    if not file_paths:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = file_paths[0]
    
    try:
        # Load data
        if file_path.name.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.name.endswith('.json'):
            data = pd.read_json(file_path)
        elif file_path.name.endswith('.parquet'):
            data = pd.read_parquet(file_path)
        
        # Create configuration
        config = SyntheticDataConfig(**request.config)
        if request.num_samples:
            config.num_samples = request.num_samples
        
        # Generate synthetic data
        generator = GeneratorFactory.create_generator(config)
        result = generator.fit_generate(data, request.num_samples)
        
        # Validation
        validation_results = None
        if request.validate_output:
            validator = SyntheticDataValidator(data, result.synthetic_data)
            validation_results = validator.validate_all()
        
        # Return synthetic data as JSON
        return {
            "synthetic_data": result.synthetic_data.to_dict('records'),
            "metadata": {
                "generation_time": result.generation_time,
                "model_info": result.model_info,
                "config": result.config.to_dict()
            },
            "validation": validation_results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# Asynchronous generation
@app.post("/generate/async")
async def generate_async(background_tasks: BackgroundTasks, request: GenerationRequest, file_id: str):
    """Start asynchronous synthetic data generation"""
    
    job_id = str(uuid.uuid4())
    
    # Create job entry
    job = GenerationJob(
        job_id=job_id,
        status="pending",
        config=request.config,
        created_at=datetime.now()
    )
    
    jobs_storage[job_id] = job
    
    # Start background generation
    background_tasks.add_task(
        _generate_background,
        job_id,
        file_id,
        request
    )
    
    return {"job_id": job_id, "status": "pending"}


async def _generate_background(job_id: str, file_id: str, request: GenerationRequest):
    """Background task for data generation"""
    
    try:
        # Update job status
        jobs_storage[job_id].status = "running"
        
        # Load data
        temp_dir = Path(tempfile.gettempdir()) / "inferloop_uploads"
        file_paths = list(temp_dir.glob(f"{file_id}_*"))
        
        if not file_paths:
            raise Exception("Input file not found")
        
        file_path = file_paths[0]
        
        # Load data based on file type
        if file_path.name.endswith('.csv'):
            data = pd.read_csv(file_path)
        elif file_path.name.endswith('.json'):
            data = pd.read_json(file_path)
        elif file_path.name.endswith('.parquet'):
            data = pd.read_parquet(file_path)
        
        # Create configuration and generate
        config = SyntheticDataConfig(**request.config)
        if request.num_samples:
            config.num_samples = request.num_samples
        
        generator = GeneratorFactory.create_generator(config)
        result = generator.fit_generate(data, request.num_samples)
        
        # Save result
        results_dir = Path(tempfile.gettempdir()) / "inferloop_results"
        results_dir.mkdir(exist_ok=True)
        
        result_path = results_dir / f"{job_id}_result.csv"
        result.save(result_path)
        
        # Update job
        jobs_storage[job_id].status = "completed"
        jobs_storage[job_id].completed_at = datetime.now()
        jobs_storage[job_id].result_path = str(result_path)
    
    except Exception as e:
        jobs_storage[job_id].status = "failed"
        jobs_storage[job_id].error_message = str(e)
        jobs_storage[job_id].completed_at = datetime.now()


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and results"""
    
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_storage[job_id]
    
    response = {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at,
        "completed_at": job.completed_at
    }
    
    if job.status == "failed":
        response["error_message"] = job.error_message
    elif job.status == "completed":
        response["download_url"] = f"/jobs/{job_id}/download"
    
    return response


@app.get("/jobs/{job_id}/download")
async def download_result(job_id: str):
    """Download generated synthetic data"""
    
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_storage[job_id]
    
    if job.status != "completed" or not job.result_path:
        raise HTTPException(status_code=400, detail="Result not available")
    
    result_path = Path(job.result_path)
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        result_path,
        media_type="text/csv",
        filename=f"synthetic_data_{job_id}.csv"
    )


# Validation endpoint
@app.post("/validate")
async def validate_data(real_file_id: str, synthetic_file_id: str):
    """Validate synthetic data against real data"""
    
    temp_dir = Path(tempfile.gettempdir()) / "inferloop_uploads"
    
    # Load real data
    real_file_paths = list(temp_dir.glob(f"{real_file_id}_*"))
    if not real_file_paths:
        raise HTTPException(status_code=404, detail="Real data file not found")
    
    # Load synthetic data
    synthetic_file_paths = list(temp_dir.glob(f"{synthetic_file_id}_*"))
    if not synthetic_file_paths:
        raise HTTPException(status_code=404, detail="Synthetic data file not found")
    
    try:
        # Load datasets
        real_data = pd.read_csv(real_file_paths[0])
        synthetic_data = pd.read_csv(synthetic_file_paths[0])
        
        # Run validation
        validator = SyntheticDataValidator(real_data, synthetic_data)
        results = validator.validate_all()
        
        return {
            "validation_results": results,
            "report": validator.generate_report()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


# Configuration templates
@app.get("/config/templates")
async def get_config_templates():
    """Get configuration templates for different generators"""
    
    templates = {
        "sdv_gaussian_copula": {
            "generator_type": "sdv",
            "model_type": "gaussian_copula",
            "num_samples": 1000,
            "hyperparameters": {},
            "epochs": 300,
            "batch_size": 500
        },
        "sdv_ctgan": {
            "generator_type": "sdv",
            "model_type": "ctgan",
            "num_samples": 1000,
            "hyperparameters": {},
            "epochs": 300,
            "batch_size": 500
        },
        "ctgan": {
            "generator_type": "ctgan",
            "model_type": "ctgan",
            "num_samples": 1000,
            "hyperparameters": {},
            "epochs": 300,
            "batch_size": 500
        },
        "ydata_wgan": {
            "generator_type": "ydata",
            "model_type": "wgan_gp",
            "num_samples": 1000,
            "hyperparameters": {
                "noise_dim": 32,
                "layers_dim": 128
            },
            "epochs": 500,
            "batch_size": 128,
            "learning_rate": 2e-4
        }
    }
    
    return {"templates": templates}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)