# audio_synth/api/server.py
"""
FastAPI server for Audio Synthetic Data Framework
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import torch
import torchaudio
import tempfile
import os
import uuid
import json
from pathlib import Path
from datetime import datetime
import asyncio
import logging

from ..sdk.client import AudioSynthSDK
from ..core.utils.config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models for API
# ============================================================================

class AudioConfigModel(BaseModel):
    sample_rate: int = Field(default=22050, ge=8000, le=48000)
    duration: float = Field(default=5.0, ge=0.1, le=300.0)
    channels: int = Field(default=1, ge=1, le=2)
    format: str = Field(default="wav", regex="^(wav|mp3|flac|ogg)$")

class DemographicsModel(BaseModel):
    gender: Optional[str] = Field(None, regex="^(male|female|other)$")
    age_group: Optional[str] = Field(None, regex="^(child|adult|elderly)$")
    accent: Optional[str] = None
    language: Optional[str] = Field(default="en")

class GenerationRequest(BaseModel):
    method: str = Field(default="diffusion", regex="^(diffusion|gan|vae|tts|vocoder)$")
    prompt: Optional[str] = None
    num_samples: int = Field(default=1, ge=1, le=100)
    audio_config: AudioConfigModel = AudioConfigModel()
    privacy_level: str = Field(default="medium", regex="^(low|medium|high)$")
    seed: Optional[int] = None
    speaker_id: Optional[str] = None
    demographics: Optional[DemographicsModel] = None
    conditions: Optional[Dict[str, Any]] = None

class ValidationRequest(BaseModel):
    validators: List[str] = Field(default=["quality", "privacy", "fairness"])
    thresholds: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

class BatchGenerationRequest(BaseModel):
    requests: List[GenerationRequest]
    batch_id: Optional[str] = None

class GenerationResponse(BaseModel):
    job_id: str
    status: str
    num_samples: int
    audio_urls: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ValidationResponse(BaseModel):
    job_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ============================================================================
# Global Variables and Setup
# ============================================================================

app = FastAPI(
    title="Audio Synthetic Data API",
    description="Generate and validate synthetic audio data with privacy and fairness guarantees",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
sdk: Optional[AudioSynthSDK] = None
jobs: Dict[str, JobStatus] = {}
temp_dir = tempfile.mkdtemp()

# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize SDK and create necessary directories"""
    global sdk
    
    try:
        # Initialize SDK
        config_path = os.getenv("AUDIO_SYNTH_CONFIG", None)
        sdk = AudioSynthSDK(config_path)
        
        # Create output directory
        os.makedirs(temp_dir, exist_ok=True)
        
        logger.info("Audio Synthesis API server started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources"""
    import shutil
    
    try:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Audio Synthesis API server shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Audio Synthetic Data API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "generate": "/api/v1/generate",
            "validate": "/api/v1/validate",
            "jobs": "/api/v1/jobs",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "sdk_initialized": sdk is not None,
        "active_jobs": len([j for j in jobs.values() if j.status in ["pending", "running"]])
    }

@app.post("/api/v1/generate", response_model=GenerationResponse)
async def generate_audio(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate synthetic audio samples"""
    
    if sdk is None:
        raise HTTPException(status_code=500, detail="SDK not initialized")
    
    # Create job
    job_id = str(uuid.uuid4())
    job = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.utcnow()
    )
    jobs[job_id] = job
    
    # Start background generation task
    background_tasks.add_task(run_generation_task, job_id, request)
    
    return GenerationResponse(
        job_id=job_id,
        status="pending",
        num_samples=request.num_samples
    )

@app.post("/api/v1/generate/batch", response_model=GenerationResponse)
async def generate_audio_batch(request: BatchGenerationRequest, background_tasks: BackgroundTasks):
    """Generate multiple batches of synthetic audio samples"""
    
    if sdk is None:
        raise HTTPException(status_code=500, detail="SDK not initialized")
    
    # Create batch job
    batch_id = request.batch_id or str(uuid.uuid4())
    job = JobStatus(
        job_id=batch_id,
        status="pending",
        created_at=datetime.utcnow()
    )
    jobs[batch_id] = job
    
    # Start background batch generation task
    background_tasks.add_task(run_batch_generation_task, batch_id, request)
    
    total_samples = sum(req.num_samples for req in request.requests)
    
    return GenerationResponse(
        job_id=batch_id,
        status="pending",
        num_samples=total_samples
    )

@app.post("/api/v1/validate", response_model=ValidationResponse)
async def validate_audio(
    request: ValidationRequest,
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """Validate uploaded audio files"""
    
    if sdk is None:
        raise HTTPException(status_code=500, detail="SDK not initialized")
    
    # Create job
    job_id = str(uuid.uuid4())
    job = JobStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.utcnow()
    )
    jobs[job_id] = job
    
    # Start background validation task
    background_tasks.add_task(run_validation_task, job_id, request, files)
    
    return ValidationResponse(
        job_id=job_id,
        status="pending"
    )

@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get status of a specific job"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]

@app.get("/api/v1/jobs", response_model=List[JobStatus])
async def list_jobs(status: Optional[str] = None, limit: int = 100):
    """List all jobs with optional status filter"""
    
    filtered_jobs = list(jobs.values())
    
    if status:
        filtered_jobs = [job for job in filtered_jobs if job.status == status]
    
    # Sort by creation time (newest first) and limit
    filtered_jobs.sort(key=lambda x: x.created_at, reverse=True)
    
    return filtered_jobs[:limit]

@app.delete("/api/v1/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a completed job and its associated files"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.status in ["pending", "running"]:
        raise HTTPException(status_code=400, detail="Cannot delete running job")
    
    # Clean up associated files
    job_dir = Path(temp_dir) / job_id
    if job_dir.exists():
        import shutil
        shutil.rmtree(job_dir)
    
    # Remove job from memory
    del jobs[job_id]
    
    return {"message": "Job deleted successfully"}

@app.get("/api/v1/jobs/{job_id}/download/{filename}")
async def download_file(job_id: str, filename: str):
    """Download generated audio file"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    file_path = Path(temp_dir) / job_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="audio/wav"
    )

@app.get("/api/v1/models")
async def list_models():
    """List available generation models"""
    
    return {
        "generation_methods": [
            {
                "name": "diffusion",
                "description": "Diffusion-based audio generation",
                "capabilities": ["text-to-speech", "music", "sound-effects"],
                "privacy_levels": ["low", "medium", "high"]
            },
            {
                "name": "gan",
                "description": "GAN-based audio generation",
                "capabilities": ["voice-synthesis", "music"],
                "privacy_levels": ["low", "medium", "high"]
            },
            {
                "name": "vae",
                "description": "VAE-based audio generation",
                "capabilities": ["voice-conversion", "style-transfer"],
                "privacy_levels": ["medium", "high"]
            },
            {
                "name": "tts",
                "description": "Text-to-speech synthesis",
                "capabilities": ["speech-synthesis"],
                "privacy_levels": ["low", "medium", "high"]
            }
        ],
        "validators": [
            {
                "name": "quality",
                "description": "Audio quality assessment",
                "metrics": ["snr", "spectral_centroid", "realism_score"]
            },
            {
                "name": "privacy",
                "description": "Privacy preservation assessment",
                "metrics": ["speaker_anonymity", "privacy_leakage"]
            },
            {
                "name": "fairness",
                "description": "Fairness across demographics",
                "metrics": ["demographic_parity", "diversity_score"]
            }
        ]
    }

@app.get("/api/v1/config")
async def get_config():
    """Get current API configuration"""
    
    if sdk is None:
        raise HTTPException(status_code=500, detail="SDK not initialized")
    
    return {
        "audio_config": sdk.config.get("audio", {}),
        "generation_config": sdk.config.get("generation", {}),
        "validation_config": sdk.config.get("validation", {}),
        "supported_formats": ["wav", "mp3", "flac", "ogg"],
        "max_duration": 300.0,
        "max_samples_per_request": 100
    }

# ============================================================================
# Background Task Functions
# ============================================================================

async def run_generation_task(job_id: str, request: GenerationRequest):
    """Background task for audio generation"""
    
    job = jobs[job_id]
    job.status = "running"
    
    try:
        # Prepare generation parameters
        conditions = request.conditions or {}
        
        if request.speaker_id:
            conditions['speaker_id'] = request.speaker_id
        
        if request.demographics:
            conditions['demographics'] = request.demographics.dict(exclude_none=True)
        
        generation_params = {
            'conditions': conditions if conditions else None
        }
        
        if request.seed:
            generation_params['seed'] = request.seed
            torch.manual_seed(request.seed)
        
        # Generate audio
        job.progress = 0.2
        audios = sdk.generate(
            method=request.method,
            prompt=request.prompt,
            num_samples=request.num_samples,
            **generation_params
        )
        
        job.progress = 0.8
        
        # Save generated audio files
        job_dir = Path(temp_dir) / job_id
        job_dir.mkdir(exist_ok=True)
        
        audio_urls = []
        metadata_list = []
        
        for i, audio in enumerate(audios):
            filename = f"generated_{i+1:03d}.wav"
            filepath = job_dir / filename
            
            # Save audio
            torchaudio.save(
                str(filepath), 
                audio.unsqueeze(0), 
                request.audio_config.sample_rate
            )
            
            audio_urls.append(f"/api/v1/jobs/{job_id}/download/{filename}")
            
            # Prepare metadata
            metadata = {
                'filename': filename,
                'method': request.method,
                'prompt': request.prompt,
                'audio_config': request.audio_config.dict(),
                'privacy_level': request.privacy_level,
                'generation_params': generation_params
            }
            metadata_list.append(metadata)
        
        # Save metadata
        metadata_file = job_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        # Update job status
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.progress = 1.0
        job.result = {
            "audio_urls": audio_urls,
            "metadata": metadata_list,
            "num_samples": len(audios)
        }
        
    except Exception as e:
        logger.error(f"Generation task failed for job {job_id}: {e}")
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.utcnow()

async def run_batch_generation_task(batch_id: str, request: BatchGenerationRequest):
    """Background task for batch audio generation"""
    
    job = jobs[batch_id]
    job.status = "running"
    
    try:
        all_audio_urls = []
        all_metadata = []
        total_requests = len(request.requests)
        
        # Create batch directory
        batch_dir = Path(temp_dir) / batch_id
        batch_dir.mkdir(exist_ok=True)
        
        for req_idx, gen_request in enumerate(request.requests):
            # Update progress
            job.progress = req_idx / total_requests
            
            # Generate for this request
            conditions = gen_request.conditions or {}
            
            if gen_request.speaker_id:
                conditions['speaker_id'] = gen_request.speaker_id
            
            if gen_request.demographics:
                conditions['demographics'] = gen_request.demographics.dict(exclude_none=True)
            
            generation_params = {
                'conditions': conditions if conditions else None
            }
            
            if gen_request.seed:
                generation_params['seed'] = gen_request.seed
                torch.manual_seed(gen_request.seed)
            
            # Generate audio for this batch
            audios = sdk.generate(
                method=gen_request.method,
                prompt=gen_request.prompt,
                num_samples=gen_request.num_samples,
                **generation_params
            )
            
            # Save generated audio files
            for i, audio in enumerate(audios):
                filename = f"batch_{req_idx+1:03d}_sample_{i+1:03d}.wav"
                filepath = batch_dir / filename
                
                # Save audio
                torchaudio.save(
                    str(filepath), 
                    audio.unsqueeze(0), 
                    gen_request.audio_config.sample_rate
                )
                
                audio_url = f"/api/v1/jobs/{batch_id}/download/{filename}"
                all_audio_urls.append(audio_url)
                
                # Prepare metadata
                metadata = {
                    'batch_index': req_idx,
                    'sample_index': i,
                    'filename': filename,
                    'method': gen_request.method,
                    'prompt': gen_request.prompt,
                    'audio_config': gen_request.audio_config.dict(),
                    'privacy_level': gen_request.privacy_level,
                    'generation_params': generation_params
                }
                all_metadata.append(metadata)
        
        # Save batch metadata
        metadata_file = batch_dir / "batch_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        # Update job status
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.progress = 1.0
        job.result = {
            "audio_urls": all_audio_urls,
            "metadata": all_metadata,
            "num_samples": len(all_audio_urls),
            "num_batches": total_requests
        }
        
    except Exception as e:
        logger.error(f"Batch generation task failed for job {batch_id}: {e}")
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.utcnow()

async def run_validation_task(job_id: str, request: ValidationRequest, files: List[UploadFile]):
    """Background task for audio validation"""
    
    job = jobs[job_id]
    job.status = "running"
    
    try:
        audios = []
        metadata_list = []
        
        # Create job directory
        job_dir = Path(temp_dir) / job_id
        job_dir.mkdir(exist_ok=True)
        
        # Process uploaded files
        for i, file in enumerate(files):
            job.progress = (i / len(files)) * 0.5  # First half for file processing
            
            # Save uploaded file
            file_path = job_dir / file.filename
            content = await file.read()
            
            with open(file_path, 'wb') as f:
                f.write(content)
            
            # Load audio
            try:
                audio, sample_rate = torchaudio.load(str(file_path))
                audio = audio.squeeze()  # Remove batch dimension
                audios.append(audio)
                
                # Prepare metadata
                metadata = request.metadata or {}
                metadata.update({
                    'filename': file.filename,
                    'sample_rate': sample_rate,
                    'duration': len(audio) / sample_rate,
                    'channels': audio.dim()
                })
                metadata_list.append(metadata)
                
            except Exception as e:
                logger.warning(f"Failed to load audio file {file.filename}: {e}")
                continue
        
        if not audios:
            raise ValueError("No valid audio files found")
        
        # Run validation
        job.progress = 0.7
        validation_results = sdk.validate(
            audios=audios,
            metadata=metadata_list,
            validators=request.validators
        )
        
        # Calculate summary statistics
        summary = calculate_validation_summary(validation_results, request.thresholds)
        
        # Save validation results
        results_file = job_dir / "validation_results.json"
        results_data = {
            'validation_results': validation_results,
            'summary': summary,
            'thresholds': request.thresholds,
            'files_processed': len(audios)
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Update job status
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.progress = 1.0
        job.result = {
            "validation_results": validation_results,
            "summary": summary,
            "files_processed": len(audios)
        }
        
    except Exception as e:
        logger.error(f"Validation task failed for job {job_id}: {e}")
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.utcnow()

def calculate_validation_summary(validation_results: Dict[str, List[Dict]], 
                                thresholds: Optional[Dict[str, float]]) -> Dict[str, Any]:
    """Calculate summary statistics for validation results"""
    
    summary = {}
    default_thresholds = {
        'quality': 0.7,
        'privacy': 0.8,
        'fairness': 0.75
    }
    
    thresholds = thresholds or default_thresholds
    
    for validator_name, results_list in validation_results.items():
        if not results_list:
            continue
        
        threshold = thresholds.get(validator_name, 0.5)
        
        # Calculate statistics for each metric
        all_metrics = {}
        for result in results_list:
            for metric, value in result.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        metrics_stats = {}
        for metric, values in all_metrics.items():
            metrics_stats[metric] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
        
        # Calculate pass/fail counts
        passed = 0
        failed = 0
        
        for result in results_list:
            avg_score = sum(result.values()) / len(result) if result else 0
            if avg_score >= threshold:
                passed += 1
            else:
                failed += 1
        
        summary[validator_name] = {
            'total_samples': len(results_list),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(results_list) if results_list else 0,
            'threshold': threshold,
            'metrics': metrics_stats
        }
    
    return summary

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# ============================================================================
# Main Application Runner
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "audio_synth.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )