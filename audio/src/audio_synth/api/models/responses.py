# audio_synth/api/models/responses.py
"""
API response models
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

class GenerationResponse(BaseModel):
    """Response for audio generation request"""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Job status")
    num_samples: int = Field(..., description="Number of samples requested")
    audio_urls: Optional[List[str]] = Field(None, description="URLs to generated audio files")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Generation metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")

class ValidationResponse(BaseModel):
    """Response for validation request"""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Job status")
    results: Optional[Dict[str, List[Dict[str, float]]]] = Field(None, description="Validation results")
    summary: Optional[Dict[str, Any]] = Field(None, description="Validation summary")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")

class JobStatusResponse(BaseModel):
    """Response for job status request"""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Job progress (0-1)")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    result: Optional[Dict[str, Any]] = Field(None, description="Job result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")

class AudioFileInfo(BaseModel):
    """Information about an audio file"""
    filename: str = Field(..., description="File name")
    url: str = Field(..., description="Download URL")
    format: str = Field(..., description="Audio format")
    duration: float = Field(..., description="Duration in seconds")
    sample_rate: int = Field(..., description="Sample rate")
    channels: int = Field(..., description="Number of channels")
    size_bytes: int = Field(..., description="File size in bytes")

class MetricsResponse(BaseModel):
    """Response containing validation metrics"""
    metrics: Dict[str, float] = Field(..., description="Validation metrics")
    threshold: float = Field(..., description="Threshold used for validation")
    passed: bool = Field(..., description="Whether validation passed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Validation timestamp")

class BatchMetricsResponse(BaseModel):
    """Response containing batch validation metrics"""
    individual_results: List[MetricsResponse] = Field(..., description="Individual validation results")
    summary: Dict[str, Any] = Field(..., description="Batch validation summary")
    pass_rate: float = Field(..., description="Overall pass rate")

class ModelInfo(BaseModel):
    """Information about a generation model"""
    name: str = Field(..., description="Model name")
    description: str = Field(..., description="Model description")
    capabilities: List[str] = Field(..., description="Model capabilities")
    privacy_levels: List[PrivacyLevel] = Field(..., description="Supported privacy levels")
    supported_languages: Optional[List[str]] = Field(None, description="Supported languages")
    max_duration: Optional[float] = Field(None, description="Maximum generation duration")

class ValidatorInfo(BaseModel):
    """Information about a validator"""
    name: str = Field(..., description="Validator name")
    description: str = Field(..., description="Validator description")
    metrics: List[str] = Field(..., description="Metrics provided by validator")
    default_threshold: float = Field(..., description="Default threshold")

class ModelsResponse(BaseModel):
    """Response listing available models and validators"""
    generation_methods: List[ModelInfo] = Field(..., description="Available generation methods")
    validators: List[ValidatorInfo] = Field(..., description="Available validators")

class ConfigResponse(BaseModel):
    """Response containing API configuration"""
    audio_config: Dict[str, Any] = Field(..., description="Default audio configuration")
    generation_config: Dict[str, Any] = Field(..., description="Default generation configuration")
    validation_config: Dict[str, Any] = Field(..., description="Default validation configuration")
    supported_formats: List[str] = Field(..., description="Supported audio formats")
    max_duration: float = Field(..., description="Maximum audio duration")
    max_samples_per_request: int = Field(..., description="Maximum samples per request")
    rate_limits: Optional[Dict[str, Any]] = Field(None, description="Rate limiting information")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(..., description="API version")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")
    dependencies: Optional[Dict[str, str]] = Field(None, description="Dependency status")

class StatsResponse(BaseModel):
    """Statistics response"""
    requests_total: int = Field(..., description="Total requests processed")
    requests_last_hour: int = Field(..., description="Requests in last hour")
    success_rate: float = Field(..., description="Overall success rate")
    avg_processing_time: float = Field(..., description="Average processing time")
    active_jobs: int = Field(..., description="Currently active jobs")
    queue_length: int = Field(..., description="Current queue length")