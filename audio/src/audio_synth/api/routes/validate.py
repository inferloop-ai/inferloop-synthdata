# audio_synth/api/routes/validate.py
"""
Audio validation API routes
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Depends
from typing import List
import uuid
import logging

from ..models.requests import ValidationRequest
from ..models.responses import ValidationResponse
from ...sdk.client import AudioSynthSDK

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/validate", tags=["validation"])

async def get_sdk() -> AudioSynthSDK:
    return AudioSynthSDK()

@router.post("/", response_model=ValidationResponse)
async def validate_audio(
    request: ValidationRequest,
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
    sdk: AudioSynthSDK = Depends(get_sdk)
):
    """Validate uploaded audio files"""
    
    if not files:
        raise HTTPException(status_code=400, detail="No audio files provided")
    
    # Create job
    job_id = str(uuid.uuid4())
    
    # Start background validation task
    background_tasks.add_task(
        _run_validation_task,
        job_id,
        request,
        files,
        sdk
    )
    
    return ValidationResponse(
        job_id=job_id,
        status="pending"
    )

async def _run_validation_task(job_id: str, request: ValidationRequest, files: List[UploadFile], sdk: AudioSynthSDK):
    """Background task for validation"""
    # Implementation would be similar to the server.py version
    pass

