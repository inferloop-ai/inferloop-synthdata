# audio_synth/api/routes/health.py
"""
Health check API routes
"""

from fastapi import APIRouter
from datetime import datetime
import psutil
import torch

from ..models.responses import HealthResponse

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    dependencies = {
        "pytorch": "available" if torch.cuda.is_available() else "cpu_only",
        "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB" if torch.cuda.is_available() else "n/a",
        "system_memory": f"{psutil.virtual_memory().total // (1024**3)}GB",
        "cpu_count": str(psutil.cpu_count())
    }
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime=None,  # Would track actual uptime
        dependencies=dependencies
    )

@router.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    return {"status": "ready"}

@router.get("/live")
async def liveness_check():
    """Liveness check for Kubernetes"""
    return {"status": "alive"}