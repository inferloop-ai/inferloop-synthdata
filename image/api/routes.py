from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional, Dict, Any
import json
from pathlib import Path
import asyncio

# Import your modules here
from realtime.profiler.generate_profile_json import ProfileGenerator
from generation.generate_diffusion import DiffusionGenerator

# Create routers for different functionality
profile_router = APIRouter(prefix="/profile", tags=["profiling"])
generation_router = APIRouter(prefix="/generate", tags=["generation"])
validation_router = APIRouter(prefix="/validate", tags=["validation"])
data_router = APIRouter(prefix="/data", tags=["data"])

# ===== PROFILE ROUTES =====

@profile_router.get("/sources")
async def get_profile_sources():
    """Get available profiling sources."""
    return {
        "sources": {
            "unsplash": {
                "description": "Fetch images from Unsplash API",
                "parameters": ["query", "count", "orientation"]
            },
            "webcam": {
                "description": "Capture from webcam",
                "parameters": ["device_id", "count", "interval"]
            },
            "upload": {
                "description": "Upload your own images",
                "parameters": ["files"]
            }
        }
    }

@profile_router.post("/stream/start")
async def start_profiling_stream(
    source: str,
    config: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Start a continuous profiling stream."""
    
    if source not in ["unsplash", "webcam"]:
        raise HTTPException(status_code=400, detail=f"Unsupported source: {source}")
    
    # Add background task
    stream_id = f"stream_{source}_{id(config)}"
    background_tasks.add_task(_run_profiling_stream, stream_id, source, config)
    
    return {
        "message": "Profiling stream started",
        "stream_id": stream_id,
        "source": source,
        "config": config
    }

async def _run_profiling_stream(stream_id: str, source: str, config: Dict):
    """Background task for continuous profiling."""
    # Implementation would depend on your specific streaming requirements
    pass

# ===== GENERATION ROUTES =====

@generation_router.get("/methods")
async def get_generation_methods():
    """Get available generation methods."""
    return {
        "methods": {
            "diffusion": {
                "description": "Stable Diffusion text-to-image",
                "parameters": ["prompts", "num_inference_steps", "guidance_scale"],
                "supports_conditioning": True
            },
            "gan": {
                "description": "GAN-based generation",
                "parameters": ["truncation_psi", "noise_mode"],
                "supports_conditioning": True
            },
            "simulation": {
                "description": "3D simulation engines",
                "parameters": ["environment", "weather", "time_of_day"],
                "supports_conditioning": False
            }
        }
    }

@generation_router.post("/preview")
async def generate_preview(
    method: str = "diffusion",
    prompt: str = "test image",
    num_images: int = 1
):
    """Generate a small preview batch for testing."""
    if num_images > 5:
        raise HTTPException(status_code=400, detail="Preview limited to 5 images")
    
    # Implementation would call appropriate generator
    return {
        "message": f"Generated {num_images} preview images",
        "method": method,
        "prompt": prompt
    }

# ===== VALIDATION ROUTES =====

@validation_router.get("/metrics")
async def get_validation_metrics():
    """Get available validation metrics."""
    return {
        "metrics": {
            "quality": ["fid", "ssim", "lpips", "psnr"],
            "diversity": ["entropy", "cluster_score"],
            "privacy": ["face_detection", "pii_detection"],
            "fairness": ["demographic_parity", "equalized_odds"]
        }
    }

@validation_router.post("/compare")
async def compare_datasets(
    dataset_a: str,
    dataset_b: str,
    metrics: List[str]
):
    """Compare two datasets using specified metrics."""
    
    # Validate datasets exist
    dataset_a_path = Path(f"./data/generated/{dataset_a}")
    dataset_b_path = Path(f"./data/generated/{dataset_b}")
    
    if not dataset_a_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset A not found: {dataset_a}")
    if not dataset_b_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset B not found: {dataset_b}")
    
    # Implementation would run comparison
    return {
        "comparison": f"{dataset_a} vs {dataset_b}",
        "metrics": metrics,
        "results": {}  # Actual comparison results would go here
    }

# ===== DATA ROUTES =====

@data_router.get("/stats")
async def get_data_statistics():
    """Get statistics about stored data."""
    stats = {
        "profiles": 0,
        "generated_datasets": 0,
        "total_images": 0,
        "storage_used": "0 MB"
    }
    
    # Count profiles
    profiles_dir = Path("./profiles")
    if profiles_dir.exists():
        stats["profiles"] = len(list(profiles_dir.glob("stream_*.json")))
    
    # Count datasets
    datasets_dir = Path("./data/generated")
    if datasets_dir.exists():
        for dataset_dir in datasets_dir.iterdir():
            if dataset_dir.is_dir():
                stats["generated_datasets"] += 1
                # Count images in this dataset
                images = list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpg"))
                stats["total_images"] += len(images)
    
    return stats

@data_router.delete("/cleanup")
async def cleanup_old_data(
    days_old: int = 7,
    confirm: bool = False
):
    """Clean up data older than specified days."""
    if not confirm:
        raise HTTPException(
            status_code=400, 
            detail="Set confirm=true to proceed with cleanup"
        )
    
    # Implementation would clean up old files
    return {
        "message": f"Cleaned up data older than {days_old} days",
        "files_removed": 0,
        "space_freed": "0 MB"
    }

# Create main router that includes all sub-routers
def create_router():
    main_router = APIRouter()
    
    main_router.include_router(profile_router)
    main_router.include_router(generation_router)
    main_router.include_router(validation_router)
    main_router.include_router(data_router)
    
    return main_router
