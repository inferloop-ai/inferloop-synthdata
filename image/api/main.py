from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json
import tempfile
from pathlib import Path
from typing import List, Optional, Dict
import cv2
import numpy as np
from PIL import Image
import io

from realtime.profiler.generate_profile_json import ProfileGenerator
from realtime.ingest_unsplash import UnsplashIngester
from realtime.ingest_webcam import WebcamIngester
from generation.generate_diffusion import DiffusionGenerator
from validation.validate_quality import QualityValidator

app = FastAPI(
    title="Agentic AI Synthetic Image Generator",
    description="Generate synthetic images for Agentic AI testing and validation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
profile_generator = ProfileGenerator()
diffusion_generator = None
quality_validator = QualityValidator()

@app.on_event("startup")
async def startup_event():
    """Initialize heavy components on startup."""
    global diffusion_generator
    try:
        diffusion_generator = DiffusionGenerator()
    except Exception as e:
        print(f"Warning: Could not initialize diffusion generator: {e}")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Agentic AI Synthetic Image Generator API", "status": "running"}

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "components": {
            "profile_generator": True,
            "diffusion_generator": diffusion_generator is not None,
            "quality_validator": True
        }
    }

# ===== PROFILING ENDPOINTS =====

@app.post("/profile/upload")
async def profile_uploaded_images(
    files: List[UploadFile] = File(...),
    source_name: str = "upload"
):
    """Profile uploaded images and generate distribution profile."""
    try:
        images = []
        
        # Process uploaded files
        for file in files:
            # Read image
            contents = await file.read()
            
            # Convert to numpy array
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                images.append(image)
        
        if not images:
            raise HTTPException(status_code=400, detail="No valid images provided")
        
        # Generate profile
        profile_path = profile_generator.process_and_save(images, source_name)
        profile = profile_generator.load_profile(source_name)
        
        return {
            "message": f"Profiled {len(images)} images",
            "source_name": source_name,
            "profile_path": profile_path,
            "profile_summary": {
                "sample_count": profile["metadata"]["sample_count"],
                "top_objects": profile.get("conditioning_profile", {})
                                     .get("generation_hints", {})
                                     .get("object_guidance", {})
                                     .get("preferred_classes", [])[:5],
                "scene_type": profile.get("conditioning_profile", {})
                                   .get("generation_hints", {})
                                   .get("scene_guidance", {})
                                   .get("preferred_scene", "unknown")
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling failed: {str(e)}")

@app.post("/profile/unsplash")
async def profile_unsplash_images(
    query: str = "urban",
    count: int = 10,
    source_name: Optional[str] = None
):
    """Profile images from Unsplash API."""
    try:
        if not source_name:
            source_name = f"unsplash_{query}"
        
        # Initialize Unsplash ingester
        ingester = UnsplashIngester()
        
        # Fetch images
        images = ingester.fetch_random_images(query=query, count=count)
        
        if not images:
            raise HTTPException(status_code=400, detail="Failed to fetch images from Unsplash")
        
        # Generate profile
        profile_path = profile_generator.process_and_save(images, source_name)
        profile = profile_generator.load_profile(source_name)
        
        return {
            "message": f"Profiled {len(images)} Unsplash images",
            "query": query,
            "source_name": source_name,
            "profile_path": profile_path,
            "profile_summary": {
                "sample_count": profile["metadata"]["sample_count"],
                "generation_guidance": profile.get("generation_guidance", {})
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unsplash profiling failed: {str(e)}")

@app.get("/profile/{source_name}")
async def get_profile(source_name: str):
    """Get an existing profile."""
    profile = profile_generator.load_profile(source_name)
    
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile '{source_name}' not found")
    
    return profile

@app.get("/profiles/list")
async def list_profiles():
    """List all available profiles."""
    profiles_dir = Path("./profiles")
    if not profiles_dir.exists():
        return {"profiles": []}
    
    profiles = []
    for file_path in profiles_dir.glob("stream_*.json"):
        if not file_path.name.endswith("_latest.json"):  # Skip timestamped files
            continue
            
        source_name = file_path.stem.replace("stream_", "")
        try:
            with open(file_path, 'r') as f:
                profile = json.load(f)
            
            profiles.append({
                "source_name": source_name,
                "timestamp": profile["metadata"]["generation_timestamp"],
                "sample_count": profile["metadata"]["sample_count"],
                "file_path": str(file_path)
            })
        except Exception as e:
            continue
    
    return {"profiles": profiles}

# ===== GENERATION ENDPOINTS =====

@app.post("/generate/diffusion")
async def generate_diffusion_images(
    num_images: int = 10,
    profile_source: Optional[str] = None,
    custom_prompts: Optional[List[str]] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    batch_size: int = 4
):
    """Generate images using Stable Diffusion."""
    if diffusion_generator is None:
        raise HTTPException(status_code=503, detail="Diffusion generator not available")
    
    try:
        if profile_source:
            # Generate from profile
            profile_path = f"./profiles/stream_{profile_source}.json"
            if not Path(profile_path).exists():
                raise HTTPException(status_code=404, detail=f"Profile '{profile_source}' not found")
            
            images = diffusion_generator.generate_from_profile(
                profile_path=profile_path,
                num_images=num_images,
                batch_size=batch_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
        elif custom_prompts:
            # Generate from custom prompts
            images = diffusion_generator.generate_from_prompts(
                prompts=custom_prompts,
                num_images=num_images,
                batch_size=batch_size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
        else:
            raise HTTPException(status_code=400, detail="Either profile_source or custom_prompts required")
        
        # Save images
        output_dir = "./data/generated"
        saved_paths = diffusion_generator.save_images(images, output_dir, "api_diffusion")
        
        return {
            "message": f"Generated {len(images)} images",
            "generation_method": "diffusion",
            "num_images": len(images),
            "saved_paths": saved_paths,
            "parameters": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "batch_size": batch_size
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate/batch")
async def generate_batch_with_validation(
    background_tasks: BackgroundTasks,
    num_images: int = 100,
    profile_source: Optional[str] = None,
    generation_method: str = "diffusion",
    validation_enabled: bool = True
):
    """Generate a large batch of images with optional validation."""
    
    # Add background task for generation
    task_id = f"batch_{num_images}_{generation_method}"
    
    background_tasks.add_task(
        _generate_batch_background,
        task_id,
        num_images,
        profile_source,
        generation_method,
        validation_enabled
    )
    
    return {
        "message": "Batch generation started",
        "task_id": task_id,
        "num_images": num_images,
        "status": "queued"
    }

async def _generate_batch_background(
    task_id: str,
    num_images: int,
    profile_source: Optional[str],
    generation_method: str,
    validation_enabled: bool
):
    """Background task for batch generation."""
    try:
        # This would typically be implemented with a proper task queue like Celery
        # For now, just run synchronously
        
        if generation_method == "diffusion" and diffusion_generator:
            if profile_source:
                profile_path = f"./profiles/stream_{profile_source}.json"
                images = diffusion_generator.generate_from_profile(
                    profile_path=profile_path,
                    num_images=num_images
                )
            else:
                default_prompts = ["high quality realistic image"]
                images = diffusion_generator.generate_from_prompts(
                    prompts=default_prompts,
                    num_images=num_images
                )
            
            # Save images
            output_dir = f"./data/generated/{task_id}"
            saved_paths = diffusion_generator.save_images(images, output_dir, "batch")
            
            # Run validation if enabled
            if validation_enabled:
                validation_results = quality_validator.validate_batch(saved_paths)
                
                # Save validation report
                report_path = f"{output_dir}/validation_report.json"
                with open(report_path, 'w') as f:
                    json.dump(validation_results, f, indent=2)
        
    except Exception as e:
        print(f"Background task {task_id} failed: {e}")

# ===== VALIDATION ENDPOINTS =====

@app.post("/validate/quality")
async def validate_image_quality(
    files: List[UploadFile] = File(...),
    reference_files: Optional[List[UploadFile]] = File(None)
):
    """Validate quality of uploaded images."""
    try:
        # Process uploaded files
        image_paths = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files
            for i, file in enumerate(files):
                file_path = Path(temp_dir) / f"image_{i}.jpg"
                
                contents = await file.read()
                with open(file_path, 'wb') as f:
                    f.write(contents)
                
                image_paths.append(str(file_path))
            
            # Process reference files if provided
            reference_paths = []
            if reference_files:
                for i, file in enumerate(reference_files):
                    file_path = Path(temp_dir) / f"reference_{i}.jpg"
                    
                    contents = await file.read()
                    with open(file_path, 'wb') as f:
                        f.write(contents)
                    
                    reference_paths.append(str(file_path))
            
            # Run validation
            if reference_paths:
                results = quality_validator.validate_against_reference(image_paths, reference_paths)
            else:
                results = quality_validator.validate_batch(image_paths)
            
            return {
                "message": f"Validated {len(image_paths)} images",
                "validation_results": results
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

# ===== UTILITY ENDPOINTS =====

@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """Download a generated file."""
    full_path = Path(file_path)
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(full_path),
        media_type='application/octet-stream',
        filename=full_path.name
    )

@app.get("/datasets/list")
async def list_generated_datasets():
    """List all generated datasets."""
    datasets_dir = Path("./data/generated")
    if not datasets_dir.exists():
        return {"datasets": []}
    
    datasets = []
    for dataset_dir in datasets_dir.iterdir():
        if dataset_dir.is_dir():
            # Count images
            image_count = len(list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpg")))
            
            datasets.append({
                "name": dataset_dir.name,
                "path": str(dataset_dir),
                "image_count": image_count,
                "created": dataset_dir.stat().st_mtime
            })
    
    return {"datasets": datasets}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
