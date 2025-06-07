from fastapi import APIRouter, HTTPException, Depends, Query, Body, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import uuid
from pathlib import Path

# Import generation module
from generation.generate_diffusion import generate_image, generate_image_batch

# Create router
router = APIRouter()

# Data models
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    num_inference_steps: int = Field(default=50, ge=1, le=150)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: Optional[int] = None
    model_id: Optional[str] = "stabilityai/stable-diffusion-2-1"

class BatchGenerationRequest(BaseModel):
    prompts: List[str]
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    num_inference_steps: int = Field(default=50, ge=1, le=150)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: Optional[int] = None
    model_id: Optional[str] = "stabilityai/stable-diffusion-2-1"

class GenerationResponse(BaseModel):
    image_path: str
    prompt: str
    parameters: dict

class BatchGenerationResponse(BaseModel):
    images: List[GenerationResponse]

# Routes
@router.post("/generate", response_model=GenerationResponse)
async def create_image(request: GenerationRequest):
    """
    Generate a single image from a text prompt
    """
    try:
        # Prepare output directory
        output_dir = Path("data/generated")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate unique filename
        filename = f"{uuid.uuid4()}.png"
        
        # Generate image
        image_path = generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            model_id=request.model_id,
            output_dir=str(output_dir),
            filename=filename
        )
        
        # Return response
        return {
            "image_path": image_path,
            "prompt": request.prompt,
            "parameters": request.dict(exclude={"prompt"})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate/batch", response_model=BatchGenerationResponse)
async def create_batch(request: BatchGenerationRequest, background_tasks: BackgroundTasks):
    """
    Generate multiple images from a list of prompts
    """
    try:
        # Prepare output directory
        batch_id = uuid.uuid4().hex
        output_dir = Path(f"data/generated/batch_{batch_id}")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Extract parameters
        params = request.dict(exclude={"prompts"})
        
        # Generate images
        results = generate_image_batch(
            prompts=request.prompts,
            output_dir=str(output_dir),
            **params
        )
        
        # Create response
        response_items = []
        for idx, result in enumerate(results):
            response_items.append({
                "image_path": result,
                "prompt": request.prompts[idx],
                "parameters": params
            })
            
        return {"images": response_items}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/image/{image_id}")
async def get_image(image_id: str):
    """
    Retrieve a generated image by ID
    """
    # Extract the UUID part and reconstruct the path
    image_path = Path(f"data/generated/{image_id}.png")
    
    # Check for alternative locations
    if not image_path.exists():
        # Check in batch folders
        for batch_dir in Path("data/generated").glob("batch_*"):
            alt_path = batch_dir / f"{image_id}.png"
            if alt_path.exists():
                image_path = alt_path
                break
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
        
    return FileResponse(str(image_path))

@router.get("/health")
async def health_check():
    """
    API health check endpoint
    """
    return {"status": "ok"}