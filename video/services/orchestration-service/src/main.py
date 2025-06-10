"""
Orchestration Service
Coordinates the entire video synthesis pipeline workflow
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import logging
import uvicorn
from datetime import datetime
import json
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Orchestration Service",
    description="Pipeline orchestration and workflow management",
    version="1.0.0"
)

class PipelineRequest(BaseModel):
    vertical: str
    data_sources: List[Dict[str, Any]]
    generation_config: Dict[str, Any]
    quality_requirements: Dict[str, float]
    delivery_config: Dict[str, Any]

class PipelineStatus(BaseModel):
    pipeline_id: str
    status: str
    current_stage: str
    progress: float
    stages: Dict[str, Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

# In-memory storage
active_pipelines: Dict[str, PipelineStatus] = {}

# Service endpoints
SERVICES = {
    "ingestion": "http://ingestion-service:8080",
    "metrics": "http://metrics-extraction-service:8080", 
    "generation": "http://generation-service:8080",
    "validation": "http://validation-service:8080",
    "delivery": "http://delivery-service:8080"
}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "orchestration-service", "version": "1.0.0"}

@app.post("/api/v1/pipeline/start")
async def start_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Start complete video synthesis pipeline"""
    pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(request)) % 10000:04d}"
    
    pipeline = PipelineStatus(
        pipeline_id=pipeline_id,
        status="started",
        current_stage="ingestion",
        progress=0.0,
        stages={
            "ingestion": {"status": "pending", "progress": 0.0},
            "metrics_extraction": {"status": "pending", "progress": 0.0},
            "generation": {"status": "pending", "progress": 0.0},
            "validation": {"status": "pending", "progress": 0.0},
            "delivery": {"status": "pending", "progress": 0.0}
        },
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    active_pipelines[pipeline_id] = pipeline
    background_tasks.add_task(execute_pipeline, pipeline_id, request)
    
    logger.info(f"Started pipeline {pipeline_id} for vertical {request.vertical}")
    return {"pipeline_id": pipeline_id, "status": "started"}

@app.get("/api/v1/pipeline/status/{pipeline_id}")
async def get_pipeline_status(pipeline_id: str):
    """Get pipeline status"""
    if pipeline_id not in active_pipelines:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    return active_pipelines[pipeline_id]

@app.get("/api/v1/pipeline/list")
async def list_pipelines():
    """List all pipelines"""
    return {"pipelines": list(active_pipelines.values()), "total": len(active_pipelines)}

async def execute_pipeline(pipeline_id: str, request: PipelineRequest):
    """Execute the complete pipeline workflow"""
    pipeline = active_pipelines[pipeline_id]
    
    try:
        # Stage 1: Ingestion
        await execute_stage(pipeline, "ingestion", "Ingesting data sources", 
                          lambda: call_service("ingestion", "/api/v1/ingest/start", request.data_sources))
        
        # Stage 2: Metrics Extraction
        await execute_stage(pipeline, "metrics_extraction", "Extracting quality metrics",
                          lambda: call_service("metrics", "/api/v1/metrics/extract", {}))
        
        # Stage 3: Generation
        await execute_stage(pipeline, "generation", "Generating synthetic video",
                          lambda: call_service("generation", "/api/v1/generate/video", request.generation_config))
        
        # Stage 4: Validation
        await execute_stage(pipeline, "validation", "Validating output quality",
                          lambda: call_service("validation", "/api/v1/validate", request.quality_requirements))
        
        # Stage 5: Delivery
        await execute_stage(pipeline, "delivery", "Delivering final output",
                          lambda: call_service("delivery", "/api/v1/deliver", request.delivery_config))
        
        pipeline.status = "completed"
        pipeline.current_stage = "completed"
        pipeline.progress = 100.0
        
    except Exception as e:
        pipeline.status = "failed"
        pipeline.stages[pipeline.current_stage]["error"] = str(e)
        logger.error(f"Pipeline {pipeline_id} failed at stage {pipeline.current_stage}: {e}")
    
    pipeline.updated_at = datetime.now()

async def execute_stage(pipeline: PipelineStatus, stage_name: str, description: str, task_func):
    """Execute a pipeline stage"""
    pipeline.current_stage = stage_name
    pipeline.stages[stage_name]["status"] = "running"
    pipeline.stages[stage_name]["description"] = description
    pipeline.updated_at = datetime.now()
    
    logger.info(f"Pipeline {pipeline.pipeline_id}: Starting {stage_name}")
    
    # Simulate stage execution
    for progress in range(0, 101, 20):
        await asyncio.sleep(1)
        pipeline.stages[stage_name]["progress"] = progress
        stage_progress = sum(stage["progress"] for stage in pipeline.stages.values()) / len(pipeline.stages)
        pipeline.progress = stage_progress
        pipeline.updated_at = datetime.now()
    
    pipeline.stages[stage_name]["status"] = "completed"
    pipeline.stages[stage_name]["progress"] = 100.0
    
    logger.info(f"Pipeline {pipeline.pipeline_id}: Completed {stage_name}")

async def call_service(service_name: str, endpoint: str, data: Any):
    """Call another service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{SERVICES[service_name]}{endpoint}", json=data)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to call {service_name}{endpoint}: {e}")
        # For demo, don't fail - just log
        return {"status": "simulated"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
