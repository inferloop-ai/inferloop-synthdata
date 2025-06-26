"""
TextNLP API Application - Unified Infrastructure Version

This is the refactored version of the TextNLP API that uses the unified
cloud deployment infrastructure for text and NLP synthetic data generation.
"""

import os
import asyncio
import json
from typing import Optional, Dict, Any, List, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
import uuid

from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator

# Import from unified infrastructure
from unified_cloud_deployment.auth import AuthMiddleware, get_current_user, User, require_permissions
from unified_cloud_deployment.monitoring import (
    TelemetryMiddleware,
    MetricsCollector,
    create_counter,
    create_histogram,
    create_gauge
)
from unified_cloud_deployment.storage import StorageClient, get_storage_client
from unified_cloud_deployment.cache import CacheClient, get_cache_client
from unified_cloud_deployment.database import DatabaseClient, get_db_session
from unified_cloud_deployment.config import get_service_config
from unified_cloud_deployment.ratelimit import RateLimitMiddleware
from unified_cloud_deployment.websocket import WebSocketManager

# Import service-specific logic
from textnlp.sdk import GPT2Generator, LangChainTemplate
from textnlp.sdk.validation import BLEUROUGEValidator
from textnlp.infrastructure.adapter import TextNLPServiceAdapter, ServiceTier


# Initialize metrics
tokens_generated_counter = create_counter(
    "textnlp_tokens_generated_total",
    "Total number of tokens generated",
    labels=["model", "tier", "user_id"]
)

generation_duration_histogram = create_histogram(
    "textnlp_generation_duration_seconds",
    "Duration of text generation",
    labels=["model", "tier", "prompt_length"],
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]
)

active_streams_gauge = create_gauge(
    "textnlp_active_streams",
    "Number of active streaming connections",
    labels=["model"]
)

# Service configuration
service_config = get_service_config()
service_tier = ServiceTier(service_config.get("tier", "starter"))
adapter = TextNLPServiceAdapter(service_tier)

# Initialize components
storage_client = StorageClient(service_name="textnlp")
cache_client = CacheClient(namespace="textnlp")
metrics_collector = MetricsCollector(service_name="textnlp")
ws_manager = WebSocketManager()

# Model registry
MODEL_REGISTRY = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print(f"Starting TextNLP API Service (Tier: {service_tier.value})")
    
    # Initialize storage buckets
    await storage_client.ensure_bucket("prompts")
    await storage_client.ensure_bucket("generations")
    await storage_client.ensure_bucket("templates")
    await storage_client.ensure_bucket("validations")
    
    # Load models based on tier
    model_config = adapter.get_model_config()
    if model_config["model_loading"]["preload"]:
        print("Preloading models...")
        for model_name in model_config["available_models"]:
            try:
                MODEL_REGISTRY[model_name] = await load_model(model_name)
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")
    
    yield
    
    # Shutdown
    print("Shutting down TextNLP API Service")
    await storage_client.close()
    await cache_client.close()
    await ws_manager.close_all()


# Create FastAPI app
app = FastAPI(
    title="Inferloop TextNLP API",
    description="AI-powered synthetic text and NLP data generation",
    version=adapter.SERVICE_VERSION,
    lifespan=lifespan
)

# Add middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.inferloop.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    AuthMiddleware,
    service_name="textnlp",
    required_permissions=["textnlp:read"]
)

app.add_middleware(
    RateLimitMiddleware,
    service_name="textnlp",
    tier_config=adapter.get_api_config()["rate_limiting"]
)

app.add_middleware(
    TelemetryMiddleware,
    service_name="textnlp",
    enable_tracing=service_tier != ServiceTier.STARTER,
    enable_metrics=True
)


# Request/Response models
class GenerationRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Input prompt for generation")
    model: str = Field("gpt2", description="Model to use for generation")
    max_tokens: int = Field(100, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    stream: bool = Field(False, description="Stream the response")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v, values):
        # Validate against model limits
        model = values.get('model', 'gpt2')
        # Model-specific validation would go here
        return v


class GenerationResponse(BaseModel):
    """Response model for text generation"""
    id: str
    model: str
    created: int
    text: str
    usage: Dict[str, int]
    metadata: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request model for chat completion"""
    messages: List[ChatMessage]
    model: str = Field("gpt2-large", description="Model to use")
    max_tokens: int = Field(500, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stream: bool = Field(False)


class ValidationRequest(BaseModel):
    """Request model for text validation"""
    reference_texts: List[str]
    generated_texts: List[str]
    metrics: List[str] = Field(["bleu", "rouge"], description="Metrics to calculate")


class ValidationResponse(BaseModel):
    """Response model for validation results"""
    id: str
    metrics: Dict[str, Any]
    summary: Dict[str, float]
    created: int


# Helper functions
async def load_model(model_name: str):
    """Load a model if not already loaded"""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    
    # Check cache for model
    cache_key = f"model:{model_name}"
    cached_path = await cache_client.get(cache_key)
    
    if cached_path:
        model_path = cached_path
    else:
        # Download from storage if needed
        model_path = f"/models/{model_name}"
    
    # Initialize model based on type
    if model_name.startswith("gpt2"):
        model = GPT2Generator(model_name)
    else:
        # Add other model types as needed
        raise ValueError(f"Unknown model type: {model_name}")
    
    MODEL_REGISTRY[model_name] = model
    return model


async def check_model_access(user: User, model_name: str) -> bool:
    """Check if user has access to the requested model"""
    user_tier = ServiceTier(user.tier)
    model_config = adapter.get_model_config()
    
    if model_name not in model_config["available_models"]:
        return False
    
    model_info = next(
        (m for m in adapter.MODELS if m.name == model_name),
        None
    )
    
    if not model_info:
        return False
    
    return model_info.min_tier.value <= user_tier.value


# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "textnlp",
        "version": adapter.SERVICE_VERSION,
        "tier": service_tier.value
    }


@app.get("/ready")
async def readiness_check(db=Depends(get_db_session)):
    """Readiness check endpoint"""
    try:
        # Check database
        await db.execute("SELECT 1")
        
        # Check cache
        await cache_client.ping()
        
        # Check storage
        await storage_client.check_connection()
        
        # Check at least one model is loaded
        if service_tier != ServiceTier.STARTER and not MODEL_REGISTRY:
            raise Exception("No models loaded")
        
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")


@app.post("/api/textnlp/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db_session)
):
    """
    Generate synthetic text.
    
    Requires permissions: textnlp:generate
    """
    require_permissions(current_user, ["textnlp:generate"])
    
    # Check model access
    if not await check_model_access(current_user, request.model):
        raise HTTPException(
            status_code=403,
            detail=f"Model '{request.model}' not available for {current_user.tier} tier"
        )
    
    # Check token limits
    model_config = adapter.get_model_config()
    model_info = model_config["available_models"].get(request.model, {})
    max_allowed_tokens = model_info.get("max_tokens", 1024)
    
    if request.max_tokens > max_allowed_tokens:
        raise HTTPException(
            status_code=400,
            detail=f"Max tokens ({request.max_tokens}) exceeds limit ({max_allowed_tokens}) for model {request.model}"
        )
    
    if request.stream:
        # Return streaming response
        return StreamingResponse(
            generate_stream(request, current_user, db),
            media_type="text/event-stream"
        )
    
    # Non-streaming generation
    import time
    start_time = time.time()
    
    # Generate unique ID
    generation_id = f"gen-{uuid.uuid4()}"
    
    # Load model
    model = await load_model(request.model)
    
    # Generate text
    result = await model.generate_async(
        prompt=request.prompt,
        max_length=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        seed=request.seed
    )
    
    # Calculate tokens
    prompt_tokens = len(request.prompt.split())  # Simple approximation
    completion_tokens = len(result.split())
    total_tokens = prompt_tokens + completion_tokens
    
    # Track metrics
    duration = time.time() - start_time
    tokens_generated_counter.labels(
        model=request.model,
        tier=service_tier.value,
        user_id=str(current_user.id)
    ).inc(completion_tokens)
    
    generation_duration_histogram.labels(
        model=request.model,
        tier=service_tier.value,
        prompt_length=len(request.prompt)
    ).observe(duration)
    
    # Store generation record
    await db.execute(
        """
        INSERT INTO textnlp.generations 
        (id, user_id, model, prompt, generated_text, tokens_used, duration, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        generation_id, current_user.id, request.model,
        request.prompt, result, total_tokens, duration, datetime.utcnow()
    )
    
    # Cache result if enabled
    if model_info.get("cache_config", {}).get("enabled", False):
        cache_key = f"generation:{request.model}:{hash(request.prompt)}"
        await cache_client.set(
            cache_key,
            result,
            ttl=model_info["cache_config"].get("ttl", 3600)
        )
    
    return GenerationResponse(
        id=generation_id,
        model=request.model,
        created=int(time.time()),
        text=result,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        },
        metadata={
            "temperature": request.temperature,
            "duration_seconds": round(duration, 3)
        }
    )


async def generate_stream(
    request: GenerationRequest,
    user: User,
    db
) -> AsyncGenerator[str, None]:
    """Stream text generation"""
    generation_id = f"gen-stream-{uuid.uuid4()}"
    model = await load_model(request.model)
    
    # Increment active streams
    active_streams_gauge.labels(model=request.model).inc()
    
    try:
        token_count = 0
        all_text = ""
        
        async for token in model.generate_stream_async(
            prompt=request.prompt,
            max_length=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k
        ):
            token_count += 1
            all_text += token
            
            # Send SSE event
            event = {
                "id": generation_id,
                "model": request.model,
                "choices": [{
                    "text": token,
                    "index": 0,
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(event)}\n\n"
        
        # Send completion event
        completion_event = {
            "id": generation_id,
            "model": request.model,
            "choices": [{
                "text": "",
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "completion_tokens": token_count,
                "total_tokens": token_count + len(request.prompt.split())
            }
        }
        yield f"data: {json.dumps(completion_event)}\n\n"
        yield "data: [DONE]\n\n"
        
        # Track usage
        tokens_generated_counter.labels(
            model=request.model,
            tier=service_tier.value,
            user_id=str(user.id)
        ).inc(token_count)
        
        # Store generation
        await db.execute(
            """
            INSERT INTO textnlp.generations 
            (id, user_id, model, prompt, generated_text, tokens_used, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            generation_id, user.id, request.model,
            request.prompt, all_text, token_count, datetime.utcnow()
        )
        
    finally:
        # Decrement active streams
        active_streams_gauge.labels(model=request.model).dec()


@app.post("/api/textnlp/chat")
async def chat_completion(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db_session)
):
    """
    Generate chat completion.
    
    Requires permissions: textnlp:generate
    """
    require_permissions(current_user, ["textnlp:generate"])
    
    # Check if chat is available for tier
    if service_tier == ServiceTier.STARTER:
        raise HTTPException(
            status_code=403,
            detail="Chat completion requires Professional tier or higher"
        )
    
    # Convert chat format to prompt
    prompt = "\n".join([
        f"{msg.role}: {msg.content}"
        for msg in request.messages
    ])
    prompt += "\nassistant:"
    
    # Use generation endpoint logic
    gen_request = GenerationRequest(
        prompt=prompt,
        model=request.model,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=request.stream
    )
    
    return await generate_text(gen_request, current_user, db)


@app.post("/api/textnlp/validate", response_model=ValidationResponse)
async def validate_text(
    request: ValidationRequest,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db_session)
):
    """
    Validate generated text quality.
    
    Requires permissions: textnlp:validate
    """
    require_permissions(current_user, ["textnlp:validate"])
    
    validation_id = f"val-{uuid.uuid4()}"
    
    # Initialize validator
    validator = BLEUROUGEValidator()
    
    # Calculate metrics
    results = {}
    
    if "bleu" in request.metrics:
        results["bleu"] = validator.calculate_bleu(
            request.reference_texts,
            request.generated_texts
        )
    
    if "rouge" in request.metrics:
        results["rouge"] = validator.calculate_rouge(
            request.reference_texts,
            request.generated_texts
        )
    
    # Calculate summary scores
    summary = {}
    if "bleu" in results:
        summary["bleu_avg"] = sum(results["bleu"]) / len(results["bleu"])
    if "rouge" in results:
        summary["rouge_l_avg"] = sum(
            r["rouge-l"]["f"] for r in results["rouge"]
        ) / len(results["rouge"])
    
    # Store validation
    await db.execute(
        """
        INSERT INTO textnlp.validations
        (id, user_id, metrics, results, created_at)
        VALUES ($1, $2, $3, $4, $5)
        """,
        validation_id, current_user.id, request.metrics,
        json.dumps(results), datetime.utcnow()
    )
    
    return ValidationResponse(
        id=validation_id,
        metrics=results,
        summary=summary,
        created=int(time.time())
    )


@app.get("/api/textnlp/models")
async def list_models(
    current_user: User = Depends(get_current_user)
):
    """List available models for user's tier"""
    user_tier = ServiceTier(current_user.tier)
    model_config = adapter.get_model_config()
    
    models = []
    for model in adapter.MODELS:
        available = model.enabled and model.min_tier.value <= user_tier.value
        models.append({
            "id": model.name,
            "name": model.name,
            "available": available,
            "min_tier": model.min_tier.value,
            "max_tokens": model.max_tokens,
            "is_commercial": model.is_commercial,
            "description": f"{model.name} language model"
        })
    
    return {
        "user_tier": user_tier.value,
        "default_model": model_config["default_model"],
        "models": models
    }


@app.get("/api/textnlp/templates")
async def list_templates(
    current_user: User = Depends(get_current_user),
    storage: StorageClient = Depends(get_storage_client)
):
    """List available prompt templates"""
    # Check tier
    if ServiceTier(current_user.tier) == ServiceTier.STARTER:
        raise HTTPException(
            status_code=403,
            detail="Templates require Professional tier or higher"
        )
    
    # List templates from storage
    templates = await storage.list_objects("templates/")
    
    return {
        "templates": [
            {
                "name": t["name"],
                "description": t.get("metadata", {}).get("description", ""),
                "category": t.get("metadata", {}).get("category", "general"),
                "created": t["created"]
            }
            for t in templates
        ]
    }


@app.websocket("/ws/textnlp/stream")
async def websocket_stream(
    websocket: WebSocket,
    current_user: User = Depends(get_current_user)
):
    """WebSocket endpoint for streaming generation"""
    # Check if WebSocket is enabled for tier
    if ServiceTier(current_user.tier) == ServiceTier.STARTER:
        await websocket.close(code=4003, reason="WebSocket requires Professional tier or higher")
        return
    
    await websocket.accept()
    connection_id = f"ws-{uuid.uuid4()}"
    
    try:
        await ws_manager.connect(connection_id, websocket)
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            if data.get("type") == "generate":
                request = GenerationRequest(**data["payload"])
                
                # Stream generation
                async for chunk in generate_stream(request, current_user, None):
                    await websocket.send_text(chunk)
            
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        ws_manager.disconnect(connection_id)
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
        await websocket.close()


@app.get("/api/textnlp/usage")
async def get_usage_stats(
    period: str = "30d",
    current_user: User = Depends(get_current_user),
    db=Depends(get_db_session)
):
    """Get usage statistics"""
    if ServiceTier(current_user.tier) == ServiceTier.STARTER:
        raise HTTPException(
            status_code=403,
            detail="Usage statistics require Professional tier or higher"
        )
    
    # Parse period
    period_map = {"24h": "1 day", "7d": "7 days", "30d": "30 days"}
    interval = period_map.get(period, "30 days")
    
    # Get stats
    stats = await db.fetch_one(
        f"""
        SELECT 
            COUNT(*) as total_generations,
            SUM(tokens_used) as total_tokens,
            COUNT(DISTINCT model) as models_used,
            AVG(duration) as avg_duration
        FROM textnlp.generations
        WHERE user_id = $1 
        AND created_at >= NOW() - INTERVAL '{interval}'
        """,
        current_user.id
    )
    
    return {
        "period": period,
        "usage": {
            "total_generations": stats["total_generations"] or 0,
            "total_tokens": stats["total_tokens"] or 0,
            "models_used": stats["models_used"] or 0,
            "avg_duration_seconds": round(stats["avg_duration"] or 0, 2)
        },
        "limits": adapter.get_api_config()["rate_limiting"]
    }


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Expose metrics in Prometheus format"""
    return await metrics_collector.get_metrics()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_unified:app",
        host="0.0.0.0",
        port=8000,
        reload=bool(os.getenv("DEBUG", False))
    )