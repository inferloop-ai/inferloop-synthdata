# api/routes.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

# Import SDK components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdk import GPT2Generator, LangChainTemplate, DataFormatter
from sdk.validation import BLEUROUGEValidator, GPT4Validator

# Configure logging
logger = logging.getLogger(__name__)

# Pydantic models
class GenerateRequest(BaseModel):
    prompts: List[str]
    model_name: str = "gpt2"
    max_length: int = 100
    temperature: float = 0.8
    top_p: float = 0.9

class ValidateRequest(BaseModel):
    references: List[str]
    candidates: List[str]
    validation_type: str = "bleu_rouge"  # or "gpt4"

class FormatRequest(BaseModel):
    data: List[Dict]
    format_type: str = "jsonl"  # jsonl, csv, markdown
    filepath: str

# Create router
router = APIRouter()

# Global generators (in production, use dependency injection)
generators = {}
validator = BLEUROUGEValidator()
gpt4_validator = GPT4Validator()

# Helper function to get or create generator
def get_generator(model_name: str):
    if model_name not in generators:
        generators[model_name] = GPT2Generator(model_name)
    return generators[model_name]

@router.post("/generate")
async def generate_text(request: GenerateRequest):
    """Generate synthetic text"""
    try:
        generator = get_generator(request.model_name)
        
        results = generator.batch_generate(
            request.prompts,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return {
            "success": True,
            "generated_texts": results,
            "model_used": request.model_name
        }
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate")
async def validate_text(request: ValidateRequest):
    """Validate synthetic text quality"""
    try:
        if request.validation_type == "bleu_rouge":
            scores = validator.validate_batch(request.references, request.candidates)
        elif request.validation_type == "gpt4":
            scores = gpt4_validator.batch_evaluate(request.candidates)
        else:
            raise ValueError(f"Unknown validation type: {request.validation_type}")
        
        return {
            "success": True,
            "validation_scores": scores,
            "validation_type": request.validation_type
        }
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/format")
async def format_data(request: FormatRequest):
    """Format data to specified format"""
    try:
        formatter = DataFormatter()
        
        if request.format_type == "jsonl":
            formatter.to_jsonl(request.data, request.filepath)
        elif request.format_type == "csv":
            formatter.to_csv(request.data, request.filepath)
        elif request.format_type == "markdown":
            formatter.to_markdown(request.data, request.filepath)
        else:
            raise ValueError(f"Unknown format type: {request.format_type}")
        
        return {
            "success": True,
            "message": f"Data formatted and saved to {request.filepath}"
        }
        
    except Exception as e:
        logger.error(f"Formatting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": list(generators.keys())}
