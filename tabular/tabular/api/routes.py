from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import pandas as pd
import json
import io
import yaml
from pydantic import BaseModel

from ..sdk.factory import SyntheticDataGeneratorFactory
from ..sdk.validator import DataValidator

router = APIRouter(prefix="/api/v1", tags=["synthetic-data"])

class GenerationRequest(BaseModel):
    generator_type: str = "sdv"  # sdv, ctgan, ydata
    config: Optional[Dict[str, Any]] = None
    num_rows: int = 100
    
class ValidationRequest(BaseModel):
    metrics: List[str] = ["column_correlations", "column_shapes"]

@router.post("/generate")
async def generate_synthetic_data(
    request: GenerationRequest,
    file: UploadFile = File(...),
):
    """Generate synthetic data based on uploaded CSV data"""
    try:
        # Read the uploaded CSV file
        content = await file.read()
        data = pd.read_csv(io.BytesIO(content))
        
        # Create generator
        factory = SyntheticDataGeneratorFactory()
        generator = factory.create_generator(request.generator_type)
        
        # Configure and fit
        if request.config:
            generator.configure(request.config)
        generator.fit(data)
        
        # Generate synthetic data
        synthetic_data = generator.generate(request.num_rows)
        
        # Convert to JSON response
        result = synthetic_data.to_dict(orient="records")
        return {"data": result, "rows": len(result)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating synthetic data: {str(e)}")

@router.post("/validate")
async def validate_synthetic_data(
    validation_request: ValidationRequest,
    original_file: UploadFile = File(...),
    synthetic_file: UploadFile = File(...),
):
    """Validate synthetic data against original data"""
    try:
        # Read uploaded files
        original_content = await original_file.read()
        synthetic_content = await synthetic_file.read()
        
        original_data = pd.read_csv(io.BytesIO(original_content))
        synthetic_data = pd.read_csv(io.BytesIO(synthetic_content))
        
        # Validate data
        validator = DataValidator()
        validation_results = validator.validate(
            original_data, 
            synthetic_data,
            metrics=validation_request.metrics
        )
        
        return validation_results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating data: {str(e)}")

@router.get("/config-templates/{generator_type}")
async def get_config_template(generator_type: str):
    """Get configuration template for a specific generator"""
    try:
        valid_types = ["sdv", "ctgan", "ydata"]
        if generator_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid generator type. Must be one of: {valid_types}")
        
        # Load template from sample templates
        try:
            with open(f"data/sample_templates/{generator_type}_config.yaml", "r") as f:
                config = yaml.safe_load(f)
                return config
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Config template for {generator_type} not found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving config template: {str(e)}")
