"""
Data profiling API endpoints
"""

import os
import tempfile
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import pandas as pd

from sdk.profiler import DataProfiler, DatasetProfile
from api.deps import get_current_user
from api.middleware.rate_limiter import rate_limit
from api.middleware.security_middleware import SecurityMiddleware

router = APIRouter(prefix="/profile", tags=["profiling"])


class ProfileRequest(BaseModel):
    """Request model for profiling configuration"""
    detect_patterns: bool = Field(default=True, description="Detect data patterns")
    detect_distributions: bool = Field(default=True, description="Detect statistical distributions")
    outlier_method: str = Field(default='iqr', regex='^(iqr|zscore)$')
    cardinality_threshold: float = Field(default=0.95, ge=0, le=1)
    sample_size: Optional[int] = Field(default=None, description="Sample size for large datasets")


class ProfileResponse(BaseModel):
    """Response model for profile results"""
    name: str
    shape: List[int]
    memory_usage_mb: float
    column_count: int
    numerical_columns: int
    categorical_columns: int
    missing_data_percentage: float
    profile_summary: Dict[str, Any]


@router.post("/analyze", response_model=ProfileResponse)
@rate_limit(requests_per_minute=20)
async def profile_dataset(
    file: UploadFile = File(...),
    config: ProfileRequest = ProfileRequest(),
    current_user = Depends(get_current_user)
):
    """
    Profile a dataset to understand its characteristics
    """
    # Validate file
    security_middleware = SecurityMiddleware()
    validation_result = await security_middleware.validate_file_upload(file)
    
    if not validation_result['valid']:
        raise HTTPException(
            status_code=400,
            detail=f"File validation failed: {', '.join(validation_result['errors'])}"
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Load data
        if file.filename.endswith('.csv'):
            if config.sample_size:
                df = pd.read_csv(tmp_path, nrows=config.sample_size)
            else:
                df = pd.read_csv(tmp_path)
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(tmp_path)
            if config.sample_size and len(df) > config.sample_size:
                df = df.sample(n=config.sample_size)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Create profiler
        profiler = DataProfiler(
            detect_patterns=config.detect_patterns,
            detect_distributions=config.detect_distributions,
            outlier_method=config.outlier_method,
            cardinality_threshold=config.cardinality_threshold
        )
        
        # Generate profile
        profile = profiler.profile_dataset(df, name=file.filename)
        
        # Calculate summary statistics
        total_nulls = sum(col.null_count for col in profile.column_profiles.values())
        total_cells = profile.shape[0] * profile.shape[1]
        missing_percentage = (total_nulls / total_cells * 100) if total_cells > 0 else 0
        
        # Create summary
        profile_summary = {
            'columns': {
                col_name: {
                    'dtype': col_profile.dtype,
                    'null_percentage': col_profile.null_percentage,
                    'unique_count': col_profile.unique_count,
                    'mean': col_profile.mean,
                    'std': col_profile.std,
                    'min': col_profile.min,
                    'max': col_profile.max,
                    'outliers_percentage': col_profile.outliers_percentage
                }
                for col_name, col_profile in profile.column_profiles.items()
            },
            'duplicates': {
                'count': profile.duplicates_count,
                'percentage': profile.duplicates_percentage
            },
            'potential_keys': profile.potential_keys,
            'constant_columns': profile.constant_columns,
            'high_cardinality_columns': profile.high_cardinality_columns
        }
        
        return ProfileResponse(
            name=profile.name,
            shape=list(profile.shape),
            memory_usage_mb=profile.memory_usage_mb,
            column_count=profile.shape[1],
            numerical_columns=len(profile.numerical_columns),
            categorical_columns=len(profile.categorical_columns),
            missing_data_percentage=missing_percentage,
            profile_summary=profile_summary
        )
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/detailed")
async def get_detailed_profile(
    file: UploadFile = File(...),
    config: ProfileRequest = ProfileRequest(),
    current_user = Depends(get_current_user)
):
    """
    Get detailed profiling results including all statistics and patterns
    """
    # Validate file
    security_middleware = SecurityMiddleware()
    validation_result = await security_middleware.validate_file_upload(file)
    
    if not validation_result['valid']:
        raise HTTPException(
            status_code=400,
            detail=f"File validation failed: {', '.join(validation_result['errors'])}"
        )
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Load data
        if file.filename.endswith('.csv'):
            df = pd.read_csv(tmp_path, nrows=config.sample_size)
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(tmp_path)
            if config.sample_size and len(df) > config.sample_size:
                df = df.sample(n=config.sample_size)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Generate profile
        profiler = DataProfiler(
            detect_patterns=config.detect_patterns,
            detect_distributions=config.detect_distributions,
            outlier_method=config.outlier_method,
            cardinality_threshold=config.cardinality_threshold
        )
        
        profile = profiler.profile_dataset(df, name=file.filename)
        
        # Return full profile
        return profile.to_dict()
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/column/{column_name}")
async def profile_column(
    column_name: str,
    file: UploadFile = File(...),
    current_user = Depends(get_current_user)
):
    """
    Profile a specific column in the dataset
    """
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Load data
        if file.filename.endswith('.csv'):
            df = pd.read_csv(tmp_path)
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Check if column exists
        if column_name not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{column_name}' not found")
        
        # Profile column
        profiler = DataProfiler()
        column_profile = profiler.profile_column(df[column_name])
        
        return column_profile.to_dict()
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/compare")
async def compare_datasets(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    current_user = Depends(get_current_user)
):
    """
    Compare profiles of two datasets
    """
    # Save both files
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file1.filename).suffix) as tmp1:
        content1 = await file1.read()
        tmp1.write(content1)
        tmp1_path = tmp1.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file2.filename).suffix) as tmp2:
        content2 = await file2.read()
        tmp2.write(content2)
        tmp2_path = tmp2.name
    
    try:
        # Load datasets
        df1 = pd.read_csv(tmp1_path) if file1.filename.endswith('.csv') else pd.read_parquet(tmp1_path)
        df2 = pd.read_csv(tmp2_path) if file2.filename.endswith('.csv') else pd.read_parquet(tmp2_path)
        
        # Generate profiles
        profiler = DataProfiler()
        profile1 = profiler.profile_dataset(df1, name=file1.filename)
        profile2 = profiler.profile_dataset(df2, name=file2.filename)
        
        # Compare profiles
        comparison = profiler.compare_profiles(profile1, profile2)
        
        return comparison
        
    finally:
        for path in [tmp1_path, tmp2_path]:
            if os.path.exists(path):
                os.unlink(path)


@router.post("/report")
async def generate_profile_report(
    file: UploadFile = File(...),
    format: str = Query(default="text", regex="^(text|json|html)$"),
    current_user = Depends(get_current_user)
):
    """
    Generate a comprehensive profiling report
    """
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Load data
        if file.filename.endswith('.csv'):
            df = pd.read_csv(tmp_path)
        elif file.filename.endswith('.parquet'):
            df = pd.read_parquet(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Generate profile
        profiler = DataProfiler()
        profile = profiler.profile_dataset(df, name=file.filename)
        
        if format == "json":
            return profile.to_dict()
        
        elif format == "text":
            report = profiler.generate_report(profile)
            return {"report": report}
        
        elif format == "html":
            # Generate HTML report
            report_text = profiler.generate_report(profile)
            html_content = f"""
            <html>
            <head>
                <title>Data Profile Report - {file.filename}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    pre {{ background: #f5f5f5; padding: 10px; border-radius: 5px; }}
                    .metric {{ margin: 10px 0; }}
                </style>
            </head>
            <body>
                <pre>{report_text}</pre>
            </body>
            </html>
            """
            
            # Save HTML to temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as html_file:
                html_file.write(html_content)
                html_path = html_file.name
            
            return FileResponse(
                html_path,
                media_type="text/html",
                filename=f"profile_report_{file.filename}.html"
            )
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/templates")
async def get_profiling_templates(current_user = Depends(get_current_user)):
    """
    Get profiling configuration templates
    """
    templates = {
        "quick": {
            "detect_patterns": False,
            "detect_distributions": False,
            "outlier_method": "iqr",
            "cardinality_threshold": 0.95,
            "description": "Quick profiling for large datasets"
        },
        "comprehensive": {
            "detect_patterns": True,
            "detect_distributions": True,
            "outlier_method": "iqr",
            "cardinality_threshold": 0.95,
            "description": "Full profiling with pattern and distribution detection"
        },
        "data_quality": {
            "detect_patterns": True,
            "detect_distributions": False,
            "outlier_method": "zscore",
            "cardinality_threshold": 0.90,
            "description": "Focus on data quality issues"
        }
    }
    
    return templates


@router.post("/quality-check")
async def check_data_quality(
    file: UploadFile = File(...),
    rules: Optional[Dict[str, Any]] = None,
    current_user = Depends(get_current_user)
):
    """
    Check data quality against specified rules
    """
    # Default quality rules
    default_rules = {
        "max_null_percentage": 10.0,
        "max_duplicate_percentage": 5.0,
        "required_columns": [],
        "column_types": {},
        "value_ranges": {}
    }
    
    if rules:
        default_rules.update(rules)
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Load data
        df = pd.read_csv(tmp_path) if file.filename.endswith('.csv') else pd.read_parquet(tmp_path)
        
        # Generate profile
        profiler = DataProfiler()
        profile = profiler.profile_dataset(df, name=file.filename)
        
        # Check quality rules
        issues = []
        
        # Check null percentage
        for col_name, col_profile in profile.column_profiles.items():
            if col_profile.null_percentage > default_rules["max_null_percentage"]:
                issues.append({
                    "type": "high_null_percentage",
                    "column": col_name,
                    "value": col_profile.null_percentage,
                    "threshold": default_rules["max_null_percentage"]
                })
        
        # Check duplicates
        if profile.duplicates_percentage > default_rules["max_duplicate_percentage"]:
            issues.append({
                "type": "high_duplicate_percentage",
                "value": profile.duplicates_percentage,
                "threshold": default_rules["max_duplicate_percentage"]
            })
        
        # Check required columns
        missing_columns = set(default_rules["required_columns"]) - set(df.columns)
        if missing_columns:
            issues.append({
                "type": "missing_required_columns",
                "columns": list(missing_columns)
            })
        
        # Check column types
        for col, expected_type in default_rules["column_types"].items():
            if col in profile.column_profiles:
                actual_type = profile.column_profiles[col].dtype
                if not actual_type.startswith(expected_type):
                    issues.append({
                        "type": "incorrect_column_type",
                        "column": col,
                        "expected": expected_type,
                        "actual": actual_type
                    })
        
        # Check value ranges
        for col, range_spec in default_rules["value_ranges"].items():
            if col in profile.column_profiles:
                col_profile = profile.column_profiles[col]
                if col_profile.min is not None:
                    if "min" in range_spec and col_profile.min < range_spec["min"]:
                        issues.append({
                            "type": "value_below_minimum",
                            "column": col,
                            "min_value": col_profile.min,
                            "expected_min": range_spec["min"]
                        })
                    if "max" in range_spec and col_profile.max > range_spec["max"]:
                        issues.append({
                            "type": "value_above_maximum",
                            "column": col,
                            "max_value": col_profile.max,
                            "expected_max": range_spec["max"]
                        })
        
        return {
            "quality_score": 1.0 - (len(issues) / 10.0),  # Simple scoring
            "issues": issues,
            "summary": {
                "total_issues": len(issues),
                "passed": len(issues) == 0
            }
        }
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)