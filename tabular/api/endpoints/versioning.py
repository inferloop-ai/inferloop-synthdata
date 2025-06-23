"""
Model versioning API endpoints
"""

import os
import tempfile
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from sdk.versioning import get_version_manager, ModelVersion
from api.deps import get_current_user
from api.auth.models import UserRole

router = APIRouter(prefix="/models", tags=["versioning"])


class ModelVersionInfo(BaseModel):
    """Model version information"""
    version_id: str
    version_number: int
    created_at: datetime
    generator_type: str
    model_type: str
    data_hash: str
    metrics: Dict[str, float]
    tags: List[str]
    description: str
    is_active: bool


class ModelListResponse(BaseModel):
    """Response for model list"""
    model_id: str
    generator_type: str
    model_type: str
    version_count: int
    active_version: Optional[int]
    latest_version: int


class VersionComparisonResponse(BaseModel):
    """Response for version comparison"""
    version1: int
    version2: int
    created_at_diff_seconds: float
    config_changes: Dict[str, Any]
    metrics_comparison: Dict[str, Any]
    tags_added: List[str]
    tags_removed: List[str]


@router.get("/", response_model=List[ModelListResponse])
async def list_models(current_user = Depends(get_current_user)):
    """List all models with version information"""
    manager = get_version_manager()
    models = []
    
    for model_id in manager.list_models():
        versions = manager.list_versions(model_id)
        active_version = manager.get_active_version(model_id)
        
        if versions:
            first_version = versions[0]
            models.append(ModelListResponse(
                model_id=model_id,
                generator_type=first_version.generator_type,
                model_type=first_version.model_type,
                version_count=len(versions),
                active_version=active_version.version_number if active_version else None,
                latest_version=max(v.version_number for v in versions)
            ))
    
    return models


@router.get("/{model_id}/versions", response_model=List[ModelVersionInfo])
async def list_versions(
    model_id: str,
    tag: Optional[str] = Query(None, description="Filter by tag"),
    current_user = Depends(get_current_user)
):
    """List all versions of a model"""
    manager = get_version_manager()
    
    try:
        versions = manager.list_versions(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    # Filter by tag if provided
    if tag:
        versions = [v for v in versions if tag in v.tags]
    
    return [
        ModelVersionInfo(
            version_id=v.version_id,
            version_number=v.version_number,
            created_at=v.created_at,
            generator_type=v.generator_type,
            model_type=v.model_type,
            data_hash=v.data_hash,
            metrics=v.metrics,
            tags=v.tags,
            description=v.description,
            is_active=v.is_active
        )
        for v in versions
    ]


@router.get("/{model_id}/versions/{version}", response_model=ModelVersionInfo)
async def get_version(
    model_id: str,
    version: int,
    current_user = Depends(get_current_user)
):
    """Get specific version information"""
    manager = get_version_manager()
    
    try:
        versions = manager.list_versions(model_id)
        version_info = next((v for v in versions if v.version_number == version), None)
        
        if not version_info:
            raise HTTPException(status_code=404, detail=f"Version {version} not found")
        
        return ModelVersionInfo(
            version_id=version_info.version_id,
            version_number=version_info.version_number,
            created_at=version_info.created_at,
            generator_type=version_info.generator_type,
            model_type=version_info.model_type,
            data_hash=version_info.data_hash,
            metrics=version_info.metrics,
            tags=version_info.tags,
            description=version_info.description,
            is_active=version_info.is_active
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{model_id}/versions/{version}/activate")
async def activate_version(
    model_id: str,
    version: int,
    current_user = Depends(get_current_user)
):
    """Set a version as active"""
    manager = get_version_manager()
    
    try:
        manager.set_active_version(model_id, version)
        return {"message": f"Version {version} activated for model {model_id}"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{model_id}/rollback/{version}")
async def rollback_version(
    model_id: str,
    version: int,
    current_user = Depends(get_current_user)
):
    """Rollback to a specific version"""
    manager = get_version_manager()
    
    try:
        manager.rollback(model_id, version)
        return {"message": f"Rolled back to version {version} for model {model_id}"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{model_id}/versions/{version}")
async def delete_version(
    model_id: str,
    version: int,
    current_user = Depends(get_current_user)
):
    """Delete a specific version"""
    # Only admins can delete versions
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    manager = get_version_manager()
    
    try:
        manager.delete_version(model_id, version)
        return {"message": f"Version {version} deleted for model {model_id}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{model_id}/versions/{version}/tags")
async def add_tags(
    model_id: str,
    version: int,
    tags: List[str],
    current_user = Depends(get_current_user)
):
    """Add tags to a version"""
    manager = get_version_manager()
    
    try:
        manager.tag_version(model_id, version, tags)
        return {"message": f"Tags added to version {version}"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{model_id}/compare", response_model=VersionComparisonResponse)
async def compare_versions(
    model_id: str,
    v1: int = Query(..., description="First version number"),
    v2: int = Query(..., description="Second version number"),
    current_user = Depends(get_current_user)
):
    """Compare two versions of a model"""
    manager = get_version_manager()
    
    try:
        comparison = manager.compare_versions(model_id, v1, v2)
        return VersionComparisonResponse(
            version1=comparison['version1'],
            version2=comparison['version2'],
            created_at_diff_seconds=comparison['created_at_diff'],
            config_changes=comparison['config_changes'],
            metrics_comparison=comparison['metrics_comparison'],
            tags_added=comparison['tags_added'],
            tags_removed=comparison['tags_removed']
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{model_id}/versions/{version}/export")
async def export_version(
    model_id: str,
    version: int,
    current_user = Depends(get_current_user)
):
    """Export a model version"""
    manager = get_version_manager()
    
    try:
        # Create temporary directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / f"{model_id}_v{version}"
            
            # Export version
            export_file = manager.export_version(model_id, version, str(export_path))
            
            return FileResponse(
                export_file,
                media_type='application/zip',
                filename=f"{model_id}_v{version}.zip"
            )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{model_id}/import")
async def import_version(
    model_id: str,
    file: UploadFile = File(...),
    set_active: bool = Query(False, description="Set as active version"),
    current_user = Depends(get_current_user)
):
    """Import a model version from file"""
    # Only admins can import versions
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported")
    
    manager = get_version_manager()
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        content = await file.read()
        tmp.write(content)
        import_path = tmp.name
    
    try:
        # Import version
        version = manager.import_version(model_id, import_path, set_active)
        
        return {
            "message": "Version imported successfully",
            "version_number": version.version_number,
            "version_id": version.version_id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(import_path):
            os.unlink(import_path)


@router.get("/search")
async def search_models(
    generator_type: Optional[str] = None,
    model_type: Optional[str] = None,
    tag: Optional[str] = None,
    current_user = Depends(get_current_user)
):
    """Search for models"""
    manager = get_version_manager()
    results = []
    
    for model_id in manager.list_models():
        versions = manager.list_versions(model_id)
        
        if not versions:
            continue
        
        # Filter by generator type
        if generator_type and not any(v.generator_type == generator_type for v in versions):
            continue
        
        # Filter by model type
        if model_type and not any(v.model_type == model_type for v in versions):
            continue
        
        # Filter by tag
        if tag and not any(tag in v.tags for v in versions):
            continue
        
        # Get active version
        active_version = manager.get_active_version(model_id)
        latest_version = versions[0]  # Assumes sorted by version number desc
        
        results.append({
            'model_id': model_id,
            'generator_type': latest_version.generator_type,
            'model_type': latest_version.model_type,
            'version_count': len(versions),
            'active_version': active_version.version_number if active_version else None,
            'latest_version': latest_version.version_number,
            'tags': list(set(tag for v in versions for tag in v.tags))
        })
    
    return results


@router.post("/cleanup")
async def cleanup_old_versions(
    keep_versions: int = Query(5, description="Number of versions to keep per model"),
    dry_run: bool = Query(True, description="Preview what would be deleted"),
    current_user = Depends(get_current_user)
):
    """Clean up old model versions"""
    # Only admins can cleanup
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    manager = get_version_manager()
    cleanup_summary = []
    
    for model_id in manager.list_models():
        versions = manager.list_versions(model_id)
        active_version = manager.get_active_version(model_id)
        
        # Sort by version number descending
        versions.sort(key=lambda v: v.version_number, reverse=True)
        
        # Determine versions to delete
        versions_to_delete = []
        kept_count = 0
        
        for version in versions:
            # Always keep active version
            if version.is_active:
                continue
            
            # Keep requested number of versions
            if kept_count < keep_versions:
                kept_count += 1
                continue
            
            versions_to_delete.append(version.version_number)
        
        if versions_to_delete:
            cleanup_summary.append({
                'model_id': model_id,
                'total_versions': len(versions),
                'versions_to_delete': versions_to_delete,
                'versions_to_keep': [v.version_number for v in versions if v.version_number not in versions_to_delete]
            })
            
            # Actually delete if not dry run
            if not dry_run:
                for version_num in versions_to_delete:
                    try:
                        manager.delete_version(model_id, version_num)
                    except Exception:
                        pass
    
    return {
        'dry_run': dry_run,
        'models_affected': len(cleanup_summary),
        'cleanup_summary': cleanup_summary
    }