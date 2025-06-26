"""
Model Version Manager for TextNLP
Handles versioning, tracking, and management of model versions
"""

import os
import json
import hashlib
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import semantic_version
import asyncio
import aiofiles
from pathlib import Path
import logging
import sqlite3
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model version status"""
    DRAFT = "draft"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ModelFormat(Enum):
    """Supported model formats"""
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    TENSORFLOW = "tensorflow"
    JAX = "jax"


@dataclass
class ModelMetadata:
    """Metadata for a model version"""
    name: str
    version: str
    format: ModelFormat
    framework_version: str
    architecture: str
    parameters: int
    size_bytes: int
    checksum: str
    created_at: datetime
    created_by: str
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, str] = field(default_factory=dict)
    

@dataclass
class ModelVersion:
    """Complete model version information"""
    id: str
    metadata: ModelMetadata
    status: ModelStatus
    storage_path: str
    parent_version: Optional[str] = None
    children_versions: List[str] = field(default_factory=list)
    deployment_info: Dict[str, Any] = field(default_factory=dict)
    changelog: str = ""
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    

class ModelVersionManager:
    """Manages model versions with full lifecycle support"""
    
    def __init__(self, storage_backend: Any, db_path: str = "model_versions.db"):
        self.storage = storage_backend
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for version tracking"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                status TEXT NOT NULL,
                metadata JSON NOT NULL,
                storage_path TEXT NOT NULL,
                parent_version TEXT,
                changelog TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                UNIQUE(name, version)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_deployments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                environment TEXT NOT NULL,
                deployed_at TIMESTAMP NOT NULL,
                deployed_by TEXT NOT NULL,
                configuration JSON,
                FOREIGN KEY (model_id) REFERENCES model_versions(id)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                measured_at TIMESTAMP NOT NULL,
                FOREIGN KEY (model_id) REFERENCES model_versions(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    @asynccontextmanager
    async def _get_db_connection(self):
        """Get async database connection"""
        loop = asyncio.get_event_loop()
        conn = await loop.run_in_executor(None, sqlite3.connect, self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _generate_model_id(self, name: str, version: str) -> str:
        """Generate unique model ID"""
        return f"{name}-{version}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    
    def _validate_version(self, version: str) -> semantic_version.Version:
        """Validate and parse semantic version"""
        try:
            return semantic_version.Version(version)
        except ValueError:
            raise ValueError(f"Invalid semantic version: {version}")
    
    async def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of model file"""
        sha256_hash = hashlib.sha256()
        
        async with aiofiles.open(file_path, "rb") as f:
            while chunk := await f.read(8192):
                sha256_hash.update(chunk)
                
        return sha256_hash.hexdigest()
    
    async def register_model(self, model_path: str, metadata: ModelMetadata,
                           parent_version: Optional[str] = None) -> ModelVersion:
        """Register a new model version"""
        # Validate version
        version_obj = self._validate_version(metadata.version)
        
        # Check if version already exists
        existing = await self.get_model(metadata.name, metadata.version)
        if existing:
            raise ValueError(f"Model {metadata.name} version {metadata.version} already exists")
        
        # Validate parent version if specified
        if parent_version:
            parent = await self.get_model_by_id(parent_version)
            if not parent:
                raise ValueError(f"Parent version {parent_version} not found")
            
            # Ensure version is higher than parent
            parent_version_obj = self._validate_version(parent.metadata.version)
            if version_obj <= parent_version_obj:
                raise ValueError(f"Version {metadata.version} must be higher than parent {parent.metadata.version}")
        
        # Calculate checksum
        checksum = await self._calculate_checksum(model_path)
        metadata.checksum = checksum
        
        # Generate model ID
        model_id = self._generate_model_id(metadata.name, metadata.version)
        
        # Upload model to storage
        storage_path = f"models/{metadata.name}/{metadata.version}/model.{metadata.format.value}"
        await self.storage.upload_file(model_path, storage_path)
        
        # Create model version
        model_version = ModelVersion(
            id=model_id,
            metadata=metadata,
            status=ModelStatus.DRAFT,
            storage_path=storage_path,
            parent_version=parent_version
        )
        
        # Save to database
        async with self._get_db_connection() as conn:
            await asyncio.get_event_loop().run_in_executor(
                None,
                conn.execute,
                """
                INSERT INTO model_versions 
                (id, name, version, status, metadata, storage_path, parent_version, 
                 changelog, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    metadata.name,
                    metadata.version,
                    ModelStatus.DRAFT.value,
                    json.dumps(asdict(metadata), default=str),
                    storage_path,
                    parent_version,
                    "",
                    datetime.utcnow(),
                    datetime.utcnow()
                )
            )
            await asyncio.get_event_loop().run_in_executor(None, conn.commit)
        
        # Update parent's children if applicable
        if parent_version:
            await self._add_child_version(parent_version, model_id)
        
        logger.info(f"Registered model {metadata.name} version {metadata.version}")
        
        return model_version
    
    async def _add_child_version(self, parent_id: str, child_id: str):
        """Add child version to parent"""
        parent = await self.get_model_by_id(parent_id)
        if parent:
            parent.children_versions.append(child_id)
            # Update in database (simplified - would need proper JSON handling)
    
    async def get_model(self, name: str, version: str) -> Optional[ModelVersion]:
        """Get specific model version"""
        async with self._get_db_connection() as conn:
            cursor = await asyncio.get_event_loop().run_in_executor(
                None,
                conn.execute,
                "SELECT * FROM model_versions WHERE name = ? AND version = ?",
                (name, version)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_model_version(row)
            return None
    
    async def get_model_by_id(self, model_id: str) -> Optional[ModelVersion]:
        """Get model by ID"""
        async with self._get_db_connection() as conn:
            cursor = await asyncio.get_event_loop().run_in_executor(
                None,
                conn.execute,
                "SELECT * FROM model_versions WHERE id = ?",
                (model_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_model_version(row)
            return None
    
    def _row_to_model_version(self, row: sqlite3.Row) -> ModelVersion:
        """Convert database row to ModelVersion"""
        metadata_dict = json.loads(row['metadata'])
        metadata = ModelMetadata(
            name=metadata_dict['name'],
            version=metadata_dict['version'],
            format=ModelFormat(metadata_dict['format']),
            framework_version=metadata_dict['framework_version'],
            architecture=metadata_dict['architecture'],
            parameters=metadata_dict['parameters'],
            size_bytes=metadata_dict['size_bytes'],
            checksum=metadata_dict['checksum'],
            created_at=datetime.fromisoformat(metadata_dict['created_at']),
            created_by=metadata_dict['created_by'],
            tags=metadata_dict.get('tags', []),
            metrics=metadata_dict.get('metrics', {}),
            config=metadata_dict.get('config', {}),
            dependencies=metadata_dict.get('dependencies', {})
        )
        
        return ModelVersion(
            id=row['id'],
            metadata=metadata,
            status=ModelStatus(row['status']),
            storage_path=row['storage_path'],
            parent_version=row['parent_version'],
            changelog=row['changelog']
        )
    
    async def list_models(self, name: Optional[str] = None,
                         status: Optional[ModelStatus] = None,
                         tags: Optional[List[str]] = None) -> List[ModelVersion]:
        """List models with optional filtering"""
        query = "SELECT * FROM model_versions WHERE 1=1"
        params = []
        
        if name:
            query += " AND name = ?"
            params.append(name)
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        query += " ORDER BY name, version DESC"
        
        async with self._get_db_connection() as conn:
            cursor = await asyncio.get_event_loop().run_in_executor(
                None,
                conn.execute,
                query,
                params
            )
            rows = cursor.fetchall()
            
            models = [self._row_to_model_version(row) for row in rows]
            
            # Filter by tags if specified
            if tags:
                models = [m for m in models if any(tag in m.metadata.tags for tag in tags)]
            
            return models
    
    async def get_latest_version(self, name: str, 
                               status: Optional[ModelStatus] = None) -> Optional[ModelVersion]:
        """Get latest version of a model"""
        models = await self.list_models(name=name, status=status)
        
        if not models:
            return None
        
        # Sort by semantic version
        sorted_models = sorted(
            models,
            key=lambda m: self._validate_version(m.metadata.version),
            reverse=True
        )
        
        return sorted_models[0]
    
    async def promote_model(self, model_id: str, new_status: ModelStatus,
                          approved_by: str, changelog: str = "") -> ModelVersion:
        """Promote model to new status"""
        model = await self.get_model_by_id(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Validate status transition
        valid_transitions = {
            ModelStatus.DRAFT: [ModelStatus.TESTING],
            ModelStatus.TESTING: [ModelStatus.STAGING, ModelStatus.DRAFT],
            ModelStatus.STAGING: [ModelStatus.PRODUCTION, ModelStatus.TESTING],
            ModelStatus.PRODUCTION: [ModelStatus.DEPRECATED],
            ModelStatus.DEPRECATED: [ModelStatus.ARCHIVED]
        }
        
        if new_status not in valid_transitions.get(model.status, []):
            raise ValueError(f"Invalid status transition from {model.status} to {new_status}")
        
        # Update model
        model.status = new_status
        model.approved_by = approved_by
        model.approved_at = datetime.utcnow()
        if changelog:
            model.changelog = changelog
        
        # Update database
        async with self._get_db_connection() as conn:
            await asyncio.get_event_loop().run_in_executor(
                None,
                conn.execute,
                """
                UPDATE model_versions 
                SET status = ?, changelog = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_status.value, changelog, datetime.utcnow(), model_id)
            )
            await asyncio.get_event_loop().run_in_executor(None, conn.commit)
        
        logger.info(f"Promoted model {model_id} to {new_status}")
        
        return model
    
    async def deploy_model(self, model_id: str, environment: str,
                         deployed_by: str, configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy model to environment"""
        model = await self.get_model_by_id(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Only production models can be deployed to production environment
        if environment == "production" and model.status != ModelStatus.PRODUCTION:
            raise ValueError("Only production models can be deployed to production environment")
        
        # Record deployment
        async with self._get_db_connection() as conn:
            cursor = await asyncio.get_event_loop().run_in_executor(
                None,
                conn.execute,
                """
                INSERT INTO model_deployments 
                (model_id, environment, deployed_at, deployed_by, configuration)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    environment,
                    datetime.utcnow(),
                    deployed_by,
                    json.dumps(configuration)
                )
            )
            deployment_id = cursor.lastrowid
            await asyncio.get_event_loop().run_in_executor(None, conn.commit)
        
        deployment_info = {
            "deployment_id": deployment_id,
            "model_id": model_id,
            "environment": environment,
            "deployed_at": datetime.utcnow().isoformat(),
            "deployed_by": deployed_by,
            "configuration": configuration
        }
        
        logger.info(f"Deployed model {model_id} to {environment}")
        
        return deployment_info
    
    async def get_deployment_history(self, model_id: Optional[str] = None,
                                   environment: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get deployment history"""
        query = "SELECT * FROM model_deployments WHERE 1=1"
        params = []
        
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        
        if environment:
            query += " AND environment = ?"
            params.append(environment)
        
        query += " ORDER BY deployed_at DESC"
        
        async with self._get_db_connection() as conn:
            cursor = await asyncio.get_event_loop().run_in_executor(
                None,
                conn.execute,
                query,
                params
            )
            rows = cursor.fetchall()
            
            deployments = []
            for row in rows:
                deployments.append({
                    "deployment_id": row['id'],
                    "model_id": row['model_id'],
                    "environment": row['environment'],
                    "deployed_at": row['deployed_at'],
                    "deployed_by": row['deployed_by'],
                    "configuration": json.loads(row['configuration'] or '{}')
                })
            
            return deployments
    
    async def add_metrics(self, model_id: str, metrics: Dict[str, float]):
        """Add performance metrics to model"""
        model = await self.get_model_by_id(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Update model metrics
        model.metadata.metrics.update(metrics)
        
        # Record metrics in database
        async with self._get_db_connection() as conn:
            for metric_name, metric_value in metrics.items():
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    conn.execute,
                    """
                    INSERT INTO model_metrics 
                    (model_id, metric_name, metric_value, measured_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (model_id, metric_name, metric_value, datetime.utcnow())
                )
            
            # Update model metadata
            await asyncio.get_event_loop().run_in_executor(
                None,
                conn.execute,
                """
                UPDATE model_versions 
                SET metadata = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(asdict(model.metadata), default=str), datetime.utcnow(), model_id)
            )
            
            await asyncio.get_event_loop().run_in_executor(None, conn.commit)
        
        logger.info(f"Added metrics to model {model_id}: {metrics}")
    
    async def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Get complete lineage of a model"""
        model = await self.get_model_by_id(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        lineage = {
            "model": model,
            "ancestors": [],
            "descendants": []
        }
        
        # Trace ancestors
        current = model
        while current.parent_version:
            parent = await self.get_model_by_id(current.parent_version)
            if parent:
                lineage["ancestors"].append(parent)
                current = parent
            else:
                break
        
        # Trace descendants
        async def get_descendants(model_id: str, collected: List):
            model = await self.get_model_by_id(model_id)
            if model:
                for child_id in model.children_versions:
                    child = await self.get_model_by_id(child_id)
                    if child:
                        collected.append(child)
                        await get_descendants(child_id, collected)
        
        await get_descendants(model_id, lineage["descendants"])
        
        return lineage
    
    async def compare_models(self, model_id1: str, model_id2: str) -> Dict[str, Any]:
        """Compare two model versions"""
        model1 = await self.get_model_by_id(model_id1)
        model2 = await self.get_model_by_id(model_id2)
        
        if not model1 or not model2:
            raise ValueError("One or both models not found")
        
        comparison = {
            "model1": model1,
            "model2": model2,
            "differences": {
                "version": {
                    "model1": model1.metadata.version,
                    "model2": model2.metadata.version
                },
                "size": {
                    "model1": model1.metadata.size_bytes,
                    "model2": model2.metadata.size_bytes,
                    "difference": model2.metadata.size_bytes - model1.metadata.size_bytes
                },
                "parameters": {
                    "model1": model1.metadata.parameters,
                    "model2": model2.metadata.parameters,
                    "difference": model2.metadata.parameters - model1.metadata.parameters
                },
                "metrics": {}
            }
        }
        
        # Compare metrics
        all_metrics = set(model1.metadata.metrics.keys()) | set(model2.metadata.metrics.keys())
        for metric in all_metrics:
            val1 = model1.metadata.metrics.get(metric, 0)
            val2 = model2.metadata.metrics.get(metric, 0)
            comparison["differences"]["metrics"][metric] = {
                "model1": val1,
                "model2": val2,
                "difference": val2 - val1,
                "improvement": ((val2 - val1) / val1 * 100) if val1 != 0 else 0
            }
        
        return comparison
    
    async def cleanup_old_versions(self, name: str, keep_versions: int = 5,
                                 keep_production: bool = True) -> List[str]:
        """Clean up old model versions"""
        models = await self.list_models(name=name)
        
        # Sort by version
        sorted_models = sorted(
            models,
            key=lambda m: self._validate_version(m.metadata.version),
            reverse=True
        )
        
        # Identify models to delete
        models_to_delete = []
        kept_count = 0
        
        for model in sorted_models:
            if kept_count >= keep_versions:
                if keep_production and model.status == ModelStatus.PRODUCTION:
                    continue
                if model.status not in [ModelStatus.ARCHIVED, ModelStatus.DEPRECATED]:
                    models_to_delete.append(model.id)
            else:
                kept_count += 1
        
        # Delete models
        deleted = []
        for model_id in models_to_delete:
            try:
                await self.delete_model(model_id)
                deleted.append(model_id)
            except Exception as e:
                logger.error(f"Failed to delete model {model_id}: {e}")
        
        logger.info(f"Cleaned up {len(deleted)} old versions of {name}")
        
        return deleted
    
    async def delete_model(self, model_id: str):
        """Delete a model version"""
        model = await self.get_model_by_id(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Don't delete production models
        if model.status == ModelStatus.PRODUCTION:
            raise ValueError("Cannot delete production models")
        
        # Delete from storage
        await self.storage.delete_file(model.storage_path)
        
        # Delete from database
        async with self._get_db_connection() as conn:
            await asyncio.get_event_loop().run_in_executor(
                None,
                conn.execute,
                "DELETE FROM model_versions WHERE id = ?",
                (model_id,)
            )
            await asyncio.get_event_loop().run_in_executor(None, conn.commit)
        
        logger.info(f"Deleted model {model_id}")


class ModelRegistry:
    """High-level model registry interface"""
    
    def __init__(self, version_manager: ModelVersionManager):
        self.version_manager = version_manager
    
    async def publish_model(self, model_path: str, name: str, version: str,
                          architecture: str, created_by: str,
                          tags: Optional[List[str]] = None,
                          parent_version: Optional[str] = None) -> ModelVersion:
        """Publish a new model to the registry"""
        # Detect model format
        if model_path.endswith('.safetensors'):
            format = ModelFormat.SAFETENSORS
        elif model_path.endswith('.onnx'):
            format = ModelFormat.ONNX
        elif model_path.endswith('.pt') or model_path.endswith('.pth'):
            format = ModelFormat.PYTORCH
        else:
            format = ModelFormat.PYTORCH  # Default
        
        # Get model info
        size_bytes = os.path.getsize(model_path)
        
        # Create metadata
        metadata = ModelMetadata(
            name=name,
            version=version,
            format=format,
            framework_version="2.0.0",  # Would detect actual version
            architecture=architecture,
            parameters=0,  # Would calculate from model
            size_bytes=size_bytes,
            checksum="",  # Will be calculated
            created_at=datetime.utcnow(),
            created_by=created_by,
            tags=tags or []
        )
        
        # Register model
        return await self.version_manager.register_model(
            model_path=model_path,
            metadata=metadata,
            parent_version=parent_version
        )
    
    async def get_model_for_inference(self, name: str, 
                                    version: Optional[str] = None) -> Tuple[str, ModelVersion]:
        """Get model path for inference"""
        if version:
            model = await self.version_manager.get_model(name, version)
        else:
            # Get latest production model
            model = await self.version_manager.get_latest_version(
                name, 
                status=ModelStatus.PRODUCTION
            )
            
            if not model:
                # Fall back to latest staging
                model = await self.version_manager.get_latest_version(
                    name,
                    status=ModelStatus.STAGING
                )
        
        if not model:
            raise ValueError(f"No suitable model found for {name}")
        
        # Download model if needed
        local_path = f"/tmp/models/{model.id}"
        if not os.path.exists(local_path):
            await self.version_manager.storage.download_file(
                model.storage_path,
                local_path
            )
        
        return local_path, model


# Example usage
if __name__ == "__main__":
    async def example():
        # Initialize storage backend (placeholder)
        class DummyStorage:
            async def upload_file(self, local_path: str, remote_path: str):
                print(f"Uploading {local_path} to {remote_path}")
            
            async def download_file(self, remote_path: str, local_path: str):
                print(f"Downloading {remote_path} to {local_path}")
            
            async def delete_file(self, remote_path: str):
                print(f"Deleting {remote_path}")
        
        storage = DummyStorage()
        
        # Create version manager
        version_manager = ModelVersionManager(storage)
        
        # Create registry
        registry = ModelRegistry(version_manager)
        
        # Publish a model
        model = await registry.publish_model(
            model_path="/path/to/model.safetensors",
            name="gpt2-custom",
            version="1.0.0",
            architecture="transformer",
            created_by="john.doe",
            tags=["nlp", "text-generation"]
        )
        
        print(f"Published model: {model.id}")
        
        # Promote to testing
        model = await version_manager.promote_model(
            model.id,
            ModelStatus.TESTING,
            approved_by="jane.smith",
            changelog="Initial testing release"
        )
        
        # Add metrics
        await version_manager.add_metrics(
            model.id,
            {
                "perplexity": 15.2,
                "accuracy": 0.94,
                "latency_ms": 45
            }
        )
        
        # Get model for inference
        model_path, model_info = await registry.get_model_for_inference("gpt2-custom")
        print(f"Model ready for inference at: {model_path}")
    
    # Run example
    # asyncio.run(example())