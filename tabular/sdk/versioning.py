"""
Model versioning and rollback capabilities
"""

import os
import json
import pickle
import shutil
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
import hashlib
from dataclasses import dataclass, field
import threading

import pandas as pd

from .base import BaseSyntheticGenerator, SyntheticDataConfig


@dataclass
class ModelVersion:
    """Information about a model version"""
    version_id: str
    version_number: int
    created_at: datetime
    generator_type: str
    model_type: str
    config: Dict[str, Any]
    data_hash: str
    model_path: str
    metadata_path: str
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    is_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'version_id': self.version_id,
            'version_number': self.version_number,
            'created_at': self.created_at.isoformat(),
            'generator_type': self.generator_type,
            'model_type': self.model_type,
            'config': self.config,
            'data_hash': self.data_hash,
            'model_path': self.model_path,
            'metadata_path': self.metadata_path,
            'metrics': self.metrics,
            'tags': self.tags,
            'description': self.description,
            'is_active': self.is_active
        }


class ModelVersionManager:
    """Manage model versions with rollback support"""
    
    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else Path.home() / '.inferloop' / 'models'
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.storage_dir / 'registry.json'
        self.lock = threading.Lock()
        self._load_registry()
    
    def _load_registry(self):
        """Load model registry"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                registry_data = json.load(f)
                self.registry = {}
                for model_id, versions in registry_data.items():
                    self.registry[model_id] = {}
                    for version_num, version_data in versions.items():
                        version = ModelVersion(
                            version_id=version_data['version_id'],
                            version_number=int(version_num),
                            created_at=datetime.fromisoformat(version_data['created_at']),
                            generator_type=version_data['generator_type'],
                            model_type=version_data['model_type'],
                            config=version_data['config'],
                            data_hash=version_data['data_hash'],
                            model_path=version_data['model_path'],
                            metadata_path=version_data['metadata_path'],
                            metrics=version_data.get('metrics', {}),
                            tags=version_data.get('tags', []),
                            description=version_data.get('description', ''),
                            is_active=version_data.get('is_active', False)
                        )
                        self.registry[model_id][int(version_num)] = version
        else:
            self.registry = {}
    
    def _save_registry(self):
        """Save model registry"""
        registry_data = {}
        for model_id, versions in self.registry.items():
            registry_data[model_id] = {}
            for version_num, version in versions.items():
                registry_data[model_id][str(version_num)] = version.to_dict()
        
        with open(self.registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def _generate_model_id(self, generator_type: str, model_type: str, data: pd.DataFrame) -> str:
        """Generate unique model ID"""
        # Create hash from data characteristics
        data_info = f"{data.shape}:{sorted(data.columns.tolist())}:{data.dtypes.to_dict()}"
        data_hash = hashlib.md5(data_info.encode()).hexdigest()[:8]
        
        return f"{generator_type}_{model_type}_{data_hash}"
    
    def _generate_version_id(self) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(os.urandom(16)).hexdigest()[:6]
        return f"v_{timestamp}_{random_suffix}"
    
    def _hash_data(self, data: pd.DataFrame) -> str:
        """Create hash of dataset"""
        # Use data shape and sample for hashing
        shape_str = f"{data.shape[0]}x{data.shape[1]}"
        columns_str = ",".join(sorted(data.columns))
        
        # Sample data for hashing
        sample_size = min(100, len(data))
        sample_data = data.sample(n=sample_size, random_state=42).to_json()
        
        combined = f"{shape_str}:{columns_str}:{sample_data}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def save_model(self,
                   generator: BaseSyntheticGenerator,
                   data: pd.DataFrame,
                   metrics: Optional[Dict[str, float]] = None,
                   tags: Optional[List[str]] = None,
                   description: str = "",
                   set_active: bool = True) -> ModelVersion:
        """Save a trained model as a new version"""
        with self.lock:
            # Generate model ID
            model_id = self._generate_model_id(
                generator.config.generator_type,
                generator.config.model_type,
                data
            )
            
            # Get next version number
            if model_id not in self.registry:
                self.registry[model_id] = {}
            
            existing_versions = self.registry[model_id]
            next_version = max(existing_versions.keys(), default=0) + 1
            
            # Generate version ID
            version_id = self._generate_version_id()
            
            # Create version directory
            version_dir = self.storage_dir / model_id / f"v{next_version}"
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = version_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(generator.model, f)
            
            # Save metadata
            metadata = {
                'version_id': version_id,
                'version_number': next_version,
                'created_at': datetime.now().isoformat(),
                'generator_type': generator.config.generator_type,
                'model_type': generator.config.model_type,
                'config': generator.config.to_dict(),
                'data_shape': data.shape,
                'data_columns': data.columns.tolist(),
                'data_dtypes': data.dtypes.astype(str).to_dict(),
                'model_info': generator.get_model_info() if hasattr(generator, 'get_model_info') else {},
                'metrics': metrics or {},
                'tags': tags or [],
                'description': description
            }
            
            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create version object
            version = ModelVersion(
                version_id=version_id,
                version_number=next_version,
                created_at=datetime.now(),
                generator_type=generator.config.generator_type,
                model_type=generator.config.model_type,
                config=generator.config.to_dict(),
                data_hash=self._hash_data(data),
                model_path=str(model_path),
                metadata_path=str(metadata_path),
                metrics=metrics or {},
                tags=tags or [],
                description=description,
                is_active=False
            )
            
            # Add to registry
            self.registry[model_id][next_version] = version
            
            # Set as active if requested
            if set_active:
                self.set_active_version(model_id, next_version)
            
            # Save registry
            self._save_registry()
            
            return version
    
    def load_model(self,
                   model_id: str,
                   version: Optional[int] = None,
                   generator_class: Optional[type] = None) -> BaseSyntheticGenerator:
        """Load a specific model version"""
        with self.lock:
            if model_id not in self.registry:
                raise ValueError(f"Model '{model_id}' not found in registry")
            
            versions = self.registry[model_id]
            
            # Get version to load
            if version is None:
                # Get active version
                active_versions = [v for v in versions.values() if v.is_active]
                if not active_versions:
                    # Get latest version
                    version = max(versions.keys())
                else:
                    version = active_versions[0].version_number
            
            if version not in versions:
                raise ValueError(f"Version {version} not found for model '{model_id}'")
            
            version_info = versions[version]
            
            # Load model
            with open(version_info.model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Create generator instance
            if generator_class is None:
                # Try to determine from config
                from .factory import GeneratorFactory
                config = SyntheticDataConfig(**version_info.config)
                generator = GeneratorFactory.create_generator(config)
            else:
                config = SyntheticDataConfig(**version_info.config)
                generator = generator_class(config)
            
            # Set model
            generator.model = model
            generator.is_fitted = True
            
            return generator
    
    def set_active_version(self, model_id: str, version: int):
        """Set a specific version as active"""
        with self.lock:
            if model_id not in self.registry:
                raise ValueError(f"Model '{model_id}' not found")
            
            if version not in self.registry[model_id]:
                raise ValueError(f"Version {version} not found for model '{model_id}'")
            
            # Deactivate all versions
            for v in self.registry[model_id].values():
                v.is_active = False
            
            # Activate specified version
            self.registry[model_id][version].is_active = True
            
            # Save registry
            self._save_registry()
    
    def rollback(self, model_id: str, target_version: int):
        """Rollback to a specific version"""
        self.set_active_version(model_id, target_version)
    
    def list_models(self) -> List[str]:
        """List all model IDs"""
        return list(self.registry.keys())
    
    def list_versions(self, model_id: str) -> List[ModelVersion]:
        """List all versions of a model"""
        if model_id not in self.registry:
            raise ValueError(f"Model '{model_id}' not found")
        
        versions = list(self.registry[model_id].values())
        versions.sort(key=lambda v: v.version_number, reverse=True)
        return versions
    
    def get_active_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get the active version of a model"""
        if model_id not in self.registry:
            return None
        
        for version in self.registry[model_id].values():
            if version.is_active:
                return version
        
        return None
    
    def delete_version(self, model_id: str, version: int):
        """Delete a specific version"""
        with self.lock:
            if model_id not in self.registry:
                raise ValueError(f"Model '{model_id}' not found")
            
            if version not in self.registry[model_id]:
                raise ValueError(f"Version {version} not found")
            
            version_info = self.registry[model_id][version]
            
            if version_info.is_active:
                raise ValueError("Cannot delete active version. Set another version as active first.")
            
            # Delete files
            version_dir = Path(version_info.model_path).parent
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            # Remove from registry
            del self.registry[model_id][version]
            
            # If no versions left, remove model entry
            if not self.registry[model_id]:
                del self.registry[model_id]
            
            # Save registry
            self._save_registry()
    
    def compare_versions(self, model_id: str, version1: int, version2: int) -> Dict[str, Any]:
        """Compare two versions of a model"""
        if model_id not in self.registry:
            raise ValueError(f"Model '{model_id}' not found")
        
        v1 = self.registry[model_id].get(version1)
        v2 = self.registry[model_id].get(version2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'created_at_diff': (v2.created_at - v1.created_at).total_seconds(),
            'config_changes': self._diff_configs(v1.config, v2.config),
            'metrics_comparison': self._compare_metrics(v1.metrics, v2.metrics),
            'tags_added': list(set(v2.tags) - set(v1.tags)),
            'tags_removed': list(set(v1.tags) - set(v2.tags))
        }
        
        return comparison
    
    def _diff_configs(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """Find differences between configurations"""
        changes = {}
        
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            if key not in config1:
                changes[key] = {'added': config2[key]}
            elif key not in config2:
                changes[key] = {'removed': config1[key]}
            elif config1[key] != config2[key]:
                changes[key] = {'old': config1[key], 'new': config2[key]}
        
        return changes
    
    def _compare_metrics(self, metrics1: Dict[str, float], metrics2: Dict[str, float]) -> Dict[str, Any]:
        """Compare metrics between versions"""
        comparison = {}
        
        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        
        for metric in all_metrics:
            if metric in metrics1 and metric in metrics2:
                diff = metrics2[metric] - metrics1[metric]
                pct_change = (diff / metrics1[metric] * 100) if metrics1[metric] != 0 else 0
                comparison[metric] = {
                    'v1': metrics1[metric],
                    'v2': metrics2[metric],
                    'diff': diff,
                    'pct_change': pct_change
                }
            elif metric in metrics1:
                comparison[metric] = {'v1': metrics1[metric], 'v2': None}
            else:
                comparison[metric] = {'v1': None, 'v2': metrics2[metric]}
        
        return comparison
    
    def tag_version(self, model_id: str, version: int, tags: List[str]):
        """Add tags to a version"""
        with self.lock:
            if model_id not in self.registry:
                raise ValueError(f"Model '{model_id}' not found")
            
            if version not in self.registry[model_id]:
                raise ValueError(f"Version {version} not found")
            
            version_info = self.registry[model_id][version]
            version_info.tags.extend(tags)
            version_info.tags = list(set(version_info.tags))  # Remove duplicates
            
            # Update metadata file
            metadata_path = Path(version_info.metadata_path)
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                metadata['tags'] = version_info.tags
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Save registry
            self._save_registry()
    
    def export_version(self, model_id: str, version: int, export_path: str):
        """Export a model version to a file"""
        with self.lock:
            if model_id not in self.registry:
                raise ValueError(f"Model '{model_id}' not found")
            
            if version not in self.registry[model_id]:
                raise ValueError(f"Version {version} not found")
            
            version_info = self.registry[model_id][version]
            version_dir = Path(version_info.model_path).parent
            
            # Create archive
            export_file = Path(export_path)
            shutil.make_archive(
                export_file.stem,
                'zip',
                version_dir
            )
            
            return f"{export_file.stem}.zip"
    
    def import_version(self, model_id: str, import_path: str, set_active: bool = False) -> ModelVersion:
        """Import a model version from file"""
        with self.lock:
            # Extract archive
            import_file = Path(import_path)
            temp_dir = self.storage_dir / 'temp' / import_file.stem
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.unpack_archive(import_path, temp_dir)
            
            # Load metadata
            metadata_path = temp_dir / 'metadata.json'
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Get next version number
            if model_id not in self.registry:
                self.registry[model_id] = {}
            
            next_version = max(self.registry[model_id].keys(), default=0) + 1
            
            # Create new version directory
            version_dir = self.storage_dir / model_id / f"v{next_version}"
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files
            shutil.copytree(temp_dir, version_dir, dirs_exist_ok=True)
            
            # Update paths in metadata
            metadata['version_number'] = next_version
            metadata['model_path'] = str(version_dir / 'model.pkl')
            metadata['metadata_path'] = str(version_dir / 'metadata.json')
            
            # Save updated metadata
            with open(version_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create version object
            version = ModelVersion(
                version_id=metadata['version_id'],
                version_number=next_version,
                created_at=datetime.fromisoformat(metadata['created_at']),
                generator_type=metadata['generator_type'],
                model_type=metadata['model_type'],
                config=metadata['config'],
                data_hash=metadata.get('data_hash', ''),
                model_path=str(version_dir / 'model.pkl'),
                metadata_path=str(version_dir / 'metadata.json'),
                metrics=metadata.get('metrics', {}),
                tags=metadata.get('tags', []),
                description=metadata.get('description', ''),
                is_active=False
            )
            
            # Add to registry
            self.registry[model_id][next_version] = version
            
            # Set active if requested
            if set_active:
                self.set_active_version(model_id, next_version)
            
            # Clean up temp directory
            shutil.rmtree(temp_dir.parent)
            
            # Save registry
            self._save_registry()
            
            return version


# Global version manager instance
_version_manager = None

def get_version_manager(storage_dir: Optional[str] = None) -> ModelVersionManager:
    """Get global version manager instance"""
    global _version_manager
    if _version_manager is None:
        _version_manager = ModelVersionManager(storage_dir)
    return _version_manager


class VersionedGenerator:
    """Wrapper for generators with versioning support"""
    
    def __init__(self, generator: BaseSyntheticGenerator, 
                 version_manager: Optional[ModelVersionManager] = None):
        self.generator = generator
        self.version_manager = version_manager or get_version_manager()
        self.current_version: Optional[ModelVersion] = None
    
    def fit(self, data: pd.DataFrame, save_version: bool = True, **kwargs) -> None:
        """Fit model and optionally save version"""
        # Fit the model
        self.generator.fit(data)
        
        # Save version if requested
        if save_version:
            self.current_version = self.version_manager.save_model(
                self.generator,
                data,
                **kwargs
            )
    
    def load_version(self, model_id: str, version: Optional[int] = None) -> None:
        """Load a specific version"""
        loaded_generator = self.version_manager.load_model(
            model_id,
            version,
            type(self.generator)
        )
        
        # Update generator
        self.generator.model = loaded_generator.model
        self.generator.is_fitted = loaded_generator.is_fitted
        self.generator.config = loaded_generator.config
        
        # Update current version
        versions = self.version_manager.list_versions(model_id)
        if version is None:
            self.current_version = self.version_manager.get_active_version(model_id)
        else:
            self.current_version = next((v for v in versions if v.version_number == version), None)
    
    def rollback(self, version: int) -> None:
        """Rollback to a previous version"""
        if not self.current_version:
            raise ValueError("No current version set")
        
        model_id = f"{self.current_version.generator_type}_{self.current_version.model_type}"
        self.load_version(model_id, version)