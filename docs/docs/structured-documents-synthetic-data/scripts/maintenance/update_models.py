#!/usr/bin/env python3
"""
Model update and management script for structured document synthesis.

Provides automated model downloading, updating, version management,
and dependency resolution for ML models used in the system.
"""

import asyncio
import json
import hashlib
import requests
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import subprocess
import aiohttp
import aiofiles
from packaging import version

# Model configuration
MODEL_CONFIG = {
    'default_model_dir': Path.home() / '.structured_docs_synth' / 'models',
    'model_registry_url': 'https://api.huggingface.co/models',
    'download_timeout': 3600,  # 1 hour
    'verification_enabled': True,
    'auto_backup_enabled': True,
    'parallel_downloads': 3,
    'retry_attempts': 3
}

# Supported model types and their configurations
SUPPORTED_MODELS = {
    'text_generation': {
        'models': [
            'microsoft/DialoGPT-medium',
            'microsoft/DialoGPT-large',
            'EleutherAI/gpt-neo-1.3B',
            'EleutherAI/gpt-neo-2.7B'
        ],
        'required_files': ['pytorch_model.bin', 'config.json', 'tokenizer.json']
    },
    'document_understanding': {
        'models': [
            'microsoft/layoutlm-base-uncased',
            'microsoft/layoutlmv2-base-uncased',
            'microsoft/layoutlmv3-base'
        ],
        'required_files': ['pytorch_model.bin', 'config.json']
    },
    'ocr': {
        'models': [
            'microsoft/trocr-base-printed',
            'microsoft/trocr-large-printed',
            'microsoft/trocr-base-handwritten'
        ],
        'required_files': ['pytorch_model.bin', 'config.json', 'preprocessor_config.json']
    },
    'embedding': {
        'models': [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        ],
        'required_files': ['pytorch_model.bin', 'config.json']
    }
}


class ModelUpdater:
    """Comprehensive model update and management system"""
    
    def __init__(self, model_dir: Optional[Path] = None, 
                 config: Optional[Dict[str, Any]] = None):
        self.model_dir = model_dir or MODEL_CONFIG['default_model_dir']
        self.config = {**MODEL_CONFIG, **(config or {})}
        
        # Ensure model directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model registry
        self.model_registry_file = self.model_dir / 'model_registry.json'
        self.model_registry = self._load_model_registry()
    
    async def update_all_models(self, force_update: bool = False) -> Dict[str, Any]:
        """
        Update all configured models.
        
        Args:
            force_update: Force update even if model is current
        
        Returns:
            Update results
        """
        print("= Starting model update process...")
        
        try:
            update_results = {
                'timestamp': datetime.now().isoformat(),
                'total_models': 0,
                'updated_models': 0,
                'failed_models': 0,
                'results': {}
            }
            
            # Process each model type
            for model_type, type_config in SUPPORTED_MODELS.items():
                print(f"=Â Processing {model_type} models...")
                
                type_results = []
                
                # Process models in parallel batches
                models = type_config['models']
                for i in range(0, len(models), self.config['parallel_downloads']):
                    batch = models[i:i + self.config['parallel_downloads']]
                    
                    # Create tasks for batch
                    tasks = [
                        self._update_single_model(model_name, model_type, type_config, force_update)
                        for model_name in batch
                    ]
                    
                    # Execute batch
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for model_name, result in zip(batch, batch_results):
                        if isinstance(result, Exception):
                            print(f"L Error updating {model_name}: {result}")
                            type_results.append({
                                'model': model_name,
                                'success': False,
                                'error': str(result)
                            })
                            update_results['failed_models'] += 1
                        else:
                            type_results.append(result)
                            if result['success'] and result.get('updated', False):
                                update_results['updated_models'] += 1
                        
                        update_results['total_models'] += 1
                
                update_results['results'][model_type] = type_results
            
            # Update model registry
            await self._save_model_registry()
            
            print(f" Model update completed:")
            print(f"=Ê Total: {update_results['total_models']}")
            print(f"  Updated: {update_results['updated_models']}")
            print(f"L Failed: {update_results['failed_models']}")
            
            return update_results
            
        except Exception as e:
            print(f"L Model update process failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def update_model(self, model_name: str, model_type: Optional[str] = None, 
                          force_update: bool = False) -> Dict[str, Any]:
        """
        Update a specific model.
        
        Args:
            model_name: Name of the model to update
            model_type: Type of model (if known)
            force_update: Force update even if current
        
        Returns:
            Update result
        """
        # Determine model type if not provided
        if not model_type:
            model_type = self._get_model_type(model_name)
            if not model_type:
                return {
                    'success': False,
                    'model': model_name,
                    'error': 'Unknown model type'
                }
        
        type_config = SUPPORTED_MODELS.get(model_type, {})
        return await self._update_single_model(model_name, model_type, type_config, force_update)
    
    async def list_models(self, include_details: bool = False) -> List[Dict[str, Any]]:
        """
        List all installed models.
        
        Args:
            include_details: Include detailed information
        
        Returns:
            List of model information
        """
        models = []
        
        for model_dir in self.model_dir.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith('.'):
                model_info = await self._get_model_info(model_dir, include_details)
                if model_info:
                    models.append(model_info)
        
        return sorted(models, key=lambda x: x['name'])
    
    async def verify_models(self) -> Dict[str, Any]:
        """
        Verify integrity of all installed models.
        
        Returns:
            Verification results
        """
        print("= Verifying model integrity...")
        
        verification_results = {
            'timestamp': datetime.now().isoformat(),
            'total_models': 0,
            'valid_models': 0,
            'invalid_models': 0,
            'results': {}
        }
        
        models = await self.list_models(include_details=True)
        
        for model in models:
            model_name = model['name']
            model_path = Path(model['path'])
            
            print(f"= Verifying {model_name}...")
            
            result = await self._verify_single_model(model_path, model)
            verification_results['results'][model_name] = result
            verification_results['total_models'] += 1
            
            if result['valid']:
                verification_results['valid_models'] += 1
                print(f" {model_name}: Valid")
            else:
                verification_results['invalid_models'] += 1
                print(f"L {model_name}: Invalid - {result.get('error', 'Unknown error')}")
        
        print(f"=Ê Verification completed:")
        print(f"Total: {verification_results['total_models']}")
        print(f"Valid: {verification_results['valid_models']}")
        print(f"Invalid: {verification_results['invalid_models']}")
        
        return verification_results
    
    async def cleanup_models(self, keep_versions: int = 2) -> Dict[str, Any]:
        """
        Clean up old model versions.
        
        Args:
            keep_versions: Number of versions to keep per model
        
        Returns:
            Cleanup results
        """
        print(f">ù Cleaning up old model versions (keeping {keep_versions} versions)...")
        
        cleanup_results = {
            'timestamp': datetime.now().isoformat(),
            'models_processed': 0,
            'versions_removed': 0,
            'space_freed_mb': 0,
            'results': {}
        }
        
        # Group models by base name
        model_groups = {}
        for model_dir in self.model_dir.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith('.'):
                base_name = self._get_base_model_name(model_dir.name)
                if base_name not in model_groups:
                    model_groups[base_name] = []
                model_groups[base_name].append(model_dir)
        
        # Clean up each model group
        for base_name, model_versions in model_groups.items():
            if len(model_versions) <= keep_versions:
                continue
            
            # Sort by modification time (newest first)
            model_versions.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old versions
            versions_to_remove = model_versions[keep_versions:]
            
            for model_dir in versions_to_remove:
                try:
                    # Calculate size before removal
                    size_mb = self._get_dir_size(model_dir) / (1024 * 1024)
                    
                    # Remove directory
                    shutil.rmtree(model_dir)
                    
                    cleanup_results['versions_removed'] += 1
                    cleanup_results['space_freed_mb'] += size_mb
                    
                    print(f"=Ñ  Removed {model_dir.name} ({size_mb:.1f} MB)")
                    
                except Exception as e:
                    print(f"   Error removing {model_dir.name}: {e}")
            
            cleanup_results['models_processed'] += 1
            cleanup_results['results'][base_name] = len(versions_to_remove)
        
        print(f" Cleanup completed:")
        print(f"=Ê Models processed: {cleanup_results['models_processed']}")
        print(f"=Ñ  Versions removed: {cleanup_results['versions_removed']}")
        print(f"=¾ Space freed: {cleanup_results['space_freed_mb']:.1f} MB")
        
        return cleanup_results
    
    async def download_model(self, model_name: str, model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Download a specific model.
        
        Args:
            model_name: Name of the model to download
            model_type: Type of model
        
        Returns:
            Download result
        """
        if not model_type:
            model_type = self._get_model_type(model_name)
            if not model_type:
                return {
                    'success': False,
                    'model': model_name,
                    'error': 'Unknown model type'
                }
        
        print(f"  Downloading {model_name}...")
        
        try:
            model_path = self.model_dir / model_name.replace('/', '_')
            
            # Create model directory
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Download model files
            type_config = SUPPORTED_MODELS.get(model_type, {})
            required_files = type_config.get('required_files', [])
            
            download_results = []
            
            for file_name in required_files:
                file_result = await self._download_model_file(model_name, file_name, model_path)
                download_results.append(file_result)
                
                if not file_result['success']:
                    # Cleanup on failure
                    if model_path.exists():
                        shutil.rmtree(model_path)
                    return {
                        'success': False,
                        'model': model_name,
                        'error': f"Failed to download {file_name}: {file_result.get('error', 'Unknown error')}"
                    }
            
            # Update model registry
            self.model_registry[model_name] = {
                'type': model_type,
                'path': str(model_path),
                'version': 'latest',
                'downloaded_at': datetime.now().isoformat(),
                'files': required_files,
                'size_mb': self._get_dir_size(model_path) / (1024 * 1024)
            }
            
            await self._save_model_registry()
            
            print(f" Downloaded {model_name}")
            
            return {
                'success': True,
                'model': model_name,
                'path': str(model_path),
                'files_downloaded': len(required_files),
                'size_mb': self.model_registry[model_name]['size_mb']
            }
            
        except Exception as e:
            print(f"L Error downloading {model_name}: {e}")
            return {
                'success': False,
                'model': model_name,
                'error': str(e)
            }
    
    # Private methods
    
    async def _update_single_model(self, model_name: str, model_type: str, 
                                  type_config: Dict[str, Any], force_update: bool) -> Dict[str, Any]:
        """Update a single model"""
        try:
            model_path = self.model_dir / model_name.replace('/', '_')
            
            # Check if model exists and is current
            if model_path.exists() and not force_update:
                is_current = await self._is_model_current(model_name, model_path)
                if is_current:
                    return {
                        'success': True,
                        'model': model_name,
                        'updated': False,
                        'reason': 'Already current'
                    }
            
            # Backup existing model if enabled
            if self.config['auto_backup_enabled'] and model_path.exists():
                await self._backup_model(model_path)
            
            # Download/update model
            download_result = await self.download_model(model_name, model_type)
            
            if download_result['success']:
                # Verify downloaded model
                if self.config['verification_enabled']:
                    model_info = await self._get_model_info(model_path, include_details=True)
                    verify_result = await self._verify_single_model(model_path, model_info)
                    
                    if not verify_result['valid']:
                        return {
                            'success': False,
                            'model': model_name,
                            'error': f"Verification failed: {verify_result.get('error', 'Unknown error')}"
                        }
                
                return {
                    'success': True,
                    'model': model_name,
                    'updated': True,
                    'size_mb': download_result.get('size_mb', 0)
                }
            else:
                return download_result
                
        except Exception as e:
            return {
                'success': False,
                'model': model_name,
                'error': str(e)
            }
    
    async def _download_model_file(self, model_name: str, file_name: str, 
                                  model_path: Path) -> Dict[str, Any]:
        """Download a single model file"""
        try:
            # Construct download URL (Hugging Face format)
            url = f"https://huggingface.co/{model_name}/resolve/main/{file_name}"
            file_path = model_path / file_name
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.config['download_timeout']) as response:
                    if response.status == 200:
                        async with aiofiles.open(file_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                await f.write(chunk)
                        
                        return {
                            'success': True,
                            'file': file_name,
                            'size_mb': file_path.stat().st_size / (1024 * 1024)
                        }
                    else:
                        return {
                            'success': False,
                            'file': file_name,
                            'error': f"HTTP {response.status}"
                        }
        
        except Exception as e:
            return {
                'success': False,
                'file': file_name,
                'error': str(e)
            }
    
    async def _is_model_current(self, model_name: str, model_path: Path) -> bool:
        """Check if model is current"""
        # Simple check - could be enhanced with version comparison
        if model_name in self.model_registry:
            model_info = self.model_registry[model_name]
            last_update = datetime.fromisoformat(model_info.get('downloaded_at', '1970-01-01'))
            days_old = (datetime.now() - last_update).days
            return days_old < 7  # Consider current if less than 7 days old
        
        return False
    
    async def _verify_single_model(self, model_path: Path, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a single model"""
        try:
            # Check if all required files exist
            model_type = model_info.get('type', 'unknown')
            type_config = SUPPORTED_MODELS.get(model_type, {})
            required_files = type_config.get('required_files', [])
            
            missing_files = []
            for file_name in required_files:
                file_path = model_path / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
            
            if missing_files:
                return {
                    'valid': False,
                    'error': f"Missing files: {', '.join(missing_files)}"
                }
            
            # Check file sizes (basic corruption check)
            for file_name in required_files:
                file_path = model_path / file_name
                if file_path.stat().st_size == 0:
                    return {
                        'valid': False,
                        'error': f"Zero-size file: {file_name}"
                    }
            
            return {'valid': True}
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e)
            }
    
    async def _get_model_info(self, model_path: Path, include_details: bool = False) -> Optional[Dict[str, Any]]:
        """Get model information"""
        try:
            model_name = model_path.name
            
            # Basic info
            info = {
                'name': model_name,
                'path': str(model_path),
                'size_mb': self._get_dir_size(model_path) / (1024 * 1024),
                'modified': datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
            }
            
            # Registry info if available
            registry_key = model_name.replace('_', '/')
            if registry_key in self.model_registry:
                registry_info = self.model_registry[registry_key]
                info.update({
                    'type': registry_info.get('type', 'unknown'),
                    'version': registry_info.get('version', 'unknown'),
                    'downloaded_at': registry_info.get('downloaded_at', 'unknown')
                })
            
            if include_details:
                # Count files
                info['file_count'] = len(list(model_path.iterdir()))
                
                # List files
                info['files'] = [f.name for f in model_path.iterdir() if f.is_file()]
            
            return info
            
        except Exception:
            return None
    
    async def _backup_model(self, model_path: Path):
        """Create backup of existing model"""
        backup_name = f"{model_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = model_path.parent / backup_name
        
        shutil.copytree(model_path, backup_path)
        print(f"=æ Created backup: {backup_name}")
    
    def _get_model_type(self, model_name: str) -> Optional[str]:
        """Determine model type from name"""
        for model_type, type_config in SUPPORTED_MODELS.items():
            if model_name in type_config['models']:
                return model_type
        return None
    
    def _get_base_model_name(self, model_dir_name: str) -> str:
        """Get base model name without version suffixes"""
        # Remove common version suffixes
        base_name = model_dir_name
        for suffix in ['_backup', '_v1', '_v2', '_v3', '_latest']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        return base_name
    
    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def _load_model_registry(self) -> Dict[str, Any]:
        """Load model registry from file"""
        if self.model_registry_file.exists():
            try:
                with open(self.model_registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"   Warning: Could not load model registry: {e}")
        
        return {}
    
    async def _save_model_registry(self):
        """Save model registry to file"""
        try:
            with open(self.model_registry_file, 'w') as f:
                json.dump(self.model_registry, f, indent=2)
        except Exception as e:
            print(f"   Warning: Could not save model registry: {e}")


async def main():
    """
    Main model update script function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Update and manage ML models for structured document synthesis'
    )
    parser.add_argument(
        'action',
        choices=['update', 'download', 'list', 'verify', 'cleanup'],
        help='Action to perform'
    )
    parser.add_argument(
        '--model',
        help='Specific model name to update/download'
    )
    parser.add_argument(
        '--model-type',
        choices=['text_generation', 'document_understanding', 'ocr', 'embedding'],
        help='Model type'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force update even if model is current'
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        help='Custom model directory'
    )
    parser.add_argument(
        '--details',
        action='store_true',
        help='Include detailed information (for list action)'
    )
    parser.add_argument(
        '--keep-versions',
        type=int,
        default=2,
        help='Number of versions to keep per model (for cleanup action)'
    )
    
    args = parser.parse_args()
    
    # Initialize model updater
    config = MODEL_CONFIG.copy()
    if args.model_dir:
        config['default_model_dir'] = args.model_dir
    
    updater = ModelUpdater(config=config)
    
    if args.action == 'update':
        if args.model:
            result = await updater.update_model(args.model, args.model_type, args.force)
            if result['success']:
                if result.get('updated', False):
                    print(f"\n Model updated successfully: {args.model}")
                else:
                    print(f"\n=Ë Model already current: {args.model}")
            else:
                print(f"\nL Model update failed: {result.get('error', 'Unknown error')}")
                return 1
        else:
            result = await updater.update_all_models(args.force)
            if result.get('failed_models', 0) > 0:
                return 1
    
    elif args.action == 'download':
        if not args.model:
            print("L Model name required for download")
            return 1
        
        result = await updater.download_model(args.model, args.model_type)
        
        if result['success']:
            print(f"\n Model downloaded successfully: {args.model}")
            print(f"=Á Path: {result['path']}")
            print(f"=Ê Size: {result['size_mb']:.1f} MB")
        else:
            print(f"\nL Model download failed: {result.get('error', 'Unknown error')}")
            return 1
    
    elif args.action == 'list':
        models = await updater.list_models(args.details)
        
        if not models:
            print("=Ë No models found")
            return 0
        
        print(f"=Ë Installed Models ({len(models)}):")
        print("=" * 80)
        
        for model in models:
            print(f"=æ {model['name']}")
            print(f"   Type: {model.get('type', 'unknown')}")
            print(f"   Size: {model['size_mb']:.1f} MB")
            print(f"   Modified: {model['modified']}")
            if args.details and 'files' in model:
                print(f"   Files: {model['file_count']} ({', '.join(model['files'][:5])}{'...' if len(model['files']) > 5 else ''})")
            print(f"   Path: {model['path']}")
            print()
    
    elif args.action == 'verify':
        result = await updater.verify_models()
        
        if result['invalid_models'] > 0:
            print(f"\n   {result['invalid_models']} invalid models found")
            return 1
        else:
            print(f"\n All {result['valid_models']} models are valid")
    
    elif args.action == 'cleanup':
        result = await updater.cleanup_models(args.keep_versions)
        
        print(f"\n Cleanup completed successfully")
        print(f"=Ñ  Removed {result['versions_removed']} old versions")
        print(f"=¾ Freed {result['space_freed_mb']:.1f} MB")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(asyncio.run(main()))