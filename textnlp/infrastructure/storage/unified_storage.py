"""
Unified Storage Interface for TextNLP
Provides a unified interface for model storage across cloud providers
"""

import os
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass
import logging
from pathlib import Path

from .model_shard_manager import ModelShardManager, AdaptiveShardManager, ModelShardConfig
from .chunked_transfer import ChunkedTransferManager, TransferConfig
from .model_version_manager import ModelVersionManager, ModelRegistry
from .delta_update_manager import DeltaUpdateManager

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Unified storage configuration"""
    provider: str  # "aws", "gcp", "azure", "local"
    bucket_name: Optional[str] = None
    region: Optional[str] = None
    credentials: Optional[Dict[str, str]] = None
    
    # Feature flags
    enable_sharding: bool = True
    enable_chunked_transfer: bool = True
    enable_versioning: bool = True
    enable_delta_updates: bool = True
    
    # Performance settings
    max_shard_size_gb: float = 2.0
    chunk_size_mb: int = 10
    parallel_transfers: int = 4
    compression: bool = True
    

class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    async def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a file to storage"""
        pass
    
    @abstractmethod
    async def download_file(self, remote_path: str, local_path: str) -> None:
        """Download a file from storage"""
        pass
    
    @abstractmethod
    async def delete_file(self, remote_path: str) -> None:
        """Delete a file from storage"""
        pass
    
    @abstractmethod
    async def list_files(self, prefix: str) -> List[str]:
        """List files with given prefix"""
        pass
    
    @abstractmethod
    async def file_exists(self, remote_path: str) -> bool:
        """Check if file exists"""
        pass
    
    @abstractmethod
    async def get_file_info(self, remote_path: str) -> Dict[str, Any]:
        """Get file metadata"""
        pass
    
    @abstractmethod
    async def stream_download(self, remote_path: str, chunk_size: int) -> AsyncIterator[bytes]:
        """Stream download file in chunks"""
        pass


class S3StorageBackend(StorageBackend):
    """AWS S3 storage backend implementation"""
    
    def __init__(self, bucket_name: str, region: str, credentials: Optional[Dict[str, str]] = None):
        import boto3
        
        self.bucket_name = bucket_name
        self.region = region
        
        if credentials:
            self.s3_client = boto3.client(
                's3',
                region_name=region,
                aws_access_key_id=credentials.get('access_key_id'),
                aws_secret_access_key=credentials.get('secret_access_key')
            )
        else:
            self.s3_client = boto3.client('s3', region_name=region)
    
    async def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload file to S3"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self.s3_client.upload_file,
            local_path,
            self.bucket_name,
            remote_path
        )
    
    async def download_file(self, remote_path: str, local_path: str) -> None:
        """Download file from S3"""
        loop = asyncio.get_event_loop()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        await loop.run_in_executor(
            None,
            self.s3_client.download_file,
            self.bucket_name,
            remote_path,
            local_path
        )
    
    async def delete_file(self, remote_path: str) -> None:
        """Delete file from S3"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self.s3_client.delete_object,
            Bucket=self.bucket_name,
            Key=remote_path
        )
    
    async def list_files(self, prefix: str) -> List[str]:
        """List files in S3 with prefix"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self.s3_client.list_objects_v2,
            Bucket=self.bucket_name,
            Prefix=prefix
        )
        
        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents']]
        return []
    
    async def file_exists(self, remote_path: str) -> bool:
        """Check if file exists in S3"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.s3_client.head_object,
                Bucket=self.bucket_name,
                Key=remote_path
            )
            return True
        except:
            return False
    
    async def get_file_info(self, remote_path: str) -> Dict[str, Any]:
        """Get S3 object metadata"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self.s3_client.head_object,
            Bucket=self.bucket_name,
            Key=remote_path
        )
        
        return {
            'size': response['ContentLength'],
            'last_modified': response['LastModified'],
            'etag': response['ETag']
        }
    
    async def stream_download(self, remote_path: str, chunk_size: int) -> AsyncIterator[bytes]:
        """Stream download from S3"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self.s3_client.get_object,
            Bucket=self.bucket_name,
            Key=remote_path
        )
        
        body = response['Body']
        while True:
            chunk = await loop.run_in_executor(None, body.read, chunk_size)
            if not chunk:
                break
            yield chunk


class GCSStorageBackend(StorageBackend):
    """Google Cloud Storage backend implementation"""
    
    def __init__(self, bucket_name: str, project_id: str, credentials: Optional[Dict[str, str]] = None):
        from google.cloud import storage
        
        self.bucket_name = bucket_name
        self.project_id = project_id
        
        if credentials and 'service_account_key' in credentials:
            self.client = storage.Client.from_service_account_json(
                credentials['service_account_key'],
                project=project_id
            )
        else:
            self.client = storage.Client(project=project_id)
        
        self.bucket = self.client.bucket(bucket_name)
    
    async def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload file to GCS"""
        blob = self.bucket.blob(remote_path)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, blob.upload_from_filename, local_path)
    
    async def download_file(self, remote_path: str, local_path: str) -> None:
        """Download file from GCS"""
        blob = self.bucket.blob(remote_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, blob.download_to_filename, local_path)
    
    async def delete_file(self, remote_path: str) -> None:
        """Delete file from GCS"""
        blob = self.bucket.blob(remote_path)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, blob.delete)
    
    async def list_files(self, prefix: str) -> List[str]:
        """List files in GCS with prefix"""
        loop = asyncio.get_event_loop()
        blobs = await loop.run_in_executor(
            None,
            list,
            self.bucket.list_blobs(prefix=prefix)
        )
        return [blob.name for blob in blobs]
    
    async def file_exists(self, remote_path: str) -> bool:
        """Check if file exists in GCS"""
        blob = self.bucket.blob(remote_path)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, blob.exists)
    
    async def get_file_info(self, remote_path: str) -> Dict[str, Any]:
        """Get GCS object metadata"""
        blob = self.bucket.blob(remote_path)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, blob.reload)
        
        return {
            'size': blob.size,
            'last_modified': blob.updated,
            'etag': blob.etag
        }
    
    async def stream_download(self, remote_path: str, chunk_size: int) -> AsyncIterator[bytes]:
        """Stream download from GCS"""
        blob = self.bucket.blob(remote_path)
        loop = asyncio.get_event_loop()
        
        with await loop.run_in_executor(None, blob.open, 'rb') as f:
            while True:
                chunk = await loop.run_in_executor(None, f.read, chunk_size)
                if not chunk:
                    break
                yield chunk


class AzureStorageBackend(StorageBackend):
    """Azure Blob Storage backend implementation"""
    
    def __init__(self, account_name: str, container_name: str, credentials: Optional[Dict[str, str]] = None):
        from azure.storage.blob.aio import BlobServiceClient
        
        self.container_name = container_name
        
        if credentials and 'account_key' in credentials:
            connection_string = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={account_name};"
                f"AccountKey={credentials['account_key']};"
                f"EndpointSuffix=core.windows.net"
            )
            self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        else:
            self.blob_service = BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net"
            )
    
    async def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload file to Azure Blob Storage"""
        async with self.blob_service:
            container_client = self.blob_service.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(remote_path)
            
            with open(local_path, 'rb') as data:
                await blob_client.upload_blob(data, overwrite=True)
    
    async def download_file(self, remote_path: str, local_path: str) -> None:
        """Download file from Azure Blob Storage"""
        async with self.blob_service:
            container_client = self.blob_service.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(remote_path)
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                stream = await blob_client.download_blob()
                data = await stream.readall()
                f.write(data)
    
    async def delete_file(self, remote_path: str) -> None:
        """Delete file from Azure Blob Storage"""
        async with self.blob_service:
            container_client = self.blob_service.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(remote_path)
            await blob_client.delete_blob()
    
    async def list_files(self, prefix: str) -> List[str]:
        """List files in Azure with prefix"""
        async with self.blob_service:
            container_client = self.blob_service.get_container_client(self.container_name)
            blobs = []
            async for blob in container_client.list_blobs(name_starts_with=prefix):
                blobs.append(blob.name)
            return blobs
    
    async def file_exists(self, remote_path: str) -> bool:
        """Check if file exists in Azure"""
        async with self.blob_service:
            container_client = self.blob_service.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(remote_path)
            return await blob_client.exists()
    
    async def get_file_info(self, remote_path: str) -> Dict[str, Any]:
        """Get Azure blob metadata"""
        async with self.blob_service:
            container_client = self.blob_service.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(remote_path)
            properties = await blob_client.get_blob_properties()
            
            return {
                'size': properties.size,
                'last_modified': properties.last_modified,
                'etag': properties.etag
            }
    
    async def stream_download(self, remote_path: str, chunk_size: int) -> AsyncIterator[bytes]:
        """Stream download from Azure"""
        async with self.blob_service:
            container_client = self.blob_service.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(remote_path)
            
            stream = await blob_client.download_blob()
            async for chunk in stream.chunks():
                yield chunk


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend for development"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def upload_file(self, local_path: str, remote_path: str) -> None:
        """Copy file to local storage"""
        dest_path = self.base_path / remote_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, shutil.copy2, local_path, dest_path)
    
    async def download_file(self, remote_path: str, local_path: str) -> None:
        """Copy file from local storage"""
        src_path = self.base_path / remote_path
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, shutil.copy2, src_path, local_path)
    
    async def delete_file(self, remote_path: str) -> None:
        """Delete file from local storage"""
        file_path = self.base_path / remote_path
        if file_path.exists():
            file_path.unlink()
    
    async def list_files(self, prefix: str) -> List[str]:
        """List files with prefix"""
        prefix_path = self.base_path / prefix
        if not prefix_path.exists():
            return []
        
        files = []
        for path in prefix_path.rglob('*'):
            if path.is_file():
                relative_path = path.relative_to(self.base_path)
                files.append(str(relative_path))
        
        return files
    
    async def file_exists(self, remote_path: str) -> bool:
        """Check if file exists"""
        return (self.base_path / remote_path).exists()
    
    async def get_file_info(self, remote_path: str) -> Dict[str, Any]:
        """Get file metadata"""
        file_path = self.base_path / remote_path
        stat = file_path.stat()
        
        return {
            'size': stat.st_size,
            'last_modified': datetime.fromtimestamp(stat.st_mtime),
            'etag': str(stat.st_mtime)
        }
    
    async def stream_download(self, remote_path: str, chunk_size: int) -> AsyncIterator[bytes]:
        """Stream read file"""
        file_path = self.base_path / remote_path
        
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(chunk_size):
                yield chunk


class UnifiedModelStorage:
    """Unified interface for model storage with all features integrated"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        
        # Initialize storage backend
        self.backend = self._create_backend(config)
        
        # Initialize managers based on config
        if config.enable_sharding:
            shard_config = ModelShardConfig(
                max_shard_size_gb=config.max_shard_size_gb,
                compression=config.compression,
                chunk_size_mb=config.chunk_size_mb
            )
            self.shard_manager = AdaptiveShardManager(self.backend, shard_config)
        
        if config.enable_chunked_transfer:
            transfer_config = TransferConfig(
                chunk_size_mb=config.chunk_size_mb,
                parallel_chunks=config.parallel_transfers,
                compression=config.compression
            )
            self.transfer_manager = ChunkedTransferManager(transfer_config)
        
        if config.enable_versioning:
            self.version_manager = ModelVersionManager(self.backend)
            self.registry = ModelRegistry(self.version_manager)
        
        if config.enable_delta_updates:
            self.delta_manager = DeltaUpdateManager(self.backend)
    
    def _create_backend(self, config: StorageConfig) -> StorageBackend:
        """Create appropriate storage backend"""
        if config.provider == "aws":
            return S3StorageBackend(
                bucket_name=config.bucket_name,
                region=config.region,
                credentials=config.credentials
            )
        elif config.provider == "gcp":
            return GCSStorageBackend(
                bucket_name=config.bucket_name,
                project_id=config.credentials.get('project_id'),
                credentials=config.credentials
            )
        elif config.provider == "azure":
            return AzureStorageBackend(
                account_name=config.credentials.get('account_name'),
                container_name=config.bucket_name,
                credentials=config.credentials
            )
        elif config.provider == "local":
            return LocalStorageBackend(config.bucket_name or "./storage")
        else:
            raise ValueError(f"Unknown storage provider: {config.provider}")
    
    async def upload_model(self, model_path: str, model_name: str, 
                         version: str, **kwargs) -> Dict[str, Any]:
        """Upload a model with all features"""
        result = {
            "model_name": model_name,
            "version": version,
            "features_used": []
        }
        
        # Check if sharding is needed
        model_size = os.path.getsize(model_path)
        use_sharding = (
            self.config.enable_sharding and 
            model_size > self.config.max_shard_size_gb * 1024 * 1024 * 1024
        )
        
        if use_sharding:
            # Use sharding for large models
            logger.info(f"Using sharding for large model ({model_size / 1e9:.2f}GB)")
            manifest = await self.shard_manager.shard_model_adaptive(
                model_path=model_path,
                output_dir=f"/tmp/shards/{model_name}/{version}",
                model_id=f"{model_name}-{version}",
                strategy=kwargs.get('sharding_strategy', 'layer_based')
            )
            result['sharding_manifest'] = manifest
            result['features_used'].append('sharding')
            
        elif self.config.enable_chunked_transfer:
            # Use chunked transfer for medium models
            logger.info(f"Using chunked transfer for model ({model_size / 1e6:.2f}MB)")
            remote_path = f"models/{model_name}/{version}/model.bin"
            transfer_result = await self.transfer_manager.upload_file_chunked(
                local_path=model_path,
                remote_path=remote_path,
                storage_backend=self.backend
            )
            result['transfer_result'] = transfer_result
            result['features_used'].append('chunked_transfer')
            
        else:
            # Direct upload for small models
            remote_path = f"models/{model_name}/{version}/model.bin"
            await self.backend.upload_file(model_path, remote_path)
            result['remote_path'] = remote_path
        
        # Register in version manager if enabled
        if self.config.enable_versioning:
            model_version = await self.registry.publish_model(
                model_path=model_path,
                name=model_name,
                version=version,
                architecture=kwargs.get('architecture', 'unknown'),
                created_by=kwargs.get('created_by', 'system'),
                tags=kwargs.get('tags', []),
                parent_version=kwargs.get('parent_version')
            )
            result['model_version'] = model_version
            result['features_used'].append('versioning')
        
        # Create delta patch if parent version exists
        if self.config.enable_delta_updates and kwargs.get('parent_version'):
            parent_path = kwargs.get('parent_model_path')
            if parent_path:
                patch = await self.delta_manager.create_delta_patch(
                    old_model_path=parent_path,
                    new_model_path=model_path,
                    source_version=kwargs.get('parent_version'),
                    target_version=version
                )
                result['delta_patch'] = patch
                result['features_used'].append('delta_updates')
        
        return result
    
    async def download_model(self, model_name: str, version: Optional[str] = None,
                           **kwargs) -> str:
        """Download a model with appropriate method"""
        
        # Use version manager to get model info
        if self.config.enable_versioning:
            local_path, model_info = await self.registry.get_model_for_inference(
                name=model_name,
                version=version
            )
            
            # Check if model is sharded
            manifest_path = f"models/{model_name}/{model_info.metadata.version}_manifest.json"
            if await self.backend.file_exists(manifest_path):
                # Download and load sharded model
                logger.info("Downloading sharded model")
                state_dict = await self.shard_manager.load_sharded_model(
                    manifest_path=manifest_path,
                    device=kwargs.get('device')
                )
                # Save reconstructed model
                import torch
                torch.save(state_dict, local_path)
            
            return local_path
            
        else:
            # Direct download
            remote_path = f"models/{model_name}/{version or 'latest'}/model.bin"
            local_path = f"/tmp/models/{model_name}_{version or 'latest'}.bin"
            
            if self.config.enable_chunked_transfer:
                await self.transfer_manager.download_file_chunked(
                    remote_path=remote_path,
                    local_path=local_path,
                    storage_backend=self.backend
                )
            else:
                await self.backend.download_file(remote_path, local_path)
            
            return local_path
    
    async def list_models(self, **filters) -> List[Dict[str, Any]]:
        """List available models"""
        if self.config.enable_versioning:
            models = await self.version_manager.list_models(**filters)
            return [
                {
                    "name": m.metadata.name,
                    "version": m.metadata.version,
                    "status": m.status.value,
                    "size_gb": m.metadata.size_bytes / 1e9,
                    "created_at": m.metadata.created_at.isoformat()
                }
                for m in models
            ]
        else:
            # List from storage directly
            files = await self.backend.list_files("models/")
            models = {}
            
            for file in files:
                parts = file.split('/')
                if len(parts) >= 3:
                    name = parts[1]
                    version = parts[2]
                    if name not in models:
                        models[name] = []
                    models[name].append(version)
            
            return [
                {"name": name, "versions": versions}
                for name, versions in models.items()
            ]


# Example usage
if __name__ == "__main__":
    import shutil
    from datetime import datetime
    import aiofiles
    
    async def example():
        # Configure unified storage
        config = StorageConfig(
            provider="local",  # Use local for example
            bucket_name="./test_storage",
            enable_sharding=True,
            enable_chunked_transfer=True,
            enable_versioning=True,
            enable_delta_updates=True
        )
        
        # Create unified storage
        storage = UnifiedModelStorage(config)
        
        # Example: Upload a model
        result = await storage.upload_model(
            model_path="/path/to/model.safetensors",
            model_name="gpt2-custom",
            version="1.0.0",
            architecture="transformer",
            created_by="john.doe",
            tags=["nlp", "text-generation"]
        )
        
        print(f"Upload result: {result}")
        
        # Example: Download model
        model_path = await storage.download_model(
            model_name="gpt2-custom",
            version="1.0.0"
        )
        
        print(f"Downloaded model to: {model_path}")
        
        # Example: List models
        models = await storage.list_models()
        print(f"Available models: {models}")
    
    # Run example
    # asyncio.run(example())