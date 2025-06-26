"""
Unified Storage Module

Provides unified storage abstraction for all Inferloop services.
Supports multiple storage backends (S3, GCS, Azure Blob, MinIO) with a consistent interface.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional, BinaryIO, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import json

import boto3
import aiofiles
from azure.storage.blob.aio import BlobServiceClient as AzureBlobServiceClient
from google.cloud import storage as gcs
import httpx


@dataclass
class StorageObject:
    """Represents a storage object"""
    key: str
    size: int
    last_modified: datetime
    etag: Optional[str] = None
    content_type: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class StorageConfig:
    """Storage configuration"""
    provider: str  # s3, gcs, azure, minio
    bucket_name: str
    region: Optional[str] = None
    endpoint_url: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    credentials_path: Optional[str] = None


class StorageBackend(ABC):
    """Abstract base class for storage backends"""
    
    @abstractmethod
    async def get_object(self, key: str) -> bytes:
        """Get object by key"""
        pass
    
    @abstractmethod
    async def put_object(self, key: str, data: Union[bytes, BinaryIO], 
                        content_type: Optional[str] = None,
                        metadata: Optional[Dict[str, str]] = None) -> str:
        """Put object with key"""
        pass
    
    @abstractmethod
    async def delete_object(self, key: str) -> bool:
        """Delete object by key"""
        pass
    
    @abstractmethod
    async def list_objects(self, prefix: str = "", limit: int = 1000) -> List[StorageObject]:
        """List objects with optional prefix"""
        pass
    
    @abstractmethod
    async def object_exists(self, key: str) -> bool:
        """Check if object exists"""
        pass
    
    @abstractmethod
    async def get_presigned_url(self, key: str, expiration: int = 3600, 
                               method: str = "GET") -> str:
        """Get presigned URL for object"""
        pass
    
    @abstractmethod
    async def copy_object(self, source_key: str, dest_key: str) -> bool:
        """Copy object to new key"""
        pass


class S3Backend(StorageBackend):
    """S3-compatible storage backend"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.session = boto3.Session(
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            region_name=config.region
        )
        
        # Create S3 client
        self.s3_client = self.session.client(
            's3',
            endpoint_url=config.endpoint_url
        )
    
    async def get_object(self, key: str) -> bytes:
        """Get object from S3"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.s3_client.get_object(
                    Bucket=self.config.bucket_name,
                    Key=key
                )
            )
            return response['Body'].read()
        except Exception as e:
            raise StorageError(f"Failed to get object {key}: {str(e)}")
    
    async def put_object(self, key: str, data: Union[bytes, BinaryIO],
                        content_type: Optional[str] = None,
                        metadata: Optional[Dict[str, str]] = None) -> str:
        """Put object to S3"""
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            if metadata:
                extra_args['Metadata'] = metadata
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.s3_client.put_object(
                    Bucket=self.config.bucket_name,
                    Key=key,
                    Body=data,
                    **extra_args
                )
            )
            return response.get('ETag', '').strip('"')
        except Exception as e:
            raise StorageError(f"Failed to put object {key}: {str(e)}")
    
    async def delete_object(self, key: str) -> bool:
        """Delete object from S3"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.delete_object(
                    Bucket=self.config.bucket_name,
                    Key=key
                )
            )
            return True
        except Exception as e:
            raise StorageError(f"Failed to delete object {key}: {str(e)}")
    
    async def list_objects(self, prefix: str = "", limit: int = 1000) -> List[StorageObject]:
        """List objects in S3"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.s3_client.list_objects_v2(
                    Bucket=self.config.bucket_name,
                    Prefix=prefix,
                    MaxKeys=limit
                )
            )
            
            objects = []
            for obj in response.get('Contents', []):
                objects.append(StorageObject(
                    key=obj['Key'],
                    size=obj['Size'],
                    last_modified=obj['LastModified'],
                    etag=obj.get('ETag', '').strip('"')
                ))
            
            return objects
        except Exception as e:
            raise StorageError(f"Failed to list objects: {str(e)}")
    
    async def object_exists(self, key: str) -> bool:
        """Check if object exists in S3"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.head_object(
                    Bucket=self.config.bucket_name,
                    Key=key
                )
            )
            return True
        except:
            return False
    
    async def get_presigned_url(self, key: str, expiration: int = 3600,
                               method: str = "GET") -> str:
        """Get presigned URL for S3 object"""
        try:
            operation = 'get_object' if method == 'GET' else 'put_object'
            loop = asyncio.get_event_loop()
            url = await loop.run_in_executor(
                None,
                lambda: self.s3_client.generate_presigned_url(
                    operation,
                    Params={'Bucket': self.config.bucket_name, 'Key': key},
                    ExpiresIn=expiration
                )
            )
            return url
        except Exception as e:
            raise StorageError(f"Failed to generate presigned URL: {str(e)}")
    
    async def copy_object(self, source_key: str, dest_key: str) -> bool:
        """Copy object in S3"""
        try:
            copy_source = {
                'Bucket': self.config.bucket_name,
                'Key': source_key
            }
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.s3_client.copy_object(
                    CopySource=copy_source,
                    Bucket=self.config.bucket_name,
                    Key=dest_key
                )
            )
            return True
        except Exception as e:
            raise StorageError(f"Failed to copy object: {str(e)}")


class GCSBackend(StorageBackend):
    """Google Cloud Storage backend"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        if config.credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config.credentials_path
        
        self.client = gcs.Client()
        self.bucket = self.client.bucket(config.bucket_name)
    
    async def get_object(self, key: str) -> bytes:
        """Get object from GCS"""
        try:
            blob = self.bucket.blob(key)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, blob.download_as_bytes)
        except Exception as e:
            raise StorageError(f"Failed to get object {key}: {str(e)}")
    
    async def put_object(self, key: str, data: Union[bytes, BinaryIO],
                        content_type: Optional[str] = None,
                        metadata: Optional[Dict[str, str]] = None) -> str:
        """Put object to GCS"""
        try:
            blob = self.bucket.blob(key)
            if content_type:
                blob.content_type = content_type
            if metadata:
                blob.metadata = metadata
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, blob.upload_from_string, data)
            return blob.etag or ""
        except Exception as e:
            raise StorageError(f"Failed to put object {key}: {str(e)}")
    
    async def delete_object(self, key: str) -> bool:
        """Delete object from GCS"""
        try:
            blob = self.bucket.blob(key)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, blob.delete)
            return True
        except Exception as e:
            raise StorageError(f"Failed to delete object {key}: {str(e)}")
    
    async def list_objects(self, prefix: str = "", limit: int = 1000) -> List[StorageObject]:
        """List objects in GCS"""
        try:
            loop = asyncio.get_event_loop()
            blobs = await loop.run_in_executor(
                None,
                lambda: list(self.client.list_blobs(
                    self.config.bucket_name,
                    prefix=prefix,
                    max_results=limit
                ))
            )
            
            objects = []
            for blob in blobs:
                objects.append(StorageObject(
                    key=blob.name,
                    size=blob.size or 0,
                    last_modified=blob.time_created or datetime.utcnow(),
                    etag=blob.etag,
                    content_type=blob.content_type
                ))
            
            return objects
        except Exception as e:
            raise StorageError(f"Failed to list objects: {str(e)}")
    
    async def object_exists(self, key: str) -> bool:
        """Check if object exists in GCS"""
        try:
            blob = self.bucket.blob(key)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, blob.exists)
        except:
            return False
    
    async def get_presigned_url(self, key: str, expiration: int = 3600,
                               method: str = "GET") -> str:
        """Get presigned URL for GCS object"""
        try:
            blob = self.bucket.blob(key)
            loop = asyncio.get_event_loop()
            url = await loop.run_in_executor(
                None,
                lambda: blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(seconds=expiration),
                    method=method
                )
            )
            return url
        except Exception as e:
            raise StorageError(f"Failed to generate presigned URL: {str(e)}")
    
    async def copy_object(self, source_key: str, dest_key: str) -> bool:
        """Copy object in GCS"""
        try:
            source_blob = self.bucket.blob(source_key)
            dest_blob = self.bucket.blob(dest_key)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.bucket.copy_blob(source_blob, self.bucket, dest_key)
            )
            return True
        except Exception as e:
            raise StorageError(f"Failed to copy object: {str(e)}")


class AzureBackend(StorageBackend):
    """Azure Blob Storage backend"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not connection_string and config.access_key and config.secret_key:
            connection_string = f"DefaultEndpointsProtocol=https;AccountName={config.access_key};AccountKey={config.secret_key};EndpointSuffix=core.windows.net"
        
        self.blob_service_client = AzureBlobServiceClient.from_connection_string(connection_string)
        self.container_name = config.bucket_name
    
    async def get_object(self, key: str) -> bytes:
        """Get object from Azure Blob Storage"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=key
            )
            
            download_stream = await blob_client.download_blob()
            return await download_stream.readall()
        except Exception as e:
            raise StorageError(f"Failed to get object {key}: {str(e)}")
    
    async def put_object(self, key: str, data: Union[bytes, BinaryIO],
                        content_type: Optional[str] = None,
                        metadata: Optional[Dict[str, str]] = None) -> str:
        """Put object to Azure Blob Storage"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=key
            )
            
            kwargs = {}
            if content_type:
                kwargs['content_type'] = content_type
            if metadata:
                kwargs['metadata'] = metadata
            
            result = await blob_client.upload_blob(data, overwrite=True, **kwargs)
            return result.get('etag', '').strip('"')
        except Exception as e:
            raise StorageError(f"Failed to put object {key}: {str(e)}")
    
    async def delete_object(self, key: str) -> bool:
        """Delete object from Azure Blob Storage"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=key
            )
            await blob_client.delete_blob()
            return True
        except Exception as e:
            raise StorageError(f"Failed to delete object {key}: {str(e)}")
    
    async def list_objects(self, prefix: str = "", limit: int = 1000) -> List[StorageObject]:
        """List objects in Azure Blob Storage"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            
            objects = []
            async for blob in container_client.list_blobs(name_starts_with=prefix):
                if len(objects) >= limit:
                    break
                
                objects.append(StorageObject(
                    key=blob.name,
                    size=blob.size,
                    last_modified=blob.last_modified,
                    etag=blob.etag.strip('"') if blob.etag else None,
                    content_type=blob.content_settings.content_type if blob.content_settings else None
                ))
            
            return objects
        except Exception as e:
            raise StorageError(f"Failed to list objects: {str(e)}")
    
    async def object_exists(self, key: str) -> bool:
        """Check if object exists in Azure Blob Storage"""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=key
            )
            properties = await blob_client.get_blob_properties()
            return True
        except:
            return False
    
    async def get_presigned_url(self, key: str, expiration: int = 3600,
                               method: str = "GET") -> str:
        """Get presigned URL for Azure blob"""
        # Azure uses SAS tokens for presigned URLs
        # This is a simplified implementation
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=key
            )
            # Generate SAS token (simplified)
            return blob_client.url
        except Exception as e:
            raise StorageError(f"Failed to generate presigned URL: {str(e)}")
    
    async def copy_object(self, source_key: str, dest_key: str) -> bool:
        """Copy object in Azure Blob Storage"""
        try:
            source_blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=source_key
            )
            dest_blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=dest_key
            )
            
            copy_props = await dest_blob_client.start_copy_from_url(source_blob_client.url)
            return True
        except Exception as e:
            raise StorageError(f"Failed to copy object: {str(e)}")


class StorageError(Exception):
    """Storage operation error"""
    pass


class StorageClient:
    """Unified storage client"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.config = self._load_config()
        self.backend = self._create_backend()
        self._buckets_cache = {}
    
    def _load_config(self) -> StorageConfig:
        """Load storage configuration"""
        provider = os.getenv('STORAGE_PROVIDER', 's3').lower()
        bucket_name = os.getenv('STORAGE_BUCKET_NAME', f'inferloop-{self.service_name}')
        
        return StorageConfig(
            provider=provider,
            bucket_name=bucket_name,
            region=os.getenv('STORAGE_REGION', 'us-east-1'),
            endpoint_url=os.getenv('STORAGE_ENDPOINT_URL'),
            access_key=os.getenv('STORAGE_ACCESS_KEY'),
            secret_key=os.getenv('STORAGE_SECRET_KEY'),
            credentials_path=os.getenv('STORAGE_CREDENTIALS_PATH')
        )
    
    def _create_backend(self) -> StorageBackend:
        """Create storage backend based on configuration"""
        if self.config.provider == 's3' or self.config.provider == 'minio':
            return S3Backend(self.config)
        elif self.config.provider == 'gcs':
            return GCSBackend(self.config)
        elif self.config.provider == 'azure':
            return AzureBackend(self.config)
        else:
            raise ValueError(f"Unsupported storage provider: {self.config.provider}")
    
    async def get(self, key: str) -> bytes:
        """Get object by key"""
        full_key = f"{self.service_name}/{key}"
        return await self.backend.get_object(full_key)
    
    async def put(self, key: str, data: Union[bytes, str, BinaryIO],
                  content_type: Optional[str] = None,
                  metadata: Optional[Dict[str, str]] = None) -> str:
        """Put object with key"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        full_key = f"{self.service_name}/{key}"
        return await self.backend.put_object(full_key, data, content_type, metadata)
    
    async def delete(self, key: str) -> bool:
        """Delete object by key"""
        full_key = f"{self.service_name}/{key}"
        return await self.backend.delete_object(full_key)
    
    async def list(self, prefix: str = "", limit: int = 1000) -> List[StorageObject]:
        """List objects with prefix"""
        full_prefix = f"{self.service_name}/{prefix}"
        objects = await self.backend.list_objects(full_prefix, limit)
        
        # Remove service prefix from keys
        for obj in objects:
            if obj.key.startswith(f"{self.service_name}/"):
                obj.key = obj.key[len(f"{self.service_name}/"):]
        
        return objects
    
    async def exists(self, key: str) -> bool:
        """Check if object exists"""
        full_key = f"{self.service_name}/{key}"
        return await self.backend.object_exists(full_key)
    
    async def get_url(self, key: str, expiration: int = 3600, method: str = "GET") -> str:
        """Get presigned URL for object"""
        full_key = f"{self.service_name}/{key}"
        return await self.backend.get_presigned_url(full_key, expiration, method)
    
    async def copy(self, source_key: str, dest_key: str) -> bool:
        """Copy object to new key"""
        full_source_key = f"{self.service_name}/{source_key}"
        full_dest_key = f"{self.service_name}/{dest_key}"
        return await self.backend.copy_object(full_source_key, full_dest_key)
    
    async def ensure_bucket(self, bucket_path: str = ""):
        """Ensure bucket/container exists (for certain operations)"""
        # This is mainly for organizing data within the bucket
        # Most cloud providers auto-create buckets or require manual creation
        if bucket_path:
            self._buckets_cache[bucket_path] = True
    
    async def get_json(self, key: str) -> Dict[str, Any]:
        """Get and parse JSON object"""
        data = await self.get(key)
        return json.loads(data.decode('utf-8'))
    
    async def put_json(self, key: str, data: Dict[str, Any]) -> str:
        """Put JSON object"""
        json_data = json.dumps(data, indent=2)
        return await self.put(key, json_data, content_type="application/json")
    
    async def get_text(self, key: str) -> str:
        """Get text object"""
        data = await self.get(key)
        return data.decode('utf-8')
    
    async def put_text(self, key: str, text: str) -> str:
        """Put text object"""
        return await self.put(key, text, content_type="text/plain")
    
    async def check_connection(self) -> bool:
        """Check if storage connection is working"""
        try:
            # Try to list objects with a limit of 1
            await self.backend.list_objects("", limit=1)
            return True
        except Exception:
            return False
    
    async def close(self):
        """Close storage client connections"""
        # Close any open connections
        if hasattr(self.backend, 'close'):
            await self.backend.close()


# Dependency for FastAPI
async def get_storage_client() -> StorageClient:
    """Dependency to get storage client"""
    # This would typically be cached or managed by a dependency injection container
    return StorageClient("default")


# Global storage clients cache
_storage_clients: Dict[str, StorageClient] = {}


def get_storage_client_sync(service_name: str) -> StorageClient:
    """Get storage client synchronously (cached)"""
    if service_name not in _storage_clients:
        _storage_clients[service_name] = StorageClient(service_name)
    return _storage_clients[service_name]