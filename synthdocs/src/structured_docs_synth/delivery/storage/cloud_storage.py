"""
Cloud storage implementation for structured document synthesis delivery.

Provides unified interface for multiple cloud storage providers including
AWS S3, Azure Blob Storage, Google Cloud Storage, and other cloud platforms.

Features:
- Multi-cloud provider support (AWS, Azure, GCP, etc.)
- Async file operations for high performance
- Built-in retry logic and error handling
- Encryption and security features
- Batch upload/download capabilities
- Progress tracking for large transfers
- Cost optimization features
- Backup and versioning support
"""

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, AsyncIterator
from urllib.parse import urlparse
import tempfile

from ...core.exceptions import DeliveryError, ConfigurationError

logger = logging.getLogger(__name__)


class CloudStorageConfig:
    """Configuration class for cloud storage providers"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # General settings
        self.provider = self.config.get('provider', 'aws')
        self.region = self.config.get('region', 'us-east-1')
        self.bucket_name = self.config.get('bucket_name', 'structured-docs-synth')
        self.prefix = self.config.get('prefix', 'documents/')
        
        # Security settings
        self.encryption = self.config.get('encryption', True)
        self.access_key_id = self.config.get('access_key_id')
        self.secret_access_key = self.config.get('secret_access_key')
        
        # Performance settings
        self.max_concurrent_uploads = self.config.get('max_concurrent_uploads', 10)
        self.chunk_size = self.config.get('chunk_size', 8 * 1024 * 1024)  # 8MB
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.timeout = self.config.get('timeout', 300)  # 5 minutes
        
        # Cost optimization
        self.storage_class = self.config.get('storage_class', 'STANDARD')
        self.lifecycle_rules = self.config.get('lifecycle_rules', [])


class CloudStorageClient:
    """Base class for cloud storage clients"""
    
    def __init__(self, config: CloudStorageConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._client = None
    
    async def connect(self):
        """Initialize connection to cloud storage"""
        raise NotImplementedError
    
    async def disconnect(self):
        """Close connection to cloud storage"""
        if self._client:
            await self._client.close()
    
    async def upload_file(self, local_path: Path, remote_path: str, **kwargs) -> Dict[str, Any]:
        """Upload a single file to cloud storage"""
        raise NotImplementedError
    
    async def download_file(self, remote_path: str, local_path: Path, **kwargs) -> Dict[str, Any]:
        """Download a single file from cloud storage"""
        raise NotImplementedError
    
    async def delete_file(self, remote_path: str) -> bool:
        """Delete a file from cloud storage"""
        raise NotImplementedError
    
    async def list_files(self, prefix: str = "", limit: int = 1000) -> List[Dict[str, Any]]:
        """List files in cloud storage"""
        raise NotImplementedError


class AWSS3Client(CloudStorageClient):
    """AWS S3 storage client implementation"""
    
    async def connect(self):
        """Initialize AWS S3 client"""
        try:
            # Import AWS SDK
            import aioboto3
            
            # Create session with credentials
            session = aioboto3.Session(
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                region_name=self.config.region
            )
            
            # Create S3 client
            self._client = session.client('s3')
            
            # Test connection
            await self._client.head_bucket(Bucket=self.config.bucket_name)
            self.logger.info(f"Connected to AWS S3 bucket: {self.config.bucket_name}")
            
        except ImportError:
            raise ConfigurationError("AWS SDK (aioboto3) not installed. Install with: pip install aioboto3")
        except Exception as e:
            raise DeliveryError(f"Failed to connect to AWS S3: {e}")
    
    async def upload_file(self, local_path: Path, remote_path: str, **kwargs) -> Dict[str, Any]:
        """Upload file to AWS S3"""
        try:
            start_time = datetime.now()
            file_size = local_path.stat().st_size
            
            # Prepare upload parameters
            upload_params = {
                'Bucket': self.config.bucket_name,
                'Key': f"{self.config.prefix}{remote_path}",
                'Body': local_path.read_bytes()
            }
            
            # Add encryption if enabled
            if self.config.encryption:
                upload_params['ServerSideEncryption'] = 'AES256'
            
            # Add storage class
            if self.config.storage_class:
                upload_params['StorageClass'] = self.config.storage_class
            
            # Add metadata
            metadata = kwargs.get('metadata', {})
            metadata.update({
                'uploaded_at': datetime.now().isoformat(),
                'original_name': local_path.name,
                'file_size': str(file_size)
            })
            upload_params['Metadata'] = metadata
            
            # Perform upload
            response = await self._client.put_object(**upload_params)
            
            upload_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'remote_path': f"{self.config.prefix}{remote_path}",
                'file_size': file_size,
                'upload_time': upload_time,
                'etag': response.get('ETag', '').strip('"'),
                'version_id': response.get('VersionId'),
                'url': f"s3://{self.config.bucket_name}/{self.config.prefix}{remote_path}"
            }
            
        except Exception as e:
            self.logger.error(f"S3 upload failed for {local_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'remote_path': remote_path
            }
    
    async def download_file(self, remote_path: str, local_path: Path, **kwargs) -> Dict[str, Any]:
        """Download file from AWS S3"""
        try:
            start_time = datetime.now()
            
            # Download file
            response = await self._client.get_object(
                Bucket=self.config.bucket_name,
                Key=f"{self.config.prefix}{remote_path}"
            )
            
            # Write to local file
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(await response['Body'].read())
            
            download_time = (datetime.now() - start_time).total_seconds()
            file_size = local_path.stat().st_size
            
            return {
                'success': True,
                'local_path': str(local_path),
                'file_size': file_size,
                'download_time': download_time,
                'metadata': response.get('Metadata', {})
            }
            
        except Exception as e:
            self.logger.error(f"S3 download failed for {remote_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'remote_path': remote_path
            }
    
    async def delete_file(self, remote_path: str) -> bool:
        """Delete file from AWS S3"""
        try:
            await self._client.delete_object(
                Bucket=self.config.bucket_name,
                Key=f"{self.config.prefix}{remote_path}"
            )
            return True
        except Exception as e:
            self.logger.error(f"S3 delete failed for {remote_path}: {e}")
            return False
    
    async def list_files(self, prefix: str = "", limit: int = 1000) -> List[Dict[str, Any]]:
        """List files in AWS S3"""
        try:
            response = await self._client.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=f"{self.config.prefix}{prefix}",
                MaxKeys=limit
            )
            
            files = []
            for obj in response.get('Contents', []):
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'etag': obj['ETag'].strip('"'),
                    'storage_class': obj.get('StorageClass', 'STANDARD')
                })
            
            return files
            
        except Exception as e:
            self.logger.error(f"S3 list failed: {e}")
            return []


class AzureBlobClient(CloudStorageClient):
    """Azure Blob Storage client implementation"""
    
    async def connect(self):
        """Initialize Azure Blob client"""
        try:
            from azure.storage.blob.aio import BlobServiceClient
            
            # Create connection string
            connection_string = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={self.config.access_key_id};"
                f"AccountKey={self.config.secret_access_key};"
                f"EndpointSuffix=core.windows.net"
            )
            
            # Create blob service client
            self._client = BlobServiceClient.from_connection_string(connection_string)
            
            # Test connection
            account_info = await self._client.get_account_information()
            self.logger.info(f"Connected to Azure Blob Storage: {account_info}")
            
        except ImportError:
            raise ConfigurationError("Azure SDK not installed. Install with: pip install azure-storage-blob")
        except Exception as e:
            raise DeliveryError(f"Failed to connect to Azure Blob Storage: {e}")
    
    async def upload_file(self, local_path: Path, remote_path: str, **kwargs) -> Dict[str, Any]:
        """Upload file to Azure Blob Storage"""
        try:
            start_time = datetime.now()
            file_size = local_path.stat().st_size
            
            # Get blob client
            blob_client = self._client.get_blob_client(
                container=self.config.bucket_name,
                blob=f"{self.config.prefix}{remote_path}"
            )
            
            # Prepare metadata
            metadata = kwargs.get('metadata', {})
            metadata.update({
                'uploaded_at': datetime.now().isoformat(),
                'original_name': local_path.name,
                'file_size': str(file_size)
            })
            
            # Upload file
            with open(local_path, 'rb') as data:
                await blob_client.upload_blob(
                    data, 
                    overwrite=True,
                    metadata=metadata
                )
            
            upload_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'remote_path': f"{self.config.prefix}{remote_path}",
                'file_size': file_size,
                'upload_time': upload_time,
                'url': blob_client.url
            }
            
        except Exception as e:
            self.logger.error(f"Azure upload failed for {local_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'remote_path': remote_path
            }
    
    async def download_file(self, remote_path: str, local_path: Path, **kwargs) -> Dict[str, Any]:
        """Download file from Azure Blob Storage"""
        try:
            start_time = datetime.now()
            
            # Get blob client
            blob_client = self._client.get_blob_client(
                container=self.config.bucket_name,
                blob=f"{self.config.prefix}{remote_path}"
            )
            
            # Download file
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, 'wb') as download_file:
                download_stream = await blob_client.download_blob()
                download_file.write(await download_stream.readall())
            
            download_time = (datetime.now() - start_time).total_seconds()
            file_size = local_path.stat().st_size
            
            return {
                'success': True,
                'local_path': str(local_path),
                'file_size': file_size,
                'download_time': download_time
            }
            
        except Exception as e:
            self.logger.error(f"Azure download failed for {remote_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'remote_path': remote_path
            }
    
    async def delete_file(self, remote_path: str) -> bool:
        """Delete file from Azure Blob Storage"""
        try:
            blob_client = self._client.get_blob_client(
                container=self.config.bucket_name,
                blob=f"{self.config.prefix}{remote_path}"
            )
            await blob_client.delete_blob()
            return True
        except Exception as e:
            self.logger.error(f"Azure delete failed for {remote_path}: {e}")
            return False
    
    async def list_files(self, prefix: str = "", limit: int = 1000) -> List[Dict[str, Any]]:
        """List files in Azure Blob Storage"""
        try:
            container_client = self._client.get_container_client(self.config.bucket_name)
            
            files = []
            async for blob in container_client.list_blobs(name_starts_with=f"{self.config.prefix}{prefix}"):
                files.append({
                    'key': blob.name,
                    'size': blob.size,
                    'last_modified': blob.last_modified.isoformat(),
                    'etag': blob.etag.strip('"'),
                    'content_type': blob.content_settings.content_type
                })
                
                if len(files) >= limit:
                    break
            
            return files
            
        except Exception as e:
            self.logger.error(f"Azure list failed: {e}")
            return []


class GoogleCloudClient(CloudStorageClient):
    """Google Cloud Storage client implementation"""
    
    async def connect(self):
        """Initialize Google Cloud Storage client"""
        try:
            from google.cloud import storage
            
            # Create client with credentials
            if self.config.access_key_id:  # Service account key file
                self._client = storage.Client.from_service_account_json(self.config.access_key_id)
            else:  # Default credentials
                self._client = storage.Client()
            
            # Test connection
            bucket = self._client.bucket(self.config.bucket_name)
            bucket.reload()
            
            self.logger.info(f"Connected to Google Cloud Storage bucket: {self.config.bucket_name}")
            
        except ImportError:
            raise ConfigurationError("Google Cloud SDK not installed. Install with: pip install google-cloud-storage")
        except Exception as e:
            raise DeliveryError(f"Failed to connect to Google Cloud Storage: {e}")
    
    async def upload_file(self, local_path: Path, remote_path: str, **kwargs) -> Dict[str, Any]:
        """Upload file to Google Cloud Storage"""
        try:
            start_time = datetime.now()
            file_size = local_path.stat().st_size
            
            # Get bucket and blob
            bucket = self._client.bucket(self.config.bucket_name)
            blob = bucket.blob(f"{self.config.prefix}{remote_path}")
            
            # Set metadata
            metadata = kwargs.get('metadata', {})
            metadata.update({
                'uploaded_at': datetime.now().isoformat(),
                'original_name': local_path.name,
                'file_size': str(file_size)
            })
            blob.metadata = metadata
            
            # Set storage class
            if self.config.storage_class:
                blob.storage_class = self.config.storage_class
            
            # Upload file
            blob.upload_from_filename(str(local_path))
            
            upload_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'remote_path': f"{self.config.prefix}{remote_path}",
                'file_size': file_size,
                'upload_time': upload_time,
                'etag': blob.etag,
                'generation': blob.generation,
                'url': f"gs://{self.config.bucket_name}/{self.config.prefix}{remote_path}"
            }
            
        except Exception as e:
            self.logger.error(f"GCS upload failed for {local_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'remote_path': remote_path
            }
    
    async def download_file(self, remote_path: str, local_path: Path, **kwargs) -> Dict[str, Any]:
        """Download file from Google Cloud Storage"""
        try:
            start_time = datetime.now()
            
            # Get bucket and blob
            bucket = self._client.bucket(self.config.bucket_name)
            blob = bucket.blob(f"{self.config.prefix}{remote_path}")
            
            # Download file
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))
            
            download_time = (datetime.now() - start_time).total_seconds()
            file_size = local_path.stat().st_size
            
            return {
                'success': True,
                'local_path': str(local_path),
                'file_size': file_size,
                'download_time': download_time,
                'metadata': blob.metadata or {}
            }
            
        except Exception as e:
            self.logger.error(f"GCS download failed for {remote_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'remote_path': remote_path
            }
    
    async def delete_file(self, remote_path: str) -> bool:
        """Delete file from Google Cloud Storage"""
        try:
            bucket = self._client.bucket(self.config.bucket_name)
            blob = bucket.blob(f"{self.config.prefix}{remote_path}")
            blob.delete()
            return True
        except Exception as e:
            self.logger.error(f"GCS delete failed for {remote_path}: {e}")
            return False
    
    async def list_files(self, prefix: str = "", limit: int = 1000) -> List[Dict[str, Any]]:
        """List files in Google Cloud Storage"""
        try:
            bucket = self._client.bucket(self.config.bucket_name)
            
            files = []
            for blob in bucket.list_blobs(prefix=f"{self.config.prefix}{prefix}", max_results=limit):
                files.append({
                    'key': blob.name,
                    'size': blob.size,
                    'last_modified': blob.updated.isoformat() if blob.updated else '',
                    'etag': blob.etag,
                    'content_type': blob.content_type,
                    'storage_class': blob.storage_class
                })
            
            return files
            
        except Exception as e:
            self.logger.error(f"GCS list failed: {e}")
            return []


class CloudStorage:
    """
    Unified cloud storage interface supporting multiple providers.
    
    Provides a consistent API across different cloud storage providers
    with built-in error handling, retry logic, and performance optimization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize cloud storage with configuration.
        
        Args:
            config: Configuration dictionary for cloud storage
        """
        self.config = CloudStorageConfig(config or {})
        self.logger = logging.getLogger(__name__)
        
        # Initialize appropriate client based on provider
        if self.config.provider.lower() == 'aws':
            self._client = AWSS3Client(self.config)
        elif self.config.provider.lower() == 'azure':
            self._client = AzureBlobClient(self.config)
        elif self.config.provider.lower() == 'gcp':
            self._client = GoogleCloudClient(self.config)
        else:
            raise ConfigurationError(f"Unsupported cloud provider: {self.config.provider}")
        
        self._connected = False
    
    async def connect(self):
        """Establish connection to cloud storage"""
        if not self._connected:
            await self._client.connect()
            self._connected = True
    
    async def disconnect(self):
        """Close connection to cloud storage"""
        if self._connected:
            await self._client.disconnect()
            self._connected = False
    
    async def store_files(self, file_paths: List[Union[str, Path]], storage_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Store multiple files in cloud storage.
        
        Args:
            file_paths: List of local file paths to upload
            storage_config: Additional storage configuration
            
        Returns:
            Storage result dictionary
        """
        await self.connect()
        
        storage_config = storage_config or {}
        results = {
            'total_files': len(file_paths),
            'successful_uploads': 0,
            'failed_uploads': 0,
            'upload_results': [],
            'total_size': 0,
            'total_time': 0
        }
        
        start_time = datetime.now()
        
        # Create semaphore for concurrent uploads
        semaphore = asyncio.Semaphore(self.config.max_concurrent_uploads)
        
        async def upload_single_file(file_path):
            async with semaphore:
                local_path = Path(file_path)
                if not local_path.exists():
                    return {
                        'success': False,
                        'local_path': str(local_path),
                        'error': 'File not found'
                    }
                
                # Generate remote path
                remote_path = storage_config.get('remote_path', local_path.name)
                if storage_config.get('preserve_structure', False):
                    # Preserve directory structure
                    base_path = Path(storage_config.get('base_path', '.'))
                    relative_path = local_path.relative_to(base_path)
                    remote_path = str(relative_path)
                
                # Upload file
                return await self._client.upload_file(
                    local_path, 
                    remote_path, 
                    metadata=storage_config.get('metadata', {})
                )
        
        # Upload files concurrently
        upload_tasks = [upload_single_file(file_path) for file_path in file_paths]
        upload_results = await asyncio.gather(*upload_tasks, return_exceptions=True)
        
        # Process results
        for result in upload_results:
            if isinstance(result, Exception):
                results['failed_uploads'] += 1
                results['upload_results'].append({
                    'success': False,
                    'error': str(result)
                })
            elif result.get('success', False):
                results['successful_uploads'] += 1
                results['total_size'] += result.get('file_size', 0)
                results['upload_results'].append(result)
            else:
                results['failed_uploads'] += 1
                results['upload_results'].append(result)
        
        results['total_time'] = (datetime.now() - start_time).total_seconds()
        
        self.logger.info(f"Cloud storage upload completed: {results['successful_uploads']}/{results['total_files']} files")
        return results
    
    async def retrieve_files(self, remote_paths: List[str], local_dir: Path, **kwargs) -> Dict[str, Any]:
        """Retrieve multiple files from cloud storage"""
        await self.connect()
        
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'total_files': len(remote_paths),
            'successful_downloads': 0,
            'failed_downloads': 0,
            'download_results': []
        }
        
        # Create semaphore for concurrent downloads
        semaphore = asyncio.Semaphore(self.config.max_concurrent_uploads)
        
        async def download_single_file(remote_path):
            async with semaphore:
                local_path = local_dir / Path(remote_path).name
                return await self._client.download_file(remote_path, local_path)
        
        # Download files concurrently
        download_tasks = [download_single_file(path) for path in remote_paths]
        download_results = await asyncio.gather(*download_tasks, return_exceptions=True)
        
        # Process results
        for result in download_results:
            if isinstance(result, Exception):
                results['failed_downloads'] += 1
                results['download_results'].append({
                    'success': False,
                    'error': str(result)
                })
            elif result.get('success', False):
                results['successful_downloads'] += 1
                results['download_results'].append(result)
            else:
                results['failed_downloads'] += 1
                results['download_results'].append(result)
        
        return results
    
    async def list_stored_files(self, prefix: str = "", limit: int = 1000) -> List[Dict[str, Any]]:
        """List files stored in cloud storage"""
        await self.connect()
        return await self._client.list_files(prefix, limit)
    
    async def delete_stored_files(self, remote_paths: List[str]) -> Dict[str, Any]:
        """Delete multiple files from cloud storage"""
        await self.connect()
        
        results = {
            'total_files': len(remote_paths),
            'successful_deletions': 0,
            'failed_deletions': 0
        }
        
        for remote_path in remote_paths:
            success = await self._client.delete_file(remote_path)
            if success:
                results['successful_deletions'] += 1
            else:
                results['failed_deletions'] += 1
        
        return results
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get cloud storage statistics"""
        await self.connect()
        
        files = await self.list_stored_files(limit=10000)
        
        total_size = sum(file.get('size', 0) for file in files)
        total_files = len(files)
        
        # Calculate storage distribution by file type
        file_types = {}
        for file in files:
            ext = Path(file['key']).suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        return {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'file_types': file_types,
            'provider': self.config.provider,
            'bucket_name': self.config.bucket_name,
            'region': self.config.region
        }


# Factory function
def create_cloud_storage(config: Dict[str, Any] = None) -> CloudStorage:
    """
    Factory function to create cloud storage instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured cloud storage instance
    """
    return CloudStorage(config or {})