#!/usr/bin/env python3
"""
Google Cloud Storage implementation for document storage.

Provides GCS storage capabilities with lifecycle management,
encryption, and versioning support.
"""

import os
import io
import json
from typing import Dict, List, Optional, Any, Union, BinaryIO, Iterator
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...core.logging import get_logger
from ...core.exceptions import StorageError, ValidationError


logger = get_logger(__name__)


@dataclass
class GCSObject:
    """GCS object metadata"""
    name: str
    size: int
    updated: datetime
    etag: str
    md5_hash: str
    crc32c: str
    storage_class: str = 'STANDARD'
    metadata: Dict[str, str] = None
    generation: Optional[str] = None
    metageneration: Optional[str] = None


class GCSStorage:
    """
    Google Cloud Storage implementation.
    
    Features:
    - Blob upload/download
    - Bucket management
    - Versioning support
    - Customer-managed encryption keys
    - Lifecycle policies
    - Resumable uploads
    - Signed URLs
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize GCS storage"""
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # GCS credentials
        self.project_id = self.config.get('project_id') or os.environ.get('GOOGLE_CLOUD_PROJECT')
        self.credentials_path = self.config.get('credentials_path') or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        self.use_service_account = self.config.get('use_service_account', True)
        
        # Bucket settings
        self.bucket_name = self.config.get('bucket_name')
        self.prefix = self.config.get('prefix', '')
        self.location = self.config.get('location', 'US')
        
        # Storage settings
        self.storage_class = self.config.get('storage_class', 'STANDARD')
        self.uniform_bucket_level_access = self.config.get('uniform_bucket_level_access', False)
        
        # Encryption settings
        self.encryption_key = self.config.get('encryption_key')  # Customer-managed encryption key
        
        # Upload settings
        self.chunk_size = self.config.get('chunk_size', 8 * 1024 * 1024)  # 8MB
        self.resumable_threshold = self.config.get('resumable_threshold', 5 * 1024 * 1024)  # 5MB
        self.max_workers = self.config.get('max_workers', 10)
        
        # Versioning
        self.enable_versioning = self.config.get('enable_versioning', False)
        
        # Initialize client
        self._client = None
        self._bucket = None
        
        self.logger.info(f"GCS storage initialized for bucket: {self.bucket_name}")
    
    @property
    def client(self):
        """Get or create GCS client"""
        if self._client is None:
            try:
                from google.cloud import storage
                from google.oauth2 import service_account
                
                if self.use_service_account and self.credentials_path:
                    credentials = service_account.Credentials.from_service_account_file(
                        self.credentials_path
                    )
                    self._client = storage.Client(
                        project=self.project_id,
                        credentials=credentials
                    )
                else:
                    # Use default credentials
                    self._client = storage.Client(project=self.project_id)
                    
            except ImportError:
                raise StorageError("google-cloud-storage library not installed")
            except Exception as e:
                raise StorageError(f"Failed to create GCS client: {e}")
        
        return self._client
    
    @property
    def bucket(self):
        """Get or create bucket"""
        if self._bucket is None:
            try:
                self._bucket = self.client.bucket(self.bucket_name)
                
                # Check if bucket exists
                if not self._bucket.exists():
                    # Create bucket
                    self._bucket.location = self.location
                    self._bucket.storage_class = self.storage_class
                    self._bucket.versioning_enabled = self.enable_versioning
                    
                    if self.uniform_bucket_level_access:
                        self._bucket.iam_configuration.uniform_bucket_level_access_enabled = True
                    
                    self._bucket = self.client.create_bucket(self._bucket)
                    self.logger.info(f"Created bucket: {self.bucket_name}")
                    
            except Exception as e:
                raise StorageError(f"Failed to access bucket: {e}")
        
        return self._bucket
    
    def upload_file(
        self,
        file_path: Union[str, Path],
        blob_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
        cache_control: Optional[str] = None,
        content_encoding: Optional[str] = None
    ) -> GCSObject:
        """
        Upload file to GCS.
        
        Args:
            file_path: Local file path
            blob_name: Blob name (default: filename)
            metadata: Custom metadata
            content_type: MIME type
            cache_control: Cache control header
            content_encoding: Content encoding
            
        Returns:
            GCSObject with upload details
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine blob name
        if not blob_name:
            blob_name = file_path.name
        
        # Add prefix if configured
        if self.prefix:
            blob_name = f"{self.prefix.rstrip('/')}/{blob_name}"
        
        try:
            # Create blob
            blob = self.bucket.blob(blob_name, chunk_size=self.chunk_size)
            
            # Set properties
            if metadata:
                blob.metadata = metadata
            
            if content_type:
                blob.content_type = content_type
            elif not blob.content_type:
                import mimetypes
                content_type, _ = mimetypes.guess_type(str(file_path))
                if content_type:
                    blob.content_type = content_type
            
            if cache_control:
                blob.cache_control = cache_control
            
            if content_encoding:
                blob.content_encoding = content_encoding
            
            # Set storage class
            blob.storage_class = self.storage_class
            
            # Set encryption key if provided
            if self.encryption_key:
                blob.encryption_key = self.encryption_key
            
            # Upload file
            file_size = file_path.stat().st_size
            
            if file_size > self.resumable_threshold:
                # Use resumable upload for large files
                blob.upload_from_filename(
                    str(file_path),
                    if_generation_match=0 if not blob.exists() else None
                )
            else:
                # Regular upload
                with open(file_path, 'rb') as f:
                    blob.upload_from_file(
                        f,
                        if_generation_match=0 if not blob.exists() else None
                    )
            
            # Reload to get updated metadata
            blob.reload()
            
            return GCSObject(
                name=blob.name,
                size=blob.size,
                updated=blob.updated,
                etag=blob.etag,
                md5_hash=blob.md5_hash,
                crc32c=blob.crc32c,
                storage_class=blob.storage_class,
                metadata=blob.metadata,
                generation=str(blob.generation),
                metageneration=str(blob.metageneration)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to upload file: {e}")
            raise StorageError(f"Upload failed: {e}")
    
    def upload_bytes(
        self,
        data: bytes,
        blob_name: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: str = 'application/octet-stream',
        cache_control: Optional[str] = None
    ) -> GCSObject:
        """
        Upload bytes data to GCS.
        
        Args:
            data: Bytes data
            blob_name: Blob name
            metadata: Custom metadata
            content_type: MIME type
            cache_control: Cache control header
            
        Returns:
            GCSObject with upload details
        """
        # Add prefix if configured
        if self.prefix:
            blob_name = f"{self.prefix.rstrip('/')}/{blob_name}"
        
        try:
            # Create blob
            blob = self.bucket.blob(blob_name)
            
            # Set properties
            if metadata:
                blob.metadata = metadata
            
            blob.content_type = content_type
            
            if cache_control:
                blob.cache_control = cache_control
            
            # Set storage class
            blob.storage_class = self.storage_class
            
            # Set encryption key if provided
            if self.encryption_key:
                blob.encryption_key = self.encryption_key
            
            # Upload data
            blob.upload_from_string(
                data,
                content_type=content_type,
                if_generation_match=0 if not blob.exists() else None
            )
            
            # Reload to get updated metadata
            blob.reload()
            
            return GCSObject(
                name=blob.name,
                size=len(data),
                updated=blob.updated,
                etag=blob.etag,
                md5_hash=blob.md5_hash,
                crc32c=blob.crc32c,
                storage_class=blob.storage_class,
                metadata=blob.metadata,
                generation=str(blob.generation),
                metageneration=str(blob.metageneration)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to upload bytes: {e}")
            raise StorageError(f"Upload failed: {e}")
    
    def download_file(
        self,
        blob_name: str,
        file_path: Union[str, Path],
        generation: Optional[str] = None
    ) -> Path:
        """
        Download blob to file.
        
        Args:
            blob_name: Blob name
            file_path: Local file path
            generation: Specific generation to download
            
        Returns:
            Downloaded file path
        """
        # Add prefix if configured
        if self.prefix and not blob_name.startswith(self.prefix):
            blob_name = f"{self.prefix.rstrip('/')}/{blob_name}"
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get blob
            blob = self.bucket.blob(blob_name)
            
            # Set encryption key if provided
            if self.encryption_key:
                blob.encryption_key = self.encryption_key
            
            # Download file
            if generation:
                blob = self.bucket.blob(blob_name, generation=int(generation))
            
            blob.download_to_filename(str(file_path))
            
            self.logger.info(f"Downloaded {blob_name} to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to download file: {e}")
            raise StorageError(f"Download failed: {e}")
    
    def download_bytes(
        self,
        blob_name: str,
        generation: Optional[str] = None
    ) -> bytes:
        """
        Download blob as bytes.
        
        Args:
            blob_name: Blob name
            generation: Specific generation to download
            
        Returns:
            Blob data as bytes
        """
        # Add prefix if configured
        if self.prefix and not blob_name.startswith(self.prefix):
            blob_name = f"{self.prefix.rstrip('/')}/{blob_name}"
        
        try:
            # Get blob
            if generation:
                blob = self.bucket.blob(blob_name, generation=int(generation))
            else:
                blob = self.bucket.blob(blob_name)
            
            # Set encryption key if provided
            if self.encryption_key:
                blob.encryption_key = self.encryption_key
            
            # Download as bytes
            return blob.download_as_bytes()
            
        except Exception as e:
            self.logger.error(f"Failed to download bytes: {e}")
            raise StorageError(f"Download failed: {e}")
    
    def list_blobs(
        self,
        prefix: Optional[str] = None,
        delimiter: Optional[str] = None,
        max_results: Optional[int] = None,
        include_versions: bool = False
    ) -> List[GCSObject]:
        """
        List blobs in bucket.
        
        Args:
            prefix: Filter by prefix
            delimiter: Delimiter for hierarchy
            max_results: Maximum blobs to return
            include_versions: Include all versions
            
        Returns:
            List of GCSObject instances
        """
        # Combine configured prefix with provided prefix
        if self.prefix:
            if prefix:
                prefix = f"{self.prefix.rstrip('/')}/{prefix}"
            else:
                prefix = self.prefix
        
        blobs = []
        
        try:
            # List blobs
            blob_iter = self.bucket.list_blobs(
                prefix=prefix,
                delimiter=delimiter,
                max_results=max_results,
                versions=include_versions
            )
            
            for blob in blob_iter:
                blobs.append(GCSObject(
                    name=blob.name,
                    size=blob.size,
                    updated=blob.updated,
                    etag=blob.etag,
                    md5_hash=blob.md5_hash,
                    crc32c=blob.crc32c,
                    storage_class=blob.storage_class,
                    metadata=blob.metadata,
                    generation=str(blob.generation) if blob.generation else None,
                    metageneration=str(blob.metageneration) if blob.metageneration else None
                ))
            
            return blobs
            
        except Exception as e:
            self.logger.error(f"Failed to list blobs: {e}")
            raise StorageError(f"List failed: {e}")
    
    def delete_blob(
        self,
        blob_name: str,
        generation: Optional[str] = None
    ) -> bool:
        """
        Delete blob.
        
        Args:
            blob_name: Blob name
            generation: Specific generation to delete
            
        Returns:
            True if deleted successfully
        """
        # Add prefix if configured
        if self.prefix and not blob_name.startswith(self.prefix):
            blob_name = f"{self.prefix.rstrip('/')}/{blob_name}"
        
        try:
            # Get blob
            if generation:
                blob = self.bucket.blob(blob_name, generation=int(generation))
            else:
                blob = self.bucket.blob(blob_name)
            
            # Delete blob
            blob.delete()
            
            self.logger.info(f"Deleted blob: {blob_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete blob: {e}")
            raise StorageError(f"Delete failed: {e}")
    
    def delete_blobs(
        self,
        blob_names: List[str]
    ) -> Dict[str, bool]:
        """
        Delete multiple blobs.
        
        Args:
            blob_names: List of blob names
            
        Returns:
            Dict of blob_name -> success status
        """
        results = {}
        
        # Use batch operations for efficiency
        with self.client.batch():
            for blob_name in blob_names:
                try:
                    # Add prefix if needed
                    full_name = blob_name
                    if self.prefix and not blob_name.startswith(self.prefix):
                        full_name = f"{self.prefix.rstrip('/')}/{blob_name}"
                    
                    blob = self.bucket.blob(full_name)
                    blob.delete()
                    results[blob_name] = True
                except Exception as e:
                    results[blob_name] = False
                    self.logger.error(f"Failed to delete {blob_name}: {e}")
        
        return results
    
    def blob_exists(
        self,
        blob_name: str
    ) -> bool:
        """
        Check if blob exists.
        
        Args:
            blob_name: Blob name
            
        Returns:
            True if blob exists
        """
        # Add prefix if configured
        if self.prefix and not blob_name.startswith(self.prefix):
            blob_name = f"{self.prefix.rstrip('/')}/{blob_name}"
        
        try:
            blob = self.bucket.blob(blob_name)
            return blob.exists()
        except Exception:
            return False
    
    def copy_blob(
        self,
        source_blob: str,
        dest_blob: str,
        source_bucket: Optional[str] = None,
        dest_bucket: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> GCSObject:
        """
        Copy blob within GCS.
        
        Args:
            source_blob: Source blob name
            dest_blob: Destination blob name
            source_bucket: Source bucket (default: same bucket)
            dest_bucket: Destination bucket (default: same bucket)
            metadata: New metadata
            
        Returns:
            Copied GCSObject
        """
        # Add prefix to blob names
        if self.prefix:
            if not source_blob.startswith(self.prefix):
                source_blob = f"{self.prefix.rstrip('/')}/{source_blob}"
            if not dest_blob.startswith(self.prefix):
                dest_blob = f"{self.prefix.rstrip('/')}/{dest_blob}"
        
        source_bucket = source_bucket or self.bucket_name
        dest_bucket = dest_bucket or self.bucket_name
        
        try:
            # Get source blob
            source_bucket_obj = self.client.bucket(source_bucket)
            source_blob_obj = source_bucket_obj.blob(source_blob)
            
            # Get destination bucket
            dest_bucket_obj = self.client.bucket(dest_bucket)
            
            # Copy blob
            dest_blob_obj = source_bucket_obj.copy_blob(
                source_blob_obj,
                dest_bucket_obj,
                dest_blob
            )
            
            # Update metadata if provided
            if metadata:
                dest_blob_obj.metadata = metadata
                dest_blob_obj.patch()
            
            # Reload to get updated metadata
            dest_blob_obj.reload()
            
            return GCSObject(
                name=dest_blob_obj.name,
                size=dest_blob_obj.size,
                updated=dest_blob_obj.updated,
                etag=dest_blob_obj.etag,
                md5_hash=dest_blob_obj.md5_hash,
                crc32c=dest_blob_obj.crc32c,
                storage_class=dest_blob_obj.storage_class,
                metadata=dest_blob_obj.metadata,
                generation=str(dest_blob_obj.generation),
                metageneration=str(dest_blob_obj.metageneration)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to copy blob: {e}")
            raise StorageError(f"Copy failed: {e}")
    
    def generate_signed_url(
        self,
        blob_name: str,
        expiration: timedelta = timedelta(hours=1),
        method: str = 'GET',
        content_type: Optional[str] = None,
        response_disposition: Optional[str] = None,
        generation: Optional[str] = None
    ) -> str:
        """
        Generate signed URL for blob.
        
        Args:
            blob_name: Blob name
            expiration: URL expiration time
            method: HTTP method
            content_type: Required content type
            response_disposition: Content-Disposition header
            generation: Specific generation
            
        Returns:
            Signed URL
        """
        # Add prefix if configured
        if self.prefix and not blob_name.startswith(self.prefix):
            blob_name = f"{self.prefix.rstrip('/')}/{blob_name}"
        
        try:
            # Get blob
            if generation:
                blob = self.bucket.blob(blob_name, generation=int(generation))
            else:
                blob = self.bucket.blob(blob_name)
            
            # Generate signed URL
            url = blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method=method,
                content_type=content_type,
                response_disposition=response_disposition
            )
            
            return url
            
        except Exception as e:
            self.logger.error(f"Failed to generate signed URL: {e}")
            raise StorageError(f"Signed URL generation failed: {e}")
    
    def create_lifecycle_rule(
        self,
        action: str,
        conditions: Dict[str, Any],
        action_params: Optional[Dict[str, Any]] = None
    ):
        """
        Create lifecycle rule for bucket.
        
        Args:
            action: Action type (Delete, SetStorageClass)
            conditions: Rule conditions
            action_params: Action parameters
        """
        try:
            from google.cloud.storage import LifecycleRuleDelete, LifecycleRuleSetStorageClass
            
            # Get current rules
            rules = list(self.bucket.lifecycle_rules)
            
            # Create new rule
            if action == 'Delete':
                rule = LifecycleRuleDelete(**conditions)
            elif action == 'SetStorageClass':
                storage_class = action_params.get('storage_class', 'NEARLINE')
                rule = LifecycleRuleSetStorageClass(
                    storage_class=storage_class,
                    **conditions
                )
            else:
                raise ValueError(f"Unknown action: {action}")
            
            # Add rule
            rules.append(rule)
            
            # Update bucket
            self.bucket.lifecycle_rules = rules
            self.bucket.patch()
            
            self.logger.info(f"Created lifecycle rule: {action}")
            
        except Exception as e:
            self.logger.error(f"Failed to create lifecycle rule: {e}")
            raise StorageError(f"Lifecycle rule creation failed: {e}")
    
    def enable_bucket_versioning(self, enabled: bool = True):
        """
        Enable or disable bucket versioning.
        
        Args:
            enabled: Whether to enable versioning
        """
        try:
            self.bucket.versioning_enabled = enabled
            self.bucket.patch()
            
            status = "enabled" if enabled else "disabled"
            self.logger.info(f"Bucket versioning {status}")
            
        except Exception as e:
            self.logger.error(f"Failed to update versioning: {e}")
            raise StorageError(f"Versioning update failed: {e}")
    
    def batch_upload(
        self,
        file_mapping: Dict[str, Union[str, Path]],
        metadata: Optional[Dict[str, Dict[str, str]]] = None
    ) -> Dict[str, Union[GCSObject, Exception]]:
        """
        Upload multiple files in parallel.
        
        Args:
            file_mapping: Dict of blob_name -> file_path
            metadata: Optional metadata for each blob
            
        Returns:
            Dict of blob_name -> GCSObject or Exception
        """
        results = {}
        metadata = metadata or {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for blob_name, file_path in file_mapping.items():
                blob_metadata = metadata.get(blob_name)
                future = executor.submit(
                    self.upload_file,
                    file_path,
                    blob_name,
                    blob_metadata
                )
                futures[future] = blob_name
            
            for future in as_completed(futures):
                blob_name = futures[future]
                try:
                    results[blob_name] = future.result()
                except Exception as e:
                    results[blob_name] = e
                    self.logger.error(f"Failed to upload {blob_name}: {e}")
        
        return results
    
    def set_blob_metadata(
        self,
        blob_name: str,
        metadata: Dict[str, str],
        merge: bool = True
    ) -> bool:
        """
        Set blob metadata.
        
        Args:
            blob_name: Blob name
            metadata: Metadata to set
            merge: Merge with existing metadata
            
        Returns:
            True if successful
        """
        # Add prefix if configured
        if self.prefix and not blob_name.startswith(self.prefix):
            blob_name = f"{self.prefix.rstrip('/')}/{blob_name}"
        
        try:
            blob = self.bucket.blob(blob_name)
            blob.reload()  # Get current metadata
            
            if merge and blob.metadata:
                blob.metadata.update(metadata)
            else:
                blob.metadata = metadata
            
            blob.patch()
            
            self.logger.info(f"Updated metadata for {blob_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set metadata: {e}")
            raise StorageError(f"Metadata update failed: {e}")


# Factory function
def create_gcs_storage(config: Optional[Dict[str, Any]] = None) -> GCSStorage:
    """Create and return a GCS storage instance"""
    return GCSStorage(config)