#!/usr/bin/env python3
"""
Amazon S3 Storage implementation for document storage.

Provides S3 storage capabilities with versioning, encryption,
and lifecycle management support.
"""

import os
import json
import mimetypes
from typing import Dict, List, Optional, Any, Union, BinaryIO, Iterator
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...core.logging import get_logger
from ...core.exceptions import StorageError, ValidationError


logger = get_logger(__name__)


@dataclass
class S3Object:
    """S3 object metadata"""
    key: str
    size: int
    last_modified: datetime
    etag: str
    storage_class: str = 'STANDARD'
    metadata: Dict[str, str] = None
    version_id: Optional[str] = None


class S3Storage:
    """
    Amazon S3 storage implementation.
    
    Features:
    - Object upload/download
    - Versioning support
    - Server-side encryption
    - Lifecycle policies
    - Multipart uploads
    - Pre-signed URLs
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize S3 storage"""
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # AWS credentials
        self.aws_access_key_id = self.config.get('aws_access_key_id') or os.environ.get('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = self.config.get('aws_secret_access_key') or os.environ.get('AWS_SECRET_ACCESS_KEY')
        self.region_name = self.config.get('region_name', 'us-east-1')
        
        # S3 settings
        self.bucket_name = self.config.get('bucket_name')
        self.prefix = self.config.get('prefix', '')
        self.storage_class = self.config.get('storage_class', 'STANDARD')
        
        # Encryption settings
        self.encryption = self.config.get('encryption', 'AES256')  # AES256 or aws:kms
        self.kms_key_id = self.config.get('kms_key_id')
        
        # Upload settings
        self.multipart_threshold = self.config.get('multipart_threshold', 100 * 1024 * 1024)  # 100MB
        self.multipart_chunksize = self.config.get('multipart_chunksize', 10 * 1024 * 1024)  # 10MB
        self.max_concurrency = self.config.get('max_concurrency', 10)
        
        # Versioning
        self.enable_versioning = self.config.get('enable_versioning', False)
        
        # Initialize client
        self._client = None
        self._resource = None
        
        self.logger.info(f"S3 storage initialized for bucket: {self.bucket_name}")
    
    @property
    def client(self):
        """Get or create S3 client"""
        if self._client is None:
            try:
                import boto3
                
                self._client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.region_name
                )
            except ImportError:
                raise StorageError("boto3 library not installed")
            except Exception as e:
                raise StorageError(f"Failed to create S3 client: {e}")
        
        return self._client
    
    @property
    def resource(self):
        """Get or create S3 resource"""
        if self._resource is None:
            try:
                import boto3
                
                self._resource = boto3.resource(
                    's3',
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.region_name
                )
            except ImportError:
                raise StorageError("boto3 library not installed")
            except Exception as e:
                raise StorageError(f"Failed to create S3 resource: {e}")
        
        return self._resource
    
    def upload_file(
        self,
        file_path: Union[str, Path],
        key: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None
    ) -> S3Object:
        """
        Upload file to S3.
        
        Args:
            file_path: Local file path
            key: S3 object key (default: filename)
            metadata: Object metadata
            tags: Object tags
            content_type: MIME type
            
        Returns:
            S3Object with upload details
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine key
        if not key:
            key = file_path.name
        
        # Add prefix if configured
        if self.prefix:
            key = f"{self.prefix.rstrip('/')}/{key}"
        
        # Determine content type
        if not content_type:
            content_type, _ = mimetypes.guess_type(str(file_path))
            content_type = content_type or 'application/octet-stream'
        
        try:
            # Prepare extra args
            extra_args = {
                'ContentType': content_type,
                'StorageClass': self.storage_class
            }
            
            # Add encryption
            if self.encryption == 'AES256':
                extra_args['ServerSideEncryption'] = 'AES256'
            elif self.encryption == 'aws:kms' and self.kms_key_id:
                extra_args['ServerSideEncryption'] = 'aws:kms'
                extra_args['SSEKMSKeyId'] = self.kms_key_id
            
            # Add metadata
            if metadata:
                extra_args['Metadata'] = metadata
            
            # Upload file
            file_size = file_path.stat().st_size
            
            if file_size > self.multipart_threshold:
                # Use multipart upload
                self._multipart_upload(file_path, key, extra_args)
            else:
                # Regular upload
                self.client.upload_file(
                    str(file_path),
                    self.bucket_name,
                    key,
                    ExtraArgs=extra_args
                )
            
            # Add tags if provided
            if tags:
                self._set_object_tags(key, tags)
            
            # Get object info
            response = self.client.head_object(Bucket=self.bucket_name, Key=key)
            
            return S3Object(
                key=key,
                size=response['ContentLength'],
                last_modified=response['LastModified'],
                etag=response['ETag'].strip('"'),
                storage_class=response.get('StorageClass', 'STANDARD'),
                metadata=response.get('Metadata', {}),
                version_id=response.get('VersionId')
            )
            
        except Exception as e:
            self.logger.error(f"Failed to upload file: {e}")
            raise StorageError(f"Upload failed: {e}")
    
    def upload_bytes(
        self,
        data: bytes,
        key: str,
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        content_type: str = 'application/octet-stream'
    ) -> S3Object:
        """
        Upload bytes data to S3.
        
        Args:
            data: Bytes data
            key: S3 object key
            metadata: Object metadata
            tags: Object tags
            content_type: MIME type
            
        Returns:
            S3Object with upload details
        """
        # Add prefix if configured
        if self.prefix:
            key = f"{self.prefix.rstrip('/')}/{key}"
        
        try:
            # Prepare args
            put_args = {
                'Bucket': self.bucket_name,
                'Key': key,
                'Body': data,
                'ContentType': content_type,
                'StorageClass': self.storage_class
            }
            
            # Add encryption
            if self.encryption == 'AES256':
                put_args['ServerSideEncryption'] = 'AES256'
            elif self.encryption == 'aws:kms' and self.kms_key_id:
                put_args['ServerSideEncryption'] = 'aws:kms'
                put_args['SSEKMSKeyId'] = self.kms_key_id
            
            # Add metadata
            if metadata:
                put_args['Metadata'] = metadata
            
            # Upload
            response = self.client.put_object(**put_args)
            
            # Add tags if provided
            if tags:
                self._set_object_tags(key, tags)
            
            return S3Object(
                key=key,
                size=len(data),
                last_modified=datetime.now(),
                etag=response['ETag'].strip('"'),
                storage_class=self.storage_class,
                metadata=metadata or {},
                version_id=response.get('VersionId')
            )
            
        except Exception as e:
            self.logger.error(f"Failed to upload bytes: {e}")
            raise StorageError(f"Upload failed: {e}")
    
    def download_file(
        self,
        key: str,
        file_path: Union[str, Path],
        version_id: Optional[str] = None
    ) -> Path:
        """
        Download file from S3.
        
        Args:
            key: S3 object key
            file_path: Local file path
            version_id: Specific version to download
            
        Returns:
            Downloaded file path
        """
        # Add prefix if configured
        if self.prefix and not key.startswith(self.prefix):
            key = f"{self.prefix.rstrip('/')}/{key}"
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Prepare download args
            download_args = {
                'Bucket': self.bucket_name,
                'Key': key
            }
            
            if version_id:
                download_args['VersionId'] = version_id
            
            # Download file
            self.client.download_file(
                self.bucket_name,
                key,
                str(file_path),
                ExtraArgs={'VersionId': version_id} if version_id else None
            )
            
            self.logger.info(f"Downloaded {key} to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to download file: {e}")
            raise StorageError(f"Download failed: {e}")
    
    def download_bytes(
        self,
        key: str,
        version_id: Optional[str] = None
    ) -> bytes:
        """
        Download object as bytes.
        
        Args:
            key: S3 object key
            version_id: Specific version to download
            
        Returns:
            Object data as bytes
        """
        # Add prefix if configured
        if self.prefix and not key.startswith(self.prefix):
            key = f"{self.prefix.rstrip('/')}/{key}"
        
        try:
            # Prepare get args
            get_args = {
                'Bucket': self.bucket_name,
                'Key': key
            }
            
            if version_id:
                get_args['VersionId'] = version_id
            
            # Download object
            response = self.client.get_object(**get_args)
            
            return response['Body'].read()
            
        except Exception as e:
            self.logger.error(f"Failed to download bytes: {e}")
            raise StorageError(f"Download failed: {e}")
    
    def list_objects(
        self,
        prefix: Optional[str] = None,
        max_keys: int = 1000,
        delimiter: Optional[str] = None
    ) -> List[S3Object]:
        """
        List objects in bucket.
        
        Args:
            prefix: Filter by prefix
            max_keys: Maximum objects to return
            delimiter: Delimiter for hierarchy
            
        Returns:
            List of S3Objects
        """
        # Combine configured prefix with provided prefix
        if self.prefix:
            if prefix:
                prefix = f"{self.prefix.rstrip('/')}/{prefix}"
            else:
                prefix = self.prefix
        
        objects = []
        
        try:
            # Prepare list args
            list_args = {
                'Bucket': self.bucket_name,
                'MaxKeys': max_keys
            }
            
            if prefix:
                list_args['Prefix'] = prefix
            
            if delimiter:
                list_args['Delimiter'] = delimiter
            
            # List objects
            paginator = self.client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(**list_args):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects.append(S3Object(
                            key=obj['Key'],
                            size=obj['Size'],
                            last_modified=obj['LastModified'],
                            etag=obj['ETag'].strip('"'),
                            storage_class=obj.get('StorageClass', 'STANDARD')
                        ))
                
                if len(objects) >= max_keys:
                    break
            
            return objects[:max_keys]
            
        except Exception as e:
            self.logger.error(f"Failed to list objects: {e}")
            raise StorageError(f"List failed: {e}")
    
    def delete_object(
        self,
        key: str,
        version_id: Optional[str] = None
    ) -> bool:
        """
        Delete object from S3.
        
        Args:
            key: S3 object key
            version_id: Specific version to delete
            
        Returns:
            True if deleted successfully
        """
        # Add prefix if configured
        if self.prefix and not key.startswith(self.prefix):
            key = f"{self.prefix.rstrip('/')}/{key}"
        
        try:
            # Prepare delete args
            delete_args = {
                'Bucket': self.bucket_name,
                'Key': key
            }
            
            if version_id:
                delete_args['VersionId'] = version_id
            
            # Delete object
            self.client.delete_object(**delete_args)
            
            self.logger.info(f"Deleted object: {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete object: {e}")
            raise StorageError(f"Delete failed: {e}")
    
    def delete_objects(
        self,
        keys: List[str]
    ) -> Dict[str, bool]:
        """
        Delete multiple objects.
        
        Args:
            keys: List of S3 object keys
            
        Returns:
            Dict of key -> success status
        """
        results = {}
        
        # Add prefix to keys
        prefixed_keys = []
        for key in keys:
            if self.prefix and not key.startswith(self.prefix):
                prefixed_keys.append(f"{self.prefix.rstrip('/')}/{key}")
            else:
                prefixed_keys.append(key)
        
        try:
            # Batch delete (max 1000 at a time)
            for i in range(0, len(prefixed_keys), 1000):
                batch = prefixed_keys[i:i+1000]
                
                delete_objects = {
                    'Objects': [{'Key': key} for key in batch]
                }
                
                response = self.client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete=delete_objects
                )
                
                # Process results
                for deleted in response.get('Deleted', []):
                    original_key = keys[prefixed_keys.index(deleted['Key'])]
                    results[original_key] = True
                
                for error in response.get('Errors', []):
                    original_key = keys[prefixed_keys.index(error['Key'])]
                    results[original_key] = False
                    self.logger.error(f"Failed to delete {error['Key']}: {error['Message']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to delete objects: {e}")
            raise StorageError(f"Batch delete failed: {e}")
    
    def object_exists(
        self,
        key: str
    ) -> bool:
        """
        Check if object exists.
        
        Args:
            key: S3 object key
            
        Returns:
            True if object exists
        """
        # Add prefix if configured
        if self.prefix and not key.startswith(self.prefix):
            key = f"{self.prefix.rstrip('/')}/{key}"
        
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except self.client.exceptions.NoSuchKey:
            return False
        except Exception as e:
            self.logger.error(f"Failed to check object existence: {e}")
            raise StorageError(f"Existence check failed: {e}")
    
    def copy_object(
        self,
        source_key: str,
        dest_key: str,
        source_bucket: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> S3Object:
        """
        Copy object within S3.
        
        Args:
            source_key: Source object key
            dest_key: Destination object key
            source_bucket: Source bucket (default: same bucket)
            metadata: New metadata
            
        Returns:
            Copied S3Object
        """
        # Add prefix to keys
        if self.prefix:
            if not source_key.startswith(self.prefix):
                source_key = f"{self.prefix.rstrip('/')}/{source_key}"
            if not dest_key.startswith(self.prefix):
                dest_key = f"{self.prefix.rstrip('/')}/{dest_key}"
        
        source_bucket = source_bucket or self.bucket_name
        
        try:
            # Prepare copy source
            copy_source = {
                'Bucket': source_bucket,
                'Key': source_key
            }
            
            # Prepare copy args
            copy_args = {
                'CopySource': copy_source,
                'Bucket': self.bucket_name,
                'Key': dest_key,
                'StorageClass': self.storage_class
            }
            
            # Add encryption
            if self.encryption == 'AES256':
                copy_args['ServerSideEncryption'] = 'AES256'
            elif self.encryption == 'aws:kms' and self.kms_key_id:
                copy_args['ServerSideEncryption'] = 'aws:kms'
                copy_args['SSEKMSKeyId'] = self.kms_key_id
            
            # Add metadata
            if metadata:
                copy_args['Metadata'] = metadata
                copy_args['MetadataDirective'] = 'REPLACE'
            else:
                copy_args['MetadataDirective'] = 'COPY'
            
            # Copy object
            response = self.client.copy_object(**copy_args)
            
            # Get object info
            head_response = self.client.head_object(Bucket=self.bucket_name, Key=dest_key)
            
            return S3Object(
                key=dest_key,
                size=head_response['ContentLength'],
                last_modified=head_response['LastModified'],
                etag=response['CopyObjectResult']['ETag'].strip('"'),
                storage_class=head_response.get('StorageClass', 'STANDARD'),
                metadata=head_response.get('Metadata', {}),
                version_id=response.get('VersionId')
            )
            
        except Exception as e:
            self.logger.error(f"Failed to copy object: {e}")
            raise StorageError(f"Copy failed: {e}")
    
    def generate_presigned_url(
        self,
        key: str,
        operation: str = 'get_object',
        expires_in: int = 3600,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate pre-signed URL.
        
        Args:
            key: S3 object key
            operation: Operation (get_object, put_object)
            expires_in: URL expiration in seconds
            params: Additional parameters
            
        Returns:
            Pre-signed URL
        """
        # Add prefix if configured
        if self.prefix and not key.startswith(self.prefix):
            key = f"{self.prefix.rstrip('/')}/{key}"
        
        try:
            # Prepare parameters
            url_params = {
                'Bucket': self.bucket_name,
                'Key': key
            }
            
            if params:
                url_params.update(params)
            
            # Generate URL
            url = self.client.generate_presigned_url(
                ClientMethod=operation,
                Params=url_params,
                ExpiresIn=expires_in
            )
            
            return url
            
        except Exception as e:
            self.logger.error(f"Failed to generate presigned URL: {e}")
            raise StorageError(f"URL generation failed: {e}")
    
    def _multipart_upload(
        self,
        file_path: Path,
        key: str,
        extra_args: Dict[str, Any]
    ):
        """Perform multipart upload for large files"""
        try:
            # Create multipart upload
            create_response = self.client.create_multipart_upload(
                Bucket=self.bucket_name,
                Key=key,
                **extra_args
            )
            
            upload_id = create_response['UploadId']
            parts = []
            
            # Upload parts
            file_size = file_path.stat().st_size
            part_number = 1
            
            with open(file_path, 'rb') as f:
                with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                    futures = []
                    
                    while True:
                        data = f.read(self.multipart_chunksize)
                        if not data:
                            break
                        
                        future = executor.submit(
                            self._upload_part,
                            data,
                            self.bucket_name,
                            key,
                            upload_id,
                            part_number
                        )
                        futures.append((part_number, future))
                        part_number += 1
                    
                    # Collect results
                    for part_num, future in futures:
                        etag = future.result()
                        parts.append({
                            'PartNumber': part_num,
                            'ETag': etag
                        })
            
            # Complete multipart upload
            self.client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            
        except Exception as e:
            # Abort multipart upload on error
            try:
                self.client.abort_multipart_upload(
                    Bucket=self.bucket_name,
                    Key=key,
                    UploadId=upload_id
                )
            except:
                pass
            
            raise StorageError(f"Multipart upload failed: {e}")
    
    def _upload_part(
        self,
        data: bytes,
        bucket: str,
        key: str,
        upload_id: str,
        part_number: int
    ) -> str:
        """Upload single part in multipart upload"""
        response = self.client.upload_part(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            PartNumber=part_number,
            Body=data
        )
        return response['ETag']
    
    def _set_object_tags(
        self,
        key: str,
        tags: Dict[str, str]
    ):
        """Set object tags"""
        tag_set = [{'Key': k, 'Value': v} for k, v in tags.items()]
        
        self.client.put_object_tagging(
            Bucket=self.bucket_name,
            Key=key,
            Tagging={'TagSet': tag_set}
        )
    
    def create_bucket_lifecycle_rule(
        self,
        rule_id: str,
        prefix: str,
        expiration_days: int,
        storage_class_transitions: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Create lifecycle rule for automatic object management.
        
        Args:
            rule_id: Rule identifier
            prefix: Object prefix filter
            expiration_days: Days until expiration
            storage_class_transitions: Storage class transition rules
        """
        try:
            # Get existing configuration
            try:
                response = self.client.get_bucket_lifecycle_configuration(
                    Bucket=self.bucket_name
                )
                rules = response.get('Rules', [])
            except self.client.exceptions.NoSuchLifecycleConfiguration:
                rules = []
            
            # Create new rule
            rule = {
                'ID': rule_id,
                'Status': 'Enabled',
                'Filter': {'Prefix': prefix},
                'Expiration': {'Days': expiration_days}
            }
            
            # Add transitions
            if storage_class_transitions:
                rule['Transitions'] = storage_class_transitions
            
            # Add/update rule
            rules = [r for r in rules if r['ID'] != rule_id]
            rules.append(rule)
            
            # Update configuration
            self.client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket_name,
                LifecycleConfiguration={'Rules': rules}
            )
            
            self.logger.info(f"Created lifecycle rule: {rule_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to create lifecycle rule: {e}")
            raise StorageError(f"Lifecycle rule creation failed: {e}")


# Factory function
def create_s3_storage(config: Optional[Dict[str, Any]] = None) -> S3Storage:
    """Create and return an S3 storage instance"""
    return S3Storage(config)