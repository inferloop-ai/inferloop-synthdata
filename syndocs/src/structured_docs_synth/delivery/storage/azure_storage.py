#!/usr/bin/env python3
"""
Azure Blob Storage implementation for document storage.

Provides Azure Blob Storage capabilities with tiering, encryption,
and lifecycle management support.
"""

import os
import io
from typing import Dict, List, Optional, Any, Union, BinaryIO, Iterator
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib

from ...core.logging import get_logger
from ...core.exceptions import StorageError, ValidationError


logger = get_logger(__name__)


@dataclass
class BlobInfo:
    """Azure Blob metadata"""
    name: str
    size: int
    last_modified: datetime
    etag: str
    content_type: str = 'application/octet-stream'
    blob_tier: str = 'Hot'
    metadata: Dict[str, str] = None
    tags: Dict[str, str] = None
    version_id: Optional[str] = None


class AzureStorage:
    """
    Azure Blob Storage implementation.
    
    Features:
    - Blob upload/download
    - Container management
    - Access tiers (Hot, Cool, Archive)
    - Blob versioning
    - Soft delete
    - Encryption
    - SAS token generation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Azure storage"""
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Azure credentials
        self.account_name = self.config.get('account_name') or os.environ.get('AZURE_STORAGE_ACCOUNT')
        self.account_key = self.config.get('account_key') or os.environ.get('AZURE_STORAGE_KEY')
        self.connection_string = self.config.get('connection_string') or os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
        self.sas_token = self.config.get('sas_token')
        
        # Container settings
        self.container_name = self.config.get('container_name')
        self.prefix = self.config.get('prefix', '')
        
        # Storage settings
        self.blob_tier = self.config.get('blob_tier', 'Hot')  # Hot, Cool, Archive
        self.enable_versioning = self.config.get('enable_versioning', False)
        self.enable_soft_delete = self.config.get('enable_soft_delete', False)
        self.soft_delete_days = self.config.get('soft_delete_days', 7)
        
        # Upload settings
        self.max_single_put_size = self.config.get('max_single_put_size', 64 * 1024 * 1024)  # 64MB
        self.max_block_size = self.config.get('max_block_size', 4 * 1024 * 1024)  # 4MB
        self.max_concurrency = self.config.get('max_concurrency', 10)
        
        # Initialize clients
        self._blob_service_client = None
        self._container_client = None
        
        self.logger.info(f"Azure storage initialized for container: {self.container_name}")
    
    @property
    def blob_service_client(self):
        """Get or create blob service client"""
        if self._blob_service_client is None:
            try:
                from azure.storage.blob import BlobServiceClient
                
                if self.connection_string:
                    self._blob_service_client = BlobServiceClient.from_connection_string(
                        self.connection_string
                    )
                elif self.account_name and self.account_key:
                    account_url = f"https://{self.account_name}.blob.core.windows.net"
                    self._blob_service_client = BlobServiceClient(
                        account_url=account_url,
                        credential=self.account_key
                    )
                elif self.account_name and self.sas_token:
                    account_url = f"https://{self.account_name}.blob.core.windows.net"
                    self._blob_service_client = BlobServiceClient(
                        account_url=account_url,
                        credential=self.sas_token
                    )
                else:
                    # Try default credential
                    from azure.identity import DefaultAzureCredential
                    account_url = f"https://{self.account_name}.blob.core.windows.net"
                    self._blob_service_client = BlobServiceClient(
                        account_url=account_url,
                        credential=DefaultAzureCredential()
                    )
                    
            except ImportError:
                raise StorageError("azure-storage-blob library not installed")
            except Exception as e:
                raise StorageError(f"Failed to create blob service client: {e}")
        
        return self._blob_service_client
    
    @property
    def container_client(self):
        """Get or create container client"""
        if self._container_client is None:
            self._container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            
            # Ensure container exists
            try:
                self._container_client.get_container_properties()
            except Exception:
                # Create container if it doesn't exist
                try:
                    self._container_client.create_container()
                    self.logger.info(f"Created container: {self.container_name}")
                except Exception as e:
                    if "ContainerAlreadyExists" not in str(e):
                        raise StorageError(f"Failed to create container: {e}")
        
        return self._container_client
    
    def upload_file(
        self,
        file_path: Union[str, Path],
        blob_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
        tier: Optional[str] = None
    ) -> BlobInfo:
        """
        Upload file to Azure Blob Storage.
        
        Args:
            file_path: Local file path
            blob_name: Blob name (default: filename)
            metadata: Blob metadata
            tags: Blob tags
            content_type: MIME type
            tier: Access tier
            
        Returns:
            BlobInfo with upload details
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
        
        # Determine content type
        if not content_type:
            import mimetypes
            content_type, _ = mimetypes.guess_type(str(file_path))
            content_type = content_type or 'application/octet-stream'
        
        # Determine tier
        tier = tier or self.blob_tier
        
        try:
            # Get blob client
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Upload file
            with open(file_path, 'rb') as data:
                blob_client.upload_blob(
                    data,
                    overwrite=True,
                    content_settings={'content_type': content_type},
                    metadata=metadata,
                    standard_blob_tier=tier,
                    max_concurrency=self.max_concurrency
                )
            
            # Set tags if provided
            if tags:
                blob_client.set_blob_tags(tags)
            
            # Get blob properties
            properties = blob_client.get_blob_properties()
            
            return BlobInfo(
                name=blob_name,
                size=properties.size,
                last_modified=properties.last_modified,
                etag=properties.etag,
                content_type=properties.content_settings.content_type,
                blob_tier=properties.blob_tier,
                metadata=properties.metadata,
                tags=tags,
                version_id=properties.get('version_id')
            )
            
        except Exception as e:
            self.logger.error(f"Failed to upload file: {e}")
            raise StorageError(f"Upload failed: {e}")
    
    def upload_bytes(
        self,
        data: bytes,
        blob_name: str,
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        content_type: str = 'application/octet-stream',
        tier: Optional[str] = None
    ) -> BlobInfo:
        """
        Upload bytes data to Azure Blob Storage.
        
        Args:
            data: Bytes data
            blob_name: Blob name
            metadata: Blob metadata
            tags: Blob tags
            content_type: MIME type
            tier: Access tier
            
        Returns:
            BlobInfo with upload details
        """
        # Add prefix if configured
        if self.prefix:
            blob_name = f"{self.prefix.rstrip('/')}/{blob_name}"
        
        # Determine tier
        tier = tier or self.blob_tier
        
        try:
            # Get blob client
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Upload data
            blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings={'content_type': content_type},
                metadata=metadata,
                standard_blob_tier=tier
            )
            
            # Set tags if provided
            if tags:
                blob_client.set_blob_tags(tags)
            
            # Get blob properties
            properties = blob_client.get_blob_properties()
            
            return BlobInfo(
                name=blob_name,
                size=len(data),
                last_modified=properties.last_modified,
                etag=properties.etag,
                content_type=content_type,
                blob_tier=tier,
                metadata=metadata,
                tags=tags,
                version_id=properties.get('version_id')
            )
            
        except Exception as e:
            self.logger.error(f"Failed to upload bytes: {e}")
            raise StorageError(f"Upload failed: {e}")
    
    def download_file(
        self,
        blob_name: str,
        file_path: Union[str, Path],
        version_id: Optional[str] = None
    ) -> Path:
        """
        Download blob to file.
        
        Args:
            blob_name: Blob name
            file_path: Local file path
            version_id: Specific version to download
            
        Returns:
            Downloaded file path
        """
        # Add prefix if configured
        if self.prefix and not blob_name.startswith(self.prefix):
            blob_name = f"{self.prefix.rstrip('/')}/{blob_name}"
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get blob client
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Download blob
            with open(file_path, 'wb') as file:
                download_stream = blob_client.download_blob(version_id=version_id)
                file.write(download_stream.readall())
            
            self.logger.info(f"Downloaded {blob_name} to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to download file: {e}")
            raise StorageError(f"Download failed: {e}")
    
    def download_bytes(
        self,
        blob_name: str,
        version_id: Optional[str] = None
    ) -> bytes:
        """
        Download blob as bytes.
        
        Args:
            blob_name: Blob name
            version_id: Specific version to download
            
        Returns:
            Blob data as bytes
        """
        # Add prefix if configured
        if self.prefix and not blob_name.startswith(self.prefix):
            blob_name = f"{self.prefix.rstrip('/')}/{blob_name}"
        
        try:
            # Get blob client
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Download blob
            download_stream = blob_client.download_blob(version_id=version_id)
            return download_stream.readall()
            
        except Exception as e:
            self.logger.error(f"Failed to download bytes: {e}")
            raise StorageError(f"Download failed: {e}")
    
    def list_blobs(
        self,
        prefix: Optional[str] = None,
        max_results: Optional[int] = None,
        include_metadata: bool = False,
        include_tags: bool = False
    ) -> List[BlobInfo]:
        """
        List blobs in container.
        
        Args:
            prefix: Filter by prefix
            max_results: Maximum blobs to return
            include_metadata: Include blob metadata
            include_tags: Include blob tags
            
        Returns:
            List of BlobInfo objects
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
            blob_list = self.container_client.list_blobs(
                name_starts_with=prefix,
                include=['metadata'] if include_metadata else None
            )
            
            for blob in blob_list:
                blob_info = BlobInfo(
                    name=blob.name,
                    size=blob.size,
                    last_modified=blob.last_modified,
                    etag=blob.etag,
                    content_type=blob.content_settings.content_type if blob.content_settings else 'application/octet-stream',
                    blob_tier=blob.blob_tier,
                    metadata=blob.metadata if include_metadata else None,
                    version_id=blob.get('version_id')
                )
                
                # Get tags if requested
                if include_tags:
                    try:
                        blob_client = self.container_client.get_blob_client(blob.name)
                        tags = blob_client.get_blob_tags()
                        blob_info.tags = dict(tags)
                    except:
                        pass
                
                blobs.append(blob_info)
                
                if max_results and len(blobs) >= max_results:
                    break
            
            return blobs
            
        except Exception as e:
            self.logger.error(f"Failed to list blobs: {e}")
            raise StorageError(f"List failed: {e}")
    
    def delete_blob(
        self,
        blob_name: str,
        delete_snapshots: bool = True
    ) -> bool:
        """
        Delete blob.
        
        Args:
            blob_name: Blob name
            delete_snapshots: Delete snapshots as well
            
        Returns:
            True if deleted successfully
        """
        # Add prefix if configured
        if self.prefix and not blob_name.startswith(self.prefix):
            blob_name = f"{self.prefix.rstrip('/')}/{blob_name}"
        
        try:
            # Get blob client
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Delete blob
            blob_client.delete_blob(
                delete_snapshots='include' if delete_snapshots else 'only'
            )
            
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
        
        for blob_name in blob_names:
            try:
                self.delete_blob(blob_name)
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
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.get_blob_properties()
            return True
        except Exception:
            return False
    
    def copy_blob(
        self,
        source_blob: str,
        dest_blob: str,
        source_container: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        tier: Optional[str] = None
    ) -> BlobInfo:
        """
        Copy blob within Azure Storage.
        
        Args:
            source_blob: Source blob name
            dest_blob: Destination blob name
            source_container: Source container (default: same container)
            metadata: New metadata
            tier: Access tier for destination
            
        Returns:
            Copied BlobInfo
        """
        # Add prefix to blob names
        if self.prefix:
            if not source_blob.startswith(self.prefix):
                source_blob = f"{self.prefix.rstrip('/')}/{source_blob}"
            if not dest_blob.startswith(self.prefix):
                dest_blob = f"{self.prefix.rstrip('/')}/{dest_blob}"
        
        source_container = source_container or self.container_name
        tier = tier or self.blob_tier
        
        try:
            # Get source URL
            source_blob_client = self.blob_service_client.get_blob_client(
                container=source_container,
                blob=source_blob
            )
            source_url = source_blob_client.url
            
            # Get destination blob client
            dest_blob_client = self.container_client.get_blob_client(dest_blob)
            
            # Start copy
            copy_result = dest_blob_client.start_copy_from_url(
                source_url,
                metadata=metadata,
                standard_blob_tier=tier
            )
            
            # Wait for copy to complete
            properties = dest_blob_client.get_blob_properties()
            while properties.copy.status == 'pending':
                import time
                time.sleep(1)
                properties = dest_blob_client.get_blob_properties()
            
            if properties.copy.status != 'success':
                raise StorageError(f"Copy failed: {properties.copy.status}")
            
            return BlobInfo(
                name=dest_blob,
                size=properties.size,
                last_modified=properties.last_modified,
                etag=properties.etag,
                content_type=properties.content_settings.content_type,
                blob_tier=tier,
                metadata=metadata or properties.metadata,
                version_id=properties.get('version_id')
            )
            
        except Exception as e:
            self.logger.error(f"Failed to copy blob: {e}")
            raise StorageError(f"Copy failed: {e}")
    
    def set_blob_tier(
        self,
        blob_name: str,
        tier: str
    ) -> bool:
        """
        Set blob access tier.
        
        Args:
            blob_name: Blob name
            tier: Access tier (Hot, Cool, Archive)
            
        Returns:
            True if successful
        """
        # Add prefix if configured
        if self.prefix and not blob_name.startswith(self.prefix):
            blob_name = f"{self.prefix.rstrip('/')}/{blob_name}"
        
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.set_standard_blob_tier(tier)
            
            self.logger.info(f"Set blob {blob_name} tier to {tier}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set blob tier: {e}")
            raise StorageError(f"Set tier failed: {e}")
    
    def generate_sas_url(
        self,
        blob_name: str,
        permissions: str = 'r',
        expiry_hours: int = 1,
        start_time: Optional[datetime] = None
    ) -> str:
        """
        Generate SAS URL for blob.
        
        Args:
            blob_name: Blob name
            permissions: Permissions (r=read, w=write, d=delete)
            expiry_hours: Hours until expiry
            start_time: Start time (default: now)
            
        Returns:
            SAS URL
        """
        # Add prefix if configured
        if self.prefix and not blob_name.startswith(self.prefix):
            blob_name = f"{self.prefix.rstrip('/')}/{blob_name}"
        
        try:
            from azure.storage.blob import generate_blob_sas, BlobSasPermissions
            
            # Set times
            if not start_time:
                start_time = datetime.utcnow()
            expiry_time = start_time + timedelta(hours=expiry_hours)
            
            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=self.account_name,
                container_name=self.container_name,
                blob_name=blob_name,
                account_key=self.account_key,
                permission=BlobSasPermissions(
                    read='r' in permissions,
                    write='w' in permissions,
                    delete='d' in permissions
                ),
                expiry=expiry_time,
                start=start_time
            )
            
            # Construct URL
            blob_client = self.container_client.get_blob_client(blob_name)
            sas_url = f"{blob_client.url}?{sas_token}"
            
            return sas_url
            
        except Exception as e:
            self.logger.error(f"Failed to generate SAS URL: {e}")
            raise StorageError(f"SAS generation failed: {e}")
    
    def create_container_lifecycle_policy(
        self,
        rule_name: str,
        prefix_filter: str,
        delete_after_days: Optional[int] = None,
        tier_to_cool_after_days: Optional[int] = None,
        tier_to_archive_after_days: Optional[int] = None
    ):
        """
        Create lifecycle management policy.
        
        Args:
            rule_name: Rule name
            prefix_filter: Blob prefix filter
            delete_after_days: Days until deletion
            tier_to_cool_after_days: Days until tier to Cool
            tier_to_archive_after_days: Days until tier to Archive
        """
        try:
            from azure.storage.blob import ManagementPolicy, ManagementPolicyRule, ManagementPolicyAction
            
            # Build actions
            actions = {}
            
            if tier_to_cool_after_days:
                actions['baseBlob'] = {
                    'tierToCool': {'daysAfterModificationGreaterThan': tier_to_cool_after_days}
                }
            
            if tier_to_archive_after_days:
                if 'baseBlob' not in actions:
                    actions['baseBlob'] = {}
                actions['baseBlob']['tierToArchive'] = {
                    'daysAfterModificationGreaterThan': tier_to_archive_after_days
                }
            
            if delete_after_days:
                if 'baseBlob' not in actions:
                    actions['baseBlob'] = {}
                actions['baseBlob']['delete'] = {
                    'daysAfterModificationGreaterThan': delete_after_days
                }
            
            # Create rule
            rule = ManagementPolicyRule(
                name=rule_name,
                enabled=True,
                type='Lifecycle',
                definition={
                    'actions': actions,
                    'filters': {
                        'blobTypes': ['blockBlob'],
                        'prefixMatch': [prefix_filter]
                    }
                }
            )
            
            # Get or create policy
            try:
                policy = self.blob_service_client.get_management_policy()
                rules = policy.rules
            except:
                rules = []
            
            # Add/update rule
            rules = [r for r in rules if r.name != rule_name]
            rules.append(rule)
            
            # Set policy
            policy = ManagementPolicy(rules=rules)
            self.blob_service_client.set_management_policy(policy)
            
            self.logger.info(f"Created lifecycle policy: {rule_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create lifecycle policy: {e}")
            raise StorageError(f"Lifecycle policy creation failed: {e}")


# Factory function
def create_azure_storage(config: Optional[Dict[str, Any]] = None) -> AzureStorage:
    """Create and return an Azure storage instance"""
    return AzureStorage(config)