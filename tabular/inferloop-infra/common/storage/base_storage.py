"""Base storage abstraction interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, List, Optional, Any, Union, Generator
from datetime import datetime
from pathlib import Path
import pandas as pd
import io
import json


@dataclass
class StorageMetadata:
    """Storage object metadata."""
    
    size: int
    content_type: str
    etag: Optional[str] = None
    last_modified: Optional[datetime] = None
    storage_class: Optional[str] = None
    encryption: Optional[str] = None
    custom_metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class StorageObject:
    """Storage object information."""
    
    bucket: str
    key: str
    metadata: StorageMetadata
    version_id: Optional[str] = None
    is_delete_marker: bool = False


class BaseStorage(ABC):
    """Abstract storage interface for cloud and on-premise storage systems."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize storage provider."""
        self.config = config
    
    @abstractmethod
    def create_bucket(
        self,
        bucket_name: str,
        region: Optional[str] = None,
        versioning: bool = False,
        encryption: bool = True,
        lifecycle_rules: Optional[List[Dict[str, Any]]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Create a storage bucket."""
        pass
    
    @abstractmethod
    def delete_bucket(self, bucket_name: str, force: bool = False) -> bool:
        """Delete a storage bucket."""
        pass
    
    @abstractmethod
    def list_buckets(self) -> List[Dict[str, Any]]:
        """List all buckets."""
        pass
    
    @abstractmethod
    def bucket_exists(self, bucket_name: str) -> bool:
        """Check if a bucket exists."""
        pass
    
    @abstractmethod
    def upload_file(
        self,
        file_path: Union[str, Path],
        bucket: str,
        key: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
        encryption: Optional[str] = None,
    ) -> StorageObject:
        """Upload a file to storage."""
        pass
    
    @abstractmethod
    def upload_fileobj(
        self,
        file_obj: BinaryIO,
        bucket: str,
        key: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
        encryption: Optional[str] = None,
    ) -> StorageObject:
        """Upload a file object to storage."""
        pass
    
    @abstractmethod
    def download_file(
        self,
        bucket: str,
        key: str,
        file_path: Union[str, Path],
        version_id: Optional[str] = None,
    ) -> bool:
        """Download a file from storage."""
        pass
    
    @abstractmethod
    def download_fileobj(
        self,
        bucket: str,
        key: str,
        file_obj: BinaryIO,
        version_id: Optional[str] = None,
    ) -> bool:
        """Download to a file object from storage."""
        pass
    
    @abstractmethod
    def get_object(
        self, bucket: str, key: str, version_id: Optional[str] = None
    ) -> bytes:
        """Get object content as bytes."""
        pass
    
    @abstractmethod
    def put_object(
        self,
        bucket: str,
        key: str,
        data: bytes,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
        encryption: Optional[str] = None,
    ) -> StorageObject:
        """Put object data directly."""
        pass
    
    @abstractmethod
    def delete_object(
        self, bucket: str, key: str, version_id: Optional[str] = None
    ) -> bool:
        """Delete an object."""
        pass
    
    @abstractmethod
    def delete_objects(
        self, bucket: str, keys: List[str]
    ) -> Dict[str, Union[bool, str]]:
        """Delete multiple objects."""
        pass
    
    @abstractmethod
    def list_objects(
        self,
        bucket: str,
        prefix: Optional[str] = None,
        delimiter: Optional[str] = None,
        max_keys: int = 1000,
        continuation_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List objects in a bucket."""
        pass
    
    @abstractmethod
    def get_object_metadata(
        self, bucket: str, key: str, version_id: Optional[str] = None
    ) -> StorageMetadata:
        """Get object metadata."""
        pass
    
    @abstractmethod
    def copy_object(
        self,
        source_bucket: str,
        source_key: str,
        dest_bucket: str,
        dest_key: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObject:
        """Copy an object."""
        pass
    
    @abstractmethod
    def generate_presigned_url(
        self,
        bucket: str,
        key: str,
        operation: str = "get_object",
        expires_in: int = 3600,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a presigned URL."""
        pass
    
    @abstractmethod
    def enable_versioning(self, bucket: str) -> bool:
        """Enable versioning on a bucket."""
        pass
    
    @abstractmethod
    def set_lifecycle_policy(
        self, bucket: str, rules: List[Dict[str, Any]]
    ) -> bool:
        """Set lifecycle policy on a bucket."""
        pass
    
    def upload_dataframe(
        self,
        df: pd.DataFrame,
        bucket: str,
        key: str,
        format: str = "parquet",
        compression: Optional[str] = "snappy",
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageObject:
        """Upload a pandas DataFrame."""
        buffer = io.BytesIO()
        
        if format == "parquet":
            df.to_parquet(buffer, compression=compression, index=False)
            content_type = "application/octet-stream"
        elif format == "csv":
            compression_ext = f".{compression}" if compression else ""
            df.to_csv(buffer, index=False, compression=compression)
            content_type = "text/csv"
        elif format == "json":
            df.to_json(buffer, orient="records", lines=True)
            content_type = "application/json"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        buffer.seek(0)
        
        # Add DataFrame metadata
        df_metadata = {
            "rows": str(len(df)),
            "columns": str(len(df.columns)),
            "format": format,
            "compression": compression or "none",
        }
        
        if metadata:
            df_metadata.update(metadata)
        
        return self.upload_fileobj(
            buffer,
            bucket,
            key,
            metadata=df_metadata,
            content_type=content_type,
        )
    
    def download_dataframe(
        self,
        bucket: str,
        key: str,
        format: Optional[str] = None,
        version_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Download a pandas DataFrame."""
        # Get object metadata to determine format
        if not format:
            metadata = self.get_object_metadata(bucket, key, version_id)
            format = metadata.custom_metadata.get("format", "parquet")
        
        # Download object
        data = self.get_object(bucket, key, version_id)
        buffer = io.BytesIO(data)
        
        # Read DataFrame based on format
        if format == "parquet":
            return pd.read_parquet(buffer)
        elif format == "csv":
            return pd.read_csv(buffer)
        elif format == "json":
            return pd.read_json(buffer, orient="records", lines=True)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def upload_json(
        self,
        data: Union[Dict, List],
        bucket: str,
        key: str,
        metadata: Optional[Dict[str, str]] = None,
        indent: int = 2,
    ) -> StorageObject:
        """Upload JSON data."""
        json_bytes = json.dumps(data, indent=indent).encode("utf-8")
        
        return self.put_object(
            bucket,
            key,
            json_bytes,
            metadata=metadata,
            content_type="application/json",
        )
    
    def download_json(
        self, bucket: str, key: str, version_id: Optional[str] = None
    ) -> Union[Dict, List]:
        """Download JSON data."""
        data = self.get_object(bucket, key, version_id)
        return json.loads(data.decode("utf-8"))
    
    def list_all_objects(
        self, bucket: str, prefix: Optional[str] = None
    ) -> Generator[StorageObject, None, None]:
        """List all objects with pagination."""
        continuation_token = None
        
        while True:
            response = self.list_objects(
                bucket,
                prefix=prefix,
                continuation_token=continuation_token,
            )
            
            for obj in response.get("objects", []):
                yield obj
            
            continuation_token = response.get("next_continuation_token")
            if not continuation_token:
                break
    
    def sync_directory(
        self,
        local_path: Union[str, Path],
        bucket: str,
        prefix: str = "",
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[StorageObject]:
        """Sync a local directory to storage."""
        local_path = Path(local_path)
        uploaded = []
        
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                # Check exclude patterns
                if exclude_patterns:
                    skip = any(
                        pattern in str(file_path) for pattern in exclude_patterns
                    )
                    if skip:
                        continue
                
                # Calculate relative path for key
                relative_path = file_path.relative_to(local_path)
                key = f"{prefix}/{relative_path}".replace("\\", "/").lstrip("/")
                
                # Upload file
                obj = self.upload_file(file_path, bucket, key)
                uploaded.append(obj)
        
        return uploaded
    
    def get_bucket_size(self, bucket: str, prefix: Optional[str] = None) -> int:
        """Get total size of objects in a bucket."""
        total_size = 0
        
        for obj in self.list_all_objects(bucket, prefix):
            total_size += obj.metadata.size
        
        return total_size
    
    def empty_bucket(self, bucket: str, prefix: Optional[str] = None) -> int:
        """Empty a bucket by deleting all objects."""
        objects_to_delete = []
        
        for obj in self.list_all_objects(bucket, prefix):
            objects_to_delete.append(obj.key)
            
            # Delete in batches of 1000
            if len(objects_to_delete) >= 1000:
                self.delete_objects(bucket, objects_to_delete)
                objects_to_delete = []
        
        # Delete remaining objects
        if objects_to_delete:
            self.delete_objects(bucket, objects_to_delete)
        
        return len(objects_to_delete)