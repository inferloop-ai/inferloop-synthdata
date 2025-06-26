"""
Chunked Transfer Manager for TextNLP
Handles efficient chunked upload/download of large model files with resume capability
"""

import os
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncIterator, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import logging
from tqdm.asyncio import tqdm
import boto3
from botocore.exceptions import ClientError
from azure.storage.blob.aio import BlobServiceClient
from google.cloud import storage as gcs
from google.api_core import retry

logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """Information about a file chunk"""
    chunk_id: int
    offset: int
    size: int
    checksum: str
    uploaded: bool = False
    upload_time: Optional[float] = None
    retry_count: int = 0


@dataclass
class TransferConfig:
    """Configuration for chunked transfers"""
    chunk_size_mb: int = 10
    parallel_chunks: int = 4
    max_retries: int = 3
    retry_delay: float = 1.0
    checksum_algorithm: str = "md5"
    resume_enabled: bool = True
    compression: bool = False
    encryption: bool = True


@dataclass
class TransferProgress:
    """Track transfer progress"""
    total_bytes: int
    transferred_bytes: int = 0
    chunks_total: int = 0
    chunks_completed: int = 0
    start_time: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)
    
    @property
    def progress_percent(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return (self.transferred_bytes / self.total_bytes) * 100
    
    @property
    def transfer_rate_mbps(self) -> float:
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return (self.transferred_bytes / 1024 / 1024) / elapsed
    
    @property
    def eta_seconds(self) -> float:
        if self.transfer_rate_mbps == 0:
            return float('inf')
        remaining_mb = (self.total_bytes - self.transferred_bytes) / 1024 / 1024
        return remaining_mb / self.transfer_rate_mbps


class ChunkedTransferManager:
    """Manages chunked file transfers with resume capability"""
    
    def __init__(self, config: Optional[TransferConfig] = None):
        self.config = config or TransferConfig()
        self.chunk_size = self.config.chunk_size_mb * 1024 * 1024
        self.executor = ThreadPoolExecutor(max_workers=self.config.parallel_chunks)
        self._transfers: Dict[str, TransferProgress] = {}
        
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate checksum of data"""
        if self.config.checksum_algorithm == "md5":
            return hashlib.md5(data).hexdigest()
        elif self.config.checksum_algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        else:
            raise ValueError(f"Unsupported checksum algorithm: {self.config.checksum_algorithm}")
    
    def _get_resume_info_path(self, file_path: str) -> str:
        """Get path for resume information file"""
        return f"{file_path}.transfer_resume"
    
    async def _save_resume_info(self, file_path: str, chunks: List[ChunkInfo]):
        """Save resume information"""
        if not self.config.resume_enabled:
            return
            
        resume_info = {
            "file_path": file_path,
            "chunk_size": self.chunk_size,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "offset": c.offset,
                    "size": c.size,
                    "checksum": c.checksum,
                    "uploaded": c.uploaded
                }
                for c in chunks
            ],
            "timestamp": time.time()
        }
        
        resume_path = self._get_resume_info_path(file_path)
        async with aiofiles.open(resume_path, 'w') as f:
            await f.write(json.dumps(resume_info, indent=2))
    
    async def _load_resume_info(self, file_path: str) -> Optional[List[ChunkInfo]]:
        """Load resume information if available"""
        if not self.config.resume_enabled:
            return None
            
        resume_path = self._get_resume_info_path(file_path)
        if not os.path.exists(resume_path):
            return None
            
        try:
            async with aiofiles.open(resume_path, 'r') as f:
                resume_info = json.loads(await f.read())
            
            # Verify file hasn't changed
            if os.path.getmtime(file_path) > resume_info['timestamp']:
                logger.warning("File has been modified since last transfer, ignoring resume info")
                return None
            
            chunks = [
                ChunkInfo(
                    chunk_id=c['chunk_id'],
                    offset=c['offset'],
                    size=c['size'],
                    checksum=c['checksum'],
                    uploaded=c['uploaded']
                )
                for c in resume_info['chunks']
            ]
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load resume info: {e}")
            return None
    
    async def _prepare_chunks(self, file_path: str) -> List[ChunkInfo]:
        """Prepare chunk information for a file"""
        file_size = os.path.getsize(file_path)
        chunks = []
        
        # Check for resume information
        resume_chunks = await self._load_resume_info(file_path)
        if resume_chunks:
            logger.info(f"Resuming transfer with {len(resume_chunks)} chunks")
            return resume_chunks
        
        # Calculate chunks
        num_chunks = (file_size + self.chunk_size - 1) // self.chunk_size
        
        async with aiofiles.open(file_path, 'rb') as f:
            for i in range(num_chunks):
                offset = i * self.chunk_size
                size = min(self.chunk_size, file_size - offset)
                
                # Read chunk to calculate checksum
                await f.seek(offset)
                data = await f.read(size)
                checksum = self._calculate_checksum(data)
                
                chunks.append(ChunkInfo(
                    chunk_id=i,
                    offset=offset,
                    size=size,
                    checksum=checksum
                ))
        
        return chunks
    
    async def upload_file_chunked(self, local_path: str, remote_path: str,
                                 storage_backend: Any,
                                 progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Upload a file in chunks with resume capability"""
        logger.info(f"Starting chunked upload: {local_path} -> {remote_path}")
        
        # Prepare chunks
        chunks = await self._prepare_chunks(local_path)
        file_size = os.path.getsize(local_path)
        
        # Initialize progress
        progress = TransferProgress(
            total_bytes=file_size,
            chunks_total=len(chunks)
        )
        self._transfers[local_path] = progress
        
        # Count already uploaded chunks
        uploaded_chunks = [c for c in chunks if c.uploaded]
        progress.chunks_completed = len(uploaded_chunks)
        progress.transferred_bytes = sum(c.size for c in uploaded_chunks)
        
        # Upload remaining chunks
        remaining_chunks = [c for c in chunks if not c.uploaded]
        
        if remaining_chunks:
            # Create upload tasks
            semaphore = asyncio.Semaphore(self.config.parallel_chunks)
            
            async def upload_chunk(chunk: ChunkInfo):
                async with semaphore:
                    return await self._upload_chunk(
                        local_path, remote_path, chunk, storage_backend, progress, progress_callback
                    )
            
            # Upload chunks concurrently
            tasks = [upload_chunk(chunk) for chunk in remaining_chunks]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for errors
            errors = [str(r) for r in results if isinstance(r, Exception)]
            if errors:
                progress.errors.extend(errors)
                logger.error(f"Upload failed with {len(errors)} errors")
                raise Exception(f"Upload failed: {errors[0]}")
        
        # Clean up resume file on success
        resume_path = self._get_resume_info_path(local_path)
        if os.path.exists(resume_path):
            os.remove(resume_path)
        
        # Final progress callback
        if progress_callback:
            progress_callback(progress)
        
        return {
            "file_path": local_path,
            "remote_path": remote_path,
            "total_bytes": file_size,
            "chunks": len(chunks),
            "transfer_time": time.time() - progress.start_time,
            "average_speed_mbps": progress.transfer_rate_mbps
        }
    
    async def _upload_chunk(self, local_path: str, remote_path: str,
                          chunk: ChunkInfo, storage_backend: Any,
                          progress: TransferProgress,
                          progress_callback: Optional[Callable]) -> None:
        """Upload a single chunk with retry logic"""
        retry_count = 0
        last_error = None
        
        while retry_count <= self.config.max_retries:
            try:
                # Read chunk data
                async with aiofiles.open(local_path, 'rb') as f:
                    await f.seek(chunk.offset)
                    data = await f.read(chunk.size)
                
                # Verify checksum
                if self._calculate_checksum(data) != chunk.checksum:
                    raise ValueError(f"Checksum mismatch for chunk {chunk.chunk_id}")
                
                # Upload chunk
                chunk_remote_path = f"{remote_path}.chunk_{chunk.chunk_id:06d}"
                await storage_backend.upload_chunk(chunk_remote_path, data, chunk.offset)
                
                # Update progress
                chunk.uploaded = True
                chunk.upload_time = time.time()
                progress.chunks_completed += 1
                progress.transferred_bytes += chunk.size
                
                # Save resume info
                await self._save_resume_info(local_path, 
                    [c for c in await self._prepare_chunks(local_path)])
                
                # Progress callback
                if progress_callback:
                    progress_callback(progress)
                
                return
                
            except Exception as e:
                last_error = e
                retry_count += 1
                chunk.retry_count = retry_count
                
                if retry_count <= self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * retry_count)
                    logger.warning(f"Retrying chunk {chunk.chunk_id} (attempt {retry_count})")
        
        raise Exception(f"Failed to upload chunk {chunk.chunk_id} after {retry_count} attempts: {last_error}")
    
    async def download_file_chunked(self, remote_path: str, local_path: str,
                                   storage_backend: Any,
                                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Download a file in chunks with resume capability"""
        logger.info(f"Starting chunked download: {remote_path} -> {local_path}")
        
        # Get file metadata
        file_info = await storage_backend.get_file_info(remote_path)
        file_size = file_info['size']
        
        # Calculate chunks
        num_chunks = (file_size + self.chunk_size - 1) // self.chunk_size
        
        # Check for partial download
        partial_path = f"{local_path}.partial"
        start_chunk = 0
        
        if os.path.exists(partial_path) and self.config.resume_enabled:
            existing_size = os.path.getsize(partial_path)
            start_chunk = existing_size // self.chunk_size
            logger.info(f"Resuming download from chunk {start_chunk}")
        
        # Initialize progress
        progress = TransferProgress(
            total_bytes=file_size,
            chunks_total=num_chunks,
            chunks_completed=start_chunk,
            transferred_bytes=start_chunk * self.chunk_size
        )
        self._transfers[remote_path] = progress
        
        # Download chunks
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        async with aiofiles.open(partial_path, 'ab' if start_chunk > 0 else 'wb') as f:
            # Create download tasks
            semaphore = asyncio.Semaphore(self.config.parallel_chunks)
            
            async def download_chunk(chunk_id: int):
                async with semaphore:
                    offset = chunk_id * self.chunk_size
                    size = min(self.chunk_size, file_size - offset)
                    
                    return await self._download_chunk(
                        remote_path, chunk_id, offset, size,
                        storage_backend, progress, progress_callback
                    )
            
            # Download chunks in order
            for chunk_id in range(start_chunk, num_chunks):
                chunk_data = await download_chunk(chunk_id)
                await f.write(chunk_data)
        
        # Rename to final path
        os.rename(partial_path, local_path)
        
        # Final progress callback
        if progress_callback:
            progress_callback(progress)
        
        return {
            "file_path": local_path,
            "remote_path": remote_path,
            "total_bytes": file_size,
            "chunks": num_chunks,
            "transfer_time": time.time() - progress.start_time,
            "average_speed_mbps": progress.transfer_rate_mbps
        }
    
    async def _download_chunk(self, remote_path: str, chunk_id: int,
                            offset: int, size: int,
                            storage_backend: Any,
                            progress: TransferProgress,
                            progress_callback: Optional[Callable]) -> bytes:
        """Download a single chunk with retry logic"""
        retry_count = 0
        last_error = None
        
        while retry_count <= self.config.max_retries:
            try:
                # Download chunk
                chunk_remote_path = f"{remote_path}.chunk_{chunk_id:06d}"
                data = await storage_backend.download_chunk(chunk_remote_path, offset, size)
                
                # Update progress
                progress.chunks_completed += 1
                progress.transferred_bytes += len(data)
                
                # Progress callback
                if progress_callback:
                    progress_callback(progress)
                
                return data
                
            except Exception as e:
                last_error = e
                retry_count += 1
                
                if retry_count <= self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * retry_count)
                    logger.warning(f"Retrying chunk {chunk_id} download (attempt {retry_count})")
        
        raise Exception(f"Failed to download chunk {chunk_id} after {retry_count} attempts: {last_error}")
    
    async def stream_upload(self, data_stream: AsyncIterator[bytes],
                          remote_path: str, storage_backend: Any,
                          total_size: Optional[int] = None) -> Dict[str, Any]:
        """Stream upload data in chunks"""
        logger.info(f"Starting stream upload to {remote_path}")
        
        progress = TransferProgress(total_bytes=total_size or 0)
        chunk_id = 0
        
        async for chunk_data in data_stream:
            chunk_path = f"{remote_path}.chunk_{chunk_id:06d}"
            
            # Upload chunk
            await storage_backend.upload_chunk(chunk_path, chunk_data, chunk_id * self.chunk_size)
            
            # Update progress
            progress.transferred_bytes += len(chunk_data)
            progress.chunks_completed += 1
            chunk_id += 1
        
        return {
            "remote_path": remote_path,
            "total_bytes": progress.transferred_bytes,
            "chunks": chunk_id,
            "transfer_time": time.time() - progress.start_time
        }


class S3ChunkedBackend:
    """AWS S3 backend for chunked transfers"""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=region)
    
    async def upload_chunk(self, key: str, data: bytes, offset: int):
        """Upload a chunk to S3"""
        # For multipart upload
        await asyncio.get_event_loop().run_in_executor(
            None,
            self.s3_client.put_object,
            self.bucket_name,
            key,
            data
        )
    
    async def download_chunk(self, key: str, offset: int, size: int) -> bytes:
        """Download a chunk from S3"""
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=key,
                Range=f'bytes={offset}-{offset + size - 1}'
            )
        )
        return response['Body'].read()
    
    async def get_file_info(self, key: str) -> Dict[str, Any]:
        """Get file information from S3"""
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            self.s3_client.head_object,
            self.bucket_name,
            key
        )
        return {
            'size': response['ContentLength'],
            'etag': response['ETag'],
            'last_modified': response['LastModified']
        }
    
    async def initiate_multipart_upload(self, key: str) -> str:
        """Initiate multipart upload"""
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            self.s3_client.create_multipart_upload,
            self.bucket_name,
            key
        )
        return response['UploadId']
    
    async def upload_part(self, key: str, upload_id: str, part_number: int, data: bytes) -> str:
        """Upload a part in multipart upload"""
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.s3_client.upload_part(
                Bucket=self.bucket_name,
                Key=key,
                UploadId=upload_id,
                PartNumber=part_number,
                Body=data
            )
        )
        return response['ETag']
    
    async def complete_multipart_upload(self, key: str, upload_id: str, parts: List[Dict]):
        """Complete multipart upload"""
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.s3_client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
        )


class AzureChunkedBackend:
    """Azure Blob Storage backend for chunked transfers"""
    
    def __init__(self, account_name: str, container_name: str, credential: Any):
        self.container_name = container_name
        self.blob_service = BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            credential=credential
        )
    
    async def upload_chunk(self, blob_name: str, data: bytes, offset: int):
        """Upload a chunk to Azure Blob"""
        async with self.blob_service:
            container_client = self.blob_service.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(blob_name)
            
            # For block blob
            block_id = f"{offset:010d}".encode('utf-8').hex()
            await blob_client.stage_block(block_id, data)
    
    async def download_chunk(self, blob_name: str, offset: int, size: int) -> bytes:
        """Download a chunk from Azure Blob"""
        async with self.blob_service:
            container_client = self.blob_service.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(blob_name)
            
            download_stream = await blob_client.download_blob(
                offset=offset,
                length=size
            )
            return await download_stream.readall()
    
    async def get_file_info(self, blob_name: str) -> Dict[str, Any]:
        """Get blob information"""
        async with self.blob_service:
            container_client = self.blob_service.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(blob_name)
            
            properties = await blob_client.get_blob_properties()
            return {
                'size': properties.size,
                'etag': properties.etag,
                'last_modified': properties.last_modified
            }


class GCSChunkedBackend:
    """Google Cloud Storage backend for chunked transfers"""
    
    def __init__(self, bucket_name: str, project_id: str):
        self.bucket_name = bucket_name
        self.client = gcs.Client(project=project_id)
        self.bucket = self.client.bucket(bucket_name)
    
    async def upload_chunk(self, blob_name: str, data: bytes, offset: int):
        """Upload a chunk to GCS"""
        blob = self.bucket.blob(blob_name)
        
        # For resumable upload
        await asyncio.get_event_loop().run_in_executor(
            None,
            blob.upload_from_string,
            data,
            content_type='application/octet-stream'
        )
    
    async def download_chunk(self, blob_name: str, offset: int, size: int) -> bytes:
        """Download a chunk from GCS"""
        blob = self.bucket.blob(blob_name)
        
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: blob.download_as_bytes(start=offset, end=offset + size - 1)
        )
    
    async def get_file_info(self, blob_name: str) -> Dict[str, Any]:
        """Get blob information"""
        blob = self.bucket.blob(blob_name)
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            blob.reload
        )
        
        return {
            'size': blob.size,
            'etag': blob.etag,
            'last_modified': blob.updated
        }


# Example usage
if __name__ == "__main__":
    async def example_upload():
        # Configure transfer
        config = TransferConfig(
            chunk_size_mb=50,
            parallel_chunks=4,
            resume_enabled=True
        )
        
        manager = ChunkedTransferManager(config)
        
        # Setup storage backend
        backend = S3ChunkedBackend(
            bucket_name="textnlp-models",
            region="us-east-1"
        )
        
        # Progress callback
        def progress_callback(progress: TransferProgress):
            print(f"Progress: {progress.progress_percent:.1f}% "
                  f"({progress.chunks_completed}/{progress.chunks_total} chunks) "
                  f"Speed: {progress.transfer_rate_mbps:.1f} MB/s")
        
        # Upload large file
        result = await manager.upload_file_chunked(
            local_path="/path/to/large_model.bin",
            remote_path="models/llama-13b/model.bin",
            storage_backend=backend,
            progress_callback=progress_callback
        )
        
        print(f"Upload complete: {result}")
    
    # Run example
    # asyncio.run(example_upload())