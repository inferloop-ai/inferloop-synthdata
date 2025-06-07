import boto3
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable
import logging
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

logger = logging.getLogger(__name__)

class S3Exporter:
    """Export synthetic image datasets to AWS S3."""
    
    def __init__(self, 
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 region_name: str = 'us-east-1'):
        
        # Initialize S3 client
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=region_name
        )
        
        self.s3_client = session.client('s3')
        self.s3_resource = session.resource('s3')
        
        self.uploaded_count = 0
        self.failed_count = 0
        self.total_size = 0
        
        # Thread-safe counters
        self._lock = threading.Lock()
    
    def export_dataset(self,
                      dataset_dir: str,
                      bucket_name: str,
                      s3_prefix: str = "",
                      max_workers: int = 10,
                      include_metadata: bool = True,
                      public_read: bool = False,
                      storage_class: str = 'STANDARD',
                      progress_callback: Optional[Callable] = None) -> Dict:
        """Export a complete dataset to S3."""
        
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
        
        # Verify bucket exists
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
        except Exception as e:
            raise ValueError(f"Cannot access S3 bucket '{bucket_name}': {e}")
        
        # Get all files to upload
        files_to_upload = []
        
        # Image files
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']:
            files_to_upload.extend(dataset_path.glob(ext))
        
        # Metadata files
        for ext in ['*.json', '*.txt', '*.csv']:
            files_to_upload.extend(dataset_path.glob(f"**/{ext}"))
        
        if not files_to_upload:
            raise ValueError(f"No files found in {dataset_dir}")
        
        logger.info(f"Uploading {len(files_to_upload)} files to s3://{bucket_name}/{s3_prefix}")
        
        # Upload files concurrently
        upload_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    self._upload_file,
                    file_path,
                    bucket_name,
                    s3_prefix,
                    dataset_path,
                    public_read,
                    storage_class
                ): file_path for file_path in files_to_upload
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    upload_results.append(result)
                    
                    with self._lock:
                        if result['success']:
                            self.uploaded_count += 1
                            self.total_size += result['file_size']
                        else:
                            self.failed_count += 1
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(self.uploaded_count, self.failed_count, len(files_to_upload))
                    
                    # Log progress
                    if self.uploaded_count % 100 == 0:
                        logger.info(f"Uploaded {self.uploaded_count}/{len(files_to_upload)} files")
                        
                except Exception as e:
                    logger.error(f"Upload failed for {file_path}: {e}")
                    with self._lock:
                        self.failed_count += 1
        
        # Create dataset manifest
        manifest = self._create_dataset_manifest(
            upload_results, 
            bucket_name, 
            s3_prefix,
            dataset_path
        )
        
        # Upload manifest
        manifest_key = f"{s3_prefix}/dataset_manifest.json" if s3_prefix else "dataset_manifest.json"
        
        try:
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=manifest_key,
                Body=json.dumps(manifest, indent=2),
                ContentType='application/json'
            )
            logger.info(f"Dataset manifest uploaded to s3://{bucket_name}/{manifest_key}")
        except Exception as e:
            logger.error(f"Failed to upload manifest: {e}")
        
        # Create summary
        summary = {
            'export_timestamp': datetime.now().isoformat(),
            'bucket_name': bucket_name,
            's3_prefix': s3_prefix,
            'dataset_directory': str(dataset_path),
            'total_files': len(files_to_upload),
            'uploaded_files': self.uploaded_count,
            'failed_files': self.failed_count,
            'total_size_mb': self.total_size / (1024 * 1024),
            'manifest_key': manifest_key,
            'storage_class': storage_class,
            'public_read': public_read,
            'success_rate': (self.uploaded_count / len(files_to_upload)) * 100 if files_to_upload else 0
        }
        
        logger.info(f"S3 export complete: {self.uploaded_count} files uploaded, {self.failed_count} failed")
        
        return summary
    
    def _upload_file(self,
                    file_path: Path,
                    bucket_name: str,
                    s3_prefix: str,
                    dataset_path: Path,
                    public_read: bool,
                    storage_class: str) -> Dict:
        """Upload a single file to S3."""
        
        try:
            # Calculate S3 key
            relative_path = file_path.relative_to(dataset_path)
            s3_key = f"{s3_prefix}/{relative_path}" if s3_prefix else str(relative_path)
            s3_key = s3_key.replace("\\", "/")  # Ensure forward slashes
            
            # Determine content type
            content_type = self._get_content_type(file_path)
            
            # Upload parameters
            upload_params = {
                'Bucket': bucket_name,
                'Key': s3_key,
                'Body': file_path.read_bytes(),
                'ContentType': content_type,
                'StorageClass': storage_class
            }
            
            # Set ACL if public read
            if public_read:
                upload_params['ACL'] = 'public-read'
            
            # Upload file
            self.s3_client.put_object(**upload_params)
            
            file_size = file_path.stat().st_size
            
            return {
                'success': True,
                'local_path': str(file_path),
                's3_key': s3_key,
                's3_url': f"s3://{bucket_name}/{s3_key}",
                'file_size': file_size,
                'content_type': content_type
            }
            
        except Exception as e:
            return {
                'success': False,
                'local_path': str(file_path),
                'error': str(e),
                'file_size': 0
            }
    
    def _get_content_type(self, file_path: Path) -> str:
        """Determine content type based on file extension."""
        
        extension = file_path.suffix.lower()
        
        content_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.tiff': 'image/tiff',
            '.bmp': 'image/bmp',
            '.json': 'application/json',
            '.txt': 'text/plain',
            '.csv': 'text/csv',
            '.parquet': 'application/octet-stream',
            '.jsonl': 'application/jsonlines'
        }
        
        return content_types.get(extension, 'application/octet-stream')
    
    def _create_dataset_manifest(self,
                               upload_results: List[Dict],
                               bucket_name: str,
                               s3_prefix: str,
                               dataset_path: Path) -> Dict:
        """Create a dataset manifest with all uploaded files."""
        
        successful_uploads = [r for r in upload_results if r['success']]
        
        # Organize by file type
        file_types = {}
        for result in successful_uploads:
            extension = Path(result['local_path']).suffix.lower()
            if extension not in file_types:
                file_types[extension] = []
            file_types[extension].append(result)
        
        manifest = {
            'dataset_info': {
                'name': dataset_path.name,
                'created_at': datetime.now().isoformat(),
                'bucket_name': bucket_name,
                's3_prefix': s3_prefix,
                'total_files': len(successful_uploads),
                'total_size_bytes': sum(r['file_size'] for r in successful_uploads)
            },
            'file_types': {
                ext: {
                    'count': len(files),
                    'total_size_bytes': sum(f['file_size'] for f in files),
                    'files': files
                } for ext, files in file_types.items()
            },
            'access_info': {
                'base_url': f"s3://{bucket_name}/{s3_prefix}" if s3_prefix else f"s3://{bucket_name}",
                'region': self.s3_client.meta.region_name
            }
        }
        
        return manifest
    
    def download_dataset(self,
                        bucket_name: str,
                        s3_prefix: str,
                        local_dir: str,
                        max_workers: int = 10) -> Dict:
        """Download a dataset from S3."""
        
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        # List all objects with the prefix
        objects = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
            if 'Contents' in page:
                objects.extend(page['Contents'])
        
        if not objects:
            raise ValueError(f"No objects found in s3://{bucket_name}/{s3_prefix}")
        
        logger.info(f"Downloading {len(objects)} objects from S3...")
        
        downloaded_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_obj = {
                executor.submit(
                    self._download_file,
                    bucket_name,
                    obj['Key'],
                    local_path,
                    s3_prefix
                ): obj for obj in objects
            }
            
            for future in as_completed(future_to_obj):
                try:
                    result = future.result()
                    if result['success']:
                        downloaded_count += 1
                    else:
                        failed_count += 1
                        
                    if downloaded_count % 100 == 0:
                        logger.info(f"Downloaded {downloaded_count}/{len(objects)} files")
                        
                except Exception as e:
                    logger.error(f"Download failed: {e}")
                    failed_count += 1
        
        summary = {
            'download_timestamp': datetime.now().isoformat(),
            'bucket_name': bucket_name,
            's3_prefix': s3_prefix,
            'local_directory': str(local_path),
            'total_objects': len(objects),
            'downloaded_files': downloaded_count,
            'failed_files': failed_count,
            'success_rate': (downloaded_count / len(objects)) * 100 if objects else 0
        }
        
        logger.info(f"S3 download complete: {downloaded_count} files downloaded, {failed_count} failed")
        
        return summary
    
    def _download_file(self,
                      bucket_name: str,
                      s3_key: str,
                      local_dir: Path,
                      s3_prefix: str) -> Dict:
        """Download a single file from S3."""
        
        try:
            # Calculate local file path
            relative_key = s3_key[len(s3_prefix):].lstrip('/')
            local_file = local_dir / relative_key
            
            # Create parent directories
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            self.s3_client.download_file(bucket_name, s3_key, str(local_file))
            
            return {
                'success': True,
                's3_key': s3_key,
                'local_path': str(local_file)
            }
            
        except Exception as e:
            return {
                'success': False,
                's3_key': s3_key,
                'error': str(e)
            }

def export_to_s3(dataset_dir: str,
                bucket_name: str,
                s3_prefix: str = "",
                **kwargs) -> Dict:
    """Convenience function for S3 export."""
    
    exporter = S3Exporter()
    return exporter.export_dataset(dataset_dir, bucket_name, s3_prefix, **kwargs)

if __name__ == "__main__":
    # Example usage
    try:
        def progress_callback(uploaded, failed, total):
            print(f"Progress: {uploaded}/{total} uploaded, {failed} failed")
        
        result = export_to_s3(
            dataset_dir="./data/generated/test_dataset",
            bucket_name="my-synthetic-data-bucket",
            s3_prefix="datasets/test_v1",
            max_workers=5,
            progress_callback=progress_callback
        )
        print(f"Export completed: {result}")
    except Exception as e:
        print(f"Export failed: {e}")
