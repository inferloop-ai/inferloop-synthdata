#!/usr/bin/env python3
"""
Batch exporter for processing large datasets efficiently.

Provides batch processing capabilities for exporting large numbers of documents
with progress tracking, error handling, and resource management.

Features:
- Batch processing with configurable batch sizes
- Progress tracking and reporting
- Error handling and retry mechanisms
- Memory-efficient processing
- Parallel processing support
- Export job management
- Resume capabilities for interrupted exports
"""

import asyncio
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from uuid import uuid4

from ...core import get_logger, get_config
from ...core.exceptions import ProcessingError, ValidationError
from ..storage import DatabaseStorage, CacheManager
from .format_exporters import create_format_exporter


logger = get_logger(__name__)
config = get_config()


class ExportJob:
    """Represents a batch export job"""
    
    def __init__(self, job_id: str, config: Dict[str, Any]):
        self.job_id = job_id
        self.config = config
        self.status = 'pending'
        self.progress = 0.0
        self.total_documents = 0
        self.processed_documents = 0
        self.failed_documents = 0
        self.start_time = None
        self.end_time = None
        self.error_message = None
        self.output_path = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary"""
        return {
            'job_id': self.job_id,
            'config': self.config,
            'status': self.status,
            'progress': self.progress,
            'total_documents': self.total_documents,
            'processed_documents': self.processed_documents,
            'failed_documents': self.failed_documents,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error_message': self.error_message,
            'output_path': self.output_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExportJob':
        """Create job from dictionary"""
        job = cls(data['job_id'], data['config'])
        job.status = data.get('status', 'pending')
        job.progress = data.get('progress', 0.0)
        job.total_documents = data.get('total_documents', 0)
        job.processed_documents = data.get('processed_documents', 0)
        job.failed_documents = data.get('failed_documents', 0)
        job.start_time = datetime.fromisoformat(data['start_time']) if data.get('start_time') else None
        job.end_time = datetime.fromisoformat(data['end_time']) if data.get('end_time') else None
        job.error_message = data.get('error_message')
        job.output_path = data.get('output_path')
        return job


class BatchExporter:
    """Main batch exporter class"""
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.storage = DatabaseStorage()
        self.cache = CacheManager()
        self.active_jobs = {}
        self.job_callbacks = {}
        
    async def create_export_job(self, document_ids: List[str], format_type: str, 
                              output_path: str, options: Optional[Dict[str, Any]] = None,
                              user_id: Optional[str] = None) -> str:
        """Create a new export job"""
        job_id = str(uuid4())
        
        config = {
            'document_ids': document_ids,
            'format_type': format_type,
            'output_path': output_path,
            'options': options or {},
            'user_id': user_id,
            'batch_size': self.batch_size,
            'max_workers': self.max_workers
        }
        
        job = ExportJob(job_id, config)
        job.total_documents = len(document_ids)
        
        # Store job in database
        await self.storage.store_export_job(job.to_dict())
        
        # Cache job for quick access
        self.active_jobs[job_id] = job
        
        logger.info(f"Created export job {job_id} for {len(document_ids)} documents")
        return job_id
    
    async def start_export_job(self, job_id: str, progress_callback: Optional[callable] = None) -> bool:
        """Start an export job"""
        try:
            # Get job from cache or database
            job = await self.get_job(job_id)
            if not job:
                logger.error(f"Export job {job_id} not found")
                return False
            
            if job.status != 'pending':
                logger.warning(f"Export job {job_id} is not in pending status: {job.status}")
                return False
            
            # Register progress callback
            if progress_callback:
                self.job_callbacks[job_id] = progress_callback
            
            # Start background processing
            asyncio.create_task(self._process_export_job(job))
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting export job {job_id}: {e}")
            return False
    
    async def get_job(self, job_id: str) -> Optional[ExportJob]:
        """Get export job by ID"""
        # Check cache first
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        # Load from database
        job_data = await self.storage.get_export_job(job_id)
        if job_data:
            job = ExportJob.from_dict(job_data)
            self.active_jobs[job_id] = job
            return job
        
        return None
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status information"""
        job = await self.get_job(job_id)
        if not job:
            return None
        
        return {
            'job_id': job_id,
            'status': job.status,
            'progress': job.progress,
            'total_documents': job.total_documents,
            'processed_documents': job.processed_documents,
            'failed_documents': job.failed_documents,
            'start_time': job.start_time.isoformat() if job.start_time else None,
            'end_time': job.end_time.isoformat() if job.end_time else None,
            'error_message': job.error_message,
            'output_path': job.output_path,
            'elapsed_time': (datetime.utcnow() - job.start_time).total_seconds() if job.start_time else None
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel an active export job"""
        job = await self.get_job(job_id)
        if not job:
            return False
        
        if job.status in ['completed', 'failed', 'cancelled']:
            return False
        
        job.status = 'cancelled'
        job.end_time = datetime.utcnow()
        
        # Update in database
        await self.storage.update_export_job(job_id, job.to_dict())
        
        # Notify callback
        await self._notify_progress(job_id, job)
        
        logger.info(f"Export job {job_id} cancelled")
        return True
    
    async def _process_export_job(self, job: ExportJob):
        """Process export job in background"""
        try:
            job.status = 'running'
            job.start_time = datetime.utcnow()
            
            # Update job status
            await self.storage.update_export_job(job.job_id, job.to_dict())
            await self._notify_progress(job.job_id, job)
            
            # Create output directory
            output_path = Path(job.config['output_path'])
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create format exporter
            exporter = create_format_exporter(
                job.config['format_type'],
                privacy_protection=job.config['options'].get('privacy_protection', True)
            )
            
            # Process documents in batches
            document_ids = job.config['document_ids']
            batch_size = job.config['batch_size']
            
            # Group documents into batches
            batches = [
                document_ids[i:i + batch_size] 
                for i in range(0, len(document_ids), batch_size)
            ]
            
            logger.info(f"Processing {len(document_ids)} documents in {len(batches)} batches")
            
            # Process batches
            if job.config.get('parallel_processing', True) and len(batches) > 1:
                await self._process_batches_parallel(job, batches, exporter, output_path)
            else:
                await self._process_batches_sequential(job, batches, exporter, output_path)
            
            # Finalize export
            if job.status != 'cancelled':
                await self._finalize_export(job, exporter, output_path)
            
        except Exception as e:
            logger.error(f"Error processing export job {job.job_id}: {e}")
            job.status = 'failed'
            job.error_message = str(e)
            job.end_time = datetime.utcnow()
            
            await self.storage.update_export_job(job.job_id, job.to_dict())
            await self._notify_progress(job.job_id, job)
        
        finally:
            # Clean up active job reference
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            if job.job_id in self.job_callbacks:
                del self.job_callbacks[job.job_id]
    
    async def _process_batches_sequential(self, job: ExportJob, batches: List[List[str]], 
                                        exporter, output_path: Path):
        """Process batches sequentially"""
        for batch_idx, batch_ids in enumerate(batches):
            if job.status == 'cancelled':
                break
                
            try:
                # Load documents for this batch
                documents = await self._load_documents(batch_ids)
                
                # Create batch output directory
                batch_output_path = output_path / f"batch_{batch_idx:04d}"
                
                # Export batch
                result = await exporter.export(
                    documents,
                    str(batch_output_path),
                    job.config['options']
                )
                
                # Update progress
                job.processed_documents += len(documents)
                job.progress = job.processed_documents / job.total_documents
                
                # Update job status
                await self.storage.update_export_job(job.job_id, job.to_dict())
                await self._notify_progress(job.job_id, job)
                
                logger.info(f"Processed batch {batch_idx + 1}/{len(batches)} for job {job.job_id}")
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx} for job {job.job_id}: {e}")
                job.failed_documents += len(batch_ids)
                
                # Continue with next batch instead of failing entire job
                continue
    
    async def _process_batches_parallel(self, job: ExportJob, batches: List[List[str]], 
                                      exporter, output_path: Path):
        """Process batches in parallel"""
        max_workers = min(job.config['max_workers'], len(batches))
        
        # Create semaphore to limit concurrent batches
        semaphore = asyncio.Semaphore(max_workers)
        
        async def process_batch_with_semaphore(batch_idx: int, batch_ids: List[str]):
            async with semaphore:
                return await self._process_single_batch(job, batch_idx, batch_ids, exporter, output_path)
        
        # Start all batch processing tasks
        tasks = [
            process_batch_with_semaphore(batch_idx, batch_ids)
            for batch_idx, batch_ids in enumerate(batches)
        ]
        
        # Wait for all batches to complete
        for task in asyncio.as_completed(tasks):
            if job.status == 'cancelled':
                # Cancel remaining tasks
                for remaining_task in tasks:
                    if not remaining_task.done():
                        remaining_task.cancel()
                break
                
            try:
                batch_result = await task
                
                # Update progress
                job.processed_documents += batch_result['processed_count']
                job.failed_documents += batch_result['failed_count']
                job.progress = job.processed_documents / job.total_documents
                
                # Update job status
                await self.storage.update_export_job(job.job_id, job.to_dict())
                await self._notify_progress(job.job_id, job)
                
            except Exception as e:
                logger.error(f"Error in parallel batch processing for job {job.job_id}: {e}")
    
    async def _process_single_batch(self, job: ExportJob, batch_idx: int, batch_ids: List[str], 
                                  exporter, output_path: Path) -> Dict[str, int]:
        """Process a single batch"""
        try:
            # Load documents for this batch
            documents = await self._load_documents(batch_ids)
            
            if not documents:
                return {'processed_count': 0, 'failed_count': len(batch_ids)}
            
            # Create batch output directory
            batch_output_path = output_path / f"batch_{batch_idx:04d}"
            
            # Export batch
            result = await exporter.export(
                documents,
                str(batch_output_path),
                job.config['options']
            )
            
            logger.info(f"Processed batch {batch_idx} for job {job.job_id}: {len(documents)} documents")
            return {'processed_count': len(documents), 'failed_count': 0}
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx} for job {job.job_id}: {e}")
            return {'processed_count': 0, 'failed_count': len(batch_ids)}
    
    async def _load_documents(self, document_ids: List[str]) -> List[Dict[str, Any]]:
        """Load documents from storage"""
        documents = []
        
        for doc_id in document_ids:
            try:
                doc = await self.storage.get_document(doc_id)
                if doc:
                    documents.append(doc)
                else:
                    logger.warning(f"Document {doc_id} not found")
            except Exception as e:
                logger.error(f"Error loading document {doc_id}: {e}")
        
        return documents
    
    async def _finalize_export(self, job: ExportJob, exporter, output_path: Path):
        """Finalize export job"""
        try:
            # Create manifest file
            manifest = {
                'job_id': job.job_id,
                'format_type': job.config['format_type'],
                'total_documents': job.total_documents,
                'processed_documents': job.processed_documents,
                'failed_documents': job.failed_documents,
                'export_time': job.start_time.isoformat() if job.start_time else None,
                'options': job.config['options']
            }
            
            manifest_path = output_path / 'export_manifest.json'
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Update job completion
            job.status = 'completed'
            job.end_time = datetime.utcnow()
            job.output_path = str(output_path)
            
            await self.storage.update_export_job(job.job_id, job.to_dict())
            await self._notify_progress(job.job_id, job)
            
            logger.info(f"Export job {job.job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error finalizing export job {job.job_id}: {e}")
            job.status = 'failed'
            job.error_message = f"Finalization failed: {str(e)}"
            job.end_time = datetime.utcnow()
            
            await self.storage.update_export_job(job.job_id, job.to_dict())
            await self._notify_progress(job.job_id, job)
    
    async def _notify_progress(self, job_id: str, job: ExportJob):
        """Notify progress callback if registered"""
        if job_id in self.job_callbacks:
            try:
                callback = self.job_callbacks[job_id]
                if asyncio.iscoroutinefunction(callback):
                    await callback(job.to_dict())
                else:
                    callback(job.to_dict())
            except Exception as e:
                logger.error(f"Error in progress callback for job {job_id}: {e}")
    
    async def list_jobs(self, user_id: Optional[str] = None, status: Optional[str] = None,
                       limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List export jobs with optional filtering"""
        try:
            jobs = await self.storage.list_export_jobs(
                user_id=user_id,
                status=status,
                limit=limit,
                offset=offset
            )
            return jobs
        except Exception as e:
            logger.error(f"Error listing export jobs: {e}")
            return []
    
    async def cleanup_completed_jobs(self, older_than_days: int = 7) -> int:
        """Clean up old completed jobs"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
            deleted_count = await self.storage.cleanup_export_jobs(cutoff_date)
            
            logger.info(f"Cleaned up {deleted_count} old export jobs")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up export jobs: {e}")
            return 0


# Factory function
def create_batch_exporter(batch_size: int = 100, max_workers: int = 4) -> BatchExporter:
    """Create batch exporter instance"""
    return BatchExporter(batch_size=batch_size, max_workers=max_workers)


__all__ = [
    'ExportJob',
    'BatchExporter',
    'create_batch_exporter'
]