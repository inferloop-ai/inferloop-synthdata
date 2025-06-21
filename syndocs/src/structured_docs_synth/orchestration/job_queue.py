#!/usr/bin/env python3
"""
Job Queue implementation for managing document generation tasks.

Provides a priority-based queue system for handling document generation jobs
with support for job scheduling, retries, and status tracking.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from queue import PriorityQueue
import threading
from concurrent.futures import ThreadPoolExecutor, Future

from pydantic import BaseModel, Field

from ..core.config import get_config
from ..core.logging import get_logger
from ..core.exceptions import ProcessingError, ValidationError


logger = get_logger(__name__)


class JobStatus(Enum):
    """Job execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(Enum):
    """Job priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class JobMetadata:
    """Job metadata information"""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    result: Optional[Any] = None
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class Job:
    """Individual job representation"""
    job_id: str
    name: str
    job_type: str
    payload: Dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    metadata: JobMetadata = field(default_factory=JobMetadata)
    callback: Optional[Callable] = None
    timeout: Optional[float] = None
    
    def __lt__(self, other):
        """Compare jobs based on priority and creation time"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.metadata.created_at < other.metadata.created_at


class JobQueueConfig(BaseModel):
    """Job queue configuration"""
    max_queue_size: int = Field(10000, description="Maximum queue size")
    max_workers: int = Field(10, description="Maximum concurrent workers")
    job_timeout: float = Field(3600.0, description="Default job timeout in seconds")
    retry_delay: float = Field(60.0, description="Delay between retries in seconds")
    enable_persistence: bool = Field(True, description="Enable job persistence")
    persistence_path: str = Field("./job_queue_state.json", description="Persistence file path")
    cleanup_interval: float = Field(300.0, description="Cleanup interval in seconds")
    max_job_age_days: int = Field(7, description="Maximum job age in days")


class JobQueue:
    """
    Priority-based job queue for managing document generation tasks.
    
    Features:
    - Priority-based execution
    - Async and sync job support
    - Job dependencies
    - Automatic retries
    - Status tracking
    - Persistence support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize job queue"""
        self.config = JobQueueConfig(**(config or {}))
        self.logger = get_logger(__name__)
        
        # Job storage
        self.jobs: Dict[str, Job] = {}
        self.queue: PriorityQueue = PriorityQueue(maxsize=self.config.max_queue_size)
        
        # Worker management
        self.workers: List[threading.Thread] = []
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.running_jobs: Dict[str, Future] = {}
        
        # Control flags
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Cleanup thread
        self.cleanup_thread: Optional[threading.Thread] = None
        
        # Load persisted state if enabled
        if self.config.enable_persistence:
            self._load_state()
        
        self.logger.info(f"Job queue initialized with {self.config.max_workers} workers")
    
    def submit_job(
        self,
        name: str,
        job_type: str,
        payload: Dict[str, Any],
        priority: Union[JobPriority, str] = JobPriority.NORMAL,
        callback: Optional[Callable] = None,
        timeout: Optional[float] = None,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None
    ) -> str:
        """
        Submit a new job to the queue.
        
        Args:
            name: Job name
            job_type: Type of job (e.g., 'document_generation')
            payload: Job payload data
            priority: Job priority
            callback: Optional callback function
            timeout: Job timeout in seconds
            tags: Optional job tags
            dependencies: List of job IDs that must complete first
            
        Returns:
            Job ID
        """
        # Generate job ID
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        
        # Convert string priority to enum
        if isinstance(priority, str):
            priority = JobPriority[priority.upper()]
        
        # Create job metadata
        metadata = JobMetadata(
            tags=tags or [],
            dependencies=dependencies or []
        )
        
        # Create job
        job = Job(
            job_id=job_id,
            name=name,
            job_type=job_type,
            payload=payload,
            priority=priority,
            metadata=metadata,
            callback=callback,
            timeout=timeout or self.config.job_timeout
        )
        
        # Store job
        self.jobs[job_id] = job
        
        # Check dependencies
        if self._can_queue_job(job):
            self._queue_job(job)
        
        self.logger.info(f"Job submitted: {job_id} - {name} (priority: {priority.name})")
        
        # Persist state
        if self.config.enable_persistence:
            self._save_state()
        
        return job_id
    
    def _can_queue_job(self, job: Job) -> bool:
        """Check if job can be queued based on dependencies"""
        if not job.metadata.dependencies:
            return True
        
        # Check all dependencies
        for dep_id in job.metadata.dependencies:
            if dep_id not in self.jobs:
                self.logger.warning(f"Job {job.job_id} has unknown dependency: {dep_id}")
                return False
            
            dep_job = self.jobs[dep_id]
            if dep_job.status not in [JobStatus.COMPLETED]:
                return False
        
        return True
    
    def _queue_job(self, job: Job):
        """Add job to the priority queue"""
        if self.queue.full():
            raise ProcessingError("Job queue is full")
        
        job.status = JobStatus.QUEUED
        job.metadata.updated_at = datetime.now()
        self.queue.put(job)
    
    def start(self):
        """Start the job queue workers"""
        if self.is_running:
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start worker threads
        for i in range(self.config.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"JobQueueWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        # Start cleanup thread
        if self.config.cleanup_interval > 0:
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                name="JobQueueCleanup",
                daemon=True
            )
            self.cleanup_thread.start()
        
        self.logger.info("Job queue started")
    
    def stop(self, wait: bool = True, timeout: float = 30.0):
        """Stop the job queue"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping job queue...")
        self.is_running = False
        self.shutdown_event.set()
        
        # Cancel pending jobs
        while not self.queue.empty():
            try:
                job = self.queue.get_nowait()
                job.status = JobStatus.CANCELLED
                job.metadata.updated_at = datetime.now()
            except:
                break
        
        # Wait for workers to finish
        if wait:
            for worker in self.workers:
                worker.join(timeout=timeout)
        
        # Shutdown executor
        self.executor.shutdown(wait=wait)
        
        # Save final state
        if self.config.enable_persistence:
            self._save_state()
        
        self.logger.info("Job queue stopped")
    
    def _worker_loop(self):
        """Worker thread main loop"""
        while self.is_running:
            try:
                # Get job from queue with timeout
                job = self.queue.get(timeout=1.0)
                
                # Process job
                self._process_job(job)
                
                # Check for jobs with satisfied dependencies
                self._check_pending_jobs()
                
            except:
                # Queue empty or timeout
                continue
    
    def _process_job(self, job: Job):
        """Process a single job"""
        self.logger.info(f"Processing job: {job.job_id} - {job.name}")
        
        # Update job status
        job.status = JobStatus.RUNNING
        job.metadata.started_at = datetime.now()
        job.metadata.updated_at = datetime.now()
        
        try:
            # Submit to executor
            future = self.executor.submit(self._execute_job, job)
            self.running_jobs[job.job_id] = future
            
            # Wait for completion with timeout
            result = future.result(timeout=job.timeout)
            
            # Update job with result
            job.metadata.result = result
            job.status = JobStatus.COMPLETED
            job.metadata.completed_at = datetime.now()
            
            self.logger.info(f"Job completed: {job.job_id}")
            
            # Execute callback if provided
            if job.callback:
                try:
                    job.callback(job)
                except Exception as e:
                    self.logger.error(f"Job callback failed: {e}")
        
        except Exception as e:
            self.logger.error(f"Job failed: {job.job_id} - {str(e)}")
            job.metadata.error_message = str(e)
            
            # Check retry
            if job.metadata.retry_count < job.metadata.max_retries:
                job.status = JobStatus.RETRYING
                job.metadata.retry_count += 1
                
                # Schedule retry
                threading.Timer(
                    self.config.retry_delay,
                    lambda: self._queue_job(job)
                ).start()
                
                self.logger.info(f"Job scheduled for retry: {job.job_id} (attempt {job.metadata.retry_count})")
            else:
                job.status = JobStatus.FAILED
                self.logger.error(f"Job failed after {job.metadata.retry_count} retries: {job.job_id}")
        
        finally:
            # Clean up
            job.metadata.updated_at = datetime.now()
            self.running_jobs.pop(job.job_id, None)
            
            # Persist state
            if self.config.enable_persistence:
                self._save_state()
    
    def _execute_job(self, job: Job) -> Any:
        """Execute the actual job logic"""
        job_type = job.job_type
        payload = job.payload
        
        # Route to appropriate handler based on job type
        if job_type == "document_generation":
            return self._handle_document_generation(payload)
        elif job_type == "batch_processing":
            return self._handle_batch_processing(payload)
        elif job_type == "validation":
            return self._handle_validation(payload)
        elif job_type == "export":
            return self._handle_export(payload)
        elif job_type == "custom":
            # Execute custom function if provided
            if "function" in payload and callable(payload["function"]):
                func = payload["function"]
                args = payload.get("args", [])
                kwargs = payload.get("kwargs", {})
                return func(*args, **kwargs)
            else:
                raise ValueError("Custom job requires callable function in payload")
        else:
            raise ValueError(f"Unknown job type: {job_type}")
    
    def _handle_document_generation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document generation job"""
        # Simulate document generation
        time.sleep(1.0)  # Placeholder for actual generation
        
        return {
            "status": "success",
            "document_id": f"doc_{uuid.uuid4().hex[:8]}",
            "generation_time": 1.0
        }
    
    def _handle_batch_processing(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle batch processing job"""
        batch_size = payload.get("batch_size", 10)
        
        # Simulate batch processing
        results = []
        for i in range(batch_size):
            time.sleep(0.1)  # Placeholder for actual processing
            results.append(f"item_{i}")
        
        return {
            "status": "success",
            "processed_items": results,
            "batch_size": batch_size
        }
    
    def _handle_validation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle validation job"""
        # Simulate validation
        time.sleep(0.5)  # Placeholder for actual validation
        
        return {
            "status": "success",
            "valid": True,
            "validation_time": 0.5
        }
    
    def _handle_export(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle export job"""
        export_format = payload.get("format", "pdf")
        
        # Simulate export
        time.sleep(2.0)  # Placeholder for actual export
        
        return {
            "status": "success",
            "export_path": f"/exports/export_{uuid.uuid4().hex[:8]}.{export_format}",
            "export_time": 2.0
        }
    
    def _check_pending_jobs(self):
        """Check pending jobs for satisfied dependencies"""
        pending_jobs = [
            job for job in self.jobs.values()
            if job.status == JobStatus.PENDING
        ]
        
        for job in pending_jobs:
            if self._can_queue_job(job):
                try:
                    self._queue_job(job)
                except Exception as e:
                    self.logger.error(f"Failed to queue job {job.job_id}: {e}")
    
    def _cleanup_loop(self):
        """Cleanup old jobs periodically"""
        while self.is_running:
            try:
                # Wait for cleanup interval
                self.shutdown_event.wait(timeout=self.config.cleanup_interval)
                
                if not self.is_running:
                    break
                
                # Clean up old jobs
                self._cleanup_old_jobs()
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
    
    def _cleanup_old_jobs(self):
        """Remove old completed/failed jobs"""
        cutoff_date = datetime.now() - timedelta(days=self.config.max_job_age_days)
        jobs_to_remove = []
        
        for job_id, job in self.jobs.items():
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                if job.metadata.updated_at < cutoff_date:
                    jobs_to_remove.append(job_id)
        
        # Remove old jobs
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
        
        if jobs_to_remove:
            self.logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
            
            # Persist state
            if self.config.enable_persistence:
                self._save_state()
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self.jobs.get(job_id)
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status"""
        job = self.jobs.get(job_id)
        return job.status if job else None
    
    def get_job_result(self, job_id: str) -> Optional[Any]:
        """Get job result"""
        job = self.jobs.get(job_id)
        return job.metadata.result if job else None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.PENDING, JobStatus.QUEUED]:
            job.status = JobStatus.CANCELLED
            job.metadata.updated_at = datetime.now()
            
            # Remove from queue if queued
            if job.status == JobStatus.QUEUED:
                # Note: PriorityQueue doesn't support removal, so we mark it cancelled
                pass
            
            self.logger.info(f"Job cancelled: {job_id}")
            return True
        
        elif job.status == JobStatus.RUNNING:
            # Cancel running job
            future = self.running_jobs.get(job_id)
            if future:
                future.cancel()
                job.status = JobStatus.CANCELLED
                job.metadata.updated_at = datetime.now()
                self.logger.info(f"Running job cancelled: {job_id}")
                return True
        
        return False
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        status_counts = {}
        for job in self.jobs.values():
            status = job.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_jobs": len(self.jobs),
            "queue_size": self.queue.qsize(),
            "running_jobs": len(self.running_jobs),
            "status_counts": status_counts,
            "workers": self.config.max_workers,
            "is_running": self.is_running
        }
    
    def wait_for_job(self, job_id: str, timeout: Optional[float] = None) -> Optional[Any]:
        """Wait for job completion and return result"""
        start_time = time.time()
        
        while True:
            job = self.get_job(job_id)
            if not job:
                return None
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return job.metadata.result
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            time.sleep(0.1)
    
    def _save_state(self):
        """Save queue state to disk"""
        try:
            import json
            from pathlib import Path
            
            state = {
                "jobs": {
                    job_id: {
                        "job_id": job.job_id,
                        "name": job.name,
                        "job_type": job.job_type,
                        "payload": job.payload,
                        "priority": job.priority.name,
                        "status": job.status.name,
                        "metadata": {
                            "created_at": job.metadata.created_at.isoformat(),
                            "updated_at": job.metadata.updated_at.isoformat(),
                            "started_at": job.metadata.started_at.isoformat() if job.metadata.started_at else None,
                            "completed_at": job.metadata.completed_at.isoformat() if job.metadata.completed_at else None,
                            "retry_count": job.metadata.retry_count,
                            "max_retries": job.metadata.max_retries,
                            "error_message": job.metadata.error_message,
                            "tags": job.metadata.tags,
                            "dependencies": job.metadata.dependencies
                        }
                    }
                    for job_id, job in self.jobs.items()
                    if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
                }
            }
            
            Path(self.config.persistence_path).write_text(json.dumps(state, indent=2))
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def _load_state(self):
        """Load queue state from disk"""
        try:
            import json
            from pathlib import Path
            
            state_file = Path(self.config.persistence_path)
            if not state_file.exists():
                return
            
            state = json.loads(state_file.read_text())
            
            # Restore jobs
            for job_data in state.get("jobs", {}).values():
                metadata = JobMetadata(
                    created_at=datetime.fromisoformat(job_data["metadata"]["created_at"]),
                    updated_at=datetime.fromisoformat(job_data["metadata"]["updated_at"]),
                    started_at=datetime.fromisoformat(job_data["metadata"]["started_at"]) if job_data["metadata"]["started_at"] else None,
                    completed_at=datetime.fromisoformat(job_data["metadata"]["completed_at"]) if job_data["metadata"]["completed_at"] else None,
                    retry_count=job_data["metadata"]["retry_count"],
                    max_retries=job_data["metadata"]["max_retries"],
                    error_message=job_data["metadata"]["error_message"],
                    tags=job_data["metadata"]["tags"],
                    dependencies=job_data["metadata"]["dependencies"]
                )
                
                job = Job(
                    job_id=job_data["job_id"],
                    name=job_data["name"],
                    job_type=job_data["job_type"],
                    payload=job_data["payload"],
                    priority=JobPriority[job_data["priority"]],
                    status=JobStatus.PENDING,  # Reset to pending
                    metadata=metadata
                )
                
                self.jobs[job.job_id] = job
                
                # Re-queue if it was queued or running
                if self._can_queue_job(job):
                    self._queue_job(job)
            
            self.logger.info(f"Loaded {len(self.jobs)} jobs from state")
            
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")


# Factory function
def create_job_queue(config: Optional[Dict[str, Any]] = None) -> JobQueue:
    """Create and return a job queue instance"""
    return JobQueue(config)