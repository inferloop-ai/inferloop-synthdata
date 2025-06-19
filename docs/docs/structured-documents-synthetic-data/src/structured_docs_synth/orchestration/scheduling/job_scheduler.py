#!/usr/bin/env python3
"""
Job scheduler for queue-based job processing.

Provides comprehensive job queue management with priority scheduling,
retry logic, and distributed processing capabilities.
"""

import asyncio
import heapq
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import pickle
import json

from pydantic import BaseModel, Field

from ...core import get_logger, BaseConfig

logger = get_logger(__name__)


class JobPriority(int, Enum):
    """Job priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class JobState(str, Enum):
    """Job execution states"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class Job:
    """Job definition"""
    id: str
    name: str
    function: str  # Function name or callable reference
    args: tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: JobPriority = JobPriority.NORMAL
    max_retries: int = 3
    retry_delay_seconds: int = 60
    timeout_seconds: Optional[int] = None
    scheduled_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __lt__(self, other):
        """For priority queue ordering"""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value  # Higher priority first
        return self.created_at < other.created_at  # FIFO for same priority


@dataclass
class JobResult:
    """Job execution result"""
    job_id: str
    state: JobState
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    worker_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get execution duration"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class JobSchedulerConfig(BaseConfig):
    """Job scheduler configuration"""
    max_workers: int = 4
    max_queue_size: int = 1000
    worker_check_interval_seconds: int = 5
    
    # Job defaults
    default_timeout_seconds: int = 3600
    default_max_retries: int = 3
    default_retry_delay_seconds: int = 60
    
    # Queue management
    enable_persistent_queue: bool = False
    queue_persistence_file: Optional[str] = None
    
    # Monitoring
    enable_job_metrics: bool = True
    job_history_retention_days: int = 7
    
    # Distributed processing
    enable_distributed: bool = False
    redis_url: Optional[str] = None
    worker_heartbeat_interval: int = 30


class JobScheduler:
    """Comprehensive job queue and scheduling system"""
    
    def __init__(self, config: JobSchedulerConfig):
        self.config = config
        self.job_queue: List[Job] = []  # Priority queue
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.job_results: Dict[str, JobResult] = {}
        self.job_history: List[JobResult] = []
        self.registered_functions: Dict[str, Callable] = {}
        self.workers: List[asyncio.Task] = []
        self.running = False
        self.worker_id = str(uuid.uuid4())
        self.job_callbacks: Dict[str, List[Callable]] = {}
        
        # Job metrics
        self.metrics = {
            'jobs_queued': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'jobs_cancelled': 0,
            'total_processing_time': 0.0
        }
    
    async def start(self):
        """Start job scheduler"""
        self.running = True
        logger.info(f"Job scheduler started with {self.config.max_workers} workers")
        
        # Start worker tasks
        for i in range(self.config.max_workers):
            worker_task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker_task)
        
        # Start management tasks
        asyncio.create_task(self._cleanup_worker())
        asyncio.create_task(self._metrics_worker())
        
        # Load persistent queue if enabled
        if self.config.enable_persistent_queue:
            await self._load_persistent_queue()
    
    async def stop(self):
        """Stop job scheduler"""
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Cancel running jobs
        for task in self.running_jobs.values():
            task.cancel()
        
        # Save persistent queue if enabled
        if self.config.enable_persistent_queue:
            await self._save_persistent_queue()
        
        logger.info("Job scheduler stopped")
    
    def register_function(self, name: str, function: Callable):
        """
        Register function for job execution.
        
        Args:
            name: Function name
            function: Callable function
        """
        self.registered_functions[name] = function
        logger.info(f"Registered function: {name}")
    
    async def submit_job(self, job: Job) -> str:
        """
        Submit job to queue.
        
        Args:
            job: Job to submit
        
        Returns:
            Job ID
        """
        try:
            if len(self.job_queue) >= self.config.max_queue_size:
                raise ValueError("Job queue is full")
            
            # Generate ID if not provided
            if not job.id:
                job.id = str(uuid.uuid4())
            
            # Set default scheduled time
            if job.scheduled_time is None:
                job.scheduled_time = datetime.now()
            
            # Add to priority queue
            heapq.heappush(self.job_queue, job)
            
            # Create initial result
            result = JobResult(
                job_id=job.id,
                state=JobState.QUEUED
            )
            self.job_results[job.id] = result
            
            self.metrics['jobs_queued'] += 1
            
            logger.info(f"Submitted job: {job.id} - {job.name} (priority: {job.priority.name})")
            return job.id
            
        except Exception as e:
            logger.error(f"Failed to submit job: {e}")
            raise
    
    async def submit_function_job(self, function_name: str, *args, 
                                name: Optional[str] = None,
                                priority: JobPriority = JobPriority.NORMAL,
                                **kwargs) -> str:
        """
        Submit job for registered function.
        
        Args:
            function_name: Name of registered function
            *args: Function arguments
            name: Job name
            priority: Job priority
            **kwargs: Function keyword arguments
        
        Returns:
            Job ID
        """
        if function_name not in self.registered_functions:
            raise ValueError(f"Function '{function_name}' not registered")
        
        job = Job(
            id=str(uuid.uuid4()),
            name=name or f"Function: {function_name}",
            function=function_name,
            args=args,
            kwargs=kwargs,
            priority=priority
        )
        
        return await self.submit_job(job)
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel job.
        
        Args:
            job_id: ID of job to cancel
        
        Returns:
            True if job was cancelled
        """
        try:
            # Check if job is running
            if job_id in self.running_jobs:
                task = self.running_jobs[job_id]
                task.cancel()
                del self.running_jobs[job_id]
                
                if job_id in self.job_results:
                    self.job_results[job_id].state = JobState.CANCELLED
                    self.job_results[job_id].end_time = datetime.now()
                
                self.metrics['jobs_cancelled'] += 1
                logger.info(f"Cancelled running job: {job_id}")
                return True
            
            # Check if job is in queue
            for i, job in enumerate(self.job_queue):
                if job.id == job_id:
                    del self.job_queue[i]
                    heapq.heapify(self.job_queue)  # Restore heap property
                    
                    if job_id in self.job_results:
                        self.job_results[job_id].state = JobState.CANCELLED
                        self.job_results[job_id].end_time = datetime.now()
                    
                    self.metrics['jobs_cancelled'] += 1
                    logger.info(f"Cancelled queued job: {job_id}")
                    return True
            
            logger.warning(f"Job not found for cancellation: {job_id}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status.
        
        Args:
            job_id: Job ID
        
        Returns:
            Job status dictionary
        """
        if job_id not in self.job_results:
            return None
        
        result = self.job_results[job_id]
        
        # Find job in queue
        queue_position = None
        for i, job in enumerate(self.job_queue):
            if job.id == job_id:
                queue_position = i
                break
        
        return {
            'job_id': job_id,
            'state': result.state.value,
            'queue_position': queue_position,
            'is_running': job_id in self.running_jobs,
            'start_time': result.start_time.isoformat() if result.start_time else None,
            'end_time': result.end_time.isoformat() if result.end_time else None,
            'duration_seconds': result.duration_seconds,
            'retry_count': result.retry_count,
            'worker_id': result.worker_id,
            'error': result.error
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get queue status.
        
        Returns:
            Queue status dictionary
        """
        # Count jobs by priority
        priority_counts = {priority.name: 0 for priority in JobPriority}
        for job in self.job_queue:
            priority_counts[job.priority.name] += 1
        
        # Count jobs by state
        state_counts = {state.name: 0 for state in JobState}
        for result in self.job_results.values():
            state_counts[result.state.name] += 1
        
        return {
            'queue_size': len(self.job_queue),
            'running_jobs': len(self.running_jobs),
            'max_workers': self.config.max_workers,
            'available_workers': self.config.max_workers - len(self.running_jobs),
            'priority_distribution': priority_counts,
            'state_distribution': state_counts,
            'metrics': self.metrics.copy()
        }
    
    def add_job_callback(self, job_id: str, callback: Callable[[JobResult], None]):
        """
        Add callback for job completion.
        
        Args:
            job_id: Job ID or '*' for all jobs
            callback: Callback function
        """
        if job_id not in self.job_callbacks:
            self.job_callbacks[job_id] = []
        self.job_callbacks[job_id].append(callback)
    
    async def _worker(self, worker_id: str):
        """Worker task for processing jobs"""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get next job from queue
                job = await self._get_next_job()
                if job is None:
                    await asyncio.sleep(self.config.worker_check_interval_seconds)
                    continue
                
                # Execute job
                await self._execute_job(job, worker_id)
                
            except Exception as e:
                logger.error(f"Error in worker {worker_id}: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _get_next_job(self) -> Optional[Job]:
        """Get next job from priority queue"""
        try:
            # Check if there are jobs and if we have available workers
            if not self.job_queue or len(self.running_jobs) >= self.config.max_workers:
                return None
            
            # Get highest priority job that's ready to run
            now = datetime.now()
            
            for i, job in enumerate(self.job_queue):
                if job.scheduled_time and job.scheduled_time > now:
                    continue  # Job not ready yet
                
                # Remove job from queue
                del self.job_queue[i]
                heapq.heapify(self.job_queue)
                return job
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting next job: {e}")
            return None
    
    async def _execute_job(self, job: Job, worker_id: str):
        """Execute job"""
        job_result = self.job_results[job.id]
        job_result.state = JobState.RUNNING
        job_result.start_time = datetime.now()
        job_result.worker_id = worker_id
        
        # Add to running jobs
        execution_task = asyncio.create_task(self._run_job_function(job))
        self.running_jobs[job.id] = execution_task
        
        try:
            logger.info(f"Worker {worker_id} executing job: {job.id} - {job.name}")
            
            # Execute with timeout
            timeout = job.timeout_seconds or self.config.default_timeout_seconds
            result = await asyncio.wait_for(execution_task, timeout=timeout)
            
            job_result.state = JobState.COMPLETED
            job_result.result = result
            self.metrics['jobs_completed'] += 1
            
            logger.info(f"Job completed: {job.id}")
            
        except asyncio.TimeoutError:
            job_result.state = JobState.FAILED
            job_result.error = f"Job timed out after {timeout} seconds"
            self.metrics['jobs_failed'] += 1
            
            logger.error(f"Job timed out: {job.id}")
            
        except asyncio.CancelledError:
            job_result.state = JobState.CANCELLED
            self.metrics['jobs_cancelled'] += 1
            
            logger.info(f"Job cancelled: {job.id}")
            
        except Exception as e:
            job_result.state = JobState.FAILED
            job_result.error = str(e)
            self.metrics['jobs_failed'] += 1
            
            logger.error(f"Job failed: {job.id} - {e}")
            
            # Handle retry
            if job_result.retry_count < job.max_retries:
                await self._schedule_retry(job, job_result)
                return
        
        finally:
            # Clean up
            job_result.end_time = datetime.now()
            if job.id in self.running_jobs:
                del self.running_jobs[job.id]
            
            # Update metrics
            if job_result.duration_seconds:
                self.metrics['total_processing_time'] += job_result.duration_seconds
            
            # Add to history
            self.job_history.append(job_result)
            
            # Call callbacks
            await self._call_job_callbacks(job.id, job_result)
    
    async def _run_job_function(self, job: Job) -> Any:
        """Run job function"""
        if job.function not in self.registered_functions:
            raise ValueError(f"Function '{job.function}' not registered")
        
        function = self.registered_functions[job.function]
        
        if asyncio.iscoroutinefunction(function):
            return await function(*job.args, **job.kwargs)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: function(*job.args, **job.kwargs)
            )
    
    async def _schedule_retry(self, job: Job, job_result: JobResult):
        """Schedule job retry"""
        try:
            job_result.retry_count += 1
            job_result.state = JobState.RETRYING
            
            # Calculate retry delay
            delay = job.retry_delay_seconds * (2 ** (job_result.retry_count - 1))  # Exponential backoff
            
            # Schedule retry
            retry_job = Job(
                id=job.id,
                name=f"{job.name} (retry {job_result.retry_count})",
                function=job.function,
                args=job.args,
                kwargs=job.kwargs,
                priority=job.priority,
                max_retries=job.max_retries,
                retry_delay_seconds=job.retry_delay_seconds,
                timeout_seconds=job.timeout_seconds,
                scheduled_time=datetime.now() + timedelta(seconds=delay),
                metadata=job.metadata,
                created_at=job.created_at
            )
            
            heapq.heappush(self.job_queue, retry_job)
            
            logger.info(f"Scheduled retry for job {job.id} in {delay} seconds (attempt {job_result.retry_count})")
            
        except Exception as e:
            logger.error(f"Failed to schedule retry for job {job.id}: {e}")
    
    async def _call_job_callbacks(self, job_id: str, job_result: JobResult):
        """Call job callbacks"""
        try:
            # Job-specific callbacks
            if job_id in self.job_callbacks:
                for callback in self.job_callbacks[job_id]:
                    try:
                        await callback(job_result)
                    except Exception as e:
                        logger.error(f"Error in job callback for {job_id}: {e}")
            
            # Global callbacks
            if '*' in self.job_callbacks:
                for callback in self.job_callbacks['*']:
                    try:
                        await callback(job_result)
                    except Exception as e:
                        logger.error(f"Error in global job callback: {e}")
        
        except Exception as e:
            logger.error(f"Failed to call job callbacks: {e}")
    
    async def _cleanup_worker(self):
        """Background worker for cleanup"""
        while self.running:
            try:
                await self._cleanup_old_jobs()
                await asyncio.sleep(24 * 3600)  # Daily cleanup
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    async def _cleanup_old_jobs(self):
        """Clean up old job results"""
        try:
            cutoff = datetime.now() - timedelta(days=self.config.job_history_retention_days)
            
            # Clean job results
            old_results = [
                job_id for job_id, result in self.job_results.items()
                if result.end_time and result.end_time < cutoff
            ]
            
            for job_id in old_results:
                del self.job_results[job_id]
            
            # Clean job history
            original_count = len(self.job_history)
            self.job_history = [
                result for result in self.job_history
                if not result.end_time or result.end_time >= cutoff
            ]
            
            cleaned_count = original_count - len(self.job_history) + len(old_results)
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old job records")
        
        except Exception as e:
            logger.error(f"Failed to cleanup old jobs: {e}")
    
    async def _metrics_worker(self):
        """Background worker for metrics collection"""
        while self.running:
            try:
                # This could export metrics to monitoring systems
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Error in metrics worker: {e}")
    
    async def _load_persistent_queue(self):
        """Load persistent job queue"""
        try:
            if self.config.queue_persistence_file:
                # Implementation would load from file/database
                pass
        except Exception as e:
            logger.error(f"Failed to load persistent queue: {e}")
    
    async def _save_persistent_queue(self):
        """Save persistent job queue"""
        try:
            if self.config.queue_persistence_file:
                # Implementation would save to file/database
                pass
        except Exception as e:
            logger.error(f"Failed to save persistent queue: {e}")


def create_job_scheduler(config: Optional[JobSchedulerConfig] = None) -> JobScheduler:
    """Factory function to create job scheduler"""
    if config is None:
        config = JobSchedulerConfig()
    return JobScheduler(config)


__all__ = [
    'JobScheduler',
    'JobSchedulerConfig',
    'Job',
    'JobResult',
    'JobPriority',
    'JobState',
    'create_job_scheduler'
]