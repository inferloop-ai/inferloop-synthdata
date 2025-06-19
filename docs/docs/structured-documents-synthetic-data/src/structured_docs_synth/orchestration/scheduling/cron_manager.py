#!/usr/bin/env python3
"""
Cron manager for cron-like job scheduling.

Provides cron-style scheduling capabilities with support for
complex schedules, job management, and execution monitoring.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import croniter

from pydantic import BaseModel, Field

from ...core import get_logger, BaseConfig

logger = get_logger(__name__)


class JobStatus(str, Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CronJob:
    """Cron job definition"""
    id: str
    name: str
    cron_expression: str
    function: Callable
    args: tuple = ()
    kwargs: Dict[str, Any] = None
    enabled: bool = True
    max_retries: int = 3
    timeout_seconds: Optional[int] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


@dataclass
class JobExecution:
    """Job execution record"""
    job_id: str
    execution_id: str
    status: JobStatus
    scheduled_time: datetime
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['scheduled_time'] = self.scheduled_time.isoformat()
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get execution duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class CronManagerConfig(BaseConfig):
    """Cron manager configuration"""
    check_interval_seconds: int = 60
    max_concurrent_jobs: int = 10
    execution_history_days: int = 30
    
    # Job defaults
    default_timeout_seconds: int = 3600
    default_max_retries: int = 3
    
    # Monitoring
    enable_job_monitoring: bool = True
    alert_on_job_failure: bool = True
    
    # Persistence
    persist_job_history: bool = True
    job_history_file: Optional[str] = None


class CronManager:
    """Comprehensive cron-style job scheduler"""
    
    def __init__(self, config: CronManagerConfig):
        self.config = config
        self.jobs: Dict[str, CronJob] = {}
        self.executions: Dict[str, JobExecution] = {}
        self.execution_history: List[JobExecution] = []
        self.running = False
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.job_callbacks: Dict[str, List[Callable]] = {}
        
    async def start(self):
        """Start cron manager"""
        self.running = True
        logger.info("Cron manager started")
        
        # Start scheduler worker
        asyncio.create_task(self._scheduler_worker())
        asyncio.create_task(self._cleanup_worker())
    
    async def stop(self):
        """Stop cron manager"""
        self.running = False
        
        # Cancel running jobs
        for task in self.running_jobs.values():
            task.cancel()
        
        logger.info("Cron manager stopped")
    
    def add_job(self, job: CronJob) -> bool:
        """
        Add cron job.
        
        Args:
            job: Cron job to add
        
        Returns:
            True if job was added successfully
        """
        try:
            # Validate cron expression
            croniter.croniter(job.cron_expression)
            
            self.jobs[job.id] = job
            logger.info(f"Added cron job: {job.id} - {job.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add cron job {job.id}: {e}")
            return False
    
    def remove_job(self, job_id: str) -> bool:
        """
        Remove cron job.
        
        Args:
            job_id: ID of job to remove
        
        Returns:
            True if job was removed
        """
        if job_id in self.jobs:
            # Cancel if running
            if job_id in self.running_jobs:
                self.running_jobs[job_id].cancel()
                del self.running_jobs[job_id]
            
            del self.jobs[job_id]
            logger.info(f"Removed cron job: {job_id}")
            return True
        
        return False
    
    def enable_job(self, job_id: str) -> bool:
        """
        Enable cron job.
        
        Args:
            job_id: ID of job to enable
        
        Returns:
            True if job was enabled
        """
        if job_id in self.jobs:
            self.jobs[job_id].enabled = True
            logger.info(f"Enabled cron job: {job_id}")
            return True
        return False
    
    def disable_job(self, job_id: str) -> bool:
        """
        Disable cron job.
        
        Args:
            job_id: ID of job to disable
        
        Returns:
            True if job was disabled
        """
        if job_id in self.jobs:
            self.jobs[job_id].enabled = False
            logger.info(f"Disabled cron job: {job_id}")
            return True
        return False
    
    async def run_job_now(self, job_id: str) -> Optional[JobExecution]:
        """
        Run job immediately.
        
        Args:
            job_id: ID of job to run
        
        Returns:
            Job execution record
        """
        if job_id not in self.jobs:
            logger.error(f"Job not found: {job_id}")
            return None
        
        job = self.jobs[job_id]
        execution = JobExecution(
            job_id=job_id,
            execution_id=f"{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_manual",
            status=JobStatus.PENDING,
            scheduled_time=datetime.now()
        )
        
        await self._execute_job(job, execution)
        return execution
    
    def get_next_run_time(self, job_id: str) -> Optional[datetime]:
        """
        Get next scheduled run time for job.
        
        Args:
            job_id: ID of job
        
        Returns:
            Next run time or None
        """
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        if not job.enabled:
            return None
        
        try:
            cron = croniter.croniter(job.cron_expression, datetime.now())
            return cron.get_next(datetime)
        except Exception as e:
            logger.error(f"Failed to get next run time for {job_id}: {e}")
            return None
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status and statistics.
        
        Args:
            job_id: ID of job
        
        Returns:
            Job status dictionary
        """
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        
        # Get recent executions
        recent_executions = [
            ex for ex in self.execution_history 
            if ex.job_id == job_id
        ][-10:]  # Last 10 executions
        
        # Calculate statistics
        total_runs = len(recent_executions)
        successful_runs = len([ex for ex in recent_executions if ex.status == JobStatus.COMPLETED])
        failed_runs = len([ex for ex in recent_executions if ex.status == JobStatus.FAILED])
        
        avg_duration = None
        if recent_executions:
            durations = [ex.duration_seconds for ex in recent_executions if ex.duration_seconds]
            if durations:
                avg_duration = sum(durations) / len(durations)
        
        return {
            'job_id': job_id,
            'name': job.name,
            'enabled': job.enabled,
            'cron_expression': job.cron_expression,
            'next_run': self.get_next_run_time(job_id).isoformat() if self.get_next_run_time(job_id) else None,
            'is_running': job_id in self.running_jobs,
            'statistics': {
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'failed_runs': failed_runs,
                'success_rate': (successful_runs / total_runs * 100) if total_runs > 0 else 0,
                'average_duration_seconds': avg_duration
            },
            'recent_executions': [ex.to_dict() for ex in recent_executions]
        }
    
    def get_all_jobs_status(self) -> List[Dict[str, Any]]:
        """
        Get status for all jobs.
        
        Returns:
            List of job status dictionaries
        """
        return [self.get_job_status(job_id) for job_id in self.jobs.keys()]
    
    def add_job_callback(self, job_id: str, callback: Callable[[JobExecution], None]):
        """
        Add callback for job completion.
        
        Args:
            job_id: Job ID or '*' for all jobs
            callback: Callback function
        """
        if job_id not in self.job_callbacks:
            self.job_callbacks[job_id] = []
        self.job_callbacks[job_id].append(callback)
    
    async def _scheduler_worker(self):
        """Background worker for job scheduling"""
        while self.running:
            try:
                current_time = datetime.now()
                
                for job_id, job in self.jobs.items():
                    if not job.enabled:
                        continue
                    
                    # Skip if already running
                    if job_id in self.running_jobs:
                        continue
                    
                    # Check if job should run
                    if self._should_run_job(job, current_time):
                        await self._schedule_job_execution(job)
                
                await asyncio.sleep(self.config.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in scheduler worker: {e}")
    
    def _should_run_job(self, job: CronJob, current_time: datetime) -> bool:
        """Check if job should run at current time"""
        try:
            cron = croniter.croniter(job.cron_expression)
            
            # Get previous run time
            prev_time = cron.get_prev(datetime)
            
            # Check if we're within the execution window
            window_start = prev_time
            window_end = prev_time + timedelta(seconds=self.config.check_interval_seconds)
            
            return window_start <= current_time <= window_end
            
        except Exception as e:
            logger.error(f"Failed to check job schedule for {job.id}: {e}")
            return False
    
    async def _schedule_job_execution(self, job: CronJob):
        """Schedule job for execution"""
        try:
            if len(self.running_jobs) >= self.config.max_concurrent_jobs:
                logger.warning(f"Max concurrent jobs reached, skipping {job.id}")
                return
            
            execution = JobExecution(
                job_id=job.id,
                execution_id=f"{job.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                status=JobStatus.PENDING,
                scheduled_time=datetime.now()
            )
            
            # Start job execution
            task = asyncio.create_task(self._execute_job(job, execution))
            self.running_jobs[job.id] = task
            
            logger.info(f"Scheduled job execution: {job.id}")
            
        except Exception as e:
            logger.error(f"Failed to schedule job {job.id}: {e}")
    
    async def _execute_job(self, job: CronJob, execution: JobExecution):
        """Execute job"""
        try:
            execution.status = JobStatus.RUNNING
            execution.start_time = datetime.now()
            
            self.executions[execution.execution_id] = execution
            
            logger.info(f"Starting job execution: {job.id} ({execution.execution_id})")
            
            # Determine timeout
            timeout = job.timeout_seconds or self.config.default_timeout_seconds
            
            # Execute job function
            try:
                if asyncio.iscoroutinefunction(job.function):
                    result = await asyncio.wait_for(
                        job.function(*job.args, **job.kwargs),
                        timeout=timeout
                    )
                else:
                    # Run sync function in thread pool
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, lambda: job.function(*job.args, **job.kwargs)
                        ),
                        timeout=timeout
                    )
                
                execution.status = JobStatus.COMPLETED
                execution.result = result
                
                logger.info(f"Job completed successfully: {job.id}")
                
            except asyncio.TimeoutError:
                execution.status = JobStatus.FAILED
                execution.error = f"Job timed out after {timeout} seconds"
                logger.error(f"Job timed out: {job.id}")
                
            except Exception as e:
                execution.status = JobStatus.FAILED
                execution.error = str(e)
                logger.error(f"Job failed: {job.id} - {e}")
                
                # Retry if configured
                if execution.retry_count < job.max_retries:
                    execution.retry_count += 1
                    logger.info(f"Retrying job {job.id} (attempt {execution.retry_count})")
                    await asyncio.sleep(60)  # Wait before retry
                    return await self._execute_job(job, execution)
            
            execution.end_time = datetime.now()
            
            # Store in history
            self.execution_history.append(execution)
            
            # Call callbacks
            await self._call_job_callbacks(job.id, execution)
            
        except Exception as e:
            logger.error(f"Error executing job {job.id}: {e}")
            execution.status = JobStatus.FAILED
            execution.error = str(e)
            execution.end_time = datetime.now()
            
        finally:
            # Remove from running jobs
            if job.id in self.running_jobs:
                del self.running_jobs[job.id]
    
    async def _call_job_callbacks(self, job_id: str, execution: JobExecution):
        """Call job callbacks"""
        try:
            # Job-specific callbacks
            if job_id in self.job_callbacks:
                for callback in self.job_callbacks[job_id]:
                    try:
                        await callback(execution)
                    except Exception as e:
                        logger.error(f"Error in job callback for {job_id}: {e}")
            
            # Global callbacks
            if '*' in self.job_callbacks:
                for callback in self.job_callbacks['*']:
                    try:
                        await callback(execution)
                    except Exception as e:
                        logger.error(f"Error in global job callback: {e}")
        
        except Exception as e:
            logger.error(f"Failed to call job callbacks: {e}")
    
    async def _cleanup_worker(self):
        """Background worker for cleanup"""
        while self.running:
            try:
                await self._cleanup_old_executions()
                await asyncio.sleep(24 * 3600)  # Daily cleanup
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    async def _cleanup_old_executions(self):
        """Clean up old job executions"""
        try:
            cutoff = datetime.now() - timedelta(days=self.config.execution_history_days)
            
            original_count = len(self.execution_history)
            self.execution_history = [
                ex for ex in self.execution_history 
                if ex.scheduled_time >= cutoff
            ]
            
            cleaned_count = original_count - len(self.execution_history)
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old job executions")
        
        except Exception as e:
            logger.error(f"Failed to cleanup old executions: {e}")


def create_cron_manager(config: Optional[CronManagerConfig] = None) -> CronManager:
    """Factory function to create cron manager"""
    if config is None:
        config = CronManagerConfig()
    return CronManager(config)


__all__ = [
    'CronManager',
    'CronManagerConfig',
    'CronJob',
    'JobExecution',
    'JobStatus',
    'create_cron_manager'
]