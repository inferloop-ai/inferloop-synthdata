#!/usr/bin/env python3
"""
Pipeline manager for data processing pipeline orchestration.

Provides comprehensive pipeline management with support for
linear and parallel data processing workflows.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, Iterator
from dataclasses import dataclass, asdict, field
from enum import Enum
import json

from pydantic import BaseModel, Field

from ...core import get_logger, BaseConfig

logger = get_logger(__name__)


class PipelineStatus(str, Enum):
    """Pipeline execution status"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StageStatus(str, Enum):
    """Stage execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class ProcessingMode(str, Enum):
    """Data processing modes"""
    BATCH = "batch"
    STREAM = "stream"
    MICRO_BATCH = "micro_batch"


@dataclass
class DataBatch:
    """Data batch for processing"""
    id: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    size: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class StageDefinition:
    """Pipeline stage definition"""
    id: str
    name: str
    processor: Callable
    config: Dict[str, Any] = field(default_factory=dict)
    parallel: bool = False
    max_workers: int = 1
    batch_size: Optional[int] = None
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    on_error: str = "stop"  # stop, skip, retry
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageExecution:
    """Stage execution record"""
    stage_id: str
    execution_id: str
    pipeline_id: str
    status: StageStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processed_batches: int = 0
    failed_batches: int = 0
    total_items_processed: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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


@dataclass
class PipelineDefinition:
    """Pipeline definition"""
    id: str
    name: str
    description: str
    stages: List[StageDefinition]
    processing_mode: ProcessingMode = ProcessingMode.BATCH
    global_timeout_seconds: Optional[int] = None
    max_parallel_stages: int = 5
    default_batch_size: int = 100
    error_handling: str = "stop"  # stop, continue
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PipelineExecution:
    """Pipeline execution record"""
    pipeline_id: str
    execution_id: str
    status: PipelineStatus
    processing_mode: ProcessingMode
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    stage_executions: Dict[str, StageExecution] = field(default_factory=dict)
    total_input_items: int = 0
    total_output_items: int = 0
    total_failed_items: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        
        # Convert stage executions
        data['stage_executions'] = {
            stage_id: execution.to_dict() 
            for stage_id, execution in self.stage_executions.items()
        }
        
        return data


class PipelineManagerConfig(BaseConfig):
    """Pipeline manager configuration"""
    max_concurrent_pipelines: int = 3
    max_concurrent_stages: int = 10
    default_stage_timeout_seconds: int = 3600
    default_pipeline_timeout_seconds: int = 7200
    
    # Batch processing
    default_batch_size: int = 100
    max_batch_size: int = 10000
    
    # Stream processing
    stream_buffer_size: int = 1000
    stream_flush_interval_seconds: int = 30
    
    # Error handling
    enable_automatic_retry: bool = True
    default_retry_delay_seconds: int = 60
    
    # Monitoring
    enable_pipeline_monitoring: bool = True
    metrics_collection_interval_seconds: int = 30
    
    # Persistence
    execution_retention_days: int = 30
    enable_checkpoint: bool = False
    checkpoint_interval_batches: int = 100


class PipelineManager:
    """Comprehensive pipeline management system"""
    
    def __init__(self, config: PipelineManagerConfig):
        self.config = config
        self.pipelines: Dict[str, PipelineDefinition] = {}
        self.executions: Dict[str, PipelineExecution] = {}
        self.execution_history: List[PipelineExecution] = []
        self.running_pipelines: Dict[str, asyncio.Task] = {}
        self.running_stages: Dict[str, asyncio.Task] = {}
        self.data_queues: Dict[str, asyncio.Queue] = {}
        self.running = False
        
        # Pipeline callbacks
        self.pipeline_callbacks: Dict[str, List[Callable]] = {}
        self.stage_callbacks: Dict[str, List[Callable]] = {}
        
        # Metrics
        self.metrics = {
            'pipelines_executed': 0,
            'pipelines_completed': 0,
            'pipelines_failed': 0,
            'total_items_processed': 0,
            'total_processing_time': 0.0
        }
    
    async def start(self):
        """Start pipeline manager"""
        self.running = True
        logger.info("Pipeline manager started")
        
        # Start management tasks
        asyncio.create_task(self._cleanup_worker())
        asyncio.create_task(self._metrics_worker())
    
    async def stop(self):
        """Stop pipeline manager"""
        self.running = False
        
        # Cancel all running pipelines
        for task in self.running_pipelines.values():
            task.cancel()
        
        # Cancel all running stages
        for task in self.running_stages.values():
            task.cancel()
        
        logger.info("Pipeline manager stopped")
    
    def register_pipeline(self, pipeline: PipelineDefinition) -> bool:
        """
        Register pipeline definition.
        
        Args:
            pipeline: Pipeline definition to register
        
        Returns:
            True if pipeline was registered
        """
        try:
            # Validate pipeline
            self._validate_pipeline(pipeline)
            
            self.pipelines[pipeline.id] = pipeline
            logger.info(f"Registered pipeline: {pipeline.id} - {pipeline.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register pipeline {pipeline.id}: {e}")
            return False
    
    async def execute_pipeline(self, pipeline_id: str, 
                             input_data: Any,
                             processing_mode: Optional[ProcessingMode] = None) -> str:
        """
        Execute pipeline with input data.
        
        Args:
            pipeline_id: ID of pipeline to execute
            input_data: Input data for processing
            processing_mode: Override processing mode
        
        Returns:
            Execution ID
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_id}' not found")
        
        if len(self.running_pipelines) >= self.config.max_concurrent_pipelines:
            raise RuntimeError("Maximum concurrent pipelines reached")
        
        pipeline = self.pipelines[pipeline_id]
        execution_id = str(uuid.uuid4())
        
        # Determine processing mode
        mode = processing_mode or pipeline.processing_mode
        
        execution = PipelineExecution(
            pipeline_id=pipeline_id,
            execution_id=execution_id,
            status=PipelineStatus.CREATED,
            processing_mode=mode
        )
        
        self.executions[execution_id] = execution
        
        # Start pipeline execution
        pipeline_task = asyncio.create_task(
            self._execute_pipeline_async(execution, input_data)
        )
        self.running_pipelines[execution_id] = pipeline_task
        
        self.metrics['pipelines_executed'] += 1
        
        logger.info(f"Started pipeline execution: {pipeline_id} ({execution_id})")
        return execution_id
    
    async def cancel_pipeline(self, execution_id: str) -> bool:
        """
        Cancel pipeline execution.
        
        Args:
            execution_id: Execution ID to cancel
        
        Returns:
            True if pipeline was cancelled
        """
        try:
            if execution_id in self.running_pipelines:
                task = self.running_pipelines[execution_id]
                task.cancel()
                
                if execution_id in self.executions:
                    self.executions[execution_id].status = PipelineStatus.CANCELLED
                    self.executions[execution_id].end_time = datetime.now()
                
                logger.info(f"Cancelled pipeline execution: {execution_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel pipeline {execution_id}: {e}")
            return False
    
    def get_pipeline_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get pipeline execution status.
        
        Args:
            execution_id: Execution ID
        
        Returns:
            Pipeline status dictionary
        """
        if execution_id not in self.executions:
            return None
        
        execution = self.executions[execution_id]
        pipeline = self.pipelines.get(execution.pipeline_id)
        
        # Calculate progress
        total_stages = len(pipeline.stages) if pipeline else 0
        completed_stages = len([
            s for s in execution.stage_executions.values() 
            if s.status == StageStatus.COMPLETED
        ])
        
        # Calculate throughput
        throughput = 0.0
        if execution.start_time:
            elapsed = (datetime.now() - execution.start_time).total_seconds()
            if elapsed > 0:
                throughput = execution.total_output_items / elapsed
        
        return {
            'execution_id': execution_id,
            'pipeline_id': execution.pipeline_id,
            'pipeline_name': pipeline.name if pipeline else 'Unknown',
            'status': execution.status.value,
            'processing_mode': execution.processing_mode.value,
            'start_time': execution.start_time.isoformat() if execution.start_time else None,
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'duration_seconds': (
                (execution.end_time - execution.start_time).total_seconds()
                if execution.start_time and execution.end_time else None
            ),
            'progress': {
                'total_stages': total_stages,
                'completed_stages': completed_stages,
                'progress_percent': (completed_stages / total_stages * 100) if total_stages > 0 else 0
            },
            'statistics': {
                'total_input_items': execution.total_input_items,
                'total_output_items': execution.total_output_items,
                'total_failed_items': execution.total_failed_items,
                'success_rate': (
                    (execution.total_output_items / execution.total_input_items * 100) 
                    if execution.total_input_items > 0 else 0
                ),
                'throughput_items_per_second': throughput
            },
            'stage_statuses': {
                stage_id: execution.status.value 
                for stage_id, execution in execution.stage_executions.items()
            },
            'error': execution.error
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get pipeline manager system status.
        
        Returns:
            System status dictionary
        """
        return {
            'running': self.running,
            'registered_pipelines': len(self.pipelines),
            'running_pipelines': len(self.running_pipelines),
            'running_stages': len(self.running_stages),
            'total_executions': len(self.execution_history),
            'metrics': self.metrics.copy()
        }
    
    def add_pipeline_callback(self, pipeline_id: str, 
                            callback: Callable[[PipelineExecution], None]):
        """
        Add callback for pipeline completion.
        
        Args:
            pipeline_id: Pipeline ID or '*' for all pipelines
            callback: Callback function
        """
        if pipeline_id not in self.pipeline_callbacks:
            self.pipeline_callbacks[pipeline_id] = []
        self.pipeline_callbacks[pipeline_id].append(callback)
    
    def add_stage_callback(self, stage_id: str, 
                         callback: Callable[[StageExecution], None]):
        """
        Add callback for stage completion.
        
        Args:
            stage_id: Stage ID or '*' for all stages
            callback: Callback function
        """
        if stage_id not in self.stage_callbacks:
            self.stage_callbacks[stage_id] = []
        self.stage_callbacks[stage_id].append(callback)
    
    def _validate_pipeline(self, pipeline: PipelineDefinition):
        """Validate pipeline definition"""
        if not pipeline.stages:
            raise ValueError("Pipeline must have at least one stage")
        
        # Validate stage processors
        for stage in pipeline.stages:
            if not callable(stage.processor):
                raise ValueError(f"Stage {stage.id} processor must be callable")
    
    async def _execute_pipeline_async(self, execution: PipelineExecution, 
                                    input_data: Any):
        """Execute pipeline asynchronously"""
        try:
            execution.status = PipelineStatus.RUNNING
            execution.start_time = datetime.now()
            
            pipeline = self.pipelines[execution.pipeline_id]
            
            logger.info(f"Executing pipeline: {pipeline.name} ({execution.execution_id})")
            
            # Execute based on processing mode
            if execution.processing_mode == ProcessingMode.BATCH:
                await self._execute_batch_pipeline(execution, input_data)
            elif execution.processing_mode == ProcessingMode.STREAM:
                await self._execute_stream_pipeline(execution, input_data)
            elif execution.processing_mode == ProcessingMode.MICRO_BATCH:
                await self._execute_micro_batch_pipeline(execution, input_data)
            
            execution.status = PipelineStatus.COMPLETED
            self.metrics['pipelines_completed'] += 1
            
            logger.info(f"Pipeline completed: {pipeline.name} ({execution.execution_id})")
            
        except asyncio.CancelledError:
            execution.status = PipelineStatus.CANCELLED
            logger.info(f"Pipeline cancelled: {execution.execution_id}")
            
        except Exception as e:
            execution.status = PipelineStatus.FAILED
            execution.error = str(e)
            self.metrics['pipelines_failed'] += 1
            
            logger.error(f"Pipeline failed: {execution.execution_id} - {e}")
        
        finally:
            execution.end_time = datetime.now()
            
            # Update metrics
            if execution.start_time and execution.end_time:
                duration = (execution.end_time - execution.start_time).total_seconds()
                self.metrics['total_processing_time'] += duration
            
            self.metrics['total_items_processed'] += execution.total_output_items
            
            # Move to history
            self.execution_history.append(execution)
            
            # Clean up
            if execution.execution_id in self.running_pipelines:
                del self.running_pipelines[execution.execution_id]
            
            # Call callbacks
            await self._call_pipeline_callbacks(execution)
    
    async def _execute_batch_pipeline(self, execution: PipelineExecution, 
                                     input_data: Any):
        """Execute pipeline in batch mode"""
        pipeline = self.pipelines[execution.pipeline_id]
        
        # Convert input data to batches
        batches = self._create_batches(input_data, pipeline.default_batch_size)
        execution.total_input_items = len(input_data) if hasattr(input_data, '__len__') else 0
        
        current_data = batches
        
        # Execute stages sequentially
        for stage in pipeline.stages:
            stage_execution = StageExecution(
                stage_id=stage.id,
                execution_id=str(uuid.uuid4()),
                pipeline_id=execution.pipeline_id,
                status=StageStatus.PENDING
            )
            
            execution.stage_executions[stage.id] = stage_execution
            
            # Execute stage
            current_data = await self._execute_stage(stage, stage_execution, current_data)
            
            # Check for failures
            if stage_execution.status == StageStatus.FAILED and pipeline.error_handling == "stop":
                raise RuntimeError(f"Stage {stage.id} failed: {stage_execution.error}")
        
        # Calculate final output count
        execution.total_output_items = sum(
            len(batch.data) if hasattr(batch.data, '__len__') else 1 
            for batch in current_data
        )
    
    async def _execute_stream_pipeline(self, execution: PipelineExecution, 
                                     input_data: Any):
        """Execute pipeline in stream mode"""
        pipeline = self.pipelines[execution.pipeline_id]
        
        # Create data queues for each stage
        stage_queues = {}
        for i, stage in enumerate(pipeline.stages):
            if i == 0:
                # First stage gets input data
                queue = asyncio.Queue(maxsize=self.config.stream_buffer_size)
                stage_queues[stage.id] = queue
                
                # Feed input data to first queue
                if hasattr(input_data, '__iter__'):
                    for item in input_data:
                        await queue.put(DataBatch(
                            id=str(uuid.uuid4()),
                            data=item
                        ))
                else:
                    await queue.put(DataBatch(
                        id=str(uuid.uuid4()),
                        data=input_data
                    ))
                
                await queue.put(None)  # End marker
            else:
                stage_queues[stage.id] = asyncio.Queue(maxsize=self.config.stream_buffer_size)
        
        # Start all stages concurrently
        stage_tasks = []
        for i, stage in enumerate(pipeline.stages):
            stage_execution = StageExecution(
                stage_id=stage.id,
                execution_id=str(uuid.uuid4()),
                pipeline_id=execution.pipeline_id,
                status=StageStatus.PENDING
            )
            
            execution.stage_executions[stage.id] = stage_execution
            
            input_queue = stage_queues[stage.id]
            output_queue = stage_queues.get(pipeline.stages[i + 1].id) if i + 1 < len(pipeline.stages) else None
            
            task = asyncio.create_task(
                self._execute_stream_stage(stage, stage_execution, input_queue, output_queue)
            )
            stage_tasks.append(task)
        
        # Wait for all stages to complete
        await asyncio.gather(*stage_tasks)
    
    async def _execute_micro_batch_pipeline(self, execution: PipelineExecution, 
                                          input_data: Any):
        """Execute pipeline in micro-batch mode"""
        # Micro-batch is similar to batch but with smaller, time-based batches
        # This is a simplified implementation
        await self._execute_batch_pipeline(execution, input_data)
    
    def _create_batches(self, data: Any, batch_size: int) -> List[DataBatch]:
        """Create data batches from input data"""
        batches = []
        
        if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
            # Iterable data
            current_batch = []
            for item in data:
                current_batch.append(item)
                
                if len(current_batch) >= batch_size:
                    batches.append(DataBatch(
                        id=str(uuid.uuid4()),
                        data=current_batch,
                        size=len(current_batch)
                    ))
                    current_batch = []
            
            # Add remaining items
            if current_batch:
                batches.append(DataBatch(
                    id=str(uuid.uuid4()),
                    data=current_batch,
                    size=len(current_batch)
                ))
        else:
            # Single item
            batches.append(DataBatch(
                id=str(uuid.uuid4()),
                data=data,
                size=1
            ))
        
        return batches
    
    async def _execute_stage(self, stage: StageDefinition, 
                           stage_execution: StageExecution,
                           input_batches: List[DataBatch]) -> List[DataBatch]:
        """Execute pipeline stage"""
        try:
            stage_execution.status = StageStatus.RUNNING
            stage_execution.start_time = datetime.now()
            
            logger.info(f"Executing stage: {stage.name} ({stage.id})")
            
            output_batches = []
            
            if stage.parallel and len(input_batches) > 1:
                # Execute batches in parallel
                semaphore = asyncio.Semaphore(stage.max_workers)
                
                async def process_batch(batch: DataBatch) -> DataBatch:
                    async with semaphore:
                        return await self._process_batch(stage, batch)
                
                tasks = [process_batch(batch) for batch in input_batches]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        stage_execution.failed_batches += 1
                        if stage.on_error == "stop":
                            raise result
                    else:
                        output_batches.append(result)
                        stage_execution.processed_batches += 1
            else:
                # Execute batches sequentially
                for batch in input_batches:
                    try:
                        result = await self._process_batch(stage, batch)
                        output_batches.append(result)
                        stage_execution.processed_batches += 1
                    except Exception as e:
                        stage_execution.failed_batches += 1
                        if stage.on_error == "stop":
                            raise e
                        elif stage.on_error == "skip":
                            continue
            
            # Calculate total items processed
            stage_execution.total_items_processed = sum(
                batch.size or 1 for batch in output_batches
            )
            
            stage_execution.status = StageStatus.COMPLETED
            logger.info(f"Stage completed: {stage.name} ({stage.id})")
            
            return output_batches
            
        except Exception as e:
            stage_execution.status = StageStatus.FAILED
            stage_execution.error = str(e)
            
            logger.error(f"Stage failed: {stage.name} ({stage.id}) - {e}")
            
            # Handle retry
            if stage_execution.retry_count < stage.max_retries:
                stage_execution.retry_count += 1
                logger.info(f"Retrying stage {stage.id} (attempt {stage_execution.retry_count})")
                await asyncio.sleep(self.config.default_retry_delay_seconds)
                return await self._execute_stage(stage, stage_execution, input_batches)
            
            raise
        
        finally:
            stage_execution.end_time = datetime.now()
            
            # Call stage callbacks
            await self._call_stage_callbacks(stage_execution)
    
    async def _process_batch(self, stage: StageDefinition, 
                           batch: DataBatch) -> DataBatch:
        """Process single batch through stage"""
        try:
            # Execute stage processor
            if asyncio.iscoroutinefunction(stage.processor):
                result = await stage.processor(batch.data, **stage.config)
            else:
                # Run in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: stage.processor(batch.data, **stage.config)
                )
            
            # Create output batch
            output_batch = DataBatch(
                id=str(uuid.uuid4()),
                data=result,
                metadata={
                    **batch.metadata,
                    'processed_by': stage.id,
                    'processing_time': datetime.now().isoformat()
                },
                size=len(result) if hasattr(result, '__len__') else 1
            )
            
            return output_batch
            
        except Exception as e:
            logger.error(f"Batch processing failed in stage {stage.id}: {e}")
            raise
    
    async def _execute_stream_stage(self, stage: StageDefinition,
                                  stage_execution: StageExecution,
                                  input_queue: asyncio.Queue,
                                  output_queue: Optional[asyncio.Queue]):
        """Execute stage in streaming mode"""
        try:
            stage_execution.status = StageStatus.RUNNING
            stage_execution.start_time = datetime.now()
            
            logger.info(f"Executing stream stage: {stage.name} ({stage.id})")
            
            while True:
                # Get batch from input queue
                batch = await input_queue.get()
                
                if batch is None:  # End marker
                    if output_queue:
                        await output_queue.put(None)
                    break
                
                try:
                    # Process batch
                    result_batch = await self._process_batch(stage, batch)
                    
                    # Send to output queue
                    if output_queue:
                        await output_queue.put(result_batch)
                    
                    stage_execution.processed_batches += 1
                    stage_execution.total_items_processed += batch.size or 1
                    
                except Exception as e:
                    stage_execution.failed_batches += 1
                    if stage.on_error == "stop":
                        raise e
            
            stage_execution.status = StageStatus.COMPLETED
            logger.info(f"Stream stage completed: {stage.name} ({stage.id})")
            
        except Exception as e:
            stage_execution.status = StageStatus.FAILED
            stage_execution.error = str(e)
            logger.error(f"Stream stage failed: {stage.name} ({stage.id}) - {e}")
        
        finally:
            stage_execution.end_time = datetime.now()
            await self._call_stage_callbacks(stage_execution)
    
    async def _call_pipeline_callbacks(self, execution: PipelineExecution):
        """Call pipeline callbacks"""
        try:
            # Pipeline-specific callbacks
            if execution.pipeline_id in self.pipeline_callbacks:
                for callback in self.pipeline_callbacks[execution.pipeline_id]:
                    try:
                        await callback(execution)
                    except Exception as e:
                        logger.error(f"Error in pipeline callback: {e}")
            
            # Global callbacks
            if '*' in self.pipeline_callbacks:
                for callback in self.pipeline_callbacks['*']:
                    try:
                        await callback(execution)
                    except Exception as e:
                        logger.error(f"Error in global pipeline callback: {e}")
        
        except Exception as e:
            logger.error(f"Failed to call pipeline callbacks: {e}")
    
    async def _call_stage_callbacks(self, stage_execution: StageExecution):
        """Call stage callbacks"""
        try:
            # Stage-specific callbacks
            if stage_execution.stage_id in self.stage_callbacks:
                for callback in self.stage_callbacks[stage_execution.stage_id]:
                    try:
                        await callback(stage_execution)
                    except Exception as e:
                        logger.error(f"Error in stage callback: {e}")
            
            # Global callbacks
            if '*' in self.stage_callbacks:
                for callback in self.stage_callbacks['*']:
                    try:
                        await callback(stage_execution)
                    except Exception as e:
                        logger.error(f"Error in global stage callback: {e}")
        
        except Exception as e:
            logger.error(f"Failed to call stage callbacks: {e}")
    
    async def _cleanup_worker(self):
        """Background worker for cleanup"""
        while self.running:
            try:
                await self._cleanup_old_executions()
                await asyncio.sleep(24 * 3600)  # Daily cleanup
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    async def _cleanup_old_executions(self):
        """Clean up old pipeline executions"""
        try:
            cutoff = datetime.now() - timedelta(days=self.config.execution_retention_days)
            
            original_count = len(self.execution_history)
            self.execution_history = [
                ex for ex in self.execution_history 
                if not ex.end_time or ex.end_time >= cutoff
            ]
            
            cleaned_count = original_count - len(self.execution_history)
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old pipeline executions")
        
        except Exception as e:
            logger.error(f"Failed to cleanup old executions: {e}")
    
    async def _metrics_worker(self):
        """Background worker for metrics collection"""
        while self.running:
            try:
                # Collect and export metrics
                await asyncio.sleep(self.config.metrics_collection_interval_seconds)
            except Exception as e:
                logger.error(f"Error in metrics worker: {e}")


def create_pipeline_manager(config: Optional[PipelineManagerConfig] = None) -> PipelineManager:
    """Factory function to create pipeline manager"""
    if config is None:
        config = PipelineManagerConfig()
    return PipelineManager(config)


__all__ = [
    'PipelineManager',
    'PipelineManagerConfig',
    'PipelineDefinition',
    'StageDefinition',
    'PipelineExecution',
    'StageExecution',
    'DataBatch',
    'PipelineStatus',
    'StageStatus',
    'ProcessingMode',
    'create_pipeline_manager'
]