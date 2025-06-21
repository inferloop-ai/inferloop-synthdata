#!/usr/bin/env python3
"""
Pipeline Coordinator for managing end-to-end document generation pipelines.

Provides high-level coordination of all orchestration components to create
seamless document generation pipelines with monitoring and optimization.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import threading
import json
from pathlib import Path

from pydantic import BaseModel, Field

from ..core.config import get_config
from ..core.logging import get_logger
from ..core.exceptions import ProcessingError, ValidationError
from .job_queue import JobQueue, Job, JobPriority
from .resource_manager import ResourceManager, ResourceType
from .scheduler import Scheduler, SchedulingStrategy
from .state_manager import StateManager, StateType
from .task_manager import TaskManager, TaskType
from .workflow_manager import WorkflowManager, WorkflowStatus


logger = get_logger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages"""
    INITIALIZATION = "initialization"
    DATA_INGESTION = "data_ingestion"
    TEMPLATE_SELECTION = "template_selection"
    DOCUMENT_GENERATION = "document_generation"
    OCR_PROCESSING = "ocr_processing"
    VALIDATION = "validation"
    PRIVACY_ENFORCEMENT = "privacy_enforcement"
    EXPORT = "export"
    DELIVERY = "delivery"
    CLEANUP = "cleanup"


class PipelineStatus(Enum):
    """Pipeline execution status"""
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    total_duration: float = 0.0
    stage_durations: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    throughput: float = 0.0
    error_rate: float = 0.0


@dataclass
class PipelineDefinition:
    """Pipeline definition"""
    pipeline_id: str
    name: str
    description: Optional[str] = None
    stages: List[PipelineStage] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    workflow_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineInstance:
    """Running pipeline instance"""
    instance_id: str
    pipeline_id: str
    status: PipelineStatus
    current_stage: Optional[PipelineStage] = None
    completed_stages: List[PipelineStage] = field(default_factory=list)
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class PipelineCoordinatorConfig(BaseModel):
    """Pipeline coordinator configuration"""
    # Pipeline settings
    max_concurrent_pipelines: int = Field(10, description="Maximum concurrent pipelines")
    default_batch_size: int = Field(100, description="Default document batch size")
    stage_timeout: float = Field(3600.0, description="Default stage timeout in seconds")
    
    # Optimization settings
    enable_optimization: bool = Field(True, description="Enable pipeline optimization")
    optimization_interval: float = Field(300.0, description="Optimization interval in seconds")
    
    # Monitoring settings
    enable_monitoring: bool = Field(True, description="Enable pipeline monitoring")
    metrics_interval: float = Field(30.0, description="Metrics collection interval")
    
    # Recovery settings
    enable_recovery: bool = Field(True, description="Enable failure recovery")
    recovery_attempts: int = Field(3, description="Maximum recovery attempts")
    
    # Persistence settings
    enable_persistence: bool = Field(True, description="Enable pipeline persistence")
    persistence_path: str = Field("./pipelines", description="Pipeline storage path")


class PipelineCoordinator:
    """
    Master coordinator for document generation pipelines.
    
    Features:
    - End-to-end pipeline management
    - Component orchestration
    - Performance monitoring
    - Resource optimization
    - Failure recovery
    - Pipeline templates
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize pipeline coordinator"""
        self.config = PipelineCoordinatorConfig(**(config or {}))
        self.logger = get_logger(__name__)
        
        # Initialize orchestration components
        self.job_queue = JobQueue()
        self.resource_manager = ResourceManager()
        self.state_manager = StateManager()
        self.scheduler = Scheduler(
            resource_manager=self.resource_manager
        )
        self.task_manager = TaskManager(
            job_queue=self.job_queue,
            state_manager=self.state_manager,
            resource_manager=self.resource_manager
        )
        self.workflow_manager = WorkflowManager(
            task_manager=self.task_manager,
            state_manager=self.state_manager,
            resource_manager=self.resource_manager
        )
        
        # Pipeline storage
        self.pipeline_definitions: Dict[str, PipelineDefinition] = {}
        self.pipeline_instances: Dict[str, PipelineInstance] = {}
        self.pipeline_templates: Dict[str, Dict[str, Any]] = {}
        
        # Execution tracking
        self.running_pipelines: Dict[str, asyncio.Task] = {}
        self.pipeline_locks: Dict[str, threading.Lock] = {}
        
        # Monitoring
        self.metrics_history: defaultdict = defaultdict(list)
        self.optimization_thread: Optional[threading.Thread] = None
        
        # Control
        self.is_running = True
        self._lock = threading.RLock()
        
        # Initialize components
        self._initialize_components()
        
        # Load templates
        self._load_pipeline_templates()
        
        # Create persistence directory
        if self.config.enable_persistence:
            self.persistence_path = Path(self.config.persistence_path)
            self.persistence_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Pipeline coordinator initialized")
    
    def _initialize_components(self):
        """Initialize all orchestration components"""
        # Start job queue
        self.job_queue.start()
        
        # Start resource monitoring
        self.resource_manager.start_monitoring()
        
        # Start scheduler
        self.scheduler.start()
        
        # Register task functions
        self._register_task_functions()
        
        # Start optimization thread
        if self.config.enable_optimization:
            self.optimization_thread = threading.Thread(
                target=self._optimization_loop,
                name="PipelineOptimizer",
                daemon=True
            )
            self.optimization_thread.start()
    
    def _register_task_functions(self):
        """Register pipeline task functions"""
        self.task_manager.register_function("ingest_data", self._task_ingest_data)
        self.task_manager.register_function("select_template", self._task_select_template)
        self.task_manager.register_function("generate_document", self._task_generate_document)
        self.task_manager.register_function("process_ocr", self._task_process_ocr)
        self.task_manager.register_function("validate_document", self._task_validate_document)
        self.task_manager.register_function("enforce_privacy", self._task_enforce_privacy)
        self.task_manager.register_function("export_document", self._task_export_document)
        self.task_manager.register_function("deliver_document", self._task_deliver_document)
    
    def _load_pipeline_templates(self):
        """Load predefined pipeline templates"""
        self.pipeline_templates = {
            "standard": {
                "name": "Standard Document Generation",
                "stages": [
                    PipelineStage.INITIALIZATION,
                    PipelineStage.DATA_INGESTION,
                    PipelineStage.TEMPLATE_SELECTION,
                    PipelineStage.DOCUMENT_GENERATION,
                    PipelineStage.VALIDATION,
                    PipelineStage.EXPORT,
                    PipelineStage.CLEANUP
                ]
            },
            "ocr_enhanced": {
                "name": "OCR-Enhanced Document Generation",
                "stages": [
                    PipelineStage.INITIALIZATION,
                    PipelineStage.DATA_INGESTION,
                    PipelineStage.TEMPLATE_SELECTION,
                    PipelineStage.DOCUMENT_GENERATION,
                    PipelineStage.OCR_PROCESSING,
                    PipelineStage.VALIDATION,
                    PipelineStage.EXPORT,
                    PipelineStage.CLEANUP
                ]
            },
            "privacy_compliant": {
                "name": "Privacy-Compliant Document Generation",
                "stages": [
                    PipelineStage.INITIALIZATION,
                    PipelineStage.DATA_INGESTION,
                    PipelineStage.PRIVACY_ENFORCEMENT,
                    PipelineStage.TEMPLATE_SELECTION,
                    PipelineStage.DOCUMENT_GENERATION,
                    PipelineStage.VALIDATION,
                    PipelineStage.PRIVACY_ENFORCEMENT,
                    PipelineStage.EXPORT,
                    PipelineStage.DELIVERY,
                    PipelineStage.CLEANUP
                ]
            }
        }
    
    def create_pipeline(
        self,
        name: str,
        template: Optional[str] = None,
        stages: Optional[List[PipelineStage]] = None,
        configuration: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new pipeline definition.
        
        Args:
            name: Pipeline name
            template: Template name to use
            stages: Custom stage list (if not using template)
            configuration: Pipeline configuration
            workflow_id: Associated workflow ID
            description: Pipeline description
            metadata: Additional metadata
            
        Returns:
            Pipeline ID
        """
        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name.lower().replace(' ', '_')}"
        
        # Get stages from template or custom
        if template and template in self.pipeline_templates:
            template_def = self.pipeline_templates[template]
            pipeline_stages = template_def["stages"]
            if not description:
                description = template_def["name"]
        elif stages:
            pipeline_stages = stages
        else:
            pipeline_stages = self.pipeline_templates["standard"]["stages"]
        
        # Create pipeline definition
        pipeline = PipelineDefinition(
            pipeline_id=pipeline_id,
            name=name,
            description=description,
            stages=pipeline_stages,
            configuration=configuration or {},
            workflow_id=workflow_id,
            metadata=metadata or {}
        )
        
        # Create associated workflow if not provided
        if not workflow_id:
            workflow_id = self._create_pipeline_workflow(pipeline)
            pipeline.workflow_id = workflow_id
        
        # Store pipeline
        with self._lock:
            self.pipeline_definitions[pipeline_id] = pipeline
        
        # Persist if enabled
        if self.config.enable_persistence:
            self._save_pipeline(pipeline)
        
        self.logger.info(f"Pipeline created: {pipeline_id} - {name}")
        
        return pipeline_id
    
    def _create_pipeline_workflow(self, pipeline: PipelineDefinition) -> str:
        """Create workflow for pipeline"""
        steps = []
        
        for i, stage in enumerate(pipeline.stages):
            step = {
                "id": f"stage_{stage.value}",
                "name": stage.value.replace("_", " ").title(),
                "type": "task",
                "config": {
                    "function": f"execute_{stage.value}",
                    "task_type": "DOCUMENT_GENERATION",
                    "timeout": self.config.stage_timeout
                },
                "next_steps": [f"stage_{pipeline.stages[i+1].value}"] if i < len(pipeline.stages) - 1 else []
            }
            steps.append(step)
        
        # Register stage execution functions
        for stage in pipeline.stages:
            self.task_manager.register_function(
                f"execute_{stage.value}",
                lambda s=stage: self._execute_stage(pipeline.pipeline_id, s)
            )
        
        # Create workflow
        workflow_id = self.workflow_manager.define_workflow(
            name=f"{pipeline.name} Workflow",
            version="1.0",
            steps=steps,
            start_step=f"stage_{pipeline.stages[0].value}" if pipeline.stages else None,
            description=f"Workflow for {pipeline.name}",
            metadata={"pipeline_id": pipeline.pipeline_id}
        )
        
        return workflow_id
    
    def start_pipeline(
        self,
        pipeline_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a pipeline instance.
        
        Args:
            pipeline_id: Pipeline definition ID
            input_data: Input data for the pipeline
            parameters: Pipeline parameters
            
        Returns:
            Instance ID
        """
        if pipeline_id not in self.pipeline_definitions:
            raise ValidationError(f"Pipeline not found: {pipeline_id}")
        
        pipeline = self.pipeline_definitions[pipeline_id]
        
        # Create instance
        instance_id = f"instance_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{pipeline_id}"
        instance = PipelineInstance(
            instance_id=instance_id,
            pipeline_id=pipeline_id,
            status=PipelineStatus.CREATED,
            context={
                "input_data": input_data or {},
                "parameters": parameters or {},
                "configuration": pipeline.configuration
            }
        )
        
        with self._lock:
            self.pipeline_instances[instance_id] = instance
            self.pipeline_locks[instance_id] = threading.Lock()
        
        # Start execution
        if asyncio.get_event_loop().is_running():
            task = asyncio.create_task(self._execute_pipeline(instance_id))
        else:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            task = loop.create_task(self._execute_pipeline(instance_id))
        
        self.running_pipelines[instance_id] = task
        
        self.logger.info(f"Pipeline started: {instance_id} ({pipeline.name})")
        
        return instance_id
    
    async def _execute_pipeline(self, instance_id: str):
        """Execute pipeline instance"""
        instance = self.pipeline_instances[instance_id]
        pipeline = self.pipeline_definitions[instance.pipeline_id]
        
        try:
            # Initialize
            instance.status = PipelineStatus.INITIALIZED
            instance.start_time = datetime.now()
            
            # Check resources
            if not await self._check_resources():
                raise ProcessingError("Insufficient resources to start pipeline")
            
            # Start workflow if defined
            if pipeline.workflow_id:
                workflow_instance_id = self.workflow_manager.start_workflow(
                    workflow_id=pipeline.workflow_id,
                    parameters=instance.context.get("parameters", {}),
                    context=instance.context
                )
                
                # Monitor workflow
                await self._monitor_workflow(instance_id, workflow_instance_id)
            else:
                # Execute stages manually
                instance.status = PipelineStatus.RUNNING
                
                for stage in pipeline.stages:
                    instance.current_stage = stage
                    stage_start = time.time()
                    
                    try:
                        await self._execute_stage(instance_id, stage)
                        
                        instance.completed_stages.append(stage)
                        instance.metrics.stage_durations[stage.value] = time.time() - stage_start
                        
                    except Exception as e:
                        self.logger.error(f"Stage {stage.value} failed: {e}")
                        
                        if self.config.enable_recovery:
                            if not await self._recover_stage(instance_id, stage):
                                raise
                        else:
                            raise
            
            # Complete pipeline
            instance.status = PipelineStatus.COMPLETED
            instance.end_time = datetime.now()
            instance.metrics.total_duration = (instance.end_time - instance.start_time).total_seconds()
            
            # Calculate final metrics
            self._calculate_pipeline_metrics(instance)
            
        except Exception as e:
            instance.status = PipelineStatus.FAILED
            instance.error = str(e)
            instance.end_time = datetime.now()
            self.logger.error(f"Pipeline {instance_id} failed: {e}")
        
        finally:
            # Clean up
            del self.running_pipelines[instance_id]
            
            # Store metrics
            if self.config.enable_monitoring:
                self.metrics_history[instance.pipeline_id].append({
                    "instance_id": instance_id,
                    "metrics": instance.metrics,
                    "timestamp": datetime.now()
                })
            
            # Persist state
            if self.state_manager:
                self.state_manager.set(
                    f"pipeline_instance:{instance_id}",
                    instance,
                    StateType.JOB
                )
    
    async def _execute_stage(self, pipeline_id: str, stage: PipelineStage) -> Any:
        """Execute a pipeline stage"""
        self.logger.info(f"Executing stage: {stage.value}")
        
        # Simulate stage execution
        if stage == PipelineStage.INITIALIZATION:
            return await self._stage_initialization(pipeline_id)
        elif stage == PipelineStage.DATA_INGESTION:
            return await self._stage_data_ingestion(pipeline_id)
        elif stage == PipelineStage.TEMPLATE_SELECTION:
            return await self._stage_template_selection(pipeline_id)
        elif stage == PipelineStage.DOCUMENT_GENERATION:
            return await self._stage_document_generation(pipeline_id)
        elif stage == PipelineStage.OCR_PROCESSING:
            return await self._stage_ocr_processing(pipeline_id)
        elif stage == PipelineStage.VALIDATION:
            return await self._stage_validation(pipeline_id)
        elif stage == PipelineStage.PRIVACY_ENFORCEMENT:
            return await self._stage_privacy_enforcement(pipeline_id)
        elif stage == PipelineStage.EXPORT:
            return await self._stage_export(pipeline_id)
        elif stage == PipelineStage.DELIVERY:
            return await self._stage_delivery(pipeline_id)
        elif stage == PipelineStage.CLEANUP:
            return await self._stage_cleanup(pipeline_id)
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    async def _stage_initialization(self, instance_id: str) -> Dict[str, Any]:
        """Initialize pipeline execution"""
        instance = self.pipeline_instances[instance_id]
        
        # Set up working directories
        work_dir = Path(f"./work/{instance_id}")
        work_dir.mkdir(parents=True, exist_ok=True)
        
        instance.context["work_dir"] = str(work_dir)
        
        return {"status": "initialized", "work_dir": str(work_dir)}
    
    async def _stage_data_ingestion(self, instance_id: str) -> Dict[str, Any]:
        """Ingest data for processing"""
        instance = self.pipeline_instances[instance_id]
        
        # Simulate data ingestion
        input_data = instance.context.get("input_data", {})
        
        # Process input data
        processed_data = {
            "records": input_data.get("records", []),
            "count": len(input_data.get("records", [])),
            "metadata": input_data.get("metadata", {})
        }
        
        instance.context["processed_data"] = processed_data
        instance.metrics.total_documents = processed_data["count"]
        
        return {"status": "ingested", "record_count": processed_data["count"]}
    
    async def _stage_template_selection(self, instance_id: str) -> Dict[str, Any]:
        """Select document templates"""
        instance = self.pipeline_instances[instance_id]
        
        # Simulate template selection
        template_type = instance.context.get("parameters", {}).get("template_type", "default")
        
        selected_templates = {
            "main_template": f"template_{template_type}",
            "sub_templates": [f"sub_template_{i}" for i in range(3)]
        }
        
        instance.context["templates"] = selected_templates
        
        return {"status": "selected", "templates": selected_templates}
    
    async def _stage_document_generation(self, instance_id: str) -> Dict[str, Any]:
        """Generate documents"""
        instance = self.pipeline_instances[instance_id]
        
        # Simulate document generation
        processed_data = instance.context.get("processed_data", {})
        templates = instance.context.get("templates", {})
        
        generated_docs = []
        for i in range(processed_data.get("count", 0)):
            doc = {
                "id": f"doc_{i}",
                "template": templates.get("main_template"),
                "content": f"Generated content for document {i}",
                "metadata": {"index": i}
            }
            generated_docs.append(doc)
            
            instance.metrics.processed_documents += 1
            
            # Simulate processing time
            await asyncio.sleep(0.1)
        
        instance.context["generated_documents"] = generated_docs
        
        return {"status": "generated", "document_count": len(generated_docs)}
    
    async def _stage_ocr_processing(self, instance_id: str) -> Dict[str, Any]:
        """Process documents with OCR"""
        instance = self.pipeline_instances[instance_id]
        
        # Simulate OCR processing
        documents = instance.context.get("generated_documents", [])
        
        ocr_results = []
        for doc in documents:
            result = {
                "document_id": doc["id"],
                "ocr_text": f"OCR extracted text for {doc['id']}",
                "confidence": 0.95
            }
            ocr_results.append(result)
        
        instance.context["ocr_results"] = ocr_results
        
        return {"status": "processed", "ocr_count": len(ocr_results)}
    
    async def _stage_validation(self, instance_id: str) -> Dict[str, Any]:
        """Validate generated documents"""
        instance = self.pipeline_instances[instance_id]
        
        # Simulate validation
        documents = instance.context.get("generated_documents", [])
        
        validation_results = []
        for doc in documents:
            result = {
                "document_id": doc["id"],
                "valid": True,
                "issues": []
            }
            validation_results.append(result)
        
        instance.context["validation_results"] = validation_results
        
        return {"status": "validated", "valid_count": len(validation_results)}
    
    async def _stage_privacy_enforcement(self, instance_id: str) -> Dict[str, Any]:
        """Enforce privacy rules"""
        instance = self.pipeline_instances[instance_id]
        
        # Simulate privacy enforcement
        documents = instance.context.get("generated_documents", [])
        
        privacy_results = []
        for doc in documents:
            result = {
                "document_id": doc["id"],
                "privacy_applied": True,
                "redactions": 0
            }
            privacy_results.append(result)
        
        instance.context["privacy_results"] = privacy_results
        
        return {"status": "privacy_enforced", "processed_count": len(privacy_results)}
    
    async def _stage_export(self, instance_id: str) -> Dict[str, Any]:
        """Export documents"""
        instance = self.pipeline_instances[instance_id]
        
        # Simulate export
        documents = instance.context.get("generated_documents", [])
        work_dir = Path(instance.context.get("work_dir", "."))
        
        export_paths = []
        for doc in documents:
            export_path = work_dir / f"{doc['id']}.pdf"
            export_paths.append(str(export_path))
        
        instance.context["export_paths"] = export_paths
        
        return {"status": "exported", "export_count": len(export_paths)}
    
    async def _stage_delivery(self, instance_id: str) -> Dict[str, Any]:
        """Deliver documents"""
        instance = self.pipeline_instances[instance_id]
        
        # Simulate delivery
        export_paths = instance.context.get("export_paths", [])
        
        delivery_results = []
        for path in export_paths:
            result = {
                "path": path,
                "delivered": True,
                "timestamp": datetime.now().isoformat()
            }
            delivery_results.append(result)
        
        instance.context["delivery_results"] = delivery_results
        
        return {"status": "delivered", "delivery_count": len(delivery_results)}
    
    async def _stage_cleanup(self, instance_id: str) -> Dict[str, Any]:
        """Clean up resources"""
        instance = self.pipeline_instances[instance_id]
        
        # Simulate cleanup
        work_dir = Path(instance.context.get("work_dir", "."))
        
        # In real implementation, would clean up files and resources
        
        return {"status": "cleaned", "work_dir": str(work_dir)}
    
    async def _check_resources(self) -> bool:
        """Check if resources are available"""
        if not self.resource_manager:
            return True
        
        # Check CPU and memory
        required_resources = {
            ResourceType.CPU: 20.0,  # 20% CPU
            ResourceType.MEMORY: 10.0  # 10% memory
        }
        
        for resource_type, required in required_resources.items():
            if not self.resource_manager.is_resource_available(resource_type, required):
                return False
        
        return True
    
    async def _monitor_workflow(self, pipeline_instance_id: str, workflow_instance_id: str):
        """Monitor workflow execution"""
        pipeline_instance = self.pipeline_instances[pipeline_instance_id]
        
        while True:
            workflow_instance = self.workflow_manager.get_workflow_instance(workflow_instance_id)
            
            if not workflow_instance:
                break
            
            # Update pipeline status based on workflow
            if workflow_instance.status == WorkflowStatus.RUNNING:
                pipeline_instance.status = PipelineStatus.RUNNING
            elif workflow_instance.status == WorkflowStatus.COMPLETED:
                pipeline_instance.status = PipelineStatus.COMPLETED
                break
            elif workflow_instance.status == WorkflowStatus.FAILED:
                pipeline_instance.status = PipelineStatus.FAILED
                pipeline_instance.error = workflow_instance.error
                break
            elif workflow_instance.status == WorkflowStatus.CANCELLED:
                pipeline_instance.status = PipelineStatus.CANCELLED
                break
            
            # Update metrics
            pipeline_instance.metrics.processed_documents = len(workflow_instance.completed_steps)
            
            await asyncio.sleep(1)
    
    async def _recover_stage(self, instance_id: str, stage: PipelineStage) -> bool:
        """Attempt to recover from stage failure"""
        instance = self.pipeline_instances[instance_id]
        
        for attempt in range(self.config.recovery_attempts):
            try:
                self.logger.info(f"Recovery attempt {attempt + 1} for stage {stage.value}")
                
                # Wait before retry
                await asyncio.sleep(30 * (attempt + 1))
                
                # Retry stage
                await self._execute_stage(instance_id, stage)
                
                return True
                
            except Exception as e:
                self.logger.error(f"Recovery attempt {attempt + 1} failed: {e}")
        
        return False
    
    def _calculate_pipeline_metrics(self, instance: PipelineInstance):
        """Calculate final pipeline metrics"""
        if instance.metrics.total_documents > 0:
            instance.metrics.throughput = (
                instance.metrics.processed_documents / instance.metrics.total_duration
            )
            instance.metrics.error_rate = (
                instance.metrics.failed_documents / instance.metrics.total_documents
            )
        
        # Get resource usage
        if self.resource_manager:
            current_metrics = self.resource_manager.get_current_metrics()
            for resource_type, metric in current_metrics.items():
                instance.metrics.resource_usage[resource_type.value] = metric.usage_percent
    
    def _optimization_loop(self):
        """Background optimization loop"""
        while self.is_running:
            try:
                time.sleep(self.config.optimization_interval)
                
                # Analyze metrics
                self._analyze_pipeline_performance()
                
                # Optimize resource allocation
                self._optimize_resource_allocation()
                
                # Adjust scheduling strategy
                self._adjust_scheduling_strategy()
                
            except Exception as e:
                self.logger.error(f"Optimization error: {e}")
    
    def _analyze_pipeline_performance(self):
        """Analyze pipeline performance metrics"""
        with self._lock:
            for pipeline_id, history in self.metrics_history.items():
                if not history:
                    continue
                
                # Calculate average metrics
                recent_metrics = history[-10:]  # Last 10 runs
                
                avg_duration = sum(m["metrics"].total_duration for m in recent_metrics) / len(recent_metrics)
                avg_throughput = sum(m["metrics"].throughput for m in recent_metrics) / len(recent_metrics)
                
                self.logger.info(f"Pipeline {pipeline_id}: avg_duration={avg_duration:.2f}s, avg_throughput={avg_throughput:.2f}")
    
    def _optimize_resource_allocation(self):
        """Optimize resource allocation based on usage"""
        if not self.resource_manager:
            return
        
        # Get current resource usage
        current_metrics = self.resource_manager.get_current_metrics()
        
        # Adjust limits based on usage patterns
        for resource_type, metric in current_metrics.items():
            if metric.usage_percent > 80:
                self.logger.warning(f"High {resource_type.value} usage: {metric.usage_percent:.1f}%")
            elif metric.usage_percent < 20:
                self.logger.info(f"Low {resource_type.value} usage: {metric.usage_percent:.1f}%")
    
    def _adjust_scheduling_strategy(self):
        """Adjust scheduling strategy based on workload"""
        if not self.scheduler:
            return
        
        stats = self.scheduler.get_scheduler_stats()
        
        # Switch strategy based on queue size
        if stats["queued_tasks"] > 100:
            # High load - switch to resource-aware
            self.scheduler.config.strategy = SchedulingStrategy.RESOURCE_AWARE
        elif stats["queued_tasks"] < 10:
            # Low load - switch to priority
            self.scheduler.config.strategy = SchedulingStrategy.PRIORITY
    
    # Task functions for pipeline stages
    async def _task_ingest_data(self, **kwargs):
        """Task: Ingest data"""
        return {"status": "ingested", "records": 100}
    
    async def _task_select_template(self, **kwargs):
        """Task: Select template"""
        return {"status": "selected", "template": "standard"}
    
    async def _task_generate_document(self, **kwargs):
        """Task: Generate document"""
        return {"status": "generated", "document_id": "doc_123"}
    
    async def _task_process_ocr(self, **kwargs):
        """Task: Process OCR"""
        return {"status": "processed", "confidence": 0.95}
    
    async def _task_validate_document(self, **kwargs):
        """Task: Validate document"""
        return {"status": "validated", "valid": True}
    
    async def _task_enforce_privacy(self, **kwargs):
        """Task: Enforce privacy"""
        return {"status": "enforced", "redactions": 5}
    
    async def _task_export_document(self, **kwargs):
        """Task: Export document"""
        return {"status": "exported", "path": "/exports/doc.pdf"}
    
    async def _task_deliver_document(self, **kwargs):
        """Task: Deliver document"""
        return {"status": "delivered", "timestamp": datetime.now().isoformat()}
    
    def get_pipeline_status(self, instance_id: str) -> Optional[PipelineStatus]:
        """Get pipeline status"""
        with self._lock:
            if instance_id in self.pipeline_instances:
                return self.pipeline_instances[instance_id].status
        return None
    
    def get_pipeline_metrics(self, instance_id: str) -> Optional[PipelineMetrics]:
        """Get pipeline metrics"""
        with self._lock:
            if instance_id in self.pipeline_instances:
                return self.pipeline_instances[instance_id].metrics
        return None
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all pipeline definitions"""
        with self._lock:
            return [
                {
                    "pipeline_id": p.pipeline_id,
                    "name": p.name,
                    "description": p.description,
                    "stages": [s.value for s in p.stages]
                }
                for p in self.pipeline_definitions.values()
            ]
    
    def list_instances(
        self,
        pipeline_id: Optional[str] = None,
        status: Optional[PipelineStatus] = None
    ) -> List[Dict[str, Any]]:
        """List pipeline instances"""
        with self._lock:
            instances = []
            for instance in self.pipeline_instances.values():
                if pipeline_id and instance.pipeline_id != pipeline_id:
                    continue
                if status and instance.status != status:
                    continue
                
                instances.append({
                    "instance_id": instance.instance_id,
                    "pipeline_id": instance.pipeline_id,
                    "status": instance.status.value,
                    "current_stage": instance.current_stage.value if instance.current_stage else None,
                    "progress": len(instance.completed_stages),
                    "metrics": {
                        "processed": instance.metrics.processed_documents,
                        "total": instance.metrics.total_documents,
                        "throughput": instance.metrics.throughput
                    }
                })
            
            return instances
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        with self._lock:
            return {
                "total_pipelines": len(self.pipeline_definitions),
                "total_instances": len(self.pipeline_instances),
                "running_instances": len(self.running_pipelines),
                "job_queue_stats": self.job_queue.get_queue_stats(),
                "scheduler_stats": self.scheduler.get_scheduler_stats(),
                "resource_stats": self.resource_manager.get_resource_summary(),
                "workflow_stats": self.workflow_manager.get_stats()
            }
    
    def _save_pipeline(self, pipeline: PipelineDefinition):
        """Save pipeline definition"""
        pipeline_file = self.persistence_path / f"{pipeline.pipeline_id}.json"
        with open(pipeline_file, 'w') as f:
            json.dump({
                "pipeline_id": pipeline.pipeline_id,
                "name": pipeline.name,
                "description": pipeline.description,
                "stages": [s.value for s in pipeline.stages],
                "configuration": pipeline.configuration,
                "workflow_id": pipeline.workflow_id,
                "metadata": pipeline.metadata
            }, f, indent=2)
    
    def cleanup(self):
        """Clean up coordinator resources"""
        self.is_running = False
        
        # Stop optimization thread
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5.0)
        
        # Clean up components
        self.workflow_manager.cleanup()
        self.task_manager.cleanup()
        self.scheduler.stop()
        self.state_manager.cleanup()
        self.resource_manager.cleanup()
        self.job_queue.stop()
        
        self.logger.info("Pipeline coordinator cleaned up")


# Factory function
def create_pipeline_coordinator(
    config: Optional[Dict[str, Any]] = None
) -> PipelineCoordinator:
    """Create and return a pipeline coordinator instance"""
    return PipelineCoordinator(config)