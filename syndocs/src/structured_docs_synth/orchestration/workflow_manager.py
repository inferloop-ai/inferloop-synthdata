#!/usr/bin/env python3
"""
Workflow Manager for orchestrating complex document generation pipelines.

Provides workflow definition, execution, monitoring, and management capabilities
with support for conditional logic, loops, and parallel execution branches.
"""

import asyncio
import time
import uuid
import yaml
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable, Set, Tuple, Type
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import threading
from pathlib import Path

from pydantic import BaseModel, Field

from ..core.config import get_config
from ..core.logging import get_logger
from ..core.exceptions import ProcessingError, ValidationError
from .task_manager import TaskManager, Task, TaskType, TaskStatus, TaskResult
from .state_manager import StateManager, StateType
from .resource_manager import ResourceManager


logger = get_logger(__name__)


class WorkflowStepType(Enum):
    """Types of workflow steps"""
    TASK = "task"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    SUB_WORKFLOW = "sub_workflow"
    WAIT = "wait"
    NOTIFICATION = "notification"


class WorkflowStatus(Enum):
    """Workflow execution status"""
    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """Individual workflow step"""
    step_id: str
    name: str
    step_type: WorkflowStepType
    config: Dict[str, Any]
    next_steps: List[str] = field(default_factory=list)
    error_handler: Optional[str] = None
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    """Workflow definition"""
    workflow_id: str
    name: str
    version: str
    description: Optional[str] = None
    steps: Dict[str, WorkflowStep] = field(default_factory=dict)
    start_step: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowInstance:
    """Running workflow instance"""
    instance_id: str
    workflow_id: str
    status: WorkflowStatus
    parameters: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)
    current_steps: Set[str] = field(default_factory=set)
    completed_steps: Set[str] = field(default_factory=set)
    step_results: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    error: Optional[str] = None


class WorkflowManagerConfig(BaseModel):
    """Workflow manager configuration"""
    # Execution settings
    max_concurrent_workflows: int = Field(20, description="Maximum concurrent workflows")
    max_concurrent_steps: int = Field(50, description="Maximum concurrent steps")
    default_timeout: float = Field(3600.0, description="Default workflow timeout")
    
    # Retry settings
    enable_retries: bool = Field(True, description="Enable automatic retries")
    max_retries: int = Field(3, description="Maximum retry attempts")
    retry_delay: float = Field(60.0, description="Retry delay in seconds")
    
    # Persistence settings
    enable_persistence: bool = Field(True, description="Enable workflow persistence")
    persistence_path: str = Field("./workflows", description="Workflow storage path")
    
    # Monitoring settings
    enable_monitoring: bool = Field(True, description="Enable workflow monitoring")
    metrics_interval: float = Field(30.0, description="Metrics collection interval")
    
    # History settings
    keep_history: bool = Field(True, description="Keep workflow execution history")
    history_retention_days: int = Field(30, description="History retention period")


class WorkflowManager:
    """
    Advanced workflow manager for complex document generation pipelines.
    
    Features:
    - Workflow definition and versioning
    - Conditional execution
    - Loop support
    - Parallel branches
    - Sub-workflows
    - Error handling
    - State persistence
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        task_manager: Optional[TaskManager] = None,
        state_manager: Optional[StateManager] = None,
        resource_manager: Optional[ResourceManager] = None
    ):
        """Initialize workflow manager"""
        self.config = WorkflowManagerConfig(**(config or {}))
        self.logger = get_logger(__name__)
        
        # External managers
        self.task_manager = task_manager or TaskManager()
        self.state_manager = state_manager
        self.resource_manager = resource_manager
        
        # Workflow storage
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.workflow_instances: Dict[str, WorkflowInstance] = {}
        self.workflow_history: deque = deque(maxlen=10000)
        
        # Execution tracking
        self.running_workflows: Dict[str, asyncio.Task] = {}
        self.workflow_locks: Dict[str, threading.Lock] = {}
        
        # Step handlers
        self.step_handlers: Dict[WorkflowStepType, Callable] = {
            WorkflowStepType.TASK: self._execute_task_step,
            WorkflowStepType.CONDITION: self._execute_condition_step,
            WorkflowStepType.LOOP: self._execute_loop_step,
            WorkflowStepType.PARALLEL: self._execute_parallel_step,
            WorkflowStepType.SEQUENTIAL: self._execute_sequential_step,
            WorkflowStepType.SUB_WORKFLOW: self._execute_sub_workflow_step,
            WorkflowStepType.WAIT: self._execute_wait_step,
            WorkflowStepType.NOTIFICATION: self._execute_notification_step
        }
        
        # Control
        self.is_running = True
        self._lock = threading.RLock()
        
        # Create persistence directory
        if self.config.enable_persistence:
            self.persistence_path = Path(self.config.persistence_path)
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            self._load_workflows()
        
        self.logger.info("Workflow manager initialized")
    
    def define_workflow(
        self,
        name: str,
        version: str,
        steps: List[Dict[str, Any]],
        start_step: str,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Define a new workflow.
        
        Args:
            name: Workflow name
            version: Workflow version
            steps: List of step definitions
            start_step: ID of the starting step
            description: Workflow description
            parameters: Workflow parameters schema
            outputs: Workflow outputs schema
            metadata: Additional metadata
            
        Returns:
            Workflow ID
        """
        workflow_id = f"{name}_v{version}_{uuid.uuid4().hex[:8]}"
        
        # Create step objects
        workflow_steps = {}
        for step_def in steps:
            step = WorkflowStep(
                step_id=step_def["id"],
                name=step_def["name"],
                step_type=WorkflowStepType(step_def["type"]),
                config=step_def.get("config", {}),
                next_steps=step_def.get("next_steps", []),
                error_handler=step_def.get("error_handler"),
                retry_policy=step_def.get("retry_policy", {}),
                metadata=step_def.get("metadata", {})
            )
            workflow_steps[step.step_id] = step
        
        # Create workflow definition
        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=name,
            version=version,
            description=description,
            steps=workflow_steps,
            start_step=start_step,
            parameters=parameters or {},
            outputs=outputs or {},
            metadata=metadata or {}
        )
        
        # Validate workflow
        self._validate_workflow(workflow)
        
        # Store workflow
        with self._lock:
            self.workflow_definitions[workflow_id] = workflow
        
        # Persist if enabled
        if self.config.enable_persistence:
            self._save_workflow(workflow)
        
        self.logger.info(f"Workflow defined: {workflow_id} - {name} v{version}")
        
        return workflow_id
    
    def define_workflow_from_yaml(self, yaml_path: str) -> str:
        """Define workflow from YAML file"""
        with open(yaml_path, 'r') as f:
            workflow_def = yaml.safe_load(f)
        
        return self.define_workflow(**workflow_def)
    
    def start_workflow(
        self,
        workflow_id: str,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a workflow instance.
        
        Args:
            workflow_id: Workflow definition ID
            parameters: Workflow parameters
            context: Initial context
            
        Returns:
            Instance ID
        """
        if workflow_id not in self.workflow_definitions:
            raise ValidationError(f"Workflow not found: {workflow_id}")
        
        workflow = self.workflow_definitions[workflow_id]
        
        # Validate parameters
        if workflow.parameters:
            self._validate_parameters(parameters or {}, workflow.parameters)
        
        # Create instance
        instance_id = f"instance_{uuid.uuid4().hex[:12]}"
        instance = WorkflowInstance(
            instance_id=instance_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.READY,
            parameters=parameters or {},
            context=context or {}
        )
        
        with self._lock:
            self.workflow_instances[instance_id] = instance
            self.workflow_locks[instance_id] = threading.Lock()
        
        # Start execution
        if asyncio.get_event_loop().is_running():
            task = asyncio.create_task(self._execute_workflow(instance_id))
        else:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            task = loop.create_task(self._execute_workflow(instance_id))
        
        self.running_workflows[instance_id] = task
        
        self.logger.info(f"Workflow started: {instance_id} ({workflow.name})")
        
        return instance_id
    
    async def _execute_workflow(self, instance_id: str):
        """Execute workflow instance"""
        instance = self.workflow_instances[instance_id]
        workflow = self.workflow_definitions[instance.workflow_id]
        
        try:
            # Update status
            instance.status = WorkflowStatus.RUNNING
            instance.start_time = datetime.now()
            
            # Start from the beginning
            if workflow.start_step:
                await self._execute_step(instance_id, workflow.start_step)
            
            # Check if workflow completed successfully
            if instance.status == WorkflowStatus.RUNNING:
                instance.status = WorkflowStatus.COMPLETED
                instance.end_time = datetime.now()
                
                # Process outputs
                if workflow.outputs:
                    instance.context["outputs"] = self._process_outputs(
                        instance,
                        workflow.outputs
                    )
            
        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.error = str(e)
            instance.end_time = datetime.now()
            self.logger.error(f"Workflow {instance_id} failed: {e}")
        
        finally:
            # Clean up
            del self.running_workflows[instance_id]
            
            # Add to history
            if self.config.keep_history:
                self.workflow_history.append({
                    "instance": asdict(instance),
                    "timestamp": datetime.now()
                })
            
            # Persist state
            if self.state_manager:
                self.state_manager.set(
                    f"workflow_instance:{instance_id}",
                    instance,
                    StateType.JOB
                )
    
    async def _execute_step(self, instance_id: str, step_id: str):
        """Execute a workflow step"""
        instance = self.workflow_instances[instance_id]
        workflow = self.workflow_definitions[instance.workflow_id]
        
        if step_id not in workflow.steps:
            raise ValidationError(f"Step not found: {step_id}")
        
        step = workflow.steps[step_id]
        
        # Check if already completed
        if step_id in instance.completed_steps:
            return
        
        # Mark as current
        instance.current_steps.add(step_id)
        
        try:
            # Get handler
            handler = self.step_handlers.get(step.step_type)
            if not handler:
                raise ValueError(f"Unknown step type: {step.step_type}")
            
            # Execute step
            result = await handler(instance, step)
            
            # Store result
            instance.step_results[step_id] = result
            
            # Mark as completed
            instance.completed_steps.add(step_id)
            instance.current_steps.remove(step_id)
            
            # Execute next steps
            for next_step_id in step.next_steps:
                await self._execute_step(instance_id, next_step_id)
            
        except Exception as e:
            self.logger.error(f"Step {step_id} failed: {e}")
            
            # Handle error
            if step.error_handler:
                await self._execute_step(instance_id, step.error_handler)
            else:
                raise
    
    async def _execute_task_step(
        self,
        instance: WorkflowInstance,
        step: WorkflowStep
    ) -> Any:
        """Execute a task step"""
        config = step.config
        
        # Create task
        task_id = self.task_manager.create_task(
            name=f"{instance.instance_id}_{step.name}",
            task_type=TaskType(config.get("task_type", "DOCUMENT_GENERATION")),
            function=config["function"],
            args=config.get("args", []),
            kwargs=self._resolve_parameters(config.get("kwargs", {}), instance),
            retry_policy=step.retry_policy,
            timeout=config.get("timeout", self.config.default_timeout),
            metadata={"workflow_instance": instance.instance_id, "step": step.step_id}
        )
        
        # Execute task
        result = await self.task_manager.execute_task_async(task_id)
        
        if result.status != TaskStatus.SUCCEEDED:
            raise ProcessingError(f"Task failed: {result.error}")
        
        # Update context with result
        if "output_key" in config:
            instance.context[config["output_key"]] = result.output
        
        return result.output
    
    async def _execute_condition_step(
        self,
        instance: WorkflowInstance,
        step: WorkflowStep
    ) -> Any:
        """Execute a condition step"""
        config = step.config
        
        # Evaluate condition
        condition = config["condition"]
        context = instance.context.copy()
        context.update(instance.parameters)
        
        try:
            # Simple expression evaluation (in production, use a safe evaluator)
            result = eval(condition, {"__builtins__": {}}, context)
        except Exception as e:
            raise ProcessingError(f"Condition evaluation failed: {e}")
        
        # Override next steps based on result
        if result:
            step.next_steps = config.get("true_branch", [])
        else:
            step.next_steps = config.get("false_branch", [])
        
        return result
    
    async def _execute_loop_step(
        self,
        instance: WorkflowInstance,
        step: WorkflowStep
    ) -> Any:
        """Execute a loop step"""
        config = step.config
        
        # Get loop configuration
        items = self._resolve_parameters(config["items"], instance)
        loop_var = config.get("loop_var", "item")
        body_steps = config["body_steps"]
        
        results = []
        
        # Execute loop body for each item
        for index, item in enumerate(items):
            # Set loop variables in context
            instance.context[loop_var] = item
            instance.context[f"{loop_var}_index"] = index
            
            # Execute body steps
            for body_step_id in body_steps:
                await self._execute_step(instance.instance_id, body_step_id)
            
            # Collect results if specified
            if "result_key" in config:
                result_key = config["result_key"]
                if result_key in instance.context:
                    results.append(instance.context[result_key])
        
        # Clean up loop variables
        instance.context.pop(loop_var, None)
        instance.context.pop(f"{loop_var}_index", None)
        
        return results
    
    async def _execute_parallel_step(
        self,
        instance: WorkflowInstance,
        step: WorkflowStep
    ) -> Any:
        """Execute parallel branches"""
        config = step.config
        branches = config["branches"]
        
        # Create tasks for each branch
        branch_tasks = []
        for branch in branches:
            branch_steps = branch["steps"]
            branch_task = asyncio.create_task(
                self._execute_branch(instance, branch_steps)
            )
            branch_tasks.append(branch_task)
        
        # Wait for all branches
        results = await asyncio.gather(*branch_tasks, return_exceptions=True)
        
        # Check for failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise ProcessingError(f"Branch {i} failed: {result}")
        
        return results
    
    async def _execute_branch(
        self,
        instance: WorkflowInstance,
        steps: List[str]
    ) -> Any:
        """Execute a sequence of steps"""
        results = []
        for step_id in steps:
            await self._execute_step(instance.instance_id, step_id)
            if step_id in instance.step_results:
                results.append(instance.step_results[step_id])
        return results
    
    async def _execute_sequential_step(
        self,
        instance: WorkflowInstance,
        step: WorkflowStep
    ) -> Any:
        """Execute sequential steps"""
        config = step.config
        steps = config["steps"]
        
        return await self._execute_branch(instance, steps)
    
    async def _execute_sub_workflow_step(
        self,
        instance: WorkflowInstance,
        step: WorkflowStep
    ) -> Any:
        """Execute a sub-workflow"""
        config = step.config
        
        # Start sub-workflow
        sub_instance_id = self.start_workflow(
            workflow_id=config["workflow_id"],
            parameters=self._resolve_parameters(config.get("parameters", {}), instance),
            context={"parent_instance": instance.instance_id}
        )
        
        # Wait for completion
        sub_instance = self.workflow_instances[sub_instance_id]
        while sub_instance.status in [WorkflowStatus.READY, WorkflowStatus.RUNNING]:
            await asyncio.sleep(1)
        
        # Check result
        if sub_instance.status != WorkflowStatus.COMPLETED:
            raise ProcessingError(f"Sub-workflow failed: {sub_instance.error}")
        
        return sub_instance.context.get("outputs", {})
    
    async def _execute_wait_step(
        self,
        instance: WorkflowInstance,
        step: WorkflowStep
    ) -> Any:
        """Execute a wait step"""
        config = step.config
        
        # Wait for specified duration
        duration = config.get("duration", 1)
        await asyncio.sleep(duration)
        
        return {"waited": duration}
    
    async def _execute_notification_step(
        self,
        instance: WorkflowInstance,
        step: WorkflowStep
    ) -> Any:
        """Execute a notification step"""
        config = step.config
        
        # Send notification (simplified)
        message = self._resolve_parameters(config["message"], instance)
        channel = config.get("channel", "log")
        
        if channel == "log":
            self.logger.info(f"Notification: {message}")
        
        return {"message": message, "channel": channel}
    
    def _resolve_parameters(
        self,
        value: Any,
        instance: WorkflowInstance
    ) -> Any:
        """Resolve parameters with context values"""
        if isinstance(value, str) and value.startswith("$"):
            # Variable reference
            var_path = value[1:].split(".")
            context = instance.context.copy()
            context.update(instance.parameters)
            
            result = context
            for part in var_path:
                if isinstance(result, dict) and part in result:
                    result = result[part]
                else:
                    return None
            return result
        
        elif isinstance(value, dict):
            return {k: self._resolve_parameters(v, instance) for k, v in value.items()}
        
        elif isinstance(value, list):
            return [self._resolve_parameters(v, instance) for v in value]
        
        return value
    
    def _validate_workflow(self, workflow: WorkflowDefinition):
        """Validate workflow definition"""
        # Check start step exists
        if workflow.start_step and workflow.start_step not in workflow.steps:
            raise ValidationError(f"Start step not found: {workflow.start_step}")
        
        # Check all referenced steps exist
        for step in workflow.steps.values():
            for next_step in step.next_steps:
                if next_step not in workflow.steps:
                    raise ValidationError(f"Referenced step not found: {next_step}")
            
            if step.error_handler and step.error_handler not in workflow.steps:
                raise ValidationError(f"Error handler step not found: {step.error_handler}")
        
        # Check for cycles (simplified)
        # In production, use proper cycle detection
    
    def _validate_parameters(self, provided: Dict[str, Any], schema: Dict[str, Any]):
        """Validate parameters against schema"""
        # Simple validation - in production use proper schema validation
        required = schema.get("required", [])
        for param in required:
            if param not in provided:
                raise ValidationError(f"Required parameter missing: {param}")
    
    def _process_outputs(
        self,
        instance: WorkflowInstance,
        output_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process workflow outputs"""
        outputs = {}
        
        for output_name, output_config in output_schema.items():
            if "from" in output_config:
                value = self._resolve_parameters(output_config["from"], instance)
                outputs[output_name] = value
        
        return outputs
    
    def pause_workflow(self, instance_id: str):
        """Pause a running workflow"""
        with self._lock:
            if instance_id in self.workflow_instances:
                instance = self.workflow_instances[instance_id]
                if instance.status == WorkflowStatus.RUNNING:
                    instance.status = WorkflowStatus.PAUSED
                    self.logger.info(f"Workflow paused: {instance_id}")
    
    def resume_workflow(self, instance_id: str):
        """Resume a paused workflow"""
        with self._lock:
            if instance_id in self.workflow_instances:
                instance = self.workflow_instances[instance_id]
                if instance.status == WorkflowStatus.PAUSED:
                    instance.status = WorkflowStatus.RUNNING
                    self.logger.info(f"Workflow resumed: {instance_id}")
    
    def cancel_workflow(self, instance_id: str):
        """Cancel a workflow"""
        with self._lock:
            if instance_id in self.workflow_instances:
                instance = self.workflow_instances[instance_id]
                instance.status = WorkflowStatus.CANCELLED
                instance.end_time = datetime.now()
                
                # Cancel running task
                if instance_id in self.running_workflows:
                    self.running_workflows[instance_id].cancel()
                
                self.logger.info(f"Workflow cancelled: {instance_id}")
    
    def get_workflow_status(self, instance_id: str) -> Optional[WorkflowStatus]:
        """Get workflow status"""
        with self._lock:
            if instance_id in self.workflow_instances:
                return self.workflow_instances[instance_id].status
        return None
    
    def get_workflow_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """Get workflow instance"""
        with self._lock:
            return self.workflow_instances.get(instance_id)
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflow definitions"""
        with self._lock:
            return [
                {
                    "workflow_id": wf.workflow_id,
                    "name": wf.name,
                    "version": wf.version,
                    "description": wf.description,
                    "steps": len(wf.steps)
                }
                for wf in self.workflow_definitions.values()
            ]
    
    def list_instances(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None
    ) -> List[Dict[str, Any]]:
        """List workflow instances"""
        with self._lock:
            instances = []
            for instance in self.workflow_instances.values():
                if workflow_id and instance.workflow_id != workflow_id:
                    continue
                if status and instance.status != status:
                    continue
                
                instances.append({
                    "instance_id": instance.instance_id,
                    "workflow_id": instance.workflow_id,
                    "status": instance.status.value,
                    "start_time": instance.start_time.isoformat(),
                    "end_time": instance.end_time.isoformat() if instance.end_time else None,
                    "progress": len(instance.completed_steps)
                })
            
            return instances
    
    def get_stats(self) -> Dict[str, Any]:
        """Get workflow manager statistics"""
        with self._lock:
            status_counts = defaultdict(int)
            for instance in self.workflow_instances.values():
                status_counts[instance.status.value] += 1
            
            return {
                "total_workflows": len(self.workflow_definitions),
                "total_instances": len(self.workflow_instances),
                "running_instances": len(self.running_workflows),
                "status_counts": dict(status_counts),
                "history_size": len(self.workflow_history)
            }
    
    def _save_workflow(self, workflow: WorkflowDefinition):
        """Save workflow definition to disk"""
        workflow_file = self.persistence_path / f"{workflow.workflow_id}.json"
        with open(workflow_file, 'w') as f:
            json.dump(asdict(workflow), f, indent=2, default=str)
    
    def _load_workflows(self):
        """Load workflow definitions from disk"""
        for workflow_file in self.persistence_path.glob("*.json"):
            try:
                with open(workflow_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct workflow
                steps = {}
                for step_id, step_data in data["steps"].items():
                    steps[step_id] = WorkflowStep(
                        step_id=step_data["step_id"],
                        name=step_data["name"],
                        step_type=WorkflowStepType(step_data["step_type"]),
                        config=step_data["config"],
                        next_steps=step_data["next_steps"],
                        error_handler=step_data.get("error_handler"),
                        retry_policy=step_data.get("retry_policy", {}),
                        metadata=step_data.get("metadata", {})
                    )
                
                workflow = WorkflowDefinition(
                    workflow_id=data["workflow_id"],
                    name=data["name"],
                    version=data["version"],
                    description=data.get("description"),
                    steps=steps,
                    start_step=data.get("start_step"),
                    parameters=data.get("parameters", {}),
                    outputs=data.get("outputs", {}),
                    metadata=data.get("metadata", {})
                )
                
                self.workflow_definitions[workflow.workflow_id] = workflow
                
            except Exception as e:
                self.logger.error(f"Failed to load workflow {workflow_file}: {e}")
        
        self.logger.info(f"Loaded {len(self.workflow_definitions)} workflows")
    
    def cleanup(self):
        """Clean up workflow manager"""
        self.is_running = False
        
        # Cancel running workflows
        for task in self.running_workflows.values():
            task.cancel()
        
        # Clean up task manager
        if self.task_manager:
            self.task_manager.cleanup()
        
        self.logger.info("Workflow manager cleaned up")


# Factory function
def create_workflow_manager(
    config: Optional[Dict[str, Any]] = None,
    task_manager: Optional[TaskManager] = None,
    state_manager: Optional[StateManager] = None,
    resource_manager: Optional[ResourceManager] = None
) -> WorkflowManager:
    """Create and return a workflow manager instance"""
    return WorkflowManager(config, task_manager, state_manager, resource_manager)