#!/usr/bin/env python3
"""
Custom orchestrator for complex workflow management.

Provides comprehensive workflow orchestration with support for
conditional execution, parallel processing, and error handling.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import json

from pydantic import BaseModel, Field

from ...core import get_logger, BaseConfig

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskType(str, Enum):
    """Task types"""
    FUNCTION = "function"
    SUBPROCESS = "subprocess"
    HTTP_REQUEST = "http_request"
    CONDITION = "condition"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    WAIT = "wait"


@dataclass
class TaskDefinition:
    """Task definition"""
    id: str
    name: str
    task_type: TaskType
    config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: Optional[int] = None
    on_failure: Optional[str] = None  # Task to run on failure
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecution:
    """Task execution record"""
    task_id: str
    execution_id: str
    workflow_id: str
    status: TaskStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
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
class WorkflowDefinition:
    """Workflow definition"""
    id: str
    name: str
    description: str
    tasks: List[TaskDefinition]
    global_timeout_seconds: Optional[int] = None
    max_parallel_tasks: int = 10
    on_failure: str = "stop"  # stop, continue, rollback
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowExecution:
    """Workflow execution record"""
    workflow_id: str
    execution_id: str
    status: WorkflowStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    task_executions: Dict[str, TaskExecution] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        
        # Convert task executions
        data['task_executions'] = {
            task_id: execution.to_dict() 
            for task_id, execution in self.task_executions.items()
        }
        
        return data


class CustomOrchestratorConfig(BaseConfig):
    """Custom orchestrator configuration"""
    max_concurrent_workflows: int = 5
    max_concurrent_tasks: int = 20
    default_task_timeout_seconds: int = 3600
    default_workflow_timeout_seconds: int = 7200
    
    # Execution history
    execution_retention_days: int = 30
    
    # Error handling
    enable_automatic_retry: bool = True
    default_retry_delay_seconds: int = 60
    
    # Monitoring
    enable_workflow_monitoring: bool = True
    health_check_interval_seconds: int = 60
    
    # Persistence
    enable_workflow_persistence: bool = False
    persistence_backend: str = "memory"  # memory, file, database


class CustomOrchestrator:
    """Comprehensive workflow orchestration engine"""
    
    def __init__(self, config: CustomOrchestratorConfig):
        self.config = config
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: List[WorkflowExecution] = []
        self.running_workflows: Dict[str, asyncio.Task] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.registered_functions: Dict[str, Callable] = {}
        self.running = False
        
        # Workflow callbacks
        self.workflow_callbacks: Dict[str, List[Callable]] = {}
        self.task_callbacks: Dict[str, List[Callable]] = {}
    
    async def start(self):
        """Start orchestrator"""
        self.running = True
        logger.info("Custom orchestrator started")
        
        # Start management tasks
        asyncio.create_task(self._cleanup_worker())
        asyncio.create_task(self._monitoring_worker())
    
    async def stop(self):
        """Stop orchestrator"""
        self.running = False
        
        # Cancel all running workflows
        for task in self.running_workflows.values():
            task.cancel()
        
        # Cancel all running tasks
        for task in self.running_tasks.values():
            task.cancel()
        
        logger.info("Custom orchestrator stopped")
    
    def register_workflow(self, workflow: WorkflowDefinition) -> bool:
        """
        Register workflow definition.
        
        Args:
            workflow: Workflow definition to register
        
        Returns:
            True if workflow was registered
        """
        try:
            # Validate workflow
            self._validate_workflow(workflow)
            
            self.workflows[workflow.id] = workflow
            logger.info(f"Registered workflow: {workflow.id} - {workflow.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register workflow {workflow.id}: {e}")
            return False
    
    def register_function(self, name: str, function: Callable):
        """
        Register function for task execution.
        
        Args:
            name: Function name
            function: Callable function
        """
        self.registered_functions[name] = function
        logger.info(f"Registered function: {name}")
    
    async def execute_workflow(self, workflow_id: str, 
                             context: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute workflow.
        
        Args:
            workflow_id: ID of workflow to execute
            context: Initial execution context
        
        Returns:
            Execution ID
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow '{workflow_id}' not found")
        
        if len(self.running_workflows) >= self.config.max_concurrent_workflows:
            raise RuntimeError("Maximum concurrent workflows reached")
        
        execution_id = str(uuid.uuid4())
        
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            execution_id=execution_id,
            status=WorkflowStatus.CREATED,
            context=context or {}
        )
        
        self.executions[execution_id] = execution
        
        # Start workflow execution
        workflow_task = asyncio.create_task(
            self._execute_workflow_async(execution)
        )
        self.running_workflows[execution_id] = workflow_task
        
        logger.info(f"Started workflow execution: {workflow_id} ({execution_id})")
        return execution_id
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """
        Cancel workflow execution.
        
        Args:
            execution_id: Execution ID to cancel
        
        Returns:
            True if workflow was cancelled
        """
        try:
            if execution_id in self.running_workflows:
                task = self.running_workflows[execution_id]
                task.cancel()
                
                if execution_id in self.executions:
                    self.executions[execution_id].status = WorkflowStatus.CANCELLED
                    self.executions[execution_id].end_time = datetime.now()
                
                logger.info(f"Cancelled workflow execution: {execution_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel workflow {execution_id}: {e}")
            return False
    
    async def pause_workflow(self, execution_id: str) -> bool:
        """
        Pause workflow execution.
        
        Args:
            execution_id: Execution ID to pause
        
        Returns:
            True if workflow was paused
        """
        try:
            if execution_id in self.executions:
                execution = self.executions[execution_id]
                if execution.status == WorkflowStatus.RUNNING:
                    execution.status = WorkflowStatus.PAUSED
                    logger.info(f"Paused workflow execution: {execution_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to pause workflow {execution_id}: {e}")
            return False
    
    async def resume_workflow(self, execution_id: str) -> bool:
        """
        Resume paused workflow execution.
        
        Args:
            execution_id: Execution ID to resume
        
        Returns:
            True if workflow was resumed
        """
        try:
            if execution_id in self.executions:
                execution = self.executions[execution_id]
                if execution.status == WorkflowStatus.PAUSED:
                    execution.status = WorkflowStatus.RUNNING
                    logger.info(f"Resumed workflow execution: {execution_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resume workflow {execution_id}: {e}")
            return False
    
    def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get workflow execution status.
        
        Args:
            execution_id: Execution ID
        
        Returns:
            Workflow status dictionary
        """
        if execution_id not in self.executions:
            return None
        
        execution = self.executions[execution_id]
        workflow = self.workflows.get(execution.workflow_id)
        
        # Calculate progress
        total_tasks = len(workflow.tasks) if workflow else 0
        completed_tasks = len([
            t for t in execution.task_executions.values() 
            if t.status == TaskStatus.COMPLETED
        ])
        
        return {
            'execution_id': execution_id,
            'workflow_id': execution.workflow_id,
            'workflow_name': workflow.name if workflow else 'Unknown',
            'status': execution.status.value,
            'start_time': execution.start_time.isoformat() if execution.start_time else None,
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'duration_seconds': (
                (execution.end_time - execution.start_time).total_seconds()
                if execution.start_time and execution.end_time else None
            ),
            'progress': {
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'progress_percent': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            },
            'task_statuses': {
                task_id: execution.status.value 
                for task_id, execution in execution.task_executions.items()
            },
            'error': execution.error
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get orchestrator system status.
        
        Returns:
            System status dictionary
        """
        return {
            'running': self.running,
            'registered_workflows': len(self.workflows),
            'running_workflows': len(self.running_workflows),
            'running_tasks': len(self.running_tasks),
            'total_executions': len(self.execution_history),
            'registered_functions': len(self.registered_functions)
        }
    
    def add_workflow_callback(self, workflow_id: str, 
                            callback: Callable[[WorkflowExecution], None]):
        """
        Add callback for workflow completion.
        
        Args:
            workflow_id: Workflow ID or '*' for all workflows
            callback: Callback function
        """
        if workflow_id not in self.workflow_callbacks:
            self.workflow_callbacks[workflow_id] = []
        self.workflow_callbacks[workflow_id].append(callback)
    
    def add_task_callback(self, task_id: str, 
                        callback: Callable[[TaskExecution], None]):
        """
        Add callback for task completion.
        
        Args:
            task_id: Task ID or '*' for all tasks
            callback: Callback function
        """
        if task_id not in self.task_callbacks:
            self.task_callbacks[task_id] = []
        self.task_callbacks[task_id].append(callback)
    
    def _validate_workflow(self, workflow: WorkflowDefinition):
        """Validate workflow definition"""
        # Check for circular dependencies
        visited = set()
        
        def check_dependencies(task_id: str, path: Set[str]):
            if task_id in path:
                raise ValueError(f"Circular dependency detected: {task_id}")
            
            if task_id in visited:
                return
            
            visited.add(task_id)
            path.add(task_id)
            
            task = next((t for t in workflow.tasks if t.id == task_id), None)
            if task:
                for dep_id in task.dependencies:
                    check_dependencies(dep_id, path.copy())
            
            path.remove(task_id)
        
        for task in workflow.tasks:
            check_dependencies(task.id, set())
    
    async def _execute_workflow_async(self, execution: WorkflowExecution):
        """Execute workflow asynchronously"""
        try:
            execution.status = WorkflowStatus.RUNNING
            execution.start_time = datetime.now()
            
            workflow = self.workflows[execution.workflow_id]
            
            logger.info(f"Executing workflow: {workflow.name} ({execution.execution_id})")
            
            # Create task dependency graph
            task_graph = self._build_task_graph(workflow.tasks)
            
            # Execute tasks based on dependencies
            await self._execute_task_graph(execution, task_graph)
            
            execution.status = WorkflowStatus.COMPLETED
            logger.info(f"Workflow completed: {workflow.name} ({execution.execution_id})")
            
        except asyncio.CancelledError:
            execution.status = WorkflowStatus.CANCELLED
            logger.info(f"Workflow cancelled: {execution.execution_id}")
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            logger.error(f"Workflow failed: {execution.execution_id} - {e}")
        
        finally:
            execution.end_time = datetime.now()
            
            # Move to history
            self.execution_history.append(execution)
            
            # Clean up
            if execution.execution_id in self.running_workflows:
                del self.running_workflows[execution.execution_id]
            
            # Call callbacks
            await self._call_workflow_callbacks(execution)
    
    def _build_task_graph(self, tasks: List[TaskDefinition]) -> Dict[str, TaskDefinition]:
        """Build task dependency graph"""
        return {task.id: task for task in tasks}
    
    async def _execute_task_graph(self, execution: WorkflowExecution, 
                                task_graph: Dict[str, TaskDefinition]):
        """Execute tasks based on dependency graph"""
        completed_tasks = set()
        running_tasks = {}
        
        while len(completed_tasks) < len(task_graph):
            # Check for paused workflow
            if execution.status == WorkflowStatus.PAUSED:
                await asyncio.sleep(1)
                continue
            
            # Find tasks ready to run
            ready_tasks = []
            for task_id, task in task_graph.items():
                if (task_id not in completed_tasks and 
                    task_id not in running_tasks and
                    all(dep in completed_tasks for dep in task.dependencies)):
                    
                    # Check conditions
                    if self._check_task_conditions(task, execution.context):
                        ready_tasks.append(task)
            
            # Start ready tasks
            for task in ready_tasks:
                if len(running_tasks) >= self.config.max_concurrent_tasks:
                    break
                
                task_execution = TaskExecution(
                    task_id=task.id,
                    execution_id=str(uuid.uuid4()),
                    workflow_id=execution.workflow_id,
                    status=TaskStatus.PENDING
                )
                
                execution.task_executions[task.id] = task_execution
                
                # Start task
                task_task = asyncio.create_task(
                    self._execute_task(task, task_execution, execution.context)
                )
                running_tasks[task.id] = task_task
            
            # Wait for at least one task to complete
            if running_tasks:
                done_tasks = []
                for task_id, task_task in list(running_tasks.items()):
                    if task_task.done():
                        done_tasks.append(task_id)
                        completed_tasks.add(task_id)
                        del running_tasks[task_id]
                
                if not done_tasks:
                    # Wait for any task to complete
                    done, pending = await asyncio.wait(
                        running_tasks.values(),
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Find completed task
                    for task_id, task_task in list(running_tasks.items()):
                        if task_task in done:
                            completed_tasks.add(task_id)
                            del running_tasks[task_id]
                            break
            else:
                # No tasks ready and none running - check if we can continue
                if len(completed_tasks) < len(task_graph):
                    remaining_tasks = [t for t in task_graph.keys() if t not in completed_tasks]
                    raise RuntimeError(f"Workflow deadlock detected. Remaining tasks: {remaining_tasks}")
    
    def _check_task_conditions(self, task: TaskDefinition, 
                             context: Dict[str, Any]) -> bool:
        """Check if task conditions are met"""
        if not task.conditions:
            return True
        
        # Simple condition evaluation - could be enhanced
        for condition in task.conditions:
            # This is a placeholder for condition evaluation
            # In a real implementation, you'd have a more sophisticated evaluator
            if condition not in context or not context[condition]:
                return False
        
        return True
    
    async def _execute_task(self, task: TaskDefinition, 
                          task_execution: TaskExecution,
                          context: Dict[str, Any]):
        """Execute individual task"""
        try:
            task_execution.status = TaskStatus.RUNNING
            task_execution.start_time = datetime.now()
            
            logger.info(f"Executing task: {task.name} ({task.id})")
            
            # Execute based on task type
            if task.task_type == TaskType.FUNCTION:
                result = await self._execute_function_task(task, context)
            elif task.task_type == TaskType.SUBPROCESS:
                result = await self._execute_subprocess_task(task, context)
            elif task.task_type == TaskType.HTTP_REQUEST:
                result = await self._execute_http_request_task(task, context)
            elif task.task_type == TaskType.CONDITION:
                result = await self._execute_condition_task(task, context)
            elif task.task_type == TaskType.WAIT:
                result = await self._execute_wait_task(task, context)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            task_execution.status = TaskStatus.COMPLETED
            task_execution.result = result
            
            # Update context with result
            if isinstance(result, dict):
                context.update(result)
            else:
                context[f"task_{task.id}_result"] = result
            
            logger.info(f"Task completed: {task.name} ({task.id})")
            
        except Exception as e:
            task_execution.status = TaskStatus.FAILED
            task_execution.error = str(e)
            
            logger.error(f"Task failed: {task.name} ({task.id}) - {e}")
            
            # Handle retry
            if task_execution.retry_count < task.max_retries:
                task_execution.retry_count += 1
                logger.info(f"Retrying task {task.id} (attempt {task_execution.retry_count})")
                await asyncio.sleep(self.config.default_retry_delay_seconds)
                return await self._execute_task(task, task_execution, context)
        
        finally:
            task_execution.end_time = datetime.now()
            
            # Call task callbacks
            await self._call_task_callbacks(task_execution)
    
    async def _execute_function_task(self, task: TaskDefinition, 
                                   context: Dict[str, Any]) -> Any:
        """Execute function task"""
        function_name = task.config.get('function')
        if function_name not in self.registered_functions:
            raise ValueError(f"Function '{function_name}' not registered")
        
        function = self.registered_functions[function_name]
        args = task.config.get('args', [])
        kwargs = task.config.get('kwargs', {})
        
        # Substitute context variables
        args = self._substitute_context_variables(args, context)
        kwargs = self._substitute_context_variables(kwargs, context)
        
        if asyncio.iscoroutinefunction(function):
            return await function(*args, **kwargs)
        else:
            # Run in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: function(*args, **kwargs)
            )
    
    async def _execute_subprocess_task(self, task: TaskDefinition, 
                                     context: Dict[str, Any]) -> Any:
        """Execute subprocess task"""
        command = task.config.get('command')
        if not command:
            raise ValueError("No command specified for subprocess task")
        
        # Substitute context variables
        command = self._substitute_context_variables(command, context)
        
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"Subprocess failed: {stderr.decode()}")
        
        return {
            'stdout': stdout.decode(),
            'stderr': stderr.decode(),
            'returncode': process.returncode
        }
    
    async def _execute_http_request_task(self, task: TaskDefinition, 
                                       context: Dict[str, Any]) -> Any:
        """Execute HTTP request task"""
        import aiohttp
        
        url = task.config.get('url')
        method = task.config.get('method', 'GET')
        headers = task.config.get('headers', {})
        data = task.config.get('data')
        
        # Substitute context variables
        url = self._substitute_context_variables(url, context)
        headers = self._substitute_context_variables(headers, context)
        data = self._substitute_context_variables(data, context)
        
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers, json=data) as response:
                result = {
                    'status': response.status,
                    'headers': dict(response.headers),
                    'text': await response.text()
                }
                
                try:
                    result['json'] = await response.json()
                except:
                    pass
                
                return result
    
    async def _execute_condition_task(self, task: TaskDefinition, 
                                    context: Dict[str, Any]) -> Any:
        """Execute condition task"""
        condition = task.config.get('condition')
        if not condition:
            raise ValueError("No condition specified for condition task")
        
        # Simple condition evaluation
        # In a real implementation, you'd have a more sophisticated evaluator
        try:
            # This is a very basic evaluator - should be enhanced for production
            result = eval(condition, {}, context)
            return {'condition_result': bool(result)}
        except Exception as e:
            raise ValueError(f"Condition evaluation failed: {e}")
    
    async def _execute_wait_task(self, task: TaskDefinition, 
                               context: Dict[str, Any]) -> Any:
        """Execute wait task"""
        seconds = task.config.get('seconds', 0)
        seconds = self._substitute_context_variables(seconds, context)
        
        await asyncio.sleep(float(seconds))
        return {'waited_seconds': seconds}
    
    def _substitute_context_variables(self, value: Any, context: Dict[str, Any]) -> Any:
        """Substitute context variables in value"""
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            var_name = value[2:-1]
            return context.get(var_name, value)
        elif isinstance(value, dict):
            return {k: self._substitute_context_variables(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._substitute_context_variables(item, context) for item in value]
        else:
            return value
    
    async def _call_workflow_callbacks(self, execution: WorkflowExecution):
        """Call workflow callbacks"""
        try:
            # Workflow-specific callbacks
            if execution.workflow_id in self.workflow_callbacks:
                for callback in self.workflow_callbacks[execution.workflow_id]:
                    try:
                        await callback(execution)
                    except Exception as e:
                        logger.error(f"Error in workflow callback: {e}")
            
            # Global callbacks
            if '*' in self.workflow_callbacks:
                for callback in self.workflow_callbacks['*']:
                    try:
                        await callback(execution)
                    except Exception as e:
                        logger.error(f"Error in global workflow callback: {e}")
        
        except Exception as e:
            logger.error(f"Failed to call workflow callbacks: {e}")
    
    async def _call_task_callbacks(self, task_execution: TaskExecution):
        """Call task callbacks"""
        try:
            # Task-specific callbacks
            if task_execution.task_id in self.task_callbacks:
                for callback in self.task_callbacks[task_execution.task_id]:
                    try:
                        await callback(task_execution)
                    except Exception as e:
                        logger.error(f"Error in task callback: {e}")
            
            # Global callbacks
            if '*' in self.task_callbacks:
                for callback in self.task_callbacks['*']:
                    try:
                        await callback(task_execution)
                    except Exception as e:
                        logger.error(f"Error in global task callback: {e}")
        
        except Exception as e:
            logger.error(f"Failed to call task callbacks: {e}")
    
    async def _cleanup_worker(self):
        """Background worker for cleanup"""
        while self.running:
            try:
                await self._cleanup_old_executions()
                await asyncio.sleep(24 * 3600)  # Daily cleanup
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    async def _cleanup_old_executions(self):
        """Clean up old workflow executions"""
        try:
            cutoff = datetime.now() - timedelta(days=self.config.execution_retention_days)
            
            original_count = len(self.execution_history)
            self.execution_history = [
                ex for ex in self.execution_history 
                if not ex.end_time or ex.end_time >= cutoff
            ]
            
            cleaned_count = original_count - len(self.execution_history)
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old workflow executions")
        
        except Exception as e:
            logger.error(f"Failed to cleanup old executions: {e}")
    
    async def _monitoring_worker(self):
        """Background worker for monitoring"""
        while self.running:
            try:
                # Monitor workflow health
                await asyncio.sleep(self.config.health_check_interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring worker: {e}")


def create_custom_orchestrator(config: Optional[CustomOrchestratorConfig] = None) -> CustomOrchestrator:
    """Factory function to create custom orchestrator"""
    if config is None:
        config = CustomOrchestratorConfig()
    return CustomOrchestrator(config)


__all__ = [
    'CustomOrchestrator',
    'CustomOrchestratorConfig',
    'WorkflowDefinition',
    'TaskDefinition',
    'WorkflowExecution',
    'TaskExecution',
    'WorkflowStatus',
    'TaskStatus',
    'TaskType',
    'create_custom_orchestrator'
]