#!/usr/bin/env python3
"""
Task Manager for orchestrating complex document generation workflows.

Provides high-level task management with support for task graphs,
parallel execution, failure recovery, and progress tracking.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable, Set, Tuple, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import threading
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import networkx as nx

from pydantic import BaseModel, Field

from ..core.config import get_config
from ..core.logging import get_logger
from ..core.exceptions import ProcessingError, ValidationError
from .job_queue import JobQueue, Job, JobStatus, JobPriority
from .state_manager import StateManager, StateType
from .resource_manager import ResourceManager, ResourceType


logger = get_logger(__name__)


class TaskType(Enum):
    """Types of tasks"""
    DOCUMENT_GENERATION = "document_generation"
    DATA_PROCESSING = "data_processing"
    VALIDATION = "validation"
    EXPORT = "export"
    NOTIFICATION = "notification"
    CLEANUP = "cleanup"
    COMPOSITE = "composite"


class TaskStatus(Enum):
    """Task execution status"""
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    status: TaskStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Individual task definition"""
    task_id: str
    name: str
    task_type: TaskType
    function: Union[Callable, str]  # Callable or function name
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 300.0
    priority: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskGraph:
    """Task execution graph"""
    graph_id: str
    name: str
    tasks: Dict[str, Task]
    edges: List[Tuple[str, str]]  # (from_task, to_task)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph"""
        G = nx.DiGraph()
        for task_id, task in self.tasks.items():
            G.add_node(task_id, task=task)
        G.add_edges_from(self.edges)
        return G


class TaskManagerConfig(BaseModel):
    """Task manager configuration"""
    # Execution settings
    max_concurrent_tasks: int = Field(50, description="Maximum concurrent tasks")
    task_timeout: float = Field(300.0, description="Default task timeout in seconds")
    enable_async: bool = Field(True, description="Enable async task execution")
    
    # Retry settings
    default_max_retries: int = Field(3, description="Default maximum retries")
    default_retry_delay: float = Field(60.0, description="Default retry delay in seconds")
    exponential_backoff: bool = Field(True, description="Use exponential backoff for retries")
    
    # Progress tracking
    enable_progress_tracking: bool = Field(True, description="Enable progress tracking")
    progress_update_interval: float = Field(5.0, description="Progress update interval")
    
    # Persistence
    enable_persistence: bool = Field(True, description="Enable task persistence")
    persistence_path: str = Field("./task_manager_state.json", description="Persistence file path")
    
    # Monitoring
    enable_monitoring: bool = Field(True, description="Enable task monitoring")
    metrics_interval: float = Field(60.0, description="Metrics collection interval")


class TaskManager:
    """
    Advanced task manager for complex workflows.
    
    Features:
    - Task graph execution
    - Dependency management
    - Parallel execution
    - Failure recovery
    - Progress tracking
    - Task composition
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        job_queue: Optional[JobQueue] = None,
        state_manager: Optional[StateManager] = None,
        resource_manager: Optional[ResourceManager] = None
    ):
        """Initialize task manager"""
        self.config = TaskManagerConfig(**(config or {}))
        self.logger = get_logger(__name__)
        
        # External managers
        self.job_queue = job_queue
        self.state_manager = state_manager
        self.resource_manager = resource_manager
        
        # Task storage
        self.tasks: Dict[str, Task] = {}
        self.task_graphs: Dict[str, TaskGraph] = {}
        self.task_results: Dict[str, TaskResult] = {}
        
        # Execution tracking
        self.running_tasks: Dict[str, Future] = {}
        self.task_futures: Dict[str, asyncio.Future] = {}
        
        # Function registry
        self.function_registry: Dict[str, Callable] = {}
        self._register_builtin_functions()
        
        # Executors
        self.thread_executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_tasks)
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Progress tracking
        self.progress_callbacks: List[Callable] = []
        self.task_progress: Dict[str, float] = {}
        
        # Control
        self.is_running = False
        self._lock = threading.RLock()
        
        # Load state if persistence enabled
        if self.config.enable_persistence:
            self._load_state()
        
        self.logger.info("Task manager initialized")
    
    def _register_builtin_functions(self):
        """Register built-in task functions"""
        self.function_registry.update({
            "generate_document": self._builtin_generate_document,
            "validate_document": self._builtin_validate_document,
            "export_document": self._builtin_export_document,
            "send_notification": self._builtin_send_notification,
            "cleanup_resources": self._builtin_cleanup_resources
        })
    
    def register_function(self, name: str, function: Callable):
        """Register a task function"""
        self.function_registry[name] = function
        self.logger.debug(f"Registered function: {name}")
    
    def create_task(
        self,
        name: str,
        task_type: TaskType,
        function: Union[Callable, str],
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new task.
        
        Args:
            name: Task name
            task_type: Type of task
            function: Task function or registered function name
            args: Function arguments
            kwargs: Function keyword arguments
            dependencies: List of task IDs this task depends on
            retry_policy: Retry configuration
            timeout: Task timeout
            priority: Task priority
            metadata: Additional metadata
            
        Returns:
            Task ID
        """
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        # Default retry policy
        if retry_policy is None:
            retry_policy = {
                "max_retries": self.config.default_max_retries,
                "retry_delay": self.config.default_retry_delay,
                "exponential_backoff": self.config.exponential_backoff
            }
        
        task = Task(
            task_id=task_id,
            name=name,
            task_type=task_type,
            function=function,
            args=args or [],
            kwargs=kwargs or {},
            dependencies=set(dependencies or []),
            retry_policy=retry_policy,
            timeout=timeout or self.config.task_timeout,
            priority=priority,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.tasks[task_id] = task
        
        self.logger.info(f"Task created: {task_id} - {name}")
        
        return task_id
    
    def create_task_graph(
        self,
        name: str,
        tasks: List[Dict[str, Any]],
        edges: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a task graph.
        
        Args:
            name: Graph name
            tasks: List of task definitions
            edges: List of edges (from_task, to_task)
            metadata: Additional metadata
            
        Returns:
            Graph ID
        """
        graph_id = f"graph_{uuid.uuid4().hex[:12]}"
        
        # Create tasks
        graph_tasks = {}
        task_id_map = {}  # Map from provided IDs to generated IDs
        
        for task_def in tasks:
            provided_id = task_def.pop("id", None)
            task_id = self.create_task(**task_def)
            
            if provided_id:
                task_id_map[provided_id] = task_id
            
            graph_tasks[task_id] = self.tasks[task_id]
        
        # Map edges to actual task IDs
        mapped_edges = []
        for from_id, to_id in edges:
            from_task_id = task_id_map.get(from_id, from_id)
            to_task_id = task_id_map.get(to_id, to_id)
            mapped_edges.append((from_task_id, to_task_id))
            
            # Update dependencies
            if to_task_id in graph_tasks:
                graph_tasks[to_task_id].dependencies.add(from_task_id)
        
        # Create graph
        task_graph = TaskGraph(
            graph_id=graph_id,
            name=name,
            tasks=graph_tasks,
            edges=mapped_edges,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.task_graphs[graph_id] = task_graph
        
        # Validate graph (check for cycles)
        G = task_graph.to_networkx()
        if not nx.is_directed_acyclic_graph(G):
            raise ValidationError(f"Task graph {graph_id} contains cycles")
        
        self.logger.info(f"Task graph created: {graph_id} - {name} ({len(graph_tasks)} tasks)")
        
        return graph_id
    
    async def execute_task_async(self, task_id: str) -> TaskResult:
        """Execute a single task asynchronously"""
        if task_id not in self.tasks:
            raise ValidationError(f"Task not found: {task_id}")
        
        task = self.tasks[task_id]
        
        # Check dependencies
        for dep_id in task.dependencies:
            dep_result = self.task_results.get(dep_id)
            if not dep_result or dep_result.status != TaskStatus.SUCCEEDED:
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.SKIPPED,
                    error=f"Dependency {dep_id} not satisfied"
                )
        
        # Execute task
        return await self._execute_task_with_retry(task)
    
    def execute_task(self, task_id: str) -> TaskResult:
        """Execute a single task synchronously"""
        if self.config.enable_async and self.event_loop:
            future = asyncio.run_coroutine_threadsafe(
                self.execute_task_async(task_id),
                self.event_loop
            )
            return future.result()
        else:
            return self._execute_task_sync(task_id)
    
    def _execute_task_sync(self, task_id: str) -> TaskResult:
        """Execute task synchronously"""
        if task_id not in self.tasks:
            raise ValidationError(f"Task not found: {task_id}")
        
        task = self.tasks[task_id]
        
        # Check dependencies
        for dep_id in task.dependencies:
            dep_result = self.task_results.get(dep_id)
            if not dep_result or dep_result.status != TaskStatus.SUCCEEDED:
                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.SKIPPED,
                    error=f"Dependency {dep_id} not satisfied"
                )
        
        # Execute task with retries
        max_retries = task.retry_policy.get("max_retries", 0)
        retry_delay = task.retry_policy.get("retry_delay", 60)
        
        for attempt in range(max_retries + 1):
            try:
                result = self._execute_single_task(task)
                if result.status == TaskStatus.SUCCEEDED:
                    return result
                
                if attempt < max_retries:
                    self.logger.warning(f"Task {task_id} failed, retrying ({attempt + 1}/{max_retries})")
                    if task.retry_policy.get("exponential_backoff"):
                        time.sleep(retry_delay * (2 ** attempt))
                    else:
                        time.sleep(retry_delay)
                    
                    result.status = TaskStatus.RETRYING
                    self._update_task_result(result)
                
            except Exception as e:
                if attempt == max_retries:
                    return TaskResult(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error=str(e),
                        start_time=datetime.now(),
                        end_time=datetime.now()
                    )
        
        return result
    
    async def _execute_task_with_retry(self, task: Task) -> TaskResult:
        """Execute task with retry logic"""
        max_retries = task.retry_policy.get("max_retries", 0)
        retry_delay = task.retry_policy.get("retry_delay", 60)
        
        for attempt in range(max_retries + 1):
            try:
                result = await self._execute_single_task_async(task)
                if result.status == TaskStatus.SUCCEEDED:
                    return result
                
                if attempt < max_retries:
                    self.logger.warning(f"Task {task.task_id} failed, retrying ({attempt + 1}/{max_retries})")
                    if task.retry_policy.get("exponential_backoff"):
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                    else:
                        await asyncio.sleep(retry_delay)
                    
                    result.status = TaskStatus.RETRYING
                    self._update_task_result(result)
                
            except Exception as e:
                if attempt == max_retries:
                    return TaskResult(
                        task_id=task.task_id,
                        status=TaskStatus.FAILED,
                        error=str(e),
                        start_time=datetime.now(),
                        end_time=datetime.now()
                    )
        
        return result
    
    def _execute_single_task(self, task: Task) -> TaskResult:
        """Execute a single task without retries"""
        start_time = datetime.now()
        
        try:
            # Get function
            if isinstance(task.function, str):
                if task.function not in self.function_registry:
                    raise ValueError(f"Function not found: {task.function}")
                func = self.function_registry[task.function]
            else:
                func = task.function
            
            # Update progress
            self._update_progress(task.task_id, 0.0)
            
            # Execute with timeout
            future = self.thread_executor.submit(func, *task.args, **task.kwargs)
            output = future.result(timeout=task.timeout)
            
            # Update progress
            self._update_progress(task.task_id, 1.0)
            
            end_time = datetime.now()
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.SUCCEEDED,
                output=output,
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds()
            )
            
        except Exception as e:
            end_time = datetime.now()
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds()
            )
        
        self._update_task_result(result)
        return result
    
    async def _execute_single_task_async(self, task: Task) -> TaskResult:
        """Execute a single task asynchronously"""
        start_time = datetime.now()
        
        try:
            # Get function
            if isinstance(task.function, str):
                if task.function not in self.function_registry:
                    raise ValueError(f"Function not found: {task.function}")
                func = self.function_registry[task.function]
            else:
                func = task.function
            
            # Update progress
            self._update_progress(task.task_id, 0.0)
            
            # Execute with timeout
            if asyncio.iscoroutinefunction(func):
                output = await asyncio.wait_for(
                    func(*task.args, **task.kwargs),
                    timeout=task.timeout
                )
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(
                    self.thread_executor,
                    func,
                    *task.args,
                    **task.kwargs
                )
            
            # Update progress
            self._update_progress(task.task_id, 1.0)
            
            end_time = datetime.now()
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.SUCCEEDED,
                output=output,
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds()
            )
            
        except asyncio.TimeoutError:
            end_time = datetime.now()
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=f"Task timed out after {task.timeout} seconds",
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds()
            )
        except Exception as e:
            end_time = datetime.now()
            result = TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                start_time=start_time,
                end_time=end_time,
                duration=(end_time - start_time).total_seconds()
            )
        
        self._update_task_result(result)
        return result
    
    async def execute_task_graph_async(self, graph_id: str) -> Dict[str, TaskResult]:
        """Execute a task graph asynchronously"""
        if graph_id not in self.task_graphs:
            raise ValidationError(f"Task graph not found: {graph_id}")
        
        task_graph = self.task_graphs[graph_id]
        G = task_graph.to_networkx()
        
        # Topological sort for execution order
        try:
            execution_order = list(nx.topological_sort(G))
        except nx.NetworkXError:
            raise ValidationError(f"Task graph {graph_id} contains cycles")
        
        # Execute tasks in parallel where possible
        results = {}
        completed = set()
        
        while len(completed) < len(execution_order):
            # Find tasks ready to execute
            ready_tasks = []
            for task_id in execution_order:
                if task_id in completed:
                    continue
                
                # Check if all dependencies are completed
                task = task_graph.tasks[task_id]
                if all(dep_id in completed for dep_id in task.dependencies):
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                # No tasks ready, might be a dependency issue
                break
            
            # Execute ready tasks in parallel
            task_futures = []
            for task_id in ready_tasks:
                future = asyncio.create_task(self.execute_task_async(task_id))
                task_futures.append((task_id, future))
            
            # Wait for tasks to complete
            for task_id, future in task_futures:
                result = await future
                results[task_id] = result
                if result.status == TaskStatus.SUCCEEDED:
                    completed.add(task_id)
        
        return results
    
    def execute_task_graph(self, graph_id: str) -> Dict[str, TaskResult]:
        """Execute a task graph synchronously"""
        if self.config.enable_async:
            # Create event loop if needed
            if not self.event_loop:
                self.event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.event_loop)
            
            return self.event_loop.run_until_complete(
                self.execute_task_graph_async(graph_id)
            )
        else:
            return self._execute_task_graph_sync(graph_id)
    
    def _execute_task_graph_sync(self, graph_id: str) -> Dict[str, TaskResult]:
        """Execute task graph synchronously"""
        if graph_id not in self.task_graphs:
            raise ValidationError(f"Task graph not found: {graph_id}")
        
        task_graph = self.task_graphs[graph_id]
        G = task_graph.to_networkx()
        
        # Topological sort for execution order
        try:
            execution_order = list(nx.topological_sort(G))
        except nx.NetworkXError:
            raise ValidationError(f"Task graph {graph_id} contains cycles")
        
        # Execute tasks
        results = {}
        
        for task_id in execution_order:
            result = self.execute_task(task_id)
            results[task_id] = result
            
            if result.status == TaskStatus.FAILED:
                self.logger.error(f"Task {task_id} failed, stopping graph execution")
                break
        
        return results
    
    def _update_task_result(self, result: TaskResult):
        """Update task result"""
        with self._lock:
            self.task_results[result.task_id] = result
        
        # Update state if state manager available
        if self.state_manager:
            self.state_manager.set(
                f"task_result:{result.task_id}",
                result,
                StateType.JOB
            )
    
    def _update_progress(self, task_id: str, progress: float):
        """Update task progress"""
        with self._lock:
            self.task_progress[task_id] = progress
        
        # Notify callbacks
        for callback in self.progress_callbacks:
            try:
                callback(task_id, progress)
            except Exception as e:
                self.logger.error(f"Progress callback failed: {e}")
    
    def register_progress_callback(self, callback: Callable):
        """Register progress callback"""
        self.progress_callbacks.append(callback)
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get task result"""
        with self._lock:
            return self.task_results.get(task_id)
    
    def get_task_progress(self, task_id: str) -> float:
        """Get task progress"""
        with self._lock:
            return self.task_progress.get(task_id, 0.0)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        with self._lock:
            if task_id in self.running_tasks:
                future = self.running_tasks[task_id]
                if future.cancel():
                    result = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.CANCELLED,
                        end_time=datetime.now()
                    )
                    self._update_task_result(result)
                    return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics"""
        with self._lock:
            status_counts = defaultdict(int)
            for result in self.task_results.values():
                status_counts[result.status.value] += 1
            
            return {
                "total_tasks": len(self.tasks),
                "total_graphs": len(self.task_graphs),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len(self.task_results),
                "status_counts": dict(status_counts),
                "registered_functions": len(self.function_registry)
            }
    
    def _builtin_generate_document(self, doc_type: str, **kwargs) -> Dict[str, Any]:
        """Built-in document generation function"""
        time.sleep(2)  # Simulate work
        return {
            "status": "success",
            "document_id": f"doc_{uuid.uuid4().hex[:8]}",
            "type": doc_type
        }
    
    def _builtin_validate_document(self, document_id: str, **kwargs) -> Dict[str, Any]:
        """Built-in document validation function"""
        time.sleep(1)  # Simulate work
        return {
            "status": "success",
            "document_id": document_id,
            "valid": True
        }
    
    def _builtin_export_document(self, document_id: str, format: str, **kwargs) -> Dict[str, Any]:
        """Built-in document export function"""
        time.sleep(3)  # Simulate work
        return {
            "status": "success",
            "document_id": document_id,
            "export_path": f"/exports/{document_id}.{format}"
        }
    
    def _builtin_send_notification(self, message: str, **kwargs) -> Dict[str, Any]:
        """Built-in notification function"""
        time.sleep(0.5)  # Simulate work
        return {
            "status": "success",
            "message": message,
            "sent_at": datetime.now().isoformat()
        }
    
    def _builtin_cleanup_resources(self, **kwargs) -> Dict[str, Any]:
        """Built-in cleanup function"""
        time.sleep(1)  # Simulate work
        return {
            "status": "success",
            "cleaned": True
        }
    
    def _save_state(self):
        """Save task manager state"""
        # Implementation would persist tasks and results
        pass
    
    def _load_state(self):
        """Load task manager state"""
        # Implementation would restore tasks and results
        pass
    
    def cleanup(self):
        """Clean up task manager"""
        self.is_running = False
        
        # Cancel running tasks
        for future in self.running_tasks.values():
            future.cancel()
        
        # Shutdown executors
        self.thread_executor.shutdown(wait=True)
        
        # Save state
        if self.config.enable_persistence:
            self._save_state()
        
        self.logger.info("Task manager cleaned up")


# Factory function
def create_task_manager(
    config: Optional[Dict[str, Any]] = None,
    job_queue: Optional[JobQueue] = None,
    state_manager: Optional[StateManager] = None,
    resource_manager: Optional[ResourceManager] = None
) -> TaskManager:
    """Create and return a task manager instance"""
    return TaskManager(config, job_queue, state_manager, resource_manager)