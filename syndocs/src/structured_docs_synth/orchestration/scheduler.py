#!/usr/bin/env python3
"""
Advanced Scheduler for orchestrating document generation tasks.

Provides sophisticated scheduling capabilities including priority-based scheduling,
dependency management, resource-aware scheduling, and automatic load balancing.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import heapq
import threading
from concurrent.futures import ThreadPoolExecutor, Future

from pydantic import BaseModel, Field

from ..core.config import get_config
from ..core.logging import get_logger
from ..core.exceptions import ProcessingError, ValidationError
from .job_queue import Job, JobStatus, JobPriority
from .resource_manager import ResourceManager, ResourceType


logger = get_logger(__name__)


class SchedulingStrategy(Enum):
    """Scheduling strategies"""
    FIFO = "fifo"                    # First In First Out
    PRIORITY = "priority"            # Priority-based
    ROUND_ROBIN = "round_robin"      # Round-robin across queues
    RESOURCE_AWARE = "resource_aware"  # Consider resource availability
    DEADLINE = "deadline"            # Deadline-driven
    FAIR_SHARE = "fair_share"        # Fair share among users/groups


class TaskState(Enum):
    """Task execution states"""
    WAITING = "waiting"
    READY = "ready"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class ScheduledTask:
    """Scheduled task representation"""
    task_id: str
    name: str
    priority: int
    dependencies: Set[str] = field(default_factory=set)
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    estimated_duration: float = 60.0
    deadline: Optional[datetime] = None
    state: TaskState = TaskState.WAITING
    scheduled_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    assigned_worker: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare tasks for priority queue"""
        # Higher priority = lower number (min heap)
        if self.priority != other.priority:
            return self.priority < other.priority
        # If same priority, prefer earlier deadline
        if self.deadline and other.deadline:
            return self.deadline < other.deadline
        return self.task_id < other.task_id


@dataclass
class WorkerNode:
    """Worker node representation"""
    worker_id: str
    capacity: int = 1
    current_load: int = 0
    resource_capacity: Dict[ResourceType, float] = field(default_factory=dict)
    active_tasks: Set[str] = field(default_factory=set)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SchedulerConfig(BaseModel):
    """Scheduler configuration"""
    strategy: SchedulingStrategy = Field(SchedulingStrategy.RESOURCE_AWARE, description="Scheduling strategy")
    max_concurrent_tasks: int = Field(100, description="Maximum concurrent tasks")
    max_queue_size: int = Field(10000, description="Maximum queue size")
    schedule_interval: float = Field(1.0, description="Scheduling interval in seconds")
    
    # Worker settings
    worker_heartbeat_timeout: float = Field(60.0, description="Worker heartbeat timeout")
    load_balance_interval: float = Field(30.0, description="Load balancing interval")
    
    # Resource settings
    resource_check_enabled: bool = Field(True, description="Enable resource checking")
    resource_buffer_percent: float = Field(10.0, description="Resource buffer percentage")
    
    # Deadline settings
    deadline_slack_minutes: int = Field(5, description="Deadline slack time in minutes")
    
    # Fair share settings
    fair_share_window_minutes: int = Field(60, description="Fair share calculation window")
    
    # Persistence
    enable_persistence: bool = Field(True, description="Enable state persistence")
    persistence_path: str = Field("./scheduler_state.json", description="State persistence path")


class Scheduler:
    """
    Advanced task scheduler with multiple scheduling strategies.
    
    Features:
    - Multiple scheduling strategies
    - Dependency management
    - Resource-aware scheduling
    - Deadline-based scheduling
    - Load balancing
    - Worker management
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        resource_manager: Optional[ResourceManager] = None
    ):
        """Initialize scheduler"""
        self.config = SchedulerConfig(**(config or {}))
        self.logger = get_logger(__name__)
        
        # Resource manager
        self.resource_manager = resource_manager
        
        # Task storage
        self.tasks: Dict[str, ScheduledTask] = {}
        self.task_queue: List[ScheduledTask] = []  # Priority queue
        self.ready_queue: deque = deque()
        
        # Worker management
        self.workers: Dict[str, WorkerNode] = {}
        self.worker_lock = threading.Lock()
        
        # Dependency tracking
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        # Execution tracking
        self.running_tasks: Dict[str, Future] = {}
        self.completed_tasks: Set[str] = set()
        
        # Fair share tracking
        self.user_usage: Dict[str, float] = defaultdict(float)
        self.group_usage: Dict[str, float] = defaultdict(float)
        
        # Control
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_tasks)
        
        # Load state if persistence enabled
        if self.config.enable_persistence:
            self._load_state()
        
        self.logger.info(f"Scheduler initialized with strategy: {self.config.strategy.value}")
    
    def add_task(
        self,
        name: str,
        priority: int = 5,
        dependencies: Optional[List[str]] = None,
        resource_requirements: Optional[Dict[ResourceType, float]] = None,
        estimated_duration: float = 60.0,
        deadline: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a task to the scheduler.
        
        Args:
            name: Task name
            priority: Task priority (1=highest, 10=lowest)
            dependencies: List of task IDs this task depends on
            resource_requirements: Required resources
            estimated_duration: Estimated duration in seconds
            deadline: Task deadline
            metadata: Additional metadata
            
        Returns:
            Task ID
        """
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        task = ScheduledTask(
            task_id=task_id,
            name=name,
            priority=priority,
            dependencies=set(dependencies or []),
            resource_requirements=resource_requirements or {},
            estimated_duration=estimated_duration,
            deadline=deadline,
            metadata=metadata or {}
        )
        
        # Store task
        self.tasks[task_id] = task
        
        # Update dependency graph
        for dep_id in task.dependencies:
            self.dependency_graph[task_id].add(dep_id)
            self.reverse_dependencies[dep_id].add(task_id)
        
        # Add to queue if ready
        if self._is_task_ready(task):
            task.state = TaskState.READY
            heapq.heappush(self.task_queue, task)
        
        self.logger.info(f"Task added: {task_id} - {name} (priority: {priority})")
        
        # Persist state
        if self.config.enable_persistence:
            self._save_state()
        
        return task_id
    
    def add_worker(
        self,
        worker_id: str,
        capacity: int = 1,
        resource_capacity: Optional[Dict[ResourceType, float]] = None
    ) -> WorkerNode:
        """Add a worker node"""
        with self.worker_lock:
            worker = WorkerNode(
                worker_id=worker_id,
                capacity=capacity,
                resource_capacity=resource_capacity or {}
            )
            self.workers[worker_id] = worker
            
            self.logger.info(f"Worker added: {worker_id} (capacity: {capacity})")
            return worker
    
    def remove_worker(self, worker_id: str):
        """Remove a worker node"""
        with self.worker_lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                
                # Reschedule any active tasks
                for task_id in worker.active_tasks:
                    if task_id in self.tasks:
                        task = self.tasks[task_id]
                        task.state = TaskState.READY
                        task.assigned_worker = None
                        heapq.heappush(self.task_queue, task)
                
                del self.workers[worker_id]
                self.logger.info(f"Worker removed: {worker_id}")
    
    def start(self):
        """Start the scheduler"""
        if self.is_running:
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="Scheduler",
            daemon=True
        )
        self.scheduler_thread.start()
        
        self.logger.info("Scheduler started")
    
    def stop(self, wait: bool = True):
        """Stop the scheduler"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping scheduler...")
        self.is_running = False
        
        if wait and self.scheduler_thread:
            self.scheduler_thread.join(timeout=10.0)
        
        # Cancel running tasks
        for future in self.running_tasks.values():
            future.cancel()
        
        self.executor.shutdown(wait=wait)
        
        # Save state
        if self.config.enable_persistence:
            self._save_state()
        
        self.logger.info("Scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        last_load_balance = time.time()
        
        while self.is_running:
            try:
                # Update worker heartbeats
                self._check_worker_health()
                
                # Check completed tasks
                self._check_completed_tasks()
                
                # Schedule ready tasks
                scheduled_count = self._schedule_tasks()
                
                # Load balance if needed
                if time.time() - last_load_balance > self.config.load_balance_interval:
                    self._load_balance()
                    last_load_balance = time.time()
                
                # Sleep if no tasks scheduled
                if scheduled_count == 0:
                    time.sleep(self.config.schedule_interval)
                
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(self.config.schedule_interval)
    
    def _is_task_ready(self, task: ScheduledTask) -> bool:
        """Check if task is ready to run"""
        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        
        # Check resources if enabled
        if self.config.resource_check_enabled and self.resource_manager:
            for resource_type, required in task.resource_requirements.items():
                if not self.resource_manager.is_resource_available(resource_type, required):
                    return False
        
        return True
    
    def _schedule_tasks(self) -> int:
        """Schedule ready tasks based on strategy"""
        scheduled_count = 0
        
        if self.config.strategy == SchedulingStrategy.FIFO:
            scheduled_count = self._schedule_fifo()
        elif self.config.strategy == SchedulingStrategy.PRIORITY:
            scheduled_count = self._schedule_priority()
        elif self.config.strategy == SchedulingStrategy.RESOURCE_AWARE:
            scheduled_count = self._schedule_resource_aware()
        elif self.config.strategy == SchedulingStrategy.DEADLINE:
            scheduled_count = self._schedule_deadline()
        elif self.config.strategy == SchedulingStrategy.FAIR_SHARE:
            scheduled_count = self._schedule_fair_share()
        else:
            scheduled_count = self._schedule_priority()  # Default
        
        return scheduled_count
    
    def _schedule_priority(self) -> int:
        """Priority-based scheduling"""
        scheduled_count = 0
        
        # Process tasks from priority queue
        temp_queue = []
        
        while self.task_queue and scheduled_count < self.config.max_concurrent_tasks:
            if not self.task_queue:
                break
            
            task = heapq.heappop(self.task_queue)
            
            # Check if still ready
            if not self._is_task_ready(task):
                task.state = TaskState.BLOCKED
                temp_queue.append(task)
                continue
            
            # Find available worker
            worker = self._find_available_worker(task)
            if worker:
                self._assign_task_to_worker(task, worker)
                scheduled_count += 1
            else:
                temp_queue.append(task)
                break
        
        # Re-add unscheduled tasks
        for task in temp_queue:
            heapq.heappush(self.task_queue, task)
        
        return scheduled_count
    
    def _schedule_resource_aware(self) -> int:
        """Resource-aware scheduling"""
        scheduled_count = 0
        
        if not self.resource_manager:
            return self._schedule_priority()
        
        # Get current resource availability
        current_resources = self.resource_manager.get_current_metrics()
        
        # Sort tasks by resource efficiency
        ready_tasks = []
        temp_queue = []
        
        while self.task_queue:
            task = heapq.heappop(self.task_queue)
            if self._is_task_ready(task):
                ready_tasks.append(task)
            else:
                temp_queue.append(task)
        
        # Sort by resource fit
        ready_tasks.sort(key=lambda t: self._calculate_resource_fit(t, current_resources))
        
        # Schedule tasks
        for task in ready_tasks:
            worker = self._find_available_worker(task)
            if worker and self._check_resources_available(task):
                self._assign_task_to_worker(task, worker)
                scheduled_count += 1
            else:
                temp_queue.append(task)
        
        # Re-add unscheduled tasks
        for task in temp_queue:
            heapq.heappush(self.task_queue, task)
        
        return scheduled_count
    
    def _schedule_deadline(self) -> int:
        """Deadline-based scheduling (EDF - Earliest Deadline First)"""
        scheduled_count = 0
        
        # Get all ready tasks with deadlines
        ready_tasks = []
        temp_queue = []
        
        while self.task_queue:
            task = heapq.heappop(self.task_queue)
            if self._is_task_ready(task):
                ready_tasks.append(task)
            else:
                temp_queue.append(task)
        
        # Sort by deadline (earliest first)
        ready_tasks.sort(key=lambda t: t.deadline or datetime.max)
        
        # Schedule tasks
        for task in ready_tasks:
            # Check if deadline can be met
            if task.deadline:
                estimated_completion = datetime.now() + timedelta(seconds=task.estimated_duration)
                if estimated_completion > task.deadline:
                    self.logger.warning(f"Task {task.task_id} may miss deadline")
            
            worker = self._find_available_worker(task)
            if worker:
                self._assign_task_to_worker(task, worker)
                scheduled_count += 1
            else:
                temp_queue.append(task)
        
        # Re-add unscheduled tasks
        for task in temp_queue:
            heapq.heappush(self.task_queue, task)
        
        return scheduled_count
    
    def _schedule_fair_share(self) -> int:
        """Fair share scheduling"""
        scheduled_count = 0
        
        # Calculate fair share quotas
        window_start = datetime.now() - timedelta(minutes=self.config.fair_share_window_minutes)
        
        # Group tasks by user/group
        user_tasks = defaultdict(list)
        temp_queue = []
        
        while self.task_queue:
            task = heapq.heappop(self.task_queue)
            if self._is_task_ready(task):
                user = task.metadata.get("user", "default")
                user_tasks[user].append(task)
            else:
                temp_queue.append(task)
        
        # Calculate usage ratios
        total_usage = sum(self.user_usage.values()) or 1.0
        
        # Schedule tasks fairly
        for user, tasks in user_tasks.items():
            user_ratio = self.user_usage[user] / total_usage
            allowed_tasks = max(1, int((1.0 - user_ratio) * len(tasks)))
            
            for i, task in enumerate(tasks[:allowed_tasks]):
                worker = self._find_available_worker(task)
                if worker:
                    self._assign_task_to_worker(task, worker)
                    scheduled_count += 1
                    self.user_usage[user] += task.estimated_duration
                else:
                    temp_queue.extend(tasks[i:])
                    break
            
            # Re-queue remaining tasks
            if len(tasks) > allowed_tasks:
                temp_queue.extend(tasks[allowed_tasks:])
        
        # Re-add unscheduled tasks
        for task in temp_queue:
            heapq.heappush(self.task_queue, task)
        
        return scheduled_count
    
    def _schedule_fifo(self) -> int:
        """FIFO scheduling"""
        # Convert to FIFO by ignoring priority
        scheduled_count = 0
        all_tasks = []
        
        while self.task_queue:
            all_tasks.append(heapq.heappop(self.task_queue))
        
        # Sort by task creation time (task_id contains timestamp)
        all_tasks.sort(key=lambda t: t.task_id)
        
        temp_queue = []
        for task in all_tasks:
            if self._is_task_ready(task):
                worker = self._find_available_worker(task)
                if worker:
                    self._assign_task_to_worker(task, worker)
                    scheduled_count += 1
                else:
                    temp_queue.append(task)
            else:
                temp_queue.append(task)
        
        # Re-add unscheduled tasks
        for task in temp_queue:
            heapq.heappush(self.task_queue, task)
        
        return scheduled_count
    
    def _find_available_worker(self, task: ScheduledTask) -> Optional[WorkerNode]:
        """Find available worker for task"""
        with self.worker_lock:
            best_worker = None
            min_load = float('inf')
            
            for worker in self.workers.values():
                # Check capacity
                if worker.current_load >= worker.capacity:
                    continue
                
                # Check resource capacity
                can_handle = True
                for resource_type, required in task.resource_requirements.items():
                    if resource_type in worker.resource_capacity:
                        if worker.resource_capacity[resource_type] < required:
                            can_handle = False
                            break
                
                if can_handle and worker.current_load < min_load:
                    best_worker = worker
                    min_load = worker.current_load
            
            return best_worker
    
    def _assign_task_to_worker(self, task: ScheduledTask, worker: WorkerNode):
        """Assign task to worker"""
        task.state = TaskState.SCHEDULED
        task.scheduled_time = datetime.now()
        task.assigned_worker = worker.worker_id
        
        worker.current_load += 1
        worker.active_tasks.add(task.task_id)
        
        # Submit task for execution
        future = self.executor.submit(self._execute_task, task)
        self.running_tasks[task.task_id] = future
        
        self.logger.info(f"Task {task.task_id} scheduled on worker {worker.worker_id}")
    
    def _execute_task(self, task: ScheduledTask) -> Any:
        """Execute a scheduled task"""
        task.state = TaskState.RUNNING
        task.start_time = datetime.now()
        
        try:
            # Simulate task execution
            time.sleep(min(task.estimated_duration, 5.0))  # Cap at 5 seconds for demo
            
            # Mark as completed
            task.state = TaskState.COMPLETED
            task.completion_time = datetime.now()
            
            result = {
                "task_id": task.task_id,
                "status": "completed",
                "duration": (task.completion_time - task.start_time).total_seconds()
            }
            
            self.logger.info(f"Task {task.task_id} completed")
            return result
            
        except Exception as e:
            task.state = TaskState.FAILED
            task.completion_time = datetime.now()
            self.logger.error(f"Task {task.task_id} failed: {e}")
            raise
    
    def _check_completed_tasks(self):
        """Check and process completed tasks"""
        completed = []
        
        for task_id, future in list(self.running_tasks.items()):
            if future.done():
                completed.append(task_id)
                
                task = self.tasks[task_id]
                worker_id = task.assigned_worker
                
                # Update worker
                if worker_id and worker_id in self.workers:
                    worker = self.workers[worker_id]
                    worker.current_load -= 1
                    worker.active_tasks.discard(task_id)
                
                # Add to completed set
                self.completed_tasks.add(task_id)
                
                # Check dependent tasks
                self._check_dependent_tasks(task_id)
        
        # Remove from running tasks
        for task_id in completed:
            del self.running_tasks[task_id]
    
    def _check_dependent_tasks(self, completed_task_id: str):
        """Check tasks dependent on completed task"""
        dependent_tasks = self.reverse_dependencies.get(completed_task_id, set())
        
        for task_id in dependent_tasks:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.state == TaskState.WAITING and self._is_task_ready(task):
                    task.state = TaskState.READY
                    heapq.heappush(self.task_queue, task)
    
    def _check_worker_health(self):
        """Check worker health and remove stale workers"""
        with self.worker_lock:
            stale_workers = []
            current_time = datetime.now()
            
            for worker_id, worker in self.workers.items():
                time_since_heartbeat = (current_time - worker.last_heartbeat).total_seconds()
                if time_since_heartbeat > self.config.worker_heartbeat_timeout:
                    stale_workers.append(worker_id)
            
            for worker_id in stale_workers:
                self.remove_worker(worker_id)
    
    def _load_balance(self):
        """Perform load balancing across workers"""
        with self.worker_lock:
            if len(self.workers) < 2:
                return
            
            # Calculate average load
            total_load = sum(w.current_load for w in self.workers.values())
            avg_load = total_load / len(self.workers)
            
            # Find overloaded and underloaded workers
            overloaded = []
            underloaded = []
            
            for worker in self.workers.values():
                if worker.current_load > avg_load * 1.5:
                    overloaded.append(worker)
                elif worker.current_load < avg_load * 0.5:
                    underloaded.append(worker)
            
            # Rebalance if needed
            if overloaded and underloaded:
                self.logger.info("Performing load balancing")
                # In a real implementation, would migrate tasks
    
    def _calculate_resource_fit(
        self,
        task: ScheduledTask,
        current_resources: Dict[ResourceType, Any]
    ) -> float:
        """Calculate how well task fits current resources"""
        if not task.resource_requirements:
            return 0.0
        
        fit_score = 0.0
        for resource_type, required in task.resource_requirements.items():
            if resource_type in current_resources:
                metric = current_resources[resource_type]
                available = 100.0 - metric.usage_percent
                fit_score += max(0, available - required)
        
        return fit_score
    
    def _check_resources_available(self, task: ScheduledTask) -> bool:
        """Check if resources are available for task"""
        if not self.resource_manager:
            return True
        
        for resource_type, required in task.resource_requirements.items():
            buffer = self.config.resource_buffer_percent
            if not self.resource_manager.is_resource_available(resource_type, required + buffer):
                return False
        
        return True
    
    def update_worker_heartbeat(self, worker_id: str):
        """Update worker heartbeat"""
        with self.worker_lock:
            if worker_id in self.workers:
                self.workers[worker_id].last_heartbeat = datetime.now()
    
    def get_task_status(self, task_id: str) -> Optional[TaskState]:
        """Get task status"""
        task = self.tasks.get(task_id)
        return task.state if task else None
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        with self.worker_lock:
            total_capacity = sum(w.capacity for w in self.workers.values())
            total_load = sum(w.current_load for w in self.workers.values())
        
        return {
            "total_tasks": len(self.tasks),
            "queued_tasks": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "workers": len(self.workers),
            "total_capacity": total_capacity,
            "current_load": total_load,
            "utilization": (total_load / total_capacity * 100) if total_capacity > 0 else 0,
            "strategy": self.config.strategy.value
        }
    
    def _save_state(self):
        """Save scheduler state"""
        # Implementation would persist task and worker state
        pass
    
    def _load_state(self):
        """Load scheduler state"""
        # Implementation would restore task and worker state
        pass


# Factory function
def create_scheduler(
    config: Optional[Dict[str, Any]] = None,
    resource_manager: Optional[ResourceManager] = None
) -> Scheduler:
    """Create and return a scheduler instance"""
    return Scheduler(config, resource_manager)