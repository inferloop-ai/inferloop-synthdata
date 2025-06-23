"""
Resource lifecycle management for infrastructure resources
"""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

from ..abstractions.base import ResourceState, BaseResource


logger = logging.getLogger(__name__)


class LifecycleState(Enum):
    """Lifecycle states for resources"""
    INITIALIZING = "initializing"
    CREATING = "creating"
    CONFIGURING = "configuring"
    READY = "ready"
    UPDATING = "updating"
    MAINTAINING = "maintaining"
    DEGRADED = "degraded"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    ERROR = "error"


class LifecycleEvent(Enum):
    """Lifecycle events"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    START = "start"
    STOP = "stop"
    RESTART = "restart"
    HEALTH_CHECK = "health_check"
    BACKUP = "backup"
    RESTORE = "restore"
    SCALE = "scale"


@dataclass
class ResourceLifecycle:
    """Lifecycle information for a resource"""
    resource_id: str
    resource_type: str
    current_state: LifecycleState
    previous_state: Optional[LifecycleState] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    health_status: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_event(self, event: LifecycleEvent, details: Dict[str, Any] = None):
        """Add lifecycle event"""
        self.events.append({
            'event': event.value,
            'timestamp': datetime.now(),
            'details': details or {}
        })
        self.updated_at = datetime.now()
    
    def transition_state(self, new_state: LifecycleState):
        """Transition to new state"""
        self.previous_state = self.current_state
        self.current_state = new_state
        self.updated_at = datetime.now()


class ResourceLifecycleManager:
    """Manages lifecycle of infrastructure resources"""
    
    def __init__(self):
        self._resources: Dict[str, ResourceLifecycle] = {}
        self._hooks: Dict[LifecycleEvent, List[Callable]] = {}
        self._health_check_intervals: Dict[str, int] = {}
        self._health_check_tasks: Dict[str, asyncio.Task] = {}
        
    def register_resource(self, resource_id: str, resource_type: str,
                         initial_state: LifecycleState = LifecycleState.INITIALIZING) -> ResourceLifecycle:
        """Register a new resource for lifecycle management"""
        lifecycle = ResourceLifecycle(
            resource_id=resource_id,
            resource_type=resource_type,
            current_state=initial_state
        )
        self._resources[resource_id] = lifecycle
        logger.info(f"Registered resource {resource_id} of type {resource_type}")
        return lifecycle
    
    def unregister_resource(self, resource_id: str) -> bool:
        """Unregister resource from lifecycle management"""
        if resource_id in self._resources:
            # Cancel health check if running
            if resource_id in self._health_check_tasks:
                self._health_check_tasks[resource_id].cancel()
                del self._health_check_tasks[resource_id]
            
            del self._resources[resource_id]
            logger.info(f"Unregistered resource {resource_id}")
            return True
        return False
    
    def get_resource_lifecycle(self, resource_id: str) -> Optional[ResourceLifecycle]:
        """Get resource lifecycle information"""
        return self._resources.get(resource_id)
    
    def list_resources(self, resource_type: Optional[str] = None,
                      state: Optional[LifecycleState] = None) -> List[ResourceLifecycle]:
        """List resources with optional filters"""
        resources = list(self._resources.values())
        
        if resource_type:
            resources = [r for r in resources if r.resource_type == resource_type]
        
        if state:
            resources = [r for r in resources if r.current_state == state]
        
        return resources
    
    async def transition_state(self, resource_id: str, new_state: LifecycleState,
                             event: Optional[LifecycleEvent] = None,
                             details: Optional[Dict[str, Any]] = None) -> bool:
        """Transition resource to new state"""
        lifecycle = self.get_resource_lifecycle(resource_id)
        if not lifecycle:
            logger.error(f"Resource {resource_id} not found")
            return False
        
        old_state = lifecycle.current_state
        lifecycle.transition_state(new_state)
        
        if event:
            lifecycle.add_event(event, details)
        
        logger.info(f"Resource {resource_id} transitioned from {old_state} to {new_state}")
        
        # Execute hooks
        await self._execute_hooks(event, resource_id, old_state, new_state)
        
        return True
    
    def register_hook(self, event: LifecycleEvent, hook: Callable) -> None:
        """Register a lifecycle hook"""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(hook)
        logger.info(f"Registered hook for event {event}")
    
    async def _execute_hooks(self, event: Optional[LifecycleEvent], resource_id: str,
                           old_state: LifecycleState, new_state: LifecycleState) -> None:
        """Execute registered hooks for an event"""
        if event and event in self._hooks:
            for hook in self._hooks[event]:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook(resource_id, old_state, new_state)
                    else:
                        hook(resource_id, old_state, new_state)
                except Exception as e:
                    logger.error(f"Error executing hook for {event}: {str(e)}")
    
    async def start_health_monitoring(self, resource_id: str, resource: BaseResource,
                                    interval: int = 60) -> None:
        """Start health monitoring for a resource"""
        if resource_id in self._health_check_tasks:
            logger.warning(f"Health monitoring already active for {resource_id}")
            return
        
        self._health_check_intervals[resource_id] = interval
        task = asyncio.create_task(self._health_check_loop(resource_id, resource))
        self._health_check_tasks[resource_id] = task
        logger.info(f"Started health monitoring for {resource_id} with interval {interval}s")
    
    def stop_health_monitoring(self, resource_id: str) -> None:
        """Stop health monitoring for a resource"""
        if resource_id in self._health_check_tasks:
            self._health_check_tasks[resource_id].cancel()
            del self._health_check_tasks[resource_id]
            del self._health_check_intervals[resource_id]
            logger.info(f"Stopped health monitoring for {resource_id}")
    
    async def _health_check_loop(self, resource_id: str, resource: BaseResource) -> None:
        """Health check loop for a resource"""
        interval = self._health_check_intervals.get(resource_id, 60)
        
        while True:
            try:
                # Perform health check
                health_status = await self._perform_health_check(resource_id, resource)
                
                # Update lifecycle
                lifecycle = self.get_resource_lifecycle(resource_id)
                if lifecycle:
                    lifecycle.health_status = health_status
                    lifecycle.add_event(LifecycleEvent.HEALTH_CHECK, health_status)
                    
                    # Update state based on health
                    if health_status.get('healthy', False):
                        if lifecycle.current_state == LifecycleState.DEGRADED:
                            await self.transition_state(resource_id, LifecycleState.READY)
                    else:
                        if lifecycle.current_state == LifecycleState.READY:
                            await self.transition_state(resource_id, LifecycleState.DEGRADED)
                
                # Wait for next check
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check for {resource_id}: {str(e)}")
                await asyncio.sleep(interval)
    
    async def _perform_health_check(self, resource_id: str, resource: BaseResource) -> Dict[str, Any]:
        """Perform health check on a resource"""
        try:
            state = await resource.get_state(resource_id)
            
            health_status = {
                'healthy': state == ResourceState.RUNNING,
                'state': state.value,
                'timestamp': datetime.now().isoformat(),
                'checks': {}
            }
            
            # Additional health checks can be added here
            # For example, checking connectivity, performance metrics, etc.
            
            return health_status
            
        except Exception as e:
            return {
                'healthy': False,
                'state': 'unknown',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    async def execute_maintenance(self, resource_id: str, maintenance_type: str,
                                 params: Dict[str, Any] = None) -> bool:
        """Execute maintenance operation on a resource"""
        lifecycle = self.get_resource_lifecycle(resource_id)
        if not lifecycle:
            logger.error(f"Resource {resource_id} not found")
            return False
        
        # Transition to maintaining state
        await self.transition_state(resource_id, LifecycleState.MAINTAINING)
        
        try:
            # Execute maintenance operation
            # This would call specific maintenance methods based on type
            logger.info(f"Executing {maintenance_type} maintenance on {resource_id}")
            
            # Simulate maintenance
            await asyncio.sleep(5)
            
            # Transition back to ready
            await self.transition_state(resource_id, LifecycleState.READY)
            return True
            
        except Exception as e:
            logger.error(f"Maintenance failed for {resource_id}: {str(e)}")
            await self.transition_state(resource_id, LifecycleState.ERROR)
            return False
    
    def get_resource_age(self, resource_id: str) -> Optional[timedelta]:
        """Get age of a resource"""
        lifecycle = self.get_resource_lifecycle(resource_id)
        if lifecycle:
            return datetime.now() - lifecycle.created_at
        return None
    
    def get_resources_by_age(self, min_age: Optional[timedelta] = None,
                           max_age: Optional[timedelta] = None) -> List[ResourceLifecycle]:
        """Get resources filtered by age"""
        resources = []
        now = datetime.now()
        
        for lifecycle in self._resources.values():
            age = now - lifecycle.created_at
            
            if min_age and age < min_age:
                continue
            if max_age and age > max_age:
                continue
            
            resources.append(lifecycle)
        
        return resources
    
    async def cleanup_terminated_resources(self, retention_period: timedelta) -> int:
        """Clean up terminated resources older than retention period"""
        cleaned = 0
        now = datetime.now()
        
        resources_to_remove = []
        for resource_id, lifecycle in self._resources.items():
            if lifecycle.current_state == LifecycleState.TERMINATED:
                age = now - lifecycle.updated_at
                if age > retention_period:
                    resources_to_remove.append(resource_id)
        
        for resource_id in resources_to_remove:
            self.unregister_resource(resource_id)
            cleaned += 1
        
        logger.info(f"Cleaned up {cleaned} terminated resources")
        return cleaned
    
    def get_lifecycle_metrics(self) -> Dict[str, Any]:
        """Get lifecycle metrics for all resources"""
        metrics = {
            'total_resources': len(self._resources),
            'by_state': {},
            'by_type': {},
            'health_monitoring': len(self._health_check_tasks),
            'average_age': None
        }
        
        # Count by state
        for lifecycle in self._resources.values():
            state = lifecycle.current_state.value
            metrics['by_state'][state] = metrics['by_state'].get(state, 0) + 1
            
            rtype = lifecycle.resource_type
            metrics['by_type'][rtype] = metrics['by_type'].get(rtype, 0) + 1
        
        # Calculate average age
        if self._resources:
            total_age = sum(
                (datetime.now() - lifecycle.created_at).total_seconds()
                for lifecycle in self._resources.values()
            )
            metrics['average_age'] = total_age / len(self._resources)
        
        return metrics