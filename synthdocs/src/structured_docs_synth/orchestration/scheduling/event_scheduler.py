#!/usr/bin/env python3
"""
Event scheduler for event-driven task scheduling.

Provides event-based scheduling capabilities with support for
custom events, webhooks, and reactive programming patterns.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import re

from pydantic import BaseModel, Field

from ...core import get_logger, BaseConfig

logger = get_logger(__name__)


class EventType(str, Enum):
    """Event types"""
    CUSTOM = "custom"
    TIMER = "timer"
    FILE_CHANGE = "file_change"
    WEBHOOK = "webhook"
    SYSTEM = "system"
    JOB_COMPLETION = "job_completion"
    METRIC_THRESHOLD = "metric_threshold"


class TriggerCondition(str, Enum):
    """Trigger condition types"""
    IMMEDIATE = "immediate"
    DEBOUNCED = "debounced"
    THROTTLED = "throttled"
    BATCH = "batch"
    CONDITIONAL = "conditional"


@dataclass
class Event:
    """Event data structure"""
    id: str
    event_type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class EventHandler:
    """Event handler configuration"""
    id: str
    name: str
    event_pattern: str  # Pattern to match events
    handler_function: Callable
    condition: TriggerCondition = TriggerCondition.IMMEDIATE
    condition_config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0
    max_retries: int = 3
    timeout_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class HandlerExecution:
    """Handler execution record"""
    execution_id: str
    handler_id: str
    event_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


class EventSchedulerConfig(BaseConfig):
    """Event scheduler configuration"""
    max_concurrent_handlers: int = 10
    event_buffer_size: int = 1000
    handler_timeout_seconds: int = 300
    
    # Event retention
    event_retention_hours: int = 24
    execution_retention_days: int = 7
    
    # Debouncing/throttling
    default_debounce_seconds: int = 5
    default_throttle_seconds: int = 60
    default_batch_size: int = 10
    default_batch_timeout_seconds: int = 30
    
    # Webhook server
    enable_webhook_server: bool = False
    webhook_host: str = "localhost"
    webhook_port: int = 8080
    webhook_path: str = "/webhook"
    
    # File watching
    enable_file_watching: bool = False
    watched_directories: List[str] = field(default_factory=list)


class EventScheduler:
    """Comprehensive event-driven scheduling system"""
    
    def __init__(self, config: EventSchedulerConfig):
        self.config = config
        self.handlers: Dict[str, EventHandler] = {}
        self.events: List[Event] = []
        self.executions: Dict[str, HandlerExecution] = {}
        self.execution_history: List[HandlerExecution] = []
        self.running = False
        self.running_handlers: Dict[str, asyncio.Task] = {}
        
        # Event processing
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=config.event_buffer_size)
        
        # Debouncing/throttling state
        self.debounce_timers: Dict[str, asyncio.Task] = {}
        self.throttle_last_run: Dict[str, datetime] = {}
        self.batch_buffers: Dict[str, List[Event]] = {}
        self.batch_timers: Dict[str, asyncio.Task] = {}
        
        # File watching
        self.file_watchers: List[asyncio.Task] = []
        
        # Webhook server
        self.webhook_server = None
    
    async def start(self):
        """Start event scheduler"""
        self.running = True
        logger.info("Event scheduler started")
        
        # Start event processing workers
        for i in range(self.config.max_concurrent_handlers):
            asyncio.create_task(self._event_processor(f"processor-{i}"))
        
        # Start management tasks
        asyncio.create_task(self._cleanup_worker())
        
        # Start webhook server if enabled
        if self.config.enable_webhook_server:
            await self._start_webhook_server()
        
        # Start file watchers if enabled
        if self.config.enable_file_watching:
            await self._start_file_watchers()
    
    async def stop(self):
        """Stop event scheduler"""
        self.running = False
        
        # Cancel all running handlers
        for task in self.running_handlers.values():
            task.cancel()
        
        # Cancel debounce/batch timers
        for timer in self.debounce_timers.values():
            timer.cancel()
        for timer in self.batch_timers.values():
            timer.cancel()
        
        # Stop file watchers
        for watcher in self.file_watchers:
            watcher.cancel()
        
        # Stop webhook server
        if self.webhook_server:
            await self.webhook_server.stop()
        
        logger.info("Event scheduler stopped")
    
    def register_handler(self, handler: EventHandler) -> bool:
        """
        Register event handler.
        
        Args:
            handler: Event handler to register
        
        Returns:
            True if handler was registered
        """
        try:
            # Validate event pattern (basic regex validation)
            re.compile(handler.event_pattern)
            
            self.handlers[handler.id] = handler
            logger.info(f"Registered event handler: {handler.id} - {handler.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register handler {handler.id}: {e}")
            return False
    
    def unregister_handler(self, handler_id: str) -> bool:
        """
        Unregister event handler.
        
        Args:
            handler_id: ID of handler to unregister
        
        Returns:
            True if handler was unregistered
        """
        if handler_id in self.handlers:
            # Cancel any running executions
            for execution_id, task in list(self.running_handlers.items()):
                if self.executions.get(execution_id, {}).get('handler_id') == handler_id:
                    task.cancel()
                    del self.running_handlers[execution_id]
            
            del self.handlers[handler_id]
            logger.info(f"Unregistered event handler: {handler_id}")
            return True
        
        return False
    
    async def emit_event(self, event: Event):
        """
        Emit event for processing.
        
        Args:
            event: Event to emit
        """
        try:
            # Add to event list for history
            self.events.append(event)
            
            # Add to processing queue
            await self.event_queue.put(event)
            
            logger.debug(f"Emitted event: {event.id} - {event.event_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to emit event {event.id}: {e}")
    
    async def emit_custom_event(self, source: str, data: Dict[str, Any], 
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Emit custom event.
        
        Args:
            source: Event source
            data: Event data
            metadata: Event metadata
        
        Returns:
            Event ID
        """
        event = Event(
            id=str(uuid.uuid4()),
            event_type=EventType.CUSTOM,
            source=source,
            data=data,
            metadata=metadata or {}
        )
        
        await self.emit_event(event)
        return event.id
    
    async def schedule_timer_event(self, delay_seconds: int, source: str, 
                                 data: Dict[str, Any]) -> str:
        """
        Schedule timer-based event.
        
        Args:
            delay_seconds: Delay before event fires
            source: Event source
            data: Event data
        
        Returns:
            Event ID
        """
        async def timer_task():
            await asyncio.sleep(delay_seconds)
            event = Event(
                id=str(uuid.uuid4()),
                event_type=EventType.TIMER,
                source=source,
                data=data
            )
            await self.emit_event(event)
        
        asyncio.create_task(timer_task())
        return f"timer_{uuid.uuid4()}"
    
    def get_handler_status(self, handler_id: str) -> Optional[Dict[str, Any]]:
        """
        Get handler status.
        
        Args:
            handler_id: Handler ID
        
        Returns:
            Handler status dictionary
        """
        if handler_id not in self.handlers:
            return None
        
        handler = self.handlers[handler_id]
        
        # Get execution statistics
        handler_executions = [
            ex for ex in self.execution_history 
            if ex.handler_id == handler_id
        ]
        
        total_executions = len(handler_executions)
        successful_executions = len([ex for ex in handler_executions if ex.status == "completed"])
        failed_executions = len([ex for ex in handler_executions if ex.status == "failed"])
        
        # Calculate average execution time
        durations = []
        for ex in handler_executions:
            if ex.end_time:
                duration = (ex.end_time - ex.start_time).total_seconds()
                durations.append(duration)
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            'handler_id': handler_id,
            'name': handler.name,
            'enabled': handler.enabled,
            'event_pattern': handler.event_pattern,
            'condition': handler.condition.value,
            'priority': handler.priority,
            'is_running': any(ex.handler_id == handler_id for ex in self.executions.values() if ex.status == "running"),
            'statistics': {
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'failed_executions': failed_executions,
                'success_rate': (successful_executions / total_executions * 100) if total_executions > 0 else 0,
                'average_duration_seconds': avg_duration
            }
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status.
        
        Returns:
            System status dictionary
        """
        return {
            'running': self.running,
            'total_handlers': len(self.handlers),
            'enabled_handlers': len([h for h in self.handlers.values() if h.enabled]),
            'running_executions': len(self.running_handlers),
            'events_processed': len(self.events),
            'queue_size': self.event_queue.qsize(),
            'total_executions': len(self.execution_history)
        }
    
    async def _event_processor(self, processor_id: str):
        """Event processing worker"""
        logger.info(f"Event processor {processor_id} started")
        
        while self.running:
            try:
                # Get event from queue
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Find matching handlers
                matching_handlers = self._find_matching_handlers(event)
                
                # Process handlers
                for handler in matching_handlers:
                    if not handler.enabled:
                        continue
                    
                    await self._process_handler(handler, event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in event processor {processor_id}: {e}")
        
        logger.info(f"Event processor {processor_id} stopped")
    
    def _find_matching_handlers(self, event: Event) -> List[EventHandler]:
        """Find handlers that match event"""
        matching_handlers = []
        
        for handler in self.handlers.values():
            if self._event_matches_pattern(event, handler.event_pattern):
                matching_handlers.append(handler)
        
        # Sort by priority
        matching_handlers.sort(key=lambda h: h.priority, reverse=True)
        
        return matching_handlers
    
    def _event_matches_pattern(self, event: Event, pattern: str) -> bool:
        """Check if event matches handler pattern"""
        try:
            # Simple pattern matching - could be enhanced with more complex rules
            event_str = f"{event.event_type.value}:{event.source}"
            return re.match(pattern, event_str) is not None
        except Exception as e:
            logger.error(f"Error matching event pattern: {e}")
            return False
    
    async def _process_handler(self, handler: EventHandler, event: Event):
        """Process handler for event"""
        try:
            if handler.condition == TriggerCondition.IMMEDIATE:
                await self._execute_handler_immediate(handler, event)
            elif handler.condition == TriggerCondition.DEBOUNCED:
                await self._execute_handler_debounced(handler, event)
            elif handler.condition == TriggerCondition.THROTTLED:
                await self._execute_handler_throttled(handler, event)
            elif handler.condition == TriggerCondition.BATCH:
                await self._execute_handler_batch(handler, event)
            elif handler.condition == TriggerCondition.CONDITIONAL:
                await self._execute_handler_conditional(handler, event)
        
        except Exception as e:
            logger.error(f"Error processing handler {handler.id}: {e}")
    
    async def _execute_handler_immediate(self, handler: EventHandler, event: Event):
        """Execute handler immediately"""
        await self._execute_handler(handler, [event])
    
    async def _execute_handler_debounced(self, handler: EventHandler, event: Event):
        """Execute handler with debouncing"""
        debounce_key = f"{handler.id}:{event.source}"
        debounce_delay = handler.condition_config.get(
            'debounce_seconds', self.config.default_debounce_seconds
        )
        
        # Cancel existing timer
        if debounce_key in self.debounce_timers:
            self.debounce_timers[debounce_key].cancel()
        
        # Start new timer
        async def debounced_execution():
            await asyncio.sleep(debounce_delay)
            await self._execute_handler(handler, [event])
            if debounce_key in self.debounce_timers:
                del self.debounce_timers[debounce_key]
        
        self.debounce_timers[debounce_key] = asyncio.create_task(debounced_execution())
    
    async def _execute_handler_throttled(self, handler: EventHandler, event: Event):
        """Execute handler with throttling"""
        throttle_key = f"{handler.id}:{event.source}"
        throttle_interval = handler.condition_config.get(
            'throttle_seconds', self.config.default_throttle_seconds
        )
        
        now = datetime.now()
        last_run = self.throttle_last_run.get(throttle_key)
        
        if last_run is None or (now - last_run).total_seconds() >= throttle_interval:
            self.throttle_last_run[throttle_key] = now
            await self._execute_handler(handler, [event])
    
    async def _execute_handler_batch(self, handler: EventHandler, event: Event):
        """Execute handler with batching"""
        batch_key = f"{handler.id}:{event.source}"
        batch_size = handler.condition_config.get('batch_size', self.config.default_batch_size)
        batch_timeout = handler.condition_config.get(
            'batch_timeout_seconds', self.config.default_batch_timeout_seconds
        )
        
        # Add event to batch buffer
        if batch_key not in self.batch_buffers:
            self.batch_buffers[batch_key] = []
        
        self.batch_buffers[batch_key].append(event)
        
        # Check if batch is full
        if len(self.batch_buffers[batch_key]) >= batch_size:
            events = self.batch_buffers[batch_key]
            del self.batch_buffers[batch_key]
            
            # Cancel timeout timer
            if batch_key in self.batch_timers:
                self.batch_timers[batch_key].cancel()
                del self.batch_timers[batch_key]
            
            await self._execute_handler(handler, events)
        else:
            # Start timeout timer if not already running
            if batch_key not in self.batch_timers:
                async def batch_timeout_handler():
                    await asyncio.sleep(batch_timeout)
                    if batch_key in self.batch_buffers:
                        events = self.batch_buffers[batch_key]
                        del self.batch_buffers[batch_key]
                        await self._execute_handler(handler, events)
                    if batch_key in self.batch_timers:
                        del self.batch_timers[batch_key]
                
                self.batch_timers[batch_key] = asyncio.create_task(batch_timeout_handler())
    
    async def _execute_handler_conditional(self, handler: EventHandler, event: Event):
        """Execute handler with conditions"""
        # This would implement custom condition logic
        # For now, just execute immediately
        await self._execute_handler(handler, [event])
    
    async def _execute_handler(self, handler: EventHandler, events: List[Event]):
        """Execute handler with events"""
        execution_id = str(uuid.uuid4())
        
        execution = HandlerExecution(
            execution_id=execution_id,
            handler_id=handler.id,
            event_id=events[0].id if events else "batch",
            status="running",
            start_time=datetime.now()
        )
        
        self.executions[execution_id] = execution
        
        try:
            logger.info(f"Executing handler {handler.id} for {len(events)} events")
            
            # Execute handler function
            timeout = handler.timeout_seconds or self.config.handler_timeout_seconds
            
            if asyncio.iscoroutinefunction(handler.handler_function):
                result = await asyncio.wait_for(
                    handler.handler_function(events),
                    timeout=timeout
                )
            else:
                # Run sync function in thread pool
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: handler.handler_function(events)
                    ),
                    timeout=timeout
                )
            
            execution.status = "completed"
            execution.result = result
            
            logger.info(f"Handler {handler.id} completed successfully")
            
        except asyncio.TimeoutError:
            execution.status = "failed"
            execution.error = f"Handler timed out after {timeout} seconds"
            logger.error(f"Handler {handler.id} timed out")
            
        except Exception as e:
            execution.status = "failed"
            execution.error = str(e)
            logger.error(f"Handler {handler.id} failed: {e}")
        
        finally:
            execution.end_time = datetime.now()
            
            # Move to history
            self.execution_history.append(execution)
            del self.executions[execution_id]
    
    async def _start_webhook_server(self):
        """Start webhook server"""
        try:
            from aiohttp import web
            
            async def webhook_handler(request):
                try:
                    data = await request.json()
                    
                    event = Event(
                        id=str(uuid.uuid4()),
                        event_type=EventType.WEBHOOK,
                        source=request.remote,
                        data=data,
                        metadata={
                            'headers': dict(request.headers),
                            'method': request.method,
                            'path': request.path
                        }
                    )
                    
                    await self.emit_event(event)
                    
                    return web.json_response({'status': 'ok', 'event_id': event.id})
                    
                except Exception as e:
                    logger.error(f"Webhook handler error: {e}")
                    return web.json_response({'error': str(e)}, status=400)
            
            app = web.Application()
            app.router.add_post(self.config.webhook_path, webhook_handler)
            
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(runner, self.config.webhook_host, self.config.webhook_port)
            await site.start()
            
            self.webhook_server = runner
            
            logger.info(f"Webhook server started on {self.config.webhook_host}:{self.config.webhook_port}")
            
        except Exception as e:
            logger.error(f"Failed to start webhook server: {e}")
    
    async def _start_file_watchers(self):
        """Start file watchers"""
        try:
            # This would use a file watching library like watchdog
            # For now, it's a placeholder
            logger.info("File watchers started")
        except Exception as e:
            logger.error(f"Failed to start file watchers: {e}")
    
    async def _cleanup_worker(self):
        """Background worker for cleanup"""
        while self.running:
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # Hourly cleanup
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old events and executions"""
        try:
            # Clean old events
            event_cutoff = datetime.now() - timedelta(hours=self.config.event_retention_hours)
            original_event_count = len(self.events)
            self.events = [e for e in self.events if e.timestamp >= event_cutoff]
            
            # Clean old executions
            execution_cutoff = datetime.now() - timedelta(days=self.config.execution_retention_days)
            original_execution_count = len(self.execution_history)
            self.execution_history = [
                ex for ex in self.execution_history 
                if ex.start_time >= execution_cutoff
            ]
            
            cleaned_events = original_event_count - len(self.events)
            cleaned_executions = original_execution_count - len(self.execution_history)
            
            if cleaned_events > 0 or cleaned_executions > 0:
                logger.info(f"Cleaned up {cleaned_events} events and {cleaned_executions} executions")
        
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")


def create_event_scheduler(config: Optional[EventSchedulerConfig] = None) -> EventScheduler:
    """Factory function to create event scheduler"""
    if config is None:
        config = EventSchedulerConfig()
    return EventScheduler(config)


__all__ = [
    'EventScheduler',
    'EventSchedulerConfig',
    'Event',
    'EventHandler',
    'HandlerExecution',
    'EventType',
    'TriggerCondition',
    'create_event_scheduler'
]