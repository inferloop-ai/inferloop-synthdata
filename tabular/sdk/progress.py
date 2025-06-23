"""
Progress tracking and callback system for long-running operations
"""

import time
import threading
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from contextlib import contextmanager


class ProgressStage(Enum):
    """Stages of synthetic data generation"""
    INITIALIZING = "initializing"
    LOADING_DATA = "loading_data"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    GENERATING = "generating"
    VALIDATING = "validating"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressInfo:
    """Information about current progress"""
    stage: ProgressStage
    current: int = 0
    total: int = 0
    percentage: float = 0.0
    message: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    eta: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'stage': self.stage.value,
            'current': self.current,
            'total': self.total,
            'percentage': self.percentage,
            'message': self.message,
            'started_at': self.started_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'eta': str(self.eta) if self.eta else None,
            'metadata': self.metadata
        }


class ProgressTracker:
    """Track progress of operations with callback support"""
    
    def __init__(self, 
                 callback: Optional[Callable[[ProgressInfo], None]] = None,
                 update_interval: float = 0.1,
                 async_mode: bool = False):
        self.callback = callback
        self.update_interval = update_interval
        self.async_mode = async_mode
        self._progress = ProgressInfo(stage=ProgressStage.INITIALIZING)
        self._lock = threading.Lock()
        self._last_update = 0
        self._stage_times: Dict[ProgressStage, float] = {}
        self._cancelled = False
        
    def set_stage(self, stage: ProgressStage, message: str = ""):
        """Set the current stage"""
        with self._lock:
            # Record stage timing
            if self._progress.stage in self._stage_times:
                self._stage_times[self._progress.stage] = time.time() - self._stage_times[self._progress.stage]
            
            self._progress.stage = stage
            self._progress.message = message
            self._progress.current = 0
            self._progress.total = 0
            self._progress.percentage = 0.0
            self._progress.updated_at = datetime.now()
            
            # Start timing new stage
            self._stage_times[stage] = time.time()
            
        self._notify_callback()
    
    def update(self, current: int, total: int, message: str = "", 
              force: bool = False, **metadata):
        """Update progress within current stage"""
        with self._lock:
            self._progress.current = current
            self._progress.total = total
            self._progress.percentage = (current / total * 100) if total > 0 else 0
            if message:
                self._progress.message = message
            self._progress.updated_at = datetime.now()
            self._progress.metadata.update(metadata)
            
            # Calculate ETA
            if current > 0 and total > 0:
                elapsed = (datetime.now() - self._progress.started_at).total_seconds()
                rate = current / elapsed if elapsed > 0 else 0
                remaining = total - current
                eta_seconds = remaining / rate if rate > 0 else 0
                self._progress.eta = timedelta(seconds=int(eta_seconds))
        
        # Throttle updates unless forced
        current_time = time.time()
        if force or (current_time - self._last_update) >= self.update_interval:
            self._last_update = current_time
            self._notify_callback()
    
    def increment(self, amount: int = 1, message: str = "", **metadata):
        """Increment current progress"""
        with self._lock:
            current = self._progress.current + amount
            total = self._progress.total
        self.update(current, total, message, **metadata)
    
    def set_total(self, total: int):
        """Set total items for current stage"""
        with self._lock:
            self._progress.total = total
    
    def complete(self, message: str = "Operation completed"):
        """Mark operation as completed"""
        self.set_stage(ProgressStage.COMPLETED, message)
    
    def fail(self, error: str = "Operation failed"):
        """Mark operation as failed"""
        self.set_stage(ProgressStage.FAILED, error)
    
    def cancel(self):
        """Cancel the operation"""
        with self._lock:
            self._cancelled = True
        self.set_stage(ProgressStage.CANCELLED, "Operation cancelled")
    
    def is_cancelled(self) -> bool:
        """Check if operation was cancelled"""
        with self._lock:
            return self._cancelled
    
    def get_progress(self) -> ProgressInfo:
        """Get current progress information"""
        with self._lock:
            return ProgressInfo(
                stage=self._progress.stage,
                current=self._progress.current,
                total=self._progress.total,
                percentage=self._progress.percentage,
                message=self._progress.message,
                started_at=self._progress.started_at,
                updated_at=self._progress.updated_at,
                eta=self._progress.eta,
                metadata=dict(self._progress.metadata)
            )
    
    def get_stage_times(self) -> Dict[str, float]:
        """Get time spent in each stage"""
        with self._lock:
            times = {}
            for stage, start_time in self._stage_times.items():
                if stage == self._progress.stage:
                    # Current stage still running
                    times[stage.value] = time.time() - start_time
                else:
                    times[stage.value] = start_time
            return times
    
    def _notify_callback(self):
        """Notify callback of progress update"""
        if self.callback:
            progress = self.get_progress()
            if self.async_mode and asyncio.iscoroutinefunction(self.callback):
                # Handle async callback
                loop = asyncio.get_event_loop()
                loop.create_task(self.callback(progress))
            else:
                # Handle sync callback
                try:
                    self.callback(progress)
                except Exception as e:
                    # Don't let callback errors affect operation
                    pass
    
    @contextmanager
    def stage(self, stage: ProgressStage, message: str = ""):
        """Context manager for stage tracking"""
        self.set_stage(stage, message)
        try:
            yield self
        except Exception as e:
            self.fail(f"Error in {stage.value}: {str(e)}")
            raise
        finally:
            if self._progress.stage == stage and stage != ProgressStage.COMPLETED:
                # Auto-progress to next logical stage if not completed
                pass


class MultiProgressTracker:
    """Track progress of multiple parallel operations"""
    
    def __init__(self, callback: Optional[Callable[[str, ProgressInfo], None]] = None):
        self.callback = callback
        self.trackers: Dict[str, ProgressTracker] = {}
        self._lock = threading.Lock()
    
    def create_tracker(self, operation_id: str) -> ProgressTracker:
        """Create a new progress tracker"""
        def tracker_callback(progress: ProgressInfo):
            if self.callback:
                self.callback(operation_id, progress)
        
        tracker = ProgressTracker(callback=tracker_callback)
        
        with self._lock:
            self.trackers[operation_id] = tracker
        
        return tracker
    
    def get_tracker(self, operation_id: str) -> Optional[ProgressTracker]:
        """Get tracker for operation"""
        with self._lock:
            return self.trackers.get(operation_id)
    
    def remove_tracker(self, operation_id: str):
        """Remove completed tracker"""
        with self._lock:
            if operation_id in self.trackers:
                del self.trackers[operation_id]
    
    def get_all_progress(self) -> Dict[str, ProgressInfo]:
        """Get progress for all operations"""
        with self._lock:
            return {
                op_id: tracker.get_progress() 
                for op_id, tracker in self.trackers.items()
            }
    
    def cancel_all(self):
        """Cancel all operations"""
        with self._lock:
            for tracker in self.trackers.values():
                tracker.cancel()


# Progress callback types
ProgressCallback = Callable[[ProgressInfo], None]
AsyncProgressCallback = Callable[[ProgressInfo], asyncio.Task]
MultiProgressCallback = Callable[[str, ProgressInfo], None]


def console_progress_callback(progress: ProgressInfo):
    """Simple console progress callback"""
    bar_length = 40
    filled = int(bar_length * progress.percentage / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f"\r{progress.stage.value}: [{bar}] {progress.percentage:.1f}% - {progress.message}", 
          end='', flush=True)
    
    if progress.stage in [ProgressStage.COMPLETED, ProgressStage.FAILED, ProgressStage.CANCELLED]:
        print()  # New line at end


def create_rich_progress_callback():
    """Create a rich progress callback using rich library"""
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
    from rich.console import Console
    
    console = Console()
    progress_bars = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        def callback(info: ProgressInfo):
            stage_name = info.stage.value
            
            if stage_name not in progress_bars:
                progress_bars[stage_name] = progress.add_task(
                    f"{stage_name}: {info.message}",
                    total=info.total or 100
                )
            
            task_id = progress_bars[stage_name]
            progress.update(
                task_id,
                completed=info.current,
                total=info.total,
                description=f"{stage_name}: {info.message}"
            )
        
        return callback


class ProgressMixin:
    """Mixin to add progress tracking to generators"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_tracker: Optional[ProgressTracker] = None
        self._progress_callback: Optional[ProgressCallback] = None
    
    def set_progress_callback(self, callback: ProgressCallback):
        """Set progress callback"""
        self._progress_callback = callback
        if not self._progress_tracker:
            self._progress_tracker = ProgressTracker(callback=callback)
        else:
            self._progress_tracker.callback = callback
    
    def _get_progress_tracker(self) -> ProgressTracker:
        """Get or create progress tracker"""
        if not self._progress_tracker:
            self._progress_tracker = ProgressTracker(callback=self._progress_callback)
        return self._progress_tracker
    
    def _update_progress(self, stage: ProgressStage, current: int = 0, 
                        total: int = 0, message: str = "", **metadata):
        """Update progress if tracker is set"""
        if self._progress_tracker:
            if stage != self._progress_tracker._progress.stage:
                self._progress_tracker.set_stage(stage, message)
            else:
                self._progress_tracker.update(current, total, message, **metadata)
    
    def _check_cancelled(self):
        """Check if operation was cancelled"""
        if self._progress_tracker and self._progress_tracker.is_cancelled():
            raise RuntimeError("Operation cancelled by user")


# Decorators for progress tracking
def with_progress(stage: ProgressStage, message: str = ""):
    """Decorator to track progress of a method"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if hasattr(self, '_progress_tracker') and self._progress_tracker:
                with self._progress_tracker.stage(stage, message):
                    return func(self, *args, **kwargs)
            else:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator


def track_progress(total_items: Optional[int] = None):
    """Decorator to track progress of iterations"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if hasattr(self, '_progress_tracker') and self._progress_tracker:
                tracker = self._progress_tracker
                
                # Set total if provided
                if total_items:
                    tracker.set_total(total_items)
                
                # Wrap the generator/iterator
                result = func(self, *args, **kwargs)
                
                if hasattr(result, '__iter__'):
                    def progress_iterator():
                        for i, item in enumerate(result):
                            tracker.update(i + 1, tracker._progress.total or total_items or 0)
                            yield item
                            # Check for cancellation
                            if tracker.is_cancelled():
                                break
                    return progress_iterator()
                else:
                    return result
            else:
                return func(self, *args, **kwargs)
        return wrapper
    return decorator