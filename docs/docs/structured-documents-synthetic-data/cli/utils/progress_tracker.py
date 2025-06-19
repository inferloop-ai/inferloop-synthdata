#!/usr/bin/env python3
"""
Progress tracker for CLI operations.

Provides progress tracking and reporting for long-running CLI operations
with support for multiple progress bars, time estimation, and rate calculation.
"""

import sys
import time
from typing import Optional, TextIO
from threading import RLock


class ProgressTracker:
    """Thread-safe progress tracker with rate calculation and ETA"""
    
    def __init__(self, total: int, desc: str = "", unit: str = "items", 
                 file: Optional[TextIO] = None, disable: bool = False,
                 bar_width: int = 40, update_interval: float = 0.1):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.file = file or sys.stderr
        self.disable = disable
        self.bar_width = bar_width
        self.update_interval = update_interval
        
        # Progress state
        self.n = 0  # Current progress
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_print_time = self.start_time
        
        # Thread safety
        self._lock = RLock()
        
        # Display state
        self._closed = False
        self._last_line_length = 0
        
        # Rate calculation
        self._rate_samples = []
        self._max_samples = 10
    
    def update(self, n: int = 1):
        """Update progress by n items"""
        if self.disable or self._closed:
            return
        
        with self._lock:
            self.n = min(self.n + n, self.total)
            current_time = time.time()
            
            # Update rate samples
            if len(self._rate_samples) == 0:
                self._rate_samples.append((current_time, self.n))
            else:
                # Add new sample
                self._rate_samples.append((current_time, self.n))
                
                # Keep only recent samples
                if len(self._rate_samples) > self._max_samples:
                    self._rate_samples.pop(0)
                
                # Remove old samples (older than 5 seconds)
                self._rate_samples = [
                    (t, count) for t, count in self._rate_samples 
                    if current_time - t <= 5.0
                ]
            
            # Update display if enough time has passed
            if (current_time - self.last_print_time >= self.update_interval or 
                self.n >= self.total):
                self._refresh()
                self.last_print_time = current_time
            
            self.last_update_time = current_time
    
    def set_progress(self, n: int):
        """Set absolute progress"""
        if self.disable or self._closed:
            return
        
        with self._lock:
            old_n = self.n
            self.n = min(max(n, 0), self.total)
            
            # Update rate calculation
            if self.n != old_n:
                current_time = time.time()
                self._rate_samples.append((current_time, self.n))
                
                if len(self._rate_samples) > self._max_samples:
                    self._rate_samples.pop(0)
            
            self._refresh()
    
    def close(self):
        """Close the progress tracker"""
        if self.disable or self._closed:
            return
        
        with self._lock:
            self._closed = True
            self._refresh(final=True)
            if self.file.isatty():
                self.file.write('\n')
                self.file.flush()
    
    def _refresh(self, final: bool = False):
        """Refresh the progress display"""
        if self.disable:
            return
        
        # Calculate progress percentage
        percent = (self.n / self.total) if self.total > 0 else 0
        
        # Calculate rate and ETA
        rate = self._calculate_rate()
        eta = self._calculate_eta(rate)
        
        # Build progress bar
        bar = self._build_progress_bar(percent)
        
        # Build status text
        status_parts = []
        
        if self.desc:
            status_parts.append(self.desc)
        
        status_parts.append(f"{percent:.1%}")
        status_parts.append(f"{self.n}/{self.total} {self.unit}")
        
        if rate > 0:
            if rate >= 1:
                status_parts.append(f"{rate:.1f} {self.unit}/s")
            else:
                status_parts.append(f"{1/rate:.1f}s/{self.unit}")
        
        if eta is not None and not final:
            status_parts.append(f"ETA: {self._format_time(eta)}")
        
        if final:
            elapsed = time.time() - self.start_time
            status_parts.append(f"Total: {self._format_time(elapsed)}")
        
        # Combine bar and status
        status_text = " | ".join(status_parts)
        progress_line = f"{bar} {status_text}"
        
        # Clear previous line and print new one
        if self.file.isatty() and not final:
            # Clear previous line
            if self._last_line_length > 0:
                self.file.write('\r' + ' ' * self._last_line_length + '\r')
            
            self.file.write(progress_line)
            self.file.flush()
            self._last_line_length = len(progress_line)
        else:
            # Non-TTY or final update
            if self._last_line_length > 0:
                self.file.write('\r')
            self.file.write(progress_line)
            if final:
                self.file.write('\n')
            self.file.flush()
    
    def _build_progress_bar(self, percent: float) -> str:
        """Build visual progress bar"""
        filled_width = int(self.bar_width * percent)
        remaining_width = self.bar_width - filled_width
        
        # Unicode block characters for smooth progress
        filled_char = 'ˆ'  # Full block
        empty_char = '‘'   # Light shade
        
        bar = filled_char * filled_width + empty_char * remaining_width
        return f"[{bar}]"
    
    def _calculate_rate(self) -> float:
        """Calculate current processing rate"""
        if len(self._rate_samples) < 2:
            return 0.0
        
        # Use linear regression for smooth rate calculation
        times = [t for t, _ in self._rate_samples]
        counts = [c for _, c in self._rate_samples]
        
        if len(times) < 2:
            return 0.0
        
        # Simple rate calculation using first and last samples
        time_diff = times[-1] - times[0]
        count_diff = counts[-1] - counts[0]
        
        if time_diff <= 0:
            return 0.0
        
        return count_diff / time_diff
    
    def _calculate_eta(self, rate: float) -> Optional[float]:
        """Calculate estimated time to completion"""
        if rate <= 0 or self.n >= self.total:
            return None
        
        remaining = self.total - self.n
        return remaining / rate
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration for display"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    @property
    def percentage(self) -> float:
        """Get current percentage complete"""
        return (self.n / self.total) if self.total > 0 else 0
    
    @property
    def rate(self) -> float:
        """Get current processing rate"""
        return self._calculate_rate()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time since start"""
        return time.time() - self.start_time


class MultiProgressTracker:
    """Manage multiple progress trackers"""
    
    def __init__(self, file: Optional[TextIO] = None):
        self.file = file or sys.stderr
        self.trackers = {}
        self._lock = RLock()
        self._closed = False
    
    def add_tracker(self, name: str, total: int, desc: str = "", 
                   unit: str = "items") -> ProgressTracker:
        """Add a new progress tracker"""
        with self._lock:
            if self._closed:
                raise RuntimeError("MultiProgressTracker is closed")
            
            tracker = ProgressTracker(
                total=total,
                desc=desc or name,
                unit=unit,
                file=self.file,
                disable=False
            )
            
            self.trackers[name] = tracker
            return tracker
    
    def update(self, name: str, n: int = 1):
        """Update specific tracker"""
        with self._lock:
            if name in self.trackers:
                self.trackers[name].update(n)
    
    def set_progress(self, name: str, n: int):
        """Set absolute progress for specific tracker"""
        with self._lock:
            if name in self.trackers:
                self.trackers[name].set_progress(n)
    
    def close_tracker(self, name: str):
        """Close specific tracker"""
        with self._lock:
            if name in self.trackers:
                self.trackers[name].close()
                del self.trackers[name]
    
    def close_all(self):
        """Close all trackers"""
        with self._lock:
            for tracker in self.trackers.values():
                tracker.close()
            self.trackers.clear()
            self._closed = True
    
    def get_tracker(self, name: str) -> Optional[ProgressTracker]:
        """Get tracker by name"""
        return self.trackers.get(name)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close_all()


def create_progress_tracker(total: int, desc: str = "", unit: str = "items", 
                           disable: bool = False) -> ProgressTracker:
    """Factory function to create progress tracker"""
    return ProgressTracker(
        total=total,
        desc=desc,
        unit=unit,
        disable=disable
    )


def create_multi_progress_tracker(file: Optional[TextIO] = None) -> MultiProgressTracker:
    """Factory function to create multi progress tracker"""
    return MultiProgressTracker(file=file)


__all__ = [
    'ProgressTracker',
    'MultiProgressTracker',
    'create_progress_tracker',
    'create_multi_progress_tracker'
]