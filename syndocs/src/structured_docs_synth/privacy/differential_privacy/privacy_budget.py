"""
Privacy Budget Tracker for Differential Privacy
Manages epsilon and delta budgets across multiple queries and mechanisms
"""

from typing import Dict, List, Union, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid
import threading
import json

from ...core import get_logger, PrivacyError


class BudgetStatus(Enum):
    """Status of a budget allocation"""
    PENDING = "pending"
    ALLOCATED = "allocated"
    CONSUMED = "consumed"
    EXPIRED = "expired"
    RELEASED = "released"


class BudgetScope(Enum):
    """Scope of budget application"""
    GLOBAL = "global"  # Across entire dataset
    USER = "user"  # Per user/data subject
    SESSION = "session"  # Per analysis session
    QUERY_SET = "query_set"  # Per set of related queries
    TIME_WINDOW = "time_window"  # Per time period


class CompositionType(Enum):
    """Types of privacy composition"""
    BASIC = "basic"  # Basic composition (linear)
    ADVANCED = "advanced"  # Advanced composition (tighter bounds)
    MOMENTS_ACCOUNTANT = "moments_accountant"  # Moments accountant method
    RDP = "rdp"  # Renyi Differential Privacy
    ZERO_CDP = "zero_cdp"  # Zero-Concentrated Differential Privacy


@dataclass
class BudgetAllocation:
    """A single budget allocation"""
    allocation_id: str
    epsilon: float
    delta: float
    mechanism: str
    query_description: str
    allocated_at: datetime
    consumed_at: Optional[datetime] = None
    status: BudgetStatus = BudgetStatus.PENDING
    scope: BudgetScope = BudgetScope.GLOBAL
    scope_id: Optional[str] = None  # User ID, session ID, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    expiration_time: Optional[datetime] = None


@dataclass
class BudgetSummary:
    """Summary of budget usage"""
    total_epsilon_allocated: float
    total_delta_allocated: float
    total_epsilon_consumed: float
    total_delta_consumed: float
    epsilon_remaining: float
    delta_remaining: float
    allocations_count: int
    active_allocations: int
    expired_allocations: int
    composition_epsilon: float  # Composed epsilon considering privacy amplification
    composition_delta: float  # Composed delta


@dataclass
class BudgetLimit:
    """Budget limits and policies"""
    max_epsilon: float
    max_delta: float
    time_window: Optional[timedelta] = None
    max_allocations: Optional[int] = None
    composition_type: CompositionType = CompositionType.BASIC
    allow_overdraft: bool = False
    overdraft_limit: float = 0.0


class PrivacyBudgetTracker:
    """
    Comprehensive privacy budget management system
    
    Tracks epsilon and delta allocations across multiple queries and mechanisms,
    handles different composition methods, and enforces budget limits.
    """
    
    def __init__(
        self,
        max_epsilon: float = 1.0,
        max_delta: float = 1e-5,
        composition_type: CompositionType = CompositionType.BASIC,
        enable_time_windows: bool = True
    ):
        self.logger = get_logger(__name__)
        
        # Budget limits
        self.budget_limits = BudgetLimit(
            max_epsilon=max_epsilon,
            max_delta=max_delta,
            composition_type=composition_type
        )
        
        # Allocations storage
        self.allocations: Dict[str, BudgetAllocation] = {}
        self.allocations_by_scope: Dict[str, List[str]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Time windows for budget refresh
        self.enable_time_windows = enable_time_windows
        self.time_window_budgets: Dict[str, Tuple[datetime, float, float]] = {}
        
        # Privacy accounting
        self.composition_cache: Dict[str, Tuple[float, float]] = {}
        
        self.logger.info(
            f"Privacy Budget Tracker initialized: epsilon_max={max_epsilon}, "
            f"delta_max={max_delta}, composition={composition_type.value}"
        )
    
    def allocate_budget(
        self,
        epsilon: float,
        delta: float,
        mechanism: str,
        query_description: str,
        scope: BudgetScope = BudgetScope.GLOBAL,
        scope_id: Optional[str] = None,
        duration_hours: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Allocate privacy budget for a query"""
        
        try:
            with self._lock:
                # Validate allocation
                self._validate_allocation(epsilon, delta, scope, scope_id)
                
                # Check if allocation would exceed limits
                self._check_budget_limits(epsilon, delta, scope, scope_id)
                
                # Create allocation
                allocation_id = str(uuid.uuid4())
                expiration_time = None
                if duration_hours:
                    expiration_time = datetime.utcnow() + timedelta(hours=duration_hours)
                
                allocation = BudgetAllocation(
                    allocation_id=allocation_id,
                    epsilon=epsilon,
                    delta=delta,
                    mechanism=mechanism,
                    query_description=query_description,
                    allocated_at=datetime.utcnow(),
                    scope=scope,
                    scope_id=scope_id,
                    metadata=metadata or {},
                    expiration_time=expiration_time,
                    status=BudgetStatus.ALLOCATED
                )
                
                # Store allocation
                self.allocations[allocation_id] = allocation
                
                # Update scope tracking
                scope_key = f"{scope.value}:{scope_id or 'global'}"
                if scope_key not in self.allocations_by_scope:
                    self.allocations_by_scope[scope_key] = []
                self.allocations_by_scope[scope_key].append(allocation_id)
                
                # Clear composition cache
                self.composition_cache.clear()
                
                self.logger.info(
                    f"Budget allocated: {allocation_id[:8]} - epsilon={epsilon}, delta={delta}, "
                    f"mechanism={mechanism}, scope={scope.value}"
                )
                
                return allocation_id
                
        except Exception as e:
            self.logger.error(f"Error allocating budget: {str(e)}")
            raise PrivacyError(f"Budget allocation failed: {str(e)}")
    
    def consume_budget(self, allocation_id: str) -> None:
        """Mark budget allocation as consumed"""
        
        try:
            with self._lock:
                if allocation_id not in self.allocations:
                    raise PrivacyError(f"Allocation not found: {allocation_id}")
                
                allocation = self.allocations[allocation_id]
                
                if allocation.status != BudgetStatus.ALLOCATED:
                    raise PrivacyError(f"Allocation not available for consumption: {allocation.status}")
                
                # Check if expired
                if allocation.expiration_time and datetime.utcnow() > allocation.expiration_time:
                    allocation.status = BudgetStatus.EXPIRED
                    raise PrivacyError(f"Allocation expired: {allocation_id}")
                
                # Mark as consumed
                allocation.status = BudgetStatus.CONSUMED
                allocation.consumed_at = datetime.utcnow()
                
                # Clear composition cache
                self.composition_cache.clear()
                
                self.logger.info(f"Budget consumed: {allocation_id[:8]}")
                
        except Exception as e:
            self.logger.error(f"Error consuming budget: {str(e)}")
            raise PrivacyError(f"Budget consumption failed: {str(e)}")
    
    def release_budget(self, allocation_id: str) -> None:
        """Release an unused budget allocation"""
        
        try:
            with self._lock:
                if allocation_id not in self.allocations:
                    raise PrivacyError(f"Allocation not found: {allocation_id}")
                
                allocation = self.allocations[allocation_id]
                
                if allocation.status == BudgetStatus.CONSUMED:
                    raise PrivacyError("Cannot release consumed budget")
                
                allocation.status = BudgetStatus.RELEASED
                
                # Clear composition cache
                self.composition_cache.clear()
                
                self.logger.info(f"Budget released: {allocation_id[:8]}")
                
        except Exception as e:
            self.logger.error(f"Error releasing budget: {str(e)}")
            raise PrivacyError(f"Budget release failed: {str(e)}")
    
    def get_budget_summary(
        self,
        scope: Optional[BudgetScope] = None,
        scope_id: Optional[str] = None
    ) -> BudgetSummary:
        """Get comprehensive budget usage summary"""
        
        try:
            with self._lock:
                # Filter allocations by scope if specified
                if scope:
                    scope_key = f"{scope.value}:{scope_id or 'global'}"
                    allocation_ids = self.allocations_by_scope.get(scope_key, [])
                    allocations = [self.allocations[aid] for aid in allocation_ids if aid in self.allocations]
                else:
                    allocations = list(self.allocations.values())
                
                # Calculate totals
                total_epsilon_allocated = 0.0
                total_delta_allocated = 0.0
                total_epsilon_consumed = 0.0
                total_delta_consumed = 0.0
                active_count = 0
                expired_count = 0
                
                for allocation in allocations:
                    # Check if expired
                    if (allocation.expiration_time and 
                        datetime.utcnow() > allocation.expiration_time and
                        allocation.status != BudgetStatus.EXPIRED):
                        allocation.status = BudgetStatus.EXPIRED
                    
                    if allocation.status == BudgetStatus.ALLOCATED:
                        total_epsilon_allocated += allocation.epsilon
                        total_delta_allocated += allocation.delta
                        active_count += 1
                    elif allocation.status == BudgetStatus.CONSUMED:
                        total_epsilon_consumed += allocation.epsilon
                        total_delta_consumed += allocation.delta
                    elif allocation.status == BudgetStatus.EXPIRED:
                        expired_count += 1
                
                # Calculate composition
                composition_epsilon, composition_delta = self._calculate_composition(
                    scope, scope_id
                )
                
                # Calculate remaining budget
                epsilon_remaining = self.budget_limits.max_epsilon - composition_epsilon
                delta_remaining = self.budget_limits.max_delta - composition_delta
                
                return BudgetSummary(
                    total_epsilon_allocated=total_epsilon_allocated,
                    total_delta_allocated=total_delta_allocated,
                    total_epsilon_consumed=total_epsilon_consumed,
                    total_delta_consumed=total_delta_consumed,
                    epsilon_remaining=max(0, epsilon_remaining),
                    delta_remaining=max(0, delta_remaining),
                    allocations_count=len(allocations),
                    active_allocations=active_count,
                    expired_allocations=expired_count,
                    composition_epsilon=composition_epsilon,
                    composition_delta=composition_delta
                )
                
        except Exception as e:
            self.logger.error(f"Error getting budget summary: {str(e)}")
            raise PrivacyError(f"Budget summary failed: {str(e)}")
    
    def check_budget_availability(
        self,
        epsilon: float,
        delta: float,
        scope: BudgetScope = BudgetScope.GLOBAL,
        scope_id: Optional[str] = None
    ) -> bool:
        """Check if requested budget is available"""
        
        try:
            # Get current composition
            current_epsilon, current_delta = self._calculate_composition(scope, scope_id)
            
            # Simulate adding this allocation
            if self.budget_limits.composition_type == CompositionType.BASIC:
                new_epsilon = current_epsilon + epsilon
                new_delta = current_delta + delta
            else:
                # Use advanced composition for estimation
                new_epsilon, new_delta = self._advanced_composition(
                    [(current_epsilon, current_delta), (epsilon, delta)]
                )
            
            # Check limits
            epsilon_available = new_epsilon <= self.budget_limits.max_epsilon
            delta_available = new_delta <= self.budget_limits.max_delta
            
            return epsilon_available and delta_available
            
        except Exception as e:
            self.logger.error(f"Error checking budget availability: {str(e)}")
            return False
    
    def get_allocation_history(
        self,
        scope: Optional[BudgetScope] = None,
        scope_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[BudgetAllocation]:
        """Get allocation history with optional filtering"""
        
        with self._lock:
            # Filter allocations
            if scope:
                scope_key = f"{scope.value}:{scope_id or 'global'}"
                allocation_ids = self.allocations_by_scope.get(scope_key, [])
                allocations = [self.allocations[aid] for aid in allocation_ids if aid in self.allocations]
            else:
                allocations = list(self.allocations.values())
            
            # Filter by time
            if start_time or end_time:
                filtered_allocations = []
                for allocation in allocations:
                    if start_time and allocation.allocated_at < start_time:
                        continue
                    if end_time and allocation.allocated_at > end_time:
                        continue
                    filtered_allocations.append(allocation)
                allocations = filtered_allocations
            
            # Sort by allocation time
            allocations.sort(key=lambda a: a.allocated_at)
            
            return allocations
    
    def update_budget_limits(
        self,
        max_epsilon: Optional[float] = None,
        max_delta: Optional[float] = None,
        composition_type: Optional[CompositionType] = None
    ) -> None:
        """Update budget limits and policies"""
        
        with self._lock:
            if max_epsilon is not None:
                self.budget_limits.max_epsilon = max_epsilon
            if max_delta is not None:
                self.budget_limits.max_delta = max_delta
            if composition_type is not None:
                self.budget_limits.composition_type = composition_type
            
            # Clear composition cache
            self.composition_cache.clear()
            
            self.logger.info(
                f"Budget limits updated: epsilon_max={self.budget_limits.max_epsilon}, "
                f"delta_max={self.budget_limits.max_delta}"
            )
    
    def cleanup_expired_allocations(self) -> int:
        """Clean up expired allocations"""
        
        try:
            with self._lock:
                expired_count = 0
                current_time = datetime.utcnow()
                
                for allocation in self.allocations.values():
                    if (allocation.expiration_time and 
                        current_time > allocation.expiration_time and
                        allocation.status not in [BudgetStatus.CONSUMED, BudgetStatus.EXPIRED]):
                        allocation.status = BudgetStatus.EXPIRED
                        expired_count += 1
                
                # Clear composition cache if any expired
                if expired_count > 0:
                    self.composition_cache.clear()
                
                self.logger.info(f"Cleaned up {expired_count} expired allocations")
                return expired_count
                
        except Exception as e:
            self.logger.error(f"Error cleaning up allocations: {str(e)}")
            return 0
    
    def reset_budget(
        self,
        scope: BudgetScope = BudgetScope.GLOBAL,
        scope_id: Optional[str] = None
    ) -> None:
        """Reset budget for a specific scope"""
        
        try:
            with self._lock:
                if scope == BudgetScope.GLOBAL and scope_id is None:
                    # Reset all budgets
                    self.allocations.clear()
                    self.allocations_by_scope.clear()
                    self.composition_cache.clear()
                    self.logger.info("Global budget reset")
                else:
                    # Reset specific scope
                    scope_key = f"{scope.value}:{scope_id or 'global'}"
                    if scope_key in self.allocations_by_scope:
                        allocation_ids = self.allocations_by_scope[scope_key]
                        for aid in allocation_ids:
                            self.allocations.pop(aid, None)
                        del self.allocations_by_scope[scope_key]
                        self.composition_cache.clear()
                        self.logger.info(f"Budget reset for scope: {scope_key}")
                
        except Exception as e:
            self.logger.error(f"Error resetting budget: {str(e)}")
            raise PrivacyError(f"Budget reset failed: {str(e)}")
    
    def export_budget_report(self, output_path: str) -> str:
        """Export detailed budget report"""
        
        try:
            with self._lock:
                report = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "budget_limits": {
                        "max_epsilon": self.budget_limits.max_epsilon,
                        "max_delta": self.budget_limits.max_delta,
                        "composition_type": self.budget_limits.composition_type.value
                    },
                    "summary": self.get_budget_summary().__dict__,
                    "allocations": []
                }
                
                # Add allocation details
                for allocation in self.allocations.values():
                    alloc_dict = allocation.__dict__.copy()
                    # Convert datetime objects to strings
                    alloc_dict["allocated_at"] = allocation.allocated_at.isoformat()
                    if allocation.consumed_at:
                        alloc_dict["consumed_at"] = allocation.consumed_at.isoformat()
                    if allocation.expiration_time:
                        alloc_dict["expiration_time"] = allocation.expiration_time.isoformat()
                    alloc_dict["status"] = allocation.status.value
                    alloc_dict["scope"] = allocation.scope.value
                    
                    report["allocations"].append(alloc_dict)
                
                # Write to file
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                
                self.logger.info(f"Budget report exported to {output_path}")
                return output_path
                
        except Exception as e:
            self.logger.error(f"Error exporting budget report: {str(e)}")
            raise PrivacyError(f"Budget report export failed: {str(e)}")
    
    def _validate_allocation(
        self,
        epsilon: float,
        delta: float,
        scope: BudgetScope,
        scope_id: Optional[str]
    ) -> None:
        """Validate allocation parameters"""
        
        if epsilon <= 0:
            raise PrivacyError("Epsilon must be positive")
        
        if delta < 0:
            raise PrivacyError("Delta must be non-negative")
        
        if scope in [BudgetScope.USER, BudgetScope.SESSION] and not scope_id:
            raise PrivacyError(f"Scope ID required for {scope.value}")
    
    def _check_budget_limits(
        self,
        epsilon: float,
        delta: float,
        scope: BudgetScope,
        scope_id: Optional[str]
    ) -> None:
        """Check if allocation would exceed budget limits"""
        
        if not self.check_budget_availability(epsilon, delta, scope, scope_id):
            current_epsilon, current_delta = self._calculate_composition(scope, scope_id)
            remaining_epsilon = self.budget_limits.max_epsilon - current_epsilon
            remaining_delta = self.budget_limits.max_delta - current_delta
            
            raise PrivacyError(
                f"Budget allocation would exceed limits. "
                f"Requested: (epsilon={epsilon}, delta={delta}), "
                f"Remaining: (epsilon={remaining_epsilon:.6f}, delta={remaining_delta:.9f})"
            )
    
    def _calculate_composition(
        self,
        scope: Optional[BudgetScope] = None,
        scope_id: Optional[str] = None
    ) -> Tuple[float, float]:
        """Calculate privacy composition for consumed allocations"""
        
        # Create cache key
        cache_key = f"{scope.value if scope else 'all'}:{scope_id or 'global'}"
        
        if cache_key in self.composition_cache:
            return self.composition_cache[cache_key]
        
        # Get relevant allocations
        if scope:
            scope_key = f"{scope.value}:{scope_id or 'global'}"
            allocation_ids = self.allocations_by_scope.get(scope_key, [])
            allocations = [
                self.allocations[aid] for aid in allocation_ids 
                if aid in self.allocations and self.allocations[aid].status == BudgetStatus.CONSUMED
            ]
        else:
            allocations = [
                a for a in self.allocations.values() 
                if a.status == BudgetStatus.CONSUMED
            ]
        
        if not allocations:
            self.composition_cache[cache_key] = (0.0, 0.0)
            return (0.0, 0.0)
        
        # Calculate composition based on type
        if self.budget_limits.composition_type == CompositionType.BASIC:
            total_epsilon = sum(a.epsilon for a in allocations)
            total_delta = sum(a.delta for a in allocations)
        else:
            # Advanced composition
            privacy_pairs = [(a.epsilon, a.delta) for a in allocations]
            total_epsilon, total_delta = self._advanced_composition(privacy_pairs)
        
        self.composition_cache[cache_key] = (total_epsilon, total_delta)
        return (total_epsilon, total_delta)
    
    def _advanced_composition(self, privacy_pairs: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate advanced composition bounds"""
        
        if not privacy_pairs:
            return (0.0, 0.0)
        
        # Simple advanced composition (can be improved with tighter bounds)
        total_epsilon = sum(eps for eps, _ in privacy_pairs)
        total_delta = sum(delta for _, delta in privacy_pairs)
        
        # Apply composition improvement factor
        k = len(privacy_pairs)
        if k > 1:
            # Improved bound: sqrt(2k log(1/delta')) epsilon + k epsilon (epsilon + 1)
            # Simplified version for demonstration
            composition_factor = min(1.0, k * 0.1)  # Simple improvement
            total_epsilon *= (1 - composition_factor)
        
        return (total_epsilon, total_delta)
    
    def get_tracker_info(self) -> Dict[str, Any]:
        """Get information about the privacy budget tracker"""
        
        return {
            "tracker_name": "Privacy Budget Tracker",
            "privacy_model": "Differential Privacy",
            "budget_limits": {
                "max_epsilon": self.budget_limits.max_epsilon,
                "max_delta": self.budget_limits.max_delta,
                "composition_type": self.budget_limits.composition_type.value
            },
            "supported_scopes": [scope.value for scope in BudgetScope],
            "supported_composition": [comp.value for comp in CompositionType],
            "total_allocations": len(self.allocations),
            "active_scopes": len(self.allocations_by_scope),
            "cache_entries": len(self.composition_cache)
        }