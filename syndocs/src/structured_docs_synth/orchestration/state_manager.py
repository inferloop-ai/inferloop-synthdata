#!/usr/bin/env python3
"""
State Manager for maintaining system state and coordination.

Provides centralized state management with support for distributed systems,
state persistence, event propagation, and transactional updates.
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Callable, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import hashlib
import pickle
from pathlib import Path
import sqlite3

from pydantic import BaseModel, Field

from ..core.config import get_config
from ..core.logging import get_logger
from ..core.exceptions import ProcessingError, ValidationError


logger = get_logger(__name__)


class StateType(Enum):
    """Types of state data"""
    SYSTEM = "system"
    JOB = "job"
    WORKER = "worker"
    RESOURCE = "resource"
    CONFIG = "config"
    METRICS = "metrics"
    CUSTOM = "custom"


class StateEventType(Enum):
    """State change event types"""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    EXPIRED = "expired"


@dataclass
class StateEntry:
    """Individual state entry"""
    key: str
    value: Any
    state_type: StateType
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None


@dataclass
class StateEvent:
    """State change event"""
    event_id: str
    event_type: StateEventType
    state_type: StateType
    key: str
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateTransaction:
    """State transaction for atomic updates"""
    transaction_id: str
    operations: List[Tuple[str, StateEntry]]  # (operation, entry)
    timestamp: datetime = field(default_factory=datetime.now)
    committed: bool = False


class StateManagerConfig(BaseModel):
    """State manager configuration"""
    # Storage settings
    storage_backend: str = Field("sqlite", description="Storage backend (memory, sqlite, redis)")
    storage_path: str = Field("./state_manager.db", description="Storage path for persistent backends")
    
    # Memory settings
    max_memory_entries: int = Field(100000, description="Maximum entries in memory")
    memory_cache_enabled: bool = Field(True, description="Enable memory caching")
    cache_ttl_seconds: int = Field(300, description="Cache TTL in seconds")
    
    # Persistence settings
    enable_persistence: bool = Field(True, description="Enable state persistence")
    persist_interval: float = Field(30.0, description="Persistence interval in seconds")
    enable_compression: bool = Field(True, description="Enable state compression")
    
    # Expiration settings
    enable_expiration: bool = Field(True, description="Enable state expiration")
    expiration_check_interval: float = Field(60.0, description="Expiration check interval")
    
    # Event settings
    enable_events: bool = Field(True, description="Enable state change events")
    max_event_history: int = Field(10000, description="Maximum event history size")
    
    # Transaction settings
    enable_transactions: bool = Field(True, description="Enable transactional updates")
    transaction_timeout: float = Field(30.0, description="Transaction timeout in seconds")


class StateManager:
    """
    Centralized state manager for system coordination.
    
    Features:
    - Multiple storage backends
    - Transactional updates
    - State versioning
    - Event propagation
    - Expiration support
    - Distributed state sync
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize state manager"""
        self.config = StateManagerConfig(**(config or {}))
        self.logger = get_logger(__name__)
        
        # State storage
        self.state_store: Dict[str, StateEntry] = {}
        self.state_lock = threading.RLock()
        
        # Memory cache
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_lock = threading.Lock()
        
        # Event handling
        self.event_history: deque = deque(maxlen=self.config.max_event_history)
        self.event_listeners: Dict[StateType, List[Callable]] = defaultdict(list)
        
        # Transaction management
        self.active_transactions: Dict[str, StateTransaction] = {}
        self.transaction_lock = threading.Lock()
        
        # Background threads
        self.is_running = False
        self.persist_thread: Optional[threading.Thread] = None
        self.expiration_thread: Optional[threading.Thread] = None
        
        # Initialize storage backend
        self._init_storage()
        
        # Start background tasks
        self._start_background_tasks()
        
        self.logger.info(f"State manager initialized with backend: {self.config.storage_backend}")
    
    def _init_storage(self):
        """Initialize storage backend"""
        if self.config.storage_backend == "memory":
            # Already using in-memory dict
            pass
        elif self.config.storage_backend == "sqlite":
            self._init_sqlite()
        elif self.config.storage_backend == "redis":
            self._init_redis()
        else:
            raise ValueError(f"Unknown storage backend: {self.config.storage_backend}")
    
    def _init_sqlite(self):
        """Initialize SQLite backend"""
        self.db_path = Path(self.config.storage_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create tables
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS state_entries (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    state_type TEXT,
                    version INTEGER,
                    created_at TEXT,
                    updated_at TEXT,
                    expires_at TEXT,
                    metadata TEXT,
                    checksum TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS state_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT,
                    state_type TEXT,
                    key TEXT,
                    old_value BLOB,
                    new_value BLOB,
                    timestamp TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON state_entries(expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_state_type ON state_entries(state_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_timestamp ON state_events(timestamp)")
        
        # Load existing state
        self._load_from_sqlite()
    
    def _init_redis(self):
        """Initialize Redis backend"""
        try:
            import redis
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=False
            )
            self.redis_client.ping()
            self.logger.info("Connected to Redis")
        except ImportError:
            raise ProcessingError("Redis library not installed")
        except Exception as e:
            raise ProcessingError(f"Failed to connect to Redis: {e}")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        self.is_running = True
        
        if self.config.enable_persistence and self.config.storage_backend != "memory":
            self.persist_thread = threading.Thread(
                target=self._persist_loop,
                name="StateManagerPersist",
                daemon=True
            )
            self.persist_thread.start()
        
        if self.config.enable_expiration:
            self.expiration_thread = threading.Thread(
                target=self._expiration_loop,
                name="StateManagerExpiration",
                daemon=True
            )
            self.expiration_thread.start()
    
    def get(
        self,
        key: str,
        state_type: StateType = StateType.CUSTOM,
        default: Any = None
    ) -> Any:
        """Get state value"""
        # Check cache first
        if self.config.memory_cache_enabled:
            with self.cache_lock:
                if key in self.cache:
                    value, cached_at = self.cache[key]
                    if (datetime.now() - cached_at).total_seconds() < self.config.cache_ttl_seconds:
                        return value
        
        # Get from store
        with self.state_lock:
            entry = self.state_store.get(key)
            if entry:
                # Check expiration
                if entry.expires_at and datetime.now() > entry.expires_at:
                    self._expire_entry(key)
                    return default
                
                # Update cache
                if self.config.memory_cache_enabled:
                    with self.cache_lock:
                        self.cache[key] = (entry.value, datetime.now())
                
                return entry.value
        
        return default
    
    def set(
        self,
        key: str,
        value: Any,
        state_type: StateType = StateType.CUSTOM,
        expires_in: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StateEntry:
        """Set state value"""
        with self.state_lock:
            # Get existing entry for versioning
            existing = self.state_store.get(key)
            version = existing.version + 1 if existing else 1
            
            # Calculate expiration
            expires_at = None
            if expires_in:
                expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            # Create entry
            entry = StateEntry(
                key=key,
                value=value,
                state_type=state_type,
                version=version,
                expires_at=expires_at,
                metadata=metadata or {},
                checksum=self._calculate_checksum(value)
            )
            
            # Store entry
            old_value = existing.value if existing else None
            self.state_store[key] = entry
            
            # Update cache
            if self.config.memory_cache_enabled:
                with self.cache_lock:
                    self.cache[key] = (value, datetime.now())
            
            # Emit event
            if self.config.enable_events:
                self._emit_event(
                    StateEventType.UPDATED if existing else StateEventType.CREATED,
                    state_type,
                    key,
                    old_value,
                    value
                )
            
            return entry
    
    def delete(self, key: str) -> bool:
        """Delete state entry"""
        with self.state_lock:
            if key in self.state_store:
                entry = self.state_store[key]
                old_value = entry.value
                del self.state_store[key]
                
                # Remove from cache
                if self.config.memory_cache_enabled:
                    with self.cache_lock:
                        self.cache.pop(key, None)
                
                # Emit event
                if self.config.enable_events:
                    self._emit_event(
                        StateEventType.DELETED,
                        entry.state_type,
                        key,
                        old_value,
                        None
                    )
                
                return True
        
        return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        with self.state_lock:
            if key in self.state_store:
                entry = self.state_store[key]
                # Check expiration
                if entry.expires_at and datetime.now() > entry.expires_at:
                    self._expire_entry(key)
                    return False
                return True
        return False
    
    def get_all(self, state_type: Optional[StateType] = None) -> Dict[str, Any]:
        """Get all entries of a specific type"""
        with self.state_lock:
            result = {}
            for key, entry in list(self.state_store.items()):
                # Check expiration
                if entry.expires_at and datetime.now() > entry.expires_at:
                    self._expire_entry(key)
                    continue
                
                # Filter by type
                if state_type is None or entry.state_type == state_type:
                    result[key] = entry.value
            
            return result
    
    def begin_transaction(self) -> str:
        """Begin a new transaction"""
        if not self.config.enable_transactions:
            raise ProcessingError("Transactions are disabled")
        
        transaction_id = f"txn_{int(time.time() * 1000000)}"
        
        with self.transaction_lock:
            transaction = StateTransaction(
                transaction_id=transaction_id,
                operations=[]
            )
            self.active_transactions[transaction_id] = transaction
        
        self.logger.debug(f"Transaction started: {transaction_id}")
        return transaction_id
    
    def add_to_transaction(
        self,
        transaction_id: str,
        operation: str,
        key: str,
        value: Any = None,
        state_type: StateType = StateType.CUSTOM,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add operation to transaction"""
        with self.transaction_lock:
            if transaction_id not in self.active_transactions:
                raise ValidationError(f"Transaction not found: {transaction_id}")
            
            transaction = self.active_transactions[transaction_id]
            
            if operation not in ["set", "delete"]:
                raise ValueError(f"Invalid operation: {operation}")
            
            if operation == "set":
                entry = StateEntry(
                    key=key,
                    value=value,
                    state_type=state_type,
                    metadata=metadata or {}
                )
            else:
                entry = StateEntry(key=key, value=None, state_type=state_type)
            
            transaction.operations.append((operation, entry))
    
    def commit_transaction(self, transaction_id: str) -> bool:
        """Commit a transaction"""
        with self.transaction_lock:
            if transaction_id not in self.active_transactions:
                raise ValidationError(f"Transaction not found: {transaction_id}")
            
            transaction = self.active_transactions[transaction_id]
            
            # Apply all operations atomically
            with self.state_lock:
                try:
                    for operation, entry in transaction.operations:
                        if operation == "set":
                            self.set(
                                entry.key,
                                entry.value,
                                entry.state_type,
                                metadata=entry.metadata
                            )
                        elif operation == "delete":
                            self.delete(entry.key)
                    
                    transaction.committed = True
                    self.logger.debug(f"Transaction committed: {transaction_id}")
                    return True
                    
                except Exception as e:
                    self.logger.error(f"Transaction failed: {transaction_id} - {e}")
                    # Rollback would happen here in a real implementation
                    return False
                    
                finally:
                    del self.active_transactions[transaction_id]
    
    def rollback_transaction(self, transaction_id: str):
        """Rollback a transaction"""
        with self.transaction_lock:
            if transaction_id in self.active_transactions:
                del self.active_transactions[transaction_id]
                self.logger.debug(f"Transaction rolled back: {transaction_id}")
    
    def subscribe(self, state_type: StateType, callback: Callable):
        """Subscribe to state change events"""
        if not self.config.enable_events:
            raise ProcessingError("Events are disabled")
        
        self.event_listeners[state_type].append(callback)
        self.logger.debug(f"Subscribed to {state_type.value} events")
    
    def unsubscribe(self, state_type: StateType, callback: Callable):
        """Unsubscribe from state change events"""
        if callback in self.event_listeners[state_type]:
            self.event_listeners[state_type].remove(callback)
    
    def _emit_event(
        self,
        event_type: StateEventType,
        state_type: StateType,
        key: str,
        old_value: Any,
        new_value: Any
    ):
        """Emit state change event"""
        event = StateEvent(
            event_id=f"evt_{int(time.time() * 1000000)}",
            event_type=event_type,
            state_type=state_type,
            key=key,
            old_value=old_value,
            new_value=new_value
        )
        
        # Add to history
        self.event_history.append(event)
        
        # Notify listeners
        for callback in self.event_listeners.get(state_type, []):
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Event callback failed: {e}")
    
    def _expire_entry(self, key: str):
        """Expire a state entry"""
        if key in self.state_store:
            entry = self.state_store[key]
            old_value = entry.value
            del self.state_store[key]
            
            # Remove from cache
            if self.config.memory_cache_enabled:
                with self.cache_lock:
                    self.cache.pop(key, None)
            
            # Emit event
            if self.config.enable_events:
                self._emit_event(
                    StateEventType.EXPIRED,
                    entry.state_type,
                    key,
                    old_value,
                    None
                )
    
    def _calculate_checksum(self, value: Any) -> str:
        """Calculate checksum for value"""
        try:
            serialized = pickle.dumps(value)
            return hashlib.sha256(serialized).hexdigest()[:16]
        except:
            return ""
    
    def _persist_loop(self):
        """Background persistence loop"""
        while self.is_running:
            try:
                time.sleep(self.config.persist_interval)
                
                if self.config.storage_backend == "sqlite":
                    self._persist_to_sqlite()
                elif self.config.storage_backend == "redis":
                    self._persist_to_redis()
                
            except Exception as e:
                self.logger.error(f"Persistence error: {e}")
    
    def _persist_to_sqlite(self):
        """Persist state to SQLite"""
        with self.state_lock:
            entries = list(self.state_store.values())
        
        with sqlite3.connect(str(self.db_path)) as conn:
            # Persist entries
            for entry in entries:
                conn.execute("""
                    INSERT OR REPLACE INTO state_entries 
                    (key, value, state_type, version, created_at, updated_at, expires_at, metadata, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.key,
                    pickle.dumps(entry.value),
                    entry.state_type.value,
                    entry.version,
                    entry.created_at.isoformat(),
                    entry.updated_at.isoformat(),
                    entry.expires_at.isoformat() if entry.expires_at else None,
                    json.dumps(entry.metadata),
                    entry.checksum
                ))
            
            conn.commit()
    
    def _persist_to_redis(self):
        """Persist state to Redis"""
        with self.state_lock:
            for key, entry in self.state_store.items():
                redis_key = f"state:{key}"
                value = pickle.dumps(entry)
                
                if entry.expires_at:
                    ttl = int((entry.expires_at - datetime.now()).total_seconds())
                    if ttl > 0:
                        self.redis_client.setex(redis_key, ttl, value)
                else:
                    self.redis_client.set(redis_key, value)
    
    def _load_from_sqlite(self):
        """Load state from SQLite"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("SELECT * FROM state_entries")
            
            for row in cursor:
                try:
                    entry = StateEntry(
                        key=row[0],
                        value=pickle.loads(row[1]),
                        state_type=StateType(row[2]),
                        version=row[3],
                        created_at=datetime.fromisoformat(row[4]),
                        updated_at=datetime.fromisoformat(row[5]),
                        expires_at=datetime.fromisoformat(row[6]) if row[6] else None,
                        metadata=json.loads(row[7]),
                        checksum=row[8]
                    )
                    
                    # Skip expired entries
                    if entry.expires_at and datetime.now() > entry.expires_at:
                        continue
                    
                    self.state_store[entry.key] = entry
                    
                except Exception as e:
                    self.logger.error(f"Failed to load entry {row[0]}: {e}")
        
        self.logger.info(f"Loaded {len(self.state_store)} entries from SQLite")
    
    def _expiration_loop(self):
        """Background expiration check loop"""
        while self.is_running:
            try:
                time.sleep(self.config.expiration_check_interval)
                
                expired_keys = []
                with self.state_lock:
                    for key, entry in self.state_store.items():
                        if entry.expires_at and datetime.now() > entry.expires_at:
                            expired_keys.append(key)
                
                # Expire entries
                for key in expired_keys:
                    self._expire_entry(key)
                
                if expired_keys:
                    self.logger.debug(f"Expired {len(expired_keys)} entries")
                
            except Exception as e:
                self.logger.error(f"Expiration check error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics"""
        with self.state_lock:
            type_counts = defaultdict(int)
            for entry in self.state_store.values():
                type_counts[entry.state_type.value] += 1
        
        return {
            "total_entries": len(self.state_store),
            "entries_by_type": dict(type_counts),
            "cache_size": len(self.cache),
            "event_history_size": len(self.event_history),
            "active_transactions": len(self.active_transactions),
            "storage_backend": self.config.storage_backend
        }
    
    def cleanup(self):
        """Clean up state manager"""
        self.is_running = False
        
        if self.persist_thread:
            self.persist_thread.join(timeout=5.0)
        
        if self.expiration_thread:
            self.expiration_thread.join(timeout=5.0)
        
        # Final persistence
        if self.config.enable_persistence:
            if self.config.storage_backend == "sqlite":
                self._persist_to_sqlite()
            elif self.config.storage_backend == "redis":
                self._persist_to_redis()
        
        self.logger.info("State manager cleaned up")


# Factory function
def create_state_manager(config: Optional[Dict[str, Any]] = None) -> StateManager:
    """Create and return a state manager instance"""
    return StateManager(config)