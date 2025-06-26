"""
Audit Logging Implementation for TextNLP
Comprehensive audit logging for security, compliance, and monitoring
"""

import json
import logging
import asyncio
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timezone
import uuid
import os
from pathlib import Path
import sqlite3
import aiofiles
import aiofiles.os
from contextlib import asynccontextmanager
import threading
from queue import Queue, Empty
import time

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of events that can be audited"""
    # Content Generation Events
    GENERATION_REQUEST = "generation_request"
    GENERATION_COMPLETED = "generation_completed"
    GENERATION_FAILED = "generation_failed"
    
    # Safety and Compliance Events
    PII_DETECTED = "pii_detected"
    TOXICITY_DETECTED = "toxicity_detected"
    BIAS_DETECTED = "bias_detected"
    COMPLIANCE_VIOLATION = "compliance_violation"
    CONTENT_FILTERED = "content_filtered"
    
    # Security Events
    AUTHENTICATION_SUCCESS = "authentication_success"
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHORIZATION_DENIED = "authorization_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    
    # System Events
    SERVICE_STARTED = "service_started"
    SERVICE_STOPPED = "service_stopped"
    CONFIGURATION_CHANGED = "configuration_changed"
    ERROR_OCCURRED = "error_occurred"
    
    # Data Events
    DATA_CREATED = "data_created"
    DATA_MODIFIED = "data_modified"
    DATA_DELETED = "data_deleted"
    DATA_EXPORTED = "data_exported"
    
    # Model Events
    MODEL_LOADED = "model_loaded"
    MODEL_UNLOADED = "model_unloaded"
    MODEL_UPDATED = "model_updated"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditStatus(Enum):
    """Status of audit events"""
    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    CANCELLED = "cancelled"


@dataclass
class AuditEvent:
    """Individual audit event"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: AuditEventType = AuditEventType.GENERATION_REQUEST
    severity: AuditSeverity = AuditSeverity.INFO
    status: AuditStatus = AuditStatus.SUCCESS
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    service_name: str = "textnlp"
    component: str = "unknown"
    action: str = ""
    resource: str = ""
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['event_type'] = self.event_type.value
        result['severity'] = self.severity.value
        result['status'] = self.status.value
        return result
    
    def to_json(self) -> str:
        """Convert audit event to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(',', ':'))


@dataclass
class AuditConfiguration:
    """Configuration for audit logging"""
    enabled: bool = True
    log_level: str = "INFO"
    storage_backend: str = "file"  # "file", "database", "elasticsearch", "cloudwatch"
    storage_config: Dict[str, Any] = field(default_factory=dict)
    retention_days: int = 90
    max_file_size_mb: int = 100
    max_files: int = 10
    enable_encryption: bool = True
    encryption_key: Optional[str] = None
    async_logging: bool = True
    buffer_size: int = 1000
    flush_interval: int = 30  # seconds
    include_sensitive_data: bool = False
    exclude_events: List[str] = field(default_factory=list)
    include_events: List[str] = field(default_factory=list)


class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, config: AuditConfiguration):
        self.config = config
        self.is_running = False
        self.event_buffer = Queue(maxsize=config.buffer_size)
        self.worker_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Initialize storage backend
        self._initialize_storage()
        
        # Start background worker if async logging is enabled
        if config.async_logging:
            self._start_background_worker()
    
    def _initialize_storage(self):
        """Initialize the storage backend"""
        backend = self.config.storage_backend.lower()
        
        if backend == "file":
            self._initialize_file_storage()
        elif backend == "database":
            self._initialize_database_storage()
        elif backend == "elasticsearch":
            self._initialize_elasticsearch_storage()
        elif backend == "cloudwatch":
            self._initialize_cloudwatch_storage()
        else:
            raise ValueError(f"Unsupported storage backend: {backend}")
    
    def _initialize_file_storage(self):
        """Initialize file-based storage"""
        storage_config = self.config.storage_config
        self.log_directory = Path(storage_config.get("directory", "logs/audit"))
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Current log file
        self.current_log_file = self.log_directory / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.current_file_size = 0
        
        if self.current_log_file.exists():
            self.current_file_size = self.current_log_file.stat().st_size
    
    def _initialize_database_storage(self):
        """Initialize database storage"""
        storage_config = self.config.storage_config
        db_path = storage_config.get("path", "audit.db")
        
        self.db_path = db_path
        self._create_audit_table()
    
    def _create_audit_table(self):
        """Create audit table in SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    request_id TEXT,
                    service_name TEXT NOT NULL,
                    component TEXT NOT NULL,
                    action TEXT,
                    resource TEXT,
                    source_ip TEXT,
                    user_agent TEXT,
                    message TEXT,
                    details TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity)")
    
    def _initialize_elasticsearch_storage(self):
        """Initialize Elasticsearch storage"""
        # Placeholder for Elasticsearch integration
        storage_config = self.config.storage_config
        self.es_host = storage_config.get("host", "localhost:9200")
        self.es_index = storage_config.get("index", "textnlp-audit")
        logger.info(f"Elasticsearch storage configured: {self.es_host}")
    
    def _initialize_cloudwatch_storage(self):
        """Initialize AWS CloudWatch storage"""
        # Placeholder for CloudWatch integration
        storage_config = self.config.storage_config
        self.log_group = storage_config.get("log_group", "/textnlp/audit")
        self.log_stream = storage_config.get("log_stream", "default")
        logger.info(f"CloudWatch storage configured: {self.log_group}")
    
    def _start_background_worker(self):
        """Start background worker thread for async logging"""
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.worker_thread.start()
        logger.info("Audit logging background worker started")
    
    def _background_worker(self):
        """Background worker for processing audit events"""
        last_flush = time.time()
        
        while self.is_running:
            try:
                # Process events from buffer
                events_to_process = []
                
                # Collect events with timeout
                try:
                    # Get first event with timeout
                    event = self.event_buffer.get(timeout=1.0)
                    events_to_process.append(event)
                    
                    # Get additional events without blocking
                    while not self.event_buffer.empty() and len(events_to_process) < 100:
                        try:
                            event = self.event_buffer.get_nowait()
                            events_to_process.append(event)
                        except Empty:
                            break
                    
                except Empty:
                    pass
                
                # Process collected events
                if events_to_process:
                    self._write_events_batch(events_to_process)
                
                # Periodic flush
                current_time = time.time()
                if current_time - last_flush >= self.config.flush_interval:
                    self._flush_storage()
                    last_flush = current_time
                    
            except Exception as e:
                logger.error(f"Error in audit logging worker: {e}")
    
    def _write_events_batch(self, events: List[AuditEvent]):
        """Write a batch of events to storage"""
        backend = self.config.storage_backend.lower()
        
        try:
            if backend == "file":
                self._write_to_file_batch(events)
            elif backend == "database":
                self._write_to_database_batch(events)
            elif backend == "elasticsearch":
                self._write_to_elasticsearch_batch(events)
            elif backend == "cloudwatch":
                self._write_to_cloudwatch_batch(events)
        except Exception as e:
            logger.error(f"Failed to write audit events batch: {e}")
    
    def _write_to_file_batch(self, events: List[AuditEvent]):
        """Write events to file storage"""
        # Check if we need to rotate log file
        self._check_file_rotation()
        
        with open(self.current_log_file, "a", encoding="utf-8") as f:
            for event in events:
                json_line = event.to_json()
                if self.config.enable_encryption and self.config.encryption_key:
                    json_line = self._encrypt_data(json_line)
                
                f.write(json_line + "\n")
                self.current_file_size += len(json_line.encode("utf-8")) + 1
    
    def _write_to_database_batch(self, events: List[AuditEvent]):
        """Write events to database storage"""
        with sqlite3.connect(self.db_path) as conn:
            for event in events:
                conn.execute("""
                    INSERT INTO audit_events (
                        event_id, timestamp, event_type, severity, status,
                        user_id, session_id, request_id, service_name,
                        component, action, resource, source_ip, user_agent,
                        message, details, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.event_type.value,
                    event.severity.value,
                    event.status.value,
                    event.user_id,
                    event.session_id,
                    event.request_id,
                    event.service_name,
                    event.component,
                    event.action,
                    event.resource,
                    event.source_ip,
                    event.user_agent,
                    event.message,
                    json.dumps(event.details),
                    json.dumps(event.metadata)
                ))
    
    def _write_to_elasticsearch_batch(self, events: List[AuditEvent]):
        """Write events to Elasticsearch"""
        # Placeholder implementation
        logger.info(f"Would write {len(events)} events to Elasticsearch")
    
    def _write_to_cloudwatch_batch(self, events: List[AuditEvent]):
        """Write events to CloudWatch"""
        # Placeholder implementation
        logger.info(f"Would write {len(events)} events to CloudWatch")
    
    def _check_file_rotation(self):
        """Check if log file needs rotation"""
        max_size_bytes = self.config.max_file_size_mb * 1024 * 1024
        
        if self.current_file_size >= max_size_bytes:
            # Rotate to new file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.current_log_file = self.log_directory / f"audit_{timestamp}.jsonl"
            self.current_file_size = 0
            
            # Clean up old files
            self._cleanup_old_files()
    
    def _cleanup_old_files(self):
        """Clean up old audit log files"""
        if self.config.max_files <= 0:
            return
        
        # Get all audit log files
        audit_files = sorted(
            self.log_directory.glob("audit_*.jsonl"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        
        # Remove excess files
        if len(audit_files) > self.config.max_files:
            for old_file in audit_files[self.config.max_files:]:
                try:
                    old_file.unlink()
                    logger.info(f"Removed old audit log: {old_file}")
                except Exception as e:
                    logger.error(f"Failed to remove old audit log {old_file}: {e}")
    
    def _encrypt_data(self, data: str) -> str:
        """Encrypt sensitive audit data"""
        # Placeholder for encryption implementation
        # In practice, you would use a proper encryption library
        return data
    
    def _flush_storage(self):
        """Flush storage backend"""
        # For file storage, this would sync to disk
        # For database storage, this would commit transactions
        pass
    
    def _should_log_event(self, event: AuditEvent) -> bool:
        """Check if event should be logged based on configuration"""
        if not self.config.enabled:
            return False
        
        event_type_str = event.event_type.value
        
        # Check exclude list
        if event_type_str in self.config.exclude_events:
            return False
        
        # Check include list (if specified)
        if self.config.include_events and event_type_str not in self.config.include_events:
            return False
        
        return True
    
    def log_event(self, event: AuditEvent):
        """Log an audit event"""
        if not self._should_log_event(event):
            return
        
        # Sanitize sensitive data if needed
        if not self.config.include_sensitive_data:
            event = self._sanitize_event(event)
        
        if self.config.async_logging:
            try:
                self.event_buffer.put_nowait(event)
            except Exception as e:
                logger.error(f"Failed to queue audit event: {e}")
                # Fall back to synchronous logging
                self._write_events_batch([event])
        else:
            self._write_events_batch([event])
    
    def _sanitize_event(self, event: AuditEvent) -> AuditEvent:
        """Remove sensitive data from event"""
        # Create a copy of the event
        sanitized = AuditEvent(
            event_id=event.event_id,
            timestamp=event.timestamp,
            event_type=event.event_type,
            severity=event.severity,
            status=event.status,
            user_id=event.user_id,
            session_id=event.session_id,
            request_id=event.request_id,
            service_name=event.service_name,
            component=event.component,
            action=event.action,
            resource=event.resource,
            source_ip=event.source_ip,
            user_agent=event.user_agent,
            message=event.message,
            details=dict(event.details),
            metadata=dict(event.metadata)
        )
        
        # Remove sensitive fields
        sensitive_keys = ["password", "token", "key", "secret", "credit_card", "ssn"]
        
        def sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            sanitized_dict = {}
            for key, value in d.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    sanitized_dict[key] = "[REDACTED]"
                elif isinstance(value, dict):
                    sanitized_dict[key] = sanitize_dict(value)
                elif isinstance(value, str) and len(value) > 100:
                    # Truncate very long strings
                    sanitized_dict[key] = value[:100] + "..."
                else:
                    sanitized_dict[key] = value
            return sanitized_dict
        
        sanitized.details = sanitize_dict(sanitized.details)
        sanitized.metadata = sanitize_dict(sanitized.metadata)
        
        return sanitized
    
    async def query_events(self, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          event_types: Optional[List[AuditEventType]] = None,
                          user_id: Optional[str] = None,
                          severity: Optional[AuditSeverity] = None,
                          limit: int = 1000) -> List[AuditEvent]:
        """Query audit events from storage"""
        backend = self.config.storage_backend.lower()
        
        if backend == "file":
            return await self._query_file_storage(start_time, end_time, event_types, user_id, severity, limit)
        elif backend == "database":
            return await self._query_database_storage(start_time, end_time, event_types, user_id, severity, limit)
        else:
            raise NotImplementedError(f"Query not implemented for {backend}")
    
    async def _query_file_storage(self, start_time, end_time, event_types, user_id, severity, limit) -> List[AuditEvent]:
        """Query events from file storage"""
        events = []
        
        # Get relevant log files
        log_files = sorted(self.log_directory.glob("audit_*.jsonl"), reverse=True)
        
        for log_file in log_files:
            if len(events) >= limit:
                break
            
            try:
                async with aiofiles.open(log_file, 'r', encoding='utf-8') as f:
                    async for line in f:
                        if len(events) >= limit:
                            break
                        
                        try:
                            event_data = json.loads(line.strip())
                            
                            # Apply filters
                            if start_time and datetime.fromisoformat(event_data['timestamp']) < start_time:
                                continue
                            if end_time and datetime.fromisoformat(event_data['timestamp']) > end_time:
                                continue
                            if event_types and event_data['event_type'] not in [et.value for et in event_types]:
                                continue
                            if user_id and event_data.get('user_id') != user_id:
                                continue
                            if severity and event_data['severity'] != severity.value:
                                continue
                            
                            # Convert back to AuditEvent
                            event = AuditEvent(
                                event_id=event_data['event_id'],
                                timestamp=datetime.fromisoformat(event_data['timestamp']),
                                event_type=AuditEventType(event_data['event_type']),
                                severity=AuditSeverity(event_data['severity']),
                                status=AuditStatus(event_data['status']),
                                user_id=event_data.get('user_id'),
                                session_id=event_data.get('session_id'),
                                request_id=event_data.get('request_id'),
                                service_name=event_data['service_name'],
                                component=event_data['component'],
                                action=event_data.get('action', ''),
                                resource=event_data.get('resource', ''),
                                source_ip=event_data.get('source_ip'),
                                user_agent=event_data.get('user_agent'),
                                message=event_data.get('message', ''),
                                details=event_data.get('details', {}),
                                metadata=event_data.get('metadata', {})
                            )
                            events.append(event)
                            
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            logger.warning(f"Failed to parse audit event: {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"Failed to read audit log file {log_file}: {e}")
        
        return events
    
    async def _query_database_storage(self, start_time, end_time, event_types, user_id, severity, limit) -> List[AuditEvent]:
        """Query events from database storage"""
        # Build query
        where_conditions = []
        params = []
        
        if start_time:
            where_conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())
        
        if end_time:
            where_conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())
        
        if event_types:
            placeholders = ",".join("?" * len(event_types))
            where_conditions.append(f"event_type IN ({placeholders})")
            params.extend([et.value for et in event_types])
        
        if user_id:
            where_conditions.append("user_id = ?")
            params.append(user_id)
        
        if severity:
            where_conditions.append("severity = ?")
            params.append(severity.value)
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        query = f"""
            SELECT * FROM audit_events 
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """
        params.append(limit)
        
        events = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            for row in cursor:
                event = AuditEvent(
                    event_id=row['event_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    event_type=AuditEventType(row['event_type']),
                    severity=AuditSeverity(row['severity']),
                    status=AuditStatus(row['status']),
                    user_id=row['user_id'],
                    session_id=row['session_id'],
                    request_id=row['request_id'],
                    service_name=row['service_name'],
                    component=row['component'],
                    action=row['action'],
                    resource=row['resource'],
                    source_ip=row['source_ip'],
                    user_agent=row['user_agent'],
                    message=row['message'],
                    details=json.loads(row['details']) if row['details'] else {},
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                events.append(event)
        
        return events
    
    def create_audit_report(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Create audit report from events"""
        if not events:
            return {"summary": {"total_events": 0}}
        
        # Count events by type
        event_type_counts = {}
        for event in events:
            event_type_counts[event.event_type.value] = event_type_counts.get(event.event_type.value, 0) + 1
        
        # Count events by severity
        severity_counts = {}
        for event in events:
            severity_counts[event.severity.value] = severity_counts.get(event.severity.value, 0) + 1
        
        # Count events by status
        status_counts = {}
        for event in events:
            status_counts[event.status.value] = status_counts.get(event.status.value, 0) + 1
        
        # Find time range
        timestamps = [event.timestamp for event in events]
        start_time = min(timestamps)
        end_time = max(timestamps)
        
        # Count unique users
        unique_users = set(event.user_id for event in events if event.user_id)
        
        return {
            "summary": {
                "total_events": len(events),
                "unique_users": len(unique_users),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                }
            },
            "event_type_distribution": event_type_counts,
            "severity_distribution": severity_counts,
            "status_distribution": status_counts,
            "top_users": list(unique_users)[:10],
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
    
    def stop(self):
        """Stop the audit logger"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("Audit logger stopped")


class AuditLoggerFactory:
    """Factory for creating audit loggers"""
    
    @staticmethod
    def create_logger(config: Dict[str, Any]) -> AuditLogger:
        """Create audit logger from configuration"""
        audit_config = AuditConfiguration(
            enabled=config.get("enabled", True),
            log_level=config.get("log_level", "INFO"),
            storage_backend=config.get("storage_backend", "file"),
            storage_config=config.get("storage_config", {}),
            retention_days=config.get("retention_days", 90),
            max_file_size_mb=config.get("max_file_size_mb", 100),
            max_files=config.get("max_files", 10),
            enable_encryption=config.get("enable_encryption", True),
            encryption_key=config.get("encryption_key"),
            async_logging=config.get("async_logging", True),
            buffer_size=config.get("buffer_size", 1000),
            flush_interval=config.get("flush_interval", 30),
            include_sensitive_data=config.get("include_sensitive_data", False),
            exclude_events=config.get("exclude_events", []),
            include_events=config.get("include_events", [])
        )
        
        return AuditLogger(audit_config)


# Convenience functions for common audit events
def log_generation_request(audit_logger: AuditLogger, user_id: str, prompt: str, 
                         model: str, request_id: str, session_id: str = None):
    """Log a generation request"""
    event = AuditEvent(
        event_type=AuditEventType.GENERATION_REQUEST,
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
        component="generator",
        action="generate",
        resource=model,
        message=f"Generation request for model {model}",
        details={
            "model": model,
            "prompt_length": len(prompt),
            "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()
        }
    )
    audit_logger.log_event(event)


def log_safety_violation(audit_logger: AuditLogger, violation_type: str, 
                        confidence: float, user_id: str = None, request_id: str = None):
    """Log a safety violation"""
    event = AuditEvent(
        event_type=AuditEventType.CONTENT_FILTERED,
        severity=AuditSeverity.WARNING if confidence < 0.8 else AuditSeverity.ERROR,
        user_id=user_id,
        request_id=request_id,
        component="safety",
        action="filter",
        message=f"Content filtered due to {violation_type}",
        details={
            "violation_type": violation_type,
            "confidence": confidence
        }
    )
    audit_logger.log_event(event)


def log_authentication_event(audit_logger: AuditLogger, user_id: str, success: bool, 
                            source_ip: str = None, user_agent: str = None):
    """Log an authentication event"""
    event = AuditEvent(
        event_type=AuditEventType.AUTHENTICATION_SUCCESS if success else AuditEventType.AUTHENTICATION_FAILED,
        severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
        status=AuditStatus.SUCCESS if success else AuditStatus.FAILURE,
        user_id=user_id,
        source_ip=source_ip,
        user_agent=user_agent,
        component="auth",
        action="authenticate",
        message=f"Authentication {'successful' if success else 'failed'} for user {user_id}"
    )
    audit_logger.log_event(event)


# Example usage
if __name__ == "__main__":
    async def example():
        # Configure audit logger
        config = {
            "enabled": True,
            "storage_backend": "file",
            "storage_config": {"directory": "logs/audit"},
            "async_logging": True,
            "include_sensitive_data": False
        }
        
        # Create audit logger
        audit_logger = AuditLoggerFactory.create_logger(config)
        
        # Log some example events
        log_generation_request(
            audit_logger, 
            user_id="user123", 
            prompt="Write a story about...", 
            model="gpt2",
            request_id="req456"
        )
        
        log_safety_violation(
            audit_logger,
            violation_type="toxicity",
            confidence=0.85,
            user_id="user123",
            request_id="req456"
        )
        
        log_authentication_event(
            audit_logger,
            user_id="user123",
            success=True,
            source_ip="192.168.1.1"
        )
        
        # Wait a moment for async processing
        await asyncio.sleep(2)
        
        # Query events
        events = await audit_logger.query_events(limit=10)
        print(f"Found {len(events)} audit events")
        
        # Create report
        report = audit_logger.create_audit_report(events)
        print(f"Audit report: {json.dumps(report, indent=2)}")
        
        # Stop logger
        audit_logger.stop()
    
    # Run example
    # asyncio.run(example())