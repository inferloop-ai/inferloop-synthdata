"""
Audit Logger for compliance tracking and forensic analysis
Implements comprehensive audit trails for privacy and compliance monitoring
"""

import json
import hashlib
from typing import Dict, List, Set, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import uuid
import threading
from pathlib import Path

from ...core import get_logger, ComplianceError, PrivacyError


class AuditEventType(Enum):
    """Types of audit events"""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    PRIVACY_ASSESSMENT = "privacy_assessment"
    COMPLIANCE_CHECK = "compliance_check"
    MASKING_OPERATION = "masking_operation"
    ANONYMIZATION = "anonymization"
    CONSENT_MANAGEMENT = "consent_management"
    BREACH_DETECTION = "breach_detection"
    SYSTEM_ACCESS = "system_access"
    CONFIGURATION_CHANGE = "configuration_change"
    POLICY_VIOLATION = "policy_violation"
    ESCALATION = "escalation"


class AuditSeverity(Enum):
    """Audit event severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ComplianceFramework(Enum):
    """Compliance frameworks for audit context"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    SOX = "sox"
    CCPA = "ccpa"
    PIPEDA = "pipeda"
    LGPD = "lgpd"
    GENERAL = "general"


class DataClassification(Enum):
    """Data classification levels for audit context"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class AuditStatus(Enum):
    """Audit record processing status"""
    PENDING = "pending"
    PROCESSED = "processed"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    ARCHIVED = "archived"


@dataclass
class AuditContext:
    """Context information for audit events"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    operation_id: Optional[str] = None
    compliance_framework: Optional[ComplianceFramework] = None
    data_classification: Optional[DataClassification] = None
    business_justification: Optional[str] = None
    approval_id: Optional[str] = None


@dataclass
class DataSubject:
    """Data subject information for privacy audits"""
    subject_id: str
    subject_type: str  # individual, employee, customer, etc.
    jurisdiction: Optional[str] = None
    consent_status: Optional[str] = None
    retention_period: Optional[str] = None


@dataclass
class AuditRecord:
    """Comprehensive audit record"""
    record_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    message: str
    details: Dict[str, Any]
    context: AuditContext
    data_subject: Optional[DataSubject] = None
    before_value: Optional[str] = None
    after_value: Optional[str] = None
    affected_fields: List[str] = None
    risk_level: Optional[str] = None
    compliance_impact: Optional[str] = None
    remediation_required: bool = False
    status: AuditStatus = AuditStatus.PENDING
    checksum: Optional[str] = None


@dataclass
class AuditQuery:
    """Query parameters for audit search"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    severity_levels: Optional[List[AuditSeverity]] = None
    user_ids: Optional[List[str]] = None
    compliance_frameworks: Optional[List[ComplianceFramework]] = None
    data_subjects: Optional[List[str]] = None
    search_text: Optional[str] = None
    risk_levels: Optional[List[str]] = None
    limit: int = 1000


@dataclass
class AuditSummary:
    """Summary statistics for audit analysis"""
    total_events: int
    events_by_type: Dict[str, int]
    events_by_severity: Dict[str, int]
    events_by_framework: Dict[str, int]
    high_risk_events: int
    compliance_violations: int
    data_subjects_affected: int
    unique_users: int
    time_range: str


class AuditLogger:
    """Comprehensive audit logging system for compliance and forensics"""
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        retention_days: int = 2555,  # 7 years default
        encryption_key: Optional[str] = None,
        auto_archive: bool = True
    ):
        self.logger = get_logger(__name__)
        self.storage_path = Path(storage_path) if storage_path else Path("audit_logs")
        self.retention_days = retention_days
        self.encryption_key = encryption_key
        self.auto_archive = auto_archive
        
        # In-memory storage for active audit records
        self.audit_records: List[AuditRecord] = []
        self.audit_index: Dict[str, AuditRecord] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Event handlers
        self.event_handlers: Dict[AuditEventType, List[Callable]] = {}
        
        # Compliance rules
        self.compliance_rules: Dict[ComplianceFramework, Dict[str, Any]] = {
            ComplianceFramework.GDPR: {
                "retention_days": 2555,  # 7 years max
                "required_fields": ["data_subject", "lawful_basis"],
                "breach_notification_hours": 72
            },
            ComplianceFramework.HIPAA: {
                "retention_days": 2555,  # 7 years
                "required_fields": ["covered_entity", "baa_status"],
                "audit_controls_required": True
            },
            ComplianceFramework.SOX: {
                "retention_days": 2555,  # 7 years
                "required_fields": ["financial_impact", "internal_controls"],
                "management_certification": True
            }
        }
        
        # Initialize storage
        self._initialize_storage()
        
        self.logger.info(f"Audit Logger initialized with {retention_days} days retention")
    
    def log_event(
        self,
        event_type: AuditEventType,
        message: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        context: Optional[AuditContext] = None,
        data_subject: Optional[DataSubject] = None,
        details: Optional[Dict[str, Any]] = None,
        before_value: Optional[str] = None,
        after_value: Optional[str] = None,
        affected_fields: Optional[List[str]] = None
    ) -> str:
        """Log an audit event"""
        
        try:
            with self._lock:
                # Generate unique record ID
                record_id = str(uuid.uuid4())
                
                # Create audit record
                record = AuditRecord(
                    record_id=record_id,
                    timestamp=datetime.utcnow(),
                    event_type=event_type,
                    severity=severity,
                    message=message,
                    details=details or {},
                    context=context or AuditContext(),
                    data_subject=data_subject,
                    before_value=before_value,
                    after_value=after_value,
                    affected_fields=affected_fields or [],
                    risk_level=self._assess_risk_level(event_type, severity, details),
                    compliance_impact=self._assess_compliance_impact(event_type, context),
                    remediation_required=self._requires_remediation(event_type, severity)
                )
                
                # Calculate checksum for integrity
                record.checksum = self._calculate_checksum(record)
                
                # Store record
                self.audit_records.append(record)
                self.audit_index[record_id] = record
                
                # Trigger event handlers
                self._trigger_event_handlers(record)
                
                # Check for compliance violations
                self._check_compliance_violations(record)
                
                # Auto-archive if needed
                if self.auto_archive and len(self.audit_records) > 10000:
                    self._archive_old_records()
                
                self.logger.debug(f"Audit event logged: {record_id} - {event_type.value}")
                
                return record_id
                
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {str(e)}")
            raise PrivacyError(f"Audit logging failed: {str(e)}")
    
    def log_privacy_assessment(
        self,
        assessment_type: str,
        data_processed: Dict[str, Any],
        results: Dict[str, Any],
        context: Optional[AuditContext] = None,
        data_subject: Optional[DataSubject] = None
    ) -> str:
        """Log privacy assessment event"""
        
        details = {
            "assessment_type": assessment_type,
            "data_fields_count": len(data_processed),
            "compliance_score": results.get("compliance_score"),
            "violations_found": results.get("violations", []),
            "risk_level": results.get("risk_level")
        }
        
        severity = AuditSeverity.WARNING if results.get("violations") else AuditSeverity.INFO
        
        return self.log_event(
            event_type=AuditEventType.PRIVACY_ASSESSMENT,
            message=f"Privacy assessment completed: {assessment_type}",
            severity=severity,
            context=context,
            data_subject=data_subject,
            details=details
        )
    
    def log_data_access(
        self,
        accessed_data: Dict[str, Any],
        purpose: str,
        context: Optional[AuditContext] = None,
        data_subject: Optional[DataSubject] = None
    ) -> str:
        """Log data access event"""
        
        details = {
            "access_purpose": purpose,
            "data_fields": list(accessed_data.keys()),
            "record_count": len(accessed_data) if isinstance(accessed_data, list) else 1,
            "access_time": datetime.utcnow().isoformat()
        }
        
        return self.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            message=f"Data accessed for: {purpose}",
            severity=AuditSeverity.INFO,
            context=context,
            data_subject=data_subject,
            details=details
        )
    
    def log_compliance_violation(
        self,
        violation_type: str,
        framework: ComplianceFramework,
        violation_details: Dict[str, Any],
        context: Optional[AuditContext] = None,
        data_subject: Optional[DataSubject] = None
    ) -> str:
        """Log compliance violation"""
        
        notification_deadline = self._get_notification_deadline(framework)
        details = {
            "violation_type": violation_type,
            "compliance_framework": framework.value,
            "violation_details": violation_details,
            "requires_notification": self._requires_breach_notification(framework, violation_details),
            "notification_deadline": notification_deadline.isoformat() if notification_deadline else None
        }
        
        # Update context with compliance framework
        if context:
            context.compliance_framework = framework
        else:
            context = AuditContext(compliance_framework=framework)
        
        return self.log_event(
            event_type=AuditEventType.POLICY_VIOLATION,
            message=f"Compliance violation detected: {violation_type}",
            severity=AuditSeverity.CRITICAL,
            context=context,
            data_subject=data_subject,
            details=details
        )
    
    def log_masking_operation(
        self,
        masking_method: str,
        fields_masked: List[str],
        success: bool,
        context: Optional[AuditContext] = None,
        data_subject: Optional[DataSubject] = None
    ) -> str:
        """Log data masking operation"""
        
        details = {
            "masking_method": masking_method,
            "fields_masked": fields_masked,
            "fields_count": len(fields_masked),
            "operation_success": success,
            "reversible": "reversible" in masking_method.lower()
        }
        
        severity = AuditSeverity.INFO if success else AuditSeverity.ERROR
        
        return self.log_event(
            event_type=AuditEventType.MASKING_OPERATION,
            message=f"Data masking applied: {masking_method}",
            severity=severity,
            context=context,
            data_subject=data_subject,
            details=details,
            affected_fields=fields_masked
        )
    
    def query_audit_records(self, query: AuditQuery) -> List[AuditRecord]:
        """Query audit records with filtering"""
        
        with self._lock:
            results = []
            
            for record in self.audit_records:
                if self._matches_query(record, query):
                    results.append(record)
                    if len(results) >= query.limit:
                        break
            
            return results
    
    def get_audit_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> AuditSummary:
        """Generate audit summary statistics"""
        
        with self._lock:
            # Filter records by time range
            records = self.audit_records
            if start_time or end_time:
                records = [
                    r for r in records
                    if (not start_time or r.timestamp >= start_time) and
                       (not end_time or r.timestamp <= end_time)
                ]
            
            # Calculate statistics
            events_by_type = {}
            events_by_severity = {}
            events_by_framework = {}
            data_subjects = set()
            users = set()
            high_risk_count = 0
            violation_count = 0
            
            for record in records:
                # Event type stats
                event_type = record.event_type.value
                events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
                
                # Severity stats
                severity = record.severity.value
                events_by_severity[severity] = events_by_severity.get(severity, 0) + 1
                
                # Framework stats
                if record.context.compliance_framework:
                    framework = record.context.compliance_framework.value
                    events_by_framework[framework] = events_by_framework.get(framework, 0) + 1
                
                # Risk and violations
                if record.risk_level in ["HIGH", "CRITICAL"]:
                    high_risk_count += 1
                
                if record.event_type == AuditEventType.POLICY_VIOLATION:
                    violation_count += 1
                
                # Data subjects and users
                if record.data_subject:
                    data_subjects.add(record.data_subject.subject_id)
                
                if record.context.user_id:
                    users.add(record.context.user_id)
            
            # Time range
            time_range = "All time"
            if records:
                earliest = min(r.timestamp for r in records)
                latest = max(r.timestamp for r in records)
                time_range = f"{earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}"
            
            return AuditSummary(
                total_events=len(records),
                events_by_type=events_by_type,
                events_by_severity=events_by_severity,
                events_by_framework=events_by_framework,
                high_risk_events=high_risk_count,
                compliance_violations=violation_count,
                data_subjects_affected=len(data_subjects),
                unique_users=len(users),
                time_range=time_range
            )
    
    def register_event_handler(
        self,
        event_type: AuditEventType,
        handler: Callable[[AuditRecord], None]
    ) -> None:
        """Register event handler for specific audit events"""
        
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        self.logger.info(f"Event handler registered for {event_type.value}")
    
    def export_audit_trail(
        self,
        output_path: str,
        query: Optional[AuditQuery] = None,
        format: str = "json"
    ) -> str:
        """Export audit trail to file"""
        
        records = self.query_audit_records(query) if query else self.audit_records
        
        output_file = Path(output_path)
        
        if format.lower() == "json":
            with open(output_file, 'w') as f:
                json.dump([asdict(record) for record in records], f, indent=2, default=str)
        elif format.lower() == "csv":
            import csv
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header
                writer.writerow([
                    'record_id', 'timestamp', 'event_type', 'severity', 'message',
                    'user_id', 'compliance_framework', 'risk_level'
                ])
                # Data
                for record in records:
                    writer.writerow([
                        record.record_id,
                        record.timestamp.isoformat(),
                        record.event_type.value,
                        record.severity.value,
                        record.message,
                        record.context.user_id or '',
                        record.context.compliance_framework.value if record.context.compliance_framework else '',
                        record.risk_level or ''
                    ])
        
        self.logger.info(f"Audit trail exported to {output_file}")
        return str(output_file)
    
    def verify_audit_integrity(self) -> Dict[str, Any]:
        """Verify audit trail integrity"""
        
        with self._lock:
            total_records = len(self.audit_records)
            corrupted_records = []
            
            for record in self.audit_records:
                expected_checksum = self._calculate_checksum(record, exclude_checksum=True)
                if record.checksum != expected_checksum:
                    corrupted_records.append(record.record_id)
            
            return {
                "total_records": total_records,
                "corrupted_records": len(corrupted_records),
                "corrupted_record_ids": corrupted_records,
                "integrity_percentage": ((total_records - len(corrupted_records)) / total_records * 100) if total_records > 0 else 100
            }
    
    def _initialize_storage(self) -> None:
        """Initialize audit storage"""
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Audit storage initialized at {self.storage_path}")
    
    def _assess_risk_level(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        details: Optional[Dict[str, Any]]
    ) -> str:
        """Assess risk level for audit event"""
        
        if severity == AuditSeverity.CRITICAL or severity == AuditSeverity.FATAL:
            return "CRITICAL"
        
        if event_type in [AuditEventType.POLICY_VIOLATION, AuditEventType.BREACH_DETECTION]:
            return "HIGH"
        
        if event_type in [AuditEventType.DATA_DELETION, AuditEventType.DATA_EXPORT]:
            return "MEDIUM"
        
        return "LOW"
    
    def _assess_compliance_impact(
        self,
        event_type: AuditEventType,
        context: Optional[AuditContext]
    ) -> str:
        """Assess compliance impact"""
        
        if not context or not context.compliance_framework:
            return "NONE"
        
        framework = context.compliance_framework
        
        if event_type == AuditEventType.POLICY_VIOLATION:
            return "CRITICAL"
        
        if framework == ComplianceFramework.GDPR and event_type == AuditEventType.DATA_EXPORT:
            return "HIGH"
        
        if framework == ComplianceFramework.HIPAA and event_type == AuditEventType.DATA_ACCESS:
            return "MEDIUM"
        
        return "LOW"
    
    def _requires_remediation(self, event_type: AuditEventType, severity: AuditSeverity) -> bool:
        """Determine if event requires remediation"""
        
        return (
            severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL, AuditSeverity.FATAL] or
            event_type in [AuditEventType.POLICY_VIOLATION, AuditEventType.BREACH_DETECTION]
        )
    
    def _calculate_checksum(self, record: AuditRecord, exclude_checksum: bool = False) -> str:
        """Calculate checksum for audit record integrity"""
        
        # Create a serializable version of the record
        record_dict = {
            'record_id': record.record_id,
            'timestamp': record.timestamp.isoformat(),
            'event_type': record.event_type.value,
            'severity': record.severity.value,
            'message': record.message,
            'details': record.details,
            'before_value': record.before_value,
            'after_value': record.after_value,
            'affected_fields': record.affected_fields,
            'risk_level': record.risk_level,
            'compliance_impact': record.compliance_impact,
            'remediation_required': record.remediation_required,
            'status': record.status.value
        }
        
        # Add context if present
        if record.context:
            record_dict['context'] = {
                'user_id': record.context.user_id,
                'session_id': record.context.session_id,
                'ip_address': record.context.ip_address,
                'user_agent': record.context.user_agent,
                'request_id': record.context.request_id,
                'operation_id': record.context.operation_id,
                'compliance_framework': record.context.compliance_framework.value if record.context.compliance_framework else None,
                'data_classification': record.context.data_classification.value if record.context.data_classification else None,
                'business_justification': record.context.business_justification,
                'approval_id': record.context.approval_id
            }
        
        # Add data subject if present
        if record.data_subject:
            record_dict['data_subject'] = {
                'subject_id': record.data_subject.subject_id,
                'subject_type': record.data_subject.subject_type,
                'jurisdiction': record.data_subject.jurisdiction,
                'consent_status': record.data_subject.consent_status,
                'retention_period': record.data_subject.retention_period
            }
        
        if not exclude_checksum and record.checksum:
            record_dict['checksum'] = record.checksum
        
        record_json = json.dumps(record_dict, sort_keys=True)
        return hashlib.sha256(record_json.encode()).hexdigest()
    
    def _trigger_event_handlers(self, record: AuditRecord) -> None:
        """Trigger registered event handlers"""
        
        handlers = self.event_handlers.get(record.event_type, [])
        for handler in handlers:
            try:
                handler(record)
            except Exception as e:
                self.logger.error(f"Event handler failed: {str(e)}")
    
    def _check_compliance_violations(self, record: AuditRecord) -> None:
        """Check for compliance violations in audit record"""
        
        if not record.context.compliance_framework:
            return
        
        framework = record.context.compliance_framework
        rules = self.compliance_rules.get(framework, {})
        
        # Check required fields
        required_fields = rules.get("required_fields", [])
        for field in required_fields:
            if field not in record.details:
                self.log_event(
                    event_type=AuditEventType.POLICY_VIOLATION,
                    message=f"Missing required field for {framework.value}: {field}",
                    severity=AuditSeverity.WARNING,
                    details={"missing_field": field, "original_record": record.record_id}
                )
    
    def _requires_breach_notification(
        self,
        framework: ComplianceFramework,
        violation_details: Dict[str, Any]
    ) -> bool:
        """Determine if breach notification is required"""
        
        if framework == ComplianceFramework.GDPR:
            return violation_details.get("affects_data_subjects", False)
        
        if framework == ComplianceFramework.HIPAA:
            return violation_details.get("unsecured_phi", False)
        
        return False
    
    def _get_notification_deadline(self, framework: ComplianceFramework) -> Optional[datetime]:
        """Get breach notification deadline"""
        
        rules = self.compliance_rules.get(framework, {})
        notification_hours = rules.get("breach_notification_hours")
        
        if notification_hours:
            return datetime.utcnow() + timedelta(hours=notification_hours)
        
        return None
    
    def _matches_query(self, record: AuditRecord, query: AuditQuery) -> bool:
        """Check if record matches query criteria"""
        
        # Time range
        if query.start_time and record.timestamp < query.start_time:
            return False
        if query.end_time and record.timestamp > query.end_time:
            return False
        
        # Event types
        if query.event_types and record.event_type not in query.event_types:
            return False
        
        # Severity levels
        if query.severity_levels and record.severity not in query.severity_levels:
            return False
        
        # User IDs
        if query.user_ids and record.context.user_id not in query.user_ids:
            return False
        
        # Compliance frameworks
        if (query.compliance_frameworks and 
            record.context.compliance_framework not in query.compliance_frameworks):
            return False
        
        # Data subjects
        if (query.data_subjects and 
            (not record.data_subject or record.data_subject.subject_id not in query.data_subjects)):
            return False
        
        # Risk levels
        if query.risk_levels and record.risk_level not in query.risk_levels:
            return False
        
        # Search text
        if query.search_text:
            search_text = query.search_text.lower()
            searchable_text = f"{record.message} {json.dumps(record.details)}".lower()
            if search_text not in searchable_text:
                return False
        
        return True
    
    def _archive_old_records(self) -> None:
        """Archive old audit records"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        
        with self._lock:
            active_records = []
            archived_count = 0
            
            for record in self.audit_records:
                if record.timestamp < cutoff_date:
                    archived_count += 1
                    # Remove from index
                    self.audit_index.pop(record.record_id, None)
                else:
                    active_records.append(record)
            
            self.audit_records = active_records
            
            if archived_count > 0:
                self.logger.info(f"Archived {archived_count} old audit records")
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit compliance report"""
        
        return {
            "audit_events_monitored": [event.value for event in AuditEventType],
            "severity_levels": [sev.value for sev in AuditSeverity],
            "compliance_frameworks": [fw.value for fw in ComplianceFramework],
            "retention_days": self.retention_days,
            "total_audit_records": len(self.audit_records),
            "event_handlers_registered": sum(len(handlers) for handlers in self.event_handlers.values()),
            "storage_path": str(self.storage_path)
        }