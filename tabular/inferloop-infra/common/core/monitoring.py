"""Monitoring and observability components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum


class MetricType(Enum):
    """Types of metrics."""
    
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class LogLevel(Enum):
    """Log levels."""
    
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MetricData:
    """Metric data point."""
    
    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    unit: str = "count"
    dimensions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEntry:
    """Log entry."""
    
    timestamp: datetime
    level: LogLevel
    message: str
    logger: str
    resource_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert definition."""
    
    name: str
    description: str
    metric_name: str
    threshold: float
    comparison_operator: str  # >, <, >=, <=, ==, !=
    evaluation_periods: int = 1
    datapoints_to_alarm: int = 1
    severity: AlertSeverity = AlertSeverity.MEDIUM
    actions: List[str] = field(default_factory=list)
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Dashboard:
    """Monitoring dashboard definition."""
    
    name: str
    description: str
    widgets: List[Dict[str, Any]]
    refresh_interval: int = 300  # seconds
    time_range: str = "1h"  # 1h, 3h, 6h, 12h, 1d, 3d, 7d
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class TraceSpan:
    """Distributed trace span."""
    
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: datetime
    status: str  # ok, error
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)


class BaseMonitoring(ABC):
    """Abstract base class for monitoring providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize monitoring provider."""
        self.config = config
    
    @abstractmethod
    def put_metric(self, metric: MetricData) -> None:
        """Send a metric data point."""
        pass
    
    @abstractmethod
    def put_metrics(self, metrics: List[MetricData]) -> None:
        """Send multiple metric data points."""
        pass
    
    @abstractmethod
    def get_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        dimensions: Optional[Dict[str, str]] = None,
        statistic: str = "Average",
    ) -> List[Dict[str, Any]]:
        """Retrieve metric data."""
        pass
    
    @abstractmethod
    def put_log(self, log_entry: LogEntry) -> None:
        """Send a log entry."""
        pass
    
    @abstractmethod
    def put_logs(self, log_entries: List[LogEntry]) -> None:
        """Send multiple log entries."""
        pass
    
    @abstractmethod
    def query_logs(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
    ) -> List[LogEntry]:
        """Query logs."""
        pass
    
    @abstractmethod
    def create_alert(self, alert: Alert) -> str:
        """Create an alert."""
        pass
    
    @abstractmethod
    def delete_alert(self, alert_name: str) -> None:
        """Delete an alert."""
        pass
    
    @abstractmethod
    def list_alerts(self) -> List[Alert]:
        """List all alerts."""
        pass
    
    @abstractmethod
    def create_dashboard(self, dashboard: Dashboard) -> str:
        """Create a monitoring dashboard."""
        pass
    
    @abstractmethod
    def update_dashboard(self, dashboard_id: str, dashboard: Dashboard) -> None:
        """Update a dashboard."""
        pass
    
    @abstractmethod
    def delete_dashboard(self, dashboard_id: str) -> None:
        """Delete a dashboard."""
        pass
    
    @abstractmethod
    def put_trace(self, span: TraceSpan) -> None:
        """Send a trace span."""
        pass
    
    @abstractmethod
    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace."""
        pass
    
    def record_api_latency(self, endpoint: str, latency_ms: float, status_code: int) -> None:
        """Record API endpoint latency."""
        self.put_metric(
            MetricData(
                name="api_latency",
                value=latency_ms,
                timestamp=datetime.utcnow(),
                metric_type=MetricType.HISTOGRAM,
                unit="milliseconds",
                dimensions={
                    "endpoint": endpoint,
                    "status_code": str(status_code),
                },
            )
        )
    
    def record_synthetic_data_generation(
        self,
        generator_type: str,
        rows_generated: int,
        duration_seconds: float,
        success: bool,
    ) -> None:
        """Record synthetic data generation metrics."""
        timestamp = datetime.utcnow()
        
        # Record count
        self.put_metric(
            MetricData(
                name="synthetic_data_rows_generated",
                value=rows_generated,
                timestamp=timestamp,
                metric_type=MetricType.COUNTER,
                unit="count",
                dimensions={
                    "generator_type": generator_type,
                    "status": "success" if success else "failure",
                },
            )
        )
        
        # Record duration
        self.put_metric(
            MetricData(
                name="synthetic_data_generation_duration",
                value=duration_seconds,
                timestamp=timestamp,
                metric_type=MetricType.HISTOGRAM,
                unit="seconds",
                dimensions={
                    "generator_type": generator_type,
                    "status": "success" if success else "failure",
                },
            )
        )
        
        # Record throughput
        if duration_seconds > 0:
            throughput = rows_generated / duration_seconds
            self.put_metric(
                MetricData(
                    name="synthetic_data_generation_throughput",
                    value=throughput,
                    timestamp=timestamp,
                    metric_type=MetricType.GAUGE,
                    unit="rows/second",
                    dimensions={"generator_type": generator_type},
                )
            )
    
    def record_resource_utilization(
        self,
        resource_id: str,
        cpu_percent: float,
        memory_percent: float,
        disk_percent: Optional[float] = None,
    ) -> None:
        """Record resource utilization metrics."""
        timestamp = datetime.utcnow()
        
        metrics = [
            MetricData(
                name="resource_cpu_utilization",
                value=cpu_percent,
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                unit="percent",
                dimensions={"resource_id": resource_id},
            ),
            MetricData(
                name="resource_memory_utilization",
                value=memory_percent,
                timestamp=timestamp,
                metric_type=MetricType.GAUGE,
                unit="percent",
                dimensions={"resource_id": resource_id},
            ),
        ]
        
        if disk_percent is not None:
            metrics.append(
                MetricData(
                    name="resource_disk_utilization",
                    value=disk_percent,
                    timestamp=timestamp,
                    metric_type=MetricType.GAUGE,
                    unit="percent",
                    dimensions={"resource_id": resource_id},
                )
            )
        
        self.put_metrics(metrics)
    
    def log_event(
        self,
        level: LogLevel,
        message: str,
        resource_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Log an event."""
        self.put_log(
            LogEntry(
                timestamp=datetime.utcnow(),
                level=level,
                message=message,
                logger=self.__class__.__name__,
                resource_id=resource_id,
                metadata=kwargs,
            )
        )
    
    def create_standard_alerts(self, resource_name: str) -> List[str]:
        """Create standard alerts for a resource."""
        alerts = [
            Alert(
                name=f"{resource_name}-high-cpu",
                description=f"High CPU utilization on {resource_name}",
                metric_name="resource_cpu_utilization",
                threshold=80,
                comparison_operator=">=",
                evaluation_periods=2,
                severity=AlertSeverity.HIGH,
            ),
            Alert(
                name=f"{resource_name}-high-memory",
                description=f"High memory utilization on {resource_name}",
                metric_name="resource_memory_utilization",
                threshold=85,
                comparison_operator=">=",
                evaluation_periods=2,
                severity=AlertSeverity.HIGH,
            ),
            Alert(
                name=f"{resource_name}-api-errors",
                description=f"High API error rate on {resource_name}",
                metric_name="api_error_rate",
                threshold=5,
                comparison_operator=">=",
                evaluation_periods=1,
                severity=AlertSeverity.CRITICAL,
            ),
        ]
        
        alert_ids = []
        for alert in alerts:
            alert_id = self.create_alert(alert)
            alert_ids.append(alert_id)
        
        return alert_ids