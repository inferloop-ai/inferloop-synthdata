"""
Monitoring resource abstractions for logs, metrics, and alerting
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from .base import BaseResource, ResourceConfig, ResourceType


class MetricUnit(Enum):
    """Units for metrics"""
    COUNT = "count"
    BYTES = "bytes"
    SECONDS = "seconds"
    PERCENT = "percent"
    BYTES_PER_SECOND = "bytes_per_second"
    COUNT_PER_SECOND = "count_per_second"
    MILLISECONDS = "milliseconds"
    NONE = "none"


class AlertSeverity(Enum):
    """Severity levels for alerts"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class LogLevel(Enum):
    """Log levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricAggregation(Enum):
    """Metric aggregation types"""
    AVG = "average"
    SUM = "sum"
    MIN = "minimum"
    MAX = "maximum"
    COUNT = "count"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"


@dataclass
class MonitoringConfig(ResourceConfig):
    """Base configuration for monitoring resources"""
    retention_days: int = 30
    encryption_enabled: bool = True
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if self.retention_days < 1:
            errors.append("Retention days must be at least 1")
        return errors


@dataclass
class LogGroupConfig(MonitoringConfig):
    """Configuration for log groups"""
    log_stream_prefix: str = ""
    metric_filters: List[Dict[str, Any]] = field(default_factory=list)
    subscription_filters: List[Dict[str, Any]] = field(default_factory=list)
    
    def validate(self) -> List[str]:
        errors = super().validate()
        return errors


@dataclass
class MetricConfig(MonitoringConfig):
    """Configuration for custom metrics"""
    namespace: str = ""
    metric_name: str = ""
    dimensions: Dict[str, str] = field(default_factory=dict)
    unit: MetricUnit = MetricUnit.NONE
    storage_resolution: int = 60  # seconds
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if not self.namespace:
            errors.append("Namespace is required")
        if not self.metric_name:
            errors.append("Metric name is required")
        if self.storage_resolution not in [1, 60]:
            errors.append("Storage resolution must be 1 or 60 seconds")
        return errors


@dataclass
class AlertConfig(ResourceConfig):
    """Configuration for alerts"""
    metric_namespace: str = ""
    metric_name: str = ""
    threshold: float = 0.0
    comparison_operator: str = "GreaterThanThreshold"
    evaluation_periods: int = 1
    period: int = 300  # seconds
    statistic: MetricAggregation = MetricAggregation.AVG
    treat_missing_data: str = "missing"
    actions: List[str] = field(default_factory=list)
    severity: AlertSeverity = AlertSeverity.MEDIUM
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if not self.metric_namespace:
            errors.append("Metric namespace is required")
        if not self.metric_name:
            errors.append("Metric name is required")
        if self.evaluation_periods < 1:
            errors.append("Evaluation periods must be at least 1")
        if self.period < 60:
            errors.append("Period must be at least 60 seconds")
        return errors


@dataclass
class DashboardConfig(ResourceConfig):
    """Configuration for monitoring dashboards"""
    widgets: List[Dict[str, Any]] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    refresh_interval: int = 300  # seconds
    
    def validate(self) -> List[str]:
        errors = super().validate()
        if not self.widgets:
            errors.append("At least one widget is required")
        return errors


@dataclass
class MonitoringResource:
    """Representation of a monitoring resource"""
    id: str
    name: str
    type: str
    state: str
    config: MonitoringConfig
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEntry:
    """Representation of a log entry"""
    timestamp: datetime
    message: str
    level: LogLevel
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricDataPoint:
    """Representation of a metric data point"""
    timestamp: datetime
    value: float
    unit: MetricUnit
    dimensions: Dict[str, str] = field(default_factory=dict)


class BaseMonitoring(BaseResource[MonitoringConfig, MonitoringResource]):
    """Base class for monitoring resources"""
    
    def get_resource_type(self) -> ResourceType:
        return ResourceType.MONITORING
    
    @abstractmethod
    async def create_dashboard(self, config: DashboardConfig) -> str:
        """Create monitoring dashboard"""
        pass
    
    @abstractmethod
    async def update_dashboard(self, dashboard_id: str, config: DashboardConfig) -> bool:
        """Update monitoring dashboard"""
        pass
    
    @abstractmethod
    async def create_alert(self, config: AlertConfig) -> str:
        """Create alert/alarm"""
        pass
    
    @abstractmethod
    async def update_alert(self, alert_id: str, config: AlertConfig) -> bool:
        """Update alert/alarm"""
        pass
    
    @abstractmethod
    async def enable_alert(self, alert_id: str) -> bool:
        """Enable alert"""
        pass
    
    @abstractmethod
    async def disable_alert(self, alert_id: str) -> bool:
        """Disable alert"""
        pass
    
    @abstractmethod
    async def get_alert_history(self, alert_id: str, start_time: datetime, 
                               end_time: datetime) -> List[Dict[str, Any]]:
        """Get alert history"""
        pass


class BaseLogging(BaseMonitoring):
    """Base class for logging services"""
    
    @abstractmethod
    async def create_log_group(self, config: LogGroupConfig) -> str:
        """Create log group"""
        pass
    
    @abstractmethod
    async def create_log_stream(self, log_group: str, stream_name: str) -> bool:
        """Create log stream"""
        pass
    
    @abstractmethod
    async def put_log_events(self, log_group: str, stream_name: str, 
                            events: List[LogEntry]) -> bool:
        """Write log events"""
        pass
    
    @abstractmethod
    async def get_log_events(self, log_group: str, stream_name: str,
                            start_time: datetime, end_time: datetime,
                            filter_pattern: Optional[str] = None) -> List[LogEntry]:
        """Retrieve log events"""
        pass
    
    @abstractmethod
    async def create_metric_filter(self, log_group: str, filter_name: str,
                                  filter_pattern: str, metric_config: MetricConfig) -> bool:
        """Create metric filter from logs"""
        pass
    
    @abstractmethod
    async def create_subscription_filter(self, log_group: str, filter_name: str,
                                       filter_pattern: str, destination_arn: str) -> bool:
        """Create subscription filter"""
        pass
    
    @abstractmethod
    async def tail_logs(self, log_group: str, stream_name: Optional[str] = None,
                       filter_pattern: Optional[str] = None) -> None:
        """Tail logs in real-time"""
        pass
    
    @abstractmethod
    async def export_logs(self, log_group: str, start_time: datetime,
                         end_time: datetime, destination: str) -> str:
        """Export logs to storage"""
        pass


class BaseMetrics(BaseMonitoring):
    """Base class for metrics services"""
    
    @abstractmethod
    async def put_metric_data(self, namespace: str, metrics: List[MetricDataPoint]) -> bool:
        """Write metric data"""
        pass
    
    @abstractmethod
    async def get_metric_data(self, namespace: str, metric_name: str,
                             start_time: datetime, end_time: datetime,
                             period: int, statistic: MetricAggregation,
                             dimensions: Optional[Dict[str, str]] = None) -> List[MetricDataPoint]:
        """Retrieve metric data"""
        pass
    
    @abstractmethod
    async def list_metrics(self, namespace: Optional[str] = None,
                          metric_name: Optional[str] = None,
                          dimensions: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """List available metrics"""
        pass
    
    @abstractmethod
    async def get_metric_statistics(self, namespace: str, metric_name: str,
                                   start_time: datetime, end_time: datetime,
                                   period: int, statistics: List[MetricAggregation],
                                   dimensions: Optional[Dict[str, str]] = None) -> Dict[str, List[float]]:
        """Get metric statistics"""
        pass
    
    @abstractmethod
    async def create_composite_alarm(self, name: str, alarm_rule: str,
                                   actions: List[str]) -> str:
        """Create composite alarm from multiple metrics"""
        pass
    
    @abstractmethod
    async def put_anomaly_detector(self, namespace: str, metric_name: str,
                                  dimensions: Optional[Dict[str, str]] = None) -> bool:
        """Create anomaly detector for metric"""
        pass