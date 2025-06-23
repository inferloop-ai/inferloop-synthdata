"""AWS monitoring resources implementation."""

import boto3
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from common.core.monitoring import (
    BaseMonitoring,
    MetricData,
    LogEntry,
    Alert,
    Dashboard,
    TraceSpan,
    MetricType,
    LogLevel,
    AlertSeverity,
)
from common.core.config import InfrastructureConfig
from common.core.exceptions import (
    MonitoringError,
    ResourceCreationError,
    ResourceNotFoundError,
)


class AWSMonitoring(BaseMonitoring):
    """AWS CloudWatch monitoring implementation."""
    
    def __init__(self, session: boto3.Session, config: InfrastructureConfig):
        """Initialize AWS monitoring manager."""
        super().__init__(config.to_provider_config())
        self.session = session
        self.config = config
        self.cloudwatch_client = session.client("cloudwatch")
        self.logs_client = session.client("logs")
        self.xray_client = session.client("xray")
        self.sns_client = session.client("sns")
        
        # Create default log group
        self.default_log_group = f"/aws/{config.project_name}/{config.environment.value}"
        self._ensure_log_group(self.default_log_group)
        
        # Create SNS topic for alerts
        self.alert_topic_arn = self._ensure_sns_topic()
    
    def put_metric(self, metric: MetricData) -> None:
        """Send a metric data point to CloudWatch."""
        try:
            # Prepare metric data
            metric_data = {
                "MetricName": metric.name,
                "Value": metric.value,
                "Unit": self._convert_unit(metric.unit),
                "Timestamp": metric.timestamp,
                "Dimensions": [
                    {"Name": k, "Value": v}
                    for k, v in metric.dimensions.items()
                ],
            }
            
            # Add storage resolution if specified
            if metric.metadata.get("storage_resolution"):
                metric_data["StorageResolution"] = metric.metadata["storage_resolution"]
            
            # Send metric
            self.cloudwatch_client.put_metric_data(
                Namespace=f"{self.config.project_name}/{self.config.environment.value}",
                MetricData=[metric_data],
            )
            
        except Exception as e:
            raise MonitoringError("put_metric", str(e))
    
    def put_metrics(self, metrics: List[MetricData]) -> None:
        """Send multiple metric data points to CloudWatch."""
        try:
            # CloudWatch accepts max 20 metrics per request
            for i in range(0, len(metrics), 20):
                batch = metrics[i:i + 20]
                
                metric_data = []
                for metric in batch:
                    data = {
                        "MetricName": metric.name,
                        "Value": metric.value,
                        "Unit": self._convert_unit(metric.unit),
                        "Timestamp": metric.timestamp,
                        "Dimensions": [
                            {"Name": k, "Value": v}
                            for k, v in metric.dimensions.items()
                        ],
                    }
                    
                    if metric.metadata.get("storage_resolution"):
                        data["StorageResolution"] = metric.metadata["storage_resolution"]
                    
                    metric_data.append(data)
                
                self.cloudwatch_client.put_metric_data(
                    Namespace=f"{self.config.project_name}/{self.config.environment.value}",
                    MetricData=metric_data,
                )
                
        except Exception as e:
            raise MonitoringError("put_metrics", str(e))
    
    def get_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        dimensions: Optional[Dict[str, str]] = None,
        statistic: str = "Average",
    ) -> List[Dict[str, Any]]:
        """Retrieve metric data from CloudWatch."""
        try:
            params = {
                "Namespace": f"{self.config.project_name}/{self.config.environment.value}",
                "MetricName": metric_name,
                "StartTime": start_time,
                "EndTime": end_time,
                "Period": 300,  # 5 minutes
                "Statistics": [statistic],
            }
            
            if dimensions:
                params["Dimensions"] = [
                    {"Name": k, "Value": v}
                    for k, v in dimensions.items()
                ]
            
            response = self.cloudwatch_client.get_metric_statistics(**params)
            
            # Format response
            datapoints = []
            for point in response["Datapoints"]:
                datapoints.append({
                    "timestamp": point["Timestamp"],
                    "value": point.get(statistic, 0),
                    "unit": point.get("Unit", "None"),
                })
            
            # Sort by timestamp
            datapoints.sort(key=lambda x: x["timestamp"])
            
            return datapoints
            
        except Exception as e:
            raise MonitoringError("get_metrics", str(e))
    
    def put_log(self, log_entry: LogEntry) -> None:
        """Send a log entry to CloudWatch Logs."""
        try:
            # Ensure log stream exists
            log_stream = f"{self.config.environment.value}/{log_entry.logger}"
            self._ensure_log_stream(self.default_log_group, log_stream)
            
            # Prepare log event
            log_event = {
                "timestamp": int(log_entry.timestamp.timestamp() * 1000),
                "message": json.dumps({
                    "level": log_entry.level.value,
                    "message": log_entry.message,
                    "resource_id": log_entry.resource_id,
                    "trace_id": log_entry.trace_id,
                    "span_id": log_entry.span_id,
                    "metadata": log_entry.metadata,
                }),
            }
            
            # Send log
            self.logs_client.put_log_events(
                logGroupName=self.default_log_group,
                logStreamName=log_stream,
                logEvents=[log_event],
            )
            
        except Exception as e:
            raise MonitoringError("put_log", str(e))
    
    def put_logs(self, log_entries: List[LogEntry]) -> None:
        """Send multiple log entries to CloudWatch Logs."""
        try:
            # Group logs by stream
            logs_by_stream = {}
            for entry in log_entries:
                stream = f"{self.config.environment.value}/{entry.logger}"
                if stream not in logs_by_stream:
                    logs_by_stream[stream] = []
                
                logs_by_stream[stream].append({
                    "timestamp": int(entry.timestamp.timestamp() * 1000),
                    "message": json.dumps({
                        "level": entry.level.value,
                        "message": entry.message,
                        "resource_id": entry.resource_id,
                        "trace_id": entry.trace_id,
                        "span_id": entry.span_id,
                        "metadata": entry.metadata,
                    }),
                })
            
            # Send logs by stream
            for stream, events in logs_by_stream.items():
                self._ensure_log_stream(self.default_log_group, stream)
                
                # Sort events by timestamp
                events.sort(key=lambda x: x["timestamp"])
                
                # CloudWatch accepts max 10,000 log events per request
                for i in range(0, len(events), 10000):
                    batch = events[i:i + 10000]
                    
                    self.logs_client.put_log_events(
                        logGroupName=self.default_log_group,
                        logStreamName=stream,
                        logEvents=batch,
                    )
                    
        except Exception as e:
            raise MonitoringError("put_logs", str(e))
    
    def query_logs(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
    ) -> List[LogEntry]:
        """Query logs using CloudWatch Insights."""
        try:
            # Start query
            response = self.logs_client.start_query(
                logGroupName=self.default_log_group,
                startTime=int(start_time.timestamp()),
                endTime=int(end_time.timestamp()),
                queryString=query,
                limit=limit,
            )
            
            query_id = response["queryId"]
            
            # Wait for query to complete
            status = "Running"
            while status == "Running":
                response = self.logs_client.get_query_results(queryId=query_id)
                status = response["status"]
                
                if status == "Running":
                    import time
                    time.sleep(1)
            
            # Parse results
            log_entries = []
            for result in response["results"]:
                # Convert result to dict
                entry_data = {}
                for field in result:
                    entry_data[field["field"]] = field["value"]
                
                # Parse JSON message if present
                if "@message" in entry_data:
                    try:
                        message_data = json.loads(entry_data["@message"])
                        log_entries.append(
                            LogEntry(
                                timestamp=datetime.fromtimestamp(
                                    float(entry_data.get("@timestamp", 0)) / 1000
                                ),
                                level=LogLevel(message_data.get("level", "info")),
                                message=message_data.get("message", ""),
                                logger=entry_data.get("@logStream", "").split("/")[-1],
                                resource_id=message_data.get("resource_id"),
                                trace_id=message_data.get("trace_id"),
                                span_id=message_data.get("span_id"),
                                metadata=message_data.get("metadata", {}),
                            )
                        )
                    except:
                        pass
            
            return log_entries
            
        except Exception as e:
            raise MonitoringError("query_logs", str(e))
    
    def create_alert(self, alert: Alert) -> str:
        """Create a CloudWatch alarm."""
        try:
            # Create alarm
            self.cloudwatch_client.put_metric_alarm(
                AlarmName=alert.name,
                ComparisonOperator=self._convert_comparison_operator(alert.comparison_operator),
                EvaluationPeriods=alert.evaluation_periods,
                MetricName=alert.metric_name,
                Namespace=f"{self.config.project_name}/{self.config.environment.value}",
                Period=300,  # 5 minutes
                Statistic="Average",
                Threshold=alert.threshold,
                ActionsEnabled=alert.enabled,
                AlarmActions=[self.alert_topic_arn] if alert.actions else [],
                AlarmDescription=alert.description,
                DatapointsToAlarm=alert.datapoints_to_alarm,
                Tags=self._format_tags(alert.tags),
            )
            
            # Subscribe to SNS topic if email provided
            if self.config.alert_email and alert.enabled:
                self.sns_client.subscribe(
                    TopicArn=self.alert_topic_arn,
                    Protocol="email",
                    Endpoint=self.config.alert_email,
                )
            
            return alert.name
            
        except Exception as e:
            raise ResourceCreationError("CloudWatch Alarm", str(e))
    
    def delete_alert(self, alert_name: str) -> None:
        """Delete a CloudWatch alarm."""
        try:
            self.cloudwatch_client.delete_alarms(AlarmNames=[alert_name])
        except Exception as e:
            raise MonitoringError("delete_alert", str(e))
    
    def list_alerts(self) -> List[Alert]:
        """List all CloudWatch alarms."""
        try:
            alerts = []
            
            paginator = self.cloudwatch_client.get_paginator("describe_alarms")
            for page in paginator.paginate():
                for alarm in page["MetricAlarms"]:
                    # Only include alarms for our namespace
                    if alarm["Namespace"] == f"{self.config.project_name}/{self.config.environment.value}":
                        alerts.append(
                            Alert(
                                name=alarm["AlarmName"],
                                description=alarm.get("AlarmDescription", ""),
                                metric_name=alarm["MetricName"],
                                threshold=alarm["Threshold"],
                                comparison_operator=self._reverse_comparison_operator(
                                    alarm["ComparisonOperator"]
                                ),
                                evaluation_periods=alarm["EvaluationPeriods"],
                                datapoints_to_alarm=alarm.get("DatapointsToAlarm", 1),
                                severity=AlertSeverity.MEDIUM,  # CloudWatch doesn't have severity
                                actions=alarm.get("AlarmActions", []),
                                enabled=alarm.get("ActionsEnabled", True),
                            )
                        )
            
            return alerts
            
        except Exception as e:
            raise MonitoringError("list_alerts", str(e))
    
    def create_dashboard(self, dashboard: Dashboard) -> str:
        """Create a CloudWatch dashboard."""
        try:
            # Convert widgets to CloudWatch format
            cw_widgets = []
            
            for widget in dashboard.widgets:
                if widget["type"] == "metric":
                    cw_widget = {
                        "type": "metric",
                        "properties": {
                            "metrics": widget["metrics"],
                            "period": widget.get("period", 300),
                            "stat": widget.get("stat", "Average"),
                            "region": self.config.region,
                            "title": widget.get("title", ""),
                        },
                    }
                elif widget["type"] == "log":
                    cw_widget = {
                        "type": "log",
                        "properties": {
                            "query": widget["query"],
                            "region": self.config.region,
                            "title": widget.get("title", ""),
                        },
                    }
                else:
                    continue
                
                # Add position and size
                cw_widget.update({
                    "x": widget.get("x", 0),
                    "y": widget.get("y", 0),
                    "width": widget.get("width", 12),
                    "height": widget.get("height", 6),
                })
                
                cw_widgets.append(cw_widget)
            
            # Create dashboard
            self.cloudwatch_client.put_dashboard(
                DashboardName=dashboard.name,
                DashboardBody=json.dumps({"widgets": cw_widgets}),
            )
            
            return dashboard.name
            
        except Exception as e:
            raise ResourceCreationError("CloudWatch Dashboard", str(e))
    
    def update_dashboard(self, dashboard_id: str, dashboard: Dashboard) -> None:
        """Update a CloudWatch dashboard."""
        try:
            # CloudWatch uses name as ID
            self.create_dashboard(dashboard)
        except Exception as e:
            raise MonitoringError("update_dashboard", str(e))
    
    def delete_dashboard(self, dashboard_id: str) -> None:
        """Delete a CloudWatch dashboard."""
        try:
            self.cloudwatch_client.delete_dashboards(DashboardNames=[dashboard_id])
        except Exception as e:
            raise MonitoringError("delete_dashboard", str(e))
    
    def put_trace(self, span: TraceSpan) -> None:
        """Send a trace span to X-Ray."""
        try:
            # Convert to X-Ray format
            segment = {
                "trace_id": span.trace_id,
                "id": span.span_id,
                "parent_id": span.parent_span_id,
                "name": span.operation_name,
                "start_time": span.start_time.timestamp(),
                "end_time": span.end_time.timestamp(),
                "in_progress": False,
                "annotations": span.attributes,
            }
            
            # Add error flag if status is error
            if span.status == "error":
                segment["error"] = True
                segment["cause"] = {
                    "exceptions": [
                        {
                            "message": span.attributes.get("error.message", "Unknown error"),
                            "type": span.attributes.get("error.type", "Error"),
                        }
                    ]
                }
            
            # Send to X-Ray
            self.xray_client.put_trace_segments(
                TraceSegmentDocuments=[json.dumps(segment)]
            )
            
        except Exception as e:
            # X-Ray might not be available, don't fail
            pass
    
    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace from X-Ray."""
        try:
            # Get trace summaries
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            
            response = self.xray_client.get_trace_summaries(
                TimeRangeType="TraceId",
                TraceIds=[trace_id],
                StartTime=start_time,
                EndTime=end_time,
            )
            
            if not response["TraceSummaries"]:
                return []
            
            # Get full trace
            trace_response = self.xray_client.batch_get_traces(TraceIds=[trace_id])
            
            spans = []
            for trace in trace_response["Traces"]:
                for segment in trace["Segments"]:
                    doc = json.loads(segment["Document"])
                    
                    spans.append(
                        TraceSpan(
                            trace_id=doc["trace_id"],
                            span_id=doc["id"],
                            parent_span_id=doc.get("parent_id"),
                            operation_name=doc["name"],
                            start_time=datetime.fromtimestamp(doc["start_time"]),
                            end_time=datetime.fromtimestamp(doc["end_time"]),
                            status="error" if doc.get("error") else "ok",
                            attributes=doc.get("annotations", {}),
                        )
                    )
            
            return spans
            
        except Exception:
            return []
    
    def create_synthdata_dashboard(self) -> str:
        """Create a dashboard for synthetic data operations."""
        dashboard = Dashboard(
            name=f"{self.config.resource_name}-synthdata-dashboard",
            description="Synthetic Data Generation Monitoring",
            widgets=[
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "title": "Data Generation Rate",
                    "metrics": [
                        [
                            f"{self.config.project_name}/{self.config.environment.value}",
                            "synthetic_data_rows_generated",
                            {"stat": "Sum", "period": 300},
                        ]
                    ],
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "title": "Generation Duration",
                    "metrics": [
                        [
                            f"{self.config.project_name}/{self.config.environment.value}",
                            "synthetic_data_generation_duration",
                            {"stat": "Average", "period": 300},
                        ]
                    ],
                },
                {
                    "type": "metric",
                    "x": 0,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "title": "API Latency",
                    "metrics": [
                        [
                            f"{self.config.project_name}/{self.config.environment.value}",
                            "api_latency",
                            {"stat": "Average", "period": 300},
                        ]
                    ],
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "title": "Resource Utilization",
                    "metrics": [
                        [
                            f"{self.config.project_name}/{self.config.environment.value}",
                            "resource_cpu_utilization",
                            {"stat": "Average", "period": 300},
                        ],
                        [
                            ".",
                            "resource_memory_utilization",
                            {"stat": "Average", "period": 300},
                        ],
                    ],
                },
                {
                    "type": "log",
                    "x": 0,
                    "y": 12,
                    "width": 24,
                    "height": 6,
                    "title": "Recent Errors",
                    "query": f"""
                        fields @timestamp, message, resource_id
                        | filter level = "error"
                        | sort @timestamp desc
                        | limit 20
                    """,
                },
            ],
        )
        
        return self.create_dashboard(dashboard)
    
    def _ensure_log_group(self, log_group_name: str) -> None:
        """Ensure a log group exists."""
        try:
            self.logs_client.create_log_group(
                logGroupName=log_group_name,
                tags=self.config.default_tags,
            )
            
            # Set retention
            self.logs_client.put_retention_policy(
                logGroupName=log_group_name,
                retentionInDays=self.config.log_retention_days,
            )
        except self.logs_client.exceptions.ResourceAlreadyExistsException:
            pass
    
    def _ensure_log_stream(self, log_group_name: str, log_stream_name: str) -> None:
        """Ensure a log stream exists."""
        try:
            self.logs_client.create_log_stream(
                logGroupName=log_group_name,
                logStreamName=log_stream_name,
            )
        except self.logs_client.exceptions.ResourceAlreadyExistsException:
            pass
    
    def _ensure_sns_topic(self) -> str:
        """Ensure SNS topic exists for alerts."""
        topic_name = f"{self.config.resource_name}-alerts"
        
        try:
            # Check if topic exists
            response = self.sns_client.list_topics()
            for topic in response["Topics"]:
                if topic_name in topic["TopicArn"]:
                    return topic["TopicArn"]
            
            # Create topic
            response = self.sns_client.create_topic(
                Name=topic_name,
                Tags=self._format_tags(self.config.default_tags),
            )
            
            return response["TopicArn"]
            
        except Exception:
            # Return a placeholder if SNS is not available
            return f"arn:aws:sns:{self.config.region}:123456789012:placeholder"
    
    def _convert_unit(self, unit: str) -> str:
        """Convert unit to CloudWatch format."""
        unit_map = {
            "count": "Count",
            "seconds": "Seconds",
            "milliseconds": "Milliseconds",
            "bytes": "Bytes",
            "kilobytes": "Kilobytes",
            "megabytes": "Megabytes",
            "gigabytes": "Gigabytes",
            "percent": "Percent",
            "rows": "Count",
            "rows/second": "Count/Second",
        }
        
        return unit_map.get(unit.lower(), "None")
    
    def _convert_comparison_operator(self, operator: str) -> str:
        """Convert comparison operator to CloudWatch format."""
        operator_map = {
            ">": "GreaterThanThreshold",
            ">=": "GreaterThanOrEqualToThreshold",
            "<": "LessThanThreshold",
            "<=": "LessThanOrEqualToThreshold",
            "==": "EqualToThreshold",
            "!=": "NotEqualToThreshold",
        }
        
        return operator_map.get(operator, "GreaterThanThreshold")
    
    def _reverse_comparison_operator(self, cw_operator: str) -> str:
        """Convert CloudWatch operator back to standard format."""
        operator_map = {
            "GreaterThanThreshold": ">",
            "GreaterThanOrEqualToThreshold": ">=",
            "LessThanThreshold": "<",
            "LessThanOrEqualToThreshold": "<=",
            "EqualToThreshold": "==",
            "NotEqualToThreshold": "!=",
        }
        
        return operator_map.get(cw_operator, ">")
    
    def _format_tags(self, tags: Dict[str, str]) -> List[Dict[str, str]]:
        """Format tags for AWS API."""
        return [{"Key": k, "Value": v} for k, v in tags.items()]