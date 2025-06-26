"""
Business Metrics Dashboard for TextNLP
Comprehensive business intelligence and analytics dashboard
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
import statistics
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
import contextlib

logger = logging.getLogger(__name__)


class MetricCategory(Enum):
    """Categories of business metrics"""
    USAGE = "usage"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    COST = "cost"
    REVENUE = "revenue"
    USER_ENGAGEMENT = "user_engagement"
    SAFETY = "safety"
    COMPLIANCE = "compliance"


class TimeGranularity(Enum):
    """Time granularities for metrics aggregation"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


@dataclass
class BusinessMetric:
    """Individual business metric"""
    metric_name: str
    category: MetricCategory
    value: Union[float, int]
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    title: str
    widget_type: str  # "chart", "table", "kpi", "alert"
    metrics: List[str]
    config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=dict)


@dataclass
class BusinessReport:
    """Business intelligence report"""
    report_id: str
    title: str
    generated_at: datetime
    time_period: Dict[str, datetime]
    executive_summary: Dict[str, Any]
    detailed_metrics: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    data_sources: List[str]


class BusinessMetricsCollector:
    """Collects and aggregates business metrics from various sources"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics_history: List[BusinessMetric] = []
        self.aggregated_metrics: Dict[str, Any] = {}
        
        # Database for persistent storage
        self.db_path = self.config.get("db_path", "business_metrics.db")
        self._initialize_database()
        
        # Data sources configuration
        self.data_sources = self.config.get("data_sources", {})
        
        logger.info("Business metrics collector initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database for metrics storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS business_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metric_timestamp ON business_metrics(metric_name, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category_timestamp ON business_metrics(category, timestamp)")
    
    def record_metric(self, metric: BusinessMetric):
        """Record a business metric"""
        self.metrics_history.append(metric)
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO business_metrics 
                (metric_name, category, value, unit, timestamp, metadata, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.metric_name,
                metric.category.value,
                metric.value,
                metric.unit,
                metric.timestamp.isoformat(),
                json.dumps(metric.metadata),
                json.dumps(metric.tags)
            ))
    
    def record_usage_metrics(self, generation_data: Dict[str, Any]):
        """Record usage-related metrics"""
        timestamp = datetime.now(timezone.utc)
        
        # API requests
        self.record_metric(BusinessMetric(
            metric_name="api_requests",
            category=MetricCategory.USAGE,
            value=1,
            unit="count",
            timestamp=timestamp,
            tags={"model": generation_data.get("model", "unknown")}
        ))
        
        # Tokens generated
        if "tokens_generated" in generation_data:
            self.record_metric(BusinessMetric(
                metric_name="tokens_generated",
                category=MetricCategory.USAGE,
                value=generation_data["tokens_generated"],
                unit="tokens",
                timestamp=timestamp,
                tags={"model": generation_data.get("model", "unknown")}
            ))
        
        # Active users (unique user tracking)
        if "user_id" in generation_data:
            self.record_metric(BusinessMetric(
                metric_name="active_users",
                category=MetricCategory.USER_ENGAGEMENT,
                value=1,
                unit="count",
                timestamp=timestamp,
                metadata={"user_id": generation_data["user_id"]}
            ))
    
    def record_performance_metrics(self, performance_data: Dict[str, Any]):
        """Record performance-related metrics"""
        timestamp = datetime.now(timezone.utc)
        
        # Response time
        if "response_time" in performance_data:
            self.record_metric(BusinessMetric(
                metric_name="response_time",
                category=MetricCategory.PERFORMANCE,
                value=performance_data["response_time"],
                unit="seconds",
                timestamp=timestamp
            ))
        
        # Throughput
        if "throughput" in performance_data:
            self.record_metric(BusinessMetric(
                metric_name="throughput",
                category=MetricCategory.PERFORMANCE,
                value=performance_data["throughput"],
                unit="requests_per_second",
                timestamp=timestamp
            ))
        
        # Error rate
        if "errors" in performance_data:
            self.record_metric(BusinessMetric(
                metric_name="error_rate",
                category=MetricCategory.PERFORMANCE,
                value=performance_data["errors"],
                unit="percentage",
                timestamp=timestamp
            ))
    
    def record_quality_metrics(self, quality_data: Dict[str, Any]):
        """Record quality-related metrics"""
        timestamp = datetime.now(timezone.utc)
        
        quality_metrics = ["bleu_score", "rouge_score", "semantic_similarity", "fluency_score"]
        
        for metric_name in quality_metrics:
            if metric_name in quality_data:
                self.record_metric(BusinessMetric(
                    metric_name=metric_name,
                    category=MetricCategory.QUALITY,
                    value=quality_data[metric_name],
                    unit="score",
                    timestamp=timestamp
                ))
    
    def record_cost_metrics(self, cost_data: Dict[str, Any]):
        """Record cost-related metrics"""
        timestamp = datetime.now(timezone.utc)
        
        # Total cost
        if "total_cost" in cost_data:
            self.record_metric(BusinessMetric(
                metric_name="total_cost",
                category=MetricCategory.COST,
                value=cost_data["total_cost"],
                unit="usd",
                timestamp=timestamp
            ))
        
        # Cost per request
        if "cost_per_request" in cost_data:
            self.record_metric(BusinessMetric(
                metric_name="cost_per_request",
                category=MetricCategory.COST,
                value=cost_data["cost_per_request"],
                unit="usd",
                timestamp=timestamp
            ))
        
        # Resource utilization costs
        if "resource_costs" in cost_data:
            for resource, cost in cost_data["resource_costs"].items():
                self.record_metric(BusinessMetric(
                    metric_name=f"{resource}_cost",
                    category=MetricCategory.COST,
                    value=cost,
                    unit="usd",
                    timestamp=timestamp,
                    tags={"resource_type": resource}
                ))
    
    def record_safety_metrics(self, safety_data: Dict[str, Any]):
        """Record safety-related metrics"""
        timestamp = datetime.now(timezone.utc)
        
        # Safety violations
        if "violations" in safety_data:
            for violation_type, count in safety_data["violations"].items():
                self.record_metric(BusinessMetric(
                    metric_name=f"{violation_type}_violations",
                    category=MetricCategory.SAFETY,
                    value=count,
                    unit="count",
                    timestamp=timestamp
                ))
        
        # Overall safety score
        if "safety_score" in safety_data:
            self.record_metric(BusinessMetric(
                metric_name="safety_score",
                category=MetricCategory.SAFETY,
                value=safety_data["safety_score"],
                unit="score",
                timestamp=timestamp
            ))
    
    def get_metric_aggregation(self, metric_name: str, 
                              start_time: datetime, end_time: datetime,
                              granularity: TimeGranularity = TimeGranularity.DAY,
                              aggregation_function: str = "sum") -> List[Dict[str, Any]]:
        """Get aggregated metrics for time series analysis"""
        with sqlite3.connect(self.db_path) as conn:
            # Query metrics in time range
            query = """
                SELECT value, timestamp, metadata, tags
                FROM business_metrics
                WHERE metric_name = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            
            cursor = conn.execute(query, (
                metric_name,
                start_time.isoformat(),
                end_time.isoformat()
            ))
            
            results = []
            for row in cursor:
                results.append({
                    "value": row[0],
                    "timestamp": datetime.fromisoformat(row[1]),
                    "metadata": json.loads(row[2]) if row[2] else {},
                    "tags": json.loads(row[3]) if row[3] else {}
                })
        
        if not results:
            return []
        
        # Group by time period
        grouped_data = defaultdict(list)
        
        for result in results:
            timestamp = result["timestamp"]
            
            if granularity == TimeGranularity.HOUR:
                period_key = timestamp.replace(minute=0, second=0, microsecond=0)
            elif granularity == TimeGranularity.DAY:
                period_key = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            elif granularity == TimeGranularity.WEEK:
                days_since_monday = timestamp.weekday()
                period_key = timestamp.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
            elif granularity == TimeGranularity.MONTH:
                period_key = timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                period_key = timestamp
            
            grouped_data[period_key].append(result["value"])
        
        # Apply aggregation function
        aggregated_results = []
        for period, values in grouped_data.items():
            if aggregation_function == "sum":
                aggregated_value = sum(values)
            elif aggregation_function == "avg":
                aggregated_value = statistics.mean(values)
            elif aggregation_function == "max":
                aggregated_value = max(values)
            elif aggregation_function == "min":
                aggregated_value = min(values)
            elif aggregation_function == "count":
                aggregated_value = len(values)
            else:
                aggregated_value = sum(values)  # Default to sum
            
            aggregated_results.append({
                "timestamp": period,
                "value": aggregated_value,
                "sample_count": len(values)
            })
        
        return sorted(aggregated_results, key=lambda x: x["timestamp"])


class BusinessDashboard:
    """Interactive business metrics dashboard"""
    
    def __init__(self, metrics_collector: BusinessMetricsCollector, 
                 config: Optional[Dict[str, Any]] = None):
        self.metrics_collector = metrics_collector
        self.config = config or {}
        self.widgets: List[DashboardWidget] = []
        
        # Initialize default widgets
        self._initialize_default_widgets()
        
        logger.info("Business dashboard initialized")
    
    def _initialize_default_widgets(self):
        """Initialize default dashboard widgets"""
        default_widgets = [
            DashboardWidget(
                widget_id="api_requests_chart",
                title="API Requests Over Time",
                widget_type="chart",
                metrics=["api_requests"],
                config={"chart_type": "line", "time_range": "24h"}
            ),
            DashboardWidget(
                widget_id="response_time_kpi",
                title="Average Response Time",
                widget_type="kpi",
                metrics=["response_time"],
                config={"aggregation": "avg", "format": "ms"}
            ),
            DashboardWidget(
                widget_id="error_rate_alert",
                title="Error Rate Monitor",
                widget_type="alert",
                metrics=["error_rate"],
                config={"threshold": 5.0, "comparison": "greater_than"}
            ),
            DashboardWidget(
                widget_id="cost_breakdown_table",
                title="Cost Breakdown",
                widget_type="table",
                metrics=["cpu_cost", "gpu_cost", "memory_cost", "storage_cost"],
                config={"time_range": "30d", "group_by": "resource_type"}
            ),
            DashboardWidget(
                widget_id="quality_metrics_chart",
                title="Quality Metrics Trends",
                widget_type="chart",
                metrics=["bleu_score", "rouge_score", "fluency_score"],
                config={"chart_type": "multi_line", "time_range": "7d"}
            ),
            DashboardWidget(
                widget_id="safety_violations_chart",
                title="Safety Violations",
                widget_type="chart",
                metrics=["toxicity_violations", "bias_violations", "pii_violations"],
                config={"chart_type": "stacked_bar", "time_range": "30d"}
            )
        ]
        
        self.widgets.extend(default_widgets)
    
    async def get_dashboard_data(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get all dashboard data"""
        end_time = datetime.now(timezone.utc)
        
        # Parse time range
        if time_range.endswith("h"):
            hours = int(time_range[:-1])
            start_time = end_time - timedelta(hours=hours)
            granularity = TimeGranularity.HOUR
        elif time_range.endswith("d"):
            days = int(time_range[:-1])
            start_time = end_time - timedelta(days=days)
            granularity = TimeGranularity.DAY
        elif time_range.endswith("w"):
            weeks = int(time_range[:-1])
            start_time = end_time - timedelta(weeks=weeks)
            granularity = TimeGranularity.DAY
        elif time_range.endswith("m"):
            months = int(time_range[:-1])
            start_time = end_time - timedelta(days=months*30)  # Approximate
            granularity = TimeGranularity.DAY
        else:
            start_time = end_time - timedelta(hours=24)
            granularity = TimeGranularity.HOUR
        
        dashboard_data = {
            "timestamp": end_time.isoformat(),
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "granularity": granularity.value
            },
            "widgets": []
        }
        
        # Get data for each widget
        for widget in self.widgets:
            widget_data = await self._get_widget_data(widget, start_time, end_time, granularity)
            dashboard_data["widgets"].append(widget_data)
        
        return dashboard_data
    
    async def _get_widget_data(self, widget: DashboardWidget, 
                              start_time: datetime, end_time: datetime,
                              granularity: TimeGranularity) -> Dict[str, Any]:
        """Get data for a specific widget"""
        widget_data = {
            "widget_id": widget.widget_id,
            "title": widget.title,
            "type": widget.widget_type,
            "config": widget.config,
            "data": {}
        }
        
        if widget.widget_type == "kpi":
            # Single value metric
            metric_name = widget.metrics[0]
            aggregation = widget.config.get("aggregation", "avg")
            
            aggregated_data = self.metrics_collector.get_metric_aggregation(
                metric_name, start_time, end_time, granularity, aggregation
            )
            
            if aggregated_data:
                latest_value = aggregated_data[-1]["value"]
                # Calculate change from previous period
                if len(aggregated_data) > 1:
                    previous_value = aggregated_data[-2]["value"]
                    change = ((latest_value - previous_value) / previous_value) * 100 if previous_value != 0 else 0
                else:
                    change = 0
                
                widget_data["data"] = {
                    "value": latest_value,
                    "change": change,
                    "trend": "up" if change > 0 else "down" if change < 0 else "stable"
                }
            else:
                widget_data["data"] = {"value": 0, "change": 0, "trend": "stable"}
        
        elif widget.widget_type == "chart":
            # Time series data
            chart_data = {}
            
            for metric_name in widget.metrics:
                aggregation = widget.config.get("aggregation", "sum")
                aggregated_data = self.metrics_collector.get_metric_aggregation(
                    metric_name, start_time, end_time, granularity, aggregation
                )
                
                chart_data[metric_name] = {
                    "timestamps": [d["timestamp"].isoformat() for d in aggregated_data],
                    "values": [d["value"] for d in aggregated_data]
                }
            
            widget_data["data"] = chart_data
        
        elif widget.widget_type == "table":
            # Tabular data
            table_data = []
            
            for metric_name in widget.metrics:
                aggregated_data = self.metrics_collector.get_metric_aggregation(
                    metric_name, start_time, end_time, granularity, "sum"
                )
                
                total_value = sum(d["value"] for d in aggregated_data)
                table_data.append({
                    "metric": metric_name,
                    "value": total_value,
                    "unit": "varies"  # Would be retrieved from metric definition
                })
            
            widget_data["data"] = {"rows": table_data}
        
        elif widget.widget_type == "alert":
            # Alert status
            metric_name = widget.metrics[0]
            threshold = widget.config.get("threshold", 0)
            comparison = widget.config.get("comparison", "greater_than")
            
            # Get latest value
            aggregated_data = self.metrics_collector.get_metric_aggregation(
                metric_name, start_time, end_time, TimeGranularity.HOUR, "avg"
            )
            
            alert_status = "normal"
            if aggregated_data:
                latest_value = aggregated_data[-1]["value"]
                
                if comparison == "greater_than" and latest_value > threshold:
                    alert_status = "warning"
                elif comparison == "less_than" and latest_value < threshold:
                    alert_status = "warning"
            
            widget_data["data"] = {
                "status": alert_status,
                "value": aggregated_data[-1]["value"] if aggregated_data else 0,
                "threshold": threshold
            }
        
        return widget_data
    
    def add_widget(self, widget: DashboardWidget):
        """Add a widget to the dashboard"""
        self.widgets.append(widget)
    
    def remove_widget(self, widget_id: str):
        """Remove a widget from the dashboard"""
        self.widgets = [w for w in self.widgets if w.widget_id != widget_id]
    
    def get_widget_config(self, widget_id: str) -> Optional[DashboardWidget]:
        """Get widget configuration"""
        for widget in self.widgets:
            if widget.widget_id == widget_id:
                return widget
        return None


class BusinessReportGenerator:
    """Generates comprehensive business intelligence reports"""
    
    def __init__(self, metrics_collector: BusinessMetricsCollector, 
                 config: Optional[Dict[str, Any]] = None):
        self.metrics_collector = metrics_collector
        self.config = config or {}
        
        logger.info("Business report generator initialized")
    
    async def generate_executive_report(self, 
                                      start_time: datetime, 
                                      end_time: datetime) -> BusinessReport:
        """Generate executive summary report"""
        report_id = f"executive_{int(datetime.now().timestamp())}"
        
        # Calculate key metrics
        executive_summary = await self._calculate_executive_summary(start_time, end_time)
        detailed_metrics = await self._calculate_detailed_metrics(start_time, end_time)
        insights = await self._generate_insights(detailed_metrics)
        recommendations = await self._generate_recommendations(detailed_metrics, insights)
        
        return BusinessReport(
            report_id=report_id,
            title="Executive Business Report",
            generated_at=datetime.now(timezone.utc),
            time_period={"start": start_time, "end": end_time},
            executive_summary=executive_summary,
            detailed_metrics=detailed_metrics,
            insights=insights,
            recommendations=recommendations,
            data_sources=["business_metrics", "performance_data", "cost_data"]
        )
    
    async def _calculate_executive_summary(self, 
                                         start_time: datetime, 
                                         end_time: datetime) -> Dict[str, Any]:
        """Calculate executive summary metrics"""
        summary = {}
        
        # Usage metrics
        api_requests = self.metrics_collector.get_metric_aggregation(
            "api_requests", start_time, end_time, TimeGranularity.DAY, "sum"
        )
        total_requests = sum(d["value"] for d in api_requests)
        
        tokens_generated = self.metrics_collector.get_metric_aggregation(
            "tokens_generated", start_time, end_time, TimeGranularity.DAY, "sum"
        )
        total_tokens = sum(d["value"] for d in tokens_generated)
        
        # Performance metrics
        response_times = self.metrics_collector.get_metric_aggregation(
            "response_time", start_time, end_time, TimeGranularity.DAY, "avg"
        )
        avg_response_time = statistics.mean([d["value"] for d in response_times]) if response_times else 0
        
        # Cost metrics
        costs = self.metrics_collector.get_metric_aggregation(
            "total_cost", start_time, end_time, TimeGranularity.DAY, "sum"
        )
        total_cost = sum(d["value"] for d in costs)
        
        # Quality metrics
        quality_scores = self.metrics_collector.get_metric_aggregation(
            "bleu_score", start_time, end_time, TimeGranularity.DAY, "avg"
        )
        avg_quality = statistics.mean([d["value"] for d in quality_scores]) if quality_scores else 0
        
        summary = {
            "usage": {
                "total_api_requests": total_requests,
                "total_tokens_generated": total_tokens,
                "requests_per_day": total_requests / max((end_time - start_time).days, 1)
            },
            "performance": {
                "average_response_time": avg_response_time,
                "uptime_percentage": 99.5  # Would be calculated from actual uptime data
            },
            "financial": {
                "total_cost": total_cost,
                "cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
                "cost_per_token": total_cost / total_tokens if total_tokens > 0 else 0
            },
            "quality": {
                "average_quality_score": avg_quality,
                "quality_trend": "stable"  # Would be calculated from trend analysis
            }
        }
        
        return summary
    
    async def _calculate_detailed_metrics(self, 
                                        start_time: datetime, 
                                        end_time: datetime) -> Dict[str, Any]:
        """Calculate detailed metrics breakdown"""
        detailed = {
            "usage_trends": {},
            "performance_breakdown": {},
            "cost_analysis": {},
            "quality_metrics": {},
            "safety_metrics": {}
        }
        
        # Usage trends
        daily_requests = self.metrics_collector.get_metric_aggregation(
            "api_requests", start_time, end_time, TimeGranularity.DAY, "sum"
        )
        detailed["usage_trends"]["daily_requests"] = daily_requests
        
        # Performance breakdown
        hourly_response_times = self.metrics_collector.get_metric_aggregation(
            "response_time", start_time, end_time, TimeGranularity.HOUR, "avg"
        )
        detailed["performance_breakdown"]["response_times"] = hourly_response_times
        
        # Cost analysis by resource type
        for resource_type in ["cpu_cost", "gpu_cost", "memory_cost", "storage_cost"]:
            resource_costs = self.metrics_collector.get_metric_aggregation(
                resource_type, start_time, end_time, TimeGranularity.DAY, "sum"
            )
            detailed["cost_analysis"][resource_type] = resource_costs
        
        # Quality metrics
        for quality_metric in ["bleu_score", "rouge_score", "fluency_score"]:
            quality_data = self.metrics_collector.get_metric_aggregation(
                quality_metric, start_time, end_time, TimeGranularity.DAY, "avg"
            )
            detailed["quality_metrics"][quality_metric] = quality_data
        
        # Safety metrics
        for safety_metric in ["toxicity_violations", "bias_violations", "pii_violations"]:
            safety_data = self.metrics_collector.get_metric_aggregation(
                safety_metric, start_time, end_time, TimeGranularity.DAY, "sum"
            )
            detailed["safety_metrics"][safety_metric] = safety_data
        
        return detailed
    
    async def _generate_insights(self, detailed_metrics: Dict[str, Any]) -> List[str]:
        """Generate business insights from metrics"""
        insights = []
        
        # Usage patterns
        usage_data = detailed_metrics.get("usage_trends", {}).get("daily_requests", [])
        if len(usage_data) > 7:
            recent_avg = statistics.mean([d["value"] for d in usage_data[-7:]])
            previous_avg = statistics.mean([d["value"] for d in usage_data[-14:-7]])
            
            if recent_avg > previous_avg * 1.1:
                insights.append("API usage has increased by more than 10% in the last week")
            elif recent_avg < previous_avg * 0.9:
                insights.append("API usage has decreased by more than 10% in the last week")
        
        # Cost trends
        cost_data = detailed_metrics.get("cost_analysis", {})
        total_cost_trend = []
        for resource_costs in cost_data.values():
            if resource_costs:
                total_cost_trend.extend([d["value"] for d in resource_costs])
        
        if len(total_cost_trend) > 14:
            recent_cost = sum(total_cost_trend[-7:])
            previous_cost = sum(total_cost_trend[-14:-7])
            
            if recent_cost > previous_cost * 1.15:
                insights.append("Infrastructure costs have increased significantly (>15%) in the last week")
        
        # Quality trends
        quality_data = detailed_metrics.get("quality_metrics", {})
        for metric_name, metric_data in quality_data.items():
            if len(metric_data) > 7:
                recent_quality = statistics.mean([d["value"] for d in metric_data[-7:]])
                if recent_quality < 0.7:
                    insights.append(f"{metric_name} has dropped below acceptable threshold (0.7)")
        
        # Safety violations
        safety_data = detailed_metrics.get("safety_metrics", {})
        for violation_type, violation_data in safety_data.items():
            total_violations = sum(d["value"] for d in violation_data)
            if total_violations > 0:
                insights.append(f"Detected {total_violations} {violation_type} in the reporting period")
        
        return insights
    
    async def _generate_recommendations(self, detailed_metrics: Dict[str, Any], 
                                       insights: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Cost optimization
        cost_data = detailed_metrics.get("cost_analysis", {})
        if cost_data:
            # Find highest cost resource
            total_costs = {}
            for resource, cost_data_points in cost_data.items():
                total_costs[resource] = sum(d["value"] for d in cost_data_points)
            
            if total_costs:
                highest_cost_resource = max(total_costs.items(), key=lambda x: x[1])
                recommendations.append(
                    f"Consider optimizing {highest_cost_resource[0]} usage to reduce costs "
                    f"(${highest_cost_resource[1]:.2f} total)"
                )
        
        # Performance optimization
        performance_data = detailed_metrics.get("performance_breakdown", {}).get("response_times", [])
        if performance_data:
            avg_response_time = statistics.mean([d["value"] for d in performance_data])
            if avg_response_time > 2.0:  # 2 seconds threshold
                recommendations.append(
                    "Response times are above optimal threshold. Consider implementing caching "
                    "or model optimization strategies"
                )
        
        # Quality improvements
        quality_data = detailed_metrics.get("quality_metrics", {})
        low_quality_metrics = []
        for metric_name, metric_values in quality_data.items():
            if metric_values:
                avg_quality = statistics.mean([d["value"] for d in metric_values])
                if avg_quality < 0.8:
                    low_quality_metrics.append(metric_name)
        
        if low_quality_metrics:
            recommendations.append(
                f"Quality metrics {', '.join(low_quality_metrics)} are below target. "
                "Consider model fine-tuning or prompt optimization"
            )
        
        # Safety improvements
        safety_data = detailed_metrics.get("safety_metrics", {})
        violation_counts = {}
        for violation_type, violation_data in safety_data.items():
            total = sum(d["value"] for d in violation_data)
            if total > 0:
                violation_counts[violation_type] = total
        
        if violation_counts:
            recommendations.append(
                "Implement stricter content filtering or model fine-tuning to reduce safety violations"
            )
        
        return recommendations
    
    def export_report(self, report: BusinessReport, format: str = "json") -> str:
        """Export business report in specified format"""
        if format.lower() == "json":
            report_dict = {
                "report_id": report.report_id,
                "title": report.title,
                "generated_at": report.generated_at.isoformat(),
                "time_period": {
                    "start": report.time_period["start"].isoformat(),
                    "end": report.time_period["end"].isoformat()
                },
                "executive_summary": report.executive_summary,
                "detailed_metrics": report.detailed_metrics,
                "insights": report.insights,
                "recommendations": report.recommendations,
                "data_sources": report.data_sources
            }
            return json.dumps(report_dict, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Example usage
if __name__ == "__main__":
    async def example():
        # Initialize business metrics system
        collector = BusinessMetricsCollector({"db_path": "example_business.db"})
        dashboard = BusinessDashboard(collector)
        report_generator = BusinessReportGenerator(collector)
        
        # Simulate some business metrics
        import random
        
        for i in range(100):
            # Usage metrics
            collector.record_usage_metrics({
                "tokens_generated": random.randint(50, 500),
                "model": random.choice(["gpt2", "gpt-j", "llama"]),
                "user_id": f"user_{random.randint(1, 20)}"
            })
            
            # Performance metrics
            collector.record_performance_metrics({
                "response_time": random.uniform(0.5, 3.0),
                "throughput": random.uniform(10, 50),
                "errors": random.uniform(0, 5)
            })
            
            # Cost metrics
            collector.record_cost_metrics({
                "total_cost": random.uniform(0.01, 0.50),
                "cost_per_request": random.uniform(0.001, 0.05),
                "resource_costs": {
                    "cpu": random.uniform(0.005, 0.02),
                    "gpu": random.uniform(0.01, 0.30),
                    "memory": random.uniform(0.001, 0.01)
                }
            })
        
        # Get dashboard data
        dashboard_data = await dashboard.get_dashboard_data("24h")
        print("Dashboard Data:")
        print(f"Widgets: {len(dashboard_data['widgets'])}")
        
        # Generate business report
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=7)
        
        report = await report_generator.generate_executive_report(start_time, end_time)
        
        print(f"\nBusiness Report: {report.title}")
        print(f"Insights: {len(report.insights)}")
        print(f"Recommendations: {len(report.recommendations)}")
        
        for insight in report.insights:
            print(f"- {insight}")
        
        for recommendation in report.recommendations:
            print(f"* {recommendation}")
        
        print("\nBusiness metrics system example completed")
    
    # Run example
    # asyncio.run(example())