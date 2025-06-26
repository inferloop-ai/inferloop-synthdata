"""
TextNLP Metrics Module

Comprehensive metrics collection and analysis for text generation:
- Generation Metrics (performance, quality, resource usage)
- Quality Metrics (BLEU, ROUGE, semantic similarity, etc.)
- Resource Utilization Tracking
- Business Intelligence Dashboard
"""

from .generation_metrics import MetricsCollector, GenerationMetrics, MetricType, MetricValue
from .quality_metrics import QualityMetricsCalculator, QualityEvaluation, QualityMetricType, QualityScore
from .resource_tracker import ResourceTracker, ResourceUtilization, ResourceType, ResourceAlert
from .business_dashboard import BusinessMetricsCollector, BusinessDashboard, BusinessReportGenerator, BusinessMetric

__version__ = "1.0.0"
__all__ = [
    # Generation Metrics
    "MetricsCollector", "GenerationMetrics", "MetricType", "MetricValue",
    
    # Quality Metrics
    "QualityMetricsCalculator", "QualityEvaluation", "QualityMetricType", "QualityScore",
    
    # Resource Tracking
    "ResourceTracker", "ResourceUtilization", "ResourceType", "ResourceAlert",
    
    # Business Dashboard
    "BusinessMetricsCollector", "BusinessDashboard", "BusinessReportGenerator", "BusinessMetric"
]