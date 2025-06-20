#!/usr/bin/env python3
"""
Orchestration package for managing workflows, scheduling, and monitoring.

Provides comprehensive orchestration capabilities including job scheduling,
workflow management, system monitoring, and performance tracking.
"""

from .monitoring import (
    create_alert_manager,
    create_performance_monitor,
    create_metrics_collector,
    create_health_checker
)
from .scheduling import (
    create_cron_manager,
    create_job_scheduler,
    create_event_scheduler
)
from .workflow import (
    create_custom_orchestrator,
    create_pipeline_manager
)

__version__ = "1.0.0"

__all__ = [
    # Monitoring
    'create_alert_manager',
    'create_performance_monitor', 
    'create_metrics_collector',
    'create_health_checker',
    
    # Scheduling
    'create_cron_manager',
    'create_job_scheduler',
    'create_event_scheduler',
    
    # Workflow
    'create_custom_orchestrator',
    'create_pipeline_manager'
]