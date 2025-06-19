#!/usr/bin/env python3
"""
Alert manager for system notifications and alerting.

Provides comprehensive alerting capabilities including email notifications,
Slack integration, webhook delivery, and alert escalation management.
"""

import asyncio
import json
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, asdict

import aiohttp
from pydantic import BaseModel, Field

from ...core import get_logger, BaseConfig

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


class AlertStatus(str, Enum):
    """Alert status"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    title: str
    message: str
    severity: AlertSeverity
    source: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    escalation_level: int = 0
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        if self.acknowledged_at:
            data['acknowledged_at'] = self.acknowledged_at.isoformat()
        return data


class AlertManagerConfig(BaseConfig):
    """Alert manager configuration"""
    email_enabled: bool = False
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    from_email: str = "alerts@system.local"
    
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = "#alerts"
    
    webhook_enabled: bool = False
    webhook_urls: List[str] = Field(default_factory=list)
    
    escalation_enabled: bool = True
    escalation_delay_minutes: int = 30
    max_escalation_level: int = 3
    
    alert_retention_days: int = 30
    batch_size: int = 10
    send_interval_seconds: int = 60


class AlertManager:
    """Comprehensive alert management system"""
    
    def __init__(self, config: AlertManagerConfig):
        self.config = config
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_handlers: Dict[str, Callable] = {}
        self.running = False
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup notification handlers"""
        if self.config.email_enabled:
            self.notification_handlers['email'] = self._send_email_alert
        if self.config.slack_enabled:
            self.notification_handlers['slack'] = self._send_slack_alert
        if self.config.webhook_enabled:
            self.notification_handlers['webhook'] = self._send_webhook_alert
    
    async def start(self):
        """Start alert manager"""
        self.running = True
        logger.info("Alert manager started")
        
        # Start background tasks
        asyncio.create_task(self._escalation_worker())
        asyncio.create_task(self._cleanup_worker())
        
    async def stop(self):
        """Stop alert manager"""
        self.running = False
        logger.info("Alert manager stopped")
    
    async def create_alert(self, title: str, message: str, 
                         severity: AlertSeverity, source: str,
                         metadata: Optional[Dict[str, Any]] = None,
                         tags: Optional[List[str]] = None) -> Alert:
        """
        Create new alert.
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity level
            source: Alert source system
            metadata: Additional alert metadata
            tags: Alert tags
        
        Returns:
            Created alert
        """
        try:
            alert_id = f"{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.active_alerts)}"
            
            alert = Alert(
                id=alert_id,
                title=title,
                message=message,
                severity=severity,
                source=source,
                timestamp=datetime.now(),
                metadata=metadata or {},
                tags=tags or []
            )
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Send notifications
            await self._send_notifications(alert)
            
            logger.info(f"Created alert: {alert_id} - {title}")
            return alert
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            raise
    
    async def resolve_alert(self, alert_id: str, 
                          resolution_message: Optional[str] = None) -> bool:
        """
        Resolve an active alert.
        
        Args:
            alert_id: Alert ID to resolve
            resolution_message: Optional resolution message
        
        Returns:
            True if alert was resolved
        """
        try:
            if alert_id not in self.active_alerts:
                logger.warning(f"Alert not found: {alert_id}")
                return False
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            
            if resolution_message:
                alert.metadata['resolution_message'] = resolution_message
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            # Send resolution notification
            await self._send_resolution_notification(alert)
            
            logger.info(f"Resolved alert: {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False
    
    async def acknowledge_alert(self, alert_id: str, 
                              acknowledger: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID to acknowledge
            acknowledger: Person acknowledging the alert
        
        Returns:
            True if alert was acknowledged
        """
        try:
            if alert_id not in self.active_alerts:
                logger.warning(f"Alert not found: {alert_id}")
                return False
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.metadata['acknowledger'] = acknowledger
            
            logger.info(f"Acknowledged alert: {alert_id} by {acknowledger}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            return False
    
    async def get_active_alerts(self, 
                              severity: Optional[AlertSeverity] = None,
                              source: Optional[str] = None) -> List[Alert]:
        """
        Get active alerts with optional filtering.
        
        Args:
            severity: Filter by severity
            source: Filter by source
        
        Returns:
            List of matching active alerts
        """
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if source:
            alerts = [a for a in alerts if a.source == source]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    async def get_alert_history(self, 
                              days: int = 7,
                              severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """
        Get alert history.
        
        Args:
            days: Number of days to look back
            severity: Filter by severity
        
        Returns:
            List of historical alerts
        """
        cutoff = datetime.now() - timedelta(days=days)
        alerts = [a for a in self.alert_history if a.timestamp >= cutoff]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    async def get_alert_stats(self) -> Dict[str, Any]:
        """
        Get alert statistics.
        
        Returns:
            Alert statistics dictionary
        """
        active_count = len(self.active_alerts)
        
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                a for a in self.active_alerts.values() 
                if a.severity == severity
            ])
        
        source_counts = {}
        for alert in self.active_alerts.values():
            source_counts[alert.source] = source_counts.get(alert.source, 0) + 1
        
        # Historical stats (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        recent_alerts = [a for a in self.alert_history if a.timestamp >= cutoff]
        
        return {
            'active_alerts': active_count,
            'severity_breakdown': severity_counts,
            'source_breakdown': source_counts,
            'alerts_last_24h': len(recent_alerts),
            'total_historical': len(self.alert_history)
        }
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications for alert"""
        try:
            for handler_name, handler in self.notification_handlers.items():
                try:
                    await handler(alert)
                except Exception as e:
                    logger.error(f"Failed to send {handler_name} notification: {e}")
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.from_email
            msg['To'] = ", ".join(alert.metadata.get('email_recipients', []))
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
            Alert Details:
            
            Title: {alert.title}
            Severity: {alert.severity.value.upper()}
            Source: {alert.source}
            Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            Message:
            {alert.message}
            
            Metadata:
            {json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                if self.config.smtp_username:
                    server.starttls()
                    server.login(self.config.smtp_username, self.config.smtp_password)
                
                server.send_message(msg)
            
            logger.debug(f"Sent email alert for {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert notification"""
        try:
            color_map = {
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.WARNING: "warning", 
                AlertSeverity.INFO: "good",
                AlertSeverity.DEBUG: "#808080"
            }
            
            payload = {
                "channel": self.config.slack_channel,
                "attachments": [{
                    "color": color_map.get(alert.severity, "good"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                    ]
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.config.slack_webhook_url, 
                                      json=payload) as response:
                    if response.status == 200:
                        logger.debug(f"Sent Slack alert for {alert.id}")
                    else:
                        logger.error(f"Failed to send Slack alert: {response.status}")
        
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert notification"""
        try:
            payload = {
                "alert": alert.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                for webhook_url in self.config.webhook_urls:
                    try:
                        async with session.post(webhook_url, json=payload) as response:
                            if response.status == 200:
                                logger.debug(f"Sent webhook alert to {webhook_url}")
                            else:
                                logger.warning(f"Webhook alert failed: {response.status}")
                    except Exception as e:
                        logger.error(f"Failed to send webhook to {webhook_url}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to send webhook alerts: {e}")
    
    async def _send_resolution_notification(self, alert: Alert):
        """Send alert resolution notification"""
        try:
            resolution_alert = Alert(
                id=f"{alert.id}_resolved",
                title=f"RESOLVED: {alert.title}",
                message=f"Alert {alert.id} has been resolved.",
                severity=AlertSeverity.INFO,
                source=alert.source,
                timestamp=datetime.now(),
                metadata={'original_alert': alert.to_dict()}
            )
            
            await self._send_notifications(resolution_alert)
            
        except Exception as e:
            logger.error(f"Failed to send resolution notification: {e}")
    
    async def _escalation_worker(self):
        """Background worker for alert escalation"""
        while self.running:
            try:
                if self.config.escalation_enabled:
                    await self._process_escalations()
                await asyncio.sleep(self.config.escalation_delay_minutes * 60)
            except Exception as e:
                logger.error(f"Error in escalation worker: {e}")
    
    async def _process_escalations(self):
        """Process alert escalations"""
        try:
            cutoff = datetime.now() - timedelta(minutes=self.config.escalation_delay_minutes)
            
            for alert in self.active_alerts.values():
                if (alert.status == AlertStatus.ACTIVE and 
                    alert.timestamp <= cutoff and
                    alert.escalation_level < self.config.max_escalation_level):
                    
                    alert.escalation_level += 1
                    
                    # Create escalation alert
                    escalation_alert = Alert(
                        id=f"{alert.id}_escalation_{alert.escalation_level}",
                        title=f"ESCALATION #{alert.escalation_level}: {alert.title}",
                        message=f"Alert {alert.id} has been escalated (level {alert.escalation_level})",
                        severity=alert.severity,
                        source=alert.source,
                        timestamp=datetime.now(),
                        metadata={'original_alert': alert.to_dict()}
                    )
                    
                    await self._send_notifications(escalation_alert)
                    logger.info(f"Escalated alert {alert.id} to level {alert.escalation_level}")
        
        except Exception as e:
            logger.error(f"Failed to process escalations: {e}")
    
    async def _cleanup_worker(self):
        """Background worker for alert cleanup"""
        while self.running:
            try:
                await self._cleanup_old_alerts()
                await asyncio.sleep(24 * 3600)  # Run daily
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    async def _cleanup_old_alerts(self):
        """Clean up old alerts from history"""
        try:
            cutoff = datetime.now() - timedelta(days=self.config.alert_retention_days)
            
            original_count = len(self.alert_history)
            self.alert_history = [a for a in self.alert_history if a.timestamp >= cutoff]
            
            cleaned_count = original_count - len(self.alert_history)
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old alerts")
        
        except Exception as e:
            logger.error(f"Failed to cleanup old alerts: {e}")


def create_alert_manager(config: Optional[AlertManagerConfig] = None) -> AlertManager:
    """Factory function to create alert manager"""
    if config is None:
        config = AlertManagerConfig()
    return AlertManager(config)


__all__ = ['AlertManager', 'AlertManagerConfig', 'Alert', 'AlertSeverity', 'AlertStatus', 'create_alert_manager']