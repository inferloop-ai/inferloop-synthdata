# audio_synth/monitoring/alerting.py
"""
Alerting system for audio synthesis monitoring
"""

import smtplib
import time
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum
import requests
import json
import logging

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    metadata: Dict[str, Any]
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "resolved": self.resolved
        }

class AlertRule:
    """Base class for alert rules"""
    
    def __init__(self, 
                 name: str, 
                 severity: AlertSeverity,
                 cooldown_minutes: int = 15):
        self.name = name
        self.severity = severity
        self.cooldown_minutes = cooldown_minutes
        self.last_triggered = 0
        
    def should_trigger(self, metrics: Dict[str, Any]) -> bool:
        """Check if this rule should trigger an alert"""
        raise NotImplementedError
    
    def get_message(self, metrics: Dict[str, Any]) -> str:
        """Get alert message"""
        raise NotImplementedError
    
    def can_trigger(self) -> bool:
        """Check if enough time has passed since last trigger"""
        return time.time() - self.last_triggered > (self.cooldown_minutes * 60)

class HighErrorRateRule(AlertRule):
    """Alert when error rate exceeds threshold"""
    
    def __init__(self, threshold: float = 0.1, **kwargs):
        super().__init__("high_error_rate", AlertSeverity.ERROR, **kwargs)
        self.threshold = threshold
    
    def should_trigger(self, metrics: Dict[str, Any]) -> bool:
        success_rate = metrics.get("success_rate", 1.0)
        error_rate = 1.0 - success_rate
        return error_rate > self.threshold and metrics.get("requests_last_hour", 0) > 10
    
    def get_message(self, metrics: Dict[str, Any]) -> str:
        error_rate = 1.0 - metrics.get("success_rate", 1.0)
        return f"High error rate detected: {error_rate:.1%} (threshold: {self.threshold:.1%})"

class SlowResponseRule(AlertRule):
    """Alert when response time is too slow"""
    
    def __init__(self, threshold_seconds: float = 30.0, **kwargs):
        super().__init__("slow_response", AlertSeverity.WARNING, **kwargs)
        self.threshold_seconds = threshold_seconds
    
    def should_trigger(self, metrics: Dict[str, Any]) -> bool:
        avg_duration = metrics.get("avg_duration", 0.0)
        return avg_duration > self.threshold_seconds and metrics.get("requests_last_hour", 0) > 5
    
    def get_message(self, metrics: Dict[str, Any]) -> str:
        avg_duration = metrics.get("avg_duration", 0.0)
        return f"Slow response time: {avg_duration:.1f}s (threshold: {self.threshold_seconds}s)"

class HighMemoryUsageRule(AlertRule):
    """Alert when memory usage is high"""
    
    def __init__(self, threshold_percent: float = 85.0, **kwargs):
        super().__init__("high_memory_usage", AlertSeverity.WARNING, **kwargs)
        self.threshold_percent = threshold_percent
    
    def should_trigger(self, metrics: Dict[str, Any]) -> bool:
        system = metrics.get("system", {})
        memory_percent = system.get("memory_percent", 0.0)
        return memory_percent > self.threshold_percent
    
    def get_message(self, metrics: Dict[str, Any]) -> str:
        system = metrics.get("system", {})
        memory_percent = system.get("memory_percent", 0.0)
        return f"High memory usage: {memory_percent:.1f}% (threshold: {self.threshold_percent}%)"

class LowQualityRule(AlertRule):
    """Alert when quality scores are consistently low"""
    
    def __init__(self, threshold: float = 0.7, **kwargs):
        super().__init__("low_quality", AlertSeverity.WARNING, **kwargs)
        self.threshold = threshold
    
    def should_trigger(self, metrics: Dict[str, Any]) -> bool:
        # This would need access to quality analysis data
        # For now, just a placeholder
        return False
    
    def get_message(self, metrics: Dict[str, Any]) -> str:
        return f"Quality scores below threshold: {self.threshold}"

class AlertManager:
    """Manages alert rules and notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels = []
        
        # Setup notification channels
        self._setup_notification_channels()
        
        # Setup default rules
        self._setup_default_rules()
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Alert manager initialized")
    
    def _setup_notification_channels(self):
        """Setup notification channels"""
        
        channels_config = self.config.get("notification_channels", {})
        
        # Email notifications
        if "email" in channels_config:
            email_config = channels_config["email"]
            self.notification_channels.append(
                EmailNotifier(
                    smtp_server=email_config["smtp_server"],
                    smtp_port=email_config["smtp_port"],
                    username=email_config["username"],
                    password=email_config["password"],
                    recipients=email_config["recipients"]
                )
            )
        
        # Slack notifications
        if "slack" in channels_config:
            slack_config = channels_config["slack"]
            self.notification_channels.append(
                SlackNotifier(webhook_url=slack_config["webhook_url"])
            )
        
        # PagerDuty notifications
        if "pagerduty" in channels_config:
            pd_config = channels_config["pagerduty"]
            self.notification_channels.append(
                PagerDutyNotifier(integration_key=pd_config["integration_key"])
            )
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        
        rules_config = self.config.get("alert_rules", {})
        
        # High error rate
        if rules_config.get("high_error_rate", {}).get("enabled", True):
            threshold = rules_config.get("high_error_rate", {}).get("threshold", 0.1)
            self.add_rule(HighErrorRateRule(threshold=threshold))
        
        # Slow response
        if rules_config.get("slow_response", {}).get("enabled", True):
            threshold = rules_config.get("slow_response", {}).get("threshold_seconds", 30.0)
            self.add_rule(SlowResponseRule(threshold_seconds=threshold))
        
        # High memory usage
        if rules_config.get("high_memory_usage", {}).get("enabled", True):
            threshold = rules_config.get("high_memory_usage", {}).get("threshold_percent", 85.0)
            self.add_rule(HighMemoryUsageRule(threshold_percent=threshold))
        
        # Low quality
        if rules_config.get("low_quality", {}).get("enabled", False):
            threshold = rules_config.get("low_quality", {}).get("threshold", 0.7)
            self.add_rule(LowQualityRule(threshold=threshold))
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all rules against current metrics"""
        
        for rule in self.rules:
            if rule.should_trigger(metrics) and rule.can_trigger():
                # Create alert
                alert = Alert(
                    name=rule.name,
                    severity=rule.severity,
                    message=rule.get_message(metrics),
                    timestamp=time.time(),
                    metadata={"metrics": metrics}
                )
                
                # Add to active alerts
                self.active_alerts[rule.name] = alert
                self.alert_history.append(alert)
                
                # Update rule trigger time
                rule.last_triggered = time.time()
                
                # Send notifications
                self._send_alert(alert)
                
                logger.warning(f"Alert triggered: {rule.name} - {alert.message}")
    
    def resolve_alert(self, alert_name: str):
        """Manually resolve an alert"""
        if alert_name in self.active_alerts:
            alert = self.active_alerts[alert_name]
            alert.resolved = True
            del self.active_alerts[alert_name]
            
            # Send resolution notification
            self._send_resolution(alert)
            
            logger.info(f"Alert resolved: {alert_name}")
    
    def _send_alert(self, alert: Alert):
        """Send alert to all notification channels"""
        for channel in self.notification_channels:
            try:
                channel.send_alert(alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {type(channel).__name__}: {e}")
    
    def _send_resolution(self, alert: Alert):
        """Send resolution notification"""
        for channel in self.notification_channels:
            try:
                if hasattr(channel, 'send_resolution'):
                    channel.send_resolution(alert)
            except Exception as e:
                logger.error(f"Failed to send resolution via {type(channel).__name__}: {e}")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        # This would be connected to the MetricsCollector
        # For now, just a placeholder
        pass
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history"""
        cutoff_time = time.time() - (hours * 3600)
        recent_alerts = [alert for alert in self.alert_history 
                        if alert.timestamp >= cutoff_time]
        return [alert.to_dict() for alert in recent_alerts]

class EmailNotifier:
    """Email notification channel"""
    
    def __init__(self, smtp_server: str, smtp_port: int, 
                 username: str, password: str, recipients: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients
    
    def send_alert(self, alert: Alert):
        """Send alert via email"""
        
        subject = f"[{alert.severity.value.upper()}] Audio Synthesis Alert: {alert.name}"
        
        body = f"""
Alert: {alert.name}
Severity: {alert.severity.value.upper()}
Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}

Message: {alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}

---
Audio Synthesis Monitoring System
        """
        
        msg = MIMEMultipart()
        msg['From'] = self.username
        msg['To'] = ', '.join(self.recipients)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.username, self.recipients, msg.as_string())
            
            logger.info(f"Email alert sent: {alert.name}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

class SlackNotifier:
    """Slack notification channel"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_alert(self, alert: Alert):
        """Send alert via Slack"""
        
        color_map = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "danger",
            AlertSeverity.CRITICAL: "danger"
        }
        
        payload = {
            "text": f"Audio Synthesis Alert: {alert.name}",
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "danger"),
                    "fields": [
                        {
                            "title": "Severity",
                            "value": alert.severity.value.upper(),
                            "short": True
                        },
                        {
                            "title": "Time",
                            "value": time.strftime('%Y-%m-%d %H:%M:%S', 
                                               time.localtime(alert.timestamp)),
                            "short": True
                        },
                        {
                            "title": "Message",
                            "value": alert.message,
                            "short": False
                        }
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(self.webhook_url, 
                                   data=json.dumps(payload),
                                   headers={'Content-Type': 'application/json'})
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {alert.name}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

class PagerDutyNotifier:
    """PagerDuty notification channel"""
    
    def __init__(self, integration_key: str):
        self.integration_key = integration_key
        self.api_url = "https://events.pagerduty.com/v2/enqueue"
    
    def send_alert(self, alert: Alert):
        """Send alert via PagerDuty"""
        
        severity_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning", 
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical"
        }
        
        payload = {
            "routing_key": self.integration_key,
            "event_action": "trigger",
            "dedup_key": f"audio_synthesis_{alert.name}",
            "payload": {
                "summary": f"Audio Synthesis Alert: {alert.name}",
                "severity": severity_map.get(alert.severity, "error"),
                "source": "audio_synthesis_monitoring",
                "component": "audio_generator",
                "group": "production",
                "class": "alert",
                "custom_details": {
                    "message": alert.message,
                    "metadata": alert.metadata
                }
            }
        }
        
        try:
            response = requests.post(self.api_url,
                                   data=json.dumps(payload),
                                   headers={'Content-Type': 'application/json'})
            response.raise_for_status()
            
            logger.info(f"PagerDuty alert sent: {alert.name}")
            
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")

# ============================================================================
