#!/usr/bin/env python3
"""
Health checker for monitoring system and service health.

Provides comprehensive health checking capabilities including service
health monitoring, dependency checks, and automated recovery actions.
"""

import asyncio
import aiohttp
import psutil
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum

from pydantic import BaseModel, Field

from ...core import get_logger, BaseConfig

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckType(str, Enum):
    """Health check types"""
    HTTP = "http"
    TCP = "tcp"
    PROCESS = "process"
    DATABASE = "database"
    DISK_SPACE = "disk_space"
    MEMORY = "memory"
    CPU = "cpu"
    CUSTOM = "custom"


@dataclass
class HealthCheckResult:
    """Health check result"""
    check_name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class HealthCheck:
    """Health check configuration"""
    name: str
    check_type: CheckType
    config: Dict[str, Any]
    interval_seconds: int = 60
    timeout_seconds: int = 30
    retries: int = 3
    enabled: bool = True
    critical: bool = False  # Whether failure affects overall health
    recovery_action: Optional[str] = None


class HealthCheckerConfig(BaseConfig):
    """Health checker configuration"""
    check_interval_seconds: int = 30
    result_retention_hours: int = 24
    
    # Global thresholds
    cpu_threshold_percent: float = 90.0
    memory_threshold_percent: float = 90.0
    disk_threshold_percent: float = 90.0
    
    # Service discovery
    enable_service_discovery: bool = False
    service_discovery_interval: int = 300
    
    # Recovery actions
    enable_auto_recovery: bool = False
    recovery_cooldown_minutes: int = 10
    
    # Notifications
    enable_notifications: bool = True
    notification_channels: List[str] = Field(default_factory=list)


class HealthChecker:
    """Comprehensive system and service health monitoring"""
    
    def __init__(self, config: HealthCheckerConfig):
        self.config = config
        self.health_checks: Dict[str, HealthCheck] = {}
        self.check_results: Dict[str, List[HealthCheckResult]] = {}
        self.overall_status = HealthStatus.UNKNOWN
        self.running = False
        self.notification_callbacks: List[Callable] = []
        self.recovery_last_run: Dict[str, datetime] = {}
        
        # Setup default health checks
        self._setup_default_checks()
    
    def _setup_default_checks(self):
        """Setup default system health checks"""
        # System resource checks
        self.add_health_check(HealthCheck(
            name="system_cpu",
            check_type=CheckType.CPU,
            config={"threshold_percent": self.config.cpu_threshold_percent},
            interval_seconds=30,
            critical=True
        ))
        
        self.add_health_check(HealthCheck(
            name="system_memory",
            check_type=CheckType.MEMORY,
            config={"threshold_percent": self.config.memory_threshold_percent},
            interval_seconds=30,
            critical=True
        ))
        
        self.add_health_check(HealthCheck(
            name="system_disk",
            check_type=CheckType.DISK_SPACE,
            config={
                "path": "/",
                "threshold_percent": self.config.disk_threshold_percent
            },
            interval_seconds=60,
            critical=True
        ))
    
    async def start(self):
        """Start health checker"""
        self.running = True
        logger.info("Health checker started")
        
        # Start background tasks
        asyncio.create_task(self._health_check_worker())
        asyncio.create_task(self._cleanup_worker())
        
        if self.config.enable_service_discovery:
            asyncio.create_task(self._service_discovery_worker())
    
    async def stop(self):
        """Stop health checker"""
        self.running = False
        logger.info("Health checker stopped")
    
    def add_health_check(self, health_check: HealthCheck):
        """
        Add health check.
        
        Args:
            health_check: Health check configuration
        """
        self.health_checks[health_check.name] = health_check
        self.check_results[health_check.name] = []
        logger.info(f"Added health check: {health_check.name}")
    
    def remove_health_check(self, check_name: str) -> bool:
        """
        Remove health check.
        
        Args:
            check_name: Name of check to remove
        
        Returns:
            True if check was removed
        """
        if check_name in self.health_checks:
            del self.health_checks[check_name]
            del self.check_results[check_name]
            logger.info(f"Removed health check: {check_name}")
            return True
        return False
    
    def add_notification_callback(self, callback: Callable[[HealthCheckResult], None]):
        """
        Add notification callback.
        
        Args:
            callback: Function to call on health status changes
        """
        self.notification_callbacks.append(callback)
    
    async def run_health_check(self, check_name: str) -> HealthCheckResult:
        """
        Run specific health check.
        
        Args:
            check_name: Name of check to run
        
        Returns:
            Health check result
        """
        if check_name not in self.health_checks:
            raise ValueError(f"Health check '{check_name}' not found")
        
        health_check = self.health_checks[check_name]
        
        if not health_check.enabled:
            return HealthCheckResult(
                check_name=check_name,
                status=HealthStatus.UNKNOWN,
                message="Check disabled",
                timestamp=datetime.now()
            )
        
        start_time = datetime.now()
        
        try:
            result = await self._execute_health_check(health_check)
            
        except Exception as e:
            logger.error(f"Health check {check_name} failed: {e}")
            result = HealthCheckResult(
                check_name=check_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check execution failed: {str(e)}",
                timestamp=datetime.now()
            )
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        result.response_time_ms = response_time
        
        # Store result
        self.check_results[check_name].append(result)
        
        # Trigger notifications
        await self._notify_status_change(result)
        
        # Trigger recovery if needed
        if (result.status == HealthStatus.UNHEALTHY and 
            health_check.recovery_action and 
            self.config.enable_auto_recovery):
            await self._trigger_recovery(health_check, result)
        
        return result
    
    async def _execute_health_check(self, health_check: HealthCheck) -> HealthCheckResult:
        """Execute health check based on type"""
        if health_check.check_type == CheckType.HTTP:
            return await self._check_http(health_check)
        elif health_check.check_type == CheckType.TCP:
            return await self._check_tcp(health_check)
        elif health_check.check_type == CheckType.PROCESS:
            return await self._check_process(health_check)
        elif health_check.check_type == CheckType.DATABASE:
            return await self._check_database(health_check)
        elif health_check.check_type == CheckType.DISK_SPACE:
            return await self._check_disk_space(health_check)
        elif health_check.check_type == CheckType.MEMORY:
            return await self._check_memory(health_check)
        elif health_check.check_type == CheckType.CPU:
            return await self._check_cpu(health_check)
        elif health_check.check_type == CheckType.CUSTOM:
            return await self._check_custom(health_check)
        else:
            raise ValueError(f"Unknown check type: {health_check.check_type}")
    
    async def _check_http(self, health_check: HealthCheck) -> HealthCheckResult:
        """Perform HTTP health check"""
        config = health_check.config
        url = config['url']
        expected_status = config.get('expected_status', 200)
        expected_content = config.get('expected_content')
        
        try:
            timeout = aiohttp.ClientTimeout(total=health_check.timeout_seconds)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == expected_status:
                        if expected_content:
                            content = await response.text()
                            if expected_content in content:
                                status = HealthStatus.HEALTHY
                                message = f"HTTP check passed: {response.status}"
                            else:
                                status = HealthStatus.UNHEALTHY
                                message = f"Expected content not found"
                        else:
                            status = HealthStatus.HEALTHY
                            message = f"HTTP check passed: {response.status}"
                    else:
                        status = HealthStatus.UNHEALTHY
                        message = f"Unexpected status: {response.status} (expected {expected_status})"
                    
                    return HealthCheckResult(
                        check_name=health_check.name,
                        status=status,
                        message=message,
                        timestamp=datetime.now(),
                        metadata={'status_code': response.status}
                    )
        
        except Exception as e:
            return HealthCheckResult(
                check_name=health_check.name,
                status=HealthStatus.UNHEALTHY,
                message=f"HTTP check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _check_tcp(self, health_check: HealthCheck) -> HealthCheckResult:
        """Perform TCP health check"""
        config = health_check.config
        host = config['host']
        port = config['port']
        
        try:
            future = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(
                future, timeout=health_check.timeout_seconds
            )
            writer.close()
            await writer.wait_closed()
            
            return HealthCheckResult(
                check_name=health_check.name,
                status=HealthStatus.HEALTHY,
                message=f"TCP connection successful to {host}:{port}",
                timestamp=datetime.now()
            )
        
        except Exception as e:
            return HealthCheckResult(
                check_name=health_check.name,
                status=HealthStatus.UNHEALTHY,
                message=f"TCP connection failed to {host}:{port}: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _check_process(self, health_check: HealthCheck) -> HealthCheckResult:
        """Perform process health check"""
        config = health_check.config
        process_name = config['process_name']
        min_instances = config.get('min_instances', 1)
        
        try:
            running_processes = []
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if process_name.lower() in proc.info['name'].lower():
                        running_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if len(running_processes) >= min_instances:
                status = HealthStatus.HEALTHY
                message = f"Found {len(running_processes)} instances of {process_name}"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Only {len(running_processes)} instances running (minimum: {min_instances})"
            
            return HealthCheckResult(
                check_name=health_check.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                metadata={'running_instances': len(running_processes), 'processes': running_processes}
            )
        
        except Exception as e:
            return HealthCheckResult(
                check_name=health_check.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Process check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _check_database(self, health_check: HealthCheck) -> HealthCheckResult:
        """Perform database health check"""
        # This is a placeholder - in real implementation, you'd connect to specific databases
        config = health_check.config
        db_type = config.get('type', 'postgresql')
        
        return HealthCheckResult(
            check_name=health_check.name,
            status=HealthStatus.HEALTHY,
            message=f"Database {db_type} check passed",
            timestamp=datetime.now()
        )
    
    async def _check_disk_space(self, health_check: HealthCheck) -> HealthCheckResult:
        """Perform disk space health check"""
        config = health_check.config
        path = config.get('path', '/')
        threshold_percent = config.get('threshold_percent', 90.0)
        
        try:
            disk_usage = psutil.disk_usage(path)
            used_percent = (disk_usage.used / disk_usage.total) * 100
            
            if used_percent < threshold_percent:
                status = HealthStatus.HEALTHY
                message = f"Disk usage {used_percent:.1f}% (threshold: {threshold_percent}%)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Disk usage {used_percent:.1f}% exceeds threshold {threshold_percent}%"
            
            return HealthCheckResult(
                check_name=health_check.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                metadata={
                    'used_percent': used_percent,
                    'used_gb': disk_usage.used / (1024**3),
                    'free_gb': disk_usage.free / (1024**3),
                    'total_gb': disk_usage.total / (1024**3)
                }
            )
        
        except Exception as e:
            return HealthCheckResult(
                check_name=health_check.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Disk space check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _check_memory(self, health_check: HealthCheck) -> HealthCheckResult:
        """Perform memory health check"""
        config = health_check.config
        threshold_percent = config.get('threshold_percent', 90.0)
        
        try:
            memory = psutil.virtual_memory()
            used_percent = memory.percent
            
            if used_percent < threshold_percent:
                status = HealthStatus.HEALTHY
                message = f"Memory usage {used_percent:.1f}% (threshold: {threshold_percent}%)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage {used_percent:.1f}% exceeds threshold {threshold_percent}%"
            
            return HealthCheckResult(
                check_name=health_check.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                metadata={
                    'used_percent': used_percent,
                    'used_gb': memory.used / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'total_gb': memory.total / (1024**3)
                }
            )
        
        except Exception as e:
            return HealthCheckResult(
                check_name=health_check.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _check_cpu(self, health_check: HealthCheck) -> HealthCheckResult:
        """Perform CPU health check"""
        config = health_check.config
        threshold_percent = config.get('threshold_percent', 90.0)
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent < threshold_percent:
                status = HealthStatus.HEALTHY
                message = f"CPU usage {cpu_percent:.1f}% (threshold: {threshold_percent}%)"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"CPU usage {cpu_percent:.1f}% exceeds threshold {threshold_percent}%"
            
            return HealthCheckResult(
                check_name=health_check.name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                metadata={'cpu_percent': cpu_percent}
            )
        
        except Exception as e:
            return HealthCheckResult(
                check_name=health_check.name,
                status=HealthStatus.UNHEALTHY,
                message=f"CPU check failed: {str(e)}",
                timestamp=datetime.now()
            )
    
    async def _check_custom(self, health_check: HealthCheck) -> HealthCheckResult:
        """Perform custom health check"""
        # This would allow custom check functions to be registered
        config = health_check.config
        check_function = config.get('function')
        
        if check_function and callable(check_function):
            try:
                result = await check_function(health_check)
                return result
            except Exception as e:
                return HealthCheckResult(
                    check_name=health_check.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Custom check failed: {str(e)}",
                    timestamp=datetime.now()
                )
        
        return HealthCheckResult(
            check_name=health_check.name,
            status=HealthStatus.UNKNOWN,
            message="No custom check function defined",
            timestamp=datetime.now()
        )
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Overall health summary
        """
        try:
            critical_checks = []
            all_results = []
            
            for check_name, health_check in self.health_checks.items():
                if check_name in self.check_results and self.check_results[check_name]:
                    latest_result = self.check_results[check_name][-1]
                    all_results.append(latest_result)
                    
                    if health_check.critical:
                        critical_checks.append(latest_result)
            
            # Determine overall status
            if not all_results:
                overall_status = HealthStatus.UNKNOWN
            elif any(r.status == HealthStatus.UNHEALTHY for r in critical_checks):
                overall_status = HealthStatus.UNHEALTHY
            elif any(r.status == HealthStatus.DEGRADED for r in all_results):
                overall_status = HealthStatus.DEGRADED
            elif all(r.status == HealthStatus.HEALTHY for r in all_results):
                overall_status = HealthStatus.HEALTHY
            else:
                overall_status = HealthStatus.UNKNOWN
            
            self.overall_status = overall_status
            
            # Status counts
            status_counts = {}
            for status in HealthStatus:
                status_counts[status.value] = len([
                    r for r in all_results if r.status == status
                ])
            
            return {
                'overall_status': overall_status.value,
                'timestamp': datetime.now().isoformat(),
                'total_checks': len(all_results),
                'critical_checks': len(critical_checks),
                'status_counts': status_counts,
                'checks': [r.to_dict() for r in all_results]
            }
            
        except Exception as e:
            logger.error(f"Failed to get overall health: {e}")
            return {
                'overall_status': HealthStatus.UNKNOWN.value,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _notify_status_change(self, result: HealthCheckResult):
        """Notify on status changes"""
        try:
            if self.config.enable_notifications:
                for callback in self.notification_callbacks:
                    try:
                        await callback(result)
                    except Exception as e:
                        logger.error(f"Error in notification callback: {e}")
        
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")
    
    async def _trigger_recovery(self, health_check: HealthCheck, result: HealthCheckResult):
        """Trigger recovery action"""
        try:
            # Check cooldown
            if health_check.name in self.recovery_last_run:
                last_run = self.recovery_last_run[health_check.name]
                cooldown = timedelta(minutes=self.config.recovery_cooldown_minutes)
                if datetime.now() - last_run < cooldown:
                    logger.debug(f"Recovery action for {health_check.name} in cooldown")
                    return
            
            # Execute recovery action
            recovery_action = health_check.recovery_action
            logger.info(f"Triggering recovery action for {health_check.name}: {recovery_action}")
            
            # This is a placeholder - in real implementation, you'd execute the recovery action
            # Examples: restart service, clear cache, send alert, etc.
            
            self.recovery_last_run[health_check.name] = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to trigger recovery for {health_check.name}: {e}")
    
    async def _health_check_worker(self):
        """Background worker for running health checks"""
        while self.running:
            try:
                # Run all enabled health checks
                tasks = []
                for check_name, health_check in self.health_checks.items():
                    if health_check.enabled:
                        tasks.append(self.run_health_check(check_name))
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(self.config.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in health check worker: {e}")
    
    async def _service_discovery_worker(self):
        """Background worker for service discovery"""
        while self.running:
            try:
                # This would discover new services and add health checks
                await asyncio.sleep(self.config.service_discovery_interval)
            except Exception as e:
                logger.error(f"Error in service discovery worker: {e}")
    
    async def _cleanup_worker(self):
        """Background worker for cleanup"""
        while self.running:
            try:
                await self._cleanup_old_results()
                await asyncio.sleep(3600)  # Cleanup hourly
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    async def _cleanup_old_results(self):
        """Clean up old health check results"""
        try:
            cutoff = datetime.now() - timedelta(hours=self.config.result_retention_hours)
            
            for check_name, results in self.check_results.items():
                original_count = len(results)
                self.check_results[check_name] = [
                    r for r in results if r.timestamp >= cutoff
                ]
                cleaned_count = original_count - len(self.check_results[check_name])
                
                if cleaned_count > 0:
                    logger.debug(f"Cleaned {cleaned_count} old results for {check_name}")
        
        except Exception as e:
            logger.error(f"Failed to cleanup old results: {e}")


def create_health_checker(config: Optional[HealthCheckerConfig] = None) -> HealthChecker:
    """Factory function to create health checker"""
    if config is None:
        config = HealthCheckerConfig()
    return HealthChecker(config)


__all__ = [
    'HealthChecker',
    'HealthCheckerConfig',
    'HealthCheck',
    'HealthCheckResult',
    'HealthStatus',
    'CheckType',
    'create_health_checker'
]