"""
System monitoring and health check functionality.
Provides comprehensive monitoring of system components and operational visibility.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from ..utils.logging_config import get_logger
from ..utils.exceptions import StockPredictorError


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check result data model."""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    response_time_ms: Optional[float] = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: HealthStatus
    checks: List[HealthCheck]
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.overall_status == HealthStatus.HEALTHY
    
    @property
    def critical_issues(self) -> List[HealthCheck]:
        """Get list of critical health issues."""
        return [check for check in self.checks if check.status == HealthStatus.CRITICAL]
    
    @property
    def warnings(self) -> List[HealthCheck]:
        """Get list of warning health issues."""
        return [check for check in self.checks if check.status == HealthStatus.WARNING]


class SystemMonitor:
    """
    System monitoring and health checking service.
    Provides comprehensive monitoring of system components and resources.
    """
    
    def __init__(self, check_interval: float = 60.0, memory_threshold_gb: float = 8.0):
        """
        Initialize system monitor.
        
        Args:
            check_interval: Interval between health checks in seconds
            memory_threshold_gb: Memory usage threshold in GB for warnings
        """
        self.check_interval = check_interval
        self.memory_threshold_gb = memory_threshold_gb
        self.logger = get_logger("SystemMonitor")
        
        # Health check registry
        self._health_checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._last_health_status: Optional[SystemHealth] = None
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self) -> None:
        """Register default system health checks."""
        self.register_health_check("memory_usage", self._check_memory_usage)
        self.register_health_check("disk_space", self._check_disk_space)
        self.register_health_check("cpu_usage", self._check_cpu_usage)
        self.register_health_check("system_load", self._check_system_load)
    
    def register_health_check(self, name: str, check_func: Callable[[], HealthCheck]) -> None:
        """
        Register a health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns a HealthCheck result
        """
        self._health_checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    def unregister_health_check(self, name: str) -> None:
        """
        Unregister a health check.
        
        Args:
            name: Name of the health check to remove
        """
        if name in self._health_checks:
            del self._health_checks[name]
            self.logger.info(f"Unregistered health check: {name}")
    
    def run_health_checks(self) -> SystemHealth:
        """
        Run all registered health checks and return system health status.
        
        Returns:
            SystemHealth object with overall status and individual check results
        """
        checks = []
        start_time = time.time()
        
        for name, check_func in self._health_checks.items():
            try:
                check_start = time.time()
                check_result = check_func()
                check_result.response_time_ms = (time.time() - check_start) * 1000
                checks.append(check_result)
                
            except Exception as e:
                self.logger.error(f"Health check {name} failed: {str(e)}")
                checks.append(HealthCheck(
                    component=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    response_time_ms=(time.time() - check_start) * 1000
                ))
        
        # Determine overall status
        overall_status = self._determine_overall_status(checks)
        
        system_health = SystemHealth(
            overall_status=overall_status,
            checks=checks
        )
        
        self._last_health_status = system_health
        
        total_time = (time.time() - start_time) * 1000
        self.logger.debug(f"Health checks completed in {total_time:.2f}ms, status: {overall_status.value}")
        
        return system_health
    
    def _determine_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """
        Determine overall system health status from individual checks.
        
        Args:
            checks: List of individual health check results
            
        Returns:
            Overall health status
        """
        if not checks:
            return HealthStatus.UNKNOWN
        
        # If any check is critical, system is critical
        if any(check.status == HealthStatus.CRITICAL for check in checks):
            return HealthStatus.CRITICAL
        
        # If any check has warnings, system has warnings
        if any(check.status == HealthStatus.WARNING for check in checks):
            return HealthStatus.WARNING
        
        # All checks are healthy
        return HealthStatus.HEALTHY
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring in background thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self.logger.warning("Monitoring is already running")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        self.logger.info(f"Started system monitoring with {self.check_interval}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5.0)
            self.logger.info("Stopped system monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        while not self._stop_monitoring.is_set():
            try:
                health_status = self.run_health_checks()
                
                # Log critical issues
                if health_status.critical_issues:
                    for issue in health_status.critical_issues:
                        self.logger.critical(f"Critical health issue in {issue.component}: {issue.message}")
                
                # Log warnings
                if health_status.warnings:
                    for warning in health_status.warnings:
                        self.logger.warning(f"Health warning in {warning.component}: {warning.message}")
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
            
            # Wait for next check or stop signal
            self._stop_monitoring.wait(self.check_interval)
    
    def get_last_health_status(self) -> Optional[SystemHealth]:
        """Get the last health check results."""
        return self._last_health_status
    
    def _check_memory_usage(self) -> HealthCheck:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            memory_gb = memory.used / (1024**3)
            memory_percent = memory.percent
            
            if memory_gb > self.memory_threshold_gb:
                status = HealthStatus.WARNING if memory_percent < 90 else HealthStatus.CRITICAL
                message = f"High memory usage: {memory_gb:.2f}GB ({memory_percent:.1f}%)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_gb:.2f}GB ({memory_percent:.1f}%)"
            
            return HealthCheck(
                component="memory_usage",
                status=status,
                message=message,
                details={
                    "used_gb": memory_gb,
                    "percent": memory_percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                component="memory_usage",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check memory usage: {str(e)}"
            )
    
    def _check_disk_space(self) -> HealthCheck:
        """Check available disk space."""
        try:
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            percent_used = (disk.used / disk.total) * 100
            
            if percent_used > 90:
                status = HealthStatus.CRITICAL
                message = f"Critical disk space: {free_gb:.2f}GB free ({100-percent_used:.1f}% available)"
            elif percent_used > 80:
                status = HealthStatus.WARNING
                message = f"Low disk space: {free_gb:.2f}GB free ({100-percent_used:.1f}% available)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space normal: {free_gb:.2f}GB free ({100-percent_used:.1f}% available)"
            
            return HealthCheck(
                component="disk_space",
                status=status,
                message=message,
                details={
                    "free_gb": free_gb,
                    "used_percent": percent_used,
                    "total_gb": disk.total / (1024**3)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                component="disk_space",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check disk space: {str(e)}"
            )
    
    def _check_cpu_usage(self) -> HealthCheck:
        """Check CPU usage."""
        try:
            # Get CPU usage over 1 second interval
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Critical CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent > 80:
                status = HealthStatus.WARNING
                message = f"High CPU usage: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return HealthCheck(
                component="cpu_usage",
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "cpu_count": psutil.cpu_count()
                }
            )
            
        except Exception as e:
            return HealthCheck(
                component="cpu_usage",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check CPU usage: {str(e)}"
            )
    
    def _check_system_load(self) -> HealthCheck:
        """Check system load average."""
        try:
            load_avg = psutil.getloadavg()
            cpu_count = psutil.cpu_count()
            load_ratio = load_avg[0] / cpu_count if cpu_count > 0 else 0
            
            if load_ratio > 2.0:
                status = HealthStatus.CRITICAL
                message = f"Critical system load: {load_avg[0]:.2f} (ratio: {load_ratio:.2f})"
            elif load_ratio > 1.5:
                status = HealthStatus.WARNING
                message = f"High system load: {load_avg[0]:.2f} (ratio: {load_ratio:.2f})"
            else:
                status = HealthStatus.HEALTHY
                message = f"System load normal: {load_avg[0]:.2f} (ratio: {load_ratio:.2f})"
            
            return HealthCheck(
                component="system_load",
                status=status,
                message=message,
                details={
                    "load_1min": load_avg[0],
                    "load_5min": load_avg[1],
                    "load_15min": load_avg[2],
                    "cpu_count": cpu_count,
                    "load_ratio": load_ratio
                }
            )
            
        except Exception as e:
            return HealthCheck(
                component="system_load",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system load: {str(e)}"
            )