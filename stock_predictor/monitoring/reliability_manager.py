"""
Reliability manager that coordinates all monitoring and reliability components.
Provides unified interface for system reliability and monitoring features.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from ..utils.logging_config import get_logger
from ..utils.exceptions import StockPredictorError
from ..config import get_config
from .system_monitor import SystemMonitor, SystemHealth, HealthStatus
from .metrics_collector import MetricsCollector, SystemMetrics
from .model_versioning import ModelVersionManager
from .concurrent_processor import ConcurrentStockProcessor


@dataclass
class ReliabilityStatus:
    """Overall system reliability status."""
    timestamp: datetime
    system_health: SystemHealth
    system_metrics: SystemMetrics
    processor_stats: Dict[str, Any]
    model_versions_count: int
    is_reliable: bool
    issues: List[str]


class ReliabilityManager:
    """
    Unified reliability manager for the Stock Direction Predictor system.
    Coordinates monitoring, metrics, versioning, and concurrent processing.
    """
    
    def __init__(self):
        """Initialize reliability manager with all components."""
        self.logger = get_logger("ReliabilityManager")
        self.config = get_config()
        
        # Initialize components
        self.system_monitor = SystemMonitor(
            check_interval=60.0,
            memory_threshold_gb=self.config.system.memory_limit_gb
        )
        
        self.metrics_collector = MetricsCollector(
            retention_hours=24,
            collection_interval=30.0
        )
        
        self.model_version_manager = ModelVersionManager(
            base_path=self.config.system.model_save_path,
            max_versions=10
        )
        
        self.concurrent_processor = ConcurrentStockProcessor(
            max_workers=self.config.system.max_workers,
            queue_size=100,
            timeout_seconds=300.0,
            metrics_collector=self.metrics_collector
        )
        
        # Register custom health checks
        self._register_custom_health_checks()
        
        # Register custom metrics collectors
        self._register_custom_metrics_collectors()
        
        # State tracking
        self._is_running = False
        self._shutdown_event = threading.Event()
    
    def start(self) -> None:
        """Start all reliability and monitoring components."""
        if self._is_running:
            self.logger.warning("Reliability manager is already running")
            return
        
        try:
            # Start system monitoring
            self.system_monitor.start_monitoring()
            
            # Start metrics collection
            self.metrics_collector.start_collection()
            
            # Start concurrent processor
            self.concurrent_processor.start()
            
            self._is_running = True
            self._shutdown_event.clear()
            
            self.logger.info("Started reliability manager with all components")
            
            # Record startup metrics
            self.metrics_collector.increment_counter("reliability_manager.starts")
            self.metrics_collector.set_gauge("reliability_manager.status", 1)
            
        except Exception as e:
            self.logger.error(f"Failed to start reliability manager: {str(e)}")
            self.stop()  # Clean up any partially started components
            raise
    
    def stop(self) -> None:
        """Stop all reliability and monitoring components."""
        if not self._is_running:
            return
        
        self._shutdown_event.set()
        
        try:
            # Stop concurrent processor
            self.concurrent_processor.stop(wait=True, timeout=30.0)
            
            # Stop metrics collection
            self.metrics_collector.stop_collection()
            
            # Stop system monitoring
            self.system_monitor.stop_monitoring()
            
            self._is_running = False
            
            self.logger.info("Stopped reliability manager")
            
            # Record shutdown metrics
            self.metrics_collector.increment_counter("reliability_manager.stops")
            self.metrics_collector.set_gauge("reliability_manager.status", 0)
            
        except Exception as e:
            self.logger.error(f"Error during reliability manager shutdown: {str(e)}")
    
    def get_reliability_status(self) -> ReliabilityStatus:
        """
        Get comprehensive reliability status.
        
        Returns:
            ReliabilityStatus object with all system status information
        """
        timestamp = datetime.now()
        
        # Get system health
        system_health = self.system_monitor.run_health_checks()
        
        # Get system metrics
        system_metrics = self.metrics_collector.collect_metrics()
        
        # Get processor stats
        processor_stats = self.concurrent_processor.get_processing_stats()
        
        # Count model versions
        model_versions_count = self._count_model_versions()
        
        # Determine overall reliability
        is_reliable, issues = self._assess_reliability(system_health, system_metrics, processor_stats)
        
        return ReliabilityStatus(
            timestamp=timestamp,
            system_health=system_health,
            system_metrics=system_metrics,
            processor_stats=processor_stats,
            model_versions_count=model_versions_count,
            is_reliable=is_reliable,
            issues=issues
        )
    
    def register_task_processor(self, task_type: str, processor_func: Callable) -> None:
        """
        Register a task processor for concurrent processing.
        
        Args:
            task_type: Type of task to process
            processor_func: Function that processes the task
        """
        self.concurrent_processor.register_task_processor(task_type, processor_func)
    
    def submit_analysis_task(
        self,
        symbol: str,
        task_type: str,
        parameters: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """
        Submit a stock analysis task for concurrent processing.
        
        Args:
            symbol: Stock symbol to analyze
            task_type: Type of analysis to perform
            parameters: Analysis parameters
            priority: Task priority
            
        Returns:
            Task ID for tracking
        """
        return self.concurrent_processor.submit_task(
            symbol=symbol,
            task_type=task_type,
            parameters=parameters,
            priority=priority
        )
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None):
        """Get the result of a submitted task."""
        return self.concurrent_processor.get_task_result(task_id, timeout)
    
    def wait_for_tasks(self, task_ids: List[str], timeout: Optional[float] = None):
        """Wait for multiple tasks to complete."""
        return self.concurrent_processor.wait_for_tasks(task_ids, timeout)
    
    def save_model_version(self, model, model_config, performance_metrics=None, metadata=None) -> str:
        """Save a model with versioning."""
        return self.model_version_manager.save_model(
            model, model_config, performance_metrics, metadata
        )
    
    def load_model_version(self, model_type: str, pattern_length: int, version_id: Optional[str] = None):
        """Load a model version."""
        return self.model_version_manager.load_model(model_type, pattern_length, version_id)
    
    def rollback_model(self, model_type: str, pattern_length: int, version_id: str) -> None:
        """Rollback to a previous model version."""
        self.model_version_manager.rollback_to_version(model_type, pattern_length, version_id)
    
    def get_model_versions(self, model_type: str, pattern_length: int):
        """Get all versions for a model."""
        return self.model_version_manager.list_versions(model_type, pattern_length)
    
    def record_metric(self, name: str, value: float, metric_type: str = "gauge", tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a custom metric.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric (counter, gauge, histogram, timer)
            tags: Optional tags
        """
        if metric_type == "counter":
            self.metrics_collector.increment_counter(name, value, tags)
        elif metric_type == "gauge":
            self.metrics_collector.set_gauge(name, value, tags)
        elif metric_type == "histogram":
            self.metrics_collector.record_histogram(name, value, tags)
        elif metric_type == "timer":
            self.metrics_collector.record_timer(name, value, tags)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
    
    def get_metrics_history(self, hours: Optional[int] = None):
        """Get historical metrics."""
        return self.metrics_collector.get_metrics_history(hours)
    
    def register_health_check(self, name: str, check_func: Callable) -> None:
        """Register a custom health check."""
        self.system_monitor.register_health_check(name, check_func)
    
    def _register_custom_health_checks(self) -> None:
        """Register custom health checks for stock predictor components."""
        
        def check_model_versions():
            """Check if we have active model versions."""
            from .system_monitor import HealthCheck, HealthStatus
            
            try:
                # Check if we have any active models
                active_count = 0
                for model_type in self.config.models.model_types:
                    for pattern_length in self.config.features.pattern_lengths:
                        active_version = self.model_version_manager.get_active_version(model_type, pattern_length)
                        if active_version:
                            active_count += 1
                
                if active_count == 0:
                    return HealthCheck(
                        component="model_versions",
                        status=HealthStatus.WARNING,
                        message="No active model versions found"
                    )
                else:
                    return HealthCheck(
                        component="model_versions",
                        status=HealthStatus.HEALTHY,
                        message=f"{active_count} active model versions available"
                    )
                    
            except Exception as e:
                return HealthCheck(
                    component="model_versions",
                    status=HealthStatus.CRITICAL,
                    message=f"Failed to check model versions: {str(e)}"
                )
        
        def check_concurrent_processor():
            """Check concurrent processor health."""
            from .system_monitor import HealthCheck, HealthStatus
            
            try:
                stats = self.concurrent_processor.get_processing_stats()
                active_tasks = stats.get("active_tasks", 0)
                failed_tasks = stats.get("tasks_failed", 0)
                completed_tasks = stats.get("tasks_completed", 0)
                
                total_tasks = failed_tasks + completed_tasks
                failure_rate = failed_tasks / total_tasks if total_tasks > 0 else 0
                
                if failure_rate > 0.5:
                    return HealthCheck(
                        component="concurrent_processor",
                        status=HealthStatus.CRITICAL,
                        message=f"High task failure rate: {failure_rate:.2%}"
                    )
                elif failure_rate > 0.2:
                    return HealthCheck(
                        component="concurrent_processor",
                        status=HealthStatus.WARNING,
                        message=f"Elevated task failure rate: {failure_rate:.2%}"
                    )
                else:
                    return HealthCheck(
                        component="concurrent_processor",
                        status=HealthStatus.HEALTHY,
                        message=f"Processor healthy: {active_tasks} active, {failure_rate:.2%} failure rate"
                    )
                    
            except Exception as e:
                return HealthCheck(
                    component="concurrent_processor",
                    status=HealthStatus.CRITICAL,
                    message=f"Failed to check processor: {str(e)}"
                )
        
        self.system_monitor.register_health_check("model_versions", check_model_versions)
        self.system_monitor.register_health_check("concurrent_processor", check_concurrent_processor)
    
    def _register_custom_metrics_collectors(self) -> None:
        """Register custom metrics collectors for stock predictor components."""
        
        def collect_processor_metrics():
            """Collect concurrent processor metrics."""
            from .metrics_collector import Metric, MetricType
            
            metrics = []
            timestamp = datetime.now()
            
            try:
                stats = self.concurrent_processor.get_processing_stats()
                
                metrics.extend([
                    Metric("processor.active_tasks", stats.get("active_tasks", 0), MetricType.GAUGE, timestamp),
                    Metric("processor.queue_size", stats.get("queue_size", 0), MetricType.GAUGE, timestamp),
                    Metric("processor.tasks_completed", stats.get("tasks_completed", 0), MetricType.COUNTER, timestamp),
                    Metric("processor.tasks_failed", stats.get("tasks_failed", 0), MetricType.COUNTER, timestamp),
                    Metric("processor.avg_processing_time_ms", stats.get("avg_processing_time_ms", 0), MetricType.GAUGE, timestamp)
                ])
                
            except Exception as e:
                self.logger.error(f"Failed to collect processor metrics: {str(e)}")
            
            return metrics
        
        def collect_model_version_metrics():
            """Collect model version metrics."""
            from .metrics_collector import Metric, MetricType
            
            metrics = []
            timestamp = datetime.now()
            
            try:
                total_versions = 0
                active_versions = 0
                
                for model_type in self.config.models.model_types:
                    for pattern_length in self.config.features.pattern_lengths:
                        versions = self.model_version_manager.list_versions(model_type, pattern_length)
                        total_versions += len(versions)
                        
                        active_version = self.model_version_manager.get_active_version(model_type, pattern_length)
                        if active_version:
                            active_versions += 1
                
                metrics.extend([
                    Metric("models.total_versions", total_versions, MetricType.GAUGE, timestamp),
                    Metric("models.active_versions", active_versions, MetricType.GAUGE, timestamp)
                ])
                
            except Exception as e:
                self.logger.error(f"Failed to collect model version metrics: {str(e)}")
            
            return metrics
        
        self.metrics_collector.register_custom_collector("processor_metrics", collect_processor_metrics)
        self.metrics_collector.register_custom_collector("model_version_metrics", collect_model_version_metrics)
    
    def _count_model_versions(self) -> int:
        """Count total number of model versions."""
        total_count = 0
        
        try:
            for model_type in self.config.models.model_types:
                for pattern_length in self.config.features.pattern_lengths:
                    versions = self.model_version_manager.list_versions(model_type, pattern_length)
                    total_count += len(versions)
        except Exception as e:
            self.logger.error(f"Failed to count model versions: {str(e)}")
        
        return total_count
    
    def _assess_reliability(
        self,
        system_health: SystemHealth,
        system_metrics: SystemMetrics,
        processor_stats: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Assess overall system reliability.
        
        Returns:
            Tuple of (is_reliable, list_of_issues)
        """
        issues = []
        
        # Check system health
        if system_health.overall_status == HealthStatus.CRITICAL:
            issues.append("Critical system health issues detected")
        elif system_health.overall_status == HealthStatus.WARNING:
            issues.append("System health warnings detected")
        
        # Check processor failure rate
        failed_tasks = processor_stats.get("tasks_failed", 0)
        completed_tasks = processor_stats.get("tasks_completed", 0)
        total_tasks = failed_tasks + completed_tasks
        
        if total_tasks > 0:
            failure_rate = failed_tasks / total_tasks
            if failure_rate > 0.3:
                issues.append(f"High task failure rate: {failure_rate:.2%}")
        
        # Check if we have active models
        active_models = 0
        try:
            for model_type in self.config.models.model_types:
                for pattern_length in self.config.features.pattern_lengths:
                    active_version = self.model_version_manager.get_active_version(model_type, pattern_length)
                    if active_version:
                        active_models += 1
        except Exception:
            pass
        
        if active_models == 0:
            issues.append("No active model versions available")
        
        # System is reliable if no critical issues
        is_reliable = len(issues) == 0 or all("warning" in issue.lower() for issue in issues)
        
        return is_reliable, issues