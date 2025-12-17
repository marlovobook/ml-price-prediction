"""
Metrics collection and operational visibility for the Stock Direction Predictor system.
Provides comprehensive metrics collection and monitoring capabilities.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
from ..utils.logging_config import get_logger


class MetricType(Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Individual metric data model."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __str__(self) -> str:
        tags_str = ",".join([f"{k}={v}" for k, v in self.tags.items()]) if self.tags else ""
        return f"{self.name}[{tags_str}] = {self.value} ({self.metric_type.value})"


@dataclass
class SystemMetrics:
    """System-wide metrics collection."""
    timestamp: datetime
    metrics: List[Metric]
    
    def get_metric(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[Metric]:
        """Get a specific metric by name and optional tags."""
        for metric in self.metrics:
            if metric.name == name:
                if tags is None or metric.tags == tags:
                    return metric
        return None
    
    def get_metrics_by_type(self, metric_type: MetricType) -> List[Metric]:
        """Get all metrics of a specific type."""
        return [metric for metric in self.metrics if metric.metric_type == metric_type]


class MetricsCollector:
    """
    Comprehensive metrics collection and monitoring service.
    Collects and aggregates system and application metrics.
    """
    
    def __init__(self, retention_hours: int = 24, collection_interval: float = 30.0):
        """
        Initialize metrics collector.
        
        Args:
            retention_hours: How long to retain metrics in memory
            collection_interval: Interval between metric collections in seconds
        """
        self.retention_hours = retention_hours
        self.collection_interval = collection_interval
        self.logger = get_logger("MetricsCollector")
        
        # Metrics storage
        self._metrics_history: deque = deque(maxlen=int(retention_hours * 3600 / collection_interval))
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Collection thread
        self._collection_thread: Optional[threading.Thread] = None
        self._stop_collection = threading.Event()
        
        # Custom metric collectors
        self._custom_collectors: Dict[str, Callable[[], List[Metric]]] = {}
    
    def register_custom_collector(self, name: str, collector_func: Callable[[], List[Metric]]) -> None:
        """
        Register a custom metric collector function.
        
        Args:
            name: Name of the collector
            collector_func: Function that returns a list of metrics
        """
        with self._lock:
            self._custom_collectors[name] = collector_func
        self.logger.info(f"Registered custom metric collector: {name}")
    
    def unregister_custom_collector(self, name: str) -> None:
        """
        Unregister a custom metric collector.
        
        Args:
            name: Name of the collector to remove
        """
        with self._lock:
            if name in self._custom_collectors:
                del self._custom_collectors[name]
        self.logger.info(f"Unregistered custom metric collector: {name}")
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Counter name
            value: Value to increment by
            tags: Optional tags for the metric
        """
        with self._lock:
            key = self._make_key(name, tags)
            self._counters[key] += value
    
    def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric value.
        
        Args:
            name: Gauge name
            value: Value to set
            tags: Optional tags for the metric
        """
        with self._lock:
            key = self._make_key(name, tags)
            self._gauges[key] = value
    
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a value in a histogram metric.
        
        Args:
            name: Histogram name
            value: Value to record
            tags: Optional tags for the metric
        """
        with self._lock:
            key = self._make_key(name, tags)
            self._histograms[key].append(value)
            
            # Keep only recent values (last 1000)
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]
    
    def record_timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a timing metric.
        
        Args:
            name: Timer name
            duration_ms: Duration in milliseconds
            tags: Optional tags for the metric
        """
        with self._lock:
            key = self._make_key(name, tags)
            self._timers[key].append(duration_ms)
            
            # Keep only recent values (last 1000)
            if len(self._timers[key]) > 1000:
                self._timers[key] = self._timers[key][-1000:]
    
    def timer_context(self, name: str, tags: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.
        
        Args:
            name: Timer name
            tags: Optional tags for the metric
        
        Returns:
            Context manager that records timing
        """
        return TimerContext(self, name, tags)
    
    def collect_metrics(self) -> SystemMetrics:
        """
        Collect all current metrics and return as SystemMetrics object.
        
        Returns:
            SystemMetrics object with current metric values
        """
        metrics = []
        timestamp = datetime.now()
        
        with self._lock:
            # Collect counters
            for key, value in self._counters.items():
                name, tags = self._parse_key(key)
                metrics.append(Metric(
                    name=name,
                    value=value,
                    metric_type=MetricType.COUNTER,
                    timestamp=timestamp,
                    tags=tags
                ))
            
            # Collect gauges
            for key, value in self._gauges.items():
                name, tags = self._parse_key(key)
                metrics.append(Metric(
                    name=name,
                    value=value,
                    metric_type=MetricType.GAUGE,
                    timestamp=timestamp,
                    tags=tags
                ))
            
            # Collect histogram statistics
            for key, values in self._histograms.items():
                if values:
                    name, tags = self._parse_key(key)
                    # Add mean, min, max, count
                    metrics.extend([
                        Metric(f"{name}.mean", sum(values) / len(values), MetricType.HISTOGRAM, timestamp, tags),
                        Metric(f"{name}.min", min(values), MetricType.HISTOGRAM, timestamp, tags),
                        Metric(f"{name}.max", max(values), MetricType.HISTOGRAM, timestamp, tags),
                        Metric(f"{name}.count", len(values), MetricType.HISTOGRAM, timestamp, tags)
                    ])
            
            # Collect timer statistics
            for key, values in self._timers.items():
                if values:
                    name, tags = self._parse_key(key)
                    # Add mean, min, max, count
                    metrics.extend([
                        Metric(f"{name}.mean", sum(values) / len(values), MetricType.TIMER, timestamp, tags),
                        Metric(f"{name}.min", min(values), MetricType.TIMER, timestamp, tags),
                        Metric(f"{name}.max", max(values), MetricType.TIMER, timestamp, tags),
                        Metric(f"{name}.count", len(values), MetricType.TIMER, timestamp, tags)
                    ])
        
        # Collect custom metrics
        for collector_name, collector_func in self._custom_collectors.items():
            try:
                custom_metrics = collector_func()
                metrics.extend(custom_metrics)
            except Exception as e:
                self.logger.error(f"Error collecting custom metrics from {collector_name}: {str(e)}")
        
        system_metrics = SystemMetrics(timestamp=timestamp, metrics=metrics)
        
        # Store in history
        with self._lock:
            self._metrics_history.append(system_metrics)
        
        return system_metrics
    
    def get_metrics_history(self, hours: Optional[int] = None) -> List[SystemMetrics]:
        """
        Get historical metrics.
        
        Args:
            hours: Number of hours of history to return (None for all available)
            
        Returns:
            List of SystemMetrics objects
        """
        with self._lock:
            if hours is None:
                return list(self._metrics_history)
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            return [
                metrics for metrics in self._metrics_history
                if metrics.timestamp >= cutoff_time
            ]
    
    def start_collection(self) -> None:
        """Start automatic metrics collection in background thread."""
        if self._collection_thread and self._collection_thread.is_alive():
            self.logger.warning("Metrics collection is already running")
            return
        
        self._stop_collection.clear()
        self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collection_thread.start()
        self.logger.info(f"Started metrics collection with {self.collection_interval}s interval")
    
    def stop_collection(self) -> None:
        """Stop automatic metrics collection."""
        if self._collection_thread and self._collection_thread.is_alive():
            self._stop_collection.set()
            self._collection_thread.join(timeout=5.0)
            self.logger.info("Stopped metrics collection")
    
    def _collection_loop(self) -> None:
        """Main collection loop running in background thread."""
        while not self._stop_collection.is_set():
            try:
                self.collect_metrics()
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {str(e)}")
            
            # Wait for next collection or stop signal
            self._stop_collection.wait(self.collection_interval)
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create a unique key for a metric with tags."""
        if not tags:
            return name
        
        tag_str = ",".join([f"{k}={v}" for k, v in sorted(tags.items())])
        return f"{name}[{tag_str}]"
    
    def _parse_key(self, key: str) -> tuple[str, Dict[str, str]]:
        """Parse a metric key back into name and tags."""
        if '[' not in key:
            return key, {}
        
        name, tag_part = key.split('[', 1)
        tag_part = tag_part.rstrip(']')
        
        tags = {}
        if tag_part:
            for tag_pair in tag_part.split(','):
                k, v = tag_pair.split('=', 1)
                tags[k] = v
        
        return name, tags
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()
            self._metrics_history.clear()
        self.logger.info("Reset all metrics")


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.time() - self.start_time) * 1000
            self.collector.record_timer(self.name, duration_ms, self.tags)