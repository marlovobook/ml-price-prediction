"""
Concurrent processing support for multiple stock analysis.
Provides thread-safe concurrent processing capabilities with proper resource management.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from queue import Queue, Empty
from ..utils.logging_config import get_logger
from ..utils.exceptions import StockPredictorError
from ..utils.error_handler import retry_on_exception, CircuitBreaker
from .metrics_collector import MetricsCollector


@dataclass
class ProcessingTask:
    """Task for concurrent processing."""
    task_id: str
    symbol: str
    task_type: str
    parameters: Dict[str, Any]
    priority: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ProcessingResult:
    """Result of a processing task."""
    task_id: str
    symbol: str
    task_type: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time_ms: float = 0
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.now()


class ConcurrentStockProcessor:
    """
    Concurrent processing engine for multiple stock analysis.
    Provides thread-safe processing with resource management and monitoring.
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        queue_size: int = 100,
        timeout_seconds: float = 300.0,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize concurrent processor.
        
        Args:
            max_workers: Maximum number of worker threads
            queue_size: Maximum size of task queue
            timeout_seconds: Timeout for individual tasks
            metrics_collector: Optional metrics collector for monitoring
        """
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.timeout_seconds = timeout_seconds
        self.metrics_collector = metrics_collector
        self.logger = get_logger("ConcurrentStockProcessor")
        
        # Task queue and processing
        self.task_queue: Queue[ProcessingTask] = Queue(maxsize=queue_size)
        self.executor: Optional[ThreadPoolExecutor] = None
        self.active_tasks: Dict[str, Future] = {}
        self.completed_tasks: Dict[str, ProcessingResult] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        self._shutdown = threading.Event()
        
        # Task processors registry
        self._task_processors: Dict[str, Callable] = {}
        
        # Circuit breakers for different task types
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Processing statistics
        self._stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_processing_time_ms": 0.0
        }
    
    def register_task_processor(self, task_type: str, processor_func: Callable) -> None:
        """
        Register a task processor function.
        
        Args:
            task_type: Type of task to process
            processor_func: Function that processes the task
        """
        with self._lock:
            self._task_processors[task_type] = processor_func
            self._circuit_breakers[task_type] = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
        
        self.logger.info(f"Registered task processor for type: {task_type}")
    
    def start(self) -> None:
        """Start the concurrent processing engine."""
        if self.executor is not None:
            self.logger.warning("Processor is already running")
            return
        
        self._shutdown.clear()
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="StockProcessor")
        
        self.logger.info(f"Started concurrent processor with {self.max_workers} workers")
        
        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.set_gauge("concurrent_processor.workers", self.max_workers)
            self.metrics_collector.increment_counter("concurrent_processor.starts")
    
    def stop(self, wait: bool = True, timeout: float = 30.0) -> None:
        """
        Stop the concurrent processing engine.
        
        Args:
            wait: Whether to wait for active tasks to complete
            timeout: Maximum time to wait for shutdown
        """
        if self.executor is None:
            return
        
        self._shutdown.set()
        
        if wait:
            # Wait for active tasks to complete
            self.logger.info("Waiting for active tasks to complete...")
            with self._lock:
                active_futures = list(self.active_tasks.values())
            
            for future in active_futures:
                try:
                    future.result(timeout=timeout / len(active_futures) if active_futures else timeout)
                except Exception as e:
                    self.logger.warning(f"Task did not complete cleanly: {str(e)}")
        
        # Shutdown executor
        self.executor.shutdown(wait=wait)
        self.executor = None
        
        self.logger.info("Stopped concurrent processor")
        
        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.increment_counter("concurrent_processor.stops")
    
    def submit_task(
        self,
        symbol: str,
        task_type: str,
        parameters: Dict[str, Any],
        priority: int = 0,
        task_id: Optional[str] = None
    ) -> str:
        """
        Submit a task for processing.
        
        Args:
            symbol: Stock symbol to process
            task_type: Type of task to perform
            parameters: Task parameters
            priority: Task priority (higher = more priority)
            task_id: Optional custom task ID
            
        Returns:
            Task ID for tracking
        """
        if self.executor is None:
            raise StockPredictorError("Processor is not running", error_code="PROCESSOR_NOT_RUNNING")
        
        if task_type not in self._task_processors:
            raise StockPredictorError(
                f"No processor registered for task type: {task_type}",
                error_code="UNKNOWN_TASK_TYPE"
            )
        
        # Generate task ID if not provided
        if task_id is None:
            task_id = f"{task_type}_{symbol}_{int(time.time() * 1000)}"
        
        # Create task
        task = ProcessingTask(
            task_id=task_id,
            symbol=symbol,
            task_type=task_type,
            parameters=parameters,
            priority=priority
        )
        
        try:
            # Submit to queue (non-blocking)
            self.task_queue.put_nowait(task)
            
            # Submit to executor
            future = self.executor.submit(self._process_task, task)
            
            with self._lock:
                self.active_tasks[task_id] = future
                self._stats["tasks_submitted"] += 1
            
            self.logger.debug(f"Submitted task {task_id} for {symbol} ({task_type})")
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.increment_counter("concurrent_processor.tasks_submitted")
                self.metrics_collector.increment_counter(
                    "concurrent_processor.tasks_by_type",
                    tags={"task_type": task_type}
                )
            
            return task_id
            
        except Exception as e:
            raise StockPredictorError(
                f"Failed to submit task: {str(e)}",
                error_code="TASK_SUBMISSION_ERROR",
                details={"task_id": task_id, "symbol": symbol, "task_type": task_type}
            )
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> ProcessingResult:
        """
        Get the result of a task.
        
        Args:
            task_id: Task ID to get result for
            timeout: Optional timeout for waiting
            
        Returns:
            ProcessingResult object
        """
        # Check if already completed
        with self._lock:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            if task_id not in self.active_tasks:
                raise StockPredictorError(
                    f"Task {task_id} not found",
                    error_code="TASK_NOT_FOUND"
                )
            
            future = self.active_tasks[task_id]
        
        try:
            # Wait for completion
            result = future.result(timeout=timeout)
            return result
            
        except Exception as e:
            raise StockPredictorError(
                f"Failed to get task result: {str(e)}",
                error_code="TASK_RESULT_ERROR",
                details={"task_id": task_id}
            )
    
    def wait_for_tasks(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, ProcessingResult]:
        """
        Wait for multiple tasks to complete.
        
        Args:
            task_ids: List of task IDs to wait for
            timeout: Optional timeout for waiting
            
        Returns:
            Dictionary mapping task IDs to results
        """
        results = {}
        start_time = time.time()
        
        for task_id in task_ids:
            remaining_timeout = None
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining_timeout = max(0, timeout - elapsed)
                if remaining_timeout <= 0:
                    break
            
            try:
                result = self.get_task_result(task_id, remaining_timeout)
                results[task_id] = result
            except Exception as e:
                self.logger.error(f"Failed to get result for task {task_id}: {str(e)}")
        
        return results
    
    def get_active_tasks(self) -> List[str]:
        """Get list of active task IDs."""
        with self._lock:
            return list(self.active_tasks.keys())
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self._lock:
            stats = self._stats.copy()
            stats["active_tasks"] = len(self.active_tasks)
            stats["queue_size"] = self.task_queue.qsize()
            stats["avg_processing_time_ms"] = (
                stats["total_processing_time_ms"] / stats["tasks_completed"]
                if stats["tasks_completed"] > 0 else 0
            )
        
        return stats
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel an active task.
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if task was cancelled, False otherwise
        """
        with self._lock:
            if task_id not in self.active_tasks:
                return False
            
            future = self.active_tasks[task_id]
            cancelled = future.cancel()
            
            if cancelled:
                del self.active_tasks[task_id]
                self.logger.info(f"Cancelled task {task_id}")
                
                # Record metrics
                if self.metrics_collector:
                    self.metrics_collector.increment_counter("concurrent_processor.tasks_cancelled")
            
            return cancelled
    
    @retry_on_exception(max_attempts=3, delay=1.0)
    def _process_task(self, task: ProcessingTask) -> ProcessingResult:
        """
        Process a single task.
        
        Args:
            task: Task to process
            
        Returns:
            ProcessingResult object
        """
        start_time = time.time()
        
        try:
            # Check if shutdown requested
            if self._shutdown.is_set():
                raise StockPredictorError("Processor is shutting down", error_code="SHUTDOWN_REQUESTED")
            
            # Get processor function
            processor_func = self._task_processors[task.task_type]
            circuit_breaker = self._circuit_breakers[task.task_type]
            
            # Process through circuit breaker
            result = circuit_breaker.call(processor_func, task.symbol, task.parameters)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create success result
            processing_result = ProcessingResult(
                task_id=task.task_id,
                symbol=task.symbol,
                task_type=task.task_type,
                success=True,
                result=result,
                processing_time_ms=processing_time_ms
            )
            
            # Update statistics
            with self._lock:
                self._stats["tasks_completed"] += 1
                self._stats["total_processing_time_ms"] += processing_time_ms
                
                # Move from active to completed
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.completed_tasks[task.task_id] = processing_result
            
            self.logger.debug(f"Completed task {task.task_id} in {processing_time_ms:.2f}ms")
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.increment_counter("concurrent_processor.tasks_completed")
                self.metrics_collector.record_timer(
                    "concurrent_processor.task_duration",
                    processing_time_ms,
                    tags={"task_type": task.task_type, "symbol": task.symbol}
                )
            
            return processing_result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create error result
            processing_result = ProcessingResult(
                task_id=task.task_id,
                symbol=task.symbol,
                task_type=task.task_type,
                success=False,
                error=str(e),
                processing_time_ms=processing_time_ms
            )
            
            # Update statistics
            with self._lock:
                self._stats["tasks_failed"] += 1
                
                # Move from active to completed
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.completed_tasks[task.task_id] = processing_result
            
            self.logger.error(f"Failed task {task.task_id} after {processing_time_ms:.2f}ms: {str(e)}")
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.increment_counter("concurrent_processor.tasks_failed")
                self.metrics_collector.record_timer(
                    "concurrent_processor.task_duration",
                    processing_time_ms,
                    tags={"task_type": task.task_type, "symbol": task.symbol, "status": "failed"}
                )
            
            return processing_result