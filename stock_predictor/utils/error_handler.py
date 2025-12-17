"""
Error handling utilities and retry mechanisms for the Stock Direction Predictor system.
Provides decorators and utilities for robust error handling and recovery.
"""

import time
import functools
from typing import Callable, Type, Tuple, Optional, Any
from .logging_config import get_logger
from .exceptions import StockPredictorError, NetworkError


def retry_on_exception(
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0
):
    """
    Decorator that retries a function on specified exceptions with exponential backoff.
    
    Args:
        exceptions: Tuple of exception types to retry on
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each attempt
        max_delay: Maximum delay between retries
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = get_logger(f"{func.__module__}.{func.__name__}")
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts: {str(e)}")
                        raise
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. Retrying in {current_delay:.2f}s")
                    time.sleep(current_delay)
                    current_delay = min(current_delay * backoff_factor, max_delay)
                except Exception as e:
                    # Don't retry on unexpected exceptions
                    logger.error(f"Unexpected exception in {func.__name__}: {str(e)}")
                    raise
            
            return None  # Should never reach here
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    default_return: Any = None,
    log_errors: bool = True,
    reraise: bool = False
) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        default_return: Value to return if function fails
        log_errors: Whether to log errors
        reraise: Whether to reraise exceptions after logging
    
    Returns:
        Function result or default_return if function fails
    """
    logger = get_logger("safe_execute")
    
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger.error(f"Error executing function {func.__name__}: {str(e)}")
        
        if reraise:
            raise
        
        return default_return


def validate_and_execute(
    func: Callable,
    validation_func: Optional[Callable] = None,
    *args,
    **kwargs
) -> Any:
    """
    Execute a function with optional pre-validation.
    
    Args:
        func: Function to execute
        validation_func: Optional validation function to run first
        *args: Arguments to pass to func
        **kwargs: Keyword arguments to pass to func
    
    Returns:
        Function result
    
    Raises:
        Exception: If validation fails or function execution fails
    """
    logger = get_logger("validate_and_execute")
    
    try:
        # Run validation if provided
        if validation_func:
            logger.debug(f"Running validation for {func.__name__}")
            validation_func(*args, **kwargs)
        
        # Execute the main function
        logger.debug(f"Executing {func.__name__}")
        return func(*args, **kwargs)
    
    except Exception as e:
        logger.error(f"Error in validate_and_execute for {func.__name__}: {str(e)}")
        raise


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for handling repeated failures.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.logger = get_logger("CircuitBreaker")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to function
            **kwargs: Keyword arguments to pass to function
        
        Returns:
            Function result
        
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker moving to HALF_OPEN state")
            else:
                raise StockPredictorError(
                    "Circuit breaker is OPEN - too many recent failures",
                    error_code="CIRCUIT_BREAKER_OPEN"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful execution."""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.logger.info("Circuit breaker reset to CLOSED state")
        
        self.failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


def circuit_breaker(failure_threshold: int = 5, recovery_timeout: float = 60.0):
    """
    Decorator that applies circuit breaker pattern to a function.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting to close circuit
    """
    breaker = CircuitBreaker(failure_threshold, recovery_timeout)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return breaker.call(func, *args, **kwargs)
        
        return wrapper
    
    return decorator


class ErrorContext:
    """
    Context manager for structured error handling and logging.
    """
    
    def __init__(self, operation_name: str, logger_name: Optional[str] = None):
        self.operation_name = operation_name
        self.logger = get_logger(logger_name or "ErrorContext")
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0
        
        if exc_type is None:
            self.logger.info(f"Operation completed successfully: {self.operation_name} ({duration:.2f}s)")
        else:
            self.logger.error(f"Operation failed: {self.operation_name} ({duration:.2f}s) - {str(exc_val)}")
        
        # Don't suppress exceptions
        return False