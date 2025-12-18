"""
Performance Optimizer for VectorBT Visualization Enhancement.

This module provides performance optimization for large datasets including data sampling
strategies, memory management, cleanup for plot objects, and performance monitoring.
"""

import logging
import time
import gc
import psutil
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

from ..utils.exceptions import DataValidationError, BacktestingError


class SamplingStrategy(Enum):
    """Available data sampling strategies for large datasets."""
    UNIFORM = "uniform"
    RANDOM = "random"
    SYSTEMATIC = "systematic"
    STRATIFIED = "stratified"
    TIME_BASED = "time_based"
    ADAPTIVE = "adaptive"


class MemoryManagementMode(Enum):
    """Memory management modes for different scenarios."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MINIMAL = "minimal"


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring operations."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    memory_before_mb: Optional[float] = None
    memory_after_mb: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    data_size_mb: Optional[float] = None
    optimization_applied: bool = False
    sampling_ratio: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    max_data_points: int = 50000
    max_memory_mb: float = 1024.0  # 1GB
    enable_sampling: bool = True
    sampling_strategy: SamplingStrategy = SamplingStrategy.ADAPTIVE
    memory_management_mode: MemoryManagementMode = MemoryManagementMode.BALANCED
    enable_caching: bool = True
    cache_size_limit_mb: float = 512.0  # 512MB
    performance_monitoring: bool = True
    cleanup_threshold_mb: float = 100.0  # Cleanup when using > 100MB
    warning_threshold_seconds: float = 30.0  # Warn if operation takes > 30s


class VisualizationPerformanceOptimizer:
    """
    Performance optimizer for VectorBT visualization operations.
    
    Provides data sampling strategies for large datasets, memory management and cleanup
    for plot objects, and performance monitoring with optimization recommendations.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize the performance optimizer.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.cache: Dict[str, Any] = {}
        self.cache_size_mb = 0.0
        
        # Memory monitoring
        self.process = psutil.Process()
        self.baseline_memory_mb = self._get_memory_usage_mb()
        
        self.logger.info(
            f"Performance Optimizer initialized: "
            f"max_data_points={self.config.max_data_points}, "
            f"max_memory={self.config.max_memory_mb}MB, "
            f"baseline_memory={self.baseline_memory_mb:.1f}MB"
        )
    
    def optimize_dataset_for_visualization(
        self,
        data: Union[pd.DataFrame, pd.Series, np.ndarray],
        operation_name: str = "visualization",
        target_size: Optional[int] = None
    ) -> Tuple[Union[pd.DataFrame, pd.Series, np.ndarray], Dict[str, Any]]:
        """
        Optimize dataset for visualization by applying appropriate sampling strategy.
        
        Args:
            data: Input dataset to optimize
            operation_name: Name of the operation for tracking
            target_size: Target size after optimization (uses config default if None)
            
        Returns:
            Tuple of (optimized_data, optimization_info)
        """
        try:
            start_time = time.time()
            memory_before = self._get_memory_usage_mb()
            
            # Calculate data size
            data_size_mb = self._calculate_data_size_mb(data)
            
            # Determine if optimization is needed
            current_size = len(data)
            max_size = target_size or self.config.max_data_points
            
            optimization_info = {
                'original_size': current_size,
                'target_size': max_size,
                'data_size_mb': data_size_mb,
                'optimization_applied': False,
                'sampling_strategy': None,
                'sampling_ratio': 1.0,
                'memory_saved_mb': 0.0
            }
            
            # Check if optimization is needed
            if not self.config.enable_sampling or current_size <= max_size:
                self.logger.debug(f"No optimization needed for {operation_name}: {current_size} <= {max_size}")
                return data, optimization_info
            
            # Apply sampling strategy
            self.logger.info(
                f"Optimizing dataset for {operation_name}: {current_size} -> {max_size} points "
                f"({data_size_mb:.1f}MB)"
            )
            
            optimized_data, sampling_info = self._apply_sampling_strategy(
                data, max_size, self.config.sampling_strategy
            )
            
            # Update optimization info
            optimization_info.update({
                'optimization_applied': True,
                'sampling_strategy': sampling_info['strategy'],
                'sampling_ratio': len(optimized_data) / current_size,
                'final_size': len(optimized_data),
                'memory_saved_mb': data_size_mb * (1 - optimization_info['sampling_ratio'])
            })
            
            # Log optimization results
            duration = time.time() - start_time
            memory_after = self._get_memory_usage_mb()
            
            self.logger.info(
                f"Dataset optimization completed in {duration:.2f}s: "
                f"{current_size} -> {len(optimized_data)} points "
                f"(ratio: {optimization_info['sampling_ratio']:.3f})"
            )
            
            # Record performance metrics
            if self.config.performance_monitoring:
                self._record_performance_metrics(
                    operation_name=f"optimize_{operation_name}",
                    start_time=start_time,
                    end_time=time.time(),
                    memory_before_mb=memory_before,
                    memory_after_mb=memory_after,
                    data_size_mb=data_size_mb,
                    optimization_applied=True,
                    sampling_ratio=optimization_info['sampling_ratio']
                )
            
            return optimized_data, optimization_info
            
        except Exception as e:
            self.logger.error(f"Error optimizing dataset for {operation_name}: {str(e)}")
            raise BacktestingError(f"Dataset optimization failed: {str(e)}")
    
    def monitor_operation_performance(
        self,
        operation_func: Callable,
        operation_name: str,
        *args,
        **kwargs
    ) -> Tuple[Any, PerformanceMetrics]:
        """
        Monitor performance of a visualization operation.
        
        Args:
            operation_func: Function to monitor
            operation_name: Name of the operation
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Tuple of (function_result, performance_metrics)
        """
        try:
            # Start monitoring
            start_time = time.time()
            memory_before = self._get_memory_usage_mb()
            
            # Execute operation
            try:
                result = operation_func(*args, **kwargs)
                success = True
                error_message = None
            except Exception as e:
                result = None
                success = False
                error_message = str(e)
                raise
            
        except Exception as e:
            success = False
            error_message = str(e)
            result = None
        
        finally:
            # Record metrics regardless of success/failure
            end_time = time.time()
            memory_after = self._get_memory_usage_mb()
            duration = end_time - start_time
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                memory_before_mb=memory_before,
                memory_after_mb=memory_after,
                memory_peak_mb=max(memory_before, memory_after),
                success=success,
                error_message=error_message
            )
            
            # Log performance
            if success:
                self.logger.info(
                    f"Operation '{operation_name}' completed in {duration:.2f}s "
                    f"(memory: {memory_before:.1f} -> {memory_after:.1f} MB)"
                )
            else:
                self.logger.error(
                    f"Operation '{operation_name}' failed after {duration:.2f}s: {error_message}"
                )
            
            # Check for performance warnings
            if duration > self.config.warning_threshold_seconds:
                self.logger.warning(
                    f"Operation '{operation_name}' took {duration:.2f}s "
                    f"(exceeds threshold of {self.config.warning_threshold_seconds}s)"
                )
            
            # Record metrics
            if self.config.performance_monitoring:
                self.performance_history.append(metrics)
            
            # Trigger cleanup if needed
            if memory_after > self.baseline_memory_mb + self.config.cleanup_threshold_mb:
                self._perform_memory_cleanup()
        
        return result, metrics
    
    def manage_plot_memory(
        self,
        plot_objects: List[Any],
        operation_name: str = "plot_management"
    ) -> Dict[str, Any]:
        """
        Manage memory for plot objects with cleanup strategies.
        
        Args:
            plot_objects: List of plot objects to manage
            operation_name: Name of the operation for tracking
            
        Returns:
            Dictionary with memory management results
        """
        try:
            memory_before = self._get_memory_usage_mb()
            
            management_result = {
                'objects_processed': len(plot_objects),
                'memory_before_mb': memory_before,
                'memory_after_mb': 0.0,
                'memory_freed_mb': 0.0,
                'cleanup_actions': [],
                'success': True
            }
            
            # Apply memory management based on mode
            if self.config.memory_management_mode == MemoryManagementMode.AGGRESSIVE:
                management_result['cleanup_actions'].extend(
                    self._aggressive_plot_cleanup(plot_objects)
                )
            elif self.config.memory_management_mode == MemoryManagementMode.CONSERVATIVE:
                management_result['cleanup_actions'].extend(
                    self._conservative_plot_cleanup(plot_objects)
                )
            elif self.config.memory_management_mode == MemoryManagementMode.MINIMAL:
                management_result['cleanup_actions'].extend(
                    self._minimal_plot_cleanup(plot_objects)
                )
            else:  # BALANCED
                management_result['cleanup_actions'].extend(
                    self._balanced_plot_cleanup(plot_objects)
                )
            
            # Force garbage collection
            gc.collect()
            
            # Measure memory after cleanup
            memory_after = self._get_memory_usage_mb()
            management_result.update({
                'memory_after_mb': memory_after,
                'memory_freed_mb': max(0, memory_before - memory_after)
            })
            
            self.logger.info(
                f"Plot memory management for {operation_name}: "
                f"{len(plot_objects)} objects, "
                f"freed {management_result['memory_freed_mb']:.1f}MB"
            )
            
            return management_result
            
        except Exception as e:
            self.logger.error(f"Error in plot memory management: {str(e)}")
            return {
                'objects_processed': len(plot_objects),
                'success': False,
                'error': str(e)
            }
    
    def get_performance_recommendations(self) -> Dict[str, Any]:
        """
        Generate performance optimization recommendations based on monitoring data.
        
        Returns:
            Dictionary with performance analysis and recommendations
        """
        try:
            if not self.performance_history:
                return {
                    'recommendations': ['No performance data available yet'],
                    'analysis': {},
                    'statistics': {}
                }
            
            # Analyze performance patterns
            analysis = self._analyze_performance_patterns()
            
            # Generate recommendations
            recommendations = self._generate_performance_recommendations(analysis)
            
            # Calculate statistics
            statistics = self._calculate_performance_statistics()
            
            return {
                'recommendations': recommendations,
                'analysis': analysis,
                'statistics': statistics,
                'current_memory_mb': self._get_memory_usage_mb(),
                'baseline_memory_mb': self.baseline_memory_mb,
                'cache_size_mb': self.cache_size_mb
            }
            
        except Exception as e:
            self.logger.error(f"Error generating performance recommendations: {str(e)}")
            return {'error': str(e)}
    
    def clear_cache(self) -> Dict[str, Any]:
        """
        Clear performance cache to free memory.
        
        Returns:
            Dictionary with cache clearing results
        """
        try:
            memory_before = self._get_memory_usage_mb()
            cache_items = len(self.cache)
            cache_size_before = self.cache_size_mb
            
            # Clear cache
            self.cache.clear()
            self.cache_size_mb = 0.0
            
            # Force garbage collection
            gc.collect()
            
            memory_after = self._get_memory_usage_mb()
            
            result = {
                'cache_items_cleared': cache_items,
                'cache_size_freed_mb': cache_size_before,
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after,
                'memory_freed_mb': max(0, memory_before - memory_after)
            }
            
            self.logger.info(
                f"Cache cleared: {cache_items} items, "
                f"{cache_size_before:.1f}MB cache, "
                f"{result['memory_freed_mb']:.1f}MB total memory freed"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            return {'error': str(e)}
    
    def _apply_sampling_strategy(
        self,
        data: Union[pd.DataFrame, pd.Series, np.ndarray],
        target_size: int,
        strategy: SamplingStrategy
    ) -> Tuple[Union[pd.DataFrame, pd.Series, np.ndarray], Dict[str, Any]]:
        """Apply the specified sampling strategy to reduce data size."""
        try:
            current_size = len(data)
            
            if target_size >= current_size:
                return data, {'strategy': 'none', 'ratio': 1.0}
            
            sampling_info = {'strategy': strategy.value, 'ratio': target_size / current_size}
            
            if strategy == SamplingStrategy.UNIFORM:
                # Uniform sampling - take every nth element
                step = current_size // target_size
                if isinstance(data, pd.DataFrame):
                    sampled_data = data.iloc[::step][:target_size]
                elif isinstance(data, pd.Series):
                    sampled_data = data.iloc[::step][:target_size]
                else:  # numpy array
                    sampled_data = data[::step][:target_size]
            
            elif strategy == SamplingStrategy.RANDOM:
                # Random sampling
                indices = np.random.choice(current_size, target_size, replace=False)
                indices = np.sort(indices)  # Maintain time order for time series
                
                if isinstance(data, pd.DataFrame):
                    sampled_data = data.iloc[indices]
                elif isinstance(data, pd.Series):
                    sampled_data = data.iloc[indices]
                else:  # numpy array
                    sampled_data = data[indices]
            
            elif strategy == SamplingStrategy.SYSTEMATIC:
                # Systematic sampling - start at random point, then regular intervals
                start = np.random.randint(0, current_size // target_size)
                step = current_size // target_size
                indices = np.arange(start, current_size, step)[:target_size]
                
                if isinstance(data, pd.DataFrame):
                    sampled_data = data.iloc[indices]
                elif isinstance(data, pd.Series):
                    sampled_data = data.iloc[indices]
                else:  # numpy array
                    sampled_data = data[indices]
            
            elif strategy == SamplingStrategy.TIME_BASED:
                # Time-based sampling - preserve recent data, sample older data more aggressively
                if isinstance(data, (pd.DataFrame, pd.Series)) and hasattr(data.index, 'to_pydatetime'):
                    # Keep last 20% of data, sample the rest
                    recent_cutoff = int(current_size * 0.8)
                    recent_data = data.iloc[recent_cutoff:]
                    older_data = data.iloc[:recent_cutoff]
                    
                    # Sample older data
                    older_target = target_size - len(recent_data)
                    if older_target > 0 and len(older_data) > 0:
                        step = len(older_data) // older_target
                        sampled_older = older_data.iloc[::max(1, step)][:older_target]
                        sampled_data = pd.concat([sampled_older, recent_data])
                    else:
                        sampled_data = recent_data.iloc[-target_size:]
                else:
                    # Fallback to uniform sampling
                    return self._apply_sampling_strategy(data, target_size, SamplingStrategy.UNIFORM)
            
            elif strategy == SamplingStrategy.ADAPTIVE:
                # Adaptive sampling based on data characteristics
                if isinstance(data, (pd.DataFrame, pd.Series)):
                    # Check if data has time index
                    if hasattr(data.index, 'to_pydatetime'):
                        return self._apply_sampling_strategy(data, target_size, SamplingStrategy.TIME_BASED)
                    else:
                        return self._apply_sampling_strategy(data, target_size, SamplingStrategy.UNIFORM)
                else:
                    return self._apply_sampling_strategy(data, target_size, SamplingStrategy.UNIFORM)
            
            else:
                # Default to uniform sampling
                return self._apply_sampling_strategy(data, target_size, SamplingStrategy.UNIFORM)
            
            return sampled_data, sampling_info
            
        except Exception as e:
            self.logger.error(f"Error applying sampling strategy {strategy}: {str(e)}")
            # Fallback to simple uniform sampling
            step = max(1, current_size // target_size)
            if isinstance(data, pd.DataFrame):
                fallback_data = data.iloc[::step][:target_size]
            elif isinstance(data, pd.Series):
                fallback_data = data.iloc[::step][:target_size]
            else:
                fallback_data = data[::step][:target_size]
            
            return fallback_data, {'strategy': 'fallback_uniform', 'ratio': len(fallback_data) / current_size}
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _calculate_data_size_mb(self, data: Union[pd.DataFrame, pd.Series, np.ndarray]) -> float:
        """Calculate approximate data size in MB."""
        try:
            if isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum() / (1024 * 1024)
            elif isinstance(data, pd.Series):
                return data.memory_usage(deep=True) / (1024 * 1024)
            elif isinstance(data, np.ndarray):
                return data.nbytes / (1024 * 1024)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _perform_memory_cleanup(self) -> None:
        """Perform memory cleanup operations."""
        try:
            memory_before = self._get_memory_usage_mb()
            
            # Clear cache if it's too large
            if self.cache_size_mb > self.config.cache_size_limit_mb:
                self.clear_cache()
            
            # Force garbage collection
            gc.collect()
            
            memory_after = self._get_memory_usage_mb()
            memory_freed = max(0, memory_before - memory_after)
            
            if memory_freed > 1.0:  # Only log if significant memory was freed
                self.logger.info(f"Memory cleanup freed {memory_freed:.1f}MB")
            
        except Exception as e:
            self.logger.warning(f"Error during memory cleanup: {str(e)}")
    
    def _aggressive_plot_cleanup(self, plot_objects: List[Any]) -> List[str]:
        """Aggressive cleanup strategy for plot objects."""
        actions = []
        
        try:
            for i, plot_obj in enumerate(plot_objects):
                try:
                    # Close matplotlib figures
                    if hasattr(plot_obj, 'close'):
                        plot_obj.close()
                        actions.append(f"Closed plot object {i}")
                    
                    # Clear plotly figures
                    if hasattr(plot_obj, 'data'):
                        plot_obj.data = []
                        actions.append(f"Cleared plotly data for object {i}")
                    
                    # Delete references
                    del plot_obj
                    actions.append(f"Deleted reference to object {i}")
                    
                except Exception as e:
                    actions.append(f"Failed to cleanup object {i}: {str(e)}")
            
            # Force garbage collection
            gc.collect()
            actions.append("Forced garbage collection")
            
        except Exception as e:
            actions.append(f"Error in aggressive cleanup: {str(e)}")
        
        return actions
    
    def _conservative_plot_cleanup(self, plot_objects: List[Any]) -> List[str]:
        """Conservative cleanup strategy for plot objects."""
        actions = []
        
        try:
            # Only clear data, don't close objects
            for i, plot_obj in enumerate(plot_objects):
                try:
                    if hasattr(plot_obj, 'data') and hasattr(plot_obj.data, 'clear'):
                        plot_obj.data.clear()
                        actions.append(f"Cleared data for object {i}")
                except Exception as e:
                    actions.append(f"Failed to clear data for object {i}: {str(e)}")
            
            # Gentle garbage collection
            gc.collect()
            actions.append("Performed garbage collection")
            
        except Exception as e:
            actions.append(f"Error in conservative cleanup: {str(e)}")
        
        return actions
    
    def _balanced_plot_cleanup(self, plot_objects: List[Any]) -> List[str]:
        """Balanced cleanup strategy for plot objects."""
        actions = []
        
        try:
            for i, plot_obj in enumerate(plot_objects):
                try:
                    # Clear data but keep object structure
                    if hasattr(plot_obj, 'data'):
                        if hasattr(plot_obj.data, 'clear'):
                            plot_obj.data.clear()
                        else:
                            plot_obj.data = []
                        actions.append(f"Cleared data for object {i}")
                    
                    # Close if it's a matplotlib figure
                    if hasattr(plot_obj, 'close') and hasattr(plot_obj, 'number'):
                        plot_obj.close()
                        actions.append(f"Closed matplotlib figure {i}")
                    
                except Exception as e:
                    actions.append(f"Failed to cleanup object {i}: {str(e)}")
            
            gc.collect()
            actions.append("Performed garbage collection")
            
        except Exception as e:
            actions.append(f"Error in balanced cleanup: {str(e)}")
        
        return actions
    
    def _minimal_plot_cleanup(self, plot_objects: List[Any]) -> List[str]:
        """Minimal cleanup strategy for plot objects."""
        actions = []
        
        try:
            # Only force garbage collection, don't modify objects
            gc.collect()
            actions.append("Performed minimal garbage collection")
            
        except Exception as e:
            actions.append(f"Error in minimal cleanup: {str(e)}")
        
        return actions
    
    def _record_performance_metrics(
        self,
        operation_name: str,
        start_time: float,
        end_time: float,
        memory_before_mb: float,
        memory_after_mb: float,
        data_size_mb: Optional[float] = None,
        optimization_applied: bool = False,
        sampling_ratio: Optional[float] = None
    ) -> None:
        """Record performance metrics for monitoring."""
        try:
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=end_time - start_time,
                memory_before_mb=memory_before_mb,
                memory_after_mb=memory_after_mb,
                memory_peak_mb=max(memory_before_mb, memory_after_mb),
                data_size_mb=data_size_mb,
                optimization_applied=optimization_applied,
                sampling_ratio=sampling_ratio,
                success=True
            )
            
            self.performance_history.append(metrics)
            
            # Limit history size to prevent memory growth
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]
            
        except Exception as e:
            self.logger.warning(f"Error recording performance metrics: {str(e)}")
    
    def _analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze performance patterns from historical data."""
        try:
            if not self.performance_history:
                return {}
            
            # Calculate averages and trends
            durations = [m.duration_seconds for m in self.performance_history if m.duration_seconds]
            memory_usage = [m.memory_after_mb - m.memory_before_mb for m in self.performance_history 
                          if m.memory_after_mb and m.memory_before_mb]
            
            analysis = {
                'total_operations': len(self.performance_history),
                'avg_duration_seconds': np.mean(durations) if durations else 0,
                'max_duration_seconds': max(durations) if durations else 0,
                'avg_memory_change_mb': np.mean(memory_usage) if memory_usage else 0,
                'max_memory_change_mb': max(memory_usage) if memory_usage else 0,
                'operations_with_optimization': sum(1 for m in self.performance_history if m.optimization_applied),
                'slow_operations': [m.operation_name for m in self.performance_history 
                                  if m.duration_seconds and m.duration_seconds > self.config.warning_threshold_seconds],
                'memory_intensive_operations': [m.operation_name for m in self.performance_history 
                                              if m.memory_after_mb and m.memory_before_mb and 
                                              (m.memory_after_mb - m.memory_before_mb) > 100]
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance patterns: {str(e)}")
            return {'error': str(e)}
    
    def _generate_performance_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on analysis."""
        recommendations = []
        
        try:
            if not analysis:
                return ["No performance data available for recommendations"]
            
            # Duration-based recommendations
            avg_duration = analysis.get('avg_duration_seconds', 0)
            if avg_duration > self.config.warning_threshold_seconds:
                recommendations.append(
                    f"Average operation duration ({avg_duration:.1f}s) exceeds threshold. "
                    "Consider enabling more aggressive data sampling."
                )
            
            # Memory-based recommendations
            avg_memory_change = analysis.get('avg_memory_change_mb', 0)
            if avg_memory_change > 50:
                recommendations.append(
                    f"High average memory usage ({avg_memory_change:.1f}MB per operation). "
                    "Consider more aggressive memory management."
                )
            
            # Optimization recommendations
            total_ops = analysis.get('total_operations', 0)
            optimized_ops = analysis.get('operations_with_optimization', 0)
            if total_ops > 0 and optimized_ops / total_ops < 0.5:
                recommendations.append(
                    "Less than 50% of operations used optimization. "
                    "Consider lowering max_data_points threshold."
                )
            
            # Slow operations
            slow_ops = analysis.get('slow_operations', [])
            if slow_ops:
                unique_slow_ops = list(set(slow_ops))
                recommendations.append(
                    f"Slow operations detected: {', '.join(unique_slow_ops)}. "
                    "Consider optimization for these specific operations."
                )
            
            # Memory intensive operations
            memory_intensive = analysis.get('memory_intensive_operations', [])
            if memory_intensive:
                unique_memory_ops = list(set(memory_intensive))
                recommendations.append(
                    f"Memory-intensive operations: {', '.join(unique_memory_ops)}. "
                    "Consider data sampling or memory management improvements."
                )
            
            # General recommendations
            if not recommendations:
                recommendations.append("Performance looks good! No specific optimizations needed.")
            
            return recommendations
            
        except Exception as e:
            return [f"Error generating recommendations: {str(e)}"]
    
    def _calculate_performance_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance statistics."""
        try:
            if not self.performance_history:
                return {}
            
            successful_ops = [m for m in self.performance_history if m.success]
            failed_ops = [m for m in self.performance_history if not m.success]
            
            durations = [m.duration_seconds for m in successful_ops if m.duration_seconds]
            memory_changes = [m.memory_after_mb - m.memory_before_mb for m in successful_ops 
                            if m.memory_after_mb and m.memory_before_mb]
            
            statistics = {
                'total_operations': len(self.performance_history),
                'successful_operations': len(successful_ops),
                'failed_operations': len(failed_ops),
                'success_rate': len(successful_ops) / len(self.performance_history) if self.performance_history else 0,
                'duration_stats': {
                    'mean': np.mean(durations) if durations else 0,
                    'median': np.median(durations) if durations else 0,
                    'std': np.std(durations) if durations else 0,
                    'min': min(durations) if durations else 0,
                    'max': max(durations) if durations else 0
                },
                'memory_stats': {
                    'mean_change_mb': np.mean(memory_changes) if memory_changes else 0,
                    'median_change_mb': np.median(memory_changes) if memory_changes else 0,
                    'std_change_mb': np.std(memory_changes) if memory_changes else 0,
                    'min_change_mb': min(memory_changes) if memory_changes else 0,
                    'max_change_mb': max(memory_changes) if memory_changes else 0
                },
                'optimization_stats': {
                    'operations_optimized': sum(1 for m in self.performance_history if m.optimization_applied),
                    'avg_sampling_ratio': np.mean([m.sampling_ratio for m in self.performance_history 
                                                 if m.sampling_ratio]) if any(m.sampling_ratio for m in self.performance_history) else 1.0
                }
            }
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance statistics: {str(e)}")
            return {'error': str(e)}