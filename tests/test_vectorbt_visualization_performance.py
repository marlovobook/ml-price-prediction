"""
Performance benchmarking tests for VectorBT Visualization Enhancement.

This module tests visualization generation times across different dataset sizes,
optimizes memory usage and processing efficiency, and creates performance 
monitoring and alerting systems.
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from unittest.mock import Mock, patch
import gc

# Import visualization components
from stock_predictor.visualization import (
    VectorBTVisualizationEngine,
    EnhancedPortfolioEngine,
    SignalAlignmentEngine,
    PortfolioConfig,
    PlotConfig
)


@pytest.mark.benchmark
@pytest.mark.visualization
@pytest.mark.slow
class TestVectorBTVisualizationPerformance:
    """Performance benchmarking tests for VectorBT visualization enhancement."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Performance tracking
        self.performance_results = {}
        self.memory_samples = []
        
        # Initialize components with performance-optimized settings
        self.portfolio_config = PortfolioConfig(
            init_cash=100000.0,
            fees=0.0025,
            slippage=0.0025,
            size_strategy='fixed_amount',
            size_value=10000.0
        )
        
        self.plot_config = PlotConfig(
            width=800,  # Smaller for performance testing
            height=400,
            show_trades=True,
            show_positions=False  # Disable for performance
        )
        
        self.viz_engine = VectorBTVisualizationEngine(
            portfolio_config=self.portfolio_config,
            plot_config=self.plot_config,
            enable_performance_optimization=True
        )
        
        self.portfolio_engine = EnhancedPortfolioEngine(
            portfolio_config=self.portfolio_config
        )
        
        self.signal_aligner = SignalAlignmentEngine()
        
        self.logger = logging.getLogger(__name__)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Save performance results
        if self.performance_results:
            results_file = Path(self.temp_dir).parent / "visualization_performance_results.json"
            try:
                with open(results_file, 'w') as f:
                    json.dump(self.performance_results, f, indent=2, default=str)
                self.logger.info(f"Performance results saved to: {results_file}")
            except Exception as e:
                self.logger.warning(f"Could not save performance results: {e}")
    
    def _create_benchmark_data(self, num_days: int, symbol: str = "BENCHMARK") -> pd.DataFrame:
        """Create benchmark market data of specified size."""
        dates = pd.date_range(start="2023-01-01", periods=num_days, freq='D')
        
        np.random.seed(hash(symbol + str(num_days)) % 2**32)
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, num_days)
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            open_price = close * (1 + np.random.normal(0, 0.005))
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
            volume = int(np.random.lognormal(15, 0.5))
            
            data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume,
                'Adj Close': close * (1 + np.random.normal(0, 0.001))
            })
        
        return pd.DataFrame(data, index=dates)
    
    def _create_benchmark_predictions(self, num_predictions: int, strategy: str = "balanced") -> np.ndarray:
        """Create benchmark predictions with different strategies."""
        np.random.seed(42)
        
        if strategy == "conservative":
            probabilities = [0.1, 0.8, 0.1]  # Mostly hold
        elif strategy == "aggressive":
            probabilities = [0.3, 0.4, 0.3]  # More trading
        else:  # balanced
            probabilities = [0.2, 0.6, 0.2]  # Balanced approach
        
        return np.random.choice([0, 1, 2], size=num_predictions, p=probabilities)
    
    def _measure_performance_metrics(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Measure comprehensive performance metrics for a function call."""
        process = psutil.Process(os.getpid())
        
        # Initial measurements
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu_percent = process.cpu_percent()
        
        # Force garbage collection before measurement
        gc.collect()
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        # Final measurements
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        end_cpu_percent = process.cpu_percent()
        
        # Force garbage collection after measurement
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'execution_time': end_time - start_time,
            'memory_usage_mb': end_memory - start_memory,
            'peak_memory_mb': end_memory,
            'final_memory_mb': final_memory,
            'memory_cleanup_mb': end_memory - final_memory,
            'cpu_percent': end_cpu_percent,
            'success': success,
            'error': error,
            'result': result
        }
    
    def test_visualization_generation_scalability(self):
        """
        Test visualization generation times across different dataset sizes.
        
        This test validates Requirements 9.1, 9.3 by measuring performance
        across various dataset sizes and ensuring scalability.
        """
        dataset_sizes = [30, 90, 180, 365, 730, 1460]  # Days of data
        prediction_ratios = [0.2, 0.3, 0.4]  # Ratio of predictions to total data
        
        scalability_results = {}
        
        for num_days in dataset_sizes:
            for pred_ratio in prediction_ratios:
                test_name = f"{num_days}d_{int(pred_ratio*100)}p"
                
                # Create test data
                price_data = self._create_benchmark_data(num_days)
                num_predictions = max(10, int(num_days * pred_ratio))
                predictions = self._create_benchmark_predictions(num_predictions)
                test_start_idx = num_days - num_predictions
                
                # Measure signal alignment performance
                alignment_metrics = self._measure_performance_metrics(
                    self.signal_aligner.align_predictions_to_timeline,
                    predictions, price_data, test_start_idx
                )
                
                if not alignment_metrics['success']:
                    self.logger.warning(f"Signal alignment failed for {test_name}: {alignment_metrics['error']}")
                    continue
                
                # Measure portfolio creation performance
                portfolio_metrics = self._measure_performance_metrics(
                    self.portfolio_engine.create_portfolio_from_predictions,
                    predictions, price_data, test_start_idx
                )
                
                if not portfolio_metrics['success']:
                    self.logger.warning(f"Portfolio creation failed for {test_name}: {portfolio_metrics['error']}")
                    continue
                
                portfolio = portfolio_metrics['result'].portfolio
                
                # Measure visualization generation performance
                viz_metrics = self._measure_performance_metrics(
                    self.viz_engine.generate_portfolio_plot,
                    portfolio, f"Performance Test - {test_name}"
                )
                
                # Measure drawdown visualization performance
                drawdown_metrics = self._measure_performance_metrics(
                    self.viz_engine.generate_drawdown_plot,
                    portfolio
                )
                
                # Compile results
                scalability_results[test_name] = {
                    'dataset_size': num_days,
                    'prediction_count': num_predictions,
                    'prediction_ratio': pred_ratio,
                    'alignment_time': alignment_metrics['execution_time'],
                    'alignment_memory': alignment_metrics['memory_usage_mb'],
                    'portfolio_time': portfolio_metrics['execution_time'],
                    'portfolio_memory': portfolio_metrics['memory_usage_mb'],
                    'visualization_time': viz_metrics['execution_time'] if viz_metrics['success'] else None,
                    'visualization_memory': viz_metrics['memory_usage_mb'] if viz_metrics['success'] else None,
                    'drawdown_time': drawdown_metrics['execution_time'] if drawdown_metrics['success'] else None,
                    'drawdown_memory': drawdown_metrics['memory_usage_mb'] if drawdown_metrics['success'] else None,
                    'total_time': (
                        alignment_metrics['execution_time'] + 
                        portfolio_metrics['execution_time'] + 
                        (viz_metrics['execution_time'] if viz_metrics['success'] else 0) +
                        (drawdown_metrics['execution_time'] if drawdown_metrics['success'] else 0)
                    ),
                    'success': all([
                        alignment_metrics['success'],
                        portfolio_metrics['success'],
                        viz_metrics['success'],
                        drawdown_metrics['success']
                    ])
                }
                
                # Performance assertions (Requirements 9.1, 9.3)
                if scalability_results[test_name]['success']:
                    total_time = scalability_results[test_name]['total_time']
                    
                    # Time should scale reasonably with dataset size
                    time_per_day = total_time / num_days
                    assert time_per_day < 0.1, f"Processing time per day ({time_per_day:.4f}s) too high for {test_name}"
                    
                    # Visualization should complete within reasonable time limits
                    if num_days <= 365:  # 1 year
                        assert total_time < 30, f"Total processing time ({total_time:.2f}s) too high for {test_name}"
                    elif num_days <= 730:  # 2 years
                        assert total_time < 60, f"Total processing time ({total_time:.2f}s) too high for {test_name}"
                    else:  # > 2 years
                        assert total_time < 120, f"Total processing time ({total_time:.2f}s) too high for {test_name}"
        
        self.performance_results['scalability'] = scalability_results
        
        # Analyze scaling characteristics
        successful_results = {k: v for k, v in scalability_results.items() if v['success']}
        if len(successful_results) >= 3:
            sizes = [v['dataset_size'] for v in successful_results.values()]
            times = [v['total_time'] for v in successful_results.values()]
            
            # Check that time doesn't scale exponentially
            for i in range(1, len(sizes)):
                size_ratio = sizes[i] / sizes[0]
                time_ratio = times[i] / times[0]
                scaling_factor = time_ratio / size_ratio
                
                # Should scale better than linearly due to optimizations
                assert scaling_factor < 2.0, f"Poor scaling detected: {scaling_factor:.2f}x"
        
        self.logger.info(f"Scalability test completed: {len(successful_results)}/{len(scalability_results)} scenarios successful")
    
    def test_memory_usage_optimization(self):
        """
        Test memory usage and optimization effectiveness.
        
        This test validates memory efficiency and garbage collection
        for visualization operations.
        """
        # Test with progressively larger datasets
        dataset_sizes = [100, 500, 1000, 2000]
        memory_results = {}
        
        for num_days in dataset_sizes:
            test_name = f"memory_{num_days}d"
            
            # Monitor memory throughout the process
            process = psutil.Process(os.getpid())
            memory_timeline = []
            
            def record_memory(stage: str):
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_timeline.append({'stage': stage, 'memory_mb': memory_mb, 'timestamp': time.time()})
            
            record_memory('start')
            
            # Create large dataset
            price_data = self._create_benchmark_data(num_days)
            record_memory('data_created')
            
            predictions = self._create_benchmark_predictions(int(num_days * 0.3))
            test_start_idx = int(num_days * 0.7)
            record_memory('predictions_created')
            
            # Signal alignment
            aligned_signals = self.signal_aligner.align_predictions_to_timeline(
                predictions, price_data, test_start_idx
            )
            record_memory('signals_aligned')
            
            # Portfolio creation
            portfolio_result = self.portfolio_engine.create_portfolio_from_predictions(
                predictions, price_data, test_start_idx
            )
            record_memory('portfolio_created')
            
            if portfolio_result.success:
                portfolio = portfolio_result.portfolio
                
                # Visualization generation
                viz_result = self.viz_engine.generate_portfolio_plot(portfolio)
                record_memory('visualization_created')
                
                # Force cleanup
                del viz_result
                gc.collect()
                record_memory('after_viz_cleanup')
                
                # Drawdown visualization
                drawdown_result = self.viz_engine.generate_drawdown_plot(portfolio)
                record_memory('drawdown_created')
                
                # Force cleanup
                del drawdown_result
                del portfolio
                gc.collect()
                record_memory('after_drawdown_cleanup')
            
            # Final cleanup
            del price_data, predictions, aligned_signals, portfolio_result
            gc.collect()
            record_memory('final_cleanup')
            
            # Analyze memory usage
            memory_results[test_name] = {
                'dataset_size': num_days,
                'memory_timeline': memory_timeline,
                'peak_memory': max(m['memory_mb'] for m in memory_timeline),
                'final_memory': memory_timeline[-1]['memory_mb'],
                'initial_memory': memory_timeline[0]['memory_mb'],
                'memory_growth': memory_timeline[-1]['memory_mb'] - memory_timeline[0]['memory_mb'],
                'cleanup_effectiveness': (
                    memory_timeline[-3]['memory_mb'] - memory_timeline[-1]['memory_mb']
                ) if len(memory_timeline) >= 3 else 0
            }
            
            # Memory efficiency assertions
            peak_memory = memory_results[test_name]['peak_memory']
            memory_growth = memory_results[test_name]['memory_growth']
            
            # Peak memory should be reasonable for dataset size
            memory_per_day = peak_memory / num_days
            assert memory_per_day < 10.0, f"Memory per day ({memory_per_day:.3f}MB) too high for {test_name}"
            
            # Memory growth should be controlled (not indicating major leaks)
            assert memory_growth < 500, f"Memory growth ({memory_growth:.1f}MB) suggests potential leak for {test_name}"
            
            # Cleanup should be effective
            cleanup_effectiveness = memory_results[test_name]['cleanup_effectiveness']
            if cleanup_effectiveness > 0:
                assert cleanup_effectiveness > 10, f"Poor memory cleanup ({cleanup_effectiveness:.1f}MB) for {test_name}"
        
        self.performance_results['memory_usage'] = memory_results
        
        self.logger.info(f"Memory optimization test completed for {len(dataset_sizes)} dataset sizes")
    
    def test_concurrent_visualization_performance(self):
        """
        Test performance under concurrent visualization requests.
        
        This test validates system behavior when multiple visualizations
        are generated simultaneously.
        """
        import concurrent.futures
        import threading
        
        # Create test scenarios
        scenarios = [
            {'name': 'small', 'days': 90, 'predictions': 18},
            {'name': 'medium', 'days': 180, 'predictions': 36},
            {'name': 'large', 'days': 365, 'predictions': 73}
        ]
        
        def create_visualization(scenario: Dict[str, Any]) -> Dict[str, Any]:
            """Create a single visualization and measure performance."""
            start_time = time.time()
            thread_id = threading.current_thread().ident
            
            try:
                # Create data
                price_data = self._create_benchmark_data(scenario['days'], f"CONC_{thread_id}")
                predictions = self._create_benchmark_predictions(scenario['predictions'])
                test_start_idx = scenario['days'] - scenario['predictions']
                
                # Create portfolio
                portfolio_result = self.portfolio_engine.create_portfolio_from_predictions(
                    predictions, price_data, test_start_idx
                )
                
                if not portfolio_result.success:
                    return {
                        'scenario': scenario['name'],
                        'thread_id': thread_id,
                        'success': False,
                        'error': portfolio_result.error_message,
                        'execution_time': time.time() - start_time
                    }
                
                # Create visualization
                viz_result = self.viz_engine.generate_portfolio_plot(
                    portfolio_result.portfolio, f"Concurrent Test - {scenario['name']} - {thread_id}"
                )
                
                return {
                    'scenario': scenario['name'],
                    'thread_id': thread_id,
                    'success': viz_result.success,
                    'error': viz_result.error_message if not viz_result.success else None,
                    'execution_time': time.time() - start_time,
                    'visualization_time': viz_result.generation_time
                }
                
            except Exception as e:
                return {
                    'scenario': scenario['name'],
                    'thread_id': thread_id,
                    'success': False,
                    'error': str(e),
                    'execution_time': time.time() - start_time
                }
        
        # Test with different concurrency levels
        concurrency_levels = [1, 2, 4]
        concurrent_results = {}
        
        for max_workers in concurrency_levels:
            test_name = f"concurrent_{max_workers}w"
            
            # Create tasks (multiple scenarios per worker)
            tasks = []
            for _ in range(max_workers):
                for scenario in scenarios:
                    tasks.append(scenario)
            
            # Execute concurrently
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_scenario = {
                    executor.submit(create_visualization, scenario): scenario 
                    for scenario in tasks
                }
                
                results = []
                for future in concurrent.futures.as_completed(future_to_scenario):
                    result = future.result()
                    results.append(result)
            
            total_time = time.time() - start_time
            
            # Analyze results
            successful_results = [r for r in results if r['success']]
            failed_results = [r for r in results if not r['success']]
            
            concurrent_results[test_name] = {
                'max_workers': max_workers,
                'total_tasks': len(tasks),
                'successful_tasks': len(successful_results),
                'failed_tasks': len(failed_results),
                'total_execution_time': total_time,
                'average_task_time': np.mean([r['execution_time'] for r in successful_results]) if successful_results else 0,
                'max_task_time': max([r['execution_time'] for r in successful_results]) if successful_results else 0,
                'min_task_time': min([r['execution_time'] for r in successful_results]) if successful_results else 0,
                'success_rate': len(successful_results) / len(tasks) if tasks else 0,
                'results': results
            }
            
            # Performance assertions
            success_rate = concurrent_results[test_name]['success_rate']
            assert success_rate >= 0.8, f"Low success rate ({success_rate:.2%}) for {test_name}"
            
            # Concurrent execution should not cause extreme slowdown
            avg_task_time = concurrent_results[test_name]['average_task_time']
            if max_workers == 1:
                baseline_time = avg_task_time
            else:
                # Allow some overhead for concurrency, but not extreme slowdown
                slowdown_factor = avg_task_time / baseline_time if 'baseline_time' in locals() else 1.0
                assert slowdown_factor < 3.0, f"Extreme slowdown ({slowdown_factor:.2f}x) for {test_name}"
        
        self.performance_results['concurrent_performance'] = concurrent_results
        
        self.logger.info(f"Concurrent performance test completed for {len(concurrency_levels)} concurrency levels")
    
    def test_performance_monitoring_and_alerting(self):
        """
        Test performance monitoring and alerting system.
        
        This test validates that the system can monitor its own performance
        and provide alerts when performance degrades.
        """
        # Create performance monitoring system
        performance_monitor = VisualizationPerformanceMonitor()
        
        # Test scenarios with different performance characteristics
        test_scenarios = [
            {'name': 'fast', 'days': 50, 'predictions': 10, 'expected_time': 5.0},
            {'name': 'medium', 'days': 200, 'predictions': 40, 'expected_time': 15.0},
            {'name': 'slow', 'days': 500, 'predictions': 100, 'expected_time': 30.0}
        ]
        
        monitoring_results = {}
        
        for scenario in test_scenarios:
            test_name = scenario['name']
            
            # Create test data
            price_data = self._create_benchmark_data(scenario['days'])
            predictions = self._create_benchmark_predictions(scenario['predictions'])
            test_start_idx = scenario['days'] - scenario['predictions']
            
            # Monitor performance
            with performance_monitor.monitor_operation(f"visualization_{test_name}") as monitor:
                # Create portfolio
                portfolio_result = self.portfolio_engine.create_portfolio_from_predictions(
                    predictions, price_data, test_start_idx
                )
                
                monitor.checkpoint("portfolio_created")
                
                if portfolio_result.success:
                    # Create visualization
                    viz_result = self.viz_engine.generate_portfolio_plot(
                        portfolio_result.portfolio, f"Monitoring Test - {test_name}"
                    )
                    
                    monitor.checkpoint("visualization_created")
                    
                    # Create drawdown plot
                    drawdown_result = self.viz_engine.generate_drawdown_plot(
                        portfolio_result.portfolio
                    )
                    
                    monitor.checkpoint("drawdown_created")
            
            # Get monitoring results
            operation_metrics = performance_monitor.get_operation_metrics(f"visualization_{test_name}")
            
            monitoring_results[test_name] = {
                'scenario': scenario,
                'metrics': operation_metrics,
                'total_time': operation_metrics['total_duration'],
                'checkpoints': operation_metrics['checkpoints'],
                'memory_usage': operation_metrics['peak_memory_mb'],
                'performance_alerts': performance_monitor.check_performance_alerts(
                    f"visualization_{test_name}", scenario['expected_time']
                )
            }
            
            # Validate monitoring functionality
            assert operation_metrics['total_duration'] > 0
            assert len(operation_metrics['checkpoints']) >= 2
            assert operation_metrics['peak_memory_mb'] > 0
            
            # Check performance alerts
            alerts = monitoring_results[test_name]['performance_alerts']
            if operation_metrics['total_duration'] > scenario['expected_time']:
                assert len(alerts) > 0, f"Expected performance alert for slow {test_name} scenario"
            
        self.performance_results['performance_monitoring'] = monitoring_results
        
        self.logger.info(f"Performance monitoring test completed for {len(test_scenarios)} scenarios")


class VisualizationPerformanceMonitor:
    """Simple performance monitoring system for visualization operations."""
    
    def __init__(self):
        self.operations = {}
        self.current_operation = None
    
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring an operation."""
        return PerformanceMonitorContext(self, operation_name)
    
    def start_operation(self, operation_name: str):
        """Start monitoring an operation."""
        self.current_operation = operation_name
        self.operations[operation_name] = {
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024,
            'checkpoints': [],
            'end_time': None,
            'peak_memory_mb': 0
        }
    
    def checkpoint(self, checkpoint_name: str):
        """Record a checkpoint in the current operation."""
        if self.current_operation and self.current_operation in self.operations:
            current_time = time.time()
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            operation = self.operations[self.current_operation]
            operation['checkpoints'].append({
                'name': checkpoint_name,
                'time': current_time,
                'elapsed': current_time - operation['start_time'],
                'memory_mb': current_memory
            })
            
            # Update peak memory
            operation['peak_memory_mb'] = max(operation['peak_memory_mb'], current_memory)
    
    def end_operation(self):
        """End monitoring the current operation."""
        if self.current_operation and self.current_operation in self.operations:
            self.operations[self.current_operation]['end_time'] = time.time()
            self.current_operation = None
    
    def get_operation_metrics(self, operation_name: str) -> Dict[str, Any]:
        """Get metrics for a completed operation."""
        if operation_name not in self.operations:
            return {}
        
        operation = self.operations[operation_name]
        
        return {
            'total_duration': operation['end_time'] - operation['start_time'] if operation['end_time'] else 0,
            'start_memory_mb': operation['start_memory'],
            'peak_memory_mb': operation['peak_memory_mb'],
            'checkpoints': operation['checkpoints'],
            'checkpoint_count': len(operation['checkpoints'])
        }
    
    def check_performance_alerts(self, operation_name: str, expected_time: float) -> List[str]:
        """Check for performance alerts."""
        alerts = []
        
        if operation_name not in self.operations:
            return alerts
        
        metrics = self.get_operation_metrics(operation_name)
        
        if metrics['total_duration'] > expected_time:
            alerts.append(f"Operation exceeded expected time: {metrics['total_duration']:.2f}s > {expected_time:.2f}s")
        
        if metrics['peak_memory_mb'] > 1000:  # 1GB
            alerts.append(f"High memory usage detected: {metrics['peak_memory_mb']:.1f}MB")
        
        return alerts


class PerformanceMonitorContext:
    """Context manager for performance monitoring."""
    
    def __init__(self, monitor: VisualizationPerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
    
    def __enter__(self):
        self.monitor.start_operation(self.operation_name)
        return self.monitor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monitor.end_operation()


if __name__ == "__main__":
    # Run performance benchmarking tests
    pytest.main([__file__, "-v", "-m", "benchmark"])