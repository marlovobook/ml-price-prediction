"""
Property-based tests for system reliability and monitoring.

Feature: stock-direction-predictor, Property 9: System Reliability and Monitoring
Validates: Requirements 7.5, 8.1, 8.4, 8.5
"""

import pytest
import time
import threading
import tempfile
import os
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, settings, HealthCheck
from typing import Dict, Any, List

from stock_predictor.monitoring.reliability_manager import ReliabilityManager
from stock_predictor.monitoring.system_monitor import SystemMonitor, HealthStatus
from stock_predictor.monitoring.metrics_collector import MetricsCollector, MetricType
from stock_predictor.monitoring.model_versioning import ModelVersionManager, ModelVersion
from stock_predictor.monitoring.concurrent_processor import ConcurrentStockProcessor
from stock_predictor.interfaces import ModelConfiguration


# Test data generators
@st.composite
def generate_stock_symbols(draw):
    """Generate realistic stock symbols."""
    symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'SPY']
    return draw(st.lists(st.sampled_from(symbols), min_size=1, max_size=5, unique=True))


@st.composite
def generate_task_parameters(draw):
    """Generate realistic task parameters."""
    return {
        'start_date': draw(st.sampled_from(['2020-01-01', '2021-01-01', '2022-01-01'])),
        'end_date': draw(st.sampled_from(['2023-01-01', '2024-01-01'])),
        'pattern_length': draw(st.sampled_from([3, 5, 7, 14])),
        'indicators': draw(st.lists(
            st.sampled_from(['RSI', 'MACD', 'EMA20', 'EMA50', 'EMA200']),
            min_size=1, max_size=3, unique=True
        ))
    }


@st.composite
def generate_model_config(draw):
    """Generate realistic model configuration."""
    model_type = draw(st.sampled_from(['xgboost', 'random_forest', 'svm']))
    pattern_length = draw(st.sampled_from([3, 5, 7, 14]))
    
    return ModelConfiguration(
        model_type=model_type,
        pattern_length=pattern_length,
        hyperparameters={
            'n_estimators': draw(st.integers(min_value=50, max_value=200)),
            'max_depth': draw(st.integers(min_value=3, max_value=10))
        },
        feature_set=['rsi', 'macd', 'ema_20'],
        version=f"v1.0.{draw(st.integers(min_value=0, max_value=99))}"
    )


@st.composite
def generate_performance_metrics(draw):
    """Generate realistic performance metrics."""
    return {
        'accuracy': draw(st.floats(min_value=0.4, max_value=0.95)),
        'precision': draw(st.floats(min_value=0.3, max_value=0.9)),
        'recall': draw(st.floats(min_value=0.3, max_value=0.9)),
        'f1_score': draw(st.floats(min_value=0.3, max_value=0.9)),
        'mse': draw(st.floats(min_value=0.1, max_value=2.0)),
        'mae': draw(st.floats(min_value=0.1, max_value=1.5))
    }


class MockModel:
    """Mock model for testing."""
    
    def __init__(self):
        self.fitted = False
    
    def fit(self, X, y):
        self.fitted = True
    
    def predict(self, X):
        return [0] * len(X)
    
    def predict_proba(self, X):
        return [[0.33, 0.34, 0.33]] * len(X)


class TestSystemReliabilityAndMonitoring:
    """Property-based tests for system reliability and monitoring."""
    
    @given(st.data())
    @settings(
        max_examples=10,
        deadline=30000,
        suppress_health_check=[HealthCheck.large_base_example]
    )
    def test_property_9_concurrent_processing_reliability(self, data):
        """
        Feature: stock-direction-predictor, Property 9: System Reliability and Monitoring
        
        For any system operation, the system should handle concurrent processing correctly,
        support model versioning and rollback, provide comprehensive error logging,
        and generate operational metrics.
        
        Validates: Requirements 7.5, 8.1, 8.4, 8.5
        """
        # Generate test data
        symbols = data.draw(generate_stock_symbols())
        task_params = data.draw(generate_task_parameters())
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize concurrent processor
            processor = ConcurrentStockProcessor(
                max_workers=2,
                queue_size=10,
                timeout_seconds=5.0
            )
            
            # Mock task processor function
            def mock_task_processor(symbol: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
                # Simulate some processing time
                time.sleep(0.1)
                return {
                    'symbol': symbol,
                    'processed': True,
                    'parameters': parameters,
                    'result_count': len(parameters.get('indicators', []))
                }
            
            # Register task processor
            processor.register_task_processor('analysis', mock_task_processor)
            
            try:
                # Test 1: Concurrent processing support (Requirement 8.1)
                processor.start()
                
                # Submit multiple tasks concurrently
                task_ids = []
                for symbol in symbols:
                    task_id = processor.submit_task(
                        symbol=symbol,
                        task_type='analysis',
                        parameters=task_params,
                        priority=1
                    )
                    task_ids.append(task_id)
                
                # Verify tasks are submitted
                assert len(task_ids) == len(symbols), "All tasks should be submitted"
                assert len(set(task_ids)) == len(task_ids), "Task IDs should be unique"
                
                # Wait for all tasks to complete
                results = processor.wait_for_tasks(task_ids, timeout=10.0)
                
                # Verify all tasks completed successfully
                assert len(results) == len(task_ids), "All tasks should complete"
                for task_id, result in results.items():
                    assert result.success, f"Task {task_id} should succeed"
                    assert result.result is not None, f"Task {task_id} should have result"
                    assert result.processing_time_ms > 0, f"Task {task_id} should have processing time"
                
                # Test 2: Processing statistics and monitoring (Requirement 8.5)
                stats = processor.get_processing_stats()
                
                # Verify statistics are collected
                assert isinstance(stats, dict), "Stats should be dictionary"
                assert stats['tasks_completed'] >= len(symbols), "Should track completed tasks"
                assert stats['tasks_submitted'] >= len(symbols), "Should track submitted tasks"
                assert stats['tasks_failed'] >= 0, "Should track failed tasks"
                assert stats['avg_processing_time_ms'] > 0, "Should calculate average processing time"
                
                # Test 3: Graceful error handling (Requirement 7.5)
                # Submit a task that will fail
                def failing_task_processor(symbol: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
                    raise ValueError(f"Simulated failure for {symbol}")
                
                processor.register_task_processor('failing_analysis', failing_task_processor)
                
                failing_task_id = processor.submit_task(
                    symbol=symbols[0],
                    task_type='failing_analysis',
                    parameters=task_params
                )
                
                # Wait for failing task
                failing_result = processor.get_task_result(failing_task_id, timeout=5.0)
                
                # Verify error is handled gracefully
                assert not failing_result.success, "Failing task should not succeed"
                assert failing_result.error is not None, "Failing task should have error message"
                assert "Simulated failure" in failing_result.error, "Error message should be preserved"
                
                # Verify system continues to work after failure
                recovery_task_id = processor.submit_task(
                    symbol=symbols[0],
                    task_type='analysis',
                    parameters=task_params
                )
                
                recovery_result = processor.get_task_result(recovery_task_id, timeout=5.0)
                assert recovery_result.success, "System should recover from failures"
                
            finally:
                processor.stop(wait=True, timeout=5.0)
    
    @given(st.data())
    @settings(
        max_examples=5,
        deadline=20000,
        suppress_health_check=[HealthCheck.large_base_example]
    )
    def test_property_9_model_versioning_and_rollback(self, data):
        """
        Test model versioning system with rollback capabilities.
        
        Validates: Requirements 8.4 (model versioning and rollback)
        """
        # Generate test data
        model_config = data.draw(generate_model_config())
        performance_metrics = data.draw(generate_performance_metrics())
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize model version manager
            version_manager = ModelVersionManager(base_path=temp_dir, max_versions=5)
            
            # Create mock model
            mock_model = MockModel()
            
            # Test 1: Model saving with versioning
            version_id_1 = version_manager.save_model(
                model=mock_model,
                model_config=model_config,
                performance_metrics=performance_metrics,
                metadata={'test': 'data'}
            )
            
            # Verify version is saved
            assert version_id_1 is not None, "Version ID should be generated"
            assert isinstance(version_id_1, str), "Version ID should be string"
            
            # Test 2: Model loading
            loaded_model, loaded_version = version_manager.load_model(
                model_type=model_config.model_type,
                pattern_length=model_config.pattern_length,
                version_id=version_id_1
            )
            
            # Verify loaded model and version
            assert loaded_model is not None, "Model should be loaded"
            assert isinstance(loaded_version, ModelVersion), "Version metadata should be returned"
            assert loaded_version.version_id == version_id_1, "Version ID should match"
            assert loaded_version.model_type == model_config.model_type, "Model type should match"
            assert loaded_version.pattern_length == model_config.pattern_length, "Pattern length should match"
            
            # Test 3: Multiple versions and rollback capability
            # Save another version with different performance
            modified_config = ModelConfiguration(
                model_type=model_config.model_type,
                pattern_length=model_config.pattern_length,
                hyperparameters={'n_estimators': 150},
                feature_set=model_config.feature_set,
                version=f"v2.0.0"
            )
            
            better_metrics = {k: v * 1.1 for k, v in performance_metrics.items()}
            
            version_id_2 = version_manager.save_model(
                model=mock_model,
                model_config=modified_config,
                performance_metrics=better_metrics
            )
            
            # Set second version as active
            version_manager.set_active_version(
                model_type=model_config.model_type,
                pattern_length=model_config.pattern_length,
                version_id=version_id_2
            )
            
            # Verify active version is set
            active_version = version_manager.get_active_version(
                model_type=model_config.model_type,
                pattern_length=model_config.pattern_length
            )
            assert active_version is not None, "Active version should be set"
            assert active_version.version_id == version_id_2, "Active version should be the second one"
            
            # Test 4: Rollback functionality
            version_manager.rollback_to_version(
                model_type=model_config.model_type,
                pattern_length=model_config.pattern_length,
                version_id=version_id_1
            )
            
            # Verify rollback worked
            active_after_rollback = version_manager.get_active_version(
                model_type=model_config.model_type,
                pattern_length=model_config.pattern_length
            )
            assert active_after_rollback.version_id == version_id_1, "Should rollback to first version"
            
            # Test 5: Version listing and comparison
            versions = version_manager.list_versions(
                model_type=model_config.model_type,
                pattern_length=model_config.pattern_length
            )
            
            # Verify version listing
            assert len(versions) == 2, "Should have two versions"
            assert all(isinstance(v, ModelVersion) for v in versions), "All should be ModelVersion objects"
            
            # Verify versions are sorted by creation date (newest first)
            assert versions[0].created_at >= versions[1].created_at, "Versions should be sorted by date"
            
            # Test version comparison
            comparison = version_manager.get_version_comparison(
                model_type=model_config.model_type,
                pattern_length=model_config.pattern_length,
                version_ids=[version_id_1, version_id_2]
            )
            
            assert isinstance(comparison, dict), "Comparison should be dictionary"
            assert 'versions' in comparison, "Should include version details"
            assert 'performance_comparison' in comparison, "Should include performance comparison"
            assert len(comparison['versions']) == 2, "Should compare both versions"
    
    @given(st.data())
    @settings(
        max_examples=5,
        deadline=15000,
        suppress_health_check=[HealthCheck.large_base_example]
    )
    def test_property_9_system_monitoring_and_health_checks(self, data):
        """
        Test system monitoring and health check functionality.
        
        Validates: Requirements 8.5 (operational visibility and monitoring)
        """
        # Initialize system monitor
        monitor = SystemMonitor(check_interval=1.0, memory_threshold_gb=1.0)
        
        # Test 1: Health check registration and execution
        def custom_health_check():
            from stock_predictor.monitoring.system_monitor import HealthCheck, HealthStatus
            return HealthCheck(
                component="test_component",
                status=HealthStatus.HEALTHY,
                message="Test component is working"
            )
        
        # Register custom health check
        monitor.register_health_check("test_check", custom_health_check)
        
        # Run health checks
        health_status = monitor.run_health_checks()
        
        # Verify health check results
        assert health_status is not None, "Health status should be returned"
        assert hasattr(health_status, 'overall_status'), "Should have overall status"
        assert hasattr(health_status, 'checks'), "Should have individual checks"
        assert len(health_status.checks) > 0, "Should have at least one health check"
        
        # Verify our custom check is included
        test_check = next((check for check in health_status.checks if check.component == "test_component"), None)
        assert test_check is not None, "Custom health check should be included"
        assert test_check.status == HealthStatus.HEALTHY, "Custom check should be healthy"
        
        # Test 2: Default system health checks
        default_checks = ['memory_usage', 'disk_space', 'cpu_usage', 'system_load']
        check_components = [check.component for check in health_status.checks]
        
        for default_check in default_checks:
            assert default_check in check_components, f"Should include {default_check} health check"
        
        # Test 3: Health check response times
        for check in health_status.checks:
            assert hasattr(check, 'response_time_ms'), "Each check should have response time"
            if check.response_time_ms is not None:
                assert check.response_time_ms >= 0, "Response time should be non-negative"
        
        # Test 4: System health status determination
        assert health_status.overall_status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.CRITICAL], \
            "Overall status should be valid"
        
        # Test critical and warning issue detection
        critical_issues = health_status.critical_issues
        warnings = health_status.warnings
        
        assert isinstance(critical_issues, list), "Critical issues should be list"
        assert isinstance(warnings, list), "Warnings should be list"
        
        # If there are critical issues, overall status should reflect that
        if critical_issues:
            assert health_status.overall_status == HealthStatus.CRITICAL, \
                "Overall status should be critical if there are critical issues"
    
    @given(st.data())
    @settings(
        max_examples=5,
        deadline=10000,
        suppress_health_check=[HealthCheck.large_base_example]
    )
    def test_property_9_metrics_collection_and_operational_visibility(self, data):
        """
        Test metrics collection and operational visibility.
        
        Validates: Requirements 8.5 (metrics collection and operational visibility)
        """
        # Initialize metrics collector
        collector = MetricsCollector(retention_hours=1, collection_interval=0.5)
        
        # Generate test metric values
        counter_value = data.draw(st.floats(min_value=1.0, max_value=100.0))
        gauge_value = data.draw(st.floats(min_value=0.0, max_value=1000.0))
        histogram_values = data.draw(st.lists(st.floats(min_value=0.1, max_value=10.0), min_size=5, max_size=20))
        timer_values = data.draw(st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=3, max_size=10))
        
        # Test 1: Metric recording
        collector.increment_counter("test_counter", counter_value)
        collector.set_gauge("test_gauge", gauge_value)
        
        for value in histogram_values:
            collector.record_histogram("test_histogram", value)
        
        for value in timer_values:
            collector.record_timer("test_timer", value)
        
        # Test 2: Metric collection and aggregation
        system_metrics = collector.collect_metrics()
        
        # Verify metrics are collected
        assert system_metrics is not None, "System metrics should be returned"
        assert hasattr(system_metrics, 'metrics'), "Should have metrics list"
        assert hasattr(system_metrics, 'timestamp'), "Should have timestamp"
        assert len(system_metrics.metrics) > 0, "Should have collected metrics"
        
        # Test 3: Metric retrieval and validation
        counter_metric = system_metrics.get_metric("test_counter")
        gauge_metric = system_metrics.get_metric("test_gauge")
        
        assert counter_metric is not None, "Counter metric should be found"
        assert gauge_metric is not None, "Gauge metric should be found"
        
        assert counter_metric.metric_type == MetricType.COUNTER, "Should be counter type"
        assert gauge_metric.metric_type == MetricType.GAUGE, "Should be gauge type"
        
        assert counter_metric.value == counter_value, "Counter value should match"
        assert gauge_metric.value == gauge_value, "Gauge value should match"
        
        # Test 4: Histogram and timer statistics
        histogram_metrics = [m for m in system_metrics.metrics if m.name.startswith("test_histogram")]
        timer_metrics = [m for m in system_metrics.metrics if m.name.startswith("test_timer")]
        
        # Should have mean, min, max, count for both histogram and timer
        histogram_names = [m.name for m in histogram_metrics]
        timer_names = [m.name for m in timer_metrics]
        
        expected_stats = ['.mean', '.min', '.max', '.count']
        for stat in expected_stats:
            assert any(name.endswith(stat) for name in histogram_names), f"Should have histogram{stat}"
            assert any(name.endswith(stat) for name in timer_names), f"Should have timer{stat}"
        
        # Test 5: Custom metric collectors
        def custom_collector():
            from stock_predictor.monitoring.metrics_collector import Metric, MetricType
            from datetime import datetime
            return [
                Metric("custom_metric", 42.0, MetricType.GAUGE, datetime.now())
            ]
        
        collector.register_custom_collector("test_collector", custom_collector)
        
        # Collect metrics again
        updated_metrics = collector.collect_metrics()
        
        # Verify custom metric is included
        custom_metric = updated_metrics.get_metric("custom_metric")
        assert custom_metric is not None, "Custom metric should be collected"
        assert custom_metric.value == 42.0, "Custom metric value should be correct"
        
        # Test 6: Timer context manager
        with collector.timer_context("context_timer"):
            time.sleep(0.01)  # Small delay to ensure measurable time
        
        # Collect metrics to get timer result
        final_metrics = collector.collect_metrics()
        context_timer_metrics = [m for m in final_metrics.metrics if m.name.startswith("context_timer")]
        
        assert len(context_timer_metrics) > 0, "Context timer should record metrics"
        
        # Find the mean timing metric
        mean_timer = next((m for m in context_timer_metrics if m.name.endswith('.mean')), None)
        assert mean_timer is not None, "Should have mean timer metric"
        assert mean_timer.value > 0, "Timer should record positive duration"


if __name__ == "__main__":
    pytest.main([__file__])