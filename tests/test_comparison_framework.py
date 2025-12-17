"""
Tests for the comprehensive comparison framework.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os

from stock_predictor.evaluation.comparison_framework import (
    ComparisonFramework, ComparisonResult, StatisticalTest
)
from stock_predictor.interfaces import BacktestResult


class TestComparisonFramework:
    """Test cases for the ComparisonFramework class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.framework = ComparisonFramework(confidence_level=0.95)
        
        # Create sample results for testing
        self.sample_results = [
            {
                'model_type': 'XGBoost',
                'pattern_length': 3,
                'financial_metrics': {
                    'total_return': 0.15,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': -0.08,
                    'win_rate': 0.65,
                    'volatility': 0.12
                },
                'prediction_metrics': {
                    'accuracy': 0.72,
                    'mse': 0.05,
                    'mae': 0.18,
                    'rmse': 0.22
                },
                'composite_score': 0.85
            },
            {
                'model_type': 'RandomForest',
                'pattern_length': 5,
                'financial_metrics': {
                    'total_return': 0.12,
                    'sharpe_ratio': 1.0,
                    'max_drawdown': -0.06,
                    'win_rate': 0.68,
                    'volatility': 0.10
                },
                'prediction_metrics': {
                    'accuracy': 0.70,
                    'mse': 0.06,
                    'mae': 0.20,
                    'rmse': 0.24
                },
                'composite_score': 0.78
            },
            {
                'model_type': 'SVM',
                'pattern_length': 7,
                'financial_metrics': {
                    'total_return': 0.10,
                    'sharpe_ratio': 0.8,
                    'max_drawdown': -0.10,
                    'win_rate': 0.60,
                    'volatility': 0.15
                },
                'prediction_metrics': {
                    'accuracy': 0.68,
                    'mse': 0.08,
                    'mae': 0.25,
                    'rmse': 0.28
                },
                'composite_score': 0.65
            }
        ]
    
    def test_initialization(self):
        """Test framework initialization."""
        framework = ComparisonFramework(confidence_level=0.99)
        assert framework.confidence_level == 0.99
        assert abs(framework.alpha - 0.01) < 1e-10  # Handle floating point precision
        assert framework.performance_evaluator is not None
    
    def test_compare_all_combinations_success(self):
        """Test successful comparison of all combinations."""
        result = self.framework.compare_all_combinations(self.sample_results)
        
        # Check structure
        assert 'executive_summary' in result
        assert 'detailed_results' in result
        assert 'pattern_length_analysis' in result
        assert 'model_type_analysis' in result
        assert 'statistical_tests' in result
        assert 'recommendations' in result
        
        # Check executive summary
        summary = result['executive_summary']
        assert summary['total_configurations'] == 3
        assert 'best_configuration' in summary
        
        # Check detailed results
        detailed = result['detailed_results']
        assert len(detailed) == 3
        assert all('recommendation_score' in r for r in detailed)
        
        # Check recommendations
        recommendations = result['recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    def test_compare_insufficient_data(self):
        """Test comparison with insufficient data."""
        result = self.framework.compare_all_combinations([self.sample_results[0]])
        assert 'error' in result
        assert 'Insufficient data' in result['error']
    
    def test_extract_performance_metrics(self):
        """Test performance metrics extraction."""
        result = self.sample_results[0]
        metrics = self.framework._extract_performance_metrics(result)
        
        # Should combine financial and prediction metrics
        assert 'total_return' in metrics
        assert 'accuracy' in metrics
        assert 'composite_score' in metrics
        assert metrics['total_return'] == 0.15
        assert metrics['accuracy'] == 0.72
    
    def test_calculate_recommendation_score(self):
        """Test recommendation score calculation."""
        result = self.sample_results[0]
        score = self.framework._calculate_recommendation_score(result)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
    
    def test_generate_visualization_data(self):
        """Test visualization data generation."""
        # Create comparison results
        comparison_results = []
        for result in self.sample_results:
            comp_result = ComparisonResult(
                model_type=result['model_type'],
                pattern_length=result['pattern_length'],
                performance_metrics=self.framework._extract_performance_metrics(result),
                statistical_significance={},
                rank=1,
                recommendation_score=self.framework._calculate_recommendation_score(result)
            )
            comparison_results.append(comp_result)
        
        viz_data = self.framework.generate_visualization_data(comparison_results)
        
        assert 'performance_comparison' in viz_data
        assert 'heatmap_data' in viz_data
        assert 'risk_return_scatter' in viz_data
        assert 'ranking_data' in viz_data
        
        # Check performance comparison data
        perf_data = viz_data['performance_comparison']
        assert len(perf_data) == 3
        assert all('model_type' in item for item in perf_data)
        assert all('pattern_length' in item for item in perf_data)
    
    def test_select_best_configuration(self):
        """Test best configuration selection."""
        # Create comparison results
        comparison_results = []
        for i, result in enumerate(self.sample_results):
            comp_result = ComparisonResult(
                model_type=result['model_type'],
                pattern_length=result['pattern_length'],
                performance_metrics=self.framework._extract_performance_metrics(result),
                statistical_significance={},
                rank=i + 1,
                recommendation_score=self.framework._calculate_recommendation_score(result)
            )
            comparison_results.append(comp_result)
        
        best_config = self.framework.select_best_configuration(comparison_results)
        
        assert isinstance(best_config, ComparisonResult)
        assert best_config.model_type in ['XGBoost', 'RandomForest', 'SVM']
        assert best_config.pattern_length in [3, 5, 7]
    
    def test_select_best_configuration_empty_list(self):
        """Test best configuration selection with empty list."""
        with pytest.raises(ValueError, match="No comparison results provided"):
            self.framework.select_best_configuration([])
    
    def test_create_performance_charts(self):
        """Test performance chart creation."""
        # Create comparison results
        comparison_results = []
        for result in self.sample_results:
            comp_result = ComparisonResult(
                model_type=result['model_type'],
                pattern_length=result['pattern_length'],
                performance_metrics=self.framework._extract_performance_metrics(result),
                statistical_significance={},
                rank=1,
                recommendation_score=self.framework._calculate_recommendation_score(result)
            )
            comparison_results.append(comp_result)
        
        # Test without saving (should not raise errors)
        chart_paths = self.framework.create_performance_charts(comparison_results)
        assert isinstance(chart_paths, dict)
        
        # Test with saving to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            chart_paths = self.framework.create_performance_charts(comparison_results, temp_dir)
            
            # Check that files were created
            expected_charts = ['recommendation_scores', 'risk_return_analysis', 'performance_heatmap']
            for chart_name in expected_charts:
                if chart_name in chart_paths:
                    assert os.path.exists(chart_paths[chart_name])
    
    def test_statistical_tests_integration(self):
        """Test statistical tests integration."""
        # This is a basic integration test to ensure statistical tests don't crash
        statistical_tests = self.framework._perform_statistical_tests(self.sample_results)
        
        assert isinstance(statistical_tests, dict)
        # Should have pairwise tests at minimum
        assert 'pairwise_tests' in statistical_tests
    
    def test_pattern_performance_analysis(self):
        """Test pattern performance analysis."""
        comparison_results = []
        for result in self.sample_results:
            comp_result = ComparisonResult(
                model_type=result['model_type'],
                pattern_length=result['pattern_length'],
                performance_metrics=self.framework._extract_performance_metrics(result),
                statistical_significance={},
                rank=1,
                recommendation_score=self.framework._calculate_recommendation_score(result)
            )
            comparison_results.append(comp_result)
        
        pattern_analysis = self.framework._analyze_pattern_performance(comparison_results)
        
        assert isinstance(pattern_analysis, dict)
        # Should have entries for each pattern length
        expected_keys = ['3_day', '5_day', '7_day']
        for key in expected_keys:
            if key in pattern_analysis:
                assert 'avg_recommendation_score' in pattern_analysis[key]
                assert 'best_model' in pattern_analysis[key]
    
    def test_model_performance_analysis(self):
        """Test model performance analysis."""
        comparison_results = []
        for result in self.sample_results:
            comp_result = ComparisonResult(
                model_type=result['model_type'],
                pattern_length=result['pattern_length'],
                performance_metrics=self.framework._extract_performance_metrics(result),
                statistical_significance={},
                rank=1,
                recommendation_score=self.framework._calculate_recommendation_score(result)
            )
            comparison_results.append(comp_result)
        
        model_analysis = self.framework._analyze_model_performance(comparison_results)
        
        assert isinstance(model_analysis, dict)
        # Should have entries for each model type
        expected_models = ['XGBoost', 'RandomForest', 'SVM']
        for model in expected_models:
            if model in model_analysis:
                assert 'avg_recommendation_score' in model_analysis[model]
                assert 'best_pattern_length' in model_analysis[model]


if __name__ == '__main__':
    pytest.main([__file__])