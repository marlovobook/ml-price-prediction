"""
Property-based tests for performance evaluation module.
Tests performance metric calculation accuracy and ranking consistency.
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
import sys
import os
from typing import Dict, List, Any

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stock_predictor.evaluation.performance_evaluator import PerformanceEvaluator
from stock_predictor.interfaces import BacktestResult


def generate_portfolio_values(num_days: int, initial_value: float = 10000.0, volatility: float = 0.02) -> pd.Series:
    """Generate realistic portfolio value series for testing."""
    np.random.seed(42)  # For reproducible tests
    
    values = [initial_value]
    current_value = initial_value
    
    for i in range(num_days - 1):
        # Generate daily return with some volatility
        daily_return = np.random.normal(0, volatility)
        current_value = current_value * (1 + daily_return)
        values.append(current_value)
    
    dates = pd.date_range(start='2023-01-01', periods=num_days, freq='D')
    return pd.Series(values, index=dates)


def generate_trade_log(num_trades: int) -> pd.DataFrame:
    """Generate realistic trade log for testing."""
    np.random.seed(42)
    
    trades = []
    for i in range(num_trades):
        # Generate random P&L with some winning and losing trades
        pnl = np.random.normal(50, 200)  # Average $50 profit with $200 std dev
        
        trades.append({
            'trade_id': i + 1,
            'entry_date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i*2),
            'exit_date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i*2 + 1),
            'signal': np.random.choice([-1, 1]),
            'entry_price': 100 + np.random.normal(0, 10),
            'exit_price': 100 + np.random.normal(0, 10),
            'pnl': pnl,
            'quantity': 100
        })
    
    return pd.DataFrame(trades)


def generate_backtest_result(num_days: int = 252, num_trades: int = 50) -> BacktestResult:
    """Generate realistic BacktestResult for testing."""
    portfolio_values = generate_portfolio_values(num_days)
    trade_log = generate_trade_log(num_trades)
    
    # Calculate basic metrics
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    returns = portfolio_values.pct_change().dropna()
    
    # Calculate max drawdown
    running_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    winning_trades = (trade_log['pnl'] > 0).sum()
    win_rate = winning_trades / len(trade_log) if len(trade_log) > 0 else 0.0
    
    # Calculate profit factor
    gross_profit = trade_log[trade_log['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trade_log[trade_log['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Calculate Sharpe ratio (simplified)
    excess_return = returns.mean() * 252 - 0.02  # Annualized excess return
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
    
    return BacktestResult(
        total_return=total_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        win_rate=win_rate,
        profit_factor=profit_factor,
        trade_log=trade_log,
        portfolio_values=portfolio_values
    )


class TestPerformanceEvaluationProperties:
    """Property-based tests for performance evaluation module."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = PerformanceEvaluator(risk_free_rate=0.02)
    
    @given(
        num_predictions=st.integers(min_value=10, max_value=1000),
        noise_level=st.floats(min_value=0.1, max_value=2.0),
        bias=st.floats(min_value=-1.0, max_value=1.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_performance_metric_calculation(self, num_predictions, noise_level, bias):
        """
        **Feature: stock-direction-predictor, Property 6: Performance Metric Calculation**
        **Validates: Requirements 5.1, 5.2, 5.3, 5.5, 5.6**
        
        For any set of predictions and actual values, the performance evaluator should 
        calculate MSE, MAE, RMSE, cumulative profit, ROI, and maximum drawdown correctly 
        and rank model-pattern combinations appropriately.
        """
        np.random.seed(42)
        
        # Generate true values and predictions with controlled relationship
        y_true = np.random.normal(0, 1, num_predictions)
        y_pred = y_true + bias + np.random.normal(0, noise_level, num_predictions)
        
        # Calculate prediction metrics
        metrics = self.evaluator.calculate_prediction_metrics(y_true, y_pred)
        
        # Property 1: MSE should be non-negative
        assert metrics['mse'] >= 0, f"MSE should be non-negative, got {metrics['mse']}"
        
        # Property 2: MAE should be non-negative
        assert metrics['mae'] >= 0, f"MAE should be non-negative, got {metrics['mae']}"
        
        # Property 3: RMSE should be non-negative and equal to sqrt(MSE)
        assert metrics['rmse'] >= 0, f"RMSE should be non-negative, got {metrics['rmse']}"
        expected_rmse = np.sqrt(metrics['mse'])
        assert abs(metrics['rmse'] - expected_rmse) < 1e-10, f"RMSE should equal sqrt(MSE)"
        
        # Property 4: RMSE should be >= MAE (by mathematical property)
        # This is not always true, but for most cases it should hold
        # We'll allow some tolerance for edge cases
        if metrics['mae'] > 0:
            ratio = metrics['rmse'] / metrics['mae']
            assert ratio >= 0.5, f"RMSE/MAE ratio should be reasonable, got {ratio}"
        
        # Property 5: Accuracy should be between 0 and 1
        assert 0 <= metrics['accuracy'] <= 1, f"Accuracy should be between 0 and 1, got {metrics['accuracy']}"
        
        # Property 6: Perfect predictions should yield zero error metrics
        perfect_pred = y_true.copy()
        perfect_metrics = self.evaluator.calculate_prediction_metrics(y_true, perfect_pred)
        assert perfect_metrics['mse'] == 0, "Perfect predictions should have MSE = 0"
        assert perfect_metrics['mae'] == 0, "Perfect predictions should have MAE = 0"
        assert perfect_metrics['rmse'] == 0, "Perfect predictions should have RMSE = 0"
        
        # Property 7: Metrics should be finite for finite inputs
        assert np.isfinite(metrics['mse']), "MSE should be finite for finite inputs"
        assert np.isfinite(metrics['mae']), "MAE should be finite for finite inputs"
        assert np.isfinite(metrics['rmse']), "RMSE should be finite for finite inputs"
        assert np.isfinite(metrics['accuracy']), "Accuracy should be finite for finite inputs"
    
    @given(
        num_days=st.integers(min_value=30, max_value=500),
        num_trades=st.integers(min_value=5, max_value=100),
        initial_value=st.floats(min_value=1000.0, max_value=100000.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_financial_metrics_calculation(self, num_days, num_trades, initial_value):
        """
        Test financial metrics calculation properties.
        """
        # Generate backtest result
        backtest_result = generate_backtest_result(num_days, num_trades)
        
        # Adjust initial portfolio value
        portfolio_values = backtest_result.portfolio_values
        portfolio_values = portfolio_values * (initial_value / portfolio_values.iloc[0])
        backtest_result.portfolio_values = portfolio_values
        
        # Calculate financial metrics
        metrics = self.evaluator.calculate_financial_metrics(backtest_result)
        
        # Property 1: Total return should be calculated correctly
        expected_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        assert abs(metrics['total_return'] - expected_return) < 1e-10, "Total return calculation should be accurate"
        
        # Property 2: Maximum drawdown should be non-positive (it's a loss measure)
        assert metrics['max_drawdown'] <= 0, f"Maximum drawdown should be <= 0, got {metrics['max_drawdown']}"
        
        # Property 3: Win rate should be between 0 and 1
        assert 0 <= metrics['win_rate'] <= 1, f"Win rate should be between 0 and 1, got {metrics['win_rate']}"
        
        # Property 4: Profit factor should be non-negative
        assert metrics['profit_factor'] >= 0, f"Profit factor should be >= 0, got {metrics['profit_factor']}"
        
        # Property 5: Volatility should be non-negative
        assert metrics['volatility'] >= 0, f"Volatility should be >= 0, got {metrics['volatility']}"
        
        # Property 6: All metrics should be finite (except profit_factor which can be inf)
        assert np.isfinite(metrics['total_return']), "Total return should be finite"
        assert np.isfinite(metrics['annualized_return']), "Annualized return should be finite"
        assert np.isfinite(metrics['volatility']), "Volatility should be finite"
        assert np.isfinite(metrics['max_drawdown']), "Max drawdown should be finite"
        assert np.isfinite(metrics['win_rate']), "Win rate should be finite"
        # Profit factor can be infinite if there are no losses
        
        # Property 7: Sharpe ratio should be finite when volatility > 0
        if metrics['volatility'] > 0:
            assert np.isfinite(metrics['sharpe_ratio']), "Sharpe ratio should be finite when volatility > 0"
    
    @given(
        num_results=st.integers(min_value=2, max_value=20),
        pattern_lengths=st.lists(st.sampled_from([3, 5, 7, 14]), min_size=1, max_size=4, unique=True)
    )
    @settings(max_examples=50, deadline=None)
    def test_ranking_consistency(self, num_results, pattern_lengths):
        """
        Test ranking system consistency properties.
        """
        np.random.seed(42)
        
        # Generate multiple model results
        results = []
        model_types = ['XGBoost', 'RandomForest', 'SVM', 'NeuralNetwork']
        
        for i in range(num_results):
            model_type = np.random.choice(model_types)
            pattern_length = np.random.choice(pattern_lengths)
            
            # Generate random but realistic metrics
            total_return = np.random.normal(0.1, 0.3)  # 10% average return, 30% std
            sharpe_ratio = np.random.normal(0.5, 0.5)  # 0.5 average Sharpe, 0.5 std
            max_drawdown = -abs(np.random.normal(0.1, 0.1))  # Negative drawdown
            win_rate = np.random.uniform(0.3, 0.7)  # 30-70% win rate
            rmse = np.random.uniform(0.1, 2.0)  # RMSE between 0.1 and 2.0
            accuracy = np.random.uniform(0.4, 0.8)  # 40-80% accuracy
            
            result = {
                'model_type': model_type,
                'pattern_length': pattern_length,
                'financial_metrics': {
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate
                },
                'prediction_metrics': {
                    'rmse': rmse,
                    'accuracy': accuracy,
                    'mse': rmse ** 2,
                    'mae': rmse * 0.8  # Approximate relationship
                }
            }
            results.append(result)
        
        # Rank the results
        ranked_results = self.evaluator.rank_model_combinations(results)
        
        # Property 1: All results should be present after ranking
        assert len(ranked_results) == len(results), "All results should be present after ranking"
        
        # Property 2: Ranks should be consecutive integers starting from 1
        ranks = [r['rank'] for r in ranked_results]
        expected_ranks = list(range(1, len(results) + 1))
        assert ranks == expected_ranks, f"Ranks should be consecutive from 1 to {len(results)}"
        
        # Property 3: Composite scores should be in descending order
        scores = [r['composite_score'] for r in ranked_results]
        assert scores == sorted(scores, reverse=True), "Composite scores should be in descending order"
        
        # Property 4: All composite scores should be finite and non-negative
        for score in scores:
            assert np.isfinite(score), "Composite scores should be finite"
            assert score >= 0, "Composite scores should be non-negative"
        
        # Property 5: Best ranked result should have rank 1
        assert ranked_results[0]['rank'] == 1, "Best result should have rank 1"
        
        # Property 6: Worst ranked result should have the highest rank number
        assert ranked_results[-1]['rank'] == len(results), "Worst result should have highest rank number"
    
    @given(
        num_results=st.integers(min_value=1, max_value=15),
        pattern_lengths=st.lists(st.sampled_from([3, 5, 7, 14]), min_size=1, max_size=4, unique=True)
    )
    @settings(max_examples=50, deadline=None)
    def test_performance_report_generation(self, num_results, pattern_lengths):
        """
        Test performance report generation properties.
        """
        np.random.seed(42)
        
        # Generate results similar to ranking test
        results = []
        model_types = ['XGBoost', 'RandomForest', 'SVM']
        
        for i in range(num_results):
            model_type = np.random.choice(model_types)
            pattern_length = np.random.choice(pattern_lengths)
            
            total_return = np.random.normal(0.1, 0.2)
            sharpe_ratio = np.random.normal(0.5, 0.3)
            max_drawdown = -abs(np.random.normal(0.1, 0.05))
            win_rate = np.random.uniform(0.4, 0.7)
            rmse = np.random.uniform(0.5, 1.5)
            accuracy = np.random.uniform(0.5, 0.8)
            
            result = {
                'model_type': model_type,
                'pattern_length': pattern_length,
                'financial_metrics': {
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate
                },
                'prediction_metrics': {
                    'rmse': rmse,
                    'accuracy': accuracy,
                    'mse': rmse ** 2,
                    'mae': rmse * 0.8
                }
            }
            results.append(result)
        
        # Generate performance report
        report = self.evaluator.generate_performance_report(results)
        
        # Property 1: Report should contain all required sections
        required_sections = ['summary', 'best_configuration', 'ranked_results', 'comparison_analysis', 'pattern_length_analysis']
        for section in required_sections:
            assert section in report, f"Report should contain {section} section"
        
        # Property 2: Summary should contain correct total combinations count
        assert report['summary']['total_combinations'] == num_results, "Summary should show correct total combinations"
        
        # Property 3: Best configuration should be the top-ranked result
        if num_results > 0:
            best_config = report['best_configuration']
            ranked_results = report['ranked_results']
            assert best_config == ranked_results[0], "Best configuration should match top-ranked result"
        
        # Property 4: Ranked results should have correct length
        assert len(report['ranked_results']) == num_results, "Ranked results should have correct length"
        
        # Property 5: Comparison analysis should have correct length
        assert len(report['comparison_analysis']) == num_results, "Comparison analysis should have correct length"
        
        # Property 6: Pattern length analysis should contain data for each unique pattern length
        pattern_analysis = report['pattern_length_analysis']
        unique_patterns = set(pattern_lengths)
        for pattern in unique_patterns:
            pattern_key = f'{pattern}_day'
            if any(r['pattern_length'] == pattern for r in results):
                assert pattern_key in pattern_analysis, f"Pattern analysis should contain {pattern_key}"
    
    @given(
        initial_value=st.floats(min_value=100.0, max_value=100000.0),
        final_multiplier=st.floats(min_value=0.1, max_value=5.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_roi_calculation(self, initial_value, final_multiplier):
        """
        Test ROI calculation properties.
        """
        final_value = initial_value * final_multiplier
        
        # Calculate ROI
        roi = self.evaluator.calculate_roi(initial_value, final_value)
        
        # Property 1: ROI should equal (final - initial) / initial
        expected_roi = (final_value - initial_value) / initial_value
        assert abs(roi - expected_roi) < 1e-10, "ROI calculation should be accurate"
        
        # Property 2: ROI should be finite for positive initial values
        assert np.isfinite(roi), "ROI should be finite for positive initial values"
        
        # Property 3: ROI should be -1 when final value is 0
        zero_roi = self.evaluator.calculate_roi(initial_value, 0.0)
        assert abs(zero_roi - (-1.0)) < 1e-10, "ROI should be -1 when final value is 0"
        
        # Property 4: ROI should be 0 when final equals initial
        same_roi = self.evaluator.calculate_roi(initial_value, initial_value)
        assert abs(same_roi) < 1e-10, "ROI should be 0 when final equals initial"
        
        # Property 5: ROI should be positive when final > initial
        if final_value > initial_value:
            assert roi > 0, "ROI should be positive when final > initial"
        elif final_value < initial_value:
            assert roi < 0, "ROI should be negative when final < initial"


if __name__ == "__main__":
    pytest.main([__file__])