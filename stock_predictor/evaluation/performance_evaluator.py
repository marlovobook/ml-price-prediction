"""
Performance evaluation module for the Stock Direction Predictor system.
Implements comprehensive performance metrics calculation and model comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging

from ..interfaces import IPerformanceEvaluator, BacktestResult


@dataclass
class PerformanceReport:
    """Data model for performance evaluation reports."""
    model_type: str
    pattern_length: int
    prediction_metrics: Dict[str, float]
    financial_metrics: Dict[str, float]
    overall_score: float
    rank: int


class PerformanceEvaluator(IPerformanceEvaluator):
    """
    Performance evaluator implementing comprehensive metric calculations
    for model comparison and ranking.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the performance evaluator.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
    
    def calculate_prediction_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate prediction accuracy metrics (MSE, MAE, RMSE).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing MSE, MAE, and RMSE metrics
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        # Calculate prediction accuracy metrics
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        
        # Calculate additional metrics
        accuracy = np.mean(np.sign(y_true) == np.sign(y_pred)) if len(y_true) > 0 else 0.0
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'accuracy': float(accuracy)
        }
        
        self.logger.info(f"Calculated prediction metrics: {metrics}")
        return metrics
    
    def calculate_financial_metrics(self, backtest_result: BacktestResult) -> Dict[str, float]:
        """
        Calculate financial performance metrics from backtesting results.
        
        Args:
            backtest_result: Results from backtesting simulation
            
        Returns:
            Dictionary containing financial performance metrics
        """
        if backtest_result.portfolio_values.empty:
            raise ValueError("Portfolio values cannot be empty")
        
        portfolio_values = backtest_result.portfolio_values
        returns = portfolio_values.pct_change().dropna()
        
        # Calculate cumulative return
        cumulative_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        
        # Calculate annualized return (assuming daily data)
        trading_days = len(portfolio_values)
        annualized_return = (1 + cumulative_return) ** (252 / trading_days) - 1
        
        # Calculate volatility
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate Sharpe ratio
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
        
        # Calculate maximum drawdown
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate win rate from trade log
        if not backtest_result.trade_log.empty and 'pnl' in backtest_result.trade_log.columns:
            winning_trades = (backtest_result.trade_log['pnl'] > 0).sum()
            total_trades = len(backtest_result.trade_log)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        else:
            win_rate = backtest_result.win_rate
        
        # Calculate profit factor
        if not backtest_result.trade_log.empty and 'pnl' in backtest_result.trade_log.columns:
            gross_profit = backtest_result.trade_log[backtest_result.trade_log['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(backtest_result.trade_log[backtest_result.trade_log['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            profit_factor = backtest_result.profit_factor
        
        metrics = {
            'total_return': float(cumulative_return),
            'annualized_return': float(annualized_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor)
        }
        
        self.logger.info(f"Calculated financial metrics: {metrics}")
        return metrics
    
    def calculate_roi(self, initial_value: float, final_value: float) -> float:
        """
        Calculate Return on Investment (ROI).
        
        Args:
            initial_value: Initial portfolio value
            final_value: Final portfolio value
            
        Returns:
            ROI as a percentage
        """
        if initial_value <= 0:
            raise ValueError("Initial value must be positive")
        
        roi = (final_value - initial_value) / initial_value
        return float(roi)
    
    def compare_pattern_lengths(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create comparative analysis for different pattern lengths.
        
        Args:
            results: List of result dictionaries containing model performance data
            
        Returns:
            DataFrame with comparative analysis across pattern lengths
        """
        if not results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for result in results:
            comparison_data.append({
                'model_type': result.get('model_type', 'Unknown'),
                'pattern_length': result.get('pattern_length', 0),
                'total_return': result.get('financial_metrics', {}).get('total_return', 0.0),
                'sharpe_ratio': result.get('financial_metrics', {}).get('sharpe_ratio', 0.0),
                'max_drawdown': result.get('financial_metrics', {}).get('max_drawdown', 0.0),
                'win_rate': result.get('financial_metrics', {}).get('win_rate', 0.0),
                'rmse': result.get('prediction_metrics', {}).get('rmse', float('inf')),
                'accuracy': result.get('prediction_metrics', {}).get('accuracy', 0.0)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by pattern length for better visualization
        df = df.sort_values(['model_type', 'pattern_length'])
        
        self.logger.info(f"Created comparison analysis for {len(df)} model-pattern combinations")
        return df
    
    def rank_model_combinations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank model-pattern combinations by overall performance.
        
        Args:
            results: List of result dictionaries containing model performance data
            
        Returns:
            Sorted list of results with ranking information
        """
        if not results:
            return []
        
        # Calculate composite score for each result
        scored_results = []
        
        for result in results:
            financial_metrics = result.get('financial_metrics', {})
            prediction_metrics = result.get('prediction_metrics', {})
            
            # Normalize metrics (higher is better for most metrics)
            sharpe_score = max(0, financial_metrics.get('sharpe_ratio', 0))
            return_score = max(0, financial_metrics.get('total_return', 0))
            drawdown_score = max(0, 1 + financial_metrics.get('max_drawdown', 0))  # Convert negative to positive
            accuracy_score = prediction_metrics.get('accuracy', 0)
            rmse_score = 1 / (1 + prediction_metrics.get('rmse', 1))  # Lower RMSE is better
            
            # Weighted composite score
            composite_score = (
                0.3 * sharpe_score +
                0.25 * return_score +
                0.2 * drawdown_score +
                0.15 * accuracy_score +
                0.1 * rmse_score
            )
            
            result_copy = result.copy()
            result_copy['composite_score'] = float(composite_score)
            scored_results.append(result_copy)
        
        # Sort by composite score (descending)
        ranked_results = sorted(scored_results, key=lambda x: x['composite_score'], reverse=True)
        
        # Add rank information
        for i, result in enumerate(ranked_results):
            result['rank'] = i + 1
        
        self.logger.info(f"Ranked {len(ranked_results)} model-pattern combinations")
        return ranked_results
    
    def generate_performance_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive performance report with best configuration identification.
        
        Args:
            results: List of result dictionaries containing model performance data
            
        Returns:
            Dictionary containing comprehensive performance report
        """
        if not results:
            return {'error': 'No results provided'}
        
        # Rank all combinations
        ranked_results = self.rank_model_combinations(results)
        
        # Get best configuration
        best_config = ranked_results[0] if ranked_results else None
        
        # Create comparison DataFrame
        comparison_df = self.compare_pattern_lengths(results)
        
        # Calculate summary statistics
        summary_stats = {
            'total_combinations': len(results),
            'best_model_type': best_config.get('model_type', 'Unknown') if best_config else 'Unknown',
            'best_pattern_length': best_config.get('pattern_length', 0) if best_config else 0,
            'best_total_return': best_config.get('financial_metrics', {}).get('total_return', 0.0) if best_config else 0.0,
            'best_sharpe_ratio': best_config.get('financial_metrics', {}).get('sharpe_ratio', 0.0) if best_config else 0.0,
            'avg_total_return': comparison_df['total_return'].mean() if not comparison_df.empty else 0.0,
            'avg_sharpe_ratio': comparison_df['sharpe_ratio'].mean() if not comparison_df.empty else 0.0
        }
        
        report = {
            'summary': summary_stats,
            'best_configuration': best_config,
            'ranked_results': ranked_results,
            'comparison_analysis': comparison_df.to_dict('records') if not comparison_df.empty else [],
            'pattern_length_analysis': self._analyze_pattern_lengths(comparison_df)
        }
        
        self.logger.info(f"Generated performance report for {len(results)} configurations")
        return report
    
    def _analyze_pattern_lengths(self, comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze performance across different pattern lengths.
        
        Args:
            comparison_df: DataFrame with comparison data
            
        Returns:
            Dictionary with pattern length analysis
        """
        if comparison_df.empty:
            return {}
        
        pattern_analysis = {}
        
        for pattern_length in comparison_df['pattern_length'].unique():
            pattern_data = comparison_df[comparison_df['pattern_length'] == pattern_length]
            
            pattern_analysis[f'{pattern_length}_day'] = {
                'avg_return': float(pattern_data['total_return'].mean()),
                'avg_sharpe': float(pattern_data['sharpe_ratio'].mean()),
                'avg_accuracy': float(pattern_data['accuracy'].mean()),
                'best_model': pattern_data.loc[pattern_data['total_return'].idxmax(), 'model_type'] if not pattern_data.empty else 'Unknown'
            }
        
        return pattern_analysis