"""
Example usage of the comprehensive comparison framework.
Demonstrates how to compare model-pattern combinations and generate reports.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from stock_predictor.evaluation.comparison_framework import ComparisonFramework
from stock_predictor.interfaces import BacktestResult


def create_sample_results():
    """Create sample results for demonstration."""
    
    # Sample portfolio values for backtesting results
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    results = []
    
    # XGBoost with 3-day patterns
    portfolio_values_1 = pd.Series(
        10000 * (1 + np.cumsum(np.random.normal(0.001, 0.02, len(dates)))),
        index=dates
    )
    trade_dates_1 = dates[::10]  # Trade every 10 days
    trade_log_1 = pd.DataFrame({
        'date': trade_dates_1,
        'action': (['buy', 'sell'] * (len(trade_dates_1) // 2 + 1))[:len(trade_dates_1)],
        'price': np.random.uniform(100, 200, len(trade_dates_1)),
        'pnl': np.random.normal(50, 100, len(trade_dates_1))
    })
    
    backtest_result_1 = BacktestResult(
        total_return=0.15,
        max_drawdown=-0.08,
        sharpe_ratio=1.2,
        win_rate=0.65,
        profit_factor=1.8,
        trade_log=trade_log_1,
        portfolio_values=portfolio_values_1
    )
    
    results.append({
        'model_type': 'XGBoost',
        'pattern_length': 3,
        'financial_metrics': {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'win_rate': 0.65,
            'volatility': 0.12,
            'annualized_return': 0.16,
            'profit_factor': 1.8
        },
        'prediction_metrics': {
            'accuracy': 0.72,
            'mse': 0.05,
            'mae': 0.18,
            'rmse': 0.22
        },
        'composite_score': 0.85,
        'backtest_result': backtest_result_1
    })
    
    # Random Forest with 5-day patterns
    portfolio_values_2 = pd.Series(
        10000 * (1 + np.cumsum(np.random.normal(0.0008, 0.018, len(dates)))),
        index=dates
    )
    trade_dates_2 = dates[::12]
    trade_log_2 = pd.DataFrame({
        'date': trade_dates_2,
        'action': (['buy', 'sell'] * (len(trade_dates_2) // 2 + 1))[:len(trade_dates_2)],
        'price': np.random.uniform(100, 200, len(trade_dates_2)),
        'pnl': np.random.normal(40, 80, len(trade_dates_2))
    })
    
    backtest_result_2 = BacktestResult(
        total_return=0.12,
        max_drawdown=-0.06,
        sharpe_ratio=1.0,
        win_rate=0.68,
        profit_factor=1.6,
        trade_log=trade_log_2,
        portfolio_values=portfolio_values_2
    )
    
    results.append({
        'model_type': 'RandomForest',
        'pattern_length': 5,
        'financial_metrics': {
            'total_return': 0.12,
            'sharpe_ratio': 1.0,
            'max_drawdown': -0.06,
            'win_rate': 0.68,
            'volatility': 0.10,
            'annualized_return': 0.13,
            'profit_factor': 1.6
        },
        'prediction_metrics': {
            'accuracy': 0.70,
            'mse': 0.06,
            'mae': 0.20,
            'rmse': 0.24
        },
        'composite_score': 0.78,
        'backtest_result': backtest_result_2
    })
    
    # SVM with 7-day patterns
    portfolio_values_3 = pd.Series(
        10000 * (1 + np.cumsum(np.random.normal(0.0006, 0.025, len(dates)))),
        index=dates
    )
    trade_dates_3 = dates[::15]
    trade_log_3 = pd.DataFrame({
        'date': trade_dates_3,
        'action': (['buy', 'sell'] * (len(trade_dates_3) // 2 + 1))[:len(trade_dates_3)],
        'price': np.random.uniform(100, 200, len(trade_dates_3)),
        'pnl': np.random.normal(30, 120, len(trade_dates_3))
    })
    
    backtest_result_3 = BacktestResult(
        total_return=0.10,
        max_drawdown=-0.10,
        sharpe_ratio=0.8,
        win_rate=0.60,
        profit_factor=1.4,
        trade_log=trade_log_3,
        portfolio_values=portfolio_values_3
    )
    
    results.append({
        'model_type': 'SVM',
        'pattern_length': 7,
        'financial_metrics': {
            'total_return': 0.10,
            'sharpe_ratio': 0.8,
            'max_drawdown': -0.10,
            'win_rate': 0.60,
            'volatility': 0.15,
            'annualized_return': 0.11,
            'profit_factor': 1.4
        },
        'prediction_metrics': {
            'accuracy': 0.68,
            'mse': 0.08,
            'mae': 0.25,
            'rmse': 0.28
        },
        'composite_score': 0.65,
        'backtest_result': backtest_result_3
    })
    
    # Neural Network with 14-day patterns
    portfolio_values_4 = pd.Series(
        10000 * (1 + np.cumsum(np.random.normal(0.0009, 0.022, len(dates)))),
        index=dates
    )
    trade_dates_4 = dates[::8]
    trade_log_4 = pd.DataFrame({
        'date': trade_dates_4,
        'action': (['buy', 'sell'] * (len(trade_dates_4) // 2 + 1))[:len(trade_dates_4)],
        'price': np.random.uniform(100, 200, len(trade_dates_4)),
        'pnl': np.random.normal(45, 90, len(trade_dates_4))
    })
    
    backtest_result_4 = BacktestResult(
        total_return=0.13,
        max_drawdown=-0.07,
        sharpe_ratio=1.1,
        win_rate=0.63,
        profit_factor=1.7,
        trade_log=trade_log_4,
        portfolio_values=portfolio_values_4
    )
    
    results.append({
        'model_type': 'NeuralNetwork',
        'pattern_length': 14,
        'financial_metrics': {
            'total_return': 0.13,
            'sharpe_ratio': 1.1,
            'max_drawdown': -0.07,
            'win_rate': 0.63,
            'volatility': 0.11,
            'annualized_return': 0.14,
            'profit_factor': 1.7
        },
        'prediction_metrics': {
            'accuracy': 0.71,
            'mse': 0.055,
            'mae': 0.19,
            'rmse': 0.23
        },
        'composite_score': 0.82,
        'backtest_result': backtest_result_4
    })
    
    return results


def main():
    """Main example function."""
    print("=== Comprehensive Comparison Framework Example ===\n")
    
    # Create sample results
    print("1. Creating sample model-pattern combination results...")
    results = create_sample_results()
    print(f"   Created {len(results)} model-pattern combinations\n")
    
    # Initialize comparison framework
    print("2. Initializing comparison framework...")
    framework = ComparisonFramework(confidence_level=0.95)
    print("   Framework initialized with 95% confidence level\n")
    
    # Perform comprehensive comparison
    print("3. Performing comprehensive comparison analysis...")
    comparison_report = framework.compare_all_combinations(results)
    print("   Analysis complete!\n")
    
    # Display executive summary
    print("=== EXECUTIVE SUMMARY ===")
    summary = comparison_report['executive_summary']
    print(f"Total Configurations Analyzed: {summary['total_configurations']}")
    
    best_config = summary['best_configuration']
    print(f"Best Configuration: {best_config['model_type']} with {best_config['pattern_length']}-day patterns")
    print(f"Recommendation Score: {best_config['recommendation_score']:.1f}/100")
    print(f"Total Return: {best_config['key_metrics'].get('total_return', 0):.2%}")
    print(f"Sharpe Ratio: {best_config['key_metrics'].get('sharpe_ratio', 0):.2f}")
    print()
    
    # Display detailed rankings
    print("=== DETAILED RANKINGS ===")
    detailed_results = comparison_report['detailed_results']
    print(f"{'Rank':<4} {'Model':<15} {'Pattern':<8} {'Score':<8} {'Return':<8} {'Sharpe':<8}")
    print("-" * 60)
    
    for result in detailed_results[:5]:  # Top 5
        print(f"{result['rank']:<4} {result['model_type']:<15} "
              f"{result['pattern_length']}d{'':<6} {result['recommendation_score']:<8.1f} "
              f"{result['performance_metrics'].get('total_return', 0):<8.2%} "
              f"{result['performance_metrics'].get('sharpe_ratio', 0):<8.2f}")
    print()
    
    # Display pattern length analysis
    print("=== PATTERN LENGTH ANALYSIS ===")
    pattern_analysis = comparison_report['pattern_length_analysis']
    for pattern, stats in pattern_analysis.items():
        print(f"{pattern.replace('_', ' ').title()}:")
        print(f"  Average Return: {stats['avg_total_return']:.2%}")
        print(f"  Average Sharpe: {stats['avg_sharpe_ratio']:.2f}")
        print(f"  Best Model: {stats['best_model']}")
        print()
    
    # Display model type analysis
    print("=== MODEL TYPE ANALYSIS ===")
    model_analysis = comparison_report['model_type_analysis']
    for model, stats in model_analysis.items():
        print(f"{model}:")
        print(f"  Average Return: {stats['avg_total_return']:.2%}")
        print(f"  Average Sharpe: {stats['avg_sharpe_ratio']:.2f}")
        print(f"  Best Pattern Length: {stats['best_pattern_length']} days")
        print()
    
    # Display recommendations
    print("=== RECOMMENDATIONS ===")
    recommendations = comparison_report['recommendations']
    for i, recommendation in enumerate(recommendations, 1):
        print(f"{i}. {recommendation}")
    print()
    
    # Generate visualization data
    print("4. Generating visualization data...")
    comparison_results = []
    for result in detailed_results:
        from stock_predictor.evaluation.comparison_framework import ComparisonResult
        comp_result = ComparisonResult(
            model_type=result['model_type'],
            pattern_length=result['pattern_length'],
            performance_metrics=result['performance_metrics'],
            statistical_significance=result['statistical_significance'],
            rank=result['rank'],
            recommendation_score=result['recommendation_score']
        )
        comparison_results.append(comp_result)
    
    viz_data = framework.generate_visualization_data(comparison_results)
    print(f"   Generated visualization data with {len(viz_data)} chart types\n")
    
    # Select best configuration with custom criteria
    print("5. Selecting best configuration with custom criteria...")
    custom_weights = {
        'total_return': 0.40,  # Emphasize returns
        'sharpe_ratio': 0.30,
        'max_drawdown': 0.15,
        'accuracy': 0.10,
        'win_rate': 0.05
    }
    
    best_config_custom = framework.select_best_configuration(
        comparison_results, 
        criteria_weights=custom_weights
    )
    
    print(f"   Best configuration (custom weights): {best_config_custom.model_type} "
          f"with {best_config_custom.pattern_length}-day patterns")
    print(f"   Recommendation Score: {best_config_custom.recommendation_score:.1f}/100\n")
    
    # Statistical significance summary
    print("=== STATISTICAL ANALYSIS ===")
    statistical_tests = comparison_report['statistical_tests']
    
    if 'friedman_test' in statistical_tests:
        friedman = statistical_tests['friedman_test']
        print(f"Friedman Test: {friedman.interpretation}")
    
    pairwise_tests = statistical_tests.get('pairwise_tests', {})
    significant_pairs = [test for test in pairwise_tests.values() if test.is_significant]
    print(f"Significant pairwise differences found: {len(significant_pairs)}")
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("The comparison framework has successfully analyzed all model-pattern combinations")
    print("and provided comprehensive insights for optimal configuration selection.")


if __name__ == '__main__':
    # Set random seed for reproducible results
    np.random.seed(42)
    main()