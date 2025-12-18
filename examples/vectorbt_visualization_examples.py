"""
VectorBT Visualization Enhancement Examples

This module demonstrates all visualization types and features of the
VectorBT Visualization Enhancement system.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Any

# Import visualization components
from stock_predictor.visualization import (
    VectorBTVisualizationEngine,
    EnhancedPortfolioEngine,
    SignalAlignmentEngine,
    PortfolioConfig,
    PlotConfig,
    PlotExportEngine
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_market_data(num_days: int = 252, symbol: str = "SAMPLE", scenario: str = "normal") -> pd.DataFrame:
    """
    Create realistic sample market data for demonstrations.
    
    Args:
        num_days: Number of trading days to generate
        symbol: Stock symbol for seeding randomness
        scenario: Market scenario ('bull', 'bear', 'volatile', 'normal')
    
    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start="2023-01-01", periods=num_days, freq='D')
    
    # Seed for reproducible results
    np.random.seed(hash(symbol + scenario) % 2**32)
    
    # Base parameters
    base_price = 100.0
    
    # Scenario-specific parameters
    if scenario == "bull":
        trend = 0.0008  # Strong upward trend
        volatility = 0.015
    elif scenario == "bear":
        trend = -0.0006  # Downward trend
        volatility = 0.025
    elif scenario == "volatile":
        trend = 0.0002  # Slight upward trend
        volatility = 0.035
    else:  # normal
        trend = 0.0004  # Moderate upward trend
        volatility = 0.02
    
    # Generate price series
    returns = np.random.normal(trend, volatility, num_days)
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1.0))  # Prevent negative prices
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Calculate open (with small gap)
        prev_close = prices[i-1] if i > 0 else close
        gap = np.random.normal(0, 0.002)
        open_price = prev_close * (1 + gap)
        
        # Calculate high and low
        intraday_range = close * volatility * np.random.uniform(0.5, 1.5)
        high = max(open_price, close) + intraday_range * np.random.uniform(0, 0.8)
        low = min(open_price, close) - intraday_range * np.random.uniform(0, 0.8)
        
        # Ensure OHLC relationships
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume
        base_volume = 1000000
        volume_factor = 1 + abs(returns[i]) * 10  # Higher volume on volatile days
        volume = int(base_volume * volume_factor * np.random.uniform(0.5, 2.0))
        
        data.append({
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume,
            'Adj Close': round(close * (1 + np.random.normal(0, 0.001)), 2)
        })
    
    return pd.DataFrame(data, index=dates)


def create_sample_predictions(num_predictions: int, strategy_type: str = "balanced") -> np.ndarray:
    """
    Create sample ML predictions for demonstrations.
    
    Args:
        num_predictions: Number of predictions to generate
        strategy_type: Type of strategy ('conservative', 'aggressive', 'balanced', 'trend_following')
    
    Returns:
        Array of predictions (0=sell, 1=hold, 2=buy)
    """
    np.random.seed(42)  # For reproducible results
    
    if strategy_type == "conservative":
        # Mostly hold with occasional trades
        probabilities = [0.1, 0.8, 0.1]
    elif strategy_type == "aggressive":
        # More frequent trading
        probabilities = [0.3, 0.4, 0.3]
    elif strategy_type == "trend_following":
        # Bias towards buy signals
        probabilities = [0.15, 0.5, 0.35]
    else:  # balanced
        probabilities = [0.2, 0.6, 0.2]
    
    return np.random.choice([0, 1, 2], size=num_predictions, p=probabilities)


def example_1_basic_portfolio_visualization():
    """
    Example 1: Basic Portfolio Visualization
    
    Demonstrates the fundamental workflow from ML predictions to portfolio visualization.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Portfolio Visualization")
    print("="*60)
    
    # Step 1: Create sample data
    logger.info("Creating sample market data...")
    price_data = create_sample_market_data(150, "BASIC_EXAMPLE", "normal")
    predictions = create_sample_predictions(30, "balanced")
    test_start_idx = 120  # Last 30 days for testing
    
    # Step 2: Configure portfolio parameters
    portfolio_config = PortfolioConfig(
        init_cash=100000.0,
        fees=0.0025,
        slippage=0.0025,
        size_strategy='fixed_amount',
        size_value=10000.0,
        stop_loss=0.1
    )
    
    plot_config = PlotConfig(
        width=1200,
        height=600,
        show_trades=True,
        show_positions=True
    )
    
    # Step 3: Initialize engines
    viz_engine = VectorBTVisualizationEngine(
        portfolio_config=portfolio_config,
        plot_config=plot_config
    )
    
    # Step 4: Create portfolio from predictions
    logger.info("Creating portfolio from ML predictions...")
    portfolio = viz_engine.create_portfolio_from_predictions(
        predictions, price_data, test_start_idx, symbol="BASIC"
    )
    
    # Step 5: Generate visualization
    logger.info("Generating portfolio visualization...")
    viz_result = viz_engine.generate_portfolio_plot(
        portfolio, title="Basic Portfolio Performance Example"
    )
    
    if viz_result.success:
        logger.info(f"Visualization generated successfully in {viz_result.generation_time:.2f}s")
        
        # Display key metrics
        metrics = viz_result.metrics_summary
        print(f"\nPortfolio Performance Metrics:")
        print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  Number of Trades: {portfolio.trades.count()}")
        
        # Show plot (in interactive environments)
        try:
            viz_engine.show_plot(viz_result)
        except Exception as e:
            logger.warning(f"Could not display plot interactively: {e}")
        
        return viz_result
    else:
        logger.error(f"Visualization failed: {viz_result.error_message}")
        return None


def example_2_drawdown_analysis():
    """
    Example 2: Comprehensive Drawdown Analysis
    
    Demonstrates detailed drawdown visualization with recovery analysis.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Comprehensive Drawdown Analysis")
    print("="*60)
    
    # Create volatile market data for interesting drawdowns
    logger.info("Creating volatile market data...")
    price_data = create_sample_market_data(200, "DRAWDOWN_EXAMPLE", "volatile")
    predictions = create_sample_predictions(50, "aggressive")
    test_start_idx = 150
    
    # Configure for drawdown analysis
    portfolio_config = PortfolioConfig(
        init_cash=100000.0,
        fees=0.003,  # Higher fees for more realistic drawdowns
        size_strategy='percent_equity',
        size_value=0.2,  # 20% of equity per trade
        stop_loss=0.15   # 15% stop-loss
    )
    
    viz_engine = VectorBTVisualizationEngine(portfolio_config=portfolio_config)
    
    # Create portfolio
    logger.info("Creating portfolio with aggressive strategy...")
    portfolio = viz_engine.create_portfolio_from_predictions(
        predictions, price_data, test_start_idx, symbol="DRAWDOWN"
    )
    
    # Generate drawdown analysis
    logger.info("Generating comprehensive drawdown analysis...")
    drawdown_result = viz_engine.generate_drawdown_plot(portfolio)
    
    if drawdown_result.success:
        logger.info(f"Drawdown analysis generated in {drawdown_result.generation_time:.2f}s")
        
        # Display drawdown metrics
        metrics = drawdown_result.metrics_summary
        print(f"\nDrawdown Analysis Metrics:")
        print(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2%}")
        print(f"  Average Drawdown: {metrics.get('avg_drawdown_pct', 0):.2%}")
        print(f"  Max Drawdown Duration: {metrics.get('max_drawdown_duration', 0):.0f} days")
        print(f"  Average Recovery Time: {metrics.get('avg_recovery_time', 0):.1f} days")
        print(f"  Time Underwater: {metrics.get('time_underwater_pct', 0):.1%}")
        print(f"  Number of Drawdown Periods: {metrics.get('num_drawdown_periods', 0)}")
        
        try:
            viz_engine.show_plot(drawdown_result)
        except Exception as e:
            logger.warning(f"Could not display plot interactively: {e}")
        
        return drawdown_result
    else:
        logger.error(f"Drawdown analysis failed: {drawdown_result.error_message}")
        return None


def example_3_multi_strategy_comparison():
    """
    Example 3: Multi-Strategy Comparison
    
    Demonstrates comparing multiple trading strategies side-by-side.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Multi-Strategy Comparison")
    print("="*60)
    
    # Create common market data
    logger.info("Creating market data for strategy comparison...")
    price_data = create_sample_market_data(180, "COMPARISON", "normal")
    test_start_idx = 120
    
    # Define different strategies
    strategies = {
        'Conservative': {
            'predictions': create_sample_predictions(60, "conservative"),
            'config': PortfolioConfig(
                init_cash=100000.0,
                size_strategy='fixed_amount',
                size_value=5000.0,  # Smaller positions
                stop_loss=0.08,     # Tighter stop-loss
                fees=0.002
            )
        },
        'Aggressive': {
            'predictions': create_sample_predictions(60, "aggressive"),
            'config': PortfolioConfig(
                init_cash=100000.0,
                size_strategy='percent_equity',
                size_value=0.25,    # 25% of equity
                stop_loss=0.15,     # Wider stop-loss
                fees=0.003
            )
        },
        'Trend Following': {
            'predictions': create_sample_predictions(60, "trend_following"),
            'config': PortfolioConfig(
                init_cash=100000.0,
                size_strategy='fixed_amount',
                size_value=8000.0,
                stop_loss=0.12,
                take_profit=0.25,   # Add take-profit
                fees=0.0025
            )
        }
    }
    
    # Create portfolios for each strategy
    portfolios = {}
    
    for strategy_name, strategy_config in strategies.items():
        logger.info(f"Creating portfolio for {strategy_name} strategy...")
        
        viz_engine = VectorBTVisualizationEngine(
            portfolio_config=strategy_config['config']
        )
        
        portfolio = viz_engine.create_portfolio_from_predictions(
            strategy_config['predictions'], 
            price_data, 
            test_start_idx, 
            symbol=f"COMP_{strategy_name.upper()}"
        )
        
        portfolios[strategy_name] = portfolio
    
    # Generate comparison visualization
    logger.info("Generating multi-strategy comparison...")
    comparison_viz_engine = VectorBTVisualizationEngine()  # Use default config for comparison
    
    comparison_result = comparison_viz_engine.generate_comparison_plot(
        portfolios, 
        title="Multi-Strategy Performance Comparison"
    )
    
    if comparison_result.success:
        logger.info(f"Comparison visualization generated in {comparison_result.generation_time:.2f}s")
        
        # Display comparison metrics
        plot_data = comparison_result.plot_data
        metrics_df = plot_data['metrics_comparison']
        rankings = plot_data['strategy_rankings']
        
        print(f"\nStrategy Performance Comparison:")
        print(f"{'Strategy':<15} {'Return':<10} {'Sharpe':<8} {'Drawdown':<10} {'Rank':<6}")
        print("-" * 55)
        
        for strategy in portfolios.keys():
            if strategy in metrics_df.index:
                metrics = metrics_df.loc[strategy]
                return_val = metrics.get('total_return', 0)
                sharpe_val = metrics.get('sharpe_ratio', 0)
                drawdown_val = metrics.get('max_drawdown', 0)
                rank = rankings.get(strategy, 'N/A')
                
                print(f"{strategy:<15} {return_val:>8.2%} {sharpe_val:>7.2f} {drawdown_val:>9.2%} {rank:>5}")
        
        try:
            comparison_viz_engine.show_plot(comparison_result)
        except Exception as e:
            logger.warning(f"Could not display plot interactively: {e}")
        
        return comparison_result
    else:
        logger.error(f"Comparison visualization failed: {comparison_result.error_message}")
        return None


def example_4_advanced_configuration():
    """
    Example 4: Advanced Configuration and Customization
    
    Demonstrates advanced portfolio configuration options and plot customization.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Advanced Configuration and Customization")
    print("="*60)
    
    # Create market data
    logger.info("Creating market data for advanced configuration example...")
    price_data = create_sample_market_data(120, "ADVANCED", "bull")
    predictions = create_sample_predictions(30, "balanced")
    test_start_idx = 90
    
    # Advanced portfolio configuration
    advanced_portfolio_config = PortfolioConfig(
        init_cash=250000.0,
        fees=0.002,
        slippage=0.003,
        size_strategy='volatility_target',
        size_value=0.15,  # Target 15% volatility
        stop_loss=0.12,
        take_profit=0.30,
        upon_opposite_entry='close',  # Close position on opposite signal
        freq='D'
    )
    
    # Advanced plot configuration
    advanced_plot_config = PlotConfig(
        width=1400,
        height=800,
        show_trades=True,
        show_positions=True,
        show_cash=True,
        template='plotly_dark',
        color_scheme='viridis',
        show_metrics=True,
        metric_position='top_right'
    )
    
    # Initialize with advanced configuration
    viz_engine = VectorBTVisualizationEngine(
        portfolio_config=advanced_portfolio_config,
        plot_config=advanced_plot_config,
        enable_performance_optimization=True
    )
    
    # Create portfolio
    logger.info("Creating portfolio with advanced configuration...")
    portfolio = viz_engine.create_portfolio_from_predictions(
        predictions, price_data, test_start_idx, symbol="ADVANCED"
    )
    
    # Generate enhanced visualization
    logger.info("Generating enhanced visualization with custom styling...")
    viz_result = viz_engine.generate_portfolio_plot(
        portfolio, title="Advanced Configuration Example - Enhanced Styling"
    )
    
    if viz_result.success:
        logger.info(f"Enhanced visualization generated in {viz_result.generation_time:.2f}s")
        
        # Display advanced metrics
        metrics = viz_result.metrics_summary
        print(f"\nAdvanced Portfolio Metrics:")
        print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        
        # Demonstrate portfolio validation
        portfolio_engine = EnhancedPortfolioEngine(advanced_portfolio_config)
        
        # Get aligned signals for validation
        signal_aligner = SignalAlignmentEngine()
        aligned_signals = signal_aligner.align_predictions_to_timeline(
            predictions, price_data, test_start_idx
        )
        
        validation_result = portfolio_engine.validate_portfolio_parameters(
            advanced_portfolio_config,
            price_data['Close'],
            aligned_signals.entry_signals,
            aligned_signals.exit_signals
        )
        
        print(f"\nPortfolio Validation Results:")
        print(f"  Valid Configuration: {validation_result['valid']}")
        print(f"  Warnings: {len(validation_result['warnings'])}")
        print(f"  Recommendations: {len(validation_result['recommendations'])}")
        
        if validation_result['warnings']:
            print("  Warnings:")
            for warning in validation_result['warnings']:
                print(f"    - {warning}")
        
        if validation_result['recommendations']:
            print("  Recommendations:")
            for rec in validation_result['recommendations']:
                print(f"    - {rec}")
        
        try:
            viz_engine.show_plot(viz_result)
        except Exception as e:
            logger.warning(f"Could not display plot interactively: {e}")
        
        return viz_result
    else:
        logger.error(f"Enhanced visualization failed: {viz_result.error_message}")
        return None


def example_5_export_and_reporting():
    """
    Example 5: Export and Comprehensive Reporting
    
    Demonstrates plot export capabilities and comprehensive report generation.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Export and Comprehensive Reporting")
    print("="*60)
    
    # Create sample portfolio
    logger.info("Creating sample portfolio for export demonstration...")
    price_data = create_sample_market_data(100, "EXPORT", "normal")
    predictions = create_sample_predictions(25, "balanced")
    test_start_idx = 75
    
    portfolio_config = PortfolioConfig(
        init_cash=100000.0,
        size_strategy='fixed_amount',
        size_value=8000.0
    )
    
    viz_engine = VectorBTVisualizationEngine(portfolio_config=portfolio_config)
    
    portfolio = viz_engine.create_portfolio_from_predictions(
        predictions, price_data, test_start_idx, symbol="EXPORT"
    )
    
    # Generate multiple visualizations
    logger.info("Generating multiple visualizations for export...")
    
    portfolio_viz = viz_engine.generate_portfolio_plot(
        portfolio, title="Export Example - Portfolio Performance"
    )
    
    drawdown_viz = viz_engine.generate_drawdown_plot(portfolio)
    
    # Initialize export engine
    export_engine = PlotExportEngine()
    
    # Create export directory
    export_dir = Path("./visualization_exports")
    export_dir.mkdir(exist_ok=True)
    
    exported_files = []
    
    if portfolio_viz.success:
        # Export portfolio plot in multiple formats
        logger.info("Exporting portfolio visualization...")
        portfolio_exports = export_engine.export_plot(
            portfolio_viz.plot_object,
            str(export_dir / "portfolio_performance"),
            formats=['png', 'html', 'svg']
        )
        exported_files.extend(portfolio_exports.values())
        
        print(f"\nPortfolio visualization exported:")
        for format_type, file_path in portfolio_exports.items():
            print(f"  {format_type.upper()}: {file_path}")
    
    if drawdown_viz.success:
        # Export drawdown plot
        logger.info("Exporting drawdown visualization...")
        drawdown_exports = export_engine.export_plot(
            drawdown_viz.plot_object,
            str(export_dir / "drawdown_analysis"),
            formats=['png', 'html']
        )
        exported_files.extend(drawdown_exports.values())
        
        print(f"\nDrawdown visualization exported:")
        for format_type, file_path in drawdown_exports.items():
            print(f"  {format_type.upper()}: {file_path}")
    
    # Export underlying data
    logger.info("Exporting portfolio data...")
    data_export_path = export_engine.export_plot_data(
        portfolio,
        str(export_dir / "portfolio_data.csv")
    )
    exported_files.append(data_export_path)
    
    print(f"\nPortfolio data exported: {data_export_path}")
    
    # Generate comprehensive report (if supported)
    try:
        logger.info("Generating comprehensive PDF report...")
        report_path = export_engine.generate_comprehensive_report(
            portfolios={'Export Example': portfolio},
            output_path=str(export_dir / "comprehensive_report.pdf"),
            include_plots=True,
            include_metrics=True,
            include_trade_analysis=True
        )
        exported_files.append(report_path)
        
        print(f"\nComprehensive report generated: {report_path}")
        
    except Exception as e:
        logger.warning(f"Could not generate PDF report: {e}")
    
    print(f"\nTotal files exported: {len(exported_files)}")
    print(f"Export directory: {export_dir.absolute()}")
    
    return exported_files


def example_6_performance_optimization():
    """
    Example 6: Performance Optimization for Large Datasets
    
    Demonstrates performance optimization techniques for handling large datasets.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Performance Optimization for Large Datasets")
    print("="*60)
    
    # Create large dataset
    logger.info("Creating large dataset (2 years of daily data)...")
    large_price_data = create_sample_market_data(500, "LARGE", "normal")  # ~1.5 years
    large_predictions = create_sample_predictions(100, "balanced")
    test_start_idx = 400
    
    # Configure for performance optimization
    from stock_predictor.visualization.performance_optimizer import OptimizationConfig
    
    optimization_config = OptimizationConfig(
        enable_data_sampling=True,
        max_data_points=1000,
        sampling_strategy='adaptive',
        enable_caching=True,
        memory_limit_mb=500
    )
    
    portfolio_config = PortfolioConfig(
        init_cash=100000.0,
        size_strategy='fixed_amount',
        size_value=10000.0
    )
    
    # Initialize with performance optimization
    viz_engine = VectorBTVisualizationEngine(
        portfolio_config=portfolio_config,
        optimization_config=optimization_config,
        enable_performance_optimization=True
    )
    
    # Measure performance
    import time
    
    start_time = time.time()
    
    logger.info("Creating optimized portfolio...")
    portfolio = viz_engine.create_portfolio_from_predictions(
        large_predictions, large_price_data, test_start_idx, symbol="LARGE"
    )
    
    portfolio_time = time.time() - start_time
    
    logger.info("Generating optimized visualization...")
    viz_start = time.time()
    
    viz_result = viz_engine.generate_portfolio_plot(
        portfolio, title="Performance Optimization Example - Large Dataset"
    )
    
    viz_time = time.time() - viz_start
    total_time = time.time() - start_time
    
    if viz_result.success:
        logger.info(f"Large dataset processing completed successfully!")
        
        print(f"\nPerformance Metrics:")
        print(f"  Dataset Size: {len(large_price_data)} days")
        print(f"  Prediction Count: {len(large_predictions)}")
        print(f"  Portfolio Creation Time: {portfolio_time:.2f}s")
        print(f"  Visualization Time: {viz_time:.2f}s")
        print(f"  Total Processing Time: {total_time:.2f}s")
        print(f"  Processing Rate: {len(large_price_data)/total_time:.1f} days/second")
        
        # Check if optimization was applied
        if hasattr(viz_result, 'performance_metrics'):
            perf_metrics = viz_result.performance_metrics
            print(f"  Optimization Applied: {perf_metrics.get('optimization_applied', 'Unknown')}")
            print(f"  Memory Usage: {perf_metrics.get('memory_usage_mb', 0):.1f} MB")
        
        # Display portfolio metrics
        metrics = viz_result.metrics_summary
        print(f"\nPortfolio Performance:")
        print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  Number of Trades: {portfolio.trades.count()}")
        
        try:
            viz_engine.show_plot(viz_result)
        except Exception as e:
            logger.warning(f"Could not display plot interactively: {e}")
        
        return viz_result
    else:
        logger.error(f"Optimized visualization failed: {viz_result.error_message}")
        return None


def run_all_examples():
    """
    Run all visualization examples in sequence.
    """
    print("VectorBT Visualization Enhancement - Complete Examples")
    print("=" * 80)
    
    examples = [
        example_1_basic_portfolio_visualization,
        example_2_drawdown_analysis,
        example_3_multi_strategy_comparison,
        example_4_advanced_configuration,
        example_5_export_and_reporting,
        example_6_performance_optimization
    ]
    
    results = {}
    
    for i, example_func in enumerate(examples, 1):
        try:
            result = example_func()
            results[f"example_{i}"] = {
                'success': result is not None,
                'result': result
            }
        except Exception as e:
            logger.error(f"Example {i} failed: {e}")
            results[f"example_{i}"] = {
                'success': False,
                'error': str(e)
            }
    
    # Summary
    print("\n" + "="*80)
    print("EXAMPLES SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"Completed: {successful}/{total} examples")
    
    for example_name, result in results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"  {example_name}: {status}")
        if not result['success'] and 'error' in result:
            print(f"    Error: {result['error']}")
    
    return results


if __name__ == "__main__":
    # Run all examples
    results = run_all_examples()
    
    # Print final message
    successful = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    if successful == total:
        print(f"\n🎉 All {total} examples completed successfully!")
    else:
        print(f"\n⚠️  {successful}/{total} examples completed successfully.")
        print("Check the logs above for details on any failures.")
    
    print("\nFor more detailed documentation, see:")
    print("  - docs/vectorbt_visualization_user_guide.md")
    print("  - Individual example functions in this file")
    print("  - API documentation in the source code")