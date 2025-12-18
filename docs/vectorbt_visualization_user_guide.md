# VectorBT Visualization Enhancement User Guide

## Overview

The VectorBT Visualization Enhancement extends the Stock Direction Predictor system with comprehensive, interactive portfolio visualization capabilities. This enhancement leverages VectorBT's built-in plotting functionality to provide professional-grade visual analytics for trading strategy evaluation.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Components](#core-components)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Configuration Options](#configuration-options)
6. [Integration Guide](#integration-guide)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

## Quick Start

### Installation Requirements

Ensure you have the required dependencies installed:

```bash
pip install vectorbt plotly pandas numpy
```

### Basic Example

```python
import numpy as np
import pandas as pd
from stock_predictor.visualization import (
    VectorBTVisualizationEngine,
    EnhancedPortfolioEngine,
    PortfolioConfig,
    PlotConfig
)

# Create sample data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
prices = pd.DataFrame({
    'Close': np.random.randn(100).cumsum() + 100,
    'Open': np.random.randn(100).cumsum() + 99,
    'High': np.random.randn(100).cumsum() + 101,
    'Low': np.random.randn(100).cumsum() + 98,
    'Volume': np.random.randint(1000000, 5000000, 100)
}, index=dates)

# Create ML predictions (0=sell, 1=hold, 2=buy)
predictions = np.random.choice([0, 1, 2], size=20, p=[0.2, 0.6, 0.2])
test_start_idx = 80

# Initialize visualization engine
portfolio_config = PortfolioConfig(
    init_cash=100000.0,
    fees=0.0025,
    size_strategy='fixed_amount',
    size_value=10000.0
)

viz_engine = VectorBTVisualizationEngine(portfolio_config=portfolio_config)

# Create portfolio from predictions
portfolio = viz_engine.create_portfolio_from_predictions(
    predictions, prices, test_start_idx
)

# Generate interactive visualization
result = viz_engine.generate_portfolio_plot(portfolio)

# Display the plot
viz_engine.show_plot(result)
```

## Core Components

### 1. VectorBTVisualizationEngine

The main visualization engine that orchestrates the entire visualization process.

**Key Features:**
- Interactive portfolio performance plots
- Drawdown analysis visualization
- Trade performance analysis
- Multi-strategy comparison plots
- Comprehensive error handling

### 2. EnhancedPortfolioEngine

Handles portfolio creation with realistic trading parameters and advanced risk management.

**Key Features:**
- Signal alignment from ML predictions
- Multiple position sizing strategies
- Risk management (stop-loss, take-profit)
- Portfolio parameter validation and optimization

### 3. SignalAlignmentEngine

Properly aligns ML predictions with complete historical data timeline.

**Key Features:**
- Full timeline signal alignment
- Prediction-to-signal conversion
- Validation and error handling

### 4. Configuration Classes

#### PortfolioConfig
Controls portfolio creation parameters:

```python
portfolio_config = PortfolioConfig(
    init_cash=100000.0,          # Starting capital
    fees=0.0025,                 # Transaction fees (0.25%)
    slippage=0.0025,             # Slippage (0.25%)
    size_strategy='fixed_amount', # Position sizing strategy
    size_value=10000.0,          # Position size value
    stop_loss=0.1,               # Stop loss (10%)
    take_profit=None,            # Take profit (optional)
    upon_opposite_entry='ignore' # Opposite signal handling
)
```

#### PlotConfig
Controls visualization appearance:

```python
plot_config = PlotConfig(
    width=1200,                  # Plot width
    height=600,                  # Plot height
    show_trades=True,            # Show trade markers
    show_positions=True,         # Show position data
    template='plotly_white',     # Plot template
    export_formats=['png', 'html'] # Export formats
)
```

## Basic Usage

### Creating Portfolios from ML Predictions

```python
from stock_predictor.visualization import EnhancedPortfolioEngine, PortfolioConfig

# Initialize portfolio engine
portfolio_engine = EnhancedPortfolioEngine()

# Create portfolio from predictions
portfolio_result = portfolio_engine.create_portfolio_from_predictions(
    predictions=predictions,      # ML predictions array
    price_data=price_data,       # Historical price data
    test_start_idx=test_start_idx, # Where test period begins
    config=portfolio_config      # Portfolio configuration
)

if portfolio_result.success:
    portfolio = portfolio_result.portfolio
    print(f"Portfolio created with {portfolio.trades.count()} trades")
else:
    print(f"Portfolio creation failed: {portfolio_result.error_message}")
```

### Generating Visualizations

#### Portfolio Performance Plot

```python
# Generate main portfolio visualization
viz_result = viz_engine.generate_portfolio_plot(
    portfolio, 
    title="My Trading Strategy Performance"
)

if viz_result.success:
    # Display the plot
    viz_engine.show_plot(viz_result)
    
    # Access plot data
    portfolio_value = viz_result.plot_data['portfolio_value']
    metrics = viz_result.metrics_summary
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
else:
    print(f"Visualization failed: {viz_result.error_message}")
```

#### Drawdown Analysis

```python
# Generate drawdown visualization
drawdown_result = viz_engine.generate_drawdown_plot(portfolio)

if drawdown_result.success:
    viz_engine.show_plot(drawdown_result)
    
    # Access drawdown metrics
    metrics = drawdown_result.metrics_summary
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2%}")
    print(f"Avg Recovery Time: {metrics['avg_recovery_time']:.1f} days")
```

#### Trade Analysis

```python
# Generate trade analysis (if trades exist)
if portfolio.trades.count() > 0:
    trade_result = viz_engine.generate_trade_analysis_plot(portfolio)
    
    if trade_result.success:
        viz_engine.show_plot(trade_result)
        
        # Access trade metrics
        metrics = trade_result.metrics_summary
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
```

### Multi-Strategy Comparison

```python
# Create multiple portfolios for comparison
strategies = {
    'Conservative': conservative_portfolio,
    'Aggressive': aggressive_portfolio,
    'Balanced': balanced_portfolio
}

# Generate comparison visualization
comparison_result = viz_engine.generate_comparison_plot(
    strategies, 
    title="Strategy Performance Comparison"
)

if comparison_result.success:
    viz_engine.show_plot(comparison_result)
    
    # Access comparison data
    rankings = comparison_result.plot_data['strategy_rankings']
    print("Strategy Rankings:")
    for strategy, rank in rankings.items():
        print(f"  {strategy}: Rank {rank}")
```

## Advanced Features

### Custom Position Sizing Strategies

```python
# Fixed amount sizing
portfolio_config = PortfolioConfig(
    size_strategy='fixed_amount',
    size_value=10000.0  # $10,000 per trade
)

# Percentage of equity sizing
portfolio_config = PortfolioConfig(
    size_strategy='percent_equity',
    size_value=0.1  # 10% of available equity
)

# Fixed number of shares
portfolio_config = PortfolioConfig(
    size_strategy='fixed_shares',
    size_value=100  # 100 shares per trade
)

# Volatility-based sizing
portfolio_config = PortfolioConfig(
    size_strategy='volatility_target',
    size_value=0.15  # Target 15% volatility
)
```

### Risk Management

```python
# Configure stop-loss and take-profit
portfolio_config = PortfolioConfig(
    stop_loss=0.1,        # 10% stop-loss
    take_profit=0.2,      # 20% take-profit
    upon_opposite_entry='close'  # Close position on opposite signal
)
```

### Performance Optimization

```python
from stock_predictor.visualization import OptimizationConfig

# Enable performance optimization
optimization_config = OptimizationConfig(
    enable_data_sampling=True,
    max_data_points=10000,
    sampling_strategy='adaptive',
    enable_caching=True
)

viz_engine = VectorBTVisualizationEngine(
    portfolio_config=portfolio_config,
    optimization_config=optimization_config,
    enable_performance_optimization=True
)
```

### Export and Reporting

```python
from stock_predictor.visualization import PlotExportEngine

# Initialize export engine
export_engine = PlotExportEngine()

# Export plots in multiple formats
export_paths = export_engine.export_plot(
    viz_result.plot_object,
    filename="my_strategy_analysis",
    formats=['png', 'html', 'svg']
)

print(f"Plots exported to: {export_paths}")

# Export underlying data
data_path = export_engine.export_plot_data(
    portfolio,
    filename="strategy_data.csv"
)

print(f"Data exported to: {data_path}")

# Generate comprehensive PDF report
report_path = export_engine.generate_comprehensive_report(
    portfolios={'My Strategy': portfolio},
    output_path="strategy_report.pdf"
)

print(f"Report generated: {report_path}")
```

## Configuration Options

### Portfolio Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `init_cash` | float | 10000.0 | Initial capital |
| `fees` | float | 0.0025 | Transaction fees (as decimal) |
| `slippage` | float | 0.0025 | Slippage (as decimal) |
| `size_strategy` | str | 'fixed_amount' | Position sizing method |
| `size_value` | float | 40.0 | Position size value |
| `stop_loss` | float | None | Stop-loss threshold |
| `take_profit` | float | None | Take-profit threshold |
| `upon_opposite_entry` | str | 'ignore' | Opposite signal handling |
| `freq` | str | 'D' | Trading frequency |

### Plot Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | int | 1200 | Plot width in pixels |
| `height` | int | 600 | Plot height in pixels |
| `show_trades` | bool | True | Show trade markers |
| `show_positions` | bool | True | Show position data |
| `show_cash` | bool | False | Show cash levels |
| `template` | str | 'plotly_white' | Plotly template |
| `color_scheme` | str | 'default' | Color scheme |
| `export_formats` | list | ['png', 'html'] | Default export formats |

## Integration Guide

### Streamlit Dashboard Integration

```python
import streamlit as st
from stock_predictor.visualization import VectorBTVisualizationEngine

# Create visualization
viz_result = viz_engine.generate_portfolio_plot(portfolio)

if viz_result.success:
    # Display in Streamlit
    st.plotly_chart(viz_result.plot_object, use_container_width=True)
    
    # Show metrics
    metrics = viz_result.metrics_summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Return", f"{metrics['total_return']:.2%}")
    
    with col2:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    
    with col3:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
```

### Flask/FastAPI Integration

```python
from flask import Flask, jsonify, render_template_string

app = Flask(__name__)

@app.route('/api/portfolio/<strategy_name>')
def get_portfolio_visualization(strategy_name):
    # Generate visualization
    viz_result = viz_engine.generate_portfolio_plot(portfolio)
    
    if viz_result.success:
        return {
            'success': True,
            'plot_html': viz_result.plot_object.to_html(),
            'metrics': viz_result.metrics_summary,
            'generation_time': viz_result.generation_time
        }
    else:
        return {
            'success': False,
            'error': viz_result.error_message
        }, 400

@app.route('/portfolio/<strategy_name>')
def show_portfolio(strategy_name):
    viz_result = viz_engine.generate_portfolio_plot(portfolio)
    
    if viz_result.success:
        plot_html = viz_result.plot_object.to_html()
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head><title>Portfolio Analysis</title></head>
        <body>
            <h1>{{ strategy_name }} Performance</h1>
            {{ plot_html|safe }}
        </body>
        </html>
        """, strategy_name=strategy_name, plot_html=plot_html)
    else:
        return f"Error: {viz_result.error_message}", 500
```

### Jupyter Notebook Integration

```python
# In Jupyter notebook
from IPython.display import display, HTML

# Generate visualization
viz_result = viz_engine.generate_portfolio_plot(portfolio)

if viz_result.success:
    # Display inline
    viz_result.plot_object.show()
    
    # Or display as HTML
    display(HTML(viz_result.plot_object.to_html()))
    
    # Show metrics table
    import pandas as pd
    metrics_df = pd.DataFrame([viz_result.metrics_summary])
    display(metrics_df)
```

## Performance Optimization

### Large Dataset Handling

```python
# Enable data sampling for large datasets
optimization_config = OptimizationConfig(
    enable_data_sampling=True,
    max_data_points=5000,
    sampling_strategy='adaptive'
)

viz_engine = VectorBTVisualizationEngine(
    optimization_config=optimization_config
)
```

### Memory Management

```python
# Enable memory optimization
viz_engine = VectorBTVisualizationEngine(
    enable_performance_optimization=True
)

# Manual cleanup for batch processing
import gc

for strategy in strategies:
    viz_result = viz_engine.generate_portfolio_plot(strategy)
    # Process result...
    
    # Clean up
    del viz_result
    gc.collect()
```

### Caching

```python
# Enable result caching
viz_engine = VectorBTVisualizationEngine(
    enable_caching=True,
    cache_directory="./viz_cache"
)
```

## Troubleshooting

### Common Issues

#### 1. VectorBT Import Errors

```python
# Error: ModuleNotFoundError: No module named 'vectorbt'
# Solution: Install VectorBT
pip install vectorbt
```

#### 2. Widget Display Issues

```python
# Error: "Please install anywidget to use the FigureWidget class"
# Solution: Install widget dependencies
pip install anywidget ipywidgets

# Or disable widgets in headless environments
viz_engine = VectorBTVisualizationEngine(
    enable_error_handling=True  # Enables fallback to static plots
)
```

#### 3. Memory Issues with Large Datasets

```python
# Enable performance optimization
viz_engine = VectorBTVisualizationEngine(
    enable_performance_optimization=True
)

# Or reduce dataset size
optimization_config = OptimizationConfig(
    max_data_points=1000,
    enable_data_sampling=True
)
```

#### 4. Signal Alignment Errors

```python
# Ensure data lengths match
assert len(predictions) <= len(price_data) - test_start_idx

# Validate prediction values
assert all(p in [0, 1, 2] for p in predictions)

# Check test_start_idx
assert 0 <= test_start_idx < len(price_data)
```

### Error Handling

```python
# Comprehensive error handling
try:
    portfolio_result = portfolio_engine.create_portfolio_from_predictions(
        predictions, price_data, test_start_idx
    )
    
    if not portfolio_result.success:
        print(f"Portfolio creation failed: {portfolio_result.error_message}")
        return
    
    viz_result = viz_engine.generate_portfolio_plot(portfolio_result.portfolio)
    
    if not viz_result.success:
        print(f"Visualization failed: {viz_result.error_message}")
        return
    
    viz_engine.show_plot(viz_result)
    
except Exception as e:
    print(f"Unexpected error: {e}")
    # Log error details for debugging
    import traceback
    traceback.print_exc()
```

## API Reference

### VectorBTVisualizationEngine

#### Methods

- `create_portfolio_from_predictions(predictions, price_data, test_start_idx, symbol='ASSET')` → `vbt.Portfolio`
- `generate_portfolio_plot(portfolio, title=None)` → `VisualizationResult`
- `generate_drawdown_plot(portfolio)` → `VisualizationResult`
- `generate_trade_analysis_plot(portfolio)` → `VisualizationResult`
- `generate_comparison_plot(portfolios, title)` → `VisualizationResult`
- `show_plot(result)` → `None`

### EnhancedPortfolioEngine

#### Methods

- `create_portfolio_from_predictions(predictions, price_data, test_start_idx, config=None)` → `PortfolioCreationResult`
- `create_vectorbt_portfolio(close_prices, entry_signals, exit_signals, config=None)` → `vbt.Portfolio`
- `calculate_position_sizes(prices, sizing_method, capital)` → `pd.Series`
- `validate_portfolio_parameters(config, prices, entry_signals, exit_signals)` → `Dict[str, Any]`

### SignalAlignmentEngine

#### Methods

- `align_predictions_to_timeline(predictions, full_data, test_start_idx)` → `AlignedSignals`
- `convert_predictions_to_signals(predictions)` → `Tuple[np.ndarray, np.ndarray]`

For complete API documentation, see the inline docstrings in each module.

## Examples and Tutorials

See the `examples/` directory for comprehensive examples:

- `basic_visualization_example.py` - Basic usage patterns
- `advanced_features_example.py` - Advanced configuration and features
- `multi_strategy_comparison_example.py` - Comparing multiple strategies
- `dashboard_integration_example.py` - Web dashboard integration
- `performance_optimization_example.py` - Optimizing for large datasets

## Support and Contributing

For issues, questions, or contributions, please refer to the project repository and documentation.