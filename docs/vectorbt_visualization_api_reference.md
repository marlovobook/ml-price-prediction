# VectorBT Visualization Enhancement API Reference

## Overview

This document provides comprehensive API documentation for the VectorBT Visualization Enhancement system. The system consists of several key components that work together to provide professional-grade portfolio visualization capabilities.

## Core Classes

### VectorBTVisualizationEngine

The main visualization engine that orchestrates the entire visualization process.

```python
class VectorBTVisualizationEngine:
    def __init__(
        self, 
        portfolio_config: Optional[PortfolioConfig] = None,
        plot_config: Optional[PlotConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        enable_error_handling: bool = True,
        enable_performance_optimization: bool = True
    )
```

#### Parameters

- **portfolio_config** (`PortfolioConfig`, optional): Configuration for portfolio creation
- **plot_config** (`PlotConfig`, optional): Configuration for plot styling and behavior
- **optimization_config** (`OptimizationConfig`, optional): Configuration for performance optimization
- **enable_error_handling** (`bool`): Whether to enable comprehensive error handling
- **enable_performance_optimization** (`bool`): Whether to enable performance optimization

#### Methods

##### create_portfolio_from_predictions

```python
def create_portfolio_from_predictions(
    self,
    predictions: np.ndarray,
    price_data: pd.DataFrame,
    test_start_idx: int,
    symbol: str = 'ASSET'
) -> vbt.Portfolio
```

Create VectorBT portfolio from ML predictions with proper signal alignment.

**Parameters:**
- **predictions** (`np.ndarray`): ML model predictions (0=sell, 1=hold, 2=buy)
- **price_data** (`pd.DataFrame`): Historical price data with 'Close' column
- **test_start_idx** (`int`): Index where test period begins
- **symbol** (`str`): Asset symbol for labeling

**Returns:**
- `vbt.Portfolio`: VectorBT Portfolio object ready for visualization

**Raises:**
- `BacktestingError`: If portfolio creation fails

##### generate_portfolio_plot

```python
def generate_portfolio_plot(
    self, 
    portfolio: vbt.Portfolio,
    title: Optional[str] = None
) -> VisualizationResult
```

Generate interactive portfolio performance plot using VectorBT.

**Parameters:**
- **portfolio** (`vbt.Portfolio`): VectorBT portfolio object
- **title** (`str`, optional): Optional plot title

**Returns:**
- `VisualizationResult`: Result object containing plot and metadata

##### generate_drawdown_plot

```python
def generate_drawdown_plot(
    self, 
    portfolio: vbt.Portfolio
) -> VisualizationResult
```

Generate enhanced drawdown analysis plot with detailed period highlighting and recovery analysis.

**Parameters:**
- **portfolio** (`vbt.Portfolio`): VectorBT portfolio object

**Returns:**
- `VisualizationResult`: Result object with enhanced drawdown plot

##### generate_trade_analysis_plot

```python
def generate_trade_analysis_plot(
    self, 
    portfolio: vbt.Portfolio
) -> VisualizationResult
```

Generate trade performance analysis plot.

**Parameters:**
- **portfolio** (`vbt.Portfolio`): VectorBT portfolio object

**Returns:**
- `VisualizationResult`: Result object with trade analysis plot

##### generate_comparison_plot

```python
def generate_comparison_plot(
    self, 
    portfolios: Dict[str, vbt.Portfolio],
    title: str = "Multi-Strategy Comparison Analysis"
) -> VisualizationResult
```

Generate comprehensive multi-strategy comparison plot.

**Parameters:**
- **portfolios** (`Dict[str, vbt.Portfolio]`): Dictionary of named portfolio objects
- **title** (`str`): Plot title

**Returns:**
- `VisualizationResult`: Result object with enhanced comparison plot

##### show_plot

```python
def show_plot(self, result: VisualizationResult) -> None
```

Display the generated plot.

**Parameters:**
- **result** (`VisualizationResult`): VisualizationResult containing the plot object

### EnhancedPortfolioEngine

Handles portfolio creation with realistic trading parameters and advanced risk management.

```python
class EnhancedPortfolioEngine:
    def __init__(
        self, 
        portfolio_config: Optional[PortfolioConfig] = None,
        signal_aligner: Optional[SignalAlignmentEngine] = None
    )
```

#### Parameters

- **portfolio_config** (`PortfolioConfig`, optional): Configuration for portfolio creation
- **signal_aligner** (`SignalAlignmentEngine`, optional): Signal alignment engine

#### Methods

##### create_portfolio_from_predictions

```python
def create_portfolio_from_predictions(
    self,
    predictions: np.ndarray,
    price_data: pd.DataFrame,
    test_start_idx: int,
    config: Optional[PortfolioConfig] = None
) -> PortfolioCreationResult
```

Create VectorBT portfolio from ML predictions with signal alignment.

**Parameters:**
- **predictions** (`np.ndarray`): ML model predictions (0=sell, 1=hold, 2=buy)
- **price_data** (`pd.DataFrame`): Historical price data with 'Close' column
- **test_start_idx** (`int`): Index where test period begins
- **config** (`PortfolioConfig`, optional): Portfolio configuration

**Returns:**
- `PortfolioCreationResult`: Result object with portfolio and metadata

##### create_vectorbt_portfolio

```python
def create_vectorbt_portfolio(
    self,
    close_prices: pd.Series,
    entry_signals: pd.Series,
    exit_signals: pd.Series,
    config: Optional[PortfolioConfig] = None
) -> vbt.Portfolio
```

Create VectorBT portfolio with enhanced configuration.

**Parameters:**
- **close_prices** (`pd.Series`): Historical closing prices
- **entry_signals** (`pd.Series`): Boolean series for entry points
- **exit_signals** (`pd.Series`): Boolean series for exit points
- **config** (`PortfolioConfig`, optional): Portfolio configuration parameters

**Returns:**
- `vbt.Portfolio`: Configured VectorBT Portfolio object

##### calculate_position_sizes

```python
def calculate_position_sizes(
    self,
    prices: pd.Series,
    sizing_method: str,
    capital: float
) -> pd.Series
```

Calculate position sizes based on strategy.

**Parameters:**
- **prices** (`pd.Series`): Price series for sizing calculation
- **sizing_method** (`str`): 'fixed_amount', 'fixed_shares', 'percent_equity', 'volatility_target', 'risk_parity'
- **capital** (`float`): Available capital

**Returns:**
- `pd.Series`: Series of position sizes (in dollar amounts)

##### validate_portfolio_parameters

```python
def validate_portfolio_parameters(
    self,
    config: PortfolioConfig,
    prices: pd.Series,
    entry_signals: pd.Series,
    exit_signals: pd.Series
) -> Dict[str, Any]
```

Validate portfolio parameters before creation.

**Parameters:**
- **config** (`PortfolioConfig`): Portfolio configuration to validate
- **prices** (`pd.Series`): Price series
- **entry_signals** (`pd.Series`): Entry signals
- **exit_signals** (`pd.Series`): Exit signals

**Returns:**
- `Dict[str, Any]`: Dictionary with validation results and recommendations

### SignalAlignmentEngine

Properly aligns ML predictions with complete historical data timeline.

```python
class SignalAlignmentEngine:
    def __init__(self)
```

#### Methods

##### align_predictions_to_timeline

```python
def align_predictions_to_timeline(
    self, 
    predictions: np.ndarray, 
    full_data: pd.DataFrame,
    test_start_idx: int
) -> AlignedSignals
```

Align predictions to full data timeline.

**Parameters:**
- **predictions** (`np.ndarray`): ML model predictions array
- **full_data** (`pd.DataFrame`): Complete historical dataset
- **test_start_idx** (`int`): Index where test period begins

**Returns:**
- `AlignedSignals`: Aligned signals object with entry/exit signals

##### convert_predictions_to_signals

```python
def convert_predictions_to_signals(
    self, 
    predictions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]
```

Convert prediction values to entry/exit signals.

**Parameters:**
- **predictions** (`np.ndarray`): Array of prediction values (0, 1, 2)

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: Tuple of (entry_signals, exit_signals) boolean arrays

## Configuration Classes

### PortfolioConfig

Configuration class for portfolio creation parameters.

```python
@dataclass
class PortfolioConfig:
    init_cash: float = 10000.0
    size_strategy: str = 'fixed_amount'
    size_value: float = 40.0
    fees: float = 0.0025
    slippage: float = 0.0025
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    upon_opposite_entry: str = 'ignore'
    freq: str = 'D'
    accumulate: bool = False
    conflict_mode: str = 'ignore'
```

#### Parameters

- **init_cash** (`float`): Initial capital
- **size_strategy** (`str`): Position sizing method ('fixed_amount', 'fixed_shares', 'percent_equity', 'volatility_target', 'risk_parity')
- **size_value** (`float`): Position size value
- **fees** (`float`): Transaction fees (as decimal)
- **slippage** (`float`): Slippage (as decimal)
- **stop_loss** (`float`, optional): Stop-loss threshold
- **take_profit** (`float`, optional): Take-profit threshold
- **upon_opposite_entry** (`str`): Opposite signal handling ('ignore', 'close', 'reverse')
- **freq** (`str`): Trading frequency
- **accumulate** (`bool`): Whether to accumulate positions
- **conflict_mode** (`str`): Signal conflict handling

#### Methods

##### validate

```python
def validate(self) -> None
```

Validate configuration parameters.

**Raises:**
- `DataValidationError`: If configuration is invalid

##### to_vectorbt_params

```python
def to_vectorbt_params(self) -> Dict[str, Any]
```

Convert configuration to VectorBT parameters.

**Returns:**
- `Dict[str, Any]`: Dictionary of VectorBT parameters

### PlotConfig

Configuration class for plot styling and behavior.

```python
@dataclass
class PlotConfig:
    width: int = 1200
    height: int = 600
    show_trades: bool = True
    show_positions: bool = True
    show_cash: bool = False
    template: str = 'plotly_white'
    color_scheme: str = 'default'
    show_metrics: bool = True
    metric_position: str = 'top_right'
    export_formats: List[str] = field(default_factory=lambda: ['png', 'html'])
    export_directory: str = 'visualizations'
```

#### Parameters

- **width** (`int`): Plot width in pixels
- **height** (`int`): Plot height in pixels
- **show_trades** (`bool`): Show trade markers
- **show_positions** (`bool`): Show position data
- **show_cash** (`bool`): Show cash levels
- **template** (`str`): Plotly template
- **color_scheme** (`str`): Color scheme
- **show_metrics** (`bool`): Show performance metrics
- **metric_position** (`str`): Position of metrics display
- **export_formats** (`List[str]`): Default export formats
- **export_directory** (`str`): Default export directory

## Result Classes

### VisualizationResult

Result object for visualization operations.

```python
@dataclass
class VisualizationResult:
    plot_object: Any
    plot_data: Dict[str, pd.Series]
    metrics_summary: Dict[str, float]
    export_paths: Dict[str, str]
    generation_time: float
    success: bool
    error_message: Optional[str] = None
```

#### Attributes

- **plot_object** (`Any`): The generated plot object
- **plot_data** (`Dict[str, pd.Series]`): Underlying plot data
- **metrics_summary** (`Dict[str, float]`): Performance metrics
- **export_paths** (`Dict[str, str]`): Paths to exported files
- **generation_time** (`float`): Time taken to generate visualization
- **success** (`bool`): Whether operation was successful
- **error_message** (`str`, optional): Error message if failed

### PortfolioCreationResult

Result object for portfolio creation operations.

```python
@dataclass
class PortfolioCreationResult:
    portfolio: Optional[vbt.Portfolio]
    aligned_signals: Optional[AlignedSignals]
    position_sizes: Optional[pd.Series]
    creation_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

#### Attributes

- **portfolio** (`vbt.Portfolio`, optional): Created portfolio object
- **aligned_signals** (`AlignedSignals`, optional): Aligned signal data
- **position_sizes** (`pd.Series`, optional): Calculated position sizes
- **creation_time** (`float`): Time taken to create portfolio
- **success** (`bool`): Whether operation was successful
- **error_message** (`str`, optional): Error message if failed
- **metadata** (`Dict[str, Any]`, optional): Additional metadata

### AlignedSignals

Data class for aligned signal information.

```python
@dataclass
class AlignedSignals:
    entry_signals: pd.Series
    exit_signals: pd.Series
    full_timeline: pd.DatetimeIndex
    test_period_start: int
    test_period_end: int
    prediction_count: int
    alignment_metadata: Dict[str, Any]
```

#### Attributes

- **entry_signals** (`pd.Series`): Boolean series for entry signals
- **exit_signals** (`pd.Series`): Boolean series for exit signals
- **full_timeline** (`pd.DatetimeIndex`): Complete timeline index
- **test_period_start** (`int`): Start index of test period
- **test_period_end** (`int`): End index of test period
- **prediction_count** (`int`): Number of predictions
- **alignment_metadata** (`Dict[str, Any]`): Additional alignment information

## Export Classes

### PlotExportEngine

Handles plot export in multiple formats.

```python
class PlotExportEngine:
    def __init__(self)
```

#### Methods

##### export_plot

```python
def export_plot(
    self,
    plot_object: Any,
    filename: str,
    formats: List[str] = ['png', 'html']
) -> Dict[str, str]
```

Export plot in specified formats.

**Parameters:**
- **plot_object** (`Any`): VectorBT plot object
- **filename** (`str`): Base filename for export
- **formats** (`List[str]`): List of export formats

**Returns:**
- `Dict[str, str]`: Dictionary mapping format to file path

##### export_plot_data

```python
def export_plot_data(
    self,
    portfolio: vbt.Portfolio,
    filename: str
) -> str
```

Export underlying plot data as CSV.

**Parameters:**
- **portfolio** (`vbt.Portfolio`): VectorBT portfolio object
- **filename** (`str`): Output filename

**Returns:**
- `str`: Path to exported CSV file

## Error Classes

### BacktestingError

Exception raised for backtesting-related errors.

```python
class BacktestingError(Exception):
    pass
```

### DataValidationError

Exception raised for data validation errors.

```python
class DataValidationError(Exception):
    pass
```

## Usage Examples

### Basic Usage

```python
from stock_predictor.visualization import (
    VectorBTVisualizationEngine,
    PortfolioConfig,
    PlotConfig
)

# Configure portfolio
portfolio_config = PortfolioConfig(
    init_cash=100000.0,
    fees=0.0025,
    size_strategy='fixed_amount',
    size_value=10000.0
)

# Configure plots
plot_config = PlotConfig(
    width=1200,
    height=600,
    show_trades=True
)

# Initialize engine
viz_engine = VectorBTVisualizationEngine(
    portfolio_config=portfolio_config,
    plot_config=plot_config
)

# Create portfolio from predictions
portfolio = viz_engine.create_portfolio_from_predictions(
    predictions, price_data, test_start_idx
)

# Generate visualization
result = viz_engine.generate_portfolio_plot(portfolio)

if result.success:
    viz_engine.show_plot(result)
```

### Advanced Configuration

```python
from stock_predictor.visualization import (
    VectorBTVisualizationEngine,
    EnhancedPortfolioEngine,
    PortfolioConfig,
    OptimizationConfig
)

# Advanced portfolio configuration
portfolio_config = PortfolioConfig(
    init_cash=250000.0,
    size_strategy='volatility_target',
    size_value=0.15,
    stop_loss=0.12,
    take_profit=0.30,
    upon_opposite_entry='close'
)

# Performance optimization
optimization_config = OptimizationConfig(
    enable_data_sampling=True,
    max_data_points=5000,
    sampling_strategy='adaptive'
)

# Initialize with advanced configuration
viz_engine = VectorBTVisualizationEngine(
    portfolio_config=portfolio_config,
    optimization_config=optimization_config,
    enable_performance_optimization=True
)
```

### Error Handling

```python
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
    
except BacktestingError as e:
    print(f"Backtesting error: {e}")
except DataValidationError as e:
    print(f"Data validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

### Memory Management

- Use `enable_performance_optimization=True` for large datasets
- Configure `OptimizationConfig` with appropriate memory limits
- Clean up plot objects after use in batch processing

### Data Sampling

- Enable data sampling for datasets > 10,000 points
- Use 'adaptive' sampling strategy for best results
- Monitor memory usage with large datasets

### Caching

- Enable caching for repeated visualizations
- Configure cache directory for persistent storage
- Clear cache periodically to manage disk space

## Version Compatibility

This API is compatible with:
- VectorBT >= 1.0.0
- Plotly >= 5.0.0
- Pandas >= 1.3.0
- NumPy >= 1.20.0

## See Also

- [User Guide](vectorbt_visualization_user_guide.md)
- [Examples](../examples/vectorbt_visualization_examples.py)
- [Performance Benchmarks](../tests/test_vectorbt_visualization_performance.py)