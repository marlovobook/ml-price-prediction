# Design Document: VectorBT Visualization Enhancement

## Overview

The VectorBT Visualization Enhancement extends the existing Stock Direction Predictor system with comprehensive, interactive portfolio visualization capabilities. This enhancement leverages VectorBT's built-in plotting functionality to provide professional-grade visual analytics for trading strategy evaluation. The system will properly align prediction signals with historical data, create realistic portfolio simulations, and generate rich interactive visualizations that enable traders and analysts to make data-driven decisions.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VectorBT Visualization Layer                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Signal        │  │   Portfolio     │  │  Visualization  │ │
│  │   Alignment     │  │   Creation      │  │   Generation    │ │
│  │   Engine        │  │   Engine        │  │   Engine        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Existing System Components                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   ML Models     │  │   Backtesting   │  │   Performance   │ │
│  │   & Predictions │  │   Engine        │  │   Evaluator     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Component Integration

The visualization enhancement integrates with existing components:
- **ML Models**: Receives prediction arrays from trained models
- **Backtesting Engine**: Extends VectorBT engine with visualization capabilities
- **Performance Evaluator**: Provides metrics for plot annotations
- **Data Collection**: Uses historical price data for complete timeline alignment

## Components and Interfaces

### 1. Signal Alignment Engine

**Purpose**: Properly align ML predictions with complete historical data timeline

**Interface**:
```python
class SignalAlignmentEngine:
    def align_predictions_to_timeline(
        self, 
        predictions: np.ndarray, 
        full_data: pd.DataFrame,
        test_start_idx: int
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Align predictions to full data timeline.
        
        Args:
            predictions: ML model predictions array
            full_data: Complete historical dataset
            test_start_idx: Index where test period begins
            
        Returns:
            Tuple of (entry_signals, exit_signals) aligned to full timeline
        """
        pass
    
    def convert_predictions_to_signals(
        self, 
        predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert prediction values to entry/exit signals.
        
        Args:
            predictions: Array of prediction values (0, 1, 2)
            
        Returns:
            Tuple of (entry_signals, exit_signals) boolean arrays
        """
        pass
```

**Key Methods**:
- `create_full_signal_arrays()`: Initialize full-sized boolean arrays
- `populate_test_period_signals()`: Fill test period with actual predictions
- `validate_signal_alignment()`: Ensure proper array dimensions and alignment

### 2. Enhanced Portfolio Creation Engine

**Purpose**: Create VectorBT portfolios with realistic trading parameters

**Interface**:
```python
class EnhancedPortfolioEngine:
    def create_vectorbt_portfolio(
        self,
        close_prices: pd.Series,
        entry_signals: pd.Series,
        exit_signals: pd.Series,
        config: PortfolioConfig
    ) -> vbt.Portfolio:
        """
        Create VectorBT portfolio with enhanced configuration.
        
        Args:
            close_prices: Historical closing prices
            entry_signals: Boolean series for entry points
            exit_signals: Boolean series for exit points
            config: Portfolio configuration parameters
            
        Returns:
            Configured VectorBT Portfolio object
        """
        pass
    
    def calculate_position_sizes(
        self,
        prices: pd.Series,
        sizing_method: str,
        capital: float
    ) -> pd.Series:
        """
        Calculate position sizes based on strategy.
        
        Args:
            prices: Price series for sizing calculation
            sizing_method: 'fixed_amount', 'fixed_shares', 'percent_equity'
            capital: Available capital
            
        Returns:
            Series of position sizes
        """
        pass
```

**Configuration Class**:
```python
@dataclass
class PortfolioConfig:
    init_cash: float = 10000
    fees: float = 0.0025
    slippage: float = 0.0025
    size_strategy: str = 'fixed_amount'
    size_value: float = 40
    stop_loss: Optional[float] = 0.1
    take_profit: Optional[float] = None
    upon_opposite_entry: str = 'ignore'
    freq: str = 'D'
```

### 3. Visualization Generation Engine

**Purpose**: Generate comprehensive interactive visualizations

**Interface**:
```python
class VectorBTVisualizationEngine:
    def generate_portfolio_plot(
        self, 
        portfolio: vbt.Portfolio,
        config: PlotConfig = None
    ) -> Any:
        """
        Generate interactive portfolio performance plot.
        
        Args:
            portfolio: VectorBT portfolio object
            config: Plot configuration options
            
        Returns:
            Interactive plot object
        """
        pass
    
    def generate_drawdown_plot(
        self, 
        portfolio: vbt.Portfolio
    ) -> Any:
        """
        Generate drawdown analysis plot.
        
        Args:
            portfolio: VectorBT portfolio object
            
        Returns:
            Drawdown plot object
        """
        pass
    
    def generate_trade_analysis_plot(
        self, 
        portfolio: vbt.Portfolio
    ) -> Any:
        """
        Generate trade performance analysis plot.
        
        Args:
            portfolio: VectorBT portfolio object
            
        Returns:
            Trade analysis plot object
        """
        pass
    
    def generate_comparison_plot(
        self, 
        portfolios: Dict[str, vbt.Portfolio]
    ) -> Any:
        """
        Generate multi-strategy comparison plot.
        
        Args:
            portfolios: Dictionary of named portfolio objects
            
        Returns:
            Comparison plot object
        """
        pass
```

### 4. Plot Export Engine

**Purpose**: Export visualizations in multiple formats

**Interface**:
```python
class PlotExportEngine:
    def export_plot(
        self,
        plot_object: Any,
        filename: str,
        formats: List[str] = ['png', 'html']
    ) -> Dict[str, str]:
        """
        Export plot in specified formats.
        
        Args:
            plot_object: VectorBT plot object
            filename: Base filename for export
            formats: List of export formats
            
        Returns:
            Dictionary mapping format to file path
        """
        pass
    
    def export_plot_data(
        self,
        portfolio: vbt.Portfolio,
        filename: str
    ) -> str:
        """
        Export underlying plot data as CSV.
        
        Args:
            portfolio: VectorBT portfolio object
            filename: Output filename
            
        Returns:
            Path to exported CSV file
        """
        pass
```

## Data Models

### Signal Alignment Data Model

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

### Portfolio Configuration Data Model

```python
@dataclass
class PortfolioConfig:
    # Capital and sizing
    init_cash: float = 10000
    size_strategy: str = 'fixed_amount'  # 'fixed_amount', 'fixed_shares', 'percent_equity'
    size_value: float = 40
    
    # Trading costs
    fees: float = 0.0025  # 0.25%
    slippage: float = 0.0025  # 0.25%
    
    # Risk management
    stop_loss: Optional[float] = 0.1  # 10%
    take_profit: Optional[float] = None
    
    # Execution rules
    upon_opposite_entry: str = 'ignore'  # 'ignore', 'close', 'reverse'
    freq: str = 'D'  # Daily frequency
    
    # Advanced parameters
    accumulate: bool = False
    conflict_mode: str = 'ignore'
```

### Visualization Configuration Data Model

```python
@dataclass
class PlotConfig:
    # Plot dimensions
    width: int = 1200
    height: int = 600
    
    # Display options
    show_trades: bool = True
    show_positions: bool = True
    show_cash: bool = False
    
    # Styling
    template: str = 'plotly_white'
    color_scheme: str = 'default'
    
    # Annotations
    show_metrics: bool = True
    metric_position: str = 'top_right'
    
    # Export options
    export_formats: List[str] = field(default_factory=lambda: ['png', 'html'])
    export_directory: str = 'visualizations'
```

### Visualization Result Data Model

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

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Signal Array Alignment Consistency
*For any* prediction array and historical dataset, when aligning signals to the full timeline, the resulting entry_signals and exit_signals arrays should have the same length as the historical dataset and contain prediction values only in the test period indices.
**Validates: Requirements 1.1, 1.2, 1.3**

### Property 2: Signal Conversion Accuracy
*For any* prediction array with values [0, 1, 2], converting to entry/exit signals should map value 2 to entry_signals=True, value 0 to exit_signals=True, and value 1 to both signals=False.
**Validates: Requirements 1.4, 1.5**

### Property 3: Portfolio Configuration Consistency
*For any* valid PortfolioConfig, creating a VectorBT portfolio should result in a portfolio object with parameters matching the configuration values (init_cash, fees, slippage, etc.).
**Validates: Requirements 2.1, 2.6, 2.7, 2.8**

### Property 4: Position Sizing Calculation Accuracy
*For any* price series and sizing configuration, calculated position sizes should respect the specified strategy constraints and not exceed available capital limits.
**Validates: Requirements 2.2**

### Property 5: Plot Generation Completeness
*For any* valid VectorBT portfolio, generating portfolio plots should produce a plot object that can be successfully rendered without errors.
**Validates: Requirements 3.1, 3.2, 3.3**

### Property 6: Trade Visualization Accuracy
*For any* portfolio with trades, the visualization should display entry points and exit points that correspond exactly to the portfolio's trade log entries.
**Validates: Requirements 3.4, 3.5**

### Property 7: Drawdown Visualization Consistency
*For any* portfolio with drawdowns, the drawdown plot should accurately reflect the portfolio's drawdown series and highlight the maximum drawdown period.
**Validates: Requirements 4.1, 4.2, 4.3**

### Property 8: Export Format Completeness
*For any* plot object and list of export formats, the export process should generate files in all requested formats and return valid file paths.
**Validates: Requirements 7.1, 7.2**

### Property 9: Configuration Parameter Validation
*For any* portfolio configuration with invalid parameters (negative fees, invalid sizing strategy), the system should reject the configuration and provide clear error messages.
**Validates: Requirements 6.6**

### Property 10: Performance Scalability
*For any* dataset size up to 5 years of daily data, visualization generation should complete within 30 seconds and not exceed reasonable memory limits.
**Validates: Requirements 9.1, 9.3**

## Error Handling

### Signal Alignment Errors
- **Dimension Mismatch**: When prediction array length doesn't match test period
- **Index Alignment**: When test period indices are out of bounds
- **Data Type Errors**: When predictions contain invalid values

**Handling Strategy**: Validate inputs, provide clear error messages, attempt automatic correction where possible

### Portfolio Creation Errors
- **Invalid Configuration**: When portfolio parameters are out of valid ranges
- **Insufficient Data**: When price series is too short for meaningful analysis
- **VectorBT API Errors**: When VectorBT library encounters internal errors

**Handling Strategy**: Parameter validation, graceful degradation to simple backtesting, comprehensive logging

### Visualization Generation Errors
- **Rendering Failures**: When plot generation fails due to data issues
- **Memory Constraints**: When datasets are too large for visualization
- **Display Environment**: When running in headless or incompatible environments

**Handling Strategy**: Fallback to text-based output, data sampling for large datasets, environment detection

### Export Errors
- **File System Issues**: When export directory is not writable
- **Format Conversion**: When specific export formats fail
- **Storage Limitations**: When exported files exceed size limits

**Handling Strategy**: Directory validation, partial export success, compression options

## Testing Strategy

### Unit Testing
- Test signal alignment with various prediction array sizes
- Test portfolio configuration validation
- Test plot generation with minimal datasets
- Test export functionality with mock plot objects

### Property-Based Testing
- Generate random prediction arrays and verify alignment properties
- Test portfolio creation with random valid configurations
- Verify plot generation with various portfolio scenarios
- Test export with different format combinations

### Integration Testing
- End-to-end workflow from predictions to visualizations
- Integration with existing backtesting engine
- Dashboard integration testing
- Performance testing with realistic datasets

### Visual Testing
- Screenshot comparison for plot consistency
- Interactive element functionality testing
- Cross-browser compatibility for HTML exports
- Mobile responsiveness for web-based visualizations

The testing approach ensures both functional correctness and visual quality, with property-based tests providing comprehensive coverage of edge cases and integration tests validating real-world usage scenarios.