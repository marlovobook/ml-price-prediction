# Comprehensive Comparison Framework

The Comprehensive Comparison Framework provides advanced statistical analysis and visualization capabilities for evaluating model-pattern combinations in the Stock Direction Predictor system.

## Features

### 1. Statistical Significance Testing
- **Friedman Chi-Square Test**: Non-parametric test for comparing multiple groups
- **Mann-Whitney U Test**: Pairwise comparisons between configurations
- **Confidence Level Configuration**: Customizable statistical confidence levels (default: 95%)

### 2. Performance Comparison
- **Multi-Criteria Evaluation**: Combines financial and prediction metrics
- **Recommendation Scoring**: Weighted scoring system for configuration ranking
- **Pattern Length Analysis**: Comparative analysis across different candlestick pattern lengths
- **Model Type Analysis**: Performance comparison across different ML algorithms

### 3. Visualization Components
- **Performance Heatmaps**: Visual comparison of recommendation scores
- **Risk-Return Scatter Plots**: Risk vs. return analysis with bubble sizing
- **Ranking Charts**: Bar charts showing configuration performance rankings
- **Automated Chart Generation**: PNG export with customizable styling

### 4. Automated Report Generation
- **Executive Summary**: High-level insights and best configuration identification
- **Detailed Rankings**: Complete performance breakdown for all configurations
- **Statistical Analysis**: Interpretation of significance test results
- **Actionable Recommendations**: Data-driven suggestions for optimal configuration selection

## Usage

### Basic Usage

```python
from stock_predictor.evaluation.comparison_framework import ComparisonFramework

# Initialize framework
framework = ComparisonFramework(confidence_level=0.95)

# Prepare results data (from model training and backtesting)
results = [
    {
        'model_type': 'XGBoost',
        'pattern_length': 3,
        'financial_metrics': {
            'total_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': -0.08,
            'win_rate': 0.65
        },
        'prediction_metrics': {
            'accuracy': 0.72,
            'mse': 0.05,
            'rmse': 0.22
        }
    },
    # ... more results
]

# Run comprehensive comparison
comparison_report = framework.compare_all_combinations(results)

# Access results
best_config = comparison_report['executive_summary']['best_configuration']
recommendations = comparison_report['recommendations']
statistical_tests = comparison_report['statistical_tests']
```

### Advanced Usage with Visualization

```python
# Generate visualization data
comparison_results = []  # Convert to ComparisonResult objects
viz_data = framework.generate_visualization_data(comparison_results)

# Create performance charts
chart_paths = framework.create_performance_charts(
    comparison_results, 
    save_path="./charts"
)

# Select best configuration with custom criteria
custom_weights = {
    'total_return': 0.40,
    'sharpe_ratio': 0.30,
    'max_drawdown': 0.20,
    'accuracy': 0.10
}

best_config = framework.select_best_configuration(
    comparison_results,
    criteria_weights=custom_weights
)
```

### Integration with Main Application

```python
from stock_predictor.main import StockPredictorOrchestrator

# Initialize orchestrator
orchestrator = StockPredictorOrchestrator()
orchestrator.initialize()

# Run comprehensive comparison analysis
results = orchestrator.run_comprehensive_comparison(
    symbols=['AAPL', 'MSFT', 'NVDA']
)

# Access comprehensive results
comparison_report = results['comparison_report']
best_configuration = results['best_configuration']
chart_paths = results['chart_paths']
recommendations = results['recommendations']
```

## Data Models

### ComparisonResult
```python
@dataclass
class ComparisonResult:
    model_type: str
    pattern_length: int
    performance_metrics: Dict[str, float]
    statistical_significance: Dict[str, Any]
    rank: int
    recommendation_score: float
```

### StatisticalTest
```python
@dataclass
class StatisticalTest:
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    interpretation: str
```

## Metrics and Scoring

### Performance Metrics
- **Financial Metrics**: Total return, Sharpe ratio, maximum drawdown, win rate, profit factor
- **Prediction Metrics**: Accuracy, MSE, MAE, RMSE
- **Composite Score**: Weighted combination of all metrics

### Recommendation Scoring
The recommendation score (0-100) is calculated using weighted criteria:
- **Sharpe Ratio** (35%): Risk-adjusted returns
- **Total Return** (25%): Absolute performance
- **Maximum Drawdown** (25%): Risk management
- **Accuracy** (15%): Prediction quality

### Statistical Significance
- **Friedman Test**: Tests for significant differences across all configurations
- **Pairwise Tests**: Mann-Whitney U tests for individual comparisons
- **Confidence Levels**: Configurable significance thresholds

## Output Formats

### Comparison Report Structure
```json
{
  "executive_summary": {
    "total_configurations": 12,
    "best_configuration": {
      "model_type": "XGBoost",
      "pattern_length": 3,
      "recommendation_score": 78.5,
      "key_metrics": {...}
    }
  },
  "detailed_results": [...],
  "pattern_length_analysis": {...},
  "model_type_analysis": {...},
  "statistical_tests": {...},
  "recommendations": [...]
}
```

### Visualization Data
- **Performance Comparison**: Tabular data for all configurations
- **Heatmap Data**: Matrix format for pattern length vs. model type
- **Risk-Return Scatter**: Coordinates for risk-return visualization
- **Ranking Data**: Sorted performance data

## Command Line Interface

```bash
# Run comprehensive comparison analysis
python -m stock_predictor.main --mode comprehensive --symbols AAPL MSFT NVDA

# Generate charts and statistical analysis
python -m stock_predictor.main --mode comprehensive --output-dir ./results
```

## Requirements

### Core Dependencies
- `pandas >= 1.5.0`
- `numpy >= 1.21.0`
- `scipy >= 1.9.0`

### Visualization Dependencies
- `matplotlib >= 3.5.0`
- `seaborn >= 0.11.0`

### Statistical Testing
- `scipy.stats`: Friedman test, Mann-Whitney U test
- Built-in confidence interval calculations

## Testing

The framework includes comprehensive unit tests covering:
- Statistical test integration
- Visualization data generation
- Performance metric calculations
- Configuration selection algorithms
- Error handling and edge cases

Run tests with:
```bash
python -m pytest tests/test_comparison_framework.py -v
```

## Performance Considerations

- **Memory Usage**: Efficient handling of large result datasets
- **Computation Time**: Optimized statistical calculations
- **Visualization**: Lazy loading of chart generation
- **Scalability**: Supports analysis of 100+ model-pattern combinations

## Best Practices

1. **Data Quality**: Ensure complete financial and prediction metrics
2. **Statistical Power**: Use sufficient data for meaningful statistical tests
3. **Interpretation**: Consider both statistical significance and practical significance
4. **Visualization**: Use charts to communicate insights effectively
5. **Documentation**: Save comprehensive reports for reproducibility

## Troubleshooting

### Common Issues
- **Insufficient Data**: Ensure at least 2 configurations for comparison
- **Missing Metrics**: Verify all required performance metrics are present
- **Chart Generation**: Check write permissions for output directories
- **Statistical Tests**: Ensure adequate sample sizes for reliable results

### Error Handling
The framework includes robust error handling for:
- Invalid input data formats
- Missing performance metrics
- Statistical calculation failures
- File I/O operations
- Visualization rendering issues