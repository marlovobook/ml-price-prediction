# Stock Direction Predictor Orchestrator

The main application orchestrator coordinates all components of the Stock Direction Predictor system and provides configuration-driven workflow execution with multiple analysis modes.

## Features

### 1. Component Coordination
- **Data Collection**: Automated stock data retrieval from Yahoo Finance
- **Feature Engineering**: Technical indicators and chart pattern detection
- **Model Training**: Multiple ML algorithms (XGBoost, Random Forest, SVM, Neural Network)
- **Backtesting**: Trading simulation with portfolio tracking
- **Performance Evaluation**: Comprehensive metrics and ranking

### 2. Analysis Modes

#### Full Analysis (`--mode full`)
Runs complete analysis pipeline for specified symbols, pattern lengths, and model types.

```bash
python -m stock_predictor.main --mode full --symbols AAPL MSFT --patterns 3 5 7 --models xgboost random_forest
```

#### Single Symbol Analysis (`--mode single`)
Analyzes only the first specified symbol (useful for testing).

```bash
python -m stock_predictor.main --mode single --symbols AAPL
```

#### Comparison Analysis (`--mode comparison`)
Runs full analysis plus generates detailed comparison insights across:
- Pattern lengths (3, 5, 7, 14 days)
- Model types (XGBoost, Random Forest, SVM, Neural Network)
- Stock symbols

```bash
python -m stock_predictor.main --mode comparison --symbols AAPL MSFT NVDA
```

#### Batch Analysis (`--mode batch`)
Processes multiple symbol groups and time periods using batch configuration.

```bash
python -m stock_predictor.main --mode batch --batch-config examples/batch_config_example.json
```

### 3. Configuration-Driven Workflow

#### Command Line Options
- `--config`: Path to YAML configuration file
- `--symbols`: Override stock symbols from config
- `--patterns`: Override pattern lengths from config  
- `--models`: Override model types from config
- `--output-dir`: Override results directory
- `--verbose`: Enable debug logging

#### Configuration File (config.yaml)
```yaml
data:
  stock_symbols: ["AAPL", "MSFT", "NVDA", "AMZN", "META"]
  start_date: "2020-01-01"
  end_date: "2024-01-01"
  
features:
  pattern_lengths: [3, 5, 7, 14]
  
models:
  model_types: ["xgboost", "random_forest", "svm", "neural_network"]
  
backtest:
  initial_capital: 100000.0
  transaction_cost: 0.001
  slippage: 0.0005
```

### 4. Batch Processing

#### Batch Configuration Format
```json
{
  "symbol_groups": [
    ["AAPL", "MSFT"],
    ["NVDA", "AMZN"], 
    ["META"]
  ],
  "time_periods": [
    {
      "start": "2022-01-01",
      "end": "2023-01-01"
    },
    {
      "start": "2023-01-01",
      "end": "2024-01-01"
    }
  ]
}
```

This configuration will run 6 separate analyses (3 symbol groups × 2 time periods).

### 5. Results Aggregation

#### Output Files
All results are saved to the configured results directory with timestamps:

- `analysis_results_YYYYMMDD_HHMMSS.json`: Full analysis results
- `batch_results_YYYYMMDD_HHMMSS.json`: Batch analysis results  
- `comparison_results_YYYYMMDD_HHMMSS.json`: Comparison analysis results

#### Results Structure
```json
{
  "analysis_timestamp": "2024-01-01T12:00:00",
  "configuration": {
    "symbols": ["AAPL", "MSFT"],
    "pattern_lengths": [3, 5, 7, 14],
    "model_types": ["xgboost", "random_forest"]
  },
  "symbol_results": {
    "AAPL": {
      "model_results": [...]
    }
  },
  "aggregated_results": [...],
  "performance_report": {
    "best_configuration": {
      "model_type": "xgboost",
      "pattern_length": 5,
      "symbol": "AAPL",
      "financial_metrics": {
        "total_return": 0.15,
        "sharpe_ratio": 1.2,
        "max_drawdown": -0.08
      }
    },
    "ranked_results": [...],
    "comparison_analysis": [...]
  }
}
```

### 6. Performance Comparison

The orchestrator automatically generates comprehensive comparisons:

#### Pattern Length Analysis
- Average performance across all models for each pattern length
- Best performing model for each pattern length
- Statistical significance of differences

#### Model Type Analysis  
- Average performance across all pattern lengths for each model
- Best performing configuration for each model
- Consistency metrics across different market conditions

#### Symbol Analysis
- Performance characteristics of each stock
- Best model-pattern combinations per symbol
- Risk-adjusted returns comparison

## Usage Examples

### Basic Usage
```bash
# Run full analysis with default configuration
python -m stock_predictor.main

# Run with specific symbols and verbose output
python -m stock_predictor.main --symbols AAPL MSFT --verbose

# Run comparison analysis
python -m stock_predictor.main --mode comparison --symbols AAPL MSFT NVDA
```

### Advanced Usage
```bash
# Custom configuration and output directory
python -m stock_predictor.main --config my_config.yaml --output-dir ./my_results

# Batch processing with custom time periods
python -m stock_predictor.main --mode batch --batch-config batch_config.json

# Test specific model types and pattern lengths
python -m stock_predictor.main --models xgboost random_forest --patterns 3 5
```

### Programmatic Usage
```python
from stock_predictor.main import StockPredictorOrchestrator

# Initialize orchestrator
orchestrator = StockPredictorOrchestrator("config.yaml")
orchestrator.initialize()

# Run full analysis
results = orchestrator.run_full_analysis(
    symbols=["AAPL", "MSFT"],
    pattern_lengths=[3, 5, 7],
    model_types=["xgboost", "random_forest"]
)

# Access best configuration
best_config = results['performance_report']['best_configuration']
print(f"Best: {best_config['model_type']} with {best_config['pattern_length']}-day patterns")
```

## Requirements Validation

This implementation satisfies all requirements from task 11:

✅ **Main application class that coordinates all components**
- `StockPredictorOrchestrator` class coordinates data collection, feature engineering, model training, backtesting, and evaluation

✅ **Configuration-driven workflow for different analysis scenarios**  
- YAML configuration file support with CLI overrides
- Multiple analysis modes (full, single, comparison, batch)

✅ **Command-line interface for running different analysis modes**
- Comprehensive CLI with argparse supporting all modes and options
- Help documentation and parameter validation

✅ **Batch processing for multiple stocks and time periods**
- Batch configuration support with JSON files
- Automated processing of symbol groups across different time periods

✅ **Results aggregation and comparison functionality**
- Comprehensive results aggregation across all configurations
- Detailed comparison analysis by pattern length, model type, and symbol
- Performance ranking and best configuration identification
- Structured JSON output with timestamps

The orchestrator provides a complete, production-ready interface for running the Stock Direction Predictor system with full flexibility and comprehensive result analysis.