# Stock Direction Predictor Dashboard 📈

An interactive Streamlit web application for exploring machine learning model performance in stock direction prediction.

## Features

### 🎯 Core Functionality
- **Model Performance Comparison**: Compare XGBoost, Random Forest, SVM, and Neural Network models
- **Candlestick Pattern Analysis**: Analyze 3, 5, 7, and 14-day pattern strategies
- **Interactive Backtesting**: Visualize portfolio performance over time
- **Technical Indicator Analysis**: Real-time technical indicators with buy/sell signals
- **Performance Metrics Dashboard**: ROI, Sharpe ratio, maximum drawdown, and more
- **Statistical Analysis**: Statistical significance testing between model configurations

### 📊 Visualization Components
- **Performance Heatmaps**: Model vs pattern length performance comparison
- **Risk-Return Scatter Plots**: Risk-adjusted return analysis
- **Portfolio Value Charts**: Time series of portfolio performance
- **Technical Analysis Charts**: Candlestick charts with indicators
- **Radar Charts**: Multi-dimensional model comparison

### ⚡ Real-time Features
- **Live Data Updates**: Real-time stock data fetching
- **Dynamic Predictions**: Latest model predictions with confidence scores
- **Auto-refresh**: Configurable refresh intervals
- **Signal Alerts**: Buy/sell/hold signal indicators

## Quick Start

### 1. Installation

```bash
# Install dashboard requirements
pip install -r streamlit_requirements.txt

# Or install individual packages
pip install streamlit plotly pandas numpy yfinance scikit-learn xgboost
```

### 2. Launch Dashboard

```bash
# Using the launch script (recommended)
python run_dashboard.py

# Or directly with Streamlit
streamlit run streamlit_dashboard.py
```

### 3. Access Dashboard

Open your web browser and navigate to: `http://localhost:8501`

## Usage Guide

### 📋 Getting Started

1. **Select Stocks**: Choose from AAPL, MSFT, NVDA, AMZN, META in the sidebar
2. **Choose Pattern Lengths**: Select candlestick pattern lengths (3, 5, 7, 14 days)
3. **Select Models**: Pick ML models to compare (XGBoost, Random Forest, SVM, Neural Network)
4. **Set Date Range**: Configure analysis period
5. **Run Analysis**: Click "🚀 Run Analysis" to start

### 🎛️ Dashboard Sections

#### Performance Overview
- Model recommendation scores
- Risk-return analysis
- Performance heatmaps
- Detailed metrics table

#### Backtesting Results
- Portfolio value over time
- Trade analysis
- Win rate comparison
- Profit factor analysis

#### Technical Analysis
- Live candlestick charts
- Technical indicators (RSI, MACD, EMA)
- Pattern signal generation
- Buy/sell signal overlays

#### Detailed Metrics
- Pattern length analysis
- Model type comparison
- Statistical significance tests
- Performance rankings

#### Model Insights
- Prediction confidence analysis
- Real-time predictions
- Signal strength indicators
- Statistical test results

### 🔧 Configuration

Edit `dashboard_config.yaml` to customize:

```yaml
dashboard:
  default_symbols: ["AAPL", "MSFT", "NVDA"]
  default_patterns: [3, 5, 7, 14]
  default_models: ["xgboost", "random_forest"]
  
realtime:
  enabled: true
  default_interval: "15 minutes"
  
analysis:
  initial_capital: 10000
  transaction_cost: 0.001
```

## Architecture

### 🏗️ Component Structure

```
streamlit_dashboard.py
├── StockPredictorDashboard (Main Class)
├── Sidebar Configuration
├── Analysis Execution
├── Visualization Components
└── Real-time Data Integration
```

### 🔄 Data Flow

1. **User Input** → Sidebar configuration
2. **Analysis Trigger** → Stock predictor orchestrator
3. **Results Processing** → Performance metrics calculation
4. **Visualization** → Interactive charts and tables
5. **Real-time Updates** → Live data fetching and predictions

### 🎨 UI Components

- **Sidebar**: Configuration and controls
- **Main Dashboard**: Tabbed interface with multiple views
- **Executive Summary**: Key metrics and recommendations
- **Interactive Charts**: Plotly-based visualizations
- **Data Tables**: Sortable and filterable results

## Technical Details

### 📦 Dependencies

- **Streamlit**: Web application framework
- **Plotly**: Interactive charting library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **yfinance**: Real-time stock data
- **scikit-learn**: Machine learning utilities
- **XGBoost**: Gradient boosting framework

### 🔌 Integration Points

- **Stock Predictor Orchestrator**: Main analysis engine
- **Yahoo Finance API**: Real-time data source
- **Feature Engineering Module**: Technical indicator calculations
- **Backtesting Engine**: Portfolio simulation
- **Comparison Framework**: Statistical analysis

### 💾 Caching Strategy

- **Session State**: User selections and analysis results
- **Data Caching**: Stock data and technical indicators
- **Result Caching**: Analysis results for quick access

## Performance Optimization

### ⚡ Speed Improvements

- **Lazy Loading**: Components loaded on demand
- **Data Caching**: Avoid redundant API calls
- **Incremental Updates**: Update only changed components
- **Async Processing**: Non-blocking operations where possible

### 📊 Memory Management

- **Result Limiting**: Limit cached results to prevent memory issues
- **Data Cleanup**: Regular cleanup of old cached data
- **Efficient Data Structures**: Use appropriate data types

## Troubleshooting

### 🐛 Common Issues

#### Dashboard Won't Start
```bash
# Check Python version (3.8+ required)
python --version

# Install missing dependencies
pip install -r streamlit_requirements.txt

# Check for port conflicts
netstat -an | grep 8501
```

#### Analysis Fails
- Ensure internet connection for data fetching
- Check stock symbols are valid
- Verify date ranges are reasonable
- Review error messages in terminal

#### Slow Performance
- Reduce number of symbols/models/patterns
- Clear cache using sidebar button
- Check system resources
- Consider running on more powerful hardware

### 📝 Debug Mode

Enable debug logging by setting environment variable:
```bash
export STREAMLIT_LOGGER_LEVEL=debug
streamlit run streamlit_dashboard.py
```

## Customization

### 🎨 Styling

Modify CSS in the dashboard file:
```python
st.markdown("""
<style>
    .main-header {
        color: #your-color;
    }
</style>
""", unsafe_allow_html=True)
```

### 📊 Adding New Charts

1. Create new visualization function
2. Add to appropriate tab section
3. Update configuration if needed

### 🔧 Adding New Features

1. Extend `StockPredictorDashboard` class
2. Add new sidebar controls
3. Implement feature logic
4. Update documentation

## Support

### 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [Stock Predictor Documentation](./README.md)

### 🆘 Getting Help

1. Check this README for common solutions
2. Review error messages in terminal
3. Check system requirements
4. Verify data connectivity

## License

This dashboard is part of the Stock Direction Predictor project and follows the same licensing terms.