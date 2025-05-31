# Stock Forecasting Application

This project is a stock forecasting application that utilizes machine learning models to predict stock prices. It is designed to be extensible, allowing for the addition of multiple models for comparison in the future. The application also includes a Streamlit interface for visualizing prediction results.

## Project Structure

```
stock-forecasting-app
├── src
│   ├── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── model_factory.py
│   │   └── xgboost_model.py
│   ├── data
│   │   ├── __init__.py
│   │   └── data_processor.py
│   ├── backtesting
│   │   ├── __init__.py
│   │   └── portfolio.py
│   └── utils
│       ├── __init__.py
│       └── technical_indicators.py
├── streamlit_app
│   ├── __init__.py
│   ├── app.py
│   └── components
│       ├── __init__.py
│       ├── charts.py
│       └── metrics.py
├── scripts
│   └── main.py
├── config
│   └── config.yaml
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

To set up the project, create a virtual environment and install the required dependencies. You can do this by running the following commands:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

## Usage

1. **Data Processing**: The `DataProcessor` class in `src/data/data_processor.py` handles data fetching and preprocessing. You can customize the data source and parameters in `config/config.yaml`.

2. **Model Training**: The main script `scripts/main.py` uses the `ModelFactory` to create and train the specified model. You can easily add new models by implementing them in the `src/models` directory.

3. **Backtesting**: The `Portfolio` class in `src/backtesting/portfolio.py` allows you to backtest trading strategies and evaluate performance metrics.

4. **Visualization**: The Streamlit application can be launched by running `streamlit run streamlit_app/app.py`. This will start a local server where you can visualize the stock price predictions and model performance metrics.

## Contributing

Contributions are welcome! If you would like to add new features or models, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.