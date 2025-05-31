# File: /stock-forecasting-app/scripts/main.py

import pandas as pd
from src.models.model_factory import ModelFactory
from src.data.data_processor import DataProcessor
from src.backtesting.portfolio import Portfolio

def main():
    # Load configuration settings
    config = load_config()

    # Fetch and preprocess data
    data_processor = DataProcessor(config['data_source'])
    df = data_processor.fetch_data()
    df = data_processor.preprocess_data(df)

    # Create model using the factory
    model_factory = ModelFactory()
    model = model_factory.create_model('XGBoost')  # Example model type

    # Train the model
    X, y = data_processor.create_features_and_target(df)
    model.train(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Evaluate the model
    evaluation_metrics = model.evaluate(predictions, y)
    print("Model Evaluation Metrics:", evaluation_metrics)

    # Backtesting
    portfolio = Portfolio(df['Close'], predictions)
    portfolio_metrics = portfolio.evaluate()
    print("Portfolio Metrics:", portfolio_metrics)

def load_config():
    import yaml
    with open('config/config.yaml', 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    main()