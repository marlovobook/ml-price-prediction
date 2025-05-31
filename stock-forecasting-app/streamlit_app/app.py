import sys
import os
# Add the parent directory to Python path to access src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_factory import ModelFactory
from src.data.data_processor import DataProcessor
from src.backtesting.portfolio import Portfolio
import streamlit as st
import pandas as pd

def main():
    st.title("Stock Forecasting App")

    # Load configuration settings
    model_type = st.selectbox("Select Model", ["xgboost"])  # Extendable for more models
    
    # Create data processor with required parameters
    data_processor = DataProcessor(
        symbols=['NVDA'], 
        start_date='2021-01-01', 
        end_date=pd.to_datetime('today')
    )
    
    # Fetch and preprocess data
    df = data_processor.fetch_data()
    
    # Create model instance
    model_factory = ModelFactory()
    model = model_factory.create_model(model_type)
    
    # Train model
    if st.button("Train Model"):
        model.train(df)
        st.success("Model trained successfully!")

    # Make predictions
    if st.button("Make Predictions"):
        predictions = model.predict(df)
        st.write("Predictions:")
        st.line_chart(predictions)

    # Backtesting
    if st.button("Backtest"):
        portfolio = Portfolio(df, model)
        results = portfolio.backtest()
        st.write("Backtesting Results:")
        st.write(results)

if __name__ == "__main__":
    main()