"""
Example usage of the YahooFinanceDataService.
Demonstrates how to collect and validate stock data.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_predictor.data import YahooFinanceDataService
from stock_predictor.utils.logging_config import setup_logging


def main():
    """Demonstrate data collection functionality."""
    # Set up logging
    setup_logging()
    
    # Initialize the data service
    service = YahooFinanceDataService(cache_dir="example_cache")
    
    # Define parameters
    symbols = ['AAPL', 'MSFT', 'NVDA']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"Fetching data for {symbols} from {start_date} to {end_date}")
    
    try:
        # Fetch the data
        data = service.fetch_stock_data(symbols, start_date, end_date)
        
        # Display results
        for symbol, df in data.items():
            print(f"\n{symbol} Data:")
            print(f"  Shape: {df.shape}")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Sample data:")
            print(df.head(3).to_string())
            
            # Validate the data
            is_valid = service.validate_data_completeness(df)
            print(f"  Data validation: {'PASSED' if is_valid else 'FAILED'}")
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return 1
    
    print("\nData collection example completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())