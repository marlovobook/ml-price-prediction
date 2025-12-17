"""
Yahoo Finance Data Collection Service implementation.
Provides data collection functionality with error handling and retry mechanisms.
"""

import time
import logging
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import pickle
from pathlib import Path

from ..interfaces import IDataCollectionService
from ..utils.exceptions import DataCollectionError, DataValidationError


class YahooFinanceDataService(IDataCollectionService):
    """
    Yahoo Finance data collection service with retry mechanisms and data validation.
    """
    
    def __init__(self, cache_dir: str = "data_cache", max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the Yahoo Finance data service.
        
        Args:
            cache_dir: Directory for caching downloaded data
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        
        # Required columns for OHLC data
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        # Optional columns that may be present
        self.optional_columns = ['Dividends', 'Stock Splits']
    
    def fetch_stock_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical stock data for given symbols and date range.
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'MSFT'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary mapping symbol to DataFrame with OHLC data
            
        Raises:
            DataCollectionError: If data collection fails after all retries
        """
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            
            # Check cache first
            cached_data = self._load_from_cache(symbol, start_date, end_date)
            if cached_data is not None:
                self.logger.info(f"Using cached data for {symbol}")
                results[symbol] = cached_data
                continue
            
            # Fetch from Yahoo Finance with retry mechanism
            data = self._fetch_with_retry(symbol, start_date, end_date)
            
            if data is not None and not data.empty:
                # Validate and process the data
                if self.validate_data_completeness(data):
                    processed_data = self.handle_missing_values(data)
                    results[symbol] = processed_data
                    
                    # Cache the processed data
                    self._save_to_cache(symbol, start_date, end_date, processed_data)
                    self.logger.info(f"Successfully fetched and cached data for {symbol}")
                else:
                    self.logger.warning(f"Data validation failed for {symbol}")
                    raise DataValidationError(f"Invalid data structure for {symbol}")
            else:
                self.logger.error(f"No data retrieved for {symbol}")
                raise DataCollectionError(f"Failed to fetch data for {symbol}")
        
        return results
    
    def validate_data_completeness(self, data: pd.DataFrame) -> bool:
        """
        Validate that the data contains all required fields and is complete.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if data is None or data.empty:
            self.logger.warning("Data is None or empty")
            return False
        
        # Check required columns
        missing_columns = set(self.required_columns) - set(data.columns)
        if missing_columns:
            self.logger.warning(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for reasonable data ranges
        if len(data) == 0:
            self.logger.warning("Data has no rows")
            return False
        
        # Validate OHLC relationships (High >= max(Open, Close), Low <= min(Open, Close))
        invalid_ohlc = (
            (data['High'] < data[['Open', 'Close']].max(axis=1)) |
            (data['Low'] > data[['Open', 'Close']].min(axis=1))
        )
        
        if invalid_ohlc.any():
            self.logger.warning(f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships")
            return False
        
        # Check for negative values where they shouldn't exist
        if (data[['Open', 'High', 'Low', 'Close', 'Volume']] < 0).any().any():
            self.logger.warning("Found negative values in price or volume data")
            return False
        
        return True
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        data_copy = data.copy()
        
        # Log missing value statistics
        missing_stats = data_copy.isnull().sum()
        if missing_stats.sum() > 0:
            self.logger.info(f"Missing values found: {missing_stats.to_dict()}")
        
        # Forward fill missing values for price data (common approach for financial data)
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in price_columns:
            if col in data_copy.columns:
                data_copy[col] = data_copy[col].ffill()
        
        # Handle volume separately - use 0 for missing volume
        if 'Volume' in data_copy.columns:
            data_copy['Volume'] = data_copy['Volume'].fillna(0)
        
        # Ensure Adj Close exists (use Close if not available)
        if 'Adj Close' not in data_copy.columns and 'Close' in data_copy.columns:
            data_copy['Adj Close'] = data_copy['Close']
        
        # Drop any remaining rows with missing values
        initial_rows = len(data_copy)
        data_copy = data_copy.dropna()
        final_rows = len(data_copy)
        
        if initial_rows != final_rows:
            self.logger.info(f"Dropped {initial_rows - final_rows} rows with missing values")
        
        return data_copy
    
    def _fetch_with_retry(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch data with exponential backoff retry mechanism.
        
        Args:
            symbol: Stock symbol
            start_date: Start date string
            end_date: End date string
            
        Returns:
            DataFrame with stock data or None if all retries failed
        """
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    # Add Adj Close column if not present (use Close as fallback)
                    if 'Adj Close' not in data.columns:
                        data['Adj Close'] = data['Close']
                    return data
                else:
                    self.logger.warning(f"Empty data returned for {symbol} (attempt {attempt + 1})")
                    
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All retry attempts failed for {symbol}")
        
        return None
    
    def _get_cache_filename(self, symbol: str, start_date: str, end_date: str) -> Path:
        """Generate cache filename for the given parameters."""
        return self.cache_dir / f"{symbol}_{start_date}_{end_date}.pkl"
    
    def _load_from_cache(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available and not expired.
        
        Args:
            symbol: Stock symbol
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Cached DataFrame or None if not available/expired
        """
        cache_file = self._get_cache_filename(symbol, start_date, end_date)
        
        if not cache_file.exists():
            return None
        
        try:
            # Check if cache is recent (within 1 day for daily data)
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age > timedelta(days=1):
                self.logger.info(f"Cache expired for {symbol}, will fetch fresh data")
                return None
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                return data
                
        except Exception as e:
            self.logger.warning(f"Failed to load cache for {symbol}: {str(e)}")
            return None
    
    def _save_to_cache(self, symbol: str, start_date: str, end_date: str, data: pd.DataFrame) -> None:
        """
        Save data to cache.
        
        Args:
            symbol: Stock symbol
            start_date: Start date string
            end_date: End date string
            data: DataFrame to cache
        """
        cache_file = self._get_cache_filename(symbol, start_date, end_date)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.logger.debug(f"Cached data for {symbol}")
        except Exception as e:
            self.logger.warning(f"Failed to cache data for {symbol}: {str(e)}")