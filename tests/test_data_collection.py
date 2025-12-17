"""
Property-based tests for data collection service.
Tests the completeness and correctness of data collection functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.pandas import data_frames, column
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

from stock_predictor.data.yahoo_finance_service import YahooFinanceDataService
from stock_predictor.utils.exceptions import DataCollectionError, DataValidationError


class TestYahooFinanceDataService:
    """Test suite for YahooFinanceDataService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.service = YahooFinanceDataService(cache_dir=self.temp_dir, max_retries=2, retry_delay=0.1)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)


# Property-based test generators
@st.composite
def valid_ohlc_data(draw):
    """Generate valid OHLC data that satisfies market constraints."""
    # Generate base prices
    base_price = draw(st.floats(min_value=1.0, max_value=1000.0))
    
    # Generate open and close around base price
    open_price = draw(st.floats(min_value=base_price * 0.9, max_value=base_price * 1.1))
    close_price = draw(st.floats(min_value=base_price * 0.9, max_value=base_price * 1.1))
    
    # High must be >= max(open, close)
    min_high = max(open_price, close_price)
    high_price = draw(st.floats(min_value=min_high, max_value=min_high * 1.1))
    
    # Low must be <= min(open, close)
    max_low = min(open_price, close_price)
    low_price = draw(st.floats(min_value=max_low * 0.9, max_value=max_low))
    
    # Volume should be positive
    volume = draw(st.integers(min_value=0, max_value=1000000000))
    
    # Adjusted close typically close to close price
    adj_close = draw(st.floats(min_value=close_price * 0.95, max_value=close_price * 1.05))
    
    return {
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Close': close_price,
        'Volume': volume,
        'Adj Close': adj_close
    }


@st.composite
def valid_stock_dataframe(draw, min_rows=1, max_rows=100):
    """Generate a valid stock data DataFrame."""
    num_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    
    # Generate date index
    start_date = draw(st.dates(min_value=datetime(2020, 1, 1).date(), 
                              max_value=datetime(2023, 12, 31).date()))
    dates = pd.date_range(start=start_date, periods=num_rows, freq='D')
    
    # Generate OHLC data for each row
    rows = []
    for _ in range(num_rows):
        row_data = draw(valid_ohlc_data())
        rows.append(row_data)
    
    df = pd.DataFrame(rows, index=dates)
    return df


@st.composite
def valid_symbols_and_dates(draw):
    """Generate valid stock symbols and date ranges."""
    # Generate 1-5 symbols
    num_symbols = draw(st.integers(min_value=1, max_value=5))
    symbols = []
    for _ in range(num_symbols):
        symbol = draw(st.text(alphabet=st.characters(whitelist_categories=('Lu',)), 
                             min_size=1, max_size=5))
        symbols.append(symbol)
    
    # Generate valid date range
    start_date = draw(st.dates(min_value=datetime(2020, 1, 1).date(),
                              max_value=datetime(2023, 6, 1).date()))
    end_date = draw(st.dates(min_value=start_date,
                            max_value=datetime(2023, 12, 31).date()))
    
    return symbols, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


class TestDataCollectionProperties:
    """Property-based tests for data collection completeness."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.service = YahooFinanceDataService(cache_dir=self.temp_dir, max_retries=2, retry_delay=0.1)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(valid_stock_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_property_data_validation_completeness(self, stock_data):
        """
        **Feature: stock-direction-predictor, Property 1: Data Collection Completeness**
        **Validates: Requirements 1.1, 1.3, 1.4, 1.5**
        
        For any valid OHLC dataset, the data validation should correctly identify
        complete data and reject incomplete or invalid data.
        """
        # Test that valid data passes validation
        assert self.service.validate_data_completeness(stock_data) == True
        
        # Test that data with missing required columns fails validation
        incomplete_data = stock_data.drop(columns=['Volume'])
        assert self.service.validate_data_completeness(incomplete_data) == False
        
        # Test that empty data fails validation
        empty_data = stock_data.iloc[0:0]  # Empty DataFrame with same columns
        assert self.service.validate_data_completeness(empty_data) == False
    
    @given(valid_stock_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_property_missing_value_handling_preserves_structure(self, stock_data):
        """
        **Feature: stock-direction-predictor, Property 1: Data Collection Completeness**
        **Validates: Requirements 1.1, 1.3, 1.4, 1.5**
        
        For any dataset with missing values, the missing value handler should
        preserve the data structure and handle missing values appropriately.
        """
        # Introduce some missing values
        data_with_missing = stock_data.copy()
        if len(data_with_missing) > 1:
            # Add some NaN values to test handling
            data_with_missing.iloc[0, 0] = np.nan  # Missing Open price
            if len(data_with_missing) > 2:
                data_with_missing.iloc[1, 4] = np.nan  # Missing Volume
        
        # Handle missing values
        processed_data = self.service.handle_missing_values(data_with_missing)
        
        # Verify structure is preserved (columns should be the same)
        assert list(processed_data.columns) == list(stock_data.columns)
        
        # Verify no missing values remain in the result
        assert not processed_data.isnull().any().any()
        
        # Verify data types are preserved
        for col in stock_data.columns:
            if col in processed_data.columns:
                assert processed_data[col].dtype.kind in ['f', 'i']  # Numeric types
    
    @given(valid_symbols_and_dates())
    @settings(max_examples=50, deadline=None)
    def test_property_fetch_with_mock_returns_complete_structure(self, symbols_and_dates):
        """
        **Feature: stock-direction-predictor, Property 1: Data Collection Completeness**
        **Validates: Requirements 1.1, 1.3, 1.4, 1.5**
        
        For any valid symbols and date range, when data is successfully fetched,
        the result should contain all required fields and pass validation.
        """
        symbols, start_date, end_date = symbols_and_dates
        
        # Mock yfinance to return valid data
        with patch('yfinance.Ticker') as mock_ticker:
            # Create mock data for each symbol
            mock_data = {}
            for symbol in symbols:
                # Generate valid mock data
                mock_df = pd.DataFrame({
                    'Open': [100.0, 101.0, 102.0],
                    'High': [105.0, 106.0, 107.0],
                    'Low': [99.0, 100.0, 101.0],
                    'Close': [104.0, 105.0, 106.0],
                    'Volume': [1000000, 1100000, 1200000],
                    'Adj Close': [104.0, 105.0, 106.0]
                }, index=pd.date_range(start=start_date, periods=3, freq='D'))
                
                mock_ticker_instance = Mock()
                mock_ticker_instance.history.return_value = mock_df
                mock_ticker.return_value = mock_ticker_instance
                
                # Test fetch for this symbol
                result = self.service.fetch_stock_data([symbol], start_date, end_date)
                
                # Verify result structure
                assert symbol in result
                assert isinstance(result[symbol], pd.DataFrame)
                assert not result[symbol].empty
                
                # Verify all required columns are present
                for col in self.service.required_columns:
                    assert col in result[symbol].columns
                
                # Verify data passes validation
                assert self.service.validate_data_completeness(result[symbol])
    
    @given(st.data())
    @settings(max_examples=50, deadline=None)
    def test_property_ohlc_relationships_validation(self, data):
        """
        **Feature: stock-direction-predictor, Property 1: Data Collection Completeness**
        **Validates: Requirements 1.1, 1.3, 1.4, 1.5**
        
        For any dataset, OHLC validation should correctly identify invalid
        price relationships (High < max(Open,Close) or Low > min(Open,Close)).
        """
        # Generate valid OHLC data first
        valid_data = data.draw(valid_stock_dataframe(min_rows=3, max_rows=10))
        
        # This should pass validation
        assert self.service.validate_data_completeness(valid_data)
        
        # Now create invalid OHLC relationships
        invalid_data = valid_data.copy()
        if len(invalid_data) > 0:
            # Make High less than Close (invalid)
            invalid_data.iloc[0, invalid_data.columns.get_loc('High')] = \
                invalid_data.iloc[0, invalid_data.columns.get_loc('Close')] - 1.0
            
            # This should fail validation
            assert self.service.validate_data_completeness(invalid_data) == False
    
    def test_property_error_handling_with_network_failures(self):
        """
        **Feature: stock-direction-predictor, Property 1: Data Collection Completeness**
        **Validates: Requirements 1.1, 1.3, 1.4, 1.5**
        
        When network failures occur, the service should handle errors gracefully
        and raise appropriate exceptions after exhausting retries.
        """
        # Mock yfinance to raise exceptions
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.side_effect = Exception("Network error")
            mock_ticker.return_value = mock_ticker_instance
            
            # Should raise DataCollectionError after retries
            with pytest.raises(DataCollectionError):
                self.service.fetch_stock_data(['AAPL'], '2023-01-01', '2023-01-31')
    
    def test_property_cache_functionality_preserves_data_integrity(self):
        """
        **Feature: stock-direction-predictor, Property 1: Data Collection Completeness**
        **Validates: Requirements 1.1, 1.3, 1.4, 1.5**
        
        Cached data should maintain the same structure and content as original data.
        """
        # Create test data
        test_data = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0],
            'Low': [99.0, 100.0],
            'Close': [104.0, 105.0],
            'Volume': [1000000, 1100000],
            'Adj Close': [104.0, 105.0]
        }, index=pd.date_range(start='2023-01-01', periods=2, freq='D'))
        
        # Save to cache
        self.service._save_to_cache('TEST', '2023-01-01', '2023-01-02', test_data)
        
        # Load from cache
        cached_data = self.service._load_from_cache('TEST', '2023-01-01', '2023-01-02')
        
        # Verify data integrity
        assert cached_data is not None
        pd.testing.assert_frame_equal(test_data, cached_data)
        assert self.service.validate_data_completeness(cached_data)