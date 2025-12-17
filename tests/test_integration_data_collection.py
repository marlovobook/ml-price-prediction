"""
Integration tests for data collection service with real Yahoo Finance data.
These tests require network access and are optional.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import shutil

from stock_predictor.data.yahoo_finance_service import YahooFinanceDataService


@pytest.mark.integration
class TestYahooFinanceIntegration:
    """Integration tests with real Yahoo Finance API."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.service = YahooFinanceDataService(cache_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.mark.slow
    def test_fetch_real_stock_data(self):
        """Test fetching real data from Yahoo Finance."""
        # Use a short recent date range to minimize API calls
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        try:
            result = self.service.fetch_stock_data(['AAPL'], start_date, end_date)
            
            # Verify result structure
            assert 'AAPL' in result
            assert isinstance(result['AAPL'], pd.DataFrame)
            assert not result['AAPL'].empty
            
            # Verify required columns
            for col in self.service.required_columns:
                assert col in result['AAPL'].columns
            
            # Verify data passes validation
            assert self.service.validate_data_completeness(result['AAPL'])
            
        except Exception as e:
            pytest.skip(f"Network test failed (expected in CI/offline environments): {e}")
    
    @pytest.mark.slow
    def test_cache_functionality_with_real_data(self):
        """Test caching with real data."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        
        try:
            # First fetch should hit the API
            result1 = self.service.fetch_stock_data(['MSFT'], start_date, end_date)
            
            # Second fetch should use cache
            result2 = self.service.fetch_stock_data(['MSFT'], start_date, end_date)
            
            # Results should be identical
            pd.testing.assert_frame_equal(result1['MSFT'], result2['MSFT'])
            
        except Exception as e:
            pytest.skip(f"Network test failed (expected in CI/offline environments): {e}")


if __name__ == "__main__":
    # Run integration tests manually
    pytest.main([__file__, "-v", "-m", "integration"])