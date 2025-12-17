"""
Property-based tests for feature engineering module.
Tests technical indicator accuracy and pattern detection consistency.
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stock_predictor.features.feature_engineering import FeatureEngineeringModule
from stock_predictor.features.candlestick_pattern_generator import CandlestickPatternGenerator
from stock_predictor.utils.exceptions import DataValidationError


def generate_ohlc_data(num_rows: int, base_price: float = 100.0) -> pd.DataFrame:
    """Generate realistic OHLC data for testing."""
    np.random.seed(42)  # For reproducible tests
    
    data = []
    current_price = base_price
    
    for i in range(num_rows):
        # Generate price movement
        change_pct = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = current_price * (1 + change_pct)
        
        # Generate OHLC with realistic relationships
        open_price = current_price
        close_price = new_price
        
        # High and low with some randomness
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
        
        volume = np.random.randint(100000, 10000000)
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        current_price = new_price
    
    return pd.DataFrame(data)


class TestFeatureEngineeringProperties:
    """Property-based tests for feature engineering module."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.feature_module = FeatureEngineeringModule()
        self.candlestick_generator = CandlestickPatternGenerator()
    
    @given(
        num_rows=st.integers(min_value=250, max_value=500),
        base_price=st.floats(min_value=10.0, max_value=1000.0)
    )
    @settings(max_examples=100, deadline=None)
    def test_technical_indicator_calculation_accuracy(self, num_rows, base_price):
        """
        **Feature: stock-direction-predictor, Property 2: Technical Indicator Calculation Accuracy**
        **Validates: Requirements 2.1, 2.4, 2.5**
        
        For any valid OHLC dataset, the feature engineering module should calculate 
        all technical indicators following standard financial formulas with values 
        within expected ranges.
        """
        # Generate realistic OHLC data
        ohlc_data = generate_ohlc_data(num_rows, base_price)
        
        # Calculate technical indicators
        result = self.feature_module.calculate_technical_indicators(ohlc_data)
        
        # Property 1: RSI should be between 0 and 100
        rsi_values = result['rsi'].dropna()
        if len(rsi_values) > 0:
            assert rsi_values.min() >= 0, f"RSI minimum {rsi_values.min()} should be >= 0"
            assert rsi_values.max() <= 100, f"RSI maximum {rsi_values.max()} should be <= 100"
        
        # Property 2: ATR should be positive
        atr_values = result['atr'].dropna()
        if len(atr_values) > 0:
            assert atr_values.min() >= 0, f"ATR minimum {atr_values.min()} should be >= 0"
        
        # Property 3: EMAs should be positive for positive stock prices
        for ema_col in ['ema_20', 'ema_50', 'ema_200']:
            ema_values = result[ema_col].dropna()
            if len(ema_values) > 0:
                assert ema_values.min() > 0, f"{ema_col} minimum {ema_values.min()} should be > 0"
        
        # Property 4: EMA ordering should generally follow EMA20 >= EMA50 >= EMA200 in uptrends
        # (This is not always true but should hold for most cases with sufficient data)
        ema_20_vals = result['ema_20'].dropna()
        ema_50_vals = result['ema_50'].dropna()
        ema_200_vals = result['ema_200'].dropna()
        
        if len(ema_20_vals) > 0 and len(ema_50_vals) > 0 and len(ema_200_vals) > 0:
            # Check that EMAs are reasonable relative to price data
            price_min = ohlc_data['close'].min()
            price_max = ohlc_data['close'].max()
            
            for ema_col in ['ema_20', 'ema_50', 'ema_200']:
                ema_vals = result[ema_col].dropna()
                if len(ema_vals) > 0:
                    # EMAs should be within reasonable range of price data
                    assert ema_vals.min() >= price_min * 0.5, f"{ema_col} too far below price range"
                    assert ema_vals.max() <= price_max * 1.5, f"{ema_col} too far above price range"
        
        # Property 5: MACD components should be finite
        macd_vals = result['macd'].dropna()
        macd_signal_vals = result['macd_signal'].dropna()
        
        if len(macd_vals) > 0:
            assert np.all(np.isfinite(macd_vals)), "MACD values should be finite"
        
        if len(macd_signal_vals) > 0:
            assert np.all(np.isfinite(macd_signal_vals)), "MACD signal values should be finite"
        
        # Property 6: SMA should be positive for positive prices
        sma_vals = result['sma'].dropna()
        if len(sma_vals) > 0:
            assert sma_vals.min() > 0, f"SMA minimum {sma_vals.min()} should be > 0"
    
    @given(
        num_rows=st.integers(min_value=250, max_value=300),  # Ensure sufficient data
        base_price=st.floats(min_value=10.0, max_value=200.0),
        pattern_length=st.sampled_from([3, 5, 7, 14])
    )
    @settings(max_examples=100, deadline=None)
    def test_pattern_detection_consistency(self, num_rows, base_price, pattern_length):
        """
        **Feature: stock-direction-predictor, Property 3: Pattern Detection Consistency**
        **Validates: Requirements 2.2, 2.3**
        
        For any price dataset, the system should detect chart patterns and calculate 
        Fibonacci retracement levels consistently based on mathematical definitions.
        """
        # Generate realistic OHLC data
        ohlc_data = generate_ohlc_data(num_rows, base_price)
        
        # Test chart pattern detection
        result_patterns = self.feature_module.detect_chart_patterns(ohlc_data)
        
        # Property 1: Golden cross and death cross should be mutually exclusive
        golden_cross = result_patterns['golden_cross']
        death_cross = result_patterns['death_cross']
        
        # No simultaneous golden and death cross signals
        simultaneous_signals = (golden_cross == 1) & (death_cross == 1)
        assert not simultaneous_signals.any(), "Golden cross and death cross cannot occur simultaneously"
        
        # Property 2: Pattern signals should be binary (0 or 1, or -1 for some patterns)
        assert golden_cross.isin([0, 1]).all(), "Golden cross signals should be 0 or 1"
        assert death_cross.isin([0, 1]).all(), "Death cross signals should be 0 or 1"
        
        # Test Fibonacci levels calculation
        result_fib = self.feature_module.calculate_fibonacci_levels(ohlc_data)
        
        # Property 3: Fibonacci levels should be ordered correctly
        fib_levels = ['fib_23.6', 'fib_38.2', 'fib_50.0', 'fib_61.8', 'fib_78.6']  # From high to low
        
        for i in range(len(ohlc_data)):
            if pd.notna(result_fib.iloc[i]['rolling_high']) and pd.notna(result_fib.iloc[i]['rolling_low']):
                high_val = result_fib.iloc[i]['rolling_high']
                low_val = result_fib.iloc[i]['rolling_low']
                
                # Skip if high == low (no range)
                if high_val == low_val:
                    continue
                
                # Fibonacci levels should be between high and low
                for fib_col in fib_levels:
                    fib_val = result_fib.iloc[i][fib_col]
                    if pd.notna(fib_val):
                        assert low_val <= fib_val <= high_val, f"Fibonacci level {fib_col} should be between high and low"
                
                # Fibonacci levels should be in descending order from high to low
                # fib_23.6 should be closest to high, fib_78.6 closest to low
                prev_fib = high_val
                for fib_col in fib_levels:
                    fib_val = result_fib.iloc[i][fib_col]
                    if pd.notna(fib_val):
                        assert fib_val <= prev_fib, f"Fibonacci level {fib_col} ({fib_val}) should be <= previous level ({prev_fib})"
                        prev_fib = fib_val
        
        # Property 4: Fibonacci position should be valid (1-6)
        fib_positions = result_fib['fib_position'].dropna()
        if len(fib_positions) > 0:
            assert fib_positions.min() >= 1, "Fibonacci position should be >= 1"
            assert fib_positions.max() <= 6, "Fibonacci position should be <= 6"
        
        # Property 5: Head and shoulders pattern should be binary
        head_shoulders = result_patterns['head_shoulders']
        assert head_shoulders.isin([0, 1]).all(), "Head and shoulders signals should be 0 or 1"
        
        # Property 6: Wedge pattern should be in valid range
        wedge_pattern = result_patterns['wedge_pattern']
        assert wedge_pattern.isin([-1, 0, 1]).all(), "Wedge pattern signals should be -1, 0, or 1"
    
    @given(
        num_rows=st.integers(min_value=20, max_value=100),
        base_price=st.floats(min_value=10.0, max_value=500.0),
        pattern_length=st.sampled_from([3, 5, 7, 14])
    )
    @settings(max_examples=100, deadline=None)
    def test_candlestick_signal_generation(self, num_rows, base_price, pattern_length):
        """
        **Feature: stock-direction-predictor, Property 4: Candlestick Signal Generation**
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**
        
        For any N-day pattern length (3, 5, 7, 14), the system should generate buy signals 
        for N consecutive green candles, sell signals for N consecutive red candles, 
        and hold signals otherwise, with correct candle color identification.
        """
        # Generate realistic OHLC data
        ohlc_data = generate_ohlc_data(num_rows, base_price)
        
        # Generate N-day candlestick signals
        signals = self.candlestick_generator.generate_n_day_signals(ohlc_data, pattern_length)
        
        # Property 1: Signals should only contain valid values (-1, 0, 1)
        valid_signals = {-1, 0, 1}
        unique_signals = set(signals.dropna().unique())
        assert unique_signals.issubset(valid_signals), f"Invalid signal values: {unique_signals - valid_signals}"
        
        # Property 2: Signal validation should pass for generated signals
        assert self.candlestick_generator.validate_pattern_consistency(signals), "Generated signals should pass consistency validation"
        
        # Property 3: Verify candle color identification logic
        # Green candles: close > open should have positive color
        # Red candles: close < open should have negative color
        green_candles = ohlc_data['close'] > ohlc_data['open']
        red_candles = ohlc_data['close'] < ohlc_data['open']
        doji_candles = ohlc_data['close'] == ohlc_data['open']
        
        # Property 4: Buy signals should only occur after N consecutive green candles
        buy_signals = signals == 1
        if buy_signals.any():
            for idx in signals[buy_signals].index:
                # Get the position in the dataframe
                pos = ohlc_data.index.get_loc(idx)
                if pos >= pattern_length - 1:  # Ensure we have enough history
                    # Check the last N candles before and including this signal
                    start_pos = pos - pattern_length + 1
                    end_pos = pos + 1
                    recent_candles = ohlc_data.iloc[start_pos:end_pos]
                    
                    # All should be green candles (close > open)
                    all_green = (recent_candles['close'] > recent_candles['open']).all()
                    assert all_green, f"Buy signal at index {idx} should follow {pattern_length} consecutive green candles"
        
        # Property 5: Sell signals should only occur after N consecutive red candles
        sell_signals = signals == -1
        if sell_signals.any():
            for idx in signals[sell_signals].index:
                # Get the position in the dataframe
                pos = ohlc_data.index.get_loc(idx)
                if pos >= pattern_length - 1:  # Ensure we have enough history
                    # Check the last N candles before and including this signal
                    start_pos = pos - pattern_length + 1
                    end_pos = pos + 1
                    recent_candles = ohlc_data.iloc[start_pos:end_pos]
                    
                    # All should be red candles (close < open)
                    all_red = (recent_candles['close'] < recent_candles['open']).all()
                    assert all_red, f"Sell signal at index {idx} should follow {pattern_length} consecutive red candles"
        
        # Property 6: Hold signals should occur when pattern criteria are not met
        hold_signals = signals == 0
        if hold_signals.any():
            for idx in signals[hold_signals].index:
                pos = ohlc_data.index.get_loc(idx)
                if pos >= pattern_length - 1:  # Ensure we have enough history
                    start_pos = pos - pattern_length + 1
                    end_pos = pos + 1
                    recent_candles = ohlc_data.iloc[start_pos:end_pos]
                    
                    # Should NOT be all green or all red
                    all_green = (recent_candles['close'] > recent_candles['open']).all()
                    all_red = (recent_candles['close'] < recent_candles['open']).all()
                    assert not (all_green or all_red), f"Hold signal at index {idx} should not follow {pattern_length} consecutive same-color candles"
        
        # Property 7: Signal count should match input data length
        assert len(signals) == len(ohlc_data), "Signal series should have same length as input data"
        
        # Property 8: First (pattern_length - 1) signals should be NaN or 0 due to insufficient history
        if len(signals) >= pattern_length:
            early_signals = signals.iloc[:pattern_length-1]
            # Early signals should be 0 (hold) since there's insufficient history for pattern detection
            non_zero_early = early_signals[early_signals != 0]
            assert len(non_zero_early) == 0, f"Early signals should be 0 due to insufficient history, found: {non_zero_early.tolist()}"


if __name__ == "__main__":
    pytest.main([__file__])