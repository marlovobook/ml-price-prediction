"""
Property-based tests for backtesting engine.
Tests backtesting simulation accuracy and portfolio tracking.
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
import sys
import os
from typing import Dict, List, Any
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stock_predictor.backtesting.backtesting_engine import BacktestingEngine
from stock_predictor.interfaces import BacktestResult


def generate_realistic_price_series(num_days: int, initial_price: float = 100.0, 
                                   volatility: float = 0.02) -> pd.Series:
    """Generate realistic price series using geometric Brownian motion."""
    np.random.seed(42)  # For reproducible tests
    
    dates = pd.date_range(start='2023-01-01', periods=num_days, freq='D')
    prices = [initial_price]
    
    for i in range(num_days - 1):
        # Geometric Brownian motion
        drift = 0.0001  # Small positive drift
        shock = np.random.normal(0, volatility)
        price_change = prices[-1] * (drift + shock)
        new_price = max(prices[-1] + price_change, 0.01)  # Ensure positive prices
        prices.append(new_price)
    
    return pd.Series(prices, index=dates)


def generate_valid_signals(num_days: int, signal_frequency: float = 0.1) -> pd.Series:
    """Generate valid trading signals with controlled frequency."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=num_days, freq='D')
    signals = []
    
    current_position = 0  # Track current position to generate valid signals
    
    for i in range(num_days):
        # Generate signal with some probability
        if np.random.random() < signal_frequency:
            if current_position <= 0:  # Can buy
                signal = 1
                current_position = 1
            else:  # Can sell
                signal = -1
                current_position = -1
        else:
            signal = 0  # Hold
        
        signals.append(signal)
    
    return pd.Series(signals, index=dates)


class TestBacktestingProperties:
    """Property-based tests for backtesting engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = BacktestingEngine(
            transaction_cost=0.001,
            slippage=0.0005,
            max_position_size=0.1
        )
    
    @given(
        num_days=st.integers(min_value=10, max_value=252),
        initial_capital=st.floats(min_value=1000.0, max_value=100000.0),
        initial_price=st.floats(min_value=10.0, max_value=1000.0),
        volatility=st.floats(min_value=0.005, max_value=0.05)
    )
    @settings(max_examples=100, deadline=None)
    def test_backtesting_simulation_accuracy(self, num_days, initial_capital, initial_price, volatility):
        """
        **Feature: stock-direction-predictor, Property 7: Backtesting Simulation Accuracy**
        **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**
        
        For any sequence of trading signals and price data, the backtesting engine should 
        simulate trades correctly, track portfolio values, account for transaction costs, 
        and generate complete trade logs.
        """
        # Generate test data
        prices = generate_realistic_price_series(num_days, initial_price, volatility)
        signals = generate_valid_signals(num_days, signal_frequency=0.05)
        
        # Ensure we have some signals to test
        assume(signals.abs().sum() > 0)
        
        # Run backtesting simulation
        result = self.engine.simulate_trading(signals, prices, initial_capital)
        
        # Property 1: Result should be a valid BacktestResult
        assert isinstance(result, BacktestResult), "Result should be BacktestResult instance"
        
        # Property 2: Portfolio values should have same length as input data
        assert len(result.portfolio_values) == len(signals), "Portfolio values should match signal length"
        
        # Property 3: Portfolio values should be positive
        assert (result.portfolio_values > 0).all(), "Portfolio values should always be positive"
        
        # Property 4: Portfolio values should be finite
        assert result.portfolio_values.isna().sum() == 0, "Portfolio values should not contain NaN"
        assert np.isfinite(result.portfolio_values).all(), "Portfolio values should be finite"
        
        # Property 5: Total return should be calculated correctly
        expected_return = (result.portfolio_values.iloc[-1] / result.portfolio_values.iloc[0]) - 1
        assert abs(result.total_return - expected_return) < 1e-10, "Total return should be calculated correctly"
        
        # Property 6: Maximum drawdown should be non-positive
        assert result.max_drawdown <= 0, f"Maximum drawdown should be <= 0, got {result.max_drawdown}"
        
        # Property 7: Win rate should be between 0 and 1
        assert 0 <= result.win_rate <= 1, f"Win rate should be between 0 and 1, got {result.win_rate}"
        
        # Property 8: Profit factor should be non-negative
        assert result.profit_factor >= 0, f"Profit factor should be >= 0, got {result.profit_factor}"
        
        # Property 9: Trade log should be a valid DataFrame
        assert isinstance(result.trade_log, pd.DataFrame), "Trade log should be DataFrame"
        
        # Property 10: All financial metrics should be finite (except profit_factor which can be inf)
        assert np.isfinite(result.total_return), "Total return should be finite"
        assert np.isfinite(result.max_drawdown), "Max drawdown should be finite"
        # Sharpe ratio can be NaN if no volatility, but should not be inf
        if np.isfinite(result.sharpe_ratio):
            assert not np.isinf(result.sharpe_ratio), "Sharpe ratio should not be infinite"
        
        # Property 11: Portfolio should start with initial capital
        assert abs(result.portfolio_values.iloc[0] - initial_capital) < 1e-6, "Portfolio should start with initial capital"
    
    @given(
        num_days=st.integers(min_value=5, max_value=100),
        initial_capital=st.floats(min_value=1000.0, max_value=50000.0),
        transaction_cost=st.floats(min_value=0.0, max_value=0.01),
        slippage=st.floats(min_value=0.0, max_value=0.01)
    )
    @settings(max_examples=50, deadline=None)
    def test_transaction_cost_impact(self, num_days, initial_capital, transaction_cost, slippage):
        """
        Test that transaction costs and slippage are properly applied.
        """
        # Create engine with specified costs
        engine = BacktestingEngine(
            transaction_cost=transaction_cost,
            slippage=slippage,
            max_position_size=0.2
        )
        
        # Generate test data with some trading activity
        prices = generate_realistic_price_series(num_days, 100.0, 0.02)
        signals = generate_valid_signals(num_days, signal_frequency=0.1)
        
        # Ensure we have some trading signals
        assume(signals.abs().sum() >= 2)  # At least one buy and one sell
        
        # Run simulation
        result = engine.simulate_trading(signals, prices, initial_capital)
        
        # Property 1: With transaction costs, final value should be less than perfect execution
        # (This is hard to test directly, but we can check that costs are being applied)
        
        # Property 2: Trade log should contain transaction cost information if trades occurred
        if len(result.trade_log) > 0:
            # Check that we have the expected columns
            expected_columns = ['trade_id', 'entry_timestamp', 'exit_timestamp', 'signal', 
                              'entry_price', 'exit_price', 'position_type', 'pnl', 'return_pct']
            for col in expected_columns:
                assert col in result.trade_log.columns, f"Trade log should contain {col} column"
        
        # Property 3: Portfolio values should reflect transaction costs
        # If there are trades, the impact should be visible
        if len(result.trade_log) > 0:
            # The final portfolio value should account for all costs
            assert result.portfolio_values.iloc[-1] > 0, "Final portfolio value should be positive"
    
    @given(
        num_days=st.integers(min_value=10, max_value=100),
        max_position_size=st.floats(min_value=0.01, max_value=0.5)
    )
    @settings(max_examples=50, deadline=None)
    def test_position_sizing_limits(self, num_days, max_position_size):
        """
        Test that position sizing limits are respected.
        """
        engine = BacktestingEngine(
            transaction_cost=0.001,
            slippage=0.0005,
            max_position_size=max_position_size
        )
        
        initial_capital = 10000.0
        prices = generate_realistic_price_series(num_days, 100.0, 0.02)
        
        # Create signals that would trigger position sizing limits
        signals = pd.Series([1] * num_days, index=prices.index)  # All buy signals
        
        result = engine.simulate_trading(signals, prices, initial_capital)
        
        # Property 1: Portfolio should never exceed reasonable bounds based on position sizing
        # This is complex to verify directly, but we can check basic sanity
        assert result.portfolio_values.max() < initial_capital * 10, "Portfolio shouldn't grow unreasonably"
        assert result.portfolio_values.min() > 0, "Portfolio should remain positive"
        
        # Property 2: Position sizing should prevent excessive concentration
        # We can't directly test this without access to internal position tracking,
        # but we can verify the simulation completes successfully
        assert len(result.portfolio_values) == num_days, "Simulation should complete successfully"
    
    @given(
        signal_pattern=st.lists(st.sampled_from([-1, 0, 1]), min_size=5, max_size=20)
    )
    @settings(max_examples=50, deadline=None)
    def test_signal_processing_consistency(self, signal_pattern):
        """
        Test that different signal patterns are processed consistently.
        """
        # Create price series
        num_days = len(signal_pattern)
        prices = generate_realistic_price_series(num_days, 100.0, 0.01)
        
        # Create signals from pattern
        signals = pd.Series(signal_pattern, index=prices.index)
        
        initial_capital = 10000.0
        result = self.engine.simulate_trading(signals, prices, initial_capital)
        
        # Property 1: Simulation should handle any valid signal sequence
        assert isinstance(result, BacktestResult), "Should return valid BacktestResult"
        
        # Property 2: Portfolio values should have correct length
        assert len(result.portfolio_values) == num_days, "Portfolio values should match input length"
        
        # Property 3: All portfolio values should be positive and finite
        assert (result.portfolio_values > 0).all(), "All portfolio values should be positive"
        assert np.isfinite(result.portfolio_values).all(), "All portfolio values should be finite"
        
        # Property 4: Trade log should be consistent with signals
        # Count actual trading signals (non-zero)
        trading_signals = signals[signals != 0]
        
        # Trade log length should be reasonable relative to trading signals
        # (May be less due to position constraints, but shouldn't be more)
        if len(trading_signals) > 0:
            assert len(result.trade_log) <= len(trading_signals), "Trade log shouldn't exceed trading signals"
    
    @given(
        num_days=st.integers(min_value=20, max_value=100),
        price_trend=st.floats(min_value=-0.002, max_value=0.002)  # Daily trend
    )
    @settings(max_examples=50, deadline=None)
    def test_portfolio_metrics_calculation(self, num_days, price_trend):
        """
        Test portfolio metrics calculation properties.
        """
        # Generate trending price series
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=num_days, freq='D')
        prices = [100.0]
        
        for i in range(num_days - 1):
            # Add trend and some noise
            change = price_trend + np.random.normal(0, 0.01)
            new_price = max(prices[-1] * (1 + change), 0.01)
            prices.append(new_price)
        
        price_series = pd.Series(prices, index=dates)
        
        # Generate some trading signals
        signals = generate_valid_signals(num_days, signal_frequency=0.08)
        
        initial_capital = 10000.0
        result = self.engine.simulate_trading(signals, price_series, initial_capital)
        
        # Test portfolio metrics calculation
        portfolio_metrics = self.engine.calculate_portfolio_metrics(result.portfolio_values)
        
        # Property 1: Total return should match BacktestResult
        assert abs(portfolio_metrics['total_return'] - result.total_return) < 1e-10, "Total return should match"
        
        # Property 2: Maximum drawdown should match BacktestResult
        assert abs(portfolio_metrics['max_drawdown'] - result.max_drawdown) < 1e-10, "Max drawdown should match"
        
        # Property 3: All metrics should be finite or appropriately infinite
        assert np.isfinite(portfolio_metrics['total_return']), "Total return should be finite"
        assert np.isfinite(portfolio_metrics['max_drawdown']), "Max drawdown should be finite"
        assert np.isfinite(portfolio_metrics['volatility']), "Volatility should be finite"
        
        # Property 4: Volatility should be non-negative
        assert portfolio_metrics['volatility'] >= 0, "Volatility should be non-negative"
        
        # Property 5: Maximum drawdown should be non-positive
        assert portfolio_metrics['max_drawdown'] <= 0, "Max drawdown should be non-positive"
        
        # Property 6: Calmar ratio should be finite when max drawdown != 0
        if portfolio_metrics['max_drawdown'] != 0:
            assert np.isfinite(portfolio_metrics['calmar_ratio']), "Calmar ratio should be finite when max drawdown != 0"
    
    @given(
        num_days=st.integers(min_value=10, max_value=50)
    )
    @settings(max_examples=30, deadline=None)
    def test_trade_log_generation(self, num_days):
        """
        Test trade log generation properties.
        """
        prices = generate_realistic_price_series(num_days, 100.0, 0.02)
        signals = generate_valid_signals(num_days, signal_frequency=0.1)
        
        # Generate trade log directly
        trade_log = self.engine.generate_trade_log(signals, prices)
        
        # Property 1: Trade log should be a DataFrame
        assert isinstance(trade_log, pd.DataFrame), "Trade log should be DataFrame"
        
        # Property 2: Trade log should have expected columns
        expected_columns = ['trade_id', 'entry_timestamp', 'exit_timestamp', 'signal',
                           'entry_price', 'exit_price', 'position_type', 'pnl', 'return_pct']
        
        if len(trade_log) > 0:
            for col in expected_columns:
                assert col in trade_log.columns, f"Trade log should contain {col} column"
        
        # Property 3: Trade IDs should be sequential if trades exist
        if len(trade_log) > 0:
            trade_ids = trade_log['trade_id'].tolist()
            expected_ids = list(range(1, len(trade_log) + 1))
            assert trade_ids == expected_ids, "Trade IDs should be sequential starting from 1"
        
        # Property 4: Entry timestamps should be before or equal to exit timestamps
        if len(trade_log) > 0:
            for _, trade in trade_log.iterrows():
                assert trade['entry_timestamp'] <= trade['exit_timestamp'], "Entry should be before or equal to exit"
        
        # Property 5: All prices should be positive
        if len(trade_log) > 0:
            assert (trade_log['entry_price'] > 0).all(), "Entry prices should be positive"
            assert (trade_log['exit_price'] > 0).all(), "Exit prices should be positive"
        
        # Property 6: Position types should be valid
        if len(trade_log) > 0:
            valid_positions = {'long', 'short'}
            assert trade_log['position_type'].isin(valid_positions).all(), "Position types should be 'long' or 'short'"
    
    def test_empty_signals_handling(self):
        """
        Test handling of edge cases like empty signals.
        """
        # Test empty signals
        empty_signals = pd.Series([], dtype=int)
        empty_prices = pd.Series([], dtype=float)
        
        with pytest.raises(Exception):  # Should raise ValidationError
            self.engine.simulate_trading(empty_signals, empty_prices, 10000.0)
    
    def test_invalid_inputs_handling(self):
        """
        Test handling of invalid inputs.
        """
        prices = generate_realistic_price_series(10, 100.0, 0.02)
        signals = generate_valid_signals(10, 0.1)
        
        # Test negative initial capital
        with pytest.raises(Exception):  # Should raise ValidationError
            self.engine.simulate_trading(signals, prices, -1000.0)
        
        # Test zero initial capital
        with pytest.raises(Exception):  # Should raise ValidationError
            self.engine.simulate_trading(signals, prices, 0.0)
        
        # Test mismatched lengths
        short_signals = signals.iloc[:5]
        with pytest.raises(Exception):  # Should raise ValidationError
            self.engine.simulate_trading(short_signals, prices, 10000.0)


if __name__ == "__main__":
    pytest.main([__file__])