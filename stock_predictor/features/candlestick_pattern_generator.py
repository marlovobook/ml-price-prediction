"""
Candlestick Pattern Generator for Stock Direction Predictor.
Implements N-day candlestick pattern signal generation with validation.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging
from ..interfaces import ICandlestickPatternGenerator
from ..utils.exceptions import DataValidationError


class CandlestickPatternGenerator(ICandlestickPatternGenerator):
    """
    Candlestick pattern generator that creates trading signals based on
    consecutive N-day candlestick patterns.
    
    Supports pattern lengths of 3, 5, 7, and 14 days.
    Generates buy signals for N consecutive green candles,
    sell signals for N consecutive red candles, and hold signals otherwise.
    """
    
    SUPPORTED_PATTERN_LENGTHS = [3, 5, 7, 14]
    
    def __init__(self):
        """Initialize the candlestick pattern generator."""
        self.logger = logging.getLogger(__name__)
        
    def generate_n_day_signals(self, data: pd.DataFrame, n: int) -> pd.Series:
        """
        Generate N-day candlestick pattern signals.
        
        Args:
            data: DataFrame with OHLC data (must contain 'open' and 'close' columns)
            n: Pattern length (3, 5, 7, or 14 days)
            
        Returns:
            Series with trading signals:
            - 1: Buy signal (N consecutive green candles)
            - -1: Sell signal (N consecutive red candles)
            - 0: Hold signal (mixed or insufficient pattern)
            
        Raises:
            DataValidationError: If input data is invalid or pattern length unsupported
        """
        # Validate inputs
        self._validate_input_data(data)
        self._validate_pattern_length(n)
        
        if len(data) < n:
            raise DataValidationError(f"Insufficient data points. Need at least {n}, got {len(data)}")
        
        try:
            # Identify candle colors
            candle_colors = self._identify_candle_colors(data)
            
            # Generate signals based on consecutive patterns
            signals = self._generate_signals_from_colors(candle_colors, n)
            
            self.logger.debug(f"Generated {n}-day signals: {signals.value_counts().to_dict()}")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating {n}-day signals: {str(e)}")
            raise DataValidationError(f"Failed to generate {n}-day signals: {str(e)}")
    
    def validate_pattern_consistency(self, signals: pd.Series) -> bool:
        """
        Validate the consistency of generated pattern signals.
        
        Args:
            signals: Series of trading signals (-1, 0, 1)
            
        Returns:
            True if signals are consistent, False otherwise
            
        Raises:
            DataValidationError: If signals contain invalid values
        """
        if signals.empty:
            self.logger.warning("Empty signals series provided for validation")
            return False
        
        try:
            # Check that all signals are valid values (-1, 0, 1)
            valid_values = {-1, 0, 1}
            unique_signals = set(signals.dropna().unique())
            
            if not unique_signals.issubset(valid_values):
                invalid_values = unique_signals - valid_values
                raise DataValidationError(f"Invalid signal values found: {invalid_values}")
            
            # Check for reasonable signal distribution
            signal_counts = signals.value_counts()
            total_signals = len(signals.dropna())
            
            if total_signals == 0:
                self.logger.warning("No valid signals found")
                return False
            
            # For realistic market data, it's normal to have mostly hold signals (0)
            # Only flag as problematic if ALL signals are buy (1) or sell (-1)
            if len(signal_counts) == 1 and signal_counts.index[0] != 0:
                self.logger.warning(f"All signals have the same non-hold value: {signal_counts.index[0]}")
                return False
            
            # Check that hold signals (0) are the majority (expected for most market conditions)
            hold_ratio = signal_counts.get(0, 0) / total_signals
            if hold_ratio < 0.5:
                self.logger.warning(f"Hold signals ratio is unusually low: {hold_ratio:.2f}")
            
            # Check for excessive consecutive signals of the same type
            max_consecutive = self._check_consecutive_signals(signals)
            if max_consecutive > 50:  # Arbitrary threshold for excessive consecutive signals
                self.logger.warning(f"Excessive consecutive signals detected: {max_consecutive}")
            
            self.logger.debug("Signal validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal consistency: {str(e)}")
            raise DataValidationError(f"Signal validation failed: {str(e)}")
    
    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """
        Validate input DataFrame has required columns and data.
        
        Args:
            data: Input DataFrame to validate
            
        Raises:
            DataValidationError: If data is invalid
        """
        if data.empty:
            raise DataValidationError("Input data is empty")
        
        required_columns = ['open', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
        
        # Check for non-numeric data
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise DataValidationError(f"Column '{col}' must be numeric")
        
        # Check for negative prices (which would be invalid)
        for col in required_columns:
            if (data[col] <= 0).any():
                raise DataValidationError(f"Column '{col}' contains non-positive values")
    
    def _validate_pattern_length(self, n: int) -> None:
        """
        Validate that pattern length is supported.
        
        Args:
            n: Pattern length to validate
            
        Raises:
            DataValidationError: If pattern length is not supported
        """
        if n not in self.SUPPORTED_PATTERN_LENGTHS:
            raise DataValidationError(
                f"Unsupported pattern length: {n}. "
                f"Supported lengths: {self.SUPPORTED_PATTERN_LENGTHS}"
            )
    
    def _identify_candle_colors(self, data: pd.DataFrame) -> pd.Series:
        """
        Identify candle colors based on open and close prices.
        
        Args:
            data: DataFrame with 'open' and 'close' columns
            
        Returns:
            Series with candle colors (1 for green, -1 for red)
        """
        # Green candle: close > open (bullish)
        # Red candle: close < open (bearish)
        # Doji candle: close == open (neutral, treated as 0)
        candle_colors = np.where(
            data['close'] > data['open'], 1,  # Green candle
            np.where(data['close'] < data['open'], -1, 0)  # Red candle or Doji
        )
        
        return pd.Series(candle_colors, index=data.index, name='candle_color')
    
    def _generate_signals_from_colors(self, candle_colors: pd.Series, n: int) -> pd.Series:
        """
        Generate trading signals from candle colors using N-day patterns.
        
        Args:
            candle_colors: Series of candle colors (1, -1, 0)
            n: Pattern length
            
        Returns:
            Series with trading signals
        """
        # Calculate rolling sum of candle colors over N periods
        rolling_sum = candle_colors.rolling(window=n, min_periods=n).sum()
        
        # Generate signals based on consecutive patterns
        signals = np.where(
            rolling_sum == n, 1,      # N consecutive green candles -> Buy
            np.where(rolling_sum == -n, -1, 0)  # N consecutive red candles -> Sell, else Hold
        )
        
        return pd.Series(signals, index=candle_colors.index, name=f'signal_{n}d')
    
    def _check_consecutive_signals(self, signals: pd.Series) -> int:
        """
        Check for maximum consecutive signals of the same type.
        
        Args:
            signals: Series of trading signals
            
        Returns:
            Maximum number of consecutive identical signals
        """
        if signals.empty:
            return 0
        
        # Find consecutive runs of the same signal
        signal_changes = signals != signals.shift(1)
        run_ids = signal_changes.cumsum()
        run_lengths = signals.groupby(run_ids).size()
        
        return run_lengths.max() if not run_lengths.empty else 0
    
    def generate_n_day_signals_dataframe(self, data: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        Generate N-day candlestick pattern signals and return DataFrame with signals added.
        
        Args:
            data: DataFrame with OHLC data
            n: Pattern length (3, 5, 7, or 14 days)
            
        Returns:
            DataFrame with original data plus signal column
        """
        result = data.copy()
        signals = self.generate_n_day_signals(data, n)
        result[f'signal_{n}d'] = signals
        return result
    
    def get_pattern_statistics(self, data: pd.DataFrame, pattern_lengths: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Generate statistics for multiple pattern lengths.
        
        Args:
            data: DataFrame with OHLC data
            pattern_lengths: List of pattern lengths to analyze (default: all supported)
            
        Returns:
            DataFrame with pattern statistics for each length
        """
        if pattern_lengths is None:
            pattern_lengths = self.SUPPORTED_PATTERN_LENGTHS
        
        stats = []
        
        for n in pattern_lengths:
            try:
                signals = self.generate_n_day_signals(data, n)
                signal_counts = signals.value_counts()
                
                stats.append({
                    'pattern_length': n,
                    'total_signals': len(signals),
                    'buy_signals': signal_counts.get(1, 0),
                    'sell_signals': signal_counts.get(-1, 0),
                    'hold_signals': signal_counts.get(0, 0),
                    'buy_ratio': signal_counts.get(1, 0) / len(signals),
                    'sell_ratio': signal_counts.get(-1, 0) / len(signals),
                    'hold_ratio': signal_counts.get(0, 0) / len(signals)
                })
            except Exception as e:
                self.logger.error(f"Error calculating statistics for pattern length {n}: {str(e)}")
                stats.append({
                    'pattern_length': n,
                    'error': str(e)
                })
        
        return pd.DataFrame(stats)