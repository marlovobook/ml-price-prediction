"""
Feature Engineering Module for Stock Direction Predictor.
Implements technical indicators, chart pattern detection, and Fibonacci calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from ..interfaces import IFeatureEngineeringModule
from ..utils.exceptions import DataValidationError


class FeatureEngineeringModule(IFeatureEngineeringModule):
    """
    Feature engineering module that calculates technical indicators,
    detects chart patterns, and computes Fibonacci retracement levels.
    """
    
    def __init__(self):
        """Initialize the feature engineering module."""
        self.logger = logging.getLogger(__name__)
        
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given OHLC data.
        
        Args:
            data: DataFrame with OHLC data (columns: open, high, low, close, volume)
            
        Returns:
            DataFrame with original data plus technical indicators
            
        Raises:
            DataValidationError: If input data is invalid or insufficient
        """
        if data.empty:
            raise DataValidationError("Input data is empty")
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
            
        # Minimum data points required for calculations
        # Use adaptive minimum based on available data
        min_periods = 50  # Minimum for basic indicators
        if len(data) < min_periods:
            raise DataValidationError(f"Insufficient data points. Need at least {min_periods}, got {len(data)}")
            
        result = data.copy()
        
        try:
            # RSI (Relative Strength Index) - default period 14
            result['rsi'] = self._calculate_rsi(data['close'], period=14)
            
            # MACD (Moving Average Convergence Divergence)
            macd_line, macd_signal, macd_histogram = self._calculate_macd(data['close'])
            result['macd'] = macd_line
            result['macd_signal'] = macd_signal
            result['macd_histogram'] = macd_histogram
            
            # Exponential Moving Averages (adaptive to data length)
            data_length = len(data)
            result['ema_20'] = self._calculate_ema(data['close'], period=min(20, data_length // 3))
            
            if data_length >= 50:
                result['ema_50'] = self._calculate_ema(data['close'], period=50)
            else:
                result['ema_50'] = self._calculate_ema(data['close'], period=min(30, data_length // 2))
            
            if data_length >= 200:
                result['ema_200'] = self._calculate_ema(data['close'], period=200)
            else:
                # Use longest possible period for available data
                long_period = min(100, max(50, data_length // 2))
                result['ema_200'] = self._calculate_ema(data['close'], period=long_period)
            
            # ATR (Average True Range) - default period 14
            result['atr'] = self._calculate_atr(data['high'], data['low'], data['close'], period=14)
            
            # SMA (Simple Moving Average) - using 20 period as default
            result['sma'] = data['close'].rolling(window=20).mean()
            
            # Validate indicator ranges
            self._validate_indicator_ranges(result)
            
            self.logger.info("Successfully calculated technical indicators")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            raise DataValidationError(f"Failed to calculate technical indicators: {str(e)}")
    
    def generate_candlestick_signals(self, data: pd.DataFrame, pattern_length: int) -> pd.DataFrame:
        """
        Generate candlestick pattern signals for the given pattern length.
        
        Args:
            data: DataFrame with OHLC data
            pattern_length: Number of consecutive days for pattern (3, 5, 7, or 14)
            
        Returns:
            DataFrame with candlestick signals added
        """
        if pattern_length not in [3, 5, 7, 14]:
            raise DataValidationError(f"Invalid pattern length: {pattern_length}. Must be 3, 5, 7, or 14")
            
        result = data.copy()
        
        # Identify candle colors (green = 1, red = -1, doji = 0)
        result['candle_color'] = np.where(
            result['close'] > result['open'], 1,  # Green candle
            np.where(result['close'] < result['open'], -1, 0)  # Red candle or Doji
        )
        
        # Calculate rolling sum of candle colors for pattern detection
        result['color_sum'] = result['candle_color'].rolling(window=pattern_length).sum()
        
        # Generate signals based on consecutive patterns
        # Buy signal (1): N consecutive green candles
        # Sell signal (-1): N consecutive red candles  
        # Hold signal (0): Otherwise
        result[f'signal_{pattern_length}d'] = np.where(
            result['color_sum'] == pattern_length, 1,  # All green
            np.where(result['color_sum'] == -pattern_length, -1, 0)  # All red or mixed
        )
        
        return result
    
    def detect_chart_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect chart patterns including golden cross, head and shoulder, and wedge formations.
        
        Args:
            data: DataFrame with OHLC data and EMAs
            
        Returns:
            DataFrame with chart pattern signals
        """
        result = data.copy()
        
        # Ensure EMAs are calculated
        if 'ema_20' not in result.columns or 'ema_50' not in result.columns:
            result = self.calculate_technical_indicators(result)
        
        # Golden Cross Detection (EMA20 crosses above EMA50)
        result['ema_20_prev'] = result['ema_20'].shift(1)
        result['ema_50_prev'] = result['ema_50'].shift(1)
        
        result['golden_cross'] = (
            (result['ema_20'] > result['ema_50']) & 
            (result['ema_20_prev'] <= result['ema_50_prev'])
        ).astype(int)
        
        # Death Cross Detection (EMA20 crosses below EMA50)
        result['death_cross'] = (
            (result['ema_20'] < result['ema_50']) & 
            (result['ema_20_prev'] >= result['ema_50_prev'])
        ).astype(int)
        
        # Head and Shoulders Pattern Detection (simplified)
        result['head_shoulders'] = self._detect_head_shoulders(result)
        
        # Wedge Pattern Detection (simplified)
        result['wedge_pattern'] = self._detect_wedge_pattern(result)
        
        # Clean up temporary columns
        result.drop(['ema_20_prev', 'ema_50_prev'], axis=1, inplace=True)
        
        return result
    
    def calculate_fibonacci_levels(self, data: pd.DataFrame, lookback_period: int = 50) -> pd.DataFrame:
        """
        Calculate Fibonacci retracement levels based on recent price swings.
        
        Args:
            data: DataFrame with OHLC data
            lookback_period: Number of periods to look back for high/low calculation
            
        Returns:
            DataFrame with Fibonacci levels
        """
        result = data.copy()
        
        # Calculate rolling high and low over lookback period
        result['rolling_high'] = result['high'].rolling(window=lookback_period).max()
        result['rolling_low'] = result['low'].rolling(window=lookback_period).min()
        
        # Calculate Fibonacci retracement levels
        price_range = result['rolling_high'] - result['rolling_low']
        
        # Standard Fibonacci retracement levels
        result['fib_23.6'] = result['rolling_high'] - (price_range * 0.236)
        result['fib_38.2'] = result['rolling_high'] - (price_range * 0.382)
        result['fib_50.0'] = result['rolling_high'] - (price_range * 0.500)
        result['fib_61.8'] = result['rolling_high'] - (price_range * 0.618)
        result['fib_78.6'] = result['rolling_high'] - (price_range * 0.786)
        
        # Calculate current price position relative to Fibonacci levels
        current_price = result['close']
        result['fib_position'] = np.select([
            current_price >= result['fib_23.6'], 
            current_price >= result['fib_38.2'],
            current_price >= result['fib_50.0'],
            current_price >= result['fib_61.8'],
            current_price >= result['fib_78.6']
        ], [1, 2, 3, 4, 5], default=6)
        
        return result
    
    def _validate_indicator_ranges(self, data: pd.DataFrame) -> None:
        """
        Validate that technical indicators are within expected ranges.
        
        Args:
            data: DataFrame with calculated indicators
            
        Raises:
            DataValidationError: If indicators are outside expected ranges
        """
        # RSI should be between 0 and 100
        if 'rsi' in data.columns:
            rsi_values = data['rsi'].dropna()
            if len(rsi_values) > 0 and (rsi_values.min() < 0 or rsi_values.max() > 100):
                raise DataValidationError(f"RSI values out of range [0, 100]: min={rsi_values.min()}, max={rsi_values.max()}")
        
        # ATR should be positive
        if 'atr' in data.columns:
            atr_values = data['atr'].dropna()
            if len(atr_values) > 0 and atr_values.min() < 0:
                raise DataValidationError(f"ATR values should be positive, found minimum: {atr_values.min()}")
        
        # EMAs should be positive for stock prices
        for ema_col in ['ema_20', 'ema_50', 'ema_200']:
            if ema_col in data.columns:
                ema_values = data[ema_col].dropna()
                if len(ema_values) > 0 and ema_values.min() <= 0:
                    raise DataValidationError(f"{ema_col} values should be positive, found minimum: {ema_values.min()}")
    
    def _detect_head_shoulders(self, data: pd.DataFrame, window: int = 10) -> pd.Series:
        """
        Simplified head and shoulders pattern detection.
        
        Args:
            data: DataFrame with OHLC data
            window: Window size for local maxima detection
            
        Returns:
            Series with head and shoulders signals
        """
        # Find local maxima (peaks)
        highs = data['high']
        peaks = []
        
        for i in range(window, len(highs) - window):
            if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
                peaks.append(i)
        
        # Simplified pattern: look for three consecutive peaks where middle is highest
        pattern_signals = pd.Series(0, index=data.index)
        
        for i in range(len(peaks) - 2):
            left_peak = peaks[i]
            head_peak = peaks[i + 1]
            right_peak = peaks[i + 2]
            
            left_height = highs.iloc[left_peak]
            head_height = highs.iloc[head_peak]
            right_height = highs.iloc[right_peak]
            
            # Head should be higher than shoulders, shoulders should be similar height
            if (head_height > left_height and head_height > right_height and
                abs(left_height - right_height) / max(left_height, right_height) < 0.05):
                pattern_signals.iloc[right_peak] = 1
        
        return pattern_signals
    
    def _detect_wedge_pattern(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Simplified wedge pattern detection based on converging trend lines.
        
        Args:
            data: DataFrame with OHLC data
            window: Window size for trend analysis
            
        Returns:
            Series with wedge pattern signals
        """
        pattern_signals = pd.Series(0, index=data.index)
        
        for i in range(window, len(data)):
            # Get recent data window
            recent_data = data.iloc[i-window:i]
            
            # Calculate trend slopes for highs and lows
            x = np.arange(len(recent_data))
            high_slope = np.polyfit(x, recent_data['high'], 1)[0]
            low_slope = np.polyfit(x, recent_data['low'], 1)[0]
            
            # Wedge pattern: converging trend lines (slopes have opposite signs and are converging)
            if high_slope < 0 and low_slope > 0:  # Descending wedge
                pattern_signals.iloc[i] = 1
            elif high_slope > 0 and low_slope < 0:  # Ascending wedge
                pattern_signals.iloc[i] = -1
        
        return pattern_signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Series of closing prices
            period: Period for RSI calculation
            
        Returns:
            Series with RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).
        
        Args:
            prices: Series of closing prices
            period: Period for EMA calculation
            
        Returns:
            Series with EMA values
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Series of closing prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of closing prices
            period: Period for ATR calculation
            
        Returns:
            Series with ATR values
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr