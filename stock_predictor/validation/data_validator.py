"""
Comprehensive data validation for the Stock Direction Predictor system.
Validates data types, ranges, and business rules for all input data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, date
import re

from ..utils.exceptions import DataValidationError, validate_required_fields, validate_data_types, validate_numeric_range
from ..utils.logging_config import get_logger
from ..interfaces import StockData, TechnicalIndicators, CandlestickPattern, ModelConfiguration


class DataValidator:
    """Comprehensive data validator for all system components."""
    
    def __init__(self):
        self.logger = get_logger("DataValidator")
        
        # Define validation schemas
        self.stock_data_schema = {
            'required_columns': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'optional_columns': ['Adj Close', 'Dividends', 'Stock Splits'],
            'numeric_columns': ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'],
            'price_columns': ['Open', 'High', 'Low', 'Close', 'Adj Close'],
            'volume_columns': ['Volume']
        }
        
        self.technical_indicators_ranges = {
            'rsi': (0.0, 100.0),
            'macd': (-np.inf, np.inf),
            'macd_signal': (-np.inf, np.inf),
            'ema_20': (0.0, np.inf),
            'ema_50': (0.0, np.inf),
            'ema_200': (0.0, np.inf),
            'atr': (0.0, np.inf),
            'sma': (0.0, np.inf)
        }
        
        self.valid_stock_symbols = {
            'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'TSLA', 'NFLX',
            'ADBE', 'CRM', 'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'TXN'
        }
        
        self.valid_pattern_lengths = [3, 5, 7, 14]
        self.valid_signals = [-1, 0, 1]
        self.valid_model_types = ['xgboost', 'random_forest', 'svm', 'neural_network']
    
    def validate_stock_data(self, data: pd.DataFrame, symbol: Optional[str] = None) -> bool:
        """
        Validate stock OHLC data for completeness and correctness.
        
        Args:
            data: DataFrame containing stock data
            symbol: Optional stock symbol for additional validation
            
        Returns:
            bool: True if data is valid
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            self.logger.debug(f"Validating stock data for symbol: {symbol}")
            
            # Check if DataFrame is empty
            if data.empty:
                raise DataValidationError(
                    "Stock data is empty",
                    error_code="EMPTY_DATA",
                    details={"symbol": symbol}
                )
            
            # Validate required columns
            missing_columns = [col for col in self.stock_data_schema['required_columns'] 
                             if col not in data.columns]
            if missing_columns:
                raise DataValidationError(
                    f"Missing required columns: {missing_columns}",
                    error_code="MISSING_COLUMNS",
                    details={"missing_columns": missing_columns, "symbol": symbol}
                )
            
            # Validate data types (all should be numeric)
            for col in self.stock_data_schema['numeric_columns']:
                if col in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[col]):
                        raise DataValidationError(
                            f"Column {col} must be numeric",
                            error_code="INVALID_DATA_TYPE",
                            details={"column": col, "dtype": str(data[col].dtype), "symbol": symbol}
                        )
            
            # Validate OHLC relationships
            self._validate_ohlc_relationships(data, symbol)
            
            # Validate price ranges (must be positive)
            for col in self.stock_data_schema['price_columns']:
                if col in data.columns:
                    negative_prices = data[data[col] <= 0]
                    if not negative_prices.empty:
                        raise DataValidationError(
                            f"Negative or zero prices found in {col}",
                            error_code="INVALID_PRICE_RANGE",
                            details={"column": col, "negative_count": len(negative_prices), "symbol": symbol}
                        )
            
            # Validate volume (must be non-negative)
            if 'Volume' in data.columns:
                negative_volume = data[data['Volume'] < 0]
                if not negative_volume.empty:
                    raise DataValidationError(
                        "Negative volume found",
                        error_code="INVALID_VOLUME",
                        details={"negative_count": len(negative_volume), "symbol": symbol}
                    )
            
            # Validate date index
            if not isinstance(data.index, pd.DatetimeIndex):
                raise DataValidationError(
                    "Data must have DatetimeIndex",
                    error_code="INVALID_INDEX_TYPE",
                    details={"index_type": str(type(data.index)), "symbol": symbol}
                )
            
            # Check for duplicate dates
            if data.index.duplicated().any():
                duplicate_count = data.index.duplicated().sum()
                raise DataValidationError(
                    f"Duplicate dates found: {duplicate_count}",
                    error_code="DUPLICATE_DATES",
                    details={"duplicate_count": duplicate_count, "symbol": symbol}
                )
            
            # Validate symbol format if provided
            if symbol and not self._is_valid_symbol_format(symbol):
                raise DataValidationError(
                    f"Invalid symbol format: {symbol}",
                    error_code="INVALID_SYMBOL_FORMAT",
                    details={"symbol": symbol}
                )
            
            self.logger.debug(f"Stock data validation passed for symbol: {symbol}")
            return True
            
        except DataValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during stock data validation: {str(e)}")
            raise DataValidationError(
                f"Validation failed: {str(e)}",
                error_code="VALIDATION_ERROR",
                details={"symbol": symbol, "error": str(e)}
            )
    
    def validate_technical_indicators(self, indicators: Union[Dict[str, float], TechnicalIndicators]) -> bool:
        """
        Validate technical indicators are within expected ranges.
        
        Args:
            indicators: Technical indicators data
            
        Returns:
            bool: True if indicators are valid
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            self.logger.debug("Validating technical indicators")
            
            # Convert to dict if TechnicalIndicators object
            if isinstance(indicators, TechnicalIndicators):
                indicator_dict = indicators.__dict__
            else:
                indicator_dict = indicators
            
            # Validate each indicator range
            for indicator, value in indicator_dict.items():
                if indicator in self.technical_indicators_ranges:
                    min_val, max_val = self.technical_indicators_ranges[indicator]
                    
                    # Check for NaN values
                    if pd.isna(value):
                        raise DataValidationError(
                            f"Technical indicator {indicator} is NaN",
                            error_code="NAN_INDICATOR",
                            details={"indicator": indicator}
                        )
                    
                    # Check range
                    if not (min_val <= value <= max_val):
                        raise DataValidationError(
                            f"Technical indicator {indicator} out of range: {value}",
                            error_code="INDICATOR_OUT_OF_RANGE",
                            details={
                                "indicator": indicator,
                                "value": value,
                                "min_value": min_val,
                                "max_value": max_val
                            }
                        )
            
            # Validate EMA ordering (EMA20 should be more responsive than EMA50, EMA200)
            if all(key in indicator_dict for key in ['ema_20', 'ema_50', 'ema_200']):
                ema_20 = indicator_dict['ema_20']
                ema_50 = indicator_dict['ema_50']
                ema_200 = indicator_dict['ema_200']
                
                # In trending markets, shorter EMAs should be closer to current price
                # We'll just check they're all positive and reasonable
                if not (ema_20 > 0 and ema_50 > 0 and ema_200 > 0):
                    raise DataValidationError(
                        "All EMA values must be positive",
                        error_code="INVALID_EMA_VALUES",
                        details={"ema_20": ema_20, "ema_50": ema_50, "ema_200": ema_200}
                    )
            
            self.logger.debug("Technical indicators validation passed")
            return True
            
        except DataValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during technical indicators validation: {str(e)}")
            raise DataValidationError(
                f"Technical indicators validation failed: {str(e)}",
                error_code="VALIDATION_ERROR",
                details={"error": str(e)}
            )
    
    def validate_candlestick_pattern(self, pattern: Union[Dict[str, Any], CandlestickPattern]) -> bool:
        """
        Validate candlestick pattern data.
        
        Args:
            pattern: Candlestick pattern data
            
        Returns:
            bool: True if pattern is valid
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            self.logger.debug("Validating candlestick pattern")
            
            # Convert to dict if CandlestickPattern object
            if isinstance(pattern, CandlestickPattern):
                pattern_dict = pattern.__dict__
            else:
                pattern_dict = pattern
            
            # Validate required fields
            required_fields = ['pattern_length', 'signal', 'confidence', 'pattern_type']
            validate_required_fields(pattern_dict, required_fields, "candlestick pattern")
            
            # Validate pattern length
            pattern_length = pattern_dict['pattern_length']
            if pattern_length not in self.valid_pattern_lengths:
                raise DataValidationError(
                    f"Invalid pattern length: {pattern_length}",
                    error_code="INVALID_PATTERN_LENGTH",
                    details={"pattern_length": pattern_length, "valid_lengths": self.valid_pattern_lengths}
                )
            
            # Validate signal
            signal = pattern_dict['signal']
            if signal not in self.valid_signals:
                raise DataValidationError(
                    f"Invalid signal: {signal}",
                    error_code="INVALID_SIGNAL",
                    details={"signal": signal, "valid_signals": self.valid_signals}
                )
            
            # Validate confidence
            confidence = pattern_dict['confidence']
            validate_numeric_range(confidence, 0.0, 1.0, "confidence")
            
            # Validate pattern type
            pattern_type = pattern_dict['pattern_type']
            if not isinstance(pattern_type, str) or not pattern_type.strip():
                raise DataValidationError(
                    "Pattern type must be a non-empty string",
                    error_code="INVALID_PATTERN_TYPE",
                    details={"pattern_type": pattern_type}
                )
            
            self.logger.debug("Candlestick pattern validation passed")
            return True
            
        except DataValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during candlestick pattern validation: {str(e)}")
            raise DataValidationError(
                f"Candlestick pattern validation failed: {str(e)}",
                error_code="VALIDATION_ERROR",
                details={"error": str(e)}
            )
    
    def validate_model_configuration(self, config: Union[Dict[str, Any], ModelConfiguration]) -> bool:
        """
        Validate model configuration data.
        
        Args:
            config: Model configuration data
            
        Returns:
            bool: True if configuration is valid
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            self.logger.debug("Validating model configuration")
            
            # Convert to dict if ModelConfiguration object
            if isinstance(config, ModelConfiguration):
                config_dict = config.__dict__
            else:
                config_dict = config
            
            # Validate required fields
            required_fields = ['model_type', 'pattern_length', 'hyperparameters', 'feature_set', 'version']
            validate_required_fields(config_dict, required_fields, "model configuration")
            
            # Validate model type
            model_type = config_dict['model_type']
            if model_type not in self.valid_model_types:
                raise DataValidationError(
                    f"Invalid model type: {model_type}",
                    error_code="INVALID_MODEL_TYPE",
                    details={"model_type": model_type, "valid_types": self.valid_model_types}
                )
            
            # Validate pattern length
            pattern_length = config_dict['pattern_length']
            if pattern_length not in self.valid_pattern_lengths:
                raise DataValidationError(
                    f"Invalid pattern length: {pattern_length}",
                    error_code="INVALID_PATTERN_LENGTH",
                    details={"pattern_length": pattern_length, "valid_lengths": self.valid_pattern_lengths}
                )
            
            # Validate hyperparameters
            hyperparameters = config_dict['hyperparameters']
            if not isinstance(hyperparameters, dict):
                raise DataValidationError(
                    "Hyperparameters must be a dictionary",
                    error_code="INVALID_HYPERPARAMETERS_TYPE",
                    details={"type": type(hyperparameters)}
                )
            
            # Validate feature set
            feature_set = config_dict['feature_set']
            if not isinstance(feature_set, list) or not feature_set:
                raise DataValidationError(
                    "Feature set must be a non-empty list",
                    error_code="INVALID_FEATURE_SET",
                    details={"feature_set": feature_set}
                )
            
            # Validate version
            version = config_dict['version']
            if not isinstance(version, str) or not version.strip():
                raise DataValidationError(
                    "Version must be a non-empty string",
                    error_code="INVALID_VERSION",
                    details={"version": version}
                )
            
            self.logger.debug("Model configuration validation passed")
            return True
            
        except DataValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during model configuration validation: {str(e)}")
            raise DataValidationError(
                f"Model configuration validation failed: {str(e)}",
                error_code="VALIDATION_ERROR",
                details={"error": str(e)}
            )
    
    def validate_prediction_input(self, features: np.ndarray, feature_names: Optional[List[str]] = None) -> bool:
        """
        Validate prediction input features.
        
        Args:
            features: Feature array for prediction
            feature_names: Optional list of feature names
            
        Returns:
            bool: True if input is valid
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            self.logger.debug("Validating prediction input")
            
            # Check if features is numpy array
            if not isinstance(features, np.ndarray):
                raise DataValidationError(
                    "Features must be a numpy array",
                    error_code="INVALID_FEATURES_TYPE",
                    details={"type": type(features)}
                )
            
            # Check for empty array
            if features.size == 0:
                raise DataValidationError(
                    "Features array is empty",
                    error_code="EMPTY_FEATURES",
                    details={"shape": features.shape}
                )
            
            # Check for NaN or infinite values
            if np.isnan(features).any():
                nan_count = np.isnan(features).sum()
                raise DataValidationError(
                    f"Features contain NaN values: {nan_count}",
                    error_code="NAN_FEATURES",
                    details={"nan_count": int(nan_count), "shape": features.shape}
                )
            
            if np.isinf(features).any():
                inf_count = np.isinf(features).sum()
                raise DataValidationError(
                    f"Features contain infinite values: {inf_count}",
                    error_code="INF_FEATURES",
                    details={"inf_count": int(inf_count), "shape": features.shape}
                )
            
            # Validate feature names if provided
            if feature_names is not None:
                if not isinstance(feature_names, list):
                    raise DataValidationError(
                        "Feature names must be a list",
                        error_code="INVALID_FEATURE_NAMES_TYPE",
                        details={"type": type(feature_names)}
                    )
                
                # Check if number of features matches
                if len(feature_names) != features.shape[-1]:
                    raise DataValidationError(
                        "Number of feature names doesn't match features array",
                        error_code="FEATURE_NAME_MISMATCH",
                        details={
                            "feature_names_count": len(feature_names),
                            "features_shape": features.shape
                        }
                    )
            
            self.logger.debug("Prediction input validation passed")
            return True
            
        except DataValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during prediction input validation: {str(e)}")
            raise DataValidationError(
                f"Prediction input validation failed: {str(e)}",
                error_code="VALIDATION_ERROR",
                details={"error": str(e)}
            )
    
    def _validate_ohlc_relationships(self, data: pd.DataFrame, symbol: Optional[str] = None) -> None:
        """
        Validate OHLC price relationships.
        
        Args:
            data: DataFrame with OHLC data
            symbol: Optional symbol for error context
            
        Raises:
            DataValidationError: If OHLC relationships are invalid
        """
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_cols):
            return  # Skip if not all OHLC columns present
        
        # High should be >= max(Open, Close)
        max_oc = np.maximum(data['Open'], data['Close'])
        invalid_high = data['High'] < max_oc
        if invalid_high.any():
            invalid_count = invalid_high.sum()
            raise DataValidationError(
                f"High price less than max(Open, Close) in {invalid_count} rows",
                error_code="INVALID_HIGH_PRICE",
                details={"invalid_count": int(invalid_count), "symbol": symbol}
            )
        
        # Low should be <= min(Open, Close)
        min_oc = np.minimum(data['Open'], data['Close'])
        invalid_low = data['Low'] > min_oc
        if invalid_low.any():
            invalid_count = invalid_low.sum()
            raise DataValidationError(
                f"Low price greater than min(Open, Close) in {invalid_count} rows",
                error_code="INVALID_LOW_PRICE",
                details={"invalid_count": int(invalid_count), "symbol": symbol}
            )
    
    def _is_valid_symbol_format(self, symbol: str) -> bool:
        """
        Validate stock symbol format.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            bool: True if symbol format is valid
        """
        # Basic symbol format: 1-5 uppercase letters, optionally followed by a dot and more letters
        pattern = r'^[A-Z]{1,5}(\.[A-Z]{1,3})?$'
        return bool(re.match(pattern, symbol))
    
    def validate_date_range(self, start_date: Union[str, date, datetime], 
                           end_date: Union[str, date, datetime]) -> bool:
        """
        Validate date range for data collection.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            bool: True if date range is valid
            
        Raises:
            DataValidationError: If date range is invalid
        """
        try:
            self.logger.debug("Validating date range")
            
            # Convert to datetime objects
            if isinstance(start_date, str):
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            elif isinstance(start_date, date):
                start_dt = datetime.combine(start_date, datetime.min.time())
            else:
                start_dt = start_date
            
            if isinstance(end_date, str):
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            elif isinstance(end_date, date):
                end_dt = datetime.combine(end_date, datetime.min.time())
            else:
                end_dt = end_date
            
            # Validate start date is before end date
            if start_dt >= end_dt:
                raise DataValidationError(
                    "Start date must be before end date",
                    error_code="INVALID_DATE_RANGE",
                    details={"start_date": str(start_dt), "end_date": str(end_dt)}
                )
            
            # Validate dates are not in the future (with some tolerance)
            now = datetime.now()
            if start_dt > now:
                raise DataValidationError(
                    "Start date cannot be in the future",
                    error_code="FUTURE_START_DATE",
                    details={"start_date": str(start_dt), "current_date": str(now)}
                )
            
            # Validate reasonable date range (not too far in the past)
            min_date = datetime(1990, 1, 1)
            if start_dt < min_date:
                raise DataValidationError(
                    f"Start date too far in the past (before {min_date.date()})",
                    error_code="START_DATE_TOO_OLD",
                    details={"start_date": str(start_dt), "min_date": str(min_date)}
                )
            
            self.logger.debug("Date range validation passed")
            return True
            
        except DataValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during date range validation: {str(e)}")
            raise DataValidationError(
                f"Date range validation failed: {str(e)}",
                error_code="VALIDATION_ERROR",
                details={"error": str(e)}
            )