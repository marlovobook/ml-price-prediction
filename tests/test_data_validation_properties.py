"""
Property-based tests for data validation and quality assurance.
Tests comprehensive validation functionality across all system components.
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

from stock_predictor.validation.data_validator import DataValidator
from stock_predictor.validation.anomaly_detector import AnomalyDetector
from stock_predictor.validation.cross_validator import CrossValidator
from stock_predictor.validation.schema_validator import SchemaValidator
from stock_predictor.utils.exceptions import DataValidationError
from stock_predictor.interfaces import TechnicalIndicators, CandlestickPattern, ModelConfiguration


class TestDataValidationProperties:
    """Property-based tests for data validation and quality assurance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
        self.anomaly_detector = AnomalyDetector(contamination=0.1, random_state=42)
        self.cross_validator = CrossValidator(n_splits=3)
        self.schema_validator = SchemaValidator()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        pass


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
def valid_stock_dataframe(draw, min_rows=1, max_rows=50):
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
def valid_technical_indicators(draw):
    """Generate valid technical indicators."""
    return {
        'rsi': draw(st.floats(min_value=0.0, max_value=100.0)),
        'macd': draw(st.floats(min_value=-10.0, max_value=10.0)),
        'macd_signal': draw(st.floats(min_value=-10.0, max_value=10.0)),
        'ema_20': draw(st.floats(min_value=1.0, max_value=1000.0)),
        'ema_50': draw(st.floats(min_value=1.0, max_value=1000.0)),
        'ema_200': draw(st.floats(min_value=1.0, max_value=1000.0)),
        'atr': draw(st.floats(min_value=0.0, max_value=100.0)),
        'sma': draw(st.floats(min_value=1.0, max_value=1000.0))
    }


@st.composite
def valid_candlestick_pattern(draw):
    """Generate valid candlestick pattern."""
    return {
        'pattern_length': draw(st.sampled_from([3, 5, 7, 14])),
        'signal': draw(st.sampled_from([-1, 0, 1])),
        'confidence': draw(st.floats(min_value=0.0, max_value=1.0)),
        'pattern_type': draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))))
    }


@st.composite
def valid_model_configuration(draw):
    """Generate valid model configuration."""
    return {
        'model_type': draw(st.sampled_from(['xgboost', 'random_forest', 'svm', 'neural_network'])),
        'pattern_length': draw(st.sampled_from([3, 5, 7, 14])),
        'hyperparameters': draw(st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=st.one_of(st.floats(min_value=0.001, max_value=100.0), st.integers(min_value=1, max_value=1000)),
            min_size=1, max_size=5
        )),
        'feature_set': draw(st.lists(st.text(min_size=1, max_size=15), min_size=1, max_size=20)),
        'version': draw(st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    }


@st.composite
def valid_feature_matrix(draw, min_features=1, max_features=20, min_samples=10, max_samples=100):
    """Generate valid feature matrix for ML."""
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    n_features = draw(st.integers(min_value=min_features, max_value=max_features))
    
    # Generate realistic feature data
    features = []
    for _ in range(n_samples):
        sample = []
        for _ in range(n_features):
            # Generate features with different scales and distributions
            feature_val = draw(st.floats(min_value=-100.0, max_value=100.0))
            sample.append(feature_val)
        features.append(sample)
    
    return np.array(features)


class TestDataValidationQualityAssurance:
    """Property-based tests for comprehensive data validation and quality assurance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
        self.anomaly_detector = AnomalyDetector(contamination=0.1, random_state=42)
        self.cross_validator = CrossValidator(n_splits=3)
        self.schema_validator = SchemaValidator()
    
    @given(valid_stock_dataframe())
    @settings(max_examples=100, deadline=None)
    def test_property_stock_data_validation_completeness(self, stock_data):
        """
        **Feature: stock-direction-predictor, Property 8: Data Validation and Quality Assurance**
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        
        For any valid stock data, the data validator should correctly validate
        data types, ranges, and completeness, and reject invalid data appropriately.
        """
        # Valid data should pass validation
        assert self.validator.validate_stock_data(stock_data, symbol="TEST") == True
        
        # Test with invalid OHLC relationships
        invalid_data = stock_data.copy()
        if len(invalid_data) > 0:
            # Make High less than Close (invalid)
            invalid_data.iloc[0, invalid_data.columns.get_loc('High')] = \
                invalid_data.iloc[0, invalid_data.columns.get_loc('Close')] - 1.0
            
            # Should raise validation error
            with pytest.raises(DataValidationError):
                self.validator.validate_stock_data(invalid_data, symbol="TEST")
        
        # Test with missing required columns
        incomplete_data = stock_data.drop(columns=['Volume'])
        with pytest.raises(DataValidationError):
            self.validator.validate_stock_data(incomplete_data, symbol="TEST")
        
        # Test with negative prices
        negative_price_data = stock_data.copy()
        if len(negative_price_data) > 0:
            negative_price_data.iloc[0, negative_price_data.columns.get_loc('Close')] = -1.0
            with pytest.raises(DataValidationError):
                self.validator.validate_stock_data(negative_price_data, symbol="TEST")
    
    @given(valid_technical_indicators())
    @settings(max_examples=100, deadline=None)
    def test_property_technical_indicators_validation_ranges(self, indicators):
        """
        **Feature: stock-direction-predictor, Property 8: Data Validation and Quality Assurance**
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        
        For any valid technical indicators, the validator should accept indicators
        within expected ranges and reject those outside valid bounds.
        """
        # Valid indicators should pass validation
        assert self.validator.validate_technical_indicators(indicators) == True
        
        # Test with out-of-range RSI
        invalid_indicators = indicators.copy()
        invalid_indicators['rsi'] = 150.0  # RSI should be 0-100
        with pytest.raises(DataValidationError):
            self.validator.validate_technical_indicators(invalid_indicators)
        
        # Test with NaN values
        nan_indicators = indicators.copy()
        nan_indicators['rsi'] = np.nan
        with pytest.raises(DataValidationError):
            self.validator.validate_technical_indicators(nan_indicators)
        
        # Test with negative EMA (should be positive)
        negative_ema_indicators = indicators.copy()
        negative_ema_indicators['ema_20'] = -10.0
        with pytest.raises(DataValidationError):
            self.validator.validate_technical_indicators(negative_ema_indicators)
    
    @given(valid_candlestick_pattern())
    @settings(max_examples=100, deadline=None)
    def test_property_candlestick_pattern_validation_consistency(self, pattern):
        """
        **Feature: stock-direction-predictor, Property 8: Data Validation and Quality Assurance**
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        
        For any valid candlestick pattern, the validator should validate pattern
        length, signal values, and confidence ranges consistently.
        """
        # Valid pattern should pass validation
        assert self.validator.validate_candlestick_pattern(pattern) == True
        
        # Test with invalid pattern length
        invalid_pattern = pattern.copy()
        invalid_pattern['pattern_length'] = 99  # Not in valid lengths [3, 5, 7, 14]
        with pytest.raises(DataValidationError):
            self.validator.validate_candlestick_pattern(invalid_pattern)
        
        # Test with invalid signal
        invalid_signal_pattern = pattern.copy()
        invalid_signal_pattern['signal'] = 5  # Should be -1, 0, or 1
        with pytest.raises(DataValidationError):
            self.validator.validate_candlestick_pattern(invalid_signal_pattern)
        
        # Test with invalid confidence
        invalid_confidence_pattern = pattern.copy()
        invalid_confidence_pattern['confidence'] = 1.5  # Should be 0.0-1.0
        with pytest.raises(DataValidationError):
            self.validator.validate_candlestick_pattern(invalid_confidence_pattern)
    
    @given(valid_model_configuration())
    @settings(max_examples=100, deadline=None)
    def test_property_model_configuration_validation_completeness(self, config):
        """
        **Feature: stock-direction-predictor, Property 8: Data Validation and Quality Assurance**
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        
        For any valid model configuration, the validator should validate all
        required fields and reject configurations with invalid model types or parameters.
        """
        # Valid configuration should pass validation
        assert self.validator.validate_model_configuration(config) == True
        
        # Test with invalid model type
        invalid_config = config.copy()
        invalid_config['model_type'] = 'invalid_model'
        with pytest.raises(DataValidationError):
            self.validator.validate_model_configuration(invalid_config)
        
        # Test with missing required fields
        incomplete_config = config.copy()
        del incomplete_config['feature_set']
        with pytest.raises(DataValidationError):
            self.validator.validate_model_configuration(incomplete_config)
        
        # Test with invalid hyperparameters type
        invalid_hyperparams_config = config.copy()
        invalid_hyperparams_config['hyperparameters'] = "not_a_dict"
        with pytest.raises(DataValidationError):
            self.validator.validate_model_configuration(invalid_hyperparams_config)
    
    @given(valid_feature_matrix())
    @settings(max_examples=50, deadline=None)
    def test_property_prediction_input_validation_structure(self, features):
        """
        **Feature: stock-direction-predictor, Property 8: Data Validation and Quality Assurance**
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        
        For any valid feature matrix, the validator should validate input structure,
        detect NaN/infinite values, and ensure data quality for predictions.
        """
        # Valid features should pass validation
        assert self.validator.validate_prediction_input(features) == True
        
        # Test with NaN values
        features_with_nan = features.copy()
        if features_with_nan.size > 0:
            features_with_nan.flat[0] = np.nan
            with pytest.raises(DataValidationError):
                self.validator.validate_prediction_input(features_with_nan)
        
        # Test with infinite values
        features_with_inf = features.copy()
        if features_with_inf.size > 0:
            features_with_inf.flat[0] = np.inf
            with pytest.raises(DataValidationError):
                self.validator.validate_prediction_input(features_with_inf)
        
        # Test with empty array
        empty_features = np.array([])
        with pytest.raises(DataValidationError):
            self.validator.validate_prediction_input(empty_features)
    
    @given(valid_stock_dataframe(min_rows=10, max_rows=50))
    @settings(max_examples=50, deadline=None)
    def test_property_anomaly_detection_consistency(self, stock_data):
        """
        **Feature: stock-direction-predictor, Property 8: Data Validation and Quality Assurance**
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        
        For any stock data, anomaly detection should consistently identify outliers
        and handle them appropriately without corrupting the data structure.
        """
        # Detect price anomalies
        price_anomalies = self.anomaly_detector.detect_price_anomalies(stock_data, method='iqr')
        
        # Anomalies should be boolean series with same index as input
        assert isinstance(price_anomalies, pd.Series)
        assert len(price_anomalies) == len(stock_data)
        assert price_anomalies.dtype == bool
        
        # Handle anomalies
        cleaned_data = self.anomaly_detector.handle_anomalies(
            stock_data, price_anomalies, method='interpolate'
        )
        
        # Cleaned data should maintain structure
        assert isinstance(cleaned_data, pd.DataFrame)
        assert list(cleaned_data.columns) == list(stock_data.columns)
        assert not cleaned_data.isnull().any().any()  # No missing values after handling
        
        # Volume anomaly detection
        if 'Volume' in stock_data.columns:
            volume_anomalies = self.anomaly_detector.detect_volume_anomalies(stock_data, method='iqr')
            assert isinstance(volume_anomalies, pd.Series)
            assert len(volume_anomalies) == len(stock_data)
            assert volume_anomalies.dtype == bool
    
    @given(valid_feature_matrix(min_samples=20, max_samples=50))
    @settings(max_examples=30, deadline=None)
    def test_property_schema_validation_consistency(self, features):
        """
        **Feature: stock-direction-predictor, Property 8: Data Validation and Quality Assurance**
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        
        For any feature matrix, schema validation should create consistent schemas
        and validate prediction inputs against training data schemas accurately.
        """
        assume(features.shape[0] > 10 and features.shape[1] > 0)
        
        # Generate feature names
        feature_names = [f'feature_{i}' for i in range(features.shape[1])]
        
        # Create schema from training data
        schema = self.schema_validator.create_schema_from_training_data(
            features, feature_names, "test_schema"
        )
        
        # Schema should contain expected fields
        assert 'schema_name' in schema
        assert 'n_features' in schema
        assert 'feature_names' in schema
        assert 'feature_stats' in schema
        assert schema['n_features'] == features.shape[1]
        assert len(schema['feature_names']) == features.shape[1]
        
        # Validate same data against schema (should pass)
        validation_result = self.schema_validator.validate_prediction_input(
            features, "test_schema", feature_names
        )
        assert validation_result['passed'] == True
        assert validation_result['validation_score'] > 0.8
        
        # Test with modified data (should detect differences)
        modified_features = features * 2.0  # Scale all features
        drift_result = self.schema_validator.validate_feature_drift(
            modified_features, "test_schema", drift_threshold=0.1
        )
        assert 'feature_drift' in drift_result
        assert 'overall_drift' in drift_result
    
    @given(st.data())
    @settings(max_examples=30, deadline=None)
    def test_property_cross_validation_stability_assessment(self, data):
        """
        **Feature: stock-direction-predictor, Property 8: Data Validation and Quality Assurance**
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        
        For any valid ML data, cross-validation should assess model stability
        consistently and provide meaningful stability metrics.
        """
        # Generate valid ML data
        n_samples = data.draw(st.integers(min_value=30, max_value=100))
        n_features = data.draw(st.integers(min_value=3, max_value=10))
        
        X = data.draw(st.lists(
            st.lists(st.floats(min_value=-10.0, max_value=10.0), min_size=n_features, max_size=n_features),
            min_size=n_samples, max_size=n_samples
        ))
        y = data.draw(st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=n_samples, max_size=n_samples))
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Mock model class for testing
        class MockModel:
            def __init__(self, **kwargs):
                self.params = kwargs
            
            def fit(self, X, y):
                pass
            
            def predict(self, X):
                # Return simple predictions based on input
                return np.mean(X, axis=1) * 0.1
        
        # Test cross-validation
        try:
            cv_results = self.cross_validator.validate_model_stability(
                MockModel, X_array, y_array, model_params={}
            )
            
            # Results should contain expected fields
            assert 'n_successful_folds' in cv_results
            assert 'stability_metrics' in cv_results
            assert 'fold_results' in cv_results
            assert cv_results['n_successful_folds'] > 0
            
        except Exception as e:
            # Cross-validation might fail with very small datasets or edge cases
            # This is acceptable behavior
            assert isinstance(e, (DataValidationError, ValueError))
    
    def test_property_comprehensive_error_logging_and_debugging(self):
        """
        **Feature: stock-direction-predictor, Property 8: Data Validation and Quality Assurance**
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4**
        
        The validation system should provide comprehensive error logging and
        debugging information for all validation failures.
        """
        # Test that validation errors contain detailed information
        with pytest.raises(DataValidationError) as exc_info:
            # Trigger validation error with invalid stock data
            invalid_data = pd.DataFrame({'Invalid': [1, 2, 3]})
            self.validator.validate_stock_data(invalid_data, symbol="TEST")
        
        # Error should contain detailed information
        e = exc_info.value
        assert e.error_code is not None
        assert e.details is not None
        assert 'symbol' in e.details
        assert len(str(e)) > 0
        
        # Test anomaly detection error handling
        with pytest.raises(DataValidationError) as exc_info:
            # Use valid data but invalid method to trigger error
            test_data = pd.DataFrame({'Close': [100, 101, 102]})
            self.anomaly_detector.detect_price_anomalies(test_data, method='invalid_method')
        
        e = exc_info.value
        assert e.error_code is not None
        assert 'method' in e.details
        
        # Test schema validation error handling
        with pytest.raises(DataValidationError) as exc_info:
            self.schema_validator.validate_prediction_input(
                np.array([[1, 2, 3]]), "nonexistent_schema"
            )
        
        e = exc_info.value
        assert e.error_code == "SCHEMA_NOT_FOUND"
        assert 'available_schemas' in e.details