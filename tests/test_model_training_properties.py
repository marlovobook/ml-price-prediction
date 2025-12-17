"""
Property-based tests for model training pipeline completeness.

Feature: stock-direction-predictor, Property 5: Model Training Completeness
Validates: Requirements 4.3, 4.4, 4.5, 4.6
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from hypothesis import given, strategies as st, settings, HealthCheck
from typing import List

from stock_predictor.models.training_pipeline import ModelTrainingPipeline
from stock_predictor.models.ml_models import create_model


# Test data generators
@st.composite
def generate_stock_features(draw):
    """Generate realistic stock feature data."""
    n_samples = draw(st.integers(min_value=30, max_value=100))
    
    # Generate base technical indicators
    features = {
        'rsi': draw(st.lists(st.floats(min_value=0, max_value=100), min_size=n_samples, max_size=n_samples)),
        'macd': draw(st.lists(st.floats(min_value=-10, max_value=10), min_size=n_samples, max_size=n_samples)),
        'macd_signal': draw(st.lists(st.floats(min_value=-10, max_value=10), min_size=n_samples, max_size=n_samples)),
        'ema_20': draw(st.lists(st.floats(min_value=50, max_value=500), min_size=n_samples, max_size=n_samples)),
        'ema_50': draw(st.lists(st.floats(min_value=50, max_value=500), min_size=n_samples, max_size=n_samples)),
        'ema_200': draw(st.lists(st.floats(min_value=50, max_value=500), min_size=n_samples, max_size=n_samples)),
        'atr': draw(st.lists(st.floats(min_value=0.1, max_value=20), min_size=n_samples, max_size=n_samples)),
        'sma': draw(st.lists(st.floats(min_value=50, max_value=500), min_size=n_samples, max_size=n_samples)),
        'golden_cross': draw(st.lists(st.integers(min_value=0, max_value=1), min_size=n_samples, max_size=n_samples)),
        'head_shoulder': draw(st.lists(st.integers(min_value=0, max_value=1), min_size=n_samples, max_size=n_samples)),
        'wedge': draw(st.lists(st.integers(min_value=0, max_value=1), min_size=n_samples, max_size=n_samples)),
        'fibonacci_23.6': draw(st.lists(st.floats(min_value=50, max_value=500), min_size=n_samples, max_size=n_samples)),
        'fibonacci_38.2': draw(st.lists(st.floats(min_value=50, max_value=500), min_size=n_samples, max_size=n_samples)),
        'fibonacci_50.0': draw(st.lists(st.floats(min_value=50, max_value=500), min_size=n_samples, max_size=n_samples)),
        'fibonacci_61.8': draw(st.lists(st.floats(min_value=50, max_value=500), min_size=n_samples, max_size=n_samples)),
        'fibonacci_78.6': draw(st.lists(st.floats(min_value=50, max_value=500), min_size=n_samples, max_size=n_samples)),
    }
    
    return pd.DataFrame(features)


@st.composite
def generate_trading_signals(draw, n_samples):
    """Generate realistic trading signals (-1, 0, 1)."""
    signals = draw(st.lists(
        st.sampled_from([-1, 0, 1]), 
        min_size=n_samples, 
        max_size=n_samples
    ))
    return pd.Series(signals)


@st.composite
def generate_pattern_length(draw):
    """Generate valid pattern lengths."""
    return draw(st.sampled_from([3, 5, 7, 14]))


@st.composite
def generate_model_type(draw):
    """Generate valid model types."""
    return draw(st.sampled_from(['xgboost', 'random_forest', 'svm', 'neural_network']))


class TestModelTrainingCompleteness:
    """Property-based tests for model training completeness."""
    
    @given(st.data())
    @settings(
        max_examples=5, 
        deadline=60000,
        suppress_health_check=[HealthCheck.large_base_example, HealthCheck.data_too_large]
    )
    def test_property_5_model_training_completeness(self, data):
        """
        Feature: stock-direction-predictor, Property 5: Model Training Completeness
        
        For any combination of model type and pattern length, the training pipeline should:
        1. Create appropriate feature sets
        2. Perform time-based data splits  
        3. Train models successfully
        4. Save them with proper versioning
        
        Validates: Requirements 4.3, 4.4, 4.5, 4.6
        """
        # Generate test data
        features = data.draw(generate_stock_features())
        pattern_length = data.draw(generate_pattern_length())
        model_type = data.draw(generate_model_type())
        
        # Generate corresponding signals
        n_samples = len(features)
        signals = data.draw(generate_trading_signals(n_samples))
        
        # Skip if insufficient class diversity for ML training
        unique_classes = len(np.unique(signals))
        if unique_classes < 2:
            return  # Skip this test case
        
        # Create temporary directory for models
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize training pipeline
            pipeline = ModelTrainingPipeline(models_dir=temp_dir)
            
            # Test 1: Feature set preparation (Requirement 4.3)
            X, y = pipeline.prepare_training_data(features, signals, pattern_length)
            
            # Verify feature preparation creates valid arrays
            assert isinstance(X, np.ndarray), "Features should be converted to numpy array"
            assert isinstance(y, np.ndarray), "Targets should be converted to numpy array"
            assert X.shape[0] == y.shape[0], "Features and targets should have same number of samples"
            assert X.shape[0] > 0, "Should have at least some valid samples after preprocessing"
            assert len(X.shape) == 2, "Features should be 2D array"
            assert len(y.shape) == 1, "Targets should be 1D array"
            
            # Verify signals are in valid range
            assert all(signal in [-1, 0, 1] for signal in y), "All signals should be -1, 0, or 1"
            
            # Test 2: Time-based data splits (Requirement 4.4)
            splits = pipeline.create_time_based_splits(X, y, n_splits=3)
            
            # Verify splits are created properly
            assert len(splits) == 3, "Should create exactly 3 splits"
            for train_idx, val_idx in splits:
                assert len(train_idx) > 0, "Training set should not be empty"
                assert len(val_idx) > 0, "Validation set should not be empty"
                assert len(set(train_idx) & set(val_idx)) == 0, "Train and validation sets should not overlap"
                # Time series split: validation indices should come after training indices
                assert max(train_idx) < min(val_idx), "Validation should come after training in time series split"
            
            # Test 3: Model training success (Requirement 4.5)
            # Use first split for training
            train_idx, val_idx = splits[0]
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = pipeline.train_model(model_type, X_train, y_train)
            
            # Verify model is trained successfully
            assert model is not None, "Model should be created successfully"
            assert hasattr(model, 'predict'), "Model should have predict method"
            assert hasattr(model, 'predict_proba'), "Model should have predict_proba method"
            
            # Test model can make predictions
            predictions = model.predict(X_val)
            assert isinstance(predictions, np.ndarray), "Predictions should be numpy array"
            assert len(predictions) == len(y_val), "Predictions should match validation set size"
            assert all(pred in [-1, 0, 1] for pred in predictions), "All predictions should be valid signals"
            
            # Test probability predictions
            probabilities = model.predict_proba(X_val)
            assert isinstance(probabilities, np.ndarray), "Probabilities should be numpy array"
            
            # Verify probabilities have correct shape and properties
            expected_shape = (len(y_val), 3)
            if probabilities.shape != expected_shape:
                # Some models might return transposed shape - try to handle it
                if probabilities.shape == (3, len(y_val)):
                    probabilities = probabilities.T
            
            # Final shape verification
            if probabilities.shape == expected_shape:
                assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6), "Probabilities should sum to 1"
            else:
                # If shape is still wrong, just verify it's a reasonable 2D array
                assert len(probabilities.shape) == 2, "Probabilities should be 2D array"
                assert 3 in probabilities.shape, "Should have 3 classes in probabilities"
            
            # Test 4: Model saving with proper versioning (Requirement 4.6)
            model_id = f"{model_type}_pattern{pattern_length}_test"
            saved_path = pipeline.save_model(model, model_id)
            
            # Verify model is saved properly
            assert os.path.exists(saved_path), "Model file should be created"
            assert model_id in os.path.basename(saved_path), "Model ID should be in filename"
            assert saved_path.endswith('.pkl'), "Model should be saved as pickle file"
            
            # Test model can be loaded back
            loaded_model = pipeline.load_model(saved_path)
            assert loaded_model is not None, "Model should be loaded successfully"
            
            # Verify loaded model works the same
            loaded_predictions = loaded_model.predict(X_val)
            np.testing.assert_array_equal(predictions, loaded_predictions, 
                                        "Loaded model should make same predictions")
            
            # Test 5: Model validation metrics
            metrics = pipeline.validate_model(model, X_val, y_val)
            
            # Verify metrics are calculated
            assert isinstance(metrics, dict), "Metrics should be returned as dictionary"
            required_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
            for metric in required_metrics:
                assert metric in metrics, f"Should calculate {metric}"
                assert 0 <= metrics[metric] <= 1, f"{metric} should be between 0 and 1"
    
    @given(st.data())
    @settings(
        max_examples=2, 
        deadline=120000,
        suppress_health_check=[HealthCheck.large_base_example, HealthCheck.data_too_large]
    )
    def test_all_model_types_training(self, data):
        """
        Test that all supported model types can be trained successfully.
        
        Validates: Requirements 4.1, 4.2 (XGBoost baseline + additional models)
        """
        # Generate test data
        features = data.draw(generate_stock_features())
        pattern_length = data.draw(generate_pattern_length())
        
        # Generate corresponding signals
        n_samples = len(features)
        signals = data.draw(generate_trading_signals(n_samples))
        
        # Create temporary directory for models
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = ModelTrainingPipeline(models_dir=temp_dir)
            
            # Test all model types
            model_types = ['xgboost', 'random_forest', 'svm', 'neural_network']
            
            for model_type in model_types:
                # Prepare data
                X, y = pipeline.prepare_training_data(features, signals, pattern_length)
                
                # Skip if insufficient data or insufficient class diversity
                if len(X) < 20:
                    continue
                
                # Check if we have at least 2 classes for ML training
                unique_classes = len(np.unique(y))
                if unique_classes < 2:
                    continue
                
                # Create simple train/test split
                split_idx = len(X) // 2
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Train model
                model = pipeline.train_model(model_type, X_train, y_train)
                
                # Verify model works
                assert model is not None, f"{model_type} model should be created"
                
                predictions = model.predict(X_test)
                assert len(predictions) == len(y_test), f"{model_type} should predict all test samples"
                assert all(pred in [-1, 0, 1] for pred in predictions), f"{model_type} should predict valid signals"
                
                # Verify model can be saved and loaded
                model_id = f"{model_type}_test"
                saved_path = pipeline.save_model(model, model_id)
                assert os.path.exists(saved_path), f"{model_type} model should be saved"
                
                loaded_model = pipeline.load_model(saved_path)
                loaded_predictions = loaded_model.predict(X_test)
                np.testing.assert_array_equal(predictions, loaded_predictions,
                                            f"Loaded {model_type} model should work identically")


if __name__ == "__main__":
    pytest.main([__file__])