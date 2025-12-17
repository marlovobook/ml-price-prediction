"""
Model training pipeline implementation for stock direction prediction.
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..interfaces import IModelTrainingPipeline, IMLModel, ModelConfiguration
from .ml_models import create_model


class ModelTrainingPipeline(IModelTrainingPipeline):
    """Implementation of model training pipeline."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize the training pipeline.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        self.scalers = {}  # Store scalers for each pattern length
        os.makedirs(models_dir, exist_ok=True)
    
    def prepare_training_data(self, features: pd.DataFrame, targets: pd.Series, pattern_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for the given pattern length.
        
        Args:
            features: DataFrame containing all features
            targets: Series containing target signals
            pattern_length: Length of candlestick pattern (3, 5, 7, 14)
            
        Returns:
            Tuple of (X, y) arrays ready for training
        """
        # Filter features relevant to the pattern length
        feature_columns = self._get_feature_columns_for_pattern(features.columns, pattern_length)
        
        # Select relevant features
        X = features[feature_columns].copy()
        y = targets.copy()
        
        # Remove rows with NaN values (common with technical indicators)
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Convert to numpy arrays
        X_array = X.values
        y_array = y.values
        
        # Scale features
        scaler_key = f"pattern_{pattern_length}"
        if scaler_key not in self.scalers:
            self.scalers[scaler_key] = StandardScaler()
            X_scaled = self.scalers[scaler_key].fit_transform(X_array)
        else:
            X_scaled = self.scalers[scaler_key].transform(X_array)
        
        return X_scaled, y_array
    
    def _get_feature_columns_for_pattern(self, all_columns: pd.Index, pattern_length: int) -> List[str]:
        """Get relevant feature columns for a specific pattern length.
        
        Args:
            all_columns: All available feature columns
            pattern_length: Candlestick pattern length
            
        Returns:
            List of relevant feature column names
        """
        # Base technical indicators (always included)
        base_features = [
            'rsi', 'macd', 'macd_signal', 'ema_20', 'ema_50', 'ema_200', 
            'atr', 'sma', 'golden_cross', 'head_shoulder', 'wedge',
            'fibonacci_23.6', 'fibonacci_38.2', 'fibonacci_50.0', 
            'fibonacci_61.8', 'fibonacci_78.6'
        ]
        
        # Pattern-specific features
        pattern_features = [f"signal_{pattern_length}day"]
        
        # Filter to only include columns that exist
        available_features = []
        for feature in base_features + pattern_features:
            if feature in all_columns:
                available_features.append(feature)
        
        # If no pattern-specific features found, use all available technical indicators
        if not any(f"signal_{pattern_length}day" in col for col in available_features):
            available_features = [col for col in all_columns if col in base_features]
        
        return available_features
    
    def train_model(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray) -> IMLModel:
        """Train a model of the specified type.
        
        Args:
            model_type: Type of model to train ('xgboost', 'random_forest', 'svm', 'neural_network')
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained model instance
        """
        # Get default hyperparameters for each model type
        hyperparameters = self._get_default_hyperparameters(model_type)
        
        # Create and train model
        model = create_model(model_type, **hyperparameters)
        model.fit(X_train, y_train)
        
        return model
    
    def _get_default_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for each model type."""
        hyperparameters = {
            'xgboost': {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'svm': {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'probability': True,
                'random_state': 42
            },
            'neural_network': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'learning_rate': 'constant',
                'max_iter': 500,
                'random_state': 42
            }
        }
        
        return hyperparameters.get(model_type.lower(), {})
    
    def validate_model(self, model: IMLModel, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Validate the model and return performance metrics.
        
        Args:
            model: Trained model to validate
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary of performance metrics
        """
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision_macro': precision_score(y_val, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_val, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_val, y_pred, average='macro', zero_division=0)
        }
        
        # Calculate per-class metrics
        for signal_class in [-1, 0, 1]:
            class_name = {-1: 'sell', 0: 'hold', 1: 'buy'}[signal_class]
            if signal_class in y_val:
                metrics[f'precision_{class_name}'] = precision_score(
                    y_val, y_pred, labels=[signal_class], average='macro', zero_division=0
                )
                metrics[f'recall_{class_name}'] = recall_score(
                    y_val, y_pred, labels=[signal_class], average='macro', zero_division=0
                )
        
        return metrics
    
    def save_model(self, model: IMLModel, model_id: str) -> str:
        """Save the model and return the saved model path.
        
        Args:
            model: Model to save
            model_id: Unique identifier for the model
            
        Returns:
            Path to the saved model file
        """
        # Create model filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_id}_{timestamp}.pkl"
        filepath = os.path.join(self.models_dir, filename)
        
        # Save model using joblib for sklearn-based models
        try:
            joblib.dump(model, filepath)
        except Exception:
            # Fallback to pickle for other models
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        
        return filepath
    
    def load_model(self, filepath: str) -> IMLModel:
        """Load a saved model.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            Loaded model instance
        """
        try:
            return joblib.load(filepath)
        except Exception:
            # Fallback to pickle
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    
    def create_time_based_splits(self, X: np.ndarray, y: np.ndarray, n_splits: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create time-based splits for training, validation, and test sets.
        
        Args:
            X: Feature array
            y: Target array
            n_splits: Number of splits for time series cross-validation
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return list(tscv.split(X))
    
    def train_all_models_for_pattern(self, features: pd.DataFrame, targets: pd.Series, 
                                   pattern_length: int) -> Dict[str, Dict[str, Any]]:
        """Train all model types for a specific pattern length.
        
        Args:
            features: DataFrame containing all features
            targets: Series containing target signals
            pattern_length: Candlestick pattern length
            
        Returns:
            Dictionary containing trained models and their metrics
        """
        # Prepare data
        X, y = self.prepare_training_data(features, targets, pattern_length)
        
        # Create time-based splits
        splits = self.create_time_based_splits(X, y, n_splits=3)
        
        # Model types to train
        model_types = ['xgboost', 'random_forest', 'svm', 'neural_network']
        
        results = {}
        
        for model_type in model_types:
            print(f"Training {model_type} model for {pattern_length}-day pattern...")
            
            model_results = {
                'models': [],
                'metrics': [],
                'avg_metrics': {}
            }
            
            # Train on each split
            for fold, (train_idx, val_idx) in enumerate(splits):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model
                model = self.train_model(model_type, X_train, y_train)
                
                # Validate model
                metrics = self.validate_model(model, X_val, y_val)
                
                # Save model
                model_id = f"{model_type}_pattern{pattern_length}_fold{fold}"
                model_path = self.save_model(model, model_id)
                
                model_results['models'].append({
                    'model': model,
                    'path': model_path,
                    'fold': fold
                })
                model_results['metrics'].append(metrics)
            
            # Calculate average metrics
            if model_results['metrics']:
                avg_metrics = {}
                for metric_name in model_results['metrics'][0].keys():
                    avg_metrics[metric_name] = np.mean([
                        m[metric_name] for m in model_results['metrics']
                    ])
                model_results['avg_metrics'] = avg_metrics
            
            results[model_type] = model_results
        
        return results