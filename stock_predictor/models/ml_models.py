"""
Machine learning model implementations for stock direction prediction.
"""

import numpy as np
import pickle
import joblib
from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

from ..interfaces import IMLModel


class XGBoostModel(IMLModel):
    """XGBoost model implementation."""
    
    def __init__(self, **hyperparameters):
        """Initialize XGBoost model with hyperparameters."""
        default_params = {
            'objective': 'multi:softprob',
            'num_class': 3,  # -1, 0, 1 (sell, hold, buy)
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        default_params.update(hyperparameters)
        self.model = xgb.XGBClassifier(**default_params)
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the XGBoost model."""
        # Convert signals to class indices: -1 -> 0, 0 -> 1, 1 -> 2
        y_classes = y + 1
        self.model.fit(X, y_classes)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions and convert back to signal format."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        predictions = self.model.predict(X)
        # Convert back to signal format: 0 -> -1, 1 -> 0, 2 -> 1
        return predictions - 1
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)


class RandomForestModel(IMLModel):
    """Random Forest model implementation."""
    
    def __init__(self, **hyperparameters):
        """Initialize Random Forest model with hyperparameters."""
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        default_params.update(hyperparameters)
        self.model = RandomForestClassifier(**default_params)
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Random Forest model."""
        # Convert signals to class indices: -1 -> 0, 0 -> 1, 1 -> 2
        y_classes = y + 1
        self.model.fit(X, y_classes)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions and convert back to signal format."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        predictions = self.model.predict(X)
        # Convert back to signal format: 0 -> -1, 1 -> 0, 2 -> 1
        return predictions - 1
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)


class SVMModel(IMLModel):
    """Support Vector Machine model implementation."""
    
    def __init__(self, **hyperparameters):
        """Initialize SVM model with hyperparameters."""
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        }
        default_params.update(hyperparameters)
        self.model = SVC(**default_params)
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the SVM model."""
        # Convert signals to class indices: -1 -> 0, 0 -> 1, 1 -> 2
        y_classes = y + 1
        self.model.fit(X, y_classes)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions and convert back to signal format."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        predictions = self.model.predict(X)
        # Convert back to signal format: 0 -> -1, 1 -> 0, 2 -> 1
        return predictions - 1
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)


class NeuralNetworkModel(IMLModel):
    """Multi-layer Perceptron (Neural Network) model implementation."""
    
    def __init__(self, **hyperparameters):
        """Initialize Neural Network model with hyperparameters."""
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'learning_rate': 'constant',
            'max_iter': 500,
            'random_state': 42
        }
        default_params.update(hyperparameters)
        self.model = MLPClassifier(**default_params)
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Neural Network model."""
        # Convert signals to class indices: -1 -> 0, 0 -> 1, 1 -> 2
        y_classes = y + 1
        self.model.fit(X, y_classes)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions and convert back to signal format."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        predictions = self.model.predict(X)
        # Convert back to signal format: 0 -> -1, 1 -> 0, 2 -> 1
        return predictions - 1
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)


def create_model(model_type: str, **hyperparameters) -> IMLModel:
    """Factory function to create models by type."""
    model_classes = {
        'xgboost': XGBoostModel,
        'random_forest': RandomForestModel,
        'svm': SVMModel,
        'neural_network': NeuralNetworkModel
    }
    
    if model_type.lower() not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(model_classes.keys())}")
    
    return model_classes[model_type.lower()](**hyperparameters)