"""
Cross-validation functionality for model stability assessment.
Implements time-series aware cross-validation for financial data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Iterator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

from ..utils.exceptions import DataValidationError, ModelTrainingError
from ..utils.logging_config import get_logger
from ..interfaces import IMLModel


class CrossValidator:
    """Time-series aware cross-validator for model stability assessment."""
    
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None, gap: int = 0):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of cross-validation splits
            test_size: Size of test set in each split
            gap: Gap between train and test sets (to avoid data leakage)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.logger = get_logger("CrossValidator")
        
        # Initialize time series splitter
        self.ts_splitter = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap
        )
    
    def validate_model_stability(self, model_class: type, X: np.ndarray, y: np.ndarray,
                               model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate model stability using time-series cross-validation.
        
        Args:
            model_class: Model class to instantiate
            X: Feature matrix
            y: Target vector
            model_params: Parameters for model initialization
            
        Returns:
            Dict: Cross-validation results and stability metrics
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            self.logger.info(f"Starting cross-validation for {model_class.__name__}")
            
            if model_params is None:
                model_params = {}
            
            # Validate inputs
            self._validate_cv_inputs(X, y)
            
            # Perform cross-validation
            cv_results = []
            fold_metrics = []
            
            for fold, (train_idx, test_idx) in enumerate(self.ts_splitter.split(X)):
                self.logger.debug(f"Processing fold {fold + 1}/{self.n_splits}")
                
                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                try:
                    # Train model
                    model = model_class(**model_params)
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    fold_result = self._calculate_fold_metrics(y_test, y_pred, fold)
                    cv_results.append(fold_result)
                    fold_metrics.append(fold_result['metrics'])
                    
                except Exception as e:
                    self.logger.warning(f"Fold {fold + 1} failed: {str(e)}")
                    # Continue with other folds
                    continue
            
            if not cv_results:
                raise ModelTrainingError(
                    "All cross-validation folds failed",
                    error_code="ALL_FOLDS_FAILED"
                )
            
            # Calculate stability metrics
            stability_metrics = self._calculate_stability_metrics(fold_metrics)
            
            # Compile final results
            results = {
                'n_successful_folds': len(cv_results),
                'n_total_folds': self.n_splits,
                'fold_results': cv_results,
                'stability_metrics': stability_metrics,
                'model_class': model_class.__name__,
                'model_params': model_params
            }
            
            self.logger.info(f"Cross-validation completed: {len(cv_results)}/{self.n_splits} folds successful")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {str(e)}")
            raise DataValidationError(
                f"Cross-validation failed: {str(e)}",
                error_code="CROSS_VALIDATION_ERROR",
                details={"model_class": model_class.__name__, "error": str(e)}
            )
    
    def validate_feature_stability(self, X: np.ndarray, feature_names: Optional[List[str]] = None,
                                 window_size: int = 100) -> Dict[str, Any]:
        """
        Validate feature stability over time using rolling windows.
        
        Args:
            X: Feature matrix
            feature_names: Optional feature names
            window_size: Size of rolling window for stability analysis
            
        Returns:
            Dict: Feature stability metrics
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            self.logger.info("Analyzing feature stability over time")
            
            if X.size == 0:
                raise DataValidationError(
                    "Empty feature matrix provided",
                    error_code="EMPTY_FEATURES"
                )
            
            n_samples, n_features = X.shape
            
            if n_samples < window_size:
                self.logger.warning(f"Sample size ({n_samples}) smaller than window size ({window_size})")
                window_size = max(10, n_samples // 2)
            
            # Convert to DataFrame for easier manipulation
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(n_features)]
            
            df = pd.DataFrame(X, columns=feature_names)
            
            # Calculate rolling statistics
            stability_results = {}
            
            for col in df.columns:
                col_stability = self._analyze_feature_column_stability(df[col], window_size)
                stability_results[col] = col_stability
            
            # Calculate overall stability metrics
            overall_metrics = self._calculate_overall_feature_stability(stability_results)
            
            results = {
                'feature_stability': stability_results,
                'overall_metrics': overall_metrics,
                'window_size': window_size,
                'n_samples': n_samples,
                'n_features': n_features
            }
            
            self.logger.info("Feature stability analysis completed")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Feature stability validation failed: {str(e)}")
            raise DataValidationError(
                f"Feature stability validation failed: {str(e)}",
                error_code="FEATURE_STABILITY_ERROR",
                details={"error": str(e)}
            )
    
    def validate_prediction_consistency(self, model: IMLModel, X: np.ndarray,
                                      n_iterations: int = 10) -> Dict[str, Any]:
        """
        Validate prediction consistency by running multiple predictions.
        
        Args:
            model: Trained model
            X: Feature matrix for prediction
            n_iterations: Number of prediction iterations
            
        Returns:
            Dict: Prediction consistency metrics
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            self.logger.info(f"Validating prediction consistency over {n_iterations} iterations")
            
            if X.size == 0:
                raise DataValidationError(
                    "Empty feature matrix provided",
                    error_code="EMPTY_FEATURES"
                )
            
            predictions = []
            
            # Run multiple predictions
            for i in range(n_iterations):
                try:
                    pred = model.predict(X)
                    predictions.append(pred)
                except Exception as e:
                    self.logger.warning(f"Prediction iteration {i + 1} failed: {str(e)}")
                    continue
            
            if not predictions:
                raise ModelTrainingError(
                    "All prediction iterations failed",
                    error_code="ALL_PREDICTIONS_FAILED"
                )
            
            # Convert to numpy array for analysis
            predictions_array = np.array(predictions)
            
            # Calculate consistency metrics
            consistency_metrics = self._calculate_prediction_consistency_metrics(predictions_array)
            
            results = {
                'n_successful_iterations': len(predictions),
                'n_total_iterations': n_iterations,
                'consistency_metrics': consistency_metrics,
                'prediction_shape': predictions_array.shape
            }
            
            self.logger.info("Prediction consistency validation completed")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Prediction consistency validation failed: {str(e)}")
            raise DataValidationError(
                f"Prediction consistency validation failed: {str(e)}",
                error_code="PREDICTION_CONSISTENCY_ERROR",
                details={"error": str(e)}
            )
    
    def validate_temporal_stability(self, X: np.ndarray, y: np.ndarray,
                                  time_periods: List[Tuple[int, int]]) -> Dict[str, Any]:
        """
        Validate model performance across different time periods.
        
        Args:
            X: Feature matrix
            y: Target vector
            time_periods: List of (start_idx, end_idx) tuples defining time periods
            
        Returns:
            Dict: Temporal stability metrics
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            self.logger.info(f"Validating temporal stability across {len(time_periods)} periods")
            
            self._validate_cv_inputs(X, y)
            
            period_results = []
            
            for i, (start_idx, end_idx) in enumerate(time_periods):
                self.logger.debug(f"Processing time period {i + 1}: [{start_idx}:{end_idx}]")
                
                # Validate period indices
                if start_idx < 0 or end_idx > len(X) or start_idx >= end_idx:
                    self.logger.warning(f"Invalid time period {i + 1}: [{start_idx}:{end_idx}]")
                    continue
                
                # Extract period data
                X_period = X[start_idx:end_idx]
                y_period = y[start_idx:end_idx]
                
                if len(X_period) < 10:  # Minimum samples for meaningful analysis
                    self.logger.warning(f"Time period {i + 1} has too few samples: {len(X_period)}")
                    continue
                
                # Calculate period statistics
                period_stats = self._calculate_period_statistics(X_period, y_period, i + 1)
                period_results.append(period_stats)
            
            if not period_results:
                raise DataValidationError(
                    "No valid time periods for analysis",
                    error_code="NO_VALID_PERIODS"
                )
            
            # Calculate temporal stability metrics
            temporal_metrics = self._calculate_temporal_stability_metrics(period_results)
            
            results = {
                'n_periods': len(period_results),
                'period_results': period_results,
                'temporal_stability': temporal_metrics
            }
            
            self.logger.info("Temporal stability validation completed")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Temporal stability validation failed: {str(e)}")
            raise DataValidationError(
                f"Temporal stability validation failed: {str(e)}",
                error_code="TEMPORAL_STABILITY_ERROR",
                details={"error": str(e)}
            )
    
    def _validate_cv_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate cross-validation inputs."""
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise DataValidationError(
                "X and y must be numpy arrays",
                error_code="INVALID_INPUT_TYPE"
            )
        
        if X.size == 0 or y.size == 0:
            raise DataValidationError(
                "X and y cannot be empty",
                error_code="EMPTY_INPUT"
            )
        
        if len(X) != len(y):
            raise DataValidationError(
                f"X and y must have same length: {len(X)} vs {len(y)}",
                error_code="LENGTH_MISMATCH"
            )
        
        if len(X) < self.n_splits:
            raise DataValidationError(
                f"Not enough samples for {self.n_splits} splits: {len(X)}",
                error_code="INSUFFICIENT_SAMPLES"
            )
    
    def _calculate_fold_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, fold: int) -> Dict[str, Any]:
        """Calculate metrics for a single fold."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            # Additional metrics
            correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
            
            return {
                'fold': fold + 1,
                'n_samples': len(y_true),
                'metrics': {
                    'mse': float(mse),
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'correlation': float(correlation) if not np.isnan(correlation) else 0.0
                }
            }
    
    def _calculate_stability_metrics(self, fold_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate stability metrics across folds."""
        if not fold_metrics:
            return {}
        
        metrics_df = pd.DataFrame(fold_metrics)
        
        stability = {}
        for metric in metrics_df.columns:
            values = metrics_df[metric].values
            stability[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'cv': float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else float('inf')
            }
        
        return stability
    
    def _analyze_feature_column_stability(self, series: pd.Series, window_size: int) -> Dict[str, Any]:
        """Analyze stability of a single feature column."""
        rolling_mean = series.rolling(window=window_size).mean()
        rolling_std = series.rolling(window=window_size).std()
        
        # Calculate stability metrics
        mean_stability = rolling_mean.std() / rolling_mean.mean() if rolling_mean.mean() != 0 else float('inf')
        std_stability = rolling_std.std() / rolling_std.mean() if rolling_std.mean() != 0 else float('inf')
        
        return {
            'mean_stability_cv': float(mean_stability),
            'std_stability_cv': float(std_stability),
            'overall_mean': float(series.mean()),
            'overall_std': float(series.std()),
            'n_windows': len(rolling_mean.dropna())
        }
    
    def _calculate_overall_feature_stability(self, feature_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall feature stability metrics."""
        if not feature_results:
            return {}
        
        mean_cvs = [result['mean_stability_cv'] for result in feature_results.values() 
                   if not np.isinf(result['mean_stability_cv'])]
        std_cvs = [result['std_stability_cv'] for result in feature_results.values() 
                  if not np.isinf(result['std_stability_cv'])]
        
        return {
            'avg_mean_stability_cv': float(np.mean(mean_cvs)) if mean_cvs else float('inf'),
            'avg_std_stability_cv': float(np.mean(std_cvs)) if std_cvs else float('inf'),
            'n_stable_features': len(mean_cvs),
            'n_total_features': len(feature_results)
        }
    
    def _calculate_prediction_consistency_metrics(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Calculate prediction consistency metrics."""
        # Calculate variance across predictions for each sample
        prediction_variance = np.var(predictions, axis=0)
        
        # Calculate overall consistency metrics
        mean_variance = float(np.mean(prediction_variance))
        max_variance = float(np.max(prediction_variance))
        consistency_score = 1.0 / (1.0 + mean_variance)  # Higher is more consistent
        
        return {
            'mean_prediction_variance': mean_variance,
            'max_prediction_variance': max_variance,
            'consistency_score': float(consistency_score),
            'n_consistent_predictions': int(np.sum(prediction_variance < 0.01))  # Arbitrary threshold
        }
    
    def _calculate_period_statistics(self, X: np.ndarray, y: np.ndarray, period_id: int) -> Dict[str, Any]:
        """Calculate statistics for a time period."""
        return {
            'period_id': period_id,
            'n_samples': len(X),
            'feature_means': X.mean(axis=0).tolist(),
            'feature_stds': X.std(axis=0).tolist(),
            'target_mean': float(y.mean()),
            'target_std': float(y.std()),
            'target_min': float(y.min()),
            'target_max': float(y.max())
        }
    
    def _calculate_temporal_stability_metrics(self, period_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate temporal stability metrics across periods."""
        if not period_results:
            return {}
        
        # Extract target statistics across periods
        target_means = [result['target_mean'] for result in period_results]
        target_stds = [result['target_std'] for result in period_results]
        
        # Calculate stability of target distribution
        target_mean_stability = np.std(target_means) / np.mean(target_means) if np.mean(target_means) != 0 else float('inf')
        target_std_stability = np.std(target_stds) / np.mean(target_stds) if np.mean(target_stds) != 0 else float('inf')
        
        # Calculate feature stability across periods
        n_features = len(period_results[0]['feature_means'])
        feature_stability = []
        
        for i in range(n_features):
            feature_means_across_periods = [result['feature_means'][i] for result in period_results]
            feature_cv = np.std(feature_means_across_periods) / np.mean(feature_means_across_periods) \
                        if np.mean(feature_means_across_periods) != 0 else float('inf')
            feature_stability.append(feature_cv)
        
        return {
            'target_mean_stability_cv': float(target_mean_stability),
            'target_std_stability_cv': float(target_std_stability),
            'avg_feature_stability_cv': float(np.mean(feature_stability)),
            'max_feature_stability_cv': float(np.max(feature_stability)),
            'n_periods_analyzed': len(period_results)
        }