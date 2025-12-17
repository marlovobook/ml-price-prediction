"""
Schema validation for prediction inputs against training data schemas.
Ensures prediction inputs match the expected format and structure.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Set
import json
from pathlib import Path
import pickle

from ..utils.exceptions import DataValidationError
from ..utils.logging_config import get_logger


class SchemaValidator:
    """Validates prediction inputs against training data schemas."""
    
    def __init__(self):
        self.logger = get_logger("SchemaValidator")
        self._schemas: Dict[str, Dict[str, Any]] = {}
    
    def create_schema_from_training_data(self, X: np.ndarray, feature_names: List[str],
                                       schema_name: str, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Create a validation schema from training data.
        
        Args:
            X: Training feature matrix
            feature_names: List of feature names
            schema_name: Name for the schema
            y: Optional target vector
            
        Returns:
            Dict: Created schema
            
        Raises:
            DataValidationError: If schema creation fails
        """
        try:
            self.logger.info(f"Creating schema '{schema_name}' from training data")
            
            if X.size == 0:
                raise DataValidationError(
                    "Cannot create schema from empty training data",
                    error_code="EMPTY_TRAINING_DATA"
                )
            
            if len(feature_names) != X.shape[1]:
                raise DataValidationError(
                    f"Feature names count ({len(feature_names)}) doesn't match data dimensions ({X.shape[1]})",
                    error_code="FEATURE_NAME_MISMATCH"
                )
            
            # Create feature statistics
            feature_stats = {}
            for i, name in enumerate(feature_names):
                feature_data = X[:, i]
                
                stats = {
                    'name': name,
                    'index': i,
                    'dtype': str(feature_data.dtype),
                    'mean': float(np.mean(feature_data)),
                    'std': float(np.std(feature_data)),
                    'min': float(np.min(feature_data)),
                    'max': float(np.max(feature_data)),
                    'percentile_25': float(np.percentile(feature_data, 25)),
                    'percentile_75': float(np.percentile(feature_data, 75)),
                    'has_nan': bool(np.isnan(feature_data).any()),
                    'has_inf': bool(np.isinf(feature_data).any()),
                    'unique_values': int(len(np.unique(feature_data))),
                    'zero_count': int(np.sum(feature_data == 0))
                }
                
                # Calculate reasonable bounds (3 standard deviations)
                stats['lower_bound'] = stats['mean'] - 3 * stats['std']
                stats['upper_bound'] = stats['mean'] + 3 * stats['std']
                
                feature_stats[name] = stats
            
            # Create overall schema
            schema = {
                'schema_name': schema_name,
                'n_features': X.shape[1],
                'n_samples': X.shape[0],
                'feature_names': feature_names,
                'feature_stats': feature_stats,
                'data_shape': X.shape,
                'created_timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Add target statistics if provided
            if y is not None:
                target_stats = {
                    'dtype': str(y.dtype),
                    'mean': float(np.mean(y)),
                    'std': float(np.std(y)),
                    'min': float(np.min(y)),
                    'max': float(np.max(y)),
                    'unique_values': int(len(np.unique(y))),
                    'class_distribution': self._get_class_distribution(y)
                }
                schema['target_stats'] = target_stats
            
            # Store schema
            self._schemas[schema_name] = schema
            
            self.logger.info(f"Schema '{schema_name}' created successfully with {len(feature_names)} features")
            
            return schema
            
        except Exception as e:
            self.logger.error(f"Error creating schema '{schema_name}': {str(e)}")
            raise DataValidationError(
                f"Schema creation failed: {str(e)}",
                error_code="SCHEMA_CREATION_ERROR",
                details={"schema_name": schema_name, "error": str(e)}
            )
    
    def validate_prediction_input(self, X: np.ndarray, schema_name: str,
                                feature_names: Optional[List[str]] = None,
                                strict_bounds: bool = False) -> Dict[str, Any]:
        """
        Validate prediction input against a stored schema.
        
        Args:
            X: Input feature matrix
            schema_name: Name of schema to validate against
            feature_names: Optional feature names for validation
            strict_bounds: Whether to enforce strict statistical bounds
            
        Returns:
            Dict: Validation results
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            self.logger.debug(f"Validating prediction input against schema '{schema_name}'")
            
            # Get schema
            if schema_name not in self._schemas:
                raise DataValidationError(
                    f"Schema '{schema_name}' not found",
                    error_code="SCHEMA_NOT_FOUND",
                    details={"available_schemas": list(self._schemas.keys())}
                )
            
            schema = self._schemas[schema_name]
            
            # Basic shape validation
            validation_results = self._validate_basic_structure(X, schema, feature_names)
            
            # Feature-level validation
            feature_validation = self._validate_features(X, schema, strict_bounds)
            validation_results['feature_validation'] = feature_validation
            
            # Calculate overall validation score
            validation_results['validation_score'] = self._calculate_validation_score(validation_results)
            
            # Determine if validation passed
            validation_results['passed'] = validation_results['validation_score'] > 0.8
            
            if validation_results['passed']:
                self.logger.debug(f"Prediction input validation passed for schema '{schema_name}'")
            else:
                self.logger.warning(f"Prediction input validation failed for schema '{schema_name}'")
            
            return validation_results
            
        except DataValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Error validating prediction input: {str(e)}")
            raise DataValidationError(
                f"Prediction input validation failed: {str(e)}",
                error_code="VALIDATION_ERROR",
                details={"schema_name": schema_name, "error": str(e)}
            )
    
    def validate_feature_drift(self, X: np.ndarray, schema_name: str,
                             drift_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect feature drift by comparing input statistics to schema.
        
        Args:
            X: Input feature matrix
            schema_name: Name of schema to compare against
            drift_threshold: Threshold for detecting significant drift
            
        Returns:
            Dict: Drift detection results
            
        Raises:
            DataValidationError: If drift detection fails
        """
        try:
            self.logger.info(f"Detecting feature drift against schema '{schema_name}'")
            
            if schema_name not in self._schemas:
                raise DataValidationError(
                    f"Schema '{schema_name}' not found",
                    error_code="SCHEMA_NOT_FOUND"
                )
            
            schema = self._schemas[schema_name]
            
            if X.shape[1] != schema['n_features']:
                raise DataValidationError(
                    f"Feature count mismatch: expected {schema['n_features']}, got {X.shape[1]}",
                    error_code="FEATURE_COUNT_MISMATCH"
                )
            
            drift_results = {}
            significant_drift_count = 0
            
            for i, feature_name in enumerate(schema['feature_names']):
                feature_data = X[:, i]
                schema_stats = schema['feature_stats'][feature_name]
                
                # Calculate current statistics
                current_mean = float(np.mean(feature_data))
                current_std = float(np.std(feature_data))
                
                # Calculate drift metrics
                mean_drift = abs(current_mean - schema_stats['mean']) / (schema_stats['std'] + 1e-8)
                std_drift = abs(current_std - schema_stats['std']) / (schema_stats['std'] + 1e-8)
                
                # Check for distribution shift using Kolmogorov-Smirnov-like metric
                distribution_drift = self._calculate_distribution_drift(feature_data, schema_stats)
                
                feature_drift = {
                    'mean_drift': mean_drift,
                    'std_drift': std_drift,
                    'distribution_drift': distribution_drift,
                    'max_drift': max(mean_drift, std_drift, distribution_drift),
                    'significant_drift': max(mean_drift, std_drift, distribution_drift) > drift_threshold,
                    'current_mean': current_mean,
                    'current_std': current_std,
                    'schema_mean': schema_stats['mean'],
                    'schema_std': schema_stats['std']
                }
                
                if feature_drift['significant_drift']:
                    significant_drift_count += 1
                
                drift_results[feature_name] = feature_drift
            
            # Calculate overall drift metrics
            overall_drift = {
                'n_features_with_drift': significant_drift_count,
                'total_features': len(schema['feature_names']),
                'drift_percentage': (significant_drift_count / len(schema['feature_names'])) * 100,
                'avg_drift_score': np.mean([result['max_drift'] for result in drift_results.values()]),
                'max_drift_score': np.max([result['max_drift'] for result in drift_results.values()]),
                'significant_drift_detected': significant_drift_count > 0
            }
            
            results = {
                'schema_name': schema_name,
                'drift_threshold': drift_threshold,
                'feature_drift': drift_results,
                'overall_drift': overall_drift,
                'input_shape': X.shape
            }
            
            self.logger.info(f"Drift detection completed: {significant_drift_count}/{len(schema['feature_names'])} features with significant drift")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error detecting feature drift: {str(e)}")
            raise DataValidationError(
                f"Feature drift detection failed: {str(e)}",
                error_code="DRIFT_DETECTION_ERROR",
                details={"schema_name": schema_name, "error": str(e)}
            )
    
    def save_schema(self, schema_name: str, file_path: Union[str, Path]) -> None:
        """
        Save a schema to file.
        
        Args:
            schema_name: Name of schema to save
            file_path: Path to save schema file
            
        Raises:
            DataValidationError: If saving fails
        """
        try:
            if schema_name not in self._schemas:
                raise DataValidationError(
                    f"Schema '{schema_name}' not found",
                    error_code="SCHEMA_NOT_FOUND"
                )
            
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            schema = self._schemas[schema_name]
            
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'w') as f:
                    json.dump(schema, f, indent=2, default=str)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(schema, f)
            
            self.logger.info(f"Schema '{schema_name}' saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving schema '{schema_name}': {str(e)}")
            raise DataValidationError(
                f"Schema saving failed: {str(e)}",
                error_code="SCHEMA_SAVE_ERROR",
                details={"schema_name": schema_name, "file_path": str(file_path)}
            )
    
    def load_schema(self, file_path: Union[str, Path], schema_name: Optional[str] = None) -> str:
        """
        Load a schema from file.
        
        Args:
            file_path: Path to schema file
            schema_name: Optional name to assign to loaded schema
            
        Returns:
            str: Name of loaded schema
            
        Raises:
            DataValidationError: If loading fails
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise DataValidationError(
                    f"Schema file not found: {file_path}",
                    error_code="SCHEMA_FILE_NOT_FOUND"
                )
            
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r') as f:
                    schema = json.load(f)
            else:
                with open(file_path, 'rb') as f:
                    schema = pickle.load(f)
            
            # Use provided name or extract from schema
            if schema_name is None:
                schema_name = schema.get('schema_name', file_path.stem)
            
            schema['schema_name'] = schema_name
            self._schemas[schema_name] = schema
            
            self.logger.info(f"Schema '{schema_name}' loaded from {file_path}")
            
            return schema_name
            
        except Exception as e:
            self.logger.error(f"Error loading schema from {file_path}: {str(e)}")
            raise DataValidationError(
                f"Schema loading failed: {str(e)}",
                error_code="SCHEMA_LOAD_ERROR",
                details={"file_path": str(file_path)}
            )
    
    def get_schema_info(self, schema_name: str) -> Dict[str, Any]:
        """
        Get information about a stored schema.
        
        Args:
            schema_name: Name of schema
            
        Returns:
            Dict: Schema information
            
        Raises:
            DataValidationError: If schema not found
        """
        if schema_name not in self._schemas:
            raise DataValidationError(
                f"Schema '{schema_name}' not found",
                error_code="SCHEMA_NOT_FOUND",
                details={"available_schemas": list(self._schemas.keys())}
            )
        
        schema = self._schemas[schema_name]
        
        return {
            'schema_name': schema['schema_name'],
            'n_features': schema['n_features'],
            'n_samples': schema['n_samples'],
            'feature_names': schema['feature_names'],
            'created_timestamp': schema['created_timestamp'],
            'has_target_stats': 'target_stats' in schema
        }
    
    def list_schemas(self) -> List[str]:
        """
        List all available schema names.
        
        Returns:
            List: Schema names
        """
        return list(self._schemas.keys())
    
    def _validate_basic_structure(self, X: np.ndarray, schema: Dict[str, Any],
                                feature_names: Optional[List[str]]) -> Dict[str, Any]:
        """Validate basic structure of input data."""
        results = {
            'shape_validation': {},
            'feature_name_validation': {},
            'data_type_validation': {}
        }
        
        # Shape validation
        expected_features = schema['n_features']
        actual_features = X.shape[1] if len(X.shape) > 1 else 1
        
        results['shape_validation'] = {
            'expected_features': expected_features,
            'actual_features': actual_features,
            'shape_match': expected_features == actual_features,
            'input_shape': X.shape
        }
        
        # Feature name validation
        if feature_names is not None:
            expected_names = set(schema['feature_names'])
            actual_names = set(feature_names)
            
            results['feature_name_validation'] = {
                'expected_names': list(expected_names),
                'actual_names': list(actual_names),
                'names_match': expected_names == actual_names,
                'missing_features': list(expected_names - actual_names),
                'extra_features': list(actual_names - expected_names)
            }
        
        # Data type validation
        results['data_type_validation'] = {
            'is_numeric': np.issubdtype(X.dtype, np.number),
            'has_nan': bool(np.isnan(X).any()),
            'has_inf': bool(np.isinf(X).any()),
            'dtype': str(X.dtype)
        }
        
        return results
    
    def _validate_features(self, X: np.ndarray, schema: Dict[str, Any], strict_bounds: bool) -> Dict[str, Any]:
        """Validate individual features against schema statistics."""
        feature_results = {}
        
        for i, feature_name in enumerate(schema['feature_names']):
            if i >= X.shape[1]:
                break
                
            feature_data = X[:, i]
            schema_stats = schema['feature_stats'][feature_name]
            
            # Calculate current statistics
            current_mean = float(np.mean(feature_data))
            current_std = float(np.std(feature_data))
            current_min = float(np.min(feature_data))
            current_max = float(np.max(feature_data))
            
            # Validate bounds
            if strict_bounds:
                within_bounds = (current_min >= schema_stats['lower_bound'] and 
                               current_max <= schema_stats['upper_bound'])
            else:
                # More lenient bounds check
                within_bounds = (current_min >= schema_stats['min'] - 3 * schema_stats['std'] and
                               current_max <= schema_stats['max'] + 3 * schema_stats['std'])
            
            # Check for data quality issues
            has_nan = bool(np.isnan(feature_data).any())
            has_inf = bool(np.isinf(feature_data).any())
            
            feature_results[feature_name] = {
                'current_mean': current_mean,
                'current_std': current_std,
                'current_min': current_min,
                'current_max': current_max,
                'schema_mean': schema_stats['mean'],
                'schema_std': schema_stats['std'],
                'within_bounds': within_bounds,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'quality_score': self._calculate_feature_quality_score(
                    feature_data, schema_stats, strict_bounds
                )
            }
        
        return feature_results
    
    def _calculate_validation_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall validation score."""
        score = 1.0
        
        # Shape validation (critical)
        if not validation_results['shape_validation']['shape_match']:
            score *= 0.0  # Critical failure
        
        # Data type validation
        if not validation_results['data_type_validation']['is_numeric']:
            score *= 0.5
        
        if validation_results['data_type_validation']['has_nan']:
            score *= 0.7
        
        if validation_results['data_type_validation']['has_inf']:
            score *= 0.7
        
        # Feature validation
        if 'feature_validation' in validation_results:
            feature_scores = [result['quality_score'] for result in 
                            validation_results['feature_validation'].values()]
            if feature_scores:
                avg_feature_score = np.mean(feature_scores)
                score *= avg_feature_score
        
        return float(score)
    
    def _calculate_feature_quality_score(self, feature_data: np.ndarray, 
                                       schema_stats: Dict[str, Any], strict_bounds: bool) -> float:
        """Calculate quality score for a single feature."""
        score = 1.0
        
        # Bounds check
        if strict_bounds:
            within_bounds = (np.min(feature_data) >= schema_stats['lower_bound'] and
                           np.max(feature_data) <= schema_stats['upper_bound'])
        else:
            within_bounds = (np.min(feature_data) >= schema_stats['min'] - 3 * schema_stats['std'] and
                           np.max(feature_data) <= schema_stats['max'] + 3 * schema_stats['std'])
        
        if not within_bounds:
            score *= 0.5
        
        # Statistical similarity
        current_mean = np.mean(feature_data)
        current_std = np.std(feature_data)
        
        mean_diff = abs(current_mean - schema_stats['mean']) / (schema_stats['std'] + 1e-8)
        std_diff = abs(current_std - schema_stats['std']) / (schema_stats['std'] + 1e-8)
        
        # Penalize large statistical differences
        if mean_diff > 2.0:
            score *= 0.8
        if std_diff > 2.0:
            score *= 0.8
        
        # Data quality checks
        if np.isnan(feature_data).any():
            score *= 0.6
        if np.isinf(feature_data).any():
            score *= 0.6
        
        return float(score)
    
    def _calculate_distribution_drift(self, feature_data: np.ndarray, schema_stats: Dict[str, Any]) -> float:
        """Calculate distribution drift using statistical measures."""
        try:
            # Simple distribution comparison using percentiles
            current_p25 = np.percentile(feature_data, 25)
            current_p75 = np.percentile(feature_data, 75)
            
            p25_drift = abs(current_p25 - schema_stats['percentile_25']) / (schema_stats['std'] + 1e-8)
            p75_drift = abs(current_p75 - schema_stats['percentile_75']) / (schema_stats['std'] + 1e-8)
            
            return float(max(p25_drift, p75_drift))
            
        except Exception:
            return 0.0
    
    def _get_class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """Get class distribution for target variable."""
        unique_values, counts = np.unique(y, return_counts=True)
        return {str(val): int(count) for val, count in zip(unique_values, counts)}