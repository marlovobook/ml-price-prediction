"""
Custom exceptions for the Stock Direction Predictor system.
Provides specific error types for different components and scenarios.
"""

from typing import Optional, Any, Dict


class StockPredictorError(Exception):
    """Base exception class for Stock Direction Predictor system."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        error_str = self.message
        if self.error_code:
            error_str = f"[{self.error_code}] {error_str}"
        if self.details:
            error_str += f" Details: {self.details}"
        return error_str


class DataCollectionError(StockPredictorError):
    """Exception raised during data collection operations."""
    pass


class DataValidationError(StockPredictorError):
    """Exception raised during data validation."""
    pass


class FeatureEngineeringError(StockPredictorError):
    """Exception raised during feature engineering operations."""
    pass


class ModelTrainingError(StockPredictorError):
    """Exception raised during model training operations."""
    pass


class ModelPredictionError(StockPredictorError):
    """Exception raised during model prediction operations."""
    pass


class BacktestingError(StockPredictorError):
    """Exception raised during backtesting operations."""
    pass


class ConfigurationError(StockPredictorError):
    """Exception raised for configuration-related issues."""
    pass


class InsufficientDataError(DataCollectionError):
    """Exception raised when insufficient data is available."""
    pass


class InvalidSymbolError(DataCollectionError):
    """Exception raised for invalid stock symbols."""
    pass


class NetworkError(DataCollectionError):
    """Exception raised for network-related issues during data collection."""
    pass


class InvalidIndicatorError(FeatureEngineeringError):
    """Exception raised for invalid technical indicator calculations."""
    pass


class PatternDetectionError(FeatureEngineeringError):
    """Exception raised during pattern detection operations."""
    pass


class ModelConvergenceError(ModelTrainingError):
    """Exception raised when model fails to converge."""
    pass


class InvalidModelTypeError(ModelTrainingError):
    """Exception raised for unsupported model types."""
    pass


class InsufficientCapitalError(BacktestingError):
    """Exception raised when portfolio has insufficient capital for trades."""
    pass


class InvalidSignalError(BacktestingError):
    """Exception raised for invalid trading signals."""
    pass


def handle_exception(logger, exception: Exception, context: str = "") -> None:
    """
    Centralized exception handling utility.
    
    Args:
        logger: Logger instance to use for logging
        exception: The exception that occurred
        context: Additional context about where the exception occurred
    """
    error_message = f"Exception in {context}: {str(exception)}"
    
    if isinstance(exception, StockPredictorError):
        logger.error(error_message)
        if exception.details:
            logger.error(f"Error details: {exception.details}")
    else:
        logger.exception(error_message)


def validate_required_fields(data: Dict[str, Any], required_fields: list, context: str = "") -> None:
    """
    Validate that all required fields are present in the data.
    
    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        context: Context for error messages
    
    Raises:
        DataValidationError: If any required fields are missing
    """
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    
    if missing_fields:
        raise DataValidationError(
            f"Missing required fields in {context}: {missing_fields}",
            error_code="MISSING_FIELDS",
            details={"missing_fields": missing_fields, "context": context}
        )


def validate_data_types(data: Dict[str, Any], type_mapping: Dict[str, type], context: str = "") -> None:
    """
    Validate that data fields have the correct types.
    
    Args:
        data: Dictionary to validate
        type_mapping: Dictionary mapping field names to expected types
        context: Context for error messages
    
    Raises:
        DataValidationError: If any fields have incorrect types
    """
    type_errors = []
    
    for field, expected_type in type_mapping.items():
        if field in data and data[field] is not None:
            if not isinstance(data[field], expected_type):
                type_errors.append({
                    "field": field,
                    "expected_type": expected_type.__name__,
                    "actual_type": type(data[field]).__name__,
                    "value": data[field]
                })
    
    if type_errors:
        raise DataValidationError(
            f"Type validation errors in {context}",
            error_code="TYPE_VALIDATION_ERROR",
            details={"type_errors": type_errors, "context": context}
        )


def validate_numeric_range(value: float, min_value: Optional[float] = None, 
                          max_value: Optional[float] = None, field_name: str = "value") -> None:
    """
    Validate that a numeric value is within the specified range.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        field_name: Name of the field being validated
    
    Raises:
        DataValidationError: If value is outside the allowed range
    """
    if min_value is not None and value < min_value:
        raise DataValidationError(
            f"{field_name} value {value} is below minimum {min_value}",
            error_code="VALUE_BELOW_MINIMUM",
            details={"field": field_name, "value": value, "min_value": min_value}
        )
    
    if max_value is not None and value > max_value:
        raise DataValidationError(
            f"{field_name} value {value} is above maximum {max_value}",
            error_code="VALUE_ABOVE_MAXIMUM",
            details={"field": field_name, "value": value, "max_value": max_value}
        )