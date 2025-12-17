"""
Anomaly detection and outlier handling for the Stock Direction Predictor system.
Implements statistical methods to identify and handle data anomalies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..utils.exceptions import DataValidationError
from ..utils.logging_config import get_logger


class AnomalyDetector:
    """Detects and handles anomalies in financial data."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers in the data
            random_state: Random state for reproducible results
        """
        self.contamination = contamination
        self.random_state = random_state
        self.logger = get_logger("AnomalyDetector")
        
        # Initialize isolation forest
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def detect_price_anomalies(self, data: pd.DataFrame, method: str = 'iqr') -> pd.Series:
        """
        Detect price anomalies using statistical methods.
        
        Args:
            data: DataFrame with OHLC price data
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
            
        Returns:
            pd.Series: Boolean series indicating anomalies (True = anomaly)
            
        Raises:
            DataValidationError: If detection fails
        """
        try:
            self.logger.debug(f"Detecting price anomalies using {method} method")
            
            if data.empty:
                return pd.Series([], dtype=bool)
            
            # Calculate price-based features for anomaly detection
            features = self._extract_price_features(data)
            
            if method == 'iqr':
                anomalies = self._detect_iqr_anomalies(features)
            elif method == 'zscore':
                anomalies = self._detect_zscore_anomalies(features)
            elif method == 'isolation_forest':
                anomalies = self._detect_isolation_forest_anomalies(features)
            else:
                raise DataValidationError(
                    f"Unknown anomaly detection method: {method}",
                    error_code="INVALID_DETECTION_METHOD",
                    details={"method": method}
                )
            
            anomaly_count = anomalies.sum()
            self.logger.info(f"Detected {anomaly_count} price anomalies out of {len(data)} records")
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting price anomalies: {str(e)}")
            raise DataValidationError(
                f"Price anomaly detection failed: {str(e)}",
                error_code="ANOMALY_DETECTION_ERROR",
                details={"method": method, "error": str(e)}
            )
    
    def detect_volume_anomalies(self, data: pd.DataFrame, method: str = 'iqr') -> pd.Series:
        """
        Detect volume anomalies.
        
        Args:
            data: DataFrame with volume data
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
            
        Returns:
            pd.Series: Boolean series indicating anomalies (True = anomaly)
            
        Raises:
            DataValidationError: If detection fails
        """
        try:
            self.logger.debug(f"Detecting volume anomalies using {method} method")
            
            if data.empty or 'Volume' not in data.columns:
                return pd.Series([], dtype=bool)
            
            volume_data = data['Volume'].copy()
            
            # Handle zero volumes (common in some datasets)
            volume_data = volume_data[volume_data > 0]
            
            if len(volume_data) == 0:
                return pd.Series([False] * len(data), index=data.index)
            
            if method == 'iqr':
                anomalies = self._detect_iqr_anomalies_single(volume_data)
            elif method == 'zscore':
                anomalies = self._detect_zscore_anomalies_single(volume_data)
            elif method == 'isolation_forest':
                # Reshape for sklearn
                volume_array = volume_data.values.reshape(-1, 1)
                scaled_volume = self.scaler.fit_transform(volume_array)
                outliers = self.isolation_forest.fit_predict(scaled_volume)
                anomalies = pd.Series(outliers == -1, index=volume_data.index)
            else:
                raise DataValidationError(
                    f"Unknown anomaly detection method: {method}",
                    error_code="INVALID_DETECTION_METHOD",
                    details={"method": method}
                )
            
            # Reindex to match original data
            result = pd.Series([False] * len(data), index=data.index)
            result.loc[anomalies.index] = anomalies
            
            anomaly_count = result.sum()
            self.logger.info(f"Detected {anomaly_count} volume anomalies out of {len(data)} records")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting volume anomalies: {str(e)}")
            raise DataValidationError(
                f"Volume anomaly detection failed: {str(e)}",
                error_code="ANOMALY_DETECTION_ERROR",
                details={"method": method, "error": str(e)}
            )
    
    def detect_technical_indicator_anomalies(self, indicators: pd.DataFrame) -> pd.Series:
        """
        Detect anomalies in technical indicators.
        
        Args:
            indicators: DataFrame with technical indicators
            
        Returns:
            pd.Series: Boolean series indicating anomalies (True = anomaly)
            
        Raises:
            DataValidationError: If detection fails
        """
        try:
            self.logger.debug("Detecting technical indicator anomalies")
            
            if indicators.empty:
                return pd.Series([], dtype=bool)
            
            # Remove non-numeric columns
            numeric_indicators = indicators.select_dtypes(include=[np.number])
            
            if numeric_indicators.empty:
                return pd.Series([False] * len(indicators), index=indicators.index)
            
            # Use isolation forest for multivariate anomaly detection
            # Handle missing values
            clean_indicators = numeric_indicators.fillna(numeric_indicators.median())
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(clean_indicators)
            
            # Detect anomalies
            outliers = self.isolation_forest.fit_predict(scaled_data)
            anomalies = pd.Series(outliers == -1, index=indicators.index)
            
            anomaly_count = anomalies.sum()
            self.logger.info(f"Detected {anomaly_count} technical indicator anomalies out of {len(indicators)} records")
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting technical indicator anomalies: {str(e)}")
            raise DataValidationError(
                f"Technical indicator anomaly detection failed: {str(e)}",
                error_code="ANOMALY_DETECTION_ERROR",
                details={"error": str(e)}
            )
    
    def handle_anomalies(self, data: pd.DataFrame, anomalies: pd.Series, 
                        method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle detected anomalies in the data.
        
        Args:
            data: Original data
            anomalies: Boolean series indicating anomalies
            method: Handling method ('remove', 'interpolate', 'cap', 'median')
            
        Returns:
            pd.DataFrame: Data with anomalies handled
            
        Raises:
            DataValidationError: If handling fails
        """
        try:
            self.logger.debug(f"Handling anomalies using {method} method")
            
            if data.empty or anomalies.empty:
                return data.copy()
            
            result = data.copy()
            anomaly_indices = anomalies[anomalies].index
            
            if len(anomaly_indices) == 0:
                return result
            
            if method == 'remove':
                # Remove anomalous rows
                result = result.drop(anomaly_indices)
                
            elif method == 'interpolate':
                # Interpolate anomalous values
                for idx in anomaly_indices:
                    if idx in result.index:
                        # Set anomalous values to NaN and interpolate
                        result.loc[idx] = np.nan
                
                # Interpolate missing values
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                result[numeric_cols] = result[numeric_cols].interpolate(method='linear')
                
                # Forward fill any remaining NaN values at the beginning
                result[numeric_cols] = result[numeric_cols].ffill()
                
                # Backward fill any remaining NaN values at the end
                result[numeric_cols] = result[numeric_cols].bfill()
                
            elif method == 'cap':
                # Cap anomalous values at percentiles
                for col in result.select_dtypes(include=[np.number]).columns:
                    q1 = result[col].quantile(0.25)
                    q3 = result[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Cap values for anomalous indices
                    for idx in anomaly_indices:
                        if idx in result.index:
                            if result.loc[idx, col] < lower_bound:
                                result.loc[idx, col] = lower_bound
                            elif result.loc[idx, col] > upper_bound:
                                result.loc[idx, col] = upper_bound
                
            elif method == 'median':
                # Replace anomalous values with median
                for col in result.select_dtypes(include=[np.number]).columns:
                    median_val = result[col].median()
                    for idx in anomaly_indices:
                        if idx in result.index:
                            result.loc[idx, col] = median_val
                            
            else:
                raise DataValidationError(
                    f"Unknown anomaly handling method: {method}",
                    error_code="INVALID_HANDLING_METHOD",
                    details={"method": method}
                )
            
            handled_count = len(anomaly_indices)
            self.logger.info(f"Handled {handled_count} anomalies using {method} method")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error handling anomalies: {str(e)}")
            raise DataValidationError(
                f"Anomaly handling failed: {str(e)}",
                error_code="ANOMALY_HANDLING_ERROR",
                details={"method": method, "error": str(e)}
            )
    
    def _extract_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for price anomaly detection."""
        features = pd.DataFrame(index=data.index)
        
        # Basic price features
        if 'Close' in data.columns:
            features['close'] = data['Close']
            features['close_pct_change'] = data['Close'].pct_change()
            features['close_rolling_std'] = data['Close'].rolling(window=5).std()
        
        # Price range features
        if all(col in data.columns for col in ['High', 'Low']):
            features['price_range'] = data['High'] - data['Low']
            features['price_range_pct'] = (data['High'] - data['Low']) / data['Close']
        
        # OHLC relationships
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            features['body_size'] = abs(data['Close'] - data['Open'])
            features['upper_shadow'] = data['High'] - np.maximum(data['Open'], data['Close'])
            features['lower_shadow'] = np.minimum(data['Open'], data['Close']) - data['Low']
        
        # Volume-price features
        if 'Volume' in data.columns and 'Close' in data.columns:
            features['volume_price_trend'] = data['Volume'] * data['Close'].pct_change()
        
        # Remove any infinite or NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())
        
        return features
    
    def _detect_iqr_anomalies(self, data: pd.DataFrame) -> pd.Series:
        """Detect anomalies using IQR method."""
        anomalies = pd.Series([False] * len(data), index=data.index)
        
        for col in data.select_dtypes(include=[np.number]).columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            col_anomalies = (data[col] < lower_bound) | (data[col] > upper_bound)
            anomalies = anomalies | col_anomalies
        
        return anomalies
    
    def _detect_iqr_anomalies_single(self, series: pd.Series) -> pd.Series:
        """Detect anomalies in a single series using IQR method."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_zscore_anomalies(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
        """Detect anomalies using Z-score method."""
        anomalies = pd.Series([False] * len(data), index=data.index)
        
        for col in data.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs(stats.zscore(data[col]))
            col_anomalies = z_scores > threshold
            anomalies = anomalies | col_anomalies
        
        return anomalies
    
    def _detect_zscore_anomalies_single(self, series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect anomalies in a single series using Z-score method."""
        z_scores = np.abs(stats.zscore(series))
        return pd.Series(z_scores > threshold, index=series.index)
    
    def _detect_isolation_forest_anomalies(self, data: pd.DataFrame) -> pd.Series:
        """Detect anomalies using Isolation Forest."""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.select_dtypes(include=[np.number]))
        
        # Detect anomalies
        outliers = self.isolation_forest.fit_predict(scaled_data)
        
        return pd.Series(outliers == -1, index=data.index)
    
    def get_anomaly_statistics(self, data: pd.DataFrame, anomalies: pd.Series) -> Dict[str, Any]:
        """
        Get statistics about detected anomalies.
        
        Args:
            data: Original data
            anomalies: Boolean series indicating anomalies
            
        Returns:
            Dict: Statistics about anomalies
        """
        try:
            total_records = len(data)
            anomaly_count = anomalies.sum()
            anomaly_percentage = (anomaly_count / total_records * 100) if total_records > 0 else 0
            
            stats_dict = {
                'total_records': total_records,
                'anomaly_count': int(anomaly_count),
                'anomaly_percentage': round(anomaly_percentage, 2),
                'clean_records': total_records - int(anomaly_count)
            }
            
            if anomaly_count > 0:
                anomaly_indices = anomalies[anomalies].index
                stats_dict['first_anomaly'] = str(anomaly_indices[0])
                stats_dict['last_anomaly'] = str(anomaly_indices[-1])
                
                # Get statistics for anomalous vs normal data
                if not data.empty:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        normal_data = data[~anomalies][numeric_cols]
                        anomaly_data = data[anomalies][numeric_cols]
                        
                        if not normal_data.empty and not anomaly_data.empty:
                            stats_dict['normal_data_mean'] = normal_data.mean().to_dict()
                            stats_dict['anomaly_data_mean'] = anomaly_data.mean().to_dict()
            
            return stats_dict
            
        except Exception as e:
            self.logger.error(f"Error calculating anomaly statistics: {str(e)}")
            return {
                'total_records': len(data),
                'anomaly_count': int(anomalies.sum()) if not anomalies.empty else 0,
                'error': str(e)
            }