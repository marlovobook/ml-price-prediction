"""
Signal Alignment Engine for VectorBT Visualization Enhancement.

This module handles the critical task of properly aligning ML model predictions
with the complete historical data timeline for accurate VectorBT portfolio simulation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging
from dataclasses import dataclass

from ..utils.exceptions import DataValidationError


@dataclass
class AlignedSignals:
    """Data class for aligned signal results."""
    entry_signals: pd.Series
    exit_signals: pd.Series
    full_timeline: pd.DatetimeIndex
    test_period_start: int
    test_period_end: int
    prediction_count: int
    alignment_metadata: Dict[str, Any]


class SignalAlignmentEngine:
    """
    Engine for aligning ML predictions with complete historical data timeline.
    
    This class handles the critical task of mapping model predictions (which are
    typically generated only for a test period) to the full historical dataset
    required by VectorBT for proper portfolio simulation and visualization.
    """
    
    def __init__(self):
        """Initialize the Signal Alignment Engine."""
        self.logger = logging.getLogger(__name__)
    
    def align_predictions_to_timeline(
        self, 
        predictions: np.ndarray, 
        full_data: pd.DataFrame,
        test_start_idx: int
    ) -> AlignedSignals:
        """
        Align ML model predictions to the complete historical data timeline.
        
        This method creates full-sized signal arrays that match the complete
        historical dataset and populates only the test period with actual
        prediction values, as required by VectorBT.
        
        Args:
            predictions: ML model predictions array (values: 0, 1, 2)
            full_data: Complete historical dataset with full timeline
            test_start_idx: Index where test period begins in full_data
            
        Returns:
            AlignedSignals object with properly aligned entry/exit signals
            
        Raises:
            DataValidationError: If alignment parameters are invalid
        """
        try:
            # Validate inputs
            self._validate_alignment_inputs(predictions, full_data, test_start_idx)
            
            # Create full-sized signal arrays initialized to False
            entry_signals, exit_signals = self.create_full_signal_arrays(full_data)
            
            # Convert predictions to entry/exit signals
            pred_entries, pred_exits = self.convert_predictions_to_signals(predictions)
            
            # Calculate test period end index
            test_end_idx = test_start_idx + len(predictions)
            
            # Validate test period bounds
            if test_end_idx > len(full_data):
                raise DataValidationError(
                    f"Test period extends beyond data: {test_end_idx} > {len(full_data)}"
                )
            
            # Populate test period with actual predictions
            entry_signals.iloc[test_start_idx:test_end_idx] = pred_entries
            exit_signals.iloc[test_start_idx:test_end_idx] = pred_exits
            
            # Create alignment metadata
            metadata = {
                'original_prediction_count': len(predictions),
                'full_timeline_length': len(full_data),
                'test_start_idx': test_start_idx,
                'test_end_idx': test_end_idx,
                'alignment_timestamp': pd.Timestamp.now(),
                'prediction_distribution': dict(zip(*np.unique(predictions, return_counts=True)))
            }
            
            # Validate final alignment
            self.validate_signal_alignment(entry_signals, exit_signals, full_data)
            
            self.logger.info(
                f"Successfully aligned {len(predictions)} predictions to "
                f"{len(full_data)} timeline points (test period: {test_start_idx}-{test_end_idx})"
            )
            
            return AlignedSignals(
                entry_signals=entry_signals,
                exit_signals=exit_signals,
                full_timeline=full_data.index,
                test_period_start=test_start_idx,
                test_period_end=test_end_idx,
                prediction_count=len(predictions),
                alignment_metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error in signal alignment: {str(e)}")
            raise DataValidationError(f"Signal alignment failed: {str(e)}")
    
    def convert_predictions_to_signals(
        self, 
        predictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert ML prediction values to VectorBT entry/exit signals.
        
        Converts prediction values according to the standard mapping:
        - Value 2: Buy signal (entry_signal = True)
        - Value 0: Sell signal (exit_signal = True)  
        - Value 1: Hold signal (both signals = False)
        
        Args:
            predictions: Array of prediction values (0, 1, 2)
            
        Returns:
            Tuple of (entry_signals, exit_signals) boolean arrays
            
        Raises:
            DataValidationError: If predictions contain invalid values
        """
        try:
            # Validate prediction values
            valid_values = {0, 1, 2}
            unique_values = set(np.unique(predictions))
            invalid_values = unique_values - valid_values
            
            if invalid_values:
                raise DataValidationError(
                    f"Invalid prediction values found: {invalid_values}. "
                    f"Expected values: {valid_values}"
                )
            
            # Convert predictions to boolean signals
            entry_signals = (predictions == 2)  # Buy signal
            exit_signals = (predictions == 0)   # Sell signal
            # Hold signals (predictions == 1) result in both signals being False
            
            # Log signal distribution
            entry_count = np.sum(entry_signals)
            exit_count = np.sum(exit_signals)
            hold_count = len(predictions) - entry_count - exit_count
            
            self.logger.debug(
                f"Signal conversion: {entry_count} entries, {exit_count} exits, "
                f"{hold_count} holds from {len(predictions)} predictions"
            )
            
            return entry_signals, exit_signals
            
        except Exception as e:
            self.logger.error(f"Error converting predictions to signals: {str(e)}")
            raise DataValidationError(f"Signal conversion failed: {str(e)}")
    
    def create_full_signal_arrays(self, full_data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Create full-sized boolean signal arrays initialized to False.
        
        Creates pandas Series with the same index as the full dataset,
        initialized to False. These arrays will be populated with actual
        signals only in the test period.
        
        Args:
            full_data: Complete historical dataset
            
        Returns:
            Tuple of (entry_signals, exit_signals) Series initialized to False
        """
        try:
            # Create boolean series initialized to False
            entry_signals = pd.Series(False, index=full_data.index, name='entry_signals')
            exit_signals = pd.Series(False, index=full_data.index, name='exit_signals')
            
            self.logger.debug(
                f"Created full signal arrays with {len(entry_signals)} elements"
            )
            
            return entry_signals, exit_signals
            
        except Exception as e:
            self.logger.error(f"Error creating signal arrays: {str(e)}")
            raise DataValidationError(f"Signal array creation failed: {str(e)}")
    
    def validate_signal_alignment(
        self, 
        entry_signals: pd.Series, 
        exit_signals: pd.Series, 
        full_data: pd.DataFrame
    ) -> None:
        """
        Validate that signal alignment is correct and consistent.
        
        Performs comprehensive validation of the aligned signals including:
        - Array length consistency
        - Index alignment
        - Signal value validation
        - Logical consistency checks
        
        Args:
            entry_signals: Aligned entry signals
            exit_signals: Aligned exit signals
            full_data: Original full dataset
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            # Check array lengths
            if len(entry_signals) != len(full_data):
                raise DataValidationError(
                    f"Entry signals length mismatch: {len(entry_signals)} != {len(full_data)}"
                )
            
            if len(exit_signals) != len(full_data):
                raise DataValidationError(
                    f"Exit signals length mismatch: {len(exit_signals)} != {len(full_data)}"
                )
            
            # Check index alignment
            if not entry_signals.index.equals(full_data.index):
                raise DataValidationError("Entry signals index does not match full data index")
            
            if not exit_signals.index.equals(full_data.index):
                raise DataValidationError("Exit signals index does not match full data index")
            
            # Check signal types
            if not entry_signals.dtype == bool:
                raise DataValidationError(f"Entry signals must be boolean, got {entry_signals.dtype}")
            
            if not exit_signals.dtype == bool:
                raise DataValidationError(f"Exit signals must be boolean, got {exit_signals.dtype}")
            
            # Check for simultaneous entry and exit signals (logical inconsistency)
            simultaneous_signals = (entry_signals & exit_signals).sum()
            if simultaneous_signals > 0:
                self.logger.warning(
                    f"Found {simultaneous_signals} simultaneous entry/exit signals. "
                    "This may indicate signal conversion issues."
                )
            
            # Log validation summary
            entry_count = entry_signals.sum()
            exit_count = exit_signals.sum()
            
            self.logger.info(
                f"Signal validation passed: {len(entry_signals)} total points, "
                f"{entry_count} entries, {exit_count} exits"
            )
            
        except Exception as e:
            self.logger.error(f"Signal validation failed: {str(e)}")
            raise DataValidationError(f"Signal validation error: {str(e)}")
    
    def _validate_alignment_inputs(
        self, 
        predictions: np.ndarray, 
        full_data: pd.DataFrame, 
        test_start_idx: int
    ) -> None:
        """
        Validate inputs for signal alignment process.
        
        Args:
            predictions: ML model predictions
            full_data: Complete historical dataset
            test_start_idx: Test period start index
            
        Raises:
            DataValidationError: If inputs are invalid
        """
        # Validate predictions
        if predictions is None or len(predictions) == 0:
            raise DataValidationError("Predictions array is empty or None")
        
        if not isinstance(predictions, np.ndarray):
            raise DataValidationError(f"Predictions must be numpy array, got {type(predictions)}")
        
        # Validate full_data
        if full_data is None or len(full_data) == 0:
            raise DataValidationError("Full data is empty or None")
        
        if not isinstance(full_data, pd.DataFrame):
            raise DataValidationError(f"Full data must be DataFrame, got {type(full_data)}")
        
        # Validate test_start_idx
        if not isinstance(test_start_idx, int):
            raise DataValidationError(f"Test start index must be integer, got {type(test_start_idx)}")
        
        if test_start_idx < 0:
            raise DataValidationError(f"Test start index must be non-negative, got {test_start_idx}")
        
        if test_start_idx >= len(full_data):
            raise DataValidationError(
                f"Test start index {test_start_idx} exceeds data length {len(full_data)}"
            )
        
        # Validate that test period fits within data
        test_end_idx = test_start_idx + len(predictions)
        if test_end_idx > len(full_data):
            raise DataValidationError(
                f"Test period ({test_start_idx} to {test_end_idx}) exceeds "
                f"data length ({len(full_data)})"
            )
    
    def get_alignment_summary(self, aligned_signals: AlignedSignals) -> Dict[str, Any]:
        """
        Generate a summary of the signal alignment results.
        
        Args:
            aligned_signals: Aligned signals object
            
        Returns:
            Dictionary with alignment summary statistics
        """
        try:
            entry_count = aligned_signals.entry_signals.sum()
            exit_count = aligned_signals.exit_signals.sum()
            
            # Calculate signal density in test period
            test_period_length = (aligned_signals.test_period_end - 
                                aligned_signals.test_period_start)
            
            summary = {
                'total_timeline_length': len(aligned_signals.full_timeline),
                'test_period_length': test_period_length,
                'test_period_start': aligned_signals.test_period_start,
                'test_period_end': aligned_signals.test_period_end,
                'prediction_count': aligned_signals.prediction_count,
                'total_entry_signals': entry_count,
                'total_exit_signals': exit_count,
                'entry_signal_density': entry_count / test_period_length if test_period_length > 0 else 0,
                'exit_signal_density': exit_count / test_period_length if test_period_length > 0 else 0,
                'timeline_coverage': test_period_length / len(aligned_signals.full_timeline),
                'alignment_metadata': aligned_signals.alignment_metadata
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating alignment summary: {str(e)}")
            return {'error': str(e)}