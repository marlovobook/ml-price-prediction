"""
Comprehensive Error Handler for VectorBT Visualization Enhancement.

This module provides robust error handling and graceful degradation for visualization
operations, including fallback mechanisms for headless environments and VectorBT failures.
"""

import logging
import traceback
import sys
import os
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from ..utils.exceptions import StockPredictorError, DataValidationError, BacktestingError


class ErrorSeverity(Enum):
    """Error severity levels for visualization operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FallbackMode(Enum):
    """Available fallback modes for visualization failures."""
    TEXT_OUTPUT = "text_output"
    BASIC_PLOT = "basic_plot"
    MINIMAL_DATA = "minimal_data"
    GRACEFUL_SKIP = "graceful_skip"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    component: str
    input_data: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    fallback_mode: FallbackMode = FallbackMode.TEXT_OUTPUT
    additional_info: Optional[Dict[str, Any]] = None


@dataclass
class ErrorHandlingResult:
    """Result of error handling operation."""
    success: bool
    fallback_used: bool
    fallback_mode: Optional[FallbackMode] = None
    result_data: Optional[Any] = None
    error_message: Optional[str] = None
    diagnostic_info: Optional[Dict[str, Any]] = None


class VisualizationErrorHandler:
    """
    Comprehensive error handler for VectorBT visualization operations.
    
    Provides graceful degradation when VectorBT plotting fails, text-based output
    alternatives for headless environments, and detailed error logging and diagnostics.
    """
    
    def __init__(self, enable_fallbacks: bool = True, log_level: str = "INFO"):
        """
        Initialize the error handler.
        
        Args:
            enable_fallbacks: Whether to enable fallback mechanisms
            log_level: Logging level for error handler
        """
        self.enable_fallbacks = enable_fallbacks
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Environment detection
        self.is_headless = self._detect_headless_environment()
        self.has_display = self._check_display_availability()
        
        # Error statistics
        self.error_counts = {}
        self.fallback_usage = {}
        
        self.logger.info(
            f"Visualization Error Handler initialized: "
            f"headless={self.is_headless}, display={self.has_display}, "
            f"fallbacks={'enabled' if enable_fallbacks else 'disabled'}"
        )
    
    def handle_visualization_error(
        self,
        error: Exception,
        context: ErrorContext,
        fallback_function: Optional[Callable] = None
    ) -> ErrorHandlingResult:
        """
        Handle visualization errors with appropriate fallback strategies.
        
        Args:
            error: The exception that occurred
            context: Error context information
            fallback_function: Optional fallback function to execute
            
        Returns:
            ErrorHandlingResult with handling outcome
        """
        try:
            # Log the error with full context
            self._log_error_with_context(error, context)
            
            # Update error statistics
            self._update_error_statistics(error, context)
            
            # Determine appropriate fallback strategy
            fallback_mode = self._determine_fallback_mode(error, context)
            
            # Execute fallback if enabled
            if self.enable_fallbacks and fallback_mode != FallbackMode.GRACEFUL_SKIP:
                fallback_result = self._execute_fallback(
                    error, context, fallback_mode, fallback_function
                )
                
                if fallback_result.success:
                    self.logger.info(
                        f"Successfully recovered using {fallback_mode.value} fallback "
                        f"for {context.operation}"
                    )
                    return fallback_result
            
            # No fallback available or fallback failed
            diagnostic_info = self._generate_diagnostic_info(error, context)
            
            return ErrorHandlingResult(
                success=False,
                fallback_used=False,
                error_message=str(error),
                diagnostic_info=diagnostic_info
            )
            
        except Exception as fallback_error:
            self.logger.error(
                f"Error in error handler itself: {str(fallback_error)}"
            )
            return ErrorHandlingResult(
                success=False,
                fallback_used=False,
                error_message=f"Original error: {str(error)}. Handler error: {str(fallback_error)}"
            )
    
    def create_text_based_output(
        self,
        data: Dict[str, Any],
        output_type: str = "portfolio_summary"
    ) -> str:
        """
        Create text-based output alternative for headless environments.
        
        Args:
            data: Data to convert to text output
            output_type: Type of output to generate
            
        Returns:
            Formatted text output string
        """
        try:
            if output_type == "portfolio_summary":
                return self._create_portfolio_text_summary(data)
            elif output_type == "trade_analysis":
                return self._create_trade_analysis_text(data)
            elif output_type == "drawdown_analysis":
                return self._create_drawdown_analysis_text(data)
            elif output_type == "comparison_summary":
                return self._create_comparison_summary_text(data)
            else:
                return self._create_generic_text_output(data)
                
        except Exception as e:
            self.logger.error(f"Error creating text output: {str(e)}")
            return f"Error generating text output: {str(e)}"
    
    def validate_visualization_environment(self) -> Dict[str, Any]:
        """
        Validate the current environment for visualization capabilities.
        
        Returns:
            Dictionary with environment validation results
        """
        validation_result = {
            'environment_suitable': True,
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'capabilities': {}
        }
        
        try:
            # Check display availability
            validation_result['capabilities']['has_display'] = self.has_display
            if not self.has_display:
                validation_result['issues'].append("No display available for interactive plots")
                validation_result['recommendations'].append("Use text-based output or export to files")
            
            # Check headless environment
            validation_result['capabilities']['is_headless'] = self.is_headless
            if self.is_headless:
                validation_result['warnings'].append("Running in headless environment")
                validation_result['recommendations'].append("Consider using non-interactive plot backends")
            
            # Check VectorBT availability
            try:
                import vectorbt as vbt
                validation_result['capabilities']['vectorbt_available'] = True
                validation_result['capabilities']['vectorbt_version'] = vbt.__version__
            except ImportError:
                validation_result['environment_suitable'] = False
                validation_result['issues'].append("VectorBT not available")
                validation_result['recommendations'].append("Install VectorBT: pip install vectorbt")
            
            # Check Plotly availability
            try:
                import plotly
                validation_result['capabilities']['plotly_available'] = True
                validation_result['capabilities']['plotly_version'] = plotly.__version__
            except ImportError:
                validation_result['environment_suitable'] = False
                validation_result['issues'].append("Plotly not available")
                validation_result['recommendations'].append("Install Plotly: pip install plotly")
            
            # Check memory availability
            try:
                import psutil
                memory_info = psutil.virtual_memory()
                validation_result['capabilities']['available_memory_gb'] = memory_info.available / (1024**3)
                
                if memory_info.available < 1024**3:  # Less than 1GB
                    validation_result['warnings'].append("Low memory available (< 1GB)")
                    validation_result['recommendations'].append("Consider data sampling for large datasets")
            except ImportError:
                validation_result['warnings'].append("Cannot check memory usage (psutil not available)")
            
            # Determine overall suitability
            if validation_result['issues']:
                validation_result['environment_suitable'] = False
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating visualization environment: {str(e)}")
            return {
                'environment_suitable': False,
                'issues': [f"Environment validation failed: {str(e)}"],
                'warnings': [],
                'recommendations': ['Check system configuration'],
                'capabilities': {}
            }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive error statistics and diagnostics.
        
        Returns:
            Dictionary with error statistics and patterns
        """
        try:
            total_errors = sum(self.error_counts.values())
            total_fallbacks = sum(self.fallback_usage.values())
            
            statistics = {
                'total_errors': total_errors,
                'total_fallbacks_used': total_fallbacks,
                'error_breakdown': dict(self.error_counts),
                'fallback_breakdown': dict(self.fallback_usage),
                'error_rate_by_type': {},
                'most_common_errors': [],
                'fallback_success_rate': 0.0,
                'environment_info': {
                    'is_headless': self.is_headless,
                    'has_display': self.has_display,
                    'fallbacks_enabled': self.enable_fallbacks
                }
            }
            
            # Calculate error rates
            if total_errors > 0:
                for error_type, count in self.error_counts.items():
                    statistics['error_rate_by_type'][error_type] = count / total_errors
                
                # Find most common errors
                statistics['most_common_errors'] = sorted(
                    self.error_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
                
                # Calculate fallback success rate
                if total_fallbacks > 0:
                    statistics['fallback_success_rate'] = total_fallbacks / total_errors
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error generating error statistics: {str(e)}")
            return {'error': str(e)}
    
    def _detect_headless_environment(self) -> bool:
        """Detect if running in a headless environment."""
        try:
            # Check common headless indicators
            if os.environ.get('DISPLAY') is None and sys.platform.startswith('linux'):
                return True
            
            if os.environ.get('SSH_CONNECTION') is not None:
                return True
            
            # Check for common CI/CD environments
            ci_indicators = ['CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS', 'JENKINS_URL']
            if any(os.environ.get(indicator) for indicator in ci_indicators):
                return True
            
            return False
            
        except Exception:
            return False
    
    def _check_display_availability(self) -> bool:
        """Check if display is available for interactive plots."""
        try:
            if self.is_headless:
                return False
            
            # Try to import and test display-related modules
            try:
                import matplotlib.pyplot as plt
                # Try to create a figure (this will fail if no display)
                fig = plt.figure()
                plt.close(fig)
                return True
            except Exception:
                pass
            
            # Check for display environment variable
            if os.environ.get('DISPLAY'):
                return True
            
            return False
            
        except Exception:
            return False
    
    def _log_error_with_context(self, error: Exception, context: ErrorContext) -> None:
        """Log error with full context information."""
        try:
            error_info = {
                'operation': context.operation,
                'component': context.component,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'severity': context.severity.value,
                'error_code': context.error_code
            }
            
            # Log based on severity
            if context.severity == ErrorSeverity.CRITICAL:
                self.logger.critical(f"CRITICAL ERROR in {context.operation}: {str(error)}")
            elif context.severity == ErrorSeverity.HIGH:
                self.logger.error(f"HIGH SEVERITY ERROR in {context.operation}: {str(error)}")
            elif context.severity == ErrorSeverity.MEDIUM:
                self.logger.warning(f"MEDIUM SEVERITY ERROR in {context.operation}: {str(error)}")
            else:
                self.logger.info(f"LOW SEVERITY ERROR in {context.operation}: {str(error)}")
            
            # Log additional context if available
            if context.additional_info:
                self.logger.debug(f"Additional context: {context.additional_info}")
            
            # Log stack trace for debugging
            if context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                self.logger.debug(f"Stack trace: {traceback.format_exc()}")
            
        except Exception as log_error:
            # Fallback logging if structured logging fails
            print(f"Error logging failed: {str(log_error)}. Original error: {str(error)}")
    
    def _update_error_statistics(self, error: Exception, context: ErrorContext) -> None:
        """Update error statistics for monitoring."""
        try:
            error_type = type(error).__name__
            operation = context.operation
            
            # Update error counts
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            self.error_counts[f"{operation}_{error_type}"] = self.error_counts.get(f"{operation}_{error_type}", 0) + 1
            
        except Exception:
            pass  # Don't let statistics updates cause additional errors
    
    def _determine_fallback_mode(self, error: Exception, context: ErrorContext) -> FallbackMode:
        """Determine the appropriate fallback mode for the error."""
        try:
            # Use context-specified fallback mode if available
            if context.fallback_mode:
                return context.fallback_mode
            
            # Determine based on error type and environment
            if isinstance(error, (ImportError, ModuleNotFoundError)):
                return FallbackMode.TEXT_OUTPUT
            
            if self.is_headless or not self.has_display:
                return FallbackMode.TEXT_OUTPUT
            
            if isinstance(error, (DataValidationError, ValueError)):
                return FallbackMode.MINIMAL_DATA
            
            if "plot" in str(error).lower() or "display" in str(error).lower():
                return FallbackMode.BASIC_PLOT
            
            # Default fallback
            return FallbackMode.TEXT_OUTPUT
            
        except Exception:
            return FallbackMode.TEXT_OUTPUT
    
    def _execute_fallback(
        self,
        error: Exception,
        context: ErrorContext,
        fallback_mode: FallbackMode,
        fallback_function: Optional[Callable] = None
    ) -> ErrorHandlingResult:
        """Execute the appropriate fallback strategy."""
        try:
            # Update fallback usage statistics
            self.fallback_usage[fallback_mode.value] = self.fallback_usage.get(fallback_mode.value, 0) + 1
            
            if fallback_function:
                # Use provided fallback function
                result = fallback_function(error, context, fallback_mode)
                return ErrorHandlingResult(
                    success=True,
                    fallback_used=True,
                    fallback_mode=fallback_mode,
                    result_data=result
                )
            
            # Use built-in fallback strategies
            if fallback_mode == FallbackMode.TEXT_OUTPUT:
                result = self._create_text_fallback(context)
            elif fallback_mode == FallbackMode.BASIC_PLOT:
                result = self._create_basic_plot_fallback(context)
            elif fallback_mode == FallbackMode.MINIMAL_DATA:
                result = self._create_minimal_data_fallback(context)
            else:
                result = None
            
            return ErrorHandlingResult(
                success=result is not None,
                fallback_used=True,
                fallback_mode=fallback_mode,
                result_data=result
            )
            
        except Exception as fallback_error:
            self.logger.error(f"Fallback execution failed: {str(fallback_error)}")
            return ErrorHandlingResult(
                success=False,
                fallback_used=False,
                error_message=f"Fallback failed: {str(fallback_error)}"
            )
    
    def _create_text_fallback(self, context: ErrorContext) -> str:
        """Create text-based fallback output."""
        try:
            output = [
                f"=== {context.operation.upper()} SUMMARY ===",
                f"Component: {context.component}",
                f"Operation failed, showing text summary instead.",
                ""
            ]
            
            if context.input_data:
                output.append("Input Data Summary:")
                for key, value in context.input_data.items():
                    if isinstance(value, (pd.DataFrame, pd.Series)):
                        output.append(f"  {key}: {type(value).__name__} with {len(value)} elements")
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        output.append(f"  {key}: {type(value).__name__} with {len(value)} elements")
                    else:
                        output.append(f"  {key}: {str(value)[:100]}...")
                output.append("")
            
            output.extend([
                "Visualization failed - using text output instead.",
                "For full visualization, please check system configuration.",
                "=" * 50
            ])
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Text fallback creation failed: {str(e)}"
    
    def _create_basic_plot_fallback(self, context: ErrorContext) -> Any:
        """Create basic plot fallback using matplotlib."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Visualization Error in {context.operation}\n\nUsing basic plot fallback",
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title(f"Fallback Plot - {context.operation}")
            ax.axis('off')
            
            return fig
            
        except Exception as e:
            self.logger.warning(f"Basic plot fallback failed: {str(e)}")
            return None
    
    def _create_minimal_data_fallback(self, context: ErrorContext) -> Dict[str, Any]:
        """Create minimal data structure fallback."""
        return {
            'operation': context.operation,
            'component': context.component,
            'status': 'fallback_data',
            'message': 'Original operation failed, minimal data provided',
            'input_summary': self._summarize_input_data(context.input_data) if context.input_data else None
        }
    
    def _summarize_input_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of input data for fallback."""
        try:
            summary = {}
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    summary[key] = {
                        'type': 'DataFrame',
                        'shape': value.shape,
                        'columns': list(value.columns)[:5]  # First 5 columns
                    }
                elif isinstance(value, pd.Series):
                    summary[key] = {
                        'type': 'Series',
                        'length': len(value),
                        'dtype': str(value.dtype)
                    }
                elif isinstance(value, (list, tuple, np.ndarray)):
                    summary[key] = {
                        'type': type(value).__name__,
                        'length': len(value)
                    }
                else:
                    summary[key] = {
                        'type': type(value).__name__,
                        'value': str(value)[:100]
                    }
            return summary
        except Exception:
            return {'error': 'Could not summarize input data'}
    
    def _generate_diagnostic_info(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Generate comprehensive diagnostic information."""
        try:
            return {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'operation': context.operation,
                'component': context.component,
                'severity': context.severity.value,
                'environment': {
                    'is_headless': self.is_headless,
                    'has_display': self.has_display,
                    'python_version': sys.version,
                    'platform': sys.platform
                },
                'stack_trace': traceback.format_exc(),
                'suggestions': self._generate_error_suggestions(error, context)
            }
        except Exception:
            return {'error': 'Could not generate diagnostic info'}
    
    def _generate_error_suggestions(self, error: Exception, context: ErrorContext) -> List[str]:
        """Generate suggestions for resolving the error."""
        suggestions = []
        
        try:
            error_str = str(error).lower()
            
            if isinstance(error, ImportError):
                suggestions.append("Install missing dependencies")
                suggestions.append("Check virtual environment activation")
            
            if "display" in error_str or "gui" in error_str:
                suggestions.append("Run in environment with display support")
                suggestions.append("Use text-based output alternatives")
            
            if "memory" in error_str:
                suggestions.append("Reduce dataset size")
                suggestions.append("Use data sampling")
                suggestions.append("Increase available memory")
            
            if isinstance(error, DataValidationError):
                suggestions.append("Check input data format and types")
                suggestions.append("Validate data completeness")
            
            if not suggestions:
                suggestions.append("Check system logs for more details")
                suggestions.append("Verify input data and configuration")
            
        except Exception:
            suggestions = ["Unable to generate specific suggestions"]
        
        return suggestions
    
    def _create_portfolio_text_summary(self, data: Dict[str, Any]) -> str:
        """Create text summary for portfolio data."""
        try:
            lines = ["=== PORTFOLIO PERFORMANCE SUMMARY ===", ""]
            
            if 'metrics_summary' in data:
                metrics = data['metrics_summary']
                lines.extend([
                    "Performance Metrics:",
                    f"  Total Return: {metrics.get('total_return', 0):.2%}",
                    f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
                    f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}",
                    f"  Number of Trades: {metrics.get('num_trades', 0)}",
                    ""
                ])
            
            if 'plot_data' in data and 'portfolio_value' in data['plot_data']:
                portfolio_value = data['plot_data']['portfolio_value']
                if hasattr(portfolio_value, 'iloc'):
                    lines.extend([
                        "Portfolio Value:",
                        f"  Initial Value: ${portfolio_value.iloc[0]:,.2f}",
                        f"  Final Value: ${portfolio_value.iloc[-1]:,.2f}",
                        f"  Peak Value: ${portfolio_value.max():,.2f}",
                        ""
                    ])
            
            lines.append("Note: Full visualization unavailable - text summary provided")
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error creating portfolio summary: {str(e)}"
    
    def _create_trade_analysis_text(self, data: Dict[str, Any]) -> str:
        """Create text summary for trade analysis data."""
        try:
            lines = ["=== TRADE ANALYSIS SUMMARY ===", ""]
            
            if 'metrics_summary' in data:
                metrics = data['metrics_summary']
                lines.extend([
                    "Trade Statistics:",
                    f"  Total Trades: {metrics.get('num_trades', 0)}",
                    f"  Win Rate: {metrics.get('win_rate', 0):.2%}",
                    f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}",
                    f"  Best Trade: ${metrics.get('best_trade', 0):,.2f}",
                    f"  Worst Trade: ${metrics.get('worst_trade', 0):,.2f}",
                    ""
                ])
            
            lines.append("Note: Full trade visualization unavailable - text summary provided")
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error creating trade analysis summary: {str(e)}"
    
    def _create_drawdown_analysis_text(self, data: Dict[str, Any]) -> str:
        """Create text summary for drawdown analysis data."""
        try:
            lines = ["=== DRAWDOWN ANALYSIS SUMMARY ===", ""]
            
            if 'metrics_summary' in data:
                metrics = data['metrics_summary']
                lines.extend([
                    "Drawdown Statistics:",
                    f"  Maximum Drawdown: {metrics.get('max_drawdown_pct', 0):.2%}",
                    f"  Average Drawdown: {metrics.get('avg_drawdown_pct', 0):.2%}",
                    f"  Average Recovery Time: {metrics.get('avg_recovery_time', 0):.0f} days",
                    f"  Number of Drawdown Periods: {metrics.get('num_drawdown_periods', 0)}",
                    f"  Time Underwater: {metrics.get('time_underwater_pct', 0):.1%}",
                    ""
                ])
            
            lines.append("Note: Full drawdown visualization unavailable - text summary provided")
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error creating drawdown analysis summary: {str(e)}"
    
    def _create_comparison_summary_text(self, data: Dict[str, Any]) -> str:
        """Create text summary for strategy comparison data."""
        try:
            lines = ["=== STRATEGY COMPARISON SUMMARY ===", ""]
            
            if 'metrics_summary' in data:
                metrics = data['metrics_summary']
                lines.extend([
                    "Comparison Overview:",
                    f"  Strategies Analyzed: {metrics.get('num_strategies_compared', 0)}",
                    f"  Best Performer: {metrics.get('best_performer', 'N/A')}",
                    f"  Most Consistent: {metrics.get('most_consistent', 'N/A')}",
                    f"  Highest Risk: {metrics.get('highest_risk', 'N/A')}",
                    ""
                ])
            
            lines.append("Note: Full comparison visualization unavailable - text summary provided")
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error creating comparison summary: {str(e)}"
    
    def _create_generic_text_output(self, data: Dict[str, Any]) -> str:
        """Create generic text output for any data structure."""
        try:
            lines = ["=== DATA SUMMARY ===", ""]
            
            for key, value in data.items():
                if isinstance(value, dict):
                    lines.append(f"{key}:")
                    for sub_key, sub_value in value.items():
                        lines.append(f"  {sub_key}: {str(sub_value)[:100]}")
                else:
                    lines.append(f"{key}: {str(value)[:100]}")
            
            lines.extend(["", "Note: Full visualization unavailable - text summary provided"])
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error creating generic text output: {str(e)}"