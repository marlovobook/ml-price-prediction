"""
VectorBT Visualization Engine for Stock Direction Predictor.

This module provides comprehensive visualization capabilities using VectorBT's
built-in plotting functionality for portfolio analysis and trading strategy evaluation.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import time
from pathlib import Path

from .signal_alignment import SignalAlignmentEngine, AlignedSignals
from .portfolio_config import PortfolioConfig, PlotConfig, VisualizationResult
from .error_handler import VisualizationErrorHandler, ErrorContext, ErrorSeverity, FallbackMode
from .performance_optimizer import VisualizationPerformanceOptimizer, OptimizationConfig, SamplingStrategy
from ..utils.exceptions import BacktestingError, DataValidationError


class VectorBTVisualizationEngine:
    """
    Comprehensive VectorBT visualization engine for trading strategy analysis.
    
    This engine provides professional-grade portfolio visualization using VectorBT's
    built-in plotting capabilities, with proper signal alignment and realistic
    trading parameters.
    """
    
    def __init__(self, 
                 portfolio_config: Optional[PortfolioConfig] = None,
                 plot_config: Optional[PlotConfig] = None,
                 optimization_config: Optional[OptimizationConfig] = None,
                 enable_error_handling: bool = True,
                 enable_performance_optimization: bool = True):
        """
        Initialize the VectorBT Visualization Engine.
        
        Args:
            portfolio_config: Configuration for portfolio creation
            plot_config: Configuration for plot styling and behavior
            optimization_config: Configuration for performance optimization
            enable_error_handling: Whether to enable comprehensive error handling
            enable_performance_optimization: Whether to enable performance optimization
        """
        self.portfolio_config = portfolio_config or PortfolioConfig()
        self.plot_config = plot_config or PlotConfig()
        self.signal_aligner = SignalAlignmentEngine()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize error handler and performance optimizer
        self.error_handler = VisualizationErrorHandler(
            enable_fallbacks=enable_error_handling
        ) if enable_error_handling else None
        
        self.performance_optimizer = VisualizationPerformanceOptimizer(
            config=optimization_config or OptimizationConfig()
        ) if enable_performance_optimization else None
        
        # Validate configurations
        self.portfolio_config.validate()
        self.plot_config.validate()
        
        # Validate visualization environment
        if self.error_handler:
            env_validation = self.error_handler.validate_visualization_environment()
            if not env_validation['environment_suitable']:
                self.logger.warning(
                    f"Visualization environment issues detected: {env_validation['issues']}"
                )
                for recommendation in env_validation['recommendations']:
                    self.logger.info(f"Recommendation: {recommendation}")
        
        self.logger.info(
            f"VectorBT Visualization Engine initialized: "
            f"error_handling={'enabled' if enable_error_handling else 'disabled'}, "
            f"performance_optimization={'enabled' if enable_performance_optimization else 'disabled'}"
        )
    
    def create_portfolio_from_predictions(
        self,
        predictions: np.ndarray,
        price_data: pd.DataFrame,
        test_start_idx: int,
        symbol: str = 'ASSET'
    ) -> vbt.Portfolio:
        """
        Create VectorBT portfolio from ML predictions with proper signal alignment.
        
        This method implements the core functionality shown in your example:
        - Aligns predictions to full historical timeline
        - Converts predictions to entry/exit signals
        - Creates VectorBT portfolio with realistic parameters
        
        Args:
            predictions: ML model predictions (0=sell, 1=hold, 2=buy)
            price_data: Historical price data with 'Close' column
            test_start_idx: Index where test period begins
            symbol: Asset symbol for labeling
            
        Returns:
            VectorBT Portfolio object ready for visualization
            
        Raises:
            BacktestingError: If portfolio creation fails
        """
        try:
            start_time = time.time()
            
            # Ensure we have the required price column
            if 'Close' not in price_data.columns and 'close' not in price_data.columns:
                raise DataValidationError("Price data must contain 'Close' or 'close' column")
            
            # Standardize column name
            close_prices = price_data.get('Close', price_data.get('close'))
            
            # Align predictions to full timeline
            self.logger.info(f"Aligning {len(predictions)} predictions to {len(price_data)} timeline points")
            aligned_signals = self.signal_aligner.align_predictions_to_timeline(
                predictions, price_data, test_start_idx
            )
            
            # Calculate position sizes based on strategy
            # Calculate volatility for dynamic sizing if needed
            volatility = None
            if self.portfolio_config.size_strategy in ['volatility_target', 'risk_parity']:
                returns = close_prices.pct_change(fill_method=None).dropna()
                volatility = returns.rolling(window=20, min_periods=10).std() * np.sqrt(252)
            
            position_sizes = self._calculate_position_sizes(
                close_prices, 
                volatility=volatility
            )
            
            # Get VectorBT parameters
            vbt_params = self.portfolio_config.to_vectorbt_params()
            
            # Create VectorBT portfolio - this matches your example pattern
            self.logger.info("Creating VectorBT portfolio with realistic parameters")
            portfolio = vbt.Portfolio.from_signals(
                close=close_prices,
                entries=aligned_signals.entry_signals,
                exits=aligned_signals.exit_signals,
                size=position_sizes,
                size_type='amount',  # Use dollar amounts
                **vbt_params
            )
            
            creation_time = time.time() - start_time
            self.logger.info(
                f"Portfolio created successfully in {creation_time:.2f}s "
                f"({portfolio.trades.count()} trades generated)"
            )
            
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Error creating portfolio from predictions: {str(e)}")
            raise BacktestingError(f"Portfolio creation failed: {str(e)}")
    
    def generate_portfolio_plot(
        self, 
        portfolio: vbt.Portfolio,
        title: Optional[str] = None
    ) -> VisualizationResult:
        """
        Generate interactive portfolio performance plot using VectorBT.
        
        This method creates the main portfolio visualization showing:
        - Portfolio value over time
        - Entry and exit points with distinct markers
        - Performance metrics overlay
        - Interactive hover information
        
        Args:
            portfolio: VectorBT portfolio object
            title: Optional plot title
            
        Returns:
            VisualizationResult with plot object and metadata
        """
        # Use performance monitoring if available
        if self.performance_optimizer:
            return self._generate_portfolio_plot_with_monitoring(portfolio, title)
        else:
            return self._generate_portfolio_plot_basic(portfolio, title)
    
    def _generate_portfolio_plot_with_monitoring(
        self, 
        portfolio: vbt.Portfolio,
        title: Optional[str] = None
    ) -> VisualizationResult:
        """Generate portfolio plot with performance monitoring and error handling."""
        def _plot_operation():
            return self._generate_portfolio_plot_basic(portfolio, title)
        
        try:
            # Monitor performance
            result, metrics = self.performance_optimizer.monitor_operation_performance(
                _plot_operation, "generate_portfolio_plot"
            )
            
            # Add performance metrics to result
            if result and result.success:
                if not hasattr(result, 'performance_metrics'):
                    result.performance_metrics = {}
                result.performance_metrics.update({
                    'operation_duration_seconds': metrics.duration_seconds,
                    'memory_usage_mb': metrics.memory_after_mb - metrics.memory_before_mb,
                    'optimization_applied': metrics.optimization_applied
                })
            
            return result
            
        except Exception as e:
            # Handle error with fallback
            if self.error_handler:
                context = ErrorContext(
                    operation="generate_portfolio_plot",
                    component="VectorBTVisualizationEngine",
                    input_data={'portfolio_type': type(portfolio).__name__},
                    severity=ErrorSeverity.MEDIUM,
                    fallback_mode=FallbackMode.TEXT_OUTPUT
                )
                
                error_result = self.error_handler.handle_visualization_error(
                    e, context, self._create_portfolio_plot_fallback
                )
                
                if error_result.success:
                    return VisualizationResult(
                        plot_object=error_result.result_data,
                        plot_data={},
                        metrics_summary={},
                        export_paths={},
                        generation_time=0.0,
                        success=True,
                        error_message="Used fallback due to visualization error"
                    )
            
            # Fallback to basic error handling
            return VisualizationResult(
                plot_object=None,
                plot_data={},
                metrics_summary={},
                export_paths={},
                generation_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    def _generate_portfolio_plot_basic(
        self, 
        portfolio: vbt.Portfolio,
        title: Optional[str] = None
    ) -> VisualizationResult:
        """Basic portfolio plot generation without monitoring."""
        try:
            start_time = time.time()
            
            # Generate the main portfolio plot - this is your port.plot().show() equivalent
            self.logger.info("Generating interactive portfolio performance plot with trade markers")
            
            # Optimize portfolio data if performance optimizer is available
            portfolio_value = portfolio.value()
            if self.performance_optimizer:
                portfolio_value, opt_info = self.performance_optimizer.optimize_dataset_for_visualization(
                    portfolio_value, "portfolio_plot"
                )
                if opt_info['optimization_applied']:
                    self.logger.info(
                        f"Portfolio data optimized: {opt_info['original_size']} -> "
                        f"{opt_info['final_size']} points (ratio: {opt_info['sampling_ratio']:.3f})"
                    )
            
            # Create the base plot using plotly directly to avoid widget dependencies
            plot_obj = self._create_base_portfolio_plot_optimized(portfolio, portfolio_value)
            
            # Add trade visualization and annotation features (Requirements 3.4, 3.5, 3.6)
            plot_obj = self._add_trade_markers_and_annotations(portfolio, plot_obj)
            
            # Extract plot data for analysis
            plot_data = {
                'portfolio_value': portfolio_value,
                'returns': portfolio.returns(),
                'cash': portfolio.cash()
            }
            
            # Try to add positions data if available
            try:
                plot_data['positions'] = portfolio.positions.value()
            except AttributeError:
                # VectorBT API might be different, skip positions data
                self.logger.debug("Positions data not available in this VectorBT version")
            
            # Calculate key metrics for overlay
            metrics_summary = self._extract_portfolio_metrics(portfolio)
            
            # Add title if provided
            if title:
                plot_obj.update_layout(title=title)
            
            generation_time = time.time() - start_time
            
            self.logger.info(f"Enhanced portfolio plot generated successfully in {generation_time:.2f}s")
            
            return VisualizationResult(
                plot_object=plot_obj,
                plot_data=plot_data,
                metrics_summary=metrics_summary,
                export_paths={},
                generation_time=generation_time,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio plot: {str(e)}")
            return VisualizationResult(
                plot_object=None,
                plot_data={},
                metrics_summary={},
                export_paths={},
                generation_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def generate_drawdown_plot(self, portfolio: vbt.Portfolio) -> VisualizationResult:
        """
        Generate enhanced drawdown analysis plot with detailed period highlighting and recovery analysis.
        
        This method implements Requirements 4.1, 4.2, 4.3, 4.4:
        - Generate drawdown plots showing portfolio decline from peak
        - Highlight maximum drawdown period with distinct coloring
        - Visualize time to recover from drawdowns
        - Display average drawdown duration and depth
        - Provide both absolute and percentage drawdown views
        
        Args:
            portfolio: VectorBT portfolio object
            
        Returns:
            VisualizationResult with enhanced drawdown plot
        """
        try:
            start_time = time.time()
            
            self.logger.info("Generating enhanced drawdown analysis plot with recovery visualization")
            
            # Get portfolio value and calculate drawdown data
            portfolio_value = portfolio.value()
            
            # Calculate detailed drawdown metrics
            drawdown_data = self._calculate_detailed_drawdown_metrics(portfolio_value)
            
            # Ensure portfolio_value is included in drawdown_data
            drawdown_data['portfolio_value'] = portfolio_value
            
            # Create enhanced drawdown plot with recovery visualization
            plot_obj = self._create_enhanced_drawdown_plot(portfolio, drawdown_data)
            
            # Extract comprehensive plot data
            plot_data = {
                'portfolio_value': portfolio_value,
                'drawdown_pct': drawdown_data['drawdown_pct'],
                'drawdown_abs': drawdown_data['drawdown_abs'],
                'underwater_curve': drawdown_data['underwater_curve'],
                'running_max': drawdown_data['running_max'],
                'drawdown_periods': drawdown_data['drawdown_periods'],
                'recovery_periods': drawdown_data['recovery_periods']
            }
            
            # Calculate comprehensive drawdown metrics (Requirement 4.4)
            metrics_summary = {
                'max_drawdown_pct': drawdown_data['max_drawdown_pct'],
                'max_drawdown_abs': drawdown_data['max_drawdown_abs'],
                'avg_drawdown_pct': drawdown_data['avg_drawdown_pct'],
                'avg_drawdown_duration': drawdown_data['avg_drawdown_duration'],
                'avg_recovery_time': drawdown_data['avg_recovery_time'],
                'max_drawdown_duration': drawdown_data['max_drawdown_duration'],
                'max_recovery_time': drawdown_data['max_recovery_time'],
                'num_drawdown_periods': len(drawdown_data['drawdown_periods']),
                'time_underwater_pct': drawdown_data['time_underwater_pct'],
                'recovery_factor': drawdown_data['recovery_factor']
            }
            
            generation_time = time.time() - start_time
            
            self.logger.info(
                f"Enhanced drawdown plot generated in {generation_time:.2f}s "
                f"({metrics_summary['num_drawdown_periods']} drawdown periods identified)"
            )
            
            return VisualizationResult(
                plot_object=plot_obj,
                plot_data=plot_data,
                metrics_summary=metrics_summary,
                export_paths={},
                generation_time=generation_time,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error generating drawdown plot: {str(e)}")
            return VisualizationResult(
                plot_object=None,
                plot_data={},
                metrics_summary={},
                export_paths={},
                generation_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def generate_trade_analysis_plot(self, portfolio: vbt.Portfolio) -> VisualizationResult:
        """
        Generate trade performance analysis plot.
        
        Args:
            portfolio: VectorBT portfolio object
            
        Returns:
            VisualizationResult with trade analysis plot
        """
        try:
            start_time = time.time()
            
            self.logger.info("Generating trade analysis plot")
            
            # Get trade data
            trades = portfolio.trades
            
            if trades.count() == 0:
                self.logger.warning("No trades found for analysis")
                return VisualizationResult(
                    plot_object=None,
                    plot_data={},
                    metrics_summary={'num_trades': 0},
                    export_paths={},
                    generation_time=time.time() - start_time,
                    success=False,
                    error_message="No trades available for analysis"
                )
            
            # Create trade analysis plot
            plot_obj = trades.plot(
                template=self.plot_config.template,
                width=self.plot_config.width,
                height=self.plot_config.height
            )
            
            # Extract trade data
            plot_data = {
                'trade_pnl': trades.pnl,
                'trade_returns': trades.return_pct,
                'trade_duration': trades.duration
            }
            
            # Calculate trade metrics
            metrics_summary = {
                'num_trades': trades.count(),
                'win_rate': trades.win_rate(),
                'profit_factor': trades.profit_factor(),
                'avg_trade_pnl': trades.pnl.mean(),
                'best_trade': trades.pnl.max(),
                'worst_trade': trades.pnl.min()
            }
            
            generation_time = time.time() - start_time
            
            return VisualizationResult(
                plot_object=plot_obj,
                plot_data=plot_data,
                metrics_summary=metrics_summary,
                export_paths={},
                generation_time=generation_time,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error generating trade analysis plot: {str(e)}")
            return VisualizationResult(
                plot_object=None,
                plot_data={},
                metrics_summary={},
                export_paths={},
                generation_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def generate_comparison_plot(
        self, 
        portfolios: Dict[str, vbt.Portfolio],
        title: str = "Multi-Strategy Comparison Analysis"
    ) -> VisualizationResult:
        """
        Generate comprehensive multi-strategy comparison plot.
        
        This method implements Requirements 8.1, 8.2, 8.4, 8.5:
        - Generate side-by-side portfolio performance plots
        - Overlay multiple portfolio equity curves on single plot
        - Show performance ranking and statistical comparison displays
        - Rank strategies by configurable performance criteria
        
        Args:
            portfolios: Dictionary of named portfolio objects
            title: Plot title
            
        Returns:
            VisualizationResult with enhanced comparison plot
        """
        try:
            start_time = time.time()
            
            self.logger.info(f"Generating enhanced comparison plot for {len(portfolios)} strategies")
            
            if len(portfolios) == 0:
                raise DataValidationError("No portfolios provided for comparison")
            
            if len(portfolios) == 1:
                self.logger.warning("Only one portfolio provided - comparison will be limited")
            
            # Extract portfolio data and calculate comprehensive metrics
            portfolio_data = {}
            comparison_metrics = {}
            
            for name, portfolio in portfolios.items():
                portfolio_value = portfolio.value()
                portfolio_returns = portfolio.returns()
                
                portfolio_data[name] = {
                    'portfolio': portfolio,
                    'value': portfolio_value,
                    'returns': portfolio_returns,
                    'normalized_value': portfolio_value / portfolio_value.iloc[0] * 100  # Normalize to 100
                }
                
                # Calculate comprehensive metrics for ranking (Requirement 8.5)
                comparison_metrics[name] = self._calculate_strategy_comparison_metrics(portfolio)
            
            # Create enhanced comparison visualization
            plot_obj = self._create_enhanced_comparison_plot(portfolio_data, comparison_metrics, title)
            
            # Compile comprehensive plot data
            plot_data = {
                'portfolio_values': pd.DataFrame({name: data['value'] for name, data in portfolio_data.items()}),
                'normalized_values': pd.DataFrame({name: data['normalized_value'] for name, data in portfolio_data.items()}),
                'returns_comparison': pd.DataFrame({name: data['returns'] for name, data in portfolio_data.items()}),
                'metrics_comparison': pd.DataFrame(comparison_metrics).T,
                'strategy_rankings': self._calculate_strategy_rankings(comparison_metrics)
            }
            
            # Calculate summary metrics for all strategies
            metrics_summary = self._calculate_comparison_summary_metrics(comparison_metrics, portfolio_data)
            
            generation_time = time.time() - start_time
            
            self.logger.info(
                f"Enhanced comparison plot generated in {generation_time:.2f}s "
                f"(analyzed {len(portfolios)} strategies)"
            )
            
            return VisualizationResult(
                plot_object=plot_obj,
                plot_data=plot_data,
                metrics_summary=metrics_summary,
                export_paths={},
                generation_time=generation_time,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced comparison plot: {str(e)}")
            return VisualizationResult(
                plot_object=None,
                plot_data={},
                metrics_summary={},
                export_paths={},
                generation_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def generate_comparative_risk_visualization(
        self, 
        portfolios: Dict[str, vbt.Portfolio],
        title: str = "Comparative Risk Analysis"
    ) -> VisualizationResult:
        """
        Generate comparative risk visualization across multiple strategies.
        
        This method creates comprehensive risk comparison visualizations including:
        - Side-by-side drawdown analysis
        - Risk metric comparison tables
        - Recovery time analysis across strategies
        - Risk-adjusted performance comparison
        
        Args:
            portfolios: Dictionary of named portfolio objects
            title: Plot title
            
        Returns:
            VisualizationResult with comparative risk visualization
        """
        try:
            start_time = time.time()
            
            self.logger.info(f"Generating comparative risk visualization for {len(portfolios)} strategies")
            
            if len(portfolios) == 0:
                raise DataValidationError("No portfolios provided for risk comparison")
            
            # Calculate risk metrics for all portfolios
            risk_comparison_data = {}
            for name, portfolio in portfolios.items():
                portfolio_value = portfolio.value()
                drawdown_data = self._calculate_detailed_drawdown_metrics(portfolio_value)
                
                risk_comparison_data[name] = {
                    'portfolio': portfolio,
                    'drawdown_data': drawdown_data,
                    'risk_metrics': self._calculate_risk_metrics(portfolio, drawdown_data)
                }
            
            # Create comparative risk visualization
            plot_obj = self._create_comparative_risk_plot(risk_comparison_data, title)
            
            # Compile comparative plot data
            plot_data = {
                'risk_metrics_comparison': self._compile_risk_metrics_table(risk_comparison_data),
                'drawdown_comparison': self._compile_drawdown_comparison(risk_comparison_data),
                'recovery_analysis': self._compile_recovery_analysis(risk_comparison_data)
            }
            
            # Calculate comparative metrics summary
            metrics_summary = self._calculate_comparative_metrics_summary(risk_comparison_data)
            
            generation_time = time.time() - start_time
            
            self.logger.info(f"Comparative risk visualization generated in {generation_time:.2f}s")
            
            return VisualizationResult(
                plot_object=plot_obj,
                plot_data=plot_data,
                metrics_summary=metrics_summary,
                export_paths={},
                generation_time=generation_time,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error generating comparative risk visualization: {str(e)}")
            return VisualizationResult(
                plot_object=None,
                plot_data={},
                metrics_summary={},
                export_paths={},
                generation_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def show_plot(self, result: VisualizationResult) -> None:
        """
        Display the generated plot (equivalent to .show() in your example).
        
        Args:
            result: VisualizationResult containing the plot object
        """
        if not result.success or result.plot_object is None:
            self.logger.error(f"Cannot show plot: {result.error_message}")
            return
        
        try:
            # This is equivalent to your port.plot().show()
            result.plot_object.show()
            self.logger.info("Plot displayed successfully")
            
        except Exception as e:
            self.logger.error(f"Error displaying plot: {str(e)}")
    
    def _calculate_position_sizes(
        self, 
        prices: pd.Series,
        volatility: Optional[pd.Series] = None,
        risk_metrics: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Calculate position sizes based on configuration strategy.
        
        This implements enhanced size calculation with support for:
        - Fixed amount (matches your example with size=40)
        - Dynamic volatility-based sizing
        - Risk parity sizing
        
        Args:
            prices: Price series
            volatility: Optional volatility series for dynamic sizing
            risk_metrics: Optional risk metrics for advanced sizing
            
        Returns:
            Series of position sizes
        """
        return self.portfolio_config.calculate_position_sizes(
            prices=prices,
            volatility=volatility,
            risk_metrics=risk_metrics
        )
    
    def _add_trade_markers_and_annotations(
        self, 
        portfolio: vbt.Portfolio, 
        plot_obj: Any
    ) -> Any:
        """
        Add trade visualization and annotation features to portfolio plot.
        
        This method implements Requirements 3.4, 3.5, 3.6:
        - Entry points with distinct markers (green triangles)
        - Exit points with distinct markers (red triangles)  
        - Performance metrics overlay
        - Interactive hover information
        
        Args:
            portfolio: VectorBT portfolio object
            plot_obj: Base plot object from VectorBT
            
        Returns:
            Enhanced plot object with trade markers and annotations
        """
        try:
            import plotly.graph_objects as go
            
            # Get portfolio value for marker positioning
            portfolio_value = portfolio.value()
            
            # Get trade data if trades exist
            if portfolio.trades.count() > 0:
                trades = portfolio.trades
                
                # Get entry and exit points using VectorBT's correct API
                entry_indices = trades.entry_idx.values
                exit_indices = trades.exit_idx.values
                
                # Convert indices to timestamps
                entry_times = portfolio_value.index[entry_indices]
                exit_times = portfolio_value.index[exit_indices]
                
                # Get portfolio values at trade points for marker positioning
                entry_values = portfolio_value.iloc[entry_indices]
                exit_values = portfolio_value.iloc[exit_indices]
                
                # Add entry point markers (Requirement 3.4: green triangles)
                plot_obj.add_trace(go.Scatter(
                    x=entry_times,
                    y=entry_values,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='Entry Points',
                    hovertemplate='<b>Entry</b><br>' +
                                'Date: %{x}<br>' +
                                'Portfolio Value: $%{y:,.2f}<br>' +
                                '<extra></extra>',
                    showlegend=True
                ))
                
                # Add exit point markers (Requirement 3.5: red triangles)
                plot_obj.add_trace(go.Scatter(
                    x=exit_times,
                    y=exit_values,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    name='Exit Points',
                    hovertemplate='<b>Exit</b><br>' +
                                'Date: %{x}<br>' +
                                'Portfolio Value: $%{y:,.2f}<br>' +
                                '<extra></extra>',
                    showlegend=True
                ))
                
                self.logger.info(f"Added {len(entry_times)} entry and {len(exit_times)} exit markers")
            
            # Add performance metrics overlay (Requirement 3.6)
            metrics = self._extract_portfolio_metrics(portfolio)
            self._add_metrics_annotation(plot_obj, metrics)
            
            # Enhance hover information and interactive elements
            plot_obj.update_layout(
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return plot_obj
            
        except Exception as e:
            self.logger.warning(f"Error adding trade markers and annotations: {str(e)}")
            return plot_obj
    
    def _create_base_portfolio_plot(self, portfolio: vbt.Portfolio) -> Any:
        """
        Create base portfolio plot using plotly directly.
        
        This avoids VectorBT widget dependencies while still providing
        the core portfolio visualization functionality.
        
        Args:
            portfolio: VectorBT portfolio object
            
        Returns:
            Plotly figure object
        """
        try:
            import plotly.graph_objects as go
            
            # Get portfolio value over time
            portfolio_value = portfolio.value()
            
            # Create the base figure
            fig = go.Figure()
            
            # Add portfolio value line
            fig.add_trace(go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value.values,
                mode='lines',
                name='Portfolio Value',
                line=dict(width=2, color='blue'),
                hovertemplate='<b>Portfolio Value</b><br>' +
                            'Date: %{x}<br>' +
                            'Value: $%{y:,.2f}<br>' +
                            '<extra></extra>'
            ))
            
            # Configure layout
            fig.update_layout(
                template=self.plot_config.template,
                width=self.plot_config.width,
                height=self.plot_config.height,
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating base portfolio plot: {str(e)}")
            # Fallback to empty figure
            import plotly.graph_objects as go
            return go.Figure()
    
    def _add_metrics_annotation(self, plot_obj: Any, metrics: Dict[str, float]) -> None:
        """
        Add performance metrics overlay to the plot (Requirement 3.6).
        
        Args:
            plot_obj: Plot object to annotate
            metrics: Dictionary of performance metrics
        """
        try:
            # Format metrics text for display
            metrics_text = []
            
            if 'total_return' in metrics and not pd.isna(metrics['total_return']):
                metrics_text.append(f"Total Return: {metrics['total_return']:.2%}")
            
            if 'sharpe_ratio' in metrics and not pd.isna(metrics['sharpe_ratio']):
                metrics_text.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            
            if 'max_drawdown' in metrics and not pd.isna(metrics['max_drawdown']):
                metrics_text.append(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            
            if 'num_trades' in metrics:
                metrics_text.append(f"Trades: {int(metrics['num_trades'])}")
            
            if 'win_rate' in metrics and not pd.isna(metrics['win_rate']):
                metrics_text.append(f"Win Rate: {metrics['win_rate']:.2%}")
            
            # Add metrics annotation to top-right corner
            if metrics_text:
                plot_obj.add_annotation(
                    text="<br>".join(metrics_text),
                    xref="paper", yref="paper",
                    x=0.98, y=0.98,
                    xanchor="right", yanchor="top",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="gray",
                    borderwidth=1,
                    borderpad=10
                )
                
                self.logger.info(f"Added metrics annotation with {len(metrics_text)} metrics")
            
        except Exception as e:
            self.logger.warning(f"Error adding metrics annotation: {str(e)}")

    def _calculate_detailed_drawdown_metrics(self, portfolio_value: pd.Series) -> Dict[str, Any]:
        """
        Calculate detailed drawdown metrics for enhanced visualization.
        
        This method implements comprehensive drawdown analysis including:
        - Percentage and absolute drawdown calculations
        - Drawdown period identification and analysis
        - Recovery time analysis and statistics
        - Risk metric calculations
        
        Args:
            portfolio_value: Portfolio value time series
            
        Returns:
            Dictionary containing detailed drawdown metrics and periods
        """
        try:
            # Calculate running maximum and drawdowns
            running_max = portfolio_value.expanding().max()
            drawdown_pct = (portfolio_value - running_max) / running_max
            drawdown_abs = portfolio_value - running_max
            
            # Create underwater curve (time spent in drawdown)
            underwater_curve = (drawdown_pct < -0.001).astype(int)  # 0.1% threshold
            
            # Identify drawdown periods with recovery analysis
            drawdown_periods = self._identify_enhanced_drawdown_periods(
                portfolio_value, running_max, drawdown_pct
            )
            
            # Calculate recovery periods
            recovery_periods = self._calculate_recovery_periods(drawdown_periods)
            
            # Calculate aggregate statistics
            if len(drawdown_periods) > 0:
                drawdown_depths = [period['max_drawdown_pct'] for period in drawdown_periods]
                drawdown_durations = [period['duration_days'] for period in drawdown_periods]
                recovery_times = [period.get('recovery_days', 0) for period in recovery_periods]
                
                avg_drawdown_pct = np.mean(drawdown_depths)
                avg_drawdown_duration = np.mean(drawdown_durations)
                avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
                max_drawdown_duration = max(drawdown_durations)
                max_recovery_time = max(recovery_times) if recovery_times else 0
                
                # Calculate time underwater percentage
                total_underwater_days = sum(drawdown_durations)
                total_days = len(portfolio_value)
                time_underwater_pct = total_underwater_days / total_days if total_days > 0 else 0
                
                # Calculate recovery factor (how well portfolio recovers from drawdowns)
                successful_recoveries = len([p for p in recovery_periods if p.get('recovered', False)])
                recovery_factor = successful_recoveries / len(drawdown_periods) if len(drawdown_periods) > 0 else 0
                
            else:
                avg_drawdown_pct = 0
                avg_drawdown_duration = 0
                avg_recovery_time = 0
                max_drawdown_duration = 0
                max_recovery_time = 0
                time_underwater_pct = 0
                recovery_factor = 1.0
            
            return {
                'drawdown_pct': drawdown_pct,
                'drawdown_abs': drawdown_abs,
                'underwater_curve': underwater_curve,
                'running_max': running_max,
                'drawdown_periods': drawdown_periods,
                'recovery_periods': recovery_periods,
                'max_drawdown_pct': drawdown_pct.min(),
                'max_drawdown_abs': drawdown_abs.min(),
                'avg_drawdown_pct': avg_drawdown_pct,
                'avg_drawdown_duration': avg_drawdown_duration,
                'avg_recovery_time': avg_recovery_time,
                'max_drawdown_duration': max_drawdown_duration,
                'max_recovery_time': max_recovery_time,
                'time_underwater_pct': time_underwater_pct,
                'recovery_factor': recovery_factor
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating detailed drawdown metrics: {str(e)}")
            # Return minimal data structure
            return {
                'drawdown_pct': pd.Series(dtype=float),
                'drawdown_abs': pd.Series(dtype=float),
                'underwater_curve': pd.Series(dtype=int),
                'running_max': pd.Series(dtype=float),
                'drawdown_periods': [],
                'recovery_periods': [],
                'max_drawdown_pct': 0,
                'max_drawdown_abs': 0,
                'avg_drawdown_pct': 0,
                'avg_drawdown_duration': 0,
                'avg_recovery_time': 0,
                'max_drawdown_duration': 0,
                'max_recovery_time': 0,
                'time_underwater_pct': 0,
                'recovery_factor': 0
            }
    
    def _identify_enhanced_drawdown_periods(
        self, 
        portfolio_value: pd.Series, 
        running_max: pd.Series, 
        drawdown_pct: pd.Series
    ) -> List[Dict[str, Any]]:
        """
        Identify drawdown periods with enhanced analysis for recovery visualization.
        
        Args:
            portfolio_value: Portfolio value series
            running_max: Running maximum series
            drawdown_pct: Percentage drawdown series
            
        Returns:
            List of enhanced drawdown period dictionaries
        """
        try:
            drawdown_periods = []
            in_drawdown = False
            current_period = None
            drawdown_threshold = -0.001  # 0.1% threshold
            
            for i, (timestamp, dd_value) in enumerate(drawdown_pct.items()):
                if dd_value < drawdown_threshold and not in_drawdown:
                    # Start of drawdown period
                    in_drawdown = True
                    current_period = {
                        'start_date': timestamp,
                        'start_idx': i,
                        'start_value': portfolio_value.iloc[i],
                        'peak_value': running_max.iloc[i],
                        'max_drawdown_pct': dd_value,
                        'max_drawdown_abs': portfolio_value.iloc[i] - running_max.iloc[i],
                        'max_drawdown_date': timestamp,
                        'max_drawdown_idx': i,
                        'trough_value': portfolio_value.iloc[i],
                        'trough_date': timestamp,
                        'trough_idx': i
                    }
                
                elif in_drawdown:
                    # Update maximum drawdown and trough in current period
                    if dd_value < current_period['max_drawdown_pct']:
                        current_period['max_drawdown_pct'] = dd_value
                        current_period['max_drawdown_abs'] = portfolio_value.iloc[i] - running_max.iloc[i]
                        current_period['max_drawdown_date'] = timestamp
                        current_period['max_drawdown_idx'] = i
                        current_period['trough_value'] = portfolio_value.iloc[i]
                        current_period['trough_date'] = timestamp
                        current_period['trough_idx'] = i
                    
                    # Check for recovery (back to previous high or close to it)
                    if abs(dd_value) < abs(drawdown_threshold):  # Recovered
                        current_period['end_date'] = timestamp
                        current_period['end_idx'] = i
                        current_period['end_value'] = portfolio_value.iloc[i]
                        current_period['duration_days'] = (timestamp - current_period['start_date']).days
                        current_period['drawdown_to_trough_days'] = (
                            current_period['trough_date'] - current_period['start_date']
                        ).days
                        current_period['trough_to_recovery_days'] = (
                            timestamp - current_period['trough_date']
                        ).days
                        current_period['recovered'] = True
                        
                        drawdown_periods.append(current_period)
                        in_drawdown = False
                        current_period = None
            
            # Handle case where we end in a drawdown
            if in_drawdown and current_period is not None:
                last_timestamp = portfolio_value.index[-1]
                current_period['end_date'] = last_timestamp
                current_period['end_idx'] = len(portfolio_value) - 1
                current_period['end_value'] = portfolio_value.iloc[-1]
                current_period['duration_days'] = (last_timestamp - current_period['start_date']).days
                current_period['drawdown_to_trough_days'] = (
                    current_period['trough_date'] - current_period['start_date']
                ).days
                current_period['trough_to_recovery_days'] = (
                    last_timestamp - current_period['trough_date']
                ).days
                current_period['recovered'] = False
                
                drawdown_periods.append(current_period)
            
            return drawdown_periods
            
        except Exception as e:
            self.logger.error(f"Error identifying enhanced drawdown periods: {str(e)}")
            return []
    
    def _calculate_recovery_periods(self, drawdown_periods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate recovery period analysis for drawdown visualization.
        
        Args:
            drawdown_periods: List of drawdown period dictionaries
            
        Returns:
            List of recovery period analysis dictionaries
        """
        try:
            recovery_periods = []
            
            for period in drawdown_periods:
                if period.get('recovered', False):
                    recovery_analysis = {
                        'drawdown_start': period['start_date'],
                        'trough_date': period['trough_date'],
                        'recovery_date': period['end_date'],
                        'total_recovery_days': period['duration_days'],
                        'drawdown_phase_days': period['drawdown_to_trough_days'],
                        'recovery_phase_days': period['trough_to_recovery_days'],
                        'max_drawdown_pct': period['max_drawdown_pct'],
                        'recovery_strength': abs(period['max_drawdown_pct']),  # How deep was the recovery from
                        'recovered': True
                    }
                else:
                    # Ongoing drawdown - no recovery yet
                    recovery_analysis = {
                        'drawdown_start': period['start_date'],
                        'trough_date': period['trough_date'],
                        'recovery_date': None,
                        'total_recovery_days': 0,
                        'drawdown_phase_days': period['drawdown_to_trough_days'],
                        'recovery_phase_days': 0,
                        'max_drawdown_pct': period['max_drawdown_pct'],
                        'recovery_strength': 0,
                        'recovered': False
                    }
                
                recovery_periods.append(recovery_analysis)
            
            return recovery_periods
            
        except Exception as e:
            self.logger.error(f"Error calculating recovery periods: {str(e)}")
            return []
    
    def _create_enhanced_drawdown_plot(
        self, 
        portfolio: vbt.Portfolio, 
        drawdown_data: Dict[str, Any]
    ) -> Any:
        """
        Create enhanced drawdown plot with period highlighting and recovery visualization.
        
        This method implements Requirements 4.1, 4.2, 4.3, 4.5:
        - Show portfolio decline from peak
        - Highlight maximum drawdown period with distinct coloring
        - Visualize recovery times
        - Provide both absolute and percentage views
        
        Args:
            portfolio: VectorBT portfolio object
            drawdown_data: Detailed drawdown metrics and periods
            
        Returns:
            Enhanced plotly figure with drawdown analysis
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplot figure with percentage and absolute views (Requirement 4.5)
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=[
                    'Portfolio Value with Drawdown Periods',
                    'Drawdown Percentage from Peak',
                    'Underwater Curve (Time in Drawdown)'
                ],
                vertical_spacing=0.08,
                row_heights=[0.5, 0.3, 0.2]
            )
            
            portfolio_value = drawdown_data.get('portfolio_value', pd.Series(dtype=float))
            running_max = drawdown_data.get('running_max', pd.Series(dtype=float))
            drawdown_pct = drawdown_data.get('drawdown_pct', pd.Series(dtype=float))
            underwater_curve = drawdown_data.get('underwater_curve', pd.Series(dtype=int))
            
            # Validate data before plotting
            if len(portfolio_value) == 0:
                raise ValueError("Portfolio value data is empty")
            
            # Plot 1: Portfolio value with running maximum and drawdown highlighting
            fig.add_trace(
                go.Scatter(
                    x=portfolio_value.index,
                    y=portfolio_value.values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2),
                    hovertemplate='<b>Portfolio Value</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add running maximum line
            fig.add_trace(
                go.Scatter(
                    x=running_max.index,
                    y=running_max.values,
                    mode='lines',
                    name='Running Maximum',
                    line=dict(color='green', width=1, dash='dash'),
                    hovertemplate='<b>Running Max</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Highlight drawdown periods (Requirement 4.2)
            self._add_drawdown_period_highlighting(fig, drawdown_data, row=1)
            
            # Plot 2: Percentage drawdown (Requirement 4.1)
            fig.add_trace(
                go.Scatter(
                    x=drawdown_pct.index,
                    y=drawdown_pct.values * 100,  # Convert to percentage
                    mode='lines',
                    name='Drawdown %',
                    line=dict(color='red', width=2),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    hovertemplate='<b>Drawdown</b><br>Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add zero line for reference
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
            
            # Highlight maximum drawdown period (Requirement 4.2)
            self._add_max_drawdown_highlighting(fig, drawdown_data, row=2)
            
            # Plot 3: Underwater curve showing time in drawdown
            fig.add_trace(
                go.Scatter(
                    x=underwater_curve.index,
                    y=underwater_curve.values,
                    mode='lines',
                    name='Underwater',
                    line=dict(color='orange', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 165, 0, 0.3)',
                    hovertemplate='<b>Underwater</b><br>Date: %{x}<br>In Drawdown: %{y}<extra></extra>'
                ),
                row=3, col=1
            )
            
            # Add recovery time annotations (Requirement 4.3)
            self._add_recovery_time_annotations(fig, drawdown_data)
            
            # Update layout
            fig.update_layout(
                title='Enhanced Drawdown Analysis with Recovery Visualization',
                template=self.plot_config.template,
                width=self.plot_config.width,
                height=self.plot_config.height + 200,  # Extra height for subplots
                showlegend=True,
                hovermode='x unified'
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            fig.update_yaxes(title_text="Underwater", row=3, col=1)
            fig.update_xaxes(title_text="Date", row=3, col=1)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced drawdown plot: {str(e)}")
            # Return basic plot as fallback
            import plotly.graph_objects as go
            return go.Figure().add_annotation(
                text=f"Error creating drawdown plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    
    def _add_drawdown_period_highlighting(
        self, 
        fig: Any, 
        drawdown_data: Dict[str, Any], 
        row: int
    ) -> None:
        """
        Add drawdown period highlighting to the plot (Requirement 4.2).
        
        Args:
            fig: Plotly figure object
            drawdown_data: Drawdown analysis data
            row: Subplot row number
        """
        try:
            portfolio_value = drawdown_data['portfolio_value']
            
            for i, period in enumerate(drawdown_data['drawdown_periods']):
                # Determine color based on severity
                max_dd = abs(period['max_drawdown_pct'])
                if max_dd > 0.2:  # > 20%
                    color = 'rgba(255, 0, 0, 0.2)'  # Red for severe
                elif max_dd > 0.1:  # > 10%
                    color = 'rgba(255, 165, 0, 0.2)'  # Orange for moderate
                else:
                    color = 'rgba(255, 255, 0, 0.2)'  # Yellow for mild
                
                # Add shaded region for drawdown period
                fig.add_vrect(
                    x0=period['start_date'],
                    x1=period['end_date'],
                    fillcolor=color,
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    row=row, col=1
                )
                
                # Add annotation for significant drawdowns
                if max_dd > 0.05:  # Only annotate drawdowns > 5%
                    fig.add_annotation(
                        x=period['trough_date'],
                        y=period['trough_value'],
                        text=f"DD: {max_dd:.1%}<br>{period['duration_days']}d",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="red",
                        ax=0,
                        ay=-40,
                        bgcolor="white",
                        bordercolor="red",
                        borderwidth=1,
                        font=dict(size=10),
                        row=row, col=1
                    )
            
        except Exception as e:
            self.logger.warning(f"Error adding drawdown period highlighting: {str(e)}")
    
    def _add_max_drawdown_highlighting(
        self, 
        fig: Any, 
        drawdown_data: Dict[str, Any], 
        row: int
    ) -> None:
        """
        Add maximum drawdown period highlighting (Requirement 4.2).
        
        Args:
            fig: Plotly figure object
            drawdown_data: Drawdown analysis data
            row: Subplot row number
        """
        try:
            # Find the period with maximum drawdown
            max_dd_period = None
            max_dd_value = 0
            
            for period in drawdown_data['drawdown_periods']:
                if abs(period['max_drawdown_pct']) > max_dd_value:
                    max_dd_value = abs(period['max_drawdown_pct'])
                    max_dd_period = period
            
            if max_dd_period:
                # Highlight maximum drawdown period with distinct coloring
                fig.add_vrect(
                    x0=max_dd_period['start_date'],
                    x1=max_dd_period['end_date'],
                    fillcolor='rgba(139, 0, 0, 0.4)',  # Dark red for max drawdown
                    opacity=0.5,
                    layer="below",
                    line_width=2,
                    line_color="darkred",
                    row=row, col=1
                )
                
                # Add prominent annotation for maximum drawdown
                fig.add_annotation(
                    x=max_dd_period['trough_date'],
                    y=max_dd_period['max_drawdown_pct'] * 100,
                    text=f"<b>MAX DD: {max_dd_value:.1%}</b><br>"
                         f"Duration: {max_dd_period['duration_days']}d<br>"
                         f"Recovery: {max_dd_period.get('trough_to_recovery_days', 'N/A')}d",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=2,
                    arrowwidth=3,
                    arrowcolor="darkred",
                    ax=0,
                    ay=-60,
                    bgcolor="white",
                    bordercolor="darkred",
                    borderwidth=2,
                    font=dict(size=12, color="darkred"),
                    row=row, col=1
                )
            
        except Exception as e:
            self.logger.warning(f"Error adding max drawdown highlighting: {str(e)}")
    
    def _add_recovery_time_annotations(
        self, 
        fig: Any, 
        drawdown_data: Dict[str, Any]
    ) -> None:
        """
        Add recovery time visualization annotations (Requirement 4.3).
        
        Args:
            fig: Plotly figure object
            drawdown_data: Drawdown analysis data
        """
        try:
            for period in drawdown_data['recovery_periods']:
                if period.get('recovered', False):
                    # Add recovery time arrow
                    fig.add_annotation(
                        x=period['recovery_date'],
                        y=1,  # Top of underwater plot
                        text=f"Recovery: {period['recovery_phase_days']}d",
                        showarrow=True,
                        arrowhead=1,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="green",
                        ax=0,
                        ay=20,
                        bgcolor="lightgreen",
                        bordercolor="green",
                        borderwidth=1,
                        font=dict(size=9, color="green"),
                        row=3, col=1
                    )
            
        except Exception as e:
            self.logger.warning(f"Error adding recovery time annotations: {str(e)}")

    def _extract_portfolio_metrics(self, portfolio: vbt.Portfolio) -> Dict[str, float]:
        """
        Extract key portfolio metrics for display.
        
        Args:
            portfolio: VectorBT portfolio object
            
        Returns:
            Dictionary of key metrics
        """
        try:
            metrics = {
                'total_return': portfolio.total_return(),
                'annualized_return': portfolio.annualized_return(),
                'sharpe_ratio': portfolio.sharpe_ratio(),
                'max_drawdown': portfolio.max_drawdown(),
                'volatility': portfolio.annualized_volatility(),
                'num_trades': portfolio.trades.count()
            }
            
            # Add trade-specific metrics if trades exist
            if portfolio.trades.count() > 0:
                metrics.update({
                    'win_rate': portfolio.trades.win_rate(),
                    'profit_factor': portfolio.trades.profit_factor(),
                    'avg_trade_duration': portfolio.trades.duration.mean()
                })
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Error extracting portfolio metrics: {str(e)}")
            return {'error': str(e)}
    
    def create_enhanced_portfolio(
        self,
        predictions: np.ndarray,
        price_data: pd.DataFrame,
        test_start_idx: int,
        **kwargs
    ) -> Tuple[vbt.Portfolio, AlignedSignals]:
        """
        Create enhanced portfolio with signal alignment metadata.
        
        Args:
            predictions: ML predictions
            price_data: Historical price data
            test_start_idx: Test period start index
            **kwargs: Additional portfolio parameters
            
        Returns:
            Tuple of (portfolio, aligned_signals)
        """
        # Update configuration with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self.portfolio_config, key):
                setattr(self.portfolio_config, key, value)
        
        # Create portfolio
        portfolio = self.create_portfolio_from_predictions(
            predictions, price_data, test_start_idx
        )
        
        # Get aligned signals for metadata
        aligned_signals = self.signal_aligner.align_predictions_to_timeline(
            predictions, price_data, test_start_idx
        )
        
        return portfolio, aligned_signals
    
    def calculate_position_sizes(
        self,
        prices: pd.Series,
        sizing_method: Optional[str] = None,
        capital: Optional[float] = None,
        volatility: Optional[pd.Series] = None,
        risk_metrics: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Calculate position sizes with multiple sizing approaches.
        
        This method provides a direct interface to the enhanced position sizing
        functionality, supporting:
        - Fixed amount, fixed shares, and percentage-based sizing
        - Dynamic sizing based on volatility and risk metrics
        - Risk-adjusted position sizing with volatility targeting
        
        Args:
            prices: Historical price series
            sizing_method: Override default sizing method ('fixed_amount', 'fixed_shares', 
                          'percent_equity', 'volatility_target', 'risk_parity')
            capital: Override default capital amount
            volatility: Optional volatility series for dynamic sizing
            risk_metrics: Optional risk metrics for advanced sizing
            
        Returns:
            Series of position sizes aligned with price index
            
        Example:
            # Fixed amount sizing (default)
            sizes = engine.calculate_position_sizes(prices)
            
            # Volatility-targeted sizing
            sizes = engine.calculate_position_sizes(
                prices, 
                sizing_method='volatility_target',
                risk_metrics={'target_volatility': 0.12}
            )
            
            # Risk parity sizing
            sizes = engine.calculate_position_sizes(
                prices,
                sizing_method='risk_parity',
                capital=50000
            )
        """
        return self.portfolio_config.calculate_position_sizes(
            prices=prices,
            sizing_method=sizing_method,
            capital=capital,
            volatility=volatility,
            risk_metrics=risk_metrics
        )
    
    def _calculate_risk_metrics(
        self, 
        portfolio: vbt.Portfolio, 
        drawdown_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for comparative analysis.
        
        Args:
            portfolio: VectorBT portfolio object
            drawdown_data: Detailed drawdown analysis data
            
        Returns:
            Dictionary of risk metrics
        """
        try:
            # Basic portfolio metrics
            total_return = portfolio.total_return()
            sharpe_ratio = portfolio.sharpe_ratio()
            volatility = portfolio.annualized_volatility()
            
            # Drawdown-specific metrics
            max_drawdown = drawdown_data['max_drawdown_pct']
            avg_drawdown = drawdown_data['avg_drawdown_pct']
            time_underwater = drawdown_data['time_underwater_pct']
            recovery_factor = drawdown_data['recovery_factor']
            
            # Risk-adjusted metrics
            calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
            sterling_ratio = total_return / abs(avg_drawdown) if avg_drawdown != 0 else 0
            
            # Downside risk metrics
            portfolio_returns = portfolio.returns()
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = total_return / downside_volatility if downside_volatility != 0 else 0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'sterling_ratio': sterling_ratio,
                'volatility': volatility,
                'downside_volatility': downside_volatility,
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown,
                'time_underwater_pct': time_underwater,
                'recovery_factor': recovery_factor,
                'avg_recovery_time': drawdown_data['avg_recovery_time'],
                'max_recovery_time': drawdown_data['max_recovery_time'],
                'num_drawdown_periods': len(drawdown_data['drawdown_periods'])
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating risk metrics: {str(e)}")
            return {}
    
    def _create_comparative_risk_plot(
        self, 
        risk_comparison_data: Dict[str, Dict[str, Any]], 
        title: str
    ) -> Any:
        """
        Create comparative risk visualization plot.
        
        Args:
            risk_comparison_data: Risk analysis data for all strategies
            title: Plot title
            
        Returns:
            Plotly figure with comparative risk analysis
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            strategy_names = list(risk_comparison_data.keys())
            
            # Create subplot figure for comprehensive risk comparison
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Drawdown Comparison',
                    'Risk-Return Scatter',
                    'Recovery Time Analysis',
                    'Risk Metrics Heatmap'
                ],
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # Plot 1: Drawdown comparison
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            for i, (name, data) in enumerate(risk_comparison_data.items()):
                drawdown_pct = data['drawdown_data']['drawdown_pct']
                color = colors[i % len(colors)]
                
                fig.add_trace(
                    go.Scatter(
                        x=drawdown_pct.index,
                        y=drawdown_pct.values * 100,
                        mode='lines',
                        name=f'{name} DD',
                        line=dict(color=color, width=2),
                        hovertemplate=f'<b>{name}</b><br>Date: %{{x}}<br>Drawdown: %{{y:.2f}}%<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Plot 2: Risk-Return scatter
            returns = []
            risks = []
            names = []
            for name, data in risk_comparison_data.items():
                metrics = data['risk_metrics']
                returns.append(metrics.get('total_return', 0) * 100)
                risks.append(abs(metrics.get('max_drawdown', 0)) * 100)
                names.append(name)
            
            fig.add_trace(
                go.Scatter(
                    x=risks,
                    y=returns,
                    mode='markers+text',
                    text=names,
                    textposition="top center",
                    marker=dict(size=12, color='blue', opacity=0.7),
                    name='Strategies',
                    hovertemplate='<b>%{text}</b><br>Max DD: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Plot 3: Recovery time analysis
            recovery_times = []
            strategy_labels = []
            for name, data in risk_comparison_data.items():
                avg_recovery = data['risk_metrics'].get('avg_recovery_time', 0)
                max_recovery = data['risk_metrics'].get('max_recovery_time', 0)
                
                recovery_times.extend([avg_recovery, max_recovery])
                strategy_labels.extend([f'{name} Avg', f'{name} Max'])
            
            fig.add_trace(
                go.Bar(
                    x=strategy_labels,
                    y=recovery_times,
                    name='Recovery Days',
                    marker_color=['lightblue' if 'Avg' in label else 'darkblue' for label in strategy_labels],
                    hovertemplate='<b>%{x}</b><br>Recovery Time: %{y:.0f} days<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Plot 4: Risk metrics heatmap
            self._add_risk_metrics_heatmap(fig, risk_comparison_data, row=2, col=2)
            
            # Update layout
            fig.update_layout(
                title=title,
                template=self.plot_config.template,
                width=self.plot_config.width + 200,
                height=self.plot_config.height + 300,
                showlegend=True
            )
            
            # Update axis labels
            fig.update_yaxes(title_text="Drawdown (%)", row=1, col=1)
            fig.update_xaxes(title_text="Date", row=1, col=1)
            
            fig.update_yaxes(title_text="Total Return (%)", row=1, col=2)
            fig.update_xaxes(title_text="Max Drawdown (%)", row=1, col=2)
            
            fig.update_yaxes(title_text="Recovery Days", row=2, col=1)
            fig.update_xaxes(title_text="Strategy", row=2, col=1)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating comparative risk plot: {str(e)}")
            import plotly.graph_objects as go
            return go.Figure().add_annotation(
                text=f"Error creating comparative risk plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    
    def _add_risk_metrics_heatmap(
        self, 
        fig: Any, 
        risk_comparison_data: Dict[str, Dict[str, Any]], 
        row: int, 
        col: int
    ) -> None:
        """
        Add risk metrics heatmap to comparative plot.
        
        Args:
            fig: Plotly figure object
            risk_comparison_data: Risk analysis data
            row: Subplot row
            col: Subplot column
        """
        try:
            import plotly.graph_objects as go
            
            # Prepare heatmap data
            strategies = list(risk_comparison_data.keys())
            metrics = ['Sharpe Ratio', 'Calmar Ratio', 'Recovery Factor', 'Time Underwater %']
            
            heatmap_data = []
            for strategy in strategies:
                risk_metrics = risk_comparison_data[strategy]['risk_metrics']
                row_data = [
                    risk_metrics.get('sharpe_ratio', 0),
                    risk_metrics.get('calmar_ratio', 0),
                    risk_metrics.get('recovery_factor', 0),
                    risk_metrics.get('time_underwater_pct', 0) * 100
                ]
                heatmap_data.append(row_data)
            
            # Create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=heatmap_data,
                    x=metrics,
                    y=strategies,
                    colorscale='RdYlGn',
                    hoverongaps=False,
                    hovertemplate='<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>'
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.warning(f"Error adding risk metrics heatmap: {str(e)}")
    
    def _compile_risk_metrics_table(
        self, 
        risk_comparison_data: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compile risk metrics comparison table.
        
        Args:
            risk_comparison_data: Risk analysis data for all strategies
            
        Returns:
            DataFrame with risk metrics comparison
        """
        try:
            metrics_data = []
            
            for strategy_name, data in risk_comparison_data.items():
                risk_metrics = data['risk_metrics']
                metrics_data.append({
                    'Strategy': strategy_name,
                    'Total Return': f"{risk_metrics.get('total_return', 0):.2%}",
                    'Sharpe Ratio': f"{risk_metrics.get('sharpe_ratio', 0):.2f}",
                    'Sortino Ratio': f"{risk_metrics.get('sortino_ratio', 0):.2f}",
                    'Calmar Ratio': f"{risk_metrics.get('calmar_ratio', 0):.2f}",
                    'Max Drawdown': f"{risk_metrics.get('max_drawdown', 0):.2%}",
                    'Avg Recovery Time': f"{risk_metrics.get('avg_recovery_time', 0):.0f} days",
                    'Recovery Factor': f"{risk_metrics.get('recovery_factor', 0):.2f}",
                    'Time Underwater': f"{risk_metrics.get('time_underwater_pct', 0):.1%}"
                })
            
            return pd.DataFrame(metrics_data)
            
        except Exception as e:
            self.logger.error(f"Error compiling risk metrics table: {str(e)}")
            return pd.DataFrame()
    
    def _compile_drawdown_comparison(
        self, 
        risk_comparison_data: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compile drawdown comparison data.
        
        Args:
            risk_comparison_data: Risk analysis data for all strategies
            
        Returns:
            DataFrame with drawdown comparison
        """
        try:
            comparison_data = {}
            
            for strategy_name, data in risk_comparison_data.items():
                drawdown_pct = data['drawdown_data']['drawdown_pct']
                comparison_data[f'{strategy_name}_drawdown'] = drawdown_pct
            
            return pd.DataFrame(comparison_data)
            
        except Exception as e:
            self.logger.error(f"Error compiling drawdown comparison: {str(e)}")
            return pd.DataFrame()
    
    def _compile_recovery_analysis(
        self, 
        risk_comparison_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compile recovery analysis across strategies.
        
        Args:
            risk_comparison_data: Risk analysis data for all strategies
            
        Returns:
            Dictionary with recovery analysis for each strategy
        """
        try:
            recovery_analysis = {}
            
            for strategy_name, data in risk_comparison_data.items():
                recovery_periods = data['drawdown_data']['recovery_periods']
                recovery_analysis[strategy_name] = recovery_periods
            
            return recovery_analysis
            
        except Exception as e:
            self.logger.error(f"Error compiling recovery analysis: {str(e)}")
            return {}
    
    def _calculate_comparative_metrics_summary(
        self, 
        risk_comparison_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate summary metrics for comparative analysis.
        
        Args:
            risk_comparison_data: Risk analysis data for all strategies
            
        Returns:
            Dictionary with comparative summary metrics
        """
        try:
            summary = {
                'num_strategies': len(risk_comparison_data),
                'best_sharpe_strategy': '',
                'best_calmar_strategy': '',
                'lowest_drawdown_strategy': '',
                'fastest_recovery_strategy': '',
                'strategy_rankings': {}
            }
            
            # Find best performing strategies by different metrics
            best_sharpe = -float('inf')
            best_calmar = -float('inf')
            lowest_dd = float('inf')
            fastest_recovery = float('inf')
            
            for strategy_name, data in risk_comparison_data.items():
                metrics = data['risk_metrics']
                
                sharpe = metrics.get('sharpe_ratio', -float('inf'))
                calmar = metrics.get('calmar_ratio', -float('inf'))
                max_dd = abs(metrics.get('max_drawdown', float('inf')))
                avg_recovery = metrics.get('avg_recovery_time', float('inf'))
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    summary['best_sharpe_strategy'] = strategy_name
                
                if calmar > best_calmar:
                    best_calmar = calmar
                    summary['best_calmar_strategy'] = strategy_name
                
                if max_dd < lowest_dd:
                    lowest_dd = max_dd
                    summary['lowest_drawdown_strategy'] = strategy_name
                
                if avg_recovery < fastest_recovery:
                    fastest_recovery = avg_recovery
                    summary['fastest_recovery_strategy'] = strategy_name
                
                # Store individual strategy metrics for ranking
                summary['strategy_rankings'][strategy_name] = {
                    'sharpe_ratio': sharpe,
                    'calmar_ratio': calmar,
                    'max_drawdown': max_dd,
                    'avg_recovery_time': avg_recovery
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error calculating comparative metrics summary: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_strategy_comparison_metrics(self, portfolio: vbt.Portfolio) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for strategy comparison and ranking.
        
        Args:
            portfolio: VectorBT portfolio object
            
        Returns:
            Dictionary of comparison metrics
        """
        try:
            # Basic performance metrics
            total_return = portfolio.total_return()
            annualized_return = portfolio.annualized_return()
            sharpe_ratio = portfolio.sharpe_ratio()
            max_drawdown = portfolio.max_drawdown()
            volatility = portfolio.annualized_volatility()
            
            # Trade-specific metrics
            num_trades = portfolio.trades.count()
            win_rate = portfolio.trades.win_rate() if num_trades > 0 else 0
            profit_factor = portfolio.trades.profit_factor() if num_trades > 0 else 0
            
            # Risk-adjusted metrics
            calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Downside risk metrics
            returns = portfolio.returns()
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = annualized_return / downside_volatility if downside_volatility != 0 else 0
            
            # Portfolio value metrics
            portfolio_value = portfolio.value()
            final_value = portfolio_value.iloc[-1]
            initial_value = portfolio_value.iloc[0]
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'downside_volatility': downside_volatility,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'num_trades': num_trades,
                'final_value': final_value,
                'initial_value': initial_value,
                'value_ratio': final_value / initial_value if initial_value != 0 else 1
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating strategy comparison metrics: {str(e)}")
            return {}
    
    def _create_enhanced_comparison_plot(
        self, 
        portfolio_data: Dict[str, Dict[str, Any]], 
        comparison_metrics: Dict[str, Dict[str, float]], 
        title: str
    ) -> Any:
        """
        Create enhanced multi-strategy comparison plot with comprehensive analysis.
        
        This method implements Requirements 8.1, 8.2, 8.4:
        - Side-by-side performance analysis
        - Overlay plotting for multiple portfolio equity curves
        - Performance ranking and statistical comparison displays
        
        Args:
            portfolio_data: Portfolio data for all strategies
            comparison_metrics: Calculated metrics for all strategies
            title: Plot title
            
        Returns:
            Enhanced plotly figure with multi-strategy comparison
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create comprehensive subplot layout
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    'Portfolio Value Comparison (Absolute)',
                    'Normalized Performance (Base 100)',
                    'Returns Distribution Comparison',
                    'Risk-Return Scatter Analysis',
                    'Performance Metrics Comparison',
                    'Strategy Rankings'
                ],
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"colspan": 2}, None]
                ],
                vertical_spacing=0.08,
                horizontal_spacing=0.1
            )
            
            # Define colors for strategies
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            # Plot 1: Absolute portfolio values (Requirement 8.2)
            for i, (name, data) in enumerate(portfolio_data.items()):
                color = colors[i % len(colors)]
                portfolio_value = data['value']
                
                fig.add_trace(
                    go.Scatter(
                        x=portfolio_value.index,
                        y=portfolio_value.values,
                        mode='lines',
                        name=f'{name}',
                        line=dict(color=color, width=2),
                        hovertemplate=f'<b>{name}</b><br>Date: %{{x}}<br>Value: $%{{y:,.2f}}<extra></extra>',
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Plot 2: Normalized performance comparison (Requirement 8.2)
            for i, (name, data) in enumerate(portfolio_data.items()):
                color = colors[i % len(colors)]
                normalized_value = data['normalized_value']
                
                fig.add_trace(
                    go.Scatter(
                        x=normalized_value.index,
                        y=normalized_value.values,
                        mode='lines',
                        name=f'{name} (Norm)',
                        line=dict(color=color, width=2, dash='dot'),
                        hovertemplate=f'<b>{name}</b><br>Date: %{{x}}<br>Normalized: %{{y:.1f}}<extra></extra>',
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # Plot 3: Returns distribution comparison
            for i, (name, data) in enumerate(portfolio_data.items()):
                color = colors[i % len(colors)]
                returns = data['returns'] * 100  # Convert to percentage
                
                fig.add_trace(
                    go.Histogram(
                        x=returns.values,
                        name=f'{name} Returns',
                        opacity=0.7,
                        marker_color=color,
                        nbinsx=30,
                        hovertemplate=f'<b>{name}</b><br>Return: %{{x:.2f}}%<br>Count: %{{y}}<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # Plot 4: Risk-Return scatter analysis (Requirement 8.4)
            returns_list = []
            risks_list = []
            names_list = []
            colors_list = []
            
            for i, (name, metrics) in enumerate(comparison_metrics.items()):
                returns_list.append(metrics.get('total_return', 0) * 100)
                risks_list.append(abs(metrics.get('max_drawdown', 0)) * 100)
                names_list.append(name)
                colors_list.append(colors[i % len(colors)])
            
            fig.add_trace(
                go.Scatter(
                    x=risks_list,
                    y=returns_list,
                    mode='markers+text',
                    text=names_list,
                    textposition="top center",
                    marker=dict(
                        size=15,
                        color=colors_list,
                        opacity=0.8,
                        line=dict(width=2, color='white')
                    ),
                    name='Risk-Return',
                    hovertemplate='<b>%{text}</b><br>Max DD: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>',
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # Plot 5: Performance metrics comparison (Requirement 8.4)
            self._add_performance_metrics_comparison(fig, comparison_metrics, row=3, col=1)
            
            # Update layout with comprehensive styling
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    font=dict(size=16)
                ),
                template=self.plot_config.template,
                width=self.plot_config.width + 300,
                height=self.plot_config.height + 400,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.1,
                    xanchor="center",
                    x=0.5
                ),
                hovermode='closest'
            )
            
            # Update axis labels
            fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
            fig.update_xaxes(title_text="Date", row=1, col=1)
            
            fig.update_yaxes(title_text="Normalized Value", row=1, col=2)
            fig.update_xaxes(title_text="Date", row=1, col=2)
            
            fig.update_yaxes(title_text="Frequency", row=2, col=1)
            fig.update_xaxes(title_text="Daily Returns (%)", row=2, col=1)
            
            fig.update_yaxes(title_text="Total Return (%)", row=2, col=2)
            fig.update_xaxes(title_text="Max Drawdown (%)", row=2, col=2)
            
            fig.update_xaxes(title_text="Metrics", row=3, col=1)
            fig.update_yaxes(title_text="Values", row=3, col=1)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced comparison plot: {str(e)}")
            import plotly.graph_objects as go
            return go.Figure().add_annotation(
                text=f"Error creating comparison plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    
    def _add_performance_metrics_comparison(
        self, 
        fig: Any, 
        comparison_metrics: Dict[str, Dict[str, float]], 
        row: int, 
        col: int
    ) -> None:
        """
        Add performance metrics comparison bar chart.
        
        Args:
            fig: Plotly figure object
            comparison_metrics: Metrics for all strategies
            row: Subplot row
            col: Subplot column
        """
        try:
            import plotly.graph_objects as go
            
            # Select key metrics for comparison
            key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
            metric_labels = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
            
            strategies = list(comparison_metrics.keys())
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            # Create grouped bar chart
            for i, strategy in enumerate(strategies):
                metrics = comparison_metrics[strategy]
                values = []
                
                for metric in key_metrics:
                    value = metrics.get(metric, 0)
                    # Normalize values for better visualization
                    if metric == 'total_return':
                        values.append(value * 100)  # Convert to percentage
                    elif metric == 'max_drawdown':
                        values.append(abs(value) * 100)  # Convert to positive percentage
                    elif metric == 'win_rate':
                        values.append(value * 100)  # Convert to percentage
                    else:
                        values.append(value)
                
                fig.add_trace(
                    go.Bar(
                        x=metric_labels,
                        y=values,
                        name=strategy,
                        marker_color=colors[i % len(colors)],
                        opacity=0.8,
                        hovertemplate=f'<b>{strategy}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>'
                    ),
                    row=row, col=col
                )
            
        except Exception as e:
            self.logger.warning(f"Error adding performance metrics comparison: {str(e)}")
    
    def _calculate_strategy_rankings(
        self, 
        comparison_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate strategy rankings by different performance criteria.
        
        This method implements Requirement 8.5: Rank strategies by configurable performance criteria.
        
        Args:
            comparison_metrics: Metrics for all strategies
            
        Returns:
            Dictionary with strategy rankings by different criteria
        """
        try:
            rankings = {
                'by_total_return': [],
                'by_sharpe_ratio': [],
                'by_calmar_ratio': [],
                'by_max_drawdown': [],
                'by_win_rate': [],
                'composite_score': []
            }
            
            strategies = list(comparison_metrics.keys())
            
            # Rank by total return (descending)
            total_return_ranking = sorted(
                strategies, 
                key=lambda x: comparison_metrics[x].get('total_return', -float('inf')), 
                reverse=True
            )
            rankings['by_total_return'] = [
                {'rank': i+1, 'strategy': strategy, 'value': comparison_metrics[strategy].get('total_return', 0)}
                for i, strategy in enumerate(total_return_ranking)
            ]
            
            # Rank by Sharpe ratio (descending)
            sharpe_ranking = sorted(
                strategies, 
                key=lambda x: comparison_metrics[x].get('sharpe_ratio', -float('inf')), 
                reverse=True
            )
            rankings['by_sharpe_ratio'] = [
                {'rank': i+1, 'strategy': strategy, 'value': comparison_metrics[strategy].get('sharpe_ratio', 0)}
                for i, strategy in enumerate(sharpe_ranking)
            ]
            
            # Rank by Calmar ratio (descending)
            calmar_ranking = sorted(
                strategies, 
                key=lambda x: comparison_metrics[x].get('calmar_ratio', -float('inf')), 
                reverse=True
            )
            rankings['by_calmar_ratio'] = [
                {'rank': i+1, 'strategy': strategy, 'value': comparison_metrics[strategy].get('calmar_ratio', 0)}
                for i, strategy in enumerate(calmar_ranking)
            ]
            
            # Rank by max drawdown (ascending - lower is better)
            drawdown_ranking = sorted(
                strategies, 
                key=lambda x: abs(comparison_metrics[x].get('max_drawdown', float('inf')))
            )
            rankings['by_max_drawdown'] = [
                {'rank': i+1, 'strategy': strategy, 'value': comparison_metrics[strategy].get('max_drawdown', 0)}
                for i, strategy in enumerate(drawdown_ranking)
            ]
            
            # Rank by win rate (descending)
            win_rate_ranking = sorted(
                strategies, 
                key=lambda x: comparison_metrics[x].get('win_rate', -float('inf')), 
                reverse=True
            )
            rankings['by_win_rate'] = [
                {'rank': i+1, 'strategy': strategy, 'value': comparison_metrics[strategy].get('win_rate', 0)}
                for i, strategy in enumerate(win_rate_ranking)
            ]
            
            # Calculate composite score (weighted combination of metrics)
            composite_scores = {}
            for strategy in strategies:
                metrics = comparison_metrics[strategy]
                
                # Normalize metrics to 0-1 scale for composite scoring
                total_return_norm = max(0, metrics.get('total_return', 0))
                sharpe_norm = max(0, metrics.get('sharpe_ratio', 0)) / 3  # Assume max reasonable Sharpe is 3
                drawdown_norm = max(0, 1 - abs(metrics.get('max_drawdown', 1)))  # Invert drawdown (lower is better)
                win_rate_norm = metrics.get('win_rate', 0)
                
                # Weighted composite score (adjust weights as needed)
                composite_score = (
                    0.3 * total_return_norm +
                    0.3 * sharpe_norm +
                    0.2 * drawdown_norm +
                    0.2 * win_rate_norm
                )
                
                composite_scores[strategy] = composite_score
            
            composite_ranking = sorted(strategies, key=lambda x: composite_scores[x], reverse=True)
            rankings['composite_score'] = [
                {'rank': i+1, 'strategy': strategy, 'value': composite_scores[strategy]}
                for i, strategy in enumerate(composite_ranking)
            ]
            
            return rankings
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy rankings: {str(e)}")
            return {}
    
    def _calculate_comparison_summary_metrics(
        self, 
        comparison_metrics: Dict[str, Dict[str, float]], 
        portfolio_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate summary metrics for the comparison analysis.
        
        Args:
            comparison_metrics: Calculated metrics for all strategies
            portfolio_data: Portfolio data for all strategies
            
        Returns:
            Dictionary with comparison summary metrics
        """
        try:
            summary = {
                'num_strategies_compared': len(comparison_metrics),
                'analysis_period_days': 0,
                'best_performer': '',
                'worst_performer': '',
                'most_consistent': '',
                'highest_risk': '',
                'performance_spread': {},
                'correlation_analysis': {}
            }
            
            if len(comparison_metrics) == 0:
                return summary
            
            # Calculate analysis period
            first_portfolio = list(portfolio_data.values())[0]
            if 'value' in first_portfolio and len(first_portfolio['value']) > 0:
                start_date = first_portfolio['value'].index[0]
                end_date = first_portfolio['value'].index[-1]
                summary['analysis_period_days'] = (end_date - start_date).days
                summary['analysis_start_date'] = start_date.strftime('%Y-%m-%d')
                summary['analysis_end_date'] = end_date.strftime('%Y-%m-%d')
            
            # Find best and worst performers by total return
            returns = {name: metrics.get('total_return', 0) for name, metrics in comparison_metrics.items()}
            summary['best_performer'] = max(returns, key=returns.get)
            summary['worst_performer'] = min(returns, key=returns.get)
            
            # Find most consistent (lowest volatility)
            volatilities = {name: metrics.get('volatility', float('inf')) for name, metrics in comparison_metrics.items()}
            summary['most_consistent'] = min(volatilities, key=volatilities.get)
            
            # Find highest risk (highest max drawdown)
            drawdowns = {name: abs(metrics.get('max_drawdown', 0)) for name, metrics in comparison_metrics.items()}
            summary['highest_risk'] = max(drawdowns, key=drawdowns.get)
            
            # Calculate performance spread
            summary['performance_spread'] = {
                'return_range': max(returns.values()) - min(returns.values()),
                'sharpe_range': max(metrics.get('sharpe_ratio', 0) for metrics in comparison_metrics.values()) - 
                              min(metrics.get('sharpe_ratio', 0) for metrics in comparison_metrics.values()),
                'drawdown_range': max(drawdowns.values()) - min(drawdowns.values())
            }
            
            # Calculate correlation analysis if multiple strategies
            if len(portfolio_data) > 1:
                returns_df = pd.DataFrame({
                    name: data['returns'] for name, data in portfolio_data.items()
                })
                correlation_matrix = returns_df.corr()
                
                # Find most and least correlated pairs
                correlations = []
                for i, strategy1 in enumerate(returns_df.columns):
                    for j, strategy2 in enumerate(returns_df.columns):
                        if i < j:  # Avoid duplicates and self-correlation
                            corr_value = correlation_matrix.loc[strategy1, strategy2]
                            correlations.append({
                                'pair': f"{strategy1} vs {strategy2}",
                                'correlation': corr_value
                            })
                
                if correlations:
                    most_correlated = max(correlations, key=lambda x: x['correlation'])
                    least_correlated = min(correlations, key=lambda x: x['correlation'])
                    
                    summary['correlation_analysis'] = {
                        'most_correlated_pair': most_correlated,
                        'least_correlated_pair': least_correlated,
                        'average_correlation': np.mean([c['correlation'] for c in correlations])
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error calculating comparison summary metrics: {str(e)}")
            return {'error': str(e)}
    
    def generate_divergence_analysis_plot(
        self, 
        portfolios: Dict[str, vbt.Portfolio],
        title: str = "Strategy Divergence Analysis"
    ) -> VisualizationResult:
        """
        Generate divergence analysis and highlighting for multi-strategy comparison.
        
        This method implements Requirement 8.3:
        - Implement period identification where strategies diverge significantly
        - Add statistical significance testing for performance differences
        - Create automated insights and recommendations
        
        Args:
            portfolios: Dictionary of named portfolio objects
            title: Plot title
            
        Returns:
            VisualizationResult with divergence analysis plot
        """
        try:
            start_time = time.time()
            
            self.logger.info(f"Generating divergence analysis for {len(portfolios)} strategies")
            
            if len(portfolios) < 2:
                raise DataValidationError("Need at least 2 portfolios for divergence analysis")
            
            # Extract portfolio data for divergence analysis
            portfolio_data = {}
            for name, portfolio in portfolios.items():
                portfolio_data[name] = {
                    'portfolio': portfolio,
                    'value': portfolio.value(),
                    'returns': portfolio.returns(),
                    'normalized_value': portfolio.value() / portfolio.value().iloc[0]
                }
            
            # Identify divergence periods
            divergence_periods = self._identify_divergence_periods(portfolio_data)
            
            # Perform statistical significance testing
            significance_tests = self._perform_statistical_significance_testing(portfolio_data)
            
            # Generate automated insights and recommendations
            insights_and_recommendations = self._generate_automated_insights(
                portfolio_data, divergence_periods, significance_tests
            )
            
            # Create divergence analysis plot
            plot_obj = self._create_divergence_analysis_plot(
                portfolio_data, divergence_periods, significance_tests, title
            )
            
            # Compile comprehensive plot data
            plot_data = {
                'divergence_periods': divergence_periods,
                'significance_tests': significance_tests,
                'portfolio_correlations': self._calculate_rolling_correlations(portfolio_data),
                'performance_spreads': self._calculate_performance_spreads(portfolio_data),
                'insights_and_recommendations': insights_and_recommendations
            }
            
            # Calculate summary metrics
            metrics_summary = {
                'num_divergence_periods': len(divergence_periods),
                'max_divergence_magnitude': max([p['max_spread'] for p in divergence_periods]) if divergence_periods else 0,
                'avg_divergence_duration': np.mean([p['duration_days'] for p in divergence_periods]) if divergence_periods else 0,
                'significant_differences_count': len([t for t in significance_tests if t['is_significant']]),
                'insights_count': len(insights_and_recommendations['insights']),
                'recommendations_count': len(insights_and_recommendations['recommendations'])
            }
            
            generation_time = time.time() - start_time
            
            self.logger.info(
                f"Divergence analysis completed in {generation_time:.2f}s "
                f"({metrics_summary['num_divergence_periods']} divergence periods identified)"
            )
            
            return VisualizationResult(
                plot_object=plot_obj,
                plot_data=plot_data,
                metrics_summary=metrics_summary,
                export_paths={},
                generation_time=generation_time,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error generating divergence analysis: {str(e)}")
            return VisualizationResult(
                plot_object=None,
                plot_data={},
                metrics_summary={},
                export_paths={},
                generation_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _identify_divergence_periods(
        self, 
        portfolio_data: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify periods where strategies diverge significantly.
        
        This method implements the core divergence detection logic for Requirement 8.3.
        
        Args:
            portfolio_data: Portfolio data for all strategies
            
        Returns:
            List of divergence period dictionaries
        """
        try:
            divergence_periods = []
            
            if len(portfolio_data) < 2:
                return divergence_periods
            
            # Create DataFrame of normalized portfolio values
            normalized_values = pd.DataFrame({
                name: data['normalized_value'] for name, data in portfolio_data.items()
            })
            
            # Calculate rolling spread between strategies
            window_size = 20  # 20-day rolling window
            spread_threshold = 0.05  # 5% spread threshold for divergence
            
            # Calculate pairwise spreads
            strategy_names = list(normalized_values.columns)
            max_spreads = pd.Series(index=normalized_values.index, dtype=float)
            
            for i in range(len(normalized_values)):
                values_at_time = normalized_values.iloc[i]
                max_spread = values_at_time.max() - values_at_time.min()
                max_spreads.iloc[i] = max_spread
            
            # Smooth the spread using rolling average
            smoothed_spreads = max_spreads.rolling(window=5, center=True).mean()
            
            # Identify divergence periods
            in_divergence = False
            current_period = None
            
            for i, (timestamp, spread) in enumerate(smoothed_spreads.items()):
                if pd.isna(spread):
                    continue
                
                if spread > spread_threshold and not in_divergence:
                    # Start of divergence period
                    in_divergence = True
                    current_period = {
                        'start_date': timestamp,
                        'start_idx': i,
                        'max_spread': spread,
                        'max_spread_date': timestamp,
                        'strategies_at_start': normalized_values.loc[timestamp].to_dict(),
                        'leading_strategy': '',
                        'lagging_strategy': ''
                    }
                    
                    # Identify leading and lagging strategies
                    values_at_start = normalized_values.loc[timestamp]
                    current_period['leading_strategy'] = values_at_start.idxmax()
                    current_period['lagging_strategy'] = values_at_start.idxmin()
                
                elif in_divergence:
                    # Update maximum spread in current period
                    if spread > current_period['max_spread']:
                        current_period['max_spread'] = spread
                        current_period['max_spread_date'] = timestamp
                    
                    # Check for convergence (end of divergence)
                    if spread <= spread_threshold * 0.7:  # 30% below threshold for convergence
                        current_period['end_date'] = timestamp
                        current_period['end_idx'] = i
                        current_period['duration_days'] = (timestamp - current_period['start_date']).days
                        current_period['strategies_at_end'] = normalized_values.loc[timestamp].to_dict()
                        
                        # Calculate divergence characteristics
                        current_period['convergence_type'] = self._classify_convergence_type(
                            current_period, normalized_values
                        )
                        
                        divergence_periods.append(current_period)
                        in_divergence = False
                        current_period = None
            
            # Handle case where we end in divergence
            if in_divergence and current_period is not None:
                last_timestamp = smoothed_spreads.index[-1]
                current_period['end_date'] = last_timestamp
                current_period['end_idx'] = len(smoothed_spreads) - 1
                current_period['duration_days'] = (last_timestamp - current_period['start_date']).days
                current_period['strategies_at_end'] = normalized_values.loc[last_timestamp].to_dict()
                current_period['convergence_type'] = 'ongoing'
                
                divergence_periods.append(current_period)
            
            # Sort by divergence magnitude
            divergence_periods.sort(key=lambda x: x['max_spread'], reverse=True)
            
            return divergence_periods
            
        except Exception as e:
            self.logger.error(f"Error identifying divergence periods: {str(e)}")
            return []
    
    def _classify_convergence_type(
        self, 
        period: Dict[str, Any], 
        normalized_values: pd.DataFrame
    ) -> str:
        """
        Classify the type of convergence for a divergence period.
        
        Args:
            period: Divergence period data
            normalized_values: Normalized portfolio values
            
        Returns:
            String describing convergence type
        """
        try:
            start_values = period['strategies_at_start']
            end_values = period['strategies_at_end']
            
            leading_strategy = period['leading_strategy']
            lagging_strategy = period['lagging_strategy']
            
            start_leader_value = start_values[leading_strategy]
            end_leader_value = end_values[leading_strategy]
            start_lagging_value = start_values[lagging_strategy]
            end_lagging_value = end_values[lagging_strategy]
            
            # Determine convergence pattern
            if end_leader_value > end_lagging_value:
                if end_leader_value > start_leader_value and end_lagging_value > start_lagging_value:
                    return 'leader_maintained_both_up'
                elif end_leader_value < start_leader_value and end_lagging_value < start_lagging_value:
                    return 'leader_maintained_both_down'
                else:
                    return 'leader_maintained_mixed'
            else:
                return 'leadership_reversed'
            
        except Exception as e:
            self.logger.warning(f"Error classifying convergence type: {str(e)}")
            return 'unknown'
    
    def _perform_statistical_significance_testing(
        self, 
        portfolio_data: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Perform statistical significance testing for performance differences.
        
        This method implements statistical testing for Requirement 8.3.
        
        Args:
            portfolio_data: Portfolio data for all strategies
            
        Returns:
            List of statistical test results
        """
        try:
            from scipy import stats
            
            significance_tests = []
            strategy_names = list(portfolio_data.keys())
            
            # Perform pairwise t-tests on returns
            for i, strategy1 in enumerate(strategy_names):
                for j, strategy2 in enumerate(strategy_names):
                    if i < j:  # Avoid duplicates
                        returns1 = portfolio_data[strategy1]['returns'].dropna()
                        returns2 = portfolio_data[strategy2]['returns'].dropna()
                        
                        # Align returns to same dates
                        common_dates = returns1.index.intersection(returns2.index)
                        returns1_aligned = returns1.loc[common_dates]
                        returns2_aligned = returns2.loc[common_dates]
                        
                        if len(returns1_aligned) > 10 and len(returns2_aligned) > 10:
                            # Perform two-sample t-test
                            t_stat, p_value = stats.ttest_ind(returns1_aligned, returns2_aligned)
                            
                            # Calculate effect size (Cohen's d)
                            pooled_std = np.sqrt(
                                ((len(returns1_aligned) - 1) * returns1_aligned.var() + 
                                 (len(returns2_aligned) - 1) * returns2_aligned.var()) / 
                                (len(returns1_aligned) + len(returns2_aligned) - 2)
                            )
                            cohens_d = (returns1_aligned.mean() - returns2_aligned.mean()) / pooled_std
                            
                            # Perform Kolmogorov-Smirnov test for distribution differences
                            ks_stat, ks_p_value = stats.ks_2samp(returns1_aligned, returns2_aligned)
                            
                            test_result = {
                                'strategy_pair': f"{strategy1} vs {strategy2}",
                                'strategy1': strategy1,
                                'strategy2': strategy2,
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'is_significant': p_value < 0.05,
                                'cohens_d': cohens_d,
                                'effect_size': self._interpret_effect_size(abs(cohens_d)),
                                'ks_statistic': ks_stat,
                                'ks_p_value': ks_p_value,
                                'distribution_different': ks_p_value < 0.05,
                                'mean_return_diff': returns1_aligned.mean() - returns2_aligned.mean(),
                                'volatility_diff': returns1_aligned.std() - returns2_aligned.std(),
                                'sample_size': len(returns1_aligned)
                            }
                            
                            significance_tests.append(test_result)
            
            # Sort by significance (most significant first)
            significance_tests.sort(key=lambda x: x['p_value'])
            
            return significance_tests
            
        except ImportError:
            self.logger.warning("scipy not available for statistical testing")
            return []
        except Exception as e:
            self.logger.error(f"Error performing statistical significance testing: {str(e)}")
            return []
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """
        Interpret Cohen's d effect size.
        
        Args:
            cohens_d: Cohen's d value
            
        Returns:
            String interpretation of effect size
        """
        if cohens_d < 0.2:
            return 'negligible'
        elif cohens_d < 0.5:
            return 'small'
        elif cohens_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _generate_automated_insights(
        self, 
        portfolio_data: Dict[str, Dict[str, Any]], 
        divergence_periods: List[Dict[str, Any]], 
        significance_tests: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Generate automated insights and recommendations based on analysis.
        
        This method implements automated insights generation for Requirement 8.3.
        
        Args:
            portfolio_data: Portfolio data for all strategies
            divergence_periods: Identified divergence periods
            significance_tests: Statistical test results
            
        Returns:
            Dictionary with insights and recommendations
        """
        try:
            insights = []
            recommendations = []
            
            # Analyze divergence patterns
            if len(divergence_periods) > 0:
                max_divergence = max(divergence_periods, key=lambda x: x['max_spread'])
                avg_duration = np.mean([p['duration_days'] for p in divergence_periods])
                
                insights.append(
                    f"Identified {len(divergence_periods)} significant divergence periods. "
                    f"Maximum divergence reached {max_divergence['max_spread']:.1%} on "
                    f"{max_divergence['max_spread_date'].strftime('%Y-%m-%d')}."
                )
                
                insights.append(
                    f"Average divergence duration is {avg_duration:.0f} days, indicating "
                    f"{'frequent short-term' if avg_duration < 30 else 'persistent long-term'} "
                    f"strategy differences."
                )
                
                # Analyze convergence patterns
                convergence_types = [p['convergence_type'] for p in divergence_periods]
                most_common_convergence = max(set(convergence_types), key=convergence_types.count)
                
                insights.append(
                    f"Most common convergence pattern is '{most_common_convergence}', "
                    f"suggesting predictable strategy relationship dynamics."
                )
                
                # Generate recommendations based on divergence analysis
                if max_divergence['max_spread'] > 0.15:  # > 15% divergence
                    recommendations.append(
                        "Consider implementing dynamic allocation between strategies to "
                        "capitalize on large divergence periods."
                    )
                
                if avg_duration > 60:  # Long divergence periods
                    recommendations.append(
                        "Long divergence periods suggest fundamental strategy differences. "
                        "Consider separate allocation buckets for different market conditions."
                    )
            
            # Analyze statistical significance
            significant_tests = [t for t in significance_tests if t['is_significant']]
            if len(significant_tests) > 0:
                insights.append(
                    f"Found {len(significant_tests)} statistically significant performance "
                    f"differences out of {len(significance_tests)} strategy pairs tested."
                )
                
                # Identify strategies with consistently significant differences
                strategy_significance_count = {}
                for test in significant_tests:
                    for strategy in [test['strategy1'], test['strategy2']]:
                        strategy_significance_count[strategy] = strategy_significance_count.get(strategy, 0) + 1
                
                if strategy_significance_count:
                    most_different_strategy = max(strategy_significance_count, key=strategy_significance_count.get)
                    insights.append(
                        f"Strategy '{most_different_strategy}' shows significant differences "
                        f"from {strategy_significance_count[most_different_strategy]} other strategies, "
                        f"indicating unique performance characteristics."
                    )
                    
                    recommendations.append(
                        f"Consider '{most_different_strategy}' as a diversification component "
                        f"due to its distinct performance profile."
                    )
            
            # Analyze correlation patterns
            returns_df = pd.DataFrame({
                name: data['returns'] for name, data in portfolio_data.items()
            })
            correlation_matrix = returns_df.corr()
            
            # Find highly correlated strategies
            high_correlations = []
            for i, strategy1 in enumerate(correlation_matrix.columns):
                for j, strategy2 in enumerate(correlation_matrix.columns):
                    if i < j:
                        corr_value = correlation_matrix.loc[strategy1, strategy2]
                        if abs(corr_value) > 0.8:
                            high_correlations.append((strategy1, strategy2, corr_value))
            
            if high_correlations:
                insights.append(
                    f"Found {len(high_correlations)} highly correlated strategy pairs "
                    f"(correlation > 0.8), suggesting potential redundancy."
                )
                
                recommendations.append(
                    "Consider reducing allocation to highly correlated strategies to "
                    "improve diversification efficiency."
                )
            
            # Performance consistency analysis
            volatilities = {name: data['returns'].std() for name, data in portfolio_data.items()}
            most_consistent = min(volatilities, key=volatilities.get)
            least_consistent = max(volatilities, key=volatilities.get)
            
            insights.append(
                f"Strategy '{most_consistent}' shows the most consistent performance "
                f"(volatility: {volatilities[most_consistent]:.3f}), while '{least_consistent}' "
                f"is most volatile (volatility: {volatilities[least_consistent]:.3f})."
            )
            
            if volatilities[least_consistent] > 2 * volatilities[most_consistent]:
                recommendations.append(
                    f"Consider position sizing adjustments: reduce allocation to "
                    f"'{least_consistent}' and increase allocation to '{most_consistent}' "
                    f"for more stable portfolio performance."
                )
            
            return {
                'insights': insights,
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Error generating automated insights: {str(e)}")
            return {'insights': [], 'recommendations': []}
    
    def _create_divergence_analysis_plot(
        self, 
        portfolio_data: Dict[str, Dict[str, Any]], 
        divergence_periods: List[Dict[str, Any]], 
        significance_tests: List[Dict[str, Any]], 
        title: str
    ) -> Any:
        """
        Create divergence analysis plot with period highlighting.
        
        Args:
            portfolio_data: Portfolio data for all strategies
            divergence_periods: Identified divergence periods
            significance_tests: Statistical test results
            title: Plot title
            
        Returns:
            Plotly figure with divergence analysis
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplot layout
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    'Normalized Performance with Divergence Periods',
                    'Performance Spread Over Time',
                    'Rolling Correlation Analysis',
                    'Statistical Significance Heatmap',
                    'Divergence Period Analysis',
                    'Strategy Performance Distribution'
                ],
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"colspan": 2}, None]
                ],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            # Plot 1: Normalized performance with divergence highlighting
            for i, (name, data) in enumerate(portfolio_data.items()):
                color = colors[i % len(colors)]
                normalized_value = data['normalized_value']
                
                fig.add_trace(
                    go.Scatter(
                        x=normalized_value.index,
                        y=normalized_value.values,
                        mode='lines',
                        name=name,
                        line=dict(color=color, width=2),
                        hovertemplate=f'<b>{name}</b><br>Date: %{{x}}<br>Normalized: %{{y:.3f}}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # Highlight divergence periods
            for period in divergence_periods:
                fig.add_vrect(
                    x0=period['start_date'],
                    x1=period['end_date'],
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    annotation_text=f"Div: {period['max_spread']:.1%}",
                    annotation_position="top left",
                    row=1, col=1
                )
            
            # Plot 2: Performance spread over time
            self._add_performance_spread_plot(fig, portfolio_data, row=1, col=2)
            
            # Plot 3: Rolling correlation analysis
            self._add_rolling_correlation_plot(fig, portfolio_data, row=2, col=1)
            
            # Plot 4: Statistical significance heatmap
            self._add_significance_heatmap(fig, significance_tests, row=2, col=2)
            
            # Plot 5: Divergence period analysis
            self._add_divergence_period_analysis(fig, divergence_periods, row=3, col=1)
            
            # Update layout
            fig.update_layout(
                title=dict(text=title, x=0.5, font=dict(size=16)),
                template=self.plot_config.template,
                width=self.plot_config.width + 400,
                height=self.plot_config.height + 500,
                showlegend=True,
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating divergence analysis plot: {str(e)}")
            import plotly.graph_objects as go
            return go.Figure().add_annotation(
                text=f"Error creating divergence plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    
    def _add_performance_spread_plot(
        self, 
        fig: Any, 
        portfolio_data: Dict[str, Dict[str, Any]], 
        row: int, 
        col: int
    ) -> None:
        """Add performance spread plot to show divergence magnitude over time."""
        try:
            import plotly.graph_objects as go
            
            # Calculate performance spread
            normalized_values = pd.DataFrame({
                name: data['normalized_value'] for name, data in portfolio_data.items()
            })
            
            spread = normalized_values.max(axis=1) - normalized_values.min(axis=1)
            
            fig.add_trace(
                go.Scatter(
                    x=spread.index,
                    y=spread.values,
                    mode='lines',
                    name='Performance Spread',
                    line=dict(color='red', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    hovertemplate='<b>Performance Spread</b><br>Date: %{x}<br>Spread: %{y:.3f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add threshold line
            fig.add_hline(y=0.05, line_dash="dash", line_color="orange", 
                         annotation_text="Divergence Threshold", row=row, col=col)
            
        except Exception as e:
            self.logger.warning(f"Error adding performance spread plot: {str(e)}")
    
    def _add_rolling_correlation_plot(
        self, 
        fig: Any, 
        portfolio_data: Dict[str, Dict[str, Any]], 
        row: int, 
        col: int
    ) -> None:
        """Add rolling correlation analysis plot."""
        try:
            import plotly.graph_objects as go
            
            returns_df = pd.DataFrame({
                name: data['returns'] for name, data in portfolio_data.items()
            })
            
            if len(returns_df.columns) >= 2:
                # Calculate rolling correlation between first two strategies
                strategy1, strategy2 = returns_df.columns[0], returns_df.columns[1]
                rolling_corr = returns_df[strategy1].rolling(window=30).corr(returns_df[strategy2])
                
                fig.add_trace(
                    go.Scatter(
                        x=rolling_corr.index,
                        y=rolling_corr.values,
                        mode='lines',
                        name=f'{strategy1} vs {strategy2} Correlation',
                        line=dict(color='green', width=2),
                        hovertemplate=f'<b>Rolling Correlation</b><br>Date: %{{x}}<br>Correlation: %{{y:.3f}}<extra></extra>'
                    ),
                    row=row, col=col
                )
                
                # Add reference lines
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col)
                fig.add_hline(y=0.8, line_dash="dot", line_color="red", 
                             annotation_text="High Correlation", row=row, col=col)
            
        except Exception as e:
            self.logger.warning(f"Error adding rolling correlation plot: {str(e)}")
    
    def _add_significance_heatmap(
        self, 
        fig: Any, 
        significance_tests: List[Dict[str, Any]], 
        row: int, 
        col: int
    ) -> None:
        """Add statistical significance heatmap."""
        try:
            import plotly.graph_objects as go
            
            if not significance_tests:
                return
            
            # Create significance matrix
            strategies = set()
            for test in significance_tests:
                strategies.add(test['strategy1'])
                strategies.add(test['strategy2'])
            
            strategies = sorted(list(strategies))
            significance_matrix = np.zeros((len(strategies), len(strategies)))
            
            for test in significance_tests:
                i = strategies.index(test['strategy1'])
                j = strategies.index(test['strategy2'])
                p_value = test['p_value']
                significance_matrix[i, j] = -np.log10(p_value) if p_value > 0 else 10
                significance_matrix[j, i] = significance_matrix[i, j]
            
            fig.add_trace(
                go.Heatmap(
                    z=significance_matrix,
                    x=strategies,
                    y=strategies,
                    colorscale='Reds',
                    hovertemplate='<b>%{y} vs %{x}</b><br>-log10(p-value): %{z:.2f}<extra></extra>'
                ),
                row=row, col=col
            )
            
        except Exception as e:
            self.logger.warning(f"Error adding significance heatmap: {str(e)}")
    
    def _add_divergence_period_analysis(
        self, 
        fig: Any, 
        divergence_periods: List[Dict[str, Any]], 
        row: int, 
        col: int
    ) -> None:
        """Add divergence period analysis chart."""
        try:
            import plotly.graph_objects as go
            
            if not divergence_periods:
                return
            
            # Create bar chart of divergence periods
            periods_data = []
            for i, period in enumerate(divergence_periods[:10]):  # Top 10 periods
                periods_data.append({
                    'period': f"Period {i+1}",
                    'duration': period['duration_days'],
                    'magnitude': period['max_spread'] * 100
                })
            
            if periods_data:
                fig.add_trace(
                    go.Bar(
                        x=[p['period'] for p in periods_data],
                        y=[p['magnitude'] for p in periods_data],
                        name='Divergence Magnitude (%)',
                        marker_color='red',
                        opacity=0.7,
                        hovertemplate='<b>%{x}</b><br>Magnitude: %{y:.1f}%<extra></extra>'
                    ),
                    row=row, col=col
                )
            
        except Exception as e:
            self.logger.warning(f"Error adding divergence period analysis: {str(e)}")
    
    def _calculate_rolling_correlations(
        self, 
        portfolio_data: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Calculate rolling correlations between strategies."""
        try:
            returns_df = pd.DataFrame({
                name: data['returns'] for name, data in portfolio_data.items()
            })
            
            correlations = {}
            strategy_names = list(returns_df.columns)
            
            for i, strategy1 in enumerate(strategy_names):
                for j, strategy2 in enumerate(strategy_names):
                    if i < j:
                        rolling_corr = returns_df[strategy1].rolling(window=30).corr(returns_df[strategy2])
                        correlations[f'{strategy1}_vs_{strategy2}'] = rolling_corr
            
            return pd.DataFrame(correlations)
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling correlations: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_performance_spreads(
        self, 
        portfolio_data: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Calculate performance spreads over time."""
        try:
            normalized_values = pd.DataFrame({
                name: data['normalized_value'] for name, data in portfolio_data.items()
            })
            
            spreads = {
                'max_min_spread': normalized_values.max(axis=1) - normalized_values.min(axis=1),
                'std_spread': normalized_values.std(axis=1),
                'range_spread': normalized_values.max(axis=1) / normalized_values.min(axis=1) - 1
            }
            
            return pd.DataFrame(spreads)
            
        except Exception as e:
            self.logger.error(f"Error calculating performance spreads: {str(e)}")
            return pd.DataFrame()
    
    def _create_base_portfolio_plot_optimized(
        self, 
        portfolio: vbt.Portfolio, 
        portfolio_value: pd.Series
    ) -> Any:
        """
        Create optimized base portfolio plot using plotly directly.
        
        This version uses optimized data and includes performance considerations.
        
        Args:
            portfolio: VectorBT portfolio object
            portfolio_value: Optimized portfolio value series
            
        Returns:
            Plotly figure object
        """
        try:
            import plotly.graph_objects as go
            
            # Create the base figure
            fig = go.Figure()
            
            # Add portfolio value line with optimized data
            fig.add_trace(go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value.values,
                mode='lines',
                name='Portfolio Value',
                line=dict(width=2, color='blue'),
                hovertemplate='<b>Portfolio Value</b><br>' +
                               'Date: %{x}<br>' +
                               'Value: $%{y:,.2f}<br>' +
                            '<extra></extra>'
            ))
            
            # Configure layout with performance optimizations
            fig.update_layout(
                template=self.plot_config.template,
                width=self.plot_config.width,
                height=self.plot_config.height,
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                showlegend=True,
                # Performance optimizations
                hovermode='x unified',
                dragmode='pan',  # Faster than zoom for large datasets
                selectdirection='horizontal'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating optimized base portfolio plot: {str(e)}")
            # Fallback to empty figure
            import plotly.graph_objects as go
            return go.Figure()
    
    def _create_portfolio_plot_fallback(
        self, 
        error: Exception, 
        context: ErrorContext, 
        fallback_mode: FallbackMode
    ) -> Any:
        """
        Create fallback output for portfolio plot failures.
        
        Args:
            error: The original error
            context: Error context
            fallback_mode: Requested fallback mode
            
        Returns:
            Fallback output based on mode
        """
        try:
            if fallback_mode == FallbackMode.TEXT_OUTPUT:
                # Create text-based portfolio summary
                portfolio_data = context.input_data or {}
                return self.error_handler.create_text_based_output(
                    portfolio_data, "portfolio_summary"
                )
            
            elif fallback_mode == FallbackMode.BASIC_PLOT:
                # Create basic matplotlib plot
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.text(0.5, 0.5, 
                           f"Portfolio Visualization Error\n\n"
                           f"Error: {str(error)}\n\n"
                           f"Using basic plot fallback",
                           ha='center', va='center', fontsize=12, 
                           transform=ax.transAxes)
                    ax.set_title("Portfolio Plot - Fallback Mode")
                    ax.axis('off')
                    return fig
                except ImportError:
                    # Matplotlib not available, fall back to text
                    return self._create_portfolio_plot_fallback(
                        error, context, FallbackMode.TEXT_OUTPUT
                    )
            
            else:
                # Default to text output
                return "Portfolio visualization failed. Please check system configuration."
                
        except Exception as fallback_error:
            self.logger.error(f"Error in portfolio plot fallback: {str(fallback_error)}")
            return f"Portfolio visualization failed: {str(error)}"
    
    def get_performance_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance diagnostics for the visualization engine.
        
        Returns:
            Dictionary with performance diagnostics and recommendations
        """
        try:
            diagnostics = {
                'engine_status': 'operational',
                'error_handler_enabled': self.error_handler is not None,
                'performance_optimizer_enabled': self.performance_optimizer is not None,
                'environment_validation': {},
                'performance_recommendations': [],
                'error_statistics': {},
                'memory_usage': {}
            }
            
            # Get environment validation
            if self.error_handler:
                diagnostics['environment_validation'] = self.error_handler.validate_visualization_environment()
                diagnostics['error_statistics'] = self.error_handler.get_error_statistics()
            
            # Get performance recommendations
            if self.performance_optimizer:
                perf_recommendations = self.performance_optimizer.get_performance_recommendations()
                diagnostics['performance_recommendations'] = perf_recommendations.get('recommendations', [])
                diagnostics['performance_statistics'] = perf_recommendations.get('statistics', {})
                diagnostics['memory_usage'] = {
                    'current_mb': perf_recommendations.get('current_memory_mb', 0),
                    'baseline_mb': perf_recommendations.get('baseline_memory_mb', 0),
                    'cache_size_mb': perf_recommendations.get('cache_size_mb', 0)
                }
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Error getting performance diagnostics: {str(e)}")
            return {
                'engine_status': 'error',
                'error': str(e)
            }
    
    def cleanup_resources(self) -> Dict[str, Any]:
        """
        Cleanup visualization resources and free memory.
        
        Returns:
            Dictionary with cleanup results
        """
        try:
            cleanup_results = {
                'cache_cleared': False,
                'memory_freed_mb': 0.0,
                'actions_performed': []
            }
            
            # Clear performance optimizer cache
            if self.performance_optimizer:
                cache_result = self.performance_optimizer.clear_cache()
                cleanup_results['cache_cleared'] = True
                cleanup_results['memory_freed_mb'] += cache_result.get('memory_freed_mb', 0)
                cleanup_results['actions_performed'].append('Cleared performance optimizer cache')
            
            # Force garbage collection
            import gc
            gc.collect()
            cleanup_results['actions_performed'].append('Performed garbage collection')
            
            self.logger.info(
                f"Resource cleanup completed: "
                f"freed {cleanup_results['memory_freed_mb']:.1f}MB, "
                f"actions: {len(cleanup_results['actions_performed'])}"
            )
            
            return cleanup_results
            
        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {str(e)}")
            return {
                'cache_cleared': False,
                'memory_freed_mb': 0.0,
                'actions_performed': [],
                'error': str(e)
            }