"""
Enhanced Portfolio Creation Engine for VectorBT Visualization Enhancement.

This module provides comprehensive VectorBT portfolio creation with realistic parameters,
advanced trading rules, and risk management capabilities.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import time
from dataclasses import dataclass

from .signal_alignment import SignalAlignmentEngine, AlignedSignals
from .portfolio_config import PortfolioConfig, VisualizationResult
from ..utils.exceptions import BacktestingError, DataValidationError


@dataclass
class PortfolioCreationResult:
    """Result object for portfolio creation operations."""
    
    portfolio: Optional[vbt.Portfolio]
    aligned_signals: Optional[AlignedSignals]
    position_sizes: Optional[pd.Series]
    creation_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate the result object after initialization."""
        if not self.success and self.error_message is None:
            raise ValueError("Failed results must include an error message")
        
        if self.success and self.portfolio is None:
            raise ValueError("Successful results must include a portfolio object")


class EnhancedPortfolioEngine:
    """
    Enhanced VectorBT portfolio creation engine with realistic parameters.
    
    This engine provides comprehensive VectorBT portfolio creation with:
    - Proper signal alignment for ML predictions
    - Advanced position sizing strategies
    - Realistic trading costs and slippage
    - Risk management rules (stop-loss, take-profit)
    - Portfolio parameter validation and optimization
    """
    
    def __init__(self, 
                 portfolio_config: Optional[PortfolioConfig] = None,
                 signal_aligner: Optional[SignalAlignmentEngine] = None):
        """
        Initialize the Enhanced Portfolio Engine.
        
        Args:
            portfolio_config: Configuration for portfolio creation
            signal_aligner: Signal alignment engine (optional)
        """
        self.portfolio_config = portfolio_config or PortfolioConfig()
        self.signal_aligner = signal_aligner or SignalAlignmentEngine()
        
        self.logger = logging.getLogger(__name__)
        
        # Validate configuration
        self.portfolio_config.validate()
        
        self.logger.info("Enhanced Portfolio Engine initialized")
    
    def create_vectorbt_portfolio(
        self,
        close_prices: pd.Series,
        entry_signals: pd.Series,
        exit_signals: pd.Series,
        config: Optional[PortfolioConfig] = None
    ) -> vbt.Portfolio:
        """
        Create VectorBT portfolio with enhanced configuration.
        
        This method creates a VectorBT portfolio with full parameter support including:
        - Realistic trading costs and slippage
        - Advanced position sizing strategies
        - Risk management rules (stop-loss, take-profit)
        - Proper signal handling and validation
        
        Args:
            close_prices: Historical closing prices
            entry_signals: Boolean series for entry points
            exit_signals: Boolean series for exit points
            config: Portfolio configuration parameters (optional)
            
        Returns:
            Configured VectorBT Portfolio object
            
        Raises:
            BacktestingError: If portfolio creation fails
            DataValidationError: If inputs are invalid
        """
        try:
            start_time = time.time()
            
            # Use provided config or default
            portfolio_config = config or self.portfolio_config
            
            # Validate inputs
            self._validate_portfolio_inputs(close_prices, entry_signals, exit_signals)
            
            # Calculate position sizes based on strategy
            position_sizes = self.calculate_position_sizes(
                close_prices, 
                portfolio_config.size_strategy,
                portfolio_config.init_cash
            )
            
            # Get VectorBT parameters
            vbt_params = portfolio_config.to_vectorbt_params()
            
            # Log portfolio creation details
            self.logger.info(
                f"Creating VectorBT portfolio: {len(close_prices)} periods, "
                f"{entry_signals.sum()} entries, {exit_signals.sum()} exits"
            )
            
            # Create VectorBT portfolio with full parameter support
            portfolio = vbt.Portfolio.from_signals(
                close=close_prices,
                entries=entry_signals,
                exits=exit_signals,
                size=position_sizes,
                size_type='amount',  # Use dollar amounts
                **vbt_params
            )
            
            creation_time = time.time() - start_time
            
            # Log creation results
            num_trades = portfolio.trades.count()
            self.logger.info(
                f"Portfolio created successfully in {creation_time:.2f}s "
                f"({num_trades} trades generated)"
            )
            
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Error creating VectorBT portfolio: {str(e)}")
            raise BacktestingError(f"Portfolio creation failed: {str(e)}")
    
    def create_portfolio_from_predictions(
        self,
        predictions: np.ndarray,
        price_data: pd.DataFrame,
        test_start_idx: int,
        config: Optional[PortfolioConfig] = None
    ) -> PortfolioCreationResult:
        """
        Create VectorBT portfolio from ML predictions with signal alignment.
        
        This method integrates signal alignment with portfolio creation to provide
        a complete solution for converting ML predictions into VectorBT portfolios.
        
        Args:
            predictions: ML model predictions (0=sell, 1=hold, 2=buy)
            price_data: Historical price data with 'Close' column
            test_start_idx: Index where test period begins
            config: Portfolio configuration (optional)
            
        Returns:
            PortfolioCreationResult with portfolio and metadata
        """
        try:
            start_time = time.time()
            
            # Use provided config or default
            portfolio_config = config or self.portfolio_config
            
            # Validate inputs
            if 'Close' not in price_data.columns and 'close' not in price_data.columns:
                raise DataValidationError("Price data must contain 'Close' or 'close' column")
            
            # Get close prices
            close_prices = price_data.get('Close', price_data.get('close'))
            
            # Align predictions to full timeline
            self.logger.info(f"Aligning {len(predictions)} predictions to timeline")
            aligned_signals = self.signal_aligner.align_predictions_to_timeline(
                predictions, price_data, test_start_idx
            )
            
            # Create portfolio with aligned signals
            portfolio = self.create_vectorbt_portfolio(
                close_prices=close_prices,
                entry_signals=aligned_signals.entry_signals,
                exit_signals=aligned_signals.exit_signals,
                config=portfolio_config
            )
            
            # Calculate position sizes for metadata
            position_sizes = self.calculate_position_sizes(
                close_prices,
                portfolio_config.size_strategy,
                portfolio_config.init_cash
            )
            
            creation_time = time.time() - start_time
            
            # Create metadata
            metadata = {
                'prediction_count': len(predictions),
                'timeline_length': len(price_data),
                'test_start_idx': test_start_idx,
                'test_end_idx': test_start_idx + len(predictions),
                'num_trades': portfolio.trades.count(),
                'portfolio_config': portfolio_config.__dict__,
                'creation_timestamp': pd.Timestamp.now()
            }
            
            return PortfolioCreationResult(
                portfolio=portfolio,
                aligned_signals=aligned_signals,
                position_sizes=position_sizes,
                creation_time=creation_time,
                success=True,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error creating portfolio from predictions: {str(e)}")
            return PortfolioCreationResult(
                portfolio=None,
                aligned_signals=None,
                position_sizes=None,
                creation_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def calculate_position_sizes(
        self,
        prices: pd.Series,
        sizing_method: str,
        capital: float
    ) -> pd.Series:
        """
        Calculate position sizes based on strategy.
        
        This method implements multiple position sizing strategies:
        - Fixed amount: Use fixed dollar amount per trade
        - Fixed shares: Use fixed number of shares per trade
        - Percent equity: Use percentage of available equity
        - Volatility target: Adjust size based on volatility
        - Risk parity: Size inversely proportional to risk
        
        Args:
            prices: Price series for sizing calculation
            sizing_method: 'fixed_amount', 'fixed_shares', 'percent_equity', 
                          'volatility_target', 'risk_parity'
            capital: Available capital
            
        Returns:
            Series of position sizes (in dollar amounts)
            
        Raises:
            DataValidationError: If invalid sizing method or parameters
        """
        try:
            # Validate inputs
            if prices.empty:
                raise DataValidationError("Price series cannot be empty")
            
            if capital <= 0:
                raise DataValidationError(f"Capital must be positive, got {capital}")
            
            # Calculate volatility for dynamic sizing methods
            volatility = None
            if sizing_method in ['volatility_target', 'risk_parity']:
                returns = prices.pct_change(fill_method=None).dropna()
                if len(returns) >= 10:  # Need minimum data for volatility calculation
                    volatility = returns.rolling(window=20, min_periods=10).std() * np.sqrt(252)
                    volatility = volatility.reindex(prices.index, method='ffill')
                else:
                    # Use default volatility if insufficient data
                    volatility = pd.Series(0.15, index=prices.index)  # 15% default
            
            # Use portfolio config to calculate sizes
            position_sizes = self.portfolio_config.calculate_position_sizes(
                prices=prices,
                sizing_method=sizing_method,
                capital=capital,
                volatility=volatility
            )
            
            self.logger.debug(
                f"Calculated position sizes: method={sizing_method}, "
                f"mean_size=${position_sizes.mean():.2f}, "
                f"max_size=${position_sizes.max():.2f}"
            )
            
            return position_sizes
            
        except Exception as e:
            self.logger.error(f"Error calculating position sizes: {str(e)}")
            raise DataValidationError(f"Position size calculation failed: {str(e)}")
    
    def validate_portfolio_parameters(
        self,
        config: PortfolioConfig,
        prices: pd.Series,
        entry_signals: pd.Series,
        exit_signals: pd.Series
    ) -> Dict[str, Any]:
        """
        Validate portfolio parameters before creation.
        
        This method performs comprehensive validation of portfolio parameters
        and provides recommendations for optimization.
        
        Args:
            config: Portfolio configuration to validate
            prices: Price series
            entry_signals: Entry signals
            exit_signals: Exit signals
            
        Returns:
            Dictionary with validation results and recommendations
        """
        try:
            validation_result = {
                'valid': True,
                'warnings': [],
                'errors': [],
                'recommendations': [],
                'statistics': {}
            }
            
            # Validate configuration
            try:
                config.validate()
            except DataValidationError as e:
                validation_result['valid'] = False
                validation_result['errors'].append(str(e))
            
            # Validate data alignment
            if len(prices) != len(entry_signals) or len(prices) != len(exit_signals):
                validation_result['valid'] = False
                validation_result['errors'].append(
                    f"Data length mismatch: prices={len(prices)}, "
                    f"entries={len(entry_signals)}, exits={len(exit_signals)}"
                )
            
            # Check signal statistics
            entry_count = entry_signals.sum()
            exit_count = exit_signals.sum()
            simultaneous_signals = (entry_signals & exit_signals).sum()
            
            validation_result['statistics'].update({
                'total_periods': len(prices),
                'entry_signals': entry_count,
                'exit_signals': exit_count,
                'simultaneous_signals': simultaneous_signals,
                'signal_density': (entry_count + exit_count) / len(prices) if len(prices) > 0 else 0
            })
            
            # Check for potential issues
            if entry_count == 0:
                validation_result['warnings'].append("No entry signals found")
            
            if exit_count == 0:
                validation_result['warnings'].append("No exit signals found")
            
            if simultaneous_signals > 0:
                validation_result['warnings'].append(
                    f"Found {simultaneous_signals} simultaneous entry/exit signals"
                )
            
            # Check signal density
            signal_density = (entry_count + exit_count) / len(prices)
            if signal_density > 0.5:
                validation_result['warnings'].append(
                    f"High signal density ({signal_density:.2%}) may indicate overtrading"
                )
            elif signal_density < 0.01:
                validation_result['warnings'].append(
                    f"Low signal density ({signal_density:.2%}) may indicate insufficient trading"
                )
            
            # Validate position sizing
            try:
                position_sizes = self.calculate_position_sizes(
                    prices, config.size_strategy, config.init_cash
                )
                
                max_position = position_sizes.max()
                if max_position > config.init_cash:
                    validation_result['warnings'].append(
                        f"Maximum position size (${max_position:.2f}) exceeds initial capital "
                        f"(${config.init_cash:.2f})"
                    )
                
                validation_result['statistics']['max_position_size'] = max_position
                validation_result['statistics']['avg_position_size'] = position_sizes.mean()
                
            except Exception as e:
                validation_result['errors'].append(f"Position sizing validation failed: {str(e)}")
                validation_result['valid'] = False
            
            # Generate recommendations
            if validation_result['valid']:
                if signal_density < 0.05:
                    validation_result['recommendations'].append(
                        "Consider increasing signal sensitivity for more trading opportunities"
                    )
                
                if config.fees > 0.01:
                    validation_result['recommendations'].append(
                        f"High transaction fees ({config.fees:.2%}) may impact performance"
                    )
                
                if config.stop_loss is None:
                    validation_result['recommendations'].append(
                        "Consider adding stop-loss for risk management"
                    )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating portfolio parameters: {str(e)}")
            return {
                'valid': False,
                'errors': [str(e)],
                'warnings': [],
                'recommendations': [],
                'statistics': {}
            }
    
    def optimize_portfolio_parameters(
        self,
        config: PortfolioConfig,
        prices: pd.Series,
        entry_signals: pd.Series,
        exit_signals: pd.Series
    ) -> PortfolioConfig:
        """
        Optimize portfolio parameters for edge cases.
        
        This method automatically adjusts portfolio parameters to handle
        edge cases and improve portfolio performance.
        
        Args:
            config: Original portfolio configuration
            prices: Price series
            entry_signals: Entry signals
            exit_signals: Exit signals
            
        Returns:
            Optimized portfolio configuration
        """
        try:
            # Create a copy of the configuration
            optimized_config = PortfolioConfig(**config.__dict__)
            
            # Validate current configuration
            validation = self.validate_portfolio_parameters(
                config, prices, entry_signals, exit_signals
            )
            
            # Apply optimizations based on validation results
            if not validation['valid']:
                self.logger.info("Applying automatic parameter optimizations")
                
                # Adjust position sizing if it exceeds capital
                max_position = validation['statistics'].get('max_position_size', 0)
                if max_position > config.init_cash:
                    # Reduce position size to 90% of available capital
                    if config.size_strategy == 'fixed_amount':
                        optimized_config.size_value = config.init_cash * 0.9
                    elif config.size_strategy == 'percent_equity':
                        optimized_config.size_value = min(config.size_value, 0.9)
                    
                    self.logger.info(
                        f"Adjusted position size from {config.size_value} to "
                        f"{optimized_config.size_value}"
                    )
                
                # Add stop-loss if none exists and there are risky conditions
                if config.stop_loss is None and validation['statistics'].get('signal_density', 0) > 0.3:
                    optimized_config.stop_loss = 0.1  # 10% stop-loss
                    self.logger.info("Added 10% stop-loss for risk management")
                
                # Adjust fees if they're too high
                if config.fees > 0.01:  # More than 1%
                    optimized_config.fees = 0.0025  # Reduce to 0.25%
                    self.logger.info(f"Reduced fees from {config.fees:.2%} to {optimized_config.fees:.2%}")
            
            # Validate optimized configuration
            optimized_config.validate()
            
            return optimized_config
            
        except Exception as e:
            self.logger.warning(f"Error optimizing portfolio parameters: {str(e)}")
            # Return original configuration if optimization fails
            return config
    
    def create_portfolio_health_check(
        self,
        portfolio: vbt.Portfolio,
        config: PortfolioConfig
    ) -> Dict[str, Any]:
        """
        Create portfolio health checks and diagnostic reporting.
        
        This method analyzes the created portfolio and provides diagnostic
        information about its health and performance characteristics.
        
        Args:
            portfolio: VectorBT portfolio object
            config: Portfolio configuration used
            
        Returns:
            Dictionary with health check results and diagnostics
        """
        try:
            health_check = {
                'overall_health': 'good',
                'issues': [],
                'warnings': [],
                'recommendations': [],
                'metrics': {},
                'diagnostics': {}
            }
            
            # Basic portfolio metrics
            try:
                total_return = portfolio.total_return()
                sharpe_ratio = portfolio.sharpe_ratio()
                max_drawdown = portfolio.max_drawdown()
                num_trades = portfolio.trades.count()
                
                health_check['metrics'].update({
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'num_trades': num_trades
                })
                
            except Exception as e:
                health_check['issues'].append(f"Could not calculate basic metrics: {str(e)}")
                health_check['overall_health'] = 'poor'
            
            # Trade analysis
            if num_trades > 0:
                try:
                    win_rate = portfolio.trades.win_rate()
                    profit_factor = portfolio.trades.profit_factor()
                    
                    health_check['metrics'].update({
                        'win_rate': win_rate,
                        'profit_factor': profit_factor
                    })
                    
                    # Check for concerning patterns
                    if win_rate < 0.3:
                        health_check['warnings'].append(
                            f"Low win rate ({win_rate:.2%}) may indicate poor signal quality"
                        )
                    
                    if profit_factor < 1.0:
                        health_check['issues'].append(
                            f"Profit factor below 1.0 ({profit_factor:.2f}) indicates net losses"
                        )
                        health_check['overall_health'] = 'poor'
                    
                except Exception as e:
                    health_check['warnings'].append(f"Could not analyze trades: {str(e)}")
            
            else:
                health_check['issues'].append("No trades generated - check signal quality")
                health_check['overall_health'] = 'poor'
            
            # Risk analysis
            if max_drawdown < -0.5:  # More than 50% drawdown
                health_check['issues'].append(
                    f"Excessive drawdown ({max_drawdown:.2%}) indicates high risk"
                )
                health_check['overall_health'] = 'poor'
            elif max_drawdown < -0.2:  # More than 20% drawdown
                health_check['warnings'].append(
                    f"High drawdown ({max_drawdown:.2%}) - consider risk management"
                )
            
            # Configuration diagnostics
            health_check['diagnostics'].update({
                'config_used': config.__dict__,
                'portfolio_start_value': portfolio.value().iloc[0] if len(portfolio.value()) > 0 else 0,
                'portfolio_end_value': portfolio.value().iloc[-1] if len(portfolio.value()) > 0 else 0,
                'total_periods': len(portfolio.value()),
                'cash_utilization': 1 - (portfolio.cash().iloc[-1] / config.init_cash) if len(portfolio.cash()) > 0 else 0
            })
            
            # Generate recommendations
            if health_check['overall_health'] == 'poor':
                health_check['recommendations'].extend([
                    "Review signal generation strategy",
                    "Consider adjusting position sizing",
                    "Implement stricter risk management"
                ])
            elif len(health_check['warnings']) > 0:
                health_check['recommendations'].extend([
                    "Monitor risk metrics closely",
                    "Consider parameter optimization"
                ])
            
            return health_check
            
        except Exception as e:
            self.logger.error(f"Error creating portfolio health check: {str(e)}")
            return {
                'overall_health': 'unknown',
                'issues': [str(e)],
                'warnings': [],
                'recommendations': [],
                'metrics': {},
                'diagnostics': {}
            }
    
    def _validate_portfolio_inputs(
        self,
        close_prices: pd.Series,
        entry_signals: pd.Series,
        exit_signals: pd.Series
    ) -> None:
        """
        Validate inputs for portfolio creation.
        
        Args:
            close_prices: Price series
            entry_signals: Entry signals
            exit_signals: Exit signals
            
        Raises:
            DataValidationError: If inputs are invalid
        """
        # Check for None or empty inputs
        if close_prices is None or len(close_prices) == 0:
            raise DataValidationError("Close prices cannot be None or empty")
        
        if entry_signals is None or len(entry_signals) == 0:
            raise DataValidationError("Entry signals cannot be None or empty")
        
        if exit_signals is None or len(exit_signals) == 0:
            raise DataValidationError("Exit signals cannot be None or empty")
        
        # Check data types
        if not isinstance(close_prices, pd.Series):
            raise DataValidationError(f"Close prices must be pandas Series, got {type(close_prices)}")
        
        if not isinstance(entry_signals, pd.Series):
            raise DataValidationError(f"Entry signals must be pandas Series, got {type(entry_signals)}")
        
        if not isinstance(exit_signals, pd.Series):
            raise DataValidationError(f"Exit signals must be pandas Series, got {type(exit_signals)}")
        
        # Check lengths match
        if len(close_prices) != len(entry_signals):
            raise DataValidationError(
                f"Price and entry signal lengths don't match: {len(close_prices)} != {len(entry_signals)}"
            )
        
        if len(close_prices) != len(exit_signals):
            raise DataValidationError(
                f"Price and exit signal lengths don't match: {len(close_prices)} != {len(exit_signals)}"
            )
        
        # Check signal types
        if entry_signals.dtype != bool:
            raise DataValidationError(f"Entry signals must be boolean, got {entry_signals.dtype}")
        
        if exit_signals.dtype != bool:
            raise DataValidationError(f"Exit signals must be boolean, got {exit_signals.dtype}")
        
        # Check for valid prices
        if close_prices.isna().any():
            raise DataValidationError("Close prices contain NaN values")
        
        if (close_prices <= 0).any():
            raise DataValidationError("Close prices must be positive")
        
        # Check index alignment
        if not close_prices.index.equals(entry_signals.index):
            raise DataValidationError("Close prices and entry signals indices don't match")
        
        if not close_prices.index.equals(exit_signals.index):
            raise DataValidationError("Close prices and exit signals indices don't match")