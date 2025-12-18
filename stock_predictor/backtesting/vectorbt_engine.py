"""
Enhanced Backtesting Engine using VectorBT for Stock Direction Predictor.
Provides sophisticated portfolio simulation with advanced metrics and visualization.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime

from ..interfaces import IBacktestingEngine
from ..utils.exceptions import BacktestingError
from ..visualization import VectorBTVisualizationEngine, PortfolioConfig, PlotConfig


@dataclass
class VectorBTBacktestResult:
    """Enhanced backtest result using VectorBT portfolio."""
    
    # Basic metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    # Advanced metrics
    calmar_ratio: float
    sortino_ratio: float
    value_at_risk: float
    conditional_var: float
    
    # Portfolio data
    portfolio_values: pd.Series
    returns: pd.Series
    drawdowns: pd.Series
    trade_log: pd.DataFrame
    
    # VectorBT portfolio object for advanced analysis
    portfolio: Any  # vbt.Portfolio
    
    # Additional statistics
    num_trades: int
    avg_trade_duration: float
    best_trade: float
    worst_trade: float
    
    # Risk metrics
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float


class VectorBTBacktestingEngine(IBacktestingEngine):
    """
    Enhanced backtesting engine using VectorBT for sophisticated portfolio simulation.
    
    Features:
    - Advanced portfolio metrics using VectorBT
    - Multiple position sizing strategies
    - Transaction cost modeling
    - Risk management rules
    - Comprehensive performance analytics
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 max_position_size: float = 1.0,
                 risk_free_rate: float = 0.02,
                 benchmark_symbol: str = 'SPY'):
        """
        Initialize the VectorBT backtesting engine.
        
        Args:
            initial_capital: Starting portfolio value
            transaction_cost: Transaction cost as percentage of trade value
            slippage: Slippage as percentage of trade value
            max_position_size: Maximum position size as fraction of portfolio
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            benchmark_symbol: Benchmark symbol for beta/alpha calculation
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate
        self.benchmark_symbol = benchmark_symbol
        
        self.logger = logging.getLogger(__name__)
        
        # Configure VectorBT settings
        vbt.settings.portfolio['init_cash'] = initial_capital
        vbt.settings.portfolio['fees'] = transaction_cost
        vbt.settings.portfolio['slippage'] = slippage
    
    def simulate_trading(self, signals: pd.Series, prices: pd.Series, 
                        initial_capital: Optional[float] = None) -> VectorBTBacktestResult:
        """
        Simulate trading using VectorBT portfolio simulation.
        
        Args:
            signals: Trading signals (-1: sell, 0: hold, 1: buy)
            prices: Price series for the asset
            initial_capital: Override initial capital (optional)
            
        Returns:
            VectorBTBacktestResult with comprehensive metrics
        """
        try:
            # Use provided capital or default
            capital = initial_capital or self.initial_capital
            
            # Align signals and prices
            aligned_data = pd.concat([signals, prices], axis=1, join='inner')
            aligned_data.columns = ['signals', 'prices']
            aligned_data = aligned_data.dropna()
            
            if len(aligned_data) == 0:
                raise BacktestingError("No valid data points after alignment")
            
            # Ensure proper datetime index with frequency
            if not isinstance(aligned_data.index, pd.DatetimeIndex):
                aligned_data.index = pd.to_datetime(aligned_data.index)
            
            # Infer frequency if not set
            if aligned_data.index.freq is None:
                aligned_data.index.freq = pd.infer_freq(aligned_data.index)
                if aligned_data.index.freq is None:
                    # Default to business day frequency for stock data
                    aligned_data = aligned_data.asfreq('B', method='ffill')
            
            signals_clean = aligned_data['signals']
            prices_clean = aligned_data['prices']
            
            # Convert signals to entries and exits
            entries, exits = self._convert_signals_to_entries_exits(signals_clean)
            
            # Create VectorBT portfolio
            portfolio = vbt.Portfolio.from_signals(
                close=prices_clean,
                entries=entries,
                exits=exits,
                init_cash=capital,
                fees=self.transaction_cost,
                slippage=self.slippage,
                size=self._calculate_position_sizes(prices_clean, capital),
                size_type='amount',  # Use absolute amounts
                accumulate=False  # Don't accumulate positions
            )
            
            # Calculate comprehensive metrics
            result = self._calculate_comprehensive_metrics(
                portfolio, signals_clean, prices_clean
            )
            
            self.logger.info(f"Backtesting completed: {len(aligned_data)} periods, "
                           f"{result.num_trades} trades, "
                           f"{result.total_return:.2%} total return")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in backtesting simulation: {str(e)}")
            raise BacktestingError(f"Backtesting failed: {str(e)}")
    
    def _convert_signals_to_entries_exits(self, signals: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Convert trading signals to entry and exit signals for VectorBT.
        
        Args:
            signals: Trading signals (-1: sell, 0: hold, 1: buy)
            
        Returns:
            Tuple of (entries, exits) boolean series
        """
        # Initialize entry and exit series
        entries = pd.Series(False, index=signals.index)
        exits = pd.Series(False, index=signals.index)
        
        # Track current position
        position = 0  # 0: no position, 1: long, -1: short
        
        for i, signal in enumerate(signals):
            if signal == 1 and position <= 0:  # Buy signal and not already long
                entries.iloc[i] = True
                position = 1
            elif signal == -1 and position >= 0:  # Sell signal and not already short
                exits.iloc[i] = True
                position = -1
            elif signal == 0 and position != 0:  # Hold signal - close position
                exits.iloc[i] = True
                position = 0
        
        return entries, exits
    
    def _calculate_position_sizes(self, prices: pd.Series, capital: float) -> pd.Series:
        """
        Calculate position sizes based on available capital and risk management.
        
        Args:
            prices: Price series
            capital: Available capital
            
        Returns:
            Series of position sizes (in currency units)
        """
        # Simple equal-weight position sizing
        # Use a fraction of available capital per trade
        position_fraction = self.max_position_size
        
        # Calculate number of shares we can buy with available capital
        max_shares = (capital * position_fraction) / prices
        
        # Convert to currency amount
        position_sizes = max_shares * prices
        
        return position_sizes
    
    def _calculate_comprehensive_metrics(self, portfolio: Any, 
                                       signals: pd.Series, 
                                       prices: pd.Series) -> VectorBTBacktestResult:
        """
        Calculate comprehensive performance metrics using VectorBT portfolio.
        
        Args:
            portfolio: VectorBT portfolio object
            signals: Original trading signals
            prices: Price series
            
        Returns:
            VectorBTBacktestResult with all metrics
        """
        # Basic portfolio metrics
        total_return = portfolio.total_return()
        annualized_return = portfolio.annualized_return()
        volatility = portfolio.annualized_volatility()
        sharpe_ratio = portfolio.sharpe_ratio(risk_free=self.risk_free_rate)
        max_drawdown = portfolio.max_drawdown()
        
        # Portfolio values and returns
        portfolio_values = portfolio.value()
        returns = portfolio.returns()
        drawdowns = portfolio.drawdowns
        
        # Trade statistics
        trades = portfolio.trades
        num_trades = trades.count()
        
        if num_trades > 0:
            win_rate = trades.win_rate()
            profit_factor = trades.profit_factor()
            avg_trade_duration = trades.duration.mean()
            best_trade = trades.pnl.max()
            worst_trade = trades.pnl.min()
        else:
            win_rate = 0.0
            profit_factor = 1.0
            avg_trade_duration = 0.0
            best_trade = 0.0
            worst_trade = 0.0
        
        # Advanced risk metrics
        try:
            calmar_ratio = portfolio.calmar_ratio()
            # Calculate Sortino ratio manually since VectorBT API changed
            returns_series = portfolio.returns()
            if len(returns_series) > 0:
                downside_returns = returns_series[returns_series < 0]
                if len(downside_returns) > 0:
                    downside_deviation = downside_returns.std()
                    excess_return = returns_series.mean() - self.risk_free_rate / 252  # Daily risk-free rate
                    sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0.0
                else:
                    sortino_ratio = 0.0
            else:
                sortino_ratio = 0.0
            
            # Value at Risk (95% confidence)
            returns_array = returns.dropna()
            if len(returns_array) > 0:
                value_at_risk = np.percentile(returns_array, 5)
                conditional_var = returns_array[returns_array <= value_at_risk].mean()
            else:
                value_at_risk = 0.0
                conditional_var = 0.0
                
        except Exception as e:
            self.logger.warning(f"Could not calculate advanced metrics: {str(e)}")
            calmar_ratio = 0.0
            sortino_ratio = 0.0
            value_at_risk = 0.0
            conditional_var = 0.0
        
        # Beta and Alpha (simplified calculation)
        try:
            # For now, use simplified calculations
            # In production, you would fetch benchmark data
            beta = 1.0  # Placeholder
            alpha = annualized_return - (self.risk_free_rate + beta * 0.10)  # Assuming 10% market return
            information_ratio = 0.0  # Placeholder
            tracking_error = 0.0  # Placeholder
        except Exception:
            beta = 1.0
            alpha = 0.0
            information_ratio = 0.0
            tracking_error = 0.0
        
        # Create trade log
        trade_log = self._create_trade_log(portfolio, signals, prices)
        
        return VectorBTBacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            value_at_risk=value_at_risk,
            conditional_var=conditional_var,
            portfolio_values=portfolio_values,
            returns=returns,
            drawdowns=drawdowns,
            trade_log=trade_log,
            portfolio=portfolio,
            num_trades=num_trades,
            avg_trade_duration=avg_trade_duration,
            best_trade=best_trade,
            worst_trade=worst_trade,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            tracking_error=tracking_error
        )
    
    def _create_trade_log(self, portfolio: Any, signals: pd.Series, 
                         prices: pd.Series) -> pd.DataFrame:
        """
        Create detailed trade log from VectorBT portfolio.
        
        Args:
            portfolio: VectorBT portfolio object
            signals: Trading signals
            prices: Price series
            
        Returns:
            DataFrame with detailed trade information
        """
        try:
            trades = portfolio.trades
            
            if trades.count() == 0:
                return pd.DataFrame(columns=[
                    'entry_date', 'exit_date', 'entry_price', 'exit_price',
                    'size', 'pnl', 'return_pct', 'duration', 'signal'
                ])
            
            trade_log = pd.DataFrame({
                'entry_date': trades.entry_idx,
                'exit_date': trades.exit_idx,
                'entry_price': trades.entry_price,
                'exit_price': trades.exit_price,
                'size': trades.size,
                'pnl': trades.pnl,
                'return_pct': trades.return_pct,
                'duration': trades.duration
            })
            
            # Add signal information
            trade_log['signal'] = 'long'  # VectorBT default is long positions
            
            return trade_log
            
        except Exception as e:
            self.logger.warning(f"Could not create detailed trade log: {str(e)}")
            return pd.DataFrame(columns=[
                'entry_date', 'exit_date', 'entry_price', 'exit_price',
                'size', 'pnl', 'return_pct', 'duration', 'signal'
            ])
    
    def calculate_portfolio_metrics(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """
        Calculate portfolio metrics from portfolio values series.
        
        Args:
            portfolio_values: Series of portfolio values over time
            
        Returns:
            Dictionary of portfolio metrics
        """
        if len(portfolio_values) < 2:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        # Calculate returns
        returns = portfolio_values.pct_change().dropna()
        
        # Total return
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        
        # Annualized return
        periods_per_year = 252  # Trading days
        years = len(portfolio_values) / periods_per_year
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def generate_trade_log(self, signals: pd.Series, prices: pd.Series) -> pd.DataFrame:
        """
        Generate trade log from signals and prices.
        
        Args:
            signals: Trading signals
            prices: Price series
            
        Returns:
            DataFrame with trade log
        """
        # Run simulation to get trade log
        result = self.simulate_trading(signals, prices)
        return result.trade_log
    
    def create_performance_report(self, result: VectorBTBacktestResult) -> Dict[str, Any]:
        """
        Create comprehensive performance report from backtest result.
        
        Args:
            result: VectorBT backtest result
            
        Returns:
            Dictionary with performance report
        """
        return {
            'summary': {
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'volatility': result.volatility,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'calmar_ratio': result.calmar_ratio,
                'sortino_ratio': result.sortino_ratio
            },
            'trade_statistics': {
                'num_trades': result.num_trades,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'avg_trade_duration': result.avg_trade_duration,
                'best_trade': result.best_trade,
                'worst_trade': result.worst_trade
            },
            'risk_metrics': {
                'value_at_risk': result.value_at_risk,
                'conditional_var': result.conditional_var,
                'beta': result.beta,
                'alpha': result.alpha,
                'information_ratio': result.information_ratio,
                'tracking_error': result.tracking_error
            },
            'portfolio_data': {
                'start_value': result.portfolio_values.iloc[0] if len(result.portfolio_values) > 0 else 0,
                'end_value': result.portfolio_values.iloc[-1] if len(result.portfolio_values) > 0 else 0,
                'peak_value': result.portfolio_values.max() if len(result.portfolio_values) > 0 else 0,
                'periods': len(result.portfolio_values)
            }
        }
    
    def create_visualization_from_predictions(
        self,
        predictions: np.ndarray,
        price_data: pd.DataFrame,
        test_start_idx: int,
        symbol: str = 'ASSET',
        show_plot: bool = True
    ) -> Dict[str, Any]:
        """
        Create VectorBT portfolio visualization from ML predictions.
        
        This method implements the exact pattern from your example:
        - Aligns predictions to full historical timeline
        - Creates VectorBT portfolio with realistic parameters
        - Generates interactive visualization with port.plot().show()
        
        Args:
            predictions: ML model predictions (0=sell, 1=hold, 2=buy)
            price_data: Historical price data with 'Close' column
            test_start_idx: Index where test period begins
            symbol: Asset symbol for labeling
            show_plot: Whether to display the plot immediately
            
        Returns:
            Dictionary with portfolio, visualization result, and metrics
        """
        try:
            # Create portfolio configuration matching your example
            portfolio_config = PortfolioConfig(
                init_cash=10000,           # init_cash=10000
                size_strategy='fixed_amount',
                size_value=40,             # size=np.full(df.shape[0], 40)
                fees=0.0025,              # fees=0.0025
                slippage=0.0025,          # slippage=0.0025
                stop_loss=0.1,            # sl_stop=0.1 (10% stop loss)
                upon_opposite_entry='ignore',  # upon_opposite_entry='ignore'
                freq='D'                  # freq='D'
            )
            
            # Create visualization engine
            viz_engine = VectorBTVisualizationEngine(
                portfolio_config=portfolio_config,
                plot_config=PlotConfig(width=1200, height=600)
            )
            
            # Create portfolio from predictions
            self.logger.info(f"Creating VectorBT portfolio visualization for {symbol}")
            portfolio = viz_engine.create_portfolio_from_predictions(
                predictions=predictions,
                price_data=price_data,
                test_start_idx=test_start_idx,
                symbol=symbol
            )
            
            # Generate portfolio plot
            viz_result = viz_engine.generate_portfolio_plot(
                portfolio, 
                title=f"{symbol} Portfolio Performance - VectorBT Visualization"
            )
            
            # Show plot if requested (equivalent to port.plot().show())
            if show_plot and viz_result.success:
                viz_engine.show_plot(viz_result)
            
            # Create comprehensive result
            result = {
                'portfolio': portfolio,
                'visualization_result': viz_result,
                'success': viz_result.success,
                'error_message': viz_result.error_message if not viz_result.success else None,
                'metrics': viz_result.metrics_summary,
                'generation_time': viz_result.generation_time,
                'symbol': symbol,
                'test_period_start': test_start_idx,
                'prediction_count': len(predictions)
            }
            
            self.logger.info(
                f"VectorBT visualization created successfully for {symbol} "
                f"({len(predictions)} predictions, {viz_result.generation_time:.2f}s)"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating VectorBT visualization: {str(e)}")
            return {
                'portfolio': None,
                'visualization_result': None,
                'success': False,
                'error_message': str(e),
                'metrics': {},
                'generation_time': 0.0,
                'symbol': symbol,
                'test_period_start': test_start_idx,
                'prediction_count': len(predictions)
            }
    
    def create_enhanced_backtest_with_visualization(
        self,
        signals: pd.Series,
        prices: pd.Series,
        predictions: Optional[np.ndarray] = None,
        test_start_idx: Optional[int] = None,
        symbol: str = 'ASSET',
        show_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Create enhanced backtest with comprehensive visualizations.
        
        Args:
            signals: Trading signals
            prices: Price series
            predictions: Optional ML predictions for enhanced visualization
            test_start_idx: Test period start index (required if predictions provided)
            symbol: Asset symbol
            show_plots: Whether to display plots
            
        Returns:
            Dictionary with backtest results and visualizations
        """
        try:
            # Run standard backtesting
            backtest_result = self.simulate_trading(signals, prices)
            
            # Create visualization engine
            viz_engine = VectorBTVisualizationEngine()
            
            results = {
                'backtest_result': backtest_result,
                'symbol': symbol,
                'success': True
            }
            
            # Generate portfolio visualization if we have the portfolio object
            if hasattr(backtest_result, 'portfolio') and backtest_result.portfolio is not None:
                portfolio_viz = viz_engine.generate_portfolio_plot(
                    backtest_result.portfolio,
                    title=f"{symbol} Portfolio Performance"
                )
                results['portfolio_visualization'] = portfolio_viz
                
                if show_plots and portfolio_viz.success:
                    viz_engine.show_plot(portfolio_viz)
                
                # Generate drawdown analysis
                drawdown_viz = viz_engine.generate_drawdown_plot(backtest_result.portfolio)
                results['drawdown_visualization'] = drawdown_viz
                
                if show_plots and drawdown_viz.success:
                    viz_engine.show_plot(drawdown_viz)
                
                # Generate trade analysis
                trade_viz = viz_engine.generate_trade_analysis_plot(backtest_result.portfolio)
                results['trade_visualization'] = trade_viz
                
                if show_plots and trade_viz.success:
                    viz_engine.show_plot(trade_viz)
            
            # If predictions are provided, create enhanced visualization
            if predictions is not None and test_start_idx is not None:
                price_df = pd.DataFrame({'Close': prices})
                enhanced_viz = self.create_visualization_from_predictions(
                    predictions, price_df, test_start_idx, symbol, show_plots
                )
                results['enhanced_visualization'] = enhanced_viz
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in enhanced backtest with visualization: {str(e)}")
            return {
                'backtest_result': None,
                'success': False,
                'error_message': str(e),
                'symbol': symbol
            }