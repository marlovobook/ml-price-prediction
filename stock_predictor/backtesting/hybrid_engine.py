"""
Hybrid Backtesting Engine that falls back to simple backtesting when VectorBT fails.
Provides robust backtesting with enhanced metrics when possible.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from ..interfaces import IBacktestingEngine
from ..utils.exceptions import BacktestingError
from .backtesting_engine import BacktestingEngine

try:
    from .vectorbt_engine import VectorBTBacktestingEngine, VectorBTBacktestResult
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    VectorBTBacktestingEngine = None
    VectorBTBacktestResult = None


@dataclass
class HybridBacktestResult:
    """Hybrid backtest result that works with both engines."""
    
    # Basic metrics (always available)
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    
    # Enhanced metrics (when VectorBT is available)
    calmar_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    value_at_risk: Optional[float] = None
    conditional_var: Optional[float] = None
    
    # Portfolio data
    portfolio_values: Optional[pd.Series] = None
    returns: Optional[pd.Series] = None
    drawdowns: Optional[pd.Series] = None
    trade_log: Optional[pd.DataFrame] = None
    
    # Additional statistics
    avg_trade_duration: Optional[float] = None
    best_trade: Optional[float] = None
    worst_trade: Optional[float] = None
    
    # Risk metrics
    beta: Optional[float] = None
    alpha: Optional[float] = None
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None
    
    # Engine used
    engine_used: str = "simple"


class HybridBacktestingEngine(IBacktestingEngine):
    """
    Hybrid backtesting engine that tries VectorBT first, falls back to simple engine.
    
    This provides the best of both worlds:
    - Advanced metrics when VectorBT works
    - Reliable fallback when VectorBT has issues
    - Consistent interface regardless of engine used
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 max_position_size: float = 1.0,
                 risk_free_rate: float = 0.02):
        """
        Initialize the hybrid backtesting engine.
        
        Args:
            initial_capital: Starting portfolio value
            transaction_cost: Transaction cost as percentage of trade value
            slippage: Slippage as percentage of trade value
            max_position_size: Maximum position size as fraction of portfolio
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize engines
        self.simple_engine = BacktestingEngine(
            transaction_cost=transaction_cost,
            slippage=slippage,
            max_position_size=max_position_size
        )
        
        if VECTORBT_AVAILABLE:
            try:
                self.vectorbt_engine = VectorBTBacktestingEngine(
                    initial_capital=initial_capital,
                    transaction_cost=transaction_cost,
                    slippage=slippage,
                    max_position_size=max_position_size,
                    risk_free_rate=risk_free_rate
                )
                self.logger.info("VectorBT engine initialized successfully")
            except Exception as e:
                self.logger.warning(f"VectorBT engine initialization failed: {str(e)}")
                self.vectorbt_engine = None
        else:
            self.vectorbt_engine = None
            self.logger.info("VectorBT not available, using simple engine only")
    
    def simulate_trading(self, signals: pd.Series, prices: pd.Series, 
                        initial_capital: Optional[float] = None) -> HybridBacktestResult:
        """
        Simulate trading using the best available engine.
        
        Args:
            signals: Trading signals (-1: sell, 0: hold, 1: buy)
            prices: Price series for the asset
            initial_capital: Override initial capital (optional)
            
        Returns:
            HybridBacktestResult with comprehensive metrics
        """
        capital = initial_capital or self.initial_capital
        
        # Try VectorBT first if available
        if self.vectorbt_engine is not None:
            try:
                self.logger.debug("Attempting VectorBT backtesting...")
                vectorbt_result = self.vectorbt_engine.simulate_trading(signals, prices, capital)
                
                # Convert to hybrid result
                result = HybridBacktestResult(
                    total_return=vectorbt_result.total_return,
                    annualized_return=vectorbt_result.annualized_return,
                    volatility=vectorbt_result.volatility,
                    sharpe_ratio=vectorbt_result.sharpe_ratio,
                    max_drawdown=vectorbt_result.max_drawdown,
                    win_rate=vectorbt_result.win_rate,
                    profit_factor=vectorbt_result.profit_factor,
                    num_trades=vectorbt_result.num_trades,
                    calmar_ratio=vectorbt_result.calmar_ratio,
                    sortino_ratio=vectorbt_result.sortino_ratio,
                    value_at_risk=vectorbt_result.value_at_risk,
                    conditional_var=vectorbt_result.conditional_var,
                    portfolio_values=vectorbt_result.portfolio_values,
                    returns=vectorbt_result.returns,
                    drawdowns=vectorbt_result.drawdowns,
                    trade_log=vectorbt_result.trade_log,
                    avg_trade_duration=vectorbt_result.avg_trade_duration,
                    best_trade=vectorbt_result.best_trade,
                    worst_trade=vectorbt_result.worst_trade,
                    beta=vectorbt_result.beta,
                    alpha=vectorbt_result.alpha,
                    information_ratio=vectorbt_result.information_ratio,
                    tracking_error=vectorbt_result.tracking_error,
                    engine_used="vectorbt"
                )
                
                self.logger.info("VectorBT backtesting completed successfully")
                return result
                
            except Exception as e:
                self.logger.warning(f"VectorBT backtesting failed: {str(e)}, falling back to simple engine")
        
        # Fall back to simple engine
        try:
            self.logger.debug("Using simple backtesting engine...")
            simple_result = self.simple_engine.simulate_trading(signals, prices, capital)
            
            # Convert to hybrid result
            result = HybridBacktestResult(
                total_return=simple_result.total_return,
                annualized_return=self._calculate_annualized_return(simple_result.total_return, len(prices)),
                volatility=self._calculate_volatility(simple_result.portfolio_values),
                sharpe_ratio=simple_result.sharpe_ratio,
                max_drawdown=simple_result.max_drawdown,
                win_rate=simple_result.win_rate,
                profit_factor=simple_result.profit_factor,
                num_trades=len(simple_result.trade_log),
                portfolio_values=simple_result.portfolio_values,
                trade_log=simple_result.trade_log,
                engine_used="simple"
            )
            
            # Calculate additional metrics if possible
            if len(simple_result.portfolio_values) > 1:
                returns = simple_result.portfolio_values.pct_change().dropna()
                result.returns = returns
                
                # Calculate enhanced metrics
                result.calmar_ratio = self._calculate_calmar_ratio(result.annualized_return, result.max_drawdown)
                result.sortino_ratio = self._calculate_sortino_ratio(returns, self.risk_free_rate)
                result.value_at_risk = self._calculate_var(returns)
                result.conditional_var = self._calculate_cvar(returns)
            
            self.logger.info("Simple backtesting completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Both backtesting engines failed: {str(e)}")
            raise BacktestingError(f"All backtesting engines failed: {str(e)}")
    
    def _calculate_annualized_return(self, total_return: float, periods: int) -> float:
        """Calculate annualized return from total return."""
        if periods <= 0:
            return 0.0
        
        years = periods / 252  # Assuming daily data
        if years <= 0:
            return 0.0
        
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_volatility(self, portfolio_values: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(portfolio_values) < 2:
            return 0.0
        
        returns = portfolio_values.pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        
        return returns.std() * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return 0.0
        return annualized_return / abs(max_drawdown)
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        downside_deviation = downside_returns.std() * np.sqrt(252)
        if downside_deviation == 0:
            return 0.0
        
        return (excess_returns.mean() * 252) / downside_deviation
    
    def _calculate_var(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, confidence * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Conditional Value at Risk."""
        if len(returns) == 0:
            return 0.0
        
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def calculate_portfolio_metrics(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """Calculate portfolio metrics from portfolio values series."""
        return self.simple_engine.calculate_portfolio_metrics(portfolio_values)
    
    def generate_trade_log(self, signals: pd.Series, prices: pd.Series) -> pd.DataFrame:
        """Generate trade log from signals and prices."""
        return self.simple_engine.generate_trade_log(signals, prices)
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get status of available engines."""
        return {
            'vectorbt_available': VECTORBT_AVAILABLE,
            'vectorbt_engine_initialized': self.vectorbt_engine is not None,
            'simple_engine_available': True,
            'preferred_engine': 'vectorbt' if self.vectorbt_engine is not None else 'simple'
        }