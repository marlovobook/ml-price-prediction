"""
Backtesting engine implementation for stock direction predictor.
Provides comprehensive trading simulation with portfolio tracking and risk management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

from ..interfaces import IBacktestingEngine, BacktestResult
from ..utils.exceptions import DataValidationError, BacktestingError


class BacktestingEngine(IBacktestingEngine):
    """
    Backtesting engine that simulates trading strategies based on signals.
    
    Features:
    - Portfolio value tracking
    - Transaction cost and slippage modeling
    - Detailed trade logging
    - Drawdown analysis
    - Risk management rules
    """
    
    def __init__(self, 
                 transaction_cost: float = 0.001,  # 0.1% transaction cost
                 slippage: float = 0.0005,         # 0.05% slippage
                 max_position_size: float = 0.1,   # 10% max position size
                 stop_loss: Optional[float] = None, # Stop loss percentage
                 take_profit: Optional[float] = None): # Take profit percentage
        """
        Initialize backtesting engine with configuration parameters.
        
        Args:
            transaction_cost: Transaction cost as percentage of trade value
            slippage: Slippage as percentage of trade value
            max_position_size: Maximum position size as percentage of portfolio
            stop_loss: Stop loss percentage (optional)
            take_profit: Take profit percentage (optional)
        """
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.logger = logging.getLogger(__name__)
        
    def simulate_trading(self, signals: pd.Series, prices: pd.Series, initial_capital: float) -> BacktestResult:
        """
        Simulate trading based on signals and return comprehensive results.
        
        Args:
            signals: Trading signals (-1: sell, 0: hold, 1: buy)
            prices: Price series corresponding to signals
            initial_capital: Initial portfolio capital
            
        Returns:
            BacktestResult with comprehensive trading simulation results
        """
        if len(signals) != len(prices):
            raise DataValidationError("Signals and prices must have the same length")
        
        if initial_capital <= 0:
            raise DataValidationError("Initial capital must be positive")
        
        if len(signals) == 0:
            raise DataValidationError("Signals cannot be empty")
        
        # Align signals and prices by index
        aligned_data = pd.DataFrame({'signal': signals, 'price': prices}).dropna()
        
        if len(aligned_data) == 0:
            raise DataValidationError("No valid signal-price pairs found")
        
        signals = aligned_data['signal']
        prices = aligned_data['price']
        
        # Initialize portfolio tracking
        portfolio_values = []
        cash = initial_capital
        position = 0  # Number of shares held
        position_value = 0
        
        # Initialize trade tracking
        trades = []
        current_trade = None
        
        # Process each signal
        for i, (timestamp, signal) in enumerate(signals.items()):
            current_price = prices.iloc[i]
            
            # Calculate current portfolio value
            current_portfolio_value = cash + (position * current_price)
            portfolio_values.append(current_portfolio_value)
            
            # Process signal
            if signal == 1 and position <= 0:  # Buy signal
                trade_result = self._execute_buy(
                    timestamp, current_price, cash, current_portfolio_value, current_trade
                )
                cash, position, current_trade = trade_result
                
            elif signal == -1 and position >= 0:  # Sell signal
                trade_result = self._execute_sell(
                    timestamp, current_price, position, current_trade, trades, cash
                )
                cash, position, current_trade = trade_result
                
            # Check stop loss and take profit
            if current_trade and position != 0:
                stop_loss_triggered, take_profit_triggered = self._check_exit_conditions(
                    current_price, current_trade['entry_price'], position
                )
                
                if stop_loss_triggered or take_profit_triggered:
                    exit_reason = "stop_loss" if stop_loss_triggered else "take_profit"
                    trade_result = self._execute_exit(
                        timestamp, current_price, position, current_trade, trades, exit_reason
                    )
                    cash, position, current_trade = trade_result
        
        # Close any remaining position at the end
        if position != 0 and current_trade:
            final_price = prices.iloc[-1]
            final_timestamp = signals.index[-1]
            cash, position, _ = self._execute_exit(
                final_timestamp, final_price, position, current_trade, trades, "end_of_period"
            )
        
        # Create portfolio values series
        portfolio_series = pd.Series(portfolio_values, index=signals.index)
        
        # Generate trade log
        trade_log = self.generate_trade_log(signals, prices)
        
        # Calculate portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(portfolio_series)
        
        return BacktestResult(
            total_return=portfolio_metrics['total_return'],
            max_drawdown=portfolio_metrics['max_drawdown'],
            sharpe_ratio=portfolio_metrics['sharpe_ratio'],
            win_rate=portfolio_metrics['win_rate'],
            profit_factor=portfolio_metrics['profit_factor'],
            trade_log=trade_log,
            portfolio_values=portfolio_series
        )
    
    def _execute_buy(self, timestamp: datetime, price: float, cash: float, 
                     portfolio_value: float, current_trade: Optional[Dict]) -> Tuple[float, float, Dict]:
        """Execute a buy order with transaction costs and position sizing."""
        # Calculate position size (limited by max_position_size and available cash)
        max_investment = portfolio_value * self.max_position_size
        available_investment = min(cash * 0.95, max_investment)  # Keep 5% cash buffer
        
        if available_investment <= 0:
            return cash, 0, current_trade
        
        # Apply slippage to price
        execution_price = price * (1 + self.slippage)
        
        # Calculate shares to buy
        shares_to_buy = available_investment / execution_price
        
        # Apply transaction costs
        transaction_cost_amount = available_investment * self.transaction_cost
        total_cost = (shares_to_buy * execution_price) + transaction_cost_amount
        
        if total_cost > cash:
            return cash, 0, current_trade
        
        # Update cash and position
        new_cash = cash - total_cost
        new_position = shares_to_buy
        
        # Create new trade record
        new_trade = {
            'entry_timestamp': timestamp,
            'entry_price': execution_price,
            'shares': shares_to_buy,
            'entry_value': shares_to_buy * execution_price,
            'transaction_cost': transaction_cost_amount,
            'signal_type': 'buy'
        }
        
        return new_cash, new_position, new_trade
    
    def _execute_sell(self, timestamp: datetime, price: float, position: float,
                      current_trade: Optional[Dict], trades: List[Dict], current_cash: float) -> Tuple[float, float, Optional[Dict]]:
        """Execute a sell order with transaction costs."""
        if position <= 0 or not current_trade:
            return current_cash, 0, None
        
        # Apply slippage to price
        execution_price = price * (1 - self.slippage)
        
        # Calculate proceeds from sale
        gross_proceeds = position * execution_price
        transaction_cost_amount = gross_proceeds * self.transaction_cost
        net_proceeds = gross_proceeds - transaction_cost_amount
        
        # Calculate P&L
        entry_cost = current_trade['entry_value'] + current_trade['transaction_cost']
        pnl = net_proceeds - entry_cost
        
        # Record completed trade
        completed_trade = {
            'entry_timestamp': current_trade['entry_timestamp'],
            'exit_timestamp': timestamp,
            'entry_price': current_trade['entry_price'],
            'exit_price': execution_price,
            'shares': position,
            'entry_value': current_trade['entry_value'],
            'exit_value': gross_proceeds,
            'entry_transaction_cost': current_trade['transaction_cost'],
            'exit_transaction_cost': transaction_cost_amount,
            'pnl': pnl,
            'return_pct': pnl / entry_cost if entry_cost > 0 else 0,
            'signal_type': 'sell',
            'exit_reason': 'signal'
        }
        
        trades.append(completed_trade)
        
        return net_proceeds, 0, None
    
    def _execute_exit(self, timestamp: datetime, price: float, position: float,
                      current_trade: Dict, trades: List[Dict], exit_reason: str) -> Tuple[float, float, Optional[Dict]]:
        """Execute an exit order (stop loss, take profit, or end of period)."""
        if position == 0 or not current_trade:
            return 0, 0, None
        
        # Apply slippage based on position direction
        if position > 0:  # Long position - selling
            execution_price = price * (1 - self.slippage)
        else:  # Short position - buying to cover
            execution_price = price * (1 + self.slippage)
        
        # Calculate proceeds
        gross_proceeds = abs(position) * execution_price
        transaction_cost_amount = gross_proceeds * self.transaction_cost
        net_proceeds = gross_proceeds - transaction_cost_amount
        
        # For short positions, we need to handle differently
        if position < 0:
            net_proceeds = -net_proceeds  # Cost to cover short position
        
        # Calculate P&L
        entry_cost = current_trade['entry_value'] + current_trade['transaction_cost']
        if position > 0:
            pnl = net_proceeds - entry_cost
        else:
            pnl = entry_cost + net_proceeds  # For short positions
        
        # Record completed trade
        completed_trade = {
            'entry_timestamp': current_trade['entry_timestamp'],
            'exit_timestamp': timestamp,
            'entry_price': current_trade['entry_price'],
            'exit_price': execution_price,
            'shares': abs(position),
            'entry_value': current_trade['entry_value'],
            'exit_value': gross_proceeds,
            'entry_transaction_cost': current_trade['transaction_cost'],
            'exit_transaction_cost': transaction_cost_amount,
            'pnl': pnl,
            'return_pct': pnl / abs(entry_cost) if entry_cost != 0 else 0,
            'signal_type': current_trade['signal_type'],
            'exit_reason': exit_reason
        }
        
        trades.append(completed_trade)
        
        return net_proceeds if position > 0 else -net_proceeds, 0, None
    
    def _check_exit_conditions(self, current_price: float, entry_price: float, 
                               position: float) -> Tuple[bool, bool]:
        """Check if stop loss or take profit conditions are met."""
        if position == 0:
            return False, False
        
        stop_loss_triggered = False
        take_profit_triggered = False
        
        if position > 0:  # Long position
            if self.stop_loss:
                price_change = (current_price - entry_price) / entry_price
                stop_loss_triggered = price_change <= -self.stop_loss
            
            if self.take_profit:
                price_change = (current_price - entry_price) / entry_price
                take_profit_triggered = price_change >= self.take_profit
        
        else:  # Short position
            if self.stop_loss:
                price_change = (entry_price - current_price) / entry_price
                stop_loss_triggered = price_change <= -self.stop_loss
            
            if self.take_profit:
                price_change = (entry_price - current_price) / entry_price
                take_profit_triggered = price_change >= self.take_profit
        
        return stop_loss_triggered, take_profit_triggered
    
    def calculate_portfolio_metrics(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio performance metrics.
        
        Args:
            portfolio_values: Series of portfolio values over time
            
        Returns:
            Dictionary containing performance metrics
        """
        if len(portfolio_values) == 0:
            raise DataValidationError("Portfolio values cannot be empty")
        
        # Calculate returns
        returns = portfolio_values.pct_change().dropna()
        
        # Total return
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        
        # Maximum drawdown
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Volatility (annualized)
        if len(returns) > 1:
            volatility = returns.std() * np.sqrt(252)  # Assuming daily data
        else:
            volatility = 0.0
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        if volatility > 0:
            excess_return = (returns.mean() * 252) - risk_free_rate
            sharpe_ratio = excess_return / volatility
        else:
            sharpe_ratio = 0.0
        
        # Win rate and profit factor (requires trade log)
        win_rate = 0.0
        profit_factor = 1.0
        
        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(portfolio_values)) - 1,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0,
            'sortino_ratio': self._calculate_sortino_ratio(returns, risk_free_rate)
        }
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sortino ratio (downside deviation version of Sharpe ratio)."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_deviation = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)
        
        if downside_deviation == 0:
            return 0.0
        
        return (excess_returns.mean() * 252) / downside_deviation
    
    def generate_trade_log(self, signals: pd.Series, prices: pd.Series) -> pd.DataFrame:
        """
        Generate detailed trade log from signals and prices.
        
        Args:
            signals: Trading signals series
            prices: Price series
            
        Returns:
            DataFrame with detailed trade information
        """
        if len(signals) != len(prices):
            raise DataValidationError("Signals and prices must have the same length")
        
        trades = []
        position = 0
        entry_info = None
        
        # Align data
        aligned_data = pd.DataFrame({'signal': signals, 'price': prices}).dropna()
        
        for i, (timestamp, row) in enumerate(aligned_data.iterrows()):
            signal = row['signal']
            price = row['price']
            
            if signal == 1 and position <= 0:  # Buy signal
                if entry_info:  # Close previous short position
                    pnl = entry_info['entry_price'] - price  # Profit from short
                    trades.append({
                        'trade_id': len(trades) + 1,
                        'entry_timestamp': entry_info['timestamp'],
                        'exit_timestamp': timestamp,
                        'signal': entry_info['signal'],
                        'entry_price': entry_info['entry_price'],
                        'exit_price': price,
                        'position_type': 'short',
                        'pnl': pnl,
                        'return_pct': pnl / entry_info['entry_price'] if entry_info['entry_price'] > 0 else 0
                    })
                
                # Open new long position
                entry_info = {'timestamp': timestamp, 'entry_price': price, 'signal': signal}
                position = 1
                
            elif signal == -1 and position >= 0:  # Sell signal
                if entry_info:  # Close previous long position
                    pnl = price - entry_info['entry_price']  # Profit from long
                    trades.append({
                        'trade_id': len(trades) + 1,
                        'entry_timestamp': entry_info['timestamp'],
                        'exit_timestamp': timestamp,
                        'signal': entry_info['signal'],
                        'entry_price': entry_info['entry_price'],
                        'exit_price': price,
                        'position_type': 'long',
                        'pnl': pnl,
                        'return_pct': pnl / entry_info['entry_price'] if entry_info['entry_price'] > 0 else 0
                    })
                
                # Open new short position
                entry_info = {'timestamp': timestamp, 'entry_price': price, 'signal': signal}
                position = -1
        
        # Close final position if exists
        if entry_info and len(aligned_data) > 0:
            final_price = aligned_data['price'].iloc[-1]
            final_timestamp = aligned_data.index[-1]
            
            if position > 0:  # Close long position
                pnl = final_price - entry_info['entry_price']
                position_type = 'long'
            else:  # Close short position
                pnl = entry_info['entry_price'] - final_price
                position_type = 'short'
            
            trades.append({
                'trade_id': len(trades) + 1,
                'entry_timestamp': entry_info['timestamp'],
                'exit_timestamp': final_timestamp,
                'signal': entry_info['signal'],
                'entry_price': entry_info['entry_price'],
                'exit_price': final_price,
                'position_type': position_type,
                'pnl': pnl,
                'return_pct': pnl / entry_info['entry_price'] if entry_info['entry_price'] > 0 else 0
            })
        
        if not trades:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=[
                'trade_id', 'entry_timestamp', 'exit_timestamp', 'signal',
                'entry_price', 'exit_price', 'position_type', 'pnl', 'return_pct'
            ])
        
        return pd.DataFrame(trades)
    
    def identify_drawdown_periods(self, portfolio_values: pd.Series) -> List[Dict]:
        """
        Identify periods of maximum drawdown and recovery.
        
        Args:
            portfolio_values: Series of portfolio values
            
        Returns:
            List of drawdown period dictionaries
        """
        if len(portfolio_values) == 0:
            return []
        
        # Calculate running maximum and drawdown
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        current_period = None
        
        for i, (timestamp, dd_value) in enumerate(drawdown.items()):
            if dd_value < -0.001 and not in_drawdown:  # Start of drawdown (0.1% threshold)
                in_drawdown = True
                current_period = {
                    'start_date': timestamp,
                    'start_value': portfolio_values.iloc[i],
                    'peak_value': running_max.iloc[i],
                    'max_drawdown': dd_value,
                    'max_drawdown_date': timestamp
                }
            
            elif in_drawdown:
                # Update maximum drawdown in current period
                if dd_value < current_period['max_drawdown']:
                    current_period['max_drawdown'] = dd_value
                    current_period['max_drawdown_date'] = timestamp
                
                # Check for recovery (back to previous high)
                if abs(dd_value) < 0.001:  # Recovered
                    current_period['end_date'] = timestamp
                    current_period['end_value'] = portfolio_values.iloc[i]
                    current_period['recovery_days'] = (timestamp - current_period['start_date']).days
                    current_period['drawdown_duration'] = (current_period['max_drawdown_date'] - current_period['start_date']).days
                    
                    drawdown_periods.append(current_period)
                    in_drawdown = False
                    current_period = None
        
        # Handle ongoing drawdown at end of period
        if in_drawdown and current_period:
            current_period['end_date'] = portfolio_values.index[-1]
            current_period['end_value'] = portfolio_values.iloc[-1]
            current_period['recovery_days'] = None  # Still in drawdown
            current_period['drawdown_duration'] = (current_period['max_drawdown_date'] - current_period['start_date']).days
            drawdown_periods.append(current_period)
        
        return drawdown_periods