"""
Dashboard Integration Module for VectorBT Visualization Enhancement.

This module provides Streamlit-compatible plot objects and real-time visualization
updates for dashboard integration, implementing Requirements 10.2.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
from datetime import datetime, timedelta
import threading
from pathlib import Path
import hashlib
import json

from .visualization_engine import VectorBTVisualizationEngine, VisualizationResult
from .portfolio_config import PortfolioConfig, PlotConfig
from ..utils.exceptions import BacktestingError


class StreamlitVisualizationAdapter:
    """
    Adapter class for integrating VectorBT visualizations with Streamlit dashboard.
    
    This class implements Requirements 10.2:
    - Create plot objects compatible with Streamlit dashboard
    - Add real-time visualization updates for live trading
    - Implement caching for improved dashboard performance
    """
    
    def __init__(self, 
                 visualization_engine: Optional[VectorBTVisualizationEngine] = None,
                 enable_caching: bool = True,
                 cache_ttl: int = 300):  # 5 minutes default TTL
        """
        Initialize the Streamlit Visualization Adapter.
        
        Args:
            visualization_engine: VectorBT visualization engine instance
            enable_caching: Whether to enable plot caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.viz_engine = visualization_engine or VectorBTVisualizationEngine()
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize Streamlit session state for caching
        self._initialize_session_state()
        
        # Real-time update configuration
        self.real_time_enabled = False
        self.update_interval = 60  # seconds
        self.last_update = None
        
        self.logger.info("Streamlit Visualization Adapter initialized")
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables for caching."""
        if 'viz_cache' not in st.session_state:
            st.session_state.viz_cache = {}
        
        if 'viz_cache_timestamps' not in st.session_state:
            st.session_state.viz_cache_timestamps = {}
        
        if 'real_time_data' not in st.session_state:
            st.session_state.real_time_data = {}
        
        if 'last_real_time_update' not in st.session_state:
            st.session_state.last_real_time_update = None
    
    def create_streamlit_portfolio_plot(
        self,
        portfolio,
        title: Optional[str] = None,
        use_cache: bool = True
    ) -> go.Figure:
        """
        Create Streamlit-compatible portfolio plot with caching.
        
        This method implements Requirements 10.2 by creating plot objects
        compatible with Streamlit dashboard and implementing caching for
        improved performance.
        
        Args:
            portfolio: VectorBT portfolio object
            title: Optional plot title
            use_cache: Whether to use caching
            
        Returns:
            Plotly figure object ready for Streamlit display
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key('portfolio_plot', portfolio, title)
            
            # Check cache first if enabled
            if use_cache and self.enable_caching:
                cached_plot = self._get_cached_plot(cache_key)
                if cached_plot is not None:
                    self.logger.info("Returning cached portfolio plot")
                    return cached_plot
            
            # Generate new plot
            self.logger.info("Generating new Streamlit-compatible portfolio plot")
            result = self.viz_engine.generate_portfolio_plot(portfolio, title)
            
            if not result.success:
                st.error(f"Failed to generate portfolio plot: {result.error_message}")
                return self._create_error_plot("Portfolio plot generation failed")
            
            # Optimize plot for Streamlit display
            streamlit_plot = self._optimize_plot_for_streamlit(result.plot_object)
            
            # Cache the result
            if use_cache and self.enable_caching:
                self._cache_plot(cache_key, streamlit_plot)
            
            return streamlit_plot
            
        except Exception as e:
            self.logger.error(f"Error creating Streamlit portfolio plot: {str(e)}")
            st.error(f"Error generating portfolio plot: {str(e)}")
            return self._create_error_plot(f"Error: {str(e)}")
    
    def create_streamlit_drawdown_plot(
        self,
        portfolio,
        use_cache: bool = True
    ) -> go.Figure:
        """
        Create Streamlit-compatible drawdown plot with caching.
        
        Args:
            portfolio: VectorBT portfolio object
            use_cache: Whether to use caching
            
        Returns:
            Plotly figure object ready for Streamlit display
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key('drawdown_plot', portfolio)
            
            # Check cache first if enabled
            if use_cache and self.enable_caching:
                cached_plot = self._get_cached_plot(cache_key)
                if cached_plot is not None:
                    self.logger.info("Returning cached drawdown plot")
                    return cached_plot
            
            # Generate new plot
            self.logger.info("Generating new Streamlit-compatible drawdown plot")
            result = self.viz_engine.generate_drawdown_plot(portfolio)
            
            if not result.success:
                st.error(f"Failed to generate drawdown plot: {result.error_message}")
                return self._create_error_plot("Drawdown plot generation failed")
            
            # Optimize plot for Streamlit display
            streamlit_plot = self._optimize_plot_for_streamlit(result.plot_object)
            
            # Cache the result
            if use_cache and self.enable_caching:
                self._cache_plot(cache_key, streamlit_plot)
            
            return streamlit_plot
            
        except Exception as e:
            self.logger.error(f"Error creating Streamlit drawdown plot: {str(e)}")
            st.error(f"Error generating drawdown plot: {str(e)}")
            return self._create_error_plot(f"Error: {str(e)}")
    
    def create_streamlit_comparison_plot(
        self,
        portfolios: Dict[str, Any],
        title: str = "Strategy Comparison",
        use_cache: bool = True
    ) -> go.Figure:
        """
        Create Streamlit-compatible comparison plot with caching.
        
        Args:
            portfolios: Dictionary of named portfolio objects
            title: Plot title
            use_cache: Whether to use caching
            
        Returns:
            Plotly figure object ready for Streamlit display
        """
        try:
            # Generate cache key
            cache_key = self._generate_cache_key('comparison_plot', portfolios, title)
            
            # Check cache first if enabled
            if use_cache and self.enable_caching:
                cached_plot = self._get_cached_plot(cache_key)
                if cached_plot is not None:
                    self.logger.info("Returning cached comparison plot")
                    return cached_plot
            
            # Generate new plot
            self.logger.info("Generating new Streamlit-compatible comparison plot")
            result = self.viz_engine.generate_comparison_plot(portfolios, title)
            
            if not result.success:
                st.error(f"Failed to generate comparison plot: {result.error_message}")
                return self._create_error_plot("Comparison plot generation failed")
            
            # Optimize plot for Streamlit display
            streamlit_plot = self._optimize_plot_for_streamlit(result.plot_object)
            
            # Cache the result
            if use_cache and self.enable_caching:
                self._cache_plot(cache_key, streamlit_plot)
            
            return streamlit_plot
            
        except Exception as e:
            self.logger.error(f"Error creating Streamlit comparison plot: {str(e)}")
            st.error(f"Error generating comparison plot: {str(e)}")
            return self._create_error_plot(f"Error: {str(e)}")
    
    def display_portfolio_visualization(
        self,
        portfolio,
        title: Optional[str] = None,
        show_metrics: bool = True,
        show_trades: bool = True
    ) -> None:
        """
        Display comprehensive portfolio visualization in Streamlit.
        
        This method creates a complete portfolio analysis display including
        performance plot, metrics, and trade analysis.
        
        Args:
            portfolio: VectorBT portfolio object
            title: Optional plot title
            show_metrics: Whether to display performance metrics
            show_trades: Whether to display trade analysis
        """
        try:
            # Create main portfolio plot
            portfolio_plot = self.create_streamlit_portfolio_plot(portfolio, title)
            
            # Display the plot
            st.plotly_chart(portfolio_plot, use_container_width=True)
            
            # Display metrics if requested
            if show_metrics:
                self._display_portfolio_metrics(portfolio)
            
            # Display trade analysis if requested
            if show_trades and portfolio.trades.count() > 0:
                self._display_trade_analysis(portfolio)
            
        except Exception as e:
            self.logger.error(f"Error displaying portfolio visualization: {str(e)}")
            st.error(f"Error displaying portfolio visualization: {str(e)}")
    
    def display_real_time_visualization(
        self,
        data_source_func,
        update_interval: int = 60,
        auto_refresh: bool = True
    ) -> None:
        """
        Display real-time visualization with automatic updates.
        
        This method implements Requirements 10.2 for real-time visualization
        updates for live trading scenarios.
        
        Args:
            data_source_func: Function that returns updated portfolio data
            update_interval: Update interval in seconds
            auto_refresh: Whether to enable automatic refresh
        """
        try:
            # Initialize real-time container
            real_time_container = st.container()
            
            with real_time_container:
                # Display current time and update status
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.subheader("📊 Real-time Portfolio Visualization")
                
                with col2:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    st.metric("Current Time", current_time)
                
                with col3:
                    if auto_refresh:
                        st.success("🔄 Auto-refresh ON")
                    else:
                        st.info("⏸️ Auto-refresh OFF")
                
                # Manual refresh button
                if st.button("🔄 Refresh Now"):
                    self._force_real_time_update(data_source_func)
                
                # Check if update is needed
                if self._should_update_real_time(update_interval):
                    self._update_real_time_data(data_source_func)
                
                # Display current real-time data
                self._display_real_time_data()
                
                # Auto-refresh mechanism
                if auto_refresh:
                    time.sleep(1)  # Small delay to prevent excessive updates
                    st.rerun()
            
        except Exception as e:
            self.logger.error(f"Error in real-time visualization: {str(e)}")
            st.error(f"Real-time visualization error: {str(e)}")
    
    def create_cached_visualization_dashboard(
        self,
        portfolios: Dict[str, Any],
        enable_real_time: bool = False
    ) -> None:
        """
        Create a comprehensive cached visualization dashboard.
        
        This method implements Requirements 10.2 by creating a complete
        dashboard with caching for improved performance.
        
        Args:
            portfolios: Dictionary of portfolio objects to visualize
            enable_real_time: Whether to enable real-time updates
        """
        try:
            # Dashboard header
            st.title("📈 VectorBT Portfolio Analysis Dashboard")
            
            # Performance metrics overview
            st.subheader("📊 Performance Overview")
            self._display_performance_overview(portfolios)
            
            # Individual portfolio analysis
            st.subheader("🔍 Individual Portfolio Analysis")
            
            # Portfolio selector
            selected_portfolio = st.selectbox(
                "Select portfolio for detailed analysis:",
                list(portfolios.keys())
            )
            
            if selected_portfolio and selected_portfolio in portfolios:
                portfolio = portfolios[selected_portfolio]
                
                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(["📈 Performance", "📉 Drawdown", "📊 Trades"])
                
                with tab1:
                    portfolio_plot = self.create_streamlit_portfolio_plot(
                        portfolio, 
                        title=f"{selected_portfolio} Portfolio Performance"
                    )
                    st.plotly_chart(portfolio_plot, use_container_width=True)
                
                with tab2:
                    drawdown_plot = self.create_streamlit_drawdown_plot(portfolio)
                    st.plotly_chart(drawdown_plot, use_container_width=True)
                
                with tab3:
                    if portfolio.trades.count() > 0:
                        self._display_trade_analysis(portfolio)
                    else:
                        st.info("No trades available for this portfolio")
            
            # Comparison analysis
            if len(portfolios) > 1:
                st.subheader("🔄 Portfolio Comparison")
                comparison_plot = self.create_streamlit_comparison_plot(
                    portfolios,
                    title="Multi-Portfolio Comparison"
                )
                st.plotly_chart(comparison_plot, use_container_width=True)
            
            # Real-time section
            if enable_real_time:
                st.subheader("⚡ Real-time Updates")
                st.info("Real-time functionality would connect to live data sources")
            
            # Cache management
            st.subheader("🗄️ Cache Management")
            self._display_cache_management()
            
        except Exception as e:
            self.logger.error(f"Error creating cached visualization dashboard: {str(e)}")
            st.error(f"Dashboard creation error: {str(e)}")
    
    def _optimize_plot_for_streamlit(self, plot_obj: go.Figure) -> go.Figure:
        """
        Optimize plot object for Streamlit display.
        
        Args:
            plot_obj: Original plotly figure
            
        Returns:
            Optimized plotly figure for Streamlit
        """
        try:
            # Update layout for better Streamlit integration
            plot_obj.update_layout(
                # Remove margins for better fit
                margin=dict(l=0, r=0, t=30, b=0),
                
                # Optimize for responsive design
                autosize=True,
                
                # Streamlit-friendly hover mode
                hovermode='x unified',
                
                # Better legend positioning
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                
                # Streamlit theme compatibility
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            # Optimize for mobile responsiveness
            plot_obj.update_xaxes(fixedrange=False)
            plot_obj.update_yaxes(fixedrange=False)
            
            return plot_obj
            
        except Exception as e:
            self.logger.warning(f"Error optimizing plot for Streamlit: {str(e)}")
            return plot_obj
    
    def _generate_cache_key(self, plot_type: str, *args) -> str:
        """
        Generate cache key for plot caching.
        
        Args:
            plot_type: Type of plot
            *args: Arguments to include in cache key
            
        Returns:
            Cache key string
        """
        try:
            # Create a hash of the arguments
            key_data = {
                'plot_type': plot_type,
                'timestamp': int(time.time() / self.cache_ttl),  # Bucket by TTL
            }
            
            # Add serializable data from arguments
            for i, arg in enumerate(args):
                if hasattr(arg, 'value'):  # VectorBT portfolio
                    # Use portfolio value hash as identifier
                    portfolio_hash = hashlib.md5(
                        str(arg.value().sum()).encode()
                    ).hexdigest()[:8]
                    key_data[f'arg_{i}'] = portfolio_hash
                elif isinstance(arg, (str, int, float)):
                    key_data[f'arg_{i}'] = arg
                elif isinstance(arg, dict):
                    key_data[f'arg_{i}'] = len(arg)  # Use dict size as proxy
            
            # Generate hash
            key_string = json.dumps(key_data, sort_keys=True)
            cache_key = hashlib.md5(key_string.encode()).hexdigest()
            
            return cache_key
            
        except Exception as e:
            self.logger.warning(f"Error generating cache key: {str(e)}")
            return f"{plot_type}_{int(time.time())}"
    
    def _get_cached_plot(self, cache_key: str) -> Optional[go.Figure]:
        """
        Retrieve cached plot if available and not expired.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Cached plot object or None if not available/expired
        """
        try:
            if cache_key in st.session_state.viz_cache:
                # Check if cache is still valid
                cache_time = st.session_state.viz_cache_timestamps.get(cache_key, 0)
                if time.time() - cache_time < self.cache_ttl:
                    return st.session_state.viz_cache[cache_key]
                else:
                    # Remove expired cache
                    del st.session_state.viz_cache[cache_key]
                    del st.session_state.viz_cache_timestamps[cache_key]
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error retrieving cached plot: {str(e)}")
            return None
    
    def _cache_plot(self, cache_key: str, plot_obj: go.Figure) -> None:
        """
        Cache plot object with timestamp.
        
        Args:
            cache_key: Cache key
            plot_obj: Plot object to cache
        """
        try:
            st.session_state.viz_cache[cache_key] = plot_obj
            st.session_state.viz_cache_timestamps[cache_key] = time.time()
            
            # Limit cache size (keep only 50 most recent)
            if len(st.session_state.viz_cache) > 50:
                # Remove oldest entries
                sorted_keys = sorted(
                    st.session_state.viz_cache_timestamps.items(),
                    key=lambda x: x[1]
                )
                
                for key, _ in sorted_keys[:10]:  # Remove 10 oldest
                    if key in st.session_state.viz_cache:
                        del st.session_state.viz_cache[key]
                    if key in st.session_state.viz_cache_timestamps:
                        del st.session_state.viz_cache_timestamps[key]
            
        except Exception as e:
            self.logger.warning(f"Error caching plot: {str(e)}")
    
    def _create_error_plot(self, error_message: str) -> go.Figure:
        """
        Create error plot for display when visualization fails.
        
        Args:
            error_message: Error message to display
            
        Returns:
            Error plot figure
        """
        fig = go.Figure()
        fig.add_annotation(
            text=f"❌ {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Visualization Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
    
    def _display_portfolio_metrics(self, portfolio) -> None:
        """
        Display portfolio performance metrics in Streamlit.
        
        Args:
            portfolio: VectorBT portfolio object
        """
        try:
            metrics = self.viz_engine._extract_portfolio_metrics(portfolio)
            
            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'total_return' in metrics:
                    st.metric(
                        "Total Return",
                        f"{metrics['total_return']:.2%}",
                        delta=None
                    )
            
            with col2:
                if 'sharpe_ratio' in metrics:
                    st.metric(
                        "Sharpe Ratio",
                        f"{metrics['sharpe_ratio']:.2f}",
                        delta=None
                    )
            
            with col3:
                if 'max_drawdown' in metrics:
                    st.metric(
                        "Max Drawdown",
                        f"{metrics['max_drawdown']:.2%}",
                        delta=None
                    )
            
            with col4:
                if 'num_trades' in metrics:
                    st.metric(
                        "Number of Trades",
                        f"{int(metrics['num_trades'])}",
                        delta=None
                    )
            
            # Additional metrics
            if len(metrics) > 4:
                with st.expander("📊 Additional Metrics"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'win_rate' in metrics:
                            st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
                    
                    with col2:
                        if 'profit_factor' in metrics:
                            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                    
                    with col3:
                        if 'volatility' in metrics:
                            st.metric("Volatility", f"{metrics['volatility']:.2%}")
            
        except Exception as e:
            self.logger.error(f"Error displaying portfolio metrics: {str(e)}")
            st.error(f"Error displaying metrics: {str(e)}")
    
    def _display_trade_analysis(self, portfolio) -> None:
        """
        Display trade analysis in Streamlit.
        
        Args:
            portfolio: VectorBT portfolio object
        """
        try:
            trades = portfolio.trades
            
            if trades.count() == 0:
                st.info("No trades available for analysis")
                return
            
            # Trade summary
            st.subheader("📊 Trade Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Trades", trades.count())
            
            with col2:
                st.metric("Win Rate", f"{trades.win_rate():.2%}")
            
            with col3:
                st.metric("Profit Factor", f"{trades.profit_factor():.2f}")
            
            # Trade distribution
            st.subheader("📈 Trade Distribution")
            
            # Create trade PnL histogram
            trade_pnl = trades.pnl.values
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=trade_pnl,
                nbinsx=20,
                name="Trade PnL",
                marker_color='blue',
                opacity=0.7
            ))
            
            fig.update_layout(
                title="Trade PnL Distribution",
                xaxis_title="Profit/Loss ($)",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            self.logger.error(f"Error displaying trade analysis: {str(e)}")
            st.error(f"Error displaying trade analysis: {str(e)}")
    
    def _display_performance_overview(self, portfolios: Dict[str, Any]) -> None:
        """
        Display performance overview for multiple portfolios.
        
        Args:
            portfolios: Dictionary of portfolio objects
        """
        try:
            if not portfolios:
                st.info("No portfolios available for analysis")
                return
            
            # Calculate metrics for all portfolios
            performance_data = []
            
            for name, portfolio in portfolios.items():
                metrics = self.viz_engine._extract_portfolio_metrics(portfolio)
                performance_data.append({
                    'Portfolio': name,
                    'Total Return': metrics.get('total_return', 0),
                    'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                    'Max Drawdown': metrics.get('max_drawdown', 0),
                    'Trades': metrics.get('num_trades', 0)
                })
            
            # Create performance comparison DataFrame
            df = pd.DataFrame(performance_data)
            
            # Display as metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_return_idx = df['Total Return'].idxmax()
                best_portfolio = df.loc[best_return_idx, 'Portfolio']
                best_return = df.loc[best_return_idx, 'Total Return']
                st.metric(
                    "Best Performer",
                    best_portfolio,
                    f"{best_return:.2%}"
                )
            
            with col2:
                best_sharpe_idx = df['Sharpe Ratio'].idxmax()
                best_sharpe_portfolio = df.loc[best_sharpe_idx, 'Portfolio']
                best_sharpe = df.loc[best_sharpe_idx, 'Sharpe Ratio']
                st.metric(
                    "Best Risk-Adjusted",
                    best_sharpe_portfolio,
                    f"{best_sharpe:.2f}"
                )
            
            with col3:
                total_trades = df['Trades'].sum()
                st.metric(
                    "Total Trades",
                    f"{int(total_trades)}",
                    f"Across {len(portfolios)} portfolios"
                )
            
            # Display detailed table
            st.subheader("📋 Detailed Performance Table")
            
            # Format the DataFrame for display
            display_df = df.copy()
            display_df['Total Return'] = display_df['Total Return'].apply(lambda x: f"{x:.2%}")
            display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
            display_df['Max Drawdown'] = display_df['Max Drawdown'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(display_df, use_container_width=True)
            
        except Exception as e:
            self.logger.error(f"Error displaying performance overview: {str(e)}")
            st.error(f"Error displaying performance overview: {str(e)}")
    
    def _should_update_real_time(self, update_interval: int) -> bool:
        """
        Check if real-time data should be updated.
        
        Args:
            update_interval: Update interval in seconds
            
        Returns:
            True if update is needed
        """
        last_update = st.session_state.get('last_real_time_update')
        if last_update is None:
            return True
        
        return (datetime.now() - last_update).seconds >= update_interval
    
    def _update_real_time_data(self, data_source_func) -> None:
        """
        Update real-time data from data source.
        
        Args:
            data_source_func: Function to fetch updated data
        """
        try:
            # Fetch new data
            new_data = data_source_func()
            
            # Update session state
            st.session_state.real_time_data = new_data
            st.session_state.last_real_time_update = datetime.now()
            
            self.logger.info("Real-time data updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating real-time data: {str(e)}")
            st.error(f"Real-time update failed: {str(e)}")
    
    def _force_real_time_update(self, data_source_func) -> None:
        """
        Force immediate real-time data update.
        
        Args:
            data_source_func: Function to fetch updated data
        """
        with st.spinner("Updating real-time data..."):
            self._update_real_time_data(data_source_func)
        st.success("Real-time data updated!")
    
    def _display_real_time_data(self) -> None:
        """Display current real-time data."""
        try:
            real_time_data = st.session_state.get('real_time_data', {})
            
            if not real_time_data:
                st.info("No real-time data available")
                return
            
            # Display real-time metrics
            st.subheader("⚡ Live Data")
            
            # This would be customized based on actual real-time data structure
            for key, value in real_time_data.items():
                if isinstance(value, (int, float)):
                    st.metric(key, f"{value:.2f}")
                else:
                    st.text(f"{key}: {value}")
            
        except Exception as e:
            self.logger.error(f"Error displaying real-time data: {str(e)}")
            st.error(f"Error displaying real-time data: {str(e)}")
    
    def _display_cache_management(self) -> None:
        """Display cache management interface."""
        try:
            # Cache statistics
            cache_size = len(st.session_state.get('viz_cache', {}))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Cached Plots", cache_size)
            
            with col2:
                if st.button("🗑️ Clear Cache"):
                    st.session_state.viz_cache = {}
                    st.session_state.viz_cache_timestamps = {}
                    st.success("Cache cleared!")
                    st.rerun()
            
            # Cache settings
            with st.expander("⚙️ Cache Settings"):
                new_ttl = st.slider(
                    "Cache TTL (seconds)",
                    min_value=60,
                    max_value=3600,
                    value=self.cache_ttl,
                    step=60
                )
                
                if new_ttl != self.cache_ttl:
                    self.cache_ttl = new_ttl
                    st.info(f"Cache TTL updated to {new_ttl} seconds")
            
        except Exception as e:
            self.logger.error(f"Error displaying cache management: {str(e)}")
            st.error(f"Error displaying cache management: {str(e)}")


class RealTimeVisualizationManager:
    """
    Manager for real-time visualization updates in Streamlit dashboard.
    
    This class handles real-time data updates and visualization refresh
    for live trading scenarios.
    """
    
    def __init__(self, update_interval: int = 60):
        """
        Initialize real-time visualization manager.
        
        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.is_running = False
        self.data_sources = {}
        
        self.logger = logging.getLogger(__name__)
    
    def register_data_source(self, name: str, source_func) -> None:
        """
        Register a data source for real-time updates.
        
        Args:
            name: Data source name
            source_func: Function that returns updated data
        """
        self.data_sources[name] = source_func
        self.logger.info(f"Registered data source: {name}")
    
    def start_real_time_updates(self) -> None:
        """Start real-time update process."""
        self.is_running = True
        self.logger.info("Real-time updates started")
    
    def stop_real_time_updates(self) -> None:
        """Stop real-time update process."""
        self.is_running = False
        self.logger.info("Real-time updates stopped")
    
    def get_latest_data(self, source_name: str) -> Any:
        """
        Get latest data from specified source.
        
        Args:
            source_name: Name of data source
            
        Returns:
            Latest data from source
        """
        if source_name in self.data_sources:
            try:
                return self.data_sources[source_name]()
            except Exception as e:
                self.logger.error(f"Error fetching data from {source_name}: {str(e)}")
                return None
        else:
            self.logger.warning(f"Data source {source_name} not found")
            return None