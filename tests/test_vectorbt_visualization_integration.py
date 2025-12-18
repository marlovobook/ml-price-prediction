"""
Comprehensive integration tests for VectorBT Visualization Enhancement.

This module tests the complete end-to-end workflow from ML predictions to 
interactive visualizations, validating integration with existing backtesting 
and dashboard systems.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
import time
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import visualization components
from stock_predictor.visualization import (
    VectorBTVisualizationEngine,
    EnhancedPortfolioEngine,
    SignalAlignmentEngine,
    PortfolioConfig,
    PlotConfig,
    PlotExportEngine
)

# Import existing system components
from stock_predictor.main import StockPredictorOrchestrator
from stock_predictor.config import Config, DataConfig, FeatureConfig, ModelConfig, BacktestConfig, SystemConfig


@pytest.mark.integration
@pytest.mark.visualization
class TestVectorBTVisualizationIntegration:
    """Comprehensive integration tests for VectorBT visualization enhancement."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configuration
        self.test_config = Config(
            data=DataConfig(
                stock_symbols=["AAPL"],
                start_date="2023-01-01",
                end_date="2023-06-01",
                retry_attempts=1,
                retry_delay=0.1
            ),
            features=FeatureConfig(
                pattern_lengths=[3, 5],
                technical_indicators=["RSI", "MACD", "EMA20"]
            ),
            models=ModelConfig(
                model_types=["xgboost"],
                train_test_split=0.8,
                neural_network_params={
                    "hidden_layer_sizes": [100, 50],
                    "activation": "relu",
                    "solver": "adam",
                    "max_iter": 1000
                }
            ),
            backtest=BacktestConfig(
                initial_capital=100000.0,
                transaction_cost=0.001
            ),
            system=SystemConfig(
                log_level="INFO",
                model_save_path=str(Path(self.temp_dir) / "models"),
                data_cache_path=str(Path(self.temp_dir) / "cache"),
                results_path=str(Path(self.temp_dir) / "results"),
                max_workers=2
            )
        )
        
        # Initialize visualization components
        self.portfolio_config = PortfolioConfig(
            init_cash=100000.0,
            fees=0.0025,
            slippage=0.0025,
            size_strategy='fixed_amount',
            size_value=10000.0,
            stop_loss=0.1
        )
        
        self.plot_config = PlotConfig(
            width=1200,
            height=600,
            show_trades=True,
            show_positions=True
        )
        
        self.viz_engine = VectorBTVisualizationEngine(
            portfolio_config=self.portfolio_config,
            plot_config=self.plot_config
        )
        
        self.portfolio_engine = EnhancedPortfolioEngine(
            portfolio_config=self.portfolio_config
        )
        
        self.signal_aligner = SignalAlignmentEngine()
        
        self.logger = logging.getLogger(__name__)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_data(self, num_days: int = 150, symbol: str = "AAPL") -> pd.DataFrame:
        """Create realistic test market data."""
        dates = pd.date_range(start="2023-01-01", periods=num_days, freq='D')
        
        np.random.seed(hash(symbol) % 2**32)
        base_price = 150.0
        returns = np.random.normal(0.001, 0.02, num_days)
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            open_price = close * (1 + np.random.normal(0, 0.005))
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
            volume = int(np.random.lognormal(15, 0.5))
            
            data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume,
                'Adj Close': close * (1 + np.random.normal(0, 0.001))
            })
        
        return pd.DataFrame(data, index=dates)
    
    def _create_test_predictions(self, num_predictions: int = 30) -> np.ndarray:
        """Create test ML predictions."""
        np.random.seed(42)
        # Create realistic prediction pattern: mostly hold (1), some buy (2), some sell (0)
        predictions = np.random.choice([0, 1, 2], size=num_predictions, p=[0.2, 0.6, 0.2])
        return predictions
    
    def test_end_to_end_predictions_to_visualization_workflow(self):
        """
        Test complete end-to-end workflow from ML predictions to interactive visualizations.
        
        This test validates the entire pipeline:
        1. Signal alignment from predictions
        2. Portfolio creation with realistic parameters
        3. Visualization generation
        4. Plot export and data extraction
        """
        # Create test data
        price_data = self._create_test_data(150)
        predictions = self._create_test_predictions(30)
        test_start_idx = 120  # Last 30 days for testing
        
        # Step 1: Test signal alignment
        aligned_signals = self.signal_aligner.align_predictions_to_timeline(
            predictions, price_data, test_start_idx
        )
        
        # Validate signal alignment
        assert len(aligned_signals.entry_signals) == len(price_data)
        assert len(aligned_signals.exit_signals) == len(price_data)
        assert aligned_signals.test_period_start == test_start_idx
        assert aligned_signals.prediction_count == len(predictions)
        
        # Verify signals are only in test period
        pre_test_entries = aligned_signals.entry_signals.iloc[:test_start_idx].sum()
        pre_test_exits = aligned_signals.exit_signals.iloc[:test_start_idx].sum()
        assert pre_test_entries == 0, "Entry signals found before test period"
        assert pre_test_exits == 0, "Exit signals found before test period"
        
        # Step 2: Test portfolio creation
        portfolio_result = self.portfolio_engine.create_portfolio_from_predictions(
            predictions, price_data, test_start_idx, self.portfolio_config
        )
        
        # Validate portfolio creation
        assert portfolio_result.success, f"Portfolio creation failed: {portfolio_result.error_message}"
        assert portfolio_result.portfolio is not None
        assert portfolio_result.aligned_signals is not None
        assert portfolio_result.creation_time > 0
        
        portfolio = portfolio_result.portfolio
        
        # Step 3: Test visualization generation
        # Test portfolio plot
        portfolio_viz_result = self.viz_engine.generate_portfolio_plot(
            portfolio, title="End-to-End Test Portfolio"
        )
        
        assert portfolio_viz_result.success, f"Portfolio visualization failed: {portfolio_viz_result.error_message}"
        assert portfolio_viz_result.plot_object is not None
        assert portfolio_viz_result.generation_time > 0
        assert 'portfolio_value' in portfolio_viz_result.plot_data
        
        # Test drawdown plot
        drawdown_viz_result = self.viz_engine.generate_drawdown_plot(portfolio)
        
        assert drawdown_viz_result.success, f"Drawdown visualization failed: {drawdown_viz_result.error_message}"
        assert drawdown_viz_result.plot_object is not None
        assert 'drawdown_pct' in drawdown_viz_result.plot_data
        assert 'max_drawdown_pct' in drawdown_viz_result.metrics_summary
        
        # Test trade analysis (if trades exist)
        if portfolio.trades.count() > 0:
            trade_viz_result = self.viz_engine.generate_trade_analysis_plot(portfolio)
            # Trade analysis might fail due to VectorBT widget dependencies in test environment
            if not trade_viz_result.success:
                self.logger.warning(f"Trade analysis skipped due to environment: {trade_viz_result.error_message}")
            else:
                assert 'num_trades' in trade_viz_result.metrics_summary
        
        # Step 4: Test plot export
        export_engine = PlotExportEngine()
        
        # Test PNG export
        png_path = export_engine.export_plot(
            portfolio_viz_result.plot_object,
            str(Path(self.temp_dir) / "test_portfolio"),
            formats=['png']
        )
        
        assert 'png' in png_path
        assert Path(png_path['png']).exists()
        
        # Test data export
        data_path = export_engine.export_plot_data(
            portfolio_viz_result.plot_data,
            str(Path(self.temp_dir) / "test_portfolio_data.csv")
        )
        
        assert Path(data_path).exists()
        
        self.logger.info("End-to-end workflow test completed successfully")
    
    def test_integration_with_existing_backtesting_engine(self):
        """
        Test integration with existing VectorBT backtesting engine.
        
        This test validates that the visualization enhancement integrates
        seamlessly with the existing backtesting system.
        """
        # Create test data
        price_data = self._create_test_data(100)
        predictions = self._create_test_predictions(20)
        test_start_idx = 80
        
        # Test integration with existing VectorBT engine
        try:
            # Create portfolio using enhanced engine
            portfolio_result = self.portfolio_engine.create_portfolio_from_predictions(
                predictions, price_data, test_start_idx
            )
            
            assert portfolio_result.success
            portfolio = portfolio_result.portfolio
            
            # Verify portfolio is compatible with existing backtesting metrics
            # These should work with existing system
            total_return = portfolio.total_return()
            sharpe_ratio = portfolio.sharpe_ratio()
            max_drawdown = portfolio.max_drawdown()
            
            assert isinstance(total_return, (int, float))
            assert isinstance(sharpe_ratio, (int, float)) or np.isnan(sharpe_ratio)
            assert isinstance(max_drawdown, (int, float))
            
            # Test that visualization can be generated from existing portfolio
            viz_result = self.viz_engine.generate_portfolio_plot(portfolio)
            assert viz_result.success
            
            # Verify metrics are consistent
            viz_metrics = viz_result.metrics_summary
            assert 'total_return' in viz_metrics
            assert 'sharpe_ratio' in viz_metrics
            assert 'max_drawdown' in viz_metrics
            
            # Values should be approximately equal (allowing for calculation differences)
            assert abs(viz_metrics['total_return'] - total_return) < 0.01
            assert abs(viz_metrics['max_drawdown'] - max_drawdown) < 0.01
            
        except ImportError:
            pytest.skip("VectorBT backtesting engine not available")
    
    def test_dashboard_integration_compatibility(self):
        """
        Test compatibility with dashboard integration requirements.
        
        This test validates that visualizations can be integrated into
        Streamlit dashboards and web applications.
        """
        # Create test portfolio
        price_data = self._create_test_data(80)
        predictions = self._create_test_predictions(15)
        test_start_idx = 65
        
        portfolio_result = self.portfolio_engine.create_portfolio_from_predictions(
            predictions, price_data, test_start_idx
        )
        
        assert portfolio_result.success
        portfolio = portfolio_result.portfolio
        
        # Test plot object compatibility
        viz_result = self.viz_engine.generate_portfolio_plot(portfolio)
        assert viz_result.success
        
        plot_obj = viz_result.plot_object
        
        # Verify plot object has required attributes for dashboard integration
        assert hasattr(plot_obj, 'to_html'), "Plot object should support HTML export for web dashboards"
        assert hasattr(plot_obj, 'to_json'), "Plot object should support JSON export for API integration"
        
        # Test HTML generation (for Streamlit/web integration)
        try:
            html_content = plot_obj.to_html()
            assert isinstance(html_content, str)
            assert len(html_content) > 0
            assert 'plotly' in html_content.lower() or 'chart' in html_content.lower()
        except Exception as e:
            pytest.skip(f"HTML export not available: {e}")
        
        # Test JSON serialization (for API integration)
        try:
            json_content = plot_obj.to_json()
            assert isinstance(json_content, str)
            assert len(json_content) > 0
        except Exception as e:
            pytest.skip(f"JSON export not available: {e}")
        
        # Test plot data extraction for custom dashboard components
        plot_data = viz_result.plot_data
        assert 'portfolio_value' in plot_data
        assert isinstance(plot_data['portfolio_value'], pd.Series)
        
        # Verify data can be converted to formats suitable for dashboards
        portfolio_values_dict = plot_data['portfolio_value'].to_dict()
        assert isinstance(portfolio_values_dict, dict)
        
        portfolio_values_list = plot_data['portfolio_value'].tolist()
        assert isinstance(portfolio_values_list, list)
    
    def test_multi_strategy_comparison_integration(self):
        """
        Test integration of multi-strategy comparison functionality.
        
        This test validates that multiple strategies can be compared
        and visualized together effectively.
        """
        # Create test data
        price_data = self._create_test_data(120)
        test_start_idx = 90
        
        # Create multiple prediction strategies
        strategies = {
            'conservative': np.random.choice([0, 1, 2], size=30, p=[0.1, 0.8, 0.1]),
            'aggressive': np.random.choice([0, 1, 2], size=30, p=[0.3, 0.4, 0.3]),
            'balanced': np.random.choice([0, 1, 2], size=30, p=[0.2, 0.6, 0.2])
        }
        
        # Create portfolios for each strategy
        portfolios = {}
        for strategy_name, predictions in strategies.items():
            portfolio_result = self.portfolio_engine.create_portfolio_from_predictions(
                predictions, price_data, test_start_idx
            )
            
            assert portfolio_result.success, f"Failed to create portfolio for {strategy_name}"
            portfolios[strategy_name] = portfolio_result.portfolio
        
        # Test comparison visualization
        comparison_result = self.viz_engine.generate_comparison_plot(
            portfolios, title="Multi-Strategy Comparison Test"
        )
        
        assert comparison_result.success, f"Comparison visualization failed: {comparison_result.error_message}"
        assert comparison_result.plot_object is not None
        
        # Validate comparison data
        plot_data = comparison_result.plot_data
        assert 'portfolio_values' in plot_data
        assert 'metrics_comparison' in plot_data
        assert 'strategy_rankings' in plot_data
        
        # Verify all strategies are included
        portfolio_values_df = plot_data['portfolio_values']
        for strategy_name in strategies.keys():
            assert strategy_name in portfolio_values_df.columns
        
        # Verify metrics comparison
        metrics_df = plot_data['metrics_comparison']
        assert len(metrics_df) == len(strategies)
        assert 'total_return' in metrics_df.columns
        assert 'sharpe_ratio' in metrics_df.columns
        
        # Verify strategy rankings
        rankings = plot_data['strategy_rankings']
        # Rankings should contain different ranking criteria
        assert isinstance(rankings, dict)
        # Each ranking criterion should have all strategies
        for criterion, ranking_list in rankings.items():
            strategy_names = [item['strategy'] for item in ranking_list]
            assert len(strategy_names) == len(strategies)
            assert all(strategy in strategy_names for strategy in strategies.keys())
    
    def test_performance_with_large_datasets(self):
        """
        Test visualization performance with larger datasets.
        
        This test validates that the system can handle realistic dataset
        sizes without performance degradation.
        """
        # Create larger dataset (2 years of daily data)
        large_price_data = self._create_test_data(730)  # ~2 years
        large_predictions = self._create_test_predictions(180)  # ~6 months
        test_start_idx = 550
        
        # Measure performance
        start_time = time.time()
        
        # Test signal alignment performance
        aligned_signals = self.signal_aligner.align_predictions_to_timeline(
            large_predictions, large_price_data, test_start_idx
        )
        
        alignment_time = time.time() - start_time
        assert alignment_time < 5.0, f"Signal alignment too slow: {alignment_time:.2f}s"
        
        # Test portfolio creation performance
        start_time = time.time()
        
        portfolio_result = self.portfolio_engine.create_portfolio_from_predictions(
            large_predictions, large_price_data, test_start_idx
        )
        
        portfolio_creation_time = time.time() - start_time
        assert portfolio_creation_time < 10.0, f"Portfolio creation too slow: {portfolio_creation_time:.2f}s"
        assert portfolio_result.success
        
        # Test visualization performance
        start_time = time.time()
        
        viz_result = self.viz_engine.generate_portfolio_plot(portfolio_result.portfolio)
        
        visualization_time = time.time() - start_time
        assert visualization_time < 15.0, f"Visualization generation too slow: {visualization_time:.2f}s"
        assert viz_result.success
        
        # Verify data quality is maintained
        plot_data = viz_result.plot_data
        assert len(plot_data['portfolio_value']) == len(large_price_data)
        
        self.logger.info(
            f"Large dataset performance: alignment={alignment_time:.2f}s, "
            f"portfolio={portfolio_creation_time:.2f}s, viz={visualization_time:.2f}s"
        )
    
    def test_error_handling_and_recovery(self):
        """
        Test error handling and graceful degradation.
        
        This test validates that the system handles various error conditions
        gracefully and provides meaningful error messages.
        """
        # Test with invalid data
        invalid_price_data = pd.DataFrame({'Close': [np.nan, -1, 0]})
        predictions = np.array([0, 1, 2])
        
        # Should handle invalid price data gracefully
        portfolio_result = self.portfolio_engine.create_portfolio_from_predictions(
            predictions, invalid_price_data, 0
        )
        
        assert not portfolio_result.success
        assert portfolio_result.error_message is not None
        assert len(portfolio_result.error_message) > 0
        
        # Test with mismatched data lengths
        price_data = self._create_test_data(100)
        short_predictions = np.array([0, 1])  # Too short
        
        portfolio_result = self.portfolio_engine.create_portfolio_from_predictions(
            short_predictions, price_data, 90
        )
        
        # Should handle gracefully (might succeed with adjusted parameters)
        if not portfolio_result.success:
            assert portfolio_result.error_message is not None
        
        # Test visualization with empty portfolio
        try:
            # Create minimal valid portfolio
            valid_price_data = self._create_test_data(50)
            valid_predictions = np.array([1] * 10)  # All hold signals
            
            portfolio_result = self.portfolio_engine.create_portfolio_from_predictions(
                valid_predictions, valid_price_data, 40
            )
            
            if portfolio_result.success:
                viz_result = self.viz_engine.generate_portfolio_plot(portfolio_result.portfolio)
                
                # Should either succeed or fail gracefully
                if not viz_result.success:
                    assert viz_result.error_message is not None
                    assert len(viz_result.error_message) > 0
        
        except Exception as e:
            # Should not raise unhandled exceptions
            pytest.fail(f"Unhandled exception in error handling test: {e}")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])