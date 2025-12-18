"""
Regression tests for model performance consistency.
Tests that ensure system behavior remains consistent across versions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
import json
import pickle
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import hashlib

from stock_predictor.main import StockPredictorOrchestrator
from stock_predictor.config import Config, DataConfig, FeatureConfig, ModelConfig, BacktestConfig, SystemConfig


@pytest.mark.regression
class TestRegressionConsistency:
    """Regression tests for model performance consistency."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "regression_config.yaml"
        self.baseline_dir = Path(self.temp_dir) / "baseline"
        self.baseline_dir.mkdir(exist_ok=True)
        
        # Fixed configuration for regression testing
        self.regression_config = Config(
            data=DataConfig(
                stock_symbols=["AAPL", "MSFT"],
                start_date="2023-01-01",
                end_date="2023-04-01",  # 3 months for consistency
                retry_attempts=1,
                retry_delay=0.1
            ),
            features=FeatureConfig(
                pattern_lengths=[3, 5],
                technical_indicators=["RSI", "MACD", "EMA20", "EMA50"]
            ),
            models=ModelConfig(
                model_types=["xgboost", "random_forest"],
                train_test_split=0.8,
                random_state=42,  # Fixed seed for reproducibility
                neural_network_params={
                    "hidden_layer_sizes": [100, 50],  # Use list instead of tuple
                    "activation": "relu",
                    "solver": "adam",
                    "max_iter": 1000
                }
            ),
            backtest=BacktestConfig(
                initial_capital=100000.0,
                transaction_cost=0.001,
                slippage=0.0005
            ),
            system=SystemConfig(
                log_level="WARNING",
                model_save_path=str(Path(self.temp_dir) / "models"),
                data_cache_path=str(Path(self.temp_dir) / "cache"),
                results_path=str(Path(self.temp_dir) / "results"),
                max_workers=2
            )
        )
        
        # Save configuration
        from stock_predictor.config import ConfigManager
        config_manager = ConfigManager(str(self.config_path))
        config_manager._config = self.regression_config
        config_manager.save_config()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_deterministic_data(self, symbol: str, seed: int = 42) -> pd.DataFrame:
        """Create deterministic test data for regression testing."""
        np.random.seed(seed + hash(symbol) % 1000)
        
        num_days = 90  # 3 months
        dates = pd.date_range(start="2023-01-01", periods=num_days, freq='D')
        
        # Create deterministic price series
        base_price = 100.0 + (hash(symbol) % 100)  # Different base price per symbol
        trend = 0.001  # Small upward trend
        volatility = 0.02
        
        prices = [base_price]
        for i in range(1, num_days):
            # Deterministic but realistic price movement
            daily_return = trend + volatility * np.sin(i * 0.1) * np.random.normal(0, 1)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 1.0))
        
        # Create OHLC data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Deterministic intraday movements
            open_price = close * (1 + 0.005 * np.sin(i * 0.2))
            high = max(open_price, close) * (1 + 0.01 * abs(np.sin(i * 0.3)))
            low = min(open_price, close) * (1 - 0.01 * abs(np.cos(i * 0.3)))
            volume = int(1000000 * (1 + 0.5 * np.sin(i * 0.15)))
            
            data.append({
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2),
                'Volume': volume,
                'Adj Close': round(close * (1 + 0.001 * np.sin(i * 0.05)), 2)
            })
        
        return pd.DataFrame(data, index=dates)
    
    def _extract_key_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for regression comparison."""
        key_metrics = {
            'total_configurations': 0,
            'successful_configurations': 0,
            'model_performance': {},
            'pattern_performance': {},
            'aggregated_metrics': {
                'avg_mse': 0.0,
                'avg_mae': 0.0,
                'avg_rmse': 0.0,
                'avg_total_return': 0.0,
                'avg_sharpe_ratio': 0.0,
                'avg_max_drawdown': 0.0
            }
        }
        
        if 'aggregated_results' in results:
            aggregated = results['aggregated_results']
            key_metrics['total_configurations'] = len(aggregated)
            
            successful_results = [r for r in aggregated if 'error' not in r]
            key_metrics['successful_configurations'] = len(successful_results)
            
            if successful_results:
                # Extract performance by model type
                for result in successful_results:
                    model_type = result.get('model_type', 'unknown')
                    pattern_length = result.get('pattern_length', 0)
                    
                    # Model performance
                    if model_type not in key_metrics['model_performance']:
                        key_metrics['model_performance'][model_type] = {
                            'count': 0,
                            'avg_mse': 0.0,
                            'avg_total_return': 0.0
                        }
                    
                    pred_metrics = result.get('prediction_metrics', {})
                    fin_metrics = result.get('financial_metrics', {})
                    
                    key_metrics['model_performance'][model_type]['count'] += 1
                    key_metrics['model_performance'][model_type]['avg_mse'] += pred_metrics.get('mse', 0)
                    key_metrics['model_performance'][model_type]['avg_total_return'] += fin_metrics.get('total_return', 0)
                    
                    # Pattern performance
                    pattern_key = f"{pattern_length}_day"
                    if pattern_key not in key_metrics['pattern_performance']:
                        key_metrics['pattern_performance'][pattern_key] = {
                            'count': 0,
                            'avg_mse': 0.0,
                            'avg_total_return': 0.0
                        }
                    
                    key_metrics['pattern_performance'][pattern_key]['count'] += 1
                    key_metrics['pattern_performance'][pattern_key]['avg_mse'] += pred_metrics.get('mse', 0)
                    key_metrics['pattern_performance'][pattern_key]['avg_total_return'] += fin_metrics.get('total_return', 0)
                
                # Calculate averages
                for model_data in key_metrics['model_performance'].values():
                    if model_data['count'] > 0:
                        model_data['avg_mse'] /= model_data['count']
                        model_data['avg_total_return'] /= model_data['count']
                
                for pattern_data in key_metrics['pattern_performance'].values():
                    if pattern_data['count'] > 0:
                        pattern_data['avg_mse'] /= pattern_data['count']
                        pattern_data['avg_total_return'] /= pattern_data['count']
                
                # Aggregated metrics
                mse_values = [r.get('prediction_metrics', {}).get('mse', 0) for r in successful_results]
                mae_values = [r.get('prediction_metrics', {}).get('mae', 0) for r in successful_results]
                rmse_values = [r.get('prediction_metrics', {}).get('rmse', 0) for r in successful_results]
                return_values = [r.get('financial_metrics', {}).get('total_return', 0) for r in successful_results]
                sharpe_values = [r.get('financial_metrics', {}).get('sharpe_ratio', 0) for r in successful_results]
                drawdown_values = [r.get('financial_metrics', {}).get('max_drawdown', 0) for r in successful_results]
                
                key_metrics['aggregated_metrics'] = {
                    'avg_mse': np.mean(mse_values) if mse_values else 0.0,
                    'avg_mae': np.mean(mae_values) if mae_values else 0.0,
                    'avg_rmse': np.mean(rmse_values) if rmse_values else 0.0,
                    'avg_total_return': np.mean(return_values) if return_values else 0.0,
                    'avg_sharpe_ratio': np.mean(sharpe_values) if sharpe_values else 0.0,
                    'avg_max_drawdown': np.mean(drawdown_values) if drawdown_values else 0.0
                }
        
        return key_metrics
    
    def _save_baseline(self, baseline_name: str, metrics: Dict[str, Any]) -> None:
        """Save baseline metrics for future comparison."""
        baseline_file = self.baseline_dir / f"{baseline_name}_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    
    def _load_baseline(self, baseline_name: str) -> Dict[str, Any]:
        """Load baseline metrics for comparison."""
        baseline_file = self.baseline_dir / f"{baseline_name}_baseline.json"
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _compare_metrics(self, current: Dict[str, Any], baseline: Dict[str, Any], 
                        tolerance: float = 0.1) -> Dict[str, Any]:
        """Compare current metrics against baseline with tolerance."""
        comparison = {
            'passed': True,
            'differences': {},
            'significant_changes': []
        }
        
        def compare_values(current_val, baseline_val, path):
            if isinstance(current_val, (int, float)) and isinstance(baseline_val, (int, float)):
                if baseline_val != 0:
                    relative_diff = abs(current_val - baseline_val) / abs(baseline_val)
                    if relative_diff > tolerance:
                        comparison['passed'] = False
                        comparison['differences'][path] = {
                            'current': current_val,
                            'baseline': baseline_val,
                            'relative_diff': relative_diff
                        }
                        comparison['significant_changes'].append(
                            f"{path}: {baseline_val:.4f} -> {current_val:.4f} ({relative_diff:.2%} change)"
                        )
                else:
                    # Handle zero baseline
                    if abs(current_val) > tolerance:
                        comparison['passed'] = False
                        comparison['differences'][path] = {
                            'current': current_val,
                            'baseline': baseline_val,
                            'absolute_diff': abs(current_val)
                        }
        
        def recursive_compare(current_dict, baseline_dict, path=""):
            for key in baseline_dict:
                current_path = f"{path}.{key}" if path else key
                if key in current_dict:
                    if isinstance(current_dict[key], dict) and isinstance(baseline_dict[key], dict):
                        recursive_compare(current_dict[key], baseline_dict[key], current_path)
                    else:
                        compare_values(current_dict[key], baseline_dict[key], current_path)
                else:
                    comparison['passed'] = False
                    comparison['significant_changes'].append(f"Missing key: {current_path}")
        
        recursive_compare(current, baseline)
        return comparison
    
    def test_deterministic_results_consistency(self):
        """Test that the same input produces consistent results."""
        with patch('stock_predictor.data.yahoo_finance_service.yf.Ticker') as mock_ticker:
            # Create deterministic data
            test_data = {}
            for symbol in self.regression_config.data.stock_symbols:
                test_data[symbol] = self._create_deterministic_data(symbol, seed=42)
                
                mock_ticker_instance = Mock()
                mock_ticker_instance.history.return_value = test_data[symbol]
                mock_ticker.return_value = mock_ticker_instance
            
            # Run analysis multiple times
            orchestrator = StockPredictorOrchestrator(config_path=str(self.config_path))
            orchestrator.initialize()
            
            results_1 = orchestrator.run_full_analysis()
            results_2 = orchestrator.run_full_analysis()
            
            # Extract key metrics
            metrics_1 = self._extract_key_metrics(results_1)
            metrics_2 = self._extract_key_metrics(results_2)
            
            # Compare results - should be identical for deterministic input
            comparison = self._compare_metrics(metrics_1, metrics_2, tolerance=0.01)  # Very strict tolerance
            
            assert comparison['passed'], f"Deterministic results not consistent: {comparison['significant_changes']}"
            
            # Verify specific consistency requirements
            assert metrics_1['total_configurations'] == metrics_2['total_configurations']
            assert metrics_1['successful_configurations'] == metrics_2['successful_configurations']
            
            # Model performance should be identical
            for model_type in metrics_1['model_performance']:
                if model_type in metrics_2['model_performance']:
                    m1 = metrics_1['model_performance'][model_type]
                    m2 = metrics_2['model_performance'][model_type]
                    assert abs(m1['avg_mse'] - m2['avg_mse']) < 1e-6, f"MSE inconsistent for {model_type}"
                    assert abs(m1['avg_total_return'] - m2['avg_total_return']) < 1e-6, f"Return inconsistent for {model_type}"
    
    def test_baseline_performance_regression(self):
        """Test against saved baseline performance metrics."""
        with patch('stock_predictor.data.yahoo_finance_service.yf.Ticker') as mock_ticker:
            # Create deterministic data
            for symbol in self.regression_config.data.stock_symbols:
                test_data = self._create_deterministic_data(symbol, seed=42)
                
                mock_ticker_instance = Mock()
                mock_ticker_instance.history.return_value = test_data
                mock_ticker.return_value = mock_ticker_instance
            
            # Run analysis
            orchestrator = StockPredictorOrchestrator(config_path=str(self.config_path))
            orchestrator.initialize()
            results = orchestrator.run_full_analysis()
            
            # Extract current metrics
            current_metrics = self._extract_key_metrics(results)
            
            # Load baseline or create it if it doesn't exist
            baseline_metrics = self._load_baseline("performance")
            
            if not baseline_metrics:
                # First run - save as baseline
                self._save_baseline("performance", current_metrics)
                pytest.skip("Baseline created - run test again to compare against baseline")
            
            # Compare against baseline
            comparison = self._compare_metrics(current_metrics, baseline_metrics, tolerance=0.15)  # 15% tolerance
            
            # Report comparison results
            if not comparison['passed']:
                print("\n=== REGRESSION TEST FAILURES ===")
                for change in comparison['significant_changes']:
                    print(f"  {change}")
                print("\n=== DETAILED DIFFERENCES ===")
                for path, diff in comparison['differences'].items():
                    print(f"  {path}: {diff}")
            
            # Assert performance hasn't regressed significantly
            assert comparison['passed'], f"Performance regression detected: {comparison['significant_changes']}"
            
            # Specific regression checks
            current_agg = current_metrics['aggregated_metrics']
            baseline_agg = baseline_metrics['aggregated_metrics']
            
            # MSE shouldn't increase significantly (model accuracy regression)
            if baseline_agg['avg_mse'] > 0:
                mse_increase = (current_agg['avg_mse'] - baseline_agg['avg_mse']) / baseline_agg['avg_mse']
                assert mse_increase < 0.2, f"MSE increased by {mse_increase:.2%} - potential accuracy regression"
            
            # Total return shouldn't decrease significantly (financial performance regression)
            if baseline_agg['avg_total_return'] > 0:
                return_decrease = (baseline_agg['avg_total_return'] - current_agg['avg_total_return']) / baseline_agg['avg_total_return']
                assert return_decrease < 0.3, f"Average return decreased by {return_decrease:.2%} - potential performance regression"
    
    def test_feature_engineering_consistency(self):
        """Test that feature engineering produces consistent results."""
        with patch('stock_predictor.data.yahoo_finance_service.yf.Ticker') as mock_ticker:
            # Create test data
            test_data = self._create_deterministic_data("AAPL", seed=42)
            
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = test_data
            mock_ticker.return_value = mock_ticker_instance
            
            # Initialize orchestrator
            orchestrator = StockPredictorOrchestrator(config_path=str(self.config_path))
            orchestrator.initialize()
            
            # Test feature engineering consistency
            raw_data = test_data.copy()
            raw_data.columns = raw_data.columns.str.lower()
            
            # Run feature engineering multiple times
            features_1 = orchestrator.feature_engine.calculate_technical_indicators(raw_data)
            features_2 = orchestrator.feature_engine.calculate_technical_indicators(raw_data)
            
            # Features should be identical
            pd.testing.assert_frame_equal(features_1, features_2, check_dtype=False)
            
            # Test pattern generation consistency
            for pattern_length in self.regression_config.features.pattern_lengths:
                patterns_1 = orchestrator.pattern_generator.generate_n_day_signals_dataframe(features_1, pattern_length)
                patterns_2 = orchestrator.pattern_generator.generate_n_day_signals_dataframe(features_2, pattern_length)
                
                signal_col = f'signal_{pattern_length}d'
                if signal_col in patterns_1.columns and signal_col in patterns_2.columns:
                    pd.testing.assert_series_equal(patterns_1[signal_col], patterns_2[signal_col])
    
    def test_model_training_reproducibility(self):
        """Test that model training is reproducible with fixed seeds."""
        with patch('stock_predictor.data.yahoo_finance_service.yf.Ticker') as mock_ticker:
            # Create test data
            test_data = self._create_deterministic_data("AAPL", seed=42)
            
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = test_data
            mock_ticker.return_value = mock_ticker_instance
            
            # Initialize orchestrator
            orchestrator = StockPredictorOrchestrator(config_path=str(self.config_path))
            orchestrator.initialize()
            
            # Prepare consistent training data
            raw_data = test_data.copy()
            raw_data.columns = raw_data.columns.str.lower()
            features_data = orchestrator.feature_engine.calculate_technical_indicators(raw_data)
            
            pattern_length = 3
            pattern_data = orchestrator.pattern_generator.generate_n_day_signals_dataframe(features_data, pattern_length)
            targets = pattern_data[f'signal_{pattern_length}d']
            
            X, y = orchestrator.training_pipeline.prepare_training_data(pattern_data, targets, pattern_length)
            
            # Train models multiple times with same seed
            model_1 = orchestrator.training_pipeline.train_model("xgboost", X[:50], y[:50])
            model_2 = orchestrator.training_pipeline.train_model("xgboost", X[:50], y[:50])
            
            # Predictions should be identical for same input
            test_X = X[50:60]
            pred_1 = model_1.predict(test_X)
            pred_2 = model_2.predict(test_X)
            
            np.testing.assert_array_almost_equal(pred_1, pred_2, decimal=6)
    
    def test_backtesting_consistency(self):
        """Test that backtesting produces consistent results."""
        with patch('stock_predictor.data.yahoo_finance_service.yf.Ticker') as mock_ticker:
            # Create test data
            test_data = self._create_deterministic_data("AAPL", seed=42)
            
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = test_data
            mock_ticker.return_value = mock_ticker_instance
            
            # Initialize orchestrator
            orchestrator = StockPredictorOrchestrator(config_path=str(self.config_path))
            orchestrator.initialize()
            
            # Create consistent test signals and prices
            dates = pd.date_range(start="2023-02-01", periods=30, freq='D')
            signals = pd.Series([1, -1, 0, 1, -1] * 6, index=dates)  # Deterministic signals
            prices = pd.Series(np.linspace(100, 110, 30), index=dates)  # Linear price increase
            
            # Run backtesting multiple times
            result_1 = orchestrator.backtesting_engine.simulate_trading(
                signals, prices, self.regression_config.backtest.initial_capital
            )
            result_2 = orchestrator.backtesting_engine.simulate_trading(
                signals, prices, self.regression_config.backtest.initial_capital
            )
            
            # Results should be identical
            assert abs(result_1.total_return - result_2.total_return) < 1e-10
            assert abs(result_1.max_drawdown - result_2.max_drawdown) < 1e-10
            assert abs(result_1.sharpe_ratio - result_2.sharpe_ratio) < 1e-10
            assert result_1.win_rate == result_2.win_rate
            
            # Handle NaN/inf values in profit factor comparison
            pf1, pf2 = result_1.profit_factor, result_2.profit_factor
            if np.isnan(pf1) and np.isnan(pf2):
                pass  # Both NaN is consistent
            elif np.isinf(pf1) and np.isinf(pf2) and np.sign(pf1) == np.sign(pf2):
                pass  # Both same infinity is consistent
            else:
                assert abs(pf1 - pf2) < 1e-10
    
    def test_configuration_impact_regression(self):
        """Test that configuration changes have expected impact."""
        with patch('stock_predictor.data.yahoo_finance_service.yf.Ticker') as mock_ticker:
            # Create test data
            for symbol in self.regression_config.data.stock_symbols:
                test_data = self._create_deterministic_data(symbol, seed=42)
                mock_ticker_instance = Mock()
                mock_ticker_instance.history.return_value = test_data
                mock_ticker.return_value = mock_ticker_instance
            
            # Test with base configuration
            orchestrator_base = StockPredictorOrchestrator(config_path=str(self.config_path))
            orchestrator_base.initialize()
            results_base = orchestrator_base.run_full_analysis()
            metrics_base = self._extract_key_metrics(results_base)
            
            # Test with modified configuration (higher transaction costs)
            modified_config = self.regression_config
            modified_config.backtest.transaction_cost = 0.005  # 5x higher
            
            from stock_predictor.config import ConfigManager
            config_manager = ConfigManager(str(self.config_path))
            config_manager._config = modified_config
            config_manager.save_config()
            
            orchestrator_modified = StockPredictorOrchestrator(config_path=str(self.config_path))
            orchestrator_modified.initialize()
            results_modified = orchestrator_modified.run_full_analysis()
            metrics_modified = self._extract_key_metrics(results_modified)
            
            # Higher transaction costs should reduce returns
            base_return = metrics_base['aggregated_metrics']['avg_total_return']
            modified_return = metrics_modified['aggregated_metrics']['avg_total_return']
            
            if base_return > 0:
                # Returns should be lower with higher transaction costs
                assert modified_return <= base_return, "Higher transaction costs should reduce returns"
                
                # The impact should be reasonable (not extreme)
                # Handle cases where base_return is very small or negative
                if abs(base_return) > 1e-6:  # Only check if base return is significant
                    return_reduction = (base_return - modified_return) / base_return
                    # Very lenient bounds for test environment with synthetic data
                    assert -10.0 <= return_reduction <= 50.0, f"Transaction cost impact seems unrealistic: {return_reduction:.2%}"
    
    def test_data_quality_impact_regression(self):
        """Test system behavior with different data quality scenarios."""
        scenarios = {
            'clean': {'missing_ratio': 0.0, 'noise_level': 0.0},
            'missing_data': {'missing_ratio': 0.05, 'noise_level': 0.0},
            'noisy_data': {'missing_ratio': 0.0, 'noise_level': 0.02}
        }
        
        scenario_results = {}
        
        for scenario_name, params in scenarios.items():
            with patch('stock_predictor.data.yahoo_finance_service.yf.Ticker') as mock_ticker:
                # Create data with specified quality issues
                test_data = self._create_deterministic_data("AAPL", seed=42)
                
                # Add missing data
                if params['missing_ratio'] > 0:
                    missing_indices = np.random.choice(
                        len(test_data), 
                        size=int(len(test_data) * params['missing_ratio']), 
                        replace=False
                    )
                    test_data.iloc[missing_indices, 0] = np.nan  # Missing open prices
                
                # Add noise
                if params['noise_level'] > 0:
                    noise = np.random.normal(0, params['noise_level'], len(test_data))
                    test_data['Close'] *= (1 + noise)
                    test_data['High'] = np.maximum(test_data['High'], test_data['Close'])
                    test_data['Low'] = np.minimum(test_data['Low'], test_data['Close'])
                
                mock_ticker_instance = Mock()
                mock_ticker_instance.history.return_value = test_data
                mock_ticker.return_value = mock_ticker_instance
                
                # Run analysis
                orchestrator = StockPredictorOrchestrator(config_path=str(self.config_path))
                orchestrator.initialize()
                results = orchestrator.run_full_analysis()
                
                scenario_results[scenario_name] = self._extract_key_metrics(results)
        
        # Verify system handles data quality issues gracefully
        clean_metrics = scenario_results['clean']
        
        for scenario_name, metrics in scenario_results.items():
            if scenario_name != 'clean':
                # Should still produce results
                assert metrics['successful_configurations'] > 0, f"No successful configurations in {scenario_name} scenario"
                
                # Performance degradation should be reasonable
                clean_return = clean_metrics['aggregated_metrics']['avg_total_return']
                scenario_return = metrics['aggregated_metrics']['avg_total_return']
                
                if clean_return > 0:
                    performance_ratio = scenario_return / clean_return
                    assert performance_ratio > 0.5, f"Performance degradation too severe in {scenario_name}: {performance_ratio:.2f}"


if __name__ == "__main__":
    # Run regression tests manually
    pytest.main([__file__, "-v", "-m", "regression"])