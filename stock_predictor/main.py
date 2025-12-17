"""
Main application orchestrator for the Stock Direction Predictor system.
Coordinates all components and provides configuration-driven workflow execution.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import concurrent.futures
import pandas as pd

from .config import load_config, Config
from .utils.logging_config import setup_logging, get_logger
from .utils.error_handler import ErrorContext
from .utils.exceptions import ConfigurationError, DataCollectionError

# Import all components
from .data.yahoo_finance_service import YahooFinanceDataService
from .features.feature_engineering import FeatureEngineeringModule
from .features.candlestick_pattern_generator import CandlestickPatternGenerator
from .models.training_pipeline import ModelTrainingPipeline
from .backtesting.backtesting_engine import BacktestingEngine
from .evaluation.performance_evaluator import PerformanceEvaluator
from .evaluation.comparison_framework import ComparisonFramework


class StockPredictorOrchestrator:
    """
    Main application orchestrator that coordinates all components
    and provides configuration-driven workflow execution.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Stock Direction Predictor orchestrator.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path or "config.yaml"
        self.config: Optional[Config] = None
        self.logger = None
        
        # Component instances
        self.data_service = None
        self.feature_engine = None
        self.pattern_generator = None
        self.training_pipeline = None
        self.backtesting_engine = None
        self.performance_evaluator = None
        self.comparison_framework = None
        
    def initialize(self) -> None:
        """Initialize the system with configuration and logging."""
        with ErrorContext("System Initialization"):
            # Load configuration
            self.config = load_config(self.config_path)
            
            # Setup logging
            setup_logging(
                log_level=self.config.system.log_level,
                log_file=self.config.system.log_file
            )
            self.logger = get_logger("StockPredictorOrchestrator")
            
            # Create necessary directories
            self._create_directories()
            
            # Validate configuration
            self._validate_configuration()
            
            # Initialize components
            self._initialize_components()
            
            self.logger.info("Stock Direction Predictor orchestrator initialized successfully")
    
    def _create_directories(self) -> None:
        """Create necessary directories for the system."""
        if not self.config:
            raise ConfigurationError("Configuration not loaded")
        
        directories = [
            self.config.system.model_save_path,
            self.config.system.data_cache_path,
            self.config.system.results_path
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")
    
    def _validate_configuration(self) -> None:
        """Validate the loaded configuration."""
        if not self.config:
            raise ConfigurationError("Configuration not loaded")
        
        # Validate stock symbols
        if not self.config.data.stock_symbols:
            raise ConfigurationError("No stock symbols configured")
        
        # Validate pattern lengths
        if not self.config.features.pattern_lengths:
            raise ConfigurationError("No pattern lengths configured")
        
        # Validate model types
        if not self.config.models.model_types:
            raise ConfigurationError("No model types configured")
        
        # Validate date range
        if self.config.data.start_date >= self.config.data.end_date:
            raise ConfigurationError("Invalid date range: start_date must be before end_date")
        
        self.logger.info("Configuration validation completed successfully")
    
    def _initialize_components(self) -> None:
        """Initialize all system components."""
        if not self.config:
            raise ConfigurationError("Configuration not loaded")
        
        # Initialize data collection service
        self.data_service = YahooFinanceDataService(
            cache_dir=self.config.system.data_cache_path,
            max_retries=self.config.data.retry_attempts,
            retry_delay=self.config.data.retry_delay
        )
        
        # Initialize feature engineering components
        self.feature_engine = FeatureEngineeringModule()
        self.pattern_generator = CandlestickPatternGenerator()
        
        # Initialize model training pipeline
        self.training_pipeline = ModelTrainingPipeline(
            models_dir=self.config.system.model_save_path
        )
        
        # Initialize backtesting engine
        self.backtesting_engine = BacktestingEngine(
            transaction_cost=self.config.backtest.transaction_cost,
            slippage=self.config.backtest.slippage,
            max_position_size=self.config.backtest.position_size
        )
        
        # Initialize performance evaluator
        self.performance_evaluator = PerformanceEvaluator(
            risk_free_rate=self.config.backtest.risk_free_rate
        )
        
        # Initialize comparison framework
        self.comparison_framework = ComparisonFramework(confidence_level=0.95)
        
        self.logger.info("All components initialized successfully")
    
    def run_full_analysis(self, symbols: Optional[List[str]] = None, 
                         pattern_lengths: Optional[List[int]] = None,
                         model_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run complete stock direction prediction analysis.
        
        Args:
            symbols: List of stock symbols to analyze (optional, uses config default)
            pattern_lengths: List of pattern lengths to test (optional, uses config default)
            model_types: List of model types to train (optional, uses config default)
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        if not self.logger:
            raise ConfigurationError("System not initialized. Call initialize() first.")
        
        # Use configuration defaults if not specified
        symbols = symbols or self.config.data.stock_symbols
        pattern_lengths = pattern_lengths or self.config.features.pattern_lengths
        model_types = model_types or self.config.models.model_types
        
        self.logger.info(f"Starting full analysis for {len(symbols)} symbols, "
                        f"{len(pattern_lengths)} pattern lengths, {len(model_types)} model types")
        
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'configuration': {
                'symbols': symbols,
                'pattern_lengths': pattern_lengths,
                'model_types': model_types,
                'date_range': {
                    'start': self.config.data.start_date,
                    'end': self.config.data.end_date
                }
            },
            'symbol_results': {},
            'aggregated_results': [],
            'performance_report': {}
        }
        
        # Process each symbol
        for symbol in symbols:
            self.logger.info(f"Processing symbol: {symbol}")
            try:
                symbol_results = self._analyze_single_symbol(symbol, pattern_lengths, model_types)
                results['symbol_results'][symbol] = symbol_results
                
                # Add to aggregated results
                for result in symbol_results.get('model_results', []):
                    result['symbol'] = symbol
                    results['aggregated_results'].append(result)
                    
            except Exception as e:
                self.logger.error(f"Error processing symbol {symbol}: {str(e)}")
                results['symbol_results'][symbol] = {'error': str(e)}
        
        # Generate comprehensive performance report
        if results['aggregated_results']:
            results['performance_report'] = self.performance_evaluator.generate_performance_report(
                results['aggregated_results']
            )
        
        # Save results
        self._save_results(results)
        
        self.logger.info("Full analysis completed successfully")
        return results
    
    def _analyze_single_symbol(self, symbol: str, pattern_lengths: List[int], 
                              model_types: List[str]) -> Dict[str, Any]:
        """
        Analyze a single stock symbol with all pattern lengths and model types.
        
        Args:
            symbol: Stock symbol to analyze
            pattern_lengths: List of pattern lengths to test
            model_types: List of model types to train
            
        Returns:
            Dictionary containing analysis results for the symbol
        """
        self.logger.info(f"Collecting data for {symbol}")
        
        # Step 1: Collect data
        stock_data = self.data_service.fetch_stock_data(
            symbols=[symbol],
            start_date=self.config.data.start_date,
            end_date=self.config.data.end_date
        )
        
        if symbol not in stock_data or stock_data[symbol].empty:
            raise DataCollectionError(f"No data available for symbol {symbol}")
        
        raw_data = stock_data[symbol].copy()
        
        # Standardize column names to lowercase
        raw_data.columns = raw_data.columns.str.lower()
        
        # Step 2: Feature engineering
        self.logger.info(f"Generating features for {symbol}")
        features_data = self.feature_engine.calculate_technical_indicators(raw_data)
        features_data = self.feature_engine.detect_chart_patterns(features_data)
        features_data = self.feature_engine.calculate_fibonacci_levels(features_data)
        
        symbol_results = {
            'symbol': symbol,
            'data_points': len(features_data),
            'date_range': {
                'start': features_data.index.min().isoformat(),
                'end': features_data.index.max().isoformat()
            },
            'model_results': []
        }
        
        # Step 3: Train and evaluate models for each pattern length
        for pattern_length in pattern_lengths:
            self.logger.info(f"Processing {pattern_length}-day patterns for {symbol}")
            
            # Generate candlestick signals
            pattern_data = self.pattern_generator.generate_n_day_signals_dataframe(features_data, pattern_length)
            
            # Get target signals
            signal_column = f'signal_{pattern_length}d'
            if signal_column not in pattern_data.columns:
                self.logger.warning(f"Signal column {signal_column} not found, skipping pattern length {pattern_length}")
                continue
            
            targets = pattern_data[signal_column]
            
            # Train models for this pattern length
            for model_type in model_types:
                self.logger.info(f"Training {model_type} model for {symbol} with {pattern_length}-day patterns")
                
                try:
                    model_result = self._train_and_evaluate_model(
                        symbol, pattern_data, targets, pattern_length, model_type
                    )
                    symbol_results['model_results'].append(model_result)
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_type} for {symbol} "
                                    f"with {pattern_length}-day patterns: {str(e)}")
                    
                    # Add error result
                    symbol_results['model_results'].append({
                        'symbol': symbol,
                        'model_type': model_type,
                        'pattern_length': pattern_length,
                        'error': str(e)
                    })
        
        return symbol_results
    
    def _train_and_evaluate_model(self, symbol: str, features_data: pd.DataFrame, 
                                 targets: pd.Series, pattern_length: int, 
                                 model_type: str) -> Dict[str, Any]:
        """
        Train and evaluate a single model configuration.
        
        Args:
            symbol: Stock symbol
            features_data: DataFrame with features
            targets: Target signals
            pattern_length: Candlestick pattern length
            model_type: Type of model to train
            
        Returns:
            Dictionary containing model evaluation results
        """
        # Prepare training data
        X, y = self.training_pipeline.prepare_training_data(features_data, targets, pattern_length)
        
        # Create time-based splits
        splits = self.training_pipeline.create_time_based_splits(X, y, n_splits=3)
        
        # Use the last split for final evaluation
        train_idx, test_idx = splits[-1]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model = self.training_pipeline.train_model(model_type, X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate prediction metrics
        prediction_metrics = self.performance_evaluator.calculate_prediction_metrics(y_test, y_pred)
        
        # Create signals series for backtesting
        test_dates = features_data.index[test_idx]
        signals_series = pd.Series(y_pred, index=test_dates)
        prices_series = features_data.loc[test_dates, 'close']
        
        # Run backtesting
        backtest_result = self.backtesting_engine.simulate_trading(
            signals_series, prices_series, self.config.backtest.initial_capital
        )
        
        # Calculate financial metrics
        financial_metrics = self.performance_evaluator.calculate_financial_metrics(backtest_result)
        
        # Save model
        model_id = f"{symbol}_{model_type}_pattern{pattern_length}"
        model_path = self.training_pipeline.save_model(model, model_id)
        
        return {
            'symbol': symbol,
            'model_type': model_type,
            'pattern_length': pattern_length,
            'model_path': model_path,
            'prediction_metrics': prediction_metrics,
            'financial_metrics': financial_metrics,
            'backtest_result': {
                'total_return': backtest_result.total_return,
                'max_drawdown': backtest_result.max_drawdown,
                'sharpe_ratio': backtest_result.sharpe_ratio,
                'win_rate': backtest_result.win_rate,
                'profit_factor': backtest_result.profit_factor,
                'num_trades': len(backtest_result.trade_log)
            }
        }
    
    def run_batch_analysis(self, batch_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run batch processing for multiple stocks and time periods.
        
        Args:
            batch_config: Configuration for batch processing
            
        Returns:
            Dictionary containing batch analysis results
        """
        self.logger.info("Starting batch analysis")
        
        batch_results = {
            'batch_timestamp': datetime.now().isoformat(),
            'batch_config': batch_config,
            'results': []
        }
        
        # Extract batch parameters
        symbol_groups = batch_config.get('symbol_groups', [self.config.data.stock_symbols])
        time_periods = batch_config.get('time_periods', [
            {'start': self.config.data.start_date, 'end': self.config.data.end_date}
        ])
        
        # Process each combination
        for i, symbols in enumerate(symbol_groups):
            for j, time_period in enumerate(time_periods):
                self.logger.info(f"Processing batch {i+1}-{j+1}: symbols={symbols}, "
                               f"period={time_period['start']} to {time_period['end']}")
                
                # Temporarily update configuration
                original_start = self.config.data.start_date
                original_end = self.config.data.end_date
                
                try:
                    self.config.data.start_date = time_period['start']
                    self.config.data.end_date = time_period['end']
                    
                    # Run analysis for this batch
                    batch_result = self.run_full_analysis(symbols=symbols)
                    batch_result['batch_id'] = f"{i+1}-{j+1}"
                    batch_result['time_period'] = time_period
                    
                    batch_results['results'].append(batch_result)
                    
                except Exception as e:
                    self.logger.error(f"Error in batch {i+1}-{j+1}: {str(e)}")
                    batch_results['results'].append({
                        'batch_id': f"{i+1}-{j+1}",
                        'time_period': time_period,
                        'symbols': symbols,
                        'error': str(e)
                    })
                
                finally:
                    # Restore original configuration
                    self.config.data.start_date = original_start
                    self.config.data.end_date = original_end
        
        # Save batch results
        self._save_batch_results(batch_results)
        
        self.logger.info("Batch analysis completed")
        return batch_results
    
    def run_comparison_analysis(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive comparison analysis across all configurations.
        
        Args:
            symbols: List of symbols to analyze (optional)
            
        Returns:
            Dictionary containing comparison analysis results
        """
        self.logger.info("Starting comparison analysis")
        
        # Run full analysis
        results = self.run_full_analysis(symbols=symbols)
        
        # Generate additional comparison insights
        comparison_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'base_results': results,
            'pattern_length_comparison': {},
            'model_type_comparison': {},
            'symbol_comparison': {}
        }
        
        if results['aggregated_results']:
            # Analyze by pattern length
            comparison_results['pattern_length_comparison'] = self._compare_by_pattern_length(
                results['aggregated_results']
            )
            
            # Analyze by model type
            comparison_results['model_type_comparison'] = self._compare_by_model_type(
                results['aggregated_results']
            )
            
            # Analyze by symbol
            comparison_results['symbol_comparison'] = self._compare_by_symbol(
                results['aggregated_results']
            )
        
        # Save comparison results
        self._save_comparison_results(comparison_results)
        
        self.logger.info("Comparison analysis completed")
        return comparison_results
    
    def run_comprehensive_comparison(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive comparison analysis using the advanced comparison framework.
        
        Args:
            symbols: List of symbols to analyze (optional)
            
        Returns:
            Dictionary containing comprehensive comparison analysis results
        """
        self.logger.info("Starting comprehensive comparison analysis with statistical testing")
        
        # Run full analysis to get all model-pattern combinations
        results = self.run_full_analysis(symbols=symbols)
        
        if not results['aggregated_results']:
            self.logger.warning("No results available for comparison analysis")
            return {'error': 'No results available for comparison'}
        
        # Use the comparison framework for advanced analysis
        comparison_report = self.comparison_framework.compare_all_combinations(
            results['aggregated_results']
        )
        
        # Generate visualization data
        comparison_results_objects = []
        for result in comparison_report['detailed_results']:
            from .evaluation.comparison_framework import ComparisonResult
            comp_result = ComparisonResult(
                model_type=result['model_type'],
                pattern_length=result['pattern_length'],
                performance_metrics=result['performance_metrics'],
                statistical_significance=result['statistical_significance'],
                rank=result['rank'],
                recommendation_score=result['recommendation_score']
            )
            comparison_results_objects.append(comp_result)
        
        visualization_data = self.comparison_framework.generate_visualization_data(
            comparison_results_objects
        )
        
        # Select best configuration with default criteria
        best_configuration = self.comparison_framework.select_best_configuration(
            comparison_results_objects
        )
        
        # Create comprehensive results
        comprehensive_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'base_results': results,
            'comparison_report': comparison_report,
            'visualization_data': visualization_data,
            'best_configuration': {
                'model_type': best_configuration.model_type,
                'pattern_length': best_configuration.pattern_length,
                'recommendation_score': best_configuration.recommendation_score,
                'performance_metrics': best_configuration.performance_metrics,
                'rank': best_configuration.rank
            },
            'statistical_analysis': comparison_report.get('statistical_tests', {}),
            'recommendations': comparison_report.get('recommendations', [])
        }
        
        # Generate performance charts if results directory exists
        try:
            charts_dir = Path(self.config.system.results_path) / "charts"
            charts_dir.mkdir(exist_ok=True)
            
            chart_paths = self.comparison_framework.create_performance_charts(
                comparison_results_objects, 
                save_path=str(charts_dir)
            )
            comprehensive_results['chart_paths'] = chart_paths
            
        except Exception as e:
            self.logger.warning(f"Could not generate charts: {str(e)}")
            comprehensive_results['chart_paths'] = {}
        
        # Save comprehensive results
        self._save_comprehensive_comparison_results(comprehensive_results)
        
        self.logger.info("Comprehensive comparison analysis completed")
        return comprehensive_results
    
    def _compare_by_pattern_length(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare results by pattern length."""
        pattern_comparison = {}
        
        for result in results:
            if 'error' in result:
                continue
                
            pattern_length = result.get('pattern_length')
            if pattern_length not in pattern_comparison:
                pattern_comparison[pattern_length] = {
                    'results': [],
                    'avg_return': 0.0,
                    'avg_sharpe': 0.0,
                    'best_model': None
                }
            
            pattern_comparison[pattern_length]['results'].append(result)
        
        # Calculate averages and find best models
        for pattern_length, data in pattern_comparison.items():
            if data['results']:
                returns = [r.get('financial_metrics', {}).get('total_return', 0) for r in data['results']]
                sharpes = [r.get('financial_metrics', {}).get('sharpe_ratio', 0) for r in data['results']]
                
                data['avg_return'] = sum(returns) / len(returns)
                data['avg_sharpe'] = sum(sharpes) / len(sharpes)
                
                # Find best model by total return
                best_result = max(data['results'], 
                                key=lambda x: x.get('financial_metrics', {}).get('total_return', 0))
                data['best_model'] = {
                    'model_type': best_result.get('model_type'),
                    'symbol': best_result.get('symbol'),
                    'total_return': best_result.get('financial_metrics', {}).get('total_return', 0)
                }
        
        return pattern_comparison
    
    def _compare_by_model_type(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare results by model type."""
        model_comparison = {}
        
        for result in results:
            if 'error' in result:
                continue
                
            model_type = result.get('model_type')
            if model_type not in model_comparison:
                model_comparison[model_type] = {
                    'results': [],
                    'avg_return': 0.0,
                    'avg_sharpe': 0.0,
                    'best_config': None
                }
            
            model_comparison[model_type]['results'].append(result)
        
        # Calculate averages and find best configurations
        for model_type, data in model_comparison.items():
            if data['results']:
                returns = [r.get('financial_metrics', {}).get('total_return', 0) for r in data['results']]
                sharpes = [r.get('financial_metrics', {}).get('sharpe_ratio', 0) for r in data['results']]
                
                data['avg_return'] = sum(returns) / len(returns)
                data['avg_sharpe'] = sum(sharpes) / len(sharpes)
                
                # Find best configuration by total return
                best_result = max(data['results'], 
                                key=lambda x: x.get('financial_metrics', {}).get('total_return', 0))
                data['best_config'] = {
                    'pattern_length': best_result.get('pattern_length'),
                    'symbol': best_result.get('symbol'),
                    'total_return': best_result.get('financial_metrics', {}).get('total_return', 0)
                }
        
        return model_comparison
    
    def _compare_by_symbol(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare results by symbol."""
        symbol_comparison = {}
        
        for result in results:
            if 'error' in result:
                continue
                
            symbol = result.get('symbol')
            if symbol not in symbol_comparison:
                symbol_comparison[symbol] = {
                    'results': [],
                    'avg_return': 0.0,
                    'avg_sharpe': 0.0,
                    'best_config': None
                }
            
            symbol_comparison[symbol]['results'].append(result)
        
        # Calculate averages and find best configurations
        for symbol, data in symbol_comparison.items():
            if data['results']:
                returns = [r.get('financial_metrics', {}).get('total_return', 0) for r in data['results']]
                sharpes = [r.get('financial_metrics', {}).get('sharpe_ratio', 0) for r in data['results']]
                
                data['avg_return'] = sum(returns) / len(returns)
                data['avg_sharpe'] = sum(sharpes) / len(sharpes)
                
                # Find best configuration by total return
                best_result = max(data['results'], 
                                key=lambda x: x.get('financial_metrics', {}).get('total_return', 0))
                data['best_config'] = {
                    'model_type': best_result.get('model_type'),
                    'pattern_length': best_result.get('pattern_length'),
                    'total_return': best_result.get('financial_metrics', {}).get('total_return', 0)
                }
        
        return symbol_comparison
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{timestamp}.json"
        filepath = Path(self.config.system.results_path) / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def _save_batch_results(self, results: Dict[str, Any]) -> None:
        """Save batch analysis results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_results_{timestamp}.json"
        filepath = Path(self.config.system.results_path) / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Batch results saved to {filepath}")
    
    def _save_comparison_results(self, results: Dict[str, Any]) -> None:
        """Save comparison analysis results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_results_{timestamp}.json"
        filepath = Path(self.config.system.results_path) / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Comparison results saved to {filepath}")
    
    def _save_comprehensive_comparison_results(self, results: Dict[str, Any]) -> None:
        """Save comprehensive comparison analysis results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_comparison_{timestamp}.json"
        filepath = Path(self.config.system.results_path) / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive comparison results saved to {filepath}")
    
    def get_config(self) -> Config:
        """Get the system configuration."""
        if not self.config:
            raise ConfigurationError("System not initialized. Call initialize() first.")
        return self.config


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line interface parser."""
    parser = argparse.ArgumentParser(
        description="Stock Direction Predictor - ML-based stock direction prediction system"
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['full', 'batch', 'comparison', 'comprehensive', 'single'],
        default='full',
        help='Analysis mode to run (default: full)'
    )
    
    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        help='Stock symbols to analyze (overrides config)'
    )
    
    parser.add_argument(
        '--patterns', '-p',
        nargs='+',
        type=int,
        help='Pattern lengths to test (overrides config)'
    )
    
    parser.add_argument(
        '--models', '-M',
        nargs='+',
        help='Model types to train (overrides config)'
    )
    
    parser.add_argument(
        '--batch-config',
        type=str,
        help='Path to batch configuration JSON file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for results (overrides config)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main():
    """Main entry point for the application."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    try:
        # Initialize orchestrator
        orchestrator = StockPredictorOrchestrator(config_path=args.config)
        orchestrator.initialize()
        
        # Override configuration with CLI arguments
        config = orchestrator.get_config()
        if args.output_dir:
            config.system.results_path = args.output_dir
        if args.verbose:
            config.system.log_level = "DEBUG"
        
        # Get logger
        logger = get_logger("main")
        logger.info("Stock Direction Predictor orchestrator started")
        
        # Display configuration summary
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Symbols: {args.symbols or config.data.stock_symbols}")
        logger.info(f"Pattern lengths: {args.patterns or config.features.pattern_lengths}")
        logger.info(f"Model types: {args.models or config.models.model_types}")
        
        # Run analysis based on mode
        if args.mode == 'full':
            results = orchestrator.run_full_analysis(
                symbols=args.symbols,
                pattern_lengths=args.patterns,
                model_types=args.models
            )
            logger.info("Full analysis completed successfully")
            
        elif args.mode == 'batch':
            if args.batch_config:
                with open(args.batch_config, 'r') as f:
                    batch_config = json.load(f)
            else:
                # Default batch configuration
                batch_config = {
                    'symbol_groups': [config.data.stock_symbols],
                    'time_periods': [
                        {'start': config.data.start_date, 'end': config.data.end_date}
                    ]
                }
            
            results = orchestrator.run_batch_analysis(batch_config)
            logger.info("Batch analysis completed successfully")
            
        elif args.mode == 'comparison':
            results = orchestrator.run_comparison_analysis(symbols=args.symbols)
            logger.info("Comparison analysis completed successfully")
            
        elif args.mode == 'comprehensive':
            results = orchestrator.run_comprehensive_comparison(symbols=args.symbols)
            logger.info("Comprehensive comparison analysis completed successfully")
            
        elif args.mode == 'single':
            # Single symbol analysis (first symbol only)
            symbols = args.symbols or config.data.stock_symbols
            if symbols:
                results = orchestrator.run_full_analysis(symbols=[symbols[0]])
                logger.info(f"Single symbol analysis for {symbols[0]} completed successfully")
            else:
                raise ValueError("No symbols specified for single mode")
        
        # Display summary results
        if 'performance_report' in results and results['performance_report']:
            report = results['performance_report']
            if 'best_configuration' in report and report['best_configuration']:
                best = report['best_configuration']
                logger.info(f"Best configuration: {best.get('model_type')} with "
                           f"{best.get('pattern_length')}-day patterns")
                logger.info(f"Best total return: {best.get('financial_metrics', {}).get('total_return', 0):.2%}")
                logger.info(f"Best Sharpe ratio: {best.get('financial_metrics', {}).get('sharpe_ratio', 0):.3f}")
        
        logger.info("Stock Direction Predictor orchestrator completed successfully")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()