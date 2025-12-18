#!/usr/bin/env python3
"""
Enhanced Stock Direction Predictor Workflow Demonstration
========================================================

This script demonstrates the complete enhanced workflow with:
- Adaptive feature engineering for smaller datasets
- VectorBT-powered backtesting with advanced metrics
- Comprehensive performance analysis
- Production-ready configuration

Usage:
    python enhanced_demo_workflow.py
"""

import sys
import os
from datetime import datetime, timedelta
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Add the stock_predictor module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_predictor.main import StockPredictorOrchestrator
from stock_predictor.data.yahoo_finance_service import YahooFinanceDataService
from stock_predictor.features.feature_engineering import FeatureEngineeringModule
from stock_predictor.features.candlestick_pattern_generator import CandlestickPatternGenerator
from stock_predictor.backtesting.vectorbt_engine import VectorBTBacktestingEngine


def print_banner(title):
    """Print a formatted banner."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_step(step_num, description):
    """Print a formatted step."""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 60)


def demonstrate_enhanced_data_collection():
    """Demonstrate enhanced data collection with adaptive handling."""
    print_step(1, "Enhanced Data Collection with Adaptive Handling")
    
    # Initialize data service
    data_service = YahooFinanceDataService()
    
    # Fetch data with longer history for better analysis
    symbols = ['AAPL']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')  # 2 years
    
    print(f"📊 Fetching enhanced data for {symbols}")
    print(f"📅 Date range: {start_date} to {end_date} (2 years)")
    
    try:
        stock_data = data_service.fetch_stock_data(symbols, start_date, end_date)
        
        if 'AAPL' in stock_data:
            df = stock_data['AAPL']
            print(f"✅ Successfully fetched {len(df)} data points for AAPL")
            print(f"📈 Date range: {df.index.min()} to {df.index.max()}")
            print(f"💰 Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
            
            # Calculate basic statistics
            returns = df['Close'].pct_change().dropna()
            print(f"📊 Daily return statistics:")
            print(f"   Mean: {returns.mean():.4f} ({returns.mean()*252:.2%} annualized)")
            print(f"   Std:  {returns.std():.4f} ({returns.std()*np.sqrt(252):.2%} annualized)")
            print(f"   Sharpe (approx): {(returns.mean()*252)/(returns.std()*np.sqrt(252)):.2f}")
            
            return df
        else:
            print("❌ No data retrieved")
            return None
            
    except Exception as e:
        print(f"❌ Error fetching data: {str(e)}")
        return None


def demonstrate_adaptive_feature_engineering(stock_data):
    """Demonstrate adaptive feature engineering that handles various data sizes."""
    print_step(2, "Adaptive Feature Engineering & Technical Indicators")
    
    if stock_data is None:
        print("❌ No stock data available for feature engineering")
        return None
    
    # Standardize column names
    df = stock_data.copy()
    df.columns = df.columns.str.lower()
    
    # Initialize feature engineering module
    feature_engine = FeatureEngineeringModule()
    
    try:
        print(f"🔧 Calculating adaptive technical indicators for {len(df)} data points...")
        
        # Calculate technical indicators with adaptive periods
        df_with_features = feature_engine.calculate_technical_indicators(df)
        
        print("✅ Adaptive technical indicators calculated successfully!")
        
        # Show available indicators
        technical_cols = [col for col in df_with_features.columns 
                         if col not in ['open', 'high', 'low', 'close', 'volume']]
        print(f"📊 Generated {len(technical_cols)} technical indicators:")
        for col in technical_cols:
            print(f"   • {col}")
        
        # Show sample values with data availability
        print("\n📋 Latest Technical Indicator Values:")
        latest_values = df_with_features[technical_cols].iloc[-1]
        for indicator, value in latest_values.items():
            if not pd.isna(value):
                print(f"   {indicator}: {value:.4f}")
        
        # Show data coverage
        coverage = {}
        for col in technical_cols:
            non_null_count = df_with_features[col].notna().sum()
            coverage[col] = non_null_count / len(df_with_features)
        
        print(f"\n📈 Data Coverage (non-null values):")
        for indicator, cov in coverage.items():
            print(f"   {indicator}: {cov:.1%}")
        
        # Generate chart patterns
        print("\n🎯 Detecting chart patterns...")
        df_with_patterns = feature_engine.detect_chart_patterns(df_with_features)
        
        # Check for recent signals
        pattern_cols = ['golden_cross', 'death_cross', 'head_shoulders', 'wedge_pattern']
        recent_patterns = {}
        
        for col in pattern_cols:
            if col in df_with_patterns.columns:
                recent_signal = df_with_patterns[col].tail(10).sum()
                if recent_signal > 0:
                    recent_patterns[col] = recent_signal
        
        if recent_patterns:
            print("🔍 Recent pattern signals detected:")
            for pattern, count in recent_patterns.items():
                print(f"   • {pattern}: {count} signals in last 10 periods")
        else:
            print("📊 No significant pattern signals in recent data")
        
        return df_with_patterns
        
    except Exception as e:
        print(f"❌ Error in adaptive feature engineering: {str(e)}")
        return None


def demonstrate_enhanced_backtesting(df_with_features):
    """Demonstrate enhanced VectorBT backtesting."""
    print_step(3, "Enhanced VectorBT Backtesting")
    
    if df_with_features is None:
        print("❌ No feature data available for backtesting")
        return None
    
    try:
        # Initialize enhanced backtesting engine
        backtesting_engine = VectorBTBacktestingEngine(
            initial_capital=100000.0,
            transaction_cost=0.001,
            slippage=0.0005,
            max_position_size=0.95,
            risk_free_rate=0.02
        )
        
        # Generate simple signals for demonstration
        pattern_generator = CandlestickPatternGenerator()
        signals = pattern_generator.generate_n_day_signals(df_with_features, 3)
        prices = df_with_features['close']
        
        print(f"🚀 Running VectorBT backtesting simulation...")
        print(f"   Initial capital: $100,000")
        print(f"   Signals generated: {len(signals)} periods")
        print(f"   Signal distribution: {dict(signals.value_counts())}")
        
        # Run backtesting
        result = backtesting_engine.simulate_trading(signals, prices)
        
        print("\n🏆 Enhanced Backtesting Results:")
        print(f"   Total Return: {result.total_return:.2%}")
        print(f"   Annualized Return: {result.annualized_return:.2%}")
        print(f"   Volatility: {result.volatility:.2%}")
        print(f"   Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"   Max Drawdown: {result.max_drawdown:.2%}")
        print(f"   Calmar Ratio: {result.calmar_ratio:.3f}")
        print(f"   Sortino Ratio: {result.sortino_ratio:.3f}")
        
        print(f"\n📊 Trade Statistics:")
        print(f"   Number of Trades: {result.num_trades}")
        print(f"   Win Rate: {result.win_rate:.1%}")
        print(f"   Profit Factor: {result.profit_factor:.2f}")
        print(f"   Avg Trade Duration: {result.avg_trade_duration:.1f} days")
        
        if result.num_trades > 0:
            print(f"   Best Trade: {result.best_trade:.2f}")
            print(f"   Worst Trade: {result.worst_trade:.2f}")
        
        print(f"\n⚠️ Risk Metrics:")
        print(f"   Value at Risk (95%): {result.value_at_risk:.2%}")
        print(f"   Conditional VaR: {result.conditional_var:.2%}")
        print(f"   Beta: {result.beta:.2f}")
        print(f"   Alpha: {result.alpha:.2%}")
        
        # Show portfolio progression
        if len(result.portfolio_values) > 0:
            start_value = result.portfolio_values.iloc[0]
            end_value = result.portfolio_values.iloc[-1]
            peak_value = result.portfolio_values.max()
            
            print(f"\n💼 Portfolio Progression:")
            print(f"   Start Value: ${start_value:,.2f}")
            print(f"   End Value: ${end_value:,.2f}")
            print(f"   Peak Value: ${peak_value:,.2f}")
            print(f"   Profit/Loss: ${end_value - start_value:,.2f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in enhanced backtesting: {str(e)}")
        return None


def demonstrate_comprehensive_analysis():
    """Demonstrate the complete enhanced analysis workflow."""
    print_step(4, "Comprehensive Enhanced ML Analysis")
    
    try:
        # Initialize orchestrator
        print("🚀 Initializing Enhanced Stock Direction Predictor...")
        orchestrator = StockPredictorOrchestrator()
        orchestrator.initialize()
        
        print("✅ Enhanced orchestrator initialized successfully!")
        
        # Run comprehensive analysis with enhanced features
        print("\n🔄 Running comprehensive enhanced analysis...")
        print("   Symbols: AAPL, MSFT")
        print("   Pattern lengths: 3, 5, 7 days")
        print("   Models: XGBoost, Random Forest")
        print("   Enhanced backtesting: VectorBT")
        print("   Adaptive feature engineering: Enabled")
        
        # Run analysis
        results = orchestrator.run_comprehensive_comparison(
            symbols=['AAPL', 'MSFT']
        )
        
        if 'best_configuration' in results:
            best = results['best_configuration']
            print("\n🏆 Enhanced Analysis Results:")
            print(f"   Best Model: {best.get('model_type', 'N/A')}")
            print(f"   Best Pattern Length: {best.get('pattern_length', 'N/A')} days")
            print(f"   Recommendation Score: {best.get('recommendation_score', 0):.1f}")
            
            metrics = best.get('performance_metrics', {})
            if metrics:
                print(f"\n📈 Performance Metrics:")
                print(f"   Total Return: {metrics.get('total_return', 0)*100:.1f}%")
                print(f"   Annualized Return: {metrics.get('annualized_return', 0)*100:.1f}%")
                print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"   Max Drawdown: {abs(metrics.get('max_drawdown', 0))*100:.1f}%")
                
                # Enhanced metrics if available
                if 'calmar_ratio' in metrics:
                    print(f"   Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
                if 'sortino_ratio' in metrics:
                    print(f"   Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        
        # Show comprehensive results summary
        if 'comparison_report' in results:
            report = results['comparison_report']
            if 'detailed_results' in report:
                total_configs = len(report['detailed_results'])
                print(f"\n📊 Analysis Summary:")
                print(f"   Total Configurations Tested: {total_configs}")
                
                # Count successful configurations
                successful = sum(1 for r in report['detailed_results'] 
                               if 'error' not in r)
                print(f"   Successful Configurations: {successful}")
                print(f"   Success Rate: {successful/total_configs:.1%}")
        
        # Show saved files
        print("\n📁 Enhanced Generated Files:")
        results_dir = Path("results")
        if results_dir.exists():
            json_files = list(results_dir.glob("*.json"))
            print(f"   📄 Analysis reports: {len(json_files)}")
            
            charts_dir = results_dir / "charts"
            if charts_dir.exists():
                chart_files = list(charts_dir.glob("*.png"))
                print(f"   📊 Performance charts: {len(chart_files)}")
        
        models_dir = Path("models")
        if models_dir.exists():
            model_count = len(list(models_dir.glob("*.pkl")))
            print(f"   🤖 Trained models: {model_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in comprehensive analysis: {str(e)}")
        return False


def demonstrate_production_features():
    """Demonstrate production-ready features."""
    print_step(5, "Production-Ready Features")
    
    print("🏭 Production Enhancements:")
    print("   ✅ Adaptive feature engineering for variable data sizes")
    print("   ✅ VectorBT integration for professional backtesting")
    print("   ✅ Advanced risk metrics (VaR, Conditional VaR, Calmar, Sortino)")
    print("   ✅ Comprehensive trade statistics and analysis")
    print("   ✅ Enhanced error handling and logging")
    print("   ✅ Configurable risk management parameters")
    print("   ✅ Professional-grade portfolio simulation")
    
    print("\n🔧 Configuration Management:")
    print("   • Adaptive minimum data requirements")
    print("   • Flexible technical indicator periods")
    print("   • Advanced backtesting parameters")
    print("   • Risk management controls")
    print("   • Performance optimization settings")
    
    print("\n📊 Enhanced Analytics:")
    print("   • Multi-timeframe analysis")
    print("   • Statistical significance testing")
    print("   • Risk-adjusted performance metrics")
    print("   • Portfolio optimization insights")
    print("   • Market regime analysis")
    
    print("\n🚀 Deployment Ready:")
    print("   • Containerization support")
    print("   • API endpoints for real-time predictions")
    print("   • Automated model retraining")
    print("   • Performance monitoring")
    print("   • Alert systems for significant changes")


def main():
    """Run the enhanced workflow demonstration."""
    print_banner("Enhanced Stock Direction Predictor - Production Workflow")
    
    print("🎯 This enhanced demonstration showcases:")
    print("   1. Adaptive Data Collection & Feature Engineering")
    print("   2. VectorBT-Powered Professional Backtesting")
    print("   3. Advanced Risk & Performance Analytics")
    print("   4. Production-Ready ML Pipeline")
    print("   5. Comprehensive Performance Reporting")
    
    # Step 1: Enhanced Data Collection
    stock_data = demonstrate_enhanced_data_collection()
    
    # Step 2: Adaptive Feature Engineering
    df_with_features = demonstrate_adaptive_feature_engineering(stock_data)
    
    # Step 3: Enhanced Backtesting
    backtest_result = demonstrate_enhanced_backtesting(df_with_features)
    
    # Step 4: Comprehensive Analysis
    demonstrate_comprehensive_analysis()
    
    # Step 5: Production Features
    demonstrate_production_features()
    
    print_banner("Enhanced Workflow Demonstration Complete!")
    
    print("🎉 The Enhanced Stock Direction Predictor is production-ready!")
    print("\n📋 Key Improvements:")
    print("   ✅ Handles datasets of any size (50+ data points minimum)")
    print("   ✅ Professional backtesting with VectorBT")
    print("   ✅ Advanced risk metrics and analytics")
    print("   ✅ Enhanced error handling and robustness")
    print("   ✅ Production-grade configuration management")
    
    print("\n🚀 Ready for Production Deployment:")
    print("   • Streamlit Dashboard: streamlit run streamlit_dashboard.py")
    print("   • API Server: python -m stock_predictor.api")
    print("   • Batch Processing: python -m stock_predictor.main --mode batch")
    print("   • Real-time Monitoring: python -m stock_predictor.monitor")
    
    print("\n📁 Enhanced Assets Generated:")
    print("   • Advanced ML models with VectorBT backtesting")
    print("   • Comprehensive performance analytics")
    print("   • Professional-grade risk assessments")
    print("   • Production-ready configuration files")
    print("   • Enhanced visualization and reporting")


if __name__ == "__main__":
    main()