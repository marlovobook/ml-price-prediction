#!/usr/bin/env python3
"""
Complete Stock Direction Predictor Workflow Demonstration
========================================================

This script demonstrates the complete workflow from data collection 
through model training to dashboard visualization.

Usage:
    python demo_complete_workflow.py
"""

import sys
import os
from datetime import datetime, timedelta
import json
from pathlib import Path

# Add the stock_predictor module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_predictor.main import StockPredictorOrchestrator
from stock_predictor.data.yahoo_finance_service import YahooFinanceDataService
from stock_predictor.features.feature_engineering import FeatureEngineeringModule
from stock_predictor.features.candlestick_pattern_generator import CandlestickPatternGenerator


def print_banner(title):
    """Print a formatted banner."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def print_step(step_num, description):
    """Print a formatted step."""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 50)


def demonstrate_data_collection():
    """Demonstrate data collection from Yahoo Finance."""
    print_step(1, "Data Collection from Yahoo Finance")
    
    # Initialize data service
    data_service = YahooFinanceDataService()
    
    # Fetch sample data
    symbols = ['AAPL']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    print(f"📊 Fetching data for {symbols} from {start_date} to {end_date}")
    
    try:
        stock_data = data_service.fetch_stock_data(symbols, start_date, end_date)
        
        if 'AAPL' in stock_data:
            df = stock_data['AAPL']
            print(f"✅ Successfully fetched {len(df)} data points for AAPL")
            print(f"📈 Date range: {df.index.min()} to {df.index.max()}")
            print(f"💰 Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
            
            # Show sample data
            print("\n📋 Sample Data (last 5 rows):")
            print(df.tail().round(2))
            
            return df
        else:
            print("❌ No data retrieved")
            return None
            
    except Exception as e:
        print(f"❌ Error fetching data: {str(e)}")
        return None


def demonstrate_feature_engineering(stock_data):
    """Demonstrate feature engineering and technical indicators."""
    print_step(2, "Feature Engineering & Technical Indicators")
    
    if stock_data is None:
        print("❌ No stock data available for feature engineering")
        return None
    
    # Standardize column names
    df = stock_data.copy()
    df.columns = df.columns.str.lower()
    
    # Initialize feature engineering module
    feature_engine = FeatureEngineeringModule()
    
    try:
        print("🔧 Calculating technical indicators...")
        
        # Calculate technical indicators
        df_with_features = feature_engine.calculate_technical_indicators(df)
        
        print("✅ Technical indicators calculated successfully!")
        
        # Show available indicators
        technical_cols = [col for col in df_with_features.columns 
                         if col not in ['open', 'high', 'low', 'close', 'volume']]
        print(f"📊 Generated {len(technical_cols)} technical indicators:")
        for col in technical_cols:
            print(f"   • {col}")
        
        # Show sample values
        print("\n📋 Latest Technical Indicator Values:")
        latest_values = df_with_features[technical_cols].iloc[-1]
        for indicator, value in latest_values.items():
            if not pd.isna(value):
                print(f"   {indicator}: {value:.4f}")
        
        # Generate chart patterns
        print("\n🎯 Detecting chart patterns...")
        df_with_patterns = feature_engine.detect_chart_patterns(df_with_features)
        
        # Check for recent signals
        if 'golden_cross' in df_with_patterns.columns:
            recent_golden_cross = df_with_patterns['golden_cross'].tail(5).sum()
            if recent_golden_cross > 0:
                print("🟢 Golden Cross detected in recent data!")
        
        return df_with_patterns
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {str(e)}")
        return None


def demonstrate_candlestick_patterns(df_with_features):
    """Demonstrate candlestick pattern generation."""
    print_step(3, "Candlestick Pattern Analysis")
    
    if df_with_features is None:
        print("❌ No feature data available for pattern analysis")
        return None
    
    # Initialize pattern generator
    pattern_generator = CandlestickPatternGenerator()
    
    try:
        pattern_lengths = [3, 5, 7]
        
        for pattern_length in pattern_lengths:
            print(f"\n🕯️ Analyzing {pattern_length}-day candlestick patterns...")
            
            # Generate signals
            signals = pattern_generator.generate_n_day_signals(df_with_features, pattern_length)
            
            # Get latest signal
            if len(signals) > 0:
                latest_signal = signals.iloc[-1]
                signal_text = "🟢 BUY" if latest_signal == 1 else "🔴 SELL" if latest_signal == -1 else "🟡 HOLD"
                print(f"   Latest {pattern_length}d signal: {signal_text}")
                
                # Count signal distribution
                signal_counts = signals.value_counts()
                print(f"   Signal distribution: {dict(signal_counts)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in pattern analysis: {str(e)}")
        return False


def demonstrate_full_analysis():
    """Demonstrate the complete analysis workflow."""
    print_step(4, "Complete ML Analysis Workflow")
    
    try:
        # Initialize orchestrator
        print("🚀 Initializing Stock Direction Predictor...")
        orchestrator = StockPredictorOrchestrator()
        orchestrator.initialize()
        
        print("✅ Orchestrator initialized successfully!")
        
        # Run a quick analysis with limited scope for demo
        print("\n🔄 Running comprehensive analysis...")
        print("   Symbols: AAPL")
        print("   Pattern lengths: 3, 5 days")
        print("   Models: XGBoost, Random Forest")
        
        # Run analysis
        results = orchestrator.run_comprehensive_comparison(
            symbols=['AAPL']
        )
        
        if 'best_configuration' in results:
            best = results['best_configuration']
            print("\n🏆 Analysis Results:")
            print(f"   Best Model: {best.get('model_type', 'N/A')}")
            print(f"   Best Pattern Length: {best.get('pattern_length', 'N/A')} days")
            print(f"   Recommendation Score: {best.get('recommendation_score', 0):.1f}")
            
            metrics = best.get('performance_metrics', {})
            if metrics:
                print(f"   Total Return: {metrics.get('total_return', 0)*100:.1f}%")
                print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"   Max Drawdown: {abs(metrics.get('max_drawdown', 0))*100:.1f}%")
        
        # Show saved files
        print("\n📁 Generated Files:")
        results_dir = Path("results")
        if results_dir.exists():
            for file in results_dir.glob("*.json"):
                print(f"   📄 {file.name}")
        
        models_dir = Path("models")
        if models_dir.exists():
            model_count = len(list(models_dir.glob("*.pkl")))
            print(f"   🤖 {model_count} trained models saved")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in full analysis: {str(e)}")
        return False


def demonstrate_dashboard_info():
    """Show information about the dashboard."""
    print_step(5, "Interactive Dashboard")
    
    print("🌐 Streamlit Dashboard Features:")
    print("   • Interactive model performance comparison")
    print("   • Real-time technical indicator charts")
    print("   • Backtesting results visualization")
    print("   • Risk-return analysis")
    print("   • Model prediction confidence")
    
    print("\n🚀 To launch the dashboard:")
    print("   streamlit run streamlit_dashboard.py")
    print("   Then open: http://localhost:8501")
    
    print("\n📊 Dashboard Sections:")
    print("   1. Performance Overview - Compare all models")
    print("   2. Backtesting Results - Portfolio performance over time")
    print("   3. Technical Analysis - Live charts with indicators")
    print("   4. Detailed Metrics - Comprehensive performance tables")
    print("   5. Model Insights - Prediction confidence and statistics")


def main():
    """Run the complete workflow demonstration."""
    print_banner("Stock Direction Predictor - Complete Workflow Demo")
    
    print("🎯 This demonstration shows the complete workflow:")
    print("   1. Data Collection from Yahoo Finance")
    print("   2. Feature Engineering & Technical Indicators")
    print("   3. Candlestick Pattern Analysis")
    print("   4. Machine Learning Model Training & Evaluation")
    print("   5. Interactive Dashboard Visualization")
    
    # Import pandas here to avoid issues if not available
    try:
        import pandas as pd
        globals()['pd'] = pd
    except ImportError:
        print("❌ pandas not available, some features may not work")
        return
    
    # Step 1: Data Collection
    stock_data = demonstrate_data_collection()
    
    # Step 2: Feature Engineering
    df_with_features = demonstrate_feature_engineering(stock_data)
    
    # Step 3: Candlestick Patterns
    demonstrate_candlestick_patterns(df_with_features)
    
    # Step 4: Full Analysis
    demonstrate_full_analysis()
    
    # Step 5: Dashboard Info
    demonstrate_dashboard_info()
    
    print_banner("Workflow Demonstration Complete!")
    
    print("🎉 The Stock Direction Predictor system is now ready!")
    print("\n📋 Next Steps:")
    print("   1. Launch the dashboard: streamlit run streamlit_dashboard.py")
    print("   2. Explore different stocks and model configurations")
    print("   3. Analyze the generated performance reports")
    print("   4. Use the trained models for predictions")
    
    print("\n📁 Generated Assets:")
    print("   • Trained ML models in 'models/' directory")
    print("   • Analysis results in 'results/' directory")
    print("   • Performance charts in 'results/charts/' directory")
    print("   • Cached data in 'data_cache/' directory")


if __name__ == "__main__":
    main()