#!/usr/bin/env python3
"""
Test script for the Stock Direction Predictor Dashboard components.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the stock_predictor module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required imports work."""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        from stock_predictor.main import StockPredictorOrchestrator
        print("✅ Stock predictor orchestrator imported successfully")
    except ImportError as e:
        print(f"❌ Stock predictor import failed: {e}")
        return False
    
    try:
        from stock_predictor.data.yahoo_finance_service import YahooFinanceDataService
        print("✅ Yahoo Finance service imported successfully")
    except ImportError as e:
        print(f"❌ Yahoo Finance service import failed: {e}")
        return False
    
    return True


def test_data_fetching():
    """Test data fetching functionality."""
    print("\n🧪 Testing data fetching...")
    
    try:
        from stock_predictor.data.yahoo_finance_service import YahooFinanceDataService
        
        data_service = YahooFinanceDataService()
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"📅 Fetching data from {start_date} to {end_date}")
        stock_data = data_service.fetch_stock_data(['AAPL'], start_date, end_date)
        
        if 'AAPL' in stock_data and not stock_data['AAPL'].empty:
            print(f"✅ Data fetched successfully: {len(stock_data['AAPL'])} rows")
            print(f"📊 Columns: {list(stock_data['AAPL'].columns)}")
            return True
        else:
            print("❌ No data returned")
            return False
            
    except Exception as e:
        print(f"❌ Data fetching failed: {e}")
        return False


def test_technical_indicators():
    """Test technical indicator calculations."""
    print("\n🧪 Testing technical indicators...")
    
    try:
        from stock_predictor.features.feature_engineering import FeatureEngineeringModule
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'open': 100 + np.random.randn(len(dates)).cumsum(),
            'high': 102 + np.random.randn(len(dates)).cumsum(),
            'low': 98 + np.random.randn(len(dates)).cumsum(),
            'close': 101 + np.random.randn(len(dates)).cumsum(),
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        sample_data['high'] = np.maximum(sample_data['high'], 
                                       np.maximum(sample_data['open'], sample_data['close']))
        sample_data['low'] = np.minimum(sample_data['low'], 
                                      np.minimum(sample_data['open'], sample_data['close']))
        
        feature_engine = FeatureEngineeringModule()
        result = feature_engine.calculate_technical_indicators(sample_data)
        
        expected_indicators = ['rsi', 'macd', 'ema_20', 'ema_50', 'ema_200']
        found_indicators = [col for col in expected_indicators if col in result.columns]
        
        print(f"✅ Technical indicators calculated: {found_indicators}")
        print(f"📊 Result shape: {result.shape}")
        
        return len(found_indicators) > 0
        
    except Exception as e:
        print(f"❌ Technical indicator calculation failed: {e}")
        return False


def test_candlestick_patterns():
    """Test candlestick pattern generation."""
    print("\n🧪 Testing candlestick patterns...")
    
    try:
        from stock_predictor.features.candlestick_pattern_generator import CandlestickPatternGenerator
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'open': 100 + np.random.randn(len(dates)).cumsum() * 0.1,
            'close': 100 + np.random.randn(len(dates)).cumsum() * 0.1,
        }, index=dates)
        
        pattern_generator = CandlestickPatternGenerator()
        
        for pattern_length in [3, 5, 7, 14]:
            signals = pattern_generator.generate_n_day_signals(sample_data, pattern_length)
            unique_signals = signals.unique()
            
            print(f"✅ {pattern_length}-day patterns generated: {len(signals)} signals")
            print(f"   Signal distribution: {dict(zip(*np.unique(signals, return_counts=True)))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Candlestick pattern generation failed: {e}")
        return False


def test_visualization_data():
    """Test visualization data preparation."""
    print("\n🧪 Testing visualization data preparation...")
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        # Create sample performance data
        sample_results = [
            {
                'model_type': 'xgboost',
                'pattern_length': 3,
                'recommendation_score': 85.5,
                'performance_metrics': {
                    'total_return': 0.125,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': -0.08
                }
            },
            {
                'model_type': 'random_forest',
                'pattern_length': 5,
                'recommendation_score': 78.3,
                'performance_metrics': {
                    'total_return': 0.098,
                    'sharpe_ratio': 0.9,
                    'max_drawdown': -0.12
                }
            }
        ]
        
        # Test bar chart creation
        df = pd.DataFrame([
            {
                'Model': result['model_type'],
                'Pattern Length': f"{result['pattern_length']}d",
                'Recommendation Score': result['recommendation_score']
            }
            for result in sample_results
        ])
        
        fig = px.bar(df, x='Model', y='Recommendation Score', 
                    color='Pattern Length',
                    title="Sample Performance Chart")
        
        print("✅ Bar chart created successfully")
        
        # Test scatter plot creation
        scatter_df = pd.DataFrame([
            {
                'Risk': abs(result['performance_metrics']['max_drawdown']) * 100,
                'Return': result['performance_metrics']['total_return'] * 100,
                'Model': result['model_type']
            }
            for result in sample_results
        ])
        
        fig2 = px.scatter(scatter_df, x='Risk', y='Return', color='Model',
                         title="Sample Risk-Return Chart")
        
        print("✅ Scatter plot created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization data preparation failed: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    print("\n🧪 Testing configuration...")
    
    try:
        import yaml
        
        # Test if dashboard config exists and is valid
        config_path = "dashboard_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print("✅ Dashboard configuration loaded successfully")
            print(f"📋 Default symbols: {config.get('dashboard', {}).get('default_symbols', [])}")
            print(f"📏 Default patterns: {config.get('dashboard', {}).get('default_patterns', [])}")
            
            return True
        else:
            print("⚠️  Dashboard configuration file not found (optional)")
            return True
            
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def run_all_tests():
    """Run all dashboard tests."""
    print("=" * 60)
    print("🧪 Stock Direction Predictor Dashboard Tests")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Data Fetching", test_data_fetching),
        ("Technical Indicators", test_technical_indicators),
        ("Candlestick Patterns", test_candlestick_patterns),
        ("Visualization Data", test_visualization_data),
        ("Configuration", test_configuration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard should work correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)