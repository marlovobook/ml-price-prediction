#!/usr/bin/env python3
"""
Demo script for the Stock Direction Predictor Dashboard.
Shows basic usage and features.
"""

import sys
import os
import time
from datetime import datetime, timedelta

# Add the stock_predictor module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_basic_usage():
    """Demonstrate basic dashboard usage."""
    print("🎯 Stock Direction Predictor Dashboard Demo")
    print("=" * 50)
    
    print("\n📋 Basic Usage:")
    print("1. Launch the dashboard:")
    print("   python run_dashboard.py")
    print("   OR")
    print("   streamlit run streamlit_dashboard.py")
    
    print("\n2. Open your browser to: http://localhost:8501")
    
    print("\n3. Configure analysis in the sidebar:")
    print("   - Select stocks: AAPL, MSFT, NVDA, AMZN, META")
    print("   - Choose pattern lengths: 3, 5, 7, 14 days")
    print("   - Pick models: XGBoost, Random Forest, SVM, Neural Network")
    print("   - Set date range for analysis")
    
    print("\n4. Click '🚀 Run Analysis' to start")
    
    print("\n5. Explore results in different tabs:")
    print("   📊 Performance Overview - Model comparison charts")
    print("   📈 Backtesting Results - Portfolio performance")
    print("   🎯 Technical Analysis - Live indicators and signals")
    print("   📋 Detailed Metrics - Comprehensive performance data")
    print("   🔍 Model Insights - Prediction confidence and statistics")


def demo_features():
    """Demonstrate dashboard features."""
    print("\n🚀 Dashboard Features:")
    print("=" * 30)
    
    features = [
        ("📊 Interactive Charts", "Plotly-based charts with zoom, pan, and hover"),
        ("🎯 Real-time Data", "Live stock data and technical indicators"),
        ("📈 Backtesting", "Portfolio simulation with transaction costs"),
        ("🔍 Model Comparison", "Statistical comparison of ML models"),
        ("📋 Performance Metrics", "ROI, Sharpe ratio, max drawdown, etc."),
        ("🎨 Responsive Design", "Works on desktop and mobile devices"),
        ("💾 Data Caching", "Efficient caching for faster performance"),
        ("⚡ Auto-refresh", "Configurable real-time updates")
    ]
    
    for feature, description in features:
        print(f"  {feature}: {description}")


def demo_sample_analysis():
    """Show sample analysis workflow."""
    print("\n🧪 Sample Analysis Workflow:")
    print("=" * 35)
    
    print("\n1. Data Collection:")
    print("   - Fetch OHLC data from Yahoo Finance")
    print("   - Validate data completeness")
    print("   - Handle missing values")
    
    print("\n2. Feature Engineering:")
    print("   - Calculate technical indicators (RSI, MACD, EMA)")
    print("   - Generate candlestick patterns")
    print("   - Detect chart patterns")
    
    print("\n3. Model Training:")
    print("   - Train multiple ML models")
    print("   - Test different pattern lengths")
    print("   - Perform cross-validation")
    
    print("\n4. Performance Evaluation:")
    print("   - Run backtesting simulation")
    print("   - Calculate financial metrics")
    print("   - Statistical significance testing")
    
    print("\n5. Visualization:")
    print("   - Interactive performance charts")
    print("   - Risk-return analysis")
    print("   - Technical indicator plots")


def demo_configuration():
    """Show configuration options."""
    print("\n⚙️ Configuration Options:")
    print("=" * 30)
    
    print("\n📁 Files:")
    print("  - dashboard_config.yaml: Main configuration")
    print("  - streamlit_requirements.txt: Python dependencies")
    print("  - config.yaml: Stock predictor configuration")
    
    print("\n🎛️ Customizable Settings:")
    print("  - Default stock symbols")
    print("  - Pattern lengths to analyze")
    print("  - Model types to compare")
    print("  - Chart themes and colors")
    print("  - Real-time refresh intervals")
    print("  - Performance thresholds")


def demo_troubleshooting():
    """Show troubleshooting tips."""
    print("\n🔧 Troubleshooting Tips:")
    print("=" * 25)
    
    issues = [
        ("Dashboard won't start", "Check Python version (3.8+) and install requirements"),
        ("Import errors", "Run: pip install -r streamlit_requirements.txt"),
        ("Data fetching fails", "Check internet connection and stock symbols"),
        ("Analysis is slow", "Reduce number of symbols/models/patterns"),
        ("Charts not loading", "Clear browser cache and refresh page"),
        ("Port already in use", "Use different port: streamlit run --server.port 8502")
    ]
    
    for issue, solution in issues:
        print(f"  ❓ {issue}")
        print(f"     💡 {solution}")


def demo_advanced_features():
    """Show advanced features."""
    print("\n🚀 Advanced Features:")
    print("=" * 25)
    
    print("\n📊 Statistical Analysis:")
    print("  - Friedman test for model comparison")
    print("  - Mann-Whitney U test for pairwise comparison")
    print("  - Confidence intervals and p-values")
    
    print("\n🎯 Prediction Confidence:")
    print("  - Model prediction confidence scores")
    print("  - Signal strength indicators")
    print("  - Uncertainty quantification")
    
    print("\n⚡ Real-time Features:")
    print("  - Live data updates")
    print("  - Auto-refresh capabilities")
    print("  - Real-time signal generation")
    
    print("\n📈 Advanced Visualizations:")
    print("  - Heatmaps for model-pattern comparison")
    print("  - Radar charts for multi-dimensional analysis")
    print("  - Interactive candlestick charts")
    print("  - Risk-return scatter plots")


def interactive_demo():
    """Run an interactive demo."""
    print("\n🎮 Interactive Demo")
    print("=" * 20)
    
    print("\nWould you like to:")
    print("1. Launch the dashboard now")
    print("2. Run component tests")
    print("3. View sample configuration")
    print("4. Exit demo")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🚀 Launching dashboard...")
            print("Opening in your default browser...")
            os.system("python run_dashboard.py")
            
        elif choice == "2":
            print("\n🧪 Running component tests...")
            os.system("python test_dashboard.py")
            
        elif choice == "3":
            print("\n📋 Sample Configuration:")
            try:
                with open("dashboard_config.yaml", "r") as f:
                    print(f.read())
            except FileNotFoundError:
                print("Configuration file not found.")
                
        elif choice == "4":
            print("\n👋 Thanks for trying the demo!")
            
        else:
            print("\n❌ Invalid choice. Please enter 1-4.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    """Main demo function."""
    print("🎯 Stock Direction Predictor Dashboard")
    print("🎮 Interactive Demo & Tutorial")
    print("=" * 60)
    
    # Show basic information
    demo_basic_usage()
    demo_features()
    demo_sample_analysis()
    demo_configuration()
    demo_troubleshooting()
    demo_advanced_features()
    
    # Interactive section
    interactive_demo()


if __name__ == "__main__":
    main()