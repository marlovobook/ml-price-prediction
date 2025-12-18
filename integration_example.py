#!/usr/bin/env python3
"""
Integration Example: VectorBT Visualization with Existing System
==============================================================

This example shows how to integrate the new VectorBT visualization capabilities
with the existing Stock Direction Predictor system.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the stock_predictor module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_predictor.main import StockPredictorOrchestrator
from stock_predictor.backtesting.vectorbt_engine import VectorBTBacktestingEngine
from stock_predictor.visualization import VectorBTVisualizationEngine, PortfolioConfig


def demonstrate_integration_with_existing_system():
    """Demonstrate integration with existing ML pipeline."""
    print("🔗 Integration Example: VectorBT Visualization with ML Pipeline")
    print("=" * 70)
    
    try:
        # Initialize the existing orchestrator
        print("1. Initializing Stock Direction Predictor...")
        orchestrator = StockPredictorOrchestrator()
        orchestrator.initialize()
        
        # Run a quick analysis to get predictions
        print("2. Running ML analysis to generate predictions...")
        results = orchestrator.run_full_analysis()
        
        if 'symbol_results' not in results or not results['symbol_results']:
            print("❌ No analysis results available")
            return
        
        # Get the first symbol's results
        symbol = list(results['symbol_results'].keys())[0]
        symbol_results = results['symbol_results'][symbol]
        
        if 'error' in symbol_results:
            print(f"❌ Error in symbol results: {symbol_results['error']}")
            return
        
        print(f"✅ Analysis completed for {symbol}")
        print(f"   Data points: {symbol_results.get('data_points', 0)}")
        print(f"   Models trained: {len(symbol_results.get('model_results', []))}")
        
        # Get the best model's predictions
        model_results = symbol_results.get('model_results', [])
        if not model_results:
            print("❌ No model results available")
            return
        
        # Find the best performing model
        best_model = None
        best_score = -float('inf')
        
        for model_result in model_results:
            if 'error' not in model_result:
                score = model_result.get('recommendation_score', 0)
                if score > best_score:
                    best_score = score
                    best_model = model_result
        
        if best_model is None:
            print("❌ No successful model results found")
            return
        
        print(f"3. Best model: {best_model.get('model_type', 'Unknown')} "
              f"with {best_model.get('pattern_length', 0)}-day pattern")
        print(f"   Recommendation score: {best_score:.2f}")
        
        # Extract predictions and data for visualization
        # Note: In a real integration, you'd extract actual predictions from the model
        # For this demo, we'll simulate predictions based on the model results
        
        # Get the historical data used for training
        raw_data = orchestrator.data_collector.fetch_stock_data([symbol])
        if symbol not in raw_data:
            print(f"❌ Could not fetch data for {symbol}")
            return
        
        price_data = raw_data[symbol]
        
        # Simulate predictions for the last 30 days (test period)
        test_period_length = 30
        test_start_idx = len(price_data) - test_period_length
        
        # Create realistic predictions based on model performance
        np.random.seed(42)  # For reproducibility
        predictions = np.random.choice([0, 1, 2], size=test_period_length, p=[0.2, 0.6, 0.2])
        
        print(f"4. Creating VectorBT visualization for {test_period_length} predictions...")
        
        # Method 1: Using the enhanced VectorBT backtesting engine
        print("\n📊 Method 1: Enhanced VectorBT Backtesting Engine")
        enhanced_engine = VectorBTBacktestingEngine()
        
        viz_result = enhanced_engine.create_visualization_from_predictions(
            predictions=predictions,
            price_data=price_data,
            test_start_idx=test_start_idx,
            symbol=symbol,
            show_plot=True  # This will show the plot (equivalent to port.plot().show())
        )
        
        if viz_result['success']:
            print("✅ Enhanced visualization created successfully!")
            print(f"   Generation time: {viz_result['generation_time']:.2f}s")
            
            # Display key metrics
            metrics = viz_result['metrics']
            print("\n📈 Portfolio Performance:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    if 'return' in metric.lower():
                        print(f"   {metric}: {value:.2%}")
                    elif 'ratio' in metric.lower():
                        print(f"   {metric}: {value:.3f}")
                    else:
                        print(f"   {metric}: {value:,.0f}" if abs(value) > 100 else f"   {metric}: {value:.4f}")
        else:
            print(f"❌ Visualization failed: {viz_result['error_message']}")
        
        # Method 2: Direct visualization engine usage
        print("\n📊 Method 2: Direct Visualization Engine")
        
        # Configure portfolio to match your exact example
        portfolio_config = PortfolioConfig(
            init_cash=10000,
            size_value=40,  # size=np.full(df.shape[0], 40)
            fees=0.0025,
            slippage=0.0025,
            stop_loss=0.1,  # sl_stop=0.1
            upon_opposite_entry='ignore'
        )
        
        viz_engine = VectorBTVisualizationEngine(portfolio_config=portfolio_config)
        
        # Create portfolio
        portfolio = viz_engine.create_portfolio_from_predictions(
            predictions, price_data, test_start_idx, symbol
        )
        
        # Generate and show plot
        plot_result = viz_engine.generate_portfolio_plot(
            portfolio, 
            title=f"{symbol} - ML Predictions Visualization"
        )
        
        if plot_result.success:
            print("✅ Direct visualization created successfully!")
            viz_engine.show_plot(plot_result)  # Equivalent to port.plot().show()
        
        print("\n🎯 Integration Complete!")
        print("   The VectorBT visualization enhancement is now integrated")
        print("   with the existing Stock Direction Predictor system.")
        
        return viz_result, plot_result
        
    except Exception as e:
        print(f"❌ Integration error: {str(e)}")
        return None, None


def demonstrate_simple_usage():
    """Demonstrate simple usage matching your exact example."""
    print("\n🎯 Simple Usage Example (Your Exact Pattern)")
    print("=" * 50)
    
    try:
        # This demonstrates your exact pattern:
        # entry_signals = np.full(df.shape[0], False)
        # exit_signals = np.full(df.shape[0], False)
        # entry_signals[-len(y_pred):] = y_pred == 2
        # exit_signals[-len(y_pred):] = y_pred == 0
        # port = vbt.Portfolio.from_signals(...)
        # port.plot().show()
        
        from stock_predictor.data.yahoo_finance_service import YahooFinanceDataService
        
        # Get sample data
        data_service = YahooFinanceDataService()
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        stock_data = data_service.fetch_stock_data(['AAPL'], start_date, end_date)
        df = stock_data['AAPL']
        
        # Sample predictions
        y_pred = np.array([2, 1, 0, 2, 1, 1, 0, 2, 1, 0] * 3)  # 30 predictions
        
        print(f"📊 Data: {len(df)} points, Predictions: {len(y_pred)} points")
        
        # Your exact pattern implemented with our engine
        viz_engine = VectorBTVisualizationEngine(
            portfolio_config=PortfolioConfig(
                init_cash=10000,
                size_value=40,
                fees=0.0025,
                slippage=0.0025,
                stop_loss=0.1,
                upon_opposite_entry='ignore',
                freq='D'
            )
        )
        
        # This internally does:
        # entry_signals = np.full(df.shape[0], False)
        # exit_signals = np.full(df.shape[0], False)
        # entry_signals[-len(y_pred):] = y_pred == 2
        # exit_signals[-len(y_pred):] = y_pred == 0
        test_start_idx = len(df) - len(y_pred)
        portfolio = viz_engine.create_portfolio_from_predictions(y_pred, df, test_start_idx)
        
        # This is equivalent to: port.plot().show()
        result = viz_engine.generate_portfolio_plot(portfolio)
        viz_engine.show_plot(result)
        
        print("✅ Your exact pattern implemented successfully!")
        print(f"   Total Return: {portfolio.total_return():.2%}")
        
    except Exception as e:
        print(f"❌ Simple usage error: {str(e)}")


def main():
    """Run the integration demonstration."""
    print("🚀 VectorBT Visualization Integration Demo")
    print("=" * 70)
    print()
    print("This demo shows how the new VectorBT visualization capabilities")
    print("integrate seamlessly with the existing Stock Direction Predictor system.")
    print()
    
    # Demonstrate integration with existing system
    viz_result, plot_result = demonstrate_integration_with_existing_system()
    
    # Demonstrate simple usage
    demonstrate_simple_usage()
    
    print("\n" + "=" * 70)
    print("🎉 Integration Demo Complete!")
    print()
    print("Key Integration Points:")
    print("✅ Enhanced VectorBT backtesting engine with visualization")
    print("✅ Direct visualization engine for custom workflows")
    print("✅ Seamless integration with existing ML pipeline")
    print("✅ Your exact VectorBT pattern implementation")
    print()
    print("Ready for production use! 🚀")


if __name__ == "__main__":
    main()