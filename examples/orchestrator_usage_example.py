#!/usr/bin/env python3
"""
Example usage of the Stock Direction Predictor Orchestrator.
Demonstrates different analysis modes and configuration options.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import stock_predictor
sys.path.insert(0, str(Path(__file__).parent.parent))

from stock_predictor.main import StockPredictorOrchestrator


def example_single_symbol_analysis():
    """Example: Analyze a single symbol with default configuration."""
    print("=== Single Symbol Analysis Example ===")
    
    # Initialize orchestrator
    orchestrator = StockPredictorOrchestrator()
    orchestrator.initialize()
    
    # Run analysis for a single symbol
    results = orchestrator.run_full_analysis(
        symbols=["AAPL"],
        pattern_lengths=[3, 5],  # Test only 3 and 5-day patterns
        model_types=["xgboost", "random_forest"]  # Test only 2 model types
    )
    
    # Display summary
    if results['performance_report'] and 'best_configuration' in results['performance_report']:
        best = results['performance_report']['best_configuration']
        print(f"Best configuration: {best.get('model_type')} with {best.get('pattern_length')}-day patterns")
        print(f"Total return: {best.get('financial_metrics', {}).get('total_return', 0):.2%}")
        print(f"Sharpe ratio: {best.get('financial_metrics', {}).get('sharpe_ratio', 0):.3f}")
    
    return results


def example_comparison_analysis():
    """Example: Run comparison analysis across multiple symbols."""
    print("\n=== Comparison Analysis Example ===")
    
    # Initialize orchestrator
    orchestrator = StockPredictorOrchestrator()
    orchestrator.initialize()
    
    # Run comparison analysis
    results = orchestrator.run_comparison_analysis(
        symbols=["AAPL", "MSFT"]
    )
    
    # Display comparison insights
    if 'pattern_length_comparison' in results:
        print("Pattern Length Performance:")
        for pattern_length, data in results['pattern_length_comparison'].items():
            print(f"  {pattern_length}-day patterns: Avg Return = {data.get('avg_return', 0):.2%}")
    
    if 'model_type_comparison' in results:
        print("Model Type Performance:")
        for model_type, data in results['model_type_comparison'].items():
            print(f"  {model_type}: Avg Return = {data.get('avg_return', 0):.2%}")
    
    return results


def example_batch_analysis():
    """Example: Run batch analysis with custom configuration."""
    print("\n=== Batch Analysis Example ===")
    
    # Initialize orchestrator
    orchestrator = StockPredictorOrchestrator()
    orchestrator.initialize()
    
    # Define batch configuration
    batch_config = {
        "symbol_groups": [
            ["AAPL"],
            ["MSFT"]
        ],
        "time_periods": [
            {
                "start": "2023-01-01",
                "end": "2024-01-01"
            }
        ]
    }
    
    # Run batch analysis
    results = orchestrator.run_batch_analysis(batch_config)
    
    # Display batch results summary
    print(f"Processed {len(results['results'])} batches")
    for i, batch_result in enumerate(results['results']):
        if 'error' not in batch_result:
            print(f"Batch {i+1}: {len(batch_result.get('aggregated_results', []))} model configurations tested")
    
    return results


def main():
    """Run all examples."""
    try:
        # Note: These examples require actual data and may take time to run
        # Uncomment the examples you want to test
        
        print("Stock Direction Predictor Orchestrator Examples")
        print("=" * 50)
        
        # Example 1: Single symbol analysis (fastest)
        # results1 = example_single_symbol_analysis()
        
        # Example 2: Comparison analysis (moderate time)
        # results2 = example_comparison_analysis()
        
        # Example 3: Batch analysis (longer time)
        # results3 = example_batch_analysis()
        
        print("\nExamples completed successfully!")
        print("Uncomment the example functions in main() to run them.")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())