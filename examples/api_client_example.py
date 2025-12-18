#!/usr/bin/env python3
"""
Example client for the VectorBT Visualization API.

This script demonstrates how to use the REST API endpoints for
generating visualizations programmatically.
"""

import requests
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import time
from pathlib import Path


class VectorBTVisualizationClient:
    """
    Client for interacting with the VectorBT Visualization API.
    
    This client provides a convenient interface for making API requests
    to generate and manage visualizations.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        
        # Setup session with authentication
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            })
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def create_portfolio_visualization(
        self,
        predictions: List[float],
        price_data: Dict[str, List[float]],
        test_start_idx: int,
        symbol: str = "ASSET",
        title: str = None,
        portfolio_config: Dict[str, Any] = None,
        plot_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a portfolio performance visualization.
        
        Args:
            predictions: ML model predictions array
            price_data: Historical price data dictionary
            test_start_idx: Index where test period begins
            symbol: Asset symbol for labeling
            title: Optional plot title
            portfolio_config: Portfolio configuration parameters
            plot_config: Plot configuration parameters
            
        Returns:
            API response with plot ID and metadata
        """
        payload = {
            "predictions": predictions,
            "price_data": price_data,
            "test_start_idx": test_start_idx,
            "symbol": symbol,
            "title": title,
            "portfolio_config": portfolio_config,
            "plot_config": plot_config
        }
        
        response = self.session.post(
            f"{self.base_url}/api/v1/visualizations/portfolio",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def create_drawdown_visualization(
        self,
        predictions: List[float],
        price_data: Dict[str, List[float]],
        test_start_idx: int,
        symbol: str = "ASSET",
        portfolio_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a drawdown analysis visualization.
        
        Args:
            predictions: ML model predictions array
            price_data: Historical price data dictionary
            test_start_idx: Index where test period begins
            symbol: Asset symbol for labeling
            portfolio_config: Portfolio configuration parameters
            
        Returns:
            API response with plot ID and metadata
        """
        payload = {
            "predictions": predictions,
            "price_data": price_data,
            "test_start_idx": test_start_idx,
            "symbol": symbol,
            "portfolio_config": portfolio_config
        }
        
        response = self.session.post(
            f"{self.base_url}/api/v1/visualizations/drawdown",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def create_comparison_visualization(
        self,
        portfolios_data: Dict[str, Dict[str, Any]],
        title: str = "Multi-Strategy Comparison"
    ) -> Dict[str, Any]:
        """
        Create a multi-strategy comparison visualization.
        
        Args:
            portfolios_data: Dictionary of portfolio data for comparison
            title: Plot title
            
        Returns:
            API response with plot ID and metadata
        """
        payload = {
            "portfolios_data": portfolios_data,
            "title": title
        }
        
        response = self.session.post(
            f"{self.base_url}/api/v1/visualizations/comparison",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_plot_data(self, plot_id: str) -> Dict[str, Any]:
        """
        Retrieve plot data by ID.
        
        Args:
            plot_id: Plot identifier
            
        Returns:
            Plot data and metadata
        """
        response = self.session.get(f"{self.base_url}/api/v1/plots/{plot_id}")
        response.raise_for_status()
        return response.json()
    
    def get_plot_html(self, plot_id: str) -> str:
        """
        Retrieve plot as HTML.
        
        Args:
            plot_id: Plot identifier
            
        Returns:
            HTML content of the plot
        """
        response = self.session.get(f"{self.base_url}/api/v1/plots/{plot_id}/html")
        response.raise_for_status()
        return response.json()["html"]
    
    def export_plot(
        self,
        plot_id: str,
        formats: List[str] = ["png", "html"],
        include_data: bool = False
    ) -> Dict[str, Any]:
        """
        Export plot in specified formats.
        
        Args:
            plot_id: Plot identifier
            formats: List of export formats
            include_data: Whether to include underlying data
            
        Returns:
            Export response with download URLs
        """
        payload = {
            "plot_id": plot_id,
            "formats": formats,
            "include_data": include_data
        }
        
        response = self.session.post(
            f"{self.base_url}/api/v1/export",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def download_file(self, filename: str, save_path: str = None) -> str:
        """
        Download exported file.
        
        Args:
            filename: Name of file to download
            save_path: Local path to save file (optional)
            
        Returns:
            Path to downloaded file
        """
        response = self.session.get(f"{self.base_url}/api/v1/download/{filename}")
        response.raise_for_status()
        
        if save_path is None:
            save_path = filename
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        return save_path
    
    def list_plots(self) -> List[Dict[str, Any]]:
        """
        List all available plots.
        
        Returns:
            List of plot metadata
        """
        response = self.session.get(f"{self.base_url}/api/v1/plots")
        response.raise_for_status()
        return response.json()["plots"]
    
    def delete_plot(self, plot_id: str) -> Dict[str, str]:
        """
        Delete a plot.
        
        Args:
            plot_id: Plot identifier
            
        Returns:
            Deletion confirmation
        """
        response = self.session.delete(f"{self.base_url}/api/v1/plots/{plot_id}")
        response.raise_for_status()
        return response.json()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics.
        
        Returns:
            Usage statistics for current user
        """
        response = self.session.get(f"{self.base_url}/api/v1/usage")
        response.raise_for_status()
        return response.json()


def generate_sample_data():
    """Generate sample data for demonstration."""
    # Generate sample price data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Simulate stock price with trend and volatility
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = [100.0]  # Starting price
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLC data
    price_data = {
        'Open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(1000000, 5000000) for _ in prices]
    }
    
    # Generate sample predictions (0=sell, 1=hold, 2=buy)
    test_start_idx = len(prices) // 2  # Start predictions halfway through
    test_length = len(prices) - test_start_idx
    
    # Simulate ML predictions with some pattern
    predictions = []
    for i in range(test_length):
        # Simple momentum-based prediction
        if i > 5:
            recent_change = (prices[test_start_idx + i] - prices[test_start_idx + i - 5]) / prices[test_start_idx + i - 5]
            if recent_change > 0.02:
                pred = 2  # Buy
            elif recent_change < -0.02:
                pred = 0  # Sell
            else:
                pred = 1  # Hold
        else:
            pred = 1  # Hold for first few predictions
        
        # Add some noise
        if np.random.random() < 0.1:  # 10% noise
            pred = np.random.choice([0, 1, 2])
        
        predictions.append(pred)
    
    return price_data, predictions, test_start_idx


def demo_basic_usage():
    """Demonstrate basic API usage."""
    print("🚀 VectorBT Visualization API Client Demo")
    print("=" * 50)
    
    # Initialize client
    client = VectorBTVisualizationClient(
        base_url="http://localhost:8000",
        api_key="demo_key_123"  # Use demo API key
    )
    
    try:
        # Health check
        print("1. Checking API health...")
        health = client.health_check()
        print(f"   ✅ API is healthy: {health['status']}")
        
        # Generate sample data
        print("\n2. Generating sample data...")
        price_data, predictions, test_start_idx = generate_sample_data()
        print(f"   📊 Generated {len(predictions)} predictions for {len(price_data['Close'])} price points")
        
        # Create portfolio visualization
        print("\n3. Creating portfolio visualization...")
        portfolio_result = client.create_portfolio_visualization(
            predictions=predictions,
            price_data=price_data,
            test_start_idx=test_start_idx,
            symbol="DEMO",
            title="Demo Portfolio Performance",
            portfolio_config={
                "init_cash": 10000,
                "fees": 0.001,
                "size_value": 50
            }
        )
        
        portfolio_plot_id = portfolio_result["plot_id"]
        print(f"   📈 Portfolio plot created: {portfolio_plot_id}")
        print(f"   ⏱️  Generation time: {portfolio_result['generation_time']:.2f}s")
        
        # Create drawdown visualization
        print("\n4. Creating drawdown visualization...")
        drawdown_result = client.create_drawdown_visualization(
            predictions=predictions,
            price_data=price_data,
            test_start_idx=test_start_idx,
            symbol="DEMO"
        )
        
        drawdown_plot_id = drawdown_result["plot_id"]
        print(f"   📉 Drawdown plot created: {drawdown_plot_id}")
        
        # Export plots
        print("\n5. Exporting plots...")
        export_result = client.export_plot(
            plot_id=portfolio_plot_id,
            formats=["png", "html", "json"],
            include_data=True
        )
        
        if export_result["success"]:
            print("   💾 Export successful:")
            for format_type, url in export_result["download_urls"].items():
                print(f"      - {format_type.upper()}: {url}")
        
        # List all plots
        print("\n6. Listing all plots...")
        plots = client.list_plots()
        print(f"   📋 Found {len(plots)} plots:")
        for plot in plots:
            print(f"      - {plot['plot_id']}: {plot['type']} ({plot['created_at']})")
        
        # Get usage statistics
        print("\n7. Checking usage statistics...")
        usage = client.get_usage_stats()
        print(f"   📊 User: {usage['user']}")
        print(f"   🔢 Rate limit: {usage['rate_limit']} requests/hour")
        print(f"   📈 Requests this hour: {usage['requests_this_hour']}")
        
        print("\n✅ Demo completed successfully!")
        print(f"\n🌐 View plots in browser:")
        print(f"   - Portfolio: http://localhost:8000/api/v1/plots/{portfolio_plot_id}/html")
        print(f"   - Drawdown: http://localhost:8000/api/v1/plots/{drawdown_plot_id}/html")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server.")
        print("   Make sure the server is running: python run_visualization_api.py")
        
    except requests.exceptions.HTTPError as e:
        print(f"❌ API request failed: {e}")
        if e.response.status_code == 401:
            print("   Check your API key")
        elif e.response.status_code == 429:
            print("   Rate limit exceeded")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def demo_comparison_visualization():
    """Demonstrate comparison visualization."""
    print("\n🔄 Comparison Visualization Demo")
    print("=" * 40)
    
    client = VectorBTVisualizationClient(
        base_url="http://localhost:8000",
        api_key="demo_key_123"
    )
    
    try:
        # Generate data for multiple strategies
        price_data, _, test_start_idx = generate_sample_data()
        
        # Create different prediction strategies
        strategies = {
            "Conservative": [1] * (len(price_data['Close']) - test_start_idx),  # All hold
            "Aggressive": [2 if i % 3 == 0 else 0 for i in range(len(price_data['Close']) - test_start_idx)],  # Buy/sell pattern
            "Random": [np.random.choice([0, 1, 2]) for _ in range(len(price_data['Close']) - test_start_idx)]
        }
        
        # Prepare portfolios data for comparison
        portfolios_data = {}
        for strategy_name, predictions in strategies.items():
            portfolios_data[strategy_name] = {
                "predictions": predictions,
                "price_data": price_data,
                "test_start_idx": test_start_idx,
                "symbol": "DEMO",
                "portfolio_config": {
                    "init_cash": 10000,
                    "fees": 0.001,
                    "size_value": 40
                }
            }
        
        # Create comparison visualization
        print(f"Creating comparison for {len(strategies)} strategies...")
        comparison_result = client.create_comparison_visualization(
            portfolios_data=portfolios_data,
            title="Strategy Comparison Demo"
        )
        
        comparison_plot_id = comparison_result["plot_id"]
        print(f"📊 Comparison plot created: {comparison_plot_id}")
        print(f"🌐 View at: http://localhost:8000/api/v1/plots/{comparison_plot_id}/html")
        
    except Exception as e:
        print(f"❌ Error in comparison demo: {e}")


if __name__ == "__main__":
    # Run the demos
    demo_basic_usage()
    demo_comparison_visualization()
    
    print("\n" + "=" * 60)
    print("📚 For more information, visit the API documentation:")
    print("   - Swagger UI: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print("=" * 60)