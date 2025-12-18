#!/usr/bin/env python3
"""
Launch script for the Stock Direction Predictor Streamlit Dashboard.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'yfinance',
        'scikit-learn',
        'xgboost'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n💡 Install missing packages with:")
        print("pip install -r streamlit_requirements.txt")
        return False
    
    return True


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "streamlit_dashboard.py"
    
    if not dashboard_path.exists():
        print("❌ Dashboard file not found: streamlit_dashboard.py")
        return False
    
    print("🚀 Launching Stock Direction Predictor Dashboard...")
    print("📊 Dashboard will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\n⚠️  Note: First run may take a few minutes to initialize all components")
    print("📝 Check the terminal for any initialization messages or errors")
    print("\n🛑 Press Ctrl+C to stop the dashboard")
    print("-" * 60)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    
    return True


def main():
    """Main entry point."""
    print("=" * 60)
    print("📈 Stock Direction Predictor Dashboard Launcher")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Launch dashboard
    if not launch_dashboard():
        sys.exit(1)


if __name__ == "__main__":
    main()