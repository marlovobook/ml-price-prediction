"""
VectorBT Visualization Enhancement Module for Stock Direction Predictor.

This module provides comprehensive visualization capabilities using VectorBT's
built-in plotting functionality for portfolio analysis and trading strategy evaluation.
"""

from .signal_alignment import SignalAlignmentEngine, AlignedSignals
from .portfolio_config import PortfolioConfig, PlotConfig, VisualizationResult
from .visualization_engine import VectorBTVisualizationEngine
from .enhanced_portfolio_engine import EnhancedPortfolioEngine, PortfolioCreationResult
from .export_engine import PlotExportEngine
from .config_manager import ConfigurationManager, ConfigurationValidator

__all__ = [
    'SignalAlignmentEngine',
    'AlignedSignals', 
    'PortfolioConfig',
    'PlotConfig',
    'VectorBTVisualizationEngine',
    'EnhancedPortfolioEngine',
    'PortfolioCreationResult',
    'VisualizationResult',
    'PlotExportEngine',
    'ConfigurationManager',
    'ConfigurationValidator'
]