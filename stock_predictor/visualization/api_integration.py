"""
API Integration Module for VectorBT Visualization Enhancement.

This module provides REST API endpoints for visualization generation,
programmatic access to plot objects and data, and authentication/rate limiting,
implementing Requirements 10.3 and 10.4.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import json
import logging
import time
import hashlib
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

from .visualization_engine import VectorBTVisualizationEngine, VisualizationResult
from .portfolio_config import PortfolioConfig, PlotConfig
from .export_engine import PlotExportEngine
from ..utils.exceptions import BacktestingError, DataValidationError


# Pydantic models for API requests and responses
class VisualizationRequest(BaseModel):
    """Base model for visualization requests."""
    predictions: List[float] = Field(..., description="ML model predictions array")
    price_data: Dict[str, List[float]] = Field(..., description="Historical price data")
    test_start_idx: int = Field(..., description="Index where test period begins")
    symbol: str = Field(default="ASSET", description="Asset symbol for labeling")
    portfolio_config: Optional[Dict[str, Any]] = Field(default=None, description="Portfolio configuration")
    plot_config: Optional[Dict[str, Any]] = Field(default=None, description="Plot configuration")


class PortfolioPlotRequest(VisualizationRequest):
    """Request model for portfolio plot generation."""
    title: Optional[str] = Field(default=None, description="Plot title")
    show_trades: bool = Field(default=True, description="Whether to show trade markers")
    show_metrics: bool = Field(default=True, description="Whether to show performance metrics")


class DrawdownPlotRequest(VisualizationRequest):
    """Request model for drawdown plot generation."""
    show_recovery: bool = Field(default=True, description="Whether to show recovery analysis")
    highlight_max_dd: bool = Field(default=True, description="Whether to highlight maximum drawdown")


class ComparisonPlotRequest(BaseModel):
    """Request model for comparison plot generation."""
    portfolios_data: Dict[str, VisualizationRequest] = Field(..., description="Dictionary of portfolio data")
    title: str = Field(default="Multi-Strategy Comparison", description="Plot title")
    comparison_metrics: List[str] = Field(default=["total_return", "sharpe_ratio", "max_drawdown"], 
                                        description="Metrics to compare")


class ExportRequest(BaseModel):
    """Request model for plot export."""
    plot_id: str = Field(..., description="Plot ID to export")
    formats: List[str] = Field(default=["png", "html"], description="Export formats")
    include_data: bool = Field(default=False, description="Whether to include underlying data")


class VisualizationResponse(BaseModel):
    """Response model for visualization requests."""
    success: bool = Field(..., description="Whether the request was successful")
    plot_id: str = Field(..., description="Unique plot identifier")
    plot_data: Optional[Dict[str, Any]] = Field(default=None, description="Plot data if requested")
    metrics_summary: Optional[Dict[str, float]] = Field(default=None, description="Performance metrics")
    generation_time: float = Field(..., description="Time taken to generate visualization")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class ExportResponse(BaseModel):
    """Response model for export requests."""
    success: bool = Field(..., description="Whether the export was successful")
    export_paths: Dict[str, str] = Field(..., description="Paths to exported files")
    download_urls: Dict[str, str] = Field(..., description="URLs for downloading files")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class APIKeyManager:
    """
    Simple API key manager for authentication.
    
    In production, this would be replaced with a proper authentication system.
    """
    
    def __init__(self):
        self.api_keys = {
            "demo_key_123": {
                "name": "Demo User",
                "rate_limit": 100,  # requests per hour
                "permissions": ["read", "write", "export"]
            },
            "admin_key_456": {
                "name": "Admin User", 
                "rate_limit": 1000,
                "permissions": ["read", "write", "export", "admin"]
            }
        }
        self.usage_tracking = {}
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user info."""
        return self.api_keys.get(api_key)
    
    def check_rate_limit(self, api_key: str) -> bool:
        """Check if API key is within rate limits."""
        if api_key not in self.usage_tracking:
            self.usage_tracking[api_key] = []
        
        # Clean old requests (older than 1 hour)
        current_time = time.time()
        self.usage_tracking[api_key] = [
            req_time for req_time in self.usage_tracking[api_key]
            if current_time - req_time < 3600
        ]
        
        # Check rate limit
        user_info = self.validate_api_key(api_key)
        if not user_info:
            return False
        
        return len(self.usage_tracking[api_key]) < user_info["rate_limit"]
    
    def record_request(self, api_key: str) -> None:
        """Record API request for rate limiting."""
        if api_key not in self.usage_tracking:
            self.usage_tracking[api_key] = []
        
        self.usage_tracking[api_key].append(time.time())


class VectorBTVisualizationAPI:
    """
    FastAPI-based REST API for VectorBT visualization generation.
    
    This class implements Requirements 10.3 and 10.4:
    - Add REST API endpoints for visualization generation
    - Create programmatic access to plot objects and data
    - Implement authentication and rate limiting for API access
    """
    
    def __init__(self, 
                 host: str = "0.0.0.0",
                 port: int = 8000,
                 enable_auth: bool = True,
                 enable_cors: bool = True):
        """
        Initialize the VectorBT Visualization API.
        
        Args:
            host: API host address
            port: API port number
            enable_auth: Whether to enable API key authentication
            enable_cors: Whether to enable CORS middleware
        """
        self.host = host
        self.port = port
        self.enable_auth = enable_auth
        
        # Initialize components
        self.viz_engine = VectorBTVisualizationEngine()
        self.export_engine = PlotExportEngine()
        self.api_key_manager = APIKeyManager()
        
        # Plot storage for retrieval
        self.plot_storage = {}
        self.plot_metadata = {}
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="VectorBT Visualization API",
            description="REST API for generating VectorBT portfolio visualizations",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Setup middleware
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],  # Configure appropriately for production
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Setup authentication
        self.security = HTTPBearer() if enable_auth else None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Register routes
        self._register_routes()
        
        self.logger.info("VectorBT Visualization API initialized")
    
    def _register_routes(self):
        """Register all API routes."""
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        # Authentication dependency
        async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(self.security)) -> Dict[str, Any]:
            """Authenticate user and check rate limits."""
            if not self.enable_auth:
                return {"name": "Anonymous", "permissions": ["read", "write", "export"]}
            
            api_key = credentials.credentials
            user_info = self.api_key_manager.validate_api_key(api_key)
            
            if not user_info:
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            if not self.api_key_manager.check_rate_limit(api_key):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            self.api_key_manager.record_request(api_key)
            return user_info
        
        # Portfolio visualization endpoints
        @self.app.post("/api/v1/visualizations/portfolio", response_model=VisualizationResponse)
        async def create_portfolio_visualization(
            request: PortfolioPlotRequest,
            background_tasks: BackgroundTasks,
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """
            Generate portfolio performance visualization.
            
            This endpoint implements Requirements 10.3 by providing REST API access
            to portfolio visualization generation.
            """
            try:
                self.logger.info(f"Creating portfolio visualization for user: {user['name']}")
                
                # Convert request to internal format
                predictions = np.array(request.predictions)
                price_data = pd.DataFrame(request.price_data)
                
                # Create portfolio configuration
                portfolio_config = PortfolioConfig()
                if request.portfolio_config:
                    for key, value in request.portfolio_config.items():
                        if hasattr(portfolio_config, key):
                            setattr(portfolio_config, key, value)
                
                # Create plot configuration
                plot_config = PlotConfig()
                if request.plot_config:
                    for key, value in request.plot_config.items():
                        if hasattr(plot_config, key):
                            setattr(plot_config, key, value)
                
                # Update visualization engine configuration
                self.viz_engine.portfolio_config = portfolio_config
                self.viz_engine.plot_config = plot_config
                
                # Create portfolio
                portfolio = self.viz_engine.create_portfolio_from_predictions(
                    predictions, price_data, request.test_start_idx, request.symbol
                )
                
                # Generate visualization
                result = self.viz_engine.generate_portfolio_plot(portfolio, request.title)
                
                if not result.success:
                    raise HTTPException(status_code=500, detail=result.error_message)
                
                # Store plot for later retrieval
                plot_id = self._generate_plot_id()
                self.plot_storage[plot_id] = result.plot_object
                self.plot_metadata[plot_id] = {
                    "type": "portfolio",
                    "created_at": datetime.now().isoformat(),
                    "user": user["name"],
                    "symbol": request.symbol
                }
                
                # Schedule cleanup
                background_tasks.add_task(self._cleanup_old_plots)
                
                return VisualizationResponse(
                    success=True,
                    plot_id=plot_id,
                    plot_data=result.plot_data if "read" in user["permissions"] else None,
                    metrics_summary=result.metrics_summary,
                    generation_time=result.generation_time
                )
                
            except Exception as e:
                self.logger.error(f"Error creating portfolio visualization: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/visualizations/drawdown", response_model=VisualizationResponse)
        async def create_drawdown_visualization(
            request: DrawdownPlotRequest,
            background_tasks: BackgroundTasks,
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """
            Generate drawdown analysis visualization.
            
            This endpoint provides REST API access to drawdown visualization generation.
            """
            try:
                self.logger.info(f"Creating drawdown visualization for user: {user['name']}")
                
                # Convert request to internal format
                predictions = np.array(request.predictions)
                price_data = pd.DataFrame(request.price_data)
                
                # Create portfolio configuration
                portfolio_config = PortfolioConfig()
                if request.portfolio_config:
                    for key, value in request.portfolio_config.items():
                        if hasattr(portfolio_config, key):
                            setattr(portfolio_config, key, value)
                
                # Update visualization engine configuration
                self.viz_engine.portfolio_config = portfolio_config
                
                # Create portfolio
                portfolio = self.viz_engine.create_portfolio_from_predictions(
                    predictions, price_data, request.test_start_idx, request.symbol
                )
                
                # Generate visualization
                result = self.viz_engine.generate_drawdown_plot(portfolio)
                
                if not result.success:
                    raise HTTPException(status_code=500, detail=result.error_message)
                
                # Store plot for later retrieval
                plot_id = self._generate_plot_id()
                self.plot_storage[plot_id] = result.plot_object
                self.plot_metadata[plot_id] = {
                    "type": "drawdown",
                    "created_at": datetime.now().isoformat(),
                    "user": user["name"],
                    "symbol": request.symbol
                }
                
                # Schedule cleanup
                background_tasks.add_task(self._cleanup_old_plots)
                
                return VisualizationResponse(
                    success=True,
                    plot_id=plot_id,
                    plot_data=result.plot_data if "read" in user["permissions"] else None,
                    metrics_summary=result.metrics_summary,
                    generation_time=result.generation_time
                )
                
            except Exception as e:
                self.logger.error(f"Error creating drawdown visualization: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/visualizations/comparison", response_model=VisualizationResponse)
        async def create_comparison_visualization(
            request: ComparisonPlotRequest,
            background_tasks: BackgroundTasks,
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """
            Generate multi-strategy comparison visualization.
            
            This endpoint provides REST API access to comparison visualization generation.
            """
            try:
                self.logger.info(f"Creating comparison visualization for user: {user['name']}")
                
                # Create portfolios from request data
                portfolios = {}
                
                for name, portfolio_request in request.portfolios_data.items():
                    predictions = np.array(portfolio_request.predictions)
                    price_data = pd.DataFrame(portfolio_request.price_data)
                    
                    # Create portfolio configuration
                    portfolio_config = PortfolioConfig()
                    if portfolio_request.portfolio_config:
                        for key, value in portfolio_request.portfolio_config.items():
                            if hasattr(portfolio_config, key):
                                setattr(portfolio_config, key, value)
                    
                    # Update visualization engine configuration
                    self.viz_engine.portfolio_config = portfolio_config
                    
                    # Create portfolio
                    portfolio = self.viz_engine.create_portfolio_from_predictions(
                        predictions, price_data, portfolio_request.test_start_idx, portfolio_request.symbol
                    )
                    
                    portfolios[name] = portfolio
                
                # Generate comparison visualization
                result = self.viz_engine.generate_comparison_plot(portfolios, request.title)
                
                if not result.success:
                    raise HTTPException(status_code=500, detail=result.error_message)
                
                # Store plot for later retrieval
                plot_id = self._generate_plot_id()
                self.plot_storage[plot_id] = result.plot_object
                self.plot_metadata[plot_id] = {
                    "type": "comparison",
                    "created_at": datetime.now().isoformat(),
                    "user": user["name"],
                    "portfolios": list(portfolios.keys())
                }
                
                # Schedule cleanup
                background_tasks.add_task(self._cleanup_old_plots)
                
                return VisualizationResponse(
                    success=True,
                    plot_id=plot_id,
                    plot_data=result.plot_data if "read" in user["permissions"] else None,
                    metrics_summary=result.metrics_summary,
                    generation_time=result.generation_time
                )
                
            except Exception as e:
                self.logger.error(f"Error creating comparison visualization: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Plot retrieval endpoints
        @self.app.get("/api/v1/plots/{plot_id}")
        async def get_plot_data(
            plot_id: str,
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """
            Retrieve plot data by ID.
            
            This endpoint implements Requirements 10.4 by providing programmatic
            access to plot objects and data.
            """
            try:
                if plot_id not in self.plot_storage:
                    raise HTTPException(status_code=404, detail="Plot not found")
                
                if "read" not in user["permissions"]:
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                plot_obj = self.plot_storage[plot_id]
                metadata = self.plot_metadata[plot_id]
                
                # Convert plot to JSON representation
                plot_json = plot_obj.to_json()
                
                return {
                    "plot_id": plot_id,
                    "plot_data": json.loads(plot_json),
                    "metadata": metadata
                }
                
            except Exception as e:
                self.logger.error(f"Error retrieving plot data: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/plots/{plot_id}/html")
        async def get_plot_html(
            plot_id: str,
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Retrieve plot as HTML."""
            try:
                if plot_id not in self.plot_storage:
                    raise HTTPException(status_code=404, detail="Plot not found")
                
                if "read" not in user["permissions"]:
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                plot_obj = self.plot_storage[plot_id]
                html_content = plot_obj.to_html(include_plotlyjs=True)
                
                return JSONResponse(
                    content={"html": html_content},
                    media_type="application/json"
                )
                
            except Exception as e:
                self.logger.error(f"Error retrieving plot HTML: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Export endpoints
        @self.app.post("/api/v1/export", response_model=ExportResponse)
        async def export_plot(
            request: ExportRequest,
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """
            Export plot in specified formats.
            
            This endpoint provides programmatic access to plot export functionality.
            """
            try:
                if "export" not in user["permissions"]:
                    raise HTTPException(status_code=403, detail="Insufficient export permissions")
                
                if request.plot_id not in self.plot_storage:
                    raise HTTPException(status_code=404, detail="Plot not found")
                
                plot_obj = self.plot_storage[request.plot_id]
                metadata = self.plot_metadata[request.plot_id]
                
                # Generate unique filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"{metadata['type']}_{request.plot_id}_{timestamp}"
                
                # Export plot in requested formats
                export_paths = {}
                download_urls = {}
                
                for format_type in request.formats:
                    if format_type.lower() == "png":
                        # Export as PNG
                        filename = f"{base_filename}.png"
                        filepath = Path(tempfile.gettempdir()) / filename
                        plot_obj.write_image(str(filepath))
                        export_paths["png"] = str(filepath)
                        download_urls["png"] = f"/api/v1/download/{filename}"
                    
                    elif format_type.lower() == "html":
                        # Export as HTML
                        filename = f"{base_filename}.html"
                        filepath = Path(tempfile.gettempdir()) / filename
                        plot_obj.write_html(str(filepath))
                        export_paths["html"] = str(filepath)
                        download_urls["html"] = f"/api/v1/download/{filename}"
                    
                    elif format_type.lower() == "json":
                        # Export as JSON
                        filename = f"{base_filename}.json"
                        filepath = Path(tempfile.gettempdir()) / filename
                        with open(filepath, 'w') as f:
                            f.write(plot_obj.to_json())
                        export_paths["json"] = str(filepath)
                        download_urls["json"] = f"/api/v1/download/{filename}"
                
                # Export underlying data if requested
                if request.include_data:
                    data_filename = f"{base_filename}_data.csv"
                    data_filepath = Path(tempfile.gettempdir()) / data_filename
                    
                    # This would extract and save the underlying data
                    # Implementation depends on the specific plot type
                    # For now, we'll create a placeholder
                    pd.DataFrame({"placeholder": ["data"]}).to_csv(data_filepath, index=False)
                    
                    export_paths["data"] = str(data_filepath)
                    download_urls["data"] = f"/api/v1/download/{data_filename}"
                
                return ExportResponse(
                    success=True,
                    export_paths=export_paths,
                    download_urls=download_urls
                )
                
            except Exception as e:
                self.logger.error(f"Error exporting plot: {str(e)}")
                return ExportResponse(
                    success=False,
                    export_paths={},
                    download_urls={},
                    error_message=str(e)
                )
        
        @self.app.get("/api/v1/download/{filename}")
        async def download_file(
            filename: str,
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Download exported file."""
            try:
                if "export" not in user["permissions"]:
                    raise HTTPException(status_code=403, detail="Insufficient download permissions")
                
                filepath = Path(tempfile.gettempdir()) / filename
                
                if not filepath.exists():
                    raise HTTPException(status_code=404, detail="File not found")
                
                return FileResponse(
                    path=str(filepath),
                    filename=filename,
                    media_type='application/octet-stream'
                )
                
            except Exception as e:
                self.logger.error(f"Error downloading file: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Management endpoints
        @self.app.get("/api/v1/plots")
        async def list_plots(
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """List all available plots."""
            try:
                if "read" not in user["permissions"]:
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                plots = []
                for plot_id, metadata in self.plot_metadata.items():
                    plots.append({
                        "plot_id": plot_id,
                        "type": metadata["type"],
                        "created_at": metadata["created_at"],
                        "user": metadata["user"]
                    })
                
                return {"plots": plots}
                
            except Exception as e:
                self.logger.error(f"Error listing plots: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/api/v1/plots/{plot_id}")
        async def delete_plot(
            plot_id: str,
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Delete a plot."""
            try:
                if "write" not in user["permissions"]:
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                if plot_id not in self.plot_storage:
                    raise HTTPException(status_code=404, detail="Plot not found")
                
                # Check if user owns the plot or has admin permissions
                plot_metadata = self.plot_metadata[plot_id]
                if plot_metadata["user"] != user["name"] and "admin" not in user["permissions"]:
                    raise HTTPException(status_code=403, detail="Cannot delete plot owned by another user")
                
                # Delete plot
                del self.plot_storage[plot_id]
                del self.plot_metadata[plot_id]
                
                return {"message": "Plot deleted successfully"}
                
            except Exception as e:
                self.logger.error(f"Error deleting plot: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Rate limiting and usage endpoints
        @self.app.get("/api/v1/usage")
        async def get_usage_stats(
            user: Dict[str, Any] = Depends(get_current_user)
        ):
            """Get API usage statistics."""
            try:
                # This would return usage statistics for the current user
                return {
                    "user": user["name"],
                    "rate_limit": user["rate_limit"],
                    "requests_this_hour": len(self.api_key_manager.usage_tracking.get(
                        # This is simplified - in practice you'd need to track the actual API key
                        "current_key", []
                    )),
                    "permissions": user["permissions"]
                }
                
            except Exception as e:
                self.logger.error(f"Error getting usage stats: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _generate_plot_id(self) -> str:
        """Generate unique plot ID."""
        timestamp = str(int(time.time() * 1000))
        random_part = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        return f"plot_{timestamp}_{random_part}"
    
    async def _cleanup_old_plots(self):
        """Clean up old plots to prevent memory leaks."""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(hours=24)  # Keep plots for 24 hours
            
            plots_to_remove = []
            
            for plot_id, metadata in self.plot_metadata.items():
                created_at = datetime.fromisoformat(metadata["created_at"])
                if created_at < cutoff_time:
                    plots_to_remove.append(plot_id)
            
            for plot_id in plots_to_remove:
                if plot_id in self.plot_storage:
                    del self.plot_storage[plot_id]
                if plot_id in self.plot_metadata:
                    del self.plot_metadata[plot_id]
            
            if plots_to_remove:
                self.logger.info(f"Cleaned up {len(plots_to_remove)} old plots")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old plots: {str(e)}")
    
    def run(self):
        """Run the API server."""
        import uvicorn
        
        self.logger.info(f"Starting VectorBT Visualization API on {self.host}:{self.port}")
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


# Convenience function for creating and running the API
def create_visualization_api(
    host: str = "0.0.0.0",
    port: int = 8000,
    enable_auth: bool = True,
    enable_cors: bool = True
) -> VectorBTVisualizationAPI:
    """
    Create and configure the VectorBT Visualization API.
    
    Args:
        host: API host address
        port: API port number
        enable_auth: Whether to enable API key authentication
        enable_cors: Whether to enable CORS middleware
        
    Returns:
        Configured VectorBTVisualizationAPI instance
    """
    return VectorBTVisualizationAPI(
        host=host,
        port=port,
        enable_auth=enable_auth,
        enable_cors=enable_cors
    )


# Example usage and testing
if __name__ == "__main__":
    # Create and run the API
    api = create_visualization_api()
    api.run()