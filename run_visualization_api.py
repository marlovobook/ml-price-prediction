#!/usr/bin/env python3
"""
Launch script for the VectorBT Visualization API server.

This script provides a simple way to start the REST API server for
VectorBT visualization generation with configurable options.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the stock_predictor module to the path
sys.path.append(str(Path(__file__).parent))

from stock_predictor.visualization.api_integration import create_visualization_api


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('visualization_api.log')
        ]
    )


def main():
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(
        description="VectorBT Visualization API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the server"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to bind the server"
    )
    
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable API key authentication"
    )
    
    parser.add_argument(
        "--no-cors",
        action="store_true",
        help="Disable CORS middleware"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create and configure the API
        logger.info("Initializing VectorBT Visualization API...")
        
        api = create_visualization_api(
            host=args.host,
            port=args.port,
            enable_auth=not args.no_auth,
            enable_cors=not args.no_cors
        )
        
        # Display startup information
        print("=" * 60)
        print("🚀 VectorBT Visualization API Server")
        print("=" * 60)
        print(f"📡 Host: {args.host}")
        print(f"🔌 Port: {args.port}")
        print(f"🔐 Authentication: {'Enabled' if not args.no_auth else 'Disabled'}")
        print(f"🌐 CORS: {'Enabled' if not args.no_cors else 'Disabled'}")
        print(f"📝 Log Level: {args.log_level}")
        print()
        print("📚 API Documentation:")
        print(f"   - Swagger UI: http://{args.host}:{args.port}/docs")
        print(f"   - ReDoc: http://{args.host}:{args.port}/redoc")
        print()
        
        if not args.no_auth:
            print("🔑 Demo API Keys:")
            print("   - Demo User: demo_key_123 (100 req/hour)")
            print("   - Admin User: admin_key_456 (1000 req/hour)")
            print()
        
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Run the server
        api.run()
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        print("\n👋 Server stopped gracefully")
        
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        print(f"❌ Error starting server: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()