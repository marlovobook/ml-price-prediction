# VectorBT Visualization Dashboard & API Integration

This document provides comprehensive documentation for the VectorBT Visualization Enhancement dashboard integration and REST API features.

## Overview

The VectorBT Visualization Enhancement provides two primary integration methods:

1. **Streamlit Dashboard Integration** - Interactive web-based visualization with caching and real-time updates
2. **REST API** - Programmatic access to visualization generation with authentication and rate limiting

## Table of Contents

- [Streamlit Dashboard Integration](#streamlit-dashboard-integration)
- [REST API](#rest-api)
- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)
- [Deployment](#deployment)

---

## Streamlit Dashboard Integration

### Features

The Streamlit dashboard integration provides:

- ✅ **Streamlit-Compatible Plot Objects** - Optimized plotly figures for Streamlit display
- ✅ **Intelligent Caching** - Automatic plot caching with configurable TTL for improved performance
- ✅ **Real-time Updates** - Support for live data updates and automatic refresh
- ✅ **Responsive Design** - Mobile-friendly visualizations with adaptive layouts
- ✅ **Interactive Controls** - User-friendly controls for plot customization

### Quick Start

```python
import streamlit as st
from stock_predictor.visualization.dashboard_integration import StreamlitVisualizationAdapter

# Initialize the adapter
adapter = StreamlitVisualizationAdapter(
    enable_caching=True,
    cache_ttl=300  # 5 minutes
)

# Create and display a portfolio visualization
portfolio_plot = adapter.create_streamlit_portfolio_plot(
    portfolio=my_portfolio,
    title="Portfolio Performance",
    use_cache=True
)

st.plotly_chart(portfolio_plot, use_container_width=True)
```

### Dashboard Components

#### 1. StreamlitVisualizationAdapter

Main adapter class for Streamlit integration.

**Key Methods:**

- `create_streamlit_portfolio_plot()` - Create portfolio performance plot
- `create_streamlit_drawdown_plot()` - Create drawdown analysis plot
- `create_streamlit_comparison_plot()` - Create multi-strategy comparison plot
- `display_portfolio_visualization()` - Display complete portfolio analysis
- `create_cached_visualization_dashboard()` - Create full dashboard with caching

**Example:**

```python
# Create comprehensive dashboard
adapter.create_cached_visualization_dashboard(
    portfolios={
        'Strategy A': portfolio_a,
        'Strategy B': portfolio_b,
        'Strategy C': portfolio_c
    },
    enable_real_time=False
)
```

#### 2. Real-time Visualization

Support for live data updates and automatic refresh.

```python
# Enable real-time updates
adapter.display_real_time_visualization(
    data_source_func=fetch_latest_data,
    update_interval=60,  # Update every 60 seconds
    auto_refresh=True
)
```

#### 3. Caching System

Intelligent caching system for improved performance.

**Features:**
- Automatic cache key generation based on plot parameters
- Configurable TTL (time-to-live)
- Automatic cleanup of expired cache entries
- Cache size management (keeps 50 most recent plots)

**Configuration:**

```python
adapter = StreamlitVisualizationAdapter(
    enable_caching=True,
    cache_ttl=300  # 5 minutes
)

# Disable caching for specific plot
plot = adapter.create_streamlit_portfolio_plot(
    portfolio=my_portfolio,
    use_cache=False  # Skip cache
)
```

### Integration with Existing Dashboard

To integrate with the existing `streamlit_dashboard.py`:

```python
from stock_predictor.visualization.dashboard_integration import StreamlitVisualizationAdapter

# In your dashboard class
class StockPredictorDashboard:
    def __init__(self):
        # ... existing initialization ...
        self.viz_adapter = StreamlitVisualizationAdapter()
    
    def render_vectorbt_visualizations(self, portfolio):
        """Render VectorBT visualizations."""
        st.subheader("📈 VectorBT Portfolio Analysis")
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Performance", "Drawdown"])
        
        with tab1:
            plot = self.viz_adapter.create_streamlit_portfolio_plot(portfolio)
            st.plotly_chart(plot, use_container_width=True)
        
        with tab2:
            plot = self.viz_adapter.create_streamlit_drawdown_plot(portfolio)
            st.plotly_chart(plot, use_container_width=True)
```

---

## REST API

### Features

The REST API provides:

- ✅ **RESTful Endpoints** - Standard HTTP methods for visualization generation
- ✅ **Authentication** - API key-based authentication with user permissions
- ✅ **Rate Limiting** - Configurable rate limits per user
- ✅ **Multiple Export Formats** - PNG, HTML, JSON, and data export
- ✅ **Programmatic Access** - Full access to plot objects and underlying data
- ✅ **Interactive Documentation** - Swagger UI and ReDoc

### Starting the API Server

```bash
# Basic usage
python run_visualization_api.py

# Custom configuration
python run_visualization_api.py --host 0.0.0.0 --port 8000 --log-level INFO

# Disable authentication (development only)
python run_visualization_api.py --no-auth

# Disable CORS
python run_visualization_api.py --no-cors
```

### API Endpoints

#### Health Check

```http
GET /health
```

Returns API health status.

#### Portfolio Visualization

```http
POST /api/v1/visualizations/portfolio
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "predictions": [0, 1, 2, 1, 2, ...],
  "price_data": {
    "Open": [...],
    "High": [...],
    "Low": [...],
    "Close": [...],
    "Volume": [...]
  },
  "test_start_idx": 100,
  "symbol": "AAPL",
  "title": "Portfolio Performance",
  "portfolio_config": {
    "init_cash": 10000,
    "fees": 0.001,
    "size_value": 50
  }
}
```

**Response:**

```json
{
  "success": true,
  "plot_id": "plot_1234567890_abc123",
  "metrics_summary": {
    "total_return": 0.15,
    "sharpe_ratio": 1.2,
    "max_drawdown": -0.08,
    "num_trades": 25
  },
  "generation_time": 1.23
}
```

#### Drawdown Visualization

```http
POST /api/v1/visualizations/drawdown
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "predictions": [...],
  "price_data": {...},
  "test_start_idx": 100,
  "symbol": "AAPL"
}
```

#### Comparison Visualization

```http
POST /api/v1/visualizations/comparison
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "portfolios_data": {
    "Strategy A": {
      "predictions": [...],
      "price_data": {...},
      "test_start_idx": 100
    },
    "Strategy B": {
      "predictions": [...],
      "price_data": {...},
      "test_start_idx": 100
    }
  },
  "title": "Strategy Comparison"
}
```

#### Retrieve Plot Data

```http
GET /api/v1/plots/{plot_id}
Authorization: Bearer {api_key}
```

#### Retrieve Plot HTML

```http
GET /api/v1/plots/{plot_id}/html
Authorization: Bearer {api_key}
```

#### Export Plot

```http
POST /api/v1/export
Authorization: Bearer {api_key}
Content-Type: application/json

{
  "plot_id": "plot_1234567890_abc123",
  "formats": ["png", "html", "json"],
  "include_data": true
}
```

**Response:**

```json
{
  "success": true,
  "export_paths": {
    "png": "/tmp/plot_1234567890_abc123.png",
    "html": "/tmp/plot_1234567890_abc123.html",
    "json": "/tmp/plot_1234567890_abc123.json"
  },
  "download_urls": {
    "png": "/api/v1/download/plot_1234567890_abc123.png",
    "html": "/api/v1/download/plot_1234567890_abc123.html",
    "json": "/api/v1/download/plot_1234567890_abc123.json"
  }
}
```

#### Download File

```http
GET /api/v1/download/{filename}
Authorization: Bearer {api_key}
```

#### List Plots

```http
GET /api/v1/plots
Authorization: Bearer {api_key}
```

#### Delete Plot

```http
DELETE /api/v1/plots/{plot_id}
Authorization: Bearer {api_key}
```

#### Usage Statistics

```http
GET /api/v1/usage
Authorization: Bearer {api_key}
```

---

## Authentication

The API uses Bearer token authentication with API keys.

### Demo API Keys

For development and testing:

- **Demo User**: `demo_key_123`
  - Rate limit: 100 requests/hour
  - Permissions: read, write, export

- **Admin User**: `admin_key_456`
  - Rate limit: 1000 requests/hour
  - Permissions: read, write, export, admin

### Using API Keys

Include the API key in the Authorization header:

```bash
curl -X POST http://localhost:8000/api/v1/visualizations/portfolio \
  -H "Authorization: Bearer demo_key_123" \
  -H "Content-Type: application/json" \
  -d @request.json
```

### Production Deployment

For production, replace the simple API key manager with a proper authentication system:

1. Implement OAuth 2.0 or JWT-based authentication
2. Store API keys securely (encrypted database, secrets manager)
3. Implement key rotation and expiration
4. Add IP whitelisting for additional security
5. Enable HTTPS/TLS encryption

---

## Rate Limiting

Rate limiting is enforced per API key to prevent abuse.

### Default Limits

- Demo users: 100 requests/hour
- Admin users: 1000 requests/hour

### Rate Limit Headers

The API includes rate limit information in response headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640000000
```

### Handling Rate Limits

When rate limit is exceeded, the API returns:

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json

{
  "detail": "Rate limit exceeded"
}
```

---

## Examples

### Python Client Example

```python
from examples.api_client_example import VectorBTVisualizationClient

# Initialize client
client = VectorBTVisualizationClient(
    base_url="http://localhost:8000",
    api_key="demo_key_123"
)

# Create portfolio visualization
result = client.create_portfolio_visualization(
    predictions=[0, 1, 2, 1, 2, 0, 1],
    price_data={
        'Open': [100, 101, 102, 103, 104, 105, 106],
        'High': [101, 102, 103, 104, 105, 106, 107],
        'Low': [99, 100, 101, 102, 103, 104, 105],
        'Close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5],
        'Volume': [1000000] * 7
    },
    test_start_idx=3,
    symbol="DEMO"
)

print(f"Plot ID: {result['plot_id']}")
print(f"Metrics: {result['metrics_summary']}")

# Export plot
export_result = client.export_plot(
    plot_id=result['plot_id'],
    formats=["png", "html"],
    include_data=True
)

# Download exported files
for format_type, url in export_result['download_urls'].items():
    filename = url.split('/')[-1]
    client.download_file(filename, f"exports/{filename}")
```

### cURL Examples

**Create Portfolio Visualization:**

```bash
curl -X POST http://localhost:8000/api/v1/visualizations/portfolio \
  -H "Authorization: Bearer demo_key_123" \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [0, 1, 2, 1, 2],
    "price_data": {
      "Close": [100, 101, 102, 103, 104]
    },
    "test_start_idx": 2,
    "symbol": "TEST"
  }'
```

**Get Plot Data:**

```bash
curl -X GET http://localhost:8000/api/v1/plots/plot_123 \
  -H "Authorization: Bearer demo_key_123"
```

**Export Plot:**

```bash
curl -X POST http://localhost:8000/api/v1/export \
  -H "Authorization: Bearer demo_key_123" \
  -H "Content-Type: application/json" \
  -d '{
    "plot_id": "plot_123",
    "formats": ["png", "html"],
    "include_data": true
  }'
```

---

## Deployment

### Development

```bash
# Start API server
python run_visualization_api.py --log-level DEBUG

# Start Streamlit dashboard
streamlit run streamlit_dashboard.py
```

### Production

#### Using Docker

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "run_visualization_api.py", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t vectorbt-viz-api .
docker run -p 8000:8000 vectorbt-viz-api
```

#### Using Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
  stock_predictor.visualization.api_integration:app \
  --bind 0.0.0.0:8000
```

#### Environment Variables

```bash
# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export API_LOG_LEVEL=INFO

# Authentication
export API_ENABLE_AUTH=true
export API_KEY_SECRET=your-secret-key

# Rate Limiting
export API_RATE_LIMIT_DEFAULT=100
export API_RATE_LIMIT_ADMIN=1000

# CORS
export API_ENABLE_CORS=true
export API_CORS_ORIGINS=https://yourdomain.com
```

### Monitoring

#### Health Checks

```bash
# Check API health
curl http://localhost:8000/health
```

#### Logging

Logs are written to:
- Console (stdout)
- `visualization_api.log` file

#### Metrics

Monitor these key metrics:
- Request rate (requests/second)
- Response time (p50, p95, p99)
- Error rate (4xx, 5xx responses)
- Cache hit rate
- Active plot count
- Memory usage

---

## API Documentation

Interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide:
- Complete endpoint documentation
- Request/response schemas
- Interactive API testing
- Code examples in multiple languages

---

## Troubleshooting

### Common Issues

**1. Connection Refused**

```
Error: Could not connect to API server
```

**Solution**: Ensure the API server is running:
```bash
python run_visualization_api.py
```

**2. Authentication Failed**

```
HTTP 401: Invalid API key
```

**Solution**: Check your API key is correct and included in the Authorization header.

**3. Rate Limit Exceeded**

```
HTTP 429: Rate limit exceeded
```

**Solution**: Wait for the rate limit window to reset or use an API key with higher limits.

**4. Plot Not Found**

```
HTTP 404: Plot not found
```

**Solution**: Plots are automatically cleaned up after 24 hours. Generate a new plot or check the plot ID.

**5. Cache Issues**

If cached plots are stale:

```python
# Clear cache in Streamlit
st.session_state.viz_cache = {}
st.session_state.viz_cache_timestamps = {}
```

---

## Support

For issues, questions, or feature requests:

1. Check the API documentation: http://localhost:8000/docs
2. Review the examples in `examples/api_client_example.py`
3. Check the logs in `visualization_api.log`
4. Refer to the main project README

---

## License

This integration is part of the Stock Direction Predictor project and follows the same license terms.