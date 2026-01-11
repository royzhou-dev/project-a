# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Project A is a personal stock trading assistant that provides users with comprehensive market information powered by the Polygon.io API. The platform allows users to explore stock data, news, financials, dividends, splits, and more. Future plans include AI/ML-powered insights and recommendations.

## Architecture

- **fe/**: Vanilla JavaScript frontend (no frameworks)
  - Single-page application with tab-based navigation
  - Client-side caching system with configurable TTL
  - Canvas-based chart rendering
  - Responsive design with mobile support

- **be/**: Python Flask backend
  - RESTful API endpoints
  - Polygon.io API integration
  - CORS-enabled for local development
  - Environment-based configuration

## Tech Stack

### Frontend
- Vanilla JavaScript (ES6+)
- HTML5 Canvas for charts
- CSS3 (Flexbox & Grid)
- No external dependencies

### Backend
- Flask 3.0.0
- Python 3.8+
- Requests for API calls
- python-dotenv for configuration
- Flask-CORS for cross-origin support

## Key Features

1. **Stock Selection**: Searchable dropdown with thousands of tickers from `company_tickers.json`
2. **Market Status**: Real-time market open/closed indicator
3. **Overview Tab**: Key metrics (price, volume, market cap, open, high, low)
4. **Chart Tab**: Interactive price charts with multiple timeframes (1M, 3M, 6M, 1Y, 5Y)
5. **Financials Tab**: Quarterly/annual financial statements (revenue, net income, assets, liabilities)
6. **Dividends Tab**: Historical dividend payment data with ex-dates and pay dates
7. **Splits Tab**: Stock split history with ratios and types
8. **News Tab**: Latest news articles related to selected stocks
9. **Related Tab**: Similar/related companies (may require paid API)

## Data Flow

1. User selects a ticker from dropdown
2. Frontend makes parallel API calls to backend for ticker details and previous close
3. Backend fetches data from Polygon.io API
4. Data is cached client-side with appropriate TTL
5. User switches tabs → data loads from cache if available, otherwise fetches from API
6. Subsequent visits to same ticker/tab load instantly from cache

## Caching Strategy

The application uses client-side caching to respect Polygon.io's free tier rate limits (5 API calls/minute):

- **Static data** (cache until page refresh): Ticker details, dividends, splits, related companies
- **Daily data** (24-hour cache): Previous close, market status, chart data
- **Moderate** (30-minute cache): Financial statements
- **Short** (15-minute cache): News articles

Access cache stats via browser console: `stockCache.getStats()`

## API Integration

All data fetched from Polygon.io API (https://polygon.io/):
- `/v3/reference/tickers/{ticker}` - Company details
- `/v2/aggs/ticker/{ticker}/prev` - Previous day close
- `/v2/aggs/ticker/{ticker}/range/...` - Historical aggregates
- `/v3/reference/dividends` - Dividend history
- `/v3/reference/splits` - Stock split history
- `/v2/reference/news` - News articles
- `/vX/reference/financials` - Financial statements
- `/v1/marketstatus/now` - Market status

## Configuration

API key stored in `.env` file:
```
POLYGON_API_KEY=your_key_here
PORT=5000
```

## Future Enhancements

- AI-powered stock analysis and recommendations
- Portfolio tracking
- Real-time price updates (WebSocket integration)
- Advanced technical indicators
- User authentication and preferences
- Watchlists and alerts