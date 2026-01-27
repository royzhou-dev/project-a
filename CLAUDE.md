# CLAUDE.md

## Quick Start

```bash
cd be && pip install -r requirements.txt
# Add POLYGON_API_KEY and GEMINI_API_KEY to .env
python app.py
# Open http://localhost:5000
```

## Project Overview

Stock research assistant with Polygon.io data, AI chatbot (RAG-powered), and social media sentiment analysis.

## Architecture

```
fe/          Vanilla JS frontend (no frameworks)
be/          Python Flask backend
  app.py              Main Flask app, stock data routes
  chat_*.py           AI chatbot (RAG, streaming, Gemini)
  sentiment_*.py      Social sentiment (FinBERT, scrapers)
  rag_pipeline.py     FAISS vector store, embeddings
  polygon_api.py      Polygon.io wrapper
```

## Key Technical Decisions

- **Frontend**: Pure vanilla JS - no frameworks allowed
- **AI**: Google Gemini (free tier) for chat + embeddings
- **Vector DB**: FAISS local storage (`be/faiss_index/`)
- **Scraping**: cloudscraper for Cloudflare bypass (StockTwits, Reddit)
- **Sentiment**: FinBERT model (lazy-loaded, ~500MB)

## API Keys (.env)

```bash
POLYGON_API_KEY=     # Stock data (5 calls/min free tier)
GEMINI_API_KEY=      # Chat + embeddings (free)
TWITTER_BEARER_TOKEN= # Optional, paid ($100+/month)
```

## Rate Limits

Polygon.io: 5/min (handled by frontend caching). Gemini: 15 RPM chat, 1500 RPM embeddings.

## Main Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /api/chat/message` | Chat with streaming SSE |
| `POST /api/chat/scrape-articles` | RAG article indexing |
| `POST /api/sentiment/analyze` | Sentiment analysis |

## Where to Find Details

| Topic | Location |
|-------|----------|
| Frontend caching strategy | `fe/app.js` - stockCache object |
| Chart implementation | `fe/app.js` - chartState, drawChart functions |
| RAG pipeline | `be/rag_pipeline.py`, `be/chat_service.py` |
| Sentiment bias corrections | `be/sentiment_service.py` - aggregate calculation |
| Social scrapers | `be/social_scrapers.py` |
| Design system | `.design-engineer/system.md` |

## Development Rules

1. Keep frontend vanilla JS - no frameworks
2. Respect caching TTLs (see stockCache in app.js)
3. FAISS namespaces: `news:` for articles, `sentiment:` for posts
4. FinBERT has bullish bias - see sentiment_service.py for corrections
