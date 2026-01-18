"""
Flask routes for sentiment analysis API.
"""

from flask import Blueprint, request, jsonify
import logging
import traceback

from sentiment_service import get_sentiment_service

logger = logging.getLogger(__name__)

sentiment_bp = Blueprint('sentiment', __name__, url_prefix='/api/sentiment')


@sentiment_bp.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """
    Analyze social media sentiment for a ticker.

    Request body:
        { "ticker": "AAPL" }

    Response:
        {
            "aggregate": {
                "score": 0.65,
                "label": "bullish",
                "confidence": 0.78,
                "post_count": 47,
                "sources": { "stocktwits": 30, "reddit": 12, "twitter": 5 }
            },
            "posts": [...],
            "scraped": 50,
            "embedded": 47,
            "failed": 3
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body required"}), 400

        ticker = data.get('ticker', '').strip().upper()
        if not ticker:
            return jsonify({"error": "Ticker symbol required"}), 400

        logger.info(f"Sentiment analysis request for {ticker}")

        service = get_sentiment_service()
        result = service.analyze_ticker(ticker)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@sentiment_bp.route('/summary/<ticker>', methods=['GET'])
def get_sentiment_summary(ticker: str):
    """
    Get cached sentiment summary for a ticker.

    Does not re-scrape - uses existing data from FAISS.

    Response:
        {
            "ticker": "AAPL",
            "aggregate_score": 0.65,
            "label": "bullish",
            "confidence": 0.78,
            "post_count": 47,
            "last_updated": "2026-01-18T12:00:00Z"
        }
    """
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return jsonify({"error": "Ticker symbol required"}), 400

        service = get_sentiment_service()
        summary = service.get_summary(ticker)

        return jsonify(summary)

    except Exception as e:
        logger.error(f"Error getting sentiment summary: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@sentiment_bp.route('/posts/<ticker>', methods=['GET'])
def get_sentiment_posts(ticker: str):
    """
    Get sentiment posts for a ticker with filtering.

    Query params:
        platform: Filter by platform (stocktwits, reddit, twitter)
        sentiment: Filter by sentiment (positive, negative, neutral)
        limit: Max posts to return (default 50)
        offset: Pagination offset (default 0)

    Response:
        {
            "posts": [...],
            "total": 47,
            "filters": { "platform": "all", "sentiment": "all" }
        }
    """
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return jsonify({"error": "Ticker symbol required"}), 400

        # Parse query parameters
        platform = request.args.get('platform', 'all').lower()
        sentiment = request.args.get('sentiment', 'all').lower()
        limit = min(int(request.args.get('limit', 50)), 100)
        offset = int(request.args.get('offset', 0))

        service = get_sentiment_service()

        # Retrieve posts from FAISS
        contexts = service.retrieve_sentiment_context(
            query=f"{ticker} stock social media",
            ticker=ticker,
            top_k=limit + offset + 50  # Fetch extra for filtering
        )

        # Apply filters
        filtered_posts = []
        for ctx in contexts:
            meta = ctx['metadata']

            # Platform filter
            if platform != 'all' and meta.get('platform') != platform:
                continue

            # Sentiment filter
            if sentiment != 'all' and meta.get('sentiment_label') != sentiment:
                continue

            filtered_posts.append({
                "id": ctx['id'],
                "platform": meta.get('platform', 'unknown'),
                "content": meta.get('full_content', meta.get('content', ''))[:500],
                "author": meta.get('author', ''),
                "timestamp": meta.get('timestamp', ''),
                "sentiment": {
                    "label": meta.get('sentiment_label', 'neutral'),
                    "score": meta.get('sentiment_score', 0.5)
                },
                "engagement": {
                    "likes": meta.get('likes', 0),
                    "comments": meta.get('comments', 0),
                    "score": meta.get('engagement_score', 0)
                },
                "url": meta.get('url', '')
            })

        # Apply pagination
        total = len(filtered_posts)
        paginated = filtered_posts[offset:offset + limit]

        return jsonify({
            "posts": paginated,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters": {
                "platform": platform,
                "sentiment": sentiment
            }
        })

    except Exception as e:
        logger.error(f"Error getting sentiment posts: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@sentiment_bp.route('/health', methods=['GET'])
def sentiment_health():
    """Health check for sentiment service."""
    try:
        service = get_sentiment_service()

        # Check if sentiment analyzer can be accessed
        analyzer = service.sentiment_analyzer

        return jsonify({
            "status": "healthy",
            "model": "ProsusAI/finbert",
            "model_loaded": analyzer._model is not None,
            "platforms": {
                "stocktwits": True,
                "reddit": bool(service.aggregator.scrapers["reddit"].enabled),
                "twitter": bool(service.aggregator.scrapers["twitter"].enabled)
            }
        })

    except Exception as e:
        logger.error(f"Sentiment health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500
