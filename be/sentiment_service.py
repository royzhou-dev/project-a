"""
Sentiment analysis service that orchestrates:
1. Social media scraping
2. FinBERT sentiment analysis
3. FAISS vector storage for RAG
4. Aggregate sentiment calculation
"""

import math
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from sentiment_analyzer import get_sentiment_analyzer
from social_scrapers import SocialMediaAggregator
from rag_pipeline import EmbeddingGenerator, VectorStore
from config import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    TWITTER_BEARER_TOKEN
)

logger = logging.getLogger(__name__)


class SentimentService:
    """
    Main service for social media sentiment analysis.

    Coordinates scraping, sentiment analysis, embedding generation,
    and vector storage for RAG integration.
    """

    NAMESPACE = "sentiment"
    MAX_POSTS_PER_PLATFORM = 30
    MAX_WORKERS = 5

    # Bias correction: asymmetric thresholds to counteract bullish bias in data sources
    # Social media (WSB, StockTwits) skews positive; FinBERT also has slight positive bias
    BULLISH_THRESHOLD = 0.3    # Higher bar for bullish (was 0.2)
    BEARISH_THRESHOLD = -0.15  # Lower bar for bearish (was -0.2)

    # Minimum confidence to include a post in aggregate calculation
    MIN_CONFIDENCE_THRESHOLD = 0.6

    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize the sentiment service.

        Args:
            vector_store: Shared VectorStore instance (for consistency with chat service)
        """
        self.vector_store = vector_store or VectorStore()
        self.embedding_gen = EmbeddingGenerator()
        self.sentiment_analyzer = get_sentiment_analyzer()

        self.aggregator = SocialMediaAggregator(
            reddit_client_id=REDDIT_CLIENT_ID,
            reddit_client_secret=REDDIT_CLIENT_SECRET,
            reddit_user_agent=REDDIT_USER_AGENT,
            twitter_bearer_token=TWITTER_BEARER_TOKEN
        )

    def analyze_ticker(self, ticker: str) -> Dict:
        """
        Full sentiment analysis pipeline for a ticker.

        1. Scrapes social media posts
        2. Analyzes sentiment with FinBERT
        3. Embeds and stores in FAISS
        4. Calculates aggregate sentiment

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            dict with aggregate sentiment, posts, and stats
        """
        ticker = ticker.upper()
        logger.info(f"Starting sentiment analysis for {ticker}")

        # Step 1: Scrape posts from all platforms
        platform_posts = self.aggregator.scrape_all(ticker, limit_per_platform=self.MAX_POSTS_PER_PLATFORM)

        all_posts = []
        for posts in platform_posts.values():
            all_posts.extend(posts)

        if not all_posts:
            logger.warning(f"No social media posts found for {ticker}")
            return {
                "aggregate": {
                    "score": 0,
                    "label": "neutral",
                    "confidence": 0,
                    "post_count": 0,
                    "sources": {"stocktwits": 0, "reddit": 0, "twitter": 0}
                },
                "posts": [],
                "scraped": 0,
                "embedded": 0,
                "failed": 0
            }

        # Step 2: Analyze sentiment for all posts
        texts = [post["content"] for post in all_posts]
        sentiments = self.sentiment_analyzer.analyze_batch(texts)

        # Attach sentiment to posts
        for post, sentiment in zip(all_posts, sentiments):
            post["sentiment"] = sentiment
            post["sentiment_label"] = sentiment["label"]
            post["sentiment_score"] = sentiment["score"]

        # Step 3: Embed and store posts in FAISS
        embedded_count = 0
        failed_count = 0

        def embed_and_store(post):
            """Embed a single post and store in vector DB."""
            try:
                doc_id = post["id"]

                # Skip if already exists
                if self.vector_store.document_exists(doc_id, namespace=self.NAMESPACE):
                    return "skipped"

                # Generate embedding
                embedding = self.embedding_gen.generate_embedding(post["content"])
                if not embedding:
                    return "failed"

                # Prepare metadata for storage
                metadata = {
                    "ticker": ticker,
                    "type": "social_post",
                    "platform": post.get("platform", "unknown"),
                    "content": post["content"][:500],  # Truncate for storage
                    "content_preview": post["content"][:200],
                    "full_content": post["content"],
                    "author": post.get("author", ""),
                    "timestamp": post.get("timestamp", ""),
                    "likes": post.get("likes", 0),
                    "comments": post.get("comments", 0),
                    "engagement_score": post.get("engagement_score", 0),
                    "sentiment_label": post["sentiment_label"],
                    "sentiment_score": post["sentiment_score"],
                    "url": post.get("url", "")
                }

                # Store in FAISS
                success = self.vector_store.upsert_document(
                    doc_id=doc_id,
                    embedding=embedding,
                    metadata=metadata,
                    namespace=self.NAMESPACE
                )

                return "embedded" if success else "failed"

            except Exception as e:
                logger.error(f"Error embedding post {post.get('id', 'unknown')}: {e}")
                return "failed"

        # Process posts in parallel
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {executor.submit(embed_and_store, post): post for post in all_posts}

            for future in as_completed(futures):
                result = future.result()
                if result == "embedded":
                    embedded_count += 1
                elif result == "failed":
                    failed_count += 1

        # Save FAISS index after batch operation
        self.vector_store.save()

        # Step 4: Calculate aggregate sentiment
        aggregate = self._calculate_aggregate_sentiment(all_posts)
        aggregate["sources"] = self.aggregator.get_source_counts(all_posts)

        # Sort posts by engagement and recency for display
        sorted_posts = sorted(
            all_posts,
            key=lambda x: (x.get("engagement_score", 0), x.get("timestamp", "")),
            reverse=True
        )

        # Format posts for response
        formatted_posts = [self._format_post_for_response(p) for p in sorted_posts[:50]]

        logger.info(f"Sentiment analysis complete for {ticker}: {aggregate['label']} ({aggregate['score']:.2f})")

        return {
            "aggregate": aggregate,
            "posts": formatted_posts,
            "scraped": len(all_posts),
            "embedded": embedded_count,
            "failed": failed_count
        }

    def get_summary(self, ticker: str) -> Dict:
        """
        Get a quick sentiment summary without re-scraping.

        Uses existing data from FAISS if available.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Summary dict with aggregate score and metadata
        """
        ticker = ticker.upper()

        # Search for existing sentiment data
        query_embedding = self.embedding_gen.generate_query_embedding(f"{ticker} stock sentiment")
        if not query_embedding:
            return {"ticker": ticker, "error": "Failed to generate query embedding"}

        matches = self.vector_store.search(
            query_embedding=query_embedding,
            ticker=ticker,
            namespace=self.NAMESPACE,
            top_k=100
        )

        if not matches:
            return {
                "ticker": ticker,
                "aggregate_score": 0,
                "label": "neutral",
                "post_count": 0,
                "last_updated": None
            }

        # Reconstruct posts from matches
        posts = []
        latest_timestamp = None

        for match in matches:
            meta = match.metadata
            posts.append({
                "sentiment_label": meta.get("sentiment_label", "neutral"),
                "sentiment_score": meta.get("sentiment_score", 0.5),
                "engagement_score": meta.get("engagement_score", 0),
                "timestamp": meta.get("timestamp", "")
            })

            ts = meta.get("timestamp")
            if ts and (not latest_timestamp or ts > latest_timestamp):
                latest_timestamp = ts

        aggregate = self._calculate_aggregate_sentiment(posts)

        return {
            "ticker": ticker,
            "aggregate_score": aggregate["score"],
            "label": aggregate["label"],
            "confidence": aggregate["confidence"],
            "post_count": len(posts),
            "last_updated": latest_timestamp
        }

    def retrieve_sentiment_context(self, query: str, ticker: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant sentiment posts for RAG.

        Args:
            query: User query
            ticker: Stock ticker
            top_k: Number of results

        Returns:
            List of relevant posts with sentiment data
        """
        query_embedding = self.embedding_gen.generate_query_embedding(query)
        if not query_embedding:
            return []

        matches = self.vector_store.search(
            query_embedding=query_embedding,
            ticker=ticker,
            namespace=self.NAMESPACE,
            top_k=top_k
        )

        contexts = []
        for match in matches:
            contexts.append({
                "score": match.score,
                "metadata": match.metadata,
                "id": match.id
            })

        return contexts

    def _calculate_aggregate_sentiment(self, posts: List[Dict]) -> Dict:
        """
        Calculate weighted aggregate sentiment with bias correction.

        Weighting factors:
        - Recency: Posts from last 24h weighted 2x
        - Engagement: log(1 + engagement_score)
        - Confidence: FinBERT confidence score (filtered by MIN_CONFIDENCE_THRESHOLD)

        Bias corrections:
        - Asymmetric thresholds for bullish/bearish classification
        - Confidence filtering to exclude low-confidence predictions
        - Neutral posts contribute slightly negative (-0.05) to counteract positive bias
        """
        if not posts:
            return {"score": 0, "label": "neutral", "confidence": 0, "post_count": 0}

        weighted_sum = 0
        total_weight = 0
        now = datetime.now(timezone.utc)

        # Track sentiment distribution for logging
        distribution = {"positive": 0, "neutral": 0, "negative": 0}
        filtered_count = 0

        for post in posts:
            label = post.get("sentiment_label", "neutral")
            confidence = post.get("sentiment_score", 0.5)

            # Track raw distribution before filtering
            distribution[label] = distribution.get(label, 0) + 1

            # Filter out low-confidence predictions
            if confidence < self.MIN_CONFIDENCE_THRESHOLD:
                filtered_count += 1
                continue

            # Convert sentiment label to numeric score
            # Neutral posts get slight negative bias (-0.05) to counteract data source bias
            base_score = {"negative": -1, "neutral": -0.05, "positive": 1}.get(label, 0)

            # Recency weight
            recency_weight = 1.0
            timestamp = post.get("timestamp")
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        post_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    else:
                        post_time = timestamp

                    hours_old = (now - post_time).total_seconds() / 3600
                    recency_weight = 2.0 if hours_old < 24 else (1.5 if hours_old < 72 else 1.0)
                except:
                    pass

            # Engagement weight
            engagement = post.get("engagement_score", 0)
            engagement_weight = math.log(1 + engagement + 1)

            # Combined weight
            weight = confidence * recency_weight * engagement_weight
            weighted_sum += base_score * weight
            total_weight += weight

        # Log sentiment distribution for debugging
        logger.info(
            f"Sentiment distribution: {distribution} | "
            f"Filtered (low confidence): {filtered_count} | "
            f"Included: {len(posts) - filtered_count}"
        )

        avg_score = weighted_sum / total_weight if total_weight > 0 else 0

        # Determine label using asymmetric thresholds to counteract bullish bias
        if avg_score < self.BEARISH_THRESHOLD:
            label = "bearish"
        elif avg_score > self.BULLISH_THRESHOLD:
            label = "bullish"
        else:
            label = "neutral"

        included_count = len(posts) - filtered_count
        return {
            "score": round(avg_score, 3),
            "label": label,
            "confidence": round(min(1.0, total_weight / max(1, included_count) / 2), 3),
            "post_count": len(posts),
            "included_count": included_count,
            "distribution": distribution
        }

    def _format_post_for_response(self, post: Dict) -> Dict:
        """Format a post for API response."""
        return {
            "id": post.get("id", ""),
            "platform": post.get("platform", "unknown"),
            "content": post.get("content", "")[:500],
            "author": post.get("author", ""),
            "timestamp": post.get("timestamp", ""),
            "sentiment": post.get("sentiment", {}),
            "engagement": {
                "likes": post.get("likes", 0),
                "comments": post.get("comments", 0),
                "score": post.get("engagement_score", 0)
            },
            "url": post.get("url", "")
        }


# Singleton instance
_sentiment_service: Optional[SentimentService] = None


def get_sentiment_service(vector_store: Optional[VectorStore] = None) -> SentimentService:
    """Get or create the singleton sentiment service instance."""
    global _sentiment_service
    if _sentiment_service is None:
        _sentiment_service = SentimentService(vector_store=vector_store)
    return _sentiment_service
