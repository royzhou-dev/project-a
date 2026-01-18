"""
Social media scrapers for stock sentiment data.
Supports StockTwits, Reddit, and Twitter (optional).
"""

import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime, timezone
import logging
import hashlib

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Base class for social media scrapers."""

    @abstractmethod
    def scrape(self, ticker: str, limit: int = 50) -> List[Dict]:
        """
        Fetch posts for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            limit: Maximum number of posts to return

        Returns:
            List of standardized post dicts
        """
        pass

    def generate_post_id(self, platform: str, unique_id: str) -> str:
        """Generate a unique post ID."""
        return f"{platform}_{unique_id}"

    def calculate_engagement_score(self, likes: int, comments: int, retweets: int = 0) -> int:
        """Calculate total engagement score."""
        return likes + comments + retweets


class StockTwitsScraper(BaseScraper):
    """
    StockTwits API scraper.

    - FREE, no authentication required for basic access
    - Rate limit: ~200 requests/hour
    - Endpoint: https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json
    """

    BASE_URL = "https://api.stocktwits.com/api/2"
    TIMEOUT = 10

    def scrape(self, ticker: str, limit: int = 50) -> List[Dict]:
        """Fetch posts from StockTwits for a ticker."""
        url = f"{self.BASE_URL}/streams/symbol/{ticker.upper()}.json"

        try:
            response = requests.get(
                url,
                timeout=self.TIMEOUT,
                headers={"User-Agent": "StockAssistant/1.0"}
            )
            response.raise_for_status()
            data = response.json()

            if data.get("response", {}).get("status") != 200:
                logger.warning(f"StockTwits API error for {ticker}: {data}")
                return []

            posts = []
            messages = data.get("messages", [])[:limit]

            for msg in messages:
                try:
                    post = self._standardize_post(msg, ticker)
                    if post:
                        posts.append(post)
                except Exception as e:
                    logger.debug(f"Failed to parse StockTwits message: {e}")
                    continue

            logger.info(f"Scraped {len(posts)} posts from StockTwits for {ticker}")
            return posts

        except requests.exceptions.Timeout:
            logger.warning(f"StockTwits request timed out for {ticker}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"StockTwits scraping error for {ticker}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected StockTwits error for {ticker}: {e}")
            return []

    def _standardize_post(self, raw: Dict, ticker: str) -> Optional[Dict]:
        """Convert StockTwits message to standard format."""
        msg_id = raw.get("id")
        if not msg_id:
            return None

        body = raw.get("body", "").strip()
        if not body:
            return None

        user = raw.get("user", {})
        created_at = raw.get("created_at", "")

        # Parse timestamp
        timestamp = None
        if created_at:
            try:
                timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00")).isoformat()
            except:
                timestamp = created_at

        # StockTwits has built-in sentiment
        st_sentiment = raw.get("entities", {}).get("sentiment", {})
        st_sentiment_label = st_sentiment.get("basic") if st_sentiment else None

        likes_data = raw.get("likes", {})
        likes_count = likes_data.get("total", 0) if isinstance(likes_data, dict) else 0

        return {
            "id": self.generate_post_id("stocktwits", str(msg_id)),
            "platform": "stocktwits",
            "ticker": ticker.upper(),
            "content": body,
            "author": user.get("username", "unknown"),
            "author_followers": user.get("followers", 0),
            "timestamp": timestamp,
            "likes": likes_count,
            "comments": 0,  # Not available in basic API
            "retweets": 0,
            "engagement_score": self.calculate_engagement_score(likes_count, 0),
            "url": f"https://stocktwits.com/{user.get('username', '')}/message/{msg_id}",
            "stocktwits_sentiment": st_sentiment_label  # "Bullish" or "Bearish" if available
        }


class RedditScraper(BaseScraper):
    """
    Reddit API scraper using PRAW.

    - FREE with OAuth authentication
    - Rate limit: 60 requests/minute
    - Subreddits: wallstreetbets, stocks, investing, options
    """

    SUBREDDITS = ["wallstreetbets", "stocks", "investing", "options"]
    TIMEOUT = 10

    def __init__(self, client_id: str = "", client_secret: str = "", user_agent: str = "StockAssistant/1.0"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self._reddit = None
        self.enabled = bool(client_id and client_secret)

    def _get_reddit(self):
        """Lazy load Reddit client."""
        if self._reddit is None and self.enabled:
            try:
                import praw
                self._reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent
                )
                logger.info("Reddit client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Reddit client: {e}")
                self.enabled = False
        return self._reddit

    def scrape(self, ticker: str, limit: int = 50) -> List[Dict]:
        """Fetch posts from Reddit for a ticker."""
        if not self.enabled:
            logger.debug("Reddit scraper disabled - no credentials")
            return []

        reddit = self._get_reddit()
        if not reddit:
            return []

        posts = []
        per_subreddit = max(1, limit // len(self.SUBREDDITS))

        for subreddit_name in self.SUBREDDITS:
            if len(posts) >= limit:
                break

            try:
                subreddit = reddit.subreddit(subreddit_name)

                # Search for ticker mentions (with $ prefix common in finance)
                search_queries = [f"${ticker}", ticker.upper()]

                for query in search_queries:
                    if len(posts) >= limit:
                        break

                    try:
                        for submission in subreddit.search(
                            query,
                            limit=per_subreddit,
                            time_filter="week",
                            sort="relevance"
                        ):
                            post = self._standardize_post(submission, ticker, subreddit_name)
                            if post and post["id"] not in [p["id"] for p in posts]:
                                posts.append(post)

                            if len(posts) >= limit:
                                break
                    except Exception as e:
                        logger.debug(f"Reddit search failed for {query} in r/{subreddit_name}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Reddit scraping error for r/{subreddit_name}: {e}")
                continue

        logger.info(f"Scraped {len(posts)} posts from Reddit for {ticker}")
        return posts[:limit]

    def _standardize_post(self, submission, ticker: str, subreddit: str) -> Optional[Dict]:
        """Convert Reddit submission to standard format."""
        try:
            # Combine title and selftext
            content = submission.title
            if submission.selftext:
                content = f"{submission.title}\n\n{submission.selftext}"

            content = content.strip()
            if not content:
                return None

            # Truncate very long posts
            if len(content) > 2000:
                content = content[:2000] + "..."

            timestamp = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat()

            author_name = str(submission.author) if submission.author else "[deleted]"

            return {
                "id": self.generate_post_id("reddit", submission.id),
                "platform": "reddit",
                "ticker": ticker.upper(),
                "content": content,
                "author": author_name,
                "author_followers": 0,  # Not easily available
                "timestamp": timestamp,
                "likes": submission.score,
                "comments": submission.num_comments,
                "retweets": 0,
                "engagement_score": self.calculate_engagement_score(submission.score, submission.num_comments),
                "url": f"https://reddit.com{submission.permalink}",
                "subreddit": subreddit
            }

        except Exception as e:
            logger.debug(f"Failed to parse Reddit submission: {e}")
            return None


class TwitterScraper(BaseScraper):
    """
    Twitter/X API scraper.

    NOTE: Twitter API v2 requires paid access ($100+/month for Basic tier).
    This scraper is disabled by default and serves as a placeholder.
    """

    def __init__(self, bearer_token: str = ""):
        self.bearer_token = bearer_token
        self.enabled = bool(bearer_token)

    def scrape(self, ticker: str, limit: int = 50) -> List[Dict]:
        """Fetch tweets for a ticker."""
        if not self.enabled:
            logger.debug("Twitter scraper disabled - no API key (requires $100+/month)")
            return []

        # Twitter API v2 implementation would go here
        # For now, return empty list
        logger.info("Twitter scraping not implemented - API is paid")
        return []


class SocialMediaAggregator:
    """
    Aggregates posts from all social media platforms.
    """

    def __init__(
        self,
        reddit_client_id: str = "",
        reddit_client_secret: str = "",
        reddit_user_agent: str = "StockAssistant/1.0",
        twitter_bearer_token: str = ""
    ):
        self.scrapers = {
            "stocktwits": StockTwitsScraper(),
            "reddit": RedditScraper(reddit_client_id, reddit_client_secret, reddit_user_agent),
            "twitter": TwitterScraper(twitter_bearer_token)
        }

    def scrape_all(self, ticker: str, limit_per_platform: int = 30) -> Dict[str, List[Dict]]:
        """
        Scrape posts from all available platforms.

        Args:
            ticker: Stock ticker symbol
            limit_per_platform: Max posts per platform

        Returns:
            Dict mapping platform name to list of posts
        """
        results = {}

        for platform, scraper in self.scrapers.items():
            try:
                posts = scraper.scrape(ticker, limit=limit_per_platform)
                results[platform] = posts
            except Exception as e:
                logger.error(f"Failed to scrape {platform}: {e}")
                results[platform] = []

        return results

    def scrape_all_combined(self, ticker: str, total_limit: int = 50) -> List[Dict]:
        """
        Scrape from all platforms and return combined list sorted by recency.

        Args:
            ticker: Stock ticker symbol
            total_limit: Total max posts across all platforms

        Returns:
            List of posts sorted by timestamp (newest first)
        """
        per_platform = max(1, total_limit // 3)  # Distribute across 3 platforms
        results = self.scrape_all(ticker, limit_per_platform=per_platform)

        all_posts = []
        for posts in results.values():
            all_posts.extend(posts)

        # Sort by timestamp (newest first)
        all_posts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return all_posts[:total_limit]

    def get_source_counts(self, posts: List[Dict]) -> Dict[str, int]:
        """Count posts by platform."""
        counts = {"stocktwits": 0, "reddit": 0, "twitter": 0}
        for post in posts:
            platform = post.get("platform", "")
            if platform in counts:
                counts[platform] += 1
        return counts
