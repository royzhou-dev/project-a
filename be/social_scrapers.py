"""
Social media scrapers for stock sentiment data.
Supports StockTwits, Reddit, and Twitter (optional).
"""

import requests
import cloudscraper
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
    - Uses cloudscraper to bypass Cloudflare protection
    """

    BASE_URL = "https://api.stocktwits.com/api/2"
    TIMEOUT = 10

    def __init__(self):
        """Initialize with cloudscraper session to handle Cloudflare."""
        self.scraper = cloudscraper.create_scraper()

    def scrape(self, ticker: str, limit: int = 50) -> List[Dict]:
        """Fetch posts from StockTwits for a ticker."""
        url = f"{self.BASE_URL}/streams/symbol/{ticker.upper()}.json"

        try:
            response = self.scraper.get(url, timeout=self.TIMEOUT)
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
    Direct Reddit scraper using public JSON endpoints.

    - No API credentials required
    - Uses cloudscraper to bypass Cloudflare
    - Subreddits: wallstreetbets, stocks, investing, options
    """

    SUBREDDITS = ["wallstreetbets", "stocks", "investing", "options"]
    BASE_URL = "https://www.reddit.com"
    TIMEOUT = 10

    def __init__(self, client_id: str = "", client_secret: str = "", user_agent: str = "StockAssistant/1.0"):
        # Credentials no longer needed, but keep params for backwards compatibility
        self.scraper = cloudscraper.create_scraper()
        self.user_agent = user_agent

    def scrape(self, ticker: str, limit: int = 50) -> List[Dict]:
        """Fetch posts from Reddit for a ticker using public JSON endpoints."""
        posts = []
        seen_ids = set()
        per_subreddit = max(1, limit // len(self.SUBREDDITS))

        for subreddit_name in self.SUBREDDITS:
            if len(posts) >= limit:
                break

            # Search with $ prefix (common in finance subs) and plain ticker
            for query in [f"${ticker.upper()}", ticker.upper()]:
                if len(posts) >= limit:
                    break

                try:
                    url = f"{self.BASE_URL}/r/{subreddit_name}/search.json"
                    params = {
                        "q": query,
                        "limit": per_subreddit,
                        "t": "week",
                        "sort": "relevance",
                        "restrict_sr": "true"
                    }

                    response = self.scraper.get(
                        url,
                        params=params,
                        timeout=self.TIMEOUT,
                        headers={"User-Agent": self.user_agent}
                    )
                    response.raise_for_status()
                    data = response.json()

                    children = data.get("data", {}).get("children", [])

                    for child in children:
                        if len(posts) >= limit:
                            break

                        post_data = child.get("data", {})
                        post_id = post_data.get("id")

                        if post_id and post_id not in seen_ids:
                            post = self._standardize_post(post_data, ticker, subreddit_name)
                            if post:
                                posts.append(post)
                                seen_ids.add(post_id)

                except requests.exceptions.Timeout:
                    logger.warning(f"Reddit request timed out for r/{subreddit_name}")
                    continue
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Reddit scraping error for r/{subreddit_name}: {e}")
                    continue
                except Exception as e:
                    logger.debug(f"Reddit search failed for {query} in r/{subreddit_name}: {e}")
                    continue

        logger.info(f"Scraped {len(posts)} posts from Reddit for {ticker}")
        return posts[:limit]

    def _standardize_post(self, post_data: Dict, ticker: str, subreddit: str) -> Optional[Dict]:
        """Convert Reddit JSON post to standard format."""
        try:
            post_id = post_data.get("id")
            if not post_id:
                return None

            title = post_data.get("title", "").strip()
            selftext = post_data.get("selftext", "").strip()

            # Combine title and selftext
            if selftext and selftext != "[removed]" and selftext != "[deleted]":
                content = f"{title}\n\n{selftext}"
            else:
                content = title

            if not content:
                return None

            # Truncate very long posts
            if len(content) > 2000:
                content = content[:2000] + "..."

            # Parse timestamp
            created_utc = post_data.get("created_utc", 0)
            timestamp = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()

            author = post_data.get("author", "[deleted]")
            if author == "[deleted]":
                author = "[deleted]"

            score = post_data.get("score", 0)
            num_comments = post_data.get("num_comments", 0)
            permalink = post_data.get("permalink", "")

            return {
                "id": self.generate_post_id("reddit", post_id),
                "platform": "reddit",
                "ticker": ticker.upper(),
                "content": content,
                "author": author,
                "author_followers": 0,
                "timestamp": timestamp,
                "likes": score,
                "comments": num_comments,
                "retweets": 0,
                "engagement_score": self.calculate_engagement_score(score, num_comments),
                "url": f"https://reddit.com{permalink}",
                "subreddit": subreddit
            }

        except Exception as e:
            logger.debug(f"Failed to parse Reddit post: {e}")
            return None


class TwitterScraper(BaseScraper):
    """
    Twitter/X API v2 scraper.

    NOTE: Twitter API v2 requires paid access ($100+/month for Basic tier).
    This scraper is disabled by default unless TWITTER_BEARER_TOKEN is set.

    - Rate limit (Basic tier): 10,000 tweets/month
    - Endpoint: https://api.twitter.com/2/tweets/search/recent
    - Search window: Last 7 days only (recent search)
    """

    BASE_URL = "https://api.twitter.com/2"
    TIMEOUT = 15

    def __init__(self, bearer_token: str = ""):
        self.bearer_token = bearer_token
        self.enabled = bool(bearer_token)

    def scrape(self, ticker: str, limit: int = 50) -> List[Dict]:
        """
        Fetch tweets for a ticker using Twitter API v2 recent search.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            limit: Maximum number of tweets to return (max 100 per request)

        Returns:
            List of standardized post dicts
        """
        if not self.enabled:
            logger.debug("Twitter scraper disabled - no bearer token (requires $100+/month)")
            return []

        # Build search query for stock ticker
        # Search for cashtag ($AAPL) and common stock-related terms
        query = f"${ticker.upper()} (stock OR shares OR trading OR buy OR sell OR price) -is:retweet lang:en"

        # Limit to max 100 per request (Twitter API limit)
        max_results = min(limit, 100)

        url = f"{self.BASE_URL}/tweets/search/recent"
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
        params = {
            "query": query,
            "max_results": max_results,
            "tweet.fields": "created_at,public_metrics,author_id,conversation_id",
            "user.fields": "username,name,public_metrics",
            "expansions": "author_id"
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=self.TIMEOUT)

            # Handle rate limiting
            if response.status_code == 429:
                logger.warning("Twitter API rate limit reached")
                return []

            # Handle authentication errors
            if response.status_code == 401:
                logger.error("Twitter API authentication failed - check bearer token")
                return []

            if response.status_code == 403:
                logger.error("Twitter API access forbidden - may need higher tier access")
                return []

            response.raise_for_status()
            data = response.json()

            # Check for errors in response
            if "errors" in data and not data.get("data"):
                for error in data.get("errors", []):
                    logger.warning(f"Twitter API error: {error.get('message', 'Unknown error')}")
                return []

            tweets = data.get("data", [])
            if not tweets:
                logger.info(f"No tweets found for {ticker}")
                return []

            # Build user lookup map from expansions
            users = {}
            includes = data.get("includes", {})
            for user in includes.get("users", []):
                users[user["id"]] = user

            # Parse tweets
            posts = []
            for tweet in tweets:
                try:
                    post = self._standardize_post(tweet, ticker, users)
                    if post:
                        posts.append(post)
                except Exception as e:
                    logger.debug(f"Failed to parse tweet: {e}")
                    continue

            logger.info(f"Scraped {len(posts)} tweets from Twitter for {ticker}")
            return posts

        except requests.exceptions.Timeout:
            logger.warning(f"Twitter request timed out for {ticker}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Twitter API error for {ticker}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected Twitter error for {ticker}: {e}")
            return []

    def _standardize_post(self, tweet: Dict, ticker: str, users: Dict) -> Optional[Dict]:
        """
        Convert Twitter API v2 tweet to standard format.

        Args:
            tweet: Raw tweet data from API
            ticker: Stock ticker symbol
            users: User lookup map from expansions

        Returns:
            Standardized post dict or None
        """
        tweet_id = tweet.get("id")
        if not tweet_id:
            return None

        text = tweet.get("text", "").strip()
        if not text:
            return None

        # Get author info from users map
        author_id = tweet.get("author_id", "")
        author_info = users.get(author_id, {})
        username = author_info.get("username", "unknown")
        author_metrics = author_info.get("public_metrics", {})
        followers = author_metrics.get("followers_count", 0)

        # Parse timestamp
        created_at = tweet.get("created_at", "")
        timestamp = None
        if created_at:
            try:
                # Twitter API v2 uses ISO 8601 format
                timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00")).isoformat()
            except Exception:
                timestamp = created_at

        # Get engagement metrics
        metrics = tweet.get("public_metrics", {})
        likes = metrics.get("like_count", 0)
        retweets = metrics.get("retweet_count", 0)
        replies = metrics.get("reply_count", 0)
        quotes = metrics.get("quote_count", 0)

        return {
            "id": self.generate_post_id("twitter", tweet_id),
            "platform": "twitter",
            "ticker": ticker.upper(),
            "content": text,
            "author": username,
            "author_followers": followers,
            "timestamp": timestamp,
            "likes": likes,
            "comments": replies,
            "retweets": retweets + quotes,
            "engagement_score": self.calculate_engagement_score(likes, replies, retweets + quotes),
            "url": f"https://twitter.com/{username}/status/{tweet_id}"
        }


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
