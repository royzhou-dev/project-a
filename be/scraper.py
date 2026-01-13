import requests
from bs4 import BeautifulSoup
import re
import json


class ArticleScraper:
    """Web scraper for extracting full article content from news URLs"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def scrape_article(self, url, timeout=10):
        """
        Scrape full article content from a news URL

        Args:
            url: Article URL to scrape
            timeout: Request timeout in seconds

        Returns:
            Cleaned article text or None if scraping fails
        """
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'lxml')

            # Try multiple extraction methods in order of reliability
            article_content = (
                self._extract_by_schema(soup) or
                self._extract_by_selector(soup, 'article') or
                self._extract_by_selector(soup, '.article-body') or
                self._extract_by_selector(soup, '.article-content') or
                self._extract_by_selector(soup, '#article-content') or
                self._extract_by_selector(soup, '.story-body') or
                self._extract_by_selector(soup, '.entry-content') or
                self._extract_paragraphs(soup)
            )

            if article_content:
                return self._clean_text(article_content)
            else:
                print(f"Could not extract content from {url}")
                return None

        except requests.exceptions.Timeout:
            print(f"Timeout scraping {url}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error scraping {url}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error scraping {url}: {e}")
            return None

    def _extract_by_selector(self, soup, selector):
        """Extract text from a CSS selector"""
        element = soup.select_one(selector)
        if element:
            paragraphs = element.find_all('p')
            if paragraphs:
                return ' '.join(p.get_text() for p in paragraphs)
        return None

    def _extract_by_schema(self, soup):
        """Extract article body from JSON-LD schema.org metadata"""
        script_tags = soup.find_all('script', type='application/ld+json')

        for script_tag in script_tags:
            try:
                data = json.loads(script_tag.string)

                # Handle both single objects and arrays
                if isinstance(data, list):
                    for item in data:
                        if self._extract_article_body(item):
                            return self._extract_article_body(item)
                else:
                    if self._extract_article_body(data):
                        return self._extract_article_body(data)
            except (json.JSONDecodeError, AttributeError):
                continue

        return None

    def _extract_article_body(self, data):
        """Extract articleBody from JSON-LD data"""
        if isinstance(data, dict):
            if data.get('@type') in ['Article', 'NewsArticle', 'BlogPosting']:
                return data.get('articleBody')
        return None

    def _extract_paragraphs(self, soup):
        """Fallback: Extract all paragraph tags from body"""
        # Remove script, style, nav, footer, and header elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()

        # Find all paragraphs
        paragraphs = soup.find_all('p')

        if len(paragraphs) >= 3:  # Only use if we found a reasonable number of paragraphs
            text = ' '.join(p.get_text() for p in paragraphs)
            # Only return if we got substantial content
            if len(text) > 200:
                return text

        return None

    def _clean_text(self, text):
        """Clean and normalize extracted text"""
        if not text:
            return None

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove common cruft
        text = re.sub(r'(Advertisement|ADVERTISEMENT)', '', text)
        text = re.sub(r'(Read more:.*?\.)', '', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        # Only return if we have substantial content
        if len(text) > 100:
            return text

        return None
