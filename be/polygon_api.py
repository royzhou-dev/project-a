import requests
from config import POLYGON_API_KEY

BASE_URL = "https://api.polygon.io"

class PolygonAPI:
    def __init__(self):
        self.api_key = POLYGON_API_KEY

    def get_ticker_details(self, ticker):
        """Get detailed information about a ticker"""
        url = f"{BASE_URL}/v3/reference/tickers/{ticker}"
        params = {"apiKey": self.api_key}
        response = requests.get(url, params=params)
        return response.json()

    def get_previous_close(self, ticker):
        """Get previous day's close data"""
        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/prev"
        params = {"adjusted": "true", "apiKey": self.api_key}
        response = requests.get(url, params=params)
        return response.json()

    def get_aggregates(self, ticker, timespan="day", from_date=None, to_date=None):
        """Get aggregate bars for a ticker over a given date range"""
        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/{timespan}/{from_date}/{to_date}"
        params = {"adjusted": "true", "sort": "asc", "apiKey": self.api_key}
        response = requests.get(url, params=params)
        return response.json()

    def get_ticker_news(self, ticker, limit=10):
        """Get news articles for a ticker"""
        url = f"{BASE_URL}/v2/reference/news"
        params = {
            "ticker": ticker,
            "limit": limit,
            "apiKey": self.api_key
        }
        response = requests.get(url, params=params)
        return response.json()

    def get_financials(self, ticker):
        """Get financial data for a ticker"""
        url = f"{BASE_URL}/vX/reference/financials"
        params = {
            "ticker": ticker,
            "limit": 4,
            "apiKey": self.api_key
        }
        response = requests.get(url, params=params)
        return response.json()

    def get_snapshot(self, ticker):
        """Get current snapshot of a ticker"""
        url = f"{BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        params = {"apiKey": self.api_key}
        response = requests.get(url, params=params)
        return response.json()

    def get_dividends(self, ticker, limit=10):
        """Get dividend history for a ticker"""
        url = f"{BASE_URL}/v3/reference/dividends"
        params = {
            "ticker": ticker,
            "limit": limit,
            "order": "desc",
            "apiKey": self.api_key
        }
        response = requests.get(url, params=params)
        return response.json()

    def get_splits(self, ticker, limit=10):
        """Get stock split history for a ticker"""
        url = f"{BASE_URL}/v3/reference/splits"
        params = {
            "ticker": ticker,
            "limit": limit,
            "order": "desc",
            "apiKey": self.api_key
        }
        response = requests.get(url, params=params)
        return response.json()

    def get_market_status(self):
        """Get current market status"""
        url = f"{BASE_URL}/v1/marketstatus/now"
        params = {"apiKey": self.api_key}
        response = requests.get(url, params=params)
        return response.json()

    def get_related_companies(self, ticker):
        """Get related/similar companies"""
        url = f"{BASE_URL}/v1/related-companies/{ticker}"
        params = {"apiKey": self.api_key}
        response = requests.get(url, params=params)
        return response.json()
