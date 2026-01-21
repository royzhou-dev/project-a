"""
Service layer for stock price forecasting.
Orchestrates data fetching, model training, and predictions.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from polygon_api import PolygonAPI
from forecast_model import get_stock_forecaster, StockForecaster

logger = logging.getLogger(__name__)


class ForecastService:
    """
    Service for stock price forecasting.
    Manages model training, prediction, and caching.
    """

    TRAINING_DATA_YEARS = 2  # Years of historical data for training

    def __init__(self):
        self.forecaster: StockForecaster = get_stock_forecaster()
        self.polygon = PolygonAPI()
        self._forecast_cache: Dict[str, Dict] = {}  # {ticker: {timestamp, forecast}}
        self.cache_ttl_minutes = 60

    def get_forecast(self, ticker: str, force_retrain: bool = False, historical_data: list = None) -> Dict:
        """
        Get forecast for a ticker. Auto-trains if no model exists.

        Args:
            ticker: Stock ticker symbol
            force_retrain: If True, retrain even if model exists
            historical_data: Optional pre-fetched OHLCV data from frontend cache

        Returns:
            Dict with forecast data
        """
        ticker = ticker.upper()

        # Check if we need to train
        needs_training = force_retrain or not self.forecaster.has_model(ticker)

        if needs_training:
            logger.info(f"Training model for {ticker}...")
            training_result = self.train_model(ticker, historical_data)
            if training_result.get("status") != "training_complete":
                return {"error": "Training failed", "details": training_result}

        # Use provided data or fetch if not available
        if historical_data and len(historical_data) >= 60:
            recent_data = historical_data
        else:
            recent_data = self._fetch_recent_data(ticker)
            if not recent_data:
                return {"error": "Failed to fetch recent data for prediction"}

        # Generate forecast
        try:
            forecast_result = self.forecaster.predict(ticker, recent_data)
        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")
            return {"error": f"Prediction failed: {str(e)}"}

        # Get model metadata
        metadata = self.forecaster.get_model_metadata(ticker)

        # Format response with dates
        forecast = self._format_forecast(forecast_result, recent_data)

        return {
            "ticker": ticker,
            "forecast": forecast,
            "model_info": metadata,
            "historical": self._format_historical(recent_data[-60:]),  # Last 60 days for chart
            "confidence_bounds": {
                "upper": forecast_result["upper_bound"],
                "lower": forecast_result["lower_bound"]
            }
        }

    def train_model(self, ticker: str, historical_data: list = None) -> Dict:
        """
        Train/retrain model for a ticker.

        Args:
            ticker: Stock ticker symbol
            historical_data: Optional pre-fetched OHLCV data from frontend cache

        Returns:
            Dict with training results
        """
        ticker = ticker.upper()

        # Use provided data or fetch if not available
        if historical_data and len(historical_data) >= 100:
            training_data = historical_data
        else:
            training_data = self._fetch_training_data(ticker)
            if not training_data:
                return {"error": "Failed to fetch training data", "ticker": ticker}

        if len(training_data) < 100:
            return {
                "error": "Insufficient historical data",
                "ticker": ticker,
                "data_points": len(training_data),
                "required": 100
            }

        # Train the model
        try:
            result = self.forecaster.train(ticker, training_data)
            return result
        except Exception as e:
            logger.error(f"Training failed for {ticker}: {e}")
            return {"error": f"Training failed: {str(e)}", "ticker": ticker}

    def _fetch_training_data(self, ticker: str) -> Optional[list]:
        """Fetch historical data for training (2 years)."""
        to_date = datetime.now()
        from_date = to_date - timedelta(days=self.TRAINING_DATA_YEARS * 365)

        try:
            response = self.polygon.get_aggregates(
                ticker,
                timespan="day",
                from_date=from_date.strftime("%Y-%m-%d"),
                to_date=to_date.strftime("%Y-%m-%d")
            )

            if "results" not in response or not response["results"]:
                logger.error(f"Failed to fetch training data for {ticker}: {response}")
                return None

            return response["results"]

        except Exception as e:
            logger.error(f"Error fetching training data for {ticker}: {e}")
            return None

    def _fetch_recent_data(self, ticker: str) -> Optional[list]:
        """Fetch recent data for prediction (90 days to ensure enough data)."""
        to_date = datetime.now()
        from_date = to_date - timedelta(days=90)

        try:
            response = self.polygon.get_aggregates(
                ticker,
                timespan="day",
                from_date=from_date.strftime("%Y-%m-%d"),
                to_date=to_date.strftime("%Y-%m-%d")
            )

            if "results" not in response or not response["results"]:
                logger.error(f"Failed to fetch recent data for {ticker}: {response}")
                return None

            return response["results"]

        except Exception as e:
            logger.error(f"Error fetching recent data for {ticker}: {e}")
            return None

    def _format_forecast(self, forecast_result: Dict, historical_data: list) -> list:
        """Format forecast with dates."""
        predictions = forecast_result["predictions"]
        upper = forecast_result["upper_bound"]
        lower = forecast_result["lower_bound"]

        # Start from the last historical date
        if historical_data:
            last_timestamp = historical_data[-1].get("t", 0)
            last_date = datetime.fromtimestamp(last_timestamp / 1000)
        else:
            last_date = datetime.now()

        forecast = []
        current_date = last_date

        for i, pred in enumerate(predictions):
            # Skip weekends
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                current_date += timedelta(days=1)

            forecast.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "predicted_close": round(pred, 2),
                "upper_bound": round(upper[i], 2),
                "lower_bound": round(lower[i], 2),
                "day": i + 1
            })

        return forecast

    def _format_historical(self, data: list) -> list:
        """Format historical data for chart display."""
        return [
            {
                "date": datetime.fromtimestamp(d["t"] / 1000).strftime("%Y-%m-%d"),
                "close": d["c"],
                "open": d["o"],
                "high": d["h"],
                "low": d["l"],
                "volume": d["v"]
            }
            for d in data
        ]

    def get_model_status(self, ticker: str) -> Dict:
        """Get model status for a ticker."""
        ticker = ticker.upper()
        has_model = self.forecaster.has_model(ticker)
        metadata = self.forecaster.get_model_metadata(ticker) if has_model else None

        return {
            "ticker": ticker,
            "model_exists": has_model,
            "metadata": metadata
        }


# Singleton instance
_forecast_service: Optional[ForecastService] = None


def get_forecast_service() -> ForecastService:
    """Get or create the singleton forecast service instance."""
    global _forecast_service
    if _forecast_service is None:
        _forecast_service = ForecastService()
    return _forecast_service
