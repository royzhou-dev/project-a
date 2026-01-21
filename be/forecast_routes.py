"""
Flask routes for stock price forecasting.
"""

from flask import Blueprint, request, jsonify
import logging

from forecast_service import get_forecast_service

logger = logging.getLogger(__name__)

forecast_bp = Blueprint('forecast', __name__, url_prefix='/api/forecast')


@forecast_bp.route('/predict/<ticker>', methods=['POST'])
def get_forecast(ticker: str):
    """
    Get price forecast for a ticker. Auto-trains if no model exists.

    Request body (optional):
        {
            "force_retrain": false,
            "historical_data": [...]  // Optional: pre-fetched OHLCV data from frontend
        }

    Response:
        {
            "ticker": "AAPL",
            "forecast": [...],
            "model_info": {...},
            "historical": [...],
            "confidence_bounds": {...}
        }
    """
    try:
        data = request.get_json() or {}
        force_retrain = data.get('force_retrain', False)
        historical_data = data.get('historical_data', None)

        service = get_forecast_service()
        result = service.get_forecast(ticker.upper(), force_retrain=force_retrain, historical_data=historical_data)

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error getting forecast for {ticker}: {e}")
        return jsonify({"error": str(e)}), 500


@forecast_bp.route('/train/<ticker>', methods=['POST'])
def train_model(ticker: str):
    """
    Force train/retrain model for a ticker.

    Request body (optional):
        { "historical_data": [...] }  // Optional: pre-fetched OHLCV data

    Response:
        {
            "status": "training_complete",
            "ticker": "AAPL",
            "epochs": 50,
            "final_loss": 0.0023
        }
    """
    try:
        data = request.get_json() or {}
        historical_data = data.get('historical_data', None)

        service = get_forecast_service()
        result = service.train_model(ticker.upper(), historical_data=historical_data)

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error training model for {ticker}: {e}")
        return jsonify({"error": str(e)}), 500


@forecast_bp.route('/status/<ticker>', methods=['GET'])
def model_status(ticker: str):
    """
    Check model status for a ticker.

    Response:
        {
            "ticker": "AAPL",
            "model_exists": true,
            "metadata": {...}
        }
    """
    try:
        service = get_forecast_service()
        result = service.get_model_status(ticker.upper())
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error getting model status for {ticker}: {e}")
        return jsonify({"error": str(e)}), 500


@forecast_bp.route('/health', methods=['GET'])
def forecast_health():
    """
    Health check for forecast service.

    Response:
        {
            "status": "healthy",
            "service": "forecast",
            "model_dir_exists": true
        }
    """
    try:
        import os
        from forecast_model import StockForecaster

        model_dir = StockForecaster.MODEL_DIR
        model_dir_exists = os.path.exists(model_dir)

        return jsonify({
            "status": "healthy",
            "service": "forecast",
            "model_dir": model_dir,
            "model_dir_exists": model_dir_exists
        }), 200

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "service": "forecast",
            "error": str(e)
        }), 500
