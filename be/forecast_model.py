"""
LSTM-based stock price forecaster.
Uses historical OHLCV data to predict future closing prices.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM neural network for time series forecasting."""

    def __init__(self, input_size: int = 5, hidden_size: int = 128,
                 num_layers: int = 2, output_size: int = 30, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        # Predict future prices
        predictions = self.fc(last_output)
        return predictions


class MinMaxScaler:
    """Simple MinMax scaler for normalization."""

    def __init__(self):
        self.min_vals = None
        self.max_vals = None
        self.fitted = False

    def fit(self, data: np.ndarray) -> 'MinMaxScaler':
        self.min_vals = data.min(axis=0)
        self.max_vals = data.max(axis=0)
        # Avoid division by zero
        self.range_vals = self.max_vals - self.min_vals
        self.range_vals[self.range_vals == 0] = 1
        self.fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return (data - self.min_vals) / self.range_vals

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: np.ndarray, col_idx: int = 0) -> np.ndarray:
        """Inverse transform for a single column (default: close price at index 0)."""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return data * self.range_vals[col_idx] + self.min_vals[col_idx]


class StockForecaster:
    """
    Stock price forecaster using LSTM neural network.

    The model is lazy-loaded on first use to avoid slow startup times.
    Supports training on historical OHLCV data and predicting future prices.
    """

    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'forecast_models')

    def __init__(self, sequence_length: int = 60, forecast_horizon: int = 30):
        self._models: Dict[str, LSTMModel] = {}
        self._scalers: Dict[str, MinMaxScaler] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon

        # Ensure model directory exists
        os.makedirs(self.MODEL_DIR, exist_ok=True)

    def _get_ticker_dir(self, ticker: str) -> str:
        """Get the directory path for a ticker's model files."""
        return os.path.join(self.MODEL_DIR, ticker.upper())

    def _load_model(self, ticker: str) -> bool:
        """
        Load a trained model for a ticker from disk.

        Returns:
            True if model was loaded successfully, False otherwise.
        """
        ticker = ticker.upper()
        if ticker in self._models:
            return True

        ticker_dir = self._get_ticker_dir(ticker)
        model_path = os.path.join(ticker_dir, 'model.pt')
        scaler_path = os.path.join(ticker_dir, 'scaler.pkl')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return False

        try:
            logger.info(f"Loading forecast model for {ticker}...")

            # Load scaler
            with open(scaler_path, 'rb') as f:
                self._scalers[ticker] = pickle.load(f)

            # Build and load model
            model = LSTMModel(output_size=self.forecast_horizon)
            model.load_state_dict(torch.load(model_path, map_location=self._device, weights_only=True))
            model.to(self._device)
            model.eval()
            self._models[ticker] = model

            logger.info(f"Forecast model for {ticker} loaded successfully on {self._device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load forecast model for {ticker}: {e}")
            return False

    def _save_model(self, ticker: str, metadata: Dict) -> None:
        """Save trained model and scaler to disk."""
        ticker = ticker.upper()
        ticker_dir = self._get_ticker_dir(ticker)
        os.makedirs(ticker_dir, exist_ok=True)

        model_path = os.path.join(ticker_dir, 'model.pt')
        scaler_path = os.path.join(ticker_dir, 'scaler.pkl')
        metadata_path = os.path.join(ticker_dir, 'metadata.json')

        # Save model weights
        torch.save(self._models[ticker].state_dict(), model_path)

        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self._scalers[ticker], f)

        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Forecast model for {ticker} saved to {ticker_dir}")

    def _prepare_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare OHLCV data for training.

        Args:
            data: List of dicts with keys: o, h, l, c, v (open, high, low, close, volume)

        Returns:
            Tuple of (X, y) numpy arrays for training
        """
        # Extract OHLCV features - close first for easy inverse transform
        features = np.array([[d['c'], d['o'], d['h'], d['l'], d['v']] for d in data], dtype=np.float32)

        # Create sequences
        X, y = [], []
        for i in range(len(features) - self.sequence_length - self.forecast_horizon + 1):
            X.append(features[i:i + self.sequence_length])
            # Target: next forecast_horizon closing prices
            y.append(features[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon, 0])

        return np.array(X), np.array(y)

    def train(self, ticker: str, data: List[Dict], epochs: int = 50,
              learning_rate: float = 0.001, batch_size: int = 32) -> Dict:
        """
        Train the LSTM model on historical price data.

        Args:
            ticker: Stock ticker symbol
            data: List of OHLCV dicts (must have at least sequence_length + forecast_horizon entries)
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size

        Returns:
            Dict with training results (loss, metadata)
        """
        ticker = ticker.upper()

        if len(data) < self.sequence_length + self.forecast_horizon:
            raise ValueError(f"Insufficient data: need at least {self.sequence_length + self.forecast_horizon} data points")

        logger.info(f"Training forecast model for {ticker} with {len(data)} data points...")

        # Prepare data
        X, y = self._prepare_data(data)

        # Normalize features
        scaler = MinMaxScaler()
        X_flat = X.reshape(-1, X.shape[-1])
        scaler.fit(X_flat)
        X_scaled = np.array([scaler.transform(seq) for seq in X])

        # Normalize targets using close price stats
        y_scaled = (y - scaler.min_vals[0]) / scaler.range_vals[0]

        self._scalers[ticker] = scaler

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self._device)
        y_tensor = torch.FloatTensor(y_scaled).to(self._device)

        # Split train/validation (80/20)
        split_idx = int(len(X_tensor) * 0.8)
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

        # Build model
        model = LSTMModel(output_size=self.forecast_horizon)
        model.to(self._device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        best_val_loss = float('inf')
        train_losses = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0

            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / (len(X_train) // batch_size + 1)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

        self._models[ticker] = model

        # Save model and metadata
        metadata = {
            "ticker": ticker,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "training_epochs": epochs,
            "final_train_loss": float(train_losses[-1]),
            "final_val_loss": float(best_val_loss),
            "data_points": len(data),
            "sequence_length": self.sequence_length,
            "forecast_horizon": self.forecast_horizon,
            "model_version": "1.0"
        }

        self._save_model(ticker, metadata)

        logger.info(f"Training complete for {ticker}. Final val loss: {best_val_loss:.6f}")

        return {
            "status": "training_complete",
            "ticker": ticker,
            "epochs": epochs,
            "final_loss": float(train_losses[-1]),
            "validation_loss": float(best_val_loss),
            "data_points": len(data)
        }

    def predict(self, ticker: str, recent_data: List[Dict]) -> Dict:
        """
        Generate price forecast using trained model.

        Args:
            ticker: Stock ticker symbol
            recent_data: Most recent OHLCV data (at least sequence_length entries)

        Returns:
            Dict with predictions and confidence bounds
        """
        ticker = ticker.upper()

        # Load model if not in memory
        if ticker not in self._models:
            if not self._load_model(ticker):
                raise ValueError(f"No trained model found for {ticker}. Train the model first.")

        if len(recent_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points for prediction")

        # Use last sequence_length data points
        data = recent_data[-self.sequence_length:]
        features = np.array([[d['c'], d['o'], d['h'], d['l'], d['v']] for d in data], dtype=np.float32)

        # Normalize
        scaler = self._scalers[ticker]
        features_scaled = scaler.transform(features)

        # Predict
        model = self._models[ticker]
        model.eval()

        X = torch.FloatTensor(features_scaled).unsqueeze(0).to(self._device)

        with torch.no_grad():
            predictions_scaled = model(X).cpu().numpy()[0]

        # Inverse transform predictions
        predictions = scaler.inverse_transform(predictions_scaled, col_idx=0)

        # Calculate confidence bounds (simple approach: +/- percentage based on historical volatility)
        recent_closes = [d['c'] for d in recent_data[-30:]]
        volatility = np.std(recent_closes) / np.mean(recent_closes)
        confidence_pct = max(0.02, min(0.10, volatility * 2))  # 2-10% bounds

        upper_bound = predictions * (1 + confidence_pct)
        lower_bound = predictions * (1 - confidence_pct)

        # Get last date from data for generating forecast dates
        last_timestamp = recent_data[-1].get('t', 0)

        return {
            "predictions": predictions.tolist(),
            "upper_bound": upper_bound.tolist(),
            "lower_bound": lower_bound.tolist(),
            "last_timestamp": last_timestamp,
            "forecast_horizon": self.forecast_horizon
        }

    def has_model(self, ticker: str) -> bool:
        """Check if a trained model exists for the ticker."""
        ticker = ticker.upper()
        if ticker in self._models:
            return True

        ticker_dir = self._get_ticker_dir(ticker)
        return os.path.exists(os.path.join(ticker_dir, 'model.pt'))

    def get_model_metadata(self, ticker: str) -> Optional[Dict]:
        """Get metadata for a trained model."""
        ticker = ticker.upper()
        metadata_path = os.path.join(self._get_ticker_dir(ticker), 'metadata.json')

        if not os.path.exists(metadata_path):
            return None

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def unload_model(self, ticker: str) -> None:
        """Unload a model from memory to free resources."""
        ticker = ticker.upper()
        if ticker in self._models:
            del self._models[ticker]
            del self._scalers[ticker]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Forecast model for {ticker} unloaded")


# Singleton instance
_forecaster_instance: Optional[StockForecaster] = None


def get_stock_forecaster() -> StockForecaster:
    """Get or create the singleton stock forecaster instance."""
    global _forecaster_instance
    if _forecaster_instance is None:
        _forecaster_instance = StockForecaster()
    return _forecaster_instance
