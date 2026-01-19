"""
FinBERT-based sentiment analyzer for financial text.
Uses ProsusAI/finbert model for classifying text as positive, negative, or neutral.
"""

import warnings
# Suppress huggingface_hub deprecation warning about resume_download
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Financial sentiment analyzer using FinBERT.

    The model is lazy-loaded on first use to avoid slow startup times.
    Supports both single text and batch analysis for efficiency.
    """

    MODEL_NAME = "ProsusAI/finbert"
    LABELS = ["negative", "neutral", "positive"]

    def __init__(self):
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._tokenizer: Optional[AutoTokenizer] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> None:
        """Lazy load the FinBERT model on first use."""
        if self._model is not None:
            return

        logger.info(f"Loading FinBERT model ({self.MODEL_NAME})...")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
            self._model.to(self._device)
            self._model.eval()
            logger.info(f"FinBERT model loaded successfully on {self._device}")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise

    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.

        Args:
            text: The financial text to analyze

        Returns:
            dict with keys:
                - label: "positive", "neutral", or "negative"
                - score: confidence score (0-1)
                - scores: dict of all label scores
        """
        if not text or not text.strip():
            return {
                "label": "neutral",
                "score": 1.0,
                "scores": {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
            }

        self._load_model()

        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            probs = probs.cpu()
            scores = {label: float(prob) for label, prob in zip(self.LABELS, probs[0])}
            predicted_idx = probs.argmax().item()

            return {
                "label": self.LABELS[predicted_idx],
                "score": float(probs[0][predicted_idx]),
                "scores": scores
            }

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                "label": "neutral",
                "score": 0.5,
                "scores": {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
            }

    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        """
        Analyze sentiment of multiple texts efficiently.

        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process at once

        Returns:
            List of sentiment dicts (same format as analyze())
        """
        if not texts:
            return []

        self._load_model()

        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Filter out empty texts, keeping track of indices
            valid_indices = []
            valid_texts = []
            for j, text in enumerate(batch_texts):
                if text and text.strip():
                    valid_indices.append(j)
                    valid_texts.append(text)

            # Initialize results for this batch with neutral defaults
            batch_results = [{
                "label": "neutral",
                "score": 1.0,
                "scores": {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
            } for _ in batch_texts]

            if not valid_texts:
                results.extend(batch_results)
                continue

            try:
                inputs = self._tokenizer(
                    valid_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                probs = probs.cpu()

                for idx, (valid_idx, prob) in enumerate(zip(valid_indices, probs)):
                    scores = {label: float(p) for label, p in zip(self.LABELS, prob)}
                    predicted_idx = prob.argmax().item()
                    batch_results[valid_idx] = {
                        "label": self.LABELS[predicted_idx],
                        "score": float(prob[predicted_idx]),
                        "scores": scores
                    }

            except Exception as e:
                logger.error(f"Batch sentiment analysis failed: {e}")

            results.extend(batch_results)

        return results

    def convert_to_aggregate_score(self, sentiment: Dict) -> float:
        """
        Convert sentiment dict to a single score from -1 (bearish) to +1 (bullish).

        Args:
            sentiment: Sentiment dict from analyze()

        Returns:
            float: Score from -1 to +1
        """
        scores = sentiment.get("scores", {})
        positive = scores.get("positive", 0)
        negative = scores.get("negative", 0)
        return positive - negative

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("FinBERT model unloaded")


# Singleton instance for reuse
_analyzer_instance: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get or create the singleton sentiment analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SentimentAnalyzer()
    return _analyzer_instance
