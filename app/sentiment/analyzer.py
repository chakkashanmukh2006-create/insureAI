"""
Sentiment analysis for customer feedback text.

Provides a dual-strategy analyser:
  - Primary: HuggingFace Transformers (DistilBERT SST-2)
  - Fallback: VADER lexicon-based analyser
"""

from typing import Optional

from app.config.settings import settings
from app.utils.logger import logger


class SentimentAnalyzer:
    """Analyses sentiment of customer feedback text.

    On initialisation the class attempts to load the HuggingFace
    ``distilbert-base-uncased-finetuned-sst-2-english`` pipeline.
    If that fails (missing dependency, network error, etc.) it
    transparently falls back to VADER.

    Usage::

        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("Great service!")
        # {'sentiment': 'Positive', 'sentiment_score': 0.9987, 'confidence_score': 0.9987}
    """

    def __init__(self) -> None:
        self.use_huggingface: bool = settings.USE_HUGGINGFACE
        self._pipeline: Optional[object] = None
        self._vader: Optional[object] = None
        self._backend: str = "none"
        self._initialize()

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def _initialize(self) -> None:
        """Attempt to load the preferred backend, falling back as needed."""
        if self.use_huggingface:
            try:
                from transformers import pipeline  # type: ignore[import-untyped]

                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1,  # force CPU
                )
                self._backend = "huggingface"
                logger.info("HuggingFace sentiment pipeline initialised.")
                return
            except Exception as exc:
                logger.warning(
                    f"HuggingFace unavailable, falling back to VADER: {exc}"
                )

        # Fallback to VADER
        try:
            from vaderSentiment.vaderSentiment import (  # type: ignore[import-untyped]
                SentimentIntensityAnalyzer,
            )

            self._vader = SentimentIntensityAnalyzer()
            self._backend = "vader"
            logger.info("VADER sentiment analyser initialised.")
        except Exception as exc:
            logger.error(
                f"Failed to initialise VADER sentiment analyser: {exc}. "
                "Sentiment analysis will return neutral defaults."
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def analyze(self, text: Optional[str]) -> dict:
        """Analyse sentiment of *text*.

        Args:
            text: Free-form feedback string.

        Returns:
            Dictionary with keys:
            - ``sentiment``: ``'Positive'`` | ``'Neutral'`` | ``'Negative'``
            - ``sentiment_score``: float in ``[-1, 1]``
            - ``confidence_score``: float in ``[0, 1]``
        """
        if not text or not str(text).strip():
            return self._neutral_result()

        text = str(text).strip()

        if self._pipeline is not None:
            return self._analyze_huggingface(text)
        if self._vader is not None:
            return self._analyze_vader(text)

        return self._neutral_result()

    @property
    def backend(self) -> str:
        """Return the name of the active backend (``'huggingface'``, ``'vader'``, or ``'none'``)."""
        return self._backend

    # ------------------------------------------------------------------ #
    # Backend implementations
    # ------------------------------------------------------------------ #

    def _analyze_huggingface(self, text: str) -> dict:
        """Run inference through the HuggingFace pipeline."""
        try:
            # Truncate to 512 chars to stay within token limits
            result = self._pipeline(text[:512])[0]
            label: str = result['label']       # POSITIVE / NEGATIVE
            score: float = result['score']     # confidence ∈ [0, 1]

            if score < 0.6:
                # Low confidence → treat as neutral
                return {
                    'sentiment': 'Neutral',
                    'sentiment_score': 0.0,
                    'confidence_score': round(score, 4),
                }

            if label == 'POSITIVE':
                return {
                    'sentiment': 'Positive',
                    'sentiment_score': round(score, 4),
                    'confidence_score': round(score, 4),
                }
            else:
                return {
                    'sentiment': 'Negative',
                    'sentiment_score': round(-score, 4),
                    'confidence_score': round(score, 4),
                }
        except Exception as exc:
            logger.error(f"HuggingFace inference failed: {exc}")
            # Attempt VADER fallback at runtime
            if self._vader is not None:
                return self._analyze_vader(text)
            return self._neutral_result()

    def _analyze_vader(self, text: str) -> dict:
        """Run inference through VADER."""
        try:
            scores = self._vader.polarity_scores(text)
            compound: float = scores['compound']

            if compound >= 0.05:
                sentiment = 'Positive'
            elif compound <= -0.05:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'

            return {
                'sentiment': sentiment,
                'sentiment_score': round(compound, 4),
                'confidence_score': round(abs(compound), 4),
            }
        except Exception as exc:
            logger.error(f"VADER inference failed: {exc}")
            return self._neutral_result()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _neutral_result() -> dict:
        """Return a safe neutral default."""
        return {
            'sentiment': 'Neutral',
            'sentiment_score': 0.0,
            'confidence_score': 0.0,
        }
