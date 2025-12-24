"""
Feature Extraction Module

This module contains TF-IDF feature extraction for text data.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import List, Union, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sentiment_analysis.utils import load_config, get_config_value
from sentiment_analysis.exceptions import (
    ModelNotFittedError,
    ModelLoadError,
    ModelSaveError,
)

logger = logging.getLogger(__name__)

# Load configuration defaults at module level
_config = load_config()
DEFAULT_MAX_FEATURES = get_config_value(
    _config, "model", "classical", "max_features", default=5000
)
_ngram_range_list = get_config_value(
    _config, "model", "classical", "ngram_range", default=[1, 1]
)
DEFAULT_NGRAM_RANGE = tuple(_ngram_range_list)


class TFIDFFeatureExtractor:
    """
    TF-IDF feature extraction for text data.
    """

    def __init__(
        self,
        max_features: int = DEFAULT_MAX_FEATURES,
        ngram_range: Tuple[int, int] = DEFAULT_NGRAM_RANGE,
    ):
        """
        Initialize TF-IDF vectorizer.

        Args:
            max_features: Maximum number of features (default from config)
            ngram_range: Range of n-grams to extract (default from config)
        """
        if max_features < 1:
            raise ValueError(f"max_features must be positive, got {max_features}")
        if not isinstance(ngram_range, tuple) or len(ngram_range) != 2:
            raise ValueError(
                f"ngram_range must be a tuple of 2 integers, got {ngram_range}"
            )
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range
        )
        self.fitted = False  # Track if fitted to prevent calling transform, save, or get_feature_names before fitting
        logger.debug(
            f"Initialized TFIDFFeatureExtractor (max_features={max_features}, ngram_range={ngram_range})"
        )

    def fit_transform(self, texts: List[str]):
        """Fit and transform texts to TF-IDF features."""
        if not texts or len(texts) == 0:
            raise ValueError("texts cannot be empty")

        logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} texts")
        result = self.vectorizer.fit_transform(texts)
        self.fitted = True
        logger.info(f"TF-IDF features extracted: shape={result.shape}")
        return result

    def transform(self, texts: List[str]):
        """Transform texts to TF-IDF features using fitted vectorizer."""
        if not self.fitted:
            raise ModelNotFittedError("TFIDFFeatureExtractor")

        if not texts or len(texts) == 0:
            raise ValueError("texts cannot be empty")

        logger.debug(f"Transforming {len(texts)} texts to TF-IDF features")
        return self.vectorizer.transform(texts)

    def save(self, path: str):
        """Save the fitted vectorizer."""
        if not self.fitted:
            raise ModelNotFittedError("TFIDFFeatureExtractor")

        try:
            path = Path(path)
            path.parent.mkdir(
                parents=True, exist_ok=True
            )  # create directories if not exist and don't raise error if they do
            joblib.dump(self.vectorizer, path)
            logger.info(f"TF-IDF vectorizer saved to {path}")
        except Exception as e:
            raise ModelSaveError(str(path), str(e))

    def load(self, path: str):
        """Load a fitted vectorizer."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {path}")

        try:
            self.vectorizer = joblib.load(path)
            self.fitted = True
            logger.info(f"TF-IDF vectorizer loaded from {path}")
        except Exception as e:
            raise ModelLoadError(str(path), str(e))

    def get_feature_names(self):
        """Get feature names (vocabulary)."""
        if not self.fitted:
            raise ModelNotFittedError("TFIDFFeatureExtractor")
        return self.vectorizer.get_feature_names_out()
