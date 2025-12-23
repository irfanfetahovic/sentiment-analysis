"""
Inference Module

This module provides classes and functions for making predictions
with trained sentiment analysis models.
"""

import torch
import numpy as np
import json
import logging
from pathlib import Path
from typing import Union, List, Dict, Optional, Any
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    pipeline,
)
import joblib

from sentiment_analysis.text_preprocessing import TextPreprocessor
from sentiment_analysis.train_classical import ClassicalSentimentModel
from sentiment_analysis.constants import BINARY_LABEL_NAMES, THREE_CLASS_LABEL_NAMES

logger = logging.getLogger(__name__)


class SentimentPredictor:
    """
    Unified sentiment prediction interface for both transformer and classical models.
    """

    def __init__(
        self, model_path: str, model_type: str = "transformer", device: str = None
    ):
        """
        Initialize sentiment predictor.

        Args:
            model_path: Path to saved model
            model_type: Type of model ('transformer' or 'classical')
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.tokenizer = None
        self.preprocessor = None
        self.feature_extractor = None
        self.pipeline = None
        self.model_config = None

        # Load model config if available
        self._load_model_config()

        self._load_model()

    def _load_model_config(self):
        """Load model configuration if available."""
        model_path = Path(self.model_path)
        config_path = model_path / "model_config.json"
        if config_path.is_file():
            with open(config_path, "r") as f:
                self.model_config = json.load(f)
            logger.info(f"Loaded model config from {config_path}")
        elif model_path.is_dir():
            # Try in the model directory
            config_path = model_path / "model_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    self.model_config = json.load(f)
                logger.info(f"Loaded model config from {config_path}")

    def _load_model(self):
        """Load the appropriate model based on model_type."""
        if self.model_type == "transformer":
            self._load_transformer_model()
        elif self.model_type == "classical":
            self._load_classical_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _load_transformer_model(self):
        """Load transformer model and create pipeline."""
        # Check if local path exists, otherwise it's a HuggingFace model ID
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        # Convert Path to string to avoid issues with transformers library
        model_path_str = str(self.model_path)

        self.pipeline = pipeline(
            "text-classification",
            model=model_path_str,
            tokenizer=model_path_str,
            device=0 if self.device == "cuda" else -1,
            truncation=True,
            max_length=512,
        )

        # Initialize minimal preprocessor for transformers
        self.preprocessor = TextPreprocessor(mode="transformer")

        # Detect number of labels and create mapping
        num_labels = self.pipeline.model.config.num_labels
        label_names = BINARY_LABEL_NAMES if num_labels == 2 else THREE_CLASS_LABEL_NAMES
        self.label_map = {f"LABEL_{i}": label_names[i] for i in range(num_labels)}

        logger.info(
            f"Loaded transformer model from {self.model_path} ({num_labels} labels)"
        )

    def _load_classical_model(self):
        """Load classical ML model and preprocessor."""
        # Use ClassicalSentimentModel to load both model and feature extractor
        self.model = ClassicalSentimentModel()

        # Determine vectorizer path
        model_path = Path(self.model_path)
        vectorizer_path = model_path.parent / "tfidf_vectorizer.pkl"

        # Load model and vectorizer using the load method
        self.model.load(self.model_path, vectorizer_path)

        # Initialize preprocessor with classical mode (extensive preprocessing)
        self.preprocessor = TextPreprocessor(mode="classical")

        logger.info(f"Loaded classical model from {self.model_path}")

    def predict(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Predict sentiment labels for input text(s).

        Args:
            text: Single text string or list of texts

        Returns:
            Numpy array of numeric predictions (e.g., 0=NEGATIVE, 1=POSITIVE)
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        if self.model_type == "classical":
            # Classical: apply extensive preprocessing
            cleaned_texts = [self.preprocessor.clean_text(t) for t in texts]
            predictions = self.model.predict(cleaned_texts)
        else:
            # Transformer: apply minimal preprocessing (HTML/URL removal)
            cleaned_texts = [self.preprocessor.clean_text(t) for t in texts]
            results = self.pipeline(cleaned_texts, top_k=1)
            # Dynamically map LABEL_0, LABEL_1, LABEL_2, ... to 0, 1, 2, ...
            num_labels = self.pipeline.model.config.num_labels
            label_to_num = {f"LABEL_{i}": i for i in range(num_labels)}
            predictions = np.array([label_to_num[r[0]["label"]] for r in results])

        return predictions

    def predict_with_labels(
        self, text: Union[str, List[str]], return_probabilities: bool = False
    ) -> Union[Dict, List[Dict]]:
        """
        Predict sentiment with formatted labels (for API/UI use).

        Args:
            text: Single text string or list of texts
            return_probabilities: Whether to return probability scores

        Returns:
            Prediction dictionary or list of dictionaries with readable labels
        """
        if self.model_type == "transformer":
            return self._predict_transformer(text, return_probabilities)
        else:
            return self._predict_classical(text, return_probabilities)

    def _predict_transformer(
        self, text: Union[str, List[str]], return_probabilities: bool = False
    ) -> Union[Dict, List[Dict]]:
        """Make predictions using transformer model."""
        # Single text or batch
        results = self.pipeline(text, top_k=None if return_probabilities else 1)

        # Map LABEL_0, LABEL_1, etc. to actual names
        if isinstance(text, str):
            if return_probabilities:
                mapped_preds = [
                    {"label": self.label_map[p["label"]], "score": p["score"]}
                    for p in results
                ]
                return {"text": text, "predictions": mapped_preds}
            else:
                return {
                    "text": text,
                    "label": self.label_map[results[0]["label"]],
                    "score": results[0]["score"],
                }
        else:
            if return_probabilities:
                return [
                    {
                        "text": t,
                        "predictions": [
                            {"label": self.label_map[p["label"]], "score": p["score"]}
                            for p in r
                        ],
                    }
                    for t, r in zip(text, results)
                ]
            else:
                return [
                    {
                        "text": t,
                        "label": self.label_map[r[0]["label"]],
                        "score": r[0]["score"],
                    }
                    for t, r in zip(text, results)
                ]

    def _predict_classical(
        self, text: Union[str, List[str]], return_probabilities: bool = False
    ) -> Union[Dict, List[Dict]]:
        """Make predictions using classical ML model."""
        # Preprocess
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        cleaned_texts = [self.preprocessor.clean_text(t) for t in texts]

        # Predict
        predictions = self.model.predict(
            [cleaned_texts[0]] if isinstance(text, str) else cleaned_texts
        )

        # Get probabilities (always get them if model supports it)
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(
                [cleaned_texts[0]] if isinstance(text, str) else cleaned_texts
            )
        else:
            probas = None

        # Format results
        label_map = {0: "NEGATIVE", 1: "POSITIVE"}

        if isinstance(text, str):
            result = {
                "text": text,
                "label": label_map[predictions[0]],
                "score": probas[0][predictions[0]] if probas is not None else None,
            }
            return result
        else:
            results = []
            for i, t in enumerate(texts):
                results.append(
                    {
                        "text": t,
                        "label": label_map[predictions[i]],
                        "score": (
                            probas[i][predictions[i]] if probas is not None else None
                        ),
                    }
                )
            return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model configuration and metadata.

        Returns:
            Dictionary with model information
        """
        info = {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "device": self.device,
        }

        if self.model_config:
            info["config"] = self.model_config

        return info
