"""
Classical ML Model Training Module

This module contains functions for training Logistic Regression with TF-IDF features
for sentiment analysis, matching the implementation in sentiment_analysis_classicalNLP.ipynb.
"""

import numpy as np
import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from sentiment_analysis.data_preparation import load_and_prepare_data
from sentiment_analysis.feature_extraction import TFIDFFeatureExtractor
from sentiment_analysis.data_splitting import split_train_test
from sentiment_analysis.text_preprocessing import TextPreprocessor
from sentiment_analysis.evaluation import evaluate_model
from sentiment_analysis.utils import (
    load_config,
    get_config_value,
    register_trained_model,
)
from sentiment_analysis.settings import settings
from sentiment_analysis.exceptions import (
    ModelNotFittedError,
    ModelLoadError,
    ModelSaveError,
)

logger = logging.getLogger(__name__)

# Load defaults from config
_config = load_config()
DEFAULT_MAX_FEATURES = get_config_value(
    _config, "model", "classical", "max_features", default=5000
)
DEFAULT_RANDOM_STATE = get_config_value(_config, "training", "random_state", default=42)
DEFAULT_TEST_SIZE = get_config_value(_config, "training", "test_size", default=0.2)
DEFAULT_SAMPLE_FRACTION = get_config_value(_config, "data", "sample_frac", default=0.1)
CLASSICAL_SOLVER = "saga"
CLASSICAL_MAX_ITER = 1000
CLASSICAL_CLASS_WEIGHT = "balanced"


class ClassicalSentimentModel:
    """
    Logistic Regression model for sentiment analysis with TF-IDF features.
    Matches the implementation in sentiment_analysis_classicalNLP.ipynb notebook.
    """

    def __init__(self, **kwargs):
        """
        Initialize Logistic Regression model.

        Args:
            **kwargs: Additional arguments for LogisticRegression
                     Default values from constants module
        """
        self.model = LogisticRegression(
            solver=kwargs.get("solver", CLASSICAL_SOLVER),
            max_iter=kwargs.get("max_iter", CLASSICAL_MAX_ITER),
            class_weight=kwargs.get("class_weight", CLASSICAL_CLASS_WEIGHT),
            random_state=kwargs.get("random_state", DEFAULT_RANDOM_STATE),
        )
        self.feature_extractor = None
        self.fitted = False
        logger.debug(f"Initialized ClassicalSentimentModel with {kwargs}")

    def fit(self, X_train, y_train, feature_params: Dict[str, Any] = None):
        """
        Fit the model on training data.

        Args:
            X_train: Training texts (list of strings or already vectorized)
            y_train: Training labels
            feature_params: Parameters for TF-IDF extraction (max_features, ngram_range)
        """
        feature_params = feature_params or {}

        # If X_train is text, extract TF-IDF features
        if isinstance(X_train, (list, pd.Series)) and isinstance(
            X_train.iloc[0] if isinstance(X_train, pd.Series) else X_train[0], str
        ):
            self.feature_extractor = TFIDFFeatureExtractor(**feature_params)
            X_train_features = self.feature_extractor.fit_transform(X_train)
        else:
            X_train_features = X_train

        # Train model
        self.model.fit(X_train_features, y_train)
        self.fitted = True

    def predict(self, X):
        """Make predictions on new data."""
        if not self.fitted:
            raise ModelNotFittedError("ClassicalSentimentModel")

        # Extract features if needed
        if (
            self.feature_extractor
            and isinstance(X, (list, pd.Series))
            and isinstance(X.iloc[0] if isinstance(X, pd.Series) else X[0], str)
        ):
            X_features = self.feature_extractor.transform(X)
        else:
            X_features = X

        return self.model.predict(X_features)

    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.fitted:
            raise ModelNotFittedError("ClassicalSentimentModel")

        # Extract features if needed
        if (
            self.feature_extractor
            and isinstance(X, (list, pd.Series))
            and isinstance(X.iloc[0] if isinstance(X, pd.Series) else X[0], str)
        ):
            X_features = self.feature_extractor.transform(X)
        else:
            X_features = X

        return self.model.predict_proba(X_features)

    def save(
        self,
        model_path: str,
        feature_extractor_path: str = None,
        model_params: Dict = None,
    ):
        """Save the trained model and feature extractor."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)

        if self.feature_extractor and feature_extractor_path:
            self.feature_extractor.save(feature_extractor_path)

        # Save model config
        if model_params:
            model_config = {
                "model_type": "logistic_regression",
                "feature_type": "tfidf",
                "num_labels": 2,
                "label_names": ["NEGATIVE", "POSITIVE"],
                "preprocessing": {
                    "mode": "classical",
                    "description": "Extensive preprocessing for classical ML models",
                },
                "parameters": model_params,
            }
            config_path = model_path.parent / "model_config.json"
            with open(config_path, "w") as f:
                json.dump(model_config, f, indent=2)
            logger.info(f"Model config saved to {config_path}")

        logger.info(f"Model saved to {model_path}")

    def load(self, model_path: str, feature_extractor_path: str = None):
        """Load a trained model and feature extractor."""
        self.model = joblib.load(model_path)

        if feature_extractor_path and Path(feature_extractor_path).exists():
            self.feature_extractor = TFIDFFeatureExtractor()
            self.feature_extractor.load(feature_extractor_path)

        self.fitted = True
        logger.info(f"Model loaded from {model_path}")


def train_classical_model(
    data_path: str,
    output_dir: str = None,
    test_size: float = DEFAULT_TEST_SIZE,
    max_features: int = DEFAULT_MAX_FEATURES,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """
    Complete training pipeline for Logistic Regression with TF-IDF.
    Matches the workflow in sentiment_analysis_classicalNLP.ipynb notebook.

    Args:
        data_path: Path to data CSV
        output_dir: Output directory for model (default: models/classical_models)
        test_size: Test set size (default from config)
        max_features: Max features for TF-IDF (default from config)
        random_state: Random seed (default from config)

    Returns:
        Tuple of (model, results dictionary)
    """
    # Set default output_dir from config
    if output_dir is None:
        output_dir = settings.model_dir / "classical_models"
    else:
        output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting classical model training pipeline")
    logger.info(f"Output directory: {output_dir}")

    # Load and prepare data
    logger.info("Loading and preprocessing data...")
    df = load_and_prepare_data(
        data_path, sample_frac=DEFAULT_SAMPLE_FRACTION, random_state=random_state
    )

    # Split data
    train_df, test_df = split_train_test(
        df, test_size=test_size, random_state=random_state
    )
    X_train, y_train = train_df["cleaned_text"], train_df["label"]
    X_test, y_test = test_df["cleaned_text"], test_df["label"]

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Initialize and train model
    logger.info(
        f"Training Logistic Regression with TF-IDF features (max_features={max_features})..."
    )
    model = ClassicalSentimentModel()

    feature_params = {"max_features": max_features, "ngram_range": (1, 1)}
    model.fit(X_train, y_train, feature_params=feature_params)

    # Evaluate
    logger.info("Evaluating model...")
    label_names = ["negative", "positive"]
    results = evaluate_model(model, X_test, y_test, label_names=label_names)

    # Save model
    model_file = output_dir / "logistic_tfidf_model.pkl"
    feature_file = output_dir / "tfidf_vectorizer.pkl"
    model_params = {
        "model_type": "logistic_regression",
        "feature_type": "tfidf",
        "max_features": max_features,
        "test_size": test_size,
        "random_state": random_state,
        "solver": CLASSICAL_SOLVER,
        "max_iter": CLASSICAL_MAX_ITER,
        "class_weight": CLASSICAL_CLASS_WEIGHT,
    }
    model.save(model_file, feature_file, model_params)

    logger.info("Training complete!")

    # Register model in registry
    model_rel_path = output_dir.relative_to(settings.project_root)
    register_trained_model(
        model_name="Logistic Regression + TF-IDF",
        model_path=str(model_rel_path / "logistic_tfidf_model.pkl"),
        model_type="classical",
    )

    return model, results
