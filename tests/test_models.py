"""
Unit tests for model training and inference.
Tests classical models, transformer models, and inference.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from sentiment_analysis.inference import SentimentPredictor
from sentiment_analysis.evaluation import evaluate_model
from sentiment_analysis.train_classical import ClassicalSentimentModel
from sentiment_analysis.train_transformer import TransformerSentimentTrainer
from sentiment_analysis.exceptions import (
    ModelNotFittedError,
    InvalidModelTypeError,
    ModelSaveError,
    ModelLoadError,
)
from sentiment_analysis.constants import DEFAULT_TRANSFORMER_MODEL, DEFAULT_MAX_LENGTH


class TestSentimentPredictor:
    """Test cases for SentimentPredictor class."""

    @pytest.fixture
    def mock_transformer_pipeline(self):
        """Mock transformer pipeline."""
        # Patch the pipeline function from transformers.pipeline imported in inference module
        with patch("sentiment_analysis.inference.pipeline") as mock_pipe:
            mock_instance = Mock()
            mock_pipe.return_value = mock_instance
            yield mock_instance

    def test_init_transformer(self, mock_transformer_pipeline):
        """Test initialization of transformer predictor."""
        predictor = SentimentPredictor(model_path="fake/path", model_type="transformer")

        assert predictor.model_type == "transformer"
        assert predictor.pipeline is not None

    def test_predict_single_text_transformer(self, mock_transformer_pipeline):
        """Test prediction on single text with transformer (returns numpy array)."""
        # Mock 2-class model config
        mock_config = Mock()
        mock_config.num_labels = 2
        mock_transformer_pipeline.model.config = mock_config

        # When called, the mock pipeline returns a list of dicts
        mock_transformer_pipeline.return_value = [
            [{"label": "LABEL_1", "score": 0.95}]  # LABEL_1 = POSITIVE (index 1)
        ]

        predictor = SentimentPredictor(model_path="fake/path", model_type="transformer")

        result = predictor.predict("Great product!")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] == 1  # POSITIVE

    def test_predict_batch_transformer(self, mock_transformer_pipeline):
        """Test prediction on batch of texts with transformer (returns numpy array)."""
        # Mock 2-class model config
        mock_config = Mock()
        mock_config.num_labels = 2
        mock_transformer_pipeline.model.config = mock_config

        mock_transformer_pipeline.return_value = [
            [{"label": "LABEL_1", "score": 0.95}],  # POSITIVE
            [{"label": "LABEL_0", "score": 0.88}],  # NEGATIVE
        ]

        predictor = SentimentPredictor(model_path="fake/path", model_type="transformer")

        texts = ["Great product!", "Terrible quality"]
        results = predictor.predict(texts)

        assert isinstance(results, np.ndarray)
        assert results.shape == (2,)
        assert results[0] == 1  # POSITIVE
        assert results[1] == 0  # NEGATIVE

    def test_predict_with_labels_single_text(self, mock_transformer_pipeline):
        """Test predict_with_labels on single text with formatted output."""
        # Mock 2-class model config
        mock_config = Mock()
        mock_config.num_labels = 2
        mock_transformer_pipeline.model.config = mock_config

        mock_transformer_pipeline.return_value = [{"label": "LABEL_1", "score": 0.95}]

        predictor = SentimentPredictor(model_path="fake/path", model_type="transformer")

        result = predictor.predict_with_labels("Great product!")

        assert "label" in result
        assert "score" in result
        assert "text" in result
        assert result["label"] == "POSITIVE"
        assert result["score"] == 0.95

    def test_predict_with_labels_batch(self, mock_transformer_pipeline):
        """Test predict_with_labels on batch of texts with formatted output."""
        # Mock 2-class model config
        mock_config = Mock()
        mock_config.num_labels = 2
        mock_transformer_pipeline.model.config = mock_config

        mock_transformer_pipeline.return_value = [
            [{"label": "LABEL_1", "score": 0.95}],
            [{"label": "LABEL_0", "score": 0.88}],
        ]

        predictor = SentimentPredictor(model_path="fake/path", model_type="transformer")

        texts = ["Great product!", "Terrible quality"]
        results = predictor.predict_with_labels(texts)

        assert isinstance(results, list)
        assert len(results) == 2
        assert all("label" in r for r in results)
        assert all("score" in r for r in results)
        assert all("text" in r for r in results)
        assert results[0]["label"] == "POSITIVE"
        assert results[1]["label"] == "NEGATIVE"

    def test_predict_with_labels_probabilities(self, mock_transformer_pipeline):
        """Test predict_with_labels with return_probabilities=True."""
        # Mock 2-class model config
        mock_config = Mock()
        mock_config.num_labels = 2
        mock_transformer_pipeline.model.config = mock_config

        mock_transformer_pipeline.return_value = [
            {"label": "LABEL_1", "score": 0.95},
            {"label": "LABEL_0", "score": 0.05},
        ]

        predictor = SentimentPredictor(model_path="fake/path", model_type="transformer")

        result = predictor.predict_with_labels("Great!", return_probabilities=True)

        assert "predictions" in result
        assert "text" in result
        assert isinstance(result["predictions"], list)
        assert len(result["predictions"]) == 2
        assert result["predictions"][0]["label"] == "POSITIVE"
        assert result["predictions"][1]["label"] == "NEGATIVE"

    def test_predict_with_labels_three_class(self, mock_transformer_pipeline):
        """Test predict_with_labels with 3-class classification."""
        # Mock 3-class model config
        mock_config = Mock()
        mock_config.num_labels = 3
        mock_transformer_pipeline.model.config = mock_config

        mock_transformer_pipeline.return_value = [
            {"label": "LABEL_1", "score": 0.85}  # NEUTRAL
        ]

        predictor = SentimentPredictor(model_path="fake/path", model_type="transformer")

        result = predictor.predict_with_labels("It's okay, nothing special")

        assert "label" in result
        assert "score" in result
        # Should map LABEL_1 to NEUTRAL for 3-class
        assert result["label"] == "NEUTRAL"

    def test_get_model_info_transformer(self, mock_transformer_pipeline):
        """Test get_model_info method for transformer model."""
        predictor = SentimentPredictor(
            model_path="fake/path/model", model_type="transformer"
        )

        info = predictor.get_model_info()

        assert "model_path" in info
        assert "model_type" in info
        assert "device" in info
        assert info["model_path"] == "fake/path/model"
        assert info["model_type"] == "transformer"
        assert info["device"] in ["cuda", "cpu"]

    def test_get_model_info_with_config(self, mock_transformer_pipeline):
        """Test get_model_info includes model_config if available."""
        # Create temp directory with model config
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "model_config.json"
            with open(config_path, "w") as f:
                f.write('{"num_labels": 2, "model_name": "test_model"}')

            predictor = SentimentPredictor(model_path=tmpdir, model_type="transformer")

            info = predictor.get_model_info()

            assert "config" in info
            assert info["config"]["num_labels"] == 2
            assert info["config"]["model_name"] == "test_model"

    def test_predict_three_class(self, mock_transformer_pipeline):
        """Test prediction with 3-class classification (returns numpy array)."""
        # Mock 3-class model config
        mock_config = Mock()
        mock_config.num_labels = 3
        mock_transformer_pipeline.model.config = mock_config

        mock_transformer_pipeline.return_value = [
            [{"label": "LABEL_1", "score": 0.85}]  # NEUTRAL (index 1)
        ]

        predictor = SentimentPredictor(model_path="fake/path", model_type="transformer")

        result = predictor.predict("It's okay, nothing special")

        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] == 1  # NEUTRAL

    def test_predict_batch_three_class(self, mock_transformer_pipeline):
        """Test batch prediction with 3-class classification (returns numpy array)."""
        # Mock 3-class model config
        mock_config = Mock()
        mock_config.num_labels = 3
        mock_transformer_pipeline.model.config = mock_config

        mock_transformer_pipeline.return_value = [
            [{"label": "LABEL_0", "score": 0.95}],  # NEGATIVE (index 0)
            [{"label": "LABEL_1", "score": 0.70}],  # NEUTRAL (index 1)
            [{"label": "LABEL_2", "score": 0.90}],  # POSITIVE (index 2)
        ]

        predictor = SentimentPredictor(model_path="fake/path", model_type="transformer")

        texts = ["Terrible", "It's okay", "Amazing"]
        results = predictor.predict(texts)

        assert isinstance(results, np.ndarray)
        assert results.shape == (3,)
        assert results[0] == 0  # NEGATIVE
        assert results[1] == 1  # NEUTRAL
        assert results[2] == 2  # POSITIVE


# Classical Model Tests


class TestClassicalSentimentModel:
    """Test cases for ClassicalSentimentModel class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X_train = [
            "This is a great product",
            "Terrible quality",
            "Amazing experience",
            "Worst purchase ever",
            "Excellent service",
            "Very disappointed",
            "Highly recommended",
            "Would not buy again",
            "Good value",
            "Bad quality",
        ]
        y_train = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        X_test = ["Great service", "Bad experience"]
        y_test = np.array([1, 0])

        return X_train, y_train, X_test, y_test

    def test_initialization(self):
        """Test model initialization."""
        model = ClassicalSentimentModel()
        assert model.model is not None
        assert not model.fitted

    def test_fit_and_predict(self, sample_data):
        """Test fitting and prediction."""
        X_train, y_train, X_test, _ = sample_data

        model = ClassicalSentimentModel()
        model.fit(X_train, y_train)

        assert model.fitted
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)

    def test_predict_before_fit(self, sample_data):
        """Test that predict raises error before fitting."""
        _, _, X_test, _ = sample_data
        model = ClassicalSentimentModel()

        with pytest.raises(ModelNotFittedError):
            model.predict(X_test)

    def test_save_and_load(self, sample_data):
        """Test saving and loading model."""
        X_train, y_train, X_test, _ = sample_data

        model = ClassicalSentimentModel()
        model.fit(X_train, y_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pkl"
            model.save(save_path)

            loaded_model = ClassicalSentimentModel.load(save_path)
            assert loaded_model.fitted

            # Should produce same predictions
            original_preds = model.predict(X_test)
            loaded_preds = loaded_model.predict(X_test)
            assert all(original_preds == loaded_preds)

    def test_save_creates_model_config(self, sample_data):
        """Test that saving model creates model_config.json."""
        X_train, y_train, _, _ = sample_data

        model = ClassicalSentimentModel()
        model.fit(X_train, y_train)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pkl"
            model.save(save_path)

            # Check config file exists
            config_path = save_path.parent / "model_config.json"
            assert config_path.exists()

            # Check config content
            import json

            with open(config_path, "r") as f:
                config = json.load(f)

            assert "num_labels" in config
            assert config["num_labels"] == 2
            assert "model_type" in config
            assert "label_names" in config


# Transformer Model Tests


class TestTransformerSentimentTrainer:
    """Test cases for TransformerSentimentTrainer class."""

    def test_initialization_default(self):
        """Test trainer initialization with default parameters."""
        trainer = TransformerSentimentTrainer()

        assert trainer.model_name == DEFAULT_TRANSFORMER_MODEL
        assert trainer.num_labels == 2
        assert trainer.max_length == DEFAULT_MAX_LENGTH
        assert trainer.device in ["cuda", "cpu"]

    def test_initialization_custom(self):
        """Test trainer initialization with custom parameters."""
        trainer = TransformerSentimentTrainer(
            model_name="bert-base-uncased", num_labels=3, max_length=256, device="cpu"
        )

        assert trainer.model_name == "bert-base-uncased"
        assert trainer.num_labels == 3
        assert trainer.max_length == 256
        assert trainer.device == "cpu"

    # Decorator form of patch to mock DistilBertForSequenceClassification
    # mock_model_class is the mock for DistilBertForSequenceClassification
    @patch("sentiment_analysis.train_transformer.DistilBertForSequenceClassification")
    def test_build_model(self, mock_model_class):
        """Test that build_model creates a model."""
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model

        trainer = TransformerSentimentTrainer(num_labels=2)
        trainer.build_model()

        mock_model_class.from_pretrained.assert_called_once()
        assert trainer.model == mock_model

    def test_device_selection(self):
        """Test device selection logic."""
        with patch("torch.cuda.is_available", return_value=False):
            trainer = TransformerSentimentTrainer()
            assert trainer.device == "cpu"
