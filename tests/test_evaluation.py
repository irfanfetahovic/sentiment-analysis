"""
Unit tests for evaluation module.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from sentiment_analysis.evaluation import (
    evaluate_classification,
    evaluate_model,
    compare_models,
)


class TestEvaluateClassification:
    """Test cases for evaluate_classification function."""

    def test_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])

        metrics = evaluate_classification(y_true, y_pred, verbose=False)

        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0
        assert metrics["precision_macro"] == 1.0
        assert metrics["recall_macro"] == 1.0

    def test_poor_predictions(self):
        """Test evaluation with poor predictions."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1])

        metrics = evaluate_classification(y_true, y_pred, verbose=False)

        assert metrics["accuracy"] == 0.0
        assert "confusion_matrix" in metrics
        assert "classification_report" in metrics

    def test_mixed_predictions(self):
        """Test evaluation with mixed predictions."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 0, 0, 1])

        metrics = evaluate_classification(y_true, y_pred, verbose=False)

        assert 0.0 < metrics["accuracy"] < 1.0
        assert 0.0 < metrics["f1_macro"] < 1.0
        assert "confusion_matrix" in metrics

    def test_with_label_names(self):
        """Test evaluation with label names."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        metrics = evaluate_classification(
            y_true, y_pred, label_names=["negative", "positive"], verbose=False
        )

        assert "classification_report" in metrics
        assert "negative" in metrics["classification_report"]
        assert "positive" in metrics["classification_report"]

    def test_verbose_output(self, caplog):
        """Test verbose output."""
        import logging

        logger = logging.getLogger("sentiment_analysis.evaluation")
        caplog.set_level(logging.INFO, logger="sentiment_analysis.evaluation")
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        evaluate_classification(y_true, y_pred, verbose=True)

        assert "Accuracy" in caplog.text
        assert "F1 Score" in caplog.text


class TestEvaluateModel:
    """Test cases for evaluate_model function."""

    def test_evaluate_model_with_mock(self):
        """Test model evaluation with mock model."""
        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1])

        X_test = ["text1", "text2", "text3", "text4"]
        y_test = np.array([0, 1, 0, 1])

        metrics = evaluate_model(mock_model, X_test, y_test, verbose=False)

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "predictions" in metrics
        assert mock_model.predict.called

    def test_evaluate_model_without_predict_method(self):
        """Test evaluation fails without predict method."""
        mock_model = Mock(spec=[])  # No methods

        X_test = ["text1", "text2"]
        y_test = np.array([0, 1])

        with pytest.raises(AttributeError):
            evaluate_model(mock_model, X_test, y_test)

    def test_evaluate_model_with_label_names(self):
        """Test model evaluation with label names."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1])

        X_test = ["text1", "text2"]
        y_test = np.array([0, 1])

        metrics = evaluate_model(
            mock_model,
            X_test,
            y_test,
            label_names=["negative", "positive"],
            verbose=False,
        )

        assert metrics["accuracy"] == 1.0


class TestCompareModels:
    """Test cases for compare_models function."""

    def test_compare_two_models(self, caplog):
        """Test comparing two models."""
        import logging

        logger = logging.getLogger("sentiment_analysis.evaluation")
        caplog.set_level(logging.INFO, logger="sentiment_analysis.evaluation")
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])

        predictions = {
            "Model A": np.array([0, 1, 0, 1, 0, 1, 0, 1]),  # Perfect
            "Model B": np.array([0, 1, 1, 1, 0, 0, 0, 1]),  # Decent
        }

        results = compare_models(predictions, y_true)

        assert "Model A" in results
        assert "Model B" in results
        assert results["Model A"]["accuracy"] > results["Model B"]["accuracy"]

        # Check logged output
        assert "MODEL COMPARISON" in caplog.text
        assert "Model A" in caplog.text
        assert "Model B" in caplog.text

    def test_compare_multiple_models(self):
        """Test comparing multiple models."""
        y_true = np.array([0, 1, 0, 1, 0, 1])

        predictions = {
            "LogisticRegression": np.array([0, 1, 0, 1, 0, 1]),
            "DistilBERT": np.array([0, 1, 0, 1, 1, 0]),
            "RandomForest": np.array([0, 1, 1, 0, 0, 1]),
        }

        results = compare_models(predictions, y_true)

        assert len(results) == 3
        assert all("accuracy" in r for r in results.values())
        assert all("f1_macro" in r for r in results.values())

    def test_compare_with_label_names(self):
        """Test model comparison with label names."""
        y_true = np.array([0, 1, 0, 1])

        predictions = {
            "Model A": np.array([0, 1, 0, 1]),
            "Model B": np.array([0, 1, 1, 0]),
        }

        results = compare_models(
            predictions, y_true, label_names=["negative", "positive"]
        )

        assert "classification_report" in results["Model A"]
        assert "negative" in results["Model A"]["classification_report"]
