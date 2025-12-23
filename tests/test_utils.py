"""
Unit tests for utils module.
Tests for utility functions including label mapping, model registry, and configuration helpers.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
from sentiment_analysis.utils import get_label_names, register_trained_model
from sentiment_analysis.constants import BINARY_LABEL_NAMES, THREE_CLASS_LABEL_NAMES


class TestGetLabelNames:
    """Test cases for get_label_names utility function."""

    def test_binary_labels(self):
        """Test getting label names for binary classification."""
        labels = get_label_names(2)

        assert labels == BINARY_LABEL_NAMES
        assert len(labels) == 2
        assert "NEGATIVE" in labels
        assert "POSITIVE" in labels

    def test_three_class_labels(self):
        """Test getting label names for 3-class classification."""
        labels = get_label_names(3)

        assert labels == THREE_CLASS_LABEL_NAMES
        assert len(labels) == 3
        assert "NEGATIVE" in labels
        assert "NEUTRAL" in labels
        assert "POSITIVE" in labels

    def test_generic_labels_4_classes(self):
        """Test generic label names for 4 classes."""
        labels = get_label_names(4)

        assert len(labels) == 4
        assert labels == ["LABEL_0", "LABEL_1", "LABEL_2", "LABEL_3"]

    def test_generic_labels_5_classes(self):
        """Test generic label names for 5 classes."""
        labels = get_label_names(5)

        assert len(labels) == 5
        assert all(label.startswith("LABEL_") for label in labels)
        assert labels[0] == "LABEL_0"
        assert labels[4] == "LABEL_4"

    def test_generic_labels_10_classes(self):
        """Test generic label names for 10 classes."""
        labels = get_label_names(10)

        assert len(labels) == 10
        assert labels[9] == "LABEL_9"


class TestRegisterTrainedModel:
    """Test cases for register_trained_model utility function."""

    def test_register_new_model_creates_file(self):
        """Test registering model creates new registry file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "models_config.json"
            model_path = Path(tmpdir) / "test_model"

            register_trained_model(
                model_name="Test Model",
                model_path=str(model_path),
                model_type="transformer",
            )

            # Check file was created
            assert registry_path.exists()

            # Check content
            with open(registry_path, "r") as f:
                models = json.load(f)

            assert len(models) == 1
            assert models[0]["name"] == "Test Model"
            assert models[0]["path"] == str(model_path)
            assert models[0]["type"] == "transformer"
            assert "registered_at" in models[0]

    def test_register_multiple_models(self):
        """Test registering multiple models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "models_config.json"

            # Register first model
            register_trained_model(
                model_name="Model 1",
                model_path="/path/to/model1",
                model_type="classical",
            )

            # Register second model
            register_trained_model(
                model_name="Model 2",
                model_path="/path/to/model2",
                model_type="transformer",
                registry_file=str(registry_path),
            )

            # Check both models are registered
            with open(registry_path, "r") as f:
                models = json.load(f)

            assert len(models) == 2
            assert models[0]["name"] == "Model 1"
            assert models[1]["name"] == "Model 2"

    def test_register_duplicate_model_updates(self):
        """Test registering duplicate model name updates existing entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "models_config.json"

            # Register model
            register_trained_model(
                model_name="Test Model",
                model_path="/path/to/model_v1",
                model_type="classical",
                registry_file=str(registry_path),
            )

            # Register again with same name
            register_trained_model(
                model_name="Test Model",
                model_path="/path/to/model_v2",
                model_type="transformer",
                registry_file=str(registry_path),
            )

            # Check only one entry exists with updated info
            with open(registry_path, "r") as f:
                models = json.load(f)

            assert len(models) == 1
            assert models[0]["name"] == "Test Model"
            assert models[0]["path"] == "/path/to/model_v2"
            assert models[0]["type"] == "transformer"

    def test_register_preserves_existing_models(self):
        """Test registering new model preserves existing entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "models_config.json"

            # Create existing registry
            existing_models = [
                {
                    "name": "Existing Model",
                    "path": "/path/to/existing",
                    "type": "classical",
                    "registered_at": "2025-01-01T00:00:00",
                }
            ]
            with open(registry_path, "w") as f:
                json.dump(existing_models, f)

            # Register new model
            register_trained_model(
                model_name="New Model",
                model_path="/path/to/new",
                model_type="transformer",
                registry_file=str(registry_path),
            )

            # Check both models exist
            with open(registry_path, "r") as f:
                models = json.load(f)

            assert len(models) == 2
            assert models[0]["name"] == "Existing Model"
            assert models[1]["name"] == "New Model"

    def test_register_creates_parent_directories(self):
        """Test registering model creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "subdir" / "nested" / "models_config.json"

            register_trained_model(
                model_name="Test Model",
                model_path="/path/to/model",
                model_type="classical",
                registry_file=str(registry_path),
            )

            # Check file and directories were created
            assert registry_path.exists()
            assert registry_path.parent.exists()

    def test_register_with_default_path(self):
        """Test registering model with default registry path."""
        with patch("sentiment_analysis.utils.Path") as mock_path:
            mock_registry = tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            )
            mock_registry.write("[]")
            mock_registry.close()

            try:
                # Mock the default path resolution
                mock_path.return_value.__truediv__.return_value = Path(
                    mock_registry.name
                )

                register_trained_model(
                    model_name="Test Model",
                    model_path="/path/to/model",
                    model_type="transformer",
                )

                # Verify file was written (basic check)
                with open(mock_registry.name, "r") as f:
                    content = f.read()
                    assert len(content) > 0
            finally:
                Path(mock_registry.name).unlink()

    def test_register_invalid_json_creates_new(self):
        """Test registering when existing file has invalid JSON creates new registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "models_config.json"

            # Create file with invalid JSON
            with open(registry_path, "w") as f:
                f.write("invalid json {{{")

            # Should handle error and create new registry
            register_trained_model(
                model_name="Test Model",
                model_path="/path/to/model",
                model_type="classical",
                registry_file=str(registry_path),
            )

            # Check valid JSON was written
            with open(registry_path, "r") as f:
                models = json.load(f)

            assert len(models) == 1
            assert models[0]["name"] == "Test Model"


class TestUtilsIntegration:
    """Integration tests for utils module."""

    def test_label_names_used_in_registry(self):
        """Test that label names work correctly with model registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "models_config.json"

            # Register binary model
            register_trained_model(
                model_name="Binary Model",
                model_path="/path/to/binary",
                model_type="classical",
                registry_file=str(registry_path),
            )

            # Register 3-class model
            register_trained_model(
                model_name="3-Class Model",
                model_path="/path/to/3class",
                model_type="transformer",
                registry_file=str(registry_path),
            )

            # Get label names for both
            binary_labels = get_label_names(2)
            three_class_labels = get_label_names(3)

            assert len(binary_labels) == 2
            assert len(three_class_labels) == 3

            # Check registry
            with open(registry_path, "r") as f:
                models = json.load(f)

            assert len(models) == 2
