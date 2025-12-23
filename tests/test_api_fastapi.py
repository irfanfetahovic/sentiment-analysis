"""
Unit tests for FastAPI application.
"""

import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def mock_predictor():
    """Mock sentiment predictor."""
    # Create fake (mock) predictor instance with controlled behavior
    predictor = Mock()
    predictor.predict.side_effect = lambda x: (
        [
            {"text": "Great product!", "label": "POSITIVE", "score": 0.95},
            {"text": "Terrible quality", "label": "NEGATIVE", "score": 0.88},
        ]
        if isinstance(x, list)
        else {
            "text": "Great product!",
            "label": "POSITIVE",
            "score": 0.95,
        }
    )
    return predictor


@pytest.fixture
def client(mock_predictor):
    """Create FastAPI test client with mocked predictor."""
    # Temporarily replaces (patches) the SentimentPredictor used in the app with our mock
    with patch("app.app_fastapi.SentimentPredictor", return_value=mock_predictor):
        # Import after patching to ensure mock is used during startup
        from app.app_fastapi import app

        with TestClient(app) as client:
            yield client


class TestFastAPIEndpoints:
    """Test cases for FastAPI endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Sentiment Analysis API"
        assert data["version"] == "1.0.0"
        assert "docs" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "model_loaded" in data
        assert "model_type" in data
        assert "timestamp" in data

    def test_predict_endpoint_success(self, client, mock_predictor):
        """Test successful single prediction."""
        payload = {"text": "Great product!"}

        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Great product!"
        assert data["label"] == "POSITIVE"
        assert data["score"] == 0.95
        assert "processing_time" in data

        # Verify predictor was called
        mock_predictor.predict.assert_called_once_with("Great product!")

    def test_predict_endpoint_empty_text(self, client):
        """Test prediction with empty text."""
        payload = {"text": ""}

        response = client.post("/predict", json=payload)

        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_missing_text(self, client):
        """Test prediction with missing text field."""
        payload = {}

        response = client.post("/predict", json=payload)

        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_whitespace_only(self, client):
        """Test prediction with whitespace-only text."""
        payload = {"text": "   "}

        response = client.post("/predict", json=payload)

        assert response.status_code == 422  # Validation error

    def test_predict_batch_endpoint_success(self, client, mock_predictor):
        """Test successful batch prediction."""
        payload = {"texts": ["Great product!", "Terrible quality"]}

        response = client.post("/predict/batch", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        assert data["total_processed"] == 2
        assert "total_time" in data

        # Verify batch predictor was called
        mock_predictor.predict_batch.assert_called_once()

    def test_predict_batch_endpoint_empty_list(self, client):
        """Test batch prediction with empty list."""
        payload = {"texts": []}

        response = client.post("/predict/batch", json=payload)

        assert response.status_code == 422  # Validation error

    def test_predict_batch_endpoint_missing_texts(self, client):
        """Test batch prediction with missing texts field."""
        payload = {}

        response = client.post("/predict/batch", json=payload)

        assert response.status_code == 422  # Validation error

    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")

        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "model_path" in data
        assert "device" in data
        assert "max_text_length" in data
        assert "max_batch_size" in data

    def test_openapi_docs_available(self, client):
        """Test that OpenAPI documentation is available."""
        response = client.get("/docs")
        assert response.status_code == 200

        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_schema(self, client):
        """Test OpenAPI schema endpoint."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "Sentiment Analysis API"


class TestFastAPIValidation:
    """Test Pydantic validation in FastAPI."""

    def test_text_too_long(self, client):
        """Test prediction with text exceeding max length."""
        # Assuming MAX_TEXT_LENGTH is 5000
        long_text = "a" * 6000
        payload = {"text": long_text}

        response = client.post("/predict", json=payload)

        assert response.status_code == 422  # Validation error

    def test_batch_too_many_texts(self, client):
        """Test batch prediction exceeding max batch size."""
        # Assuming MAX_BATCH_SIZE is 100
        payload = {"texts": ["text"] * 101}

        response = client.post("/predict/batch", json=payload)

        assert response.status_code == 422  # Validation error

    def test_invalid_json(self, client):
        """Test request with invalid JSON."""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422
