"""
Unit tests for Flask API.
"""

import pytest
import json
from unittest.mock import Mock, patch


@pytest.fixture
def mock_predictor():
    """Mock sentiment predictor."""
    with patch("app.app.SentimentPredictor") as mock:
        instance = Mock()
        instance.predict.return_value = {
            "text": "Great product!",
            "label": "POSITIVE",
            "score": 0.95,
        }
        mock.return_value = instance
        yield instance


@pytest.fixture
def client(mock_predictor):
    """Create test client."""
    from app.app import app

    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


class TestAPI:
    """Test cases for Flask API."""

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"

    def test_predict_endpoint_success(self, client, mock_predictor):
        """Test successful prediction."""
        payload = {"text": "Great product!"}

        response = client.post(
            "/predict", data=json.dumps(payload), content_type="application/json"
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "label" in data
        assert "score" in data
        assert "text" in data

    def test_predict_endpoint_missing_text(self, client):
        """Test prediction with missing text."""
        payload = {}

        response = client.post(
            "/predict", data=json.dumps(payload), content_type="application/json"
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_predict_endpoint_empty_text(self, client):
        """Test prediction with empty text."""
        payload = {"text": ""}

        response = client.post(
            "/predict", data=json.dumps(payload), content_type="application/json"
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_predict_batch_endpoint(self, client, mock_predictor):
        """Test batch prediction."""
        mock_predictor.predict.return_value = [
            {"text": "Great!", "label": "POSITIVE", "score": 0.95},
            {"text": "Terrible!", "label": "NEGATIVE", "score": 0.92},
        ]

        payload = {"texts": ["Great!", "Terrible!"]}

        response = client.post(
            "/predict/batch", data=json.dumps(payload), content_type="application/json"
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "predictions" in data
        assert len(data["predictions"]) == 2
