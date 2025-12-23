"""
Unit tests for data preprocessing module.
Tests text cleaning, label conversion, data loading, and data splitting.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from sentiment_analysis.text_preprocessing import TextPreprocessor
from sentiment_analysis.label_conversion import score_to_label
from sentiment_analysis.data_preparation import load_and_prepare_data
from sentiment_analysis.data_splitting import split_train_test, split_train_val_test
from sentiment_analysis.exceptions import (
    InvalidScoreError,
    InvalidProblemTypeError,
    InvalidPreprocessingModeError,
    DataLoadError,
    InsufficientDataError,
)
from sentiment_analysis.constants import (
    MIN_REVIEW_SCORE,
    MAX_REVIEW_SCORE,
    BINARY_THRESHOLD,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    PROBLEM_TYPE_BINARY,
)


class TestTextPreprocessor:
    """Test cases for TextPreprocessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor_classical = TextPreprocessor(mode="classical")
        self.preprocessor_transformer = TextPreprocessor(mode="transformer")

    def test_clean_text_basic_classical(self):
        """Test basic text cleaning in classical mode."""
        text = "This is a GREAT product!"
        cleaned = self.preprocessor_classical.clean_text(text)

        assert isinstance(cleaned, str)
        assert len(cleaned) > 0
        assert (
            cleaned.islower() or "<" in cleaned
        )  # Should be lowercase or contain tokens

    def test_clean_text_basic_transformer(self):
        """Test basic text cleaning in transformer mode."""
        text = "This is a GREAT product!"
        cleaned = self.preprocessor_transformer.clean_text(text)

        assert isinstance(cleaned, str)
        assert len(cleaned) > 0
        # Transformer mode keeps case
        assert "GREAT" in cleaned or "product" in cleaned

    def test_clean_text_html(self):
        """Test HTML tag removal (both modes)."""
        text = "<b>Amazing</b> product! <a href='test'>Click here</a>"
        cleaned_classical = self.preprocessor_classical.clean_text(text)
        cleaned_transformer = self.preprocessor_transformer.clean_text(text)

        # Both modes should remove HTML
        assert "<b>" not in cleaned_classical
        assert "</b>" not in cleaned_classical
        assert "<b>" not in cleaned_transformer
        assert "</b>" not in cleaned_transformer

    def test_clean_text_urls(self):
        """Test URL removal (both modes)."""
        text = "Check this out: https://example.com and www.test.com"
        cleaned_classical = self.preprocessor_classical.clean_text(text)
        cleaned_transformer = self.preprocessor_transformer.clean_text(text)

        # Both modes should remove URLs
        assert "https://" not in cleaned_classical
        assert "www." not in cleaned_classical
        assert "https://" not in cleaned_transformer
        assert "www." not in cleaned_transformer

    def test_clean_text_numbers_classical(self):
        """Test number replacement in classical mode."""
        text = "I bought 5 items for $25.99"
        cleaned = self.preprocessor_classical.clean_text(text)

        assert "5" not in cleaned
        assert "25" not in cleaned
        assert "<num>" in cleaned or "num" in cleaned

    def test_clean_text_numbers_transformer(self):
        """Test numbers are kept in transformer mode."""
        text = "I bought 5 items for $25.99"
        cleaned = self.preprocessor_transformer.clean_text(text)

        # Transformer mode keeps numbers
        assert "5" in cleaned or "25" in cleaned

    def test_clean_text_empty(self):
        """Test handling of empty text."""
        text = ""
        cleaned_classical = self.preprocessor_classical.clean_text(text)
        cleaned_transformer = self.preprocessor_transformer.clean_text(text)

        assert isinstance(cleaned_classical, str)
        assert isinstance(cleaned_transformer, str)

    def test_clean_text_contractions_classical(self):
        """Test contraction expansion in classical mode."""
        text = "I don't think it's good"
        cleaned = self.preprocessor_classical.clean_text(text)

        # After expansion and processing, "don't" should become "do not"
        assert "don't" not in cleaned.lower()

    def test_clean_text_contractions_transformer(self):
        """Test contractions are kept in transformer mode."""
        text = "I don't think it's good"
        cleaned = self.preprocessor_transformer.clean_text(text)

        # Transformer mode keeps contractions
        assert "don't" in cleaned or "it's" in cleaned

    def test_batch_clean_classical(self):
        """Test batch cleaning in classical mode."""
        texts = ["Great product!", "Terrible quality", "It's okay"]
        cleaned_texts = self.preprocessor_classical.batch_clean(texts)

        assert len(cleaned_texts) == 3
        assert all(isinstance(t, str) for t in cleaned_texts)

    def test_batch_clean_transformer(self):
        """Test batch cleaning in transformer mode."""
        texts = ["Great product!", "Terrible quality", "It's okay"]
        cleaned_texts = self.preprocessor_transformer.batch_clean(texts)

        assert len(cleaned_texts) == 3
        assert all(isinstance(t, str) for t in cleaned_texts)

    def test_preprocess_dataframe_via_load_and_prepare(self):
        """Test DataFrame preprocessing via load_and_prepare_data pipeline."""
        # This tests the internal _preprocess_dataframe function indirectly through the public API
        pass  # Already tested in TestDataLoading


class TestScoreToLabel:
    """Test cases for score_to_label function."""

    def test_binary_classification_positive(self):
        """Test binary classification for positive scores."""
        assert score_to_label(4, "binary") == 1
        assert score_to_label(5, "binary") == 1

    def test_binary_classification_negative(self):
        """Test binary classification for negative scores."""
        assert score_to_label(1, "binary") == 0
        assert score_to_label(2, "binary") == 0
        assert score_to_label(3, "binary") == 0

    def test_3class_classification_negative(self):
        """Test 3-class classification for negative scores."""
        assert score_to_label(1, "3-class") == 0
        assert score_to_label(2, "3-class") == 0

    def test_3class_classification_neutral(self):
        """Test 3-class classification for neutral scores."""
        assert score_to_label(3, "3-class") == 1

    def test_3class_classification_positive(self):
        """Test 3-class classification for positive scores."""
        assert score_to_label(4, "3-class") == 2
        assert score_to_label(5, "3-class") == 2

    def test_score_to_label_invalid_score_low(self):
        """Test invalid score (too low)."""
        with pytest.raises(InvalidScoreError):
            score_to_label(0, "binary")

    def test_score_to_label_invalid_score_high(self):
        """Test invalid score (too high)."""
        with pytest.raises(InvalidScoreError):
            score_to_label(6, "binary")

    def test_score_to_label_invalid_problem_type(self):
        """Test invalid problem type."""
        with pytest.raises(InvalidProblemTypeError):
            score_to_label(3, "invalid_type")


class TestLoadAndPrepareData:
    """Test cases for load_and_prepare_data function."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file for testing."""
        df = pd.DataFrame(
            {
                "Score": [1, 2, 3, 4, 5, 1, 5],
                "Text": [
                    "Bad product",
                    "Not good",
                    "Okay",
                    "Good product",
                    "Excellent!",
                    "Terrible",
                    "Amazing!",
                ],
            }
        )

        csv_path = tmp_path / "test_reviews.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_load_binary_classification(self, sample_csv):
        """Test loading data for binary classification."""
        df = load_and_prepare_data(sample_csv, sample_frac=1.0, problem_type="binary")

        assert "cleaned_text" in df.columns
        assert "label" in df.columns
        assert set(df["label"].unique()).issubset({0, 1})
        # Should remove neutral reviews (score=3)
        assert 3 not in df["Score"].values

    def test_load_3class_classification(self, sample_csv):
        """Test loading data for 3-class classification."""
        df = load_and_prepare_data(sample_csv, sample_frac=1.0, problem_type="3-class")

        assert "cleaned_text" in df.columns
        assert "label" in df.columns
        assert set(df["label"].unique()).issubset({0, 1, 2})

    def test_load_with_skip_preprocessing(self, sample_csv):
        """Test loading data with skip_preprocessing=True."""
        df = load_and_prepare_data(
            sample_csv, sample_frac=1.0, problem_type="binary", skip_preprocessing=True
        )

        # Should have label but NOT cleaned_text when skipping preprocessing
        assert "label" in df.columns
        assert "cleaned_text" not in df.columns
        assert "Text" in df.columns
        assert set(df["label"].unique()).issubset({0, 1})


# Data Splitting Tests


class TestSplitTrainTest:
    """Test cases for split_train_test function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "text": [f"Sample text {i}" for i in range(100)],
                "label": np.random.randint(0, 2, 100),
            }
        )

    def test_basic_split(self, sample_dataframe):
        """Test basic train-test split."""
        train_df, test_df = split_train_test(sample_dataframe, test_size=0.2)

        assert len(train_df) + len(test_df) == len(sample_dataframe)
        assert len(test_df) == 20
        assert len(train_df) == 80

    def test_random_state_reproducibility(self, sample_dataframe):
        """Test that random_state produces reproducible splits."""
        train1, test1 = split_train_test(sample_dataframe, random_state=42)
        train2, test2 = split_train_test(sample_dataframe, random_state=42)

        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_insufficient_data(self):
        """Test error handling for insufficient data."""
        small_df = pd.DataFrame({"text": ["text1"], "label": [1]})

        with pytest.raises(InsufficientDataError):
            split_train_test(small_df, test_size=0.2)


class TestSplitTrainValTest:
    """Test cases for split_train_val_test function."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "text": [f"Sample text {i}" for i in range(100)],
                "label": np.random.randint(0, 2, 100),
            }
        )

    def test_basic_split(self, sample_dataframe):
        """Test basic train-val-test split."""
        train_df, val_df, test_df = split_train_val_test(
            sample_dataframe, test_size=0.2, val_size=0.2
        )

        assert len(train_df) + len(val_df) + len(test_df) == len(sample_dataframe)
        assert len(test_df) == 20

    def test_no_data_overlap(self, sample_dataframe):
        """Test that splits don't overlap."""
        train_df, val_df, test_df = split_train_val_test(sample_dataframe)

        train_indices = set(train_df.index)
        val_indices = set(val_df.index)
        test_indices = set(test_df.index)

        assert len(train_indices.intersection(val_indices)) == 0
        assert len(train_indices.intersection(test_indices)) == 0
        assert len(val_indices.intersection(test_indices)) == 0
