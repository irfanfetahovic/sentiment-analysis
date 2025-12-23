"""
Unit tests for feature extraction module.
Tests TF-IDF feature extraction functionality.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from scipy.sparse import csr_matrix
from sentiment_analysis.feature_extraction import TFIDFFeatureExtractor
from sentiment_analysis.exceptions import ModelNotFittedError, ModelSaveError, ModelLoadError


class TestTFIDFFeatureExtractor:
    """Test cases for TFIDFFeatureExtractor class."""
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return [
            "This is a great product",
            "Terrible quality and service",
            "Amazing experience with this",
            "Worst purchase ever made",
            "Excellent quality and value"
        ]
    
    @pytest.fixture
    def test_texts(self):
        """Test texts for transformation."""
        return [
            "Great quality",
            "Bad service"
        ]
    
    def test_initialization_default(self):
        """Test initialization with default parameters."""
        extractor = TFIDFFeatureExtractor()
        
        assert extractor.max_features is not None
        assert extractor.ngram_range is not None
        assert not extractor.fitted
        assert extractor.vectorizer is not None
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        extractor = TFIDFFeatureExtractor(max_features=1000, ngram_range=(1, 2))
        
        assert extractor.max_features == 1000
        assert extractor.ngram_range == (1, 2)
        assert not extractor.fitted
    
    def test_initialization_invalid_max_features(self):
        """Test that invalid max_features raises error."""
        with pytest.raises(ValueError, match="max_features must be positive"):
            TFIDFFeatureExtractor(max_features=0)
        
        with pytest.raises(ValueError, match="max_features must be positive"):
            TFIDFFeatureExtractor(max_features=-5)
    
    def test_initialization_invalid_ngram_range(self):
        """Test that invalid ngram_range raises error."""
        with pytest.raises(ValueError, match="ngram_range must be a tuple"):
            TFIDFFeatureExtractor(ngram_range=[1, 2])
        
        with pytest.raises(ValueError, match="ngram_range must be a tuple"):
            TFIDFFeatureExtractor(ngram_range=(1,))
    
    def test_fit_transform_basic(self, sample_texts):
        """Test basic fit_transform functionality."""
        extractor = TFIDFFeatureExtractor(max_features=100)
        result = extractor.fit_transform(sample_texts)
        
        assert extractor.fitted
        assert isinstance(result, csr_matrix)
        assert result.shape[0] == len(sample_texts)
        assert result.shape[1] <= 100  # Should respect max_features
    
    def test_fit_transform_empty_texts(self):
        """Test that fit_transform with empty texts raises error."""
        extractor = TFIDFFeatureExtractor()
        
        with pytest.raises(ValueError, match="texts cannot be empty"):
            extractor.fit_transform([])
    
    def test_fit_transform_sets_fitted_flag(self, sample_texts):
        """Test that fit_transform sets the fitted flag."""
        extractor = TFIDFFeatureExtractor()
        assert not extractor.fitted
        
        extractor.fit_transform(sample_texts)
        assert extractor.fitted
    
    def test_transform_after_fit(self, sample_texts, test_texts):
        """Test transform on new texts after fitting."""
        extractor = TFIDFFeatureExtractor()
        extractor.fit_transform(sample_texts)
        
        result = extractor.transform(test_texts)
        
        assert isinstance(result, csr_matrix)
        assert result.shape[0] == len(test_texts)
        assert result.shape[1] == extractor.vectorizer.transform(sample_texts).shape[1]
    
    def test_transform_before_fit(self, test_texts):
        """Test that transform raises error before fitting."""
        extractor = TFIDFFeatureExtractor()
        
        with pytest.raises(ModelNotFittedError):
            extractor.transform(test_texts)
    
    def test_transform_empty_texts(self, sample_texts):
        """Test that transform with empty texts raises error."""
        extractor = TFIDFFeatureExtractor()
        extractor.fit_transform(sample_texts)
        
        with pytest.raises(ValueError, match="texts cannot be empty"):
            extractor.transform([])
    
    def test_get_feature_names_after_fit(self, sample_texts):
        """Test get_feature_names after fitting."""
        extractor = TFIDFFeatureExtractor()
        extractor.fit_transform(sample_texts)
        
        features = extractor.get_feature_names()
        
        assert features is not None
        assert len(features) > 0
        assert isinstance(features, np.ndarray)
    
    def test_get_feature_names_before_fit(self):
        """Test that get_feature_names raises error before fitting."""
        extractor = TFIDFFeatureExtractor()
        
        with pytest.raises(ModelNotFittedError):
            extractor.get_feature_names()
    
    def test_save_after_fit(self, sample_texts):
        """Test saving the vectorizer after fitting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = TFIDFFeatureExtractor()
            extractor.fit_transform(sample_texts)
            
            save_path = Path(tmpdir) / "vectorizer.pkl"
            extractor.save(str(save_path))
            
            assert save_path.exists()
    
    def test_save_creates_parent_directories(self, sample_texts):
        """Test that save creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = TFIDFFeatureExtractor()
            extractor.fit_transform(sample_texts)
            
            save_path = Path(tmpdir) / "subdir" / "another" / "vectorizer.pkl"
            extractor.save(str(save_path))
            
            assert save_path.exists()
            assert save_path.parent.exists()
    
    def test_save_before_fit(self):
        """Test that save raises error before fitting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = TFIDFFeatureExtractor()
            save_path = Path(tmpdir) / "vectorizer.pkl"
            
            with pytest.raises(ModelNotFittedError):
                extractor.save(str(save_path))
    
    def test_load_fitted_vectorizer(self, sample_texts, test_texts):
        """Test loading a fitted vectorizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First, fit and save
            extractor1 = TFIDFFeatureExtractor()
            original_shape = extractor1.fit_transform(sample_texts).shape
            save_path = Path(tmpdir) / "vectorizer.pkl"
            extractor1.save(str(save_path))
            
            # Then load and use
            extractor2 = TFIDFFeatureExtractor()
            assert not extractor2.fitted
            
            extractor2.load(str(save_path))
            
            assert extractor2.fitted
            result = extractor2.transform(test_texts)
            assert result.shape[1] == original_shape[1]
    
    def test_load_nonexistent_file(self):
        """Test that load raises error for nonexistent file."""
        extractor = TFIDFFeatureExtractor()
        
        with pytest.raises(FileNotFoundError, match="Vectorizer file not found"):
            extractor.load("nonexistent/path/vectorizer.pkl")
    
    def test_load_preserves_vocabulary(self, sample_texts):
        """Test that load preserves the vocabulary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Fit and save
            extractor1 = TFIDFFeatureExtractor()
            extractor1.fit_transform(sample_texts)
            original_features = set(extractor1.get_feature_names())
            save_path = Path(tmpdir) / "vectorizer.pkl"
            extractor1.save(str(save_path))
            
            # Load
            extractor2 = TFIDFFeatureExtractor()
            extractor2.load(str(save_path))
            loaded_features = set(extractor2.get_feature_names())
            
            assert original_features == loaded_features
    
    def test_ngram_range_affects_features(self):
        """Test that different ngram_range produces different features."""
        texts = ["great product", "bad quality"]
        
        # Unigrams only
        extractor1 = TFIDFFeatureExtractor(ngram_range=(1, 1))
        result1 = extractor1.fit_transform(texts)
        
        # Unigrams and bigrams
        extractor2 = TFIDFFeatureExtractor(ngram_range=(1, 2))
        result2 = extractor2.fit_transform(texts)
        
        # Bigrams should produce more features
        assert result2.shape[1] > result1.shape[1]
    
    def test_max_features_limits_vocabulary(self):
        """Test that max_features limits the vocabulary size."""
        texts = [
            "word1 word2 word3 word4 word5",
            "word6 word7 word8 word9 word10",
            "word11 word12 word13 word14 word15"
        ]
        
        extractor = TFIDFFeatureExtractor(max_features=5)
        result = extractor.fit_transform(texts)
        
        assert result.shape[1] <= 5
        assert len(extractor.get_feature_names()) <= 5
    
    def test_consistency_across_transforms(self, sample_texts, test_texts):
        """Test that multiple transforms produce consistent results."""
        extractor = TFIDFFeatureExtractor()
        extractor.fit_transform(sample_texts)
        
        result1 = extractor.transform(test_texts)
        result2 = extractor.transform(test_texts)
        
        # Results should be identical
        assert np.array_equal(result1.toarray(), result2.toarray())
    
    def test_fit_transform_logging(self, sample_texts, caplog):
        """Test that fit_transform produces appropriate log messages."""
        import logging
        caplog.set_level(logging.INFO)
        
        extractor = TFIDFFeatureExtractor()
        extractor.fit_transform(sample_texts)
        
        assert "Fitting TF-IDF vectorizer" in caplog.text
        assert "TF-IDF features extracted" in caplog.text
    
    def test_save_and_load_cycle(self, sample_texts, test_texts):
        """Test complete save and load cycle maintains functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Original extractor
            extractor1 = TFIDFFeatureExtractor(max_features=50, ngram_range=(1, 2))
            extractor1.fit_transform(sample_texts)
            result1 = extractor1.transform(test_texts)
            
            # Save
            save_path = Path(tmpdir) / "vectorizer.pkl"
            extractor1.save(str(save_path))
            
            # Load into new extractor
            extractor2 = TFIDFFeatureExtractor()
            extractor2.load(str(save_path))
            result2 = extractor2.transform(test_texts)
            
            # Results should be identical
            assert np.allclose(result1.toarray(), result2.toarray())
