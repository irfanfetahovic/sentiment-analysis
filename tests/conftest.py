"""
Test configuration and fixtures.
"""

import pytest
import os
import sys
import tempfile
from pathlib import Path


# Creating fixtures for test data used across multiple test files.
@pytest.fixture(scope="session")
def sample_texts():
    """Sample texts for testing (binary classification)."""
    return [
        "This product is amazing! I love it.",
        "Terrible quality. Would not recommend.",
        "It's okay, nothing special.",
        "Best purchase ever! Highly recommended.",
        "Waste of money. Very disappointed.",
    ]


# Creating fixtures for test data used across multiple test files.
@pytest.fixture(scope="session")
def sample_labels():
    """Sample labels corresponding to sample texts (binary)."""
    return [1, 0, 1, 1, 0]  # 1=positive, 0=negative


@pytest.fixture(scope="session")
def sample_texts_three_class():
    """Sample texts for 3-class classification testing."""
    return [
        "This product is absolutely amazing! Best purchase ever!",
        "Terrible quality. Completely disappointed. Would not recommend.",
        "It's okay, nothing special. Average quality.",
        "Excellent service! Highly recommended!",
        "Very poor experience. Waste of money.",
        "Decent product, does what it says.",
        "Outstanding quality! Worth every penny!",
        "Mediocre at best. Nothing to write home about.",
        "Awful. Complete waste of time.",
    ]


@pytest.fixture(scope="session")
def sample_labels_three_class():
    """Sample labels for 3-class classification (0=negative, 1=neutral, 2=positive)."""
    return [2, 0, 1, 2, 0, 1, 2, 1, 0]


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_data_file():
    """Create temporary CSV file for data testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("Score,Text\n")
        f.write("5,Great product\n")
        f.write("1,Terrible quality\n")
        f.write("4,Good value\n")
        f.write("2,Not great\n")
        f.flush()
        yield Path(f.name)

    # Cleanup
    Path(f.name).unlink(missing_ok=True)
