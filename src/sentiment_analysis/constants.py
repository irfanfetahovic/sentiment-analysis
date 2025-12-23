"""
Constants Module

This module contains type constants and constraints.
All configuration values (hyperparameters, paths, etc.) are in config/config.yaml.
"""

# Score and Label Constants
MIN_REVIEW_SCORE = 1  # Minimum review score
MAX_REVIEW_SCORE = 5  # Maximum review score
BINARY_THRESHOLD = 3  # Scores <= 3 are negative, > 3 are positive

# Label Mapping
LABEL_NEGATIVE = 0
LABEL_NEUTRAL = 1  # For 3-class classification
LABEL_POSITIVE = 2  # For 3-class classification (binary uses 0, 1)
BINARY_LABEL_NAMES = ["NEGATIVE", "POSITIVE"]
THREE_CLASS_LABEL_NAMES = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

# Model Type Identifiers
MODEL_TYPE_CLASSICAL = "classical"
MODEL_TYPE_TRANSFORMER = "transformer"

# Preprocessing Mode Identifiers
PREPROCESSING_MODE_CLASSICAL = "classical"
PREPROCESSING_MODE_TRANSFORMER = "transformer"

# Device Identifiers
DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"

# Problem Type Identifiers
PROBLEM_TYPE_BINARY = "binary"
PROBLEM_TYPE_MULTICLASS = "3-class"

# API Constraints
MAX_TEXT_LENGTH = 5000  # Maximum text length for API requests
MAX_BATCH_SIZE = 100  # Maximum number of texts per batch request

# API Defaults
DEFAULT_API_HOST = "0.0.0.0"
DEFAULT_API_PORT = 5000

# File Path References (for backward compatibility)
DEFAULT_CONFIG_PATH = "config/config.yaml"

# Model and Training Defaults (for tests and modules)
DEFAULT_TRANSFORMER_MODEL = "distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 128
DEFAULT_TEST_SIZE = 0.2
