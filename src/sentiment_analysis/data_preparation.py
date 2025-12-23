"""
Data Preparation Module

This module orchestrates the complete data preparation pipeline for sentiment analysis,
including loading, validation, filtering, text preprocessing, and label conversion.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional
from sentiment_analysis.text_preprocessing import TextPreprocessor
from sentiment_analysis.label_conversion import score_to_label
from sentiment_analysis.utils import load_config, get_config_value
from sentiment_analysis.exceptions import DataLoadError, InsufficientDataError
from sentiment_analysis.constants import (
    PREPROCESSING_MODE_CLASSICAL,
    PROBLEM_TYPE_BINARY,
)

# Create logger for this module
logger = logging.getLogger(__name__)

# Load defaults from config
_config = load_config()
DEFAULT_SAMPLE_FRACTION = get_config_value(_config, "data", "sample_frac", default=0.1)
DEFAULT_RANDOM_STATE = get_config_value(_config, "training", "random_state", default=42)


def _preprocess_dataframe(
    df: pd.DataFrame,
    preprocessor: TextPreprocessor,
    text_column: str = "Text",
    output_column: str = "cleaned_text",
) -> pd.DataFrame:
    """
    Internal helper: Preprocess text column in a pandas DataFrame.

    Args:
        df: Input DataFrame
        preprocessor: TextPreprocessor instance
        text_column: Name of the column containing text
        output_column: Name for the output cleaned text column

    Returns:
        DataFrame with added cleaned text column

    Raises:
        ValueError: If text_column doesn't exist in DataFrame
    """
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found. Available: {', '.join(df.columns)}"
        )

    df = df.copy()

    logger.debug(f"Cleaning text in column '{text_column}'")
    df[output_column] = df[text_column].apply(preprocessor.clean_text)

    # Remove empty or whitespace-only texts
    initial_count = len(df)
    df = df[df[output_column].str.strip().astype(bool)]
    final_count = len(df)

    if initial_count > final_count:
        logger.warning(
            f"Removed {initial_count - final_count} rows with empty text after cleaning"
        )

    logger.info(f"Rows after cleaning: {final_count}")

    return df


def load_and_prepare_data(
    file_path: str,
    sample_frac: float = DEFAULT_SAMPLE_FRACTION,
    problem_type: str = PROBLEM_TYPE_BINARY,
    mode: str = PREPROCESSING_MODE_CLASSICAL,
    random_state: int = DEFAULT_RANDOM_STATE,
    skip_preprocessing: bool = False,
) -> pd.DataFrame:
    """
    Load and prepare data for sentiment analysis.

    Args:
        file_path: Path to CSV file
        sample_frac: Fraction of data to sample (for quick experiments)
        problem_type: 'binary' or '3-class'
        mode: 'classical' for classical ML, 'transformer' for transformers
        random_state: Random seed
        skip_preprocessing: If True, skip text preprocessing (for inference with models that preprocess internally)

    Returns:
        Prepared DataFrame with cleaned text and labels

    Raises:
        FileNotFoundError: If data file doesn't exist
        DataLoadError: If file cannot be loaded or has invalid format
        InsufficientDataError: If dataset is too small
    """
    # Validate file path
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    if file_path.suffix != ".csv":
        logger.warning(f"File {file_path} may not be a CSV file")

    # Validate parameters
    if not 0 < sample_frac <= 1.0:
        raise ValueError(f"sample_frac must be between 0 and 1, got {sample_frac}")

    logger.info(f"Loading data from {file_path}")

    try:
        # Load data
        df = pd.read_csv(file_path)
        logger.info(f"Original rows: {len(df)}")
    except Exception as e:
        raise DataLoadError(file_path, str(e))

    # Validate required columns
    if "Text" not in df.columns or "Score" not in df.columns:
        raise DataLoadError(
            file_path,
            f"Missing required columns 'Text' or 'Score'. Available: {', '.join(df.columns)}",
        )

    # Keep relevant columns and drop NA
    df = df[["Score", "Text"]].dropna()
    logger.info(f"After dropping NA: {len(df)}")

    # Sample data if needed
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)
        logger.info(f"After sampling {sample_frac*100}%: {len(df)}")

    # Remove neutral reviews for binary classification
    if problem_type == "binary":
        df = df[df["Score"] != 3]
        logger.info(f"After removing neutral reviews: {len(df)}")

    # Check if we have enough data
    if len(df) < 10:
        raise InsufficientDataError(required=10, actual=len(df))

    # Preprocess text based on mode (skip if requested for inference)
    if not skip_preprocessing:
        logger.info(f"Preprocessing mode: {mode}")
        preprocessor = TextPreprocessor(mode=mode)
        df = _preprocess_dataframe(df, preprocessor)
    else:
        logger.info("Skipping preprocessing (will be handled by model)")

    # Convert scores to labels
    logger.debug("Converting scores to labels")
    df["label"] = df["Score"].apply(lambda x: score_to_label(x, problem_type))

    logger.info(f"\nLabel distribution:\n{df['label'].value_counts()}")

    return df
