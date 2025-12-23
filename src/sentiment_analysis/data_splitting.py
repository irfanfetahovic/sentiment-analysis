"""
Data Splitting Module

This module contains functions for splitting datasets into train/validation/test sets.
"""

import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
from sentiment_analysis.utils import load_config, get_config_value
from sentiment_analysis.exceptions import InsufficientDataError

logger = logging.getLogger(__name__)

# Load configuration defaults at module level
_config = load_config()
DEFAULT_TEST_SIZE = get_config_value(_config, 'training', 'test_size', default=0.2)
DEFAULT_VAL_SIZE = get_config_value(_config, 'training', 'val_size', default=0.1)
DEFAULT_RANDOM_STATE = get_config_value(_config, 'training', 'random_state', default=42)


def split_train_test(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    stratify_column: Optional[str] = 'label',
    random_state: int = DEFAULT_RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    
    Args:
        df: Input DataFrame
        test_size: Fraction of data for test set (default from config.yaml)
        stratify_column: Column name for stratified splitting (default 'label')
        random_state: Random seed for reproducibility (default from config.yaml)
        
    Returns:
        Tuple of (train_df, test_df)
        
    Raises:
        InsufficientDataError: If dataset is too small for splitting
        ValueError: If test_size is not between 0 and 1
    """
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    
    if len(df) < 10:
        raise InsufficientDataError(required=10, actual=len(df))
    
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    # Check if we have enough data for the split
    min_samples_per_split = 2
    if len(df) * test_size < min_samples_per_split:
        raise InsufficientDataError(
            required=int(min_samples_per_split / test_size),
            actual=len(df)
        )
    
    stratify = df[stratify_column] if stratify_column and stratify_column in df.columns else None
    
    logger.info(f"Splitting {len(df)} samples into train/test (test_size={test_size})")
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )
    
    logger.info(f"Split complete: train={len(train_df)}, test={len(test_df)}")
    
    return train_df, test_df


def split_train_val_test(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    stratify_column: Optional[str] = 'label',
    random_state: int = DEFAULT_RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        test_size: Fraction of data for test set (default from config)
        val_size: Fraction of training data for validation set (default from config)
        stratify_column: Column name for stratified splitting (default 'label')
        random_state: Random seed for reproducibility (default from config)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
        
    Raises:
        InsufficientDataError: If dataset is too small for splitting
        ValueError: If test_size or val_size are not between 0 and 1
 
    """
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    
    if len(df) < 20:
        raise InsufficientDataError(required=20, actual=len(df))
    
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    if not 0 < val_size < 1:
        raise ValueError(f"val_size must be between 0 and 1, got {val_size}")
    
    if test_size + val_size >= 1:
        raise ValueError(
            f"test_size ({test_size}) + val_size ({val_size}) must be less than 1"
        )
    
    stratify = df[stratify_column] if stratify_column and stratify_column in df.columns else None
    
    logger.info(
        f"Splitting {len(df)} samples into train/val/test "
        f"(test_size={test_size}, val_size={val_size})"
    )
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state
    )
    
    # Second split: separate validation from training
    stratify_train = train_val_df[stratify_column] if stratify_column and stratify_column in train_val_df.columns else None
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        stratify=stratify_train,
        random_state=random_state
    )
    
    logger.info(
        f"Split complete: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    
    return train_df, val_df, test_df
