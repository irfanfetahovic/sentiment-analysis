"""
Label Conversion Module

This module contains functions for converting review scores to sentiment labels.
"""

import logging
from sentiment_analysis.constants import (
    MIN_REVIEW_SCORE,
    MAX_REVIEW_SCORE,
    BINARY_THRESHOLD,
)
from sentiment_analysis.exceptions import InvalidScoreError, InvalidProblemTypeError

logger = logging.getLogger(__name__)


def score_to_label(score: int, problem_type: str = "binary") -> int:
    """
    Convert review score to sentiment label.

    Args:
        score: Review score (1-5)
        problem_type: 'binary' or '3-class'

    Returns:
        Sentiment label (0: negative, 1: positive or neutral, 2: positive for 3-class)

    Raises:
        InvalidScoreError: If score is not in valid range (1-5)
        InvalidProblemTypeError: If problem_type is not 'binary' or '3-class'
    """
    # Validate score
    if not isinstance(score, (int, float)):
        raise InvalidScoreError(score, MIN_REVIEW_SCORE, MAX_REVIEW_SCORE)

    score = int(score)
    if score < MIN_REVIEW_SCORE or score > MAX_REVIEW_SCORE:
        raise InvalidScoreError(score, MIN_REVIEW_SCORE, MAX_REVIEW_SCORE)

    # Validate problem type
    valid_types = ["binary", "3-class"]
    if problem_type not in valid_types:
        raise InvalidProblemTypeError(problem_type, valid_types)

    # Convert score to label
    if problem_type == "3-class":
        if score <= 2:
            label = 0  # negative
        elif score == 3:
            label = 1  # neutral
        else:
            label = 2  # positive
    else:  # binary
        label = 0 if score <= BINARY_THRESHOLD else 1

    logger.debug(
        f"Converted score {score} to label {label} (problem_type={problem_type})"
    )
    return label
