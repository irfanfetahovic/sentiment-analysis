"""
Custom Exceptions Module

This module contains domain-specific exceptions for the sentiment analysis project.

"""


class SentimentAnalysisError(Exception):
    """Base exception for all sentiment analysis errors."""

    pass  # simply exists to be inherited


class InvalidScoreError(SentimentAnalysisError):
    """Raised when a review score is outside the valid range (1-5)."""

    def __init__(self, score: int, min_score: int = 1, max_score: int = 5):
        self.score = score
        self.min_score = min_score
        self.max_score = max_score
        super().__init__(
            f"Invalid score: {score}. Score must be between {min_score} and {max_score}."
        )


class ModelNotFittedError(SentimentAnalysisError):
    """Raised when trying to use a model that hasn't been trained yet."""

    def __init__(self, model_name: str = "Model"):
        super().__init__(
            f"{model_name} must be fitted before making predictions. "
            "Call the fit() method first."
        )


class InvalidModelTypeError(SentimentAnalysisError):
    """Raised when an invalid model type is specified."""

    def __init__(self, model_type: str, valid_types: list):
        self.model_type = model_type
        self.valid_types = valid_types
        super().__init__(
            f"Invalid model type: '{model_type}'. "
            f"Valid types are: {', '.join(valid_types)}"
        )


class InvalidPreprocessingModeError(SentimentAnalysisError):
    """Raised when an invalid preprocessing mode is specified."""

    def __init__(self, mode: str, valid_modes: list):
        self.mode = mode
        self.valid_modes = valid_modes
        super().__init__(
            f"Invalid preprocessing mode: '{mode}'. "
            f"Valid modes are: {', '.join(valid_modes)}"
        )


class DataLoadError(SentimentAnalysisError):
    """Raised when data loading fails."""

    def __init__(self, file_path: str, reason: str = None):
        self.file_path = file_path
        message = f"Failed to load data from '{file_path}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class ModelLoadError(SentimentAnalysisError):
    """Raised when model loading fails."""

    def __init__(self, model_path: str, reason: str = None):
        self.model_path = model_path
        message = f"Failed to load model from '{model_path}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class ModelSaveError(SentimentAnalysisError):
    """Raised when model saving fails."""

    def __init__(self, model_path: str, reason: str = None):
        self.model_path = model_path
        message = f"Failed to save model to '{model_path}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class InvalidProblemTypeError(SentimentAnalysisError):
    """Raised when an invalid problem type is specified."""

    def __init__(self, problem_type: str, valid_types: list):
        self.problem_type = problem_type
        self.valid_types = valid_types
        super().__init__(
            f"Invalid problem type: '{problem_type}'. "
            f"Valid types are: {', '.join(valid_types)}"
        )


class InvalidTextInputError(SentimentAnalysisError):
    """Raised when text input is invalid (empty, too long, wrong type)."""

    def __init__(self, reason: str):
        super().__init__(f"Invalid text input: {reason}")


class InsufficientDataError(SentimentAnalysisError):
    """Raised when there's not enough data for processing."""

    def __init__(self, required: int, actual: int):
        super().__init__(
            f"Insufficient data: requires at least {required} samples, "
            f"but only {actual} available."
        )
