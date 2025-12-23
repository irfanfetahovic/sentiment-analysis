"""
Sentiment Analysis Package

A production-ready sentiment analysis library using classical NLP and transformers.
"""

__version__ = "1.0.0"
__author__ = "Irfan Fetahovic"

from .inference import SentimentPredictor
from .text_preprocessing import TextPreprocessor
from .evaluation import evaluate_model, evaluate_classification, compare_models

# Define the public API of the package
__all__ = [
    "SentimentPredictor", 
    "TextPreprocessor",
    "evaluate_model",
    "evaluate_classification",
    "compare_models"
]
