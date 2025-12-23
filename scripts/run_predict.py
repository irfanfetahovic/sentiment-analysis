#!/usr/bin/env python
"""
Inference Script

This script runs sentiment prediction on input text using a trained model.
"""

import argparse
import os
import sys
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add src to path (not needed after pip install -e .)
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sentiment_analysis.inference import SentimentPredictor
from sentiment_analysis.utils import setup_logging, load_config, resolve_path
from sentiment_analysis.settings import settings
from sentiment_analysis.constants import (
    DEFAULT_CONFIG_PATH,
    MODEL_TYPE_TRANSFORMER,
    MODEL_TYPE_CLASSICAL,
    DEVICE_CPU
)
from sentiment_analysis.exceptions import InvalidTextInputError


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(
        description='Predict sentiment for text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict with default transformer model
  python scripts/run_predict.py --text "This product is amazing!"

  # Predict with classical model
  python scripts/run_predict.py --model-path models/classical_models/logistic_tfidf_model.pkl --model-type classical --text "Great quality"

  # Predict with custom model and GPU
  python scripts/run_predict.py --model-path models/distilbert_sentiment --model-type transformer --device cuda --text "Not satisfied"

  # Using sentiment-predict command (after pip install -e .)
  sentiment-predict --text "I love this product!"
        """
    )
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to config file')
    parser.add_argument('--model-path', type=str,
                        help='Path to saved model')
    parser.add_argument('--model-type', type=str,
                        choices=[MODEL_TYPE_TRANSFORMER, MODEL_TYPE_CLASSICAL],
                        help='Type of model')
    parser.add_argument('--text', type=str, required=True,
                        help='Text to analyze')
    parser.add_argument('--device', type=str,
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference')
    parser.add_argument('--log-file', type=str,
                        help='Path to log file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file=args.log_file, level='INFO')
    
    # Validate text input
    if not args.text or not args.text.strip():
        raise InvalidTextInputError("Text cannot be empty")
    
    # Load config file (auto-resolves to project root)
    config = load_config(args.config)
    paths_config = config.get('paths', {})
    inference_config = config.get('inference', {})
    
    # Resolve model path: CLI > env var > settings
    if args.model_path:
        model_path = args.model_path
    elif os.getenv('SENTIMENT_MODELS_DIR'):
        models_dir = os.getenv('SENTIMENT_MODELS_DIR')
        model_path = str(Path(models_dir) / 'distilbert_sentiment')
    else:
        model_path = str(settings.model_dir / 'distilbert_sentiment')
    
    # Resolve relative paths to project root
    model_path = resolve_path(model_path)
    
    model_type = args.model_type or MODEL_TYPE_TRANSFORMER
    device = args.device or inference_config.get('device', DEVICE_CPU)
    
    predictor = SentimentPredictor(
        model_path=model_path,
        model_type=model_type,
        device=device
    )
    
    result = predictor.predict_with_labels(args.text)
    print(f"\nText: {result['text']}")
    print(f"Sentiment: {result['label']}")
    if result['score'] is not None:
        print(f"Confidence: {result['score']:.4f}")
    else:
        print("Confidence: N/A (model does not support probability scores)")


if __name__ == "__main__":
    main()
