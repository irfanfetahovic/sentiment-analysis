#!/usr/bin/env python
"""
Train Transformer Model Script

This script trains a transformer model (DistilBERT) for sentiment analysis.
"""

import argparse
import os
import sys

# Add src to path (not needed after pip install -e .)
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sentiment_analysis.train_transformer import TransformerSentimentTrainer, train_model
from sentiment_analysis.utils import setup_logging, load_config, resolve_path
from sentiment_analysis.settings import settings
from sentiment_analysis.constants import DEFAULT_CONFIG_PATH


def main():
    """Main entry point for training transformer model."""
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to config file')
    parser.add_argument('--data-path', type=str,
                        help='Path to data CSV')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for model')
    parser.add_argument('--sample-frac', type=float,
                        help='Fraction of data to use')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int,
                        help='Training batch size')
    parser.add_argument('--log-file', type=str,
                        help='Path to log file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file=args.log_file, level='INFO')
    
    # Load config file
    config = load_config(args.config)
    data_config = config.get('data', {})
    training_config = config.get('training', {}).get('transformer', {})
    paths_config = config.get('paths', {})
    
    # Command line args override config file
    data_path = args.data_path or data_config.get('path', 'data/Reviews.csv')
    output_dir = args.output_dir or str(settings.model_dir / 'distilbert_sentiment')
    sample_frac = args.sample_frac or data_config.get('sample_frac', 0.1)
    num_epochs = args.epochs or training_config.get('num_epochs', 3)
    batch_size = args.batch_size or training_config.get('batch_size', 16)
    
    # Resolve paths relative to project root
    data_path = resolve_path(data_path)
    output_dir = resolve_path(output_dir)
    
    train_model(
        data_path=data_path,
        output_dir=output_dir,
        sample_frac=sample_frac,
        num_epochs=num_epochs,
        batch_size=batch_size
    )


if __name__ == "__main__":
    main()
