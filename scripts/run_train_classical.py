#!/usr/bin/env python
"""
Train Classical Model Script

This script trains a classical ML model (Logistic Regression + TF-IDF) for sentiment analysis.
"""

import argparse
import os
import sys

# Add src to path (not needed after pip install -e .)
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from sentiment_analysis.train_classical import train_classical_model
from sentiment_analysis.utils import setup_logging, load_config, resolve_path
from sentiment_analysis.settings import settings
from sentiment_analysis.constants import DEFAULT_CONFIG_PATH


def main():
    """Main entry point for training classical model."""
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression with TF-IDF"
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to config file"
    )
    parser.add_argument("--data-path", type=str, help="Path to data CSV")
    parser.add_argument("--output-dir", type=str, help="Output directory for model")
    parser.add_argument(
        "--max-features", type=int, help="Maximum number of TF-IDF features"
    )
    parser.add_argument("--log-file", type=str, help="Path to log file")

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_file=args.log_file, level="INFO")

    # Load config file
    config = load_config(args.config)
    data_config = config.get("data", {})
    model_config = config.get("model", {}).get("classical", {})
    paths_config = config.get("paths", {})

    # Command line args override config file
    data_path = args.data_path or data_config.get("path", "data/Reviews.csv")
    output_dir = args.output_dir or str(settings.model_dir / "classical_models")
    max_features = args.max_features or model_config.get("max_features", 5000)

    # Resolve paths relative to project root
    data_path = resolve_path(data_path)
    output_dir = resolve_path(output_dir)

    train_classical_model(
        data_path=data_path, output_dir=output_dir, max_features=max_features
    )


if __name__ == "__main__":
    main()
