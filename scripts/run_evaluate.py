#!/usr/bin/env python
"""
Model Evaluation Script

This script evaluates trained sentiment analysis models on test data.
Supports both classical and transformer models, single model evaluation,
and comparison of multiple models.
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

from sentiment_analysis.inference import SentimentPredictor
from sentiment_analysis.evaluation import evaluate_model, compare_models
from sentiment_analysis.data_preparation import load_and_prepare_data
from sentiment_analysis.data_splitting import split_train_test
from sentiment_analysis.utils import setup_logging, load_config, resolve_path, get_label_names
from sentiment_analysis.constants import (
    DEFAULT_CONFIG_PATH,
    MODEL_TYPE_TRANSFORMER,
    MODEL_TYPE_CLASSICAL,
    DEVICE_CPU
)
from sentiment_analysis.exceptions import DataLoadError


def evaluate_single_model(
    model_path: str,
    model_type: str,
    X_test: Any,
    y_test: Any,
    device: str = DEVICE_CPU,
    output_file: str = None
) -> Dict[str, Any]:
    """
    Evaluate a single model.
    
    Args:
        model_path: Path to saved model
        model_type: Type of model ('transformer' or 'classical')
        X_test: Test features
        y_test: Test labels
        device: Device for inference
        output_file: Optional path to save results as JSON
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {model_type.upper()} model")
    print(f"Model path: {model_path}")
    print(f"{'='*80}\n")
    
    # Load model
    predictor = SentimentPredictor(
        model_path=model_path,
        model_type=model_type,
        device=device
    )
    
    # Get num_labels from model configuration
    model_info = predictor.get_model_info()
    if 'config' in model_info and 'num_labels' in model_info['config']:
        num_labels = model_info['config']['num_labels']
    else:
        # Fallback: detect from test data if config not available
        import numpy as np
        num_labels = len(np.unique(y_test))
        print(f"Warning: num_labels not found in model config, detected from test data")
    
    label_names = get_label_names(num_labels)
    print(f"Using {num_labels} classes: {label_names}")
    
    # Evaluate
    metrics = evaluate_model(
        model=predictor,
        X_test=X_test,
        y_test=y_test,
        label_names=label_names,
        verbose=True
    )
    
    # Remove predictions from saved results (too large)
    metrics_to_save = {k: v for k, v in metrics.items() if k != 'predictions'}
    
    # Save results if requested
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return metrics


def compare_multiple_models(
    model_configs: list,
    X_test: Any,
    y_test: Any,
    device: str = DEVICE_CPU,
    output_file: str = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models.
    
    Args:
        model_configs: List of dicts with 'name', 'path', and 'type' keys
        X_test: Test features
        y_test: Test labels
        device: Device for inference
        output_file: Optional path to save comparison results
        
    Returns:
        Dictionary mapping model names to their metrics
    """
    print(f"\n{'='*80}")
    print(f"Comparing {len(model_configs)} models")
    print(f"{'='*80}\n")
    
    # Get predictions from all models
    all_predictions = {}
    num_labels = None
    
    for config in model_configs:
        model_name = config['name']
        model_path = config['path']
        model_type = config['type']
        
        print(f"Loading {model_name}...")
        predictor = SentimentPredictor(
            model_path=model_path,
            model_type=model_type,
            device=device
        )
        
        # Get num_labels from first model's config
        if num_labels is None:
            model_info = predictor.get_model_info()
            if 'config' in model_info and 'num_labels' in model_info['config']:
                num_labels = model_info['config']['num_labels']
        
        # Get predictions
        predictions = predictor.predict(X_test)
        
        # Handle different prediction formats
        if isinstance(predictions, list):
            # List of prediction dicts
            if predictions and isinstance(predictions[0], dict):
                all_predictions[model_name] = [p['label'] for p in predictions]
            else:
                # List of labels directly
                all_predictions[model_name] = predictions
        elif isinstance(predictions, dict):
            # Single prediction dict
            all_predictions[model_name] = [predictions['label']]
        else:
            # Assume it's an array-like of labels (numpy array, etc.)
            all_predictions[model_name] = list(predictions)
    
    # Fallback: detect from test data if config not available
    if num_labels is None:
        import numpy as np
        num_labels = len(np.unique(y_test))
        print(f"Warning: num_labels not found in model configs, detected from test data")
    
    label_names = get_label_names(num_labels)
    print(f"\nUsing {num_labels} classes: {label_names}")
    
    # Compare models
    results = compare_models(
        models_predictions=all_predictions,
        y_true=y_test,
        label_names=label_names
    )
    
    # Save comparison results
    if output_file:
        # Remove non-serializable data
        results_to_save = {}
        for model_name, metrics in results.items():
            results_to_save[model_name] = {
                k: v for k, v in metrics.items() 
                if k not in ['predictions', 'confusion_matrix']
            }
            # Convert confusion matrix to list if present
            if 'confusion_matrix' in metrics:
                results_to_save[model_name]['confusion_matrix'] = metrics['confusion_matrix']
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\nComparison results saved to: {output_file}")
    
    return results


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description='Evaluate sentiment analysis models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single model
  python scripts/run_evaluate.py --model-path models/classical_models/logistic_tfidf_model.pkl --model-type classical

  # Compare multiple models
  python scripts/run_evaluate.py --compare --models-json models_config.json

  # Evaluate with custom test data
  python scripts/run_evaluate.py --model-path models/distilbert --data-path data/Reviews.csv --test-size 0.3
        """
    )
    
    # Model arguments
    parser.add_argument('--model-path', type=str,
                        help='Path to model for single evaluation')
    parser.add_argument('--model-type', type=str,
                        choices=[MODEL_TYPE_TRANSFORMER, MODEL_TYPE_CLASSICAL],
                        help='Type of model for single evaluation')
    
    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple models')
    parser.add_argument('--models-json', type=str,
                        help='JSON file with model configurations for comparison')
    
    # Data arguments
    parser.add_argument('--data-path', type=str,
                        help='Path to data CSV')
    parser.add_argument('--test-size', type=float,
                        help='Fraction of data to use for testing')
    parser.add_argument('--sample-frac', type=float,
                        help='Fraction of data to sample (default: 0.1 for faster evaluation)')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to config file')
    
    # Output arguments
    parser.add_argument('--output-file', type=str,
                        help='Path to save evaluation results as JSON')
    parser.add_argument('--device', type=str,
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference')
    parser.add_argument('--log-file', type=str,
                        help='Path to log file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file=args.log_file, level='INFO')
    
    # Load config
    config = load_config(args.config)
    data_config = config.get('data', {})
    paths_config = config.get('paths', {})
    training_config = config.get('training', {})
    inference_config = config.get('inference', {})
    
    # Get parameters from command-line args or config file or defaults
    data_path = args.data_path or data_config.get('path', 'data/Reviews.csv')
    test_size = args.test_size or training_config.get('test_size', 0.2)
    sample_frac = args.sample_frac or 0.1  # Default to 10% for faster evaluation
    device = args.device or inference_config.get('device', DEVICE_CPU)
    
    # Resolve data path relative to project root
    data_path = resolve_path(data_path)
    
    # Load and prepare test data (without preprocessing - let predictor handle it)
    print(f"Loading test data from: {data_path} (sample: {sample_frac*100:.1f}%)")
    try:
        df = load_and_prepare_data(data_path, sample_frac=sample_frac, skip_preprocessing=True)
        _, test_df = split_train_test(df, test_size=test_size)
        # Use raw Text - let predictor handle preprocessing internally
        X_test = test_df['Text'].tolist()
        y_test = test_df['label'].values
        print(f"Test set size: {len(y_test)} samples\n")
    except Exception as e:
        raise DataLoadError(data_path, str(e))
    
    # Comparison mode
    if args.compare:
        if not args.models_json:
            print("Error: --models-json required for comparison mode")
            sys.exit(1)
        
        # Load model configurations and resolve paths
        models_json_path = resolve_path(args.models_json)
        with open(models_json_path, 'r') as f:
            model_configs = json.load(f)
        
        # Resolve model paths relative to project root
        for config in model_configs:
            config['path'] = resolve_path(config['path'])
        
        compare_multiple_models(
            model_configs=model_configs,
            X_test=X_test,
            y_test=y_test,
            device=device,
            output_file=args.output_file
        )
    
    # Single model evaluation
    else:
        if not args.model_path or not args.model_type:
            print("Error: --model-path and --model-type required for single evaluation")
            print("Use --help for usage information")
            sys.exit(1)
        
        # Resolve model path relative to project root
        model_path = resolve_path(args.model_path)
        
        evaluate_single_model(
            model_path=model_path,
            model_type=args.model_type,
            X_test=X_test,
            y_test=y_test,
            device=device,
            output_file=args.output_file
        )
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
