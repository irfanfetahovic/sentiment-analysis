"""
Model Evaluation Module

This module contains functions for evaluating sentiment analysis models.
Separated from training to allow independent evaluation of pre-trained models.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

logger = logging.getLogger(__name__)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None, # Optional names for labels
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate classification model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Names for labels (for classification report)
        verbose: Whether to print results
        
    Returns:
        Dictionary with evaluation metrics (accuracy, f1_macro, precision, recall, etc.)
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get classification report
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    
    logger.info(f"Evaluation complete: accuracy={accuracy:.4f}, f1_macro={f1_macro:.4f}")
    
    if verbose:
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score (macro): {f1_macro:.4f}")
        logger.info(f"F1 Score (weighted): {f1_weighted:.4f}")
        logger.info(f"Precision (macro): {precision:.4f}")
        logger.info(f"Recall (macro): {recall:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"\n{cm}")
        logger.info(f"\nClassification Report:")
        logger.info(f"\n{classification_report(y_true, y_pred, target_names=label_names)}")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision,
        'recall_macro': recall,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def evaluate_model(
    model: Any, # model can be of any type
    X_test: Any,
    y_test: np.ndarray,
    label_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    This is a convenience function that makes predictions and evaluates them.
    
    Args:
        model: Trained model with predict() method
        X_test: Test features
        y_test: True test labels
        label_names: Names for labels
        verbose: Whether to print results
        
    Returns:
        Dictionary with evaluation metrics and predictions
        
    Raises:
        AttributeError: If model doesn't have predict() method
    """
    if not hasattr(model, 'predict'):
        raise AttributeError("Model must have a predict() method")
    
    logger.info(f"Evaluating model on {len(y_test)} test samples")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    metrics = evaluate_classification(y_true=y_test, y_pred=y_pred, label_names=label_names, verbose=verbose)
    
    # Add predictions to results
    metrics['predictions'] = y_pred
    
    return metrics


def compare_models(
    models_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    label_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple models' predictions.
    
    Args:
        models_predictions: Dictionary mapping model names to their predictions
        y_true: True labels
        label_names: Names for labels
        
    Returns:
        Dictionary mapping model names to their evaluation metrics
        
    Examples:
        >>> predictions = {
        ...     'LogisticRegression': model1_preds,
        ...     'DistilBERT': model2_preds
        ... }
        >>> comparison = compare_models(predictions, y_true)
        >>> comparison['LogisticRegression']['accuracy']
        0.85
    """
    logger.info(f"Comparing {len(models_predictions)} models")
    
    results = {}
    
    for model_name, y_pred in models_predictions.items():
        logger.info(f"Evaluating {model_name}...")
        metrics = evaluate_classification(y_true, y_pred, label_names=label_names, verbose=False)
        results[model_name] = metrics
    
    # Print comparison table
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON")
    logger.info("="*80)
    logger.info(f"{'Model':<30} {'Accuracy':>10} {'F1-Macro':>10} {'Precision':>10} {'Recall':>10}")
    logger.info("-"*80)
    
    for model_name, metrics in results.items():
        logger.info(f"{model_name:<30} {metrics['accuracy']:>10.4f} {metrics['f1_macro']:>10.4f} "
              f"{metrics['precision_macro']:>10.4f} {metrics['recall_macro']:>10.4f}")
    
    logger.info("="*80)
    
    return results
