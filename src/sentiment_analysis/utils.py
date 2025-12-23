"""
Utility Functions Module

This module contains utility functions used across the project.

DEPRECATED: get_project_root() and resolve_path() are deprecated.
Use settings.py module instead for path management.
"""

import yaml
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from sentiment_analysis.settings import settings
from sentiment_analysis.constants import BINARY_LABEL_NAMES, THREE_CLASS_LABEL_NAMES


def setup_logging(log_file: str = None, level: str = "INFO"):
    """
    Set up logging configuration.

    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, level),
            format=log_format,
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(level=getattr(logging, level), format=log_format)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file (relative to project root or absolute)

    Returns:
        Configuration dictionary
    """
    # Import here to avoid circular dependency
    from sentiment_analysis.settings import settings

    config_file = settings.resolve_path(config_path)

    if not config_file.exists():
        return {}

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def resolve_path(path: str) -> Path:
    """
    Resolve path relative to project root if not absolute.

    DEPRECATED: Use settings.resolve_path() instead.
    This function is kept for backward compatibility.

    Args:
        path: Path to resolve (can be relative or absolute)

    Returns:
        Absolute Path object
    """
    from sentiment_analysis.settings import settings

    return settings.resolve_path(path)


def get_config_value(config: Dict[str, Any], *keys, default=None):
    """
    Safely get nested configuration value.

    Args:
        config: Configuration dictionary
        *keys: Nested keys to traverse
        default: Default value if key not found

    Returns:
        Configuration value or default

    Example:
        get_config_value(config, 'training', 'transformer', 'learning_rate', default=2e-5)
    """
    value = config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
        if value is None:
            return default
    return value


def get_label_names(num_labels: int) -> list:
    """
    Determine appropriate label names based on number of labels.

    Args:
        num_labels: Number of unique labels/classes

    Returns:
        List of label names

    Example:
        >>> get_label_names(2)
        ['NEGATIVE', 'POSITIVE']
        >>> get_label_names(3)
        ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
        >>> get_label_names(5)
        ['LABEL_0', 'LABEL_1', 'LABEL_2', 'LABEL_3', 'LABEL_4']
    """

    if num_labels == 2:
        return BINARY_LABEL_NAMES
    elif num_labels == 3:
        return THREE_CLASS_LABEL_NAMES
    else:
        # Generic labels for other cases
        return [f"LABEL_{i}" for i in range(num_labels)]


def register_trained_model(
    model_name: str,
    model_path: str,
    model_type: str,
    registry_file: str = "models_config.json",
) -> None:
    """
    Register a trained model in the models registry file.

    This function adds or updates a model entry in the registry JSON file,
    which is used for model comparison. If a model with the same path already
    exists, it updates the entry with new information.

    Args:
        model_name: Human-readable name for the model
        model_path: Path to the saved model (relative to project root)
        model_type: Type of model ('transformer' or 'classical')
        registry_file: Path to registry file (default: models_config.json in project root)
    """

    # Resolve registry file path
    registry_path = settings.project_root / registry_file

    # Load existing registry or create new one
    if registry_path.exists():
        with open(registry_path, "r") as f:
            try:
                models = json.load(f)
                if not isinstance(models, list):
                    models = []
            except json.JSONDecodeError:
                models = []
    else:
        models = []

    # Create model entry
    model_entry = {
        "name": model_name,
        "path": str(model_path),
        "type": model_type,
        "registered_at": datetime.now().isoformat(),
    }

    # Check if model already exists (by name) and update, otherwise append
    existing_index = None
    for i, model in enumerate(models):
        if model.get("name") == model_name:
            existing_index = i
            break

    if existing_index is not None:
        # Update existing entry
        models[existing_index] = model_entry
        logging.info(f"Updated model registry: {model_name} at {model_path}")
    else:
        # Add new entry
        models.append(model_entry)
        logging.info(f"Registered new model: {model_name} at {model_path}")

    # Save registry
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w") as f:
        json.dump(models, f, indent=2)

    logging.info(f"Model registry saved to {registry_path}")
