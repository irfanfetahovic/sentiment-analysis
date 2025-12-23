"""
Transformer Model Training Module

This module contains functions for training transformer-based models
(DistilBERT) for sentiment analysis.
"""

import numpy as np
import pandas as pd
import torch
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datasets import Dataset, DatasetDict
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
import evaluate

from sentiment_analysis.data_preparation import load_and_prepare_data
from sentiment_analysis.data_splitting import split_train_val_test
from sentiment_analysis.utils import load_config, get_config_value, register_trained_model
from sentiment_analysis.settings import settings
from sentiment_analysis.constants import (
    PREPROCESSING_MODE_TRANSFORMER,
    BINARY_LABEL_NAMES,
    THREE_CLASS_LABEL_NAMES
)

logger = logging.getLogger(__name__)

# Load config defaults
_config = load_config()
DEFAULT_MODEL_NAME = get_config_value(_config, 'training', 'transformer', 'model_name', default='distilbert-base-uncased')
DEFAULT_MAX_LENGTH = get_config_value(_config, 'training', 'transformer', 'max_length', default=128)
DEFAULT_BATCH_SIZE = get_config_value(_config, 'training', 'transformer', 'batch_size', default=16)
DEFAULT_NUM_EPOCHS = get_config_value(_config, 'training', 'transformer', 'num_epochs', default=3)
DEFAULT_LEARNING_RATE = get_config_value(_config, 'training', 'transformer', 'learning_rate', default=2e-5)
DEFAULT_VAL_SIZE = get_config_value(_config, 'data', 'val_size', default=0.1)
DEFAULT_TEST_SIZE = get_config_value(_config, 'data', 'test_size', default=0.2)
DEFAULT_RANDOM_STATE = get_config_value(_config, 'training', 'random_state', default=42)


class TransformerSentimentTrainer:
    """
    Trainer class for transformer-based sentiment analysis.
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_labels: int = 2,
        max_length: int = DEFAULT_MAX_LENGTH,
        device: str = None
    ):
        """
        Initialize transformer trainer.
        
        Args:
            model_name: Pre-trained model name from HuggingFace
            num_labels: Number of output labels
            max_length: Maximum sequence length
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.dataset = None
    
    def prepare_dataset(
        self,
        df: pd.DataFrame,
        text_column: str = 'Text',
        label_column: str = 'label',
        val_size: float = DEFAULT_VAL_SIZE,
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int = DEFAULT_RANDOM_STATE
    ) -> DatasetDict:
        """
        Prepare dataset for training.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            label_column: Name of label column
            val_size: Validation set size
            test_size: Test set size
            random_state: Random seed
            
        Returns:
            DatasetDict with train, validation, and test splits
        """
        # Split data using data_splitting module
        train_df, val_df, test_df = split_train_val_test(
            df,
            test_size=test_size,
            val_size=val_size,
            stratify_column=label_column,
            random_state=random_state
        )
        
        logger.info(f'Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')
        
        # Create HuggingFace datasets
        self.dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df[[text_column, label_column]].reset_index(drop=True)),
            'validation': Dataset.from_pandas(val_df[[text_column, label_column]].reset_index(drop=True)),
            'test': Dataset.from_pandas(test_df[[text_column, label_column]].reset_index(drop=True)),
        })
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
        
        # Tokenize
        def tokenize_fn(batch):
            return self.tokenizer(
                batch[text_column],
                truncation=True,
                padding=False,
                max_length=self.max_length
            )
        
        # Apply tokenization and remove text column
        self.dataset = self.dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=[text_column]
        )
        
        # Set format
        self.dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        return self.dataset
    
    def train(
        self,
        output_dir: str,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 2,
        save_total_limit: int = 2
    ):
        """
        Train the transformer model.
        
        Args:
            output_dir: Directory to save model
            learning_rate: Learning rate
            batch_size: Training batch size
            num_epochs: Number of epochs
            weight_decay: Weight decay for regularization
            early_stopping_patience: Patience for early stopping
            save_total_limit: Maximum number of checkpoints to keep
        """
        if self.dataset is None:
            raise ValueError("Dataset not prepared. Call prepare_dataset first.")
        
        # Initialize model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        )
        self.model.to(self.device)
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Metrics
        accuracy_metric = evaluate.load('accuracy')
        f1_metric = evaluate.load('f1')
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            acc = accuracy_metric.compute(predictions=preds, references=labels)
            f1 = f1_metric.compute(predictions=preds, references=labels, average='macro')
            return {'accuracy': acc['accuracy'], 'f1_macro': f1['f1']}
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            logging_steps=200,
            load_best_model_at_end=True,
            metric_for_best_model='f1_macro',
            greater_is_better=True,
            save_total_limit=save_total_limit,
            fp16=torch.cuda.is_available(),
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )
        
        # Train
        logger.info("Starting training...")
        self.trainer.train()
        
        # Save model and tokenizer
        # Save model config
        # Determine label names based on num_labels
        if self.num_labels == 2:
            label_names = BINARY_LABEL_NAMES
        elif self.num_labels == 3:
            label_names = THREE_CLASS_LABEL_NAMES
        else:
            # Generic labels for any other number
            label_names = [f'LABEL_{i}' for i in range(self.num_labels)]
        
        model_config = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'label_names': label_names,
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'label_names': ['NEGATIVE', 'POSITIVE'],
            'hyperparameters': {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'weight_decay': weight_decay,
                'early_stopping_patience': early_stopping_patience
            },
            'preprocessing': {
                'mode': 'transformer',
                'description': 'Minimal preprocessing for transformer models'
            }
        }
        config_path = Path(output_dir) / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        logger.info(f"Model config saved to {config_path}")
    
    def evaluate(self, split: str = 'test'):
        """
        Evaluate the trained model.
        
        Args:
            split: Which split to evaluate ('test' or 'validation')
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Model not trained. Call train first.")
        
        metrics = self.trainer.evaluate(self.dataset[split])
        logger.info(f"{split.capitalize()} metrics: {metrics}")
        
        return metrics


def train_model(
    data_path: str,
    output_dir: str = None,
    model_name: str = DEFAULT_MODEL_NAME,
    num_labels: int = 2,
    sample_frac: float = 0.1,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    random_state: int = DEFAULT_RANDOM_STATE
):
    """
    Complete training pipeline for transformer models.
    
    Args:
        data_path: Path to data CSV
        output_dir: Directory to save model (default: models/distilbert_sentiment)
        model_name: Pre-trained model name
        num_labels: Number of output labels (2 for binary, 3 for 3-class, etc.)
        sample_frac: Fraction of data to use
        max_length: Maximum sequence length
        batch_size: Training batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        random_state: Random seed
    """
    
    # Set default output_dir from config
    if output_dir is None:
        output_dir = settings.model_dir / 'distilbert_sentiment'
    else:
        output_dir = Path(output_dir)
    
    # Set seed
    set_seed(random_state)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading and preprocessing data...")
    df = load_and_prepare_data(
        data_path,
        sample_frac=sample_frac,
        mode=PREPROCESSING_MODE_TRANSFORMER,
        random_state=random_state
    )
    
    # Initialize trainer
    trainer = TransformerSentimentTrainer(
        model_name=model_name,
        num_labels=num_labels,
        max_length=max_length
    )
    
    # Prepare dataset
    logger.info("\nPreparing dataset...")
    trainer.prepare_dataset(df, text_column='Text', label_column='label')
    
    # Train
    logger.info("\nTraining model...")
    trainer.train(
        output_dir=output_dir,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs
    )
    
    # Evaluate
    logger.info("\nEvaluating on test set...")
    metrics = trainer.evaluate(split='test')
    
    # Register model in registry
    from pathlib import Path
    model_rel_path = Path(output_dir).relative_to(settings.project_root)
    register_trained_model(
        model_name=f"DistilBERT ({num_labels} classes)",
        model_path=str(model_rel_path),
        model_type='transformer'
    )
    
    return trainer, metrics
