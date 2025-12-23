# Model Comparison

## Overview
This document compares the performance of different sentiment analysis approaches implemented in this project.

**Note:** Performance metrics are representative examples based on similar implementations and may vary based on your specific dataset, preprocessing, and hyperparameters. Run your own evaluations for accurate performance measurements.

**Important:** Classical models and transformer models use **different preprocessing strategies**. See [PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md) for details.

## Models Evaluated

### 1. Classical NLP Models

**Preprocessing:** Extensive (lemmatization, stopword removal, negation handling, POS tagging)

#### Logistic Regression with TF-IDF
- **Type**: Classical ML
- **Features**: TF-IDF (5000 features, unigrams only)
- **Training Time**: ~5 minutes (CPU)
- **Preprocessing**: `mode='classical'`

**Pros:**
- Fast training and inference
- Interpretable feature weights
- Low resource requirements
- Good baseline performance

**Cons:**
- Limited context understanding
- Manual feature engineering required
- Struggles with complex negation and sarcasm

#### Support Vector Machine (SVM)
- **Type**: Classical ML
- **Features**: Word2Vec/GloVe embeddings
- **Training Time**: ~15 minutes (CPU)
- **Preprocessing**: `mode='classical'`

**Pros:**
- Strong performance with embeddings
- Handles non-linear relationships
- Robust to outliers

**Cons:**
- Slower training on large datasets
- Memory intensive
- Harder to interpret

### 2. Transformer Models

**Preprocessing:** Minimal (HTML/URL removal only)

#### DistilBERT
- **Type**: Transformer (fine-tuned)
- **Model**: distilbert-base-uncased
- **Training Time**: ~2 hours (GPU) / ~8 hours (CPU)
- **Preprocessing**: `mode='transformer'`
- **Hyperparameters** (from notebook):
  - Learning rate: 2e-5
  - Batch size: 16
  - Max length: 128
  - Epochs: 3
  - Early stopping patience: 2

**Pros:**
- State-of-the-art performance
- Contextual understanding
- Transfer learning from pre-training
- Handles complex language patterns
- No manual feature engineering

**Cons:**
- Requires GPU for efficient training
- Larger model size (~250MB)
- Slower inference than classical methods
- Needs more data for fine-tuning

## Performance Metrics

### Binary Classification Results (Amazon Reviews)

**Note:** These are example metrics. Actual performance will vary based on your dataset and configuration.

| Model | Accuracy | F1-Score | Precision | Recall | Inference Speed |
|-------|----------|----------|-----------|--------|-----------------||
| Logistic Regression (TF-IDF) | 91-92% | 0.91-0.92 | 0.91-0.92 | 0.91 | ~500 texts/sec |
| SVM (Word2Vec) | 90-91% | 0.89-0.90 | 0.90 | 0.89 | ~200 texts/sec |
| DistilBERT | **94-95%** | **0.94-0.95** | **0.94-0.95** | **0.94** | ~50 texts/sec (GPU) |

### Detailed Classification Report (DistilBERT)

```
              precision    recall  f1-score   support

    NEGATIVE       0.92      0.89      0.91      2845
    POSITIVE       0.96      0.98      0.97      8976

    accuracy                           0.94     11821
   macro avg       0.94      0.94      0.94     11821
weighted avg       0.94      0.94      0.94     11821
```

## Recommendations

### Use Logistic Regression when:
- Need fast training and inference
- Limited computational resources
- Want interpretable model
- Working with structured product reviews

### Use DistilBERT when:
- Maximum accuracy is priority
- Have GPU available
- Sufficient training data (>10k samples)
- Complex language understanding needed

## Resource Requirements

| Model | Disk Space | RAM | GPU | Training Data |
|-------|-----------|-----|-----|---------------|
| Logistic Regression | ~50 MB | 2-4 GB | No | 5k+ samples |
| SVM | ~100 MB | 4-8 GB | No | 10k+ samples |
| DistilBERT | ~250 MB | 8-16 GB | Recommended | 20k+ samples |

## Training Curves

For DistilBERT, typical training progression:
- Epoch 1: ~88% accuracy
- Epoch 2: ~92% accuracy
- Epoch 3: ~94% accuracy
- Early stopping typically at epoch 3-4

## Error Analysis

### Common Errors for All Models:
1. **Sarcasm**: "Oh great, another broken product" (predicted positive)
2. **Mixed sentiment**: "Good quality but terrible customer service"
3. **Domain-specific terms**: Unknown product names or technical terms

### DistilBERT Advantages:
- Better handling of negation: "not bad" correctly identified as positive
- Context understanding: "This could be better" identified as negative
- Long-range dependencies in longer reviews

## Conclusion

- **For production deployment**: DistilBERT offers the best accuracy-performance tradeoff
- **For resource-constrained environments**: Logistic Regression with TF-IDF
- **For balanced approach**: Consider ensemble of Logistic Regression + DistilBERT
- **For experimentation**: Start with Logistic Regression baseline, then fine-tune DistilBERT
