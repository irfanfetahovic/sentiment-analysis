# Text Preprocessing Guide

## Overview

This project uses **two different preprocessing strategies** depending on the model type:

1. **Classical NLP Mode** - Extensive preprocessing for TF-IDF, Word2Vec, SVM, Logistic Regression
2. **Transformer Mode** - Minimal preprocessing for DistilBERT and other transformer models

## Why Different Preprocessing?

### Classical ML Models Need Heavy Preprocessing
Classical models (SVM, Logistic Regression, Naive Bayes) don't understand language context. They need explicit feature engineering:
- **Lemmatization** to reduce words to base forms (running → run)
- **Stopword removal** to reduce noise
- **Negation handling** to mark negated words (not_good, never_bad)
- **POS tagging** for better lemmatization

### Transformers Need Minimal Preprocessing
Transformer models (BERT, DistilBERT) learn these features automatically:
- Pre-trained on massive text corpora
- Understand context and word relationships
- Benefit from keeping stopwords (they provide context)
- Too much preprocessing can hurt performance

---

## Classical Mode (`mode='classical'`)

Used for: **TF-IDF, Word2Vec, GloVe, SVM, Logistic Regression, Naive Bayes**

### Preprocessing Steps:

1. **HTML Tag Removal**
   ```python
   text = BeautifulSoup(text, "html.parser").get_text()
   ```

2. **URL Removal**
   ```python
   text = re.sub(r'http\S+|www\S+', '', text)
   ```

3. **Emoji Conversion** (demojize)
   ```python
   text = emoji.demojize(text)  # 😊 → :smiling_face:
   ```

4. **Contraction Expansion**
   ```python
   text = contractions.fix(text)  # don't → do not
   ```

5. **Lowercasing**
   ```python
   text = text.lower()
   ```

6. **Number Replacement**
   ```python
   text = re.sub(r'\d+', ' <num> ', text)  # 123 → <num>
   ```

7. **Remove Non-alphabetic Characters**
   ```python
   text = re.sub(r'[^a-zA-Z\s<>]', '', text)
   ```

8. **Tokenization**
   ```python
   tokens = word_tokenize(text)
   ```

9. **Negation Handling**
   ```python
   tokens = mark_negation(tokens)  # ["not", "good"] → ["not", "good_NEG"]
   ```

10. **POS Tagging**
    ```python
    pos_tags = pos_tag(tokens)
    ```

11. **Lemmatization with POS**
    ```python
    lemma = lemmatizer.lemmatize(word, pos)  # running → run
    ```

12. **Stopword Removal** (keep negation-marked words)
    ```python
    if word in stop_words and not word.endswith('_NEG'):
        continue
    ```

13. **Remove Short Words** (except negations)
    ```python
    clean_tokens = [w for w in tokens if len(w) > 2 or w.endswith('_NEG')]
    ```

### Example:

**Input:**
```
"This product is <b>fantastic</b>! I can't believe how great it is. 😊 https://example.com"
```

**Output (Classical Mode):**
```
"product fantastic cannot believe great"
```

### Usage:

```python
from sentiment_analysis.text_preprocessing import TextPreprocessor
from sentiment_analysis.data_preprocessing import preprocess_dataframe
from sentiment_analysis.data_loading import load_and_prepare_data

# For classical models
preprocessor = TextPreprocessor(mode='classical')
cleaned = preprocessor.clean_text(text)

# Or load full dataset
df = load_and_prepare_data(
    'data/Reviews.csv',
    mode='classical',
    problem_type='binary'
)
```

---

## Transformer Mode (`mode='transformer'`)

Used for: **DistilBERT, BERT, RoBERTa, other transformer models**

### Preprocessing Steps:

1. **HTML Tag Removal** (regex-based)
   ```python
   text = re.sub(r'<[^>]+>', '', text)
   ```

2. **URL Removal**
   ```python
   text = re.sub(r'http\S+|www\S+', '', text)
   ```

3. **Whitespace Cleanup**
   ```python
   text = re.sub(r'\s+', ' ', text).strip()
   ```

**That's it!** The transformer's tokenizer handles the rest.

### What We DON'T Do:
- ❌ Lemmatization (transformer learns word forms)
- ❌ Stopword removal (stopwords provide context)
- ❌ Lowercasing (case can be informative)
- ❌ Negation marking (transformer understands negation)
- ❌ Emoji removal (can keep sentiment signal)
- ❌ Contraction expansion (transformer handles it)

### Example:

**Input:**
```
"This product is <b>fantastic</b>! I can't believe how great it is. 😊 https://example.com"
```

**Output (Transformer Mode):**
```
"This product is fantastic! I can't believe how great it is. 😊"
```

### Usage:

```python
from sentiment_analysis.text_preprocessing import TextPreprocessor
from sentiment_analysis.data_preprocessing import preprocess_dataframe
from sentiment_analysis.data_loading import load_and_prepare_data

# For transformer models
preprocessor = TextPreprocessor(mode='transformer')
cleaned = preprocessor.clean_text(text)

# Or load full dataset
df = load_and_prepare_data(
    'data/Reviews.csv',
    mode='transformer',
    problem_type='binary'
)
```

---

## Quick Comparison Table

| Preprocessing Step | Classical Mode | Transformer Mode |
|-------------------|----------------|------------------|
| HTML removal | ✅ Yes | ✅ Yes |
| URL removal | ✅ Yes | ✅ Yes |
| Whitespace cleanup | ✅ Yes | ✅ Yes |
| Emoji handling | ✅ Convert to text | ❌ Keep as-is |
| Contraction expansion | ✅ Yes | ❌ No |
| Lowercasing | ✅ Yes | ❌ No |
| Number replacement | ✅ Yes (<num> token) | ❌ No |
| Tokenization | ✅ NLTK word_tokenize | ❌ Transformer tokenizer |
| Negation marking | ✅ Yes | ❌ No |
| Lemmatization | ✅ Yes (with POS) | ❌ No |
| Stopword removal | ✅ Yes (except NEG) | ❌ No |
| Short word removal | ✅ Yes | ❌ No |

---

## Notebook Alignment

### `sentiment_analysis_classicalNLP.ipynb`
- Uses **classical mode** preprocessing
- TF-IDF with max_features=5000
- Logistic Regression with solver='saga'
- Extensive text cleaning pipeline

### `sentiment_analysis.ipynb`
- Uses **transformer mode** preprocessing
- DistilBERT with max_length=128
- Learning rate 2e-5, batch size 16
- Minimal preprocessing, regex-based cleaning

---

## Best Practices

### For Classical Models:
```python
# Heavy preprocessing required
preprocessor = TextPreprocessor(mode='classical', use_negation=True, remove_stopwords=True)

# Use TF-IDF with limited features
from sentiment_analysis.feature_extraction import TFIDFFeatureExtractor
tfidf = TFIDFFeatureExtractor(max_features=5000, ngram_range=(1, 1))
```

### For Transformers:
```python
# Light preprocessing only
preprocessor = TextPreprocessor(mode='transformer')

# Let the model do the work
trainer = TransformerSentimentTrainer(
    model_name='distilbert-base-uncased',
    max_length=128
)
```

---

## Code Examples

### Running Data Preprocessing Script

```bash
# Test both modes
python -m sentiment_analysis.data_preprocessing
```

Output shows the difference:
```
===========================================================
CLASSICAL MODE (for TF-IDF, Word2Vec, SVM, etc.)
===========================================================
Original: This product is <b>fantastic</b>! I can't believe...
Cleaned: product fantastic cannot believe great

===========================================================
TRANSFORMER MODE (for DistilBERT, BERT, etc.)
===========================================================
Original: This product is <b>fantastic</b>! I can't believe...
Cleaned: This product is fantastic! I can't believe how great it is. 😊
```

---

## Performance Impact

### Classical Models
- **More preprocessing = Better results**
- Lemmatization improves generalization
- Stopword removal reduces noise
- Negation handling critical for sentiment

### Transformers
- **Less preprocessing = Better results**
- Pre-trained embeddings capture semantics
- Overly clean text can confuse the model
- Stopwords provide valuable context

---

## Summary

✅ **Use `mode='classical'`** when:
- Training TF-IDF, Word2Vec, SVM, Logistic Regression
- Working with classical NLP pipelines
- Need explicit feature engineering

✅ **Use `mode='transformer'`** when:
- Fine-tuning BERT, DistilBERT, RoBERTa
- Want minimal preprocessing
- Leveraging pre-trained language models

**Remember:** The preprocessing mode must match your model type for best results!
