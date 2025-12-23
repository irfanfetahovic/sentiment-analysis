# Sentiment Analysis (Classical ML + DistilBERT)

Production-focused sentiment analysis comparing classical NLP models (TF‑IDF + Logistic Regression / other scikit-learn models) with a fine‑tuned DistilBERT transformer on Amazon product reviews.

## Overview

Two modeling paradigms:
- Classical ML (TF‑IDF features + linear model) with richer preprocessing
- Transformer (DistilBERT) with minimal text normalization

Design goals: reproducible training, configurable hyperparameters (YAML + env overrides), scriptable pipeline, Dockerized deployment, CI quality gates, and lightweight API serving.

## Dataset
Amazon Fine Food Reviews ([Kaggle link](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews))
- Binary mode: score > 3 → positive, score ≤ 2 → negative (score == 3 dropped)
- Data file expected at `data/Reviews.csv` (not tracked in git)

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/irfanfetahovic/sentiment-analysis.git
cd sentiment-analysis
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### 2. Download NLTK Resources (classical mode requirements)

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"
```

### 3. Verify Installation

```bash
pytest -q
```

## Configuration & Precedence

Central config: `config/config.yaml` (hyperparameters, paths, preprocessing options).

Runtime value resolution order:
1. CLI argument
2. Environment variable (e.g. `MODEL_TYPE`, `MODEL_PATH`)
3. `config/config.yaml`
4. Internal fallback

Edit YAML for persistent defaults; use env variables or CLI arguments for runtime experimentation.

## Scripts (CLI)

All runnable under PowerShell after activation:

```powershell
# Classical model training
python scripts\run_train_classical.py --config config\config.yaml

# Transformer (DistilBERT) training
python scripts\run_train_transformer.py --config config\config.yaml

# Full pipeline (prepare + train + evaluate)
python scripts\run_pipeline.py --config config\config.yaml

# Evaluate existing model
python scripts\run_evaluate.py --model-type transformer --model-path models\distilbert_sentiment

# Single prediction
python scripts\run_predict.py --text "Fantastic quality!" --model-type classical --model-path models\classical_models\logistic_tfidf_model.pkl
```

Omit flags to fall back to env → YAML defaults.

## API Serving

Two entry points (choose one):

```powershell
# FastAPI (default)
python app\app_fastapi.py

# Flask alternative
python app\app.py
```

Sample request (PowerShell curl alias may differ, use `Invoke-RestMethod` or `curl` from Git if installed):
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"text": "Great product, very satisfied!"}'
```

Response example:
```json
{
  "text": "Great product, very satisfied!",
  "label": "POSITIVE",
  "score": 0.9876,
  "processing_time_ms": 45
}
```
## Project Structure (Core)

```
sentiment-analysis/
├── app/                      # API server(s)
├── config/                   # YAML config + (optional) extras
├── data/                     # Raw data & embeddings (not tracked)
├── models/                   # Saved model artifacts (gitignored)
├── notebooks/                # Experiment notebooks & checkpoints
├── scripts/                  # CLI entry points (train, predict, evaluate)
├── src/                      # Python package code
├── tests/                    # Pytest suite
└── .github/workflows/        # CI (tests + GHCR build)
```

Classical/transformer tunables live in `config/config.yaml` — modify without touching code.

## Metrics (Illustrative)
Example previous runs (replace with current results):
- DistilBERT: Accuracy ~96%, F1 ~0.92
- Logistic Regression (TF‑IDF): Accuracy ~90%, F1 ~0.84
Recompute after changing preprocessing or sample size.

## Development

```powershell
# Tests + coverage
pytest tests\ --cov=. --cov-report=term

# Lint
flake8 src/ tests/ scripts/ app/

# Format check
black --check src/ tests/ scripts/ app/

# Type hints (allow missing imports)
mypy src/ tests/ scripts/ app/ --ignore-missing-imports
```

Use pre‑commit (if configured): `pre-commit run --all-files`.

## Docker & GHCR

CI builds and pushes images to GHCR (GitHub Container Registry).

**Local Docker Workflow:**

```bash
# Build locally
docker build -t sentiment-analysis:local .

# Run with mounted models (both transformer and classical available)
docker run -d -p 5000:5000 \
  -v $(pwd)/models:/app/models:ro \
  -e MODEL_TYPE=transformer \
  sentiment-analysis:local

# Or download models from S3 on startup
docker run -d -p 5000:5000 \
  -e TRANSFORMER_MODEL_S3_URI=s3://bucket/distilbert_sentiment.tar.gz \
  -e CLASSICAL_MODEL_S3_URI=s3://bucket/classical_models.tar.gz \
  -e AWS_ACCESS_KEY_ID=xxx \
  -e AWS_SECRET_ACCESS_KEY=xxx \
  -e MODEL_TYPE=transformer \
  ghcr.io/irfanfetahovic/sentiment-analysis:latest

# Switch models by changing MODEL_TYPE environment variable
# transformer: Uses DistilBERT model
# classical: Uses Logistic Regression + TF-IDF
```

See [deployment.md](docs/deployment.md) for production deployment with S3, cloud platforms, and more.

## Pipeline Stages
1. Load & sample data
2. Preprocess text (mode‑dependent)
3. Feature encoding (TF‑IDF or tokenizer)
4. Train model
5. Evaluate & persist artifacts
6. Serve via API / batch inference

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| NLTK resource error | Missing corpora | Run download command above |
| GPU not used | No CUDA device | Automatically falls back to CPU |
| Env override ignored | Variable not exported | In PowerShell use `$env:VAR = "value"` before command |
| Poor transformer accuracy | Insufficient epochs/sample | Increase `num_epochs` / remove `sample_frac` |
| Classical model slow | Very high `max_features` | Reduce `max_features` in config |

## 📚 Documentation

For complete documentation, see the [Documentation Index](docs/README.md).

**Key Guides:**
- **[Configuration Guide](docs/CONFIGURATION_GUIDE.md)** - Complete configuration reference (setup.py, config.yaml, .env, Docker, etc.)
- **[Testing Guide](docs/TESTING_GUIDE.md)** - Testing infrastructure, CI/CD, and best practices
- **[Preprocessing Guide](docs/PREPROCESSING_GUIDE.md)** - Classical vs Transformer preprocessing strategies
- **[API Documentation](docs/API.md)** - REST API endpoints and usage examples
- **[Model Comparison](docs/model_comparison.md)** - Performance benchmarks and metrics
- **[Deployment Guide](docs/deployment.md)** - Docker, AWS, Azure, GCP deployment instructions

## 📝 Notebooks

- `sentiment_analysis.ipynb`: Transformer-based approach (DistilBERT)
- `sentiment_analysis_classicalNLP.ipynb`: Classical NLP methods

## CI/CD
GitHub Actions workflow:
- Matrix test (Python 3.10–3.12)
- Lint, format check, mypy (non‑blocking)
- Pytest + coverage upload (Codecov)
- GHCR image build & push on `main` branch pushes

## Contributing
1. Fork
2. Branch: `feature/<name>`
3. Commit (small, focused changes)
4. Open PR with context & before/after metrics if relevant

## License
MIT (see `LICENSE`).

## Acknowledgments
Kaggle Amazon Fine Food Reviews dataset, Hugging Face Transformers, scikit‑learn, NLTK.

---
Need a section added or more detail? Open an issue or PR.
