# Configuration Guide

Comprehensive reference for all configuration files in the sentiment analysis project.

---

## 📋 Table of Contents

1. [Quick Reference](#quick-reference)
2. [Configuration Hierarchy](#configuration-hierarchy)
3. [Core Files](#core-files)
4. [Usage Examples](#usage-examples)
5. [Best Practices](#best-practices)

---

## Quick Reference

### Configuration Files Overview

| File | Purpose | Commit to Git | Priority |
|------|---------|---------------|----------|
| `setup.py` | Package definition & dependencies | ✅ Yes | - |
| `setup.cfg` | Tool configurations (flake8, pytest, mypy) | ✅ Yes | - |
| `requirements.txt` | Production dependencies | ✅ Yes | - |
| `config/config.yaml` | ML hyperparameters & defaults | ✅ Yes | 3 |
| `settings.py` | Path resolution & environment handling | ✅ Yes | 2 |
| `.env` | Local secrets & overrides | ❌ No | 2 |
| `.env.example` | Environment variable template | ✅ Yes | - |
| `.pre-commit-config.yaml` | Git hooks for code quality | ✅ Yes | - |
| `Dockerfile` | Container build instructions | ✅ Yes | - |
| `docker-compose.yml` | Multi-container orchestration | ✅ Yes | - |

---

## Configuration Hierarchy

Values are resolved in this order (highest to lowest priority):

```
CLI Arguments  →  Environment Variables  →  config.yaml  →  Code Defaults
    (1)                    (2)                  (3)              (4)
```

**Examples:**

```bash
# 1. CLI argument (highest priority)
sentiment-train-transformer --num-epochs 5

# 2. Environment variable
export SENTIMENT_DEVICE=cuda
sentiment-train-transformer  # Uses cuda

# 3. config.yaml value
# training.transformer.num_epochs: 3

# 4. Code default (fallback)
# DEFAULT_NUM_EPOCHS = 3
```

---

## Core Files

### Package Configuration

#### `setup.py`
Defines Python package structure, dependencies, and CLI entry points.

**Key features:**
- Reads dependencies from `requirements.txt`
- Creates console scripts (e.g., `sentiment-train-transformer`)
- Enables editable install: `pip install -e .`

**Modify when:**
- Adding dependencies or CLI commands
- Updating package version

#### `setup.cfg`
Configures code quality tools (flake8, pytest, mypy).

```ini
[flake8]
max-line-length = 100

[tool:pytest]
testpaths = tests
addopts = --verbose --cov=src

[mypy]
ignore_missing_imports = True
```

#### `requirements.txt`
Production dependencies: numpy, pandas, torch, transformers, Flask, FastAPI, etc.

---

### Runtime Configuration

#### `config/config.yaml`
Central ML configuration for hyperparameters and paths.

**Key sections:**
```yaml
data:
  path: "data/Reviews.csv"
  sample_frac: 0.1  # For quick experiments

preprocessing:
  classical:  # TF-IDF, SVM
    mode: "classical"
    use_negation: true
  transformer:  # DistilBERT
    mode: "transformer"

model:
  transformer:
    model_name: "distilbert-base-uncased"
    max_length: 128
  classical:
    model_type: "logistic"
    feature_type: "tfidf"
    max_features: 5000

training:
  random_state: 42
  test_size: 0.1
  transformer:
    learning_rate: 2.0e-5
    batch_size: 16
    num_epochs: 3

api:
  host: "0.0.0.0"
  port: 5000
```

**Use for:** Project-wide ML defaults (commit to git, no secrets)

#### `src/sentiment_analysis/settings.py`
Manages environment-based configuration with auto-detection.

**Features:**
- Auto-detects project root (checks for `setup.py`, `config.yaml`, `.git`)
- Reads `SENTIMENT_*` environment variables
- Provides global `settings` instance

**Environment variables:**
- `SENTIMENT_PROJECT_ROOT`: Override project root
- `SENTIMENT_DATA_DIR`: Custom data directory
- `SENTIMENT_MODEL_DIR`: Custom model directory
- `SENTIMENT_DEVICE`: cpu or cuda
- `SENTIMENT_API_HOST`, `SENTIMENT_API_PORT`: API configuration

#### `.env` (Local, not committed)
Local environment variables for development.

**Setup:**
```bash
cp .env.example .env
# Edit with your values
```

**Example:**
```dotenv
SENTIMENT_DEVICE=cuda
SENTIMENT_API_PORT=8000
```

**⚠️ Never commit** this file to git.

#### `.env.example`
Template documenting all available environment variables.

---

### Code Quality

#### `.pre-commit-config.yaml`
Git pre-commit hooks for automated code quality.

**Hooks:**
- Trailing whitespace, end-of-file fixes
- YAML/JSON validation
- **Black**: Code formatting (100 char line length)
- **isort**: Import sorting
- **flake8**: PEP 8 linting
- **mypy**: Type checking

**Setup:**
```bash
pip install pre-commit
pre-commit install
```

---

### Deployment

#### `Dockerfile`
Container build instructions.

**Key steps:**
1. Base: `python:3.12-slim`
2. Install dependencies from `requirements.txt`
3. Download NLTK data
4. Copy code & install package
5. Run gunicorn server

#### `docker-compose.yml`
Multi-container orchestration.

**Main service:**
```yaml
sentiment-api:
  build: .
  ports: ["5000:5000"]
  volumes:
    - ./models:/app/models:ro
  environment:
    - MODEL_PATH=/app/models/distilbert_sentiment
```

---

## Usage Examples

### Development Workflow

**Initial setup:**
```bash
git clone <repo-url>
cd sentiment-analysis
cp .env.example .env  # Edit as needed
pip install -e .[dev]
pre-commit install
```

**Running scripts:**
```bash
# Uses config.yaml defaults
sentiment-train-transformer

# Override with CLI args
sentiment-train-transformer --num-epochs 5

# Override with env vars
export SENTIMENT_DEVICE=cuda
sentiment-train-transformer
```

**Testing:**
```bash
pytest                      # Run all tests
pytest --cov=src           # With coverage
black . && isort .         # Format code
flake8 && mypy src/        # Lint & type check
```

### Production Deployment

**Docker:**
```bash
docker build -t sentiment-api .
docker-compose up -d
```

**Configuration:**
- Use `config.yaml` for stable ML settings
- Use environment variables for deployment-specific settings
- Use secrets management for sensitive data (never `.env`)

### Environment-Specific Settings

**Development:**
```yaml
# config.yaml
data:
  sample_frac: 0.1  # 10% of data for speed
training:
  transformer:
    num_epochs: 3
```

**Production:**
```yaml
# config.yaml
data:
  sample_frac: 1.0  # Full dataset
training:
  transformer:
    num_epochs: 10
```

```bash
# Set via environment
export SENTIMENT_DEVICE=cuda
export SENTIMENT_MODEL_DIR=/mnt/models
```

---

## Best Practices

### ✅ Do

- Commit: `config.yaml`, `.env.example`, all config files **except** `.env`
- Document configuration changes in commit messages
- Use hierarchy: CLI args > env vars > config.yaml > defaults
- Run tests after configuration changes

### ❌ Don't

- Never commit `.env` (contains secrets)
- Don't hardcode paths or secrets
- Don't put secrets in `config.yaml`
- Don't skip pre-commit hooks

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Module not found" | Run `pip install -e .` |
| "Path not found" | Check `settings.py` auto-detection, verify project root markers |
| Tests failing | Check `setup.cfg` pytest configuration |
| Pre-commit failing | Run `black .` and `isort .` |
| Docker build fails | Verify `requirements.txt` and Dockerfile dependencies |

---

## Quick Reference Summary

**Configuration priority:**
1. CLI arguments (highest)
2. Environment variables
3. `config.yaml`
4. Code defaults (lowest)

**Files to commit:**
- ✅ `setup.py`, `setup.cfg`, `requirements.txt`
- ✅ `config.yaml`, `.env.example`
- ✅ `.pre-commit-config.yaml`
- ✅ `Dockerfile`, `docker-compose.yml`
- ❌ `.env` (local only)

**Common commands:**
```bash
pip install -e .[dev]               # Install with dev dependencies
pre-commit install                  # Setup git hooks
pytest --cov=src                   # Test with coverage
docker-compose up -d               # Deploy with Docker
```

---

**Last Updated**: December 18, 2025
