# Documentation Index

Complete documentation for the sentiment analysis project.

## 📚 Available Guides

### [Configuration Guide](CONFIGURATION_GUIDE.md)
Comprehensive reference for all configuration files:
- Package configuration (setup.py, setup.cfg)
- Runtime configuration (config.yaml)
- Environment configuration (.env)
- Code quality configuration (flake8, black, mypy)
- Deployment configuration (Docker, CI/CD)
- Configuration hierarchy and best practices

### [Testing Guide](TESTING_GUIDE.md)
Complete testing infrastructure documentation:
- Test configuration and setup
- Running tests locally and in CI/CD
- Pre-commit hooks
- Test structure and coverage
- Code quality standards
- Best practices and troubleshooting

### [Preprocessing Guide](PREPROCESSING_GUIDE.md)
Comprehensive guide to text preprocessing strategies:
- **Classical Mode**: Extensive preprocessing for TF-IDF, Word2Vec, SVM, Logistic Regression
- **Transformer Mode**: Minimal preprocessing for DistilBERT and BERT models
- Detailed step-by-step examples
- Performance impact analysis

### [API Documentation](API.md)
REST API reference for Flask and FastAPI applications:
- Health check endpoints
- Single text prediction
- Batch prediction
- Model information endpoints
- Request/response examples
- Error handling

### [Model Comparison](model_comparison.md)
Performance benchmarks and model selection guidance:
- Classical ML models (Logistic Regression, SVM)
- Transformer models (DistilBERT)
- Accuracy, F1-score, precision, recall metrics
- Inference speed comparisons
- Pros and cons for each approach

### [Deployment Guide](deployment.md)
Production deployment instructions:
- Local deployment
- Docker deployment
- Docker Compose orchestration
- Cloud deployment (AWS, Azure, GCP)
- Kubernetes configuration
- Production considerations and best practices

## 🚀 Quick Links

- [Main README](../README.md) - Project overview and quick start

## 📖 Documentation Structure

```
docs/
├── README.md                    # This file - documentation index
├── CONFIGURATION_GUIDE.md       # Complete configuration reference
├── TESTING_GUIDE.md             # Testing infrastructure guide
├── GITHUB_SETUP.md              # Git initialization guide
├── PREPROCESSING_GUIDE.md       # Text preprocessing strategies
├── API.md                       # REST API reference
├── model_comparison.md          # Performance benchmarks
└── deployment.md                # Production deployment
```

---

**Last Updated**: December 18, 2025
