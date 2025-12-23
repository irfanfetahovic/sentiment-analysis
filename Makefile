.PHONY: help install setup test lint clean train-classical train-transformer train-all evaluate pipeline docker-build docker-run

# Default target
help:
	@echo "Sentiment Analysis Project - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install              Install Python dependencies"
	@echo "  make setup                Full project setup (install + download data)"
	@echo ""
	@echo "Development:"
	@echo "  make test                 Run all tests with pytest"
	@echo "  make test-verbose         Run tests with verbose output"
	@echo "  make lint                 Run code linters"
	@echo "  make clean                Remove temporary files and artifacts"
	@echo ""
	@echo "Training:"
	@echo "  make train-classical      Train classical ML model (Logistic + TF-IDF)"
	@echo "  make train-transformer    Train transformer model (DistilBERT)"
	@echo "  make train-all            Train both models"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate-classical   Evaluate classical model"
	@echo "  make evaluate-transformer Evaluate transformer model"
	@echo "  make evaluate-all         Compare all models"
	@echo ""
	@echo "Pipeline:"
	@echo "  make pipeline             Run full pipeline (train + evaluate)"
	@echo "  make pipeline-quick       Quick pipeline run (small sample)"
	@echo ""
	@echo "API:"
	@echo "  make run-api              Start Flask API server"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build         Build Docker image"
	@echo "  make docker-run           Run Docker container"
	@echo "  make docker-shell         Open shell in Docker container"


# Setup & Installation


install:
	pip install -r requirements.txt

setup: install
	@echo "Setting up project..."
	python -m pip install --upgrade pip
	pip install -e .
	@echo "Setup complete!"


# Development


test:
	pytest tests/ -v

test-verbose:
	pytest tests/ -vv -s

test-coverage:
	pytest tests/ --cov=src/sentiment_analysis --cov-report=html --cov-report=term

lint:
	@echo "Running flake8..."
	-flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503
	@echo "Running pylint..."
	-pylint src/sentiment_analysis --disable=C0111,R0913,R0914

format:
	black src/ tests/ scripts/ --line-length=100
	isort src/ tests/ scripts/

clean:
	@echo "Cleaning up temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true
	@echo "Cleanup complete!"


# Training


train-classical:
	python scripts/run_train_classical.py \
		--data-path data/Reviews.csv \
		--models-dir models \
		--max-features 5000

train-transformer:
	python scripts/run_train_transformer.py \
		--data-path data/Reviews.csv \
		--output-dir models/distilbert_sentiment \
		--num-epochs 3 \
		--batch-size 16 \
		--sample-frac 0.1

train-all:
	@echo "Training all models..."
	$(MAKE) train-classical
	$(MAKE) train-transformer
	@echo "All models trained!"

# Quick training with small sample for testing
train-quick:
	python scripts/run_train_classical.py \
		--data-path data/Reviews.csv \
		--sample-frac 0.01 \
		--max-features 1000


# Evaluation

evaluate-classical:
	python scripts/run_evaluate.py \
		--model-path models/classical_models/logistic_tfidf_model.pkl \
		--model-type classical \
		--data-path data/Reviews.csv \
		--output-file results/classical_evaluation.json

evaluate-transformer:
	python scripts/run_evaluate.py \
		--model-path models/distilbert_sentiment \
		--model-type transformer \
		--data-path data/Reviews.csv \
		--output-file results/transformer_evaluation.json

evaluate-all:
	python scripts/run_evaluate.py \
		--compare \
		--models-json models_config.json \
		--data-path data/Reviews.csv \
		--output-file results/models_comparison.json

# Pipeline

pipeline:
	python scripts/run_pipeline.py \
		--train-all \
		--compare \
		--data-path data/Reviews.csv \
		--output-dir results

pipeline-quick:
	python scripts/run_pipeline.py \
		--train-all \
		--compare \
		--sample-frac 0.01 \
		--num-epochs 1 \
		--output-dir results


# API

run-api:
	python app/app.py


# Docker

docker-build:
	docker build -t sentiment-analysis:latest .

docker-run:
	docker run -p 5000:5000 sentiment-analysis:latest

docker-shell:
	docker run -it --rm sentiment-analysis:latest /bin/bash


# Predictions

predict-classical:
	python scripts/run_predict.py \
		--model-path models/classical_models/logistic_tfidf_model.pkl \
		--model-type classical \
		--text "This is a great product!"

predict-transformer:
	python scripts/run_predict.py \
		--model-path models/distilbert_sentiment \
		--model-type transformer \
		--text "This is a great product!"

# Data

download-data:
	@echo "Please download Reviews.csv from Kaggle and place in data/ directory"
	@echo "URL: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews"


# Utilities

check-structure:
	@echo "Project structure:"
	tree -L 3 -I '__pycache__|*.pyc|*.egg-info'

list-models:
	@echo "Available models:"
	@ls -lh models/classical_models/ 2>/dev/null || echo "No classical models found"
	@ls -lh models/ | grep distilbert 2>/dev/null || echo "No transformer models found"
