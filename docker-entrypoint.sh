#!/bin/bash

# This file is the entrypoint script for the Docker container.
# It is called each time the container starts.
# It checks for the presence of the sentiment analysis model locally.
# If the model is not found, it attempts to download it from a specified S3 URI
# or a public URL (e.g., GitHub Releases).
# Finally, it starts the application server (e.g., Gunicorn).
# Environment variables are given either from docker run -e or docker run --env-file .env or docker-compose.yml

set -e

echo "Starting sentiment-analysis application..."

# Create models directory
mkdir -p /app/models

# Download transformer model (DistilBERT) if not present
if [ ! -d "/app/models/distilbert_sentiment" ] || [ -z "$(ls -A /app/models/distilbert_sentiment 2>/dev/null)" ]; then
    if [ -n "$TRANSFORMER_MODEL_S3_URI" ]; then
        echo "Transformer model not found. Downloading from S3: $TRANSFORMER_MODEL_S3_URI"
        aws s3 cp "$TRANSFORMER_MODEL_S3_URI" /tmp/transformer_model.tar.gz
        tar -xzf /tmp/transformer_model.tar.gz -C /app/models/
        rm /tmp/transformer_model.tar.gz
        echo "Transformer model downloaded and extracted"
    elif [ -n "$TRANSFORMER_MODEL_URL" ]; then
        echo "Transformer model not found. Downloading from URL: $TRANSFORMER_MODEL_URL"
        curl -L -o /tmp/transformer_model.tar.gz "$TRANSFORMER_MODEL_URL"
        tar -xzf /tmp/transformer_model.tar.gz -C /app/models/
        rm /tmp/transformer_model.tar.gz
        echo "Transformer model downloaded and extracted"
    else
        echo "WARNING: Transformer model not found at /app/models/distilbert_sentiment"
    fi
else
    echo "Transformer model found at /app/models/distilbert_sentiment"
fi

# Download classical model if not present
if [ ! -d "/app/models/classical_models" ] || [ -z "$(ls -A /app/models/classical_models 2>/dev/null)" ]; then
    if [ -n "$CLASSICAL_MODEL_S3_URI" ]; then
        echo "Classical model not found. Downloading from S3: $CLASSICAL_MODEL_S3_URI"
        aws s3 cp "$CLASSICAL_MODEL_S3_URI" /tmp/classical_model.tar.gz
        tar -xzf /tmp/classical_model.tar.gz -C /app/models/
        rm /tmp/classical_model.tar.gz
        echo "Classical model downloaded and extracted"
    elif [ -n "$CLASSICAL_MODEL_URL" ]; then
        echo "Classical model not found. Downloading from URL: $CLASSICAL_MODEL_URL"
        curl -L -o /tmp/classical_model.tar.gz "$CLASSICAL_MODEL_URL"
        tar -xzf /tmp/classical_model.tar.gz -C /app/models/
        rm /tmp/classical_model.tar.gz
        echo "Classical model downloaded and extracted"
    else
        echo "WARNING: Classical model not found at /app/models/classical_models"
    fi
else
    echo "Classical model found at /app/models/classical_models"
fi

echo "Model setup complete. Use MODEL_TYPE env variable to switch between 'transformer' and 'classical'"

# Execute the main command (gunicorn or whatever is passed as CMD)
echo "Starting application server..."
exec "$@"
