FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/usr/local/share/nltk_data

# Install system dependencies (including AWS CLI for S3 access)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    unzip \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip \
    && rm -rf /var/lib/apt/lists/*


# Copy optimized requirements for smaller image (CPU-only PyTorch, inference deps only)
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements-docker.txt

# Download NLTK data
# RUN python -c "import nltk; \
#     nltk.download('stopwords'); \
#     nltk.download('wordnet'); \
#     nltk.download('omw-1.4'); \
#     nltk.download('punkt'); \
#     nltk.download('punkt_tab'); \
#     nltk.download('averaged_perceptron_tagger_eng')"

RUN python -m nltk.downloader -d $NLTK_DATA \
stopwords wordnet omw-1.4 punkt punkt_tab averaged_perceptron_tagger

# Copy application code
COPY src/ ./src/
COPY app/ ./app/
COPY config/ ./config/
COPY setup.py .
COPY README.md .
COPY requirements.txt .

# Install package (requirements.txt needed for setup.py to read)
RUN pip install -e .

# Create necessary directories
RUN mkdir -p logs models data

# Copy and setup entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Expose port
EXPOSE 5000

# Health check (increased start-period to allow time for model download)
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Set entrypoint to handle model downloads
ENTRYPOINT ["docker-entrypoint.sh"]

# Run the FastAPI application with uvicorn (ASGI server)
# Note: gunicorn is WSGI, FastAPI requires ASGI (uvicorn)
CMD ["uvicorn", "app.app_fastapi:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "2"]
# For Flask (WSGI), use: gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 app.app:app