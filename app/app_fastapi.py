"""
FastAPI Application for Sentiment Analysis

This module provides a REST API for sentiment analysis using trained models.
FastAPI alternative to the Flask app with auto-generated docs and validation.
"""

import os
import sys
import logging
import time
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add src directory to path (not needed after pip install -e .)
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# src_path = os.path.join(project_root, 'src')
# sys.path.insert(0, src_path)

from sentiment_analysis.inference import SentimentPredictor
from sentiment_analysis.utils import setup_logging, load_config
from sentiment_analysis.settings import settings
from sentiment_analysis.constants import (
    MAX_TEXT_LENGTH,
    MAX_BATCH_SIZE,
    DEFAULT_API_HOST,
    DEFAULT_API_PORT,
    MODEL_TYPE_TRANSFORMER,
    DEVICE_CPU
)

# Load environment variables
load_dotenv()

# Load configuration (settings.py will auto-detect config.yaml location)
config = load_config()
api_config = config.get('api', {})
inference_config = config.get('inference', {})
logging_config = config.get('logging', {})

# Configuration from environment or config file
MODEL_PATH = os.getenv('MODEL_PATH', str(settings.model_dir / 'distilbert_sentiment'))
MODEL_TYPE = os.getenv('MODEL_TYPE', inference_config.get('model_type', MODEL_TYPE_TRANSFORMER))
API_MAX_TEXT_LENGTH = int(os.getenv('MAX_TEXT_LENGTH', api_config.get('max_text_length', MAX_TEXT_LENGTH)))
LOG_FILE = os.getenv('LOG_FILE', logging_config.get('log_file', 'logs/app.log'))
HOST = os.getenv('HOST', api_config.get('host', DEFAULT_API_HOST))
PORT = int(os.getenv('PORT', api_config.get('port', DEFAULT_API_PORT)))
DEBUG = os.getenv('DEBUG', str(api_config.get('debug', False))).lower() == 'true'
DEVICE = os.getenv('DEVICE', inference_config.get('device', DEVICE_CPU))

# Setup logging
setup_logging(log_file=LOG_FILE)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class PredictRequest(BaseModel):
    """Request model for single prediction"""
    text: str = Field(..., description="Text to analyze", min_length=1, max_length=API_MAX_TEXT_LENGTH)
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This product is amazing! Highly recommend it."
            }
        }


class BatchPredictRequest(BaseModel):
    """Request model for batch predictions"""
    texts: List[str] = Field(..., description="List of texts to analyze", min_items=1, max_items=MAX_BATCH_SIZE)
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        for text in v:
            if not text or not text.strip():
                raise ValueError('Each text must be non-empty')
            if len(text) > API_MAX_TEXT_LENGTH:
                raise ValueError(f'Text length cannot exceed {API_MAX_TEXT_LENGTH} characters')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Great product!",
                    "Terrible quality",
                    "It's okay, nothing special"
                ]
            }
        }


class PredictResponse(BaseModel):
    """Response model for single prediction"""
    text: str
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    processing_time: float


class BatchPredictResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictResponse]
    total_processed: int
    total_time: float


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    model_type: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_type: str
    model_path: str
    device: str
    max_text_length: int
    max_batch_size: int


# ============================================================================
# Initialize FastAPI App
# ============================================================================

app = FastAPI(
    title="Sentiment Analysis API",
    description="Production-ready sentiment analysis using transformer models (DistilBERT) and classical ML",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc",  # ReDoc at /redoc
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # CORS in production should be restricted to specific origins
# origins = [
#     "http://localhost:3000", # common dev port of frontend apps by many frameworks
#     "https://myfrontend.com", # actual frontend domain (production)
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )





# Initialize predictor
predictor = None

# You load model only once when the app starts (not per request)
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global predictor
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        predictor = SentimentPredictor(
            model_path=MODEL_PATH,
            model_type=MODEL_TYPE,
            device=DEVICE
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


# ============================================================================
# API Endpoints
# ============================================================================

# Handling get requests to the root endpoint
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Check if the API is running and model is loaded
    
    Returns health status, model information, and timestamp
    """
    return HealthResponse(
        status="healthy" if predictor else "unhealthy",
        model_loaded=predictor is not None,
        model_type=MODEL_TYPE,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Analyze sentiment of a single text
    
    - **text**: The text to analyze (required)
    
    Returns the predicted sentiment label, confidence score, and processing time
    """
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        start_time = time.time()
        result = predictor.predict_with_labels(request.text)
        processing_time = time.time() - start_time
        
        logger.info(f"Prediction: {result['label']} (score: {result['score']:.4f})")
        
        return PredictResponse(
            text=result['text'],
            label=result['label'],
            score=result['score'],
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """
    Analyze sentiment of multiple texts in batch
    
    - **texts**: List of texts to analyze (required, max {MAX_BATCH_SIZE} items)
    
    Returns predictions for all texts with total processing time
    """
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        start_time = time.time()
        # predict_with_labels handles both single texts and lists
        results = predictor.predict_with_labels(request.texts)
        total_time = time.time() - start_time
        
        # Ensure results is a list (single text returns dict, multiple returns list)
        if isinstance(results, dict):
            results = [results]
        
        predictions = [
            PredictResponse(
                text=r['text'],
                label=r['label'],
                score=r['score'],
                processing_time=0  # Individual timing not tracked in batch
            )
            for r in results
        ]
        
        logger.info(f"Batch prediction: {len(predictions)} texts processed in {total_time:.4f}s")
        
        return BatchPredictResponse(
            predictions=predictions,
            total_processed=len(predictions),
            total_time=total_time
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """
    Get information about the loaded model
    
    Returns model type, path, device, and API limits
    """
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return ModelInfoResponse(
        model_type=MODEL_TYPE,
        model_path=MODEL_PATH,
        device=DEVICE,
        max_text_length=API_MAX_TEXT_LENGTH,
        max_batch_size=MAX_BATCH_SIZE
    )


# ============================================================================
# Run Server (use for production with Gunicorn)
# ============================================================================

# 
def main():
    """Run the FastAPI server
        Use for local testing or development only
        For production, use Gunicorn with Uvicorn workers:
            gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.app_fastapi:app
    """
    import uvicorn
    
    logger.info(f"Starting FastAPI server on {HOST}:{PORT}")
    logger.info(f"Swagger UI available at: http://{HOST}:{PORT}/docs")
    logger.info(f"ReDoc available at: http://{HOST}:{PORT}/redoc")
    
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()
