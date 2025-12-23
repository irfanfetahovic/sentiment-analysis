"""
Flask API for Sentiment Analysis

This module provides a REST API for sentiment analysis using trained models.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import logging
import time
from datetime import datetime
from dotenv import load_dotenv

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

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration (env vars override config.yaml)
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

# Initialize predictor
try:
    predictor = SentimentPredictor(model_path=MODEL_PATH, model_type=MODEL_TYPE, device=DEVICE)
    logger.info(f"Loaded {MODEL_TYPE} model from {MODEL_PATH} on {DEVICE}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    predictor = None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'model_type': MODEL_TYPE,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sentiment for a single text.
    
    Request body:
        {
            "text": "Your text here"
        }
    
    Returns:
        {
            "text": "Your text here",
            "label": "POSITIVE" or "NEGATIVE",
            "score": 0.95,
            "processing_time_ms": 45
        }
    """
    start_time = time.time()
    
    try:
        # Validate request
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        text = request.json.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        if len(text) > API_MAX_TEXT_LENGTH:
            return jsonify({'error': f'Text exceeds maximum length of {API_MAX_TEXT_LENGTH} characters'}), 400
        
        # Check if model is loaded
        if predictor is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Make prediction
        result = predictor.predict_with_labels(text)
        
        # Add processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        result['processing_time_ms'] = round(processing_time, 2)
        
        logger.info(f"Prediction: {result['label']} (score: {result['score']:.4f})")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict sentiment for multiple texts.
    
    Request body:
        {
            "texts": ["Text 1", "Text 2", "Text 3"]
        }
    
    Returns:
        {
            "predictions": [
                {"text": "Text 1", "label": "POSITIVE", "score": 0.95},
                {"text": "Text 2", "label": "NEGATIVE", "score": 0.88},
                ...
            ],
            "count": 3,
            "processing_time_ms": 120
        }
    """
    start_time = time.time()
    
    try:
        # Validate request
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        texts = request.json.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'Texts array is required'}), 400
        
        if not isinstance(texts, list):
            return jsonify({'error': 'Texts must be an array'}), 400
        
        if len(texts) > MAX_BATCH_SIZE:
            return jsonify({'error': f'Maximum {MAX_BATCH_SIZE} texts per batch'}), 400
        
        # Check if model is loaded
        if predictor is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Make predictions
        results = predictor.predict_with_labels(texts)
        
        # Add processing time
        processing_time = (time.time() - start_time) * 1000
        
        response = {
            'predictions': results,
            'count': len(results),
            'processing_time_ms': round(processing_time, 2)
        }
        
        logger.info(f"Batch prediction: {len(results)} texts processed")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    try:
        if predictor is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get base info
        info = {
            'model_type': MODEL_TYPE,
            'model_path': MODEL_PATH,
            'max_text_length': API_MAX_TEXT_LENGTH,
            'labels': ['NEGATIVE', 'POSITIVE'],
            'version': '1.0.0'
        }
        
        # Add model config if available
        model_info_data = predictor.get_model_info()
        if 'config' in model_info_data:
            info['config'] = model_info_data['config']
        
        return jsonify(info)
    
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


def main():
    """Main entry point for running the API."""
    logger.info(f"Starting sentiment analysis API on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)


if __name__ == '__main__':
    main()
