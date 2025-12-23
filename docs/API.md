# API Documentation

## Base URL
```
http://localhost:5000
```

## Endpoints

### 1. Health Check

Check if the API is running.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "transformer",
  "timestamp": "2025-11-15T10:30:00"
}
```

**Note:** The `model_type` can be either `"transformer"` (DistilBERT) or `"classical"` (Logistic Regression, SVM, etc.)

**Example:**
```bash
curl http://localhost:5000/health
```

---

### 2. Predict Sentiment (Single Text)

Analyze sentiment of a single text.

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "text": "This product is amazing! Highly recommend."
}
```

**Response:**
```json
{
  "text": "This product is amazing! Highly recommend.",
  "label": "POSITIVE",
  "score": 0.9876,
  "processing_time_ms": 45
}
```

**Error Response:**
```json
{
  "error": "Text is required",
  "status": 400
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product!"}'
```

---

### 3. Batch Prediction

Analyze sentiment of multiple texts at once.

**Endpoint:** `POST /predict/batch`

**Request Body:**
```json
{
  "texts": [
    "This product is great!",
    "Terrible quality, very disappointed.",
    "Average product, nothing special."
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "text": "This product is great!",
      "label": "POSITIVE",
      "score": 0.9823
    },
    {
      "text": "Terrible quality, very disappointed.",
      "label": "NEGATIVE",
      "score": 0.9654
    },
    {
      "text": "Average product, nothing special.",
      "label": "POSITIVE",
      "score": 0.6234
    }
  ],
  "count": 3,
  "processing_time_ms": 120
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Good!", "Bad!", "Okay"]}'
```

---

### 4. Get Model Info

Get information about the loaded model.

**Endpoint:** `GET /model/info`

**Response:**
```json
{
  "model_name": "distilbert-base-uncased",
  "model_type": "transformer",
  "num_labels": 2,
  "labels": ["NEGATIVE", "POSITIVE"],
  "max_length": 128,
  "version": "1.0.0"
}
```

**Example:**
```bash
curl http://localhost:5000/model/info
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (missing or invalid input) |
| 500 | Internal Server Error |

## Rate Limits

- **Default**: 100 requests per minute per IP
- **Batch endpoint**: 20 requests per minute per IP

## Authentication

Currently no authentication required. For production, consider implementing:
- API Key authentication
- OAuth 2.0
- JWT tokens

## Examples

### Python

```python
import requests

url = "http://localhost:5000/predict"
payload = {"text": "This product exceeded my expectations!"}
response = requests.post(url, json=payload)
print(response.json())
```

### JavaScript

```javascript
fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'Amazing product!'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

### cURL

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Fantastic product, will buy again!"
  }'
```
