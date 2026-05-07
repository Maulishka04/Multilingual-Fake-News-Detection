# API Guide

Base URL: `http://localhost:8000`

---

## Endpoints

### `GET /health`

Returns the health status of the backend and whether each model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "svm_loaded": true,
  "mbert_loaded": true
}
```

---

### `POST /predict`

Predict whether news text is fake using the specified model.

**Query parameters:**
| Parameter | Type | Default | Values |
|-----------|------|---------|--------|
| `model` | string | `svm` | `svm`, `mbert` |

**Request body:**
```json
{
  "text": "Scientists discover a new planet in the solar system.",
  "language": "en"
}
```

**Response (SVM):**
```json
{
  "prediction": 0,
  "confidence": 0.87,
  "model": "svm"
}
```

**Response (mBERT):**
```json
{
  "prediction": 0,
  "confidence": 0.94,
  "model": "mbert",
  "inference_time_ms": 487.3
}
```

`prediction`: `0` = Real News, `1` = Fake News

---

### `POST /predict-with-lime`

SVM prediction with LIME word-level explanation (SVM only).

**Request body:** same as `/predict`

**Response:**
```json
{
  "prediction": 1,
  "confidence": 0.78,
  "model": "svm",
  "explanation": {
    "positive_words": ["fake", "claim"],
    "negative_words": ["research", "study"],
    "word_scores": {
      "fake": 0.32,
      "claim": 0.21,
      "research": -0.18
    },
    "explanation_text": "Prediction is influenced towards Fake by fake, claim and towards Real by research."
  }
}
```

---

### `POST /compare`

Run the same text through both SVM and mBERT and return a side-by-side comparison.

**Request body:** same as `/predict`

**Response:**
```json
{
  "text_preview": "scientists discover a new planet...",
  "results": {
    "svm": {
      "prediction": 0,
      "confidence": 0.87,
      "inference_time_ms": 12.4
    },
    "mbert": {
      "prediction": 0,
      "confidence": 0.94,
      "inference_time_ms": 487.3
    }
  },
  "errors": {},
  "agreement": true
}
```

`agreement`: `true` if both models agree, `false` if they disagree, `null` if only one model is available.

---

### `GET /available-models`

List all models with their loading status and metadata.

**Response:**
```json
{
  "models": [
    {
      "name": "svm",
      "loaded": true,
      "description": "Calibrated Linear SVC with TF-IDF features",
      "accuracy": 0.85,
      "f1_score": 0.81,
      "inference_time_ms": 100,
      "best_for": "Speed + Explainability",
      "supports_lime": true
    },
    {
      "name": "mbert",
      "loaded": true,
      "description": "Multilingual BERT fine-tuned on 81,963 samples",
      "accuracy": 0.9115,
      "f1_score": 0.879,
      "inference_time_ms": 500,
      "best_for": "Accuracy + Context",
      "supports_lime": false
    }
  ]
}
```

---

### `GET /model-info/{model_name}`

Detailed information about a specific model.

**Path parameters:**
| Parameter | Values |
|-----------|--------|
| `model_name` | `svm`, `mbert` |

**Example:** `GET /model-info/mbert`

**Response:**
```json
{
  "name": "mbert",
  "loaded": true,
  "description": "Multilingual BERT fine-tuned on 81,963 samples",
  "accuracy": 0.9115,
  "f1_score": 0.879,
  "inference_time_ms": 500,
  "best_for": "Accuracy + Context",
  "supports_lime": false,
  "training_samples": 65570,
  "test_samples": 16393,
  "using_local_weights": true
}
```

---

## Error Codes

| Code | Meaning |
|------|---------|
| `400` | Bad request (empty text, text too short, unknown model) |
| `404` | Model not found |
| `503` | Model not loaded (artifacts missing) |
| `500` | Internal server error |

---

## Example: cURL

```bash
# SVM prediction
curl -X POST "http://localhost:8000/predict?model=svm" \
  -H "Content-Type: application/json" \
  -d '{"text": "Government announces new policy on renewable energy", "language": "en"}'

# mBERT prediction
curl -X POST "http://localhost:8000/predict?model=mbert" \
  -H "Content-Type: application/json" \
  -d '{"text": "Government announces new policy on renewable energy", "language": "en"}'

# Compare both
curl -X POST "http://localhost:8000/compare" \
  -H "Content-Type: application/json" \
  -d '{"text": "Government announces new policy on renewable energy", "language": "en"}'

# List models
curl "http://localhost:8000/available-models"
```
