"""Basic configuration for the FastAPI backend."""

from pathlib import Path

HOST = "127.0.0.1"
PORT = 8000
APP_NAME = "Fake News Backend"

# Base directory for all model artifacts
MODELS_BASE_DIR = Path(__file__).parent / "models"

# SVM model paths
SVM_DIR = MODELS_BASE_DIR / "svm"
SVM_MODEL_PATH = SVM_DIR / "linear_svc_calibrated_tfidf.pkl"
SVM_VECTORIZER_PATH = SVM_DIR / "tfidf_vectorizer.pkl"

# Legacy paths (models stored flat in models/ directory)
SVM_MODEL_PATH_LEGACY = MODELS_BASE_DIR / "linear_svc_calibrated_tfidf.pkl"
SVM_VECTORIZER_PATH_LEGACY = MODELS_BASE_DIR / "tfidf_vectorizer.pkl"

# mBERT model paths
MBERT_DIR = MODELS_BASE_DIR / "mbert"
MBERT_MODEL_NAME = "bert-base-multilingual-cased"  # Fallback HuggingFace model name
MBERT_MAX_LENGTH = 128

# Model metadata
MODEL_METADATA = {
    "svm": {
        "name": "SVM (TF-IDF)",
        "description": "Calibrated Linear SVC with TF-IDF features",
        "accuracy": 0.85,
        "f1_score": 0.81,
        "inference_time_ms": 100,
        "best_for": "Speed + Explainability",
        "supports_lime": True,
    },
    "mbert": {
        "name": "mBERT",
        "description": "Multilingual BERT fine-tuned on 81,963 samples",
        "accuracy": 0.9115,
        "f1_score": 0.8790,
        "inference_time_ms": 500,
        "best_for": "Accuracy + Context",
        "supports_lime": False,
        "training_samples": 65570,
        "test_samples": 16393,
    },
}
