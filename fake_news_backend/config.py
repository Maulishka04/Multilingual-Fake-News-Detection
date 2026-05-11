"""Backend configuration for FastAPI."""

import os
from pathlib import Path

# Server config
HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
PORT = int(os.getenv("BACKEND_PORT", "8000"))
APP_NAME = "Fake News Backend"

# Model paths
MODELS_DIR = Path(__file__).parent / "models"

# SVM model paths
SVM_MODEL_PATH = MODELS_DIR / "linear_svc_calibrated_tfidf.pkl"
SVM_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

# mBERT model path - Can be customized via environment variable
MBERT_MODEL_PATH = Path(os.getenv(
    "MBERT_MODEL_PATH",
    MODELS_DIR / "mbert_model"
))

# Model settings
MBERT_MAX_LENGTH = 128
MBERT_BATCH_SIZE = 32

# Device settings
USE_CUDA = os.getenv("USE_CUDA", "true").lower() == "true"

# LIME settings
LIME_NUM_FEATURES = 5
