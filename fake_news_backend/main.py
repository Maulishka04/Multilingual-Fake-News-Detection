import pickle
import time
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    APP_NAME,
    MBERT_DIR,
    MBERT_MAX_LENGTH,
    MBERT_MODEL_NAME,
    MODEL_METADATA,
    SVM_MODEL_PATH,
    SVM_MODEL_PATH_LEGACY,
    SVM_VECTORIZER_PATH,
    SVM_VECTORIZER_PATH_LEGACY,
)
from explainers import LIMEExplainer
from models.mbert_inference import MBertInference

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

SVM_MODEL = None
SVM_VECTORIZER = None
LIME_EXPLAINER = None
MBERT: Optional[MBertInference] = None


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    text: str
    language: Literal["en", "hi"]


class CompareRequest(BaseModel):
    text: str
    language: Literal["en", "hi"]


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def _resolve_svm_paths():
    """Return (model_path, vectorizer_path) preferring the structured svm/ subfolder."""
    if SVM_MODEL_PATH.exists() and SVM_VECTORIZER_PATH.exists():
        return SVM_MODEL_PATH, SVM_VECTORIZER_PATH
    return SVM_MODEL_PATH_LEGACY, SVM_VECTORIZER_PATH_LEGACY


@app.on_event("startup")
def load_artifacts() -> None:
    global SVM_MODEL, SVM_VECTORIZER, LIME_EXPLAINER, MBERT

    # --- SVM ---
    model_path, vectorizer_path = _resolve_svm_paths()
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"SVM model file not found: {model_path}")
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

        with model_path.open("rb") as f:
            SVM_MODEL = pickle.load(f)
        with vectorizer_path.open("rb") as f:
            SVM_VECTORIZER = pickle.load(f)

        LIME_EXPLAINER = LIMEExplainer(SVM_MODEL, SVM_VECTORIZER)
        print("SVM model, vectorizer, and LIME explainer loaded successfully.")
    except Exception as exc:
        SVM_MODEL = None
        SVM_VECTORIZER = None
        LIME_EXPLAINER = None
        print(f"Warning: SVM artifacts could not be loaded: {exc}")

    # --- mBERT ---
    try:
        mbert_instance = MBertInference(
            model_dir=MBERT_DIR,
            fallback_model_name=MBERT_MODEL_NAME,
            max_length=MBERT_MAX_LENGTH,
        )
        mbert_instance.load()
        MBERT = mbert_instance
        print("mBERT model loaded successfully.")
    except Exception as exc:
        MBERT = None
        print(f"Warning: mBERT model could not be loaded: {exc}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _run_svm_predict(cleaned_text: str):
    """Run SVM inference and return (predicted_label, confidence)."""
    if SVM_MODEL is None or SVM_VECTORIZER is None:
        raise HTTPException(status_code=503, detail="SVM model artifacts are not loaded.")
    features = SVM_VECTORIZER.transform([cleaned_text])
    predicted_label = int(SVM_MODEL.predict(features)[0])
    confidence = 0.0
    if hasattr(SVM_MODEL, "predict_proba"):
        probabilities = SVM_MODEL.predict_proba(features)[0]
        confidence = float(max(probabilities))
    return predicted_label, confidence


def _run_mbert_predict(cleaned_text: str):
    """Run mBERT inference and return the result dict."""
    if MBERT is None:
        raise HTTPException(status_code=503, detail="mBERT model is not loaded.")
    try:
        return MBERT.predict(cleaned_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    return {"message": "Fake News Backend is running"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "svm_loaded": SVM_MODEL is not None,
        "mbert_loaded": MBERT is not None and MBERT.is_loaded(),
    }


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------


@app.post("/predict")
def predict(payload: PredictRequest, model: str = "svm"):
    """Predict whether the news text is fake.

    Query parameters:
        model: ``"svm"`` (default) or ``"mbert"``
    """
    model = model.lower()
    if model not in {"svm", "mbert"}:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Use 'svm' or 'mbert'.")

    cleaned_text = LIMEExplainer.clean_text(payload.text)
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Text is empty after cleaning.")
    if len(cleaned_text) < 3:
        raise HTTPException(status_code=400, detail="Text is too short for prediction.")

    try:
        if model == "mbert":
            result = _run_mbert_predict(cleaned_text)
            return {
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "model": "mbert",
                "inference_time_ms": result.get("inference_time_ms"),
            }

        # SVM (default)
        predicted_label, confidence = _run_svm_predict(cleaned_text)
        return {"prediction": predicted_label, "confidence": confidence, "model": "svm"}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")


@app.post("/predict-with-lime")
def predict_with_lime(payload: PredictRequest):
    """SVM prediction with LIME explanation (SVM only; LIME not supported for mBERT)."""
    try:
        if SVM_MODEL is None or SVM_VECTORIZER is None:
            raise HTTPException(status_code=503, detail="SVM model artifacts are not loaded.")
        if LIME_EXPLAINER is None:
            raise HTTPException(status_code=503, detail="LIME explainer is not initialized.")

        cleaned_text = LIMEExplainer.clean_text(payload.text)
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Text is empty after cleaning.")
        if len(cleaned_text) < 3:
            raise HTTPException(status_code=400, detail="Text is too short for prediction and explanation.")

        features = SVM_VECTORIZER.transform([cleaned_text])
        predicted_label = int(SVM_MODEL.predict(features)[0])

        if not hasattr(SVM_MODEL, "predict_proba"):
            raise HTTPException(status_code=500, detail="Loaded model does not support probability output.")

        probabilities = SVM_MODEL.predict_proba(features)[0]
        confidence = float(max(probabilities))

        try:
            explanation = LIME_EXPLAINER.explain(cleaned_text, num_features=5)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"LIME input validation failed: {exc}")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LIME explanation failed: {exc}")

        return {
            "prediction": predicted_label,
            "confidence": confidence,
            "model": "svm",
            "explanation": explanation,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction with LIME failed: {exc}")


# ---------------------------------------------------------------------------
# Comparison endpoint
# ---------------------------------------------------------------------------


@app.post("/compare")
def compare_models(payload: CompareRequest):
    """Run the same text through both SVM and mBERT and return a side-by-side comparison."""
    cleaned_text = LIMEExplainer.clean_text(payload.text)
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Text is empty after cleaning.")
    if len(cleaned_text) < 3:
        raise HTTPException(status_code=400, detail="Text is too short for comparison.")

    results = {}
    errors = {}

    # SVM
    try:
        if SVM_MODEL is None or SVM_VECTORIZER is None:
            errors["svm"] = "SVM model not loaded"
        else:
            t0 = time.perf_counter()
            predicted_label, confidence = _run_svm_predict(cleaned_text)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            results["svm"] = {
                "prediction": predicted_label,
                "confidence": confidence,
                "inference_time_ms": elapsed_ms,
            }
    except Exception as exc:
        errors["svm"] = str(exc)

    # mBERT
    try:
        if MBERT is None or not MBERT.is_loaded():
            errors["mbert"] = "mBERT model not loaded"
        else:
            mbert_result = MBERT.predict(cleaned_text)
            results["mbert"] = {
                "prediction": mbert_result["prediction"],
                "confidence": mbert_result["confidence"],
                "inference_time_ms": mbert_result.get("inference_time_ms"),
            }
    except Exception as exc:
        errors["mbert"] = str(exc)

    if not results:
        raise HTTPException(status_code=503, detail="No models are available for comparison.")

    agreement = None
    if "svm" in results and "mbert" in results:
        agreement = results["svm"]["prediction"] == results["mbert"]["prediction"]

    return {
        "text_preview": cleaned_text[:200],
        "results": results,
        "errors": errors,
        "agreement": agreement,
    }


# ---------------------------------------------------------------------------
# Model info endpoints
# ---------------------------------------------------------------------------


@app.get("/available-models")
def available_models():
    """Return a list of available models with their loading status and metadata."""
    models = []
    for model_name, meta in MODEL_METADATA.items():
        loaded = False
        if model_name == "svm":
            loaded = SVM_MODEL is not None
        elif model_name == "mbert":
            loaded = MBERT is not None and MBERT.is_loaded()
        models.append({
            **meta,
            "name": model_name,
            "loaded": loaded,
        })
    return {"models": models}


@app.get("/model-info/{model_name}")
def model_info(model_name: str):
    """Return detailed information about a specific model."""
    model_name = model_name.lower()
    if model_name not in MODEL_METADATA:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(MODEL_METADATA.keys())}",
        )

    meta = MODEL_METADATA[model_name]
    loaded = False
    extra: dict = {}

    if model_name == "svm":
        loaded = SVM_MODEL is not None
    elif model_name == "mbert":
        loaded = MBERT is not None and MBERT.is_loaded()
        if MBERT is not None:
            extra["using_local_weights"] = MBERT.is_local_model()

    return {
        **meta,
        "name": model_name,
        "loaded": loaded,
        **extra,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
