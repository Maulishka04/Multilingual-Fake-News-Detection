import pickle
import sys
from typing import Literal, Dict, Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from explainers import LIMEExplainer, MBERTExplainer
from config import (
    SVM_MODEL_PATH,
    SVM_VECTORIZER_PATH,
    MBERT_MODEL_PATH,
    MBERT_MAX_LENGTH,
    USE_CUDA,
)

# Try to import the model downloader; warn if not available
try:
    from download_models import ModelDownloader
    HAS_DOWNLOADER = True
except ImportError:
    HAS_DOWNLOADER = False
    print("⚠ Warning: download_models module not available")


app = FastAPI(title="Fake News Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables
MODEL = None
VECTORIZER = None
LIME_EXPLAINER = None
MBERT_MODEL = None
MBERT_TOKENIZER = None
MBERT_EXPLAINER = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")

# Model load status tracking
MODEL_STATUS: Dict[str, Any] = {
    "svm_loaded": False,
    "mbert_loaded": False,
    "errors": [],
}


# =========================================================
# REQUEST SCHEMA
# =========================================================

class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str
    language: Literal["en", "hi"]
    prediction_model: Literal["svm", "mbert"] = Field(default="svm", alias="model_type")

# =========================================================
# STARTUP EVENT
# =========================================================

@app.on_event("startup")
def load_artifacts() -> None:
    """Load all model artifacts with automatic download fallback."""
    global MODEL, VECTORIZER, LIME_EXPLAINER
    global MBERT_MODEL, MBERT_TOKENIZER, MBERT_EXPLAINER
    global MODEL_STATUS

    print("\n" + "=" * 70)
    print("🚀 Starting Backend - Loading Models")
    print("=" * 70)

    # Step 1: Attempt to download missing models
    if HAS_DOWNLOADER:
        print("\n📥 Checking and downloading models if needed...")
        try:
            ModelDownloader.run()
        except Exception as e:
            print(f"⚠ Model download attempt failed: {e}")
            print("  Continuing with existing files...")

    # Step 2: Load SVM stack
    print("\n📦 Loading SVM model stack...")
    try:
        if not SVM_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"SVM model file not found: {SVM_MODEL_PATH}\n"
                f"  → Run: python download_models.py\n"
                f"  → Or ensure Git LFS files are pulled"
            )

        if not SVM_VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                f"Vectorizer file not found: {SVM_VECTORIZER_PATH}\n"
                f"  → Run: python download_models.py"
            )

        # Validate file sizes
        model_size = SVM_MODEL_PATH.stat().st_size
        vec_size = SVM_VECTORIZER_PATH.stat().st_size
        
        if model_size < 1000 or vec_size < 1000:
            raise ValueError(
                f"Model files appear corrupted\n"
                f"  → SVM size: {model_size} bytes\n"
                f"  → Vectorizer size: {vec_size} bytes"
            )

        with SVM_MODEL_PATH.open("rb") as model_file:
            MODEL = pickle.load(model_file)

        with SVM_VECTORIZER_PATH.open("rb") as vectorizer_file:
            VECTORIZER = pickle.load(vectorizer_file)

        LIME_EXPLAINER = LIMEExplainer(MODEL, VECTORIZER)
        MODEL_STATUS["svm_loaded"] = True

        print("  ✅ SVM model loaded successfully")
        print(f"     Size: {model_size / 1024 / 1024:.1f}MB")
        print("  ✅ TF-IDF vectorizer loaded successfully")
        print(f"     Size: {vec_size / 1024 / 1024:.1f}MB")
        print("  ✅ LIME explainer initialized")

    except Exception as exc:
        error_msg = f"Failed to load SVM: {exc}"
        MODEL_STATUS["errors"].append(error_msg)
        print(f"  ❌ {error_msg}")
        MODEL = None
        VECTORIZER = None
        LIME_EXPLAINER = None

    # Step 3: Load mBERT stack
    print("\n📦 Loading mBERT model stack...")
    try:
        if not MBERT_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"mBERT model directory not found: {MBERT_MODEL_PATH}\n"
                f"  → Run: python download_models.py"
            )

        # Validate required files
        required_files = ['config.json', 'tokenizer.json']
        for fname in required_files:
            if not (MBERT_MODEL_PATH / fname).exists():
                raise FileNotFoundError(
                    f"mBERT missing required file: {fname}"
                )

        MBERT_TOKENIZER = AutoTokenizer.from_pretrained(str(MBERT_MODEL_PATH))
        MBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(
            str(MBERT_MODEL_PATH)
        )

        MBERT_MODEL.to(DEVICE)
        MBERT_MODEL.eval()

        MBERT_EXPLAINER = MBERTExplainer(MBERT_TOKENIZER, MBERT_MODEL, DEVICE)
        MODEL_STATUS["mbert_loaded"] = True

        model_size = sum(
            f.stat().st_size for f in MBERT_MODEL_PATH.rglob('*') if f.is_file()
        ) / 1024 / 1024

        print("  ✅ mBERT tokenizer loaded successfully")
        print("  ✅ mBERT model loaded successfully")
        print(f"     Size: {model_size:.1f}MB")
        print("  ✅ mBERT explainer initialized")

    except Exception as exc:
        error_msg = f"Failed to load mBERT: {exc}"
        MODEL_STATUS["errors"].append(error_msg)
        print(f"  ❌ {error_msg}")
        MBERT_MODEL = None
        MBERT_TOKENIZER = None
        MBERT_EXPLAINER = None

    # Step 4: Summary
    print("\n" + "-" * 70)
    print(f"✅ Device: {DEVICE}")
    print(f"✅ SVM Ready: {MODEL_STATUS['svm_loaded']}")
    print(f"✅ mBERT Ready: {MODEL_STATUS['mbert_loaded']}")

    if MODEL_STATUS["errors"]:
        print(f"\n⚠ Warnings ({len(MODEL_STATUS['errors'])}):")
        for err in MODEL_STATUS["errors"]:
            print(f"  • {err.split(chr(10))[0]}")  # First line only

    if not (MODEL_STATUS["svm_loaded"] or MODEL_STATUS["mbert_loaded"]):
        print("\n❌ CRITICAL: No models loaded! Backend will not function.")
        print("   → Run: python download_models.py")
        print("   → Then restart the backend")

    print("=" * 70 + "\n")


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():
    return {
        "message": "Backend is running"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "svm_loaded": MODEL_STATUS["svm_loaded"],
        "mbert_loaded": MODEL_STATUS["mbert_loaded"],
    }


# =========================================================
# MODEL STATUS ENDPOINT
# =========================================================

@app.get("/status")
def get_status():
    """Detailed model status endpoint for diagnostics."""
    return {
        "backend": "operational",
        "device": str(DEVICE),
        "models": {
            "svm": {
                "loaded": MODEL_STATUS["svm_loaded"],
                "model_file": str(SVM_MODEL_PATH),
                "vectorizer_file": str(SVM_VECTORIZER_PATH),
                "ready": MODEL is not None and VECTORIZER is not None,
            },
            "mbert": {
                "loaded": MODEL_STATUS["mbert_loaded"],
                "model_path": str(MBERT_MODEL_PATH),
                "ready": MBERT_MODEL is not None and MBERT_TOKENIZER is not None,
            },
        },
        "errors": MODEL_STATUS["errors"] if MODEL_STATUS["errors"] else None,
    }


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict(payload: PredictRequest, model: Literal["svm", "mbert"] = "svm"):
    model_type = model if model else payload.prediction_model

    try:
        if model_type not in ["svm", "mbert"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid model type. Use 'svm' or 'mbert'."
            )

        # =========================
        # INPUT CLEANING
        # =========================

        cleaned_text = LIMEExplainer.clean_text(payload.text)

        if not cleaned_text:
            raise HTTPException(
                status_code=400,
                detail="Text is empty after cleaning."
            )

        if len(cleaned_text) < 3:
            raise HTTPException(
                status_code=400,
                detail="Text is too short for prediction."
            )

        # =====================================================
        # SVM PREDICTION
        # =====================================================

        if model_type == "svm":

            if MODEL is None or VECTORIZER is None:
                raise HTTPException(
                    status_code=500,
                    detail="SVM model artifacts are not loaded."
                )

            text_features = VECTORIZER.transform([cleaned_text])

            predicted_label = int(
                MODEL.predict(text_features)[0]
            )

            if hasattr(MODEL, "predict_proba"):

                probabilities = MODEL.predict_proba(text_features)[0]

                confidence = float(max(probabilities))

            else:
                confidence = 0.0

        # =====================================================
        # mBERT PREDICTION
        # =====================================================

        elif model_type == "mbert":

            if MBERT_MODEL is None or MBERT_TOKENIZER is None:
                raise HTTPException(
                    status_code=500,
                    detail="mBERT model is not loaded."
                )

            inputs = MBERT_TOKENIZER(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MBERT_MAX_LENGTH
            )

            inputs = {
                k: v.to(DEVICE)
                for k, v in inputs.items()
            }

            with torch.no_grad():

                outputs = MBERT_MODEL(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )

            predicted_label = int(
                torch.argmax(probabilities, dim=1).item()
            )

            confidence = float(
                torch.max(probabilities).item()
            )

        # =====================================================
        # INVALID MODEL
        # =====================================================

        else:

            raise HTTPException(
                status_code=400,
                detail="Invalid model type selected."
            )

        # =====================================================
        # RESPONSE
        # =====================================================

        return {
            "prediction": predicted_label,
            "confidence": confidence,
            "model_used": payload.prediction_model,
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")


# =========================================================
# PREDICT WITH EXPLANATIONS (SVM: LIME, mBERT: Attention)
# =========================================================

@app.post("/predict-with-lime")
def predict_with_lime(payload: PredictRequest, model: Literal["svm", "mbert"] = "svm"):
    model_type = model if model else payload.prediction_model

    try:

        # =========================================
        # SVM WITH LIME EXPLANATION
        # =========================================

        if model_type == "svm":

            if MODEL is None or VECTORIZER is None:
                raise HTTPException(
                    status_code=500,
                    detail="SVM model artifacts are not loaded."
                )

            if LIME_EXPLAINER is None:
                raise HTTPException(
                    status_code=500,
                    detail="LIME explainer is not initialized."
                )

            cleaned_text = LIMEExplainer.clean_text(payload.text)

            if not cleaned_text:
                raise HTTPException(
                    status_code=400,
                    detail="Text is empty after cleaning."
                )

            if len(cleaned_text) < 3:
                raise HTTPException(
                    status_code=400,
                    detail="Text is too short for prediction and explanation."
                )

            text_features = VECTORIZER.transform([cleaned_text])
            predicted_label = int(MODEL.predict(text_features)[0])

            if not hasattr(MODEL, "predict_proba"):
                raise HTTPException(
                    status_code=500,
                    detail="Loaded model does not support probability output."
                )

            probabilities = MODEL.predict_proba(text_features)[0]
            confidence = float(max(probabilities))

            try:
                explanation = LIME_EXPLAINER.explain(cleaned_text, num_features=5)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"LIME input validation failed: {exc}"
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"LIME explanation failed: {exc}"
                )

            return {
                "prediction": predicted_label,
                "confidence": confidence,
                "model_used": "svm",
                "explanation": explanation,
            }

        # =========================================
        # mBERT WITH ATTENTION-BASED EXPLANATION
        # =========================================

        elif model_type == "mbert":

            if MBERT_MODEL is None or MBERT_TOKENIZER is None:
                raise HTTPException(
                    status_code=500,
                    detail="mBERT model is not loaded."
                )

            if MBERT_EXPLAINER is None:
                raise HTTPException(
                    status_code=500,
                    detail="mBERT explainer is not initialized."
                )

            cleaned_text = MBERTExplainer.clean_text(payload.text)

            if not cleaned_text:
                raise HTTPException(
                    status_code=400,
                    detail="Text is empty after cleaning."
                )

            if len(cleaned_text) < 3:
                raise HTTPException(
                    status_code=400,
                    detail="Text is too short for prediction and explanation."
                )

            inputs = MBERT_TOKENIZER(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MBERT_MAX_LENGTH
            )

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = MBERT_MODEL(**inputs, output_attentions=True)

            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_label = int(torch.argmax(probabilities, dim=1).item())
            confidence = float(torch.max(probabilities).item())

            try:
                explanation = MBERT_EXPLAINER.explain(cleaned_text, top_k=5)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Explanation input validation failed: {exc}"
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"Attention explanation failed: {exc}"
                )

            return {
                "prediction": predicted_label,
                "confidence": confidence,
                "model_used": "mbert",
                "explanation": explanation,
            }

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model type: {payload.prediction_model}"
            )

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(status_code=400, detail=f"Prediction with LIME failed: {exc}")


# =========================================================
# MAIN
# =========================================================

# =========================================================
# BATCH PREDICTION ENDPOINT
# =========================================================

class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    texts: list[str]
    language: Literal["en", "hi"]
    prediction_model: Literal["svm", "mbert"] = Field(alias="model_type")


@app.post("/predict-batch")
def predict_batch(payload: BatchPredictRequest):
    """Batch predict multiple texts at once."""
    try:
        if not payload.texts:
            raise HTTPException(
                status_code=400,
                detail="No texts provided for batch prediction."
            )

        results = []
        for text in payload.texts:
            request = PredictRequest(
                text=text,
                language=payload.language,
                model_type=payload.prediction_model
            )
            result = predict(request)
            results.append(result)

        return {
            "predictions": results,
            "count": len(results),
            "model_used": payload.prediction_model,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Batch prediction failed: {exc}"
        )


# =========================================================
# MODEL INFO ENDPOINT
# =========================================================

@app.get("/model-info")
def get_model_info():
    """Get information about loaded models."""
    return {
        "svm": {
            "name": "Linear SVC (TF-IDF)",
            "accuracy": "~85%",
            "f1_score": "~0.81",
            "inference_time_ms": "~100",
            "loaded": MODEL is not None,
            "supports_lime": True,
        },
        "mbert": {
            "name": "mBERT (Multilingual BERT)",
            "accuracy": "91.15%",
            "f1_score": "0.8790",
            "inference_time_ms": "~500",
            "loaded": MBERT_MODEL is not None,
            "supports_lime": False,
            "device": str(DEVICE),
        },
    }

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )