import pickle
from typing import Literal

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
    global MODEL, VECTORIZER, LIME_EXPLAINER
    global MBERT_MODEL, MBERT_TOKENIZER

    try:

        # Load the classical ML stack used for fast predictions and LIME.

        if not SVM_MODEL_PATH.exists():
            raise FileNotFoundError(f"SVM model file not found: {SVM_MODEL_PATH}")

        if not SVM_VECTORIZER_PATH.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {SVM_VECTORIZER_PATH}")

        with SVM_MODEL_PATH.open("rb") as model_file:
            MODEL = pickle.load(model_file)

        with SVM_VECTORIZER_PATH.open("rb") as vectorizer_file:
            VECTORIZER = pickle.load(vectorizer_file)

        LIME_EXPLAINER = LIMEExplainer(MODEL, VECTORIZER)

        print("✅ SVM model loaded successfully.")
        print("✅ TF-IDF vectorizer loaded successfully.")
        print("✅ LIME explainer initialized.")

        # Load the transformer stack used for multilingual predictions.

        if not MBERT_MODEL_PATH.exists():
            raise FileNotFoundError(f"mBERT folder not found: {MBERT_MODEL_PATH}")

        MBERT_TOKENIZER = AutoTokenizer.from_pretrained(MBERT_MODEL_PATH)

        MBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(
            MBERT_MODEL_PATH
        )

        MBERT_MODEL.to(DEVICE)
        MBERT_MODEL.eval()

        MBERT_EXPLAINER = MBERTExplainer(MBERT_TOKENIZER, MBERT_MODEL, DEVICE)

        print("✅ mBERT tokenizer loaded successfully.")
        print("✅ mBERT model loaded successfully.")
        print("✅ mBERT explainer initialized.")
        print(f"✅ Running on device: {DEVICE}")

    except Exception as exc:

        MODEL = None
        VECTORIZER = None
        LIME_EXPLAINER = None

        MBERT_MODEL = None
        MBERT_TOKENIZER = None
        MBERT_EXPLAINER = None

        print(f"❌ Error loading artifacts: {exc}")


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

    return {
        "status": "healthy",
        "svm_loaded": MODEL is not None,
        "mbert_loaded": MBERT_MODEL is not None,
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