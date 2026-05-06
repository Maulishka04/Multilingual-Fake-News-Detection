from pathlib import Path
import pickle
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from explainers import LIMEExplainer

app = FastAPI(title="Fake News Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL = None
VECTORIZER = None
LIME_EXPLAINER = None
MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "linear_svc_calibrated_tfidf.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"


class PredictRequest(BaseModel):
    text: str
    language: Literal["en", "hi"]


@app.on_event("startup")
def load_artifacts() -> None:
    global MODEL, VECTORIZER, LIME_EXPLAINER

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {VECTORIZER_PATH}")

        with MODEL_PATH.open("rb") as model_file:
            MODEL = pickle.load(model_file)
        with VECTORIZER_PATH.open("rb") as vectorizer_file:
            VECTORIZER = pickle.load(vectorizer_file)

        LIME_EXPLAINER = LIMEExplainer(MODEL, VECTORIZER)
        print("LIME explainer initialized")

        print("Model and vectorizer loaded successfully.")
    except Exception as exc:
        MODEL = None
        VECTORIZER = None
        LIME_EXPLAINER = None
        print(f"Error loading model/vectorizer: {exc}")


@app.get("/")
def root():
    return {"message": "Backend is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        if MODEL is None or VECTORIZER is None:
            raise HTTPException(status_code=500, detail="Model artifacts are not loaded.")

        cleaned_text = LIMEExplainer.clean_text(payload.text)
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Text is empty after cleaning.")
        if len(cleaned_text) < 3:
            raise HTTPException(status_code=400, detail="Text is too short for prediction.")

        text_features = VECTORIZER.transform([cleaned_text])
        predicted_label = int(MODEL.predict(text_features)[0])

        if hasattr(MODEL, "predict_proba"):
            probabilities = MODEL.predict_proba(text_features)[0]
            confidence = float(max(probabilities))
        else:
            confidence = 0.0

        return {"prediction": predicted_label, "confidence": confidence}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")


@app.post("/predict-with-lime")
def predict_with_lime(payload: PredictRequest):
    try:
        if MODEL is None or VECTORIZER is None:
            raise HTTPException(status_code=500, detail="Model artifacts are not loaded.")
        if LIME_EXPLAINER is None:
            raise HTTPException(status_code=500, detail="LIME explainer is not initialized.")

        cleaned_text = LIMEExplainer.clean_text(payload.text)
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Text is empty after cleaning.")
        if len(cleaned_text) < 3:
            raise HTTPException(status_code=400, detail="Text is too short for prediction and explanation.")

        text_features = VECTORIZER.transform([cleaned_text])
        predicted_label = int(MODEL.predict(text_features)[0])

        if not hasattr(MODEL, "predict_proba"):
            raise HTTPException(status_code=500, detail="Loaded model does not support probability output.")

        probabilities = MODEL.predict_proba(text_features)[0]
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
            "explanation": explanation,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction with LIME failed: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
