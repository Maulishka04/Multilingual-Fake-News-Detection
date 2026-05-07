"""mBERT-specific FastAPI router."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/mbert", tags=["mBERT"])


class MBertPredictRequest(BaseModel):
    text: str
    language: Literal["en", "hi"]


@router.post("/predict")
def mbert_predict(payload: MBertPredictRequest, mbert_inference=None):
    """Run mBERT inference on the provided text.

    This endpoint is registered by the main application which injects the
    mBERT inference instance via a dependency.  Direct calls without the
    application context will fail — use the main ``/predict?model=mbert``
    endpoint instead.
    """
    raise HTTPException(
        status_code=501,
        detail=(
            "Use POST /predict?model=mbert via the main application. "
            "This endpoint is a placeholder for the router module."
        ),
    )
