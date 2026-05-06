from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any


MODEL_FILENAME = "linear_svc_calibrated_tfidf.pkl"
VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"


class ModelRegenerationError(Exception):
    """Raised when model/vectorizer regeneration fails."""


def _ensure_has_transformer_interface(vectorizer: Any) -> None:
    if not hasattr(vectorizer, "transform"):
        raise ModelRegenerationError(
            "Vectorizer does not look fitted or valid (missing transform method)."
        )


def _ensure_has_predict_interface(model: Any) -> None:
    if not hasattr(model, "predict"):
        raise ModelRegenerationError(
            "Model does not look fitted or valid (missing predict method)."
        )


def _save_pickle(obj: Any, output_path: Path, label: str) -> None:
    try:
        with output_path.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ {label} saved")
    except Exception as exc:
        raise ModelRegenerationError(f"Failed to save {label}: {exc}") from exc


def _verify_pickle(path: Path, label: str) -> Any:
    if not path.exists():
        raise ModelRegenerationError(f"{label} file missing after save: {path}")

    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"{label} file size: {size_bytes} bytes ({size_mb:.2f} MB)")

    if size_bytes == 0:
        raise ModelRegenerationError(f"{label} file is empty: {path}")

    try:
        with path.open("rb") as f:
            loaded = pickle.load(f)
        print(f"✅ {label} reloaded successfully: {type(loaded)}")
        return loaded
    except Exception as exc:
        raise ModelRegenerationError(f"Failed to reload {label}: {exc}") from exc


def regenerate_and_verify(model: Any, vectorizer: Any, models_dir: Path | str = "models") -> tuple[Path, Path]:
    """
    Save model and vectorizer to disk, then verify by loading them back.

    Call this from notebook after training, for example:
        from regenerate_models import regenerate_and_verify
        regenerate_and_verify(model, vectorizer)
    """
    models_path = Path(models_dir)

    if model is None:
        raise ModelRegenerationError("Model is None.")
    if vectorizer is None:
        raise ModelRegenerationError("Vectorizer is None.")

    _ensure_has_predict_interface(model)
    _ensure_has_transformer_interface(vectorizer)

    try:
        models_path.mkdir(exist_ok=True)
    except Exception as exc:
        raise ModelRegenerationError(f"Could not create models directory: {exc}") from exc

    model_path = models_path / MODEL_FILENAME
    vectorizer_path = models_path / VECTORIZER_FILENAME

    _save_pickle(model, model_path, "Model")
    _save_pickle(vectorizer, vectorizer_path, "Vectorizer")

    _verify_pickle(model_path, "Model")
    _verify_pickle(vectorizer_path, "Vectorizer")

    print("✅ Files verified and working")
    return model_path, vectorizer_path


if __name__ == "__main__":
    print("This script is intended to be called from a notebook where model and vectorizer are in memory.")
    print("Example:")
    print("from regenerate_models import regenerate_and_verify")
    print("regenerate_and_verify(model, vectorizer)")
