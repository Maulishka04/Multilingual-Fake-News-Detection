"""mBERT inference utilities for multilingual fake news classification."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class MBertInference:
    """Inference wrapper for fine-tuned mBERT sequence classification models.

    Loads a local fine-tuned model from *model_dir* if the directory exists and
    contains the required files; otherwise falls back to downloading
    ``bert-base-multilingual-cased`` from HuggingFace Hub and using it as an
    un-fine-tuned baseline (for development / CI purposes).

    The class is intentionally lazy: heavy PyTorch/Transformers imports are
    deferred to :meth:`load` so the module can be imported without those
    packages installed (they will be required only when a prediction is made).
    """

    _MIN_TEXT_LENGTH = 3

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        fallback_model_name: str = "bert-base-multilingual-cased",
        max_length: int = 128,
        device: Optional[str] = None,
    ) -> None:
        """Initialise the inference wrapper.

        Args:
            model_dir: Directory that contains ``config.json`` and the model
                weights (``pytorch_model.bin`` or safetensors shards).  When
                *None* or absent on disk the ``fallback_model_name`` is used.
            fallback_model_name: HuggingFace model identifier used when the
                local model directory is unavailable.
            max_length: Maximum token sequence length (default 128).
            device: PyTorch device string (``"cpu"``, ``"cuda"``).  Defaults to
                ``"cuda"`` when a GPU is detected, otherwise ``"cpu"``.
        """
        self.model_dir = model_dir
        self.fallback_model_name = fallback_model_name
        self.max_length = max_length
        self._device_override = device

        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None
        self._loaded = False
        self._is_local = False

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalise text while preserving Hindi Unicode characters (U+0900–U+097F)."""
        lowered = text.lower().strip()
        cleaned = re.sub(r"[^a-z0-9\u0900-\u097f\s]", " ", lowered)
        return re.sub(r"\s+", " ", cleaned).strip()

    def is_loaded(self) -> bool:
        """Return ``True`` when model artifacts have been loaded successfully."""
        return self._loaded

    def is_local_model(self) -> bool:
        """Return ``True`` when a local fine-tuned model is being used."""
        return self._is_local

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the tokenizer and model into memory.

        Raises:
            ImportError: If ``transformers`` or ``torch`` are not installed.
            RuntimeError: If the model cannot be loaded from any source.
        """
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' and 'torch' packages are required for mBERT "
                "inference.  Install them with: pip install transformers torch"
            ) from exc

        if self._device_override:
            self._device = torch.device(self._device_override)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prefer the local fine-tuned model
        source = self._resolve_model_source()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(source)
            self._model = AutoModelForSequenceClassification.from_pretrained(source)
            self._model.to(self._device)
            self._model.eval()
            self._is_local = source != self.fallback_model_name
            self._loaded = True
            print(
                f"mBERT loaded from {'local path' if self._is_local else 'Hugging Face Hub'}: "
                f"{source}"
            )
        except Exception as exc:
            self._loaded = False
            raise RuntimeError(f"Failed to load mBERT model from '{source}': {exc}") from exc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Dict[str, Any]:
        """Classify *text* and return prediction, confidence, and latency.

        Args:
            text: Raw input text (will be cleaned internally).

        Returns:
            A dict with keys ``prediction`` (int), ``confidence`` (float),
            ``inference_time_ms`` (float), and ``model_source`` (str).

        Raises:
            RuntimeError: If :meth:`load` has not been called.
            ValueError: If *text* is too short after cleaning.
        """
        if not self._loaded:
            raise RuntimeError("mBERT model is not loaded. Call load() first.")

        cleaned = self.clean_text(text)
        if not cleaned or len(cleaned) < self._MIN_TEXT_LENGTH:
            raise ValueError("Text is too short for mBERT prediction after cleaning.")

        import torch

        start = time.perf_counter()
        inputs = self._tokenizer(
            cleaned,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        elapsed_ms = (time.perf_counter() - start) * 1000
        predicted_label = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_label])

        return {
            "prediction": predicted_label,
            "confidence": confidence,
            "inference_time_ms": round(elapsed_ms, 2),
            "model_source": "local" if self._is_local else "huggingface_hub",
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify a list of texts in a single forward pass.

        Args:
            texts: List of raw input texts.

        Returns:
            List of prediction dicts (same schema as :meth:`predict`).

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError: If *texts* is empty.
        """
        if not self._loaded:
            raise RuntimeError("mBERT model is not loaded. Call load() first.")
        if not texts:
            raise ValueError("texts list must not be empty.")

        import torch

        cleaned = [self.clean_text(t) for t in texts]

        start = time.perf_counter()
        inputs = self._tokenizer(
            cleaned,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs_batch = torch.softmax(logits, dim=-1).cpu().numpy()

        elapsed_ms = (time.perf_counter() - start) * 1000
        source = "local" if self._is_local else "huggingface_hub"

        results: List[Dict[str, Any]] = []
        for probs in probs_batch:
            label = int(np.argmax(probs))
            results.append(
                {
                    "prediction": label,
                    "confidence": float(probs[label]),
                    "inference_time_ms": round(elapsed_ms / len(texts), 2),
                    "model_source": source,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_model_source(self) -> str:
        """Return the model path or HuggingFace identifier to load from."""
        if self.model_dir is not None and self.model_dir.is_dir():
            config_file = self.model_dir / "config.json"
            if config_file.exists():
                return str(self.model_dir)
        return self.fallback_model_name
