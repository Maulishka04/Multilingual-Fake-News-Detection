"""Integration tests for the /compare and /available-models endpoints.

These tests use FastAPI's TestClient and mock the global model state so that
no real model files are required.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal stubs for torch / transformers
# ---------------------------------------------------------------------------


def _make_torch_stub() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.cuda = MagicMock()
    torch.cuda.is_available = MagicMock(return_value=False)
    torch.device = lambda x: x
    torch.no_grad = MagicMock(return_value=MagicMock(__enter__=lambda s, *a: s, __exit__=lambda s, *a: None))
    torch.Tensor = MagicMock  # Required by scipy's array API compatibility layer

    def fake_softmax(logits, dim):
        arr = np.array([[0.3, 0.7]])
        return MagicMock(cpu=lambda: MagicMock(numpy=lambda: arr))

    torch.softmax = fake_softmax
    return torch


def _make_transformers_stub() -> types.ModuleType:
    transformers = types.ModuleType("transformers")
    fake_logits = MagicMock()
    fake_logits.shape = (1, 2)
    fake_output = MagicMock()
    fake_output.logits = fake_logits
    fake_model = MagicMock()
    fake_model.eval = MagicMock(return_value=None)
    fake_model.to = MagicMock(return_value=fake_model)
    fake_model.return_value = fake_output
    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = {"input_ids": MagicMock(shape=(1, 128))}
    transformers.AutoTokenizer = MagicMock()
    transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=fake_tokenizer)
    transformers.AutoModelForSequenceClassification = MagicMock()
    transformers.AutoModelForSequenceClassification.from_pretrained = MagicMock(return_value=fake_model)
    return transformers


sys.modules.setdefault("torch", _make_torch_stub())
sys.modules.setdefault("transformers", _make_transformers_stub())

BACKEND_DIR = Path(__file__).parent.parent / "fake_news_backend"
sys.path.insert(0, str(BACKEND_DIR))

# ---------------------------------------------------------------------------
# Build a test-friendly FastAPI app
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402

# We need to patch load_artifacts before the app starts
with patch("models.mbert_inference.MBertInference.load"):
    import main as app_module  # noqa: E402

client = TestClient(app_module.app, raise_server_exceptions=True)


def _make_fake_svm():
    """Return a minimal mock that behaves like a fitted CalibratedClassifierCV."""
    svm = MagicMock()
    svm.predict = MagicMock(return_value=[1])
    svm.predict_proba = MagicMock(return_value=np.array([[0.2, 0.8]]))
    return svm


def _make_fake_vectorizer():
    vec = MagicMock()
    vec.transform = MagicMock(return_value=MagicMock())
    return vec


def _make_fake_mbert():
    from models.mbert_inference import MBertInference

    mbert = MagicMock(spec=MBertInference)
    mbert.is_loaded.return_value = True
    mbert.is_local_model.return_value = True
    mbert.predict.return_value = {
        "prediction": 1,
        "confidence": 0.91,
        "inference_time_ms": 123.4,
        "model_source": "local",
    }
    return mbert


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_no_models(self):
        with patch.multiple(app_module, SVM_MODEL=None, MBERT=None):
            response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["svm_loaded"] is False
        assert data["mbert_loaded"] is False

    def test_health_with_svm(self):
        with patch.multiple(app_module, SVM_MODEL=_make_fake_svm(), MBERT=None):
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["svm_loaded"] is True

    def test_health_with_mbert(self):
        with patch.multiple(app_module, SVM_MODEL=None, MBERT=_make_fake_mbert()):
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["mbert_loaded"] is True


class TestPredictEndpoint:
    def test_svm_predict(self):
        with patch.multiple(
            app_module,
            SVM_MODEL=_make_fake_svm(),
            SVM_VECTORIZER=_make_fake_vectorizer(),
        ):
            response = client.post(
                "/predict?model=svm",
                json={"text": "Scientists discover breakthrough in cancer treatment", "language": "en"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "svm"
        assert data["prediction"] in {0, 1}

    def test_mbert_predict(self):
        with patch.multiple(app_module, MBERT=_make_fake_mbert()):
            response = client.post(
                "/predict?model=mbert",
                json={"text": "Scientists discover breakthrough in cancer treatment", "language": "en"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "mbert"
        assert "confidence" in data

    def test_unknown_model_returns_400(self):
        response = client.post(
            "/predict?model=gpt4",
            json={"text": "Some news text here", "language": "en"},
        )
        assert response.status_code == 400

    def test_empty_text_returns_400(self):
        response = client.post(
            "/predict",
            json={"text": "!!!", "language": "en"},
        )
        assert response.status_code == 400

    def test_svm_unavailable_returns_503(self):
        with patch.multiple(app_module, SVM_MODEL=None, SVM_VECTORIZER=None):
            response = client.post(
                "/predict?model=svm",
                json={"text": "Some text about politics and elections", "language": "en"},
            )
        assert response.status_code == 503


class TestCompareEndpoint:
    def test_compare_both_models(self):
        with patch.multiple(
            app_module,
            SVM_MODEL=_make_fake_svm(),
            SVM_VECTORIZER=_make_fake_vectorizer(),
            MBERT=_make_fake_mbert(),
        ):
            response = client.post(
                "/compare",
                json={"text": "A major political scandal erupted today", "language": "en"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "svm" in data["results"]
        assert "mbert" in data["results"]
        assert "agreement" in data
        assert isinstance(data["agreement"], bool)

    def test_compare_svm_only(self):
        with patch.multiple(
            app_module,
            SVM_MODEL=_make_fake_svm(),
            SVM_VECTORIZER=_make_fake_vectorizer(),
            MBERT=None,
        ):
            response = client.post(
                "/compare",
                json={"text": "A major political scandal erupted today", "language": "en"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "svm" in data["results"]
        assert "mbert" in data["errors"]
        assert data["agreement"] is None

    def test_compare_no_models_returns_503(self):
        with patch.multiple(app_module, SVM_MODEL=None, SVM_VECTORIZER=None, MBERT=None):
            response = client.post(
                "/compare",
                json={"text": "A major political scandal erupted today", "language": "en"},
            )
        assert response.status_code == 503

    def test_compare_empty_text(self):
        response = client.post("/compare", json={"text": "!!!", "language": "en"})
        assert response.status_code == 400


class TestAvailableModels:
    def test_lists_all_models(self):
        response = client.get("/available-models")
        assert response.status_code == 200
        data = response.json()
        names = {m["name"] for m in data["models"]}
        assert "svm" in names
        assert "mbert" in names

    def test_loaded_flags(self):
        with patch.multiple(
            app_module,
            SVM_MODEL=_make_fake_svm(),
            MBERT=_make_fake_mbert(),
        ):
            response = client.get("/available-models")
        assert response.status_code == 200
        data = response.json()
        model_map = {m["name"]: m for m in data["models"]}
        assert model_map["svm"]["loaded"] is True
        assert model_map["mbert"]["loaded"] is True


class TestModelInfo:
    def test_svm_info(self):
        response = client.get("/model-info/svm")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "svm"
        assert "accuracy" in data

    def test_mbert_info(self):
        response = client.get("/model-info/mbert")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "mbert"
        assert "f1_score" in data

    def test_unknown_model_returns_404(self):
        response = client.get("/model-info/gpt4")
        assert response.status_code == 404

    def test_case_insensitive(self):
        response = client.get("/model-info/SVM")
        assert response.status_code == 200
