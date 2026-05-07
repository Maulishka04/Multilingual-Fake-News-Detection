"""Unit tests for MBertInference — runs without GPU or network access.

We mock the ``transformers`` and ``torch`` packages so these tests execute
in any CI environment that does not have heavy ML libraries installed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal stubs for torch / transformers so the module can be imported
# ---------------------------------------------------------------------------


def _make_torch_stub() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.cuda = MagicMock()
    torch.cuda.is_available = MagicMock(return_value=False)
    torch.device = lambda x: x
    torch.no_grad = MagicMock(return_value=MagicMock(__enter__=lambda s, *a: s, __exit__=lambda s, *a: None))
    torch.Tensor = MagicMock  # Required by scipy's array API compatibility layer

    # Fake softmax that returns uniform probabilities
    def fake_softmax(logits, dim):
        arr = np.ones((logits.shape[0], 2)) * 0.5
        return MagicMock(cpu=lambda: MagicMock(numpy=lambda: arr))

    torch.softmax = fake_softmax
    return torch


def _make_transformers_stub() -> types.ModuleType:
    transformers = types.ModuleType("transformers")

    fake_model = MagicMock()
    fake_model.eval = MagicMock(return_value=None)
    fake_model.to = MagicMock(return_value=fake_model)

    # Return logits shaped (1, 2)
    fake_logits = MagicMock()
    fake_logits.shape = (1, 2)
    fake_logits.__getitem__ = lambda s, i: MagicMock()
    fake_output = MagicMock()
    fake_output.logits = fake_logits
    fake_model.__call__ = MagicMock(return_value=fake_output)
    fake_model.return_value = fake_output

    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = {"input_ids": MagicMock(shape=(1, 128))}

    transformers.AutoTokenizer = MagicMock()
    transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=fake_tokenizer)
    transformers.AutoModelForSequenceClassification = MagicMock()
    transformers.AutoModelForSequenceClassification.from_pretrained = MagicMock(return_value=fake_model)

    return transformers


# ---------------------------------------------------------------------------
# Import the module under test (after stubs are in place)
# ---------------------------------------------------------------------------

sys.modules.setdefault("torch", _make_torch_stub())
sys.modules.setdefault("transformers", _make_transformers_stub())

# Now import the module under test
BACKEND_DIR = Path(__file__).parent.parent / "fake_news_backend"
sys.path.insert(0, str(BACKEND_DIR))

from models.mbert_inference import MBertInference  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def unloaded_inference() -> MBertInference:
    return MBertInference(model_dir=None, fallback_model_name="bert-base-multilingual-cased")


@pytest.fixture()
def loaded_inference(tmp_path) -> MBertInference:
    """Return an MBertInference instance whose load() has been called."""
    # Create a fake local model dir with config.json to exercise local-path branch
    (tmp_path / "config.json").write_text('{"model_type": "bert"}')

    instance = MBertInference(
        model_dir=tmp_path,
        fallback_model_name="bert-base-multilingual-cased",
        max_length=32,
    )
    instance.load()
    return instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_lowercases_ascii(self):
        assert MBertInference.clean_text("Hello WORLD") == "hello world"

    def test_removes_special_chars(self):
        result = MBertInference.clean_text("news@site.com is fake!")
        assert "@" not in result
        assert "!" not in result

    def test_preserves_hindi_chars(self):
        text = "यह खबर झूठी है"
        cleaned = MBertInference.clean_text(text)
        # Hindi characters must be retained
        assert "य" in cleaned

    def test_collapses_whitespace(self):
        assert MBertInference.clean_text("  too   many   spaces  ") == "too many spaces"

    def test_empty_string(self):
        assert MBertInference.clean_text("") == ""


class TestIsLoaded:
    def test_not_loaded_initially(self, unloaded_inference):
        assert not unloaded_inference.is_loaded()

    def test_loaded_after_load(self, loaded_inference):
        assert loaded_inference.is_loaded()


class TestIsLocalModel:
    def test_local_model_detected(self, loaded_inference):
        # loaded_inference points at a tmp_path that has config.json
        assert loaded_inference.is_local_model()

    def test_not_local_when_no_dir(self):
        instance = MBertInference(model_dir=None)
        # Before loading, _is_local is False
        assert not instance.is_local_model()


class TestLoad:
    def test_raises_if_model_dir_invalid(self, tmp_path):
        """When the model dir exists but has no config.json, falls back to HF hub."""
        instance = MBertInference(model_dir=tmp_path / "nonexistent")
        # Should not raise — falls back to HuggingFace
        instance.load()
        assert instance.is_loaded()

    def test_load_succeeds(self, loaded_inference):
        assert loaded_inference.is_loaded()


class TestPredict:
    def test_raises_before_load(self, unloaded_inference):
        with pytest.raises(RuntimeError, match="not loaded"):
            unloaded_inference.predict("some text")

    def test_raises_on_short_text(self, loaded_inference):
        with pytest.raises(ValueError, match="too short"):
            loaded_inference.predict("ab")

    def test_raises_on_empty_text(self, loaded_inference):
        with pytest.raises(ValueError):
            loaded_inference.predict("!!!")  # All non-alphanumeric → empty after cleaning

    def test_returns_expected_keys(self, loaded_inference):
        result = loaded_inference.predict("This is a sample news article about politics.")
        assert "prediction" in result
        assert "confidence" in result
        assert "inference_time_ms" in result
        assert "model_source" in result

    def test_prediction_is_int(self, loaded_inference):
        result = loaded_inference.predict("This is a sample news article about politics.")
        assert isinstance(result["prediction"], int)
        assert result["prediction"] in {0, 1}

    def test_confidence_in_range(self, loaded_inference):
        result = loaded_inference.predict("This is a sample news article about politics.")
        assert 0.0 <= result["confidence"] <= 1.0


class TestPredictBatch:
    def test_raises_on_empty_list(self, loaded_inference):
        with pytest.raises(ValueError, match="empty"):
            loaded_inference.predict_batch([])

    def test_returns_list_with_correct_length(self, loaded_inference):
        texts = ["Breaking news: scientists discover new planet"]
        results = loaded_inference.predict_batch(texts)
        assert len(results) == len(texts)

    def test_each_result_has_expected_keys(self, loaded_inference):
        results = loaded_inference.predict_batch(["Sample news text for testing purposes."])
        for r in results:
            assert "prediction" in r
            assert "confidence" in r
