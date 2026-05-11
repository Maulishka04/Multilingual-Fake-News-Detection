#!/usr/bin/env python3
"""
Download and validate model artifacts for the fake news detection backend.

This script now focuses on the SVM stack for Render's free tier.
By default it verifies the cached mBERT files if they already exist,
but it does not re-download them unless explicitly enabled.
"""

import os
import sys
from pathlib import Path
from typing import Tuple
import urllib.error
import urllib.request

from config import MBERT_MODEL_PATH, MODELS_DIR, SVM_MODEL_PATH, SVM_VECTORIZER_PATH


class ModelDownloader:
    """Handle downloading and validating model files."""

    GITHUB_REPO_RAW = "https://raw.githubusercontent.com/Maulishka04/Multilingual-Fake-News-Detection/main"
    SVM_MODEL_URL = f"{GITHUB_REPO_RAW}/fake_news_backend/models/linear_svc_calibrated_tfidf.pkl"
    SVM_VECTORIZER_URL = f"{GITHUB_REPO_RAW}/fake_news_backend/models/tfidf_vectorizer.pkl"
    ENABLE_MBERT_DOWNLOAD = os.getenv("ENABLE_MBERT_DOWNLOAD", "false").lower() == "true"
    MBERT_MODEL_HF = "bert-base-multilingual-cased"

    @staticmethod
    def ensure_models_dir() -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        MBERT_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        print(f"✓ Models directory ready: {MODELS_DIR}")

    @staticmethod
    def _is_valid_binary(path: Path, minimum_size: int = 1000) -> bool:
        return path.exists() and path.is_file() and path.stat().st_size >= minimum_size

    @staticmethod
    def download_file(url: str, dest_path: Path, timeout: int = 30) -> bool:
        """Download a file using urllib.request.urlopen with timeout support."""
        try:
            print(f"  Downloading: {url}")
            print(f"  Destination: {dest_path}")

            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )

            with urllib.request.urlopen(request, timeout=timeout) as response:
                total_size = int(response.headers.get("Content-Length", "0") or 0)
                downloaded = 0
                chunk_size = 1024 * 1024
                with dest_path.open("wb") as output_file:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        output_file.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = min(downloaded * 100 / total_size, 100)
                            print(
                                f"    Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f}MB)",
                                end="\r",
                            )

            print()
            print(f"  ✓ Downloaded: {dest_path.name}")
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            print(f"  ✗ Failed to download {url}: {exc}")
            return False
        except Exception as exc:
            print(f"  ✗ Unexpected download error for {url}: {exc}")
            return False

    @staticmethod
    def download_svm_models() -> bool:
        print("\n📦 Checking SVM models...")

        if ModelDownloader._is_valid_binary(SVM_MODEL_PATH) and ModelDownloader._is_valid_binary(SVM_VECTORIZER_PATH):
            model_size = SVM_MODEL_PATH.stat().st_size / 1024 / 1024
            vec_size = SVM_VECTORIZER_PATH.stat().st_size / 1024 / 1024
            print(f"✓ SVM model already exists ({model_size:.1f}MB)")
            print(f"✓ Vectorizer already exists ({vec_size:.1f}MB)")
            return True

        print("  SVM models not found. Attempting download from GitHub raw/LFS fallback...")

        model_success = ModelDownloader.download_file(ModelDownloader.SVM_MODEL_URL, SVM_MODEL_PATH)
        vectorizer_success = ModelDownloader.download_file(ModelDownloader.SVM_VECTORIZER_URL, SVM_VECTORIZER_PATH)

        if model_success and vectorizer_success and ModelDownloader._is_valid_binary(SVM_MODEL_PATH) and ModelDownloader._is_valid_binary(SVM_VECTORIZER_PATH):
            print("✓ SVM models downloaded successfully")
            return True

        print("⚠ Could not download SVM models.")
        print("  Make sure Git LFS objects are pushed and available in the repository.")
        return False

    @staticmethod
    def download_mbert_model() -> bool:
        print("\n📦 Checking mBERT model...")

        required_files = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]
        if all((MBERT_MODEL_PATH / name).exists() for name in required_files):
            model_size = sum(file.stat().st_size for file in MBERT_MODEL_PATH.rglob("*") if file.is_file()) / 1024 / 1024
            print(f"✓ mBERT model already exists ({model_size:.1f}MB)")
            return True

        if not ModelDownloader.ENABLE_MBERT_DOWNLOAD:
            print("  Skipping mBERT download (disabled by default to preserve memory on Render)")
            return False

        print(f"  Downloading mBERT from Hugging Face: {ModelDownloader.MBERT_MODEL_HF}")
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            MBERT_MODEL_PATH.mkdir(parents=True, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(ModelDownloader.MBERT_MODEL_HF)
            model = AutoModelForSequenceClassification.from_pretrained(
                ModelDownloader.MBERT_MODEL_HF,
                num_labels=2,
                low_cpu_mem_usage=True,
            )
            tokenizer.save_pretrained(MBERT_MODEL_PATH)
            model.save_pretrained(MBERT_MODEL_PATH)

            model_size = sum(file.stat().st_size for file in MBERT_MODEL_PATH.rglob("*") if file.is_file()) / 1024 / 1024
            print(f"✓ mBERT model downloaded and saved ({model_size:.1f}MB)")
            return True
        except Exception as exc:
            print(f"✗ Failed to download mBERT model: {exc}")
            return False

    @staticmethod
    def validate_models() -> Tuple[bool, str]:
        print("\n🔍 Validating models...")

        issues = []

        if not ModelDownloader._is_valid_binary(SVM_MODEL_PATH):
            issues.append(f"SVM model missing or invalid: {SVM_MODEL_PATH}")
        if not ModelDownloader._is_valid_binary(SVM_VECTORIZER_PATH):
            issues.append(f"SVM vectorizer missing or invalid: {SVM_VECTORIZER_PATH}")

        if MBERT_MODEL_PATH.exists():
            required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
            for fname in required_files:
                if not (MBERT_MODEL_PATH / fname).exists():
                    issues.append(f"mBERT missing {fname}")

        if issues:
            return False, "\n".join(f"  ✗ {issue}" for issue in issues)

        return True, "✓ All validated model artifacts are present"

    @classmethod
    def run(cls) -> bool:
        print("=" * 60)
        print("🤖 Model Downloader")
        print("=" * 60)

        try:
            cls.ensure_models_dir()

            svm_ok = cls.download_svm_models()
            mbert_ok = cls.download_mbert_model()
            is_valid, msg = cls.validate_models()
            print(msg)

            if is_valid:
                print("\n✅ Required models are ready!")
                if not mbert_ok:
                    print("  → mBERT was intentionally skipped to reduce memory usage")
                return True

            print("\n⚠ Some models are missing or invalid")
            if not svm_ok:
                print("  → SVM models must be available for the backend to function")
            return False
        except Exception as exc:
            print(f"\n❌ Error: {exc}")
            return False


if __name__ == "__main__":
    success = ModelDownloader.run()
    sys.exit(0 if success else 1)
