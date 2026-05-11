#!/usr/bin/env python3
"""
Download and validate model artifacts for the fake news detection backend.

This script handles:
- Downloading SVM model and vectorizer from GitHub LFS (if available)
- Downloading/preparing mBERT model from Hugging Face
- Validating file integrity
- Creating necessary directory structure
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Tuple
import urllib.request
import urllib.error

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import (
    MODELS_DIR,
    SVM_MODEL_PATH,
    SVM_VECTORIZER_PATH,
    MBERT_MODEL_PATH,
)


class ModelDownloader:
    """Handle downloading and validating model files."""

    # GitHub repo LFS raw URLs (update if repo URL changes)
    GITHUB_REPO_RAW = "https://raw.githubusercontent.com/Maulishka04/Multilingual-Fake-News-Detection/main"
    
    # SVM model file URLs (from GitHub LFS)
    SVM_MODEL_URL = f"{GITHUB_REPO_RAW}/models/linear_svc_calibrated_tfidf.pkl"
    SVM_VECTORIZER_URL = f"{GITHUB_REPO_RAW}/models/tfidf_vectorizer.pkl"
    
    # Hugging Face model
    MBERT_MODEL_HF = "bert-base-multilingual-cased"

    @staticmethod
    def ensure_models_dir() -> None:
        """Create models directory if it doesn't exist."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✓ Models directory ready: {MODELS_DIR}")

    @staticmethod
    def download_file(url: str, dest_path: Path, timeout: int = 30) -> bool:
        """
        Download a file from URL with progress indication.
        
        Args:
            url: Source URL
            dest_path: Destination file path
            timeout: Download timeout in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"  Downloading: {url}")
            print(f"  Destination: {dest_path}")
            
            # Create a custom URL opener with user agent
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            
            # Download with progress
            def download_with_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(downloaded * 100 / total_size, 100)
                    print(f"    Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f}MB)", end='\r')
            
            urllib.request.urlretrieve(
                url,
                dest_path,
                reporthook=download_with_progress,
                timeout=timeout
            )
            print()  # New line after progress
            print(f"  ✓ Downloaded: {dest_path.name}")
            return True
            
        except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
            print(f"  ✗ Failed to download {url}: {e}")
            return False

    @staticmethod
    def download_svm_models() -> bool:
        """
        Download SVM model and vectorizer.
        
        Returns:
            True if both files exist (downloaded or already present), False otherwise
        """
        print("\n📦 Checking SVM models...")
        
        # Check if already exist
        if SVM_MODEL_PATH.exists() and SVM_VECTORIZER_PATH.exists():
            model_size = SVM_MODEL_PATH.stat().st_size / 1024 / 1024
            vec_size = SVM_VECTORIZER_PATH.stat().st_size / 1024 / 1024
            print(f"✓ SVM model already exists ({model_size:.1f}MB)")
            print(f"✓ Vectorizer already exists ({vec_size:.1f}MB)")
            return True
        
        print("  SVM models not found. Attempting download from GitHub LFS...")
        
        # Try to download both files
        model_success = ModelDownloader.download_file(
            ModelDownloader.SVM_MODEL_URL,
            SVM_MODEL_PATH
        )
        
        vectorizer_success = ModelDownloader.download_file(
            ModelDownloader.SVM_VECTORIZER_URL,
            SVM_VECTORIZER_PATH
        )
        
        if model_success and vectorizer_success:
            print("✓ SVM models downloaded successfully")
            return True
        
        print("⚠ Could not download SVM models from GitHub LFS.")
        print("  Note: If running on Render, ensure Git LFS files are pulled during build.")
        return False

    @staticmethod
    def download_mbert_model() -> bool:
        """
        Download mBERT model from Hugging Face.
        
        Returns:
            True if model is ready, False otherwise
        """
        print("\n📦 Checking mBERT model...")
        
        if MBERT_MODEL_PATH.exists():
            # Validate that it has required files
            required_files = ['config.json', 'tokenizer.json', 'model.safetensors']
            if all((MBERT_MODEL_PATH / f).exists() for f in required_files):
                model_size = sum(
                    f.stat().st_size for f in MBERT_MODEL_PATH.rglob('*') if f.is_file()
                ) / 1024 / 1024
                print(f"✓ mBERT model already exists ({model_size:.1f}MB)")
                return True
        
        print(f"  Downloading mBERT from Hugging Face: {ModelDownloader.MBERT_MODEL_HF}")
        
        try:
            # Download tokenizer
            print("  Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                ModelDownloader.MBERT_MODEL_HF,
                cache_dir=str(MBERT_MODEL_PATH.parent)
            )
            
            # Download model
            print("  Downloading model weights...")
            model = AutoModelForSequenceClassification.from_pretrained(
                ModelDownloader.MBERT_MODEL_HF,
                num_labels=2,
                cache_dir=str(MBERT_MODEL_PATH.parent)
            )
            
            # Save locally
            print(f"  Saving to {MBERT_MODEL_PATH}...")
            MBERT_MODEL_PATH.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(MBERT_MODEL_PATH)
            model.save_pretrained(MBERT_MODEL_PATH)
            
            model_size = sum(
                f.stat().st_size for f in MBERT_MODEL_PATH.rglob('*') if f.is_file()
            ) / 1024 / 1024
            print(f"✓ mBERT model downloaded and saved ({model_size:.1f}MB)")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download mBERT model: {e}")
            return False

    @staticmethod
    def validate_models() -> Tuple[bool, str]:
        """
        Validate that all required model files exist and are accessible.
        
        Returns:
            Tuple of (is_valid, message)
        """
        print("\n🔍 Validating models...")
        
        issues = []
        
        # Check SVM model
        if not SVM_MODEL_PATH.exists():
            issues.append(f"SVM model missing: {SVM_MODEL_PATH}")
        else:
            size = SVM_MODEL_PATH.stat().st_size
            if size < 1000:  # Less than 1KB seems invalid
                issues.append(f"SVM model too small ({size} bytes)")
        
        # Check vectorizer
        if not SVM_VECTORIZER_PATH.exists():
            issues.append(f"SVM vectorizer missing: {SVM_VECTORIZER_PATH}")
        else:
            size = SVM_VECTORIZER_PATH.stat().st_size
            if size < 1000:
                issues.append(f"Vectorizer too small ({size} bytes)")
        
        # Check mBERT
        if not MBERT_MODEL_PATH.exists():
            issues.append(f"mBERT model directory missing: {MBERT_MODEL_PATH}")
        else:
            required_files = ['config.json', 'tokenizer.json']
            for fname in required_files:
                if not (MBERT_MODEL_PATH / fname).exists():
                    issues.append(f"mBERT missing {fname}")
        
        if issues:
            return False, "\n".join(f"  ✗ {issue}" for issue in issues)
        
        return True, "✓ All models validated successfully"

    @classmethod
    def run(cls) -> bool:
        """
        Run the complete model download and validation process.
        
        Returns:
            True if all models are available, False otherwise
        """
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
                print("\n✅ All models ready!")
                return True
            else:
                print("\n⚠ Some models are missing or invalid")
                if not svm_ok:
                    print("  → SVM models need to be downloaded via Git LFS")
                return False
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False


if __name__ == "__main__":
    success = ModelDownloader.run()
    sys.exit(0 if success else 1)
