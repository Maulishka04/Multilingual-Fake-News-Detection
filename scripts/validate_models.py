#!/usr/bin/env python3
"""Validate that required model artifacts are present and loadable.

Usage:
    python scripts/validate_models.py [--backend-dir PATH]

Exit codes:
    0  All required artifacts are valid.
    1  One or more artifacts are missing or invalid.
"""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
DEFAULT_BACKEND_DIR = REPO_ROOT / "fake_news_backend"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate fake news model artifacts")
    p.add_argument(
        "--backend-dir",
        type=Path,
        default=DEFAULT_BACKEND_DIR,
        help=f"Path to the backend directory (default: {DEFAULT_BACKEND_DIR})",
    )
    return p.parse_args()


def check_pickle(path: Path, label: str) -> bool:
    """Return True if the pickle file exists and loads without error."""
    print(f"  Checking {label}: {path}")
    if not path.exists():
        print(f"    ❌ File not found")
        return False
    try:
        import pickle
        with path.open("rb") as f:
            obj = pickle.load(f)
        size_mb = path.stat().st_size / 1_048_576
        print(f"    ✅ Loaded {type(obj).__name__} ({size_mb:.2f} MB)")
        return True
    except Exception as exc:
        print(f"    ❌ Failed to load: {exc}")
        return False


def check_mbert_dir(mbert_dir: Path) -> bool:
    """Return True if the mBERT directory contains required files."""
    print(f"  Checking mBERT directory: {mbert_dir}")
    if not mbert_dir.exists():
        print(f"    ⚠️  Directory not found — mBERT will fall back to HuggingFace Hub download")
        return True  # Not a hard failure; backend handles the fallback

    config_file = mbert_dir / "config.json"
    if not config_file.exists():
        print(f"    ⚠️  config.json missing — directory is incomplete")
        return False

    size_mb = sum(f.stat().st_size for f in mbert_dir.rglob("*") if f.is_file()) / 1_048_576
    print(f"    ✅ config.json present | Total size: {size_mb:.1f} MB")
    return True


def main() -> None:
    args = parse_args()
    models_dir = args.backend_dir / "models"

    print("=" * 60)
    print("  Model Artifact Validation")
    print("=" * 60)

    all_ok = True

    # SVM — check both structured and legacy paths
    svm_dir = models_dir / "svm"
    svm_model = svm_dir / "linear_svc_calibrated_tfidf.pkl"
    svm_vectorizer = svm_dir / "tfidf_vectorizer.pkl"
    legacy_model = models_dir / "linear_svc_calibrated_tfidf.pkl"
    legacy_vectorizer = models_dir / "tfidf_vectorizer.pkl"

    print("\n📦 SVM Artifacts")
    if svm_model.exists() and svm_vectorizer.exists():
        all_ok &= check_pickle(svm_model, "SVM model (structured)")
        all_ok &= check_pickle(svm_vectorizer, "TF-IDF vectorizer (structured)")
    elif legacy_model.exists() and legacy_vectorizer.exists():
        print("  (Using legacy flat model paths)")
        all_ok &= check_pickle(legacy_model, "SVM model (legacy)")
        all_ok &= check_pickle(legacy_vectorizer, "TF-IDF vectorizer (legacy)")
    else:
        print("  ⚠️  No SVM artifacts found. Run scripts/retrain_models.py --model svm")
        all_ok = False

    # mBERT
    print("\n🤖 mBERT Artifacts")
    mbert_dir = models_dir / "mbert"
    mbert_ok = check_mbert_dir(mbert_dir)
    if not mbert_ok:
        all_ok = False

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ Validation passed — all required artifacts are present.")
    else:
        print("❌ Validation failed — see messages above for details.")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
