#!/usr/bin/env python3
"""Batch retraining script for SVM and/or mBERT models.

Usage:
    python scripts/retrain_models.py \
        --dataset path/to/dataset.csv \
        [--model svm] [--model mbert] \
        [--output-dir fake_news_backend/models]

When --model is omitted, both models are retrained.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
BACKEND_DIR = REPO_ROOT / "fake_news_backend"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain one or both fake-news detection models")
    p.add_argument("--dataset", required=True, type=Path, help="Path to CSV dataset")
    p.add_argument(
        "--model",
        choices=["svm", "mbert"],
        action="append",
        dest="models",
        help="Which model to retrain (may be specified multiple times; default: both)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=BACKEND_DIR / "models",
        help="Root output directory for model artifacts",
    )
    p.add_argument("--mbert-epochs", type=int, default=2)
    p.add_argument("--mbert-batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def retrain_svm(dataset: Path, output_dir: Path, seed: int) -> None:
    """Retrain the SVM + TF-IDF pipeline and save artifacts."""
    print("\n" + "=" * 60)
    print("🔁 Retraining SVM (TF-IDF + Calibrated LinearSVC)")
    print("=" * 60)

    try:
        import numpy as np
        import pandas as pd
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.svm import LinearSVC
        import pickle
        import re
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        sys.exit(1)

    def clean_text(text: str) -> str:
        lowered = str(text).lower().strip()
        cleaned = re.sub(r"[^a-z0-9\u0900-\u097f\s]", " ", lowered)
        return re.sub(r"\s+", " ", cleaned).strip()

    df = pd.read_csv(dataset).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=seed, stratify=df["label"]
    )

    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    base_clf = LinearSVC(max_iter=2000, random_state=seed)
    model = CalibratedClassifierCV(base_clf, cv=3)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"\n✅ Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["Real", "Fake"]))

    svm_dir = output_dir / "svm"
    svm_dir.mkdir(parents=True, exist_ok=True)
    model_path = svm_dir / "linear_svc_calibrated_tfidf.pkl"
    vectorizer_path = svm_dir / "tfidf_vectorizer.pkl"

    with model_path.open("wb") as f:
        pickle.dump(model, f)
    with vectorizer_path.open("wb") as f:
        pickle.dump(vectorizer, f)

    print(f"💾 SVM model saved to   : {model_path}")
    print(f"💾 Vectorizer saved to  : {vectorizer_path}")


def retrain_mbert(dataset: Path, output_dir: Path, epochs: int, batch_size: int, seed: int) -> None:
    print("\n" + "=" * 60)
    print("🔁 Retraining mBERT")
    print("=" * 60)

    mbert_output = output_dir / "mbert"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_mbert_local.py"),
        "--dataset", str(dataset),
        "--output-dir", str(mbert_output),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--seed", str(seed),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    models_to_train = args.models or ["svm", "mbert"]

    if not args.dataset.exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        sys.exit(1)

    print(f"📊 Dataset    : {args.dataset}")
    print(f"📁 Output dir : {args.output_dir}")
    print(f"🤖 Models     : {', '.join(models_to_train)}")

    if "svm" in models_to_train:
        retrain_svm(args.dataset, args.output_dir, args.seed)

    if "mbert" in models_to_train:
        retrain_mbert(args.dataset, args.output_dir, args.mbert_epochs, args.mbert_batch_size, args.seed)

    print("\n🎉 Retraining complete!")


if __name__ == "__main__":
    main()
