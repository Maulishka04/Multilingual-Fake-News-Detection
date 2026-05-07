#!/usr/bin/env python3
"""Fine-tune mBERT locally on the unified fake news dataset.

Usage:
    python scripts/train_mbert_local.py \
        --dataset path/to/fake_news_dataset.csv \
        --output-dir fake_news_backend/models/mbert \
        [--epochs 2] [--batch-size 16] [--max-length 128] [--sample N]

This script:
    1. Loads the dataset CSV (must have 'text' and 'label' columns).
    2. Splits into train / test (80/20, stratified).
    3. Fine-tunes bert-base-multilingual-cased for sequence classification.
    4. Evaluates and prints accuracy, precision, recall, F1.
    5. Saves the model and tokenizer to --output-dir.

Requirements:
    pip install transformers torch datasets scikit-learn pandas
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune mBERT for fake news detection")
    p.add_argument("--dataset", required=True, type=Path, help="Path to CSV dataset")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fake_news_backend/models/mbert"),
        help="Directory to save the fine-tuned model",
    )
    p.add_argument("--model-name", default="bert-base-multilingual-cased")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--sample", type=int, default=None, help="Limit dataset size (for quick tests)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-fp16", action="store_true", help="Disable mixed-precision training")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Deferred heavy imports so --help works without GPU packages
    try:
        import numpy as np
        import pandas as pd
        import torch
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.model_selection import train_test_split
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        print(f"Missing dependency: {exc}")
        print("Install with: pip install transformers torch datasets scikit-learn pandas")
        sys.exit(1)

    print("=" * 70)
    print("🤖 mBERT Fine-Tuning — Multilingual Fake News Detection")
    print("=" * 70)
    print(f"GPU available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name      : {torch.cuda.get_device_name(0)}")
    print()

    # Load dataset
    print(f"📊 Loading dataset: {args.dataset}")
    if not args.dataset.exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        sys.exit(1)
    df = pd.read_csv(args.dataset)
    print(f"   Loaded {len(df):,} rows")

    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=args.seed)
        print(f"   Sampled {len(df):,} rows")

    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        print(f"ERROR: Dataset must contain columns: {required_cols}")
        sys.exit(1)

    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    print(f"   Label distribution:\n{df['label'].value_counts().to_string()}")
    print()

    # Split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df["label"]
    )
    print(f"🔀 Train: {len(train_df):,} | Test: {len(test_df):,}")

    # Tokenizer
    print(f"\n🤗 Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True))
    train_ds = train_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Model
    print(f"🤗 Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds),
            "recall": recall_score(labels, preds),
            "f1": f1_score(labels, preds),
        }

    use_fp16 = torch.cuda.is_available() and not args.no_fp16
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=use_fp16,
        learning_rate=2e-5,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print(f"\n🚀 Starting training ({args.epochs} epoch(s), batch={args.batch_size}, fp16={use_fp16})…\n")
    trainer.train()

    print("\n" + "=" * 70)
    print("📊 Final Evaluation")
    print("=" * 70)
    results = trainer.evaluate()
    print(f"   Accuracy  : {results['eval_accuracy']:.4f}")
    print(f"   Precision : {results['eval_precision']:.4f}")
    print(f"   Recall    : {results['eval_recall']:.4f}")
    print(f"   F1-Score  : {results['eval_f1']:.4f}")

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"\n💾 Model saved to: {args.output_dir}")
    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
