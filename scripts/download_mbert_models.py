#!/usr/bin/env python3
"""Download the fine-tuned mBERT model from Hugging Face Hub.

Usage:
    python scripts/download_mbert_models.py [--repo-id REPO_ID] [--output-dir DIR]

The script downloads the model into ``fake_news_backend/models/mbert/`` by
default, which is the path expected by the backend at startup.

Prerequisites:
    pip install huggingface_hub
"""

import argparse
import sys
from pathlib import Path


DEFAULT_REPO_ID = "bert-base-multilingual-cased"  # Replace with your fine-tuned model repo
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "fake_news_backend" / "models" / "mbert"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download mBERT model weights from Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repo ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local directory to save the model (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token (required for private repos)",
    )
    return parser.parse_args()


def download(repo_id: str, output_dir: Path, token: str | None = None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{repo_id}' → {output_dir}")

    local_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(output_dir),
        token=token,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"✅ Model downloaded to: {local_path}")

    # Quick sanity check
    config_file = output_dir / "config.json"
    if config_file.exists():
        print("✅ config.json found — model directory looks valid.")
    else:
        print("⚠️  config.json not found. The download may be incomplete.")


def main() -> None:
    args = parse_args()
    download(repo_id=args.repo_id, output_dir=args.output_dir, token=args.token)


if __name__ == "__main__":
    main()
