#!/usr/bin/env bash
# setup_models.sh — Automated model setup script for Multilingual Fake News Detection
#
# Usage:
#   bash scripts/setup_models.sh [--hf-repo REPO_ID] [--hf-token TOKEN]
#
# This script:
#   1. Creates the required model directories.
#   2. Installs backend Python dependencies.
#   3. Downloads the mBERT model from Hugging Face (if a repo ID is provided).
#   4. Validates that the model artifacts are present.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/fake_news_backend"
MBERT_DIR="$BACKEND_DIR/models/mbert"
SVM_DIR="$BACKEND_DIR/models/svm"

HF_REPO="${HF_REPO:-}"
HF_TOKEN="${HF_TOKEN:-}"

# Parse optional CLI flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-repo)  HF_REPO="$2";  shift 2 ;;
    --hf-token) HF_TOKEN="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

echo "=========================================="
echo "  Multilingual Fake News — Model Setup"
echo "=========================================="
echo "Repo root : $REPO_ROOT"
echo "Backend   : $BACKEND_DIR"
echo ""

# 1. Create directories
echo "📁 Creating model directories..."
mkdir -p "$MBERT_DIR" "$SVM_DIR"
echo "   ✅ $MBERT_DIR"
echo "   ✅ $SVM_DIR"
echo ""

# 2. Install Python dependencies
echo "📦 Installing backend dependencies..."
pip install -r "$BACKEND_DIR/requirements.txt" --quiet
echo "   ✅ Dependencies installed."
echo ""

# 3. Download mBERT model
if [[ -n "$HF_REPO" ]]; then
  echo "⬇️  Downloading mBERT model from Hugging Face: $HF_REPO"
  DOWNLOAD_ARGS="--repo-id $HF_REPO --output-dir $MBERT_DIR"
  if [[ -n "$HF_TOKEN" ]]; then
    DOWNLOAD_ARGS="$DOWNLOAD_ARGS --token $HF_TOKEN"
  fi
  # shellcheck disable=SC2086
  python "$REPO_ROOT/scripts/download_mbert_models.py" $DOWNLOAD_ARGS
  echo ""
else
  echo "ℹ️  No --hf-repo provided. Skipping mBERT download."
  echo "   Set HF_REPO or pass --hf-repo REPO_ID to download the fine-tuned weights."
  echo "   Without local weights the backend will fall back to 'bert-base-multilingual-cased'."
  echo ""
fi

# 4. Validate
echo "🔍 Validating model artifacts..."
python "$REPO_ROOT/scripts/validate_models.py"

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo "Start the backend with:"
echo "  cd fake_news_backend && uvicorn main:app --reload"
