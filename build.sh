#!/bin/bash
# Build script for Render deployment
# Handles Git LFS setup and model downloads

set -e  # Exit on error

echo "=========================================="
echo "🚀 Render Build Script"
echo "=========================================="

# Step 1: Ensure Git LFS is installed
echo "\n📥 Setting up Git LFS..."
git lfs install
echo "✓ Git LFS initialized"

# Step 2: Pull LFS objects
echo "\n📥 Pulling Git LFS objects..."
git lfs pull
echo "✓ Git LFS objects pulled"

# Step 3: Install Python dependencies
echo "\n📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
cd fake_news_backend
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Step 4: Download missing models
echo "\n📥 Downloading models..."
if python download_models.py; then
    echo "✓ Models ready"
else
    echo "⚠ Warning: Model download had issues, but continuing..."
fi

# Step 5: Verify models exist
echo "\n🔍 Verifying model artifacts..."
if [ -f "models/linear_svc_calibrated_tfidf.pkl" ] && [ -d "models/mbert_model" ]; then
    echo "✓ All critical models present"
else
    echo "⚠ Warning: Some models may be missing"
    echo "  SVM model: $([ -f models/linear_svc_calibrated_tfidf.pkl ] && echo '✓' || echo '✗')"
    echo "  mBERT model: $([ -d models/mbert_model ] && echo '✓' || echo '✗')"
fi

echo "\n=========================================="
echo "✅ Build completed successfully"
echo "=========================================="
