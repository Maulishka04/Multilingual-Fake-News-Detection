# 🚀 Model Loading Guide - Fake News Detection Backend

## Overview

This guide covers how the multilingual fake news detection backend loads models, troubleshoots loading issues, and deploys to production (Render.com).

**Quick Links:**
- [Local Development](#local-development)
- [Troubleshooting](#troubleshooting)
- [Render Deployment](#render-deployment)
- [Monitoring & Testing](#monitoring--testing)
- [Git LFS Setup](#git-lfs-setup)

---

## Local Development

### Prerequisites

```bash
# Python 3.9+
python --version

# Git (with LFS support)
git --version
git lfs version
```

### Setup (First Time)

1. **Clone repository with LFS:**
```bash
git clone https://github.com/Maulishka04/Multilingual-Fake-News-Detection.git
cd Multilingual-Fake-News-Detection
git lfs install
git lfs pull
```

2. **Install dependencies:**
```bash
cd fake_news_backend
pip install -r requirements.txt
```

3. **Download models (if LFS pull didn't work):**
```bash
python download_models.py
```

4. **Verify models are loaded:**
```bash
# Check if files exist
ls -lh models/

# Output should show:
# - linear_svc_calibrated_tfidf.pkl (> 1MB)
# - tfidf_vectorizer.pkl (> 1MB)  
# - mbert_model/ (directory with >300MB total)
```

5. **Start backend:**
```bash
python main.py
```

Expected output:
```
======================================================================
🚀 Starting Backend - Loading Models
======================================================================

📥 Checking and downloading models if needed...

📦 Loading SVM model stack...
  ✅ SVM model loaded successfully
     Size: 1.2MB
  ✅ TF-IDF vectorizer loaded successfully
     Size: 0.8MB
  ✅ LIME explainer initialized

📦 Loading mBERT model stack...
  ✅ mBERT tokenizer loaded successfully
  ✅ mBERT model loaded successfully
     Size: 713.4MB
  ✅ mBERT explainer initialized

----------------------------------------------------------------------
✅ Device: cpu
✅ SVM Ready: True
✅ mBERT Ready: True
======================================================================
```

### Test Endpoints

```bash
# Check health
curl http://localhost:8000/health
# Response: {"status":"healthy","svm_loaded":true,"mbert_loaded":true}

# Get detailed status
curl http://localhost:8000/status
# Response: {"backend":"operational","device":"cpu","models":{...},"errors":null}

# Test prediction with SVM
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Breaking news! Scientists discover cure for disease",
    "language": "en",
    "model_type": "svm"
  }'

# Test prediction with mBERT
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "खबर: नई खोज की गई",
    "language": "hi",
    "model_type": "mbert"
  }'

# Get explanations
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Sample text here",
    "language": "en",
    "model_type": "svm"
  }'
```

---

## Troubleshooting

### Problem: Models Not Found

**Error message:**
```
❌ SVM model file not found: /path/to/linear_svc_calibrated_tfidf.pkl
```

**Solutions:**

1. **Download models explicitly:**
```bash
python download_models.py
```

2. **Pull Git LFS files:**
```bash
git lfs pull
git lfs install
git lfs pull
```

3. **Force download from Hugging Face (mBERT only):**
```bash
cd fake_news_backend
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
  AutoTokenizer.from_pretrained('bert-base-multilingual-cased', cache_dir='models'); \
  AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-cased', cache_dir='models')"
```

4. **Check file sizes:**
```bash
ls -lh fake_news_backend/models/
# SVM files should be > 1MB each
# mbert_model directory should be > 300MB
```

### Problem: Git LFS Pointer Files Instead of Real Files

**Symptom:**
- File size shows 100-200 bytes instead of MB
- File content starts with `version https://git-lfs.github.com/spec/v1`

**Solution:**

```bash
# Install Git LFS locally
git lfs install

# Pull LFS files
git lfs pull

# Verify
cat models/linear_svc_calibrated_tfidf.pkl | head -c 10
# Should show binary data, not "version https://..."
```

### Problem: Out of Memory When Loading mBERT

**Error:**
```
RuntimeError: CUDA out of memory or OSError: Cannot allocate memory
```

**Solutions:**

1. **Disable CUDA (use CPU):**
```bash
export USE_CUDA=false
python main.py
```

2. **Reduce batch size in config.py:**
```python
MBERT_BATCH_SIZE = 16  # Lower from 32
```

3. **Quantize model (advanced):**
```python
# In main.py, after loading model:
from transformers import AutoConfig
config = AutoConfig.from_pretrained("bert-base-multilingual-cased")
config.num_hidden_layers = 6  # Use lighter config
```

### Problem: Backend Crashes on Startup

**Diagnostic:**
```bash
# Run in verbose mode
python -c "from download_models import ModelDownloader; ModelDownloader.run()"

# Check if critical error
python main.py 2>&1 | grep "❌"
```

**Common issues:**
- Missing dependencies: `pip install -r requirements.txt`
- Python version mismatch: Use Python 3.9+
- Corrupted model files: Delete and re-download

---

## Render Deployment

### Prerequisites

- GitHub account with repository access
- Render.com account (free tier available)
- Models uploaded to GitHub via Git LFS

### Deployment Steps

#### 1. Verify Git LFS Setup

```bash
# From project root
cat .gitattributes | grep -E "pkl|mbert"

# Output should show:
# fake_news_backend/models/*.pkl filter=lfs ...
# fake_news_backend/models/mbert_model/** filter=lfs ...
```

#### 2. Ensure Models Are Uploaded

```bash
# Check that LFS objects are on GitHub
git lfs ls-files

# Output should show all model files with LFS pointers
```

#### 3. Connect Render to GitHub

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New +"
3. Select "Web Service"
4. Connect your GitHub repository
5. Select branch: `main` (or your current branch)

#### 4. Configure Build & Deploy

**Build Command:**
```bash
pip install --upgrade pip
cd fake_news_backend
pip install -r requirements.txt
python download_models.py
```

**Start Command:**
```bash
cd fake_news_backend && python main.py
```

**Environment Variables:**
```
BACKEND_HOST = 0.0.0.0
BACKEND_PORT = 8000
USE_CUDA = false
GIT_LFS_SKIP_SMUDGE = 0
```

**Plan:** Free tier (suitable for testing) or Starter/Pro (for production)

#### 5. Deploy

Click "Create Web Service" and watch the build logs.

**Expected build output:**
```
▶ Building...
  - Cloning repository...
  - Installing dependencies...
  - Downloading models...
  - Build successful ✓
▶ Starting service...
  🚀 Starting Backend - Loading Models
  ✅ Device: cpu
  ✅ SVM Ready: True
  ✅ mBERT Ready: True
```

### Monitor Deployment

After deployment, check:

```bash
# Get service URL from Render dashboard
RENDER_URL=https://your-service.render.com

# Test health endpoint
curl $RENDER_URL/health

# Get model status
curl $RENDER_URL/status

# Test prediction
curl -X POST $RENDER_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Sample","language":"en","model_type":"svm"}'
```

### Troubleshooting Render Deployment

**Issue: Models downloading very slowly**
- Normal: mBERT is ~700MB, can take 5-10 minutes on Render's free tier
- Check: `build_logs` for progress

**Issue: Service crashes after build**
```
# View live logs in Render dashboard
# Or: curl $RENDER_URL/status to get error details
```

**Issue: Git LFS files not downloaded**
- Ensure `GIT_LFS_SKIP_SMUDGE=0` in environment
- Check that `.gitattributes` is committed to repo
- Re-push changes to trigger new build

---

## Monitoring & Testing

### Health Check Endpoint

```bash
GET /health

Response:
{
  "status": "healthy",
  "svm_loaded": true,
  "mbert_loaded": true
}
```

### Detailed Status Endpoint

```bash
GET /status

Response:
{
  "backend": "operational",
  "device": "cpu",
  "models": {
    "svm": {
      "loaded": true,
      "model_file": "/path/to/linear_svc_calibrated_tfidf.pkl",
      "vectorizer_file": "/path/to/tfidf_vectorizer.pkl",
      "ready": true
    },
    "mbert": {
      "loaded": true,
      "model_path": "/path/to/mbert_model",
      "ready": true
    }
  },
  "errors": null
}
```

### Load Testing

```bash
# Using Apache Bench
ab -n 100 -c 10 http://localhost:8000/health

# Using curl in a loop
for i in {1..10}; do
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text":"Test","language":"en","model_type":"svm"}' &
done
wait
```

### Model Performance Monitoring

Monitor these metrics in production:

1. **Prediction latency:**
   - SVM: Should be < 100ms
   - mBERT: Should be 500ms - 2s (first call may be slower)

2. **Memory usage:**
   - SVM stack: ~20MB
   - mBERT stack: ~1.5GB
   - Total: ~1.7GB

3. **Error rate:**
   - Check `/status` for error field
   - 0 errors expected in normal operation

---

## Git LFS Setup

### For Contributors

If you need to modify or update models:

```bash
# 1. Initialize LFS (one-time)
git lfs install

# 2. Track model files
git lfs track "fake_news_backend/models/*.pkl"
git lfs track "fake_news_backend/models/mbert_model/**"

# 3. Add .gitattributes
git add .gitattributes

# 4. Add model files
git add fake_news_backend/models/
git commit -m "Update model artifacts"
git push origin main
```

### For CI/CD

Ensure your CI/CD pipeline:
1. Has Git LFS installed: `git lfs install`
2. Pulls LFS files before build: `git lfs pull`
3. Sets `GIT_LFS_SKIP_SMUDGE=0` if cloning fresh

---

## Quick Reference

| Task | Command |
|------|---------|
| Download models | `python download_models.py` |
| Check model status | `curl http://localhost:8000/status` |
| Run locally | `python main.py` |
| Run tests | `pytest tests/` |
| Pull Git LFS | `git lfs pull` |
| Deploy to Render | Push to GitHub → Render auto-deploys |

---

## Support & Issues

For problems:
1. Check `/status` endpoint for detailed errors
2. Review `fake_news_backend/download_models.py` logs
3. Check GitHub Issues: [Link to repo issues]
4. Review build logs on Render dashboard

---

**Last Updated:** May 2026
**Maintainer:** Maulishka04
**Backend Version:** FastAPI with PyTorch & scikit-learn
