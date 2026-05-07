# mBERT Integration Guide

## Overview

This document covers how to set up, configure, and use the mBERT (Multilingual BERT)
model alongside the existing SVM-based fake news classifier.

---

## Architecture

```
fake_news_backend/
├── main.py                  ← Dual-model FastAPI app
├── config.py                ← mBERT paths & metadata
├── explainers.py            ← LIME explainer (SVM only)
├── models/
│   ├── mbert_inference.py   ← MBertInference class
│   ├── svm/                 ← Structured SVM artifacts
│   │   ├── linear_svc_calibrated_tfidf.pkl
│   │   └── tfidf_vectorizer.pkl
│   └── mbert/               ← Fine-tuned mBERT weights (gitignored)
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       └── tokenizer_config.json
```

---

## Quick Start

### 1. Install dependencies

```bash
cd fake_news_backend
pip install -r requirements.txt
```

### 2. Download or train the mBERT model

**Option A — Download from Hugging Face (recommended)**

```bash
python scripts/download_mbert_models.py \
  --repo-id <your-hf-repo-id> \
  --output-dir fake_news_backend/models/mbert \
  --token <your-hf-token>   # only for private repos
```

**Option B — Train locally from dataset**

```bash
python scripts/train_mbert_local.py \
  --dataset path/to/fake_news_dataset.csv \
  --output-dir fake_news_backend/models/mbert \
  --epochs 2 \
  --batch-size 16
```

**Option C — Use automated setup script**

```bash
bash scripts/setup_models.sh --hf-repo <repo-id> --hf-token <token>
```

### 3. Validate artifacts

```bash
python scripts/validate_models.py
```

### 4. Start the backend

```bash
cd fake_news_backend
uvicorn main:app --reload
```

If the mBERT model directory is absent, the backend falls back to downloading
`bert-base-multilingual-cased` from Hugging Face automatically (note: this is
the un-fine-tuned base model — accuracy will be lower).

---

## API Endpoints

See [API_GUIDE.md](API_GUIDE.md) for full documentation.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict?model=svm` | SVM prediction (default) |
| `POST` | `/predict?model=mbert` | mBERT prediction |
| `POST` | `/predict-with-lime` | SVM + LIME explanation |
| `POST` | `/compare` | Side-by-side comparison |
| `GET`  | `/available-models` | List models + status |
| `GET`  | `/model-info/{name}` | Detailed model info |

---

## Frontend Model Selection

The frontend exposes a **Model** dropdown next to the Language selector.
Select **SVM (TF-IDF)** for speed + explainability or **mBERT** for higher accuracy.

---

## Notes

- mBERT model weights are listed in `.gitignore` (too large for Git).
- Use Git LFS or a Hugging Face private repo to store and share weights.
- mBERT requires `torch` and `transformers` (already in `requirements.txt`).
- Inference time is ~500 ms on CPU vs ~100 ms for SVM.
