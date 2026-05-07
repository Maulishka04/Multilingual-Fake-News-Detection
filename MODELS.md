# Model Information

## Available Models

### SVM (TF-IDF)

| Property | Value |
|----------|-------|
| **Type** | Calibrated LinearSVC |
| **Features** | TF-IDF (max 20,000 features, unigrams + bigrams) |
| **Accuracy** | ~85% |
| **F1-Score** | ~0.81 |
| **Inference Time** | ~100 ms (CPU) |
| **Explainability** | LIME word-level explanations |
| **Best For** | Speed + Explainability |
| **Languages** | English, Hindi |

**Artifacts:**
- `fake_news_backend/models/svm/linear_svc_calibrated_tfidf.pkl`
- `fake_news_backend/models/svm/tfidf_vectorizer.pkl`

---

### mBERT (bert-base-multilingual-cased, fine-tuned)

| Property | Value |
|----------|-------|
| **Base Model** | `bert-base-multilingual-cased` (Hugging Face) |
| **Training Samples** | 65,570 |
| **Test Samples** | 16,393 |
| **Total Dataset** | 81,963 samples |
| **Accuracy** | **91.15%** |
| **F1-Score** | **0.8790** |
| **Training Epochs** | 2 |
| **Batch Size** | 32 (GPU) |
| **Learning Rate** | 2e-5 |
| **Max Sequence Length** | 128 tokens |
| **Inference Time** | ~500 ms (CPU), ~50 ms (GPU) |
| **Explainability** | None (black-box) |
| **Best For** | Accuracy + Contextual understanding |
| **Languages** | Supports 104 languages |

**Artifacts (gitignored):**
- `fake_news_backend/models/mbert/config.json`
- `fake_news_backend/models/mbert/pytorch_model.bin`
- `fake_news_backend/models/mbert/tokenizer.json`
- `fake_news_backend/models/mbert/tokenizer_config.json`

---

## Comparison

| Model | Accuracy | F1-Score | Inference | Explainability |
|-------|----------|----------|-----------|----------------|
| **SVM (TF-IDF)** | ~85% | ~0.81 | Fast (~100 ms) | ✅ LIME |
| **mBERT** | **91.15%** | **0.8790** | Slower (~500 ms) | ❌ None |

### When to use SVM
- Real-time applications requiring low latency
- When word-level explanation is needed for transparency
- Environments without GPU or heavy ML libraries

### When to use mBERT
- Highest accuracy is the priority
- Complex, context-rich multilingual texts
- Batch processing where latency is less critical
