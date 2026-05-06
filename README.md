# 🔍 Multilingual Fake News Detection

![Status](https://img.shields.io/badge/Status-In%20Progress-orange)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-81%2C964%20samples-brightgreen)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset%20Uploaded-blue)

---

## 🎯 Objective

Combat misinformation by building a **robust multilingual fake news detection system** that supports **Hindi** 🇮🇳 and **English** 🌍 languages. This project combines curated datasets, intelligent preprocessing, and interpretable ML/DL models to classify news articles as genuine or fabricated with explainability.

**Mission**: Enable informed decision-making across multilingual communities by providing transparent, trustworthy AI-powered fact-checking.

---

## 📊 Dataset Information

### 📁 Data Sources

| Dataset | Language | Samples | Size | Focus |
|---------|----------|---------|------|-------|
| **HFDND** | Hindi 🇮🇳 | ~28k | 28.31 MB | Hindi Fake News Detection |
| **IFND** | Mixed (Hindi/English) | ~11k | 11.13 MB | Indian Fake News Detection |
| **LIAR** | English 🌍 | ~13k | 2.88 MB | Politician Statements Veracity |

### 📈 Unified Dataset Statistics

- **Total Samples**: 81,964 ✅
- **Cleaned Dataset**: `unified_cleaned_dataset.csv` (33.45 MB)
- **Languages**: Hindi, English
- **Class Balance**: Real (0) vs Fake (1)
- **Train/Test Split**: 80/20
- **Random State**: 42

### 🏷️ Label Mapping
- **0** = Real News ✅
- **1** = Fake News ⚠️

### 🌐 Language Distribution
- Hindi articles with Devanagari script preservation
- English articles with comprehensive text coverage
- Mixed-language datasets for robustness testing

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   RAW DATASETS (HFDND, IFND, LIAR)          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         DATA UNIFICATION & CLEANING (Phase 1) ✅            │
│   • Unicode normalization (NFKC)                            │
│   • Hindi character preservation                            │
│   • URL/email removal, whitespace normalization             │
│   • Language tagging (hi/en)                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              FEATURE EXTRACTION (Phase 2)                   │
│   • TF-IDF Vectorization (max_features=20,000) ✅           │
│   • DistilBERT Embeddings (🔄 In Progress)                  │
│   • Multilingual contextualized representations             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         MODEL TRAINING & EVALUATION (Phase 3) ✅            │
│   • Logistic Regression (TF-IDF) ✅                         │
│   • Linear SVM (Calibrated, TF-IDF) ✅                      │
│   • Naive Bayes (TF-IDF) ✅                                 │
│   • Passive Aggressive (TF-IDF) ✅                          │
│   • DistilBERT Fine-tuning (🔄 In Progress)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         EXPLAINABILITY & INTERPRETABILITY ✅                │
│   • SHAP (SHapley Additive exPlanations) ✅                 │
│   • LIME (Local Interpretable Model-agnostic) ✅            │
│   • Feature importance analysis ✅                          │
│   • Global & local explanations                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           API & DEPLOYMENT (Phase 4-6) ⏳                   │
│   • FastAPI Backend (Planned)                               │
│   • Streamlit Dashboard (Planned)                           │
│   • Cloud Deployment (Planned)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Multilingual Fake News Detection/
│
├── 📄 README.md                        # Project documentation
├── 📄 .gitignore                       # Git ignore rules (sensitive data)
├── 📄 requirements.txt                 # Python dependencies
├── 📄 app.py                           # Streamlit dashboard (Phase 4)
├── 📄 dataset-metadata.json            # Kaggle dataset metadata
│
├── 📂 dataset/                         # Datasets (⚠️ NOT pushed to Git)
│   ├── dataset-merged (HFDND).csv     # Hindi Fake News Dataset
│   ├── IFND.csv                       # Indian Fake News Dataset
│   ├── LIAR_train.tsv                 # LIAR training set
│   ├── LIAR_test.tsv                  # LIAR test set
│   ├── LIAR_valid.tsv                 # LIAR validation set
│   ├── README (LIAR)                  # LIAR dataset documentation
│   ├── unified_dataset.csv            # Combined dataset
│   └── unified_cleaned_dataset.csv    # Cleaned and preprocessed (81,964 samples)
│
├── 📂 notebooks/                       # Jupyter notebooks (EDA → Training → XAI)
│   ├── 01_dataset_overview.ipynb      # Dataset exploration & statistics
│   ├── 02_dataset_analysis.ipynb      # EDA, distributions, language analysis
│   ├── 03_preprocessing.ipynb         # Data cleaning & preprocessing pipeline
│   ├── 04_model_training.ipynb        # TF-IDF training (4 models) + evaluation
│   └── 05_explainable_ai_xai.ipynb    # SHAP + LIME explainability analysis
│
├── 📂 fake_news_backend/               # Backend API services (Phase 4)
│   ├── main.py                        # FastAPI application (⏳ Under development)
│   ├── config.py                      # Configuration & constants
│   ├── explainers.py                  # LIME explainer utilities
│   ├── requirements.txt                # Backend dependencies
│   ├── check_models.py                # Model validation script
│   ├── regenerate_models.py           # Model regeneration utilities
│   └── 📂 models/                     # Model artifacts (⚠️ NOT pushed)
│       ├── tfidf_vectorizer.pkl
│       ├── logistic_regression_tfidf.pkl
│       ├── linear_svc_calibrated_tfidf.pkl
│       ├── naive_bayes_tfidf.pkl
│       └── passive_aggressive_tfidf.pkl
│
├── 📂 frontend/                        # React + Vite Frontend (Phase 5) ⏳
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── store/
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── package.json
│
├── 📂 models/                          # Trained model artifacts (⚠️ NOT pushed)
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_regression_tfidf.pkl
│   ├── linear_svc_calibrated_tfidf.pkl
│   ├── naive_bayes_tfidf.pkl
│   └── passive_aggressive_tfidf.pkl
│
├── 📂 outputs/                         # Generated artifacts
│   └── 📂 xai/                         # Explainability outputs from Notebook 05
│       ├── 01_shap_bar_logistic_regression.png
│       ├── 02_shap_dot_logistic_regression.png
│       ├── 03_shap_bar_linear_svm.png
│       ├── 04_shap_dot_linear_svm.png
│       ├── 05_feature_importance_comparison_lr_vs_svm.png
│       ├── 06_shap_hindi_vs_english_comparison.png
│       ├── 07_lime_explanation_*.html
│       ├── 08_shap_vs_lime_comparison.csv
│       └── XAI_ANALYSIS_SUMMARY_REPORT.txt
│
├── 📂 scripts/                         # Utility scripts (reserved for future use)
│
└── 📂 news-detective/                  # Additional frontend resources (Phase 5)
    ├── index.html
    └── assets/
```

---

## ⚙️ Installation Instructions

### 🔧 Prerequisites

- **Python** 3.8 or higher
- **pip** (Python package manager)
- **Virtual Environment** (recommended)

### 📥 Setup Steps

#### 1️⃣ Clone the Repository
```bash
git clone <repository-url>
cd "Multilingual Fake News Detection"
```

#### 2️⃣ Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

#### 3️⃣ Install Dependencies
```bash
# Install main requirements
pip install -r requirements.txt

# Optional: Install backend requirements (Phase 4)
pip install -r fake_news_backend/requirements.txt

# Optional: Install frontend requirements (Phase 5)
cd frontend
npm install
```

### 📦 Key Dependencies

**Data & ML Stack:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Classical ML models
- `nltk`, `spacy` - NLP utilities

**Deep Learning (🔄 In Progress):**
- `transformers` - Hugging Face models (DistilBERT)
- `torch` - PyTorch backend

**Explainability:**
- `shap` - SHAP explanations
- `lime` - LIME local explanations

**Deployment & Visualization:**
- `streamlit` - Interactive dashboard
- `fastapi` - Backend API (Planned)
- `matplotlib`, `seaborn` - Visualization

---

## 🔄 Usage Guide

### 📊 1. Exploratory Data Analysis (EDA)

Explore dataset characteristics and distributions:
```bash
jupyter notebook notebooks/01_dataset_overview.ipynb
jupyter notebook notebooks/02_dataset_analysis.ipynb
```

**Outputs:**
- Dataset statistics (size, balance, language distribution)
- Word clouds, frequency analysis
- Overlap detection across sources

---

### 🧹 2. Data Preprocessing

Execute the multilingual-safe preprocessing pipeline:

```bash
# Run via Notebook
jupyter notebook notebooks/03_preprocessing.ipynb
```

**What it does:**
- ✅ Unicode NFKC normalization
- ✅ Hindi character preservation
- ✅ URL/email/special character removal
- ✅ Whitespace normalization
- ✅ Language tagging

**Output:** `unified_cleaned_dataset.csv` (81,964 samples)

---

### 🤖 3. Model Training (Phase 3) ✅

Train TF-IDF based models:

```bash
# Run training notebook
jupyter notebook notebooks/04_model_training.ipynb
```

**Models Trained:**
1. **Logistic Regression** ✅
2. **Linear SVM (Calibrated)** ✅
3. **Naive Bayes** ✅
4. **Passive Aggressive** ✅

**Features:**
- TF-IDF Vectorization (max_features=20,000)
- 80/20 train-test split (random_state=42)
- Performance metrics: Accuracy, Precision, Recall, F1
- Confusion matrices for each model

**Output:** Trained models saved to `models/` directory

---

### 🔍 4. Explainability Analysis (Phase 3) ✅

Generate SHAP and LIME explanations:

```bash
jupyter notebook notebooks/05_explainable_ai_xai.ipynb
```

**Explainability Techniques:**

| Technique | Type | Output |
|-----------|------|--------|
| **SHAP (Bar)** | Global | Feature importance rankings |
| **SHAP (Dot)** | Global | Feature impact distributions |
| **LIME** | Local | Word-level contributions |
| **Comparison** | Both | LR vs SVM analysis |

**Outputs:** PNG visualizations + CSV reports in `outputs/xai/`

---

### 🎯 5. Live Prediction Dashboard (Phase 4) ⏳

**Work in Progress** - Coming Soon! 

Run the Streamlit dashboard:
```bash
streamlit run app.py
```

**Features (Planned):**
- 🎤 Real-time multilingual input (Hindi + English)
- 🔮 Fake/Real predictions with confidence scores
- 🌐 Language auto-detection
- 📊 Model selection (LR, SVM, NB, PA)
- 💡 LIME word-level explanations
- 📈 Model comparison analytics
- 🔄 Confusion matrix visualization

---

### 🔌 6. Backend API (Phase 4) ⏳

**Work in Progress** - Under Development!

Start the FastAPI server:
```bash
cd fake_news_backend
python main.py
```

**API Endpoints (Planned):**
- `POST /predict` - Basic prediction
- `POST /predict-with-lime` - Prediction + explanation
- `GET /health` - Health check

---

## 📈 Current Progress & Status

### 🎯 Project Phases

```
✅ Phase 1: Data Preparation
   └─ Status: COMPLETE
      • Dataset unification (HFDND + IFND + LIAR)
      • Data cleaning & preprocessing (81,964 samples)
      • Language tagging (Hindi/English)

✅ Phase 2: Kaggle Dataset Upload
   └─ Status: COMPLETE
      • Dataset published to Kaggle
      • Metadata configured
      • Public access enabled

🔄 Phase 3: Model Training & Explainability
   └─ Status: IN PROGRESS (~15-20 minutes)
      ✅ TF-IDF based models (LR, SVM, NB, PA)
      ✅ SHAP global explanations
      ✅ LIME local explanations
      🔄 DistilBERT embeddings (fine-tuning)
      🔄 Model comparison analysis

⏳ Phase 4: API Development
   └─ Status: PLANNED
      • FastAPI backend setup
      • LIME integration for explanations
      • Real-time prediction endpoints
      • Health checks & error handling

⏳ Phase 5: Frontend Integration
   └─ Status: PLANNED
      • React + Vite frontend
      • Streamlit dashboard
      • Real-time UI updates
      • Multi-language support

⏳ Phase 6: Deployment
   └─ Status: PLANNED
      • Docker containerization
      • Cloud deployment (AWS/Azure/GCP)
      • CI/CD pipeline setup
      • Monitoring & logging
```

---

## 🚀 Key Features

### ✨ Highlights

- **🌍 Multilingual Support**: Hindi (Devanagari) + English
- **📊 Large-Scale Dataset**: 81,964 samples from 3 authoritative sources
- **🧠 Multiple Models**: Classical ML (LR, SVM, NB, PA) + Deep Learning (DistilBERT 🔄)
- **💡 Interpretability**: SHAP + LIME explanations for model transparency
- **⚡ Production-Ready**: TF-IDF models deployed in Streamlit
- **🔐 Security**: Sensitive data excluded from Git repository
- **📚 Well-Documented**: Comprehensive notebooks + API documentation

---

## ⚠️ Work in Progress Sections

### 🔄 Currently Under Development

1. **DistilBERT Model Training**
   - Status: In Progress
   - Current Approach: Using DistilBERT for efficient multilingual embeddings
   - Note: May upgrade to full mBERT or XLM-RoBERTa based on performance

2. **API Development (FastAPI)**
   - Status: Planned for Phase 4
   - Features: Real-time predictions, LIME explanations, health checks
   - Timeline: After model training completion

3. **Frontend Dashboard**
   - Status: Planned for Phase 5
   - Technology: React + Vite + Tailwind CSS
   - Features: Interactive UI, real-time predictions, history tracking

4. **Cloud Deployment**
   - Status: Planned for Phase 6
   - Target: AWS/Azure/GCP
   - CI/CD: GitHub Actions / GitLab CI

---

## 📚 References & Resources

### 📖 Datasets
- **HFDND**: [Hindi Fake News Detection Dataset](https://www.kaggle.com/datasets/techgurukulofficial/hindi-fake-news-detection)
- **IFND**: [Indian Fake News Detection Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-news)
- **LIAR**: [LIAR: A Benchmark Dataset for Fake News Detection](https://www.cs.ucsb.edu/~william/papers/liar_liar_pants_on_fire.pdf)

### 🔍 Model Documentation
- **Scikit-learn**: [Classic ML Models](https://scikit-learn.org/)
- **DistilBERT**: [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- **TF-IDF**: [Text Vectorization Guide](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

### 💡 Explainability
- **SHAP**: [SHapley Additive exPlanations](https://github.com/slundberg/shap)
- **LIME**: [Local Interpretable Model-agnostic Explanations](https://github.com/marcotcr/lime)

### 🛠️ Tools & Technologies
- **Streamlit**: [Interactive Dashboard Framework](https://streamlit.io/)
- **FastAPI**: [Modern API Framework](https://fastapi.tiangolo.com/)
- **Kaggle**: [Dataset Platform](https://www.kaggle.com/)

---

## 🔐 Security & Best Practices

### 🚫 Never Commit

- ❌ Raw datasets (`.csv`, `.tsv`, `.parquet`)
- ❌ Model artifacts (`.pkl`, `.pth`, `.bin`)
- ❌ API keys, credentials, secrets
- ❌ `.env` files with sensitive information
- ❌ Generated outputs unless necessary

### ✅ Code-Only Repository Policy

This repository maintains a **code-only policy** to ensure:
- 🔒 Security (no sensitive data exposure)
- 📦 Lightweight repository (easy cloning)
- ☁️ Scalability (datasets in cloud storage)

**Large files** are referenced via documentation or external storage (Kaggle, S3, etc.)

See [.gitignore](.gitignore) for complete exclusion rules.

---

## 📝 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

You are free to:
- ✅ Use for personal/commercial projects
- ✅ Modify and distribute
- ✅ Use in private applications

---

## 👤 Author

**Maulishka's Projects**

- 📧 Contact: [Your Email]
- 🔗 GitHub: [Your GitHub Profile]
- 📊 Kaggle: [Your Kaggle Profile]

---

## 🙏 Acknowledgments

- **Dataset Contributors**: HFDND, IFND, LIAR dataset creators
- **Framework Credits**: Scikit-learn, Transformers, Streamlit, FastAPI teams
- **Open Source**: SHAP, LIME, and all dependencies

---

## ❓ FAQ

### Q: How do I run the project locally?
**A:** Follow the [Installation Instructions](#⚙️-installation-instructions) section. Start with the Jupyter notebooks (Phase 1-3), then explore the Streamlit dashboard (Phase 4).

### Q: Can I use this for non-English languages?
**A:** Currently supports Hindi and English. The preprocessing pipeline is extensible for other Indic scripts (Gujarati, Bengali, etc.). Contributions welcome!

### Q: How accurate are the models?
**A:** Refer to [notebooks/04_model_training.ipynb](notebooks/04_model_training.ipynb) for detailed metrics. Linear SVM typically performs best with TF-IDF features. DistilBERT results coming soon (Phase 3 🔄).

### Q: Where are the trained models?
**A:** Models are saved in `models/` directory (not pushed to Git). After running `notebooks/04_model_training.ipynb`, models will be available locally.

### Q: How do I deploy this to production?
**A:** Phase 4-6 cover API development and cloud deployment. Stay tuned for updates!

---

## 📞 Support & Contributing

Have questions or want to contribute? Please:
1. 📝 Open an issue with details
2. 🔀 Submit pull requests for improvements
3. 💬 Discuss in project discussions

---

**Last Updated**: May 2026  
**Version**: 1.0 (Beta)

⭐ If you find this project useful, please give it a star! ⭐

