# Multilingual Fake News Detection

## Project Overview

This project addresses the critical challenge of detecting fake news across multiple languages, specifically focusing on **Hindi and English**. With the proliferation of misinformation in digital media, this work develops a robust machine learning and deep learning pipeline to classify news articles as genuine or fabricated, supporting informed decision-making in multilingual contexts.

The project combines multiple authoritative datasets (HFDND, IFND, and LIAR), applies rigorous preprocessing while preserving linguistic integrity, and employs both classical and transformer-based models for comprehensive fake news detection.

---

## Problem Statement

Misinformation spreads rapidly across digital platforms, creating significant societal challenges. While fake news detection has been extensively studied in English, the problem is notably underexplored in resource-limited languages like Hindi. This project tackles the following challenges:

- **Multilingual Complexity**: Handling linguistic variations and script-specific characteristics
- **Data Scarcity**: Limited availability of labeled fake news datasets in Hindi
- **Language Preservation**: Maintaining semantic integrity during text preprocessing
- **Model Interpretability**: Understanding and explaining model decisions for stakeholder trust

---

## Dataset Description

### Data Sources

| Dataset | Language | Articles | Size | Focus |
|---------|----------|----------|------|-------|
| **HFDND** | Hindi | ~28k | 28.31 MB | Hindi Fake News Detection |
| **IFND** | Mixed (Hindi/English) | ~11k | 11.13 MB | Indian Fake News Detection |
| **LIAR** | English | ~13k | 2.88 MB | Politician Statements Veracity |

### Label Mapping

- **Real News**: 0
- **Fake News**: 1

### Language Tagging

All articles are tagged with language codes:
- `hi` - Hindi
- `en` - English

### Data Unification

- **Original Unified Dataset**: `unified_dataset.csv` (33.94 MB)
- **Cleaned Dataset**: `unified_cleaned_dataset.csv` (33.45 MB)

---

## Preprocessing Pipeline

Our multilingual-safe preprocessing pipeline is designed to clean text while preserving linguistic and semantic integrity, particularly for Hindi language content.

### Preprocessing Steps

1. **Text Normalization**
   - Unicode NFKC normalization to handle various character representations
   - Consistent encoding across all text samples

2. **Hindi-Specific Handling**
   - Preservation of Hindi matras (vowel marks) to maintain word meaning
   - Script consistency for Devanagari text
   - Handling of common Hindi abbreviations and variations

3. **General Cleaning**
   - Removal of URLs, email addresses, and hyperlinks
   - Elimination of special characters while preserving punctuation semantics
   - Whitespace normalization

4. **Case and Tokenization**
   - Language-aware case conversion (preserving linguistic nuances)
   - Removal of extra whitespace and formatting artifacts

### Output

The final cleaned dataset (`unified_cleaned_dataset.csv`) is ready for feature extraction and model training.

---

## Exploratory Data Analysis

Comprehensive EDA was performed to understand dataset characteristics:

### Key Analyses

- **Class Distribution**: Balance assessment between real and fake news
- **Language Distribution**: Proportion of Hindi vs. English articles
- **Text Length Statistics**: Word count, character count, and vocabulary analysis
- **Dataset Overlap**: Identification of duplicate or similar articles across sources
- **Temporal Patterns**: Publication date trends (if available)
- **Common Keywords**: Frequency analysis and word clouds by class

### Findings

Refer to [notebooks/02_dataset_analysis.ipynb](notebooks/02_dataset_analysis.ipynb) for detailed visualizations and insights.

---

## Exploratory Data Analysis

### Outputs
- Dataset statistics and distributions
- Language-specific patterns
- Label imbalance analysis
- Data quality metrics

For detailed visualizations, see the EDA notebooks in the `notebooks/` directory.

---

## Project Structure

```
Multilingual Fake News Detection/
│
├── README.md                          # Project documentation
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies
│
├── dataset/                           # Raw and processed datasets
│   ├── dataset-merged (HFDND).csv    # Hindi Fake News Dataset
│   ├── IFND.csv                      # Indian Fake News Dataset
│   ├── LIAR_train.tsv                # LIAR training set
│   ├── LIAR_test.tsv                 # LIAR test set
│   ├── LIAR_valid.tsv                # LIAR validation set
│   ├── unified_dataset.csv           # Combined dataset
│   └── unified_cleaned_dataset.csv   # Cleaned and preprocessed dataset
│
├── notebooks/                         # Jupyter notebooks for analysis
│   ├── 01_dataset_overview.ipynb     # Dataset exploration
│   ├── 02_dataset_analysis.ipynb     # EDA and statistics
│   └── 03_preprocessing.ipynb        # Data cleaning pipeline
│
├── scripts/                           # Python scripts for processing
│   ├── preprocess.py                 # Preprocessing utilities
│   ├── train.py                      # Model training scripts
│   └── evaluate.py                   # Model evaluation utilities
│
├── models/                            # Trained models (not tracked in git)
│   ├── tfidf_lr_model.pkl            # TF-IDF + Logistic Regression
│   ├── tfidf_svm_model.pkl           # TF-IDF + SVM
│   └── transformer_models/           # Fine-tuned BERT models
│
└── outputs/                           # Analysis outputs and results
    ├── model_metrics.json            # Performance metrics
    ├── feature_importance.png        # Feature analysis plots
    └── predictions.csv               # Model predictions
```

---

## Installation Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd "Multilingual Fake News Detection"
   ```

2. **Create a Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

Key libraries include:
- **Data Processing**: pandas, numpy
- **Text Processing**: scikit-learn, nltk, spacy
- **Deep Learning**: transformers (Hugging Face), torch
- **Visualization**: matplotlib, seaborn
- **Explainability**: SHAP, LIME
- **Deployment**: flask, fastapi (planned)

For the complete list, refer to [requirements.txt](requirements.txt).

---

## How to Run

### 1. Exploratory Data Analysis

```bash
jupyter notebook notebooks/01_dataset_overview.ipynb
jupyter notebook notebooks/02_dataset_analysis.ipynb
```

### 2. Data Preprocessing

```bash
jupyter notebook notebooks/03_preprocessing.ipynb
```

Or run the preprocessing script:
```bash
python scripts/preprocess.py --input dataset/unified_dataset.csv \
                             --output dataset/unified_cleaned_dataset.csv
```

### 3. Model Training

Train TF-IDF with Logistic Regression:
```bash
python scripts/train.py --model tfidf_lr --data dataset/unified_cleaned_dataset.csv
```

Train Transformer Models (mBERT/XLM-R):
```bash
python scripts/train.py --model transformer --model-name xlm-roberta-base \
                        --data dataset/unified_cleaned_dataset.csv
```

### 4. Model Evaluation

```bash
python scripts/evaluate.py --model models/tfidf_lr_model.pkl \
                           --test-data dataset/test_set.csv
```

### 5. Generate Explanations

```bash
python scripts/explain.py --model models/transformer_models/mbert \
                          --text "Sample news article text"
```

---

## Future Work

### Phase 2: Advanced Modeling

- **TF-IDF with Classical ML**
  - Logistic Regression
  - Support Vector Machines (SVM)
  - Ensemble methods (Random Forest, Gradient Boosting)

- **Transformer Models**
  - Multilingual BERT (mBERT)
  - XLM-RoBERTa (XLM-R)
  - Language-specific fine-tuning

### Phase 3: Interpretability & Explainability

- **SHAP Values**: Global and local model explanations
- **LIME**: Local interpretable model-agnostic explanations
- **Attention Visualization**: Transformer attention mechanism analysis
- **Feature Importance**: Classical model interpretation

### Phase 4: Deployment

- **Real-time Detection API**
  - FastAPI/Flask web service
  - Docker containerization
  - Cloud deployment (AWS/Azure/GCP)

- **User Interface**
  - Web-based interface for easy testing
  - Multi-language support
  - Real-time prediction and confidence scores

### Phase 5: Extended Research

- **Multi-task Learning**: Simultaneous detection of misinformation types
- **Cross-lingual Transfer Learning**: Leveraging high-resource languages
- **Fact Verification Integration**: Combining with fact-checking databases
- **Temporal Analysis**: Detecting evolving misinformation patterns

---

## Technologies Used

### Programming & Data Science
- **Python 3.8+**: Primary programming language
- **Jupyter Notebook**: Interactive analysis and documentation
- **Pandas & NumPy**: Data manipulation and numerical computing

### Machine Learning & NLP
- **scikit-learn**: Classical ML algorithms and preprocessing
- **Hugging Face Transformers**: State-of-the-art NLP models
- **PyTorch**: Deep learning framework
- **NLTK & spaCy**: Natural language processing utilities

### Explainability & Visualization
- **SHAP**: Unified approach to explaining predictions
- **LIME**: Local interpretable model-agnostic explanations
- **Matplotlib & Seaborn**: Data visualization
- **Plotly**: Interactive visualizations

### Version Control & Documentation
- **Git**: Version control system
- **GitHub**: Repository hosting and collaboration

---

## Results & Evaluation

Model performance will be evaluated using:

- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Dataset Split**: 70% training, 15% validation, 15% testing
- **Cross-validation**: K-fold cross-validation for robustness
- **Language-specific Analysis**: Separate evaluation for Hindi and English

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact & Support

For questions, suggestions, or collaborations, please reach out:

- **Project Owner**: [Your Name]
- **Email**: [Your Email]
- **Institution**: [Your University/Organization]

---

## Acknowledgments

- **Dataset Contributors**: HFDND, IFND, and LIAR dataset creators
- **Research Community**: NLP and fake news detection researchers
- **Frameworks & Libraries**: Hugging Face, scikit-learn, PyTorch communities

---

**Last Updated**: February 19, 2026

**Status**: Active Development - Phase 1 (Data Preparation & EDA) Complete
