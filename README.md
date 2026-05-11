# Multilingual Fake News Detection

A multilingual fake news detection project for Hindi and English news text. The repository combines data preprocessing, classical machine learning, transformer-based prediction, explainability, and a React frontend.

## Project Overview

The project currently includes:

- Dataset preparation and analysis notebooks
- A Python backend with FastAPI
- An interactive React + Vite frontend
- LIME-based explanations for classical models
- Attention-based explanations for mBERT
- Persistent history and a welcome screen in the frontend

## Repository Layout

- `dataset/`: local dataset files used during research and training
- `notebooks/`: dataset analysis, preprocessing, training, and explainability notebooks
- `models/`: trained model artifacts used locally during development
- `fake_news_backend/`: FastAPI backend and model loading logic
- `frontend/`: React + TypeScript user interface
- `outputs/`: generated explainability reports and charts
- `scripts/`: helper scripts for local development

## Privacy and Git Hygiene

This repository is kept code-focused. Do not commit:

- Raw datasets such as `.csv` and `.tsv` files
- Model artifacts such as `.pkl`, `.pt`, `.bin`, and large transformer folders
- Secrets, API keys, credentials, and `.env` files
- Large generated outputs unless they are needed for documentation

See [.gitignore](.gitignore) for the current exclusion rules.

## Frontend

The frontend is built with Vite, React, TypeScript, Tailwind CSS, and Zustand.

Features:

- Mobile-first responsive layout
- Welcome screen on first load
- Persistent analysis history
- Model selection between SVM and mBERT
- Debounced text input handling
- Lazy loading for heavier visualization components
- CSS variables for the current color system

### Run the frontend

```bash
cd frontend
npm install
npm run dev
```

## Backend

The backend is a FastAPI application that loads the classical model stack and the transformer stack.

Supported endpoints include:

- `GET /`
- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict-with-lime`
- `POST /predict-batch`

### Run the backend

```bash
cd fake_news_backend
python main.py
```

## Installation

### Python dependencies

```bash
pip install -r requirements.txt
pip install -r fake_news_backend/requirements.txt
```

### Frontend dependencies

```bash
cd frontend
npm install
```

## Notebooks

Recommended notebook workflow:

1. `notebooks/01_dataset_overview.ipynb`
2. `notebooks/02_dataset_analysis.ipynb`
3. `notebooks/03_preprocessing.ipynb`
4. `notebooks/04_model_training.ipynb`
5. `notebooks/05_explainable_ai_xai.ipynb`

## Current Status

- Data preparation: complete
- Classical ML training: complete
- Explainability analysis: complete
- Backend API: in progress
- Frontend integration: in progress
- Deployment: planned

## License

MIT License

## Author

Maulishka's Projects
