import os
import re
from pathlib import Path

import joblib
import lime.lime_text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Multilingual Fake News Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_XAI_DIR = BASE_DIR / "outputs" / "xai"
DATASET_PATH = BASE_DIR / "dataset" / "unified_cleaned_dataset.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression_tfidf.pkl",
    "Linear SVM (Calibrated)": "linear_svc_calibrated_tfidf.pkl",
    "Naive Bayes": "naive_bayes_tfidf.pkl",
    "Passive Aggressive": "passive_aggressive_tfidf.pkl",
}

MODEL_DESCRIPTIONS = {
    "Logistic Regression": "Strong baseline for sparse TF-IDF text features with probabilistic outputs.",
    "Linear SVM (Calibrated)": "Best-performing model in training with calibrated confidence estimates.",
    "Naive Bayes": "Fast and lightweight probabilistic classifier, useful for quick inference.",
    "Passive Aggressive": "Online large-margin classifier optimized for speed and streaming-style updates.",
}


def clean_text_multilingual(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^0-9a-zA-Z\u0900-\u097F\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language(text: str) -> str:
    return "Hindi" if any("\u0900" <= ch <= "\u097F" for ch in text) else "English / Other"


@st.cache_resource
def load_assets():
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    models = {
        name: joblib.load(MODELS_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }
    return vectorizer, models


@st.cache_data(show_spinner=False)
def build_evaluation_artifacts():
    vectorizer, models = load_assets()
    df = pd.read_csv(DATASET_PATH).dropna(subset=["clean_text"])

    X = df["clean_text"]
    y = df["label"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_test_tfidf = vectorizer.transform(X_test)

    rows = []
    confusion_data = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        confusion_data[name] = cm
        rows.append(
            {
                "Model": name,
                "Accuracy": round(float(acc), 4),
                "Accuracy %": round(float(acc) * 100, 2),
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return comparison_df, confusion_data


def infer_with_confidence(model, vectorizer, cleaned_text: str):
    vector = vectorizer.transform([cleaned_text])
    pred = int(model.predict(vector)[0])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vector)[0]
        confidence = float(probs[pred])
        prob_fake = float(probs[1]) if len(probs) > 1 else float(pred)
    else:
        # Fallback confidence estimate for models without predict_proba.
        score = float(model.decision_function(vector)[0])
        prob_fake = 1.0 / (1.0 + np.exp(-score))
        confidence = prob_fake if pred == 1 else (1.0 - prob_fake)

    return pred, confidence, prob_fake


def lime_predict_fn(texts, model, vectorizer):
    matrix = vectorizer.transform(texts)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(matrix)

    scores = model.decision_function(matrix)
    probs_fake = 1.0 / (1.0 + np.exp(-scores))
    probs_fake = np.clip(probs_fake, 1e-6, 1 - 1e-6)
    return np.column_stack([1 - probs_fake, probs_fake])


def render_lime_bar_plot(explanation, predicted_class_idx: int):
    labels = explanation.available_labels()
    target_class = predicted_class_idx if predicted_class_idx in labels else labels[0]
    feature_weights = explanation.as_list(label=target_class)

    if not feature_weights:
        st.info("LIME explanation words are not available for this input.")
        return

    features = [fw[0] for fw in feature_weights[:10]]
    weights = [fw[1] for fw in feature_weights[:10]]
    colors = ["#2ecc71" if w < 0 else "#e74c3c" for w in weights]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, weights, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("LIME Contribution (positive -> Fake, negative -> Real)")
    ax.set_title("Top 10 Word-Level Contributions")
    ax.grid(axis="x", alpha=0.25)
    st.pyplot(fig, use_container_width=True)


st.title("Multilingual Fake News Detection Dashboard")
st.caption("Interactive deployment for real-time fake news detection with explainability.")

vectorizer, models = load_assets()

with st.sidebar:
    st.header("Project Info")
    st.write("**Team:** Maulishka's Projects")
    st.write("**Datasets:** HFDND, IFND, LIAR")

    selected_model_name = st.selectbox("Choose Classifier", list(MODEL_FILES.keys()))
    st.info(MODEL_DESCRIPTIONS[selected_model_name])

    st.markdown("---")
    st.write("**Quick Notes**")
    st.write("- Supports Hindi and English text")
    st.write("- TF-IDF vectorization + trained models")
    st.write("- LIME enabled for probability-capable models")


tab_predict, tab_shap, tab_compare = st.tabs(
    ["Live Prediction", "SHAP Global Insights", "Model Comparison"]
)

with tab_predict:
    st.subheader("Real-time News Analysis")
    user_text = st.text_area(
        "Paste news content",
        height=220,
        placeholder="Paste Hindi or English news text here for fake/real detection...",
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        analyze = st.button("Analyze News", type="primary", use_container_width=True)
    with c2:
        detected_language = detect_language(user_text)
        st.write(f"**Detected Script/Language:** {detected_language}")

    if analyze:
        if not user_text.strip():
            st.warning("Please enter news text before running analysis.")
        else:
            cleaned = clean_text_multilingual(user_text)
            if not cleaned:
                st.error("Input became empty after cleaning. Please provide meaningful text.")
            else:
                model = models[selected_model_name]
                pred, confidence, prob_fake = infer_with_confidence(model, vectorizer, cleaned)

                label_text = "FAKE NEWS" if pred == 1 else "REAL NEWS"
                confidence_pct = int(round(confidence * 100))

                if pred == 1:
                    st.error(f"🚨 {label_text}")
                else:
                    st.success(f"✅ {label_text}")

                m1, m2, m3 = st.columns(3)
                m1.metric("Model", selected_model_name)
                m2.metric("Model Confidence Score", f"{confidence_pct}%")
                m3.metric("Fake Probability", f"{prob_fake * 100:.2f}%")
                st.progress(min(max(confidence, 0.0), 1.0))

                with st.expander("🔍 See Word-level Explanation (LIME)"):
                    if selected_model_name == "Passive Aggressive":
                        st.info("LIME explanation LR aur SVM ke liye available hai. PAC model ke liye skip kiya gaya hai.")
                    else:
                        explainer = lime.lime_text.LimeTextExplainer(class_names=["Real", "Fake"])
                        explanation = explainer.explain_instance(
                            cleaned,
                            lambda txts: lime_predict_fn(txts, model, vectorizer),
                            num_features=10,
                            num_samples=300,
                        )
                        render_lime_bar_plot(explanation, predicted_class_idx=pred)

with tab_shap:
    st.subheader("SHAP Global Feature Importance")
    st.write("These plots are generated in Notebook 05 and loaded here for deployment view.")

    shap_images = [
        OUTPUTS_XAI_DIR / "01_shap_bar_logistic_regression.png",
        OUTPUTS_XAI_DIR / "02_shap_dot_logistic_regression.png",
        OUTPUTS_XAI_DIR / "03_shap_bar_linear_svm.png",
        OUTPUTS_XAI_DIR / "04_shap_dot_linear_svm.png",
        OUTPUTS_XAI_DIR / "06_shap_hindi_vs_english_comparison.png",
    ]

    for img in shap_images:
        if img.exists():
            st.image(str(img), caption=img.name, use_container_width=True)
        else:
            st.warning(f"Missing file: {img.name}")

with tab_compare:
    st.subheader("Model Performance Analytics")
    comparison_df, confusion_data = build_evaluation_artifacts()

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    fig_acc, ax_acc = plt.subplots(figsize=(8, 4))
    ax_acc.bar(comparison_df["Model"], comparison_df["Accuracy %"], color="#1f77b4")
    ax_acc.set_ylim(0, 100)
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy Comparison Across 4 Models")
    ax_acc.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig_acc, use_container_width=True)

    st.markdown("### Confusion Matrix")
    selected_cm_model = st.selectbox("Choose model for confusion matrix", list(confusion_data.keys()))
    cm = confusion_data[selected_cm_model]

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax_cm,
        xticklabels=["Pred Real", "Pred Fake"],
        yticklabels=["True Real", "True Fake"],
    )
    ax_cm.set_title(f"Confusion Matrix - {selected_cm_model}")
    st.pyplot(fig_cm)

    comparison_plot = OUTPUTS_XAI_DIR / "05_feature_importance_comparison_lr_vs_svm.png"
    if comparison_plot.exists():
        st.markdown("### Static XAI Comparison Visual")
        st.image(str(comparison_plot), caption="LR vs SVM Feature Importance", use_container_width=True)

st.markdown("---")
st.caption("Run locally with: streamlit run app.py")
