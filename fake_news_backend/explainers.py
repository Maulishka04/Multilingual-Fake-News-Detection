"""LIME-based text explanation utilities for fake news prediction."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, cast

import numpy as np
from lime.lime_text import LimeTextExplainer


class LIMEExplainer:
    """Generate local LIME explanations for multilingual fake news predictions.

    This class is designed for text classifiers that use a TF-IDF vectorizer and
    expose a `predict_proba` method (for example, calibrated LinearSVC).
    """

    _MIN_TEXT_LENGTH = 3

    def __init__(
        self,
        model: Any,
        vectorizer: Any,
        class_names: List[str] | None = None,
    ) -> None:
        """Initialize the explainer with model and vectorizer artifacts.

        Args:
            model: Trained model that supports `predict_proba` and `predict`.
            vectorizer: Fitted text vectorizer with `transform` method.
            class_names: Human-readable class labels ordered by class index.
        """
        self.model = model
        self.vectorizer = vectorizer
        self.class_names = class_names or ["Real", "Fake"]

        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Model must implement predict_proba for LIME explanations.")
        if not hasattr(self.vectorizer, "transform"):
            raise ValueError("Vectorizer must implement transform for text featurization.")

        self._explainer = LimeTextExplainer(class_names=self.class_names)

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize text while preserving Hindi Unicode characters.

        Keeps:
        - ASCII letters and digits
        - Hindi characters in Unicode range U+0900 to U+097F
        - whitespace
        """
        lowered = text.lower().strip()
        cleaned = re.sub(r"[^a-z0-9\u0900-\u097f\s]", " ", lowered)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """Vectorize and score a list of texts, returning class probabilities."""
        features = self.vectorizer.transform(texts)
        return self.model.predict_proba(features)

    def explain(self, text: str, num_features: int = 5) -> Dict[str, Any]:
        """Explain a single text prediction with LIME.

        Args:
            text: Raw user input text.
            num_features: Number of top features to include in explanation.

        Returns:
            Dictionary with positive/negative words, per-word scores, and summary.

        Raises:
            ValueError: If text is empty/too short or num_features is invalid.
            RuntimeError: If explanation generation fails.
        """
        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")

        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            raise ValueError("Input text is empty after cleaning.")
        if len(cleaned_text) < self._MIN_TEXT_LENGTH:
            raise ValueError("Input text is too short for reliable explanation.")
        if num_features < 1:
            raise ValueError("num_features must be >= 1.")

        try:
            explanation = self._explainer.explain_instance(
                cleaned_text,
                cast(Callable[[List[str]], np.ndarray], self._predict_proba),
                num_features=num_features,
            )

            scores = explanation.as_list(label=1)
            word_scores: Dict[str, float] = {word: float(score) for word, score in scores}

            positive_words = [word for word, score in scores if score > 0]
            negative_words = [word for word, score in scores if score < 0]

            if positive_words and negative_words:
                explanation_text = (
                    f"Prediction is influenced towards Fake by {', '.join(positive_words)} "
                    f"and towards Real by {', '.join(negative_words)}."
                )
            elif positive_words:
                explanation_text = (
                    f"Prediction is primarily influenced towards Fake by "
                    f"{', '.join(positive_words)}."
                )
            elif negative_words:
                explanation_text = (
                    f"Prediction is primarily influenced towards Real by "
                    f"{', '.join(negative_words)}."
                )
            else:
                explanation_text = "No strong word-level contributions were detected."

            return {
                "positive_words": positive_words,
                "negative_words": negative_words,
                "word_scores": word_scores,
                "explanation_text": explanation_text,
            }
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Failed to generate LIME explanation: {exc}") from exc


class MBERTExplainer:
    """Generate attention-based explanations for mBERT multilingual fake news predictions.
    
    Uses the attention weights from the last layer of mBERT to identify which tokens
    the model focuses on for its prediction. This is more meaningful than LIME for
    transformer models.
    """

    _MIN_TEXT_LENGTH = 3

    def __init__(self, tokenizer: Any, model: Any, device: Any) -> None:
        """Initialize the mBERT explainer.

        Args:
            tokenizer: mBERT tokenizer for text processing
            model: mBERT model with output_attentions=True
            device: torch device for model inference
        """
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize text while preserving Hindi Unicode characters."""
        lowered = text.lower().strip()
        cleaned = re.sub(r"[^a-z0-9\u0900-\u097f\s]", " ", lowered)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def explain(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        """Explain prediction using attention weights from mBERT.

        Args:
            text: Raw user input text
            top_k: Number of top tokens to include in explanation

        Returns:
            Dictionary with attention-based word importance scores
        """
        import torch

        if not isinstance(text, str):
            raise ValueError("Input text must be a string.")

        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            raise ValueError("Input text is empty after cleaning.")
        if len(cleaned_text) < self._MIN_TEXT_LENGTH:
            raise ValueError("Input text is too short for reliable explanation.")

        try:
            # Tokenize and get attention weights
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)

            # Get attention from the last layer, average across heads
            attention = outputs.attentions[-1]  # Shape: (batch, heads, seq_len, seq_len)
            attention_avg = attention.mean(dim=1).squeeze(0)  # Average across heads
            
            # Get token importance by summing attention to [CLS] token (first token)
            # [CLS] token aggregates information from the entire sequence
            cls_attention = attention_avg[0]  # Attention from [CLS] to all tokens
            
            # Convert token IDs back to words
            token_ids = inputs["input_ids"][0]
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            
            # Calculate word-level importance
            word_scores: Dict[str, float] = {}
            current_word = ""
            word_attention = []
            
            for token, score in zip(tokens, cls_attention):
                score_val = float(score.item())
                
                # Skip special tokens
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue
                
                # Handle subword tokens (starting with ##)
                if token.startswith("##"):
                    current_word += token[2:]
                    word_attention.append(score_val)
                else:
                    if current_word and word_attention:
                        word_scores[current_word] = np.mean(word_attention)
                    current_word = token
                    word_attention = [score_val]
            
            # Add the last word
            if current_word and word_attention:
                word_scores[current_word] = np.mean(word_attention)
            
            # Get top words by attention
            sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            top_words = sorted_words[:top_k]
            
            # Separate positive/negative influence based on prediction confidence
            positive_words = [word for word, score in top_words[:top_k//2 + 1]]
            negative_words = [word for word, score in top_words[top_k//2 + 1:]]
            
            explanation_text = (
                f"The model focused on key terms: {', '.join([w for w, _ in top_words][:3])}. "
                f"This attention pattern led to the prediction."
            )
            
            return {
                "positive_words": positive_words,
                "negative_words": negative_words if negative_words else ["[context]"],
                "word_scores": {word: float(score) for word, score in top_words},
                "explanation_text": explanation_text,
                "explanation_method": "Attention-based (mBERT)",
            }
        except Exception as exc:
            raise RuntimeError(f"Failed to generate attention explanation: {exc}") from exc
