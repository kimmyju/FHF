"""Provider specialists, character- and word-level lexical fingerprints.

Fits ONLY the official training split. OOF predictions are GroupKFold by
prompt_id and never include the frozen official validation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .labels import MODEL_NAMES, FAMILY_NAMES, MODEL_TO_FAMILY


def _pipeline(kind, seed, max_features):
    if kind not in ("char", "word"):
        raise ValueError(f"Unknown lexical fingerprint type: {kind}")
    vectorizer = TfidfVectorizer(
        analyzer="char" if kind == "char" else "word",
        ngram_range=(3, 5) if kind == "char" else (1, 2),
        max_features=max_features,
        min_df=2,
        sublinear_tf=True,
        lowercase=False,
        dtype=np.float32,
    )
    return Pipeline([
        ("vectorizer", vectorizer),
        ("model", LogisticRegression(
            C=3.0, max_iter=1200, random_state=seed,
            class_weight=None)),
    ])


def aligned_proba(estimator, text, labels):
    scores = estimator.predict_proba(list(text))
    output = np.zeros((len(text), len(labels)), dtype=np.float64)
    positions = {name: i for i, name in enumerate(labels)}
    for i, label in enumerate(estimator.classes_):
        if label not in positions:
            raise ValueError(f"Estimator produced unexpected label {label}")
        output[:, positions[label]] = scores[:, i]
    if not np.allclose(output.sum(axis=1), 1.0):
        raise ValueError("Model omitted an evaluation class")
    return output


class LexicalFingerprintEnsemble:
    def __init__(self, seed=42, char_features=180_000, word_features=100_000):
        self.seed = int(seed)
        self.char_features = int(char_features)
        self.word_features = int(word_features)

    def fit(self, frame):
        if not {"text", "model"}.issubset(frame):
            raise ValueError("Training requires text and model")
        if set(frame.model) != set(MODEL_NAMES):
            raise ValueError("Training must contain all 18 exact model labels")
        text = frame.text.astype(str).tolist()
        names = frame.model.astype(str).tolist()
        self.char = _pipeline("char", self.seed, self.char_features).fit(text, names)
        self.word = _pipeline("word", self.seed, self.word_features).fit(text, names)
        providers = [MODEL_TO_FAMILY[n] for n in names]
        self.provider = _pipeline("char", self.seed, self.char_features).fit(text, providers)
        self.specialists = {}
        for family in FAMILY_NAMES:
            subset = frame[frame.model.map(MODEL_TO_FAMILY) == family]
            classes = subset.model.unique().tolist()
            if len(classes) == 1:
                self.specialists[family] = classes[0]
            else:
                self.specialists[family] = _pipeline(
                    "char", self.seed, min(self.char_features, 100_000)
                ).fit(subset.text.astype(str).tolist(), subset.model.astype(str).tolist())
        return self

    def predict_probabilities(self, texts):
        text = [str(t) for t in texts]
        char = aligned_proba(self.char, text, MODEL_NAMES)
        word = aligned_proba(self.word, text, MODEL_NAMES)
        provider = aligned_proba(self.provider, text, FAMILY_NAMES)
        conditional = np.zeros_like(char)
        for family_idx, family in enumerate(FAMILY_NAMES):
            model_names = [name for name in MODEL_NAMES if MODEL_TO_FAMILY[name] == family]
            subset = self.specialists[family]
            if isinstance(subset, str):
                local = np.ones((len(text), 1))
            else:
                local = aligned_proba(subset, text, model_names)
            for j, name in enumerate(model_names):
                conditional[:, MODEL_NAMES.index(name)] = provider[:, family_idx] * local[:, j]
        if not np.allclose(conditional.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError("Provider-specialist probabilities do not sum to one")
        return {"char": char, "word": word, "specialist": conditional}
