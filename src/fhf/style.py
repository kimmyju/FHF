"""Fit-free, deterministic surface-style features and trainable feature fusion.

No label, validation statistic, or corpus-derived vocabulary is used. Raw
character 3-5-grams are feature hashed; 24 small stylistic counters encode
punctuation, length, lexical repetition and text structure.
"""
from __future__ import annotations

import re
import numpy as np
import torch
from torch import nn
from sklearn.feature_extraction.text import HashingVectorizer


STYLE_SCALAR_NAMES = (
    "log_char_length", "log_word_count", "log_line_count",
    "upper_ratio", "digit_ratio", "whitespace_ratio",
    "comma_per_100", "period_per_100", "semicolon_per_100",
    "colon_per_100", "exclamation_per_100", "question_per_100",
    "quote_per_100", "parenthesis_per_100", "bracket_per_100",
    "hyphen_per_100", "newline_per_100", "slash_per_100",
    "average_word_length", "type_token_ratio", "long_word_ratio",
    "repetition_ratio", "sentence_length_proxy", "list_line_ratio",
)


def _style_scalars(text):
    chars = max(len(text), 1)
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    word_count = len(words)
    lines = text.splitlines() or [""]
    lower_words = [w.casefold() for w in words]
    unique = len(set(lower_words))
    punctuation = [",", ".", ";", ":", "!", "?", "\"'", "()", "[]", "-–—", "\n", "/"]
    rates = []
    for group in punctuation:
        rates.append(sum(text.count(ch) for ch in group) * 100.0 / chars)
    sentence_count = max(1, sum(text.count(ch) for ch in ".!?"))
    list_lines = sum(bool(re.match(r"^\s*(?:[-*•]|\d+[.)])\s", line)) for line in lines)
    return [
        np.log1p(len(text)) / 10.0,
        np.log1p(word_count) / 8.0,
        np.log1p(len(lines)) / 5.0,
        sum(ch.isupper() for ch in text) / chars,
        sum(ch.isdigit() for ch in text) / chars,
        sum(ch.isspace() for ch in text) / chars,
        *rates,
        sum(len(w) for w in words) / max(word_count, 1) / 12.0,
        unique / max(word_count, 1),
        sum(len(w) >= 8 for w in words) / max(word_count, 1),
        (word_count - unique) / max(word_count, 1),
        word_count / sentence_count / 40.0,
        list_lines / max(len(lines), 1),
    ]


class StyleFeatureExtractor:
    """Hash n-grams without fitting on train/validation data."""

    def __init__(self, hash_dim=256):
        self.hash_dim = int(hash_dim)
        if self.hash_dim < 16 or self.hash_dim & (self.hash_dim - 1):
            raise ValueError("style.hash_dim must be a power of two >= 16")
        self.hasher = HashingVectorizer(
            analyzer="char", ngram_range=(3, 5),
            n_features=self.hash_dim, alternate_sign=False,
            norm="l2", lowercase=False, dtype=np.float32,
        )

    @property
    def dim(self):
        return self.hash_dim + len(STYLE_SCALAR_NAMES)

    def transform(self, texts):
        texts = [str(x) for x in texts]
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        hashed = self.hasher.transform(texts).toarray().astype(np.float32, copy=False)
        numeric = np.asarray([_style_scalars(t) for t in texts], dtype=np.float32)
        if numeric.shape[1] != len(STYLE_SCALAR_NAMES):
            raise RuntimeError("Internal style feature count mismatch")
        return np.concatenate((hashed, numeric), axis=1)


class MaskedAttentionPool(nn.Module):
    """Learn token weights while respecting padding positions."""

    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, attention_mask):
        logits = self.score(hidden.float()).squeeze(-1)
        logits = logits.masked_fill(attention_mask == 0, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1).to(hidden.dtype)
        return (hidden * weights.unsqueeze(-1)).sum(dim=1)


class GatedStyleFusion(nn.Module):
    """Learn a residual style correction without replacing pretrained semantics."""

    def __init__(self, feature_dim, hidden_size, style_hidden=128, dropout=0.1):
        super().__init__()
        self.style_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, style_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(style_hidden, hidden_size),
        )
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        # Begin with a modest style contribution; learn when it is useful.
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, contextual, style_features):
        style_features = style_features.to(dtype=contextual.dtype)
        style = self.style_encoder(style_features)
        gate = torch.sigmoid(self.gate(torch.cat((contextual, style), dim=-1)))
        return self.norm(contextual + gate * style), gate.mean()
