"""Fit class logit-bias calibration on official TRAIN out-of-fold predictions only.

The calibration bias is a post-hoc diagnostic; validation labels are never used
to fit it. Neural-heavy ensembles need a neural OOF prediction matrix.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

from .labels import MODEL_NAMES, MODEL_TO_ID
from .fusion import validate_probability_matrix
from .v2_ensemble import (
    read_lexical_oof, select_lexical_weights, combine,
)


def _neural_oof_matrix(frame, lexical_oof, train):
    required = {"id", "prompt_id", "true_model"}
    columns = [f"neural_prob_{model}" for model in MODEL_NAMES]
    if not required.issubset(frame) or not set(columns).issubset(frame):
        raise ValueError("Neural OOF requires id, prompt_id, true_model and all 18 probabilities")
    if frame.id.astype(str).duplicated().any():
        raise ValueError("Duplicate neural OOF IDs")
    if set(frame.id.astype(str)) != set(train.id.astype(str)):
        raise ValueError("Neural OOF must cover exactly official TRAIN IDs")
    aligned = frame.assign(id=frame.id.astype(str)).set_index("id").loc[
        lexical_oof.id.astype(str)]
    if not np.array_equal(aligned.true_model.to_numpy(), lexical_oof.true_model.to_numpy()):
        raise ValueError("Neural OOF true_model does not match lexical OOF")
    if not np.array_equal(aligned.prompt_id.astype(str).to_numpy(),
                          lexical_oof.prompt_id.astype(str).to_numpy()):
        raise ValueError("Neural OOF prompt IDs do not match lexical OOF")
    return validate_probability_matrix(
        aligned[columns].to_numpy(dtype=float), "neural OOF")


def build_oof(train, lexical_oof, neural_oof=None, neural_weight=0.7):
    if not 0 <= neural_weight <= 1:
        raise ValueError("neural_weight outside [0, 1]")
    labels, lexical = read_lexical_oof(lexical_oof, train)
    if neural_weight > 0 and neural_oof is None:
        raise ValueError(
            "Neural-heavy calibration requires a matching five-fold neural OOF. "
            "Generate it with scripts/fingerprint_v2_neural_oof.py; "
            "or use --neural-weight 0 for a lexical-only diagnostic.")
    if neural_oof is not None:
        neural = [_neural_oof_matrix(neural_oof, lexical_oof, train)]
    else:
        neural = []
    weights, lexical_score = select_lexical_weights(lexical_oof, train)
    probabilities = combine(neural, lexical, weights, neural_weight)
    return labels, probabilities, weights, lexical_score


def apply_bias(probabilities, biases):
    """Add trained log-bias to each class while preserving probability mass."""
    probabilities = validate_probability_matrix(probabilities, "precalibration")
    biases = np.asarray(biases, dtype=np.float64)
    if biases.shape != (len(MODEL_NAMES),) or not np.isfinite(biases).all():
        raise ValueError("Bias must contain 18 finite values in competition order")
    log_scores = np.log(np.clip(probabilities, 1e-12, 1)) + biases[None, :]
    adjusted = np.exp(log_scores - logsumexp(log_scores, axis=1, keepdims=True))
    return validate_probability_matrix(adjusted, "postcalibration")


def fit_bias(probabilities, labels, regularization=0.5, max_abs=1.5):
    """Balanced-class log-loss with L2-regularized, bounded per-class offsets.

    This estimates balanced class intercepts, not temperatures or new encoder
    weights. All classes must appear in the TRAIN OOF fitting subset.
    """
    p = validate_probability_matrix(probabilities, "fit probabilities")
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (len(p),) or (y < 0).any() or (y >= len(MODEL_NAMES)).any():
        raise ValueError("Invalid labels")
    counts = np.bincount(y, minlength=len(MODEL_NAMES))
    if (counts == 0).any():
        raise ValueError("All 18 classes must occur when learning calibration biases")
    if regularization <= 0 or max_abs <= 0:
        raise ValueError("Regularization and max_abs must be positive")
    weights = 1.0 / (len(MODEL_NAMES) * counts[y])
    original = np.log(np.clip(p, 1e-12, 1.0))
    nclass = len(MODEL_NAMES)

    def loss_and_grad(raw):
        centered = raw - raw.mean()
        score = original + centered[None, :]
        lse = logsumexp(score, axis=1)
        logp = score[np.arange(len(y)), y] - lse
        probs = np.exp(score - lse[:, None])
        nll = -(weights * logp).sum()
        gradient = (weights[:, None] * probs).sum(axis=0)
        np.add.at(gradient, y, -weights)
        gradient -= gradient.mean()
        loss = nll + 0.5 * regularization * centered @ centered
        gradient += regularization * centered
        return float(loss), gradient.astype(float)

    result = minimize(loss_and_grad, np.zeros(nclass, dtype=np.float64),
                      jac=True, method="L-BFGS-B",
                      bounds=[(-max_abs, max_abs)] * nclass,
                      options={"maxiter": 500, "ftol": 1e-10})
    if not result.success:
        raise RuntimeError(f"Calibration optimizer failed: {result.message}")
    center = result.x - result.x.mean()
    return center


def calibrate_oof_cv(probabilities, labels, prompt_ids, folds=5,
                     regularization=0.5, max_abs=1.5):
    """Train calibration on 4/5 OOF and evaluate on unseen 1/5 prompt groups.

    This avoids reporting the in-sample accuracy of fitted biases as a
    validation score. It is a calibration-only grouped CV estimate.
    """
    y = np.asarray(labels, dtype=int)
    groups = np.asarray(prompt_ids).astype(str)
    p = validate_probability_matrix(probabilities, "OOF")
    if len(y) != len(groups) or len(y) != len(p):
        raise ValueError("OOF labels, groups and probabilities must align")
    splitter = GroupKFold(n_splits=folds)
    recalibrated = np.full(p.shape, np.nan)
    for fit_idx, eval_idx in splitter.split(p, y, groups):
        if set(groups[fit_idx]) & set(groups[eval_idx]):
            raise RuntimeError("Calibration folds share prompts")
        bias = fit_bias(p[fit_idx], y[fit_idx], regularization, max_abs)
        recalibrated[eval_idx] = apply_bias(p[eval_idx], bias)
    if not np.isfinite(recalibrated).all():
        raise RuntimeError("Calibration CV did not cover every official TRAIN row")
    return {
        "base_bacc": float(balanced_accuracy_score(y, p.argmax(axis=1))),
        "crossfit_calibrated_bacc": float(
            balanced_accuracy_score(y, recalibrated.argmax(axis=1))),
        "n": int(len(y)),
        "folds": folds,
    }


def fit_and_save(train, lexical_oof, neural_oof, neural_weight, destination,
                 regularization=0.5, max_abs=1.5, folds=5):
    labels, probs, lexical_weights, lexical_bacc = build_oof(
        train, lexical_oof, neural_oof, neural_weight)
    prompt_groups = lexical_oof.prompt_id.astype(str).to_numpy()
    results = calibrate_oof_cv(
        probs, labels, prompt_groups, folds, regularization, max_abs)
    biases = fit_bias(probs, labels, regularization, max_abs)
    document = {
        "schema": "fhf-v2-class-log-bias-1",
        "class_order": MODEL_NAMES,
        "biases": [float(x) for x in biases],
        "neural_weight": float(neural_weight),
        "lexical_weights": lexical_weights,
        "lexical_oof_bacc": lexical_bacc,
        "regularization": regularization,
        "max_abs": max_abs,
        "train_oof_crossfit": results,
        "source": "official-train-OOF only; no frozen validation labels",
        "warning": "Biases cannot be transferred to ensembles with different component weights.",
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document


def load_bias(path):
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    if content.get("schema") != "fhf-v2-class-log-bias-1":
        raise ValueError("Unsupported calibration JSON schema")
    if content.get("class_order") != MODEL_NAMES:
        raise ValueError("Calibration class ordering differs from model ordering")
    return content
