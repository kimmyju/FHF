"""Leakage-conscious, ID-aligned probability fusion diagnostics for 18 LLM classes.

This module is inference/evaluation only. No model training or oracle labels
are used when making fused predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from .labels import MODEL_NAMES, MODEL_TO_ID
from .metrics import report


PROB_COLUMNS = [f"prob_{name}" for name in MODEL_NAMES]
DEFAULT_BERT_WEIGHTS = (1.0, 0.8, 0.7, 0.5, 0.0)


def validate_probability_matrix(values, name):
    """Require finite, nonnegative, normalized 18-class probabilities."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(MODEL_NAMES):
        raise ValueError(f"{name} must have shape [N, {len(MODEL_NAMES)}]")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} has non-finite probabilities")
    if (matrix < -1e-7).any() or (matrix > 1 + 1e-7).any():
        raise ValueError(f"{name} has probabilities outside [0, 1]")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0, atol=1e-3):
        raise ValueError(f"{name} probabilities must sum to one per row")
    return matrix


def align_style_to_validation(validation, style_predictions):
    """Align rows by id, checking label, prompt, 18-class order and uniqueness."""
    required_val = {"id", "prompt_id", "model"}
    required_style = {"id", "prompt_id", "true_model", *PROB_COLUMNS}
    if not required_val.issubset(validation.columns):
        raise ValueError(f"Validation split missing {sorted(required_val - set(validation.columns))}")
    if not required_style.issubset(style_predictions.columns):
        raise ValueError(
            f"Style predictions missing {sorted(required_style - set(style_predictions.columns))}"
        )
    valid = validation.copy()
    style = style_predictions.copy()
    for table, name in ((valid, "validation"), (style, "style")):
        if table["id"].isna().any() or table["prompt_id"].isna().any():
            raise ValueError(f"{name} has missing id or prompt_id")
        table["id"] = table["id"].astype(str)
        table["prompt_id"] = table["prompt_id"].astype(str)
        if table["id"].duplicated().any():
            raise ValueError(f"{name} contains duplicate IDs")
    if set(valid["id"]) != set(style["id"]):
        missing = len(set(valid["id"]) - set(style["id"]))
        extra = len(set(style["id"]) - set(valid["id"]))
        raise ValueError(f"Validation and style sample IDs differ: {missing} missing, {extra} extra")
    aligned = style.set_index("id").loc[valid["id"]].reset_index()
    if not np.array_equal(
        aligned["true_model"].to_numpy(), valid["model"].to_numpy()
    ):
        raise ValueError("Style ground-truth labels do not match validation labels")
    if not np.array_equal(
        aligned["prompt_id"].to_numpy(), valid["prompt_id"].to_numpy()
    ):
        raise ValueError("Style prompt_id does not match validation prompt_id")
    matrix = validate_probability_matrix(
        aligned[PROB_COLUMNS].to_numpy(dtype=np.float64), "Style"
    )
    if "pred_model" in aligned:
        expected = np.asarray(MODEL_NAMES)[matrix.argmax(axis=1)]
        if not np.array_equal(aligned["pred_model"].to_numpy(), expected):
            raise ValueError("Style pred_model disagrees with its 18-class probabilities")
    return valid, matrix


def evaluate_fusion(validation, bert_probabilities, style_predictions,
                    bert_weights=DEFAULT_BERT_WEIGHTS):
    """Compare base models, error complementarity and a fixed fusion grid.

    The oracle ensemble is a diagnostic upper bound: it uses true labels to
    choose whichever base prediction is right, unlike usable fusion scores.
    """
    valid, style = align_style_to_validation(validation, style_predictions)
    bert = validate_probability_matrix(bert_probabilities, "BERT")
    if len(bert) != len(valid):
        raise ValueError("BERT probability rows must match validation row count")
    if not len(valid):
        raise ValueError("Evaluation requires at least one validation sample")
    if not set(valid.model).issubset(MODEL_TO_ID):
        raise ValueError("Validation includes unknown model labels")

    true = valid.model.map(MODEL_TO_ID).to_numpy(dtype=int)
    pred_bert = bert.argmax(axis=1)
    pred_style = style.argmax(axis=1)
    bert_correct = pred_bert == true
    style_correct = pred_style == true
    either_correct = bert_correct | style_correct

    # This uses the ground truth for analysis only. Never export it as an inference model.
    oracle_pred = np.where(bert_correct, pred_bert, pred_style)
    base_bert = report(true, pred_bert)
    base_style = report(true, pred_style)

    predictions = pd.DataFrame({
        "id": valid.id.to_numpy(),
        "prompt_id": valid.prompt_id.to_numpy(),
        "true_model": valid.model.to_numpy(),
        "pred_bert": np.asarray(MODEL_NAMES)[pred_bert],
        "pred_style": np.asarray(MODEL_NAMES)[pred_style],
        "bert_correct": bert_correct,
        "style_correct": style_correct,
    })

    weights = [float(w) for w in bert_weights]
    if not weights or any(not np.isfinite(w) or not 0 <= w <= 1 for w in weights):
        raise ValueError("BERT fusion weights must lie in [0, 1]")
    rows = []
    for weight in weights:
        prob = weight * bert + (1.0 - weight) * style
        pred = prob.argmax(axis=1)
        score = report(true, pred)
        rows.append({
            "bert_weight": weight,
            "style_weight": 1.0 - weight,
            "accuracy": score["accuracy"],
            "balanced_accuracy": score["balanced_accuracy_observed_classes"],
            "different_from_bert": int((pred != pred_bert).sum()),
        })
        label = f"pred_fusion_bert_{weight:.2f}".replace(".", "p")
        predictions[label] = np.asarray(MODEL_NAMES)[pred]

    summary = {
        "n": len(valid),
        "bert": {
            "accuracy": base_bert["accuracy"],
            "balanced_accuracy": base_bert["balanced_accuracy_observed_classes"],
            "recall_by_model": base_bert["recall_by_model"],
        },
        "style": {
            "accuracy": base_style["accuracy"],
            "balanced_accuracy": base_style["balanced_accuracy_observed_classes"],
            "recall_by_model": base_style["recall_by_model"],
        },
        "prediction_agreement": float(np.mean(pred_bert == pred_style)),
        "both_correct": int((bert_correct & style_correct).sum()),
        "bert_only_correct": int((bert_correct & ~style_correct).sum()),
        "style_only_correct": int((~bert_correct & style_correct).sum()),
        "both_wrong": int((~either_correct).sum()),
        "both_wrong_rate": float(np.mean(~either_correct)),
        "oracle_ensemble_accuracy": float(np.mean(either_correct)),
        "oracle_ensemble_balanced_accuracy": float(
            balanced_accuracy_score(true, oracle_pred)
        ),
        "fusion_grid": rows,
        "best_grid_weight_on_this_validation_only": max(
            rows, key=lambda x: x["balanced_accuracy"]
        )["bert_weight"],
        "note": (
            "Oracle ensemble uses ground-truth labels and is not deployable. "
            "Selecting a fusion weight on this validation set is exploratory; "
            "confirm any gain on a separate held-out set."
        ),
    }
    return summary, predictions, pd.DataFrame(rows)
