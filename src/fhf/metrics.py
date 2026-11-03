"""Balanced accuracy and conditioned vs oracle within-family attribution metrics."""
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, accuracy_score
from .labels import MODEL_NAMES, MODEL_FAMILY_IDS, FAMILY_NAMES


def oracle_within_family_predictions(model_logits, true_families, family_lookup):
    """Select the top-scoring model *within the ground-truth family*.

    Unlike normal 18-way predictions, this function never uses the Family
    Head's predicted family. It is a diagnostic only: true family labels
    are not available for Kaggle test inference.
    """
    if model_logits.ndim != 2 or model_logits.shape[1] != len(MODEL_NAMES):
        raise ValueError("Expected model_logits shape [batch, 18]")
    families = torch.as_tensor(true_families, device=model_logits.device, dtype=torch.long)
    lookup = torch.as_tensor(family_lookup, device=model_logits.device, dtype=torch.long)
    if families.ndim != 1 or families.numel() != model_logits.shape[0]:
        raise ValueError("true_families must have one ID per sample")
    if lookup.ndim != 1 or lookup.numel() != model_logits.shape[1]:
        raise ValueError("family_lookup must map every exact-model class")
    allowed = lookup.unsqueeze(0).eq(families.unsqueeze(1))
    if not bool(allowed.any(dim=1).all()):
        raise ValueError("Some true family IDs have no corresponding model class")
    return model_logits.masked_fill(~allowed, float("-inf")).argmax(dim=1)


def report(y_true, y_pred, oracle_predictions=None):
    """Return standard 18-class scores and optional teacher-forced diagnostics."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        raise ValueError('Cannot evaluate an empty dataset')
    if len(y_pred) != len(y_true):
        raise ValueError('Predictions and targets must have equal length')
    labels = list(range(len(MODEL_NAMES)))
    observed = sorted(set(y_true.tolist()))
    fam_true = np.array(MODEL_FAMILY_IDS)[y_true]
    fam_pred = np.array(MODEL_FAMILY_IDS)[y_pred]
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    fam_mask = fam_true == fam_pred
    result = {
        'n': int(len(y_true)),
        'balanced_accuracy_observed_classes': float(balanced_accuracy_score(y_true, y_pred)),
        'balanced_accuracy_18_classes_zero_for_missing': float(recalls.mean()),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'family_accuracy': float(np.mean(fam_mask)),
        'within_family_accuracy_given_correct_family': (
            float(np.mean(y_true[fam_mask] == y_pred[fam_mask])) if fam_mask.any() else None
        ),
        'within_family_joint_accuracy': float(np.mean(y_true == y_pred)),
        'observed_classes': [MODEL_NAMES[i] for i in observed],
        'recall_by_model': dict(zip(MODEL_NAMES, [float(v) for v in recalls])),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if oracle_predictions is not None:
        oracle = np.asarray(oracle_predictions, dtype=int)
        if oracle.shape != y_true.shape:
            raise ValueError('Oracle predictions and targets must have equal length')
        if not np.array_equal(np.asarray(MODEL_FAMILY_IDS)[oracle], fam_true):
            raise ValueError('Oracle predictions must stay within each true family')
        result['oracle_within_family_accuracy'] = float(np.mean(oracle == y_true))
        result['oracle_within_family_accuracy_by_family'] = {
            name: float(np.mean(oracle[fam_true == i] == y_true[fam_true == i]))
            for i, name in enumerate(FAMILY_NAMES) if np.any(fam_true == i)
        }
    return result
