"""Balanced accuracy, family-level and oracle within-family accuracy, confusion analysis."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score, accuracy_score
from .labels import MODEL_NAMES, MODEL_FAMILY_IDS, FAMILY_NAMES


def report(y_true, y_pred):
    """Return robust JSON-ready metrics, including absent-class-aware per-label recalls."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        raise ValueError('Cannot evaluate an empty dataset')
    labels = list(range(len(MODEL_NAMES)))
    observed = sorted(set(y_true.tolist()))
    fam_true = np.array(MODEL_FAMILY_IDS)[y_true]
    fam_pred = np.array(MODEL_FAMILY_IDS)[y_pred]
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    fam_mask = fam_true == fam_pred
    return {
        'n': int(len(y_true)),
        'balanced_accuracy_observed_classes': float(balanced_accuracy_score(y_true, y_pred)),
        'balanced_accuracy_18_classes_zero_for_missing': float(recalls.mean()),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'family_accuracy': float(np.mean(fam_mask)),
        'within_family_accuracy_given_correct_family': float(np.mean(y_true[fam_mask]==y_pred[fam_mask])) if fam_mask.any() else None,
        'within_family_joint_accuracy': float(np.mean(y_true==y_pred)),
        'observed_classes': [MODEL_NAMES[i] for i in observed],
        'recall_by_model': dict(zip(MODEL_NAMES, [float(v) for v in recalls])),
        'confusion_matrix': confusion_matrix(y_true,y_pred,labels=labels).tolist(),
    }
