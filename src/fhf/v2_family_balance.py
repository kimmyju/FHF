"""Train-only inverse-frequency provider weights for a controlled family-CE ablation.

Full-coverage training visits each example once per epoch; consequently
model-balanced input does not imply provider-balanced supervision.
"""
from __future__ import annotations

import math
from collections import Counter

from .labels import FAMILY_NAMES, MODEL_TO_FAMILY


def family_class_weights(model_labels, power=1.0):
    """Return weights in FAMILY_NAMES order and an auditable training-only report.

    w_f proportional to (N / (K * count_f)) ** power, renormalized so
    sum_f(count_f * w_f) / N == 1.0. Does not read validation labels.
    """
    if isinstance(power, bool) or not isinstance(power, (float, int)) or not math.isfinite(power) or not 0 <= power <= 1:
        raise ValueError("family balance power must be within [0,1]")
    models = list(model_labels)
    if not models:
        raise ValueError("Empty training set")
    bad = sorted(set(models)-set(MODEL_TO_FAMILY))
    if bad:
        raise ValueError(f"Unknown exact-model labels: {bad}")
    counts = Counter(MODEL_TO_FAMILY[m] for m in models)
    absent = [f for f in FAMILY_NAMES if counts[f] == 0]
    if absent:
        raise ValueError(f"Cannot balance absent providers: {absent}")
    n, k = len(models), len(FAMILY_NAMES)
    unnormalized = [(n/(k*counts[f]))**float(power) for f in FAMILY_NAMES]
    mean = sum(counts[f]*w for f,w in zip(FAMILY_NAMES,unnormalized))/n
    weights = [float(w/mean) for w in unnormalized]
    if not all(math.isfinite(w) and w>0 for w in weights):
        raise RuntimeError("Invalid provider weights")
    report = {
        "enabled": True,
        "method": "inverse_frequency_train_only",
        "power": float(power),
        "n_train": int(n),
        "family_names": list(FAMILY_NAMES),
        "counts": {f:int(counts[f]) for f in FAMILY_NAMES},
        "weights": {f:float(w) for f,w in zip(FAMILY_NAMES, weights)},
        "train_sample_weight_mean": float(sum(counts[f]*w for f,w in zip(FAMILY_NAMES,weights))/n),
        "normalization": "Training-row weighted mean of class weights equals 1.",
        "loss_reduction": "sum(weight[family] * per_row_cross_entropy) / batch_size",
        "note": "Only auxiliary family CE is reweighted. Exact/flat/contrastive losses and sampler are unchanged.",
    }
    return weights, report
