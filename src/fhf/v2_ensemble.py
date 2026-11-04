"""Leakage-controlled lexical weights and neural/lexical ensemble inference.

Lexical weights are selected on official TRAIN out-of-fold predictions ONLY.
The neural mixing coefficient is a fixed user-specified value, not optimized
on frozen validation. Inference uses only the supplied response text.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import MODEL_NAMES, MODEL_TO_ID
from .metrics import report
from .fusion import validate_probability_matrix


SOURCES = ("char", "word", "specialist")


def read_lexical_oof(oof, official_train):
    if oof.id.astype(str).duplicated().any():
        raise ValueError("Duplicate OOF sample IDs")
    if set(oof.id.astype(str)) != set(official_train.id.astype(str)):
        raise ValueError("OOF must cover exactly the official TRAIN IDs")
    lookup = official_train.assign(id=official_train.id.astype(str)).set_index("id")
    aligned = oof.copy()
    aligned["id"] = aligned.id.astype(str)
    if not np.array_equal(
        aligned.true_model.to_numpy(), lookup.loc[aligned.id, "model"].to_numpy()
    ):
        raise ValueError("OOF true_model labels do not match official TRAIN")
    if not np.array_equal(
        aligned.prompt_id.astype(str).to_numpy(),
        lookup.loc[aligned.id, "prompt_id"].astype(str).to_numpy()
    ):
        raise ValueError("OOF prompt IDs differ from official TRAIN")
    matrices = {}
    for source in SOURCES:
        cols = [f"{source}_prob_{model}" for model in MODEL_NAMES]
        if not set(cols).issubset(aligned):
            raise ValueError(f"OOF missing {source} probabilities")
        matrices[source] = validate_probability_matrix(
            aligned[cols].to_numpy(dtype=float), source)
    return aligned.true_model.map(MODEL_TO_ID).to_numpy(), matrices


def select_lexical_weights(train_oof, official_train, step=0.1):
    y, matrices = read_lexical_oof(train_oof, official_train)
    if not 0 < step <= 1:
        raise ValueError("step must be in (0,1]")
    steps = round(1/step)
    if not np.isclose(step*steps, 1):
        raise ValueError("step must divide 1 exactly")
    best = None
    for a in range(steps+1):
        for b in range(steps+1-a):
            c = steps-a-b
            weights = np.array([a,b,c], dtype=float)/steps
            combined = sum(w*matrices[source] for w,source in zip(weights,SOURCES))
            accuracy = report(y,combined.argmax(axis=1))["balanced_accuracy_observed_classes"]
            # deterministic tie break: keep more specialist weight.
            candidate=(accuracy,weights[2],weights[0],weights)
            if best is None or candidate[:3] > best[:3]:
                best=candidate
    return dict(zip(SOURCES,best[3].tolist())),float(best[0])


def combine(neural, lexical, lexical_weights, neural_weight=0.7):
    """Fixed neural/lexical mixture, with optional multiple neural members."""
    if not 0 <= neural_weight <= 1:
        raise ValueError("neural_weight must be in [0,1]")
    if set(lexical_weights) != set(SOURCES):
        raise ValueError("Need three lexical component weights")
    ws=np.asarray([lexical_weights[source] for source in SOURCES],dtype=float)
    if (ws<0).any() or not np.isclose(ws.sum(),1):
        raise ValueError("Lexical weights must be nonnegative and sum to 1")
    checked=[validate_probability_matrix(lexical[source],source) for source in SOURCES]
    if len({x.shape[0] for x in checked})!=1:
        raise ValueError("Lexical prediction row counts disagree")
    combined_lexical=sum(w*p for w,p in zip(ws,checked))
    if neural:
        matrices=[validate_probability_matrix(item,"neural") for item in neural]
        if any(p.shape != combined_lexical.shape for p in matrices):
            raise ValueError("Neural and lexical prediction rows/classes disagree")
        mixed = neural_weight*np.mean(np.stack(matrices),axis=0) + (
            1-neural_weight)*combined_lexical
    else:
        if neural_weight != 0:
            raise ValueError("No neural models: set neural_weight=0")
        mixed=combined_lexical
    return validate_probability_matrix(mixed,"fused")



def select_neural_weight(lexical_oof, neural_oof, official_train,
                         lexical_weights, step=0.05):
    """Choose neural-vs-lexical blend solely from aligned official TRAIN OOF.

    Score on this same OOF is optimistically tuned; do not report it as an
    unbiased estimate. Confirmation must use a separate frozen validation.
    """
    true, lexical = read_lexical_oof(lexical_oof, official_train)
    if neural_oof.id.astype(str).duplicated().any():
        raise ValueError("Duplicate neural OOF IDs")
    if set(neural_oof.id.astype(str)) != set(lexical_oof.id.astype(str)):
        raise ValueError("Neural OOF must cover the same official TRAIN IDs")
    neural = neural_oof.assign(id=neural_oof.id.astype(str)).set_index("id")
    aligned = neural.loc[lexical_oof.id.astype(str)]
    if not np.array_equal(aligned.true_model.to_numpy(),
                          lexical_oof.true_model.to_numpy()):
        raise ValueError("Neural/lexical OOF labels disagree")
    columns = [f"neural_prob_{name}" for name in MODEL_NAMES]
    if not set(columns).issubset(aligned):
        raise ValueError("Neural OOF lacks 18-class probabilities")
    probabilities = validate_probability_matrix(
        aligned[columns].to_numpy(dtype=float), "neural_oof")
    resolution = round(1/step)
    if not 0<step<=1 or not np.isclose(step*resolution,1):
        raise ValueError("Step must divide one exactly")
    best=(-1.0, 0.0)
    for i in range(resolution+1):
        weight=i/resolution
        result=combine([probabilities],lexical,lexical_weights,weight)
        score=report(true,result.argmax(1))["balanced_accuracy_observed_classes"]
        if score>best[0]:
            best=(score,weight)
    return best[1],best[0]
