"""Run prompt-disjoint OOF and frozen official validation for lexical specialists.

The 5-fold OOF scores may be used to choose lexical mixture weights.
The untouched official validation is reported, never used to fit or tune.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from fhf.data import load_csv
from fhf.external_data import normalize_text
from fhf.labels import MODEL_NAMES, MODEL_TO_ID
from fhf.metrics import report
from fhf.v2_classical import LexicalFingerprintEnsemble


def _csv(frame, matrices, output):
    out = pd.DataFrame({"id": frame.id.astype(str).to_numpy()})
    if "prompt_id" in frame:
        out["prompt_id"] = frame.prompt_id.astype(str).to_numpy()
    if "model" in frame:
        out["true_model"] = frame.model.to_numpy()
    for component, values in matrices.items():
        if values.shape != (len(frame), len(MODEL_NAMES)):
            raise ValueError(f"Invalid probabilities from {component}")
        for i, name in enumerate(MODEL_NAMES):
            out[f"{component}_prob_{name}"] = values[:, i]
        out[f"{component}_pred_model"] = np.asarray(MODEL_NAMES)[values.argmax(1)]
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)


def run(train, validation, outdir, folds=5, seed=42,
        char_features=180_000, word_features=100_000, do_oof=True):
    if "prompt_id" not in train or "prompt_id" not in validation:
        raise ValueError("Prompt IDs are necessary for GroupKFold and frozen validation")
    if set(train.prompt_id.astype(str)) & set(validation.prompt_id.astype(str)):
        raise ValueError("Official train/validation prompt overlap")
    if set(train.text.map(normalize_text)) & set(validation.text.map(normalize_text)):
        raise ValueError("Official train/validation exact text overlap")
    if set(train.id.astype(str)) & set(validation.id.astype(str)):
        raise ValueError("Official train/validation ID overlap")
    if set(train.model) != set(MODEL_NAMES) or set(validation.model) != set(MODEL_NAMES):
        raise ValueError("All 18 model labels must be represented in both splits")
    if train.id.astype(str).duplicated().any() or validation.id.astype(str).duplicated().any():
        raise ValueError("Duplicate IDs")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    make = lambda: LexicalFingerprintEnsemble(
        seed=seed, char_features=char_features, word_features=word_features)
    summary = {"train_rows":len(train), "validation_rows":len(validation),
               "folds":folds, "seed":seed,
               "split": "official_train_groupkfold_oof_then_frozen_val"}
    if do_oof:
        if not 2 <= folds <= train.prompt_id.nunique():
            raise ValueError("Invalid number of folds")
        predictions = {kind: np.zeros((len(train), len(MODEL_NAMES)), dtype=np.float64)
                       for kind in ("char", "word", "specialist")}
        seen = np.zeros(len(train), dtype=int)
        groups = train.prompt_id.astype(str).to_numpy()
        splitter = GroupKFold(n_splits=folds)
        for index, (fit_idx, val_idx) in enumerate(splitter.split(train, groups=groups), 1):
            fold_train = train.iloc[fit_idx]
            if set(fold_train.prompt_id.astype(str)) & set(train.iloc[val_idx].prompt_id.astype(str)):
                raise RuntimeError("Prompt leakage in OOF fold")
            model = make().fit(fold_train)
            result = model.predict_probabilities(train.iloc[val_idx].text.astype(str))
            for kind, values in result.items():
                predictions[kind][val_idx] = values
            seen[val_idx] += 1
            print(f"Fold {index}/{folds}: train={len(fit_idx)}, OOF={len(val_idx)}")
        if not (seen == 1).all():
            raise RuntimeError("OOF predictions did not cover each train row exactly once")
        _csv(train, predictions, outdir/"oof_predictions.csv")
        true = train.model.map(MODEL_TO_ID).to_numpy()
        summary["oof"] = {
            name: report(true, value.argmax(axis=1))["balanced_accuracy_observed_classes"]
            for name, value in predictions.items()
        }
        (outdir/"oof_scores.json").write_text(json.dumps(summary["oof"],indent=2))
    final = make().fit(train)
    joblib.dump(final, outdir/"lexical_specialists.joblib")
    validation_predictions = final.predict_probabilities(validation.text.astype(str))
    _csv(validation, validation_predictions, outdir/"val_predictions.csv")
    val_true = validation.model.map(MODEL_TO_ID).to_numpy()
    summary["validation"] = {
        name: report(val_true, value.argmax(axis=1))["balanced_accuracy_observed_classes"]
        for name, value in validation_predictions.items()
    }
    (outdir/"evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="outputs/dual-path-prompt-aware/train_split.csv")
    p.add_argument("--val", default="outputs/dual-path-prompt-aware/val.csv")
    p.add_argument("--out", default="outputs/fingerprint-v2/classical")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--char-features", type=int, default=180000)
    p.add_argument("--word-features", type=int, default=100000)
    p.add_argument("--skip-oof", action="store_true")
    a=p.parse_args()
    run(load_csv(a.train), load_csv(a.val), Path(a.out), a.folds,
        a.seed, a.char_features, a.word_features, not a.skip_oof)


if __name__=="__main__":
    main()
