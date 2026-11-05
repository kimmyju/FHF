"""Fit OOF-only bias or evaluate it on SAVED ensemble validation predictions.

FIT requires neural OOF for neural-weight>0; APPLY never reads validation labels
while fitting and does not replace the original validation.csv.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from fhf.data import load_csv
from fhf.labels import MODEL_NAMES, MODEL_TO_ID
from fhf.v2_calibration import fit_and_save, load_bias, apply_bias


def run_apply(input_csv, model_json, output_csv):
    pred = pd.read_csv(input_csv, dtype={"id": str, "prompt_id": str})
    required = {"id", "true_model", "pred_model"}
    columns = [f"prob_{name}" for name in MODEL_NAMES]
    if not required.issubset(pred) or not set(columns).issubset(pred):
        raise ValueError("Saved ensemble validation CSV needs IDs, labels and all 18 probs")
    if pred.id.duplicated().any():
        raise ValueError("Duplicate validation predictions")
    document = load_bias(model_json)
    before = pred[columns].to_numpy(dtype=float)
    predicted = np.asarray(MODEL_NAMES)[before.argmax(axis=1)]
    if not np.array_equal(predicted, pred.pred_model.to_numpy()):
        raise ValueError("Stored pred_model disagrees with probabilities")
    calibrated = apply_bias(before, document["biases"])
    new = pred.copy()
    new["pred_model"] = np.asarray(MODEL_NAMES)[calibrated.argmax(axis=1)]
    for i, col in enumerate(columns):
        new[col] = calibrated[:, i]
    destination = Path(output_csv)
    if destination.resolve() == Path(input_csv).resolve():
        raise ValueError("Do not overwrite baseline validation.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(destination, index=False)
    report = {"n": len(new), "frozen_validation_used_for_fitting": False}
    if not new.true_model.isna().any():
        if not set(new.true_model).issubset(MODEL_TO_ID):
            raise ValueError("Unknown true model labels")
        y = new.true_model.map(MODEL_TO_ID).to_numpy()
        report["before_bacc"] = float(balanced_accuracy_score(y, before.argmax(1)))
        report["after_bacc"] = float(balanced_accuracy_score(y, calibrated.argmax(1)))
        report["delta_bacc"] = report["after_bacc"] - report["before_bacc"]
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fit", "apply"], required=True)
    parser.add_argument("--train", default="outputs/dual-path-prompt-aware/train_split.csv")
    parser.add_argument("--lexical-oof",
                        default="outputs/fingerprint-v2/classical/oof_predictions.csv")
    parser.add_argument("--neural-oof",
                        default="outputs/fingerprint-v2/neural-oof/neural_oof_predictions.csv")
    parser.add_argument("--neural-weight", type=float, default=0.7)
    parser.add_argument("--regularization", type=float, default=0.5)
    parser.add_argument("--max-abs", type=float, default=1.5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bias-json",
                        default="outputs/fingerprint-v2/calibration/fit.json")
    parser.add_argument("--validation-predictions",
                        default="outputs/fingerprint-v2/ensemble/validation.csv")
    parser.add_argument("--output",
                        default="outputs/fingerprint-v2/calibration/validation_calibrated.csv")
    args = parser.parse_args()
    if args.mode == "fit":
        train = load_csv(args.train)
        lexical = pd.read_csv(args.lexical_oof, dtype={"id":str,"prompt_id":str})
        neural = pd.read_csv(args.neural_oof, dtype={"id":str,"prompt_id":str}) if (
            args.neural_weight > 0) else None
        result = fit_and_save(
            train, lexical, neural, args.neural_weight, args.bias_json,
            args.regularization, args.max_abs, args.folds)
        print(json.dumps(result["train_oof_crossfit"], indent=2))
        print(f"Saved calibration to {args.bias_json}")
    else:
        run_apply(args.validation_predictions, args.bias_json, args.output)


if __name__ == "__main__":
    main()
