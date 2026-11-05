"""Confusion matrix reports from a SAVED V2 validation prediction CSV.

Rows are true labels; columns are predicted labels. No GPU or retraining.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .labels import MODEL_NAMES

FOCUS = ("glm-5.3", "glm-5.3-flash")


def diagnose(predictions, output_dir, plot=True):
    required = {"id", "true_model", "pred_model"}
    if not required.issubset(predictions):
        raise ValueError(f"Missing validation columns: {sorted(required - set(predictions))}")
    if predictions.empty or predictions.id.astype(str).duplicated().any():
        raise ValueError("Empty predictions or duplicated IDs")
    for field in ("true_model", "pred_model"):
        if predictions[field].isna().any():
            raise ValueError(f"{field} contains null values")
        unknown = set(predictions[field]) - set(MODEL_NAMES)
        if unknown:
            raise ValueError(f"Unknown {field} labels: {sorted(unknown)}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(
        predictions.true_model, predictions.pred_model, labels=MODEL_NAMES)
    counts = pd.DataFrame(matrix, index=MODEL_NAMES, columns=MODEL_NAMES)
    counts.index.name = "true_model"
    counts.columns.name = "pred_model"
    counts.to_csv(destination / "confusion_matrix_counts.csv")
    support = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, support, where=support > 0,
                           out=np.zeros(matrix.shape, dtype=float))
    fractions = pd.DataFrame(normalized, index=MODEL_NAMES, columns=MODEL_NAMES)
    fractions.index.name = "true_model"
    fractions.columns.name = "pred_model"
    fractions.to_csv(destination / "confusion_matrix_row_normalized.csv")

    cols = [k for k in ("id", "prompt_id", "true_model", "pred_model")
            if k in predictions]
    cols += [f"prob_{name}" for name in MODEL_NAMES
             if f"prob_{name}" in predictions]
    focus_rows = predictions[predictions.true_model.isin(FOCUS)]
    errors = predictions[
        (predictions.true_model != predictions.pred_model)
        & (predictions.true_model.isin(FOCUS)
           | predictions.pred_model.isin(FOCUS))
    ]
    focus_rows[cols].to_csv(destination / "glm_true_predictions.csv", index=False)
    errors[cols].to_csv(destination / "glm_errors.csv", index=False)
    counts.loc[list(FOCUS)].to_csv(destination / "glm_confusion_rows.csv")
    fractions.loc[list(FOCUS)].to_csv(
        destination / "glm_confusion_rows_normalized.csv")
    breakdown = []
    summary = {}
    for name in FOCUS:
        count = int(counts.loc[name].sum())
        correct = int(counts.loc[name, name])
        predicted = int(counts[name].sum())
        summary[name] = {
            "support": count,
            "correct": correct,
            "recall": correct / count if count else None,
            "precision": correct / predicted if predicted else None,
            "false_negatives": count - correct,
            "false_positives": predicted - correct,
            "confused_with": {
                other: int(value)
                for other, value in counts.loc[name].items()
                if other != name and value > 0
            },
        }
        if count:
            for other, value in counts.loc[name].items():
                if value:
                    breakdown.append({
                        "true_model": name, "pred_model": other,
                        "count": int(value), "rate": float(value / count),
                        "correct": other == name,
                    })
    pd.DataFrame(breakdown, columns=[
        "true_model", "pred_model", "count", "rate", "correct"
    ]).to_csv(destination / "glm_error_breakdown.csv", index=False)
    report = {"rows": int(len(predictions)),
              "orientation": "rows=true, columns=predicted",
              "glm": summary, "glm_involved_errors": int(len(errors))}
    (destination / "glm_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("Install matplotlib to export PNG; CSV/JSON are ready.")
        else:
            for filename, values in (
                ("confusion_matrix_counts.png", matrix),
                ("confusion_matrix_row_normalized.png", normalized),
            ):
                fig, ax = plt.subplots(figsize=(15, 12))
                im = ax.imshow(values, cmap="Blues", interpolation="nearest")
                fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
                ax.set_xticks(range(len(MODEL_NAMES)), MODEL_NAMES,
                              rotation=90, fontsize=7)
                ax.set_yticks(range(len(MODEL_NAMES)), MODEL_NAMES, fontsize=7)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                for name in FOCUS:
                    k = MODEL_NAMES.index(name)
                    ax.axhspan(k - 0.5, k + 0.5, color="orange", alpha=0.15)
                fig.tight_layout()
                fig.savefig(destination / filename, dpi=160)
                plt.close(fig)
    print(f"Confusion diagnostics saved to {destination}")
    for name, values in summary.items():
        print(f"{name}: {values['correct']}/{values['support']} correct; "
              f"confused with {values['confused_with']}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",
                        default="outputs/fingerprint-v2/ensemble/validation.csv")
    parser.add_argument("--output-dir",
                        default="outputs/fingerprint-v2/ensemble/diagnostics")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if not Path(args.predictions).exists():
        parser.error(f"No saved predictions: {args.predictions}")
    diagnose(pd.read_csv(args.predictions, dtype={"id": str, "prompt_id": str}),
             args.output_dir, plot=not args.no_plot)


if __name__ == "__main__":
    main()
