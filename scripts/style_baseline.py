"""Train a lightweight style-only 18-class baseline on FHF's held-out prompt split.

No ModernBERT, GPU, LoRA or augmentation is used. Texts are represented
by character n-gram TF-IDF and classified with logistic regression.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fhf.config import load_config, resolve_path
from fhf.data import grouped_split, load_csv
from fhf.labels import MODEL_NAMES, MODEL_TO_ID
from fhf.metrics import report


def load_matching_split(config):
    """Prefer the exact split exported by training; otherwise reproduce it."""
    train_file = resolve_path(config, "train_split_csv")
    valid_file = resolve_path(config, "validation_csv")
    if train_file.is_file() != valid_file.is_file():
        raise FileNotFoundError(
            "Only one saved split exists. Supply both train_split_csv and "
            "validation_csv or remove both to regenerate from train_csv."
        )
    if train_file.is_file():
        train, valid = load_csv(train_file), load_csv(valid_file)
        source = "saved_fhf_split"
    else:
        train, valid = grouped_split(
            load_csv(resolve_path(config, "train_csv")),
            float(config["training"]["validation_fraction"]),
            int(config["seed"]),
        )
        source = "regenerated_prompt_disjoint_split"

    if "prompt_id" not in train or "prompt_id" not in valid:
        raise ValueError("Both splits must have prompt_id for leakage checks")
    if train.prompt_id.isna().any() or valid.prompt_id.isna().any():
        raise ValueError("Missing prompt_id in train or validation split")
    if not set(train.prompt_id).isdisjoint(set(valid.prompt_id)):
        raise ValueError("Prompt leakage: train and validation share prompt IDs")
    if not set(train.id).isdisjoint(set(valid.id)):
        raise ValueError("Train and validation share sample IDs")
    missing = set(MODEL_NAMES) - set(train.model)
    if missing:
        raise ValueError(f"Training split is missing exact model classes: {sorted(missing)}")
    return train, valid, source


def run_style_baseline(config, output_dir="outputs/style-baseline",
                       ngram_range=(3, 5), max_features=250_000,
                       min_df=2, c=1.0):
    """Fit TF-IDF only on training text and save predictions plus diagnostics."""
    if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
        raise ValueError("Invalid ngram_range")
    if min_df < 1 or max_features < 1 or c <= 0:
        raise ValueError("min_df, max_features, and C must be positive")

    train, valid, source = load_matching_split(config)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=tuple(ngram_range),
            lowercase=False,
            min_df=int(min_df),
            max_features=int(max_features),
            sublinear_tf=True,
            dtype=np.float32,
        )),
        ("classifier", LogisticRegression(
            C=float(c), max_iter=3000, random_state=int(config["seed"]),
        )),
    ])
    pipeline.fit(train.text.astype(str).tolist(), train.model.tolist())
    inputs = valid.text.astype(str).tolist()
    pred_names = pipeline.predict(inputs)
    probabilities = pipeline.predict_proba(inputs)
    classes = list(pipeline.classes_)
    if set(classes) != set(MODEL_NAMES):
        raise RuntimeError("Classifier did not learn all 18 exact models")

    metrics = report(
        [MODEL_TO_ID[name] for name in valid.model],
        [MODEL_TO_ID[name] for name in pred_names],
    )
    metrics["split_source"] = source
    metrics["num_train"] = len(train)
    metrics["num_validation"] = len(valid)
    metrics["method"] = "character_ngram_tfidf_logistic_regression"
    metrics["ngram_range"] = list(ngram_range)
    metrics["max_features"] = int(max_features)
    metrics["min_df"] = int(min_df)
    metrics["C"] = float(c)

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out / "style_model.joblib")
    (out / "evaluation.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    prediction_table = pd.DataFrame({
        "id": valid.id.to_numpy(),
        "prompt_id": valid.prompt_id.to_numpy(),
        "true_model": valid.model.to_numpy(),
        "pred_model": pred_names,
    })
    # Save class scores to support a later (validation-only) fusion ablation.
    for name in MODEL_NAMES:
        prediction_table[f"prob_{name}"] = probabilities[:, classes.index(name)]
    prediction_table.to_csv(out / "val_predictions.csv", index=False)

    print(f"Split: {source}, train={len(train)}, val={len(valid)}")
    print(f"Style-only Accuracy: {metrics['accuracy']:.4f}")
    print(f"Style-only Balanced Accuracy: {metrics['balanced_accuracy_observed_classes']:.4f}")
    print(f"Saved model, evaluation and predictions to: {out}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/ablations/longer_training.yaml",
        help="Use this FHF experiment's existing train/validation split.",
    )
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output-dir", default="outputs/style-baseline")
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=250_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    run_style_baseline(
        config,
        output_dir=args.output_dir,
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features,
        min_df=args.min_df,
        c=args.C,
    )


if __name__ == "__main__":
    main()
