"""Read-only GLM vs Claude-Fable provenance and style audit.

This does NOT verify real API provenance. It identifies suspicious sources,
exact-label conflicts, duplicates, routing flags, and observable corpus shift.
It never relabels or modifies training examples.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from .external_data import normalize_text
from .labels import MODEL_NAMES

FOCUS = ("glm-5.3", "glm-5.3-flash", "claude-fable-5")
FLAGS = ("flag_kb_proxy", "flag_gemini37_routing_risk",
         "flag_near_train_prompt", "flag_train_prompt_regen")


def _digest(s):
    return hashlib.sha256(normalize_text(s).encode("utf-8")).hexdigest()


def _features(frame, corpus):
    result = frame[["model", "text"]].copy()
    texts = result.text.fillna("").astype(str)
    result["corpus"] = corpus
    result["words"] = texts.str.split().map(len)
    result["chars"] = texts.str.len()
    result["paragraphs"] = texts.str.count(r"\n\s*\n") + 1
    result["emdashes"] = texts.str.count("—")
    result["semicolons"] = texts.str.count(";")
    result["bullets"] = texts.str.count(r"(?m)^\s*[-*]\s")
    result["headings"] = texts.str.count(r"(?m)^#{1,6}\s")
    result["nonascii_fraction"] = texts.map(
        lambda s: sum(ord(c) > 127 for c in s) / max(len(s), 1))
    return result


def _summary_features(frame):
    result = []
    for (corpus, model), group in frame.groupby(["corpus", "model"]):
        row = {"corpus": corpus, "model": model, "n": len(group)}
        for feature in ("words", "chars", "paragraphs", "emdashes",
                        "semicolons", "bullets", "headings", "nonascii_fraction"):
            row[f"{feature}_mean"] = float(group[feature].mean())
            row[f"{feature}_median"] = float(group[feature].median())
            if feature in ("words", "chars"):
                row[f"{feature}_p10"] = float(group[feature].quantile(0.1))
                row[f"{feature}_p90"] = float(group[feature].quantile(0.9))
        result.append(row)
    return pd.DataFrame(result)


def _counts(values):
    return {str(k): int(v) for k, v in values.items()}


def _flag_count(values):
    as_str = values.astype("string").str.strip().str.lower()
    return int(as_str.isin(("1", "true", "t", "yes", "y")).sum())


def audit_selected(official_train, selected, official_validation=None, raw_pool=None,
                   include_text=False):
    required = {"id", "model", "text", "prompt_id"}
    for name, frame in (("official_train", official_train), ("selected", selected)):
        if not required.issubset(frame):
            raise ValueError(f"{name} missing {sorted(required - set(frame))}")
        if frame.empty or frame.id.astype(str).duplicated().any():
            raise ValueError(f"{name}: empty or duplicate IDs")
        unknown = set(frame.model.dropna().astype(str)) - set(MODEL_NAMES)
        if unknown:
            raise ValueError(f"{name} contains unknown labels: {sorted(unknown)}")
    for model in FOCUS:
        if (official_train.model == model).sum() == 0 or (selected.model == model).sum() == 0:
            raise ValueError(f"Missing comparison class {model}")

    a = official_train.copy()
    b = selected.copy()
    a["_hash"] = a.text.fillna("").map(_digest)
    b["_hash"] = b.text.fillna("").map(_digest)
    if official_validation is not None:
        val = official_validation.copy()
        val_hashes = set(val.text.fillna("").map(_digest))
    else:
        val_hashes = set()
    train_hashes = set(a["_hash"])
    selected_hashes = set(b["_hash"])
    exact_conflicts = (
        pd.concat([a[["_hash", "model"]].assign(corpus="official"),
                   b[["_hash", "model"]].assign(corpus="external")],
                  ignore_index=True)
        .groupby("_hash").model.nunique()
    )
    conflict_hashes = set(exact_conflicts[exact_conflicts > 1].index)
    selected_sources = b.get("source_dataset", pd.Series(
        ["<source-unrecorded>"] * len(b), index=b.index
    )).fillna("<missing>").astype(str)

    counts = b.groupby("model").size()
    source_table = pd.crosstab(b.model, selected_sources).reindex(
        index=MODEL_NAMES, fill_value=0)
    source_metrics = {}
    for name in FOCUS:
        frequencies = source_table.loc[name]
        n = int(frequencies.sum())
        ranked = frequencies[frequencies > 0].sort_values(ascending=False)
        source_metrics[name] = {
            "selected_rows": n,
            "distinct_sources": len(ranked),
            "dominant_source": str(ranked.index[0]) if len(ranked) else None,
            "dominant_source_fraction": float(ranked.iloc[0] / n) if n else None,
            "sources": _counts(ranked.to_dict()),
            "distinct_prompt_ids": int(b.loc[b.model == name, "prompt_id"].nunique()),
            "exact_duplicate_with_official_train": int(
                (b.loc[b.model == name, "_hash"].isin(train_hashes)).sum()),
            "exact_duplicate_with_official_validation": int(
                (b.loc[b.model == name, "_hash"].isin(val_hashes)).sum()),
            "exact_label_conflict_with_train_or_selected": int(
                (b.loc[b.model == name, "_hash"].isin(conflict_hashes)).sum()),
        }

    features = pd.concat([
        _features(a[a.model.isin(FOCUS)], "official_train"),
        _features(b[b.model.isin(FOCUS)], "selected_external"),
    ], ignore_index=True)
    feature_summary = _summary_features(features)

    # Inspect raw Open Pool metadata by normalized response hash, not row
    # position. Never infer a provider/version from a similar-looking text.
    raw_report = {"available": raw_pool is not None,
                  "note": "Missing risk flags or a source name is not evidence of safe routing."}
    row_review = b.loc[b.model.isin(FOCUS),
                       ["id", "model", "prompt_id", "_hash"]].copy()
    row_review["source_dataset"] = selected_sources.loc[row_review.index].to_numpy()
    if raw_pool is not None:
        if not {"text", "model"}.issubset(raw_pool):
            raise ValueError("Raw pool must contain text and model")
        raw = raw_pool.copy()
        raw["_hash"] = raw.text.fillna("").map(_digest)
        raw_focus = raw[raw["_hash"].isin(set(row_review["_hash"]))].copy()
        labels_by_hash = raw_focus.groupby("_hash").model.agg(
            lambda x: sorted(set(x.dropna().astype(str))))
        selected_group = row_review.groupby("_hash").model.agg(
            lambda x: sorted(set(x.astype(str))))
        inconsistent = [key for key, names in selected_group.items()
                        if key in labels_by_hash and
                        set(names) != set(labels_by_hash[key])]
        raw_report.update({
            "matched_rows": int(len(raw_focus)),
            "matched_unique_texts": int(raw_focus["_hash"].nunique()),
            "selected_focus_unique_texts": int(row_review["_hash"].nunique()),
            "selected_vs_raw_label_disagreement_hashes": int(len(inconsistent)),
            "unmatched_selected_hashes": int(
                len(set(row_review["_hash"]) - set(raw_focus["_hash"]))),
            "metadata_fields": [col for col in ("source_dataset", "origin_hint",
               "lab", "source_license", *FLAGS) if col in raw],
        })
        for col in FLAGS:
            if col in raw_focus:
                raw_report[f"{col}_true_count"] = _flag_count(raw_focus[col])
                raw_report[f"{col}_missing_count"] = int(raw_focus[col].isna().sum())
        if "lab" in raw_focus:
            raw_report["lab_counts"] = _counts(
                raw_focus.lab.fillna("<missing>").astype(str).value_counts().to_dict())
        if "origin_hint" in raw_focus:
            raw_report["origin_counts"] = _counts(
                raw_focus.origin_hint.fillna("<missing>").astype(str).value_counts().to_dict())
        if "source_license" in raw_focus:
            raw_report["licenses"] = _counts(
                raw_focus.source_license.fillna("<missing>").astype(str).value_counts().to_dict())
        row_review["raw_pool_matches"] = row_review["_hash"].map(
            raw_focus["_hash"].value_counts()).fillna(0).astype(int)
        row_review["raw_label_disagreement"] = row_review["_hash"].isin(inconsistent)
    if include_text:
        row_review["text"] = b.loc[row_review.index, "text"].astype(str).to_numpy()

    summary = {
        "note": "Provenance not independently verified. These are heuristic diagnostics, not proof of routing or mislabeling.",
        "focus_models": list(FOCUS),
        "official_train_rows": len(a),
        "selected_external_rows": len(b),
        "selected_duplicate_text_rows": int(b["_hash"].duplicated().sum()),
        "selected_overlap_official_train": len(selected_hashes & train_hashes),
        "selected_overlap_official_validation": len(selected_hashes & val_hashes),
        "selected_hashes_with_conflicting_labels": len(
            conflict_hashes & selected_hashes),
        "focus": source_metrics,
        "raw_pool": raw_report,
    }
    # Avoid including any text in the default outputs.
    row_review = row_review.rename(columns={"_hash": "normalized_sha256"})
    return summary, source_table, feature_summary, row_review


def save_audit(result, folder):
    summary, sources, features, rows = result
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    sources.to_csv(folder / "source_by_model.csv")
    features.to_csv(folder / "style_comparison.csv", index=False)
    rows.to_csv(folder / "focus_rows.csv", index=False)
    return folder
