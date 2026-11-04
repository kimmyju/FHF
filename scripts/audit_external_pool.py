"""Audit official train/validation + NTL v2 + NTL Open Pool, without training.

Read-only on input files. Outputs diagnostic JSON/CSV under outputs/external-audit/.
No inferred label remapping, no automatic trust based on dataset names or flags.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from fhf.external_data import normalize_text, _read_passages_prompts
from fhf.labels import MODEL_NAMES


def read_table(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


def v2_frame(passages, prompts):
    frame, join = _read_passages_prompts(
        Path(passages), Path(prompts), {"join": {}}
    )
    needed = {"text", "model", "prompt_text"}
    if not needed.issubset(frame):
        raise ValueError(f"NTL v2 missing columns: {needed - set(frame)}")
    frame = frame.copy()
    frame["prompt_key"] = frame["prompt_text"].astype(str).map(normalize_text)
    frame["source_dataset"] = "ntl-claude-gen-v2"
    frame["origin_hint"] = "ntl-v2"
    return frame, join


def pool_frame(path):
    frame = read_table(path).copy()
    required = {"text", "model", "prompt_group", "source_dataset"}
    if not required.issubset(frame):
        raise ValueError(f"Open Pool missing columns: {required - set(frame)}")
    # prompt_group is a native pool key, NOT a verified cross-dataset identity.
    frame["prompt_key"] = frame["prompt_group"].astype("string")
    return frame


def enriched(frame, source):
    for col in ("source_dataset", "origin_hint", "lab", "source_license",
                "flag_kb_proxy", "flag_gemini37_routing_risk",
                "flag_near_train_prompt", "flag_train_prompt_regen",
                "max_cos_train", "max_sim_train_prompt", "style_score"):
        if col not in frame:
            frame[col] = pd.NA
    frame = frame.copy().reset_index(drop=True)
    frame["input_dataset"] = source
    frame["source_row"] = frame.index
    frame["model"] = frame["model"].astype("string").str.strip()
    frame["text"] = frame["text"].astype("string")
    frame["text_key"] = frame["text"].fillna("").map(normalize_text)
    frame["word_count"] = frame["text"].fillna("").str.split().map(len)
    frame["known_model"] = frame["model"].isin(MODEL_NAMES)
    frame["valid_text"] = frame["text_key"].ne("")
    frame["native_prompt_missing"] = frame["prompt_key"].isna() | frame["prompt_key"].astype("string").str.strip().eq("")
    return frame


def flag_true(s):
    # Only known boolean-like true values. Missing/unknown must not silently pass.
    a = s.astype("string").str.strip().str.lower()
    return a.isin(["true", "1", "yes", "y", "t"])


def summary_frame(f):
    return {
        "rows": int(len(f)),
        "known_model_rows": int(f.known_model.sum()),
        "unknown_model_counts": {str(k): int(v) for k, v in
                                 f.loc[~f.known_model, "model"].fillna("<NULL>").value_counts().items()},
        "model_counts": {str(k): int(v) for k, v in f.model.value_counts().items()},
        "distinct_texts": int(f.text_key.nunique()),
        "duplicate_text_rows": int(f.text_key.duplicated().sum()),
        "unique_native_prompts": int(f.prompt_key.nunique()),
        "missing_native_prompt_rows": int(f.native_prompt_missing.sum()),
        "sources": {str(k): int(v) for k, v in f.source_dataset.value_counts().head(30).items()},
        "lab_values": {str(k): int(v) for k, v in f.lab.fillna("<NULL>").astype(str).value_counts().head(30).items()},
        "origin_values": {str(k): int(v) for k, v in f.origin_hint.fillna("<NULL>").astype(str).value_counts().head(30).items()},
        "license_values": {str(k): int(v) for k, v in f.source_license.fillna("<NULL>").astype(str).value_counts().head(30).items()},
    }


def audit(train, val, v2, pool, min_words=40, max_words=250):
    official_train = enriched(train.assign(prompt_key=train.prompt_id.astype("string")), "official_train")
    official_val = enriched(val.assign(prompt_key=val.prompt_id.astype("string")), "official_validation")
    # Only use official text keys to prevent content overlap; official prompt IDs
    # are anonymized, so cross-corpus prompt leakage cannot be proven absent.
    train_keys = set(official_train.text_key) - {""}
    val_keys = set(official_val.text_key) - {""}
    candidates = pd.concat([enriched(v2, "ntl_v2"), enriched(pool, "open_pool")],
                           ignore_index=True, sort=False)

    candidates["overlap_official_train"] = candidates.text_key.isin(train_keys)
    candidates["overlap_official_val"] = candidates.text_key.isin(val_keys)
    valid_external = candidates[candidates.valid_text]
    label_conflicts = valid_external.groupby("text_key").model.nunique(dropna=True)
    conflicting_texts = set(label_conflicts[label_conflicts > 1].index)
    candidates["conflicting_labels"] = candidates.text_key.isin(conflicting_texts)

    # Mark ALL copies of cross-source duplicates, not just the second copy.
    origin_count = candidates[candidates.valid_text].groupby("text_key").input_dataset.nunique()
    shared_keys = set(origin_count[origin_count > 1].index)
    candidates["shared_v2_open_pool"] = candidates.text_key.isin(shared_keys)
    candidates["external_duplicate"] = candidates.duplicated("text_key", keep=False) & candidates.valid_text

    risky = ["flag_kb_proxy", "flag_gemini37_routing_risk",
             "flag_near_train_prompt", "flag_train_prompt_regen"]
    for flag in risky:
        candidates[f"risk_{flag}"] = flag_true(candidates[flag])
    candidates["any_known_risk"] = candidates[[f"risk_{x}" for x in risky]].any(axis=1)
    candidates["unknown_risk_metadata"] = candidates[risky].isna().any(axis=1)

    # Conservative *provisional* candidates only: source labels, validity,
    # known risk flags, exact dupes, and length; NOT a certification of provenance.
    candidates["provisional_eligible"] = (
        candidates.known_model & candidates.valid_text
        & candidates.word_count.between(min_words, max_words)
        & ~candidates.overlap_official_train & ~candidates.overlap_official_val
        & ~candidates.conflicting_labels & ~candidates.any_known_risk
        & ~candidates.native_prompt_missing
    )
    # Unknown risk metadata remain visible rather than falsely treated as safe.
    # Duplicate selection prioritizes v2 for reproducibility, not because it is
    # verified to be better; source-level vetting is required before training.
    eligible = candidates[candidates.provisional_eligible].copy()
    eligible = eligible.sort_values(["input_dataset", "source_row"],
                                  key=lambda s: s.map({"ntl_v2": 0, "open_pool": 1})
                                  if s.name == "input_dataset" else s)
    eligible["selected_after_exact_dedup"] = ~eligible.duplicated("text_key", keep="first")
    candidates["selected_after_exact_dedup"] = False
    candidates.loc[eligible.index, "selected_after_exact_dedup"] = eligible.selected_after_exact_dedup

    per_model = (
        candidates.groupby(["model", "input_dataset"], dropna=False)
        .agg(raw=("model", "size"),
             provisional=("provisional_eligible", "sum"),
             unique_selected=("selected_after_exact_dedup", "sum"),
             official_train_overlap=("overlap_official_train", "sum"),
             official_val_overlap=("overlap_official_val", "sum"),
             known_risk=("any_known_risk", "sum"),
             unknown_risk_metadata=("unknown_risk_metadata", "sum"),
             shared_v2_open_pool=("shared_v2_open_pool", "sum"))
        .reset_index()
    )
    c = candidates[candidates.selected_after_exact_dedup & candidates.known_model].copy()
    prompts = (c.groupby(["input_dataset", "model"]).prompt_key.nunique()
               .rename("distinct_native_prompt_keys").reset_index())
    per_model = per_model.merge(prompts, how="left", on=["input_dataset", "model"])
    per_model["distinct_native_prompt_keys"] = per_model["distinct_native_prompt_keys"].fillna(0).astype(int)
    # Source distributions can be diagnostic of model-source confounding.
    source_by_model = (
        c.groupby(["model", "input_dataset", "source_dataset"], dropna=False)
        .size().rename("count").reset_index().sort_values("count", ascending=False)
    )
    per_model_total = (per_model.groupby("model", dropna=False)[["raw", "provisional", "unique_selected"]]
                       .sum().reindex(MODEL_NAMES, fill_value=0).reset_index())
    per_model_total.insert(1, "official_train", per_model_total["model"].map(
        official_train.model.value_counts()).fillna(0).astype(int))
    per_model_total["combined_train_if_all_provisional_used"] = (
        per_model_total["official_train"] + per_model_total["unique_selected"])
    per_model_total["deficit_to_100_external"] = (100 - per_model_total["unique_selected"]).clip(lower=0)
    per_model_total["deficit_to_300_external"] = (300 - per_model_total["unique_selected"]).clip(lower=0)

    metrics = {
        "note": ("Read-only diagnostics. 'provisional' is NOT verified trustworthiness; "
                 "inspect original providers, licenses, lab codes, source evidence, "
                 "generation API routing and near-duplicates before training. "
                 "Native prompt keys from different datasets are not directly comparable."),
        "inputs": {"official_train": summary_frame(official_train),
                   "official_validation": summary_frame(official_val),
                   "ntl_v2": summary_frame(candidates[candidates.input_dataset == "ntl_v2"]),
                   "open_pool": summary_frame(candidates[candidates.input_dataset == "open_pool"])},
        "cross_source": {
            "distinct_normalized_responses_shared_between_v2_and_open_pool": len(shared_keys),
            "rows_in_shared_responses": int(candidates.shared_v2_open_pool.sum()),
            "exact_train_overlap_rows": int(candidates.overlap_official_train.sum()),
            "exact_val_overlap_rows": int(candidates.overlap_official_val.sum()),
            "distinct_response_keys_with_conflicting_model_labels": len(conflicting_texts),
            "conflicting_label_rows": int(candidates.conflicting_labels.sum()),
            "provisional_eligible_rows_before_dedup": int(candidates.provisional_eligible.sum()),
            "provisional_selected_rows_after_dedup": int(candidates.selected_after_exact_dedup.sum()),
            "missing_risk_metadata_rows": int(candidates.unknown_risk_metadata.sum()),
            "risk_flag_true_counts": {
                x: int(candidates[f"risk_{x}"].sum()) for x in risky
            },
            "uncertainty": ("Exact response matching does not catch paraphrases; "
                            "official prompt IDs do not identify their text across corpora; "
                            "similarity columns require documented definitions and calibration.")
        },
        "criteria": {"minimum_words": min_words, "maximum_words": max_words,
                     "official_validation_untouched": True},
    }
    return candidates, per_model, per_model_total, source_by_model, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-train", default="outputs/dual-path-prompt-aware/train_split.csv")
    p.add_argument("--official-val", default="outputs/dual-path-prompt-aware/val.csv")
    p.add_argument("--v2-passages", default="data/external/ntl_v2/passages.csv")
    p.add_argument("--v2-prompts", default="data/external/ntl_v2/prompts.csv")
    p.add_argument("--pool", default="data/external/ntl_v2/ntl_v3/ntl_open_pool.parquet")
    p.add_argument("--output", default="outputs/external-audit")
    p.add_argument("--min-words", type=int, default=40)
    p.add_argument("--max-words", type=int, default=250)
    args = p.parse_args()
    train = read_table(args.official_train)
    val = read_table(args.official_val)
    v2, join = v2_frame(args.v2_passages, args.v2_prompts)
    pool = pool_frame(args.pool)
    frames = audit(train, val, v2, pool, args.min_words, args.max_words)
    detailed, per_model, totals, source_counts, metrics = frames
    metrics["v2_join"] = join
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(out / "all_external_row_flags.csv", index=False)
    per_model.to_csv(out / "per_model_by_dataset.csv", index=False)
    totals.to_csv(out / "per_model_totals.csv", index=False)
    source_counts.to_csv(out / "source_by_model.csv", index=False)
    (out / "audit_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("External audit saved to", out)
    print(json.dumps(metrics["cross_source"], indent=2, ensure_ascii=False))
    print(totals.to_string(index=False))
    print("NO training files were changed. Review before any augmentation.")


if __name__ == "__main__":
    main()
