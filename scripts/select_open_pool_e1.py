"""Select 100 external examples per model from audited NTL v2 + Open Pool.

Use with configs/ablations/within_family_style_perturb_openpool_e1.yaml.
Select only; never train or modify the frozen official split.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from audit_external_pool import audit, read_table, v2_frame, pool_frame
from fhf.external_data import normalize_text, digest
from fhf.labels import MODEL_NAMES


def pick_balanced(rows, per_model=100, seed=42):
    """Deterministic round-robin by source, favoring new prompt groups.

    The audit has already resolved cross-source exact duplicates. Native
    prompt IDs are namespaced by input dataset: no assumption that different
    sources use the same prompt identifier. Model-level coverage is enforced.
    """
    if per_model < 1:
        raise ValueError("per_model must be positive")
    rng = np.random.default_rng(seed)
    work = rows.copy()
    if not work.index.is_unique:
        raise ValueError("Candidate indices must be unique")
    # Pool prompt_group is meaningful within Open Pool. For v2 use normalized
    # prompt text; labels from different corpora never collide by accident.
    work["prompt_identity"] = (
        work.input_dataset.astype(str) + ":" + work.prompt_key.astype(str)
    )
    if work.prompt_identity.isna().any():
        raise ValueError("Missing prompt identity")
    selection = []
    for model in MODEL_NAMES:
        pool = work[work.model == model].copy()
        if len(pool) < per_model:
            raise ValueError(f"{model}: only {len(pool)} eligible, need {per_model}")
        # Seeded shuffle ensures we do not select the source file's first rows.
        pool["_random"] = rng.random(len(pool))
        pool = pool.sort_values("_random", kind="stable")
        queues = {
            str(source): frame.index.tolist()
            for source, frame in pool.groupby("source_dataset", sort=True)
        }
        # Prefer new prompts within each class to avoid single-prompt memorization.
        used_prompts = set()
        source_counts = Counter()
        remaining = set(pool.index.tolist())
        while len(selection_for_model := [i for i in selection if work.at[i, "model"] == model]) < per_model:
            available_sources = [s for s, indices in queues.items()
                                 if any(i in remaining for i in indices)]
            if not available_sources:
                raise RuntimeError(f"{model}: exhausted sources")
            min_count = min(source_counts[s] for s in available_sources)
            least_used_sources = [s for s in available_sources if source_counts[s] == min_count]
            # If equal-source candidates exist, prioritize those offering a new prompt.
            source = min(
                least_used_sources,
                key=lambda s: (
                    not any(work.at[i, "prompt_identity"] not in used_prompts
                            for i in queues[s] if i in remaining),
                    s,
                ),
            )
            choices = [i for i in queues[source] if i in remaining]
            fresh = [i for i in choices if work.at[i, "prompt_identity"] not in used_prompts]
            chosen = fresh[0] if fresh else choices[0]
            selection.append(chosen)
            remaining.remove(chosen)
            used_prompts.add(work.at[chosen, "prompt_identity"])
            source_counts[source] += 1
    picked = work.loc[selection].copy()
    assert len(picked) == per_model * len(MODEL_NAMES)
    assert picked.text_key.is_unique
    return picked


def produce(train, val, v2, pool, per_model=100, seed=42, min_words=40, max_words=250,
            exclude_unknown_risk=True):
    detailed, by_model, total, by_source, audit_report = audit(
        train, val, v2, pool, min_words=min_words, max_words=max_words
    )
    options = detailed.loc[detailed.selected_after_exact_dedup].copy()
    # NTL v2 lacks the pool's flags. Do not treat unknown as proven safe.
    # Default E1 uses pool candidates with known flags. The v2 dataset is
    # still fully audited and the nine unique v2 items may be opted into
    # only after independently checking their provenance.
    if exclude_unknown_risk:
        options = options.loc[~options.unknown_risk_metadata].copy()
    picked = pick_balanced(options, per_model=per_model, seed=seed)
    selected = pd.DataFrame({
        "id": ["e1_" + digest(f"{dataset}:{idx}:{key}")[:24]
               for dataset, idx, key in zip(picked.input_dataset,
                                            picked.source_row, picked.text_key)],
        "text": picked.text.astype(str).tolist(),
        "model": picked.model.astype(str).tolist(),
        "prompt_id": ["e1_" + digest(f"{dataset}:{prompt}")[:24]
                      for dataset, prompt in zip(picked.input_dataset,
                                                 picked.prompt_key)],
        "origin": picked.input_dataset.astype(str).tolist(),
        "source_dataset": picked.source_dataset.astype(str).tolist(),
    })
    if len(selected) != 18 * per_model or selected.id.duplicated().any():
        raise RuntimeError("Invalid selected data size or duplicate IDs")
    train_keys = set(train.text.map(normalize_text))
    val_keys = set(val.text.map(normalize_text))
    keys = selected.text.map(normalize_text)
    if keys.duplicated().any() or (set(keys) & (train_keys | val_keys)):
        raise RuntimeError("Selected external data overlaps official split or itself")
    if set(selected.prompt_id.astype(str)) & set(val.prompt_id.astype(str)):
        raise RuntimeError("Prompt ID collision with frozen validation")
    counts = selected.model.value_counts()
    if any(counts.get(model, 0) != per_model for model in MODEL_NAMES):
        raise RuntimeError("External data not model-balanced")
    stats = {
        "selection_seed": seed,
        "per_model": per_model,
        "rows": len(selected),
        "original_train_rows": len(train),
        "frozen_validation_rows": len(val),
        "combined_train_rows": len(train) + len(selected),
        "unknown_risk_metadata_excluded": exclude_unknown_risk,
        "model_counts": counts.to_dict(),
        "source_counts": selected.source_dataset.value_counts().to_dict(),
        "model_prompt_counts": selected.groupby("model").prompt_id.nunique().to_dict(),
        "audit_cross_source": audit_report["cross_source"],
        "disclaimer": ("Provisional curation, not provenance verification. "
                       "Check source/API routing, label quality, licenses and "
                       "semantic prompt/response overlap before drawing conclusions."),
    }
    return selected, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-train", default="outputs/dual-path-prompt-aware/train_split.csv")
    p.add_argument("--official-val", default="outputs/dual-path-prompt-aware/val.csv")
    p.add_argument("--v2-passages", default="data/external/ntl_v2/passages.csv")
    p.add_argument("--v2-prompts", default="data/external/ntl_v2/prompts.csv")
    p.add_argument("--pool", default="data/external/ntl_v2/ntl_v3/ntl_open_pool.parquet")
    p.add_argument("--output", default="data/external/e1/selected_100_per_model.csv")
    p.add_argument("--report", default="outputs/openpool-e1/selection_report.json")
    p.add_argument("--per-model", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train, val = read_table(args.official_train), read_table(args.official_val)
    v2, _ = v2_frame(args.v2_passages, args.v2_prompts)
    pool = pool_frame(args.pool)
    selected, report = produce(train, val, v2, pool,
                               per_model=args.per_model, seed=args.seed)
    dest, rep = Path(args.output), Path(args.report)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rep.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(dest, index=False)
    rep.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("model_counts", "source_counts", "audit_cross_source")},
                     indent=2, ensure_ascii=False))
    print(f"Selected {len(selected)} external rows -> {dest}")
    print(f"Selection report -> {rep}")
    print("Manual source/license/label verification remains necessary.")


if __name__ == "__main__":
    main()
