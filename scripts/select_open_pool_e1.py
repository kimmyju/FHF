"""Select 100 external examples per model from audited NTL v2 + Open Pool.

Use with configs/ablations/within_family_style_perturb_openpool_e1.yaml.
Select only; never train or modify the frozen official split.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd

from audit_external_pool import audit, read_table, v2_frame, pool_frame
from fhf.external_data import normalize_text, digest
from fhf.labels import MODEL_NAMES


def pick_balanced(rows, per_model=100, seed=42):
    """Deterministic source round-robin with prompt diversity, O(N log N).

    Never duplicate a row or silently undersample a model.
    """
    if per_model < 1:
        raise ValueError("per_model must be positive")
    rng = np.random.default_rng(seed)
    work = rows.copy()
    if not work.index.is_unique:
        raise ValueError("Candidate indices must be unique")
    if work.prompt_key.isna().any():
        raise ValueError("Missing prompt keys")
    work["prompt_identity"] = work.input_dataset.astype(str) + ":" + work.prompt_key.astype(str)
    selected = []
    for model in MODEL_NAMES:
        group = work.loc[work.model == model].copy()
        if len(group) < per_model:
            raise ValueError(f"{model}: only {len(group)} eligible, need {per_model}")
        group["_random"] = rng.random(len(group))
        group = group.sort_values("_random", kind="stable")
        queues = {str(source): deque(part.index.tolist())
                  for source, part in group.groupby("source_dataset", sort=True)}
        seen_prompts = set()
        usage = Counter()
        for _ in range(per_model):
            available = [s for s, q in queues.items() if q]
            if not available:
                raise RuntimeError(f"{model}: no eligible rows remain")
            source = min(available, key=lambda s: (usage[s], s))
            q = queues[source]
            # Prefer new prompt keys without making a full-corpus rescan.
            rotation_limit = min(16, len(q))
            for _ in range(rotation_limit):
                if work.at[q[0], "prompt_identity"] not in seen_prompts:
                    break
                q.rotate(-1)
            idx = q.popleft()
            selected.append(idx)
            seen_prompts.add(work.at[idx, "prompt_identity"])
            usage[source] += 1
    picked = work.loc[selected].copy()
    if len(picked) != per_model * len(MODEL_NAMES) or not picked.text_key.is_unique:
        raise RuntimeError("Selection not balanced or unique")
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
