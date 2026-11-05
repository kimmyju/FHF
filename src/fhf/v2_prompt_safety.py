"""Isolate unverified overlarge external prompt groups in contrastive learning.

Original prompt_id columns and grouped validation splits are NOT modified.
Only sampler/contrastive-view prompt identities are anonymized per row.
"""
from __future__ import annotations

import pandas as pd


def safe_prompt_ids(training, max_group_size=None):
    """Return a prompt-ID vector and group-sanitization metrics.

    A size threshold is a sensitivity-analysis heuristic, not proof of a
    wrongly grouped prompt. All examples and their true model labels remain.
    """
    required = {"id", "prompt_id"}
    if not required.issubset(training):
        raise ValueError(f"Missing {required-set(training.columns)}")
    if training.id.astype(str).duplicated().any():
        raise ValueError("Duplicate training IDs")
    ids = training.prompt_id
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError("Missing prompt IDs")
    if max_group_size is None:
        return ids.astype(str).reset_index(drop=True).tolist(), {
            "enabled": False, "affected_groups": 0,
            "affected_rows": 0, "rows_preserved": int(len(training))
        }
    if isinstance(max_group_size, bool) or int(max_group_size) != max_group_size or max_group_size < 2:
        raise ValueError("max_group_size must be an integer >= 2 or null")
    original = ids.astype(str).reset_index(drop=True)
    sizes = original.map(original.value_counts())
    masked = sizes > int(max_group_size)
    output = original.copy()
    # The synthetic prefix is distinct from native IDs, validated before use.
    artificial = "__v2_unverified_prompt_row__:"
    if original.str.startswith(artificial).any():
        raise ValueError("Original prompt ID conflicts with synthetic prefix")
    row_ids = training.id.astype(str).reset_index(drop=True)
    output.loc[masked] = artificial + row_ids.loc[masked]
    if output.isna().any() or output.str.strip().eq("").any():
        raise RuntimeError("Sanitization created invalid prompt IDs")
    return output.tolist(), {
        "enabled": True,
        "max_group_size": int(max_group_size),
        "affected_groups": int(original[masked].nunique()),
        "affected_rows": int(masked.sum()),
        "rows_preserved": int(len(training)),
        "original_unique_prompt_ids": int(original.nunique()),
        "effective_unique_prompt_ids": int(output.nunique()),
        "caution": "No original prompt texts were available to verify these groups. This is an ablation only.",
    }
