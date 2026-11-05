"""Read-only attribution diagnosis: family vs flat vs hierarchical decisions.

Never fit/adjust weights on frozen validation. This module only examines
already-trained checkpoint outputs and exports explanation-oriented metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from .labels import FAMILY_NAMES, MODEL_NAMES, MODEL_TO_FAMILY, MODEL_TO_ID

FOCUS = ("glm-5.3", "glm-5.3-flash", "claude-fable-5")
G_ZAI = [MODEL_NAMES.index("glm-5.3"), MODEL_NAMES.index("glm-5.3-flash")]
G_ANTHROPIC = [i for i, name in enumerate(MODEL_NAMES)
                if MODEL_TO_FAMILY[name] == "Anthropic"]
ZAI = FAMILY_NAMES.index("ZAI")
ANTHROPIC = FAMILY_NAMES.index("Anthropic")


def analyze_outputs(ids, true_labels, family_probs, local_model_logits,
                    final_probs, flat_probs=None, hierarchical_probs=None):
    """Inspect three failure paths without using labels during inference.

    family_probs are the 6-way family head, local_model_logits are the
    unconstrained model head (within-family), final_probs are model.log_probs.
    Flat branch probabilities can be absent in non-residual architectures.
    """
    labels = np.asarray(true_labels, dtype=str)
    ids = np.asarray(ids, dtype=str)
    fam = np.asarray(family_probs, dtype=float)
    local = np.asarray(local_model_logits, dtype=float)
    final = np.asarray(final_probs, dtype=float)
    n = len(labels)
    if len(ids) != n or len(set(ids)) != n:
        raise ValueError("Duplicate IDs or count mismatch")
    if fam.shape != (n, len(FAMILY_NAMES)) or local.shape != (n, len(MODEL_NAMES)):
        raise ValueError("Invalid family/model head matrix dimensions")
    if final.shape != (n, len(MODEL_NAMES)):
        raise ValueError("Invalid final probabilities shape")
    if any(np.asarray(x).shape != final.shape for x in
           (flat_probs, hierarchical_probs) if x is not None):
        raise ValueError("Flat/hierarchical probabilities have incorrect dimensions")
    for name, probs in (("family", fam), ("final", final),
                        ("flat", flat_probs), ("hierarchical", hierarchical_probs)):
        if probs is None:
            continue
        if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(
            probs.sum(1), 1, atol=1e-4):
            raise ValueError(f"{name}: invalid probability matrix")
    if not np.isfinite(local).all():
        raise ValueError("Invalid local model logits")
    if any(label not in MODEL_TO_ID for label in labels):
        raise ValueError("Unknown validation labels")

    indices = np.flatnonzero(np.isin(labels, FOCUS))
    focus = labels[indices]
    winners = np.asarray(FAMILY_NAMES)[fam.argmax(axis=1)]
    chosen = np.asarray(MODEL_NAMES)[final.argmax(axis=1)]
    report_rows = []
    for i in indices:
        model = labels[i]
        truth_family = MODEL_TO_FAMILY[model]
        top2 = np.argsort(-final[i])[:2]
        row = {
            "id": ids[i], "true_model": model,
            "pred_model": chosen[i],
            "true_family": truth_family,
            "family_head_pred": winners[i],
            "family_head_correct": bool(winners[i] == truth_family),
            "final_correct": bool(chosen[i] == model),
            "p_family_ZAI": float(fam[i, ZAI]),
            "p_family_Anthropic": float(fam[i, ANTHROPIC]),
            "family_ZAI_minus_Anthropic": float(fam[i, ZAI]-fam[i, ANTHROPIC]),
            "p_final_ZAI": float(final[i, G_ZAI].sum()),
            "p_final_Anthropic": float(final[i, G_ANTHROPIC].sum()),
            "p_final_claude_fable": float(final[i, MODEL_NAMES.index("claude-fable-5")]),
            "p_final_true_model": float(final[i, MODEL_TO_ID[model]]),
            "top1_prob": float(final[i, top2[0]]),
            "top2_prob": float(final[i, top2[1]]),
            "top1_minus_top2": float(final[i, top2[0]] - final[i, top2[1]]),
        }
        if truth_family == "ZAI":
            zai_winner = G_ZAI[int(np.argmax(local[i, G_ZAI]))]
            row["oracle_within_ZAI_pred"] = MODEL_NAMES[zai_winner]
            row["oracle_within_ZAI_correct"] = bool(zai_winner == MODEL_TO_ID[model])
        else:
            row["oracle_within_ZAI_pred"] = ""
            row["oracle_within_ZAI_correct"] = None
        for prefix, probs in (("flat", flat_probs), ("hierarchical", hierarchical_probs)):
            if probs is not None:
                row[f"{prefix}_pred_model"] = MODEL_NAMES[int(probs[i].argmax())]
                row[f"{prefix}_p_ZAI"] = float(probs[i, G_ZAI].sum())
                row[f"{prefix}_p_Anthropic"] = float(probs[i, G_ANTHROPIC].sum())
                row[f"{prefix}_p_fable"] = float(probs[i, MODEL_NAMES.index("claude-fable-5")])
        report_rows.append(row)

    dataframe = pd.DataFrame(report_rows)
    summary = {
        "n_total_validation": int(n),
        "n_focus": int(len(dataframe)),
        "disclaimer": "Post-hoc frozen validation diagnosis ONLY. No fitting or model selection on validation.",
        "flat_branch_present": flat_probs is not None,
        "per_model": {},
    }
    for model in FOCUS:
        r = dataframe[dataframe.true_model == model]
        if r.empty:
            continue
        summary["per_model"][model] = {
            "n": int(len(r)),
            "final_correct": int(r.final_correct.sum()),
            "family_head_correct": int(r.family_head_correct.sum()),
            "family_head_Anthropic": int((r.family_head_pred == "Anthropic").sum()),
            "final_pred_fable": int((r.pred_model == "claude-fable-5").sum()),
            "mean_family_p_ZAI": float(r.p_family_ZAI.mean()),
            "mean_family_p_Anthropic": float(r.p_family_Anthropic.mean()),
            "mean_final_p_ZAI": float(r.p_final_ZAI.mean()),
            "mean_final_p_Anthropic": float(r.p_final_Anthropic.mean()),
            "mean_final_p_fable": float(r.p_final_claude_fable.mean()),
            "mean_top1_margin": float(r.top1_minus_top2.mean()),
        }
        if model.startswith("glm-"):
            summary["per_model"][model]["oracle_within_ZAI_correct"] = int(
                r.oracle_within_ZAI_correct.sum())
            summary["per_model"][model]["oracle_within_ZAI_n"] = int(len(r))
        if flat_probs is not None:
            for prefix in ("flat", "hierarchical"):
                summary["per_model"][model][f"{prefix}_pred_fable"] = int(
                    (r[f"{prefix}_pred_model"] == "claude-fable-5").sum())
                summary["per_model"][model][f"mean_{prefix}_p_ZAI"] = float(
                    r[f"{prefix}_p_ZAI"].mean())
                summary["per_model"][model][f"mean_{prefix}_p_Anthropic"] = float(
                    r[f"{prefix}_p_Anthropic"].mean())
    glm = dataframe[dataframe.true_family == "ZAI"]
    summary["glm_aggregate"] = {
        "n": int(len(glm)),
        "family_head_Anthropic": int((glm.family_head_pred == "Anthropic").sum()),
        "final_pred_fable": int((glm.pred_model == "claude-fable-5").sum()),
        "family_head_ZAI": int((glm.family_head_pred == "ZAI").sum()),
        "oracle_within_ZAI_correct": int(glm.oracle_within_ZAI_correct.sum()),
        "final_wrong_but_family_head_correct": int(
            ((~glm.final_correct) & glm.family_head_correct).sum()),
        "final_wrong_and_family_head_wrong": int(
            ((~glm.final_correct) & (~glm.family_head_correct)).sum()),
    }
    return summary, dataframe


def save_report(summary, frame, directory):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    (directory/"head_diagnosis.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8")
    frame.to_csv(directory/"focus_head_probabilities.csv",index=False)
    return directory
