"""Focused checks for read-only cross-source external audit."""
import importlib.util
from pathlib import Path

import pandas as pd

spec=importlib.util.spec_from_file_location(
    "audit_external_pool", Path(__file__).resolve().parents[1] / "scripts/audit_external_pool.py"
)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def toy_model_frame(items, prompts, source):
    return pd.DataFrame({
        "text": [x[0] for x in items],
        "model": [x[1] for x in items],
        "prompt_key": prompts,
        "source_dataset": [source] * len(items),
    })


def test_exact_overlap_duplicate_conflict_and_provisional_balance():
    tr=pd.DataFrame({"id":["t1"],"text":["Train response"],"model":["gpt-5.5"],"prompt_id":["p1"]})
    va=pd.DataFrame({"id":["v1"],"text":["Val response"],"model":["gpt-5.5"],"prompt_id":["p2"]})
    v2=toy_model_frame([
        ("Open text one long enough to pass the filter", "gpt-5.5"),
        ("Val response", "gpt-5.5"),
        ("Shared content has several words in the response", "gpt-5.5"),
        ("Conflict answer has sufficient words for inclusion", "gpt-5.5"),
    ], ["a","b","c","d"],"v2")
    pool=toy_model_frame([
        ("Shared content has several words in the response", "gpt-5.5"),
        ("Conflict answer has sufficient words for inclusion", "gemini-3.1-pro-preview"),
        ("New valid response with sufficiently many words", "gemini-3.1-pro-preview"),
        ("Train response", "gpt-5.5"),
    ], ["e","f","g","h"],"pool")
    details, model, total, sources, metrics=mod.audit(tr,va,v2,pool,min_words=3,max_words=200)
    assert metrics["cross_source"]["distinct_normalized_responses_shared_between_v2_and_open_pool"] == 2
    assert metrics["cross_source"]["exact_val_overlap_rows"] == 1
    assert metrics["cross_source"]["exact_train_overlap_rows"] == 1
    assert metrics["cross_source"]["distinct_response_keys_with_conflicting_model_labels"] == 1
    assert metrics["cross_source"]["provisional_selected_rows_after_dedup"] == 3
    assert details.selected_after_exact_dedup.sum() == 3
    assert total.loc[total.model=="gpt-5.5","official_train"].iloc[0] == 1


def test_risk_flags_and_unknown_are_distinguished():
    tr=pd.DataFrame({"id":["t"],"text":["train"],"model":["gpt-5.5"],"prompt_id":["t"]})
    va=pd.DataFrame({"id":["v"],"text":["validation"],"model":["gpt-5.5"],"prompt_id":["v"]})
    v2=toy_model_frame([("Long safe example generated response","gpt-5.5")],["x"],"v2")
    pool=toy_model_frame([
        ("Flagged model output sample text","gpt-5.5"),
        ("Unknown flag status sample text","gpt-5.5"),
    ],["a","b"],"pool")
    pool["flag_kb_proxy"] = [True, pd.NA]
    d,*_=mod.audit(tr,va,v2,pool,min_words=2)
    flagged=d[d.text.str.startswith("Flagged")].iloc[0]
    unknown=d[d.text.str.startswith("Unknown")].iloc[0]
    assert not flagged.provisional_eligible
    assert unknown.unknown_risk_metadata
