"""Deterministic source-balanced, model-balanced E1 curation."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
spec=importlib.util.spec_from_file_location(
    "select_open_pool_e1",
    Path(__file__).resolve().parents[1] / "scripts/select_open_pool_e1.py",
)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from fhf.labels import MODEL_NAMES


def fixture_rows(n=12):
    rows=[]
    for model in MODEL_NAMES:
        for k in range(n):
            rows.append({
                "text": f"{model} artificial response {k} words " * 5,
                "model": model,
                "text_key": f"{model}-{k}",
                "input_dataset": "open_pool",
                "source_dataset": f"source-{k%3}",
                "prompt_key": f"prompt-{k}",
                "source_row": len(rows),
            })
    return pd.DataFrame(rows)


def test_source_balance_per_model_and_determinism():
    frame=fixture_rows()
    a=mod.pick_balanced(frame, per_model=9, seed=17)
    b=mod.pick_balanced(frame, per_model=9, seed=17)
    assert a.index.tolist()==b.index.tolist()
    assert a.model.value_counts().to_dict()=={m:9 for m in MODEL_NAMES}
    for _, sub in a.groupby("model"):
        counts=sub.source_dataset.value_counts()
        assert counts.max()-counts.min()<=1
        assert sub.prompt_key.nunique()==9


def test_underfilled_model_rejected():
    frame=fixture_rows(n=4)
    with pytest.raises(ValueError, match="only 4 eligible"):
        mod.pick_balanced(frame, per_model=5)
