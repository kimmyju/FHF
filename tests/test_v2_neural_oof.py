"""Neural folds are strictly prompt-disjoint and never use official val."""
import importlib.util
from pathlib import Path
import pandas as pd

def _mod():
    path=Path(__file__).resolve().parents[1]/"scripts/fingerprint_v2_neural_oof.py"
    spec=importlib.util.spec_from_file_location("neural_oof",path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_prepare_neural_folds(tmp_path):
    mod=_mod()
    frame=pd.DataFrame({
        "id":range(30),
        "text":[f"example {i}" for i in range(30)],
        "model":["gpt-5.5"]*30,
        "prompt_id":[f"prompt-{i//3}" for i in range(30)],
    })
    mod.prepare(frame,tmp_path,folds=5)
    for i in range(5):
        a=pd.read_csv(tmp_path/f"fold_{i}"/"train.csv")
        b=pd.read_csv(tmp_path/f"fold_{i}"/"val.csv")
        assert set(a.prompt_id).isdisjoint(b.prompt_id)
        assert len(a)+len(b)==len(frame)
    cmd=mod.fold_command(0,tmp_path,python="python")
    assert "paths.hard_pair_weights_json=null" in cmd
