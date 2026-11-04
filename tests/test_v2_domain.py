"""Domain shift test never consults the official validation set."""
import importlib.util
from pathlib import Path
import pandas as pd

def test_domain_audit_synthetic():
    p=Path(__file__).resolve().parents[1]/"scripts/fingerprint_v2_domain.py"
    spec=importlib.util.spec_from_file_location("domain",p)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from fhf.labels import MODEL_NAMES
    official=pd.DataFrame({
        "model":[name for name in MODEL_NAMES for j in range(12)],
        "prompt_id":[f"official_prompt_{j}" for name in MODEL_NAMES for j in range(12)],
        "text":[f"Official prose alpha sentence {name} example {j} good"
                for name in MODEL_NAMES for j in range(12)]
    })
    external=pd.DataFrame({
        "model":[name for name in MODEL_NAMES for j in range(15)],
        "prompt_id":[f"external_prompt_{j}" for name in MODEL_NAMES for j in range(15)],
        "text":[f"External prose beta example {name} numbered {j} distinct"
                for name in MODEL_NAMES for j in range(15)]
    })
    result=mod.audit_domain(official,external,max_features=2000)
    assert result["external_sampled_rows"]==len(official)
    assert 0<=result["domain_roc_auc"]<=1
