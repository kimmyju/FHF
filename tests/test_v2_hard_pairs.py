"""Hard pair weights forbid contamination and reward symmetric OOF confusions."""
import pandas as pd
import pytest
from fhf.labels import MODEL_NAMES

def test_oof_weight_formula():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/"scripts/fingerprint_v2_hard_pairs.py"
    spec=importlib.util.spec_from_file_location("hard",path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    train=pd.DataFrame({"id":[str(i) for i in range(6)],
                        "model":[MODEL_NAMES[0]]*3+[MODEL_NAMES[1]]*3})
    oof=pd.DataFrame({"id":train.id,"true_model":train.model,
                      "specialist_pred_model":[MODEL_NAMES[1]]*2+
                      [MODEL_NAMES[0]]+[MODEL_NAMES[0]]+
                      [MODEL_NAMES[1]]*2})
    w=mod.derive(train,oof)
    assert w[MODEL_NAMES[0]][MODEL_NAMES[1]]>1.0
    assert w[MODEL_NAMES[0]][MODEL_NAMES[1]]==w[MODEL_NAMES[1]][MODEL_NAMES[0]]
    with pytest.raises(ValueError,match="official TRAIN"):
        mod.derive(train, oof.iloc[:-1])
