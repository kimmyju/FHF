import numpy as np
import pandas as pd
import pytest

from fhf.v2_ensemble import read_lexical_oof, select_lexical_weights, combine
from fhf.labels import MODEL_NAMES


def make_oof():
    train = pd.DataFrame({
        "id":["1","2","3","4"],
        "prompt_id":["a","b","c","d"],
        "model": MODEL_NAMES[:4],
    })
    oof = train.rename(columns={"model":"true_model"}).copy()
    for kind in ("char","word","specialist"):
        for model in MODEL_NAMES:
            oof[f"{kind}_prob_{model}"] = 0.05 if model==MODEL_NAMES[0] else 0.95/17
    return train,oof


def test_lexical_oof_weight_alignment():
    train,oof=make_oof()
    y,m=read_lexical_oof(oof,train)
    assert len(y)==4 and m["specialist"].shape==(4,18)
    weights,score=select_lexical_weights(oof,train,step=0.5)
    assert sum(weights.values()) == 1
    assert 0 <= score <= 1
    result=combine([],m,weights,neural_weight=0)
    assert result.shape==(4,18)
    assert np.allclose(result.sum(1),1)


def test_no_holdout_leakage_or_misalignment():
    train,oof=make_oof()
    with pytest.raises(ValueError,match="official TRAIN"):
        read_lexical_oof(oof.iloc[:-1],train)
    with pytest.raises(ValueError,match="true_model"):
        tampered=oof.copy()
        tampered.loc[0,"true_model"]=MODEL_NAMES[5]
        read_lexical_oof(tampered,train)
