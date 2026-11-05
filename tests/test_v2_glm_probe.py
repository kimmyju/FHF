import numpy as np
import pandas as pd
import pytest
from fhf.v2_glm_probe import FOCUS, grouped_compare


def sample():
    rows=[]
    for p in range(15):
        for m in FOCUS:
            rows.append({"id":f"{p}_{m}","prompt_id":f"q{p}",
                         "model":m,"text":f"response {m} for prompt {p}"})
    return pd.DataFrame(rows)


def test_grouped_probe_outputs_oof_for_all_rows():
    df=sample()
    embeddings=np.zeros((len(df),4),dtype=float)
    embeddings[:,0]=np.arange(len(df))%3
    result,pred=grouped_compare(df,embeddings,folds=3)
    assert len(pred)==len(df)
    assert set(result["methods"])=={"char_tfidf","frozen_head_input"}
    assert len(result["methods"]["char_tfidf"]["folds"])==3
    assert 0<=result["methods"]["frozen_head_input"]["bacc_oof"]<=1


def test_duplicated_id_rejected():
    df=sample()
    df.loc[1,"id"]=df.loc[0,"id"]
    with pytest.raises(ValueError,match="Duplicate IDs"):
        grouped_compare(df)
