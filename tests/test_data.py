"""Test exact competition label policy and prompt-disjoint splitting."""
import pandas as pd
import pytest
from fhf.labels import MODEL_NAMES, MODEL_FAMILY_IDS, FAMILY_NAMES
from fhf.data import grouped_split, load_csv


def test_labels_and_families():
    assert len(MODEL_NAMES)==18
    assert len(set(MODEL_NAMES))==18
    assert len(FAMILY_NAMES)==6
    assert len(MODEL_FAMILY_IDS)==18


def test_prompt_disjoint():
    df=pd.DataFrame({'id':[f'i{i}' for i in range(20)],
                     'text':['hello world']*20,
                     'model':[MODEL_NAMES[i%18] for i in range(20)],
                     'prompt_id':[i//2 for i in range(20)]})
    train,val=grouped_split(df,0.2,42)
    assert set(train.prompt_id).isdisjoint(val.prompt_id)
    assert len(train)+len(val)==len(df)


def test_unknown_label_rejected(tmp_path):
    path=tmp_path/'data.csv'
    pd.DataFrame({'id':['x'],'text':['hello'],'model':['gpt-4']}).to_csv(path,index=False)
    with pytest.raises(ValueError,match='Unknown'):
        load_csv(path)
