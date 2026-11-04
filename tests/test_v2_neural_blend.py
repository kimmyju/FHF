"""Neural blend weight selection uses strictly aligned official TRAIN OOF."""
import numpy as np
import pandas as pd
from fhf.labels import MODEL_NAMES
from fhf.v2_ensemble import select_neural_weight


def test_neural_oof_weight_grid():
    train=pd.DataFrame({
        "id":[str(i) for i in range(18)],
        "prompt_id":[f"p{i}" for i in range(18)],
        "model":MODEL_NAMES,
    })
    lexical=train.rename(columns={"model":"true_model"}).copy()
    neural=lexical.copy()
    for k in ("char","word","specialist"):
        for m in MODEL_NAMES:
            lexical[f"{k}_prob_{m}"] = 1.0/18
    for i,m in enumerate(MODEL_NAMES):
        neural[f"neural_prob_{m}"]=np.array(
            [0.9 if j==i else 0.1/17 for j in range(18)])
    w,score=select_neural_weight(lexical,neural,train,
                                {"char":0,"word":0,"specialist":1},step=0.5)
    # The first optimum can be 0.5: with a uniform lexical model,
    # a 0.5 neural mixture already predicts every class correctly.
    assert 0.5 <= w <= 1.0
    assert score == 1.0
