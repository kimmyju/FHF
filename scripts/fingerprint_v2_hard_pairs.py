"""Create within-provider hard-negative weights from official TRAIN OOF only.

Never accepts validation IDs: OOF IDs must match the *entire* saved official
train split. Confusions are symmetrized and bounded.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fhf.data import load_csv
from fhf.labels import MODEL_NAMES, MODEL_TO_FAMILY


def derive(train, oof, field="specialist_pred_model", strength=8.0):
    needed={"id","true_model",field}
    if not needed.issubset(oof):
        raise ValueError(f"Missing OOF columns {needed-set(oof.columns)}")
    if oof.id.astype(str).duplicated().any():
        raise ValueError("Duplicate OOF IDs")
    if set(oof.id.astype(str)) != set(train.id.astype(str)):
        raise ValueError("OOF must cover exactly official TRAIN IDs, not validation")
    aligned=oof.copy()
    aligned["id"]=aligned.id.astype(str)
    lookup=train.assign(id=train.id.astype(str)).set_index("id")
    truth=lookup.loc[aligned.id, "model"].to_numpy()
    if not np.array_equal(truth, aligned.true_model.to_numpy()):
        raise ValueError("OOF labels disagree with official TRAIN labels")
    if not set(aligned[field]).issubset(MODEL_NAMES):
        raise ValueError("Unexpected predicted model names")
    if strength<0:
        raise ValueError("strength must be nonnegative")
    errors=pd.crosstab(aligned.true_model,aligned[field]).reindex(
        index=MODEL_NAMES,columns=MODEL_NAMES,fill_value=0)
    counts=pd.Series(truth).value_counts().reindex(MODEL_NAMES,fill_value=0)
    weights={}
    for a in MODEL_NAMES:
        row={}
        for b in MODEL_NAMES:
            if a==b or MODEL_TO_FAMILY[a]!=MODEL_TO_FAMILY[b]:
                continue
            if counts[a]==0 or counts[b]==0:
                continue
            symmetric=(errors.loc[a,b]/counts[a]+errors.loc[b,a]/counts[b])/2
            value=round(float(min(3.0,1.0+strength*symmetric)),4)
            if value>1.0:
                row[b]=value
        if row:
            weights[a]=row
    return weights


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="outputs/dual-path-prompt-aware/train_split.csv")
    p.add_argument("--oof", default="outputs/fingerprint-v2/classical/oof_predictions.csv")
    p.add_argument("--prediction-column", default="specialist_pred_model")
    p.add_argument("--strength", type=float, default=8.0)
    p.add_argument("--output", default="outputs/fingerprint-v2/hard_pairs.json")
    a=p.parse_args()
    result=derive(load_csv(a.train),pd.read_csv(a.oof,dtype={"id":str}),
                  a.prediction_column,a.strength)
    dest=Path(a.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"Saved hard-negative weights to {dest}")


if __name__=="__main__":
    main()
