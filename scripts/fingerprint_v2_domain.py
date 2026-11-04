"""Training-only domain shift diagnostics: official vs external text.

The frozen official validation is not accessed. A prompt-disjoint 80/20 split
tests whether text features trivially identify dataset origin. High ROC AUC
indicates source/domain shift, not proof of incorrect generator labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from fhf.data import load_csv
from fhf.external_data import normalize_text
from fhf.labels import MODEL_NAMES


def audit_domain(official, external, seed=42, max_features=120000):
    required = {"text", "model", "prompt_id"}
    for title, frame in (("official", official), ("external", external)):
        if not required.issubset(frame):
            raise ValueError(f"{title}: missing {required-set(frame.columns)}")
        if not set(frame.model).issubset(MODEL_NAMES):
            raise ValueError(f"{title}: unknown model labels")
    rng=np.random.default_rng(seed)
    # Equalize provider/model exposure by selecting at most official class count
    # from the external pool. Match each exact model count when possible.
    pieces=[]
    for name in MODEL_NAMES:
        number=int((official.model==name).sum())
        pool=external[external.model==name]
        if number<1 or len(pool)<number:
            raise ValueError(f"{name}: insufficient external rows for domain audit")
        picks=rng.choice(pool.index.to_numpy(),size=number,replace=False)
        pieces.append(pool.loc[picks])
    ext=pd.concat(pieces,ignore_index=True)
    if set(official.text.map(normalize_text)) & set(ext.text.map(normalize_text)):
        raise ValueError("Domain audit contains official/external exact duplicates")
    a=official[["text","prompt_id","model"]].copy()
    b=ext[["text","prompt_id","model"]].copy()
    a["domain"]=0
    b["domain"]=1
    a["group"]="official:"+a.prompt_id.astype(str)
    b["group"]="external:"+b.prompt_id.astype(str)
    frame=pd.concat([a,b],ignore_index=True)
    splitter=GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=seed)
    train_idx,eval_idx=next(splitter.split(frame,groups=frame.group))
    train=frame.iloc[train_idx]
    test=frame.iloc[eval_idx]
    if set(train.group) & set(test.group):
        raise RuntimeError("Prompt leakage in domain audit split")
    if test.domain.nunique()!=2 or train.domain.nunique()!=2:
        raise ValueError("Domain audit train/test must contain both sources")
    pipe=Pipeline([
        ("tfidf",TfidfVectorizer(
            analyzer="char",ngram_range=(3,5),lowercase=False,
            min_df=2,max_features=max_features,sublinear_tf=True)),
        ("lr",LogisticRegression(
            C=1.0,max_iter=1000,class_weight="balanced",random_state=seed))
    ])
    pipe.fit(train.text.astype(str),train.domain)
    proba=pipe.predict_proba(test.text.astype(str))[:,list(pipe.classes_).index(1)]
    report={
        "official_rows":len(official),
        "external_sampled_rows":len(ext),
        "test_rows":len(test),
        "domain_roc_auc":float(roc_auc_score(test.domain,proba)),
        "domain_balanced_accuracy":float(balanced_accuracy_score(
            test.domain,(proba>=0.5).astype(int))),
        "meaning":"High AUC means official-vs-external style is distinguishable. "
                  "It does not prove label errors. Held-out official validation "
                  "has not been used.",
    }
    per={}
    for model in MODEL_NAMES:
        mask=test.model.eq(model)
        subset=test.loc[mask]
        if subset.domain.nunique()==2 and len(subset)>=10:
            per[model]=float(roc_auc_score(subset.domain,proba[mask.to_numpy()]))
    report["auc_by_model"]=per
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official",default="outputs/dual-path-prompt-aware/train_split.csv")
    p.add_argument("--external",default="data/external/full/selected_3495_per_model.csv")
    p.add_argument("--output",default="outputs/fingerprint-v2/domain_shift.json")
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    report=audit_domain(load_csv(a.official),load_csv(a.external),seed=a.seed)
    dest=Path(a.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
