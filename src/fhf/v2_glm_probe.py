"""Leakage-controlled official-train three-way GLM/Fable grouped comparison."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

FOCUS=("glm-5.3","glm-5.3-flash","claude-fable-5")


def grouped_compare(subset, frozen_features=None, folds=5):
    """Generate per-row OOF predictions from prompt-disjoint probes."""
    required={"id","model","prompt_id","text"}
    if not required.issubset(subset):
        raise ValueError("Missing required fields")
    if set(subset.model)!=set(FOCUS):
        raise ValueError("Only the three focus labels must be selected")
    if subset.id.astype(str).duplicated().any():
        raise ValueError("Duplicate IDs")
    if subset.prompt_id.isna().any():
        raise ValueError("Missing prompt groups")
    y=subset.model.astype(str).to_numpy()
    groups=subset.prompt_id.astype(str).to_numpy()
    if not 2<=folds<=len(set(groups)):
        raise ValueError("Invalid grouped folds")
    if frozen_features is not None:
        frozen_features=np.asarray(frozen_features,dtype=np.float32)
        if len(frozen_features)!=len(y) or frozen_features.ndim!=2 or not np.isfinite(frozen_features).all():
            raise ValueError("Frozen feature dimensions/values invalid")
    kinds=["char_tfidf"]
    if frozen_features is not None:
        kinds.append("frozen_head_input")
    outputs={name:np.full(len(y),"",dtype=object) for name in kinds}
    fold_scores={name:[] for name in kinds}
    split=GroupKFold(n_splits=folds)
    for index,(fit,hold) in enumerate(split.split(subset,y,groups)):
        if set(groups[fit])&set(groups[hold]):
            raise RuntimeError("Prompt group leakage")
        if set(y[fit])!=set(FOCUS) or set(y[hold])!=set(FOCUS):
            raise ValueError("Each CV fit and holdout must contain all 3 classes")
        text_train=subset.text.iloc[fit].astype(str).tolist()
        text_test=subset.text.iloc[hold].astype(str).tolist()
        char=make_pipeline(
            TfidfVectorizer(analyzer="char",ngram_range=(3,5),
                            max_features=30000,min_df=2,
                            lowercase=False,sublinear_tf=True),
            LogisticRegression(C=1.0,max_iter=800,random_state=42))
        char.fit(text_train,y[fit])
        pred=char.predict(text_test)
        outputs["char_tfidf"][hold]=pred
        fold_scores["char_tfidf"].append({
            "fold":index,"n":int(len(hold)),
            "bacc":float(balanced_accuracy_score(y[hold],pred))})
        if frozen_features is not None:
            probe=make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1,max_iter=1000,random_state=42))
            probe.fit(frozen_features[fit],y[fit])
            pred=probe.predict(frozen_features[hold])
            outputs["frozen_head_input"][hold]=pred
            fold_scores["frozen_head_input"].append({
                "fold":index,"n":int(len(hold)),
                "bacc":float(balanced_accuracy_score(y[hold],pred))})
    summary={"n":len(y),"folds":folds,"class_counts":{
        name:int((y==name).sum()) for name in FOCUS},
        "method":"GroupKFold by official TRAIN prompt_id",
        "methods":{}}
    frame=subset[["id","prompt_id","model"]].rename(
        columns={"model":"true_model"}).reset_index(drop=True).copy()
    for name in kinds:
        pred=outputs[name]
        if any(not x for x in pred):
            raise RuntimeError("Incomplete grouped OOF predictions")
        frame[f"{name}_pred_model"]=pred
        matrix=confusion_matrix(y,pred,labels=FOCUS)
        summary["methods"][name]={
            "bacc_oof":float(balanced_accuracy_score(y,pred)),
            "correct":int((pred==y).sum()),
            "recall_by_model":{
                item:float(np.mean(pred[y==item]==item)) for item in FOCUS},
            "confusion_true_rows_pred_columns":matrix.tolist(),
            "class_order":list(FOCUS),
            "folds":fold_scores[name]}
    return summary,frame


def save_comparison(report,frame,directory):
    out=Path(directory)
    out.mkdir(parents=True,exist_ok=True)
    (out/"probe_summary.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    frame.to_csv(out/"official_train_oof_predictions.csv",index=False)
    return out
