"""Optional expensive 5-fold neural OOF, using official TRAIN only.

Modes:
  prepare: write prompt-disjoint folds (no model training).
  train: train one fold from the external-only pretraining checkpoint.
  collect: infer held-out responses from all five trained fold checkpoints.

No official frozen validation examples or OOF-derived hard-pair weights
are used to train any OOF fold.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from fhf.data import load_csv
from fhf.external_data import normalize_text
from fhf.labels import MODEL_NAMES


def prepare(train, directory, folds=5):
    if not {"id","prompt_id","model","text"}.issubset(train):
        raise ValueError("Official TRAIN requires id, prompt_id, model and text")
    if train.id.astype(str).duplicated().any():
        raise ValueError("Duplicate TRAIN IDs")
    if not 2<=folds<=train.prompt_id.nunique():
        raise ValueError("Invalid folds count")
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    split=GroupKFold(n_splits=folds)
    for fold,(ti,vi) in enumerate(split.split(train,groups=train.prompt_id)):
        fit=train.iloc[ti].copy()
        valid=train.iloc[vi].copy()
        if set(fit.prompt_id.astype(str)) & set(valid.prompt_id.astype(str)):
            raise RuntimeError("Fold prompt overlap")
        if set(fit.text.map(normalize_text)) & set(valid.text.map(normalize_text)):
            raise RuntimeError("Fold exact response text overlap")
        where=directory/f"fold_{fold}"
        where.mkdir(parents=True,exist_ok=True)
        fit.to_csv(where/"train.csv",index=False)
        valid.to_csv(where/"val.csv",index=False)
    print(f"Prepared {folds} prompt-disjoint training folds under {directory}")


def fold_command(fold, directory, python=sys.executable):
    directory=Path(directory)
    where=directory/f"fold_{fold}"
    if not (where/"train.csv").exists() or not (where/"val.csv").exists():
        raise FileNotFoundError(f"Prepare fold {fold} first")
    overrides={
        "paths.frozen_train_csv":str(where/"train.csv"),
        "paths.frozen_validation_csv":str(where/"val.csv"),
        "paths.hard_pair_weights_json":"null",
        "paths.checkpoint_dir":str(where/"checkpoint"),
        "paths.train_split_csv":str(where/"train_export.csv"),
        "paths.validation_csv":str(where/"val_export.csv"),
        "paths.history_json":str(where/"history.json"),
        "paths.config_snapshot":str(where/"resolved.yaml"),
        "paths.eval_csv":str(where/"val_export.csv"),
        "paths.eval_report":str(where/"evaluation.json"),
        "wandb.run_name":f"v2-neural-oof-fold-{fold}",
    }
    cmd=[python,"scripts/train.py","--config","configs/fingerprint_v2/finetune.yaml"]
    for key,value in overrides.items():
        cmd+=["--set",f"{key}={value}"]
    return cmd


def collect(train, directory, folds, batch_size, max_length):
    # Import at call time: preparing folds doesn't need GPU initialization.
    from fingerprint_v2_ensemble import neural_probs
    preds=np.full((len(train),len(MODEL_NAMES)),np.nan,dtype=float)
    index={str(id_):i for i,id_ in enumerate(train.id.astype(str))}
    visit=np.zeros(len(train),dtype=int)
    for fold in range(folds):
        where=Path(directory)/f"fold_{fold}"
        val=load_csv(where/"val.csv")
        probabilities=neural_probs(where/"checkpoint",val,batch_size,max_length)
        for row,id_ in enumerate(val.id.astype(str)):
            ix=index[id_]
            preds[ix]=probabilities[row]
            visit[ix]+=1
    if not np.all(visit==1) or not np.isfinite(preds).all():
        raise RuntimeError("Neural OOF predictions did not cover every TRAIN ID once")
    out=train[["id","prompt_id","model"]].rename(columns={"model":"true_model"}).copy()
    for col,label in enumerate(MODEL_NAMES):
        out[f"neural_prob_{label}"]=preds[:,col]
    out["neural_pred_model"]=np.asarray(MODEL_NAMES)[preds.argmax(1)]
    target=Path(directory)/"neural_oof_predictions.csv"
    out.to_csv(target,index=False)
    print(f"Saved {len(out)} prompt-disjoint neural OOF predictions at {target}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("prepare","train","collect"),required=True)
    p.add_argument("--train",default="outputs/dual-path-prompt-aware/train_split.csv")
    p.add_argument("--out",default="outputs/fingerprint-v2/neural-oof")
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--fold",type=int,default=0)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--max-length",type=int,default=384)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    frame=load_csv(a.train)
    if a.mode=="prepare":
        prepare(frame,a.out,a.folds)
    elif a.mode=="train":
        if not 0<=a.fold<a.folds:
            p.error("Invalid fold index")
        cmd=fold_command(a.fold,a.out)
        print(" ".join(cmd))
        if not a.dry_run:
            subprocess.run(cmd,cwd=Path(__file__).resolve().parents[1],check=True)
    else:
        collect(frame,a.out,a.folds,a.batch_size,a.max_length)


if __name__=="__main__":
    main()
