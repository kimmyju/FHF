"""Evaluate or submit V2 neural + lexical specialist ensemble.

Neural weights are FIXED, not selected on the frozen official validation.
Only official TRAIN OOF predictions determine the lexical sub-ensemble weights.
Use --mode validation for a diagnostic or --mode test for Kaggle submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fhf.data import TextDataset, load_csv, make_collator
from fhf.model import FingerprintModel
from fhf.labels import MODEL_NAMES, MODEL_TO_ID
from fhf.metrics import report
from fhf.v2_confusion import diagnose
from fhf.v2_ensemble import select_lexical_weights, select_neural_weight, combine


def neural_probs(checkpoint, frame, batch_size=16, max_length=384,
                 base_encoder_override=None):
    model, tokenizer, _ = FingerprintModel.from_checkpoint(
        checkpoint, base_encoder_override=base_encoder_override)
    if not model.heads.get("model",False):
        raise ValueError(f"{checkpoint}: no 18-class head")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    ds = TextDataset(frame, tokenizer, max_length, style_config=model.style_config)
    loader = DataLoader(ds,batch_size=batch_size,shuffle=False,
                        collate_fn=make_collator(tokenizer))
    matrices=[]
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels",None)
            batch.pop("families",None)
            output=model(**{k: v.to(device) for k,v in batch.items()})
            matrices.append(output["log_probs"].float().exp().cpu().numpy())
    result=np.concatenate(matrices,axis=0)
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run(mode, data, official_train, oof_file, classical_model, checkpoints,
        output, neural_weight=0.70, batch_size=16, max_length=384,
        base_encoder_override=None, neural_oof_file=None):
    if mode not in ("validation","test"):
        raise ValueError("mode must be validation or test")
    if not len(data) or data.id.astype(str).duplicated().any():
        raise ValueError("Empty inputs or duplicate IDs")
    if mode=="validation" and "model" not in data:
        raise ValueError("Validation requires ground-truth labels")
    if mode=="test" and "model" in data:
        raise ValueError("Test submission must be unlabeled")
    oof=pd.read_csv(oof_file,dtype={"id":str,"prompt_id":str})
    weights,oof_bacc=select_lexical_weights(oof,official_train)
    selected_neural_oof_bacc=None
    if isinstance(neural_weight,str) and neural_weight=="auto":
        if not neural_oof_file or len(checkpoints)!=1:
            raise ValueError("--neural-weight auto needs --neural-oof and exactly one checkpoint")
        neural_oof=pd.read_csv(neural_oof_file,dtype={"id":str})
        neural_weight,selected_neural_oof_bacc=select_neural_weight(
            oof,neural_oof,official_train,weights)
    neural_weight=float(neural_weight)
    if not 0<=neural_weight<=1:
        raise ValueError("neural_weight outside [0,1]")
    if not checkpoints and neural_weight>0:
        raise ValueError("Need neural checkpoints or --neural-weight 0")
    lexical=joblib.load(classical_model)
    component=lexical.predict_probabilities(data.text.astype(str).tolist())
    neural=[neural_probs(p,data,batch_size,max_length,base_encoder_override)
            for p in checkpoints]
    fused=combine(neural,component,weights,neural_weight=neural_weight)
    predictions=np.asarray(MODEL_NAMES)[fused.argmax(axis=1)]
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if mode=="test":
        prediction=pd.DataFrame({"id":data.id,"model":predictions})
        prediction.to_parquet(output,index=False)
    else:
        prediction=pd.DataFrame({
            "id":data.id,"prompt_id":data.prompt_id,
            "true_model":data.model,"pred_model":predictions
        })
        for i,name in enumerate(MODEL_NAMES):
            prediction[f"prob_{name}"]=fused[:,i]
        prediction.to_csv(output,index=False)
        diagnose(prediction, output.parent / "diagnostics")
    summary={
        "mode":mode,"n":len(data),"output":str(output),
        "neural_weight":neural_weight,
        "num_neural_models":len(checkpoints),
        "lexical_weights_selected_on_train_oof":weights,
        "lexical_oof_bacc":oof_bacc,
        "neural_mixture_tuned_oof_bacc":selected_neural_oof_bacc,
        "note":"The held-out validation was never used to select weights."
    }
    if mode=="validation":
        true=data.model.map(MODEL_TO_ID).to_numpy()
        metrics=report(true,fused.argmax(axis=1))
        summary["validation_bacc"]=metrics["balanced_accuracy_observed_classes"]
        summary["family_accuracy"]=metrics["family_accuracy"]
        summary["recall_by_model"]=metrics["recall_by_model"]
    output.with_suffix(".json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["validation","test"],default="validation")
    p.add_argument("--data", default=None)
    p.add_argument("--train",default="outputs/dual-path-prompt-aware/train_split.csv")
    p.add_argument("--oof",default="outputs/fingerprint-v2/classical/oof_predictions.csv")
    p.add_argument("--classical",default="outputs/fingerprint-v2/classical/lexical_specialists.joblib")
    p.add_argument("--checkpoint",action="append",default=[])
    p.add_argument("--neural-weight",default="0.7",
                   help="Fixed 0..1 or auto with --neural-oof")
    p.add_argument("--neural-oof",default=None)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--max-length",type=int,default=384)
    p.add_argument("--base-encoder-override",default=None)
    p.add_argument("--output",default=None)
    a=p.parse_args()
    path=a.data or ("outputs/dual-path-prompt-aware/val.csv"
                    if a.mode=="validation" else "data/test.csv")
    output=a.output or ("outputs/fingerprint-v2/ensemble/validation.csv"
                        if a.mode=="validation" else "outputs/fingerprint-v2/ensemble/submission.parquet")
    run(a.mode, load_csv(path,labeled=a.mode=="validation"),load_csv(a.train),
        a.oof,a.classical,a.checkpoint,output,a.neural_weight,
        a.batch_size,a.max_length,a.base_encoder_override,a.neural_oof)


if __name__=="__main__":
    main()
