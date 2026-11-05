"""3-class GLM-vs-Claude Fable study on official TRAIN only.

- GroupKFold over official prompt_id (never use official Validation).
- Compare raw text character TF-IDF with linear probe on frozen checkpoint
  model_head INPUT features (after style fusion), not class-head predictions.
- Default external-only pretraining checkpoint has never seen official TRAIN,
  making its representation-based grouped CV more interpretable.
- If using official-fine-tuned checkpoint, warn that upstream representation
  already saw every TRAIN example: probe CV is NOT an unbiased evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fhf.data import load_csv
from fhf.external_data import normalize_text
from fhf.labels import MODEL_NAMES
from fhf.v2_glm_probe import FOCUS, grouped_compare, save_comparison


def extract_head_features(checkpoint, subset, batch_size=16, max_length=384,
                          base_encoder_override=None):
    import torch
    from torch.utils.data import DataLoader
    from fhf.data import TextDataset, make_collator
    from fhf.model import FingerprintModel

    model, tokenizer, _ = FingerprintModel.from_checkpoint(
        checkpoint, base_encoder_override=base_encoder_override)
    if model.model_head is None:
        raise ValueError("Checkpoint does not have an 18-class model head")
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    ds=TextDataset(subset, tokenizer, max_length=max_length,
                   style_config=model.style_config)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,
                      collate_fn=make_collator(tokenizer))
    captured=[]
    def capture(_module, inputs, _output):
        if len(inputs)!=1:
            raise RuntimeError("Expected model-head input tensor")
        captured.append(inputs[0].detach().float().cpu().numpy().copy())
    handle=model.model_head.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for batch in loader:
                batch.pop("labels",None)
                batch.pop("families",None)
                model(**{k:v.to(device) for k,v in batch.items()})
    finally:
        handle.remove()
        model.cpu()
    result=np.concatenate(captured,axis=0)
    if result.shape[0]!=len(subset):
        raise RuntimeError("Feature extraction row count mismatch")
    if not np.isfinite(result).all():
        raise ValueError("Non-finite model-head features")
    return result


def run(train, destination, checkpoint=None, folds=5,
        batch_size=16, max_length=384, base_encoder_override=None,
        model_pretrained_on_official=False):
    needed={"id","text","model","prompt_id"}
    if not needed.issubset(train):
        raise ValueError(f"Missing official TRAIN fields {needed-set(train)}")
    if train.id.astype(str).duplicated().any():
        raise ValueError("Duplicate official TRAIN IDs")
    subset=train.loc[train.model.isin(FOCUS)].copy().reset_index(drop=True)
    if subset.empty or set(subset.model)!=set(FOCUS):
        raise ValueError("All three focus classes are required")
    if subset.prompt_id.isna().any():
        raise ValueError("Missing prompt IDs")
    normalized=subset.text.map(normalize_text)
    if normalized.duplicated().any():
        raise ValueError("Exact duplicated response in official focus subset")
    features=None
    if checkpoint:
        features=extract_head_features(
            checkpoint, subset, batch_size=batch_size, max_length=max_length,
            base_encoder_override=base_encoder_override)
    result,predictions=grouped_compare(
        subset, features, folds=folds)
    result["source"]="official TRAIN only; official Validation never accessed"
    result["checkpoint"]=str(checkpoint) if checkpoint else None
    result["checkpoint_exposed_to_official_train"]=bool(model_pretrained_on_official)
    if checkpoint and model_pretrained_on_official:
        result["warning"]=(
            "The checkpoint itself was trained on all official TRAIN rows. "
            "Although the diagnostic classifier uses GroupKFold, representation "
            "performance can be optimistic. Prefer external-only pretrain checkpoint.")
    else:
        result["warning"]=(
            "Frozen official TRAIN GroupKFold results diagnose separability; "
            "they do not establish future 18-class Kaggle performance.")
    save_comparison(result,predictions,destination)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",default="outputs/dual-path-prompt-aware/train_split.csv")
    p.add_argument("--checkpoint",default="outputs/fingerprint-v2/pretrain-isolated/checkpoint",
                   help="Use 'none' for lexical-only CPU run")
    p.add_argument("--checkpoint-saw-official-train",action="store_true")
    p.add_argument("--out",default="outputs/fingerprint-v2/glm-root-cause/probes")
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--max-length",type=int,default=384)
    p.add_argument("--base-encoder-override",default=None)
    a=p.parse_args()
    checkpoint=None if a.checkpoint.lower()=="none" else a.checkpoint
    result=run(load_csv(a.train),a.out,checkpoint,a.folds,a.batch_size,
               a.max_length,a.base_encoder_override,
               a.checkpoint_saw_official_train)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
